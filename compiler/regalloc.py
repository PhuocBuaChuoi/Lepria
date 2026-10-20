"""Register allocation + peephole combine for the Lepria x86-64 backend.

The backend emits a stack machine: every local and temporary lives in a slot
``[rbp-8*slot]`` and is moved to/from scratch registers around each operation.
This module removes most of that traffic in two stages:

1. **Graph-colouring register allocator.**  Slots holding 64-bit integers become
   virtual registers.  A CFG + backward liveness analysis yields an interference
   graph; the graph is coloured with Chaitin's *simplify / spill* scheme, whose
   interference graph is (near-)chordal for reducible code, so it runs in
   ~O(V+E) instead of the O(n^3) brute-force clique search.

   Values live across a ``call`` are coloured from the caller-preserved registers
   ``rbx, r12..r15``; everything else may additionally use the volatile
   ``rax, rcx, r8..r11`` (mem2reg-style: short-lived temporaries stay in
   registers for their whole scope).  A per-slot *forbidden* set records which
   physical registers are written while the value is still needed, so scratch
   registers and implicit clobbers (``rax``/``rdx`` for ``div``, all volatiles
   for ``call``...) are never assigned to a live value.  Only callee-preserved
   colours actually used are saved/restored in the frame.

   Float (``xmm``) slots are coloured separately from ``xmm6..xmm15`` (also
   nonvolatile on Win64).  Values live across a ``call``, and all float values
   in a block containing a division, stay in memory (see
   :func:`_float_keep_in_memory`); register colouring still gives ~2x on
   call-free, division-free float arithmetic.

2. **Peephole combine.**  Adjacent instructions are fused: store->load
   forwarding, ``mov+add`` -> ``lea`` (effective-address folding), constant
   folding of ``mov imm; add imm``, removal of ``mov X, X`` and local copy
   propagation (including ``xmm`` moves) that drops dead copies.

3. **List scheduling.**  Within each basic block a dependency DAG is built from
   register (with aliases), flag and memory resources; ready instructions are
   picked from a max-heap keyed by critical-path latency, then a bounded random
   adjacent-swap local search perturbs the schedule to escape local minima.

4. **Short encodings.**  Equivalent instruction forms are chosen by size:
   zeroing uses ``xor r32, r32`` / ``sub`` when flags are dead and ``mov r32, 0``
   when they are live, small non-negative constants use the zero-extending
   ``mov r32, imm32`` form, ``imul ..., 2`` becomes ``add``/``lea`` when no
   consumer reads CF/OF, and ``lea r, [r]`` collapses to ``mov`` (then
   disappears).  ``push``/``pop`` are counted as the 1-byte instructions they
   are.

5. **Cache-line packing.**  A fast O(n*64) DP tracks the offset modulo 64 and,
   at loop-header anchors, chooses a ``.p2align`` that balances padding bytes
   against the number of cache lines touched and instructions that straddle a
   line.  Only backward-jump targets are padded: the NOPs then sit outside the
   steady-state loop instead of being executed every iteration.  ``.p2align``
   is absolute, so the first padded header also anchors the DP's offset model;
   prologues and cold code pay no padding at all.

Only physical registers that the backend never touches are used, so the
transformation is conservative.  Toggles: ``LEP_NO_OPT`` (all), ``LEP_NO_SCHED``,
``LEP_NO_SHRINK``, ``LEP_NO_ALIGN``, ``LEP_NO_LAYOUT``, ``LEP_NO_FLOAT``,
``LEP_NO_VOLATILE`` (callee-saved pool only).
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple

SLOT_RE = re.compile(r"\[rbp-(\d+)\]")
CALLEE_POOL = ["rbx", "r12", "r13", "r14", "r15", "rsi", "rdi"]
VOLATILE_POOL = ["rax", "rcx", "rdx", "r8", "r9", "r10", "r11"]
POOL = CALLEE_POOL + VOLATILE_POOL
XMM_POOL = [f"xmm{i}" for i in range(6, 16)]
FLOAT_TYPES = {"float", "double", "ldouble"}

_READ_WRITE = {"add", "sub", "imul", "and", "or", "xor", "shl", "shr", "sal", "sar"}
_READ_WRITE_1 = {"neg", "not", "inc", "dec"}
_MOVES = {"mov", "movsx", "movzx", "movsxd", "lea"}
_COMPARE = {"cmp", "test"}
_FLOAT_MOVES = {"movsd", "movss", "movapd", "movaps", "movq", "movd",
                "cvtsi2sd", "cvtsi2ss", "cvttsd2si", "cvttss2si",
                "cvtsd2si", "cvtss2si", "cvtsd2ss", "cvtss2sd"}
_FLOAT_RW = {"addsd", "subsd", "mulsd", "divsd", "sqrtsd",
             "addss", "subss", "mulss", "divss", "sqrtss"}
_FLOAT_CMP = {"ucomisd", "comisd", "ucomiss", "comiss"}


def _slot_of(operand: str):
    m = SLOT_RE.fullmatch(operand.strip())
    return int(m.group(1)) if m else None


def _is_label(line: str) -> bool:
    return bool(line) and not line[0].isspace() and line.rstrip().endswith(":")


def _label_name(line: str) -> str:
    return line.strip()[:-1]


def _ends_block(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return s.startswith("j") or s == "ret" or s.startswith("ret ")


def _split(line: str):
    s = line.strip()
    if not s or " " not in s:
        return None, []
    mn, rest = s.split(None, 1)
    args = [a.strip() for a in rest.split(",")]
    return mn.lower(), args


def _use_def(line: str) -> Tuple[Set[int], Set[int]]:
    mn, args = _split(line)
    if mn is None:
        return set(), set()
    slots = [_slot_of(a) for a in args]
    reads: Set[int] = set()
    writes: Set[int] = set()

    def R(i):
        if i < len(slots) and slots[i] is not None:
            reads.add(slots[i])

    def W(i):
        if i < len(slots) and slots[i] is not None:
            writes.add(slots[i])

    if mn in _MOVES or mn in _FLOAT_MOVES:
        W(0)
        R(1)
    elif mn in _READ_WRITE or mn in _FLOAT_RW:
        R(0)
        W(0)
        R(1)
    elif mn in _COMPARE or mn in _FLOAT_CMP:
        R(0)
        R(1)
    elif mn in _READ_WRITE_1:
        R(0)
        W(0)
    else:
        for i in range(len(args)):
            R(i)
    return reads, writes


# --------------------------------------------------------------------- CFG
def _build_blocks(body: List[str]):
    n = len(body)
    bounds = {0}
    for i, line in enumerate(body):
        if _is_label(line):
            bounds.add(i)
        elif _ends_block(line):
            bounds.add(i + 1)
    starts = sorted(b for b in bounds if b < n)
    blocks = [{"start": s, "end": (starts[k + 1] - 1 if k + 1 < len(starts) else n - 1),
               "succ": []} for k, s in enumerate(starts)]

    label_to_block: Dict[str, int] = {}
    for bi, b in enumerate(blocks):
        for i in range(b["start"], b["end"] + 1):
            if _is_label(body[i]):
                label_to_block[_label_name(body[i])] = bi

    for bi, b in enumerate(blocks):
        last = body[b["end"]].strip()
        succ: List[int] = []
        if last.startswith("j"):
            parts = last.split()
            tgt = parts[1] if len(parts) > 1 else None
            if last.startswith("jmp"):
                if tgt in label_to_block:
                    succ.append(label_to_block[tgt])
            else:
                if tgt in label_to_block:
                    succ.append(label_to_block[tgt])
                if bi + 1 < len(blocks):
                    succ.append(bi + 1)
        elif bi + 1 < len(blocks):
            succ.append(bi + 1)
        b["succ"] = succ
    return blocks, label_to_block


def _analyse(body: List[str], int_slots: Set[int]):
    blocks, _ = _build_blocks(body)
    for b in blocks:
        u: Set[int] = set()
        d: Set[int] = set()
        for i in range(b["start"], b["end"] + 1):
            r, w = _use_def(body[i])
            u |= (r & int_slots) - d
            d |= (w & int_slots)
        b["use"], b["def"] = u, d
        b["live_in"], b["live_out"] = set(), set()

    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            lo: Set[int] = set()
            for s in b["succ"]:
                lo |= blocks[s]["live_in"]
            li = b["use"] | (lo - b["def"])
            if li != b["live_in"] or lo != b["live_out"]:
                b["live_in"], b["live_out"] = li, lo
                changed = True

    adj: Dict[int, Set[int]] = {s: set() for s in int_slots}
    for b in blocks:
        live = set(b["live_out"])
        for i in range(b["end"], b["start"] - 1, -1):
            r, w = _use_def(body[i])
            w &= int_slots
            for d in w:
                for x in live:
                    if x != d:
                        adj[d].add(x)
                        adj[x].add(d)
            live = (live - w) | (r & int_slots)
    return adj


# ------------------------------------------------------- graph colouring (K)
def _colour(adj: Dict[int, Set[int]], k: int) -> Dict[int, int]:
    """Chaitin simplify/spill colouring: ~O(V+E), no O(n^3) clique search."""
    work = {v: set(ns) for v, ns in adj.items()}
    order: List[int] = []
    while work:
        pick = None
        for v, ns in work.items():
            if len(ns) < k:
                pick = v
                break
        if pick is None:  # all degrees >= K -> spill the least constrained node
            pick = max(work, key=lambda v: len(work[v]))
        for ns in work.values():
            ns.discard(pick)
        del work[pick]
        order.append(pick)

    colours: Dict[int, int] = {}
    for v in reversed(order):
        used = {colours[n] for n in adj[v] if n in colours}
        c = 0
        while c in used:
            c += 1
        if c < k:
            colours[v] = c
    return colours


# ------------------------------------------------------------------ rewrite
def _colour_restricted(adj: Dict[int, Set[int]], k: int,
                       allowed) -> Dict[int, int]:
    """Like :func:`_colour`, but a node may only take colours from
    ``allowed(v)`` (an iterable of colour indices, in preference order)."""
    work = {v: set(ns) for v, ns in adj.items()}
    order: List[int] = []
    while work:
        pick = None
        for v, ns in work.items():
            if len(ns) < k:
                pick = v
                break
        if pick is None:
            pick = max(work, key=lambda v: len(work[v]))
        for ns in work.values():
            ns.discard(pick)
        del work[pick]
        order.append(pick)

    colours: Dict[int, int] = {}
    for v in reversed(order):
        used = {colours[n] for n in adj[v] if n in colours}
        for c in allowed(v):
            if c not in used:
                colours[v] = c
                break
    return colours


def _rewrite(line: str, reg_of: Dict[int, str]) -> str:
    res = SLOT_RE.sub(lambda m: reg_of.get(int(m.group(1)), m.group(0)), line)
    return re.sub(r"\b(?:qword|dword|word|byte)\s+ptr\s+([a-zA-Z0-9]+)\b", r"\1", res)


_MOV_MOV = re.compile(r"mov (\w+), (\w+)")
_MOV_XMM = re.compile(r"(movsd|movapd|movss|movaps) (\w+), (\w+)")
_ADDSUB = re.compile(r"(add|sub) (\w+), (\w+)")
_MOV_IMM = re.compile(r"mov (\w+), (-?\d+)")
_ADD_IMM = re.compile(r"add (\w+), (-?\d+)")
_MOV_REG_MEM = re.compile(r"mov (\w+), \[(rbp-\d+)\]")
_MOV_MEM_REG = re.compile(r"mov \[(rbp-\d+)\], (\w+)")
_WRITES_DST = re.compile(
    r"^\s*(?:mov|movsx|movzx|movsxd|lea|add|sub|and|or|xor|imul|shl|shr|sar|"
    r"neg|not|inc|dec|pop|cvttsd2si|cvtsi2sd)\s+(\w+)")

REG64 = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"}
XMM_REGS = {f"xmm{i}" for i in range(16)}
REGS = REG64 | XMM_REGS
_WRITE_FIRST = {"mov", "movsx", "movzx", "movsxd", "lea", "pop", "movq", "movsd",
                "movapd", "movss", "movaps", "cvttsd2si", "cvtsi2sd",
                "cvtsi2ss", "cvttss2si"}
_READ_WRITE_FIRST = _READ_WRITE | _READ_WRITE_1
_MOVE_MNEMONICS = {"mov", "movsd", "movapd", "movss", "movaps"}
_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")

_ALIAS: Dict[str, str] = {}
for _full, _lows in (("rax", ("eax", "ax", "al")), ("rbx", ("ebx", "bx", "bl")),
                     ("rcx", ("ecx", "cx", "cl")), ("rdx", ("edx", "dx", "dl")),
                     ("rsi", ("esi", "si", "sil")), ("rdi", ("edi", "di", "dil"))):
    _ALIAS[_full] = _full
    for _l in _lows:
        _ALIAS[_l] = _full
for _n in range(8, 16):
    _full = f"r{_n}"
    _ALIAS[_full] = _full
    for _suf in ("d", "w", "b"):
        _ALIAS[f"{_full}{_suf}"] = _full
for _x in range(16):
    _ALIAS[f"xmm{_x}"] = f"xmm{_x}"


def _read_regs(line: str) -> Set[str]:
    """64-bit registers read by *line* (memory operands count as reads)."""
    mn, args = _split(line)
    if mn is None:
        return set()
    toks = {_ALIAS[t] for t in _TOKEN.findall(line) if t in _ALIAS}
    if args:
        dst = _ALIAS.get(args[0], args[0])
        if dst in REGS and mn not in _READ_WRITE_FIRST:
            if mn in _WRITE_FIRST or mn.startswith("set") or mn == "movabs":
                toks.discard(dst)
    return toks


def _write_regs(line: str) -> Set[str]:
    mn, args = _split(line)
    if mn is None or not args:
        return set()
    dst = _ALIAS.get(args[0], args[0])
    if dst not in REGS:
        return set()
    if (mn in _READ_WRITE or mn in _READ_WRITE_1 or mn in _WRITE_FIRST
            or mn in _FLOAT_RW or mn in _FLOAT_MOVES
            or mn.startswith(("set", "cmov")) or mn == "movabs"):
        return {dst}
    return set()


def _implicit_write_regs(line: str) -> Set[str]:
    """Registers written implicitly by *line* (beyond ``_write_regs``)."""
    mn, args = _split(line)
    if mn is None:
        return set()
    if mn == "call":
        # Win64: all volatile GPRs may be clobbered by the callee.
        return set(VOLATILE_POOL)
    out: Set[str] = set()
    if mn in ("div", "idiv"):
        out |= {"rax", "rdx"}
    elif mn == "mul" or (mn == "imul" and len(args) == 1):
        out |= {"rax", "rdx"}
    if mn in ("cdq", "cqo", "cwd", "cdqe"):
        out |= {"rax", "rdx"}
    if mn.startswith("set"):
        out.add("rax")
    return out


def _analyse_alloc(body: List[str], int_slots: Set[int]):
    """Liveness + interference + per-slot forbidden physical registers.

    Returns ``(adj, call_live, forbidden)``:

    * ``call_live`` -- slots live across at least one ``call``; they may only
      use callee-preserved colours.
    * ``forbidden[s]`` -- physical registers written (explicitly or implicitly)
      while ``s`` is live across the instruction, so assigning ``s`` to one of
      them would lose its value.
    """
    blocks, _ = _build_blocks(body)
    for b in blocks:
        u: Set[int] = set()
        d: Set[int] = set()
        for i in range(b["start"], b["end"] + 1):
            r, w = _use_def(body[i])
            u |= (r & int_slots) - d
            d |= (w & int_slots)
        b["use"], b["def"] = u, d
        b["live_in"], b["live_out"] = set(), set()

    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            lo: Set[int] = set()
            for s in b["succ"]:
                lo |= blocks[s]["live_in"]
            li = b["use"] | (lo - b["def"])
            if li != b["live_in"] or lo != b["live_out"]:
                b["live_in"], b["live_out"] = li, lo
                changed = True

    adj: Dict[int, Set[int]] = {s: set() for s in int_slots}
    call_live: Set[int] = set()
    forbidden: Dict[int, Set[str]] = {s: set() for s in int_slots}
    for b in blocks:
        live = set(b["live_out"])
        for i in range(b["end"], b["start"] - 1, -1):
            line = body[i]
            mn, _ = _split(line)
            r, w = _use_def(line)
            r &= int_slots
            w &= int_slots

            # Any two distinct slots co-used by one instruction must not share
            # a physical register (two-address forms such as ``add r, r``
            # would otherwise compute the wrong value).
            group = r | w
            for a in group:
                adj[a].update(group - {a})

            if mn == "call":
                call_live |= live

            phys = _write_regs(line) | _implicit_write_regs(line)
            if phys:
                for s in live:
                    forbidden[s] |= phys
            if mn in _READ_WRITE or mn in _READ_WRITE_1:
                # Two-address forms read their source slot and write the
                # destination register; a source that dies here must not live
                # in that register (``sub rcx, [s]`` would become ``sub rcx, rcx``).
                wregs = _write_regs(line)
                for s in r - w:
                    forbidden[s] |= wregs
                # Memory-destination RMW: the physical source must not alias
                # the slot colour (``add [s], rax`` -> ``add rax, rax``).
                if w:
                    srcs = _read_regs(line)
                    for s in w:
                        forbidden[s] |= srcs

            for d in w:
                for x in live:
                    if x != d:
                        adj[d].add(x)
                        adj[x].add(d)
            live = (live - w) | r
    return adj, call_live, forbidden


def _phys_liveness(lines: List[str], blocks) -> List[Set[str]]:
    """Per-block live-out sets over physical registers."""
    for b in blocks:
        u: Set[str] = set()
        d: Set[str] = set()
        for i in range(b["start"], b["end"] + 1):
            r = _read_regs(lines[i])
            w = _write_regs(lines[i])
            u |= r - d
            d |= w
        b["puse"], b["pdef"] = u, d
        b["pin"], b["pout"] = set(), set()
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            po: Set[str] = set()
            for s in b["succ"]:
                po |= blocks[s]["pin"]
            pi = b["puse"] | (po - b["pdef"])
            if pi != b["pin"] or po != b["pout"]:
                b["pin"], b["pout"] = pi, po
                changed = True
    return [b["pout"] for b in blocks]


_ONE_MOVE = re.compile(r"(mov|movsd|movapd|movss|movaps) (\w+), (.+)")

# Mnemonics whose memory source operand carries no implicit size, so folding a
# ``[rbp-N]`` source into them would assemble as "ambiguous operand size".
_SIZE_AMBIG = {"cvtsi2sd", "cvtsi2ss", "movzx", "movsx", "movsxd",
               "imul", "mul", "idiv", "div"}


def _dead_or_redefined_before_use(lines: List[str], start: int, reg: str) -> bool:
    """True if every later use of *reg* from *start* on is preceded by a
    redefinition (so the value copied earlier is no longer needed)."""
    for j in range(start, len(lines)):
        if reg in _read_regs(lines[j]):
            return False
        if reg in _write_regs(lines[j]):
            return True
    return True


def _fold_block(block: List[str], live_out: Set[str]) -> List[str]:
    """Local copy propagation: forward a copy into its single following use.

    ``X = S ; ... use(X)`` becomes ``... use(S)`` and the copy is dropped when
    ``X`` is dead afterwards.  Runs to a fixpoint because each elimination can
    expose another.
    """
    lines = list(block)
    changed = True
    while changed:
        changed = False
        n = len(lines)
        reads = [_read_regs(l) for l in lines]
        out: List[str] = []
        i = 0
        while i < n:
            m1 = _ONE_MOVE.fullmatch(lines[i].strip())
            if m1 is not None and i + 1 < n:
                a, src = m1.group(2), m1.group(3).strip()
                if (a in REGS and (src in REGS
                                   or re.fullmatch(r"-?\d+", src)
                                   or src.startswith("[rbp-"))) \
                        and a not in live_out \
                        and _dead_or_redefined_before_use(lines, i + 2, a):
                    m2args = _split(lines[i + 1])[1]
                    dst2 = _ALIAS.get(m2args[0], m2args[0]) if m2args else None
                    mem_src = src.startswith("[rbp-")
                    imm_src = bool(re.fullmatch(r"-?\d+", src))
                    ok_mem = dst2 in REGS or dst2 in XMM_REGS
                    first_read = dst2 == a
                    writes_a = a in _write_regs(lines[i + 1])
                    mem_use = mem_src and re.search(
                        rf"\[[^\]]*\b{re.escape(a)}\b[^\]]*\]", lines[i + 1])
                    mn2 = _split(lines[i + 1])[0]
                    if (not writes_a and a in reads[i + 1]
                            and re.search(rf"\b{re.escape(a)}\b", lines[i + 1])
                            and not mem_use
                            and not (mem_src and mn2 in _SIZE_AMBIG)
                            and (not mem_src or (ok_mem and not first_read))
                            and (not imm_src
                                 or (dst2 in REG64 and not first_read))):
                        out.append(re.sub(rf"\b{re.escape(a)}\b", src,
                                          lines[i + 1]))
                        changed = True
                        i += 2
                        continue
                        out.append(re.sub(rf"\b{re.escape(a)}\b", src,
                                          lines[i + 1]))
                        changed = True
                        i += 2
                        continue
            out.append(lines[i])
            i += 1
        lines = out
    return lines


def _fold_copies(lines: List[str]) -> List[str]:
    """Drop a copy whose result is consumed only by the immediately following
    copy; the source is forwarded directly (local copy propagation)."""
    blocks, _ = _build_blocks(lines)
    lives = _phys_liveness(lines, blocks)
    out: List[str] = []
    for bi, b in enumerate(blocks):
        out.extend(_fold_block(lines[b["start"]:b["end"] + 1], lives[bi]))
    return out


_FLOAD0 = re.compile(r"(?:movsd|movapd) xmm0, (.+)")
_FOP_XMM1 = re.compile(r"(addsd|subsd|mulsd) xmm1, xmm0")


def _fold_float_ops(lines: List[str]) -> List[str]:
    """Fuse the stack-machine float binary sequence.

    The backend computes ``a OP b`` by loading both operands into ``xmm0`` and
    shuffling through ``xmm1`` (``xmm2`` for division).  Once the operands live
    in coloured xmm registers this becomes a chain of redundant copies; rewrite
    it to operate directly on the source registers.
    """
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        m0 = _FLOAD0.fullmatch(lines[i].strip())
        if m0 is not None:
            if i + 4 < n:
                s1, s2 = lines[i + 1].strip(), lines[i + 2].strip()
                s3, s4 = lines[i + 3].strip(), lines[i + 4].strip()
                m2 = re.fullmatch(r"(?:movsd|movapd) xmm0, (.+)", s2)
                m3 = _FOP_XMM1.fullmatch(s3)
                if (re.fullmatch(r"movapd xmm1, xmm0", s1) and m2 and m3
                        and re.fullmatch(r"movapd xmm0, xmm1", s4)):
                    out.append(f"    movsd xmm1, {m0.group(1)}")
                    out.append(f"    {m3.group(1)} xmm1, {m2.group(1)}")
                    out.append("    movapd xmm0, xmm1")
                    i += 5
                    continue
            if i + 5 < n:
                s1, s2 = lines[i + 1].strip(), lines[i + 2].strip()
                s3, s4, s5 = (lines[i + 3].strip(), lines[i + 4].strip(),
                              lines[i + 5].strip())
                m2 = re.fullmatch(r"(?:movsd|movapd) xmm0, (.+)", s2)
                if (re.fullmatch(r"movapd xmm1, xmm0", s1) and m2
                        and re.fullmatch(r"movapd xmm2, xmm0", s3)
                        and re.fullmatch(r"movapd xmm0, xmm1", s4)
                        and re.fullmatch(r"divsd xmm0, xmm2", s5)):
                    out.append(f"    movsd xmm1, {m0.group(1)}")
                    out.append(f"    movsd xmm2, {m2.group(1)}")
                    out.append("    movapd xmm0, xmm1")
                    out.append("    divsd xmm0, xmm2")
                    i += 6
                    continue
        out.append(lines[i])
        i += 1
    return out


_XMOV = re.compile(r"(movsd|movapd|movss|movaps) (\w+), (.+)")


def _fold_xmm_copies(lines: List[str]) -> List[str]:
    """Forward a copy ``xmm0 -> R`` when ``xmm0`` dies right after.

    ``movsd xmm0, Src ; movsd Dst, xmm0`` becomes ``movsd Dst, Src`` whenever
    ``xmm0`` is not read again before being redefined, which is the common shape
    the stack-machine backend produces around calls and float operators.
    """
    n = len(lines)
    reads = [_read_regs(l) for l in lines]
    writes = [_write_regs(l) for l in lines]

    live_after: List[bool] = [False] * n
    live = False
    for j in range(n - 1, -1, -1):
        live_after[j] = live
        s = lines[j].strip()
        if s.startswith("call"):
            live = False  # xmm0 is volatile, the call defines it
            continue
        if (_is_label(lines[j]) or s.startswith("j")
                or s in ("leave", "ret") or s.startswith("ret ")):
            live = True  # conservative at control-flow boundaries
            continue
        if "xmm0" in reads[j]:
            live = True
        elif "xmm0" in writes[j]:
            live = False

    out: List[str] = []
    i = 0
    while i < n:
        if i + 1 < n and not live_after[i + 1]:
            m1 = _XMOV.fullmatch(lines[i].strip())
            m2 = _XMOV.fullmatch(lines[i + 1].strip())
            if m1 and m2 and m1.group(2) == "xmm0" and m2.group(3) == "xmm0":
                src = m1.group(3).strip()
                if src != "xmm0":
                    out.append(f"    {m2.group(1)} {m2.group(2)}, {src}")
                    i += 2
                    continue
        out.append(lines[i])
        i += 1
    return out


# ------------------------------------------------ canonical / short encodings
_GPR64 = ({"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"}
          | {f"r{i}" for i in range(8, 16)})
_GPR32 = ({"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"}
          | {f"r{i}d" for i in range(8, 16)})
_GPR8 = ({"al", "bl", "cl", "dl", "sil", "dil", "bpl", "spl",
          "ah", "bh", "ch", "dh"}
         | {f"r{i}b" for i in range(8, 16)})
_GPR32_OF = {"rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx",
             "rsi": "esi", "rdi": "edi", "rbp": "ebp", "rsp": "esp"}
_GPR32_OF.update({f"r{i}": f"r{i}d" for i in range(8, 16)})
_HIGH_TOKENS = ({"sil", "dil", "bpl", "spl"}
                | {f"r{i}" for i in range(8, 16)}
                | {f"r{i}{s}" for i in range(8, 16) for s in ("d", "w", "b")}
                | {f"xmm{i}" for i in range(8, 16)})
_SSE2 = {"movsd", "movss", "movapd", "movaps", "addsd", "subsd", "mulsd",
         "divsd", "sqrtsd", "addss", "subss", "mulss", "divss", "sqrtss",
         "ucomisd", "comisd", "ucomiss", "comiss", "movq", "movd",
         "cvtsi2sd", "cvtsi2ss", "cvttsd2si", "cvttss2si", "cvtsd2ss",
         "cvtss2sd", "movdqa", "movdqu", "pxor"}
_ALU_ALL = {"add", "sub", "adc", "sbb", "and", "or", "xor", "cmp", "test"}


def _is_imm(op: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|\d+)", op.strip()))


def _imm(op: str) -> int:
    s = op.strip()
    neg = s.startswith("-")
    t = s.lstrip("+-")
    v = int(t, 16) if t.lower().startswith("0x") else int(t)
    return -v if neg else v


def _reg32(r: str) -> str:
    if r in _GPR32_OF:
        return _GPR32_OF[r]
    return r


def _mem_bytes(op: str) -> int:
    """Bytes of modrm (+sib, +displacement) for a ``[...]`` operand."""
    m = re.search(r"\[([^\]]*)\]", op)
    if not m:
        return 0
    inner = m.group(1).replace(" ", "")
    n = 1  # modrm
    if "rip" in inner:
        return n + 4
    disp = 0
    for t in re.findall(r"[+-]?[^+-]+", inner):
        if re.fullmatch(r"[+-]?\d+", t):
            disp += int(t)
    if "*" in inner or "rsp" in inner or "r12" in inner:
        n += 1  # sib
    if disp == 0:
        if "rbp" in inner or "r13" in inner or "*" in inner:
            n += 1  # mod=01 with disp8=0
    else:
        n += 1 if -128 <= disp <= 127 else 4
    return n


def _rex_bytes(args: List[str]) -> int:
    toks = set(_TOKEN.findall(" ".join(args)))
    if toks & _GPR64 or toks & _HIGH_TOKENS:
        return 1
    return 0


def _block_flags_live(block: List[str]) -> List[bool]:
    """``live[i]``: flags produced by instruction *i* are read before being
    overwritten.  Block entries are conservative (flags live-in)."""
    n = len(block)
    live = [True] * n
    cur = True
    for i in range(n - 1, -1, -1):
        live[i] = cur
        s = block[i].strip()
        if not s:
            continue
        if _is_label(block[i]) or s in ("ret", "leave") or s.startswith("ret "):
            cur = True
            continue
        mn, _ = _split(block[i])
        if mn is None or mn in _BARRIER_MN:
            if mn == "call":
                cur = False  # flags do not survive a call
            continue
        if mn == "jmp":
            continue
        if mn.startswith("j") or _flags_read(mn):
            cur = True
        elif mn in _FLAGS_W:
            cur = False
    return live


def _shrink_line(line: str, flags_dead: bool) -> str:
    """Rewrite an instruction to a shorter, semantically equal encoding."""
    mn, args = _split(line)
    if mn is None or len(args) < 2:
        return line
    dst, src = args[0], args[1]

    # ---- zero a 64-bit register: xor (2B), sub (2B) or mov r32 (5B) --------
    if mn in ("xor", "sub") and dst == src and dst in _GPR64:
        return f"    {mn} {_reg32(dst)}, {_reg32(dst)}"
    if mn == "and" and _is_imm(src) and _imm(src) == 0 and dst in _GPR64:
        # and r,0 and xor produce identical flags
        return f"    xor {_reg32(dst)}, {_reg32(dst)}"
    # ---- constants: zero -> xor/mov r32, small positive -> mov r32 --------
    if mn in ("mov", "movabs") and _is_imm(src) and dst in _GPR64:
        v = _imm(src)
        if v == 0:
            if flags_dead:
                return f"    xor {_reg32(dst)}, {_reg32(dst)}"
            return f"    mov {_reg32(dst)}, 0"
        if 0 <= v <= 0xFFFFFFFF:
            return f"    mov {_reg32(dst)}, {v}"
        if v > 0xFFFFFFFF:
            sv = v - (1 << 64)          # sign-extended imm32 form
            if -(1 << 31) <= sv <= (1 << 31) - 1:
                return f"    mov {dst}, {sv}"
        return line

    # ---- multiply by two (only when no flag consumer needs CF/OF) ---------
    if mn == "imul" and flags_dead:
        if len(args) == 3 and _is_imm(args[2]) and _imm(args[2]) == 2:
            d, s = args[0], args[1]
            if d in _GPR64 and s in _GPR64:
                return f"    lea {d}, [{s}+{s}]"  # dst is written, not read
        if len(args) == 2 and _is_imm(args[1]) and _imm(args[1]) == 2:
            d = args[0]
            if d in _GPR64:
                return f"    add {d}, {d}"

    # ---- lea r, [r] -> mov r, r (then dropped by the peephole) ------------
    if mn == "lea" and re.fullmatch(r"\[\w+\]", src):
        base = src[1:-1]
        return f"    mov {dst}, {base}"

    return line


def _shrink_encodings(lines: List[str]) -> List[str]:
    """Canonicalise to the shortest equivalent x86-64 encoding.

    Uses flags liveness so zeroing may use ``xor`` (2 bytes) instead of
    ``mov r32, 0`` (5 bytes) when no flag consumer follows; also selects the
    zero-extending 32-bit form for small non-negative constants and rewrites
    ``imul ..., 2`` to ``add``/``lea``.
    """
    blocks, _ = _build_blocks(lines)
    out: List[str] = []
    for b in blocks:
        block = lines[b["start"]:b["end"] + 1]
        live = _block_flags_live(block)
        for i, line in enumerate(block):
            out.append(_shrink_line(line, not live[i]))
    return _peephole(out)


def _store_load_forward(lines: List[str]) -> List[str]:
    known: Dict[str, str] = {}
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if _is_label(line) or s.startswith("call") or s in ("leave", "ret") or s.startswith("ret"):
            known.clear()
            out.append(line)
            continue
        if "xmm" in s:
            known.clear()
            out.append(line)
            continue

        mm = _MOV_MEM_REG.fullmatch(s)
        if mm:
            known[mm.group(1)] = mm.group(2)
            out.append(line)
            continue

        rl = _MOV_REG_MEM.fullmatch(s)
        if rl:
            dst, mem = rl.group(1), rl.group(2)
            for key in [k for k, v in known.items() if v == dst]:
                del known[key]
            if mem in known:
                src = known[mem]
                if src == dst:
                    continue  # store then load of same value -> redundant
                out.append(f"    mov {dst}, {src}")
            else:
                out.append(line)
            continue

        w = _WRITES_DST.match(line)
        if w:
            for key in [k for k, v in known.items() if v == w.group(1)]:
                del known[key]
        out.append(line)
    return out


_SETCC = re.compile(r"set([a-z]+)\s+al")
_TEST_RAX = re.compile(r"test\s+rax,\s*rax")
_MOVZX_RAX_AL = re.compile(r"movzx\s+rax,\s*al")
_CC_INV = {
    "e": "ne", "ne": "e", "z": "nz", "nz": "z",
    "l": "ge", "ge": "l", "le": "g", "g": "le",
    "b": "ae", "ae": "b", "be": "a", "a": "be",
    "c": "nc", "nc": "c", "s": "ns", "ns": "s",
    "p": "np", "np": "p", "a": "be", "g": "le",
}


def _redefined_before_read_in_block(lines: List[str], start: int, reg: str) -> bool:
    for j in range(start, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if _is_label(lines[j]) or _ends_block(lines[j]):
            return False
        if reg in _read_regs(lines[j]):
            return False
        if reg in _write_regs(lines[j]) or s.startswith("call"):
            return True
    return False


def _redefined_before_read_at_target(lines: List[str], tgt_label: str, reg: str) -> bool:
    tgt = tgt_label + ":"
    for j, l in enumerate(lines):
        if l.strip() == tgt:
            k = j + 1
            while k < len(lines) and (_is_label(lines[k]) or not lines[k].strip()):
                k += 1
            return _redefined_before_read_in_block(lines, k, reg)
    return False


def _peephole(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        s = line.strip()
        m = _MOV_MOV.fullmatch(s)
        if m and m.group(1) == m.group(2):
            continue  # mov X, X
        mx = _MOV_XMM.fullmatch(s)
        if mx and mx.group(2) == mx.group(3):
            continue  # movsd X, X
        out.append(line)

    changed = True
    while changed:
        changed = False
        res: List[str] = []
        i = 0
        while i < len(out):
            # Redundant jmp immediately before target label
            a = out[i].strip()
            parts = a.split()
            if len(parts) == 2 and parts[0] == "jmp":
                tgt = parts[1]
                j = i + 1
                while j < len(out) and not out[j].strip():
                    j += 1
                if j < len(out) and _is_label(out[j]) and _label_name(out[j]) == tgt:
                    i += 1
                    changed = True
                    continue

            # Condition branch fusion: setcc + movzx + test + j(e/ne) -> direct conditional jump
            if i + 3 < len(out):
                b = out[i + 1].strip()
                c = out[i + 2].strip()
                d = out[i + 3].strip()
                m_set = _SETCC.fullmatch(a)
                if m_set and _MOVZX_RAX_AL.fullmatch(b) and _TEST_RAX.fullmatch(c):
                    d_parts = d.split()
                    if len(d_parts) == 2 and d_parts[0] in ("je", "jne"):
                        cc = m_set.group(1)
                        if cc in _CC_INV:
                            tgt = d_parts[1]
                            if (_redefined_before_read_in_block(out, i + 4, "rax")
                                    and _redefined_before_read_at_target(out, tgt, "rax")):
                                fused_cc = _CC_INV[cc] if d_parts[0] == "je" else cc
                                res.append(f"    j{fused_cc} {tgt}")
                                i += 4
                                changed = True
                                continue

            if i + 1 < len(out):
                b = out[i + 1].strip()
                ma, mb = _MOV_MOV.fullmatch(a), _ADDSUB.fullmatch(b)
                if ma and mb and ma.group(1) == mb.group(2):
                    if mb.group(1) == "add":
                        res.append(f"    lea {ma.group(1)}, [{ma.group(2)}+{mb.group(3)}]")
                        i += 2
                        changed = True
                        continue
                    if re.fullmatch(r"-?\d+", mb.group(3)):
                        res.append(f"    lea {ma.group(1)}, [{ma.group(2)}-{mb.group(3)}]")
                        i += 2
                        changed = True
                        continue
                ma2, mb2 = _MOV_IMM.fullmatch(a), _ADD_IMM.fullmatch(b)
                if ma2 and mb2 and ma2.group(1) == mb2.group(1):
                    res.append(f"    mov {ma2.group(1)}, {int(ma2.group(2)) + int(mb2.group(2))}")
                    i += 2
                    changed = True
                    continue
            res.append(out[i])
            i += 1
        out = res
    return out


_RSP_ARG = re.compile(r"\[rsp\+(\d+)\]")


def _outgoing_bytes(body: List[str]) -> int:
    """Bytes of outgoing stack arguments the body may write at [rsp+32...]."""
    end = 0
    for line in body:
        for m in _RSP_ARG.finditer(line):
            end = max(end, int(m.group(1)) + 8)
    return max(0, end - 32)


# --------------------------------------------------------------------- entry
_FLOAT_DIV = re.compile(r"^divs[sd]\b")
_LEA_SLOT = re.compile(r"lea\s+\w+,\s*\[rbp-(\d+)\]")


def _float_keep_in_memory(body: List[str], float_slots: Set[int]) -> Set[int]:
    """Float slots that should stay in memory instead of an xmm register.

    Two allocation heuristics, both validated against the benchmarks:

    * values live across a ``call``: the callee owns the nonvolatile-xmm
      save/restore and the value must stay pinned anyway;
    * every float slot used in a basic block that performs a float division:
      the long, poorly-pipelined ``divsd`` latency makes the register-to-register
      copies on the recurrence chain cost more than a memory slot.

    Call-free, division-free float arithmetic still gains ~2x from registers.
    """
    if not float_slots:
        return set()
    blocks, _ = _build_blocks(body)
    for b in blocks:
        u: Set[int] = set()
        d: Set[int] = set()
        for i in range(b["start"], b["end"] + 1):
            r, w = _use_def(body[i])
            u |= (r & float_slots) - d
            d |= (w & float_slots)
        b["fuse"], b["fdef"] = u, d
        b["fin"], b["fout"] = set(), set()
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            fo: Set[int] = set()
            for s in b["succ"]:
                fo |= blocks[s]["fin"]
            fi = b["fuse"] | (fo - b["fdef"])
            if fi != b["fin"] or fo != b["fout"]:
                b["fin"], b["fout"] = fi, fo
                changed = True

    keep: Set[int] = set()
    for b in blocks:
        div = any(_FLOAT_DIV.match(body[i].strip())
                  for i in range(b["start"], b["end"] + 1))
        live = set(b["fout"])
        for i in range(b["end"], b["start"] - 1, -1):
            if body[i].strip().startswith("call"):
                keep |= live
            if div:
                keep |= {int(m.group(1)) for m in SLOT_RE.finditer(body[i])}
            r, w = _use_def(body[i])
            live = (live - (w & float_slots)) | (r & float_slots)
    return keep & float_slots


def _pettis_hansen(body: List[str]) -> List[str]:
    """Reorder basic blocks with the Pettis-Hansen heuristic.

    Classic Pettis-Hansen needs edge profile counts; with no profiler we use a
    static heuristic for edge weights (loop back-edges and fall-throughs are
    the likely-hot edges).  Chains are grown by repeatedly joining the
    heaviest remaining edge whose end-points are still a chain tail/head and
    which would not close a cycle, then chains are laid out to favour
    fall-through.  Where a block's original fall-through successor no longer
    follows it, an explicit ``jmp`` is emitted so semantics are unchanged.
    """
    if len(body) < 4:
        return body
    blocks, _ = _build_blocks(body)
    nb = len(blocks)
    if nb < 3:
        return body

    canon: List[str] = []
    prefix = "Lph"
    for line in body:
        if _is_label(line):
            prefix = _label_name(line)
            break
    for bi, b in enumerate(blocks):
        lbl = None
        for i in range(b["start"], b["end"] + 1):
            if _is_label(body[i]):
                lbl = _label_name(body[i])
                break
        canon.append(lbl if lbl is not None else f".L{prefix}.ph{bi}")

    def fallthrough(bi: int) -> Optional[int]:
        """Block entered by falling off the end of *bi* (not-taken branch)."""
        b = blocks[bi]
        if not _ends_block(body[b["end"]]):
            return bi + 1 if bi + 1 < nb else None
        last = body[b["end"]].strip()
        if last.startswith("j") and not last.startswith("jmp"):
            succ = b["succ"]
            if len(succ) > 1:
                return succ[1]
        return None

    weight: Dict[Tuple[int, int], int] = {}

    def add(u, v, w):
        if u is None or v is None:
            return
        weight[(u, v)] = weight.get((u, v), 0) + w

    _FALL_W, _JMP_W, _BACK_W, _TAKEN_W = 24, 20, 8, 2
    for bi, b in enumerate(blocks):
        last = body[b["end"]].strip()
        f = fallthrough(bi)
        if f is not None:
            add(bi, f, _FALL_W)  # keep physical order unless a jump is hot
        if last.startswith("jmp"):
            if b["succ"]:
                add(bi, b["succ"][0], _JMP_W)
        elif last.startswith("j"):
            tgt = b["succ"][0] if b["succ"] else None
            if tgt is not None:
                add(bi, tgt, _BACK_W if tgt <= bi else _TAKEN_W)

    chains: Dict[int, List[int]] = {bi: [bi] for bi in range(nb)}
    chain_of: Dict[int, int] = {bi: bi for bi in range(nb)}
    chosen: Dict[int, int] = {}

    def reaches(a: int, b: int) -> bool:
        seen = set()
        cur: Optional[int] = a
        while cur is not None and cur not in seen:
            if cur == b:
                return True
            seen.add(cur)
            cur = chosen.get(cur)
        return False

    for (u, v), _w in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0])):
        cu, cv = chain_of[u], chain_of[v]
        if cu == cv:
            continue
        if chains[cu][-1] != u or chains[cv][0] != v:
            continue
        if reaches(v, u):  # would close a cycle
            continue
        chosen[u] = v
        chains[cu].extend(chains[cv])
        for x in chains[cv]:
            chain_of[x] = cu
        del chains[cv]
        if len(chains) == 1:
            break

    order: List[int] = []
    remaining = set(chains.keys())
    entry = chain_of[0]
    order.extend(chains[entry])
    remaining.discard(entry)
    while remaining:
        last = order[-1]
        best = None
        f = fallthrough(last)
        cf = chain_of.get(f) if f is not None else None
        if cf in remaining and chains[cf][0] == f:
            best = cf
        if best is None:
            bestw = -1
            for v in blocks[last]["succ"]:
                c = chain_of.get(v)
                if c in remaining and chains[c][0] == v:
                    ww = weight.get((last, v), 0)
                    if ww > bestw:
                        bestw, best = ww, c
        if best is None:
            bestw = -1
            for c in remaining:
                for u in chains[c]:
                    for v in blocks[u]["succ"]:
                        if chain_of.get(v) == c:
                            continue
                        ww = weight.get((u, v), 0)
                        if ww > bestw:
                            bestw, best = ww, c
        if best is None:
            best = next(iter(remaining))
        order.extend(chains[best])
        remaining.discard(best)

    out: List[str] = []
    for idx, bi in enumerate(order):
        b = blocks[bi]
        seg = body[b["start"]:b["end"] + 1]
        if not _is_label(seg[0]):
            out.append(canon[bi] + ":")
        out.extend(seg)
        nxt = order[idx + 1] if idx + 1 < len(order) else None
        f = fallthrough(bi)
        if f is not None and nxt != f:
            out.append(f"    jmp {canon[f]}")
    return out


# -------------------------------------------------- instruction scheduling
_BARRIER_MN = {"call", "leave", "push", "pop", "int3", "syscall", "ud2"}

_FLAGS_W = {"add", "sub", "adc", "sbb", "and", "or", "xor", "cmp", "test",
            "neg", "inc", "dec", "shl", "shr", "sar", "sal", "imul", "mul",
            "div", "idiv", "ucomisd", "comisd", "ucomiss", "comiss",
            "addsd", "subsd", "mulsd", "divsd", "sqrtsd",
            "addss", "subss", "mulss", "divss", "sqrtss"}

_LAT = {"mov": 1, "movzx": 1, "movsx": 1, "movsxd": 1, "movabs": 1, "lea": 1,
        "add": 1, "sub": 1, "and": 1, "or": 1, "xor": 1, "neg": 1, "not": 1,
        "inc": 1, "dec": 1, "cmp": 1, "test": 1,
        "imul": 3, "mul": 3, "div": 20, "idiv": 20, "shl": 1, "shr": 1, "sar": 1,
        "movsd": 1, "movss": 1, "movapd": 1, "movaps": 1, "movq": 1, "movd": 1,
        "addsd": 4, "subsd": 4, "mulsd": 4, "divsd": 15, "sqrtsd": 15,
        "addss": 4, "subss": 4, "mulss": 4, "divss": 14, "sqrtss": 14,
        "cvtsi2sd": 4, "cvttsd2si": 4, "cvtsi2ss": 4, "cvttss2si": 4,
        "cvtsd2ss": 4, "cvtss2sd": 4, "ucomisd": 3, "comisd": 3,
        "ucomiss": 3, "comiss": 3, "call": 5, "nop": 0}


def _is_known_mnemonic(mn: str) -> bool:
    if mn in _READ_WRITE or mn in _READ_WRITE_1 or mn in _MOVES:
        return True
    if mn in _COMPARE or mn in _FLOAT_MOVES or mn in _FLOAT_RW or mn in _FLOAT_CMP:
        return True
    if mn.startswith(("set", "cmov", "cvt")):
        return True
    return mn in ("movabs", "nop")


def _writes_first(mn: str) -> bool:
    return (mn in _MOVES or mn in _FLOAT_MOVES or mn in _READ_WRITE
            or mn in _READ_WRITE_1 or mn in _FLOAT_RW
            or mn.startswith("cmov") or mn == "movabs")


def _flags_read(mn: str) -> bool:
    return mn in ("adc", "sbb") or mn.startswith(("set", "cmov"))


def _mem_key(operand: str):
    m = re.search(r"\[([^\]]*)\]", operand)
    if not m:
        return None
    inner = m.group(1).strip()
    if re.fullmatch(r"rbp-\d+", inner):
        return "s" + inner
    if inner.startswith("rip+"):
        return "r" + inner
    if re.fullmatch(r"rsp\+\d+", inner):
        return "p" + inner
    return "any"


def _mem_conf(a: Set[str], b: Set[str]) -> bool:
    if not a or not b:
        return False
    if "any" in a or "any" in b:
        return True
    return bool(a & b)


def _res(line: str):
    """Resource summary: (reads, writes, flags_r, flags_w, mem_r, mem_w)."""
    mn, args = _split(line)
    if mn is None or mn in _BARRIER_MN or not _is_known_mnemonic(mn):
        return ("bar",)
    r = _read_regs(line)
    w = _write_regs(line)
    fr, fw = _flags_read(mn), mn in _FLAGS_W
    if mn == "lea":
        return (r, w, fr, fw, set(), set())
    argkeys = [_mem_key(a) for a in args]
    mr: Set[str] = set()
    mw: Set[str] = set()
    if _writes_first(mn):
        if argkeys and argkeys[0] is not None:
            mw.add(argkeys[0])
        for k in argkeys[1:]:
            if k is not None:
                mr.add(k)
    else:
        for k in argkeys:
            if k is not None:
                mr.add(k)
    return (r, w, fr, fw, mr, mw)


def _res_conflict(A, B) -> bool:
    if A[0] == "bar" or B[0] == "bar":
        return True
    ra, wa, fa_r, fa_w, ma_r, ma_w = A
    rb, wb, fb_r, fb_w, mb_r, mb_w = B
    if wa & (rb | wb):
        return True
    if ra & wb:
        return True
    if fa_w and (fb_r or fb_w):
        return True
    if fa_r and fb_w:
        return True
    if _mem_conf(ma_w, mb_r | mb_w):
        return True
    if _mem_conf(ma_r, mb_w):
        return True
    return False


def _insn_latency(line: str) -> int:
    mn, args = _split(line)
    if mn is None:
        return 0
    if mn in _LAT:
        base = _LAT[mn]
    elif mn.startswith(("cmov", "set")):
        base = 1
    elif mn.startswith("cvt"):
        base = 4
    else:
        base = 1
    if mn != "lea" and any(_mem_key(a) is not None for a in args):
        base += 3
    return max(base, 0)


def _makespan(order, preds, lat) -> int:
    fin: Dict[int, int] = {}
    span = 0
    for node in order:
        st = 0
        for p in preds[node]:
            fp = fin.get(p)
            if fp is not None and fp > st:
                st = fp
        f = st + lat[node]
        fin[node] = f
        if f > span:
            span = f
    return span


def _local_swap(order, preds, succs, lat):
    """Greedy schedule -> bounded random local search to escape local minima."""
    n = len(order)
    if n < 4:
        return order
    import random
    order = list(order)
    exact = {(u, v) for u in order for v in succs[u]}
    cur = _makespan(order, preds, lat)
    best, bestord = cur, list(order)
    rng = random.Random(0xC0FFEE)
    for _ in range(min(400, 8 * n)):
        i = rng.randrange(n - 1)
        a, b = order[i], order[i + 1]
        if (a, b) in exact or (b, a) in exact:
            continue
        order[i], order[i + 1] = b, a
        new = _makespan(order, preds, lat)
        if new <= cur or rng.random() < 0.03:
            cur = new
            if new < best:
                best, bestord = new, list(order)
        else:
            order[i], order[i + 1] = a, b
    return bestord


def _schedule_block(instrs: List[str]) -> List[str]:
    n = len(instrs)
    if n < 3:
        return instrs
    res = [_res(l) for l in instrs]
    lat = [_insn_latency(l) for l in instrs]
    preds: List[List[int]] = [[] for _ in range(n)]
    succs: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _res_conflict(res[i], res[j]):
                succs[i].append(j)
                preds[j].append(i)
    prio = [0] * n
    for i in range(n - 1, -1, -1):  # critical-path length (edges go forward)
        m = 0
        for j in succs[i]:
            if prio[j] > m:
                m = prio[j]
        prio[i] = lat[i] + m
    import heapq
    indeg = [len(preds[i]) for i in range(n)]
    heap = [(-prio[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for j in succs[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (-prio[j], j))
    if len(order) != n:
        return instrs
    order = _local_swap(order, preds, succs, lat)
    return [instrs[i] for i in order]


def _schedule(body: List[str]) -> List[str]:
    """Hide latency by reordering independent instructions within each block."""
    blocks, _ = _build_blocks(body)
    out: List[str] = []
    for b in blocks:
        seg = body[b["start"]:b["end"] + 1]
        pre: List[str] = []
        while seg and _is_label(seg[0]):
            pre.append(seg.pop(0))
        suf: List[str] = []
        while seg and _ends_block(seg[-1]):
            suf.insert(0, seg.pop())
        if len(seg) >= 3:
            seg = _schedule_block(seg)
        out.extend(pre)
        out.extend(seg)
        out.extend(suf)
    return out


# ---------------------------------------------------- cache-line packing
def _insn_size(line: str) -> int:
    """Estimate the assembled byte length of one instruction.

    The cache-line DP only needs relative sizes, but the estimate follows the
    real encoding rules closely: REX presence, imm8 vs imm32 forms, 1-byte
    ``push``/``pop``, 2-byte short jumps and the zero-extending 32-bit ``mov``
    form chosen by :func:`_shrink_encodings`.
    """
    s = line.strip()
    if not s or s.startswith(".") or _is_label(line):
        return 0
    if s in ("ret", "leave", "nop", "int3"):
        return 1
    if s in ("cqo", "cdqe"):
        return 2
    if s in ("cdq", "cwd"):
        return 1

    mn, args = _split(line)
    if mn is None:
        return 1

    if mn == "call":
        if args and args[0] in _GPR64:
            return 2 + _rex_bytes(args)
        return 5
    if mn.startswith("j"):
        return 2  # rel8 (assembler relaxes only for far targets)
    if mn in ("push", "pop"):
        a = args[0] if args else ""
        return 1 + (1 if a in _HIGH_TOKENS else 0)
    if mn == "movabs":
        return 10

    dst = args[0] if args else ""
    src = args[1] if len(args) > 1 else ""
    mem_dst = dst.startswith("[")
    mem_src = src.startswith("[")
    imm_src = _is_imm(src)
    rex = _rex_bytes(args)

    if mn == "mov":
        if mem_dst:
            if imm_src:
                return 1 + rex + _mem_bytes(dst) + (1 if -128 <= _imm(src) <= 127 else 4)
            return 1 + rex + _mem_bytes(dst)
        if mem_src:
            return 1 + rex + _mem_bytes(src)
        if imm_src:
            v = _imm(src)
            if dst in _GPR64:
                if -2 ** 31 <= v <= 2 ** 31 - 1:
                    return 1 + rex + 1 + 4    # C7 /0, imm32 sign-extended
                return 10                     # movabs
            if dst in _GPR32:
                return 1 + rex + 4            # B8+rd, imm32 (zero-extends)
            if dst in _GPR8:
                return 2 + rex
            return 5
        return 1 + rex + 1

    if mn == "movsxd":
        return 1 + rex + (_mem_bytes(src) if mem_src else 1)
    if mn in ("movzx", "movsx"):
        return 2 + rex + (_mem_bytes(src) if mem_src else 1)

    if mn in _SSE2:
        if mem_dst or mem_src:
            return 3 + (_mem_bytes(dst) if mem_dst else _mem_bytes(src)) + rex
        if dst in _GPR64 or dst in _GPR32 or src in _GPR64 or src in _GPR32:
            return 5  # movq / movd between xmm and gpr
        return 4 + rex

    if mn == "test" and imm_src:
        # TEST has no imm8 form for 32/64-bit operands (only r/m8, imm8).
        if dst in _GPR8:
            return 1 + rex + 1 + 1
        return 1 + rex + 1 + 4

    if mn in _ALU_ALL:
        if imm_src:
            return 1 + rex + 1 + (1 if -128 <= _imm(src) <= 127 else 4)
        if mem_dst or mem_src:
            return 1 + rex + (_mem_bytes(dst) if mem_dst else _mem_bytes(src))
        return 1 + rex + 1

    if mn == "imul":
        if len(args) == 3:
            return 1 + rex + 1 + (1 if -128 <= _imm(args[2]) <= 127 else 4)
        if len(args) == 2 and imm_src:
            return 1 + rex + 1 + (1 if -128 <= _imm(src) <= 127 else 4)
        if mem_dst or mem_src:
            return 2 + rex + (_mem_bytes(dst) if mem_dst else _mem_bytes(src))
        return 2 + rex + 1  # 0F AF + modrm

    if mn == "lea":
        return 1 + rex + _mem_bytes(src)

    if mn in ("neg", "not", "inc", "dec"):
        if mem_dst:
            return 1 + rex + _mem_bytes(dst)
        return 1 + rex + 1

    if mn in ("shl", "shr", "sar", "sal"):
        if src in ("1", "cl"):
            return 1 + rex + 1
        return 1 + rex + 1 + 1

    if mn in ("div", "idiv"):
        return 1 + rex + ( _mem_bytes(dst) if mem_dst else 1)

    if mn in ("sete", "setne", "setl", "setle", "setg", "setge",
              "seta", "setae", "setb", "setbe", "sets", "setns", "seto",
              "setno", "setp", "setnp", "setz", "setnz"):
        return 2 + rex + 1

    if mn.startswith("cmov"):
        return 2 + rex + 1

    # conservative fallback for anything unexpected
    size = 2
    for a in args:
        if _is_imm(a):
            size += 1 if -128 <= _imm(a) <= 127 else 4
        elif a.startswith("["):
            size += _mem_bytes(a)
        else:
            size += 1
    return size


def _estimate_sizes(lines: List[str]) -> List[int]:
    """Per-line sizes with jump displacements relaxed to short or near form.

    ``_insn_size`` assumes the 2-byte rel8 form; once a function grows past
    128 bytes most jumps need rel32 (5 bytes for ``jmp``, 6 for ``jcc``).
    Iterating from the short form converges in a couple of passes.
    """
    size = [_insn_size(l) for l in lines]
    label_at = {_label_name(l): i for i, l in enumerate(lines) if _is_label(l)}
    for _ in range(4):
        offs = [0] * len(lines)
        o = 0
        for i in range(len(lines)):
            offs[i] = o
            o += size[i]
        changed = False
        for i, l in enumerate(lines):
            s = l.strip()
            if not s or _is_label(l):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            mn = parts[0]
            if mn != "jmp" and not (mn.startswith("j") and len(mn) >= 2):
                continue
            t = label_at.get(parts[1])
            if t is None:
                continue
            dist = offs[t] - (offs[i] + size[i])
            if -128 <= dist <= 127:
                new = 2
            else:
                new = 5 if mn == "jmp" else 6
            if new != size[i]:
                size[i] = new
                changed = True
        if not changed:
            break
    return size


def _backedge_targets(lines: List[str]) -> Set[str]:
    """Labels that are targets of a backward jump (natural loop headers).

    Must be called *before* block reordering, where structured code emits every
    loop's back-edge as a jump to an earlier line.
    """
    label_at = {_label_name(l): i for i, l in enumerate(lines) if _is_label(l)}
    targets: Set[str] = set()
    for i, l in enumerate(lines):
        s = l.strip()
        if not s.startswith("j"):
            continue
        parts = s.split()
        if len(parts) > 1 and parts[1] in label_at and label_at[parts[1]] <= i:
            targets.add(parts[1])
    return targets


# Cost model for the packing DP.  Padding occupies I-cache bytes and executes
# as NOPs in hot code, while an instruction straddling a 64-byte boundary costs
# an extra fetch/decode line (and splits the decoder).  The weights balance the
# two, so the DP only pads when the saved line traffic is worth the NOPs.
_ALIGN_PAD_COST = 1       # per padding byte
_ALIGN_LINE_COST = 8      # per 64-byte line touched, scaled by heat
_ALIGN_CROSS_COST = 16    # extra when one instruction spans two lines


def _align_cache_lines(lines: List[str],
                       loop_headers: Optional[Set[str]] = None) -> List[str]:
    """Fast DP choosing per-loop-header alignment so hot code packs into a line.

    A code sequence is laid out linearly; a loop header is an anchor where the
    assembler may insert padding (``.p2align k``).  The DP tracks the offset
    modulo 64 and picks the alignment that minimises
    ``heat * (lines_touched * LINE + crossing * CROSS) + pad * PAD``,
    i.e. it weighs NOP padding (and the 1-byte ``push``/``pop`` prologue
    traffic) against cache-line crossings, in O(n * 64) time.  Padding only at
    headers keeps the NOPs off the steady-state path.

    ``loop_headers`` must be the pre-layout header set (see
    :func:`_backedge_targets`); block reordering can place an exit block before
    its loop, so recomputing from the final order would misclassify it.  Since
    ``.p2align`` is absolute, a header padded to 64 bytes pins the offset model
    exactly; a smaller alignment can leave a constant 32-byte ambiguity that
    only perturbs the cost estimates (the emitted directive still aligns), so
    no separate function-entry padding is needed.
    """
    n = len(lines)
    if n < 8:
        return lines
    size = _estimate_sizes(lines)
    if loop_headers is None:
        loop_headers = _backedge_targets(lines)
    heat = [1] * n
    label_at: Dict[str, int] = {}
    for i, l in enumerate(lines):
        if _is_label(l):
            label_at[_label_name(l)] = i
    # Mark the span between a loop header and each jump back to it as hot.
    # Using the pre-layout header set keeps cold exit blocks (which may end up
    # *before* their loop after reordering) from being weighted as hot.
    for i, l in enumerate(lines):
        s = l.strip()
        if not s.startswith("j"):
            continue
        parts = s.split()
        if len(parts) > 1 and parts[1] in loop_headers and parts[1] in label_at:
            t = label_at[parts[1]]
            for k in range(min(t, i), max(t, i) + 1):
                if heat[k] < 16:
                    heat[k] = 16
    # Only loop headers are worth padding: the padding then sits outside the
    # steady-state loop and never executes per iteration.  Padding other hot
    # labels would put NOPs on the hot path.
    anchor = [not lines[i][:1].isspace() and lines[i].rstrip().endswith(":")
              and _label_name(lines[i]) in loop_headers for i in range(n)]

    def cost(off: int, i: int) -> int:
        sz = size[i]
        if sz <= 0:
            return 0
        touched = (off + sz - 1) // 64 - off // 64 + 1
        c = touched * _ALIGN_LINE_COST
        if touched > 1:
            c += _ALIGN_CROSS_COST
        return c * heat[i]

    INF = float("inf")
    stride = 64
    dp = [INF] * ((n + 1) * stride)
    for x in range(stride):
        dp[n * stride + x] = 0
    pick = [None] * (n * stride)

    for i in range(n - 1, -1, -1):
        for off in range(stride):
            best = INF
            bestc = (0, off, 0)
            cands = [(0, off, 0)]
            if anchor[i]:
                for k in (6, 5, 4, 3, 2):
                    pad = (-off) % (1 << k)
                    if pad:
                        cands.append((pad, (off + pad) % 64, k))
            for pad, o, k in cands:
                c = pad * _ALIGN_PAD_COST + cost(o, i)
                nxt = dp[(i + 1) * stride + (o + size[i]) % 64]
                if c + nxt < best:
                    best = c + nxt
                    bestc = (pad, o, k)
            dp[i * stride + off] = best
            pick[i * stride + off] = bestc

    out: List[str] = []
    off = 0
    for i in range(n):
        pad, o, k = pick[i * stride + off]
        if pad > 0 and k:
            out.append(f"    .p2align {k}")
        out.append(lines[i])
        off = (o + size[i]) % 64
    return out


def optimize_function(lines: List[str], frame: int,
                      slot_types: Optional[Dict[int, str]] = None) -> List[str]:
    if os.environ.get("LEP_NO_OPT"):
        return lines
    sub = None
    for i, line in enumerate(lines[:6]):
        if line.strip().startswith("sub rsp"):
            sub = i
            break
    if sub is None:
        return lines

    head, body = lines[:sub + 1], lines[sub + 1:]

    all_slots: Set[int] = set()
    for line in body:
        for m in SLOT_RE.finditer(line):
            all_slots.add(int(m.group(1)))

    if slot_types is not None:
        # ``all_slots`` holds byte offsets (``[rbp-N]`` -> N) while
        # ``slot_types`` is keyed by slot number (offset = 8 * slot).
        off_types = {8 * slot: typ for slot, typ in slot_types.items()}
        float_slots = {s for s in all_slots
                       if off_types.get(s) in FLOAT_TYPES}
    else:
        float_slots = {int(m.group(1)) for line in body if "xmm" in line
                       for m in SLOT_RE.finditer(line)}

    # A slot must never be rewritten if it is accessed both as a float slot and
    # as a plain 64-bit slot: keep it in memory to stay correct.  Slots whose
    # *address* is taken (``lea reg, [rbp-N]``, e.g. ``ptr(a)``) must also stay
    # in memory: a colour would turn the ``lea`` into an invalid register
    # operand and would make the address unstable.
    force_mem: Set[int] = {int(m.group(1)) for line in body
                           for m in _LEA_SLOT.finditer(line)}
    for line in body:
        if "xmm" in line:
            continue
        for m in SLOT_RE.finditer(line):
            s = int(m.group(1))
            if s in float_slots:
                force_mem.add(s)
    float_slots -= force_mem
    force_mem |= _float_keep_in_memory(body, float_slots)
    float_slots -= force_mem
    if os.environ.get("LEP_NO_FLOAT"):
        force_mem |= float_slots
        float_slots = set()
    int_slots = all_slots - float_slots - force_mem

    if not int_slots and not float_slots:
        return lines

    use_volatile = not os.environ.get("LEP_NO_VOLATILE")
    if int_slots:
        adj, call_live, forbidden = _analyse_alloc(body, int_slots)
        vol = range(len(CALLEE_POOL), len(POOL))
        cal = range(len(CALLEE_POOL))

        def allowed(s: int):
            if not use_volatile or s in call_live:
                order = list(cal)
            else:
                order = list(vol) + list(cal)
            bad = forbidden.get(s, set())
            return [c for c in order if POOL[c] not in bad]

        colours = _colour_restricted(adj, len(POOL), allowed)
    else:
        colours = {}
    fcolours = _colour(_analyse(body, float_slots), len(XMM_POOL)) if float_slots else {}

    reg_of: Dict[int, str] = {s: POOL[c] for s, c in colours.items()}
    reg_of.update({s: XMM_POOL[c] for s, c in fcolours.items()})

    new_body = [_rewrite(l, reg_of) for l in body]
    new_body = _fold_copies(new_body)
    new_body = _store_load_forward(new_body)
    new_body = _peephole(new_body)
    new_body = _fold_float_ops(new_body)
    new_body = _fold_xmm_copies(new_body)
    if not os.environ.get("LEP_NO_SHRINK"):
        new_body = _shrink_encodings(new_body)

    used = set(colours.values())
    gp_used = [r for i, r in enumerate(CALLEE_POOL) if i in used]
    fp_used = [r for i, r in enumerate(XMM_POOL) if i in set(fcolours.values())]
    n = len(gp_used) + len(fp_used)
    if n:
        # New layout (bottom -> top): shadow space, outgoing stack args,
        # saved registers, then the original frame (slots + old shadow space).
        out_bytes = _outgoing_bytes(body)
        new_frame = (frame + out_bytes + 8 * n + 15) & ~15
        head[-1] = f"    sub rsp, {new_frame}"

        saves: List[str] = []
        restores: List[str] = []
        base = frame - 32
        k = 0
        for r in gp_used:
            off = -(base + 8 * (k + 1))
            saves.append(f"    mov [rbp{off}], {r}")
            restores.append(f"    mov {r}, [rbp{off}]")
            k += 1
        for r in fp_used:
            off = -(base + 8 * (k + 1))
            saves.append(f"    movsd [rbp{off}], {r}")
            restores.append(f"    movsd {r}, [rbp{off}]")
            k += 1

        new_body[0:0] = saves
        li = max(i for i, l in enumerate(new_body) if l.strip() == "leave")
        new_body[li:li] = restores

    if not os.environ.get("LEP_NO_SCHED"):
        new_body = _schedule(new_body)

    result = head + new_body
    # Structured code emits loop back-edges as jumps to earlier lines; record
    # the headers before Pettis-Hansen reordering (which may lay an exit block
    # before its loop, making a backward jump no longer imply a loop header).
    loop_headers = _backedge_targets(result)
    if not os.environ.get("LEP_NO_LAYOUT"):
        result = _pettis_hansen(result)
    if not os.environ.get("LEP_NO_ALIGN"):
        # ``.p2align`` is absolute, so aligning the first loop header also
        # fixes the DP's offset model from that point on; no separate entry
        # padding is needed (it would only bloat cold prologues).
        result = _align_cache_lines(result, loop_headers)
    return result
