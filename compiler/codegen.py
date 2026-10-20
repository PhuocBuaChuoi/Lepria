"""x86-64 code generator for Lepria.

Target: Windows x64 ABI, GAS syntax with ``.intel_syntax noprefix``.
The generated ``.s`` is assembled and linked by ``gcc`` into a native ``.exe``.

Supported subset (milestone 1):
  * integer types (int/long/char/bool) and floating types (float/double/ldouble,
    represented internally as IEEE-754 double)
  * string literals, print/println via printf
  * arithmetic + - * / // % **, bitwise, comparisons, and/or/not, ternary
  * if/elif/else, while, for ... in [start, end, step]
  * functions, calls, return
  * casts int()/long()/float()/double()/char()
"""

import struct
from typing import Dict, List, Optional, Set, Tuple

from . import ast_nodes as A
from . import regalloc

INT_TYPES = {"int", "long", "char", "bool"}
FLOAT_TYPES = {"float", "double", "ldouble"}
STR_TYPES = {"string"}


class CodeGenError(Exception):
    def __init__(self, msg: str, line: int = 0, filename: str = "<src>"):
        super().__init__(f"{filename}:{line}: {msg}")
        self.line = line


def align16(n: int) -> int:
    return (n + 15) & ~15


def norm(t: Optional[str]) -> str:
    if t is None:
        return "long"
    if t == "int":
        return "int"
    if t == "char":
        return "char"
    if t == "bool":
        return "bool"
    if t == "long":
        return "long"
    if t in FLOAT_TYPES:
        return "float"
    if t in ("string", "void", "null", "ptr"):
        return t
    if t.startswith("ptr[") and t.endswith("]"):
        # pointer to T: same machine representation as a plain pointer.
        return "ptr"
    if t.startswith("list:"):
        return t
    if t.startswith("weak["):
        return "weak:" + norm(t[5:-1])
    if t.startswith("weak:"):
        return t
    if t.startswith("list["):
        parts = t[5:-1].split(",")
        base = norm(parts[0].strip())
        ndim = len(parts)
        for _ in range(ndim):
            base = "list:" + base
        return base
    if t.startswith("struct:") or t.startswith("class:") or t.startswith("array:"):
        return t
    if t.startswith("array["):
        inner = t[6:-1].split(",")
        name = inner[0].strip()
        dims = [d.strip() for d in inner[1:]]
        return "array:" + name + ":" + ":".join(dims)
    # named value types (structs/classes) resolve at codegen time
    return "named:" + t


def is_float(t: str) -> bool:
    return t in ("float",)


def is_int(t: str) -> bool:
    return t in ("int", "long", "char", "bool")


def is_scalar(t: str) -> bool:
    return t in ("int", "long", "float", "string", "bool", "null", "ptr")


def is_list(t: str) -> bool:
    return t.startswith("list:")


def list_elem(t: str) -> str:
    return t.split(":", 1)[1]


def list_kind(t: str) -> int:
    e = list_elem(t)
    if e == "float":
        return 1
    if e == "string":
        return 2
    return 0


def combine_types(types: List[str]) -> str:
    types = [norm(t) for t in types]
    if not types:
        return "int"
    if all(t.startswith("list:") for t in types) and len(set(types)) == 1:
        return types[0]
    if any(t == "string" for t in types):
        return "string"
    if any(t == "float" for t in types):
        return "float"
    if any(t == "long" for t in types):
        return "long"
    return "int"


def result_num(a: str, b: str) -> str:
    if is_float(a) or is_float(b):
        return "float"
    if a == "long" or b == "long":
        return "long"
    return "int"


# ---------------------------------------------------------------- asm helpers
HELPERS = r"""
lep_floordiv:
    mov r9, rdx
    mov rax, rcx
    cqo
    idiv r9
    test rdx, rdx
    je .Lfd_done
    mov r8, rdx
    xor r8, r9
    test r8, r8
    jns .Lfd_done
    dec rax
.Lfd_done:
    ret

lep_floormod:
    mov r8, rcx
    mov r9, rdx
    mov rax, rcx
    cqo
    idiv r9
    test rdx, rdx
    je .Lfm_zero
    mov r10, rdx
    xor r10, r9
    test r10, r10
    jns .Lfm_done
    dec rax
    imul rax, r9
    mov rdx, r8
    sub rdx, rax
    mov rax, rdx
    ret
.Lfm_zero:
    xor eax, eax
    ret
.Lfm_done:
    mov rax, rdx
    ret

lep_powmod64:
    mov eax, 1
    mov r8, rcx
    mov r9, rdx
.Lpm64_loop:
    test r9, r9
    je .Lpm64_done
    test r9, 1
    jz .Lpm64_skip
    imul rax, r8
.Lpm64_skip:
    imul r8, r8
    shr r9, 1
    jmp .Lpm64_loop
.Lpm64_done:
    ret

lep_powmod32:
    mov eax, 1
    mov r8d, ecx
    mov r9d, edx
.Lpm32_loop:
    test r9d, r9d
    je .Lpm32_done
    test r9d, 1
    jz .Lpm32_skip
    imul eax, r8d
.Lpm32_skip:
    imul r8d, r8d
    shr r9d, 1
    jmp .Lpm32_loop
.Lpm32_done:
    movsxd rax, eax
    ret
"""


# ---------------------------------------------------------------- function gen
class FuncGen:
    def __init__(self, cg: "CodeGen", name: str, params: List[A.Param],
                 ret_type: Optional[str]):
        self.cg = cg
        self.name = name
        self.params = params
        self.ret_type = self.resolve_type(norm(ret_type)) if ret_type else "void"
        self.body: List[str] = []
        self.slots = 0
        self.slot_types: Dict[int, str] = {}
        self.col_cache: Dict[Tuple[int, int], int] = {}
        self.scopes: List[Dict[str, Tuple[int, str]]] = [{}]
        self.loops: List[Tuple[Optional[str], str, str]] = []
        self.ret_label = cg.new_label("ret")
        self.inlining_stack: Set[str] = set()

    def _is_inlineable(self, fdef: A.FuncDef) -> bool:
        if fdef.name in self.inlining_stack or fdef.name == self.name:
            return False
        if len(self.inlining_stack) >= 2:
            return False
        if len(fdef.params) > 6:
            return False

        def has_loop(stmts: List[A.Node]) -> bool:
            for s in stmts:
                if isinstance(s, (A.While, A.ForRange, A.ForEach)):
                    return True
                if isinstance(s, A.If):
                    for _, b in s.branches:
                        if has_loop(b):
                            return True
                    if s.else_body and has_loop(s.else_body):
                        return True
            return False

        if has_loop(fdef.body):
            return False
        if len(fdef.body) > 5:
            return False
        return True

    def gen_inlined_call(self, node: A.Call, fdef: A.FuncDef) -> str:
        arg_slots: List[Tuple[str, str, int]] = []
        for arg, p in zip(node.args, fdef.params):
            ptype = norm(p.type) if p.type else "long"
            t = self.gen_expr(arg)
            self.convert(t, ptype)
            s = self.alloc_slot()
            self.store(ptype, s)
            if ptype.startswith("struct:"):
                self.load(ptype, s)
                self.copy_struct(ptype.split(":", 1)[1])
                self.store(ptype, s)
            arg_slots.append((p.name, ptype, s))

        old_ret = self.ret_label
        old_ret_t = self.ret_type
        inl_ret = self.cg.new_label("inlret")
        self.ret_label = inl_ret
        ret_t = self.resolve_type(norm(fdef.ret_type)) if fdef.ret_type else "void"
        self.ret_type = ret_t
        self.inlining_stack.add(fdef.name)

        self.scopes.append({})
        for pname, ptype, s in arg_slots:
            self.declare(pname, ptype, s)

        self.gen_block(fdef.body)
        self.label(inl_ret)

        self.scopes.pop()
        self.inlining_stack.remove(fdef.name)
        self.ret_label = old_ret
        self.ret_type = old_ret_t
        if ret_t == "int":
            self.wrap("int")
        return ret_t

    # ---- type resolution / layouts
    def resolve_type(self, t: str) -> str:
        if t.startswith("named:"):
            name = t[6:]
            if name in self.cg.layouts:
                kind = self.cg.kinds.get(name, "struct")
                return kind + ":" + name
            return "int"
        return t

    def _resolve_named(self, name: str) -> str:
        if name.startswith(("named:", "struct:", "class:", "list:", "weak:", "array:")):
            return self.resolve_type(name)
        return self.resolve_type(norm(name))

    def field_info(self, type_name: str, field: str) -> Tuple[int, str]:
        fields = self.cg.layouts.get(type_name, [])
        for i, (f, t) in enumerate(fields):
            if f == field:
                return i, self.resolve_type(t)
        self.cg.error(f"type '{type_name}' has no field '{field}'")

    def layout_size(self, type_name: str) -> int:
        return 8 * len(self.cg.layouts.get(type_name, []))

    def copy_struct(self, type_name: str):
        """Copy the struct whose address is in rax; the copy is left in rax."""
        self.emit("mov rcx, rax")
        self.emit(f"mov rdx, {self.layout_size(type_name)}")
        self.emit("call lep_struct_copy")

    def elem_type_of(self, t: str) -> str:
        if t.startswith("list:"):
            return t[5:]
        if t.startswith("array:"):
            parts = t.split(":")
            name = parts[1]
            if name in self.cg.layouts:
                return "struct:" + name
            return self.resolve_type(norm(name))
        return "int"

    def _multidim_base_name(self, index: List[A.Node]) -> str:
        base = index[0]
        if isinstance(base, A.Name):
            return base.id
        if isinstance(base, A.IntLit):
            return str(base.value)
        return "int"

    def _multidim_ndim(self, index: List[A.Node]) -> int:
        if len(index) > 1 and isinstance(index[-1], A.IntLit):
            return int(index[-1].value)
        return 1

    def _multidim_type(self, index: List[A.Node]) -> str:
        t = norm(self._multidim_base_name(index))
        for _ in range(self._multidim_ndim(index)):
            t = "list:" + t
        return t

    def _build_ndim(self, dim_slots: List[int], level: int, elem: str):
        if level == len(dim_slots) - 1:
            kind = list_kind("list:" + elem)
            self.emit(f"mov ecx, {kind}")
            self.emit(f"mov rdx, [rbp{self.off(dim_slots[level]):+d}]")
            self.emit("call lep_list_zeros")
            return

        self.emit("mov ecx, 0")
        self.emit("call lep_list_new")
        oslot = self.alloc_slot()
        self.store("long", oslot)
        idx = self.alloc_slot()
        self.emit("xor rax, rax")
        self.store("long", idx)

        top = self.cg.new_label("ndtop")
        end = self.cg.new_label("ndend")
        self.label(top)
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit(f"cmp rax, [rbp{self.off(dim_slots[level]):+d}]")
        self.emit(f"jge {end}")

        self._build_ndim(dim_slots, level + 1, elem)

        self.emit("mov rdx, rax")
        self.emit(f"mov rcx, [rbp{self.off(oslot):+d}]")
        self.emit("call lep_list_push_i")
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit("add rax, 1")
        self.store("long", idx)
        self.emit(f"jmp {top}")
        self.label(end)
        self.emit(f"mov rax, [rbp{self.off(oslot):+d}]")

    def gen_multidim_ctor(self, node: A.Call) -> str:
        index = node.func.index
        ndim = self._multidim_ndim(index)
        if len(node.args) != ndim:
            self.cg.error(
                f"list[T,{ndim}]() expects {ndim} size argument(s), got {len(node.args)}",
                node.line)
        dim_slots = []
        for a in node.args:
            t = self.gen_expr(a)
            self.convert(t, "long")
            s = self.alloc_slot()
            self.store("long", s)
            dim_slots.append(s)
        elem = norm(self._multidim_base_name(index))
        self._build_ndim(dim_slots, 0, elem)
        return self._multidim_type(index)

    # ---- emission
    def emit(self, line: str):
        self.body.append("    " + line)

    def label(self, name: str):
        self.body.append(name + ":")

    def raw(self, line: str):
        self.body.append(line)

    # ---- slots / scopes
    def alloc_slot(self) -> int:
        self.slots += 1
        return self.slots

    def declare(self, name: str, typ: str, slot: Optional[int] = None) -> int:
        if slot is None:
            slot = self.alloc_slot()
        self.slot_types[slot] = typ
        self.scopes[-1][name] = (slot, typ)
        return slot

    def invalidate_col_cache(self, slot: int):
        for key in [k for k in self.col_cache if k[0] == slot]:
            del self.col_cache[key]

    def class_col_ptr(self, obj: A.Node, idx: int):
        """Load the list pointer of SoA column *idx*, caching it per variable."""
        if isinstance(obj, A.Name):
            found = self.lookup(obj.id)
            if found is not None and not is_float(found[1]):
                key = (found[0], idx)
                cslot = self.col_cache.get(key)
                if cslot is not None:
                    self.emit(f"mov rax, [rbp{self.off(cslot):+d}]")
                    return
                self.gen_expr(obj)
                self.emit(f"mov rax, [rax+{16 + 8 * idx}]")
                cslot = self.alloc_slot()
                self.store("long", cslot)
                self.col_cache[key] = cslot
                return
        self.gen_expr(obj)
        self.emit(f"mov rax, [rax+{16 + 8 * idx}]")

    def populate_col_cache(self, slot: int, class_type: str):
        """Eagerly cache every SoA column pointer of a class binding.

        Called where a class variable is bound (assignment/parameter), i.e.
        usually outside hot loops, so later field accesses become one load.
        """
        if not class_type.startswith("class:"):
            return
        name = class_type.split(":", 1)[1]
        for idx, _ in enumerate(self.cg.layouts.get(name, [])):
            self.emit(f"mov rax, [rbp{self.off(slot):+d}]")
            self.emit(f"mov rax, [rax+{16 + 8 * idx}]")
            cslot = self.alloc_slot()
            self.store("long", cslot)
            self.col_cache[(slot, idx)] = cslot

    def lookup(self, name: str) -> Optional[Tuple[int, str]]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    @staticmethod
    def off(slot: int) -> int:
        return -8 * slot

    def store(self, typ: str, slot: int):
        self.slot_types[slot] = typ
        if is_float(typ):
            self.emit(f"movsd [rbp{self.off(slot):+d}], xmm0")
        else:
            self.emit(f"mov [rbp{self.off(slot):+d}], rax")
            self.invalidate_col_cache(slot)

    def load(self, typ: str, slot: int):
        if is_float(typ):
            self.emit(f"movsd xmm0, [rbp{self.off(slot):+d}]")
        else:
            self.emit(f"mov rax, [rbp{self.off(slot):+d}]")

    def wrap(self, typ: str):
        if typ == "int":
            self.emit("movsxd rax, eax")
        elif typ == "char":
            self.emit("movsx rax, al")
        elif typ == "bool":
            self.emit("and eax, 1")

    def convert(self, src: str, dst: str):
        if is_float(src) and is_int(dst):
            self.emit("cvttsd2si rax, xmm0")
            self.wrap(dst)
        elif is_int(src) and is_float(dst):
            self.emit("cvtsi2sd xmm0, rax")

    # ---- index-check policy (chosen by compiler flags)
    def list_get(self, elem: str) -> str:
        base = ("lep_list_get_f" if elem == "float"
                else "lep_list_get_s" if elem == "string"
                else "lep_list_get_i")
        return base + self.cg.bounds_suffix

    def list_set(self, elem: str) -> str:
        base = ("lep_list_set_f" if elem == "float"
                else "lep_list_set_s" if elem == "string"
                else "lep_list_set_i")
        return base + self.cg.bounds_suffix

    def list_addr(self) -> str:
        return "lep_list_addr" + self.cg.bounds_suffix

    # ---- statements
    def gen_block(self, stmts: List[A.Node]):
        for s in stmts:
            self.gen_stmt(s)

    def gen_stmt(self, node: A.Node):
        if isinstance(node, (A.Import, A.FromImport, A.TypeAlias, A.ClassDecl)):
            return
        if isinstance(node, A.FuncDef):
            self.cg.error(f"nested function '{node.name}' is not supported yet", node.line)
        if isinstance(node, A.VarDecl):
            self.gen_var_decl(node)
        elif isinstance(node, A.Assign):
            self.gen_assign(node)
        elif isinstance(node, A.ExprStmt):
            self.gen_expr(node.expr)
        elif isinstance(node, A.If):
            self.gen_if(node)
        elif isinstance(node, A.While):
            self.gen_while(node)
        elif isinstance(node, A.ForRange):
            self.gen_for_range(node)
        elif isinstance(node, A.ForEach):
            self.gen_for_each(node)
        elif isinstance(node, A.Return):
            if node.value is not None:
                t = self.gen_expr(node.value)
                if self.ret_type not in ("void", None):
                    if t.startswith("struct:") and self.ret_type.startswith("struct:"):
                        self.copy_struct(self.ret_type.split(":", 1)[1])
                    else:
                        self.convert(t, self.ret_type)
            self.emit(f"jmp {self.ret_label}")
        elif isinstance(node, A.Break):
            self.emit(f"jmp {self.find_loop(node.label)[1]}")
        elif isinstance(node, A.Continue):
            self.emit(f"jmp {self.find_loop(node.label)[2]}")
        elif isinstance(node, A.Pass):
            return
        elif isinstance(node, A.Raise):
            self.cg.error("raise/try is not supported yet", node.line)
        else:
            self.cg.error(f"unsupported statement {type(node).__name__}", getattr(node, "line", 0))

    def find_loop(self, label: Optional[str]) -> Tuple[Optional[str], str, str]:
        if not self.loops:
            self.cg.error("break/continue outside of a loop")
        if label is None:
            return self.loops[-1]
        for entry in reversed(self.loops):
            if entry[0] == label:
                return entry
        self.cg.error(f"unknown loop label '{label}'")

    def gen_var_decl(self, node: A.VarDecl):
        declared = self.resolve_type(norm(node.type)) if node.type else None
        if node.value is not None:
            vt = self.gen_expr(node.value)
            typ = declared or vt
            if declared and is_scalar(vt) and is_scalar(declared):
                self.convert(vt, declared)
            if typ.startswith("struct:") and vt.startswith("struct:"):
                self.copy_struct(typ.split(":", 1)[1])
        else:
            typ = declared or "long"
            if typ.startswith("struct:"):
                name = typ.split(":", 1)[1]
                self.emit(f"mov rcx, {self.layout_size(name)}")
                self.emit("call lep_struct_new")
            elif typ.startswith("array:"):
                self.gen_static_array_alloc(typ)
            elif typ.startswith("class:"):
                self.cg.error("class values need an explicit size, e.g. Particle(n)", node.line)
            elif is_float(typ):
                self.emit("xorpd xmm0, xmm0")
            else:
                self.emit("xor rax, rax")
        slot = self.declare(node.name, typ)
        self.store(typ, slot)

    def gen_static_array_alloc(self, typ: str):
        parts = typ.split(":")
        name = parts[1]
        n = int(parts[2]) if len(parts) > 2 else 0
        elem = self.elem_type_of(typ)

        if elem.startswith("struct:"):
            self.emit("mov ecx, 0")
            self.emit("call lep_list_new")
            lslot = self.alloc_slot()
            self.store("long", lslot)

            cnt = self.alloc_slot()
            self.emit(f"mov rax, {n}")
            self.store("long", cnt)
            size = self.layout_size(name)

            top = self.cg.new_label("arrtop")
            end = self.cg.new_label("arrend")
            self.label(top)
            self.emit(f"mov rax, [rbp{self.off(cnt):+d}]")
            self.emit("test rax, rax")
            self.emit(f"jle {end}")
            self.emit(f"mov rcx, {size}")
            self.emit("call lep_struct_new")
            self.emit("mov rdx, rax")
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call lep_list_push_i")
            self.emit(f"mov rax, [rbp{self.off(cnt):+d}]")
            self.emit("sub rax, 1")
            self.store("long", cnt)
            self.emit(f"jmp {top}")
            self.label(end)
            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            return

        kind = 1 if elem == "float" else (2 if elem == "string" else 0)
        self.emit(f"mov ecx, {kind}")
        self.emit(f"mov rdx, {n}")
        self.emit("call lep_list_zeros")

    def gen_assign(self, node: A.Assign):
        if (isinstance(node.target, A.Call)
                and isinstance(node.target.func, A.Name)
                and node.target.func.id == "deref"):
            if len(node.target.args) != 1:
                self.cg.error("deref() expects exactly one argument", node.line)
            self.gen_expr(node.target.args[0])
            pslot = self.alloc_slot()
            self.store("long", pslot)
            if node.op == "=":
                vt = self.gen_expr(node.value)
                self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
                if is_float(vt):
                    self.emit("movsd [rcx], xmm0")
                    return "float"
                if vt == "string":
                    self.emit("mov [rcx], rax")
                    return "string"
                self.emit("mov [rcx], rax")
                return vt
            self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
            self.emit("mov rax, [rcx]")
            cur = self.alloc_slot()
            self.store("long", cur)
            vt = self.gen_expr(node.value)
            vtmp = self.alloc_slot()
            self.store(vt, vtmp)
            res = self.gen_binary_from_slots(node.op[:-1], "long", cur, vt, vtmp)
            self.convert(res, "long")
            self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
            self.emit("mov [rcx], rax")
            return "long"

        if isinstance(node.target, A.Subscript):
            sub = node.target
            ot = self.gen_expr(sub.obj)
            if not (is_list(ot) or ot.startswith("array:")):
                self.cg.error("subscript assignment on a non-list value", node.line)
            lslot = self.alloc_slot()
            self.store("long", lslot)

            for idx_node in sub.index[:-1]:
                it = self.gen_expr(idx_node)
                self.convert(it, "long")
                islot = self.alloc_slot()
                self.store("long", islot)
                ot = self.elem_type_of(ot)
                self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
                self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
                self.emit(f"call {self.list_get('int')}")
                nl = self.alloc_slot()
                self.store("long", nl)
                lslot = nl

            elem = self.elem_type_of(ot)
            it = self.gen_expr(sub.index[-1])
            self.convert(it, "long")
            islot = self.alloc_slot()
            self.store("long", islot)

            if node.op == "=":
                vt = self.gen_expr(node.value)
                self.convert(vt, elem)
                vslot = self.alloc_slot()
                self.store(elem, vslot)
            else:
                # load current element
                self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
                self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
                cur = self.alloc_slot()
                if is_float(elem):
                    self.emit(f"call {self.list_get('float')}")
                elif elem == "string":
                    self.cg.error("compound assignment on string elements is not supported", node.line)
                else:
                    self.emit(f"call {self.list_get('int')}")
                self.store(elem, cur)

                vt = self.gen_expr(node.value)
                vslot = self.alloc_slot()
                self.store(vt, vslot)
                res = self.gen_binary_from_slots(node.op[:-1], elem, cur, vt, vslot)
                self.convert(res, elem)
                vslot2 = self.alloc_slot()
                self.store(elem, vslot2)
                vslot = vslot2

            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
            if self.cg.bounds_mode == "unchecked":
                self.emit("mov r10, [rcx]")
                if is_float(elem):
                    self.emit(f"movsd xmm0, [rbp{self.off(vslot):+d}]")
                    self.emit("movsd [r10+rdx*8], xmm0")
                else:
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit("mov [r10+rdx*8], r8")
            elif self.cg.bounds_mode == "checked":
                slow = self.cg.new_label("slow_set")
                done = self.cg.new_label("done_set")
                self.emit("test rcx, rcx")
                self.emit(f"jz {slow}")
                self.emit("cmp byte ptr [rcx+36], 0")
                self.emit(f"je {slow}")
                self.emit("cmp rdx, [rcx+8]")
                self.emit(f"jae {slow}")
                self.emit("mov r10, [rcx]")
                if is_float(elem):
                    self.emit(f"movsd xmm0, [rbp{self.off(vslot):+d}]")
                    self.emit("movsd [r10+rdx*8], xmm0")
                else:
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit("mov [r10+rdx*8], r8")
                self.emit(f"jmp {done}")
                self.label(slow)
                if is_float(elem):
                    self.emit(f"movsd xmm2, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('float')}")
                elif elem == "string":
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('string')}")
                else:
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('int')}")
                self.label(done)
            else:
                if is_float(elem):
                    self.emit(f"movsd xmm2, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('float')}")
                elif elem == "string":
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('string')}")
                else:
                    self.emit(f"mov r8, [rbp{self.off(vslot):+d}]")
                    self.emit(f"call {self.list_set('int')}")
            return

        if isinstance(node.target, A.Attribute):
            ot = self.resolve_type(self.static_type(node.target.obj))
            if not ot.startswith("struct:"):
                self.cg.error("field assignment on a non-struct value", node.line)
            name = ot.split(":", 1)[1]
            idx, ft0 = self.field_info(name, node.target.attr)
            ft = self.resolve_type(ft0)
            self.gen_expr(node.target.obj)
            pslot = self.alloc_slot()
            self.store("long", pslot)

            if node.op == "=":
                vt = self.gen_expr(node.value)
                if is_scalar(vt) and is_scalar(ft):
                    self.convert(vt, ft)
                vslot = self.alloc_slot()
                self.store(ft, vslot)
            else:
                cur = self.alloc_slot()
                self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
                if is_float(ft):
                    self.emit(f"movsd xmm0, [rcx+{8 * idx}]")
                else:
                    self.emit(f"mov rax, [rcx+{8 * idx}]")
                self.store(ft, cur)
                vt = self.gen_expr(node.value)
                vtmp = self.alloc_slot()
                self.store(vt, vtmp)
                res = self.gen_binary_from_slots(node.op[:-1], ft, cur, vt, vtmp)
                self.convert(res, ft)
                vslot = self.alloc_slot()
                self.store(ft, vslot)

            self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
            if is_float(ft):
                self.emit(f"movsd xmm0, [rbp{self.off(vslot):+d}]")
                self.emit(f"movsd [rcx+{8 * idx}], xmm0")
            else:
                self.emit(f"mov rax, [rbp{self.off(vslot):+d}]")
                self.emit(f"mov [rcx+{8 * idx}], rax")
            return

        if not isinstance(node.target, A.Name):
            self.cg.error("only variable, field and list-element assignment is supported yet", node.line)

        name = node.target.id

        if node.op == "=":
            vt = self.gen_expr(node.value)
            existing = self.lookup(name)
            if existing:
                slot, typ = existing
            else:
                typ = vt
                slot = self.declare(name, typ)
            if typ.startswith("struct:") and vt.startswith("struct:"):
                # struct is a value type: every binding gets its own record
                self.copy_struct(typ.split(":", 1)[1])
            elif is_scalar(vt) and is_scalar(typ):
                self.convert(vt, typ)
            self.store(typ, slot)
            if typ.startswith("class:"):
                self.populate_col_cache(slot, typ)
            return

        # compound assignment:  x op= value  =>  x = x op value
        existing = self.lookup(name)
        if not existing:
            self.cg.error(f"assignment to undeclared variable '{name}'", node.line)
        _, typ = existing
        self.load(typ, existing[0])
        lslot = self.alloc_slot()
        self.store(typ, lslot)

        rt = self.gen_expr(node.value)
        rslot = self.alloc_slot()
        self.store(rt, rslot)

        result_t = self.gen_binary_from_slots(node.op[:-1], typ, lslot, rt, rslot)
        existing = self.lookup(name)
        self.convert(result_t, existing[1])
        self.store(existing[1], existing[0])

    def gen_if(self, node: A.If):
        if self.try_branchless_if(node):
            return
        end = self.cg.new_label("ifend")
        num_branches = len(node.branches)
        for idx, (cond, body) in enumerate(node.branches):
            nxt = self.cg.new_label("ifnext")
            self.gen_cond(cond)
            self.emit("test rax, rax")
            self.emit(f"je {nxt}")
            self.gen_block(body)
            is_last = (idx == num_branches - 1) and (node.else_body is None)
            if not is_last:
                self.emit(f"jmp {end}")
            self.label(nxt)
        if node.else_body is not None:
            self.gen_block(node.else_body)
        self.label(end)

    # ---- if-conversion:  if c: y = A  else: y = B   ->   y = c ? A : B
    def _is_pure(self, node: A.Node) -> bool:
        """True if evaluating *node* is side-effect free and cannot trap, so it
        is safe to evaluate speculatively for a branchless select."""
        if isinstance(node, A.Call):
            return False
        if isinstance(node, A.Subscript):
            return False  # a list read may abort (bounds), so do not speculate
        if isinstance(node, (A.IntLit, A.FloatLit, A.CharLit, A.StringLit,
                             A.BoolLit, A.NullLit, A.Name)):
            return True
        if isinstance(node, A.BinOp):
            if node.op in ("//", "%"):
                return False
            if node.op == "/" and not (is_float(self.static_type(node.left))
                                       or is_float(self.static_type(node.right))):
                return False  # integer division can trap on /0
            return self._is_pure(node.left) and self._is_pure(node.right)
        if isinstance(node, A.UnaryOp):
            return self._is_pure(node.operand)
        if isinstance(node, A.Compare):
            return self._is_pure(node.left) and self._is_pure(node.right)
        if isinstance(node, A.BoolOp):
            return self._is_pure(node.left) and self._is_pure(node.right)
        if isinstance(node, A.Ternary):
            return (self._is_pure(node.cond) and self._is_pure(node.then)
                    and self._is_pure(node.other))
        if isinstance(node, A.Attribute):
            return self._is_pure(node.obj)
        if isinstance(node, A.WeakExpr):
            return self._is_pure(node.value)
        return False

    def _common_scalar(self, a: str, b: str) -> str:
        if is_float(a) or is_float(b):
            return "float"
        if a == "string" or b == "string":
            return "string"
        if a == "ptr" or b == "ptr":
            return "ptr"
        if a == "long" or b == "long":
            return "long"
        return "int"

    def try_branchless_if(self, node: A.If) -> bool:
        if len(node.branches) != 1:
            return False
        then_body = node.branches[0][1]
        if len(then_body) != 1:
            return False
        a = then_body[0]
        if not (isinstance(a, A.Assign) and a.op == "=" and isinstance(a.target, A.Name)):
            return False
        name = a.target.id

        if node.else_body is not None:
            if len(node.else_body) != 1:
                return False
            b = node.else_body[0]
            if not (isinstance(b, A.Assign) and b.op == "=" and isinstance(b.target, A.Name)):
                return False
            if a.target.id != b.target.id:
                return False
            if not (self._is_pure(a.value) and self._is_pure(b.value)):
                return False

            t1 = self.static_type(a.value)
            t2 = self.static_type(b.value)
            existing = self.lookup(name)
            common = existing[1] if existing else self._common_scalar(t1, t2)
            if not is_scalar(common):
                return False

            # condition -> 0/1
            self.gen_cond(node.branches[0][0])
            cslot = self.alloc_slot()
            self.store("long", cslot)

            t1 = self.gen_expr(a.value)
            self.convert(t1, common)
            if is_float(common):
                self.emit("movq rax, xmm0")
            s1 = self.alloc_slot()
            self.store("long", s1)

            t2 = self.gen_expr(b.value)
            self.convert(t2, common)
            if is_float(common):
                self.emit("movq rax, xmm0")
            s2 = self.alloc_slot()
            self.store("long", s2)

            self.emit(f"mov rax, [rbp{self.off(s1):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(cslot):+d}]")
            self.emit("test rcx, rcx")
            self.emit(f"cmove rax, [rbp{self.off(s2):+d}]")
            if is_float(common):
                self.emit("movq xmm0, rax")

            if existing:
                slot = existing[0]
            else:
                slot = self.declare(name, common)
            self.store(common, slot)
            return True
        else:
            # Single if without else: if cond: x = new_val (x retains current value if false)
            existing = self.lookup(name)
            if not existing:
                return False
            if not self._is_pure(a.value):
                return False
            common = existing[1]
            if not is_scalar(common):
                return False

            self.gen_cond(node.branches[0][0])
            cslot = self.alloc_slot()
            self.store("long", cslot)

            t1 = self.gen_expr(a.value)
            self.convert(t1, common)
            if is_float(common):
                self.emit("movq rax, xmm0")
            s1 = self.alloc_slot()
            self.store("long", s1)

            # Load existing value as fallback
            self.load(common, existing[0])
            if is_float(common):
                self.emit("movq rax, xmm0")
            s2 = self.alloc_slot()
            self.store("long", s2)

            self.emit(f"mov rax, [rbp{self.off(s1):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(cslot):+d}]")
            self.emit("test rcx, rcx")
            self.emit(f"cmove rax, [rbp{self.off(s2):+d}]")
            if is_float(common):
                self.emit("movq xmm0, rax")

            self.store(common, existing[0])
            return True

    def gen_while(self, node: A.While):
        top = self.cg.new_label("wtop")
        end = self.cg.new_label("wend")
        self.label(top)
        self.gen_cond(node.cond)
        self.emit("test rax, rax")
        self.emit(f"je {end}")
        self.loops.append((node.label, end, top))
        self.gen_block(node.body)
        self.loops.pop()
        self.emit(f"jmp {top}")
        self.label(end)

    # ---- AVX2 vectorization of contiguous loops --------------------------
    _VEC_PTRS = ("r8", "r9", "rcx", "rdx")

    def _vec_base_key(self, node: A.Node):
        """Stable identity of a list-valued base (``a`` or ``obj.field``)."""
        if isinstance(node, A.Name):
            return ("n", node.id)
        if isinstance(node, A.Attribute) and isinstance(node.obj, A.Name):
            return ("a", node.obj.id, node.attr)
        return None

    def _vec_base_elem(self, node: A.Node) -> Optional[str]:
        """Element class (``int``/``long``/``float``) of a list-valued base."""
        if self._vec_base_key(node) is None:
            return None
        t = self.resolve_type(self.static_type(node))
        if not is_list(t):
            return None
        e = list_elem(t)
        if e == "int":
            return "int"
        if e == "long":
            return "long"
        if e == "float":
            return "float"
        return None

    def _vec_sub_key(self, node: A.Node, var: str):
        """Key of ``base[i]`` when the subscript uses the loop variable."""
        if not (isinstance(node, A.Subscript) and len(node.index) == 1):
            return None
        idx = node.index[0]
        if not (isinstance(idx, A.Name) and idx.id == var):
            return None
        return self._vec_base_key(node.obj)

    def _try_vector_for(self, node: A.ForRange) -> bool:
        """Emit an AVX2 version of ``for i in [0, n]`` when the body is a
        contiguous elementwise or reduction update.

        Every access uses the same index, so all source loads precede the
        destination store: two bases that denote the same list are still
        lane-correct (in-place ``a[i] = a[i] + b[i]`` is fine), hence no alias
        proof is required.  A runtime length guard checks ``n`` against every
        list size and falls back to the scalar loop otherwise, which preserves
        bounds-check/abort behaviour.  ``int`` results are wrapped to 32 bits;
        reductions are vectorized only for integer data into a ``long``
        accumulator, keeping float rounding and per-step int wraps scalar-exact.
        """
        if not self.cg.avx2:
            return False
        if not (isinstance(node.start, A.IntLit) and node.start.value == 0):
            return False
        if node.step is not None and not (
                isinstance(node.step, A.IntLit) and node.step.value == 1):
            return False
        if not is_int(self.static_type(node.end)):
            return False
        if not node.body:
            return False
        stmt0 = node.body[0]

        plans = []
        elem = None
        acc = None
        val = None
        vop = None

        def first_base_elem(e) -> Optional[str]:
            k = self._vec_sub_key(e, node.var)
            if k is not None:
                return self._vec_base_elem(e.obj)
            if isinstance(e, A.BinOp):
                return first_base_elem(e.left) or first_base_elem(e.right)
            return None

        if (len(node.body) == 1 and isinstance(stmt0, A.Assign)
                and stmt0.op in ("+=", "-=")
                and isinstance(stmt0.target, A.Name)):
            got = self.lookup(stmt0.target.id)
            if got is None or got[1] != "long":
                # ``int`` accumulators wrap every step and float addition is
                # not associative: both would change results.
                return False
            elem = first_base_elem(stmt0.value)
            if elem not in ("int", "long"):
                return False
            acc, vop, val = stmt0.target.id, stmt0.op[0], stmt0.value
        else:
            for stmt in node.body:
                if not isinstance(stmt, A.Assign):
                    return False
                tk = self._vec_sub_key(stmt.target, node.var)
                if tk is None:
                    return False
                e = self._vec_base_elem(stmt.target.obj)
                if e is None or (elem is not None and e != elem):
                    return False
                elem = e
                if stmt.op == "=":
                    sv = None
                elif stmt.op in ("+=", "-=", "*=", "/="):
                    sv = stmt.op[0]
                else:
                    return False
                if elem in ("int", "long") and sv in ("*", "/"):
                    return False    # AVX2 lacks 64-bit vpmullq; int / is float
                plans.append((tk, stmt.target.obj, sv, stmt.value))

        bases: Dict[tuple, A.Node] = {}
        for tk, tbase, _, _ in plans:
            bases.setdefault(tk, tbase)

        def check(e) -> bool:
            k = self._vec_sub_key(e, node.var)
            if k is not None:
                if self._vec_base_elem(e.obj) != elem:
                    return False
                bases.setdefault(k, e.obj)
                return True
            if isinstance(e, A.IntLit):
                return True
            if isinstance(e, A.FloatLit):
                return elem == "float"
            if isinstance(e, A.Name):
                if e.id == acc or e.id == node.var:
                    # The loop index is not a lane-invariant scalar.
                    return False
                got = self.lookup(e.id)
                if got is None:
                    return False
                if elem == "float":
                    return is_float(got[1])
                return got[1] in ("int", "long", "char", "bool")
            if isinstance(e, A.BinOp):
                if e.op not in ("+", "-", "*", "/"):
                    return False
                if elem in ("int", "long") and e.op in ("*", "/"):
                    return False
                return check(e.left) and check(e.right)
            return False

        def no_int_ops(e) -> bool:
            # An int-typed operation nested in a long/float expression wraps
            # in the scalar path before being widened; SIMD would not.
            if isinstance(e, A.BinOp):
                if self.static_type(e) == "int":
                    return False
                return no_int_ops(e.left) and no_int_ops(e.right)
            return True

        wrap_red = False
        if acc is not None:
            ty = self.static_type(val)
            if elem == "int":
                if ty not in ("int", "long"):
                    return False
                if ty == "long" and not no_int_ops(val):
                    return False
                wrap_red = ty == "int"
            else:
                if ty != "long" or not no_int_ops(val):
                    return False
            if not check(val):
                return False
        else:
            for _, _, _, v in plans:
                ty = self.static_type(v)
                if elem == "float":
                    if ty != "float" or not no_int_ops(v):
                        return False
                elif elem == "long":
                    if ty != "long" or not no_int_ops(v):
                        return False
                else:
                    if ty != "int":
                        return False
                if not check(v):
                    return False
        if len(bases) > len(self._VEC_PTRS):
            return False

        def need(e) -> int:
            if (self._vec_sub_key(e, node.var) is not None
                    or isinstance(e, (A.IntLit, A.FloatLit, A.Name))):
                return 1
            return max(need(e.left), 1 + need(e.right))

        if acc is not None:
            if need(val) + 1 > 5:
                return False
        else:
            for _, _, sv, v in plans:
                if need(v) + (1 if sv is not None else 0) > 5:
                    return False

        is_flt = elem == "float"
        mov = "vmovupd" if is_flt else "vmovdqu"

        et = self.gen_expr(node.end)
        self.convert(et, "long")
        nslot = self.alloc_slot()
        self.store("long", nslot)

        islot = self.alloc_slot()
        old = self.lookup(node.var)
        self.declare(node.var, "long", islot)
        self.emit("xor eax, eax")
        self.emit(f"mov [rbp{self.off(islot):+d}], rax")

        # An empty/negative range performs no list access: skip the size and
        # handle setup entirely so dead/null handles behave exactly as in the
        # scalar loop.
        scalar = self.cg.new_label("vecscalar")
        self.emit(f"mov rax, [rbp{self.off(nslot):+d}]")
        self.emit("test rax, rax")
        self.emit(f"jle {scalar}")

        entries = []
        for key, base in bases.items():
            if key[0] == "n":
                h = self.lookup(key[1])[0]
            else:
                self.gen_expr(base)
                h = self.alloc_slot()
                self.store("long", h)
            entries.append((key, h))

        sizes: List[int] = []
        for _, h in entries:
            self.emit(f"mov rcx, [rbp{self.off(h):+d}]")
            self.emit("call lep_list_size")
            s = self.alloc_slot()
            self.store("long", s)
            sizes.append(s)

        for s in sizes:
            self.emit(f"mov rax, [rbp{self.off(nslot):+d}]")
            self.emit(f"cmp rax, [rbp{self.off(s):+d}]")
            self.emit(f"jg {scalar}")

        ptr: Dict[tuple, str] = {}
        for (key, h), reg in zip(entries, self._VEC_PTRS):
            self.emit(f"mov rax, [rbp{self.off(h):+d}]")
            self.emit(f"mov {reg}, [rax]")
            ptr[key] = reg

        def opmn(o: str) -> str:
            if is_flt:
                return {"+": "vaddpd", "-": "vsubpd",
                        "*": "vmulpd", "/": "vdivpd"}[o]
            return {"+": "vpaddq", "-": "vpsubq"}[o]

        def leaf(e, r: int):
            if isinstance(e, A.IntLit):
                if is_flt:
                    self.emit(f"mov rax, {e.value}")
                    self.emit(f"cvtsi2sd xmm{r}, rax")
                    self.emit(f"vbroadcastsd ymm{r}, xmm{r}")
                elif e.value == 0:
                    self.emit(f"vpxor ymm{r}, ymm{r}, ymm{r}")
                else:
                    self.emit(f"mov rax, {e.value}")
                    self.emit(f"vmovq xmm{r}, rax")
                    self.emit(f"vpbroadcastq ymm{r}, xmm{r}")
            elif isinstance(e, A.FloatLit):
                bits = struct.unpack("<Q", struct.pack("<d", e.value))[0]
                self.emit(f"movabs rax, 0x{bits:016x}")
                self.emit(f"vmovq xmm{r}, rax")
                self.emit(f"vbroadcastsd ymm{r}, xmm{r}")
            else:
                slot = self.lookup(e.id)[0]
                if is_flt:
                    self.emit(f"vbroadcastsd ymm{r}, [rbp{self.off(slot):+d}]")
                else:
                    self.emit(f"vmovq xmm{r}, [rbp{self.off(slot):+d}]")
                    self.emit(f"vpbroadcastq ymm{r}, xmm{r}")

        # For int data one lane register holds the vpermd index constant;
        # wrap32 then needs only two instructions.
        if elem == "int":
            if acc is not None:
                idxreg, base_free = 4, [0, 1, 2, 3]
            else:
                idxreg, base_free = 5, [0, 1, 2, 3, 4]
        else:
            idxreg = None
            base_free = list(range(5)) if acc is not None else list(range(6))
        free = list(base_free)

        def newreg() -> int:
            return free.pop(0)

        def rel(r: int):
            free.append(r)
            free.sort()

        def wrap32(r: int):
            # Gather the low dword of every qword lane, then sign-extend it
            # (AVX2 lacks an arithmetic qword shift).  Matches scalar movsxd.
            s = free[0]
            self.emit(f"vpermd ymm{s}, ymm{idxreg}, ymm{r}")
            self.emit(f"vpmovsxdq ymm{r}, xmm{s}")

        def emit_expr(e) -> int:
            k = self._vec_sub_key(e, node.var)
            if k is not None:
                r = newreg()
                self.emit(f"{mov} ymm{r}, [{ptr[k]}+r10*8]")
                return r
            if isinstance(e, (A.IntLit, A.FloatLit, A.Name)):
                r = newreg()
                leaf(e, r)
                return r
            l = emit_expr(e.left)
            r = emit_expr(e.right)
            self.emit(f"{opmn(e.op)} ymm{l}, ymm{l}, ymm{r}")
            rel(r)
            return l

        self.emit("xor r10d, r10d")
        self.emit(f"mov r11, [rbp{self.off(nslot):+d}]")
        if idxreg is not None:
            lbl = self.cg.intern_data(struct.pack("<8i", 0, 2, 4, 6, 0, 0, 0, 0))
            self.emit(f"vmovdqu ymm{idxreg}, [rip+{lbl}]")
        if acc is not None:
            self.emit("vxorpd ymm5, ymm5, ymm5" if is_flt
                      else "vpxor ymm5, ymm5, ymm5")

        top = self.cg.new_label("vec")
        done = self.cg.new_label("vecdone")
        self.label(top)
        self.emit("lea rax, [r10+4]")
        self.emit("cmp rax, r11")
        self.emit(f"jg {done}")
        if acc is not None:
            r = emit_expr(val)
            if wrap_red:
                wrap32(r)
            if vop == "-":
                self.emit(f"{'vsubpd' if is_flt else 'vpsubq'} ymm5, ymm5, ymm{r}")
            else:
                self.emit(f"{'vaddpd' if is_flt else 'vpaddq'} ymm5, ymm5, ymm{r}")
            rel(r)
        else:
            for tk, _, sv, v in plans:
                r = emit_expr(v)
                if sv is not None:
                    t = newreg()
                    self.emit(f"{mov} ymm{t}, [{ptr[tk]}+r10*8]")
                    self.emit(f"{opmn(sv)} ymm{t}, ymm{t}, ymm{r}")
                    rel(r)
                    r = t
                if elem == "int":
                    wrap32(r)
                self.emit(f"{mov} [{ptr[tk]}+r10*8], ymm{r}")
                rel(r)
        self.emit("add r10, 4")
        self.emit(f"jmp {top}")
        self.label(done)

        if acc is not None:
            if is_flt:
                self.emit("vextractf128 xmm1, ymm5, 1")
                self.emit("vaddpd xmm0, xmm5, xmm1")
                self.emit("vunpckhpd xmm1, xmm0, xmm0")
                self.emit("vaddsd xmm0, xmm0, xmm1")
                aslot = self.lookup(acc)[0]
                self.emit(f"movsd xmm1, [rbp{self.off(aslot):+d}]")
                self.emit("vaddsd xmm0, xmm0, xmm1")
                self.emit(f"movsd [rbp{self.off(aslot):+d}], xmm0")
            else:
                self.emit("vextracti128 xmm1, ymm5, 1")
                self.emit("vpaddq xmm0, xmm5, xmm1")
                self.emit("vpunpckhqdq xmm1, xmm0, xmm0")
                self.emit("vpaddq xmm0, xmm0, xmm1")
                self.emit("vmovq rax, xmm0")
                aslot = self.lookup(acc)[0]
                self.emit(f"add [rbp{self.off(aslot):+d}], rax")

        self.emit("vzeroupper")
        self.emit(f"mov [rbp{self.off(islot):+d}], r10")

        self.label(scalar)
        top2 = self.cg.new_label("ftop")
        inc2 = self.cg.new_label("finc")
        end2 = self.cg.new_label("fend")
        self.label(top2)
        self.emit(f"mov rax, [rbp{self.off(islot):+d}]")
        self.emit(f"cmp rax, [rbp{self.off(nslot):+d}]")
        self.emit(f"jge {end2}")
        self.loops.append((node.label, end2, inc2))
        self.gen_block(node.body)
        self.loops.pop()
        self.label(inc2)
        self.emit(f"mov rax, [rbp{self.off(islot):+d}]")
        self.emit("add rax, 1")
        self.emit(f"mov [rbp{self.off(islot):+d}], rax")
        self.emit(f"jmp {top2}")
        self.label(end2)

        if old is not None:
            self.declare(node.var, old[1], old[0])
        return True

    def _const_step(self, node_step) -> Optional[int]:
        if node_step is None:
            return 1
        if isinstance(node_step, A.IntLit):
            return node_step.value
        if (isinstance(node_step, A.UnaryOp) and node_step.op == "-"
                and isinstance(node_step.operand, A.IntLit)):
            return -node_step.operand.value
        return None

    def gen_for_range(self, node: A.ForRange):
        if self._try_vector_for(node):
            return
        st = self.gen_expr(node.start)
        islot = self.alloc_slot()
        self.store(st if is_int(st) or is_float(st) else "long", islot)

        et = self.gen_expr(node.end)
        eslot = self.alloc_slot()
        self.store(et if is_int(et) or is_float(et) else "long", eslot)

        c_step = self._const_step(node.step)

        # loop variable aliases the counter slot
        old = self.lookup(node.var)
        self.declare(node.var, "long", islot)

        top = self.cg.new_label("ftop")
        body_l = self.cg.new_label("fbody")
        inc = self.cg.new_label("finc")
        end = self.cg.new_label("fend")

        if c_step is not None:
            # Specialized constant step loop (no sslot, direct branch)
            self.label(top)
            self.emit(f"mov rax, [rbp{self.off(islot):+d}]")
            self.emit(f"cmp rax, [rbp{self.off(eslot):+d}]")
            if c_step > 0:
                self.emit(f"jge {end}")
            else:
                self.emit(f"jle {end}")

            self.loops.append((node.label, end, inc))
            self.gen_block(node.body)
            self.loops.pop()

            self.label(inc)
            if c_step == 1:
                self.emit(f"add qword ptr [rbp{self.off(islot):+d}], 1")
            elif c_step == -1:
                self.emit(f"sub qword ptr [rbp{self.off(islot):+d}], 1")
            elif c_step > 0:
                self.emit(f"add qword ptr [rbp{self.off(islot):+d}], {c_step}")
            else:
                self.emit(f"sub qword ptr [rbp{self.off(islot):+d}], {-c_step}")
            self.emit(f"jmp {top}")
            self.label(end)
        else:
            # Dynamic step
            pt = self.gen_expr(node.step)
            sslot = self.alloc_slot()
            self.store(pt if is_int(pt) or is_float(pt) else "long", sslot)

            self.label(top)
            self.emit(f"mov rax, [rbp{self.off(islot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(sslot):+d}]")
            self.emit("test rcx, rcx")
            self.emit(f"jg {top}_pos")
            self.emit(f"cmp rax, [rbp{self.off(eslot):+d}]")
            self.emit(f"jle {end}")
            self.emit(f"jmp {body_l}")
            self.label(f"{top}_pos")
            self.emit(f"cmp rax, [rbp{self.off(eslot):+d}]")
            self.emit(f"jge {end}")

            self.label(body_l)
            self.loops.append((node.label, end, inc))
            self.gen_block(node.body)
            self.loops.pop()

            self.label(inc)
            self.emit(f"mov rax, [rbp{self.off(sslot):+d}]")
            self.emit(f"add [rbp{self.off(islot):+d}], rax")
            self.emit(f"jmp {top}")
            self.label(end)

        if old is not None:
            self.declare(node.var, old[1], old[0])

    def gen_for_each(self, node: A.ForEach):
        if "," in node.var:
            names = [n.strip() for n in node.var.split(",")]
            if not (isinstance(node.iterable, A.Call)
                    and isinstance(node.iterable.func, A.Attribute)
                    and node.iterable.func.attr == "pairs"):
                self.cg.error("pair iteration is only supported for .pairs()", node.line)
            self.gen_for_pairs(node, names)
            return

        old = self.lookup(node.var)

        # Literal list: unroll at compile time (avoids runtime overhead).
        if isinstance(node.iterable, A.ListLit):
            for item in node.iterable.items:
                t = self.gen_expr(item)
                slot = self.declare(node.var, t)
                self.store(t, slot)
                self.gen_block(node.body)
            if old is not None:
                self.declare(node.var, old[1], old[0])
            return

        it = self.gen_expr(node.iterable)
        if not is_list(it):
            self.cg.error("'for' iterable must be a list so far", node.line)
        elem = list_elem(it)

        lslot = self.alloc_slot()
        self.store("long", lslot)
        self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
        self.emit("call lep_list_size")
        sslot = self.alloc_slot()
        self.store("long", sslot)

        idx = self.alloc_slot()
        self.emit("xor rax, rax")
        self.store("long", idx)

        vslot = self.declare(node.var, elem)
        top = self.cg.new_label("fetop")
        inc = self.cg.new_label("feinc")
        end = self.cg.new_label("feend")

        self.label(top)
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit(f"cmp rax, [rbp{self.off(sslot):+d}]")
        self.emit(f"jge {end}")
        self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
        self.emit(f"mov rdx, [rbp{self.off(idx):+d}]")
        if is_float(elem):
            self.emit(f"call {self.list_get('float')}")
        elif elem == "string":
            self.emit(f"call {self.list_get('string')}")
        else:
            self.emit(f"call {self.list_get('int')}")
        self.store(elem, vslot)

        self.loops.append((None, end, inc))
        self.gen_block(node.body)
        self.loops.pop()

        self.label(inc)
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit("add rax, 1")
        self.store("long", idx)
        self.emit(f"jmp {top}")
        self.label(end)

        if old is not None:
            self.declare(node.var, old[1], old[0])

    def gen_for_pairs(self, node: A.ForEach, names: List[str]):
        it = self.gen_expr(node.iterable.func.obj)
        if not is_list(it):
            self.cg.error("'for ... in .pairs()' needs a list", node.line)
        elem = list_elem(it)

        old1 = self.lookup(names[0])
        old2 = self.lookup(names[1])

        lslot = self.alloc_slot()
        self.store("long", lslot)
        self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
        self.emit("call lep_list_size")
        sslot = self.alloc_slot()
        self.store("long", sslot)

        idx = self.alloc_slot()
        self.emit("xor rax, rax")
        self.store("long", idx)

        islot = self.declare(names[0], "long")
        vslot = self.declare(names[1], elem)

        top = self.cg.new_label("pstop")
        inc = self.cg.new_label("psinc")
        end = self.cg.new_label("psend")

        self.label(top)
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit(f"cmp rax, [rbp{self.off(sslot):+d}]")
        self.emit(f"jge {end}")
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.store("long", islot)
        self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
        self.emit(f"mov rdx, [rbp{self.off(idx):+d}]")
        if is_float(elem):
            self.emit(f"call {self.list_get('float')}")
        elif elem == "string":
            self.emit(f"call {self.list_get('string')}")
        else:
            self.emit(f"call {self.list_get('int')}")
        self.store(elem, vslot)

        self.loops.append((None, end, inc))
        self.gen_block(node.body)
        self.loops.pop()

        self.label(inc)
        self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
        self.emit("add rax, 1")
        self.store("long", idx)
        self.emit(f"jmp {top}")
        self.label(end)

        if old1 is not None:
            self.declare(names[0], old1[1], old1[0])
        if old2 is not None:
            self.declare(names[1], old2[1], old2[0])

    # ---- conditions
    def gen_cond(self, expr: A.Node):
        t = self.gen_expr(expr)
        if is_float(t):
            self.emit("xorpd xmm1, xmm1")
            self.emit("ucomisd xmm0, xmm1")
            self.emit("setne al")
            self.emit("movzx rax, al")

    # ---- static type (no code emitted)
    def static_type(self, node: A.Node) -> str:
        if isinstance(node, A.IntLit):
            return node.type if node.type in ("int", "long") else "int"
        if isinstance(node, A.FloatLit):
            return "float"
        if isinstance(node, A.CharLit):
            return "char"
        if isinstance(node, A.StringLit):
            return "string"
        if isinstance(node, A.BoolLit):
            return "bool"
        if isinstance(node, A.NullLit):
            return "null"
        if isinstance(node, A.WeakExpr):
            return "weak:" + self.static_type(node.value)
        if isinstance(node, A.Name):
            found = self.lookup(node.id)
            return found[1] if found else "long"
        if isinstance(node, A.BinOp):
            if node.op == "/":
                return "float"
            if node.op == "*" and isinstance(node.left, A.ListLit):
                return "list:" + combine_types([self.static_type(i) for i in node.left.items])
            return result_num(self.static_type(node.left), self.static_type(node.right))
        if isinstance(node, A.UnaryOp):
            return "bool" if node.op == "not" else self.static_type(node.operand)
        if isinstance(node, A.Compare):
            return "bool"
        if isinstance(node, A.BoolOp):
            return "bool"
        if isinstance(node, A.Ternary):
            return self.static_type(node.then)
        if isinstance(node, A.Subscript):
            ot = self.resolve_type(self.static_type(node.obj))
            for _ in node.index:
                if is_list(ot) or ot.startswith("array:"):
                    ot = self.elem_type_of(ot)
                else:
                    return "int"
            return ot
        if isinstance(node, A.ListLit):
            return "list:" + combine_types([self.static_type(i) for i in node.items])
        if isinstance(node, A.Attribute):
            ot = self.resolve_type(self.static_type(node.obj))
            if ot.startswith("struct:"):
                name = ot.split(":", 1)[1]
                return self.field_info(name, node.attr)[1]
            if ot.startswith("class:"):
                name = ot.split(":", 1)[1]
                return "list:" + self.field_info(name, node.attr)[1]
            if isinstance(node.obj, A.Name) and node.obj.id in ("m", "math"):
                return "float"
            return "int"
        if isinstance(node, A.Call):
            if (isinstance(node.func, A.Subscript)
                    and isinstance(node.func.obj, A.Name)
                    and node.func.obj.id == "list"):
                return self._multidim_type(node.func.index)
            if isinstance(node.func, A.Name):
                if node.func.id in ("int", "long", "float", "double", "char", "bool"):
                    return norm(node.func.id)
                if node.func.id == "ptr":
                    return "ptr"
                if node.func.id == "deref":
                    return "long"
                if node.func.id in ("memset",):
                    return "void"
                if node.func.id == "cpu_features":
                    return "long"
                if node.func.id in ("print", "println", "write", "eprint"):
                    return "void"
                if node.func.id in self.cg.layouts:
                    kind = self.cg.kinds.get(node.func.id, "struct")
                    return kind + ":" + node.func.id
                fdef = self.cg.functions.get(node.func.id)
                if fdef is not None:
                    return norm(fdef.ret_type) if fdef.ret_type else "void"
            if isinstance(node.func, A.Attribute):
                objt = self.resolve_type(self.static_type(node.func.obj))
                if objt.startswith("class:"):
                    name = objt.split(":", 1)[1]
                    if node.func.attr in ("push", "append", "add"):
                        return "void"
                    if node.func.attr in ("count", "handle"):
                        return "long"
                    if node.func.attr == "to_aos":
                        return "list:struct:" + name
                if is_list(objt):
                    m = node.func.attr
                    if m in ("append", "push", "free", "clear"):
                        return "void"
                    if m in ("count", "handle"):
                        return "long"
                    if m in ("copy", "clone", "slice"):
                        return objt
                    if m in ("pop", "get"):
                        return list_elem(objt)
                    if m in ("contains",):
                        return "bool"
                if objt == "string":
                    m = node.func.attr
                    if m in ("strip", "upper", "lower"):
                        return "string"
                    if m in ("find", "index", "len"):
                        return "long"
                    if m == "split":
                        return "list:string"
                if isinstance(node.func.obj, A.Name) and node.func.obj.id == "rnd":
                    m = node.func.attr
                    if m == "seed":
                        return "void"
                    if m == "next_float":
                        return "float"
                    return "long"
                if isinstance(node.func.obj, A.Name) and node.func.obj.id == "lst":
                    m = node.func.attr
                    if m == "sort":
                        return "void"
                    if m == "contains":
                        return "bool"
                    return "long"
                if isinstance(node.func.obj, A.Name) and node.func.obj.id in ("m", "math"):
                    return "long" if node.func.attr == "ipow" else "float"
                if objt.startswith("weak:"):
                    inner = objt[5:]
                    if inner.endswith("?"):
                        inner = inner[:-1]
                    return self._resolve_named(inner)
            return "int"
        if isinstance(node, A.StructLit):
            return "int"
        return "int"

    # ---- expressions
    def gen_expr(self, node: A.Node) -> str:
        if isinstance(node, A.IntLit):
            typ = node.type if node.type in ("int", "long") else "int"
            value = node.value
            if typ == "int":
                value &= 0xFFFFFFFF
                if value >= 0x80000000:
                    value -= 0x100000000
                self.emit(self.cg.mov_imm(value))
            else:
                value &= 0xFFFFFFFFFFFFFFFF
                if value >= 0x8000000000000000:
                    value -= 0x10000000000000000
                self.emit(self.cg.mov_imm(value))
            return typ

        if isinstance(node, A.FloatLit):
            bits = struct.unpack("<Q", struct.pack("<d", float(node.value)))[0]
            self.emit(f"movabs rax, 0x{bits:016x}")
            self.emit("movq xmm0, rax")
            return "float"

        if isinstance(node, A.CharLit):
            self.emit(self.cg.mov_imm(node.value))
            return "char"

        if isinstance(node, A.StringLit):
            lbl = self.cg.intern_string(node.value)
            self.emit(f"lea rax, [rip+{lbl}]")
            return "string"

        if isinstance(node, A.BoolLit):
            self.emit(f"mov rax, {1 if node.value else 0}")
            return "bool"

        if isinstance(node, A.NullLit):
            self.emit("xor rax, rax")
            return "null"

        if isinstance(node, A.WeakExpr):
            t = self.gen_expr(node.value)
            return "weak:" + t

        if isinstance(node, A.Name):
            found = self.lookup(node.id)
            if not found:
                self.cg.error(f"undefined variable '{node.id}'", node.line)
            slot, typ = found
            self.load(typ, slot)
            return typ

        if isinstance(node, A.BinOp):
            return self.gen_binop(node)

        if isinstance(node, A.UnaryOp):
            return self.gen_unary(node)

        if isinstance(node, A.Compare):
            return self.gen_compare(node)

        if isinstance(node, A.BoolOp):
            return self.gen_boolop(node)

        if isinstance(node, A.Ternary):
            return self.gen_ternary(node)

        if isinstance(node, A.Call):
            return self.gen_call(node)

        if isinstance(node, A.Attribute):
            if (isinstance(node.obj, A.Name) and node.obj.id in ("m", "math")
                    and node.attr in ("pi", "e")):
                import math as _math
                val = _math.pi if node.attr == "pi" else _math.e
                bits = struct.unpack("<Q", struct.pack("<d", val))[0]
                self.emit(f"movabs rax, 0x{bits:016x}")
                self.emit("movq xmm0, rax")
                return "float"

            ot = self.resolve_type(self.static_type(node.obj))
            if ot.startswith("struct:"):
                name = ot.split(":", 1)[1]
                idx, ft = self.field_info(name, node.attr)
                self.gen_expr(node.obj)
                if is_float(ft):
                    self.emit(f"movsd xmm0, [rax+{8 * idx}]")
                else:
                    self.emit(f"mov rax, [rax+{8 * idx}]")
                return ft
            if ot.startswith("class:"):
                name = ot.split(":", 1)[1]
                idx, ft = self.field_info(name, node.attr)
                self.class_col_ptr(node.obj, idx)
                return "list:" + ft

            return self.cg.error(
                f"attribute access '.{node.attr}' is not supported yet", node.line)

        if isinstance(node, A.ListLit):
            return self.gen_list_lit(node)

        if isinstance(node, A.StructLit):
            return self.gen_struct_lit(node)

        if isinstance(node, A.Subscript):
            return self.gen_subscript(node)

        self.cg.error(f"unsupported expression {type(node).__name__}", getattr(node, "line", 0))

    def gen_list_lit(self, node: A.ListLit) -> str:
        if node.items:
            elem = combine_types([self.static_type(i) for i in node.items])
        else:
            elem = "int"
        lt = "list:" + elem
        kind = list_kind(lt)

        self.emit(f"mov ecx, {kind}")
        self.emit("call lep_list_new")
        lslot = self.alloc_slot()
        self.store("long", lslot)

        pusher = {0: "lep_list_push_i", 1: "lep_list_push_f", 2: "lep_list_push_s"}[kind]
        for item in node.items:
            t = self.gen_expr(item)
            self.convert(t, elem)
            if is_float(elem):
                self.emit("movapd xmm1, xmm0")
            else:
                self.emit("mov rdx, rax")
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {pusher}")

        self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
        return lt

    def gen_subscript(self, node: A.Subscript) -> str:
        ot = self.gen_expr(node.obj)
        if not (is_list(ot) or ot.startswith("array:")):
            self.cg.error("subscript on a non-list value", node.line)
        lslot = self.alloc_slot()
        self.store("long", lslot)

        last = len(node.index) - 1
        for k, idx_node in enumerate(node.index):
            it = self.gen_expr(idx_node)
            self.convert(it, "long")
            islot = self.alloc_slot()
            self.store("long", islot)
            elem = self.elem_type_of(ot)
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
            if k < last:
                if self.cg.bounds_mode == "unchecked":
                    self.emit("mov r10, [rcx]")
                    self.emit("mov rax, [r10+rdx*8]")
                else:
                    self.emit(f"call {self.list_get('int')}")
                nl = self.alloc_slot()
                self.store("long", nl)
                lslot = nl
                ot = elem
                continue

            if self.cg.bounds_mode == "unchecked":
                self.emit("mov r10, [rcx]")
                if is_float(elem):
                    self.emit("movsd xmm0, [r10+rdx*8]")
                else:
                    self.emit("mov rax, [r10+rdx*8]")
                return elem
            elif self.cg.bounds_mode == "checked":
                slow = self.cg.new_label("slow_get")
                done = self.cg.new_label("done_get")
                self.emit("test rcx, rcx")
                self.emit(f"jz {slow}")
                self.emit("cmp byte ptr [rcx+36], 0")
                self.emit(f"je {slow}")
                self.emit("cmp rdx, [rcx+8]")
                self.emit(f"jae {slow}")
                self.emit("mov r10, [rcx]")
                if is_float(elem):
                    self.emit("movsd xmm0, [r10+rdx*8]")
                else:
                    self.emit("mov rax, [r10+rdx*8]")
                self.emit(f"jmp {done}")
                self.label(slow)
                if is_float(elem):
                    self.emit(f"call {self.list_get('float')}")
                elif elem == "string":
                    self.emit(f"call {self.list_get('string')}")
                else:
                    self.emit(f"call {self.list_get('int')}")
                self.label(done)
                return elem
            else:
                if is_float(elem):
                    self.emit(f"call {self.list_get('float')}")
                    return elem
                if elem == "string":
                    self.emit(f"call {self.list_get('string')}")
                    return elem
                self.emit(f"call {self.list_get('int')}")
                return elem
        return "int"

    def gen_unary(self, node: A.UnaryOp) -> str:
        t = self.gen_expr(node.operand)
        if node.op == "not":
            self.emit("test rax, rax")
            self.emit("sete al")
            self.emit("movzx rax, al")
            return "bool"
        if node.op == "-":
            if is_float(t):
                self.emit("movabs rax, 0x8000000000000000")
                self.emit("movq xmm1, rax")
                self.emit("xorpd xmm0, xmm1")
            else:
                self.emit("neg rax")
                self.wrap(t)
            return t
        if node.op == "+":
            return t
        if node.op == "~":
            self.emit("not rax")
            self.wrap(t)
            return t
        self.cg.error(f"unsupported unary operator '{node.op}'", node.line)

    def gen_binop(self, node: A.BinOp) -> str:
        if node.op == "*":
            if isinstance(node.left, A.ListLit) and not isinstance(node.right, A.ListLit):
                return self.gen_list_repeat(node.left, node.right)
            if isinstance(node.right, A.ListLit) and not isinstance(node.left, A.ListLit):
                return self.gen_list_repeat(node.right, node.left)

        lt = self.gen_expr(node.left)
        lslot = self.alloc_slot()
        self.store(lt, lslot)

        rt = self.gen_expr(node.right)
        rslot = self.alloc_slot()
        self.store(rt, rslot)

        return self.gen_binary_from_slots(node.op, lt, lslot, rt, rslot)

    def gen_list_repeat(self, list_node: A.Node, count_node: A.Node) -> str:
        lt = self.gen_expr(list_node)
        if not is_list(lt):
            self.cg.error("left operand of list '*' is not a list")
        lslot = self.alloc_slot()
        self.store("long", lslot)
        ct = self.gen_expr(count_node)
        self.convert(ct, "long")
        self.emit("mov rdx, rax")
        self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
        self.emit("call lep_list_repeat")
        return lt

    def gen_binary_from_slots(self, op: str, lt: str, lslot: int,
                              rt: str, rslot: int) -> str:
        if op == "/":
            self.load_double(lt, lslot, "xmm0")
            self.load_double(rt, rslot, "xmm1")
            self.emit("divsd xmm0, xmm1")
            return "float"

        if op in ("+", "-", "*"):
            if is_float(lt) or is_float(rt):
                self.load_double(lt, lslot, "xmm0")
                self.load_double(rt, rslot, "xmm1")
                ins = {"+": "addsd", "-": "subsd", "*": "mulsd"}[op]
                self.emit(f"{ins} xmm0, xmm1")
                return "float"
            self.load(lt, lslot)
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            ins = {"+": "add", "-": "sub", "*": "imul"}[op]
            self.emit(f"{ins} rax, rcx")
            res = result_num(lt, rt)
            self.wrap(res)
            return res

        if op in ("//", "%"):
            res = result_num(lt, rt)
            if is_float(res):
                self.cg.error("floor division/modulo on floats is not supported yet")
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(rslot):+d}]")
            self.emit("call " + ("lep_floordiv" if op == "//" else "lep_floormod"))
            self.wrap(res)
            return res

        if op == "**":
            res = result_num(lt, rt)
            if is_float(res):
                self.cg.error("floating '**' is not supported yet")
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(rslot):+d}]")
            self.emit("call " + ("lep_powmod32" if res == "int" else "lep_powmod64"))
            self.wrap(res)
            return res

        if op in ("&", "|", "^"):
            res = result_num(lt, rt)
            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            ins = {"&": "and", "|": "or", "^": "xor"}[op]
            self.emit(f"{ins} rax, rcx")
            self.wrap(res)
            return res

        if op in ("<<", ">>"):
            res = result_num(lt, rt)
            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            ins = "shl" if op == "<<" else "sar"
            self.emit(f"{ins} rax, cl")
            self.wrap(res)
            return res

        self.cg.error(f"unsupported binary operator '{op}'")

    def load_double(self, typ: str, slot: int, dst: str = "xmm0"):
        if is_float(typ):
            self.emit(f"movsd {dst}, [rbp{self.off(slot):+d}]")
        else:
            self.emit(f"mov rax, [rbp{self.off(slot):+d}]")
            self.emit(f"cvtsi2sd {dst}, rax")

    def gen_compare(self, node: A.Compare) -> str:
        lt = self.gen_expr(node.left)
        lslot = self.alloc_slot()
        self.store(lt, lslot)
        rt = self.gen_expr(node.right)
        rslot = self.alloc_slot()
        self.store(rt, rslot)

        if node.op in ("in", "not in"):
            if not is_list(rt):
                self.cg.error("'in' is only supported on lists so far", node.line)
            elem = list_elem(rt)
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            if is_float(elem):
                self.emit(f"movsd xmm1, [rbp{self.off(lslot):+d}]")
                self.emit("call lep_list_contains_f")
            elif elem == "string":
                self.emit(f"mov rdx, [rbp{self.off(lslot):+d}]")
                self.emit("call lep_list_contains_s")
            else:
                self.emit(f"mov rdx, [rbp{self.off(lslot):+d}]")
                self.emit("call lep_list_contains_i")
            self.emit("test eax, eax")
            self.emit("setne al" if node.op == "in" else "sete al")
            self.emit("movzx rax, al")
            return "bool"

        if (node.op in ("==", "!=") and lt.startswith("struct:")
                and rt.startswith("struct:")):
            # struct is a value type: compare field by field, not by address
            if lt != rt:
                self.cg.error("cannot compare different struct types", node.line)
            sname = lt.split(":", 1)[1]
            self.emit(f"mov r8, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov r9, [rbp{self.off(rslot):+d}]")
            self.emit("mov al, 1")
            for idx, (_, ft0) in enumerate(self.cg.layouts.get(sname, [])):
                ft = self.resolve_type(ft0)
                if is_float(ft):
                    self.emit(f"movsd xmm0, [r8+{8 * idx}]")
                    self.emit(f"ucomisd xmm0, [r9+{8 * idx}]")
                    self.emit("sete dl")
                    self.emit("setnp dh")
                    self.emit("and dl, dh")
                else:
                    self.emit(f"mov rdx, [r8+{8 * idx}]")
                    self.emit(f"cmp rdx, [r9+{8 * idx}]")
                    self.emit("sete dl")
                self.emit("and al, dl")
            if node.op == "!=":
                self.emit("xor al, 1")
            self.emit("movzx rax, al")
            return "bool"

        int_cc = {"==": "sete", "!=": "setne", "<": "setl", "<=": "setle",
                  ">": "setg", ">=": "setge", "is": "sete", "is not": "setne"}
        float_cc = {"==": "sete", "!=": "setne", "<": "setb", "<=": "setbe",
                    ">": "seta", ">=": "setae", "is": "sete", "is not": "setne"}
        if node.op not in int_cc:
            self.cg.error(f"unsupported comparison '{node.op}'", node.line)

        if is_float(lt) or is_float(rt):
            self.load_double(lt, lslot, "xmm0")
            self.load_double(rt, rslot, "xmm1")
            self.emit("ucomisd xmm0, xmm1")
            self.emit(f"{float_cc[node.op]} al")
        else:
            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            self.emit("cmp rax, rcx")
            self.emit(f"{int_cc[node.op]} al")

        self.emit("movzx rax, al")
        return "bool"

    def gen_boolop(self, node: A.BoolOp) -> str:
        end = self.cg.new_label("bend")
        short = self.cg.new_label("bshort")
        if node.op == "and":
            self.gen_cond(node.left)
            self.emit("test rax, rax")
            self.emit(f"je {short}")
            self.gen_cond(node.right)
            self.emit("test rax, rax")
            self.emit(f"je {short}")
            self.emit("mov rax, 1")
            self.emit(f"jmp {end}")
            self.label(short)
            self.emit("xor rax, rax")
            self.label(end)
        else:
            self.gen_cond(node.left)
            self.emit("test rax, rax")
            self.emit(f"jne {short}")
            self.gen_cond(node.right)
            self.emit("test rax, rax")
            self.emit(f"jne {short}")
            self.emit("xor rax, rax")
            self.emit(f"jmp {end}")
            self.label(short)
            self.emit("mov rax, 1")
            self.label(end)
        return "bool"

    def gen_ternary(self, node: A.Ternary) -> str:
        then_t = self.static_type(node.then)
        other_t = self.static_type(node.other)
        common = self._common_scalar(then_t, other_t)
        if (is_scalar(common) and self._is_pure(node.then)
                and self._is_pure(node.other)):
            self.gen_cond(node.cond)
            cslot = self.alloc_slot()
            self.store("long", cslot)
            t = self.gen_expr(node.then)
            self.convert(t, common)
            if is_float(common):
                self.emit("movq rax, xmm0")
            s1 = self.alloc_slot()
            self.store("long", s1)
            t = self.gen_expr(node.other)
            self.convert(t, common)
            if is_float(common):
                self.emit("movq rax, xmm0")
            s2 = self.alloc_slot()
            self.store("long", s2)
            self.emit(f"mov rax, [rbp{self.off(s1):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(cslot):+d}]")
            self.emit("test rcx, rcx")
            self.emit(f"cmove rax, [rbp{self.off(s2):+d}]")
            if is_float(common):
                self.emit("movq xmm0, rax")
            return common

        other = self.cg.new_label("telse")
        end = self.cg.new_label("tend")
        self.gen_cond(node.cond)
        self.emit("test rax, rax")
        self.emit(f"je {other}")
        t1 = self.gen_expr(node.then)
        s1 = self.alloc_slot()
        self.store(t1, s1)
        self.emit(f"jmp {end}")
        self.label(other)
        t2 = self.gen_expr(node.other)
        self.convert(t2, t1)
        self.label(end)
        return t1

    # ---- calls
    def gen_call(self, node: A.Call) -> str:
        # print family
        if isinstance(node.func, A.Name) and node.func.id in ("print", "println", "write", "eprint"):
            return self.gen_print(node, newline=(node.func.id != "write"))
        if isinstance(node.func, A.Attribute):
            if (isinstance(node.func.obj, A.Name)
                    and node.func.obj.id == "io"
                    and node.func.attr in ("print", "println", "write", "eprint")):
                return self.gen_print(node, newline=(node.func.attr != "write"))
            objt = self.resolve_type(self.static_type(node.func.obj))
            if objt.startswith("class:"):
                return self.gen_class_method(node, objt)
            if is_list(objt):
                return self.gen_list_method(node, objt)
            if isinstance(node.func.obj, A.Name) and node.func.obj.id in ("m", "math"):
                return self.gen_math_call(node)
            if isinstance(node.func.obj, A.Name) and node.func.obj.id == "mem":
                return self.gen_mem_call(node)
            if objt == "string":
                return self.gen_string_method(node)
            if isinstance(node.func.obj, A.Name) and node.func.obj.id == "rnd":
                return self.gen_rand_call(node)
            if isinstance(node.func.obj, A.Name) and node.func.obj.id == "lst":
                return self.gen_lst_call(node)
            if objt.startswith("weak:"):
                return self.gen_weak_upgrade(node, objt)

        if (isinstance(node.func, A.Subscript)
                and isinstance(node.func.obj, A.Name)
                and node.func.obj.id == "list"):
            return self.gen_multidim_ctor(node)

        # pointer / memory intrinsics
        if isinstance(node.func, A.Name) and node.func.id in (
                "ptr", "deref", "memset", "cpu_features"):
            if node.func.id == "ptr":
                return self.gen_ptr_call(node)
            if node.func.id == "deref":
                return self.gen_deref(node)
            if node.func.id == "memset":
                return self.gen_memset_call(node)
            return self.gen_cpu_features()

        # struct/class construction:  Vec2(...), Particle(n)
        if isinstance(node.func, A.Name) and node.func.id in self.cg.layouts:
            name = node.func.id
            kind = self.cg.kinds.get(name, "struct")
            nfields = len(self.cg.layouts[name])
            if kind == "class":
                if len(node.args) == 1 and nfields != 1:
                    return self.gen_class_ctor(node, name)
                if len(node.args) == nfields and nfields != 1:
                    return self.gen_struct_ctor(node, name)
                return self.gen_class_ctor(node, name)
            return self.gen_struct_ctor(node, name)

        # casts
        if isinstance(node.func, A.Name) and node.func.id in ("int", "long", "float", "double", "char", "bool"):
            if len(node.args) != 1:
                self.cg.error(f"{node.func.id}() expects exactly one argument", node.line)
            target = norm(node.func.id)
            src = self.gen_expr(node.args[0])
            self.convert(src, target)
            if target == "int":
                self.wrap("int")
            elif target == "long":
                self.wrap("long")
            return target

        name = None
        if isinstance(node.func, A.Name):
            name = node.func.id
        else:
            self.cg.error("only named functions can be called", node.line)

        fdef = self.cg.functions.get(name)
        if fdef is None:
            self.cg.error(f"unknown function '{name}'", node.line)

        if len(node.args) == len(fdef.params) and self._is_inlineable(fdef):
            return self.gen_inlined_call(node, fdef)

        # evaluate arguments into slots first
        arg_slots: List[Tuple[int, str]] = []
        for arg in node.args:
            t = self.gen_expr(arg)
            slot = self.alloc_slot()
            self.store(t, slot)
            arg_slots.append((slot, t))

        # struct is a value type: pass the callee a private copy
        for slot, t in arg_slots:
            if t.startswith("struct:"):
                sname = t.split(":", 1)[1]
                self.emit(f"mov rcx, [rbp{self.off(slot):+d}]")
                self.emit(f"mov rdx, {self.layout_size(sname)}")
                self.emit("call lep_struct_copy")
                self.emit(f"mov [rbp{self.off(slot):+d}], rax")

        regs = ["rcx", "rdx", "r8", "r9"]
        for i, (slot, t) in enumerate(arg_slots):
            if is_float(t):
                self.cg.error(f"floating-point arguments to '{name}' are not supported yet", node.line)
            if i < 4:
                self.emit(f"mov {regs[i]}, [rbp{self.off(slot):+d}]")
            else:
                self.emit(f"mov rax, [rbp{self.off(slot):+d}]")
                self.emit(f"mov [rsp+{32 + 8 * (i - 4)}], rax")

        self.emit(f"call {self.cg.fn_label(name)}")

        ret = self.resolve_type(norm(fdef.ret_type)) if fdef.ret_type else "void"
        if ret == "int":
            self.wrap("int")
        return ret

    def gen_struct_ctor(self, node: A.Call, name: str) -> str:
        fields = self.cg.layouts[name]
        if len(node.args) > len(fields):
            self.cg.error(
                f"'{name}' expects at most {len(fields)} argument(s), got {len(node.args)}",
                node.line)

        self.emit(f"mov rcx, {self.layout_size(name)}")
        self.emit("call lep_struct_new")
        pslot = self.alloc_slot()
        self.store("long", pslot)

        for i, arg in enumerate(node.args):
            ft = self.resolve_type(fields[i][1])
            t = self.gen_expr(arg)
            if is_scalar(t) and is_scalar(ft):
                self.convert(t, ft)
            self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
            if is_float(ft):
                self.emit(f"movsd [rcx+{8 * i}], xmm0")
            else:
                self.emit(f"mov [rcx+{8 * i}], rax")

        self.emit(f"mov rax, [rbp{self.off(pslot):+d}]")
        return "struct:" + name

    def gen_struct_lit(self, node: A.StructLit) -> str:
        name = node.name
        if name not in self.cg.layouts:
            self.cg.error(f"unknown struct '{name}'", node.line)
        self.emit(f"mov rcx, {self.layout_size(name)}")
        self.emit("call lep_struct_new")
        pslot = self.alloc_slot()
        self.store("long", pslot)
        for fname, val in node.fields:
            idx, ft = self.field_info(name, fname)
            t = self.gen_expr(val)
            if is_scalar(t) and is_scalar(ft):
                self.convert(t, ft)
            self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
            if is_float(ft):
                self.emit(f"movsd [rcx+{8 * idx}], xmm0")
            else:
                self.emit(f"mov [rcx+{8 * idx}], rax")
        self.emit(f"mov rax, [rbp{self.off(pslot):+d}]")
        return "struct:" + name

    def gen_class_ctor(self, node: A.Call, name: str) -> str:
        fields = self.cg.layouts[name]
        if len(node.args) != 1:
            self.cg.error(f"'{name}(n)' expects exactly one size argument", node.line)

        t = self.gen_expr(node.args[0])
        self.convert(t, "long")
        nslot = self.alloc_slot()
        self.store("long", nslot)

        self.emit(f"mov rcx, [rbp{self.off(nslot):+d}]")
        self.emit(f"mov rdx, {len(fields)}")
        self.emit("call lep_soa_new")
        hslot = self.alloc_slot()
        self.store("long", hslot)

        for i, (f, ft0) in enumerate(fields):
            ft = self.resolve_type(ft0)
            kind = 1 if is_float(ft) else (2 if ft == "string" else 0)
            self.emit(f"mov ecx, {kind}")
            self.emit(f"mov rdx, [rbp{self.off(nslot):+d}]")
            self.emit("call lep_list_zeros")
            self.emit("mov r8, rax")
            self.emit(f"mov rdx, {i}")
            self.emit(f"mov rcx, [rbp{self.off(hslot):+d}]")
            self.emit("call lep_soa_set_field")

        self.emit(f"mov rax, [rbp{self.off(hslot):+d}]")
        return "class:" + name

    def gen_class_method(self, node: A.Call, objt: str) -> str:
        name = objt.split(":", 1)[1]
        fields = self.cg.layouts[name]
        attr = node.func.attr

        self.gen_expr(node.func.obj)
        hslot = self.alloc_slot()
        self.store("long", hslot)

        if attr == "count":
            self.emit(f"mov rcx, [rbp{self.off(hslot):+d}]")
            self.emit("call lep_soa_count")
            return "long"

        if attr in ("push", "append", "add"):
            if len(node.args) != len(fields):
                self.cg.error(
                    f"'{name}.{attr}()' expects {len(fields)} argument(s)", node.line)
            value_slots = []
            for i, arg in enumerate(node.args):
                ft = self.resolve_type(fields[i][1])
                t = self.gen_expr(arg)
                if is_scalar(t) and is_scalar(ft):
                    self.convert(t, ft)
                slot = self.alloc_slot()
                self.store(ft, slot)
                value_slots.append((slot, ft))

            for i, (slot, ft) in enumerate(value_slots):
                self.emit(f"mov rax, [rbp{self.off(hslot):+d}]")
                self.emit(f"mov rcx, [rax+{16 + 8 * i}]")
                if is_float(ft):
                    self.emit(f"movsd xmm1, [rbp{self.off(slot):+d}]")
                    self.emit("call lep_list_push_f")
                elif ft == "string":
                    self.emit(f"mov rdx, [rbp{self.off(slot):+d}]")
                    self.emit("call lep_list_push_s")
                else:
                    self.emit(f"mov rdx, [rbp{self.off(slot):+d}]")
                    self.emit("call lep_list_push_i")
            return "void"

        if attr == "to_aos":
            # build a list of struct blocks (one per row)
            self.emit("mov ecx, 0")
            self.emit("call lep_list_new")
            lslot = self.alloc_slot()
            self.store("long", lslot)

            self.emit(f"mov rcx, [rbp{self.off(hslot):+d}]")
            self.emit("call lep_soa_count")
            nslot = self.alloc_slot()
            self.store("long", nslot)

            idx = self.alloc_slot()
            self.emit("xor rax, rax")
            self.store("long", idx)

            top = self.cg.new_label("aostop")
            end = self.cg.new_label("aosend")
            self.label(top)
            self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
            self.emit(f"cmp rax, [rbp{self.off(nslot):+d}]")
            self.emit(f"jge {end}")

            self.emit(f"mov rcx, {self.layout_size(name)}")
            self.emit("call lep_struct_new")
            pslot = self.alloc_slot()
            self.store("long", pslot)

            for i, (f, ft0) in enumerate(fields):
                ft = self.resolve_type(ft0)
                self.emit(f"mov rax, [rbp{self.off(hslot):+d}]")
                self.emit(f"mov rcx, [rax+{16 + 8 * i}]")
                self.emit(f"mov rdx, [rbp{self.off(idx):+d}]")
                if is_float(ft):
                    self.emit(f"call {self.list_get('float')}")
                    self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
                    self.emit(f"movsd [rcx+{8 * i}], xmm0")
                else:
                    self.emit(f"call {self.list_get('int')}")
                    self.emit(f"mov rcx, [rbp{self.off(pslot):+d}]")
                    self.emit(f"mov [rcx+{8 * i}], rax")

            self.emit(f"mov rdx, [rbp{self.off(pslot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call lep_list_push_i")

            self.emit(f"mov rax, [rbp{self.off(idx):+d}]")
            self.emit("add rax, 1")
            self.store("long", idx)
            self.emit(f"jmp {top}")
            self.label(end)

            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            return "list:struct:" + name

        self.cg.error(f"unsupported class method '.{attr}'", node.line)

    def _list_axis_size(self, lslot: int, dim_node: A.Node, line: int):
        t = self.gen_expr(dim_node)
        self.convert(t, "long")
        kslot = self.alloc_slot()
        self.store("long", kslot)
        cur = self.alloc_slot()
        self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
        self.store("long", cur)

        top = self.cg.new_label("axtop")
        end = self.cg.new_label("axend")
        self.label(top)
        self.emit(f"mov rax, [rbp{self.off(kslot):+d}]")
        self.emit("test rax, rax")
        self.emit(f"je {end}")
        self.emit(f"mov rcx, [rbp{self.off(cur):+d}]")
        self.emit("xor rdx, rdx")
        self.emit(f"call {self.list_get('int')}")
        self.store("long", cur)
        self.emit(f"mov rax, [rbp{self.off(kslot):+d}]")
        self.emit("sub rax, 1")
        self.store("long", kslot)
        self.emit(f"jmp {top}")
        self.label(end)
        self.emit(f"mov rcx, [rbp{self.off(cur):+d}]")
        self.emit("call lep_list_size")

    def gen_list_method(self, node: A.Call, objt: str) -> str:
        attr = node.func.attr
        ot = self.gen_expr(node.func.obj)
        if not is_list(ot):
            self.cg.error(f"'.{attr}' called on a non-list value", node.line)
        lslot = self.alloc_slot()
        self.store("long", lslot)
        elem = list_elem(ot)

        if attr in ("append", "push"):
            if len(node.args) != 1:
                self.cg.error(f"list.{attr}() expects one argument", node.line)
            t = self.gen_expr(node.args[0])
            self.convert(t, elem)
            if is_float(elem):
                self.emit("movapd xmm1, xmm0")
                fn = "lep_list_push_f"
            elif elem == "string":
                self.emit("mov rdx, rax")
                fn = "lep_list_push_s"
            else:
                self.emit("mov rdx, rax")
                fn = "lep_list_push_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            return "void"

        if attr in ("count", "length"):
            if node.args:
                if len(node.args) != 1:
                    self.cg.error("list.count(dim) expects one argument", node.line)
                self._list_axis_size(lslot, node.args[0], node.line)
                return "long"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call lep_list_size")
            return "long"

        if attr == "slice":
            if len(node.args) != 2:
                self.cg.error("list.slice(a, b) expects two arguments", node.line)
            t = self.gen_expr(node.args[0])
            self.convert(t, "long")
            aslot = self.alloc_slot()
            self.store("long", aslot)
            t = self.gen_expr(node.args[1])
            self.convert(t, "long")
            bslot = self.alloc_slot()
            self.store("long", bslot)
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(aslot):+d}]")
            self.emit(f"mov r8, [rbp{self.off(bslot):+d}]")
            self.emit("call lep_list_slice")
            return ot

        if attr in ("copy", "clone"):
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call lep_list_copy")
            return ot

        if attr == "handle":
            self.emit(f"mov rax, [rbp{self.off(lslot):+d}]")
            return "long"

        if attr == "contains":
            if len(node.args) != 1:
                self.cg.error("list.contains() expects one argument", node.line)
            t = self.gen_expr(node.args[0])
            self.convert(t, elem)
            if is_float(elem):
                self.emit("movq rdx, xmm0")
                self.emit("movapd xmm1, xmm0")
                fn = "lep_list_contains_f"
            else:
                self.emit("mov rdx, rax")
                fn = "lep_list_contains_s" if elem == "string" else "lep_list_contains_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            self.emit("test eax, eax")
            self.emit("setne al")
            self.emit("movzx rax, al")
            return "bool"

        if attr == "pop":
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call " + ("lep_list_pop_f" if is_float(elem) else "lep_list_pop"))
            return elem

        if attr == "free":
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit("call lep_list_free")
            return "void"

        self.cg.error(f"unsupported list method '.{attr}'", node.line)

    def gen_math_call(self, node: A.Call) -> str:
        attr = node.func.attr
        unary = {"sqrt", "sin", "cos", "tan", "floor", "ceil", "fabs"}
        if attr in unary:
            t = self.gen_expr(node.args[0])
            self.convert(t, "float")
            self.emit(f"call {attr}")
            return "float"

        if attr in ("pow", "atan2"):
            t1 = self.gen_expr(node.args[0])
            self.convert(t1, "float")
            aslot = self.alloc_slot()
            self.store("float", aslot)
            t2 = self.gen_expr(node.args[1])
            self.convert(t2, "float")
            self.emit("movapd xmm1, xmm0")
            self.emit(f"movsd xmm0, [rbp{self.off(aslot):+d}]")
            self.emit(f"call {attr}")
            return "float"

        if attr == "ipow":
            t1 = self.gen_expr(node.args[0])
            self.convert(t1, "long")
            aslot = self.alloc_slot()
            self.store("long", aslot)
            t2 = self.gen_expr(node.args[1])
            self.convert(t2, "long")
            self.emit("mov rdx, rax")
            self.emit(f"mov rcx, [rbp{self.off(aslot):+d}]")
            self.emit("call lep_powmod64")
            return "long"

        if attr in ("min", "max"):
            t1 = self.gen_expr(node.args[0])
            self.convert(t1, "long")
            aslot = self.alloc_slot()
            self.store("long", aslot)
            t2 = self.gen_expr(node.args[1])
            self.convert(t2, "long")
            rslot = self.alloc_slot()
            self.store("long", rslot)
            self.emit(f"mov rax, [rbp{self.off(aslot):+d}]")
            self.emit(f"mov rcx, [rbp{self.off(rslot):+d}]")
            self.emit("cmp rax, rcx")
            self.emit("cmovg rax, rcx" if attr == "min" else "cmovl rax, rcx")
            return "long"

        if attr == "abs":
            t = self.gen_expr(node.args[0])
            if is_float(t):
                self.emit("call fabs")
                return "float"
            self.emit("mov rcx, rax")
            self.emit("neg rcx")
            self.emit("test rax, rax")
            self.emit("cmovs rax, rcx")
            return t

        self.cg.error(f"unsupported math function '.{attr}'", node.line)

    def gen_mem_call(self, node: A.Call) -> str:
        attr = node.func.attr
        if attr == "is_alive":
            t = self.gen_expr(node.args[0])
            self.convert(t, "long")
            self.emit("mov rcx, rax")
            self.emit("call lep_mem_is_alive")
            self.emit("test eax, eax")
            self.emit("setne al")
            self.emit("movzx rax, al")
            return "bool"
        if attr in ("leak_report", "init"):
            if attr == "leak_report":
                self.emit("call lep_mem_leak_report")
            return "void"
        self.cg.error(f"unsupported mem function '.{attr}'", node.line)

    def gen_string_method(self, node: A.Call) -> str:
        attr = node.func.attr
        t = self.gen_expr(node.func.obj)
        self.convert(t, "string")
        sslot = self.alloc_slot()
        self.store("string", sslot)

        if attr in ("strip", "upper", "lower"):
            self.emit(f"mov rcx, [rbp{self.off(sslot):+d}]")
            self.emit(f"call lep_str_{attr}")
            return "string"
        if attr in ("find", "index"):
            a = self.gen_expr(node.args[0])
            self.convert(a, "string")
            self.emit("mov rdx, rax")
            self.emit(f"mov rcx, [rbp{self.off(sslot):+d}]")
            self.emit("call lep_str_find")
            return "long"
        if attr == "split":
            a = self.gen_expr(node.args[0])
            self.convert(a, "string")
            self.emit("mov rdx, rax")
            self.emit(f"mov rcx, [rbp{self.off(sslot):+d}]")
            self.emit("call lep_str_split")
            return "list:string"
        if attr == "len":
            self.emit(f"mov rcx, [rbp{self.off(sslot):+d}]")
            self.emit("call lep_str_len")
            return "long"
        self.cg.error(f"unsupported string method '.{attr}'", node.line)

    def gen_rand_call(self, node: A.Call) -> str:
        attr = node.func.attr
        if attr == "seed":
            t = self.gen_expr(node.args[0])
            self.convert(t, "long")
            self.emit("mov rcx, rax")
            self.emit("call lep_rand_seed")
            return "void"
        if attr == "next_int":
            self.emit("call lep_rand_next_int")
            return "long"
        if attr == "next_float":
            self.emit("call lep_rand_next_float")
            return "float"
        if attr == "range":
            t = self.gen_expr(node.args[0])
            self.convert(t, "long")
            aslot = self.alloc_slot()
            self.store("long", aslot)
            t = self.gen_expr(node.args[1])
            self.convert(t, "long")
            self.emit("mov rcx, rax")
            self.emit("sub rcx, [rbp" + f"{self.off(aslot):+d}" + "]")
            bslot = self.alloc_slot()
            self.store("long", bslot)
            self.emit("call lep_rand_next_int")
            self.emit(f"mov rcx, [rbp{self.off(bslot):+d}]")
            self.emit("cqo")
            self.emit("idiv rcx")
            self.emit("mov rax, rdx")
            self.emit(f"add rax, [rbp{self.off(aslot):+d}]")
            return "long"
        self.cg.error(f"unsupported rand function '.{attr}'", node.line)

    def gen_lst_call(self, node: A.Call) -> str:
        attr = node.func.attr
        lt = self.gen_expr(node.args[0])
        if not is_list(lt):
            self.cg.error(f"lst.{attr}() expects a list", node.line)
        elem = list_elem(lt)
        lslot = self.alloc_slot()
        self.store("long", lslot)

        if attr == "sort":
            fn = "lep_list_sort_f" if is_float(elem) else "lep_list_sort_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            return "void"
        if attr == "sum":
            fn = "lep_list_sum_f" if is_float(elem) else "lep_list_sum_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            return "float" if is_float(elem) else "long"
        if attr == "max":
            fn = "lep_list_max_f" if is_float(elem) else "lep_list_max_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            return "float" if is_float(elem) else "long"
        if attr == "contains":
            t = self.gen_expr(node.args[1])
            self.convert(t, elem)
            if is_float(elem):
                self.emit("movapd xmm1, xmm0")
                fn = "lep_list_contains_f"
            else:
                self.emit("mov rdx, rax")
                fn = "lep_list_contains_s" if elem == "string" else "lep_list_contains_i"
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"call {fn}")
            self.emit("test eax, eax")
            self.emit("setne al")
            self.emit("movzx rax, al")
            return "bool"
        self.cg.error(f"unsupported list function '.{attr}'", node.line)

    def gen_weak_upgrade(self, node: A.Call, objt: str) -> str:
        attr = node.func.attr
        if attr not in ("upgrade", "get"):
            self.cg.error(f"unsupported weak method '.{attr}'", node.line)
        inner = objt[5:]
        if inner.endswith("?"):
            inner = inner[:-1]
        self.gen_expr(node.func.obj)
        return self._resolve_named(inner)

    # ---- pointers ---------------------------------------------------------
    def gen_addr(self, node: A.Node) -> str:
        """Materialise the address of an lvalue into rax; returns 'ptr'."""
        if isinstance(node, A.Name):
            found = self.lookup(node.id)
            if not found:
                self.cg.error(f"undefined variable '{node.id}'", node.line)
            self.emit(f"lea rax, [rbp{self.off(found[0]):+d}]")
            return "ptr"

        if isinstance(node, A.Attribute):
            ot = self.resolve_type(self.static_type(node.obj))
            if not ot.startswith("struct:"):
                self.cg.error("ptr() of a field is only supported on structs", node.line)
            name = ot.split(":", 1)[1]
            idx, _ = self.field_info(name, node.attr)
            self.gen_expr(node.obj)
            if idx:
                self.emit(f"add rax, {8 * idx}")
            return "ptr"

        if isinstance(node, A.Subscript):
            ot = self.gen_expr(node.obj)
            if not (is_list(ot) or ot.startswith("array:")):
                self.cg.error("ptr() expects a variable, struct field or list element", node.line)
            lslot = self.alloc_slot()
            self.store("long", lslot)
            for idx_node in node.index[:-1]:
                it = self.gen_expr(idx_node)
                self.convert(it, "long")
                islot = self.alloc_slot()
                self.store("long", islot)
                ot = self.elem_type_of(ot)
                self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
                self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
                self.emit(f"call {self.list_get('int')}")
                nl = self.alloc_slot()
                self.store("long", nl)
                lslot = nl
            it = self.gen_expr(node.index[-1])
            self.convert(it, "long")
            islot = self.alloc_slot()
            self.store("long", islot)
            self.emit(f"mov rcx, [rbp{self.off(lslot):+d}]")
            self.emit(f"mov rdx, [rbp{self.off(islot):+d}]")
            self.emit(f"call {self.list_addr()}")
            return "ptr"

        self.cg.error("ptr() expects a variable, struct field or list element", node.line)

    def gen_ptr_call(self, node: A.Call) -> str:
        if len(node.args) != 1:
            self.cg.error("ptr() expects exactly one argument", node.line)
        return self.gen_addr(node.args[0])

    def gen_deref(self, node: A.Call) -> str:
        if len(node.args) != 1:
            self.cg.error("deref() expects exactly one argument", node.line)
        self.gen_expr(node.args[0])
        self.emit("mov rax, [rax]")
        return "long"

    def gen_memset_call(self, node: A.Call) -> str:
        if len(node.args) != 3:
            self.cg.error("memset(dst, value, nbytes) expects three arguments", node.line)
        self.gen_expr(node.args[0])
        dslot = self.alloc_slot()
        self.store("long", dslot)

        t = self.gen_expr(node.args[1])
        self.convert(t, "long")
        vslot = self.alloc_slot()
        self.store("long", vslot)

        t = self.gen_expr(node.args[2])
        self.convert(t, "long")
        self.emit(f"mov r8, rax")
        self.emit(f"mov rdx, [rbp{self.off(vslot):+d}]")
        self.emit(f"mov rcx, [rbp{self.off(dslot):+d}]")
        self.emit("call lep_memset")
        return "void"

    def gen_cpu_features(self) -> str:
        self.emit("call lep_cpu_features")
        return "long"

    def gen_print(self, node: A.Call, newline: bool) -> str:
        for i, arg in enumerate(node.args):
            if i > 0:
                sep = self.cg.fmt("sep", " ")
                self.emit(f"lea rcx, [rip+{sep}]")
                self.emit("xor eax, eax")
                self.emit("call printf")

            t = self.gen_expr(arg)
            if is_float(t):
                lbl = self.cg.fmt("float", "%g")
                # Win64 varargs: float must also be duplicated in the GP slot.
                self.emit("movq rdx, xmm0")
                self.emit("movapd xmm1, xmm0")
                self.emit(f"lea rcx, [rip+{lbl}]")
                self.emit("mov eax, 1")
                self.emit("call printf")
            elif t == "string":
                lbl = self.cg.fmt("str", "%s")
                self.emit("mov rdx, rax")
                self.emit(f"lea rcx, [rip+{lbl}]")
                self.emit("xor eax, eax")
                self.emit("call printf")
            elif t == "char":
                lbl = self.cg.fmt("char", "%c")
                self.emit("mov rdx, rax")
                self.emit(f"lea rcx, [rip+{lbl}]")
                self.emit("xor eax, eax")
                self.emit("call printf")
            else:
                lbl = self.cg.fmt("int", "%lld")
                self.emit("mov rdx, rax")
                self.emit(f"lea rcx, [rip+{lbl}]")
                self.emit("xor eax, eax")
                self.emit("call printf")

        if newline:
            nl = self.cg.fmt("nl", "\n")
            self.emit(f"lea rcx, [rip+{nl}]")
            self.emit("xor eax, eax")
            self.emit("call printf")
        return "void"


# ---------------------------------------------------------------- top level
class CodeGen:
    def __init__(self, program: A.Program, filename: str = "<src>",
                 options: Optional[Dict[str, object]] = None):
        self.program = program
        self.filename = filename
        options = options or {}
        self.bounds_mode = str(options.get("bounds", "checked"))
        self.bounds_suffix = {"checked": "", "unchecked": "_ub",
                              "warn": "_warn"}.get(self.bounds_mode, "")
        self.avx2 = bool(options.get("avx2", False))
        self.functions: Dict[str, A.FuncDef] = {}
        self.classes: Dict[str, A.ClassDecl] = {}
        self.layouts: Dict[str, List[Tuple[str, str]]] = {}
        self.kinds: Dict[str, str] = {}
        self.rodata: List[Tuple[str, bytes]] = []
        self.fmt_cache: Dict[str, str] = {}
        self.label_counter = 0
        self.generated: List[str] = []

    # ---- utilities
    def new_label(self, prefix: str) -> str:
        self.label_counter += 1
        return f".L{prefix}{self.label_counter}"

    def error(self, msg: str, line: int = 0):
        raise CodeGenError(msg, line, self.filename)

    def mov_imm(self, value: int) -> str:
        if -0x80000000 <= value <= 0x7FFFFFFF:
            return f"mov rax, {value}"
        u = value & 0xFFFFFFFFFFFFFFFF
        return f"movabs rax, 0x{u:016x}"

    def intern_string(self, s: str) -> str:
        return self.intern_data(s.encode("utf-8") + b"\x00")

    def intern_data(self, data: bytes) -> str:
        for lbl, d in self.rodata:
            if d == data:
                return lbl
        lbl = f".Ldata{len(self.rodata)}"
        self.rodata.append((lbl, data))
        return lbl

    def fmt(self, key: str, raw: str) -> str:
        if key in self.fmt_cache:
            return self.fmt_cache[key]
        lbl = self.intern_string(raw)
        self.fmt_cache[key] = lbl
        return lbl

    def fn_label(self, name: str) -> str:
        return "lep_fn_" + name.replace(".", "_")

    # ---- driver
    def generate(self) -> str:
        for stmt in self.program.stmts:
            if isinstance(stmt, A.FuncDef):
                self.functions[stmt.name] = stmt
            elif isinstance(stmt, A.ClassDecl):
                self.classes[stmt.name] = stmt
                self.kinds[stmt.name] = stmt.kind
                self.layouts[stmt.name] = [(f, norm(t)) for (f, t) in stmt.fields]

        func_lines: List[str] = []

        # user functions
        for fdef in self.functions.values():
            if fdef.name == "main":
                continue
            func_lines.extend(self.gen_function(fdef))

        # user main
        if "main" in self.functions:
            func_lines.extend(self.gen_function(self.functions["main"]))

        # top-level statements (ignore declarations)
        top_stmts = [s for s in self.program.stmts
                     if not isinstance(s, (A.FuncDef, A.ClassDecl, A.Import,
                                           A.FromImport, A.TypeAlias))]
        has_toplevel = bool(top_stmts)
        if has_toplevel:
            func_lines.extend(self.gen_toplevel(top_stmts))

        # C entry point
        entry = ["main:", "    push rbp", "    mov rbp, rsp", "    sub rsp, 64"]
        if has_toplevel:
            entry.append("    call lep_toplevel")
        if "main" in self.functions:
            entry.append("    call " + self.fn_label("main"))
        entry += ["    xor eax, eax", "    leave", "    ret"]

        out: List[str] = []
        out.append(".intel_syntax noprefix")
        out.append(".text")
        out.append(".globl main")
        out.append(".extern printf")
        for sym in ("sqrt", "pow", "sin", "cos", "tan", "floor", "ceil",
                    "fabs", "atan2"):
            out.append(f".extern {sym}")
        out.extend(HELPERS.strip("\n").splitlines())
        out.extend(func_lines)
        out.extend(entry)

        if self.rodata:
            out.append(".section .rodata")
            for lbl, data in self.rodata:
                out.append(lbl + ":")
                out.append("    .byte " + ",".join(str(b) for b in data))

        return "\n".join(out) + "\n"

    def gen_function(self, fdef: A.FuncDef) -> List[str]:
        fg = FuncGen(self, fdef.name, fdef.params, fdef.ret_type)

        regs = ["rcx", "rdx", "r8", "r9"]
        class_params: List[Tuple[int, str]] = []
        for i, p in enumerate(fdef.params):
            typ = norm(p.type) if p.type else "long"
            slot = fg.declare(p.name, typ)
            if i < 4:
                fg.emit(f"mov [rbp{fg.off(slot):+d}], {regs[i]}")
            else:
                fg.emit(f"mov rax, [rbp+{48 + 8 * (i - 4)}]")
                fg.emit(f"mov [rbp{fg.off(slot):+d}], rax")
            rtyp = fg.resolve_type(typ)
            if rtyp.startswith("class:"):
                class_params.append((slot, rtyp))
        for slot, rtyp in class_params:
            fg.populate_col_cache(slot, rtyp)

        fg.gen_block(fdef.body)
        fg.label(fg.ret_label)
        fg.emit("leave")
        fg.emit("ret")

        frame = align16(8 * fg.slots + 32)
        lines = [self.fn_label(fdef.name) + ":"]
        lines.append("    push rbp")
        lines.append("    mov rbp, rsp")
        lines.append(f"    sub rsp, {frame}")
        lines.extend(fg.body)
        return regalloc.optimize_function(lines, frame, fg.slot_types)

    def gen_toplevel(self, stmts: List[A.Node]) -> List[str]:
        fg = FuncGen(self, "<toplevel>", [], None)
        fg.gen_block(stmts)
        fg.label(fg.ret_label)
        fg.emit("leave")
        fg.emit("ret")

        frame = align16(8 * fg.slots + 32)
        lines = ["lep_toplevel:"]
        lines.append("    push rbp")
        lines.append("    mov rbp, rsp")
        lines.append(f"    sub rsp, {frame}")
        lines.extend(fg.body)
        return regalloc.optimize_function(lines, frame, fg.slot_types)
