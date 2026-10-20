#!/usr/bin/env python3
"""lep — Lepria compiler CLI.

Usage:
  python lep.py run    <file.lep> [args...]
  python lep.py build  <file.lep> [-o app.exe] [--emit-asm out.s] [--debug]
  python lep.py emit   <file.lep> [-o out.s]
  python lep.py check  <file.lep>
"""

import argparse
import os
import sys

from compiler.driver import (assemble, build, compile_file, compile_to_asm,
                             run_source)
from compiler.lexer import LexError
from compiler.parser import ParseError
from compiler.codegen import CodeGenError


def _options(args) -> dict:
    if getattr(args, "no_bounds_check", False):
        bounds = "unchecked"
    elif getattr(args, "bounds_warn", False):
        bounds = "warn"
    else:
        bounds = "checked"
    return {"bounds": bounds, "avx2": bool(getattr(args, "mavx2", False))}


_FLAG_DESTS = {
    "--no-bounds-check": "no_bounds_check", "-fno-bounds-check": "no_bounds_check",
    "--bounds-warn": "bounds_warn", "-fbounds-warn": "bounds_warn",
    "--mavx2": "mavx2", "-avx2": "mavx2",
}


def _split_program_args(argv):
    """Pull our codegen flags out of a ``run`` REMAINDER (they may follow the
    file) and return (program_args, option_overrides)."""
    prog, over = [], {}
    for a in argv:
        dest = _FLAG_DESTS.get(a)
        if dest is not None:
            over[dest] = True
        else:
            prog.append(a)
    return prog, over


def cmd_run(args):
    prog_args, over = _split_program_args(list(args.args))
    opts = _options(args)
    if over.get("no_bounds_check"):
        opts["bounds"] = "unchecked"
    elif over.get("bounds_warn"):
        opts["bounds"] = "warn"
    if over.get("mavx2"):
        opts["avx2"] = True
    return run_source(args.file, prog_args, options=opts)


def cmd_build(args):
    exe = args.output or (os.path.splitext(args.file)[0] + ".exe")
    ok, out, asm_path = build(args.file, exe, args.emit_asm,
                              keep_asm=bool(args.emit_asm) or not args.clean,
                              debug=args.debug, options=_options(args))
    if out.strip():
        print(out.strip())
    if ok:
        print(f"built: {exe}")
        if args.emit_asm:
            print(f"asm:   {asm_path}")
        return 0
    return 1


def cmd_emit(args):
    out = args.output or (os.path.splitext(args.file)[0] + ".s")
    compile_file(args.file, out, options=_options(args))
    print(f"asm: {out}")
    return 0


def cmd_check(args):
    with open(args.file, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        asm = compile_to_asm(src, args.file, options=_options(args))
    except (LexError, ParseError, CodeGenError) as e:
        print(f"error: {e}")
        return 1
    print(f"ok: {args.file} ({len(asm.splitlines())} asm lines)")
    return 0


def _add_codegen_flags(p):
    p.add_argument("--no-bounds-check", "-fno-bounds-check", action="store_true",
                   dest="no_bounds_check",
                   help="skip array/list bounds (length) checks")
    p.add_argument("--bounds-warn", "-fbounds-warn", action="store_true",
                   dest="bounds_warn",
                   help="warn on array overflow instead of aborting (clamp index)")
    p.add_argument("--mavx2", "-avx2", action="store_true",
                   help="enable AVX2 (runtime + contiguous-loop vectorization)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="lep", description="Lepria compiler")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build and run")
    r.add_argument("file")
    _add_codegen_flags(r)
    r.add_argument("args", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("build", help="build a native executable")
    b.add_argument("file")
    b.add_argument("-o", "--output")
    b.add_argument("--emit-asm")
    b.add_argument("--debug", action="store_true")
    b.add_argument("--clean", action="store_true", help="remove intermediate .s")
    _add_codegen_flags(b)
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("emit", help="emit assembly only")
    e.add_argument("file")
    e.add_argument("-o", "--output")
    _add_codegen_flags(e)
    e.set_defaults(func=cmd_emit)

    c = sub.add_parser("check", help="parse + generate (no native build)")
    c.add_argument("file")
    _add_codegen_flags(c)
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)

    try:
        return args.func(args)
    except (LexError, ParseError, CodeGenError) as e:
        print(f"error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
