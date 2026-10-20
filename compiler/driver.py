"""Compilation driver: .lep -> .s -> .exe (via g++/gcc).

The generated assembly is assembled by the native toolchain and linked
together with the glibep runtime (``runtime/lep_runtime.cpp``).
"""

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(ROOT, "runtime")
RUNTIME_CPP = os.path.join(RUNTIME_DIR, "lep_runtime.cpp")
RUNTIME_OBJ = os.path.join(RUNTIME_DIR, "lep_runtime.o")
RUNTIME_OBJ_AVX2 = os.path.join(RUNTIME_DIR, "lep_runtime_avx2.o")


def _find(names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _find_gcc() -> str:
    path = _find(("g++", "gcc", "cc"))
    if not path:
        raise RuntimeError("g++/gcc not found on PATH; install MinGW/TDM-GCC")
    return path


def ensure_runtime(avx2: bool = False) -> str:
    """Compile the C++ runtime object if it is missing or stale."""
    gpp = _find(("g++",))
    if not gpp:
        raise RuntimeError("g++ not found on PATH; the glibep runtime needs it")
    obj = RUNTIME_OBJ_AVX2 if avx2 else RUNTIME_OBJ
    if (os.path.exists(obj)
            and os.path.getmtime(obj) >= os.path.getmtime(RUNTIME_CPP)):
        return obj
    cmd = [gpp, "-std=c++17", "-O2", "-c", RUNTIME_CPP, "-o", obj]
    if avx2:
        cmd.insert(1, "-mavx2")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("failed to build runtime:\n" + (proc.stdout or "") + (proc.stderr or ""))
    return obj


def compile_to_asm(src: str, filename: str = "<src>",
                   options: Optional[Dict[str, object]] = None) -> str:
    tokens = Lexer(src, filename).tokenize()
    program = Parser(tokens, filename).parse()
    return CodeGen(program, filename, options).generate()


def compile_file(src_path: str, asm_path: str,
                 options: Optional[Dict[str, object]] = None) -> str:
    with open(src_path, "r", encoding="utf-8-sig") as f:
        src = f.read()
    asm = compile_to_asm(src, src_path, options)
    os.makedirs(os.path.dirname(os.path.abspath(asm_path)), exist_ok=True)
    with open(asm_path, "w", encoding="ascii", newline="\n") as f:
        f.write(asm)
    return asm_path


def assemble(asm_path: str, exe_path: str, debug: bool = False,
             avx2: bool = False) -> Tuple[bool, str]:
    """Assemble+link the generated GAS file together with the runtime."""
    gpp = _find_gcc()
    runtime_obj = ensure_runtime(avx2=avx2)
    cmd: List[str] = [gpp, asm_path, runtime_obj, "-o", exe_path]
    cmd.insert(1, "-g" if debug else "-O2")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def build(src_path: str, exe_path: str, asm_path: Optional[str] = None,
          keep_asm: bool = True, debug: bool = False,
          options: Optional[Dict[str, object]] = None) -> Tuple[bool, str, str]:
    """Compile ``src_path`` to a native executable. Returns (ok, message, asm_path)."""
    options = options or {}
    if asm_path is None:
        base, _ = os.path.splitext(exe_path)
        asm_path = base + ".s"

    compile_file(src_path, asm_path, options)
    ok, out = assemble(asm_path, exe_path, debug=debug,
                       avx2=bool(options.get("avx2")))

    if not keep_asm and ok:
        try:
            os.remove(asm_path)
        except OSError:
            pass

    return ok, out, asm_path


def run_source(src_path: str, args: Optional[List[str]] = None,
               options: Optional[Dict[str, object]] = None) -> int:
    tmpdir = tempfile.mkdtemp(prefix="lep_")
    base = os.path.splitext(os.path.basename(src_path))[0] or "app"
    exe_path = os.path.join(tmpdir, base + ".exe")
    asm_path = os.path.join(tmpdir, base + ".s")

    ok, out, _ = build(src_path, exe_path, asm_path, keep_asm=False, options=options)
    if not ok:
        print(out)
        return 1

    proc = subprocess.run([exe_path] + (args or []))
    return proc.returncode
