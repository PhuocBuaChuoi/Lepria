"""AST node definitions for Lepria.

Node is a plain marker base class; every concrete node is a dataclass with a
trailing ``line`` field so construction can use keyword arguments everywhere.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


class Node:
    line: int = 0


# ---------------------------------------------------------------- top level
@dataclass
class Program:
    stmts: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class Import:
    module: str = ""
    alias: Optional[str] = None
    line: int = 0


@dataclass
class FromImport:
    module: str = ""
    names: List[str] = field(default_factory=list)
    line: int = 0


@dataclass
class Param:
    name: str = ""
    type: Optional[str] = None
    default: Optional[Any] = None
    line: int = 0


@dataclass
class FuncDef:
    name: str = ""
    params: List[Param] = field(default_factory=list)
    ret_type: Optional[str] = None
    body: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class VarDecl:
    name: str = ""
    type: Optional[str] = None
    value: Optional[Any] = None
    is_static: bool = False
    line: int = 0


@dataclass
class TypeAlias:
    name: str = ""
    type: str = ""
    line: int = 0


@dataclass
class ClassDecl:
    name: str = ""
    kind: str = "class"  # "class" or "struct"
    fields: List[Tuple[str, str]] = field(default_factory=list)
    methods: List[FuncDef] = field(default_factory=list)
    line: int = 0


# ---------------------------------------------------------------- statements
@dataclass
class Assign:
    target: Any = None
    op: str = "="
    value: Any = None
    line: int = 0


@dataclass
class ExprStmt:
    expr: Any = None
    line: int = 0


@dataclass
class If:
    branches: List[Tuple[Any, List[Any]]] = field(default_factory=list)
    else_body: Optional[List[Any]] = None
    line: int = 0


@dataclass
class While:
    cond: Any = None
    body: List[Any] = field(default_factory=list)
    label: Optional[str] = None
    line: int = 0


@dataclass
class ForRange:
    var: str = ""
    start: Any = None
    end: Any = None
    step: Any = None
    body: List[Any] = field(default_factory=list)
    label: Optional[str] = None
    line: int = 0


@dataclass
class ForEach:
    var: str = ""
    iterable: Any = None
    body: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class Return:
    value: Optional[Any] = None
    line: int = 0


@dataclass
class Break:
    label: Optional[str] = None
    line: int = 0


@dataclass
class Continue:
    label: Optional[str] = None
    line: int = 0


@dataclass
class Pass:
    line: int = 0


@dataclass
class Raise:
    value: Any = None
    line: int = 0


# ---------------------------------------------------------------- expressions
@dataclass
class IntLit:
    value: int = 0
    type: str = "int"
    line: int = 0


@dataclass
class FloatLit:
    value: float = 0.0
    type: str = "double"
    line: int = 0


@dataclass
class CharLit:
    value: int = 0
    line: int = 0


@dataclass
class StringLit:
    value: str = ""
    line: int = 0


@dataclass
class BoolLit:
    value: bool = False
    line: int = 0


@dataclass
class NullLit:
    line: int = 0


@dataclass
class WeakExpr:
    value: Any = None
    line: int = 0


@dataclass
class Name:
    id: str = ""
    line: int = 0


@dataclass
class BinOp:
    op: str = ""
    left: Any = None
    right: Any = None
    line: int = 0


@dataclass
class UnaryOp:
    op: str = ""
    operand: Any = None
    line: int = 0


@dataclass
class Compare:
    op: str = ""
    left: Any = None
    right: Any = None
    line: int = 0


@dataclass
class BoolOp:
    op: str = ""
    left: Any = None
    right: Any = None
    line: int = 0


@dataclass
class Call:
    func: Any = None
    args: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class Attribute:
    obj: Any = None
    attr: str = ""
    line: int = 0


@dataclass
class Subscript:
    obj: Any = None
    index: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class Ternary:
    cond: Any = None
    then: Any = None
    other: Any = None
    line: int = 0


@dataclass
class ListLit:
    items: List[Any] = field(default_factory=list)
    line: int = 0


@dataclass
class StructLit:
    name: str = ""
    fields: List[Tuple[str, Any]] = field(default_factory=list)
    line: int = 0
