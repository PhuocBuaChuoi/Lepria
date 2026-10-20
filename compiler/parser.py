"""Recursive-descent parser for Lepria."""

from typing import Any, List, Optional

from . import ast_nodes as A
from .lexer import Token

PRIM_TYPES = {"int", "long", "char", "float", "double", "list", "string"}
AUG_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>="}
COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}


class ParseError(Exception):
    def __init__(self, msg: str, line: int, col: int, filename: str = "<src>"):
        super().__init__(f"{filename}:{line}:{col}: {msg}")
        self.line = line
        self.col = col


class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<src>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0

    # ------------------------------------------------------------- utilities
    def peek(self, off: int = 0) -> Token:
        j = self.pos + off
        if j >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[j]

    def error(self, msg: str, tok: Optional[Token] = None):
        t = tok or self.peek()
        raise ParseError(msg, t.line, t.col, self.filename)

    def at(self, kind: str, value: Any = None) -> bool:
        t = self.peek()
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def at_op(self, value: str) -> bool:
        return self.at("OP", value)

    def at_kw(self, value: str) -> bool:
        return self.at("KW", value)

    def advance(self) -> Token:
        t = self.peek()
        if t.kind != "EOF":
            self.pos += 1
        return t

    def match(self, kind: str, value: Any = None) -> Optional[Token]:
        if self.at(kind, value):
            return self.advance()
        return None

    def expect(self, kind: str, value: Any = None, what: str = "") -> Token:
        if self.at(kind, value):
            return self.advance()
        t = self.peek()
        got = t.value if t.value != "" else t.kind
        self.error(f"expected {what or value or kind}, got {got!r}", t)

    def skip_newlines(self):
        while self.at("NEWLINE"):
            self.advance()

    def consume_newline(self):
        if self.at("NEWLINE"):
            self.advance()

    # ------------------------------------------------------------- entry
    def parse(self) -> A.Program:
        stmts = []
        self.skip_newlines()
        while not self.at("EOF"):
            if self.at("DEDENT"):
                self.advance()
                continue
            stmts.append(self.statement())
            self.skip_newlines()
        return A.Program(stmts=stmts)

    # ------------------------------------------------------------- statements
    def statement(self):
        # labeled loop:  name: while ... / name: for ...
        if self.at("IDENT") and self.peek(1).kind == "OP" and self.peek(1).value == ":":
            nxt = self.peek(2)
            if nxt.kind == "KW" and nxt.value in ("while", "for"):
                label = self.advance().value
                self.advance()  # ':'
                if self.at_kw("while"):
                    return self.parse_while(label)
                return self.parse_for(label)

        if self.at_kw("if"):
            return self.parse_if()
        if self.at_kw("while"):
            return self.parse_while(None)
        if self.at_kw("for"):
            return self.parse_for(None)
        if self.at_kw("def"):
            return self.parse_func()
        if self.at_kw("return"):
            return self.parse_return()
        if self.at_kw("break"):
            t = self.advance()
            label = self.advance().value if self.at("IDENT") else None
            self.consume_newline()
            return A.Break(label=label, line=t.line)
        if self.at_kw("continue"):
            t = self.advance()
            label = self.advance().value if self.at("IDENT") else None
            self.consume_newline()
            return A.Continue(label=label, line=t.line)
        if self.at_kw("pass"):
            t = self.advance()
            self.consume_newline()
            return A.Pass(line=t.line)
        if self.at_kw("raise"):
            t = self.advance()
            val = self.parse_expr()
            self.consume_newline()
            return A.Raise(value=val, line=t.line)
        if self.at_kw("import"):
            return self.parse_import()
        if self.at_kw("from"):
            return self.parse_from_import()
        if self.at_kw("class") or self.at_kw("struct"):
            return self.parse_class()
        if self.at_kw("type"):
            return self.parse_type_alias()
        if self.at_kw("var") or self.at_kw("static"):
            return self.parse_var_decl()

        return self.parse_simple_stmt()

    def parse_simple_stmt(self):
        t = self.peek()

        # annotated assignment / declaration:  x: int [= expr]
        if self.at("IDENT") and self.peek(1).kind == "OP" and self.peek(1).value == ":":
            name = self.advance().value
            self.advance()  # ':'
            typ = self.parse_type()
            value = None
            if self.at_op("="):
                self.advance()
                value = self.parse_expr()
            self.consume_newline()
            return A.VarDecl(name=name, type=typ, value=value, line=t.line)

        expr = self.parse_expr()

        if self.at_op("=") or (self.at("OP") and self.peek().value in AUG_OPS):
            op = self.advance().value
            value = self.parse_expr()
            self.consume_newline()
            return A.Assign(target=expr, op=op, value=value, line=t.line)

        self.consume_newline()
        return A.ExprStmt(expr=expr, line=t.line)

    def parse_var_decl(self):
        t = self.advance()
        is_static = t.value == "static"
        name = self.expect("IDENT", what="variable name").value
        typ = None
        if self.at_op(":"):
            self.advance()
            typ = self.parse_type()
        value = None
        if self.at_op("="):
            self.advance()
            value = self.parse_expr()
        self.consume_newline()
        return A.VarDecl(name=name, type=typ, value=value, is_static=is_static, line=t.line)

    def parse_if(self):
        t = self.expect("KW", "if")
        branches = []
        cond = self.parse_expr()
        body = self.parse_block()
        branches.append((cond, body))
        while self.at_kw("elif"):
            self.advance()
            cond = self.parse_expr()
            body = self.parse_block()
            branches.append((cond, body))
        else_body = None
        if self.at_kw("else"):
            self.advance()
            else_body = self.parse_block()
        return A.If(branches=branches, else_body=else_body, line=t.line)

    def parse_while(self, label: Optional[str]):
        t = self.expect("KW", "while")
        cond = self.parse_expr()
        body = self.parse_block()
        return A.While(cond=cond, body=body, label=label, line=t.line)

    def parse_for(self, label: Optional[str]):
        t = self.expect("KW", "for")

        names = [self.expect("IDENT", what="loop variable").value]
        if self.at_op(","):
            self.advance()
            names.append(self.expect("IDENT", what="loop variable").value)

        self.expect("KW", "in", what="'in'")
        iterable = self.parse_expr()
        body = self.parse_block()

        if len(names) == 1 and isinstance(iterable, A.ListLit) and len(iterable.items) in (2, 3):
            items = iterable.items
            start = items[0]
            end = items[1]
            step = items[2] if len(items) == 3 else None
            return A.ForRange(var=names[0], start=start, end=end, step=step,
                              body=body, label=label, line=t.line)

        return A.ForEach(var=",".join(names), iterable=iterable, body=body, line=t.line)

    def parse_return(self):
        t = self.expect("KW", "return")
        value = None
        if not self.at("NEWLINE") and not self.at("DEDENT") and not self.at("EOF"):
            value = self.parse_expr()
        self.consume_newline()
        return A.Return(value=value, line=t.line)

    def parse_import(self):
        t = self.expect("KW", "import")
        module = self.parse_dotted_name()
        alias = None
        if self.at_kw("as"):
            self.advance()
            alias = self.expect("IDENT", what="alias").value
        self.consume_newline()
        return A.Import(module=module, alias=alias, line=t.line)

    def parse_from_import(self):
        t = self.expect("KW", "from")
        module = self.parse_dotted_name()
        self.expect("KW", "import", what="'import'")
        names = [self.expect("IDENT", what="name").value]
        while self.at_op(","):
            self.advance()
            names.append(self.expect("IDENT", what="name").value)
        self.consume_newline()
        return A.FromImport(module=module, names=names, line=t.line)

    def parse_class(self):
        t = self.advance()
        kind = t.value
        name = self.expect("IDENT", what="type name").value
        self.expect("OP", ":")
        self.expect("NEWLINE")
        self.expect("INDENT")
        fields = []
        methods = []

        self.skip_newlines()
        while not self.at("DEDENT") and not self.at("EOF"):
            if self.at_kw("def"):
                methods.append(self.parse_func())
            elif self.at("IDENT") and self.peek(1).kind == "OP" and self.peek(1).value == ":":
                fname = self.advance().value
                self.advance()
                ftype = self.parse_type()
                fields.append((fname, ftype))
                self.consume_newline()
            elif self.at_kw("pass"):
                self.advance()
                self.consume_newline()
            else:
                # tolerate other statements inside a class body
                self.statement()
            self.skip_newlines()

        self.expect("DEDENT")
        return A.ClassDecl(name=name, kind=kind, fields=fields, methods=methods, line=t.line)

    def parse_type_alias(self):
        t = self.expect("KW", "type")
        name = self.expect("IDENT", what="type name").value
        self.expect("OP", "=")
        typ = self.parse_type()
        self.consume_newline()
        return A.TypeAlias(name=name, type=typ, line=t.line)

    def parse_func(self):
        t = self.expect("KW", "def")
        name = self.expect("IDENT", what="function name").value
        self.expect("OP", "(")
        params = []
        if not self.at_op(")"):
            while True:
                params.append(self.parse_param())
                if self.at_op(","):
                    self.advance()
                    continue
                break
        self.expect("OP", ")")
        ret_type = None
        if self.at_op("->"):
            self.advance()
            ret_type = self.parse_type()
        body = self.parse_block()
        return A.FuncDef(name=name, params=params, ret_type=ret_type, body=body, line=t.line)

    def parse_param(self):
        t = self.expect("IDENT", what="parameter name")
        typ = None
        default = None
        if self.at_op(":"):
            self.advance()
            typ = self.parse_type()
        if self.at_op("="):
            self.advance()
            default = self.parse_expr()
        return A.Param(name=t.value, type=typ, default=default, line=t.line)

    def parse_block(self) -> List[Any]:
        self.expect("OP", ":")
        self.expect("NEWLINE")
        self.expect("INDENT")
        body = []
        self.skip_newlines()
        while not self.at("DEDENT") and not self.at("EOF"):
            body.append(self.statement())
            self.skip_newlines()
        self.expect("DEDENT")
        return body

    def parse_dotted_name(self) -> str:
        parts = [self.expect("IDENT", what="name").value]
        while self.at_op("."):
            self.advance()
            parts.append(self.expect("IDENT", what="name").value)
        return ".".join(parts)

    # ------------------------------------------------------------- types
    def parse_type(self) -> str:
        t = self.peek()
        if self.peek().value in ("long", "double") or self.peek().kind == "IDENT":
            pass

        if self.at("IDENT") or self.at("KW"):
            v = self.peek().value
            if v == "long" and self.peek(1).value == "double" and self.peek(1).kind in ("IDENT", "KW"):
                self.advance()
                self.advance()
                base = "ldouble"
            elif v == "list":
                self.advance()
                self.expect("OP", "[")
                inner = self.parse_type()
                dims = []
                while self.at_op(","):
                    self.advance()
                    num = self.expect("INT", what="dimension")
                    dims.append(str(num.value))
                self.expect("OP", "]")
                base = f"list[{inner}" + ("".join("," + d for d in dims)) + "]"
            elif v == "weak":
                self.advance()
                inner = self.parse_type()
                base = f"weak[{inner}]"
            elif v == "ptr":
                self.advance()
                if self.at_op("["):
                    self.advance()
                    inner = self.parse_type()
                    self.expect("OP", "]")
                    base = f"ptr[{inner}]"
                else:
                    base = "ptr"
            else:
                self.advance()
                base = v
                if base == "double" and base == "double":
                    base = "double"
        else:
            self.error("expected a type", t)

        if self.at_op("?"):
            self.advance()
            base += "?"

        dims = []
        while self.at_op("["):
            self.advance()
            while True:
                num = self.expect("INT", what="array size")
                dims.append(str(num.value))
                if self.at_op(","):
                    self.advance()
                    continue
                break
            self.expect("OP", "]")

        if dims:
            base = f"array[{base}," + ",".join(dims) + "]"
        return base

    # ------------------------------------------------------------- expressions
    def parse_expr(self):
        node = self.parse_or()
        if self.at_kw("if"):
            self.advance()
            cond = self.parse_or()
            self.expect("KW", "else", what="'else'")
            other = self.parse_expr()
            return A.Ternary(cond=cond, then=node, other=other, line=node.line)
        return node

    def parse_or(self):
        node = self.parse_and()
        while self.at_kw("or"):
            t = self.advance()
            right = self.parse_and()
            node = A.BoolOp(op="or", left=node, right=right, line=t.line)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.at_kw("and"):
            t = self.advance()
            right = self.parse_not()
            node = A.BoolOp(op="and", left=node, right=right, line=t.line)
        return node

    def parse_not(self):
        if self.at_kw("not"):
            t = self.advance()
            return A.UnaryOp(op="not", operand=self.parse_not(), line=t.line)
        return self.parse_comparison()

    def parse_comparison(self):
        first = self.parse_bitor()
        operands = [first]
        ops = []

        while True:
            op = self.try_comp_op()
            if op is None:
                break
            ops.append(op)
            operands.append(self.parse_bitor())

        if not ops:
            return first

        node = A.Compare(op=ops[0], left=operands[0], right=operands[1], line=first.line)
        for i in range(1, len(ops)):
            nxt = A.Compare(op=ops[i], left=operands[i], right=operands[i + 1],
                            line=first.line)
            node = A.BoolOp(op="and", left=node, right=nxt, line=first.line)
        return node

    def try_comp_op(self) -> Optional[str]:
        t = self.peek()
        if t.kind == "OP" and t.value in COMPARE_OPS:
            self.advance()
            return t.value
        if t.kind == "KW" and t.value == "in":
            self.advance()
            return "in"
        if t.kind == "KW" and t.value == "is":
            self.advance()
            if self.at_kw("not"):
                self.advance()
                return "is not"
            return "is"
        if (t.kind == "KW" and t.value == "not"
                and self.peek(1).kind == "KW" and self.peek(1).value == "in"):
            self.advance()
            self.advance()
            return "not in"
        return None

    def parse_bitor(self):
        node = self.parse_bitxor()
        while self.at_op("|"):
            t = self.advance()
            node = A.BinOp(op="|", left=node, right=self.parse_bitxor(), line=t.line)
        return node

    def parse_bitxor(self):
        node = self.parse_bitand()
        while self.at_op("^"):
            t = self.advance()
            node = A.BinOp(op="^", left=node, right=self.parse_bitand(), line=t.line)
        return node

    def parse_bitand(self):
        node = self.parse_shift()
        while self.at_op("&"):
            t = self.advance()
            node = A.BinOp(op="&", left=node, right=self.parse_shift(), line=t.line)
        return node

    def parse_shift(self):
        node = self.parse_arith()
        while self.at_op("<<") or self.at_op(">>"):
            t = self.advance()
            node = A.BinOp(op=t.value, left=node, right=self.parse_arith(), line=t.line)
        return node

    def parse_arith(self):
        node = self.parse_term()
        while self.at_op("+") or self.at_op("-"):
            t = self.advance()
            node = A.BinOp(op=t.value, left=node, right=self.parse_term(), line=t.line)
        return node

    def parse_term(self):
        node = self.parse_unary()
        while self.at_op("*") or self.at_op("/") or self.at_op("//") or self.at_op("%"):
            t = self.advance()
            node = A.BinOp(op=t.value, left=node, right=self.parse_unary(), line=t.line)
        return node

    def parse_unary(self):
        if self.at_kw("weak"):
            t = self.advance()
            return A.WeakExpr(value=self.parse_unary(), line=t.line)
        if self.at_op("+") or self.at_op("-") or self.at_op("~"):
            t = self.advance()
            return A.UnaryOp(op=t.value, operand=self.parse_unary(), line=t.line)
        return self.parse_power()

    def parse_power(self):
        base = self.parse_postfix()
        if self.at_op("**"):
            t = self.advance()
            exp = self.parse_unary()
            return A.BinOp(op="**", left=base, right=exp, line=t.line)
        return base

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.at_op("("):
                t = self.advance()
                args = []
                if not self.at_op(")"):
                    while True:
                        args.append(self.parse_expr())
                        if self.at_op(","):
                            self.advance()
                            continue
                        break
                self.expect("OP", ")")
                node = A.Call(func=node, args=args, line=t.line)
            elif self.at_op("."):
                t = self.advance()
                attr = self.expect("IDENT", what="attribute").value
                node = A.Attribute(obj=node, attr=attr, line=t.line)
            elif self.at_op("["):
                t = self.advance()
                index = [self.parse_expr()]
                while self.at_op(","):
                    self.advance()
                    index.append(self.parse_expr())
                self.expect("OP", "]")
                node = A.Subscript(obj=node, index=index, line=t.line)
            else:
                break
        return node

    def parse_primary(self):
        t = self.peek()

        if t.kind == "INT":
            self.advance()
            return A.IntLit(value=t.value, type=t.type or "int", line=t.line)
        if t.kind == "FLOAT":
            self.advance()
            return A.FloatLit(value=t.value, type=t.type or "double", line=t.line)
        if t.kind == "CHAR":
            self.advance()
            return A.CharLit(value=t.value, line=t.line)
        if t.kind == "STRING":
            self.advance()
            return A.StringLit(value=t.value, line=t.line)
        if t.kind == "KW" and t.value in ("true", "false"):
            self.advance()
            return A.BoolLit(value=(t.value == "true"), line=t.line)
        if t.kind == "KW" and t.value == "null":
            self.advance()
            return A.NullLit(line=t.line)

        if t.kind == "IDENT":
            self.advance()
            if self.at_op("{"):
                # struct literal:  Name { field = expr, ... }
                self.advance()
                fields = []
                if not self.at_op("}"):
                    while True:
                        fname = self.expect("IDENT", what="field name").value
                        self.expect("OP", "=")
                        fields.append((fname, self.parse_expr()))
                        if self.at_op(","):
                            self.advance()
                            continue
                        break
                self.expect("OP", "}")
                return A.StructLit(name=t.value, fields=fields, line=t.line)
            return A.Name(id=t.value, line=t.line)

        if self.at_op("("):
            self.advance()
            node = self.parse_expr()
            self.expect("OP", ")")
            return node

        if self.at_op("["):
            self.advance()
            items = []
            if not self.at_op("]"):
                while True:
                    items.append(self.parse_expr())
                    if self.at_op(","):
                        self.advance()
                        continue
                    break
            self.expect("OP", "]")
            return A.ListLit(items=items, line=t.line)

        self.error(f"unexpected token {t.value!r}", t)
