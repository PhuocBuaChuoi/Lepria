"""Lexer for Lepria: indentation-aware tokenizer (Python-style blocks)."""

from dataclasses import dataclass
from typing import Any, List, Optional

KEYWORDS = {
    "and", "as", "break", "class", "continue", "def", "elif", "else",
    "false", "for", "from", "if", "import", "in", "is", "not", "or",
    "pass", "return", "static", "struct", "true", "var", "while", "null",
    "weak", "type", "raise", "try", "catch", "emit", "extern",
}

# Longest first so greedy matching works.
OPERATORS = [
    "**=", "//=", "<<=", ">>=",
    "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "->", "**", "//", "<<", ">>",
    "+", "-", "*", "/", "%", "=", "<", ">", "&", "|", "^", "~",
    "(", ")", "[", "]", "{", "}", ",", ":", ".", "@", "?",
]

INT_SUFFIXES = {"L": "long", "l": "long"}
FLOAT_SUFFIXES = {"f": "float", "F": "float", "d": "ldouble", "D": "ldouble"}

ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
    "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f", "v": "\v",
}


@dataclass
class Token:
    kind: str
    value: Any
    line: int
    col: int
    type: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.type:
            return f"Token({self.kind}:{self.type}, {self.value!r} @{self.line}:{self.col})"
        return f"Token({self.kind}, {self.value!r} @{self.line}:{self.col})"


class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int, filename: str = "<src>"):
        super().__init__(f"{filename}:{line}:{col}: {msg}")
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, src: str, filename: str = "<src>"):
        if src.startswith("\ufeff"):
            src = src[1:]
        self.src = src
        self.filename = filename
        self.i = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        self.indents = [0]
        self.at_line_start = True
        self.bracket = 0

    # ------------------------------------------------------------- utilities
    def error(self, msg: str, line: Optional[int] = None, col: Optional[int] = None):
        raise LexError(msg, line or self.line, col or self.col, self.filename)

    def add(self, kind: str, value: Any, line: Optional[int] = None,
            col: Optional[int] = None, type: Optional[str] = None):
        self.tokens.append(Token(kind, value, line if line is not None else self.line,
                                 col if col is not None else self.col, type))

    def peek_char(self, off: int = 0) -> str:
        j = self.i + off
        return self.src[j] if 0 <= j < len(self.src) else ""

    def advance_char(self):
        self.i += 1
        self.col += 1

    # ------------------------------------------------------------- main loop
    def tokenize(self) -> List[Token]:
        n = len(self.src)
        while self.i < n:
            if self.at_line_start and self.bracket == 0:
                if not self.handle_indent():
                    continue

            c = self.src[self.i]

            if c == "\r":
                self.i += 1
                continue
            if c == "\n":
                self.handle_newline()
                continue
            if c in " \t":
                self.advance_char()
                continue
            if c == "#":
                if self.peek_char(1) == "{":
                    self.block_comment()
                else:
                    self.line_comment()
                continue
            if c.isdigit():
                self.number()
                continue
            if c.isalpha() or c == "_":
                self.identifier()
                continue
            if c in "\"'":
                self.string_or_char(c)
                continue
            if self.operator():
                continue
            self.error(f"unexpected character {c!r}")

        # finalize
        if self.tokens and self.tokens[-1].kind not in ("NEWLINE", "INDENT", "DEDENT"):
            self.add("NEWLINE", "\n")
        while len(self.indents) > 1:
            self.indents.pop()
            self.add("DEDENT", "")
        self.add("EOF", "")
        return self.tokens

    # ------------------------------------------------------------- whitespace
    def handle_indent(self) -> bool:
        """Return True if a content line still needs processing, False if consumed."""
        n = len(self.src)
        start_line, start_col = self.line, self.col
        col = 0
        while self.i < n and self.src[self.i] in " \t":
            col += 8 if self.src[self.i] == "\t" else 1
            self.i += 1
        self.col = col + 1

        if self.i >= n:
            return True

        ch = self.src[self.i]

        # blank line
        if ch in "\r\n":
            if ch == "\r":
                self.i += 1
            if self.i < n and self.src[self.i] == "\n":
                self.i += 1
            self.line += 1
            self.col = 1
            return False

        # comment-only line (line comment or block comment)
        if ch == "#":
            if self.peek_char(1) == "{":
                self.block_comment()
                # consume trailing newline if present
                if self.i < n and self.src[self.i] == "\r":
                    self.i += 1
                if self.i < n and self.src[self.i] == "\n":
                    self.i += 1
                    self.line += 1
                self.col = 1
                return False
            self.line_comment()
            if self.i < n and self.src[self.i] == "\r":
                self.i += 1
            if self.i < n and self.src[self.i] == "\n":
                self.i += 1
                self.line += 1
            self.col = 1
            return False

        # real content line: reconcile indentation
        top = self.indents[-1]
        if col > top:
            self.indents.append(col)
            self.add("INDENT", "", start_line, start_col)
        elif col < top:
            while len(self.indents) > 1 and col < self.indents[-1]:
                self.indents.pop()
                self.add("DEDENT", "", start_line, start_col)
            if col != self.indents[-1]:
                self.error("inconsistent dedent", start_line, start_col)

        self.at_line_start = False
        return True

    def handle_newline(self):
        if self.bracket == 0:
            if not (self.tokens and self.tokens[-1].kind == "NEWLINE"):
                self.add("NEWLINE", "\n")
            self.at_line_start = True
        self.i += 1
        self.line += 1
        self.col = 1

    def line_comment(self):
        while self.i < len(self.src) and self.src[self.i] not in "\r\n":
            self.i += 1

    def block_comment(self):
        depth = 0
        n = len(self.src)
        while self.i < n:
            if self.src.startswith("#{", self.i):
                depth += 1
                self.i += 2
                self.col += 2
            elif self.src.startswith("}#", self.i):
                depth -= 1
                self.i += 2
                self.col += 2
                if depth == 0:
                    return
            elif self.src[self.i] == "\n":
                self.i += 1
                self.line += 1
                self.col = 1
            else:
                self.i += 1
                self.col += 1
        self.error("unterminated block comment")

    # ------------------------------------------------------------- literals
    def number(self):
        src = self.src
        n = len(src)
        line, col = self.line, self.col
        start = self.i

        if src[self.i] == "0" and self.i + 1 < n and src[self.i + 1] in "xXbBoO":
            base_char = src[self.i + 1].lower()
            self.i += 2
            self.col += 2
            base = {"x": 16, "b": 2, "o": 8}[base_char]
            digits = []
            while self.i < n and (src[self.i].isalnum() or src[self.i] == "_"):
                digits.append(src[self.i])
                self.i += 1
                self.col += 1
            text = "".join(d for d in digits if d != "_")
            try:
                value = int(text, base)
            except ValueError:
                self.error(f"invalid number literal {''.join(digits)!r}", line, col)
            self.emit_number(value, "int", line, col)
            return

        has_dot = False
        has_exp = False
        while self.i < n:
            c = src[self.i]
            if c.isdigit() or c == "_":
                self.i += 1
                self.col += 1
            elif c == "." and not has_dot and not has_exp:
                # only a decimal point if followed by a digit
                if self.i + 1 < n and src[self.i + 1].isdigit():
                    has_dot = True
                    self.i += 1
                    self.col += 1
                else:
                    break
            elif c in "eE" and not has_exp:
                has_exp = True
                self.i += 1
                self.col += 1
                if self.i < n and src[self.i] in "+-":
                    self.i += 1
                    self.col += 1
            else:
                break

        text = src[start:self.i].replace("_", "")
        suffix = ""
        if self.i < n and src[self.i] in "lLfFdD":
            suffix = src[self.i]
            self.i += 1
            self.col += 1

        if suffix in INT_SUFFIXES:
            try:
                value = int(text)
            except ValueError:
                self.error(f"invalid integer literal {text!r}", line, col)
            self.emit_number(value, INT_SUFFIXES[suffix], line, col)
            return

        if suffix in FLOAT_SUFFIXES:
            try:
                value = float(text)
            except ValueError:
                self.error(f"invalid float literal {text!r}", line, col)
            self.emit_number(value, FLOAT_SUFFIXES[suffix], line, col)
            return

        if has_dot or has_exp:
            self.emit_number(float(text), "double", line, col)
        else:
            self.emit_number(int(text), "int", line, col)

    def emit_number(self, value, typ: str, line: int, col: int):
        kind = "FLOAT" if typ in ("float", "double", "ldouble") else "INT"
        self.add(kind, value, line, col, typ)

    def identifier(self):
        src = self.src
        n = len(src)
        line, col = self.line, self.col
        start = self.i
        while self.i < n and (src[self.i].isalnum() or src[self.i] == "_"):
            self.i += 1
            self.col += 1
        text = src[start:self.i]
        if text in KEYWORDS:
            self.add("KW", text, line, col)
        else:
            self.add("IDENT", text, line, col)

    def string_or_char(self, quote: str):
        line, col = self.line, self.col
        triple = self.src.startswith(quote * 3, self.i)
        if triple:
            self.i += 3
            self.col += 3
            end_seq = quote * 3
        else:
            self.i += 1
            self.col += 1
            end_seq = quote

        out = []
        n = len(self.src)
        while self.i < n:
            if self.src.startswith(end_seq, self.i):
                self.i += len(end_seq)
                self.col += len(end_seq)
                break
            c = self.src[self.i]
            if c == "\n":
                if not triple:
                    self.error("unterminated string literal", line, col)
                out.append("\n")
                self.i += 1
                self.line += 1
                self.col = 1
                continue
            if c == "\\":
                self.i += 1
                self.col += 1
                out.append(self.read_escape())
                continue
            out.append(c)
            self.i += 1
            self.col += 1
        else:
            self.error("unterminated string literal", line, col)

        text = "".join(out)
        if quote == "'" and not triple:
            if len(text) != 1:
                self.error("char literal must contain exactly one character", line, col)
            self.add("CHAR", ord(text), line, col)
        else:
            self.add("STRING", text, line, col)

    def read_escape(self) -> str:
        if self.i >= len(self.src):
            self.error("unterminated escape sequence")
        c = self.src[self.i]
        self.i += 1
        self.col += 1
        if c in ESCAPES:
            return ESCAPES[c]
        if c == "x":
            hexs = self.src[self.i:self.i + 2]
            if len(hexs) != 2:
                self.error("invalid \\x escape")
            self.i += 2
            self.col += 2
            return chr(int(hexs, 16))
        if c == "u":
            hexs = self.src[self.i:self.i + 4]
            if len(hexs) != 4:
                self.error("invalid \\u escape")
            self.i += 4
            self.col += 4
            return chr(int(hexs, 16))
        return c

    # ------------------------------------------------------------- operators
    def operator(self) -> bool:
        for op in OPERATORS:
            if self.src.startswith(op, self.i):
                self.add("OP", op)
                self.i += len(op)
                self.col += len(op)
                if op in "([{":
                    self.bracket += 1
                elif op in ")]}":
                    self.bracket = max(0, self.bracket - 1)
                return True
        return False
