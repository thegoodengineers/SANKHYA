#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hock-Schittkowski test problems: published AMPL models -> .nl files (NLP stage 2).

    python bench/runners/hs_mod_to_nl.py MOD_DIR OUT_DIR hs071 hs035 ...

WHERE THE PROBLEMS COME FROM. Vanderbei's AMPL transcriptions of the CUTE collection,
https://vanderbei.princeton.edu/ampl/nlmodels/cute/, one file per problem of W. Hock and K.
Schittkowski, "Test Examples for Nonlinear Programming Codes", Lecture Notes in Economics and
Mathematical Systems 187, Springer (1981). Each file states the model, the problem's standard
starting point (`let x[i] := ...`) and the published optimal objective as the last line
(`display obj - 17.0140173;`). Instance data and a published reference optimum are allowed
by ENGINEERING_RULES.md, item 6; no solver code is involved.

WHY PARSED, NOT TYPED. A model retyped by hand is a second transcription that can be wrong
in a way nothing downstream notices. This reads the published text with a small recursive-
descent parser for exactly the AMPL subset those files use - one indexed variable, bounds,
`minimize`/`maximize`, `subject to` with one or two relations, `sum`/`prod` over an integer
range, + - * / ^, sqrt exp log sin cos, `let` and `display obj - REF` - and REFUSES any file
with anything else (`param` data, `abs`, integer variables, sets), rather than guessing.

WHY PYOMO WRITES THE .nl. Pyomo (a modeling language, not a solver) builds the model and
writes the file with its own .nl writer, so the files the solver reads were written by a
third party's implementation of the format, not by this project's - which is what makes them
a test of src/nlp/nl_reader.cpp rather than a round trip through it. Pyomo is used here only,
at data-generation time; the solver never links or calls it.

Output per problem: OUT_DIR/<name>.nl with .row/.col names beside it, and one line of
OUT_DIR/REFERENCE.csv: problem, reference objective, sense, variables, constraints, source
URL, sha256 of the .mod as downloaded, sha256 of the .nl written.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from pathlib import Path

SOURCE = "https://vanderbei.princeton.edu/ampl/nlmodels/cute/"
# A number never takes the first dot of "..": 1..4 is 1, .., 4.
TOKEN = re.compile(r"\s*(?:(\d+(?:\.(?!\.)\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|([A-Za-z_]\w*)"
                   r"|(<=|>=|:=|\.\.|[-+*/^(){}\[\],=<>:]))")


class Unsupported(Exception):
    """The file uses something outside the subset this converter reads."""


def tokenize(text: str) -> list[str]:
    out, pos = [], 0
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match or match.end() == pos:
            if text[pos:].strip() == "":
                break
            raise Unsupported(f"cannot tokenize near {text[pos:pos + 20]!r}")
        out.append(match.group(0).strip())
        pos = match.end()
    return [t for t in out if t]


class Parser:
    """Expressions to a tree: ('num', v) ('ref', name, index) ('neg', a) ('bin', op, a, b)
    ('call', f, a) ('iter', op, var, lo, hi, body)."""

    FUNCS = {"sqrt", "exp", "log", "sin", "cos"}

    def __init__(self, tokens: list[str]) -> None:
        self.t, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, want: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise Unsupported(f"expected {want!r}, found {tok!r}")
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            node = ("bin", self.take(), node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in ("*", "/"):
            node = ("bin", self.take(), node, self.unary())
        return node

    def unary(self):
        if self.peek() == "-":
            self.take()
            return ("neg", self.unary())
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.primary()
        if self.peek() == "^":
            self.take()
            return ("bin", "^", base, self.unary())  # right-associative, binds tighter than -
        return base

    def primary(self):
        tok = self.take()
        if tok == "(":
            node = self.expr()
            self.take(")")
            return node
        if tok[0].isdigit() or tok[0] == ".":
            return ("num", float(tok))
        if tok in ("sum", "prod"):
            self.take("{")
            var = self.take()
            self.take("in")
            lo = self.expr()
            self.take("..")
            hi = self.expr()
            self.take("}")
            return ("iter", tok, var, lo, hi, self.term())
        if tok in self.FUNCS:
            self.take("(")
            arg = self.expr()
            self.take(")")
            return ("call", tok, arg)
        if re.match(r"[A-Za-z_]\w*$", tok):
            if self.peek() == "[":
                self.take("[")
                index = self.expr()
                self.take("]")
                return ("ref", tok, index)
            return ("ref", tok, None)
        raise Unsupported(f"unexpected token {tok!r}")


def evaluate(node, env: dict, lib):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "neg":
        return -evaluate(node[1], env, lib)
    if kind == "bin":
        a, b = evaluate(node[2], env, lib), evaluate(node[3], env, lib)
        return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
                "^": lambda: a ** b}[node[1]]()
    if kind == "call":
        return getattr(lib, node[1])(evaluate(node[2], env, lib))
    if kind == "iter":
        lo, hi = int(evaluate(node[3], env, lib)), int(evaluate(node[4], env, lib))
        total = 0 if node[1] == "sum" else 1
        for i in range(lo, hi + 1):
            value = evaluate(node[5], {**env, node[2]: i}, lib)
            total = total + value if node[1] == "sum" else total * value
        return total
    if kind == "ref":
        name, index = node[1], node[2]
        if index is None:
            if name in env:
                return env[name]
            raise Unsupported(f"unknown name {name!r}")
        target = env.get("__vars__", {}).get(name)
        if target is None:
            raise Unsupported(f"unknown indexed name {name!r}")
        return target[int(evaluate(index, env, lib))]
    raise Unsupported(f"unknown node {kind}")


def statements(text: str) -> list[str]:
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return [s.strip() for s in text.split(";") if s.strip()]


def convert(mod_path: Path, out_dir: Path) -> dict:
    import pyomo.environ as pyo

    text = mod_path.read_text()
    model = pyo.ConcreteModel(name=mod_path.stem)
    variables, starts, reference, sense = {}, {}, None, None
    constraints = 0
    for stmt in statements(text):
        head = stmt.split()[0]
        if head in ("param", "set", "check", "fix", "integer", "binary", "printf", "function"):
            raise Unsupported(f"statement {head!r}")
        if head in ("solve", "option", "expand", "write", "reset"):
            continue
        if stmt == "data":
            continue  # a bare `data;` switches mode and states nothing
        if head == "var":
            tokens = tokenize(stmt[3:])
            name, rest = tokens[0], tokens[1:]
            if not rest or rest[0] != "{":
                raise Unsupported("a scalar variable")
            close = rest.index("}")
            inner, rest = rest[1:close], rest[close + 1:]
            index_name = None
            if len(inner) == 5 and inner[1] == "in":
                index_name, inner = inner[0], inner[2:]
            if len(inner) != 3 or inner[1] != ".." or not inner[0].isdigit() \
                    or not inner[2].isdigit():
                raise Unsupported("a variable index other than a literal a..b")
            first, last = int(inner[0]), int(inner[2])
            # Bounds are expressions, possibly of the index name (hs045: <= i), split on ",".
            bound_trees = {}
            k = 0
            while k < len(rest):
                tok = rest[k]
                if tok in (">=", "<="):
                    p = Parser(rest[k + 1:])
                    bound_trees[tok] = p.expr()
                    k += 1 + p.i
                elif tok == ",":
                    k += 1
                else:
                    raise Unsupported(f"var attribute {tok!r}")

            def bounds(_model, i, trees=bound_trees, index_name=index_name):
                env = {index_name: i} if index_name else {}
                lo = evaluate(trees[">="], env, math) if ">=" in trees else None
                hi = evaluate(trees["<="], env, math) if "<=" in trees else None
                return (lo, hi)

            setattr(model, name, pyo.Var(pyo.RangeSet(first, last), bounds=bounds))
            variables[name] = getattr(model, name)
            continue
        if head in ("minimize", "maximize"):
            if sense is not None:
                raise Unsupported("a second objective")
            sense = head
            body = stmt.split(":", 1)[1]
            p = Parser(tokenize(body))
            expr = evaluate(p.expr(), {"__vars__": variables}, pyo)
            if p.peek() is not None:
                raise Unsupported(f"trailing {p.peek()!r} in the objective")
            model.obj = pyo.Objective(expr=expr, sense=pyo.minimize if head == "minimize"
                                      else pyo.maximize)
            continue
        if head == "subject" or head == "s.t.":
            label, body = stmt.split(":", 1)
            p = Parser(tokenize(body))
            parts, relations = [p.expr()], []
            while p.peek() in ("<=", ">=", "=", "=="):
                relations.append(p.take())
                parts.append(p.expr())
            if p.peek() is not None or not relations:
                raise Unsupported(f"constraint {label!r}")
            env = {"__vars__": variables}
            values = [evaluate(x, env, pyo) for x in parts]
            constraints += 1
            if len(relations) == 1:
                a, b = values
                rel = relations[0]
                expr = a <= b if rel == "<=" else (a >= b if rel == ">=" else a == b)
            elif relations == ["<=", "<="]:
                expr = pyo.inequality(values[0], values[1], values[2])
            elif relations == [">=", ">="]:
                expr = pyo.inequality(values[2], values[1], values[0])
            else:
                raise Unsupported(f"relations {relations}")
            setattr(model, f"c{constraints}_{label.split()[-1]}", pyo.Constraint(expr=expr))
            continue
        if head == "let":
            m = re.match(r"let\s*(?:\{\s*(\w+)\s+in\s+(\d+)\s*\.\.\s*(\d+)\s*\})?\s*"
                         r"(\w+)\s*\[\s*([^\]]+?)\s*\]\s*:=\s*(.+)$", stmt, re.S)
            if not m:
                raise Unsupported(f"let statement {stmt!r}")
            loop = [None] if m.group(1) is None else range(int(m.group(2)), int(m.group(3)) + 1)
            for i in loop:
                env = {} if i is None else {m.group(1): i}
                where = int(evaluate(Parser(tokenize(m.group(5))).expr(), env, math))
                starts[(m.group(4), where)] = evaluate(Parser(tokenize(m.group(6))).expr(), env,
                                                       math)
            continue
        if head == "display":
            body = stmt[len("display"):].strip()
            if body.startswith("obj") and body != "obj":
                p = Parser(tokenize(body))
                offset = evaluate(p.expr(), {"obj": 0.0}, math)
                reference = -offset
            continue
        raise Unsupported(f"statement {head!r}")
    if sense is None or reference is None or len(variables) != 1:
        raise Unsupported("no objective, no reference value, or not exactly one variable")
    (name, var), = variables.items()
    for (vname, i), v in starts.items():
        variables[vname][i].value = v
    out_dir.mkdir(parents=True, exist_ok=True)
    nl = out_dir / f"{mod_path.stem}.nl"
    model.write(str(nl), format="nl", io_options={"symbolic_solver_labels": True})
    return {
        "problem": mod_path.stem,
        "reference_objective": repr(float(reference)),
        "sense": sense,
        "variables": len(var),
        "constraints": constraints,
        "source": SOURCE + mod_path.name,
        "mod_sha256": hashlib.sha256(mod_path.read_bytes()).hexdigest(),
        "nl_sha256": hashlib.sha256(nl.read_bytes()).hexdigest(),
    }


def main() -> int:
    mod_dir, out_dir, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    rows, refused = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            rows.append(convert(mod_dir / f"{name}.mod", out_dir))
        except Unsupported as why:
            refused.append((name, str(why)))
        except (ValueError, KeyError, IndexError, TypeError, OSError) as why:
            # Named, not swallowed: the problem is left out and the reason printed.
            refused.append((name, f"{type(why).__name__}: {why}"))
    with open(out_dir / "REFERENCE.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["problem"])
        writer.writeheader()
        writer.writerows(rows)
    for name, why in refused:
        print(f"refused {name}: {why}")
    print(f"converted {len(rows)}, refused {len(refused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
