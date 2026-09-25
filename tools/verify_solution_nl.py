# SPDX-License-Identifier: Apache-2.0
"""An independent reader and evaluator of AMPL .nl models, for verify_solution.py (NLP stage 2).

INDEPENDENT OF THE SOLVER. src/nlp/nl_reader.cpp reads these files for the solver; this is a
second reading, in another language, written from the same public specification (D. M. Gay,
"Writing .nl Files", SAND2005-7907P, 2005; the variable order of "Hooking Your Solver to
AMPL", Tables 3-4) and sharing no code with it. Derivatives are NOT taken from the solver
either: every expression is evaluated here in forward mode over dual numbers - each value
carries its gradient, and every operation applies the chain rule to it (Griewank and Walther,
"Evaluating Derivatives", 2nd ed., SIAM 2008, ch. 3) - so the checker's gradient and Jacobian
come from arithmetic the solver never did.

Only what the solver reads is read: the text form, and the operators plus, minus, mult, div,
pow, neg, sumlist, sqrt, exp, log, log10, sin, cos. Anything else raises ValueError.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

INF = math.inf


class Dual:
    """A value and its sparse gradient {column: derivative}."""

    __slots__ = ("v", "g")

    def __init__(self, v: float, g: dict[int, float] | None = None) -> None:
        self.v = v
        self.g = g or {}

    def scaled(self, factor: float) -> dict[int, float]:
        return {j: factor * d for j, d in self.g.items()}


def _combine(a: dict[int, float], fa: float, b: dict[int, float], fb: float) -> dict[int, float]:
    out = {j: fa * d for j, d in a.items()} if fa != 0.0 else {}
    if fb != 0.0:
        for j, d in b.items():
            out[j] = out.get(j, 0.0) + fb * d
    return out


def _unary(a: Dual, value: float, slope: float) -> Dual:
    if not math.isfinite(value) or not math.isfinite(slope):
        raise ValueError("an operation overflowed or has no derivative here")
    return Dual(value, a.scaled(slope))


def apply(code: int, args: list[Dual]) -> Dual:
    if code == 0:
        a, b = args
        return Dual(a.v + b.v, _combine(a.g, 1.0, b.g, 1.0))
    if code == 1:
        a, b = args
        return Dual(a.v - b.v, _combine(a.g, 1.0, b.g, -1.0))
    if code == 2:
        a, b = args
        return Dual(a.v * b.v, _combine(a.g, b.v, b.g, a.v))
    if code == 3:
        a, b = args
        if b.v == 0.0:
            raise ValueError("division by zero")
        return Dual(a.v / b.v, _combine(a.g, 1.0 / b.v, b.g, -a.v / (b.v * b.v)))
    if code == 5:
        a, b = args
        if not b.g:  # a constant exponent
            p = b.v
            if a.v < 0.0 and p != math.floor(p):
                raise ValueError("a negative base under a non-integer power")
            if a.v == 0.0 and p < 1.0 and p != 0.0:
                raise ValueError("0 raised to a power below 1 has no derivative")
            return _unary(a, a.v ** p, p * a.v ** (p - 1.0) if p != 0.0 else 0.0)
        if a.v <= 0.0:
            raise ValueError("a variable exponent over a base that is not positive")
        value = a.v ** b.v
        return Dual(value, _combine(a.g, b.v * a.v ** (b.v - 1.0), b.g, value * math.log(a.v)))
    if code == 16:
        return Dual(-args[0].v, args[0].scaled(-1.0))
    if code == 39:
        a = args[0]
        if a.v < 0.0:
            raise ValueError("sqrt of a negative number")
        root = math.sqrt(a.v)
        return _unary(a, root, 0.5 / root if root > 0.0 else INF)
    if code == 41:
        return _unary(args[0], math.sin(args[0].v), math.cos(args[0].v))
    if code == 46:
        return _unary(args[0], math.cos(args[0].v), -math.sin(args[0].v))
    if code in (42, 43):
        a = args[0]
        if a.v <= 0.0:
            raise ValueError("log of a non-positive number")
        factor = 1.0 / math.log(10.0) if code == 42 else 1.0
        return _unary(a, factor * math.log(a.v), factor / a.v)
    if code == 44:
        value = math.exp(args[0].v) if args[0].v < 709.0 else INF
        return _unary(args[0], value, value)
    if code == 54:
        total = Dual(0.0)
        for a in args:
            total = Dual(total.v + a.v, _combine(total.g, 1.0, a.g, 1.0))
        return total
    raise ValueError(f"operator o{code} is not one the checker reads")


ARITY = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 16: 1, 39: 1, 41: 1, 42: 1, 43: 1, 44: 1, 46: 1}


class NlModel:
    """What the checker needs of a .nl model: bounds, integrality, names, and functions."""

    def __init__(self) -> None:
        self.n = 0
        self.m = 0
        self.maximize = False
        self.col_lower: list[float] = []
        self.col_upper: list[float] = []
        self.integer: list[bool] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []
        self.col_names: list[str] = []
        self.row_names: list[str] = []
        self.objective_tree = None                  # nonlinear part, or None
        self.objective_linear: dict[int, float] = {}
        self.row_trees: list = []                   # nonlinear part per row, or None
        self.row_linear: list[dict[int, float]] = []
        self.defined: dict[int, tuple] = {}         # defined variable -> (linear, tree)

    # ---- evaluation ----------------------------------------------------------------------
    def _eval(self, tree, x: list[float], memo: dict[int, Dual]) -> Dual:
        kind = tree[0]
        if kind == "n":
            return Dual(tree[1])
        if kind == "v":
            j = tree[1]
            if j < self.n:
                return Dual(x[j], {j: 1.0})
            if j not in memo:
                linear, sub = self.defined[j]
                value = self._eval(sub, x, memo)
                for col, coef in linear.items():
                    value = Dual(value.v + coef * x[col], _combine(value.g, 1.0, {col: 1.0}, coef))
                memo[j] = value
            return memo[j]
        return apply(tree[1], [self._eval(t, x, memo) for t in tree[2]])

    def _with_linear(self, tree, linear: dict[int, float], x: list[float],
                     memo: dict[int, Dual]) -> Dual:
        value = self._eval(tree, x, memo) if tree is not None else Dual(0.0)
        g = dict(value.g)
        v = value.v
        for j, coef in linear.items():
            v += coef * x[j]
            g[j] = g.get(j, 0.0) + coef
        return Dual(v, g)

    def objective(self, x: list[float]) -> Dual:
        """The objective in the model's own sense, with its gradient."""
        return self._with_linear(self.objective_tree, self.objective_linear, x, {})

    def rows(self, x: list[float]) -> list[Dual]:
        memo: dict[int, Dual] = {}
        return [self._with_linear(self.row_trees[i], self.row_linear[i], x, memo)
                for i in range(self.m)]


def _tokens(text: str):
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        yield from line.split()


def _read_tree(tokens) -> tuple:
    """One prefix-notation expression, with an explicit stack (sec. 3)."""
    stack: list[list] = []  # [code, remaining, args]
    while True:
        tok = next(tokens)
        if tok[0] == "n":
            node = ("n", float(tok[1:]))
        elif tok[0] == "v":
            node = ("v", int(tok[1:]))
        elif tok[0] == "o":
            code = int(tok[1:])
            if code == 54:
                count = int(next(tokens))
            elif code in ARITY:
                count = ARITY[code]
            else:
                raise ValueError(f"operator o{code} is not one the checker reads")
            stack.append([code, count, []])
            if count > 0:
                continue
            code, _, args = stack.pop()
            node = ("o", code, args)
        else:
            raise ValueError(f"unexpected token {tok!r} in an expression")
        while True:
            if not stack:
                return node
            stack[-1][2].append(node)
            if len(stack[-1][2]) < stack[-1][1]:
                break
            code, _, args = stack.pop()
            node = ("o", code, args)


def _bound(tokens) -> tuple[float, float]:
    kind = int(next(tokens))
    if kind == 0:
        return float(next(tokens)), float(next(tokens))
    if kind == 1:
        return -INF, float(next(tokens))
    if kind == 2:
        return float(next(tokens)), INF
    if kind == 3:
        return -INF, INF
    if kind == 4:
        value = float(next(tokens))
        return value, value
    raise ValueError(f"bound type {kind} is not one the checker reads")


def read_nl(path: Path) -> NlModel:
    raw = gzip.open(path, "rt").read() if str(path).endswith(".gz") else Path(path).read_text()
    lines = raw.splitlines()
    if not lines or not lines[0].startswith("g"):
        raise ValueError("only the text ('g') form of .nl is read")
    head = [[int(v) for v in line.split("#", 1)[0].split()] for line in lines[1:10]]
    get = lambda h, k, d=0: head[h][k] if k < len(head[h]) else d  # noqa: E731
    model = NlModel()
    model.n, model.m, n_obj = head[0][0], head[0][1], head[0][2]
    n = model.n
    # Integer columns: "Hooking Your Solver to AMPL", Tables 3 and 4 (lines 5 and 7).
    nlvc, nlvo, nlvb = get(3, 0), get(3, 1), get(3, 2, 0)
    nbv, niv, nlvbi, nlvci, nlvoi = (get(5, k) for k in range(5))
    model.integer = [False] * n
    for lo, hi, tail in ((0, nlvb, nlvbi), (nlvb, nlvc, nlvci),
                         (nlvc, max(nlvc, nlvo), nlvoi)):
        for j in range(hi - tail, hi):
            model.integer[j] = True
    for j in range(n - nbv - niv, n):
        model.integer[j] = True
    model.col_lower, model.col_upper = [-INF] * n, [INF] * n
    model.row_lower, model.row_upper = [-INF] * model.m, [INF] * model.m
    model.row_trees = [None] * model.m
    model.row_linear = [{} for _ in range(model.m)]
    objectives: dict[int, tuple] = {}
    tokens = _tokens("\n".join(lines[10:]))
    for tok in tokens:
        key, rest = tok[0], tok[1:]
        if key == "C":
            model.row_trees[int(rest)] = _read_tree(tokens)
        elif key == "O":
            index = int(rest)
            sense = int(next(tokens))
            objectives[index] = (sense, _read_tree(tokens))
        elif key == "V":
            index, count, _ = int(rest), int(next(tokens)), int(next(tokens))
            linear = {}
            for _ in range(count):
                j = int(next(tokens))
                linear[j] = linear.get(j, 0.0) + float(next(tokens))
            model.defined[index] = (linear, _read_tree(tokens))
        elif key in ("J", "G"):
            index, count = int(rest), int(next(tokens))
            target = model.row_linear[index] if key == "J" else (
                model.objective_linear if index == 0 else {})
            for _ in range(count):
                j = int(next(tokens))
                target[j] = target.get(j, 0.0) + float(next(tokens))
        elif key == "r":
            for i in range(model.m):
                model.row_lower[i], model.row_upper[i] = _bound(tokens)
        elif key == "b":
            for j in range(n):
                model.col_lower[j], model.col_upper[j] = _bound(tokens)
        elif key in ("x", "d"):
            for _ in range(2 * int(rest)):
                next(tokens)
        elif key == "k":
            for _ in range(int(rest)):
                next(tokens)
        elif key == "S":
            count = int(next(tokens))
            next(tokens)
            for _ in range(2 * count):
                next(tokens)
        else:
            raise ValueError(f"segment {tok!r} is not one the checker reads")
    if n_obj > 0 and 0 in objectives:
        sense, tree = objectives[0]
        model.maximize = sense == 1
        model.objective_tree = tree
    stem = Path(str(path)[:-3] if str(path).endswith(".gz") else str(path)).with_suffix("")
    cols = _names(stem.with_suffix(".col"))
    rows = _names(stem.with_suffix(".row"))
    # The solver's naming, reproduced from the file alone: .col names when there are enough,
    # else C<j>; a row's .row name when it has one, else R<i> (src/io/writer.cpp).
    model.col_names = cols[:n] if len(cols) >= n and n > 0 else [f"C{j}" for j in range(n)]
    model.row_names = [rows[i] if i < len(rows) and rows[i] else f"R{i}" for i in range(model.m)]
    return model


def _names(path: Path) -> list[str]:
    try:
        return [line.rstrip("\r ") for line in path.read_text().splitlines()]
    except OSError:
        return []
