#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read the `.qplib` file format and write the same model as QPS, for the QPLIB runner (#492).

THE FORMAT is defined in Appendix B (Table 8) of

    F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould, L. Liberti,
    A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis, S. Vigerske and A. Wiegele,
    "QPLIB: a library of quadratic programming instances", Mathematical Programming
    Computation 11 (2019), preprint at optimization-online.org/wp-content/uploads/2017/02/5846.pdf

and on the documentation page https://qplib.zib.de/doc.html. Free format, one required value
(or index-value tuple) per line, in a fixed order; blank lines and lines starting with `!`,
`%` or `#` are ignored; anything after the required values on a line is a comment; reals may
use a `D` exponent; indices are one-based. The problem type (O, V, C letters, doc.html
"PROBTYPE") decides which sections are present: for constraint type N or B there is no
constraint count, no linear constraint part, no constraint sides and no constraint duals; for
objective type L there is no Q^0; for constraint type N, B or L there is no Q^i; for variable
type B there are no bounds; for variable type C, B or I there is no variable-type section.

THE QUADRATIC CONVENTION, and it is the trap. The paper (section 2 and the Appendix B
example) takes Q^0 SYMMETRIC and lists its lower triangle, so a listed off-diagonal (h, k, v)
would stand for Q_hk = Q_kh = v and contribute v x_h x_k to 0.5 x'Q^0 x. The files on the
website do not follow that. doc.html states the website's own convention: the Q^i "are
lower-left triangle matrices (Q^i_{row,col} = 0 for row < col)", with the footnote "In the
QPLIB paper, the Q matrices were assumed to be symmetric. For this page, it is easier to
assume them as triangle matrices." With a triangular Q^0 the objective 0.5 x'Q^0 x gives a
listed off-diagonal (h, k, v) the term 0.5 v x_h x_k - half of what the paper's reading gives.

Which one the files use was settled by measurement, not by choosing: at QPLIB's own published
solution point (the `sol/QPLIB_xxxx.sol` file linked from each instance page) the triangle
reading reproduces the published objective and the symmetric reading does not. QPLIB_0018:
triangle -6.386014981598351, symmetric -12.872950393204626, published -6.38601498159835.
QPLIB_8845: triangle 10907992.493998824, symmetric 13461747.435818076, published
10907992.4939988. The GAMS model on the site agrees (QPLIB_0018.gms writes the file's
`2 1 -19.8356` as `- 9.9178*x2*x3`). fetch_qplib.py repeats this check on EVERY instance it
fetches, through the converted QPS file and the independent verifier's MPS reader, and records
the result in the manifest; the runner does not solve an instance whose check failed.

QPS lists the lower triangle of a SYMMETRIC Hessian under 0.5 x'Qx (the Maros-Meszaros
convention src/io/mps_quadratic.cpp and tools/verify_solution_mps.py read), so the
conversion is: a diagonal entry keeps its value, an off-diagonal entry is halved. Every listed
entry is read as a term 0.5 v x_h x_k whichever triangle it sits in, and an entry the file
lists twice is refused rather than summed.

Scope: continuous variables (type *C*) and linear, box or no constraints (types **L, **B,
**N) are written. A model with integer variables or quadratic constraints is parsed but
refused by to_qps() by name - those belong to #514's runner, not this one.

    python bench/runners/qplib_format.py QPLIB_8845.qplib QPLIB_8845.qps
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

OBJECTIVE_ROW = "obj"


@dataclass
class QplibModel:
    name: str
    problem_type: str  # the three-letter O V C code
    maximize: bool
    n: int
    m: int
    q0: dict[tuple[int, int], float] = field(default_factory=dict)  # listed (h, k) -> v
    b0: list[float] = field(default_factory=list)
    constant: float = 0.0
    qi_count: int = 0  # entries of the constraint Q^i, parsed and never written
    bi: list[list[tuple[int, float]]] = field(default_factory=list)  # per constraint (j, v)
    infinity: float = math.inf
    cl: list[float] = field(default_factory=list)
    cu: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    integer: list[bool] = field(default_factory=list)
    x0: list[float] = field(default_factory=list)
    var_names: dict[int, str] = field(default_factory=dict)  # non-default names only
    con_names: dict[int, str] = field(default_factory=dict)

    def objective(self, x: list[float]) -> float:
        """0.5 x'Q^0 x + b^0'x + q^0 with Q^0 as the file lists it (the triangle reading)."""
        quadratic = sum(v * x[h] * x[k] for (h, k), v in self.q0.items())
        return 0.5 * quadratic + sum(b * xj for b, xj in zip(self.b0, x)) + self.constant


def _number(token: str) -> float:
    """A real in decimal or exponential form; the exponent may be written with D."""
    return float(token.replace("D", "E").replace("d", "e"))


class _Lines:
    """The data lines of a .qplib file, one required record at a time."""

    def __init__(self, text: str) -> None:
        self._lines = []
        for number, raw in enumerate(text.splitlines(), 1):
            stripped = raw.strip()
            if stripped and stripped[0] not in "!%#":
                self._lines.append((number, stripped.split()))
        self._at = 0

    def take(self, count: int, what: str) -> list[str]:
        if self._at >= len(self._lines):
            raise ValueError(f"file ends where {what} was expected")
        number, tokens = self._lines[self._at]
        self._at += 1
        if len(tokens) < count:
            raise ValueError(f"line {number}: {what} needs {count} value(s), found {tokens}")
        return tokens[:count]

    def integer(self, what: str) -> int:
        token = self.take(1, what)[0]
        try:
            return int(token)
        except ValueError:
            raise ValueError(f"{what}: '{token}' is not an integer") from None

    def real(self, what: str) -> float:
        return _number(self.take(1, what)[0])

    def index(self, token: str, limit: int, what: str) -> int:
        value = int(token)
        if not 1 <= value <= limit:
            raise ValueError(f"{what} index {value} outside 1..{limit}")
        return value - 1

    def dense(self, size: int, what: str) -> list[float]:
        """A default value, a count, then (index, value) pairs: the format's vector layout."""
        default = self.real(f"default {what}")
        values = [default] * size
        for _ in range(self.integer(f"number of non-default {what}")):
            j, value = self.take(2, what)
            values[self.index(j, size, what)] = _number(value)
        return values

    def names(self, size: int, what: str) -> dict[int, str]:
        names = {}
        for _ in range(self.integer(f"number of non-default {what} names")):
            j, name = self.take(2, f"{what} name")
            names[self.index(j, size, what)] = name
        return names

    def done(self) -> bool:
        return self._at >= len(self._lines)


def parse(text: str) -> QplibModel:
    lines = _Lines(text)
    name = lines.take(1, "problem name")[0]
    ptype = lines.take(1, "problem type")[0].upper()
    if len(ptype) != 3 or ptype[0] not in "LDCQ" or ptype[1] not in "CBMIG" \
            or ptype[2] not in "NBLDCQ":
        raise ValueError(f"problem type '{ptype}' is not an O V C code of doc.html")
    sense = lines.take(1, "objective sense")[0].lower()
    if sense not in ("minimize", "maximize"):
        raise ValueError(f"objective sense '{sense}' is neither minimize nor maximize")
    o_type, v_type, c_type = ptype
    has_constraints = c_type not in "NB"
    n = lines.integer("number of variables")
    m = lines.integer("number of constraints") if has_constraints else 0
    model = QplibModel(name=name, problem_type=ptype, maximize=sense == "maximize", n=n, m=m)

    if o_type != "L":
        for _ in range(lines.integer("number of Q^0 entries")):
            h, k, value = lines.take(3, "Q^0 entry")
            key = (lines.index(h, n, "Q^0 row"), lines.index(k, n, "Q^0 column"))
            if key in model.q0:
                raise ValueError(f"Q^0 lists entry ({h}, {k}) twice")
            model.q0[key] = _number(value)
    model.b0 = lines.dense(n, "b^0 entries")
    model.constant = lines.real("objective constant")
    model.bi = [[] for _ in range(m)]
    if c_type not in "NBL":
        model.qi_count = lines.integer("number of Q^i entries")
        for _ in range(model.qi_count):
            lines.take(4, "Q^i entry")
    if has_constraints:
        for _ in range(lines.integer("number of b^i entries")):
            i, j, value = lines.take(3, "b^i entry")
            model.bi[lines.index(i, m, "constraint")].append(
                (lines.index(j, n, "variable"), _number(value)))
    model.infinity = lines.real("value for infinity")
    if has_constraints:
        model.cl = lines.dense(m, "constraint left-hand sides")
        model.cu = lines.dense(m, "constraint right-hand sides")
    if v_type == "B":
        model.lower, model.upper = [0.0] * n, [1.0] * n
    else:
        model.lower = lines.dense(n, "variable lower bounds")
        model.upper = lines.dense(n, "variable upper bounds")
    model.integer = [v_type in "BI"] * n
    if v_type in "MG":
        kinds = lines.dense(n, "variable types")
        model.integer = [kind != 0 for kind in kinds]
        for j, kind in enumerate(kinds):
            if kind == 2:  # binary: bounds [0, 1] override the bound sections (Table 8)
                model.lower[j], model.upper[j] = 0.0, 1.0
    model.x0 = lines.dense(n, "starting point entries")
    if has_constraints:
        lines.dense(m, "constraint dual starting entries")
    lines.dense(n, "bound dual starting entries")
    model.var_names = lines.names(n, "variable")
    model.con_names = lines.names(m, "constraint")
    if not lines.done():
        raise ValueError("data after the constraint names; the section order was misread")

    def finite(value: float, sign: float) -> float:
        # "any bound greater than or equal to this in absolute value is infinite"
        return sign * math.inf if abs(value) >= model.infinity else value

    model.cl = [finite(v, -1.0) for v in model.cl]
    model.cu = [finite(v, +1.0) for v in model.cu]
    model.lower = [finite(v, -1.0) for v in model.lower]
    model.upper = [finite(v, +1.0) for v in model.upper]
    return model


def read(path: Path) -> QplibModel:
    return parse(path.read_text(encoding="latin-1"))


def column_name(j: int) -> str:
    """x<j>, one-based, which is also the GAMS name of variable j + 1 in QPLIB's .sol files
    (their x1 is renamed objvar, the objective variable)."""
    return f"x{j + 1}"


def row_name(i: int) -> str:
    return f"c{i + 1}"


def hessian(model: QplibModel) -> dict[tuple[int, int], float]:
    """The QPS lower triangle (row >= column) of the symmetric Hessian with the same
    0.5 x'Hx as the file's triangle reading: diagonal kept, off-diagonal halved."""
    out: dict[tuple[int, int], float] = {}
    for (h, k), value in model.q0.items():
        key = (max(h, k), min(h, k))
        out[key] = out.get(key, 0.0) + (value if h == k else 0.5 * value)
    return {key: value for key, value in out.items() if value != 0.0}


def to_qps(model: QplibModel) -> str:
    """The model as free-format QPS. Refuses what this runner does not cover."""
    if any(model.integer):
        raise ValueError(f"{model.name} ({model.problem_type}) has integer variables; the "
                         "convex continuous QP runner does not write them")
    if model.qi_count or model.problem_type[2] not in "NBL":
        raise ValueError(f"{model.name} ({model.problem_type}) has quadratic constraints; "
                         "they are #514's, not this runner's")
    num = repr
    out = [f"NAME {model.name}"]
    if model.maximize:
        out += ["OBJSENSE", "    MAX"]
    kinds, rhs, ranges = [], {}, {}
    for i in range(model.m):
        lo, hi = model.cl[i], model.cu[i]
        if lo > hi:
            raise ValueError(f"constraint {i + 1} has sides {lo} > {hi}")
        if math.isinf(lo) and math.isinf(hi):
            raise ValueError(f"constraint {i + 1} is free; not expected in QPLIB and not "
                             "written, so that no row is dropped silently")
        if lo == hi:
            kinds.append("E")
            rhs[i] = lo
        elif math.isinf(lo):
            kinds.append("L")
            rhs[i] = hi
        else:
            kinds.append("G")
            rhs[i] = lo
            if not math.isinf(hi):
                ranges[i] = hi - lo  # G row with range R: [rhs, rhs + |R|] (IBM MPS)
    out += ["ROWS", f" N  {OBJECTIVE_ROW}"]
    out += [f" {kind}  {row_name(i)}" for i, kind in enumerate(kinds)]
    by_column: list[list[tuple[int, float]]] = [[] for _ in range(model.n)]
    for i, entries in enumerate(model.bi):
        for j, value in entries:
            by_column[j].append((i, value))
    out.append("COLUMNS")
    for j in range(model.n):
        entries = [(OBJECTIVE_ROW, model.b0[j])] if model.b0[j] != 0.0 else []
        seen = set()
        for i, value in sorted(by_column[j]):
            if i in seen:
                raise ValueError(f"b^i lists variable {j + 1} in constraint {i + 1} twice")
            seen.add(i)
            if value != 0.0:
                entries.append((row_name(i), value))
        if not entries:
            entries = [(OBJECTIVE_ROW, 0.0)]  # a column needs one appearance
        out += [f"    {column_name(j)}  {row}  {num(value)}" for row, value in entries]
    out.append("RHS")
    if model.constant != 0.0:
        out.append(f"    RHS  {OBJECTIVE_ROW}  {num(-model.constant)}")  # RHS on obj = -c0
    out += [f"    RHS  {row_name(i)}  {num(value)}" for i, value in rhs.items() if value != 0.0]
    if ranges:
        out.append("RANGES")
        out += [f"    RNG  {row_name(i)}  {num(value)}" for i, value in ranges.items()]
    out.append("BOUNDS")
    for j in range(model.n):
        lo, hi, name = model.lower[j], model.upper[j], column_name(j)
        if lo > hi:
            raise ValueError(f"variable {j + 1} has bounds {lo} > {hi}")
        if lo == hi:
            out.append(f" FX BND  {name}  {num(lo)}")
            continue
        if math.isinf(lo) and math.isinf(hi):
            out.append(f" FR BND  {name}")
            continue
        # The lower bound is always written first and explicitly, so that no reader's
        # "UP < 0 with no lower bound means a lower bound of -infinity" rule can apply.
        out.append(f" MI BND  {name}" if math.isinf(lo) else f" LO BND  {name}  {num(lo)}")
        if not math.isinf(hi):
            out.append(f" UP BND  {name}  {num(hi)}")
    entries = hessian(model)
    if entries:
        out.append("QUADOBJ")
        out += [f"    {column_name(h)}  {column_name(k)}  {num(value)}"
                for (h, k), value in sorted(entries.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    out.append("ENDATA")
    return "\n".join(out) + "\n"


def read_solution(text: str, model: QplibModel) -> tuple[list[float], float | None]:
    """QPLIB's published point from its `sol/` file: `name value` lines, one per nonzero.

    The names are the site's GAMS model's: `objvar` is the objective variable, which the GAMS
    model puts where x1 would be, and `x<k>` is the file's variable k - 1. A name the
    .qplib file itself assigns (its non-default names) is also accepted. Absent means zero.
    Returns (x, objvar)."""
    x = [0.0] * model.n
    by_name = {name: j for j, name in model.var_names.items()}
    objvar = None
    for raw in text.splitlines():
        tokens = raw.split()
        if not tokens:
            continue
        name, value = tokens[0], _number(tokens[1])
        if name == "objvar":
            objvar = value
            continue
        if name in by_name:
            x[by_name[name]] = value
        elif name[:1] == "x" and name[1:].isdigit() and 2 <= int(name[1:]) <= model.n + 1:
            x[int(name[1:]) - 2] = value
        else:
            raise ValueError(f"solution names '{name}', which maps to no variable")
    return x, objvar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("qplib", type=Path)
    parser.add_argument("qps", type=Path)
    args = parser.parse_args()
    args.qps.write_text(to_qps(read(args.qplib)), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
