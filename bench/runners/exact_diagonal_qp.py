#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The exact optimum of a QPS model of one narrow shape, in rational arithmetic (#980).

    min  0.5 * sum_i q_i x_i^2 + c'x   s.t.   A x = b (equality rows only),   x >= 0,

with Q diagonal and positive. The KKT conditions are q_i x_i + c_i - a_i'lam - z_i = 0,
z_i >= 0, x_i z_i = 0, so x_i = max(0, (a_i'lam - c_i) / q_i): the multipliers lam alone
decide the point. They are found in floating point (a semismooth Newton iteration on
A x(lam) = b), the positive set read off them, and then lam is recomputed EXACTLY from the
file's own decimal strings by Gaussian elimination over the rationals on that set, and every
KKT condition is checked exactly. An optimum is reported only when all of them hold, so the
float search can choose the set but cannot make the answer wrong.

Maros-Meszaros huestis and hues-mod have this shape, and their published OPT values are
6.49e-6 above the exact optimum this computes (data/maros-meszaros/reference_errata.json).

    python bench/runners/exact_diagonal_qp.py data/maros-meszaros/huestis.qps
"""
from __future__ import annotations

import sys
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path


class Unsupported(ValueError):
    """The model is not of the shape this computes exactly."""


def read(path: Path):
    """(A as one list of row coefficients per column, c, q, b), all Fractions, from the
    QPS file's decimal strings. Refuses anything outside the shape."""
    cols: dict[str, dict[str, F]] = {}
    order: list[str] = []
    objective = None
    kinds: dict[str, str] = {}
    rhs: dict[str, F] = {}
    q: dict[str, F] = {}
    section = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                section = line.split()[0]
                continue
            t = line.split()
            if section == "ROWS":
                kinds[t[1]] = t[0]
                if t[0] == "N" and objective is None:
                    objective = t[1]
            elif section == "COLUMNS":
                if t[0] not in cols:
                    cols[t[0]] = {}
                    order.append(t[0])
                for k in range(1, len(t), 2):
                    cols[t[0]][t[k]] = cols[t[0]].get(t[k], F(0)) + F(t[k + 1])
            elif section == "RHS":
                for k in range(1, len(t), 2):
                    rhs[t[k]] = F(t[k + 1])
            elif section in ("QUADOBJ", "QMATRIX", "QSECTION"):
                if t[0] != t[1]:
                    raise Unsupported("an off-diagonal Hessian entry")
                q[t[0]] = F(t[2])
            elif section in ("RANGES", "BOUNDS"):
                raise Unsupported(f"a {section} entry (only x >= 0 is supported)")
    rows = [r for r, kind in kinds.items() if r != objective]
    if any(kinds[r] != "E" for r in rows):
        raise Unsupported("an inequality row")
    if any(q.get(c, F(0)) <= 0 for c in order):
        raise Unsupported("a column without positive curvature")
    a = [[cols[c].get(r, F(0)) for r in rows] for c in order]
    cost = [cols[c].get(objective, F(0)) for c in order]
    return a, cost, [q[c] for c in order], [rhs.get(r, F(0)) for r in rows]


def _solve(m: list[list], v: list) -> list:
    """Gaussian elimination with partial pivoting; exact on Fractions, ordinary on floats."""
    n = len(v)
    m = [row[:] + [v[i]] for i, row in enumerate(m)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(m[i][k]))
        if m[p][k] == 0:
            raise ZeroDivisionError("singular")
        m[k], m[p] = m[p], m[k]
        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            for j in range(k, n + 1):
                m[i][j] -= f * m[k][j]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def _lambda_on(a, cost, q, b, support):
    """lam with sum_{i in support} a_i (a_i'lam - c_i) / q_i = b."""
    rows = len(b)
    m = [[sum(a[i][r] * a[i][s] / q[i] for i in support) for s in range(rows)] for r in range(rows)]
    v = [b[r] + sum(a[i][r] * cost[i] / q[i] for i in support) for r in range(rows)]
    return _solve(m, v)


def exact_optimum(path: Path) -> dict:
    a, cost, q, b = read(path)
    n, rows = len(a), len(b)
    af = [[float(v) for v in row] for row in a]
    cf, qf, bf = [float(v) for v in cost], [float(v) for v in q], [float(v) for v in b]

    def support_of(lam):
        return [i for i in range(n)
                if (sum(af[i][r] * lam[r] for r in range(rows)) - cf[i]) / qf[i] > 0]

    # Float search for the positive set: semismooth Newton, starting from all columns positive.
    support = list(range(n))
    lam = _lambda_on(af, cf, qf, bf, support)
    for _ in range(100):
        new = support_of(lam)
        if new == support:
            break
        support = new
        lam = _lambda_on(af, cf, qf, bf, support)
    # Exact: lam on that set, the point, and every KKT condition.
    lam = _lambda_on(a, cost, q, b, support)
    reduced = [sum(a[i][r] * lam[r] for r in range(rows)) - cost[i] for i in range(n)]
    x = [max(F(0), reduced[i] / q[i]) for i in range(n)]
    on = set(support)
    checks = {
        "rows met exactly": all(sum(a[i][r] * x[i] for i in range(n)) == b[r] for r in range(rows)),
        "positive on its set": all(reduced[i] > 0 for i in on),
        "multiplier non-negative off it": all(reduced[i] <= 0 for i in range(n) if i not in on),
    }
    if not all(checks.values()):
        raise ArithmeticError(f"the KKT conditions do not hold exactly: {checks}")
    objective = sum(q[i] * x[i] * x[i] for i in range(n)) / 2 + sum(cost[i] * x[i] for i in range(n))
    return {"objective": objective, "positive": len(on), "columns": n, "rows": rows,
            "checks": checks}


def decimal_string(value: F, digits: int = 25) -> str:
    with localcontext() as ctx:
        ctx.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    result = exact_optimum(path)
    print(f"{path.name}: {result['rows']} rows, {result['columns']} columns, "
          f"{result['positive']} positive at the optimum; "
          + ", ".join(f"{k}: {v}" for k, v in result["checks"].items()))
    print(f"exact optimum {decimal_string(result['objective'])}")
    if len(argv) > 2:
        published = F(argv[2])
        rel = abs(result["objective"] - published) / abs(published)
        print(f"published {argv[2]}, relative difference {float(rel):.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
