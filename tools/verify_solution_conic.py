#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The conic checks of verify_solution.py (#755): second-order cones from CSECTION and the
multipliers of quadratic rows, all re-derived from the original file.

A quadratic row l <= a'x + x'Qx <= u enters the KKT conditions through its gradient
a + 2 Q x: stationarity reads c + Hx - sum_i y_i (a_i + 2 Q_i x) = d, so the only change to
the LP checks is the extra term in the reduced costs, computed here. A cone over columns K
contributes x_K in K (checked EXACTLY, in rationals: x_0 >= 0 and x_0^2 >= sum x_i^2 has no
square root), its reduced costs d_K in the dual cone - the same cone, the second-order cone
being self-dual (Alizadeh & Goldfarb, "Second-order cone programming", Math. Programming 95,
2003, section 2) - and complementarity x_K' d_K = 0. Nothing here shares code with src/.
"""
from __future__ import annotations

import math
from fractions import Fraction

from verify_solution_mps import Model


def cone_columns(model: Model) -> set[int]:
    return {j for _, _, columns in model.cones for j in columns}


def row_gradients(model: Model, x: list[float], y: list[float]) -> list[float]:
    """sum_i y_i * d(x'Q_i x)/dx_j per column j, from the QCMATRIX entries as listed: an entry
    (i, a, b, v) is the term v x_a x_b, whose gradient is v x_b on a and v x_a on b."""
    out = [0.0] * model.num_cols
    for (i, a, b), value in model.qc_entries.items():
        if y[i] == 0.0:
            continue
        out[a] += y[i] * value * x[b]
        out[b] += y[i] * value * x[a]
    return out


def _membership(kind: str, values: list[Fraction]) -> Fraction:
    """How far (exactly) a point is OUTSIDE the cone, as a non-negative rational: zero inside.
    QUAD: max(-x_0, sum_{i>=1} x_i^2 - x_0^2 if x_0 >= 0). RQUAD: max(-x_0, -x_1,
    sum_{i>=2} x_i^2 - 2 x_0 x_1). The squared forms avoid any root, so the test is exact."""
    zero = Fraction(0)
    if kind == "RQUAD":
        head = 2
        worst = max(zero, -values[0], -values[1])
        tail = sum((v * v for v in values[head:]), zero)
        return max(worst, tail - 2 * values[0] * values[1])
    worst = max(zero, -values[0])
    tail = sum((v * v for v in values[1:]), zero)
    return max(worst, tail - values[0] * values[0])


def check_primal(model: Model, x: list[float], report, primal_tol: float) -> None:
    """Every cone holds the point, exactly, up to primal_tol relative to the squares."""
    worst, where = 0.0, ""
    for name, kind, columns in model.cones:
        values = [Fraction(x[j]) for j in columns]
        outside = _membership(kind, values)
        scale = max(Fraction(1), max((v * v for v in values), default=Fraction(0)))
        relative = float(outside / scale)
        if relative > worst:
            worst, where = relative, name
    report.check(worst <= primal_tol, "cone membership (primal, exact)",
                 f"{len(model.cones)} cone(s), worst exact violation {worst:.3e} relative to "
                 "the squares" + (f" on {where}" if where else ""))


def check_dual(model: Model, x: list[float], d: list[float], lower: list[float],
               upper: list[float], report, dual_tol: float, primal_tol: float) -> None:
    """The reduced costs of each cone's columns lie in the dual cone and are complementary
    to the point. A cone column sitting at a finite bound of its own is skipped with a note:
    its reduced cost is then the cone's dual plus the bound's multiplier, and the .sol file
    does not separate the two."""
    worst_dual, where_dual, worst_comp, where_comp, skipped = 0.0, "", 0.0, "", []
    for name, kind, columns in model.cones:
        at_bound = [j for j in columns
                    if (math.isfinite(lower[j]) and abs(x[j] - lower[j]) <= primal_tol)
                    or (math.isfinite(upper[j]) and abs(x[j] - upper[j]) <= primal_tol)]
        if at_bound:
            skipped.append(name)
            continue
        values = [Fraction(d[j]) for j in columns]
        outside = _membership(kind, values)
        scale = max(Fraction(1), max((v * v for v in values), default=Fraction(0)))
        relative = float(outside / scale)
        if relative > worst_dual:
            worst_dual, where_dual = relative, name
        product = sum(x[j] * d[j] for j in columns)
        magnitude = max(1.0, max(abs(x[j]) for j in columns) * max(abs(d[j]) for j in columns))
        if abs(product) / magnitude > worst_comp:
            worst_comp, where_comp = abs(product) / magnitude, name
    report.check(worst_dual <= dual_tol, "cone membership (dual, exact)",
                 f"worst exact violation {worst_dual:.3e} relative to the squares"
                 + (f" on {where_dual}" if where_dual else ""))
    report.check(worst_comp <= 1e-6, "cone complementarity",
                 f"worst |x_K' d_K| = {worst_comp:.3e} relative to |x||d|"
                 + (f" on {where_comp}" if where_comp else ""))
    if skipped:
        report.note("cone duals", "skipped for " + ", ".join(skipped) + ": a column at its "
                    "own bound, whose reduced cost mixes the cone dual with the bound's "
                    "multiplier")
