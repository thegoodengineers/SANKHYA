#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The conic checks of verify_solution.py (#755): second-order cones from CSECTION and the
multipliers of quadratic rows, all re-derived from the original file.

A quadratic row l <= a'x + x'Qx <= u enters the KKT conditions through its gradient
a + 2 Q x: stationarity reads c + Hx - sum_i y_i (a_i + 2 Q_i x) = d, so the only change to
the LP checks is the extra term in the reduced costs, computed here.

A cone over columns K contributes x_K in K, checked EXACTLY in rationals (the squared forms
need no root). Its columns' reduced costs d_K are the cone's dual s_K plus, on a member that
sits at a finite bound of its own, that bound's multiplier mu - and the .sol file reports only
their sum. So the sum is DECOMPOSED rather than trusted or skipped (#1061 review: skipping
let a 41% suboptimal point through, because nothing then priced the cone's columns):

*   x_K strictly inside the cone: complementarity forces s_K = 0, so d_K = mu.
*   x_K on the boundary, not at the apex: s_K in the dual cone with s_K'x_K = 0 forces
    s_K = beta R(x_K), beta >= 0, where R reflects x_K across the cone's axis - (x_0, -x_1..)
    for a quadratic cone, (x_1, x_0, -x_2..) for a rotated one (Alizadeh & Goldfarb,
    "Second-order cone programming", Math. Programming 95 (2003), section 4: the second-order
    cone is self-dual and complementary boundary points are reflections). beta is fitted on
    the members at no bound, where d must equal s exactly.
*   x_K at the apex: s_K is any point of the cone, chosen inside the interval each member's
    bound leaves it (d at no bound, below d at a lower bound, above d at an upper one).

Then mu = d_K - s_K must vanish on the free members and have the sign its bound prices on
the others, and s_K must lie in the cone. Every one of those is checked; none is assumed.
Nothing here shares code with src/.
"""
from __future__ import annotations

import math
from fractions import Fraction

from verify_solution_mps import Model

# A cone's point is strictly inside when its membership margin exceeds this share of its own
# scale, and at the apex when every member is within it of zero; between the two it is on the
# boundary. The primal tolerance's own order, so a point the primal checks call feasible is
# classified the way those checks see it.
INTERIOR_SHARE = 1e-6
# Complementarity x_K's_K, relative to max(1, |x_K| |s_K|): the 1e-6 every other
# complementarity product in verify_solution.py is held to.
COMPLEMENTARITY_TOLERANCE = 1e-6


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


def _outside(kind: str, values: list[Fraction]) -> Fraction:
    """How far (exactly) a point is OUTSIDE the cone, as a non-negative rational: zero inside.
    QUAD: max(-x_0, sum_{i>=1} x_i^2 - x_0^2). RQUAD: max(-x_0, -x_1, sum_{i>=2} x_i^2 -
    2 x_0 x_1). The squared forms avoid any root, so the test is exact."""
    zero = Fraction(0)
    if kind == "RQUAD":
        worst = max(zero, -values[0], -values[1])
        tail = sum((v * v for v in values[2:]), zero)
        return max(worst, tail - 2 * values[0] * values[1])
    worst = max(zero, -values[0])
    tail = sum((v * v for v in values[1:]), zero)
    return max(worst, tail - values[0] * values[0])


def _relative_outside(kind: str, values: list[float]) -> float:
    exact = [Fraction(v) for v in values]
    scale = max(Fraction(1), max((v * v for v in exact), default=Fraction(0)))
    return float(_outside(kind, exact) / scale)


def check_primal(model: Model, x: list[float], report, primal_tol: float) -> None:
    """Every cone holds the point, exactly, up to primal_tol relative to the squares."""
    worst, where = 0.0, ""
    for name, kind, columns in model.cones:
        relative = _relative_outside(kind, [x[j] for j in columns])
        if relative > worst:
            worst, where = relative, name
    report.check(worst <= primal_tol, "cone membership (primal, exact)",
                 f"{len(model.cones)} cone(s), worst exact violation {worst:.3e} relative to "
                 "the squares" + (f" on {where}" if where else ""))


def _reflection(kind: str, v: list[float]) -> list[float]:
    if kind == "RQUAD":
        return [v[1], v[0]] + [-e for e in v[2:]]
    return [v[0]] + [-e for e in v[1:]]


def _margin(kind: str, v: list[float]) -> float:
    """How far inside the cone, relative to its own scale; negative outside."""
    if kind == "RQUAD":
        head = 2.0 * v[0] * v[1]
        tail = sum(e * e for e in v[2:])
        return min(v[0], v[1], (head - tail) / max(1.0, abs(head)))
    tail = math.sqrt(sum(e * e for e in v[1:]))
    return (v[0] - tail) / max(1.0, abs(v[0]))


def _at_bound(value: float, lower: float, upper: float, tol: float) -> str | None:
    # A fixed member is an equality: its multiplier may take either sign, as the LP checks'
    # sign_violation has it for lower == upper. Reading it as "lower" rejected the true
    # answer to the textbook disc (fixed head t = 1, multiplier -sqrt 2).
    if lower == upper:
        return "fixed"
    scale = max(1.0, abs(value))
    if math.isfinite(lower) and abs(value - lower) <= tol * scale:
        return "lower"
    if math.isfinite(upper) and abs(value - upper) <= tol * scale:
        return "upper"
    return None


def _apex_dual(kind: str, d: list[float], sides: list[str | None]) -> list[float]:
    """At the apex any s in the cone is complementary: within each member's interval
    (s = d at no bound, s <= d at a lower bound, s >= d at an upper one) take the tail as
    close to zero and the heads as large as the intervals allow."""
    heads = 2 if kind == "RQUAD" else 1
    s = list(d)
    for i in range(heads, len(d)):
        if sides[i] == "fixed":
            s[i] = 0.0
        elif sides[i] == "lower":
            s[i] = min(d[i], 0.0)
        elif sides[i] == "upper":
            s[i] = max(d[i], 0.0)
    tail = math.sqrt(sum(e * e for e in s[heads:]))
    need = tail if heads == 1 else tail / math.sqrt(2.0)
    for i in range(heads):
        if sides[i] == "upper":
            s[i] = max(d[i], need)
        elif sides[i] == "fixed":
            s[i] = need
    return s


def check_dual(model: Model, x: list[float], d: list[float], report, dual_tol: float,
               primal_tol: float) -> tuple[dict[int, float], float]:
    """Decompose each cone's reduced costs into its dual s_K and the bound multipliers mu
    (module docstring) and check every part. Returns mu per cone column, for the strong
    duality accounting, and the sum of |x_K's_K| over the cones."""
    mu: dict[int, float] = {}
    complementarity_total = 0.0
    worst_free, where_free = 0.0, ""
    worst_sign, where_sign = 0.0, ""
    worst_member, where_member = 0.0, ""
    worst_comp, where_comp = 0.0, ""
    for name, kind, columns in model.cones:
        xs = [x[j] for j in columns]
        ds = [d[j] for j in columns]
        sides = [_at_bound(x[j], model.col_lower[j], model.col_upper[j], primal_tol)
                 for j in columns]
        dscale = max(1.0, max(abs(e) for e in ds))
        if max(abs(e) for e in xs) <= INTERIOR_SHARE:
            s = _apex_dual(kind, ds, sides)
        elif _margin(kind, xs) > INTERIOR_SHARE:
            s = [0.0] * len(xs)
        else:
            r = _reflection(kind, xs)
            free = [i for i, side in enumerate(sides) if side is None] or list(range(len(xs)))
            norm = sum(r[i] * r[i] for i in free)
            beta = sum(ds[i] * r[i] for i in free) / norm if norm > 0.0 else 0.0
            if beta < 0.0:
                worst_member = max(worst_member, -beta / dscale)
                where_member = where_member or name
                beta = 0.0
            s = [beta * e for e in r]
        member = _relative_outside(kind, s)
        if member > worst_member:
            worst_member, where_member = member, name
        for i, j in enumerate(columns):
            m = ds[i] - s[i]
            mu[j] = m
            if sides[i] is None:
                if abs(m) / dscale > worst_free:
                    worst_free, where_free = abs(m) / dscale, name
            elif sides[i] != "fixed":  # a fixed member's multiplier may take either sign
                wrong = -m if sides[i] == "lower" else m  # lower prices m >= 0, upper m <= 0
                if wrong / dscale > worst_sign:
                    worst_sign, where_sign = wrong / dscale, name
        product = sum(xi * si for xi, si in zip(xs, s))
        complementarity_total += abs(product)
        magnitude = max(1.0, max(abs(e) for e in xs) * max(abs(e) for e in s))
        if abs(product) / magnitude > worst_comp:
            worst_comp, where_comp = abs(product) / magnitude, name
    report.check(worst_free <= dual_tol, "cone duals (free members)",
                 f"worst |d - s| {worst_free:.3e} relative on a member at no bound of its own"
                 + (f" on {where_free}" if where_free else ""))
    report.check(worst_sign <= dual_tol, "cone duals (bound multipliers)",
                 f"worst wrong-signed bound multiplier {worst_sign:.3e} relative"
                 + (f" on {where_sign}" if where_sign else ""))
    report.check(worst_member <= dual_tol, "cone membership (dual)",
                 f"worst violation of s_K in the dual cone {worst_member:.3e} relative"
                 + (f" on {where_member}" if where_member else ""))
    report.check(worst_comp <= COMPLEMENTARITY_TOLERANCE, "cone complementarity",
                 f"worst |x_K' s_K| = {worst_comp:.3e} relative to |x||s|"
                 + (f" on {where_comp}" if where_comp else ""))
    return mu, complementarity_total
