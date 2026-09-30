# SPDX-License-Identifier: Apache-2.0
"""The exact repair's proof that a model, as read into doubles, is infeasible (#757).

When the solver's exact repair (src/exact/exact_repair.cpp) finds that no exactly optimal
basis exists because the model itself has no feasible point, it writes Farkas multipliers
w on the rows to the .sol (`begin exact_repair_farkas`). This module re-derives the proof
from the model file alone, with Python's arbitrary-precision `fractions.Fraction` (every
coefficient and bound converted exactly from its double), sharing no code with the solver:

  every feasible point satisfies  sum_i w_i (a_i^T x - s_i) = 0,  s_i the activity of row i,
  i.e.  sum_j (sum_i w_i a_ij) x_j - sum_i w_i s_i = 0,

and with each x_j inside its column bounds and each s_i inside its row bounds, the least and
greatest values of the left side are taken term by term at the bounds. If that range
excludes 0 no point satisfies the rows and bounds together: the model is infeasible. This is
Farkas' lemma in the form of a single valid equality that the bounds cannot meet.
"""
from __future__ import annotations

from fractions import Fraction

INF = float("inf")


def farkas_proves_infeasible(model, multipliers: dict[str, str]) -> tuple[bool, str]:
    """Whether the multipliers prove the model infeasible, and a line of detail."""
    w: dict[int, Fraction] = {}
    for name, text in multipliers.items():
        if name not in model.row_index:
            return False, f"multiplier on {name}, which is not a row of the model"
        w[model.row_index[name]] = Fraction(text)
    terms: list[tuple[Fraction, float, float]] = []  # (coefficient, lower, upper)
    for j, entries in enumerate(model.entries):
        coefficient = sum((w[i] * Fraction(v) for i, v in entries if i in w), Fraction(0))
        if coefficient:
            terms.append((coefficient, model.col_lower[j], model.col_upper[j]))
    for i, wi in w.items():
        if wi:
            terms.append((-wi, model.row_lower[i], model.row_upper[i]))
    least = greatest = Fraction(0)
    least_finite = greatest_finite = True
    for coefficient, lower, upper in terms:
        low, high = (lower, upper) if coefficient > 0 else (upper, lower)
        if low in (INF, -INF):
            least_finite = False
        else:
            least += coefficient * Fraction(low)
        if high in (INF, -INF):
            greatest_finite = False
        else:
            greatest += coefficient * Fraction(high)
    if least_finite and least > 0:
        return True, (f"{len(w)} multiplier(s): the combined row is at least {float(least):.6e}"
                      " > 0 over the bounds")
    if greatest_finite and greatest < 0:
        return True, (f"{len(w)} multiplier(s): the combined row is at most {float(greatest):.6e}"
                      " < 0 over the bounds")
    return False, (f"{len(w)} multiplier(s) do not exclude 0: least "
                   f"{float(least) if least_finite else -INF:.6e}, greatest "
                   f"{float(greatest) if greatest_finite else INF:.6e}")


def check_exact_repair_farkas(model, solution, report) -> None:
    """The check verify_solution.py runs when a .sol carries the repair's Farkas row."""
    if not solution.exact_repair_farkas:
        return
    ok, detail = farkas_proves_infeasible(model, solution.exact_repair_farkas)
    report.check(ok, "exact repair: the model as read into doubles is infeasible (Farkas row)",
                 detail)
