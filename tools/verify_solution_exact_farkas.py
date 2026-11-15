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


def verify_farkas_exact(model, solution, report) -> bool:
    """The infeasibility proof of an `infeasible` verdict whose .sol carries exact multipliers.

    The exact phase 1 (src/exact/exact_repair.hpp) proves infeasibility with multipliers that
    are rationals, and on a model infeasible by less than floating point can resolve (Netlib
    cplex2) their rounded copy cannot clear verify_farkas()'s 1e-9 margin. This is the same
    check as verify_farkas(), oriented the same way (y_i > 0 leans on row i's lower bound,
    y_i < 0 on its upper), done in `Fraction` arithmetic with no tolerance anywhere: every
    coefficient, bound and multiplier is exact, a coefficient of the aggregate is zero only
    when it is exactly zero, and the contradiction must be strictly positive. So it is at
    least as strict as the floating-point check.

    Returns whether the farkas section is these multipliers rounded. When it is not (the IIS
    filter may replace it by its subsystem's certificate), it is a second claim, and the
    caller checks it as well, in floating point, as it would any other.
    """
    y: dict[int, Fraction] = {}
    for name, text in solution.exact_repair_farkas.items():
        if name not in model.row_index:
            report.check(False, "infeasibility proof",
                         f"exact multiplier on {name}, which is not a row of the model")
            return False
        if Fraction(text):
            y[model.row_index[name]] = Fraction(text)
    borrowed = [model.row_names[i] for i, v in y.items()
                if (v > 0 and model.row_lower[i] == -INF) or (v < 0 and model.row_upper[i] == INF)]
    if report.check(not borrowed, "certificate uses only real bounds",
                    "every exact multiplier leans on a finite row bound" if not borrowed
                    else f"{len(borrowed)} lean on an infinite bound: "
                    + ", ".join(borrowed[:5])):
        required = sum((v * Fraction(model.row_lower[i] if v > 0 else model.row_upper[i])
                        for i, v in y.items()), Fraction(0))
        reachable = Fraction(0)
        free = []
        for j, entries in enumerate(model.entries):
            d = sum((y[i] * Fraction(v) for i, v in entries if i in y), Fraction(0))
            if d == 0:
                continue
            bound = model.col_upper[j] if d > 0 else model.col_lower[j]
            if bound in (INF, -INF):
                free.append(model.col_names[j])
            else:
                reachable += d * Fraction(bound)
        if report.check(not free, "aggregate is bounded above",
                        "every column the aggregate uses is bounded in that direction"
                        if not free else f"{len(free)} unbounded in the direction used, so "
                        "the aggregate proves nothing: " + ", ".join(free[:5])):
            report.check(reachable < required, "infeasibility proof",
                         f"the rows aggregate to at least {float(required):.12e}, the column "
                         f"bounds allow at most {float(reachable):.12e}, a contradiction of "
                         f"{float(required - reachable):.3e}, in exact rational arithmetic")
            report.note("certificate size",
                        f"{len(y)} of {model.num_rows} rows carry an exact multiplier")
    rounded = {model.row_index[n]: v for n, v in solution.farkas.items() if n in model.row_index}
    return all(abs(rounded.get(i, 0.0) - float(y.get(i, 0))) <= 1e-12 * abs(float(y.get(i, 0)))
               for i in set(y) | set(rounded))
