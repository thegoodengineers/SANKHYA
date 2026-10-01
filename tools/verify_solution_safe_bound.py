# SPDX-License-Identifier: Apache-2.0
"""Re-derive the safe bound an optimal LP's .sol file states, in exact rational arithmetic (#763).

For  min c.x  subject to  row_lower <= A x <= row_upper,  l <= x <= u,  every feasible x and
EVERY multiplier vector y satisfy

    c.x  >=  sum_i min_{row_lower_i <= t <= row_upper_i} y_i t  +  sum_j min_{l_j <= x_j <= u_j} r_j x_j,

with r = c - A'y: weak duality with the row multipliers fixed. A. Neumaier and O. Shcherbina,
"Safe bounds in linear and mixed-integer linear programming", Math. Programming 99 (2004).

The solver evaluates the right-hand side in floating point with outward rounding. This file
evaluates it again with fractions.Fraction - no rounding at all - from the model as this
script parsed it and the row duals the file carries, and checks that the stated bound is on
the right side of the exact value. It follows the same rules the solver states in
src/core/safe_bound.hpp and lp_safe_bound.cpp, each of which only changes WHICH valid bound
is computed:

* a multiplier that prices a missing row side is priced at the row's exact activity bound
  over the box on that side, or replaced by zero when that is unbounded too;
* a column whose reduced cost needs a missing bound gets the tightest bound one row implies
  from the other columns' bounds (Savelsbergh, ORSA J. Computing 6(4), 1994) - exact here,
  so at least as tight as the solver's rounded one;
* the column bounds the file's safe_column_bounds section lists are first checked, in order,
  to be implied by their row and the box so far, exactly, and added to the box.

The multipliers are the file's safe_multipliers section: the solver's duals, or a vector
it moved them to so the bound is finite (src/core/lp_safe_bound.cpp). Any vector gives a
valid bound, so where they came from does not matter; only the arithmetic is re-done.
"""

from __future__ import annotations

import math
from fractions import Fraction


def _finite(value: float) -> bool:
    return math.isfinite(value)


class _RowRanges:
    """Exact activity range of each row over the column box [lower, upper], cached per row.

    set_bound() changes the box and forgets the rows the column is in."""

    def __init__(self, model, lower: list[float], upper: list[float]) -> None:
        self.model = model
        self.lower = list(lower)
        self.upper = list(upper)
        self.rows: list[list[tuple[int, float]]] | None = None
        self.cache: dict[int, tuple[Fraction, int, Fraction, int]] = {}

    def set_bound(self, j: int, is_upper: bool, value: float) -> None:
        (self.upper if is_upper else self.lower)[j] = value
        for i, _ in self.model.entries[j]:
            self.cache.pop(i, None)

    def entries(self, i: int) -> list[tuple[int, float]]:
        if self.rows is None:
            self.rows = [[] for _ in range(self.model.num_rows)]
            for j, column in enumerate(self.model.entries):
                for row, value in column:
                    self.rows[row].append((j, value))
        return self.rows[i]

    def range(self, i: int) -> tuple[Fraction, int, Fraction, int]:
        """(finite part of the minimum, columns infinite there, same for the maximum)."""
        if i not in self.cache:
            lo, lo_inf, hi, hi_inf = Fraction(0), 0, Fraction(0), 0
            for j, value in self.entries(i):
                at_min = self.lower[j] if value > 0 else self.upper[j]
                at_max = self.upper[j] if value > 0 else self.lower[j]
                if _finite(at_min):
                    lo += Fraction(value) * Fraction(at_min)
                else:
                    lo_inf += 1
                if _finite(at_max):
                    hi += Fraction(value) * Fraction(at_max)
                else:
                    hi_inf += 1
            self.cache[i] = (lo, lo_inf, hi, hi_inf)
        return self.cache[i]

    def implied_by_row(self, j: int, i: int, value: float, want_upper: bool) -> Fraction | None:
        """The bound on x_j (coefficient `value` in row i) row i and the box imply, or None."""
        model = self.model
        if value == 0.0:
            return None
        lo, lo_inf, hi, hi_inf = self.range(i)
        at_min = self.lower[j] if value > 0 else self.upper[j]
        at_max = self.upper[j] if value > 0 else self.lower[j]
        a = Fraction(value)
        # v x_j <= row_upper - min(rest), or v x_j >= row_lower - max(rest).
        if want_upper == (value > 0):
            side = model.row_upper[i]
            if not _finite(side):
                return None
            if _finite(at_min):
                if lo_inf:
                    return None
                rest = lo - a * Fraction(at_min)
            elif lo_inf == 1:
                rest = lo
            else:
                return None
        else:
            side = model.row_lower[i]
            if not _finite(side):
                return None
            if _finite(at_max):
                if hi_inf:
                    return None
                rest = hi - a * Fraction(at_max)
            elif hi_inf == 1:
                rest = hi
            else:
                return None
        return (Fraction(side) - rest) / a

    def implied(self, j: int, want_upper: bool) -> Fraction | None:
        """The tightest bound on x_j one row and the other columns' bounds imply, or None."""
        best: Fraction | None = None
        for i, value in self.model.entries[j]:
            candidate = self.implied_by_row(j, i, value, want_upper)
            if candidate is not None and (best is None or (
                    candidate < best if want_upper else candidate > best)):
                best = candidate
        return best


def apply_column_bounds(model, claims, ranges: _RowRanges) -> str:
    """Check each (column, side, value, row) claim in order and add it to the box.

    A claimed upper bound holds when it is at least what the row exactly implies from the box
    so far (a lower bound: at most), which makes it true of every feasible point. Returns
    the first failure, or "" when every claim holds."""
    for column, side, value, row in claims:
        j = model.col_index.get(column)
        i = model.row_index.get(row)
        if j is None or i is None or side not in ("lower", "upper"):
            return f"a column bound names {column} {side} from {row}, which the model lacks"
        want_upper = side == "upper"
        coefficient = next((v for r, v in model.entries[j] if r == i), 0.0)
        implied = ranges.implied_by_row(j, i, coefficient, want_upper)
        if implied is None or not _finite(value) or (
                Fraction(value) < implied if want_upper else Fraction(value) > implied):
            return (f"{column} {side} {value:.17g} is not implied by row {row}"
                    + ("" if implied is None else f" (it implies {float(implied):.17g})"))
        ranges.set_bound(j, want_upper, value)
    return ""


def exact_safe_bound(model, multipliers: list[float],
                     ranges: _RowRanges | None = None) -> Fraction | None:
    """The exact Neumaier-Shcherbina bound, minimise space without the offset; None = -inf.

    `multipliers` are in the model's own sense, one per model row, as the file writes them.
    `ranges` carries the column box, the model's own when not given.
    """
    if ranges is None:
        ranges = _RowRanges(model, model.col_lower, model.col_upper)
    # An int, so an exact Fraction multiplier stays a Fraction (a float sign would round it).
    sigma = -1 if model.maximize else 1
    y = [sigma * v for v in multipliers]

    total = Fraction(0)
    used: list[Fraction] = []
    for i, yi in enumerate(y):
        if yi == 0.0 or not _finite(yi):
            used.append(Fraction(0))
            continue
        side = model.row_lower[i] if yi > 0 else model.row_upper[i]
        if _finite(side):
            side_value = Fraction(side)
        else:
            # A multiplier pricing a side the row lacks: the row's exact activity bound over
            # the box on that side holds for every feasible point, so it stands in for the
            # side; with no such bound the multiplier is dropped (priced at zero).
            lo, lo_inf, hi, hi_inf = ranges.range(i)
            if (lo_inf if yi > 0 else hi_inf):
                used.append(Fraction(0))
                continue
            side_value = lo if yi > 0 else hi
        used.append(Fraction(yi))
        total += Fraction(yi) * side_value

    for j in range(model.num_cols):
        r = Fraction(sigma * model.col_cost[j])
        for i, value in model.entries[j]:
            if used[i]:
                r -= Fraction(value) * used[i]
        if r == 0:
            continue
        if r > 0:
            bound = ranges.lower[j]
            limit = Fraction(bound) if _finite(bound) else ranges.implied(j, want_upper=False)
        else:
            bound = ranges.upper[j]
            limit = Fraction(bound) if _finite(bound) else ranges.implied(j, want_upper=True)
        if limit is None:
            return None
        total += r * limit
    return total


def verify_safe_bound(model, solution, objective: float, report) -> None:
    """Check the file's safe_lower_bound against the exact bound (an optimal LP only)."""
    stated = solution.header_float("safe_lower_bound")
    if stated is None:
        report.note("safe bound", "not reported")
        return
    if math.isnan(stated):
        report.check(False, "safe bound", "stated as nan")
        return
    if (stated == -math.inf and not model.maximize) or (stated == math.inf and model.maximize):
        report.note("safe bound", "the solver proved no finite bound from its duals")
        return
    try:
        multipliers = [solution.safe_multipliers[name] for name in model.row_names]
    except KeyError as missing:
        report.check(False, "safe bound",
                     f"no safe multiplier for row {missing} to re-derive it from")
        return

    ranges = _RowRanges(model, model.col_lower, model.col_upper)
    failure = apply_column_bounds(model, solution.safe_column_bounds, ranges)
    if failure:
        report.check(False, "safe bound (column bounds)", failure)
        return
    exact = exact_safe_bound(model, multipliers, ranges)
    if exact is None:
        report.check(False, "safe bound (exact)",
                     f"the file states {stated:.17g} but these duals prove no finite bound")
        return
    offset = Fraction(model.objective_offset)
    model_sense = offset - exact if model.maximize else offset + exact
    valid = (Fraction(stated) >= model_sense) if model.maximize else \
        (Fraction(stated) <= model_sense)
    gap = (stated - objective) if model.maximize else (objective - stated)
    report.check(valid, "safe bound (exact)",
                 f"stated {stated:.17g}, exact {float(model_sense):.17g}, certified gap to the "
                 f"recomputed objective {gap:.3e} ({gap / max(1.0, abs(objective)):.3e} "
                 "relative)")
