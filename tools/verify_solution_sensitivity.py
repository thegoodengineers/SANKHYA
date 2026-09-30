#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shadow prices and sensitivity ranges re-derived from the basis in exact arithmetic (#757).

The solver reports duals and ranges in floating point from its final LU factor. This module
takes only the model (read by verify_solution_mps.py) and the basis the .sol file states, and
rebuilds everything with Python's arbitrary-precision `fractions.Fraction`: every model
coefficient is converted from its double exactly (Fraction(float) is exact), so no rounding
and no overflow happens anywhere below. It shares no code with src/exact/.

What is derived, in the conventions src/simplex/ranging.cpp documents:

  - the exact row duals y (B^T y = c_B) and reduced costs d = c - A^T y;
  - cost ranging (Chvatal, "Linear Programming", ch. 10, 1983): a nonbasic column's range is
    its reduced cost on one side; a basic column's comes from the ratio test over row p of
    B^{-1} N. A fixed variable (equal bounds) never enters and never limits a range;
  - right-hand-side ranging: v = B^{-1} e_i and the ratio test on the basic variables;
  - THE SHADOW PRICE INTERVAL. At a degenerate optimum the dual is not unique and "the"
    shadow price does not exist: the optimal value v(t) of the model with row i's interval
    shifted by t is piecewise linear and may have a kink at t = 0. Its left and right
    derivatives are the honest answer (Jansen, de Jong, Roos and Terlaky, "Sensitivity
    analysis in linear programming: just be careful!", EJOR 101, 1997). Where the ranging
    interval extends past 0 on a side, this basis stays optimal there and the derivative on
    that side is y_i. Where it does not, the lexicographic dual simplex is run in exact
    arithmetic on the infinitesimally perturbed right-hand side (Gal, "Postoptimal Analyses,
    Parametric Programming, and Related Topics", 2nd ed., 1995, ch. 3) until it reaches a
    basis optimal for small t on that side; that basis's y_i is the one-sided derivative, or
    the model has no feasible point on that side and the derivative is infinite.

Everything is in the MODEL'S sense when returned: duals and derivatives are multiplied by the
sense, and a maximize model's cost ranges exchange sides, as the .sol file writes them.
"""
from __future__ import annotations

import copy
import heapq
import math
import time
from fractions import Fraction

INF = float("inf")


class Declined(Exception):
    """The derivation was not attempted or not finished, for the stated reason."""


# ---------------------------------------------------------------------------------------
# Sparse exact LU of a square basis
# ---------------------------------------------------------------------------------------

class ExactLU:
    """Gaussian elimination in Fractions with a Markowitz pivot choice (Markowitz, Management
    Science 3, 1957): exact arithmetic needs no stability threshold, so the pivot is chosen
    for sparsity alone, which is what keeps the Fractions small enough to be practical.
    The sparsest active columns are found from a heap keyed by their nonzero count, with
    stale entries dropped when popped, rather than by sorting every active column per step.

    `columns[p]` is {row: value} for basis position p. After factorisation E B = U, with E
    the recorded row eliminations and U triangular in pivot order. U is kept both by pivot
    row and by pivot column, so both solves push from each nonzero and skip every zero.

    A basis change is a product-form eta (Dantzig and Orchard-Hays, MTAC 8, 1954):
    `with_eta` returns a new object sharing these factors, so a basis the caller still holds
    is never changed underneath it."""

    def __init__(self, columns: list[dict[int, Fraction]], deadline: float) -> None:
        m = len(columns)
        rows: dict[int, dict[int, Fraction]] = {i: {} for i in range(m)}
        col_rows: dict[int, set[int]] = {p: set() for p in range(m)}
        for p, column in enumerate(columns):
            for i, v in column.items():
                if v != 0:
                    rows[i][p] = v
                    col_rows[p].add(i)
        self.m = m
        self.pivots: list[tuple[int, int]] = []          # (row, position) in order
        self.pivot_values: list[Fraction] = []
        self.eliminations: list[list[tuple[int, Fraction]]] = []  # per step: (row, l)
        self.etas: tuple = ()                            # (position, {position: alpha})
        u_rows: list[dict[int, Fraction]] = []           # per step: {position: value}
        heap = [(len(col_rows[p]), p) for p in range(m)]
        heapq.heapify(heap)
        done: set[int] = set()
        for _ in range(m):
            if time.monotonic() > deadline:
                raise Declined("the exact factorisation ran past its time budget")
            best = None
            looked: list[int] = []
            while heap and len(looked) < 4:
                count, p = heapq.heappop(heap)
                if p in done or count != len(col_rows[p]):
                    continue  # stale
                looked.append(p)
                if not col_rows[p]:
                    raise Declined("the basis is exactly singular")
                for i in col_rows[p]:
                    cost = (len(rows[i]) - 1) * (count - 1)
                    if best is None or cost < best[0]:
                        best = (cost, i, p)
                if best[0] == 0:
                    break
            if best is None:
                raise Declined("the basis is exactly singular")
            for p in looked:
                if p != best[2]:
                    heapq.heappush(heap, (len(col_rows[p]), p))
            _, r, p = best
            pivot_row = rows.pop(r)
            pivot = pivot_row[p]
            step = []
            touched: set[int] = set()
            for i in list(col_rows[p]):
                if i == r:
                    continue
                factor = rows[i][p] / pivot
                step.append((i, factor))
                target = rows[i]
                for c, v in pivot_row.items():
                    value = target.get(c, 0) - factor * v
                    if value == 0:
                        if c in target:
                            del target[c]
                            col_rows[c].discard(i)
                            touched.add(c)
                    else:
                        if c not in target:
                            col_rows[c].add(i)
                            touched.add(c)
                        target[c] = value
            self.eliminations.append(step)
            self.pivots.append((r, p))
            self.pivot_values.append(pivot)
            u_rows.append({c: v for c, v in pivot_row.items() if c != p})
            for c in pivot_row:
                col_rows[c].discard(r)
                touched.add(c)
            done.add(p)
            del col_rows[p]
            for c in touched:
                if c not in done:
                    heapq.heappush(heap, (len(col_rows[c]), c))
        step_of = {p: k for k, (_, p) in enumerate(self.pivots)}
        # U by step: the row of step k over later steps, and the column of step k over
        # earlier steps.
        self.u_row: list[list[tuple[int, Fraction]]] = [
            [(step_of[c], v) for c, v in u.items()] for u in u_rows]
        self.u_col: list[list[tuple[int, Fraction]]] = [[] for _ in range(m)]
        for k, entries in enumerate(self.u_row):
            for l, v in entries:
                self.u_col[l].append((k, v))

    def with_eta(self, position: int, alpha: dict[int, Fraction]) -> "ExactLU":
        """The factor of the basis with column `position` replaced by a column whose
        B^{-1} a is `alpha` (by position); alpha[position] must be nonzero."""
        other = copy.copy(self)
        other.etas = self.etas + ((position, dict(alpha)),)
        return other

    def solve(self, rhs: dict[int, Fraction]) -> dict[int, Fraction]:
        """x (by basis position) with B x = rhs (by row)."""
        b = dict(rhs)
        for (r, _), step in zip(self.pivots, self.eliminations):
            br = b.get(r)
            if br:
                for i, factor in step:
                    b[i] = b.get(i, 0) - factor * br
        x: dict[int, Fraction] = {}
        pivots, values = self.pivots, self.pivot_values
        for k in range(self.m - 1, -1, -1):
            r, p = pivots[k]
            v = b.get(r)
            if v:
                xk = v / values[k]
                x[p] = xk
                for j, u in self.u_col[k]:
                    rj = pivots[j][0]
                    b[rj] = b.get(rj, 0) - u * xk
        for p, alpha in self.etas:
            xp = x.get(p)
            if not xp:
                continue
            xp = xp / alpha[p]
            for i, a in alpha.items():
                if i != p:
                    value = x.get(i, 0) - a * xp
                    if value:
                        x[i] = value
                    elif i in x:
                        del x[i]
            x[p] = xp
        return x

    def solve_transpose(self, rhs: dict[int, Fraction]) -> dict[int, Fraction]:
        """y (by row) with B^T y = rhs (by basis position)."""
        acc = dict(rhs)
        for p, alpha in reversed(self.etas):
            total = acc.get(p, 0)
            for i, a in alpha.items():
                if i != p:
                    ai = acc.get(i)
                    if ai:
                        total -= a * ai
            total = total / alpha[p]
            if total:
                acc[p] = total
            elif p in acc:
                del acc[p]
        w: dict[int, Fraction] = {}
        pivots, values = self.pivots, self.pivot_values
        for k in range(self.m):
            r, p = pivots[k]
            value = acc.get(p)
            if value:
                wr = value / values[k]
                w[r] = wr
                for l, u in self.u_row[k]:
                    pl = pivots[l][1]
                    acc[pl] = acc.get(pl, 0) - u * wr
        for (r, _), step in zip(reversed(self.pivots), reversed(self.eliminations)):
            total = w.get(r, 0)
            for i, factor in step:
                wi = w.get(i)
                if wi:
                    total -= factor * wi
            if total:
                w[r] = total
            elif r in w:
                del w[r]
        return w


# ---------------------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------------------

def _q(value: float) -> Fraction | float:
    return value if value in (INF, -INF) else Fraction(value)


class _Problem:
    """The model as [A | -I] with variables 0..n-1 structural and n+i the logical of row i."""

    def __init__(self, model) -> None:
        self.n = model.num_cols
        self.m = model.num_rows
        self.sense = -1 if model.maximize else 1
        self.cost = [self.sense * Fraction(c) for c in model.col_cost] + [Fraction(0)] * self.m
        self.lower = [_q(v) for v in model.col_lower] + [_q(v) for v in model.row_lower]
        self.upper = [_q(v) for v in model.col_upper] + [_q(v) for v in model.row_upper]
        self.columns: list[dict[int, Fraction]] = []
        for entries in model.entries:
            column: dict[int, Fraction] = {}
            for i, v in entries:
                column[i] = column.get(i, Fraction(0)) + Fraction(v)
            self.columns.append({i: v for i, v in column.items() if v != 0})
        for i in range(self.m):
            self.columns.append({i: Fraction(-1)})
        # Each column over its own least common denominator: column k is
        # int_columns[k] / scale[k] with integer entries, and row_entries holds the same
        # integers row-wise. A dot product with a vector brought to one denominator
        # (_over_common_denominator) then runs in Python's C-speed integers and is reduced
        # once, instead of costing a Fraction gcd per term.
        self.scale: list[int] = []
        self.int_columns: list[list[tuple[int, int]]] = []
        self.row_entries: list[list[tuple[int, int]]] = [[] for _ in range(self.m)]
        for k, column in enumerate(self.columns):
            scale = math.lcm(1, *(v.denominator for v in column.values()))
            entries = [(i, v.numerator * (scale // v.denominator)) for i, v in column.items()]
            self.scale.append(scale)
            self.int_columns.append(entries)
            for i, a in entries:
                self.row_entries[i].append((k, a))

    def fixed(self, k: int) -> bool:
        return self.lower[k] == self.upper[k]


def _smaller_pair(a, b):
    """The smaller of two ratios given as (numerator, positive denominator); None is +inf."""
    if a is None:
        return b
    (n1, d1), (n2, d2) = a, b
    s1, s2 = (n1 > 0) - (n1 < 0), (n2 > 0) - (n2 < 0)
    if s1 != s2:
        return a if s1 < s2 else b
    if s1 == 0:
        return a
    # |n/d| lies in (2^(e-1), 2^(e+1)) for e = bits(n) - bits(d): two exponents two or more
    # apart decide without the multiplications, which on numbers thousands of digits long
    # are most of the cost.
    e1 = abs(n1).bit_length() - d1.bit_length()
    e2 = abs(n2).bit_length() - d2.bit_length()
    if abs(e1 - e2) >= 2:
        first_larger_in_size = e1 > e2
        return b if first_larger_in_size == (s1 > 0) else a
    return a if n1 * d2 <= n2 * d1 else b


def _over_common_denominator(v: dict[int, Fraction]) -> tuple[int, dict[int, int]]:
    """The least common denominator D of v's entries and the integers v_i D."""
    lcd = math.lcm(1, *(x.denominator for x in v.values()))
    return lcd, {i: x.numerator * (lcd // x.denominator) for i, x in v.items() if x}


def _nonbasic_value(problem: _Problem, k: int, status: str):
    if status == "at_lower":
        return problem.lower[k]
    if status == "at_upper":
        return problem.upper[k]
    if status == "free":
        return Fraction(0)
    raise Declined(f"variable {k} has no value for status {status}")


def _resolve(problem: _Problem, k: int, status: str, value: float) -> str:
    """The same reading of a stated status as the solver's exact check: `fixed` with unequal
    bounds is whichever bound the value equals; `free` only when 0 is inside the bounds."""
    lower, upper = problem.lower[k], problem.upper[k]
    if status == "fixed":
        if lower == upper:
            return "at_lower"
        if value == lower:
            return "at_lower"
        if value == upper:
            return "at_upper"
        raise Declined(f"variable {k} is `fixed` but at neither bound")
    if status == "free" and not (lower <= 0 <= upper):
        raise Declined(f"variable {k} is `free` nonbasic outside its bounds")
    if status in ("at_lower", "at_upper") and _nonbasic_value(problem, k, status) in (INF, -INF):
        raise Declined(f"variable {k} is nonbasic at an infinite bound")
    return status


def _ratio(v, gap_down, gap_up):
    """How far t may go up and down when a value moves by v*t with the stated room below
    (gap_down) and above (gap_up). Exact: any nonzero v limits."""
    up = down = INF
    if v > 0:
        if gap_up != INF:
            up = gap_up / v
        if gap_down != INF:
            down = gap_down / v
    elif v < 0:
        if gap_down != INF:
            up = gap_down / -v
        if gap_up != INF:
            down = gap_up / -v
    return down, up


class _Basis:
    """A basis, its factor, the exact point, duals and reduced costs."""

    def __init__(self, problem: _Problem, basic: list[int], status: dict[int, str],
                 deadline: float) -> None:
        self.problem = problem
        self.basic = basic
        self.status = status  # nonbasic variables only
        self.position = {k: p for p, k in enumerate(basic)}
        self.lu = ExactLU([problem.columns[k] for k in basic], deadline)
        rhs: dict[int, Fraction] = {}
        for k, st in status.items():
            value = _nonbasic_value(problem, k, st)
            if value:
                for i, a in problem.columns[k].items():
                    rhs[i] = rhs.get(i, Fraction(0)) - a * value
        xb = self.lu.solve(rhs)
        self.value = {k: _nonbasic_value(problem, k, st) for k, st in status.items()}
        for p, k in enumerate(basic):
            self.value[k] = xb.get(p, Fraction(0))
        self.y = self.lu.solve_transpose(
            {p: problem.cost[k] for p, k in enumerate(basic) if problem.cost[k]})
        lcd, y_int = _over_common_denominator(self.y)
        self.d = {k: self.reduced_cost(k, lcd, y_int) for k in status}

    def reduced_cost(self, k: int, lcd: int | None = None,
                     y_int: dict[int, int] | None = None) -> Fraction:
        """c_k - a_k^T y; `lcd` and `y_int` are y over its common denominator, when the
        caller has them."""
        if y_int is None:
            lcd, y_int = _over_common_denominator(self.y)
        total = 0
        for i, a in self.problem.int_columns[k]:
            yi = y_int.get(i)
            if yi:
                total += yi * a
        if not total:
            return self.problem.cost[k]
        return self.problem.cost[k] - Fraction(total, lcd * self.problem.scale[k])

    def tableau_row(self, p: int, with_w: bool = False):
        """Row p of B^{-1} [A | -I] over the nonbasic variables, nonzeros only; with
        `with_w`, also w = e_p^T B^{-1} (by row) that it came from."""
        w, lcd, total = self._scaled_row(p)
        scale = self.problem.scale
        row = {k: Fraction(t, lcd * scale[k]) for k, t in total.items()}
        return (row, w) if with_w else row

    def tableau_row_scaled(self, p: int) -> tuple[int, dict[int, int]]:
        """Row p of the tableau unreduced: alpha_k = total[k] / (lcd * scale[k]). On a basis
        whose inverse holds numbers thousands of digits long (25fv47) reducing every alpha_k
        to a Fraction is most of the work, and a ratio test needs none of it."""
        _, lcd, total = self._scaled_row(p)
        return lcd, total

    def _scaled_row(self, p: int):
        w = self.lu.solve_transpose({p: Fraction(1)})
        lcd, w_int = _over_common_denominator(w)
        total: dict[int, int] = {}
        status = self.status
        for i, wi in w_int.items():
            for k, a in self.problem.row_entries[i]:
                if k in status:
                    total[k] = total.get(k, 0) + wi * a
        return w, lcd, {k: t for k, t in total.items() if t}

    def degenerate_pivot(self, p: int, entering: int, to: str, row: dict[int, Fraction],
                         w: dict[int, Fraction], deadline: float) -> "_Basis":
        """The basis with `entering` in position p, whose variable sits exactly on the bound
        `to` it leaves to. The primal step is zero, so every value is unchanged; the duals
        move by theta = d_q / alpha_pq: d_k -= theta alpha_pk, d_leaving = -theta, y += theta
        w. The factor takes a product-form eta, rebuilt from scratch every 32."""
        problem = self.problem
        leaving = self.basic[p]
        theta = self.d[entering] / row[entering]
        other = copy.copy(self)
        other.basic = list(self.basic)
        other.basic[p] = entering
        other.position = dict(self.position)
        del other.position[leaving]
        other.position[entering] = p
        other.status = dict(self.status)
        del other.status[entering]
        other.status[leaving] = to
        other.y = dict(self.y)
        other.d = dict(self.d)
        del other.d[entering]
        if theta:
            for i, wi in w.items():
                value = other.y.get(i, 0) + theta * wi
                if value:
                    other.y[i] = value
                elif i in other.y:
                    del other.y[i]
            for k, a in row.items():
                if k != entering:
                    other.d[k] -= theta * a
        other.d[leaving] = -theta
        if len(self.lu.etas) >= 32:
            other.lu = ExactLU([problem.columns[k] for k in other.basic], deadline)
        else:
            alpha = self.lu.solve(problem.columns[entering])
            other.lu = self.lu.with_eta(p, alpha)
        return other


def _check_optimal(basis: _Basis) -> None:
    problem = basis.problem
    for k in basis.basic:
        x = basis.value[k]
        if (problem.lower[k] != -INF and x < problem.lower[k]) or \
                (problem.upper[k] != INF and x > problem.upper[k]):
            raise Declined(f"the stated basis is not exactly primal feasible at variable {k}")
    for k, st in basis.status.items():
        d = basis.d[k]
        if problem.fixed(k):
            continue
        if (st == "at_lower" and d < 0) or (st == "at_upper" and d > 0) or \
                (st == "free" and d != 0):
            raise Declined(f"the stated basis is not exactly dual feasible at variable {k}")


def _one_sided_derivative(start: _Basis, row: int, direction: int, deadline: float):
    """d v / d t at t = 0 from the side `direction` (+1 right, -1 left), minimise space.

    Lexicographic dual simplex on the perturbation of row `row`'s interval by t: the point at
    t = 0 never changes (every pivot is degenerate at order zero), only the basis does, until
    the basis is primal feasible to first order in t on that side."""
    problem = start.problem
    logical = problem.n + row
    basis = start
    for _ in range(50 * problem.m + 50):
        if time.monotonic() > deadline:
            raise Declined("the parametric dual simplex ran past its time budget")
        # First-order motion of each basic variable and of its bounds.
        shift = {row: Fraction(direction)} if logical in basis.status else {}
        dx = basis.lu.solve(shift) if shift else {}
        leaving = None
        for p, k in sorted(enumerate(basis.basic), key=lambda e: e[1]):
            slope = dx.get(p, Fraction(0)) - (direction if k == logical else 0)
            x = basis.value[k]
            if slope < 0 and x == problem.lower[k]:
                leaving = (p, k, "at_lower")
                break
            if slope > 0 and x == problem.upper[k]:
                leaving = (p, k, "at_upper")
                break
        if leaving is None:
            return basis.y.get(row, Fraction(0))
        p, k_out, to = leaving
        alpha, w = basis.tableau_row(p, with_w=True)
        must_rise = to == "at_lower"  # the basic value must move up off its lower bound
        best = None
        for k, a in sorted(alpha.items()):
            st = basis.status[k]
            if problem.fixed(k):
                continue
            # x_p = ... - a * x_k: raising x_k moves x_p by -a.
            can_rise = st in ("at_lower", "free")
            can_fall = st in ("at_upper", "free")
            if (can_rise and (-a > 0) == must_rise) or (can_fall and (a > 0) == must_rise):
                ratio = abs(basis.d[k] / a)
                if best is None or ratio < best[0]:
                    best = (ratio, k)
        if best is None:
            # No feasible point on that side: v is +infinity there, so the derivative from
            # the right is +infinity and from the left -infinity.
            return direction * INF
        basis = basis.degenerate_pivot(p, best[1], to, alpha, w, deadline)
    raise Declined("the parametric dual simplex did not settle within its pivot cap")


def derive(model, solution, seconds: float = 60.0) -> dict:
    """Everything exact, in the model's sense, keyed by name. Raises Declined."""
    deadline = time.monotonic() + seconds
    if model.hessian or getattr(model, "qc_entries", None):
        raise Declined("the model is not an LP")
    if any(model.col_integer):
        raise Declined("the model has integer columns")
    problem = _Problem(model)
    n, m = problem.n, problem.m
    basic: list[int] = []
    status: dict[int, str] = {}
    for k in range(n + m):
        if k < n:
            name = model.col_names[k]
            st = solution.col_status.get(name)
            value = solution.col_value.get(name, 0.0)
        else:
            name = model.row_names[k - n]
            st = solution.row_status.get(name)
            value = solution.row_activity.get(name, 0.0)
        if st is None or st == "unknown":
            raise Declined(f"no basis status for {name}")
        if st == "basic":
            basic.append(k)
        else:
            status[k] = _resolve(problem, k, st, value)
    if len(basic) != m:
        raise Declined(f"the stated basis has {len(basic)} basic variables for {m} rows")
    basis = _Basis(problem, basic, status, deadline)
    _check_optimal(basis)
    sense = problem.sense

    def gaps(k):
        x = basis.value[k]
        return (x - problem.lower[k] if problem.lower[k] != -INF else INF,
                problem.upper[k] - x if problem.upper[k] != INF else INF)

    def on_time():
        if time.monotonic() > deadline:
            raise Declined("the exact derivation ran past its time budget")

    columns: dict[str, dict] = {}
    for j in range(n):
        on_time()
        name = model.col_names[j]
        if j in basis.status:
            d = basis.d[j]
            st = basis.status[j]
            if problem.fixed(j):
                lo, hi = INF, INF
            elif st == "at_lower":
                lo, hi = d, INF
            elif st == "at_upper":
                lo, hi = INF, -d
            else:
                lo, hi = Fraction(0), Fraction(0)
        else:
            d = Fraction(0)
            # Ratios as unreduced integer pairs (numerator, positive denominator), compared by
            # cross-multiplication; only the two winners become Fractions. None is +inf.
            lo_pair = hi_pair = None
            lcd, total = basis.tableau_row_scaled(basis.position[j])
            for k, t in total.items():
                st = basis.status[k]
                if problem.fixed(k):
                    continue
                if st == "free":  # free nonbasic, d = 0: no room either way
                    lo_pair = hi_pair = (0, 1)
                    continue
                dk = basis.d[k]
                # at_lower: d_k / |alpha_k|; at_upper: -d_k / |alpha_k|; alpha_k > 0 limits
                # the increase at_lower and the decrease at_upper.
                # Without the row's common factor lcd > 0, which cannot change which ratio
                # is smallest; it goes back on the two winners only.
                sign = 1 if st == "at_lower" else -1
                pair = (sign * dk.numerator * problem.scale[k], dk.denominator * abs(t))
                if (t > 0) == (st == "at_lower"):
                    hi_pair = _smaller_pair(hi_pair, pair)
                else:
                    lo_pair = _smaller_pair(lo_pair, pair)
            lo = INF if lo_pair is None else Fraction(lo_pair[0] * lcd, lo_pair[1])
            hi = INF if hi_pair is None else Fraction(hi_pair[0] * lcd, hi_pair[1])
            lo, hi = max(lo, Fraction(0)), max(hi, Fraction(0))
        if sense < 0:
            lo, hi = hi, lo
        columns[name] = {"reduced_cost": sense * d, "range_lower": lo, "range_upper": hi}

    rows: dict[str, dict] = {}
    for i in range(m):
        on_time()
        name = model.row_names[i]
        v = basis.lu.solve({i: Fraction(1)})
        down = up = INF
        for p, k in enumerate(basis.basic):
            vp = v.get(p)
            if vp:
                gd, gu = gaps(k)
                dn, u = _ratio(vp, gd, gu)
                down, up = min(down, dn), min(up, u)
        down, up = max(down, Fraction(0)), max(up, Fraction(0))
        y = basis.y.get(i, Fraction(0))
        right = y if up > 0 else _one_sided_derivative(basis, i, +1, deadline)
        left = y if down > 0 else _one_sided_derivative(basis, i, -1, deadline)
        rows[name] = {"dual": sense * y, "range_lower": down, "range_upper": up,
                      "left": sense * left, "right": sense * right}
    return {"columns": columns, "rows": rows}


def fraction_text(value) -> str:
    """The .sol spelling of an exact value: `inf`, `-inf`, or `numerator/denominator`."""
    if value == INF:
        return "inf"
    if value == -INF:
        return "-inf"
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(text: str):
    if text == "inf":
        return INF
    if text == "-inf":
        return -INF
    return Fraction(text)


# ---------------------------------------------------------------------------------------
# The check verify_solution.py runs on a .sol that carries a certified sensitivity report
# ---------------------------------------------------------------------------------------

# include/sankhya/tolerances.hpp kSensitivityAgreement: a float value is `certified` when
# within this relative distance, max(1, |exact|), of the exact one.
AGREEMENT = Fraction(1e-9)


def agrees(reported: float, exact) -> bool:
    if exact in (INF, -INF):
        return reported == exact
    if reported in (INF, -INF) or reported != reported:
        return False
    return abs(Fraction(reported) - exact) <= AGREEMENT * max(Fraction(1), abs(exact))


def check_certified(model, solution, report, seconds: float = 600.0) -> None:
    """Every exact value the solver wrote must equal the one re-derived here, and every
    `certified` / `corrected` verdict must be the one the float values earn (#757)."""
    columns = solution.exact_sensitivity_columns
    rows = solution.exact_sensitivity_rows
    if not columns and not rows:
        return
    try:
        exact = derive(model, solution, seconds)
    except Declined as why:
        report.check(False, "certified sensitivity: re-derived in exact arithmetic",
                     f"could not re-derive: {why}")
        return
    wrong: list[str] = []
    stated_certified = 0
    for name, (verdict, *values) in columns.items():
        want = exact["columns"].get(name)
        if want is None:
            wrong.append(f"column {name}: not in the model")
            continue
        keys = ("reduced_cost", "range_lower", "range_upper")
        if values != [fraction_text(want[k]) for k in keys]:
            wrong.append(f"column {name}: exact values differ")
        earned = (agrees(solution.col_dual.get(name, 0.0), want["reduced_cost"])
                  and agrees(solution.col_ranging_lower.get(name, INF), want["range_lower"])
                  and agrees(solution.col_ranging_upper.get(name, INF), want["range_upper"]))
        if (verdict == "certified") != earned:
            wrong.append(f"column {name}: stated {verdict}")
        stated_certified += verdict == "certified"
    kinks = 0
    for name, (verdict, *values) in rows.items():
        want = exact["rows"].get(name)
        if want is None:
            wrong.append(f"row {name}: not in the model")
            continue
        keys = ("dual", "range_lower", "range_upper", "left", "right")
        if values != [fraction_text(want[k]) for k in keys]:
            wrong.append(f"row {name}: exact values differ")
        earned = (agrees(solution.row_dual.get(name, 0.0), want["dual"])
                  and agrees(solution.row_ranging_lower.get(name, INF), want["range_lower"])
                  and agrees(solution.row_ranging_upper.get(name, INF), want["range_upper"]))
        if (verdict == "certified") != earned:
            wrong.append(f"row {name}: stated {verdict}")
        stated_certified += verdict == "certified"
        kinks += want["left"] != want["right"]
    report.check(
        not wrong, "certified sensitivity: every exact dual, range and shadow price interval",
        f"{len(columns)} column(s), {len(rows)} row(s) re-derived from the basis; "
        f"{stated_certified} certified; {kinks} row(s) with a two-sided shadow price"
        + (f"; MISMATCH {', '.join(wrong[:5])}" if wrong else ""))
