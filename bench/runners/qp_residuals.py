#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The three optimality measures of a convex QP answer, recomputed from the files (#491).

The QP benchmark methodology (qpbenchmark, Caron et al., the published Maros-Meszaros report
at github.com/qpsolvers/maros_meszaros_qpbenchmark, results/maros_meszaros.md) judges a
solver on three numbers at the point it returns, each against a tolerance:

    primal residual   the largest violation of any constraint or bound;
    dual residual     the largest violation of the dual feasibility condition;
    duality gap       the difference between the primal and dual objectives.

It publishes them at 1e-9 ("high accuracy") and 1e-6 ("mid accuracy"), in ABSOLUTE terms.
This module computes each of them both ways: absolute, as qpbenchmark states them, and
relative to the size of the quantities they are differences of, as #491 asks for. Every
relative measure divides by max(1, scale), so it never exceeds the absolute one and the two
agree on a well-scaled model; they part only where the data are large.

NOTHING HERE READS A NUMBER THE SOLVER COMPUTED ABOUT ITSELF. The model is re-read by the
verifier's own MPS reader (tools/verify_solution_mps.py, which shares no code with
src/io/), the point and the row multipliers are read from the .sol file, and everything
else - the activities, c + Qx, A'y, the reduced costs, both objectives - is recomputed.
The QP engine's reduced costs are not used: they are derived as d = c + Qx - A'y, the same
choice tools/verify_solution.py makes, so a solver cannot pass by reporting a convenient d.

CONVENTIONS, the .sol file's and the verifier's. In minimize space (sigma = -1 flips a
maximize model), a row multiplier y_i > 0 prices the row's lower bound and y_i < 0 its
upper bound, and likewise for a column's reduced cost d_j. Stationarity is then

    sigma (c + Qx) = A'y + d

and the dual of the convex QP (Dorn 1960; Nocedal and Wright, Numerical Optimization, 2nd
ed., ch. 12 and 16) has objective

    sum_i y_i * (bound y_i prices) + sum_j d_j * (bound d_j prices) - 0.5 x'(sigma Q)x.

*   The DUAL RESIDUAL is the distance of (y, d) from the set where every multiplier prices a
    bound that exists: a positive multiplier on a side with no lower bound, or a negative
    one on a side with no upper bound, is the violation. This is qpbenchmark's
    "||Px + q + A'y + z||" with the bound multiplier z chosen as the admissible part of d,
    which is the choice most favourable to the solver consistent with its (x, y).
*   The DUALITY GAP is |primal - dual| with those admissible multipliers. At a KKT point it
    is sum of multiplier * (value - priced bound), i.e. the complementarity, signed as
    qpbenchmark states it.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from verify_solution_mps import Model, parse_mps  # noqa: E402,F401
from verify_solution_sol import Solution, parse_sol  # noqa: E402,F401

# The two accuracy levels qpbenchmark publishes: "mid_accuracy" and "high_accuracy".
LEVELS = (1e-6, 1e-9)


@dataclass
class Residuals:
    primal: float
    dual: float
    gap: float
    primal_rel: float
    dual_rel: float
    gap_rel: float
    primal_objective: float
    dual_objective: float

    def meets(self, tolerance: float, relative: bool) -> bool:
        values = ((self.primal_rel, self.dual_rel, self.gap_rel) if relative
                  else (self.primal, self.dual, self.gap))
        return all(math.isfinite(v) and v <= tolerance for v in values)


def admissible(multiplier: float, lower: float, upper: float) -> float:
    """The part of a multiplier that prices a bound which exists. An equality or a fixed
    column admits any sign; a missing side admits none on that side."""
    if lower == upper:
        return multiplier
    if multiplier > 0.0 and not math.isfinite(lower):
        return 0.0
    if multiplier < 0.0 and not math.isfinite(upper):
        return 0.0
    return multiplier


def priced(multiplier: float, lower: float, upper: float) -> float:
    """multiplier * the bound it prices; zero for a zero multiplier. Only ever called on an
    admissible multiplier, so the bound it names is finite."""
    if multiplier > 0.0:
        return multiplier * lower
    if multiplier < 0.0:
        return multiplier * upper
    return 0.0


def _max_abs(values) -> float:
    return max((abs(v) for v in values), default=0.0)


def compute(model: Model, solution: Solution) -> Residuals:
    """The three measures for the point in `solution`, on the model as re-read."""
    sigma = -1.0 if model.maximize else 1.0
    n, m = model.num_cols, model.num_rows
    x = [solution.col_value.get(name, 0.0) for name in model.col_names]
    y = [sigma * solution.row_dual.get(name, 0.0) for name in model.row_names]

    # ---- primal: rows and column bounds, recomputed from x ------------------------------
    activity = [0.0] * m
    for j in range(n):
        if x[j] != 0.0:
            for i, value in model.entries[j]:
                activity[i] += value * x[j]
    primal = 0.0
    for i in range(m):
        primal = max(primal, model.row_lower[i] - activity[i],
                     activity[i] - model.row_upper[i])
    for j in range(n):
        primal = max(primal, model.col_lower[j] - x[j], x[j] - model.col_upper[j])
    primal_scale = max(1.0, _max_abs(activity), _max_abs(x))

    # ---- dual: d = sigma (c + Qx) - A'y, and how far (y, d) are from admissible ----------
    qx = model.hessian_times(x) if model.hessian else [0.0] * n
    aty = [0.0] * n
    for j in range(n):
        aty[j] = sum(value * y[i] for i, value in model.entries[j])
    gradient = [sigma * (model.col_cost[j] + qx[j]) for j in range(n)]
    d = [gradient[j] - aty[j] for j in range(n)]
    y_ok = [admissible(y[i], model.row_lower[i], model.row_upper[i]) for i in range(m)]
    d_ok = [admissible(d[j], model.col_lower[j], model.col_upper[j]) for j in range(n)]
    dual = max(max((abs(y[i] - y_ok[i]) for i in range(m)), default=0.0),
               max((abs(d[j] - d_ok[j]) for j in range(n)), default=0.0))
    dual_scale = max(1.0, _max_abs(model.col_cost), _max_abs(qx), _max_abs(aty))

    # ---- gap: primal objective against the Dorn dual at the admissible multipliers ------
    quadratic = 0.5 * sum(x[j] * qx[j] for j in range(n))
    primal_objective = (model.objective_offset
                        + sum(model.col_cost[j] * x[j] for j in range(n)) + quadratic)
    dual_min_space = (sum(priced(y_ok[i], model.row_lower[i], model.row_upper[i])
                          for i in range(m))
                      + sum(priced(d_ok[j], model.col_lower[j], model.col_upper[j])
                            for j in range(n)))
    # In minimize space the Hessian is sigma Q, so its 0.5 x'(sigma Q)x maps back to the
    # model's sense as sigma * sigma * quadratic = quadratic.
    dual_objective = sigma * dual_min_space + model.objective_offset - quadratic
    gap = abs(primal_objective - dual_objective)
    gap_scale = max(1.0, abs(primal_objective), abs(dual_objective))

    return Residuals(primal=primal, dual=dual, gap=gap,
                     primal_rel=primal / primal_scale, dual_rel=dual / dual_scale,
                     gap_rel=gap / gap_scale, primal_objective=primal_objective,
                     dual_objective=dual_objective)


def compute_files(model_path: Path, sol_path: Path) -> Residuals:
    return compute(parse_mps(model_path), parse_sol(sol_path))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: qp_residuals.py model.qps solution.sol")
    r = compute_files(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"primal residual {r.primal:.3e} (relative {r.primal_rel:.3e})")
    print(f"dual residual   {r.dual:.3e} (relative {r.dual_rel:.3e})")
    print(f"duality gap     {r.gap:.3e} (relative {r.gap_rel:.3e})  "
          f"primal {r.primal_objective:.12e}  dual {r.dual_objective:.12e}")
