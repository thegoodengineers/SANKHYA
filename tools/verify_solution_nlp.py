# SPDX-License-Identifier: Apache-2.0
"""The KKT checks of a nonlinear model's solution, for verify_solution.py (NLP stage 2).

Called by verify_solution.py when the model is a .nl file. The model is read by
verify_solution_nl.py (this checker's own reader and evaluator) and the answer by
verify_solution_sol.py, and every number is recomputed from (model, x, multipliers):

  column bounds and integrality, recomputed row values against their bounds and against the
  activities the file reports, the objective against the header, and - for `optimal` and
  `locally_optimal` - stationarity grad f - J^T y - d = 0, the multipliers' signs and
  complementary slackness, in minimisation space and with the scalings of the LP checks in
  verify_solution.py, the gradient and the Jacobian at x standing in for c and A.

WHAT IT CANNOT CHECK. A KKT point is a local optimum. `optimal` on a nonlinear model adds the
claim that it is GLOBAL, which the solver bases on proving the model convex; this script
does not re-derive convexity, and says so in its report rather than passing it silently.
"""

from __future__ import annotations

import math

from verify_solution_nl import INF, NlModel
from verify_solution_sol import Solution

STATUSES_WITH_A_POINT = ("optimal", "locally_optimal", "feasible", "iteration_limit",
                         "time_limit", "node_limit", "interrupted")
STATUSES_ASSERTING_FEASIBILITY = ("optimal", "locally_optimal", "feasible")
STATUSES_CLAIMING_KKT = ("optimal", "locally_optimal")
COMPLEMENTARITY_TOL = 1e-6  # verify_solution.py's constant for the same check


def _sign_violation(multiplier: float, lower: float, upper: float) -> float:
    if lower == upper:
        return 0.0
    if multiplier > 0.0 and not math.isfinite(lower):
        return multiplier
    if multiplier < 0.0 and not math.isfinite(upper):
        return -multiplier
    return 0.0


def _complementarity(multiplier: float, value: float, lower: float, upper: float) -> float:
    slack = min(value - lower if math.isfinite(lower) else INF,
                upper - value if math.isfinite(upper) else INF)
    return abs(multiplier) if not math.isfinite(slack) else abs(multiplier) * max(slack, 0.0)


def verify_nlp(model: NlModel, solution: Solution, report, primal_tol: float, dual_tol: float,
               integer_tol: float) -> None:
    status = solution.status
    report.note("class", "nonlinear (.nl), read and differentiated by this checker's own code")
    if status not in STATUSES_WITH_A_POINT:
        report.note("point", f"status {status} carries no point, and none is checked")
        return
    missing = [n for n in model.col_names if n not in solution.col_value] + \
              [n for n in model.row_names if n not in solution.row_activity]
    if not report.check(not missing, "structure",
                        f"{model.m} rows, {model.n} columns" if not missing
                        else f"missing {len(missing)} names, e.g. {missing[0]}"):
        return
    x = [solution.col_value[n] for n in model.col_names]
    try:
        objective = model.objective(x)
        rows = model.rows(x)
    except (ValueError, OverflowError, KeyError) as error:
        report.check(False, "evaluation", f"the model is undefined at the point: {error}")
        return
    feasible_claim = status in STATUSES_ASSERTING_FEASIBILITY

    def primal(worst: float, name: str, detail: str) -> None:
        if feasible_claim:
            report.check(worst <= primal_tol, name, detail)
        else:
            report.note(name, detail + f" - status {status} asserts no feasibility")

    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        v = max(model.col_lower[j] - x[j], x[j] - model.col_upper[j], 0.0) / max(1.0, abs(x[j]))
        if v > worst:
            worst, where = v, name
    primal(worst, "column bounds", f"worst {worst:.3e} relative" + (f" on {where}" if where else ""))
    if any(model.integer):
        worst, where = 0.0, ""
        for j, name in enumerate(model.col_names):
            if model.integer[j] and abs(x[j] - round(x[j])) > worst:
                worst, where = abs(x[j] - round(x[j])), name
        detail = f"worst {worst:.3e}" + (f" on {where}" if where else "")
        if feasible_claim:
            report.check(worst <= integer_tol, "integrality", detail)
        else:
            report.note("integrality", detail + f" - status {status} asserts no feasibility")
    worst, where, agree = 0.0, "", 0.0
    for i, name in enumerate(model.row_names):
        g = rows[i].v
        v = max(model.row_lower[i] - g, g - model.row_upper[i], 0.0) / max(1.0, abs(g))
        if v > worst:
            worst, where = v, name
        agree = max(agree, abs(g - solution.row_activity[name]) / max(1.0, abs(g)))
    primal(worst, "rows (recomputed)", f"worst {worst:.3e} relative" + (f" on {where}" if where else ""))
    report.check(agree <= primal_tol, "row values agree", f"worst {agree:.3e} relative")
    stated = solution.header_float("objective")
    if stated is not None:
        gap = abs(objective.v - stated) / max(1.0, abs(objective.v))
        report.check(gap <= 1e-9, "objective (recomputed)",
                     f"{objective.v:.12g} vs stated {stated:.12g}")

    if status not in STATUSES_CLAIMING_KKT:
        report.note("multipliers", f"status {status} claims no optimality; not checked")
        return
    sigma = -1.0 if model.maximize else 1.0
    y = [sigma * solution.row_dual.get(name, 0.0) for name in model.row_names]
    d = [sigma * solution.col_dual.get(name, 0.0) for name in model.col_names]
    grad = [sigma * objective.g.get(j, 0.0) for j in range(model.n)]
    jt_y = [0.0] * model.n
    scale = [max(1.0, abs(g)) for g in grad]
    for i, row in enumerate(rows):
        for j, a in row.g.items():
            term = y[i] * a
            jt_y[j] += term
            scale[j] = max(scale[j], abs(term))
    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        r = abs(grad[j] - jt_y[j] - d[j]) / scale[j]
        if r > worst:
            worst, where = r, name
    report.check(worst <= dual_tol, "stationarity",
                 f"max |grad f - J^T y - d| {worst:.3e} relative to its terms"
                 + (f" on {where}" if where else ""))
    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        v = _sign_violation(d[j], model.col_lower[j], model.col_upper[j]) / scale[j]
        if v > worst:
            worst, where = v, name
    y_norm = max([1.0] + [abs(v) for v in y])
    for i, name in enumerate(model.row_names):
        v = _sign_violation(y[i], model.row_lower[i], model.row_upper[i]) / y_norm
        if v > worst:
            worst, where = v, name
    report.check(worst <= dual_tol, "multiplier signs",
                 f"worst {worst:.3e} relative" + (f" on {where}" if where else ""))
    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        if model.col_lower[j] != model.col_upper[j]:
            c = _complementarity(d[j], x[j], model.col_lower[j], model.col_upper[j])
            if c > worst:
                worst, where = c, name
    for i, name in enumerate(model.row_names):
        if model.row_lower[i] != model.row_upper[i]:
            c = _complementarity(y[i], rows[i].v, model.row_lower[i], model.row_upper[i])
            if c > worst:
                worst, where = c, name
    report.check(worst <= COMPLEMENTARITY_TOL, "complementary slackness",
                 f"worst |multiplier| * slack = {worst:.3e}" + (f" on {where}" if where else ""))
    if status == "optimal":
        report.note("global claim", "`optimal` on a nonlinear model asserts a GLOBAL optimum, "
                    "which the solver bases on a convexity proof this checker does not re-derive; "
                    "what is verified here is a KKT point")


def main_nonlinear(args, report) -> int:
    """verify_solution.py's verdict for a .nl model: read, check, print, exit code."""
    import sys

    from verify_solution_nl import read_nl
    from verify_solution_sol import parse_sol
    try:
        model = read_nl(args.model)
    except (OSError, ValueError, StopIteration) as error:
        print(f"cannot read the model: {error}", file=sys.stderr)
        return 2
    try:
        solution = parse_sol(args.solution)
    except (OSError, ValueError) as error:
        print(f"cannot read the solution: {error}", file=sys.stderr)
        return 2
    verify_nlp(model, solution, report, args.primal_tolerance, args.dual_tolerance,
               args.integer_tolerance)
    if not args.quiet:
        print(f"model     {args.model}")
        print(f"solution  {args.solution}")
        print(f"problem   {model.m} rows x {model.n} columns  "
              f"{'maximize' if model.maximize else 'minimize'}  nonlinear")
        print(f"status    {solution.status}")
        print()
        print("Independent checks (this script shares no code with the solver):")
        print(report.render())
        print()
    if report.failures == 0:
        print(f"VERIFIED: {len(report.lines)} checks passed")
        return 0
    print(f"REJECTED: {report.failures} of {len(report.lines)} checks failed")
    return 1
