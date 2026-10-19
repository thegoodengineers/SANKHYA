# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on the conic interior point's answers (#755): a convex quadratic row
held to the KKT conditions through its gradient, and a CSECTION cone held to exact membership,
dual membership and complementarity.

Run from test_verify_solution.py's main(), which passes its `check`.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import verify_solution as vs

# min -x1 - 2 x2  s.t.  x1^2 + x2^2 <= 1, x free: x = (1, 2) / sqrt 5, objective -sqrt 5, and
# stationarity -1 - 2 y x1 = 0 gives the row's multiplier y = -sqrt 5 / 2.
DISC_QCMATRIX = """\
NAME disc
ROWS
 N obj
 L ball
COLUMNS
 x1 obj -1
 x2 obj -2
RHS
 rhs ball 1
BOUNDS
 FR bnd x1
 FR bnd x2
QCMATRIX ball
 x1 x1 1
 x2 x2 1
ENDATA
"""

# min -x1 - x2  s.t.  (t, x1, x2) in the quadratic cone, t = 1: x = (1, 1) / sqrt 2. The row
# t = 1 has multiplier y = -sqrt 2, so the cone's duals are d = c - A'y = (sqrt 2, -1, -1),
# in the cone, with x'd = sqrt 2 - sqrt 2 = 0.
DISC_CSECTION = """\
NAME disc_cone
ROWS
 N obj
 E tfix
COLUMNS
 t obj 0 tfix 1
 x1 obj -1
 x2 obj -1
RHS
 rhs tfix 1
BOUNDS
 FR bnd t
 FR bnd x1
 FR bnd x2
CSECTION disc 0.0 QUAD
 t
 x1
 x2
ENDATA
"""


def _activities(model, x: dict[str, float]) -> dict[str, float]:
    values = [x[name] for name in model.col_names]
    activity = [0.0] * model.num_rows
    for j, column in enumerate(model.entries):
        for i, value in column:
            activity[i] += value * values[j]
    model.add_quadratic_rows(values, activity, [1.0] * model.num_rows)
    return {name: activity[i] for i, name in enumerate(model.row_names)}


def _solution(model, x, objective, row_dual, col_dual, status="optimal", algorithm="conic-ipm"):
    s = vs.Solution()
    s.header = {"status": status, "objective": repr(objective), "dual_bound": repr(objective),
                "algorithm": algorithm}
    s.col_value = dict(x)
    s.col_status = {name: "unknown" for name in x}
    s.col_dual = dict(col_dual)
    s.row_activity = _activities(model, x)
    s.row_dual = dict(row_dual)
    s.row_status = {name: "unknown" for name in model.row_names}
    return s


def _verify(model, solution):
    return vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                     vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)


def run(check) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "disc.mps"
        path.write_text(DISC_QCMATRIX)
        model = vs.parse_mps(path)
        r5 = math.sqrt(5.0)
        x = {"x1": 1.0 / r5, "x2": 2.0 / r5}
        good = _solution(model, x, -r5, {"ball": -r5 / 2.0}, {"x1": 0.0, "x2": 0.0})
        report = _verify(model, good)
        check(report.failures == 0, "a convex quadratic row's KKT point verifies",
              f"{report.failures} check(s) failed")

        # The same point with the row's multiplier dropped: stationarity fails, since the
        # reduced costs the solver reports (0) no longer follow from c - y (a + 2 Q x).
        bad = _solution(model, x, -r5, {"ball": 0.0}, {"x1": 0.0, "x2": 0.0})
        check(_verify(model, bad).failures > 0, "a quadratic row with a wrong multiplier fails",
              "stationarity through the row's gradient is checked")

        # The global method's answer, with no multipliers: judged by its bound, as before.
        bound_only = _solution(model, x, -r5, {"ball": 0.0}, {"x1": 0.0, "x2": 0.0},
                               algorithm="spatial-bnb")
        check(_verify(model, bound_only).failures == 0,
              "a global answer is still judged by its bound, not by KKT",
              f"{_verify(model, bound_only).failures} check(s) failed")

        path = Path(tmp) / "disc_cone.mps"
        path.write_text(DISC_CSECTION)
        model = vs.parse_mps(path)
        check(model.cones == [("disc", "QUAD", [0, 1, 2])], "CSECTION is read",
              f"{model.cones}")
        r2 = math.sqrt(2.0)
        h = math.sqrt(0.5)
        x = {"t": 1.0, "x1": h, "x2": h}
        good = _solution(model, x, -r2, {"tfix": -r2}, {"t": r2, "x1": -1.0, "x2": -1.0})
        report = _verify(model, good)
        check(report.failures == 0, "a cone's optimum with its duals verifies",
              f"{report.failures} check(s) failed")

        # A point outside the cone: x_0^2 < x_1^2 + x_2^2, caught exactly.
        outside = _solution(model, {"t": 1.0, "x1": 0.8, "x2": 0.8}, -1.6, {"tfix": 0.0},
                            {"t": 0.0, "x1": 0.0, "x2": 0.0}, status="feasible")
        check(_verify(model, outside).failures > 0, "a point outside the cone is rejected",
              "exact membership")

        # Duals outside the dual cone: d_t = 1 < ||(-1, -1)||.
        wrong_dual = _solution(model, x, -r2, {"tfix": -1.0}, {"t": 1.0, "x1": -1.0, "x2": -1.0})
        check(_verify(model, wrong_dual).failures > 0, "duals outside the dual cone are rejected",
              "exact dual membership")
