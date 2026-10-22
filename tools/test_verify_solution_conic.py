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

# The same disc with t fixed by a bound, not by a row.
DISC_FIXED = """\
NAME disc_fixed
ROWS
 N obj
COLUMNS
 t obj 0
 x1 obj -1
 x2 obj -1
BOUNDS
 FX bnd t 1
 FR bnd x1
 FR bnd x2
CSECTION disc 0.0 QUAD
 t
 x1
 x2
ENDATA
"""


# #1061 review: min t s.t. t >= ||(x1, x2)||, x1 + x2 = 2, x1 and x2 at the MPS default lower
# bound 0. The optimum is sqrt 2 at x = (1, 1). The point t = 2, x = (2, 0) is 41% worse and
# sits with x2 at its own bound; skipping the cone's dual checks there let it pass.
AT_BOUND = """\
NAME probe
ROWS
 N obj
 E sum
COLUMNS
 t obj 1
 x1 sum 1
 x2 sum 1
RHS
 rhs sum 2
CSECTION k 0.0 QUAD
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

        # A cone member at its own bound: the reduced costs are split into the cone's dual
        # and the bound's multiplier, never skipped (#1061 review).
        path = Path(tmp) / "at_bound.mps"
        path.write_text(AT_BOUND)
        bound_model = vs.parse_mps(path)
        wrong = _solution(bound_model, {"t": 2.0, "x1": 2.0, "x2": 0.0}, 2.0, {"sum": 1.0},
                          {"t": 1.0, "x1": -1.0, "x2": -1.0})
        check(_verify(bound_model, wrong).failures > 0,
              "a 41% worse point with a cone member at its bound is rejected",
              "its bound multiplier has the wrong sign once the cone's dual is split off")
        y = 1.0 / math.sqrt(2.0)
        right = _solution(bound_model, {"t": math.sqrt(2.0), "x1": 1.0, "x2": 1.0},
                          math.sqrt(2.0), {"sum": y}, {"t": 1.0, "x1": -y, "x2": -y})
        report = _verify(bound_model, right)
        check(report.failures == 0, "the true optimum of the same model is accepted",
              f"{report.failures} check(s) failed")

        # Duals outside the dual cone: d_t = 1 < ||(-1, -1)||.
        wrong_dual = _solution(model, x, -r2, {"tfix": -1.0}, {"t": 1.0, "x1": -1.0, "x2": -1.0})
        check(_verify(model, wrong_dual).failures > 0, "duals outside the dual cone are rejected",
              "exact dual membership")

        # The same disc with t fixed by a bound instead of a row (#1146 follow-up). The cone's
        # dual is s = sqrt 2 (1, -1/sqrt 2, -1/sqrt 2); the reduced cost of t is c_t = 0, so the
        # bound multiplier is mu_t = 0 - sqrt 2. A fixed member is an equality and either sign
        # is admissible: the true answer must verify. It was rejected as a wrong-signed
        # lower-bound multiplier before.
        path = Path(tmp) / "disc_fixed.mps"
        path.write_text(DISC_FIXED)
        fixed_model = vs.parse_mps(path)
        x_fixed = {"t": 1.0, "x1": 1.0 / r2, "x2": 1.0 / r2}
        true_fixed = _solution(fixed_model, x_fixed, -r2, {}, {"t": 0.0, "x1": -1.0, "x2": -1.0})
        report = _verify(fixed_model, true_fixed)
        check(report.failures == 0, "a fixed cone member's multiplier of either sign is accepted",
              f"{report.failures} check(s) failed")
        # A wrong answer on the same model is still rejected: x = (1, 0.6, 0.8) is on the
        # cone's boundary but not optimal (objective -1.4), and no split of its reduced costs
        # makes the free members' d equal a multiple of the reflection.
        x_wrong = {"t": 1.0, "x1": 0.6, "x2": 0.8}
        wrong_fixed = _solution(fixed_model, x_wrong, -1.4, {}, {"t": 0.0, "x1": -1.0, "x2": -1.0})
        check(_verify(fixed_model, wrong_fixed).failures > 0,
              "a suboptimal point with a fixed cone member is rejected", "free members' d - s")
