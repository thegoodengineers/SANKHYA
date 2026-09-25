# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on a nonlinear (.nl) model (NLP stage 2).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too. Every expected value is derived by
hand in a comment.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import verify_solution as vs
from verify_solution_nl import read_nl
from verify_solution_nlp import verify_nlp
from verify_solution_sol import Solution

# min x0^2 + x1^2 subject to x0 + x1 >= 1, both columns free. The optimum is (1/2, 1/2): the
# gradient (1, 1) equals the row multiplier times the row's gradient (1, 1), so y = 1, and the
# row sits at its lower bound, which a positive multiplier prices.
DISC = """g3 1 1 0
 2 1 1 0 0
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0 0 0 0
 2 2
 0 0
 0 0 0 0 0
C0
n0
O0 0
o54
2
o5
v0
n2
o5
v1
n2
r
2 1
b
3
3
J0 2
0 1
1 1
G0 2
0 0
1 0
"""

# HS071 exactly as test_nl_reader.cpp writes it by hand from the specification.
HS071 = """g3 1 1 0
 4 2 1 0 1
 2 1
 0 0
 4 4 4
 0 0 0 1
 0 0 0 0 0
 8 4
 0 0
 0 0 0 0 0
C0
o2
o2
o2
v0
v1
v2
v3
C1
o54
4
o5
v0
n2
o5
v1
n2
o5
v2
n2
o5
v3
n2
O0 0
o2
o2
v0
v3
o54
3
v0
v1
v2
r
2 25
4 40
b
0 1 5
0 1 5
0 1 5
0 1 5
J0 4
0 0
1 0
2 0
3 0
J1 4
0 0
1 0
2 0
3 0
G0 4
0 0
1 0
2 1
3 0
"""


def _solution(status, x, row_value, row_dual, col_dual, objective) -> Solution:
    s = Solution()
    s.header = {"status": status, "objective": repr(objective)}
    s.col_value = {f"C{j}": v for j, v in enumerate(x)}
    s.col_dual = {f"C{j}": v for j, v in enumerate(col_dual)}
    s.row_activity = {"R0": row_value}
    s.row_dual = {"R0": row_dual}
    return s


def _verify(model, solution):
    report = vs.Report()
    verify_nlp(model, solution, report, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
               vs.DEFAULT_INTEGER_TOL)
    return report


def run(check) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        disc = Path(tmp) / "disc.nl"
        disc.write_text(DISC)
        model = read_nl(disc)
        check(model.n == 2 and model.m == 1 and model.col_names == ["C0", "C1"]
              and model.row_names == ["R0"], "a .nl model is read, with the writer's names")
        # The evaluator's gradient by hand at (0.3, -2): (2 x0, 2 x1) = (0.6, -4).
        f = model.objective([0.3, -2.0])
        check(abs(f.v - 4.09) < 1e-15 and abs(f.g[0] - 0.6) < 1e-15 and abs(f.g[1] + 4.0) < 1e-15,
              "the objective and its forward-mode gradient", f"{f.v} {f.g}")

        right = _solution("locally_optimal", [0.5, 0.5], 1.0, 1.0, [0.0, 0.0], 0.5)
        report = _verify(model, right)
        check(report.failures == 0, "a KKT point verifies", f"{report.failures} failed")

        wrong_sign = _solution("locally_optimal", [0.5, 0.5], 1.0, -1.0, [0.0, 0.0], 0.5)
        report = _verify(model, wrong_sign)
        check(report.failures >= 2, "a multiplier of the wrong sign fails stationarity and sign",
              f"{report.failures} failed, as expected")

        # (0.4, 0.4) breaks the row: 0.8 < 1, and its objective 0.32 is stated correctly.
        outside = _solution("locally_optimal", [0.4, 0.4], 0.8, 0.8, [0.0, 0.0], 0.32)
        report = _verify(model, outside)
        check(report.failures >= 1, "an infeasible point claimed optimal is rejected",
              f"{report.failures} failed, as expected")

        lying = _solution("locally_optimal", [0.5, 0.5], 1.0, 1.0, [0.0, 0.0], 0.4)
        report = _verify(model, lying)
        check(report.failures >= 1, "a misstated objective is rejected",
              f"{report.failures} failed, as expected")

        nothing = Solution()
        nothing.header = {"status": "locally_infeasible"}
        report = _verify(model, nothing)
        check(report.failures == 0, "locally_infeasible carries no point and none is judged",
              f"{report.failures} failed")

        hs = Path(tmp) / "hs071.nl"
        hs.write_text(HS071)
        model = read_nl(hs)
        # The published optimum of HS071 (Hock and Schittkowski 1981, problem 71), to the
        # digits it is published with: f = 17.0140173 at (1, 4.7429994, 3.8211503, 1.3794082).
        x = [1.0, 4.7429994, 3.8211503, 1.3794082]
        f = model.objective(x)
        rows = model.rows(x)
        check(abs(f.v - 17.0140173) < 1e-6, "HS071's published point gives its published value",
              f"{f.v:.10f}")
        check(abs(rows[0].v - 25.0) < 1e-5 and abs(rows[1].v - 40.0) < 1e-5,
              "and satisfies its rows to the published digits",
              f"{rows[0].v:.8f} {rows[1].v:.8f}")
        # The Jacobian of row 0 (x0 x1 x2 x3) is (x1 x2 x3, x0 x2 x3, x0 x1 x3, x0 x1 x2).
        expected = [x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]]
        check(all(math.isclose(rows[0].g[j], expected[j], rel_tol=1e-14) for j in range(4)),
              "the product row's Jacobian by hand")
