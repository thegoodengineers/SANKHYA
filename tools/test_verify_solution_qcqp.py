# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on a model with quadratic ROWS (#514).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too.

Haverly's first pooling problem (ACM SIGMAP Bulletin 25, 1978) in the P-formulation, with
QCMATRIX rows in the CPLEX/Gurobi convention: the full symmetric matrix, no 1/2, so the term
1 * p * f is the two entries (p, f) and (f, p) of 0.5 each. The global optimum costs -400:
crude B (1 %) through the pool, 100 units to product Y, with 100 units of C (2 %) direct to Y.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import verify_solution as vs

HAVERLY1_P = """\
NAME haverly1_p
ROWS
 N cost
 L src_1
 L src_2
 L src_3
 L dem_5
 L dem_6
 E bal_4
 L cap_4
 E pq_4_1
 L tq_5_1
 L tq_6_1
COLUMNS
 f_1_4 cost 6
 f_1_4 src_1 1
 f_1_4 bal_4 1
 f_1_4 cap_4 1
 f_1_4 pq_4_1 3
 f_2_4 cost 16
 f_2_4 src_2 1
 f_2_4 bal_4 1
 f_2_4 cap_4 1
 f_2_4 pq_4_1 1
 f_3_5 cost 1
 f_3_5 src_3 1
 f_3_5 dem_5 1
 f_3_5 tq_5_1 -0.5
 f_3_6 cost -5
 f_3_6 src_3 1
 f_3_6 dem_6 1
 f_3_6 tq_6_1 0.5
 f_4_5 cost -9
 f_4_5 dem_5 1
 f_4_5 bal_4 -1
 f_4_5 tq_5_1 -2.5
 f_4_6 cost -15
 f_4_6 dem_6 1
 f_4_6 bal_4 -1
 f_4_6 tq_6_1 -1.5
 p_4_1 cost 0
RHS
 rhs src_1 300
 rhs src_2 300
 rhs src_3 300
 rhs dem_5 100
 rhs dem_6 200
 rhs cap_4 300
BOUNDS
 UP bnd f_1_4 300
 UP bnd f_2_4 300
 UP bnd f_3_5 100
 UP bnd f_3_6 200
 UP bnd f_4_5 100
 UP bnd f_4_6 200
 LO bnd p_4_1 1
 UP bnd p_4_1 3
QCMATRIX pq_4_1
 f_4_5 p_4_1 -0.5
 p_4_1 f_4_5 -0.5
 f_4_6 p_4_1 -0.5
 p_4_1 f_4_6 -0.5
QCMATRIX tq_5_1
 f_4_5 p_4_1 0.5
 p_4_1 f_4_5 0.5
QCMATRIX tq_6_1
 f_4_6 p_4_1 0.5
 p_4_1 f_4_6 0.5
ENDATA
"""

OPTIMUM = {"f_1_4": 0.0, "f_2_4": 100.0, "f_3_5": 0.0, "f_3_6": 100.0, "f_4_5": 0.0,
           "f_4_6": 100.0, "p_4_1": 1.0}


def _activities(model: "vs.Model", point: dict[str, float]) -> dict[str, float]:
    """A x plus the products, computed here the plain way, as a solver would report them."""
    x = [point[name] for name in model.col_names]
    activity = [0.0] * model.num_rows
    for j, column in enumerate(model.entries):
        for i, value in column:
            activity[i] += value * x[j]
    model.add_quadratic_rows(x, activity, [1.0] * model.num_rows)
    return {name: activity[i] for i, name in enumerate(model.row_names)}


def _solution(model, point, objective, bound, status="optimal") -> "vs.Solution":
    solution = vs.Solution()
    solution.header = {"status": status, "objective": repr(objective),
                       "dual_bound": repr(bound), "mip_relative_gap": "0.0001",
                       "mip_absolute_gap": "1e-06"}
    solution.col_value = dict(point)
    solution.col_status = {name: "unknown" for name in point}
    solution.col_dual = {name: 0.0 for name in point}
    solution.row_activity = _activities(model, point)
    solution.row_dual = {name: 0.0 for name in model.row_names}
    return solution


def _verify(model, solution):
    return vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                     vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)


def run(check) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "haverly1_p.mps"
        path.write_text(HAVERLY1_P)
        model = vs.parse_mps(path)

        check(len(model.qc_entries) == 8, "every QCMATRIX entry is kept as listed",
              f"{len(model.qc_entries)} entries")
        activity = _activities(model, OPTIMUM)
        # pq_4_1: 3 f_1_4 + f_2_4 - p (f_4_5 + f_4_6) = 0 + 100 - 1 * 100 = 0.
        check(abs(activity["pq_4_1"]) < 1e-12, "the pool balance is 0 at the optimum",
              f"{activity['pq_4_1']}")
        # tq_6_1: p f_4_6 + 0.5 f_3_6 - 1.5 f_4_6 = 100 + 50 - 150 = 0: (c/2 + c/2) * p * f.
        check(abs(activity["tq_6_1"]) < 1e-12, "(p, f) and (f, p) of 0.5 each are one product",
              f"{activity['tq_6_1']}")

        report = _verify(model, _solution(model, OPTIMUM, -400.0, -400.0))
        check(report.failures == 0, "the global optimum verifies",
              f"{report.failures} check(s) failed")

        # Linear rows satisfied, product rows not: the pool is said to be at 1 % while all
        # of crude A (3 %) flows through it. Only a verifier that reads QCMATRIX sees it.
        wrong = dict(OPTIMUM, f_1_4=100.0, f_2_4=0.0)
        report = _verify(model, _solution(model, wrong, 6 * 100 - 15 * 100 - 5 * 100, -1e9,
                                          status="feasible"))
        check(report.failures > 0, "a point that breaks a quadratic row is rejected",
              f"{report.failures} check(s) failed, as expected")

        # `optimal` whose bound is too far from the objective: the gap claim is checked.
        report = _verify(model, _solution(model, OPTIMUM, -400.0, -450.0))
        check(report.failures > 0, "an optimal claim with an open gap is rejected",
              f"{report.failures} check(s) failed, as expected")
