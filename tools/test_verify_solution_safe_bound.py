# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py re-deriving an optimal LP's safe bound exactly (#763).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too.
"""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import verify_solution as vs
from verify_solution_safe_bound import exact_safe_bound, verify_safe_bound


def _lp(cost, upper, row_lower, row_upper, maximize=False) -> "vs.Model":
    """Columns x0.. with lower bound 0, one row r0 with every coefficient 1."""
    model = vs.Model()
    model.maximize = maximize
    model.col_names = [f"x{j}" for j in range(len(cost))]
    model.col_index = {name: j for j, name in enumerate(model.col_names)}
    model.col_cost = list(cost)
    model.col_lower = [0.0] * len(cost)
    model.col_upper = list(upper)
    model.col_integer = [False] * len(cost)
    model.row_names = ["r0"]
    model.row_index = {"r0": 0}
    model.row_lower = [row_lower]
    model.row_upper = [row_upper]
    model.entries = [[(0, 1.0)] for _ in cost]
    return model


def _verdict(model, multiplier, stated):
    solution = vs.Solution()
    solution.header = {"status": "optimal", "safe_lower_bound": repr(stated)}
    solution.safe_multipliers = {"r0": multiplier}
    report = vs.Report()
    verify_safe_bound(model, solution, 0.0, report)
    return report


def run(check) -> None:
    inf = vs.INF
    # min x0 + x1, x0 + x1 >= 2: the dual 1 proves exactly 2.
    covering = _lp([1.0, 1.0], [inf, inf], 2.0, inf)
    check(exact_safe_bound(covering, [1.0]) == 2, "safe bound: the optimal dual is exact")
    check(_verdict(covering, 1.0, 2.0).failures == 0, "safe bound: a stated 2 is accepted")
    check(_verdict(covering, 1.0, 2.0 + 1e-12).failures == 1,
          "safe bound: a stated bound one step above the exact one is rejected")

    # A multiplier moved off the optimum proves exactly what it proves, and no more.
    moved = 1.0 - 1e-7
    check(exact_safe_bound(covering, [moved]) == Fraction(moved) * 2,
          "safe bound: a moved multiplier gives its own exact bound")
    check(_verdict(covering, moved, 2.0).failures == 1,
          "safe bound: a moved multiplier does not prove the optimum")
    # The file must carry the multipliers a finite bound was proved from.
    missing = vs.Solution()
    missing.header = {"status": "optimal", "safe_lower_bound": "2"}
    report = vs.Report()
    verify_safe_bound(covering, missing, 0.0, report)
    check(report.failures == 1, "safe bound: a finite bound without its multipliers fails")

    # min -x0, x0 + x1 <= 3: x0 has no upper bound, and the dual -0.5 (not optimal) leaves it
    # a negative reduced cost. The row implies x0 <= 3, so the bound is -1.5 - 1.5 = -3.
    implied = _lp([-1.0, 0.0], [inf, inf], -inf, 3.0)
    check(exact_safe_bound(implied, [-0.5]) == -3,
          "safe bound: a missing column bound is implied by the row")
    # Without the row's side there is nothing to imply it from: no finite bound.
    free = _lp([-1.0, 0.0], [inf, inf], -inf, inf)
    check(exact_safe_bound(free, [-0.5]) is None,
          "safe bound: no row, no implied bound, no finite bound")
    check(_verdict(free, -0.5, -3.0).failures == 1,
          "safe bound: a finite bound these duals cannot prove is rejected")

    # A column bound the file claims is checked against its row before it is used.
    claimed = vs.Solution()
    claimed.header = {"status": "optimal", "safe_lower_bound": "-3"}
    claimed.safe_multipliers = {"r0": -0.5}
    claimed.safe_column_bounds = [("x0", "upper", 3.0, "r0")]
    report = vs.Report()
    verify_safe_bound(implied, claimed, 0.0, report)
    check(report.failures == 0, "safe bound: a column bound its row implies is accepted")
    claimed.safe_column_bounds = [("x0", "upper", 2.5, "r0")]
    report = vs.Report()
    verify_safe_bound(implied, claimed, 0.0, report)
    check(report.failures == 1, "safe bound: a column bound tighter than its row is rejected")

    # Both names in a column-bound record may carry spaces (Netlib forplan: "A   21 1" from
    # row "AZ  20"), so both may be quoted, and the spaces inside survive.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quoted.sol"
        path.write_text('status optimal\nbegin safe_column_bounds 2\n'
                        '"A   21 1" upper 0 "AZ  20"\nx1 lower -2.5 r0\n'
                        'end safe_column_bounds\n')
        records = vs.parse_sol(path).safe_column_bounds
    check(records == [("A   21 1", "upper", 0.0, "AZ  20"), ("x1", "lower", -2.5, "r0")],
          "safe bound: column-bound records with quoted names parse", repr(records))

    # max x0 + x1, x0 + x1 <= 2: the bound is an UPPER bound, and one below 2 is wrong.
    packing = _lp([1.0, 1.0], [inf, inf], -inf, 2.0, maximize=True)
    check(_verdict(packing, 1.0, 2.0).failures == 0,
          "safe bound: maximise, the exact upper bound is accepted")
    check(_verdict(packing, 1.0, 2.0 - 1e-12).failures == 1,
          "safe bound: maximise, a bound below the optimum is rejected")
