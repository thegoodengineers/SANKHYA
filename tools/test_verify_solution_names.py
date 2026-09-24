# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on a vector name that is also a row or column name (#590).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too.

Maros-Meszaros dpklo1 names its rows and columns "1", "2", ..., its RHS vector "1" and its
bound vector "0". The reader used to decide whether an RHS line carried a vector name by
asking whether the first token named a row; on dpklo1 it always does, so "1 77 3.577" was
read as "row 1 = 77", the last such line won, and a correct point came back rejected with
row 1 violated by 77 while the objective matched the reference to 2.9e-9. The token count
is what decides (the payload is (row, value) pairs, so an odd count means a name leads), and
the same trap on BOUNDS ("UP 1 2 10" beside a column "1") is decided the same way.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import verify_solution as vs

# A reduced dpklo1: the vector names collide with row and column names on purpose.
#   min  x1^2/2 + x2^2/2 + x3      s.t.  x1 + x2 = 4,  x2 + x3 = 7,  x1 free, x2 <= 10
# The RHS lines are written the way dpklo1 writes them, vector "1" first; under the old
# reading row 1 ends up at 2 (from "1 2 7.0") and row 2 is never set.
DPKLO1_LIKE = """\
NAME          DPKLO1-LIKE
ROWS
  E        1
  E        2
  N        3
COLUMNS
           1         1     1.0000000
           2         1     1.0000000
           2         2     1.0000000
           3         2     1.0000000
           3         3     1.0000000
RHS
           1         1     4.0000000
           1         2     7.0000000
BOUNDS
 FR        1         1
 UP        1         2    10.0000000
QUADOBJ
           1         1      1.000000
           2         2      1.000000
ENDATA
"""

# The optimum: x3 is linear with cost 1 and x3 = 7 - x2 >= 0 caps x2 at 7; the objective
# (4 - x2)^2/2 + x2^2/2 + 7 - x2 has derivative 2 x2 - 5, zero at x2 = 2.5, so x = (1.5, 2.5,
# 4.5) and the objective 1.125 + 3.125 + 4.5 = 8.75. Stationarity: column 1 is free, so
# y1 = x1 = 1.5; column 3 is basic at 4.5, so y2 = 1; column 2 then prices 2.5 - 1.5 - 1 = 0.
DPKLO1_LIKE_SOL = """\
model DPKLO1-LIKE
sense minimize
status optimal
algorithm convex-qp
objective 8.75
objective_offset 0
certificate none

begin columns 3
1 1.5 0 unknown
2 2.5 0 unknown
3 4.5 0 unknown
end columns

begin rows 2
1 4 1.5 unknown
2 7 1 unknown
end rows
"""


def run(check) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        qps = Path(tmp) / "dpklo1_like.qps"
        qps.write_text(DPKLO1_LIKE)
        model = vs.parse_mps(qps)

        check(model.row_lower == [4.0, 7.0] and model.row_upper == [4.0, 7.0],
              "an RHS vector named like a row is read as the vector name",
              f"row bounds {list(zip(model.row_lower, model.row_upper))}, expected 4 and 7")
        check(model.col_lower[0] == -vs.INF and model.col_upper[0] == vs.INF,
              "FR with a bound vector named like a column frees the named column",
              f"column 1 in [{model.col_lower[0]}, {model.col_upper[0]}]")
        check(model.col_upper[1] == 10.0 and model.col_upper[0] == vs.INF,
              "UP with a bound vector named like a column bounds the named column",
              f"upper bounds {model.col_upper}")

        sol = Path(tmp) / "dpklo1_like.sol"
        sol.write_text(DPKLO1_LIKE_SOL)
        report = vs.verify(model, vs.parse_sol(sol), 1e-7, 1e-7, 1e-6, 1e-6)
        failed = [name for ok, name, _ in report.lines if not ok]
        check(report.failures == 0, "the true optimum of the reduced dpklo1 verifies",
              str(failed))
