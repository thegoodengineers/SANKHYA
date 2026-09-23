# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on a limit that found no point (#505).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too.

MIPLIB ej: one equality row over three integer columns. In bench/results/miplib-078cb24.csv a
60 s run came back time_limit with the last node's LP relaxation written as its point, and
this script rejected it on integrality. The solver now writes no point for a limit that found
none: its status, its bound, `objective inf`, and no columns or rows section.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import verify_solution as vs

EJ_MPS = """\
NAME          ej
ROWS
 N  obj
 E  c1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x0        obj                             1
    x0        c1                          31013
    x1        c1                         -41014
    x2        c1                         -51015
    MARK0001  'MARKER'                 'INTEND'
RHS
BOUNDS
 LO bnd       x0                              1
 LI bnd       x1        0
 LI bnd       x2        0
ENDATA
"""

HEADER = [
    "# SANKHYA solution file",
    "model ej",
    "sense minimize",
    "status time_limit",
    "algorithm branch-and-bound",
    "OBJECTIVE_LINE",
    "dual_bound 1",
    "mip_relative_gap 0.0001",
    "mip_absolute_gap 9.9999999999999995e-07",
    "objective_offset 0",
    "certificate none",
    "primal_infeasibility 0",
    "dual_infeasibility 0",
    "integrality_violation 0",
]

# The file the old convention wrote (#223): the relaxation, fractional in x0.
RELAXATION_POINT = [
    "",
    "begin columns 3",
    "x0 20828.424531648019 0 unknown",
    "x1 0 1.3224776706542418 unknown",
    "x2 12662 1.6449553413084834 unknown",
    "end columns",
    "",
    "begin rows 1",
    "c1 0 3.2244542611163058e-05 unknown",
    "end rows",
]


def _verify(objective: str, body: list[str]):
    lines = [line if line != "OBJECTIVE_LINE" else f"objective {objective}" for line in HEADER]
    with tempfile.TemporaryDirectory() as tmp:
        mps = Path(tmp) / "ej.mps"
        sol = Path(tmp) / "ej.sol"
        mps.write_text(EJ_MPS)
        sol.write_text("\n".join(lines + body) + "\n")
        return vs.verify(vs.parse_mps(mps), vs.parse_sol(sol), 1e-7, 1e-7, 1e-6, 1e-6)


def _failed(report) -> list[str]:
    return [name for ok, name, _ in report.lines if not ok]


def run(check) -> None:
    report = _verify("inf", [])
    check(report.failures == 0 and [name for _, name, _ in report.lines] == ["verdict"],
          "a limit that found no point is checked as claiming nothing",
          str(report.lines))

    report = _verify("20828.424531648019", RELAXATION_POINT)
    check("integrality" in _failed(report),
          "the relaxation the old convention wrote is still rejected on integrality",
          str(_failed(report)))

    # A file that states a finite objective but lost its point is not "nothing found": it is
    # a claim with nothing behind it, and it fails the structure check.
    report = _verify("20828.424531648019", [])
    check("structure" in _failed(report),
          "a finite objective with no point is a missing point, not an empty search",
          str(_failed(report)))

    # `feasible` asserts a point, so an infinite objective and no columns is still a failure.
    lines_backup = list(HEADER)
    HEADER[3] = "status feasible"
    try:
        report = _verify("inf", [])
    finally:
        HEADER[:] = lines_backup
    check(report.failures > 0, "a status that asserts a point cannot claim nothing",
          str(_failed(report)))
