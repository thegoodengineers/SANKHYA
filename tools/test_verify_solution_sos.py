# SPDX-License-Identifier: Apache-2.0
"""verify_solution.py on semi-continuous columns and special ordered sets (#754).

Run from test_verify_solution.py's main(), which passes its `check`, so that CI's one command
(python3 tools/test_verify_solution.py) covers these too.

One model carries an SOS2 over four weights (a piecewise-linear curve) and a semi-continuous
column X in {0} or [2, 5]. A correct point verifies, with X at 0 and with X inside its run
range; a planted SOS2 violation (two members that are not adjacent) and a planted X strictly
between 0 and 2 are rejected, each on its own check. The reader's other spellings - the SETS
header, the colon form, the S1/S2 marker form in COLUMNS - are read to the same sets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import verify_solution as vs

MODEL = """\
NAME          SOSV
ROWS
 N  obj
 E  conv
 L  cap
COLUMNS
    L0        obj                0   conv               1
    L1        obj               -2   conv               1
    L2        obj                3   conv               1
    L3        obj               -4   conv               1
    X         obj                1   cap                1
RHS
    RHS       conv               1   cap               10
BOUNDS
 UP BND       L0                 1
 UP BND       L1                 1
 UP BND       L2                 1
 UP BND       L3                 1
 LO BND       X                  2
 SC BND       X                  5
SOS
 S2 SOS       curve
    curve     L0                 1
    curve     L1                 2
    curve     L2                 3
    curve     L3                 4
ENDATA
"""

NAMES = ["L0", "L1", "L2", "L3", "X"]
COSTS = [0.0, -2.0, 3.0, -4.0, 1.0]


def _sol(x: list[float]) -> str:
    objective = sum(c * v for c, v in zip(COSTS, x))
    conv = sum(x[:4])
    cap = x[4]
    lines = [
        "# SANKHYA solution file",
        "model SOSV",
        "sense minimize",
        "status optimal",
        "algorithm branch-and-bound",
        f"objective {objective!r}",
        f"dual_bound {objective!r}",
        "mip_relative_gap 0.0001",
        "mip_absolute_gap 1e-06",
        "objective_offset 0",
        "certificate none",
        "primal_infeasibility 0",
        "dual_infeasibility 0",
        "integrality_violation 0",
        "",
        f"begin columns {len(x)}",
    ]
    lines += [f"{n} {v!r} 0 unknown" for n, v in zip(NAMES, x)]
    lines += ["end columns", "", "begin rows 2", f"conv {conv!r} 0 unknown",
              f"cap {cap!r} 0 unknown", "end rows"]
    return "\n".join(lines) + "\n"


def _verify(model_text: str, x: list[float]):
    with tempfile.TemporaryDirectory() as tmp:
        mps = Path(tmp) / "sosv.mps"
        sol = Path(tmp) / "sosv.sol"
        mps.write_text(model_text)
        sol.write_text(_sol(x))
        return vs.verify(vs.parse_mps(mps), vs.parse_sol(sol), 1e-7, 1e-7, 1e-6, 1e-6)


def _parse(model_text: str):
    with tempfile.TemporaryDirectory() as tmp:
        mps = Path(tmp) / "m.mps"
        mps.write_text(model_text)
        return vs.parse_mps(mps)


def _failed(report) -> list[str]:
    return [name for ok, name, _ in report.lines if not ok]


def run(check) -> None:
    model = _parse(MODEL)
    check(model.semicontinuous == {4} and model.col_lower[4] == 2.0 and model.col_upper[4] == 5.0,
          "the SC bound is read: X in {0} or [2, 5]", str(model.semicontinuous))
    check(model.sos == [(2, "curve", [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])],
          "the SOS section is read in weight order", str(model.sos))

    report = _verify(MODEL, [0.0, 1.0, 0.0, 0.0, 0.0])
    check(report.failures == 0, "a point on the curve with X off verifies", str(_failed(report)))
    check(any(name == "semi-continuous" for _, name, _ in report.lines)
          and any(name == "special ordered sets" for _, name, _ in report.lines),
          "both conditions are checked, not skipped", str(report.lines))

    report = _verify(MODEL, [0.0, 0.5, 0.5, 0.0, 3.0])
    check(report.failures == 0, "two adjacent members and X inside its run range verify",
          str(_failed(report)))

    # Planted: L0 and L2 are both nonzero, and they are not adjacent.
    report = _verify(MODEL, [0.5, 0.0, 0.5, 0.0, 0.0])
    check(_failed(report) == ["special ordered sets"],
          "a planted SOS2 violation is rejected, on the set check alone", str(_failed(report)))

    # Planted: X = 1 is strictly between 0 and its lower bound 2.
    report = _verify(MODEL, [0.0, 1.0, 0.0, 0.0, 1.0])
    check(_failed(report) == ["semi-continuous"],
          "a planted semi-continuous value between 0 and l is rejected", str(_failed(report)))

    # X = 6 is above the run range: an ordinary bound violation, not a gap.
    report = _verify(MODEL, [0.0, 1.0, 0.0, 0.0, 6.0])
    check("column bounds" in _failed(report), "a value above the run range breaks its bound",
          str(_failed(report)))

    # The SETS spelling with colon members, in a shuffled order.
    sets_text = MODEL.replace(
        "SOS\n S2 SOS       curve\n    curve     L0                 1\n"
        "    curve     L1                 2\n    curve     L2                 3\n"
        "    curve     L3                 4\n",
        "SETS\n S2 SOS curve 7\n    curve L2:3\n    L0:1\n    curve L3 4\n    L1 2\n")
    check(_parse(sets_text).sos == [(2, "curve", [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])],
          "the SETS header and the colon form read to the same set", str(_parse(sets_text).sos))

    marker_text = """\
NAME          MARK
ROWS
 N  obj
 L  cap
COLUMNS
 S1 PICK      'MARKER'                 'SOSORG'
    A         obj                1   cap                1
    B         obj                2   cap                1
    PICKEND   'MARKER'                 'SOSEND'
    C         obj                3   cap                1
RHS
    RHS       cap                1
ENDATA
"""
    check(_parse(marker_text).sos == [(1, "PICK", [0, 1], [1.0, 2.0])],
          "the S1 marker form reads its members in order", str(_parse(marker_text).sos))

    tied = MODEL.replace("    curve     L3                 4\n", "    curve     L3                 3\n")
    try:
        _parse(tied)
        refused = False
    except ValueError:
        refused = True
    check(refused, "two members with one weight are refused: their order is undefined")
