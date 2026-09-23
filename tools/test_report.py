#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/report.py (#527): every number in the report cross-checked against the
independently-parsed .sol file, per the issue's own acceptance criterion.

    python tools/test_report.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import binding_rows, build_report, columns_at_bounds, nonbasic_columns  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402
from verify_solution_sol import parse_sol  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


_TINY_MPS = """NAME          TINY
ROWS
 N  COST
 L  CAP
 G  MIN
COLUMNS
    X         COST         -3.0   CAP          1.0
    X         MIN          1.0
    Y         COST         -2.0   CAP          1.0
RHS
    RHS       CAP          4.0    MIN          1.0
BOUNDS
 UP BND       X            10.0
 UP BND       Y            10.0
ENDATA
"""

# max 3x + 2y  s.t.  x + y <= 4,  x >= 1,  0 <= x,y <= 10.
# CAP binds at x=4,y=0 (obj 12) or x=1,y=3 (obj 9) - the LP optimum is x=4,y=0: CAP binds
# (activity 4 == upper 4), MIN does not (activity 4 > lower 1), y is nonbasic at its lower
# bound 0 with a positive reduced cost (2, in minimise-space -2 negated by maximize).
_TINY_SOL = """# SANKHYA solution file
model TINY
sense maximize
status optimal
algorithm simplex-dual
objective 12
dual_bound 12
objective_offset 0
certificate none
rows 2
columns 2
iterations 1
nodes 0
solve_seconds 0.0001
primal_infeasibility 0
dual_infeasibility 0
integrality_violation 0

begin columns 2
X 4 0 basic
Y 0 2 at_lower
end columns

begin rows 2
CAP 4 3 at_upper
MIN 4 0 basic
end rows
"""


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_binding_rows_matches_the_sol_file_exactly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        model = parse_mps(_write(directory, "tiny.mps", _TINY_MPS))
        solution = parse_sol(_write(directory, "tiny.sol", _TINY_SOL))

        rows = binding_rows(model, solution, tol=1e-7)
        check(len(rows) == 1, "exactly one binding row (CAP, not MIN)", str(len(rows)))
        if rows:
            row = rows[0]
            check(row["name"] == "CAP", "the binding row is CAP", row["name"])
            check(row["shadow_price"] == solution.row_dual["CAP"],
                  "reported shadow price equals the .sol file's row_dual['CAP']",
                  f"{row['shadow_price']} vs {solution.row_dual['CAP']}")
            check(row["activity"] == solution.row_activity["CAP"],
                  "reported activity equals the .sol file's row_activity['CAP']",
                  f"{row['activity']} vs {solution.row_activity['CAP']}")
            check(row["at"] == "upper", "CAP is binding at its upper bound", row["at"])


def test_nonbasic_columns_matches_the_sol_file_exactly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        model = parse_mps(_write(directory, "tiny.mps", _TINY_MPS))
        solution = parse_sol(_write(directory, "tiny.sol", _TINY_SOL))

        columns = nonbasic_columns(model, solution, tol=1e-7)
        check(len(columns) == 1, "exactly one nonbasic column with a nonzero reduced cost",
              str(len(columns)))
        if columns:
            col = columns[0]
            check(col["name"] == "Y", "the nonbasic column is Y", col["name"])
            check(col["reduced_cost"] == solution.col_dual["Y"],
                  "reported reduced cost equals the .sol file's col_dual['Y']",
                  f"{col['reduced_cost']} vs {solution.col_dual['Y']}")
            check(col["value"] == solution.col_value["Y"],
                  "reported value equals the .sol file's col_value['Y']",
                  f"{col['value']} vs {solution.col_value['Y']}")


def test_variables_at_bounds_is_a_column_bound_fact_not_a_row_bound_fact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        model = parse_mps(_write(directory, "tiny.mps", _TINY_MPS))
        solution = parse_sol(_write(directory, "tiny.sol", _TINY_SOL))

        columns = {c["name"]: c for c in columns_at_bounds(model, solution, tol=1e-7)}
        # X = 4 is at neither of its OWN bounds (0, 10) - only CAP (a ROW bound) makes it
        # binding, and the two lists are deliberately independent: a column-bounds fact and
        # a row-bounds fact are not the same claim.
        check("X" not in columns, "X is not reported as being at a COLUMN bound (0 or 10)")
        check("Y" in columns and columns["Y"]["at"] == "lower",
              "Y is at its own lower bound (0)")
        check(set(columns) == {"Y"}, "only Y is at a column bound", str(sorted(columns)))


def test_full_report_objective_and_status_match_the_header() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        model = parse_mps(_write(directory, "tiny.mps", _TINY_MPS))
        solution = parse_sol(_write(directory, "tiny.sol", _TINY_SOL))

        report = build_report(model, solution, tol=1e-7)
        check(report["status"] == solution.status, "report status matches the .sol header",
              f"{report['status']} vs {solution.status}")
        check(report["objective"] == solution.header_float("objective"),
              "report objective matches the .sol header",
              f"{report['objective']} vs {solution.header_float('objective')}")
        check(report["is_milp"] is False, "TINY has no integer columns")
        check(report["duals_note"] is None, "an LP report carries no MILP duals caveat")


def main() -> int:
    print("tools/report.py tests\n")
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(name)
            function()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
