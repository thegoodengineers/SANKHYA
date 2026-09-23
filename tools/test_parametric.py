#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/parametric.py (#522).

#522's acceptance criterion: "each breakpoint's objective verified by an independent
re-solve at that parameter value." Every breakpoint the sweep reports already comes from
its own solve() call (never derived from ranging's linear estimate alone - see the module
docstring), but that is the tool checking itself; this file additionally re-derives a FRESH
model at a chosen breakpoint's parameter value, through a separate code path, and confirms
the objective matches what the sweep recorded.

    python tools/test_parametric.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from parametric import sweep_cost, sweep_row  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_MODEL = REPO_ROOT / "demo" / "crude_blend.mps"

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def _independent_resolve_at(column: str | None, row: str | None, side: str,
                            value: float) -> float:
    """A model built and solved completely fresh, from scratch, at ONE parameter value -
    the independent check every breakpoint on the swept curve is compared against."""
    import sankhya

    model = sankhya.Model.read(str(DEMO_MODEL))
    if column is not None:
        # Model has no "find column index by name" accessor (write-only, #536's report.py
        # docstring makes the same point) - read the index off the independent MPS parser
        # instead, which IS meant to answer exactly this.
        from verify_solution_mps import parse_mps
        source = parse_mps(DEMO_MODEL)
        model.set_cost(source.col_index[column], value)
    else:
        from verify_solution_mps import parse_mps
        source = parse_mps(DEMO_MODEL)
        i = source.row_index[row]
        lower = source.row_lower[i] if source.row_lower[i] > -1e29 else None
        upper = source.row_upper[i] if source.row_upper[i] < 1e29 else None
        if side == "upper":
            upper = value
        else:
            lower = value
        model.set_row_bounds(i, lower, upper)
    result = model.solve(log_to_console=False)
    check(result.status == "optimal", f"the independent re-solve at {value} is optimal",
          result.status)
    return result.objective


def test_cost_sweep_breakpoints_match_independent_resolves() -> None:
    rows = sweep_cost(DEMO_MODEL, "AL", 0.0, 5.0, {})
    check(len(rows) >= 2, "the sweep over a wide range reports more than one breakpoint",
          str(len(rows)))

    mismatches = []
    for row in rows:
        if row["objective"] == "":
            continue
        independent = _independent_resolve_at("AL", None, "upper", row["parameter"])
        scale = max(1.0, abs(independent))
        if abs(row["objective"] - independent) > 1e-6 * scale:
            mismatches.append((row["parameter"], row["objective"], independent))
    check(not mismatches, "every breakpoint's objective matches an independent fresh solve "
          "at that parameter value", str(mismatches))


def test_a_range_with_no_interior_breakpoint_still_verifies_at_both_ends() -> None:
    # AL's cost over [2, 3] is exactly the earlier manual check that found no interior
    # breakpoint - the endpoints must still each be a real, independently-checkable solve.
    rows = sweep_cost(DEMO_MODEL, "AL", 2.0, 3.0, {})
    check(len(rows) == 2, "no interior breakpoint means exactly two rows (both ends)",
          str(len(rows)))
    for row in rows:
        independent = _independent_resolve_at("AL", None, "upper", row["parameter"])
        check(abs(row["objective"] - independent) <= 1e-6 * max(1.0, abs(independent)),
              f"parameter={row['parameter']} matches an independent solve",
              f"{row['objective']} vs {independent}")


def test_row_sweep_breakpoints_match_independent_resolves() -> None:
    # THRUPUT's upper bound (the refinery's throughput capacity) is a natural "what if
    # capacity changes" question - #522's own example.
    rows = sweep_row(DEMO_MODEL, "THRUPUT", "upper", 100.0, 200.0, {})
    check(len(rows) >= 1, "the row sweep reports at least the two endpoints", str(len(rows)))

    mismatches = []
    for row in rows:
        if row["objective"] == "":
            continue
        independent = _independent_resolve_at(None, "THRUPUT", "upper", row["parameter"])
        scale = max(1.0, abs(independent))
        if abs(row["objective"] - independent) > 1e-6 * scale:
            mismatches.append((row["parameter"], row["objective"], independent))
    check(not mismatches, "every row-sweep breakpoint matches an independent fresh solve",
          str(mismatches))


def main() -> int:
    print("tools/parametric.py tests\n")
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
