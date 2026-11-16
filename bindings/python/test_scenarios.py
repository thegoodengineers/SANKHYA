# SPDX-License-Identifier: Apache-2.0
"""Model.solve_scenarios (#752) through the Python bindings.

    PYTHONPATH=bindings/python python bindings/python/test_scenarios.py

Each scenario's answer must equal a cold solve of the same model edited in place with
set_cost / set_row_bounds, and an unreachable demand must never come back optimal.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import sankhya

DEMO = Path(__file__).resolve().parents[2] / "demo" / "crude_blend.mps"
FAILURES = 0


def check(condition: bool, what: str, detail: str = "") -> None:
    global FAILURES
    print(f"  {'ok  ' if condition else 'FAIL'} {what}" + (f": {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


def cold(edit) -> sankhya.Result:
    model = sankhya.Model.read(str(DEMO))
    edit(model)
    return model.solve(log_to_console=False)


def main() -> int:
    # demo/crude_blend.mps: columns AL BN MU; rows THRUPUT (90..120), DIESEL (= 40), SULPHUR.
    model = sankhya.Model.read(str(DEMO))
    scenarios = [
        [("cost", 0, 1.9)],
        [("rhs", 1, 42.0)],
        [("cost", 0, 2.9), ("row_upper", 0, 110.0)],
        [("rhs", 1, 60.0)],  # beyond what 120 kbbl/day of these crudes can yield
    ]
    edits = [
        lambda m: m.set_cost(0, 1.9),
        lambda m: m.set_row_bounds(1, 42.0, 42.0),
        lambda m: (m.set_cost(0, 2.9), m.set_row_bounds(0, 90.0, 110.0)),
    ]
    for start in ("base", "previous"):
        print(f"scenario_start={start}")
        runs = model.solve_scenarios(scenarios, log_to_console=False, scenario_start=start)
        check(len(runs) == 4, "one answer per scenario", str(len(runs)))
        for k, edit in enumerate(edits):
            reference = cold(edit)
            run = runs[k]
            check(run.verified and run.result.status == "optimal",
                  f"scenario {k} verified optimal", f"{run.result.status} {run.verified}")
            rel = abs(run.result.objective - reference.objective) / max(1.0, abs(reference.objective))
            check(rel <= 1e-8, f"scenario {k} equals the cold solve",
                  f"{run.result.objective} against {reference.objective}")
            check(math.isnan(run.safe_bound) or run.safe_bound >= reference.objective - 1e-9,
                  f"scenario {k} safe bound above the maximum", str(run.safe_bound))
            check(len(run.result.x) == model.num_cols, f"scenario {k} carries a point")
        check(runs[3].result.status != "optimal", "the unreachable demand is not optimal",
              runs[3].result.status)
    # The base model is untouched.
    check(abs(model.solve(log_to_console=False).objective - cold(lambda m: None).objective)
          <= 1e-12, "the model handle is unchanged")
    try:
        model.solve_scenarios([[("price", 0, 1.0)]])
        check(False, "an unknown target raises")
    except ValueError:
        check(True, "an unknown target raises")
    try:
        model.solve_scenarios([[("cost", 99, 1.0)]])
        check(False, "a column outside the model raises")
    except sankhya.SankhyaError:
        check(True, "a column outside the model raises")
    print("ALL TESTS PASSED" if FAILURES == 0 else f"{FAILURES} check(s) FAILED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
