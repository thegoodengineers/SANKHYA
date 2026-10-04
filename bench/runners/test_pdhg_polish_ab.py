#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/runners/pdhg_polish_ab.py: reading the engine's residuals and the
feasibility polish clause out of the result message, and the verifier's verdict and row
violation out of its report. Pure Python, no solve.

    python bench/runners/test_pdhg_polish_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdhg_polish_ab as pa  # noqa: E402

# Off: the stop reason alone.
off = pa.parse_message("stopped at relative primal 1.954e-03, dual 4.653e-04, gap 7.562e-04 "
                       "after 300 iterations and 4 restarts (target 1.0e-04)")
assert off == {"engine_relative_primal": "1.954e-03", "engine_relative_dual": "4.653e-04",
               "engine_relative_gap": "7.562e-04", "feasibility_polishes": "",
               "feasibility_polish_iterations": "", "reported_point_polished": ""}, off

# On: the polish clause restates the reported point's residuals, and the last triple wins.
on = pa.parse_message("stopped at relative primal 9.000e-03, dual 0.000e+00, gap 8.715e-03 "
                      "after 300 iterations and 4 restarts (target 1.0e-04). feasibility "
                      "polish (#483): 2 polishes, 185 of 485 iterations; reported point "
                      "polished, relative primal 1.044e-10, dual 0.000e+00, gap inf")
assert on["engine_relative_primal"] == "1.044e-10", on
assert on["engine_relative_gap"] == "inf", on
assert on["feasibility_polishes"] == 2 and on["feasibility_polish_iterations"] == 185, on
assert on["reported_point_polished"] == 1, on
kept = pa.parse_message("feasibility polish (#483): 0 polishes, 0 of 300 iterations; "
                        "reported point not polished, relative primal 3.0e-02, dual 1, gap nan")
assert kept["reported_point_polished"] == 0 and kept["engine_relative_dual"] == "1", kept
assert pa.parse_message("")["engine_relative_primal"] == ""

report = """Independent checks (this script shares no code with the solver):
  [PASS] structure                         27 rows, 32 columns present
  [PASS] row activity matches the stated   worst violation 4.997e-08 (1.035e-09 relative to the row's terms) on R13
  [PASS] objective                         recomputed -4.6e+02
VERIFIED: 7 checks passed
"""
v = pa.parse_verifier(0, report)
assert v["verifier_verdict"] == "VERIFIED" and v["independently_verified"] == 1, v
assert v["verifier_row_violation"] == "4.997e-08", v
assert v["verifier_row_violation_relative"] == "1.035e-09", v
assert v["verifier_failures"] == ""

bad = pa.parse_verifier(1, "  [FAIL] row activity   worst violation 2.0e-01 (5.0e-02 relative "
                           "to the row's terms) on X\nREJECTED: 1 of 8 checks failed\n")
assert bad["verifier_verdict"] == "REJECTED" and bad["independently_verified"] == 0, bad
assert bad["verifier_row_violation_relative"] == "5.0e-02", bad
assert bad["verifier_failures"].startswith("[FAIL] row activity"), bad
assert pa.parse_verifier(3, "")["verifier_verdict"] == "NOT A SOLUTION"

assert pa.relative_error(None, 1.0) == ""
assert pa.relative_error(101.0, 100.0) == repr(0.01)

print("pdhg_polish_ab: all checks passed")
