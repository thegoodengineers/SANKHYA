#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/runners/presolve_shrink.py: arm parsing, the profile phase split (only
the direct children of `solve` count, so a nested engine is not counted twice) and the
gap. Pure Python, no solve.

    python bench/runners/test_presolve_shrink.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import presolve_shrink as ps  # noqa: E402

assert ps.parse_arm("base") == ("base", [])
assert ps.parse_arm("x:a=1,b=true") == ("x", ["a=1", "b=true"])

profile = {"regions": [
    {"path": "solve", "inclusive_seconds": 3.0},
    {"path": "solve/presolve", "inclusive_seconds": 0.25},
    {"path": "solve/engine", "inclusive_seconds": 2.0},
    {"path": "solve/postsolve/engine", "inclusive_seconds": 9.0},  # nested: not a phase
    {"path": "solve/verification", "inclusive_seconds": 0.5},
    {"path": "solve/verification", "inclusive_seconds": 0.25},     # a second call adds
]}
phases = ps.phase_seconds(profile)
assert phases == {"presolve": 0.25, "engine": 2.0, "postsolve": 0.0, "verification": 0.75}, phases
assert ps.phase_seconds({}) == {p: 0.0 for p in ps.PHASES}

assert ps.gap(None, 1.0) is None
assert ps.gap(101.0, 100.0) == 0.01
assert ps.gap(0.5, 0.0) == 0.5  # max(1, |ref|) in the denominator

print("presolve_shrink: all checks passed")
