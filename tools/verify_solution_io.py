#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared low-level helpers for verify_solution.py's independent readers (issue #262).

Kept separate, tiny, and dependency-free: verify_solution_mps.py and verify_solution_sol.py
both need `open_text`, and only the MPS side additionally needs `normalize_infinity` - a
third module for the two lines they share is cheaper than either importing the other.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

INF = math.inf
MPS_INFINITY = 1e30


def normalize_infinity(value: float) -> float:
    """MPS uses 1e30 for infinity. Normalise so only true infinities reach the checks."""
    if value >= MPS_INFINITY:
        return INF
    if value <= -MPS_INFINITY:
        return -INF
    return value


def parse_number(text: str) -> float:
    """A numeric field of a model file, read as the solver's parse_double reads one
    (src/io/token.hpp, matched by behaviour and not by sharing code).

    Fortran-era MPS writers spell an exponent with D, `1.5D+02`, which float() refuses; a
    file the solver solves was one this script could not read at all, so `cannot read the
    model` stood where VERIFIED or REJECTED belonged. And float() accepts "nan", which the
    solver refuses in every field: a NaN bound, right-hand side or coefficient makes the
    model meaningless rather than extreme, and it fails every comparison, so a check against
    it passes without checking anything. Infinity is not refused here: a bound spelt 1e30 or
    larger is legitimately infinite, and normalize_infinity() makes it one.

    Raises ValueError, as float() does, so the fixed-format retry in parse_mps() still fires.
    """
    value = float(text.replace("D", "E").replace("d", "e"))
    if math.isnan(value):
        raise ValueError(f"{text!r} is not a number: NaN is refused in every field")
    return value


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")
