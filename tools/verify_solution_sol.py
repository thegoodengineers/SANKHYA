#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The .sol reader verify_solution.py checks a solve against (issue #262 split this out
of one 1,529-line file). Reads only the SANKHYA .sol format written by src/io/writer.cpp's
documented layout - it does not link that C++, only reads what it wrote.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from verify_solution_io import open_text

# An exact multiplier (#763) is a fraction whose parts run to thousands of digits; Python
# 3.11 refuses to convert more than 4300 by default, as a defence against quadratic parsing
# of untrusted input. These are our own files, read once.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

class Solution:
    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.col_value: dict[str, float] = {}
        self.col_dual: dict[str, float] = {}
        self.col_status: dict[str, str] = {}
        self.row_activity: dict[str, float] = {}
        self.row_dual: dict[str, float] = {}
        self.row_status: dict[str, str] = {}
        # A verdict of infeasible or unbounded carries its PROOF, not a point (#191).
        # farkas: row multipliers whose aggregate no point in the column box can satisfy.
        # ray: a direction along which the model stays feasible and the objective improves
        # without limit, checked together with the feasible point in the columns section.
        self.farkas: dict[str, float] = {}
        self.ray: dict[str, float] = {}
        # Sensitivity ranging (populated when --ranging was passed to the solver).
        self.col_ranging_lower: dict[str, float] = {}
        self.col_ranging_upper: dict[str, float] = {}
        self.row_ranging_lower: dict[str, float] = {}
        self.row_ranging_upper: dict[str, float] = {}
        # Certified sensitivity (#757): name -> [verdict, exact values as text].
        self.exact_sensitivity_columns: dict[str, list[str]] = {}
        self.exact_sensitivity_rows: dict[str, list[str]] = {}
        self.exact_repair_farkas: dict[str, str] = {}
        self.iis: list[tuple[str, str]] = []
        # One witness per IIS element: (kind, name, {column name: value}).
        self.iis_witnesses: list[tuple[str, str, dict[str, float]]] = []
        # The solution pool (#225): (rank, objective, {integer column name: value}).
        self.pool: list[tuple[int, float, dict[str, float]]] = []
        # The row multipliers an optimal LP's safe_lower_bound was proved from (#763).
        self.safe_multipliers: dict[str, float] = {}
        # ...and the column bounds it relied on: (column, "lower"|"upper", value, row).
        self.safe_column_bounds: list[tuple[str, str, float, str]] = []

    @property
    def status(self) -> str:
        return self.header.get("status", "unknown")

    def header_float(self, key: str) -> float | None:
        raw = self.header.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None


QUOTE_CHAR = '"'
BACKSLASH_CHAR = '\\'


def split_record(line: str) -> list[str]:
    """Split a .sol record, honouring a quoted leading name.

    src/io/writer.cpp quotes any name containing whitespace, because fixed-format MPS permits
    them - Netlib's forplan has a column called `DEDO3 11` - and a bare one makes the record
    undecidable: nothing in `DEDO3 11 0 0.0246 at_lower` says whether the name is one field
    or two.

    This parser is written from the format, not shared with the writer. That is the whole
    point of this script: if the two disagree about what a file means, the disagreement has
    to be able to surface, and it cannot if they run the same code.
    """
    if not line.startswith(QUOTE_CHAR):
        return line.split()
    quoted = leading_name(line)
    return line.split() if quoted is None else [quoted[0]] + quoted[1].split()


def leading_name(line: str) -> tuple[str, str] | None:
    """The record's first name, unquoted, and the unparsed rest of the line; None when the
    line opens a quote it never closes."""
    if not line.startswith(QUOTE_CHAR):
        parts = line.split(None, 1)
        return (parts[0], parts[1] if len(parts) > 1 else "") if parts else None
    name = []
    i = 1
    while i < len(line):
        c = line[i]
        if c == BACKSLASH_CHAR and i + 1 < len(line):
            name.append(line[i + 1])
            i += 2
            continue
        if c == QUOTE_CHAR:
            return "".join(name), line[i + 1:]
        name.append(c)
        i += 1
    # No closing quote. Fall back rather than silently truncating the record.
    return None


def parse_sol(path: Path) -> Solution:
    solution = Solution()
    block = ""
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_record(line)
            if fields[0] == "begin":
                block = fields[1]
                if block == "iis_witness" and len(fields) >= 4:
                    solution.iis_witnesses.append((fields[2], fields[3], {}))
                continue
            if fields[0] == "end":
                block = ""
                continue
            if block == "columns" and len(fields) >= 3:
                solution.col_value[fields[0]] = float(fields[1])
                solution.col_dual[fields[0]] = float(fields[2])
                solution.col_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif block == "farkas" and len(fields) >= 2:
                solution.farkas[fields[0]] = float(fields[1])
            elif block == "safe_multipliers" and len(fields) >= 2:
                # An exact fraction "p/q" when the solver used the basis's exact duals.
                solution.safe_multipliers[fields[0]] = (
                    Fraction(fields[1]) if "/" in fields[1] else float(fields[1]))
            elif block == "safe_column_bounds" and len(fields) >= 4:
                # `column side value row`: two names, either of which may be quoted.
                column = leading_name(line)
                parts = column[1].split(None, 2) if column else []
                row = leading_name(parts[2].strip()) if len(parts) == 3 else None
                if column is None or row is None:
                    raise ValueError(f"unreadable safe_column_bounds record: {line}")
                solution.safe_column_bounds.append(
                    (column[0], parts[0], float(parts[1]), row[0]))
            elif block == "ray" and len(fields) >= 2:
                solution.ray[fields[0]] = float(fields[1])
            elif block == "pool" and len(fields) == 3 and fields[0] == "solution":
                solution.pool.append((int(fields[1]), float(fields[2]), {}))
            elif block == "pool" and len(fields) == 2 and solution.pool:
                solution.pool[-1][2][fields[0]] = float(fields[1])
            elif block == "iis" and len(fields) >= 2:
                solution.iis.append((fields[0], fields[1]))
            elif block == "iis_witness" and len(fields) >= 2 and solution.iis_witnesses:
                solution.iis_witnesses[-1][2][fields[0]] = float(fields[1])
            elif block == "rows" and len(fields) >= 3:
                solution.row_activity[fields[0]] = float(fields[1])
                solution.row_dual[fields[0]] = float(fields[2])
                solution.row_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif block == "ranging_columns" and len(fields) >= 3:
                solution.col_ranging_lower[fields[0]] = float(fields[1])
                solution.col_ranging_upper[fields[0]] = float(fields[2])
            elif block == "ranging_rows" and len(fields) >= 3:
                solution.row_ranging_lower[fields[0]] = float(fields[1])
                solution.row_ranging_upper[fields[0]] = float(fields[2])
            elif block == "exact_sensitivity_columns" and len(fields) >= 5:
                solution.exact_sensitivity_columns[fields[0]] = fields[1:5]
            elif block == "exact_sensitivity_rows" and len(fields) >= 7:
                solution.exact_sensitivity_rows[fields[0]] = fields[1:7]
            elif block == "exact_repair_farkas" and len(fields) >= 2:
                solution.exact_repair_farkas[fields[0]] = fields[1]
            elif not block and len(fields) >= 2:
                solution.header[fields[0]] = " ".join(fields[1:])
    return solution


