#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is a certificate's model the model in an MPS file? (#518)

A proof of optimality is only as good as the statement it proves. verify_certificate.py
checks the derivation against the model written INSIDE the certificate; this module closes
the other half, by reading the MPS file with the project's independent reader
(verify_solution_mps.py, which shares no code with the solver) and comparing every number.

What is compared is the model a double-precision solver solves: each decimal in the MPS
file rounded to the nearest double, as both readers do, and that double taken exactly. So a
coefficient written 0.1 is compared as the double nearest one tenth, which is what the
certificate proves things about. The objective constant is not part of VIPR and is not
compared; the certificate's bounds exclude it.

The writer's layout, which this reproduces: every row in file order, as one E constraint when
its sides are equal and otherwise a G for a finite lower side and an L for a finite upper
side; then every column's finite lower bound (G) and finite upper bound (L).
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from verify_certificate import Certificate, Rejected
from verify_solution_mps import parse_mps


def _expected_constraints(model) -> list[tuple[str, Fraction, dict[int, Fraction]]]:
    rows: list[dict[int, Fraction]] = [dict() for _ in range(model.num_rows)]
    for j, column in enumerate(model.entries):
        for i, value in column:
            if value != 0.0:
                rows[i][j] = rows[i].get(j, Fraction(0)) + Fraction(value)
    expected = []
    for i in range(model.num_rows):
        lower, upper = model.row_lower[i], model.row_upper[i]
        if math.isfinite(lower) and lower == upper:
            expected.append(("E", Fraction(lower), rows[i]))
            continue
        if math.isfinite(lower):
            expected.append(("G", Fraction(lower), rows[i]))
        if math.isfinite(upper):
            expected.append(("L", Fraction(upper), rows[i]))
    for j in range(model.num_cols):
        if math.isfinite(model.col_lower[j]):
            expected.append(("G", Fraction(model.col_lower[j]), {j: Fraction(1)}))
        if math.isfinite(model.col_upper[j]):
            expected.append(("L", Fraction(model.col_upper[j]), {j: Fraction(1)}))
    return expected


def compare_with_mps(cert: Certificate, path: Path) -> None:
    """Raise Rejected at the first difference between the certificate's model and `path`."""
    model = parse_mps(path)
    if model.hessian:
        raise Rejected("the MPS model has a quadratic objective; the certificate is linear")
    if model.num_cols != len(cert.var_names):
        raise Rejected(f"the MPS file has {model.num_cols} columns, the certificate "
                       f"{len(cert.var_names)}")
    if model.maximize != cert.maximize:
        raise Rejected("the objective sense differs from the MPS file")
    for j in range(model.num_cols):
        if model.col_integer[j] != cert.integer[j]:
            raise Rejected(f"column {model.col_names[j]}: integrality differs")
    objective = {j: Fraction(c) for j, c in enumerate(model.col_cost) if c != 0.0}
    if objective != cert.objective:
        raise Rejected("the objective coefficients differ from the MPS file")
    expected = _expected_constraints(model)
    if len(expected) != len(cert.constraints):
        raise Rejected(f"the MPS file gives {len(expected)} constraints, the certificate "
                       f"{len(cert.constraints)}")
    for k, (sense, rhs, coefs) in enumerate(expected):
        c = cert.constraints[k]
        if c.sense != sense or c.rhs != rhs or c.coefs != {j: v for j, v in coefs.items()
                                                             if v != 0}:
            raise Rejected(f"constraint {k} ({c.name}) differs from the MPS file")
