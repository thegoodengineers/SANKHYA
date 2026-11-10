#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The refinery planning model with convex cost curves: a separable, positive definite QP
whose optimum is exact by construction.

THE MODEL is bench/runners/generate_refinery_lp.py's T-period refinery (#211): crude
purchases, CDU throughput, crude and product tanks, yields, unit capacities, quality budgets
and delivery commitments, a staircase coupled period to period by the inventory rows. What
changes is the cost of buying and of processing: each purchase column BUY[k,t] and each
throughput column RUN[k,t] carries a convex quadratic term q_j x_j^2 / 2 beside its linear
price, the shape a rising marginal cost (a crude premium that grows with the volume bought,
a unit whose cost per barrel rises toward its throughput limit) gives a cost curve. The
Hessian is diagonal with q_j > 0 on those columns and zero elsewhere, so the objective is
convex, strictly so in the purchase and processing variables.

THE OPTIMUM IS EXACT BY CONSTRUCTION, as for the LP. The LP generator chooses the plan x*,
the tight rows and their multipliers y*, and reduced costs d signed by where x* sits against
its bounds, and derives c_LP = d + A' y*. The KKT conditions of

    min  c'x + x'Qx / 2   s.t.  A x {<=,=,>=} b,  0 <= x <= u

read c + Q x* - A' y* - d = 0 with the same sign conditions, so the QP's linear prices are
c = c_LP - Q x*, and (x*, y*, d) is a KKT point of a convex QP, hence optimal; the optimum is
c'x* + x*'Q x*/2. Every q_j is a hundredth and every x*_j an integer, so the arithmetic stays
in short decimals and the value is exact. `verify_qp` re-checks every condition with the
quadratic term before a line is written.

THE SIZES the comparison runs (bench/refinery/generate.py --qp), each from the LP
generator's defaults (30 crudes, 16 products, 8 units, 10 specifications, 8 commitments,
138 columns and 89 rows a period) and seed 1:

    13k   95 periods      13,110 columns     8,455 rows
    110k  800 periods    110,400 columns    71,200 rows
    1m    7,300 periods  1,007,400 columns  649,700 rows
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
import generate_refinery_lp as lpgen  # noqa: E402

QP_SEED = 1
QP_SIZES = {"13k": 95, "110k": 800, "1m": 7300}
CURVATURE_RANGE = (1, 9)     # q_j drawn from [1, 9] hundredths per purchase / run column
CURVATURE_DENOMINATOR = 100
CURVED_PREFIXES = ("BUY_", "RUN_")


def build(periods: int, seed: int = QP_SEED):
    """The LP instance with diagonal curvature on BUY and RUN and the prices re-derived.
    Returns (inst, q) where q[j] is the Hessian diagonal (0 where the column is linear)."""
    inst = lpgen.build(periods, 30, 16, 8, 10, 8, seed)
    rng = random.Random(seed * 1_000_003 + periods)
    q = [Fraction(0)] * len(inst.cols)
    for j, name in enumerate(inst.cols):
        if name.startswith(CURVED_PREFIXES):
            q[j] = Fraction(rng.randint(*CURVATURE_RANGE), CURVATURE_DENOMINATOR)
            inst.cost[j] -= q[j] * inst.x[j]
    return inst, q


def optimum(inst, q) -> Fraction:
    return sum((c * x + qj * x * x / 2 for c, x, qj in zip(inst.cost, inst.x, q)),
               Fraction(0))


def verify_qp(inst, q) -> None:
    """Every KKT condition of the QP at (x*, y*), exactly; raises on the first that fails.
    The LP's check with the gradient c + Q x* in place of c."""
    n = len(inst.cols)
    for j in range(n):
        assert q[j] >= 0, f"{inst.cols[j]}: negative curvature"
        assert inst.lower[j] <= inst.x[j], f"{inst.cols[j]} below its lower bound"
        assert inst.upper[j] is None or inst.x[j] <= inst.upper[j], (
            f"{inst.cols[j]} above its upper bound")
    aty = [Fraction(0)] * n
    for i, entries in enumerate(inst.row_entries):
        activity = sum(c * inst.x[col] for col, c in entries.items())
        name, sense = inst.rows[i]
        if sense == "E":
            assert activity == inst.rhs[i], f"{name}: equality violated"
        elif sense == "L":
            assert activity <= inst.rhs[i], f"{name}: <= violated"
            assert inst.y[i] <= 0, f"{name}: L row needs a non-positive multiplier"
        else:
            assert activity >= inst.rhs[i], f"{name}: >= violated"
            assert inst.y[i] >= 0, f"{name}: G row needs a non-negative multiplier"
        if sense != "E" and activity != inst.rhs[i]:
            assert inst.y[i] == 0, f"{name}: slack row carries a multiplier"
        for col, c in entries.items():
            aty[col] += c * inst.y[i]
    for j in range(n):
        d = inst.cost[j] + q[j] * inst.x[j] - aty[j]
        at_lower = inst.x[j] == inst.lower[j]
        at_upper = inst.upper[j] is not None and inst.x[j] == inst.upper[j]
        if not at_lower and not at_upper:
            assert d == 0, f"{inst.cols[j]}: interior column with reduced cost {d}"
        elif at_lower and not at_upper:
            assert d >= 0, f"{inst.cols[j]}: at lower bound with reduced cost {d}"
        elif at_upper and not at_lower:
            assert d <= 0, f"{inst.cols[j]}: at upper bound with reduced cost {d}"


def write_qps(inst, q, out: Path, label: str, periods: int, seed: int) -> Fraction:
    """Free-format MPS with a QUADOBJ section (the objective's Hessian, factor 1/2 implied,
    one entry per diagonal element), the convention tools/verify_solution_mps.py reads."""
    value = optimum(inst, q)
    col_rows: list[list[tuple[int, Fraction]]] = [[] for _ in inst.cols]
    for i, entries in enumerate(inst.row_entries):
        for col, c in entries.items():
            col_rows[col].append((i, c))
    dec = lpgen.decimal
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"NAME          REFINERY_QP_{label.upper()}_S{seed}\n")
        f.write("* generator: bench/refinery/refinery_qp.py via bench/refinery/generate.py --qp\n")
        f.write(f"* structure: refinery, {periods} periods, convex quadratic cost on "
                f"purchase and processing\n")
        f.write(f"* analytic optimum: {float(value)!r} (exact: {value})\n")
        f.write("ROWS\n N  COST\n")
        for name, sense in inst.rows:
            f.write(f" {sense}  {name}\n")
        f.write("COLUMNS\n")
        for j, name in enumerate(inst.cols):
            if inst.cost[j] != 0:
                f.write(f"    {name}  COST  {dec(inst.cost[j])}\n")
            for i, c in col_rows[j]:
                f.write(f"    {name}  {inst.rows[i][0]}  {dec(c)}\n")
            if inst.cost[j] == 0 and not col_rows[j]:
                f.write(f"    {name}  COST  0\n")
        f.write("RHS\n")
        for i, (name, _) in enumerate(inst.rows):
            if inst.rhs[i] != 0:
                f.write(f"    RHS  {name}  {dec(inst.rhs[i])}\n")
        f.write("BOUNDS\n")
        for j, name in enumerate(inst.cols):
            if inst.upper[j] is not None:
                f.write(f" UP BND  {name}  {dec(inst.upper[j])}\n")
        f.write("QUADOBJ\n")
        for j, name in enumerate(inst.cols):
            if q[j] != 0:
                f.write(f"    {name}  {name}  {dec(q[j])}\n")
        f.write("ENDATA\n")
    return value
