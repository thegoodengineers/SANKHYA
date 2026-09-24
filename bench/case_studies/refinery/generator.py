#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-period refinery planning LP / MILP generator with a size ladder (#517).

SYNTHETIC DATA NOTICE
---------------------
Every numerical coefficient in the files this script produces is synthetic.
Yields, qualities, costs, capacities and demand figures were generated
pseudo-randomly from a fixed seed; they do not represent any real refinery,
crude oil stream, product specification or market price.  The model structure
(mass balance, CDU capacity, unit loads, quality budgets, delivery commitments)
follows the academic form of Floudas and Lin (2005) and Pochet and Wolsey
(2006), adapted for illustrative use.  No real company data are included.

PROBLEM STRUCTURE
-----------------
A planning horizon of T periods.  The planner buys crudes, runs them through
a crude distillation unit (CDU) and a set of processing units, and sells the
resulting products.  Inventory is held between periods.

Variables (per period t, crude k, product j):
    BUY_{k}_{t}   – barrels of crude k purchased in period t  (continuous, >= 0)
    RUN_{k}_{t}   – barrels of crude k run through the CDU    (continuous, in [0, CDU_cap])
    CS_{k}_{t}    – crude stock at end of period t             (continuous, in [0, tank_k])
    PROD_{j}_{t}  – barrels of product j made in period t      (continuous, >= 0)
    SELL_{j}_{t}  – barrels of product j sold in period t      (continuous, in [0, demand_{j,t}])
    PS_{j}_{t}    – product stock at end of period t           (continuous, in [0, tank_j])

With --milp, one binary per crude-period purchase decision is added:
    ZDEC_{k}_{t}  – 1 iff crude k is purchased in period t     (binary)
  together with the big-M linking constraint:
    BUY_{k}_{t} <= bigM * ZDEC_{k}_{t}

Constraints (per period):
    crude balance   CS_{k,t} = CS_{k,t-1} + BUY_{k,t} - RUN_{k,t}       (E)
    CDU capacity    sum_k RUN_{k,t}                            <= cap_CDU  (L)
    production      PROD_{j,t} = sum_k Y[j,k] * RUN_{k,t}                (E)
    product balance PS_{j,t} = PS_{j,t-1} + PROD_{j,t} - SELL_{j,t}     (E)
    unit capacity   sum_k H[u,k] * RUN_{k,t}                  <= cap_u_t  (L)
    quality budget  sum_k q[s,k]*Y[j_s,k] * RUN_{k,t}         <= budget   (L)
    delivery commit SELL_{j,t}                                 >= commit   (G)

Objective: minimise  sum_{k,t} c_buy[k] * BUY_{k,t}
           (+ sum_{k,t} f_{k,t} * ZDEC_{k,t}, a fixed ordering cost, with --milp)
                   - sum_{j,t} p_sell[j] * SELL_{j,t}
                   + sum_{k,t} h_crude[k] * CS_{k,t}
                   + sum_{j,t} h_prod[j] * PS_{j,t}

(purchase cost minus product revenue plus holding costs)

The LP optimum is exact by construction (KKT verified in Fractions before
writing), following the same approach as bench/runners/generate_refinery_lp.py.
The MILP file uses the same LP relaxation extended with integer markers; the
MILP optimum is not pre-verified here.

SIZE LADDER
-----------
--size small   : periods=4,  crudes=3,  products=4,  units=5,  specs=3,  commits=2
--size medium  : periods=12, crudes=8,  products=8,  units=8,  specs=6,  commits=4
--size large   : periods=52, crudes=15, products=12, units=10, specs=10, commits=8

All sizes can be overridden by the individual flags.

References
----------
Floudas, C.A. and Lin, X. (2005). Mixed Integer Linear Programming in Process
    Scheduling: Modeling, Algorithms, and Applications. Annals of Operations
    Research, 139(1), 131–162.
Pochet, Y. and Wolsey, L.A. (2006). Production Planning by Mixed Integer
    Programming. Springer Series in Operations Research and Financial
    Engineering.

Usage
-----
    python bench/case_studies/refinery/generator.py --size small --seed 1 \\
           --out refinery_small_lp.mps
    python bench/case_studies/refinery/generator.py --size small --seed 1 \\
           --milp --out refinery_small_milp.mps
    python bench/case_studies/refinery/generator.py --periods 12 --crudes 8 \\
           --seed 7 --out refinery_custom.mps
"""
from __future__ import annotations

import argparse
import random
import sys
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------------------
# Coefficient ranges: decimals with small denominators so the LP optimum is
# exact when verified in Fraction arithmetic.
# ---------------------------------------------------------------------------
YIELD_DENOM = 100       # yields in hundredths; each crude's yields sum to <=1
QUALITY_DENOM = 1000    # quality coefficients in thousandths
PLAN_RANGE = 40         # max throughput / purchase draw per crude per period
STOCK_SLACK = 5         # tank size = peak stock held + [0, STOCK_SLACK]
MULT_RANGE = 9          # |y*_i| and reduced-cost d_j drawn from [1, MULT_RANGE]
TIGHT_PROB = 0.5        # probability that a capacity / spec / demand row is tight
BIG_M_FACTOR = 4        # bigM = BIG_M_FACTOR * CDU_capacity (MILP only)
# Fixed cost of placing a crude order in a period (MILP only), as a share of what the
# largest purchase in the LP plan costs. Without it the binaries cost nothing, z = 1
# everywhere reproduces the LP optimum, and the "MILP" is the LP (review of #633).
FIXED_ORDER_SHARE = Fraction(1, 4)

# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------
SIZE_PRESETS: dict[str, dict[str, int]] = {
    "small":  dict(periods=4,  crudes=3,  products=4,  units=5,  specs=3,  commits=2),
    "medium": dict(periods=12, crudes=8,  products=8,  units=8,  specs=6,  commits=4),
    "large":  dict(periods=52, crudes=15, products=12, units=10, specs=10, commits=8),
}


# ---------------------------------------------------------------------------
# Instance container
# ---------------------------------------------------------------------------

class Instance:
    """All data needed to write the MPS file, plus the KKT witnesses."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []              # (name, sense)
        self.row_entries: list[dict[int, Fraction]] = []
        self.rhs: list[Fraction] = []
        self.cols: list[str] = []
        self.lower: list[Fraction] = []
        self.upper: list[Fraction | None] = []             # None = +inf
        self.integer: list[bool] = []
        self.x: list[Fraction] = []                        # primal plan
        self.y: list[Fraction] = []                        # dual multipliers
        self.cost: list[Fraction] = []

    def add_col(self, name: str, value: Fraction,
                upper: Fraction | None, integer: bool = False) -> int:
        self.cols.append(name)
        self.lower.append(Fraction(0))
        self.upper.append(upper)
        self.integer.append(integer)
        self.x.append(value)
        return len(self.cols) - 1

    def add_row(self, name: str, sense: str,
                entries: dict[int, Fraction], rhs: Fraction) -> int:
        self.rows.append((name, sense))
        self.row_entries.append(entries)
        self.rhs.append(rhs)
        self.y.append(Fraction(0))
        return len(self.rows) - 1


# ---------------------------------------------------------------------------
# LP builder
# ---------------------------------------------------------------------------

def build_lp(periods: int, crudes: int, products: int, units: int,
             specs: int, commits: int, seed: int) -> Instance:
    """Construct the LP instance with a KKT-feasible primal/dual pair."""
    rng = random.Random(seed)
    inst = Instance()
    fr = Fraction

    # ---- Fixed refinery data ------------------------------------------------
    # Yields Y[j][k] in hundredths, summing to <= 100 per crude (balance is loss)
    yields: list[list[Fraction]] = [[fr(0)] * crudes for _ in range(products)]
    for k in range(crudes):
        cuts = sorted(rng.sample(range(1, 100), products - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [rng.randint(cuts[-1] + 1, 100)])]
        for j in range(products):
            yields[j][k] = fr(parts[j], YIELD_DENOM)

    # Unit loads H[u][k]: fraction of a barrel that goes through unit u per barrel of crude k
    loads: list[list[Fraction]] = [[fr(0)] * crudes for _ in range(units)]
    for u in range(units):
        for k in rng.sample(range(crudes), max(1, crudes // 2)):
            loads[u][k] = fr(rng.randint(5, 60), 100)

    # Quality coefficients q[s][k] and which product pool each spec constrains
    qualities: list[list[Fraction]] = [
        [fr(rng.randint(1, 999), QUALITY_DENOM) for _ in range(crudes)]
        for _ in range(specs)
    ]
    spec_product = [rng.randrange(products) for _ in range(specs)]
    commit_products = rng.sample(range(products), min(commits, products))

    # ---- Choose operating plan x* feasible by construction -----------------
    cdu_cap = fr(PLAN_RANGE * crudes // 2)
    run = [[fr(0)] * periods for _ in range(crudes)]
    buy = [[fr(0)] * periods for _ in range(crudes)]
    cs = [[fr(0)] * periods for _ in range(crudes)]
    cs_initial = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(crudes)]

    for t in range(periods):
        draw = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(crudes)]
        total = sum(draw)
        if rng.random() < TIGHT_PROB and total > 0:
            # Scale to exactly hit CDU capacity
            draw = [fr(int(d * cdu_cap / total)) for d in draw]
            short = cdu_cap - sum(draw)
            for k in range(crudes):
                if draw[k] > 0 or short > 0:
                    draw[k] += short
                    break
        elif total > cdu_cap:
            draw = [fr(int(d * (cdu_cap - 1) / total)) for d in draw]
        for k in range(crudes):
            prev = cs_initial[k] if t == 0 else cs[k][t - 1]
            run[k][t] = draw[k]
            need = max(fr(0), run[k][t] - prev)
            buy[k][t] = need + fr(rng.randint(0, PLAN_RANGE // 4))
            cs[k][t] = prev + buy[k][t] - run[k][t]
            assert cs[k][t] >= 0

    prod = [[sum(yields[j][k] * run[k][t] for k in range(crudes))
             for t in range(periods)]
            for j in range(products)]
    sell = [[fr(0)] * periods for _ in range(products)]
    ps = [[fr(0)] * periods for _ in range(products)]
    ps_initial = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(products)]
    for j in range(products):
        for t in range(periods):
            prev = ps_initial[j] if t == 0 else ps[j][t - 1]
            available = prev + prod[j][t]
            share = fr(rng.randint(0, 10), 10)
            sell[j][t] = fr(int(available * share * 100), 100)
            ps[j][t] = available - sell[j][t]
            assert ps[j][t] >= 0

    # ---- Column bounds and indices -----------------------------------------
    crude_tank = [max(cs[k]) + fr(rng.randint(0, STOCK_SLACK)) for k in range(crudes)]
    product_tank = [max(ps[j]) + fr(rng.randint(0, STOCK_SLACK)) for j in range(products)]

    col_buy = [[0] * periods for _ in range(crudes)]
    col_run = [[0] * periods for _ in range(crudes)]
    col_cs  = [[0] * periods for _ in range(crudes)]
    col_prod = [[0] * periods for _ in range(products)]
    col_sell = [[0] * periods for _ in range(products)]
    col_ps   = [[0] * periods for _ in range(products)]

    for t in range(periods):
        for k in range(crudes):
            col_buy[k][t] = inst.add_col(f"BUY_{k}_{t}", buy[k][t], None)
            col_run[k][t] = inst.add_col(f"RUN_{k}_{t}", run[k][t], cdu_cap)
            col_cs[k][t]  = inst.add_col(f"CS_{k}_{t}",  cs[k][t],  crude_tank[k])
        for j in range(products):
            col_prod[j][t] = inst.add_col(f"PROD_{j}_{t}", prod[j][t], None)
            demand = sell[j][t] + (fr(0) if rng.random() < TIGHT_PROB
                                   else fr(rng.randint(1, PLAN_RANGE)))
            col_sell[j][t] = inst.add_col(f"SELL_{j}_{t}", sell[j][t], demand)
            col_ps[j][t]   = inst.add_col(f"PS_{j}_{t}",   ps[j][t],   product_tank[j])

    # ---- Constraints --------------------------------------------------------
    for t in range(periods):
        for k in range(crudes):
            entries: dict[int, Fraction] = {
                col_cs[k][t]: fr(1), col_buy[k][t]: fr(-1), col_run[k][t]: fr(1)
            }
            rhs = cs_initial[k] if t == 0 else fr(0)
            if t > 0:
                entries[col_cs[k][t - 1]] = fr(-1)
            inst.add_row(f"CBAL_{k}_{t}", "E", entries, rhs)

        entries = {col_run[k][t]: fr(1) for k in range(crudes)}
        inst.add_row(f"CDU_{t}", "L", entries, cdu_cap)

        for j in range(products):
            entries = {col_prod[j][t]: fr(1)}
            for k in range(crudes):
                if yields[j][k] != 0:
                    entries[col_run[k][t]] = -yields[j][k]
            inst.add_row(f"MAKE_{j}_{t}", "E", entries, fr(0))

        for j in range(products):
            entries = {
                col_ps[j][t]: fr(1), col_prod[j][t]: fr(-1), col_sell[j][t]: fr(1)
            }
            rhs = ps_initial[j] if t == 0 else fr(0)
            if t > 0:
                entries[col_ps[j][t - 1]] = fr(-1)
            inst.add_row(f"PBAL_{j}_{t}", "E", entries, rhs)

        for u in range(units):
            entries = {col_run[k][t]: loads[u][k]
                       for k in range(crudes) if loads[u][k] != 0}
            load = sum(loads[u][k] * run[k][t] for k in range(crudes))
            tight = rng.random() < TIGHT_PROB
            cap_u = load if tight else load + fr(rng.randint(1, PLAN_RANGE))
            inst.add_row(f"UNIT_{u}_{t}", "L", entries, cap_u)

        for s in range(specs):
            j = spec_product[s]
            entries = {}
            blended = fr(0)
            for k in range(crudes):
                coeff = qualities[s][k] * yields[j][k]
                if coeff != 0:
                    entries[col_run[k][t]] = coeff
                    blended += coeff * run[k][t]
            tight = rng.random() < TIGHT_PROB
            budget = blended if tight else blended + fr(rng.randint(1, PLAN_RANGE), 100)
            inst.add_row(f"SPEC_{s}_{t}", "L", entries, budget)

        for j in commit_products:
            tight = rng.random() < TIGHT_PROB
            commit = sell[j][t] if tight else max(fr(0), sell[j][t] - fr(rng.randint(1, 10)))
            inst.add_row(f"COMMIT_{j}_{t}", "G", {col_sell[j][t]: fr(1)}, commit)

    # ---- KKT multipliers and dual-feasible prices ---------------------------
    activity = [sum(c * inst.x[col] for col, c in ent.items())
                for ent in inst.row_entries]
    for i, (name, sense) in enumerate(inst.rows):
        tight = activity[i] == inst.rhs[i]
        if sense == "E":
            assert tight, f"{name}: equality violated by plan"
            inst.y[i] = fr(rng.choice([-1, 1]) * rng.randint(1, MULT_RANGE))
        elif sense == "L":
            assert activity[i] <= inst.rhs[i], f"{name}: L row violated"
            inst.y[i] = fr(-rng.randint(1, MULT_RANGE)) if tight else fr(0)
        else:
            assert activity[i] >= inst.rhs[i], f"{name}: G row violated"
            inst.y[i] = fr(rng.randint(1, MULT_RANGE)) if tight else fr(0)

    aty = [fr(0)] * len(inst.cols)
    for i, ent in enumerate(inst.row_entries):
        if inst.y[i] == 0:
            continue
        for col, c in ent.items():
            aty[col] += c * inst.y[i]

    for j in range(len(inst.cols)):
        at_lo = inst.x[j] == inst.lower[j]
        at_hi = inst.upper[j] is not None and inst.x[j] == inst.upper[j]
        if at_lo and at_hi:
            d = fr(rng.choice([-1, 1]) * rng.randint(0, MULT_RANGE))
        elif at_lo:
            d = fr(rng.randint(0, MULT_RANGE))
        elif at_hi:
            d = fr(-rng.randint(0, MULT_RANGE))
        else:
            d = fr(0)
        inst.cost.append(d + aty[j])

    return inst


# ---------------------------------------------------------------------------
# MILP extension: add binary purchase decision variables
# ---------------------------------------------------------------------------

def extend_to_milp(inst: Instance, periods: int, crudes: int) -> None:
    """Add binary ZDEC_{k}_{t} variables and BUY <= bigM * ZDEC linking rows.

    The LP optimal primal values for BUY are all >= 0; we choose bigM large
    enough that the LP relaxation is not tightened, so the LP optimum is still
    feasible for the MILP relaxation.  Each order carries a fixed cost
    (FIXED_ORDER_SHARE of the largest purchase's cost), so the MILP optimum is
    strictly above the LP optimum whenever crude must be bought.  The MILP
    optimum is not pre-verified.
    """
    fr = Fraction
    # Identify BUY column indices: they were added in the order (t=0,k=0..K-1),
    # (t=1,k=0..K-1), ... so we locate them by name.
    name_to_col = {name: j for j, name in enumerate(inst.cols)}

    # bigM: large enough to never cut off BUY values in the LP plan
    max_buy = max(inst.x[name_to_col[f"BUY_{k}_{t}"]]
                  for t in range(periods) for k in range(crudes))
    big_m = max(max_buy * fr(BIG_M_FACTOR), fr(1000))

    for t in range(periods):
        for k in range(crudes):
            buy_val = inst.x[name_to_col[f"BUY_{k}_{t}"]]
            # z* = 1 iff buy* > 0, which satisfies BUY <= bigM * z* and z binary
            z_val = fr(1) if buy_val > 0 else fr(0)
            z_col = inst.add_col(f"ZDEC_{k}_{t}", z_val, fr(1), integer=True)
            buy_col = name_to_col[f"BUY_{k}_{t}"]
            # BUY_{k,t} - bigM * ZDEC_{k,t} <= 0
            inst.add_row(f"ZLINK_{k}_{t}", "L",
                         {buy_col: fr(1), z_col: -big_m}, fr(0))
            # A fixed ordering cost: the LP relaxation pays it only in proportion
            # BUY / bigM, so the relaxation is weaker than the MILP and branching matters.
            # |c_buy|: the KKT construction gives signed prices, and a negative fixed cost
            # would reward ordering instead of charging for it.
            inst.cost.append(FIXED_ORDER_SHARE * max(abs(inst.cost[buy_col]), fr(1)) * max_buy)


# ---------------------------------------------------------------------------
# KKT verifier (LP only; called before any file is written)
# ---------------------------------------------------------------------------

def verify_lp(inst: Instance) -> None:
    """Check every KKT condition exactly before writing anything."""
    n = len(inst.cols)
    for j in range(n):
        if inst.integer[j]:
            continue  # skip binary extension columns
        assert inst.lower[j] <= inst.x[j], f"{inst.cols[j]} below lower bound"
        assert inst.upper[j] is None or inst.x[j] <= inst.upper[j], (
            f"{inst.cols[j]} above upper bound")

    aty = [Fraction(0)] * n
    for i, ent in enumerate(inst.row_entries):
        activity = sum(c * inst.x[col] for col, c in ent.items())
        name, sense = inst.rows[i]
        if sense == "E":
            assert activity == inst.rhs[i], f"{name}: equality violated"
        elif sense == "L":
            assert activity <= inst.rhs[i], f"{name}: L violated"
            assert inst.y[i] <= 0, f"{name}: L row multiplier sign"
        else:
            assert activity >= inst.rhs[i], f"{name}: G violated"
            assert inst.y[i] >= 0, f"{name}: G row multiplier sign"
        if sense != "E" and activity != inst.rhs[i]:
            assert inst.y[i] == 0, f"{name}: slack row has nonzero multiplier"
        for col, c in ent.items():
            aty[col] += c * inst.y[i]

    for j in range(n):
        if inst.integer[j]:
            continue
        d = inst.cost[j] - aty[j]
        at_lo = inst.x[j] == inst.lower[j]
        at_hi = inst.upper[j] is not None and inst.x[j] == inst.upper[j]
        if not at_lo and not at_hi:
            assert d == 0, f"{inst.cols[j]}: interior with nonzero reduced cost {d}"
        elif at_lo and not at_hi:
            assert d >= 0, f"{inst.cols[j]}: at lower bound with negative reduced cost {d}"
        elif at_hi and not at_lo:
            assert d <= 0, f"{inst.cols[j]}: at upper bound with positive reduced cost {d}"


# ---------------------------------------------------------------------------
# MPS writer
# ---------------------------------------------------------------------------

def _decimal(v: Fraction) -> str:
    """Exact decimal text for a Fraction whose denominator divides a power of 10."""
    if v.denominator == 1:
        return str(v.numerator)
    scale, digits = 1, 0
    while scale % v.denominator != 0:
        scale *= 10
        digits += 1
        if digits > 12:
            raise ValueError(f"{v} has no short decimal representation")
    sign = "-" if v < 0 else ""
    integer, frac = divmod(abs(v.numerator) * (scale // v.denominator), scale)
    text = f"{sign}{integer}.{frac:0{digits}d}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def write_mps(inst: Instance, out: Path, milp: bool,
              periods: int, crudes: int, products: int,
              units: int, seed: int) -> None:
    lp_cols = [j for j in range(len(inst.cols)) if not inst.integer[j]]
    int_cols = [j for j in range(len(inst.cols)) if inst.integer[j]]
    optimum = sum(inst.cost[j] * inst.x[j] for j in lp_cols)  # LP relaxation optimum

    col_rows: list[list[tuple[int, Fraction]]] = [[] for _ in inst.cols]
    for i, ent in enumerate(inst.row_entries):
        for col, c in ent.items():
            col_rows[col].append((i, c))

    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"NAME          REFINERY_CS_T{periods}_K{crudes}_P{products}"
                f"_U{units}_S{seed}{'_MILP' if milp else ''}\n")
        f.write("* SYNTHETIC DATA — see bench/case_studies/refinery/generator.py (#517)\n")
        f.write("* All yields, qualities, costs and capacities are pseudo-random.\n")
        f.write("* Structure follows Floudas & Lin (2005) and Pochet & Wolsey (2006).\n")
        f.write(f"* periods={periods}  crudes={crudes}  products={products}"
                f"  units={units}  seed={seed}\n")
        f.write(f"* {'MILP with binary crude-purchase decisions' if milp else 'LP relaxation'}\n")
        if not milp:
            f.write(f"* LP analytic optimum: {float(optimum)!r}\n")
        f.write("ROWS\n N  COST\n")
        for name, sense in inst.rows:
            f.write(f" {sense}  {name}\n")
        f.write("COLUMNS\n")
        # Write continuous columns first, then binary
        wrote_int_marker = False
        for j, name in enumerate(inst.cols):
            if inst.integer[j] and not wrote_int_marker:
                f.write("    MARKER  'MARKER'  'INTORG'\n")
                wrote_int_marker = True
            if not inst.integer[j] and wrote_int_marker:
                f.write("    MARKER  'MARKER'  'INTEND'\n")
                wrote_int_marker = False
            if inst.cost[j] != 0:
                f.write(f"    {name}  COST  {_decimal(inst.cost[j])}\n")
            for i, c in col_rows[j]:
                f.write(f"    {name}  {inst.rows[i][0]}  {_decimal(c)}\n")
            if inst.cost[j] == 0 and not col_rows[j]:
                f.write(f"    {name}  COST  0\n")
        if wrote_int_marker:
            f.write("    MARKER  'MARKER'  'INTEND'\n")
        f.write("RHS\n")
        for i, (name, _) in enumerate(inst.rows):
            if inst.rhs[i] != 0:
                f.write(f"    RHS  {name}  {_decimal(inst.rhs[i])}\n")
        f.write("BOUNDS\n")
        for j, name in enumerate(inst.cols):
            if inst.integer[j]:
                # Binary: 0/1 bounds with UI (upper integer) bound type
                f.write(f" UI BND  {name}  1\n")
            elif inst.upper[j] is not None:
                f.write(f" UP BND  {name}  {_decimal(inst.upper[j])}\n")
        f.write("ENDATA\n")

    n_rows = len(inst.rows)
    n_cols = len(inst.cols)
    nnz = sum(len(e) for e in inst.row_entries)
    print(f"wrote {out}")
    print(f"  rows={n_rows}  cols={n_cols}  nonzeros={nnz}"
          f"  binary={len(int_cols)}")
    if not milp:
        print(f"  LP analytic optimum: {float(optimum)!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=["small", "medium", "large"],
                        help="preset size (overridden by individual flags)")
    parser.add_argument("--periods",  type=int, help="number of time periods T")
    parser.add_argument("--crudes",   type=int, help="number of crude streams K")
    parser.add_argument("--products", type=int, help="number of products J")
    parser.add_argument("--units",    type=int, help="number of processing units U")
    parser.add_argument("--specs",    type=int, help="number of quality specifications")
    parser.add_argument("--commits",  type=int, help="products with delivery commitments")
    parser.add_argument("--seed",     type=int, default=1, help="RNG seed (default: 1)")
    parser.add_argument("--milp",     action="store_true",
                        help="add binary crude-purchase decision variables")
    parser.add_argument("--out",      type=Path, required=True,
                        help="output .mps file path")
    args = parser.parse_args(argv)

    # Start from size preset, then apply any explicit overrides
    dims = dict(periods=4, crudes=3, products=4, units=5, specs=3, commits=2)
    if args.size:
        dims.update(SIZE_PRESETS[args.size])
    for key in dims:
        val = getattr(args, key)
        if val is not None:
            dims[key] = val

    if dims["periods"] < 1 or dims["crudes"] < 2 or dims["products"] < 2:
        print("periods >= 1, crudes >= 2, products >= 2", file=sys.stderr)
        return 2

    inst = build_lp(**dims, seed=args.seed)
    verify_lp(inst)

    if args.milp:
        extend_to_milp(inst, dims["periods"], dims["crudes"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_mps(inst, args.out, milp=args.milp,
              periods=dims["periods"], crudes=dims["crudes"],
              products=dims["products"], units=dims["units"],
              seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
