#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the SANKHYA industrial case studies named in SIH PS26119.

The .mps files are committed, so a judge does not need to run this. It exists so that every
coefficient in them has a visible derivation instead of being a number somebody typed, and
so the ill-conditioned instance can be proved to be a re-scaling of a model whose answer we
already know rather than a different problem with a convenient answer.

    python data/casestudies/generate.py
"""
from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent


class Mps:
    """Minimal MPS writer. Emits the aligned fixed-format layout the committed files use."""

    def __init__(self, name: str, sense: str = "MINIMIZE") -> None:
        self.name = name
        self.sense = sense
        self.rows: list[tuple[str, str]] = []  # (type, name); N row first
        self.cols: dict[str, list[tuple[str, float]]] = {}
        self.integer: set[str] = set()
        self.rhs: list[tuple[str, float]] = []
        self.bounds: list[tuple[str, str, float]] = []  # (type, col, value)
        self.comment: list[str] = []

    def row(self, kind: str, name: str) -> None:
        self.rows.append((kind, name))

    def col(self, name: str, entries: list[tuple[str, float]], integral: bool = False) -> None:
        self.cols.setdefault(name, []).extend(entries)
        if integral:
            self.integer.add(name)

    def write(self, path: pathlib.Path) -> None:
        out: list[str] = [("* " + line).rstrip() for line in self.comment]
        out.append("NAME          " + self.name)
        if self.sense != "MINIMIZE":
            out += ["OBJSENSE", "    " + self.sense]
        out.append("ROWS")
        for kind, name in self.rows:
            out.append(" {}  {}".format(kind, name))
        out.append("COLUMNS")
        marker = 0
        in_int = False
        for name, entries in self.cols.items():
            want_int = name in self.integer
            if want_int != in_int:
                tag = "INTORG" if want_int else "INTEND"
                out.append("    MARKER{:04d}  'MARKER'                 '{}'".format(marker, tag))
                marker += 1
                in_int = want_int
            for i in range(0, len(entries), 2):
                line = "    {:<10}".format(name)
                line += "".join("{:<10}{:>14.8g}  ".format(r, v) for r, v in entries[i:i + 2])
                out.append(line.rstrip())
        if in_int:
            out.append("    MARKER{:04d}  'MARKER'                 'INTEND'".format(marker))
        if self.rhs:
            out.append("RHS")
            for i in range(0, len(self.rhs), 2):
                line = "    {:<10}".format("RHS")
                line += "".join("{:<10}{:>14.8g}  ".format(r, v) for r, v in self.rhs[i:i + 2])
                out.append(line.rstrip())
        if self.bounds:
            out.append("BOUNDS")
            for kind, colname, value in self.bounds:
                if kind in ("BV", "FR", "MI", "PL"):
                    out.append(" {} BND       {}".format(kind, colname))
                else:
                    out.append(" {} BND       {:<10}{:>14.8g}".format(kind, colname, value))
        out.append("ENDATA")
        # newline set explicitly. .gitattributes stores data/**/*.mps with -text, byte for
        # byte, so a CRLF written here on Windows is committed as CRLF, and the CI drift
        # check then regenerates LF on Ubuntu and fails on every line of every file.
        with open(path, "w", encoding="utf-8", newline=chr(10)) as handle:
            handle.write(chr(10).join(out) + chr(10))
        print("wrote " + str(path.relative_to(HERE.parent.parent)).replace("\\", "/"))


# ===========================================================================================
# 1. Power system dispatch - single-period unit commitment.
# ===========================================================================================
UNITS = [
    # name, marginal cost $/MWh, Pmin, Pmax, start-up cost $
    ("GA", 10.0, 20.0, 100.0, 100.0),
    ("GB", 12.0, 30.0, 120.0, 80.0),
    ("GC", 20.0, 10.0, 150.0, 50.0),
    ("GD", 8.0, 50.0, 60.0, 450.0),
]
DEMAND = 250.0
RESERVE_FACTOR = 1.15


def power_dispatch() -> Mps:
    m = Mps("PWRDISP")
    m.comment = [
        "SANKHYA case study - power system dispatch (single-period unit commitment).",
        "",
        "PS26119 names 'power system dispatch' in scope. This is its smallest honest form:",
        "each generating unit has a marginal cost, a minimum stable generation level it",
        "cannot run below, a maximum, and a start-up cost paid only if it is committed.",
        "The min-stable-generation level is what makes this a MILP rather than an LP - a",
        "unit is either off, or on and producing AT LEAST its minimum.",
        "",
        "  min  sum_g ( c_g * p_g + s_g * u_g )",
        "  s.t. sum_g p_g = D                     (DEMAND, met exactly)",
        "       sum_g Pmax_g * u_g >= 1.15 * D    (RESERVE, spinning reserve margin)",
        "       p_g - Pmax_g * u_g <= 0           (CAPMX, off means zero)",
        "       p_g - Pmin_g * u_g >= 0           (CAPMN, on means at least Pmin)",
        "       u_g binary",
        "",
        "  D = {:g} MW, reserve margin {:g}".format(DEMAND, RESERVE_FACTOR),
        "",
        "  unit   $/MWh   Pmin   Pmax   start-up",
    ]
    for name, cost, pmin, pmax, start in UNITS:
        m.comment.append(
            "  {}   {:6.1f} {:6.1f} {:6.1f}   {:8.1f}".format(name, cost, pmin, pmax, start))
    m.comment += [
        "",
        "GD is the trap, and the margin is deliberately thin. It burns the cheapest fuel on",
        "the system, so a merit-order rule commits it on sight. Doing so displaces 60 MW - 40",
        "from GB at 12 and 20 from GC at 20 - which saves 40*12 + 20*20 = 880 in fuel, against",
        "60*8 + 450 = 930 to run and start it. Committing GD is therefore worse by exactly 50",
        "out of 3270, about 1.5%. At a start-up cost of 400 instead of 450 the two commitments",
        "TIE at 3270 while using completely different units, which is how this instance was",
        "found: the solver and the exhaustive oracle returned the same cost and disagreed on",
        "every unit. Alternate optima are normal in dispatch models and are the reason the",
        "oracle here compares the objective, not the assignment.",
    ]
    m.row("N", "COST")
    m.row("E", "DEMAND")
    m.row("G", "RESERVE")
    for name, _, _, _, _ in UNITS:
        m.row("L", "CAPMX" + name)
        m.row("G", "CAPMN" + name)
    for name, cost, _, _, _ in UNITS:
        m.col("P" + name, [("COST", cost), ("DEMAND", 1.0),
                           ("CAPMX" + name, 1.0), ("CAPMN" + name, 1.0)])
    for name, _, pmin, pmax, start in UNITS:
        m.col("U" + name, [("COST", start), ("RESERVE", pmax),
                           ("CAPMX" + name, -pmax), ("CAPMN" + name, -pmin)], integral=True)
    m.rhs = [("DEMAND", DEMAND), ("RESERVE", RESERVE_FACTOR * DEMAND)]
    for name, _, _, pmax, _ in UNITS:
        m.bounds.append(("UP", "P" + name, pmax))
    for name, _, _, _, _ in UNITS:
        m.bounds.append(("BV", "U" + name, 0.0))
    return m


# ===========================================================================================
# 2. Supply chain distribution - balanced transportation. STRUCTURALLY DEGENERATE.
# ===========================================================================================
PLANTS = [("MANGALORE", 300.0), ("KOCHI", 400.0), ("CHENNAI", 500.0)]
DEPOTS = [("BENGALURU", 250.0), ("HYDERABAD", 350.0), ("COIMBATORE", 400.0), ("MADURAI", 200.0)]
FREIGHT = [
    [4.0, 6.0, 9.0, 5.0],
    [5.0, 3.0, 7.0, 8.0],
    [6.0, 7.0, 4.0, 3.0],
]


def _transport_rows() -> list[str]:
    names = ["SUP{}".format(i + 1) for i in range(len(PLANTS))]
    names += ["DEM{}".format(j + 1) for j in range(len(DEPOTS))]
    return names


def _transport_rhs() -> list[tuple[str, float]]:
    items = [("SUP{}".format(i + 1), s) for i, (_, s) in enumerate(PLANTS)]
    items += [("DEM{}".format(j + 1), d) for j, (_, d) in enumerate(DEPOTS)]
    return items


def supply_chain() -> Mps:
    m = Mps("SUPPLYCH")
    m.comment = [
        "SANKHYA case study - product distribution from refineries to depots.",
        "",
        "PS26119 names 'transportation and supply chain management'. This is the classical",
        "balanced transportation problem, chosen because of a property it has BY",
        "CONSTRUCTION rather than by accident:",
        "",
        "  TOTAL SUPPLY EQUALS TOTAL DEMAND, so the seven equality rows are linearly",
        "  DEPENDENT - summing the three supply rows and summing the four demand rows give",
        "  the same equation. The constraint matrix has rank 6, not 7.",
        "",
        "That redundancy is why this instance is here. A simplex implementation that assumes",
        "its basis matrix is nonsingular, or an LU that cannot handle a structurally singular",
        "column, fails on it. Every vertex is also degenerate: a basis needs m + n - 1 = 6",
        "basic variables but the tableau has 7 rows, so at least one basic variable sits at",
        "zero at every iteration and ratio-test ties are the norm rather than the exception.",
        "This is the standard cycling test bed.",
        "",
        "  min  sum_ij f_ij x_ij",
        "  s.t. sum_j x_ij = supply_i    for each plant",
        "       sum_i x_ij = demand_j    for each depot",
        "",
        "  supply {} = {:g}".format([s for _, s in PLANTS], sum(s for _, s in PLANTS)),
        "  demand {} = {:g}".format([d for _, d in DEPOTS], sum(d for _, d in DEPOTS)),
        "",
        "  freight cost per unit, plant (row) to depot (column):",
    ]
    for i, (p, _) in enumerate(PLANTS):
        m.comment.append("    {:<10} {}".format(p, FREIGHT[i]))
    m.row("N", "FREIGHT")
    for name in _transport_rows():
        m.row("E", name)
    for i in range(len(PLANTS)):
        for j in range(len(DEPOTS)):
            m.col("X{}{}".format(i + 1, j + 1),
                  [("FREIGHT", FREIGHT[i][j]),
                   ("SUP{}".format(i + 1), 1.0), ("DEM{}".format(j + 1), 1.0)])
    m.rhs = _transport_rhs()
    return m


# ===========================================================================================
# 3. The same model, deliberately ill-conditioned.
# ===========================================================================================
ROW_SCALE = [1e-6, 1e3, 1e6, 1e-5, 1e4, 1e-3, 1e5]
COL_SCALE = [1e4, 1e-4, 1e3, 1e-3, 1e5, 1e-5, 1e2, 1e-2, 1e6, 1e-6, 1e1, 1e-1]


def ill_conditioned() -> Mps:
    m = Mps("ILLCOND")
    m.comment = [
        "SANKHYA case study - the supply chain model, deliberately badly scaled.",
        "",
        "PS26119 asks for robustness on 'ill-conditioned constraint matrices'. The honest way",
        "to demonstrate that is on an instance WHOSE ANSWER IS ALREADY KNOWN, so the claim is",
        "falsifiable rather than a number nobody can check.",
        "",
        "This is data/casestudies/supply_chain.mps with row i multiplied by r_i and column j",
        "substituted x_j -> t_j * x'_j, the objective coefficient scaled to match:",
        "",
        "  row scales     10^-6 ... 10^+6",
        "  column scales  10^-6 ... 10^+4",
        "",
        "Both operations are exact re-parameterisations. The feasible set is the same set seen",
        "through a diagonal change of variables, so",
        "",
        "  THE OPTIMAL OBJECTIVE IS UNCHANGED, and    x'_j = x_j / t_j.",
        "",
        "Row and column scales COMPOUND, so the entries now span twenty-two orders of",
        "magnitude - 1e-10 to 1e+12 - and the 2-norm condition number is about 2e28. A solver",
        "that pivots on",
        "magnitude alone selects a numerically worthless pivot here, and a solver with no",
        "scaling declares it infeasible or unbounded. The demo solves both files and checks",
        "the two objectives agree - if they do not, we have a bug, and it shows.",
    ]
    m.row("N", "FREIGHT")
    row_names = _transport_rows()
    for name in row_names:
        m.row("E", name)
    scale_of = dict(zip(row_names, ROW_SCALE))
    k = 0
    for i in range(len(PLANTS)):
        for j in range(len(DEPOTS)):
            t = COL_SCALE[k]
            k += 1
            sup = "SUP{}".format(i + 1)
            dem = "DEM{}".format(j + 1)
            m.col("X{}{}".format(i + 1, j + 1),
                  [("FREIGHT", FREIGHT[i][j] * t),
                   (sup, scale_of[sup] * t), (dem, scale_of[dem] * t)])
    m.rhs = [(name, scale_of[name] * value) for name, value in _transport_rhs()]
    return m


# ===========================================================================================
# 4. Production planning - lot sizing. WEAK LP RELAXATION.
# ===========================================================================================
LOT_DEMAND = [40.0, 60.0, 30.0, 50.0]
UNIT_COST, SETUP_COST, HOLD_COST = 2.0, 150.0, 1.0


def lot_sizing(relaxed: bool = False) -> Mps:
    horizon = len(LOT_DEMAND)
    big_m = sum(LOT_DEMAND)
    m = Mps("LOTSIZLP" if relaxed else "LOTSIZE")
    m.comment = [
        "SANKHYA case study - production planning by lot sizing.",
        "",
        "PS26119 asks for robustness on 'weak LP relaxations'. This is the textbook source of",
        "one. Producing in a period costs a fixed set-up charge no matter how little is made,",
        "and the only way to write that in a MILP is the big-M link",
        "",
        "      x_t <= M * y_t,     M = total demand over the horizon",
        "",
        "In the RELAXATION y_t is free to take the value x_t / M, so a period producing one",
        "unit pays 1/180 of a set-up instead of a whole one. The relaxation therefore buys the",
        "set-up structure at a fraction of its true price and its bound sits far below the",
        "integer optimum. Branch and bound has to close that gap by search.",
        "",
        "  min  sum_t ( 2 x_t + 150 y_t + 1 s_t )",
        "  s.t. s_{t-1} + x_t - s_t = d_t     (BAL, inventory balance, s_0 = 0)",
        "       x_t - M y_t <= 0              (LINK)",
        "       y_t binary",
        "",
        "  demand {}, M = {:g}".format(LOT_DEMAND, big_m),
        "",
        "The demo reports the relaxation bound and the integer optimum side by side, so the",
        "size of the gap the search actually had to close is visible rather than asserted.",
    ]
    m.row("N", "TCOST")
    for t in range(horizon):
        m.row("E", "BAL{}".format(t + 1))
        m.row("L", "LINK{}".format(t + 1))
    for t in range(horizon):
        m.col("X{}".format(t + 1),
              [("TCOST", UNIT_COST), ("BAL{}".format(t + 1), 1.0), ("LINK{}".format(t + 1), 1.0)])
    for t in range(horizon - 1):  # no stock is carried past the end of the horizon
        m.col("S{}".format(t + 1),
              [("TCOST", HOLD_COST), ("BAL{}".format(t + 1), -1.0), ("BAL{}".format(t + 2), 1.0)])
    for t in range(horizon):
        m.col("Y{}".format(t + 1),
              [("TCOST", SETUP_COST), ("LINK{}".format(t + 1), -big_m)], integral=not relaxed)
        if relaxed:
            m.bounds.append(("UP", "Y{}".format(t + 1), 1.0))
        else:
            m.bounds.append(("BV", "Y{}".format(t + 1), 0.0))
    m.rhs = [("BAL{}".format(t + 1), LOT_DEMAND[t]) for t in range(horizon)]
    return m


# ===========================================================================================
# 5. Refinery gasoline pooling - a non-convex QCQP (#514).
# ===========================================================================================
# Five gasoline components go to two grades. Two of the components share one rundown tank and
# two share another (tankage is short, the reason pooling exists at all), so each tank's
# sulphur and octane are DECISIONS, and a tank's quality times its outflow is bilinear.
# Alkylate goes to the grades directly. Blending is linear by volume for both qualities,
# a simplification for octane. Flows in kbbl/day, money in $/bbl, so the objective is k$/day.
# Octane is a lower specification, so it is carried as -RON with an upper specification.
REFINERY_COMPONENTS = [
    # node, name, cost $/bbl, sulphur ppm, RON, available kbbl/day
    (1, "straight-run naphtha", 60.0, 8.0, 68.0, 40.0),
    (2, "FCC gasoline", 66.0, 15.0, 92.0, 60.0),
    (3, "reformate", 74.0, 1.0, 99.0, 50.0),
    (4, "isomerate", 70.0, 2.0, 88.0, 30.0),
    (5, "alkylate", 82.0, 4.0, 96.0, 25.0),
]
REFINERY_TANKS = [(6, "tank T1", (1, 2), 80.0), (7, "tank T2", (3, 4), 70.0)]
REFINERY_GRADES = [
    # node, name, price $/bbl, max sulphur ppm, min RON, max demand kbbl/day
    (8, "regular, RON 91", 78.0, 10.0, 91.0, 100.0),
    (9, "premium, RON 95", 88.0, 10.0, 95.0, 50.0),
]


def refinery_pooling() -> str:
    """The P-formulation, written by the pooling set's own generator (#516)."""
    import sys
    sys.path.insert(0, str(HERE.parent.parent / "bench" / "runners"))
    import pooling_models as pm

    arcs = []
    for tank, _, inputs, _ in REFINERY_TANKS:
        cost = {node: c for node, _, c, _, _, _ in REFINERY_COMPONENTS}
        arcs += [[i, tank, cost[i]] for i in inputs]
        arcs += [[tank, grade, -price] for grade, _, price, _, _, _ in REFINERY_GRADES]
    alkylate = REFINERY_COMPONENTS[4]
    arcs += [[alkylate[0], grade, alkylate[2] - price]
             for grade, _, price, _, _, _ in REFINERY_GRADES]
    quality = {str(node): [s, -ron] for node, _, _, s, ron, _ in REFINERY_COMPONENTS}
    quality.update({str(node): [s, -ron] for node, _, _, s, ron, _ in REFINERY_GRADES})
    upper = {str(node): avail for node, _, _, _, _, avail in REFINERY_COMPONENTS}
    upper.update({str(node): cap for node, _, _, cap in REFINERY_TANKS})
    upper.update({str(node): demand for node, _, _, _, _, demand in REFINERY_GRADES})
    net = pm.Network({
        "name": "refinery_pooling", "sources": [c[0] for c in REFINERY_COMPONENTS],
        "pools": [t[0] for t in REFINERY_TANKS], "terminals": [g[0] for g in REFINERY_GRADES],
        "num_qualities": 2, "arcs": arcs, "quality": quality, "lower": {}, "upper": upper,
    })
    lines = [
        "SANKHYA case study - refinery gasoline pooling, a NON-CONVEX QCQP (#514).",
        "GENERATED by data/casestudies/generate.py; do not edit. Numbers invented, not MRPL data.",
        "",
        "Components (node: cost $/bbl, sulphur ppm, RON, available kbbl/day):",
    ]
    lines += [f"  {n} {name}: {c:g}, {s:g}, {ron:g}, <= {a:g}"
              for n, name, c, s, ron, a in REFINERY_COMPONENTS]
    lines += ["Rundown tanks (pools): " + "; ".join(
        f"{n} {name} takes {' and '.join(str(i) for i in inputs)}, <= {cap:g}"
        for n, name, inputs, cap in REFINERY_TANKS) + "; alkylate (5) goes to both grades."]
    lines += ["Grades (node: price $/bbl, sulphur <= ppm, RON >=, demand <= kbbl/day):"]
    lines += [f"  {n} {name}: {p:g}, {s:g}, {ron:g}, {d:g}"
              for n, name, p, s, ron, d in REFINERY_GRADES]
    lines += [
        "",
        "Minimize cost - revenue (k$/day). Columns f_u_v are flows, p_t_1 a tank's sulphur",
        "and p_t_2 minus its RON. Each tank's quality balance (pq rows) and each grade's",
        "specification (tq rows) carry p_t_k * f_t_g: bilinear, so a local method can stop at a",
        "local optimum. Solve with --option nonconvex=global; add --option",
        "write_certificate=cert.json and tools/verify_global_certificate.py proves the bound in",
        "exact arithmetic from this file alone.",
        "",
        "Formulation: P (Haverly 1978; Tawarmalani and Sahinidis 2002, ch. 9), written by",
        "bench/runners/pooling_models.py. Quadratic rows use QCMATRIX (full symmetric, no 1/2).",
        "",
    ]
    return pm.write_mps(pm.p_formulation(net), lines)


if __name__ == "__main__":
    power_dispatch().write(HERE / "power_dispatch.mps")
    supply_chain().write(HERE / "supply_chain.mps")
    ill_conditioned().write(HERE / "ill_conditioned.mps")
    lot_sizing().write(HERE / "lot_sizing.mps")
    relaxation = lot_sizing(relaxed=True)
    relaxation.comment = [
        "SANKHYA case study - the LP RELAXATION of lot_sizing.mps, y continuous in [0, 1].",
        "",
        "Emitted as its own file so the weak-relaxation claim is MEASURED rather than",
        "asserted. Solving both and printing the two objectives shows the exact size of the",
        "gap branch and bound had to close by search.",
    ]
    relaxation.write(HERE / "lot_sizing_relaxed.mps")
    with open(HERE / "refinery_pooling.mps", "w", encoding="utf-8", newline=chr(10)) as handle:
        handle.write(refinery_pooling())
    print("wrote data/casestudies/refinery_pooling.mps")
