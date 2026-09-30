#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a sparse transportation LP with a KNOWN optimum, as free-format MPS (#751).

The third family of the million-row runs. The random family of generate_large_lp.py is an
expander graph, so any factorization fills; the refinery and staircase families are banded
in time. A transportation model is the other shape a planner meets at this size - plants
shipping to depots - and when every depot is served by a few NEARBY plants its normal
equations stay sparse: two plants are coupled only when they share a depot, and two depots
only when they share a plant, so A D A^T is a band whose width is the delivery window.

THE MODEL (Dantzig 1951; Hitchcock 1941). Sources i = 0..S-1, sinks j = 0..D-1, and an arc
(i, j) only when source i lies in sink j's window of `--window` sources starting at j * S / D.
Each sink samples `--arcs` distinct sources from its window. Columns are the shipments
x[i,j] >= 0; rows are

    supply   sum_j x[i,j] <= supply_i       (L, one per source)
    demand   sum_i x[i,j]  = demand_j       (E, one per sink)

THE OPTIMUM IS EXACT BY CONSTRUCTION, as in generate_large_lp.py and generate_refinery_lp.py
(Bertsimas and Tsitsiklis 1997, section 4.2 for the conditions): choose the shipping plan x*
first, derive supply and demand from it with about half the supplies tight, choose the
multipliers - u_i <= 0 on a tight supply row and 0 on a slack one, v_j free on a demand row -
and DERIVE the costs from dual feasibility:

    c[i,j] = r[i,j] + u_i + v_j,   r[i,j] = 0 where x*[i,j] > 0, r[i,j] >= 0 where it is 0.

v_j is drawn above the largest |u_i| so every cost is positive, as a freight rate is. Every
number is an integer, so c^T x* is the optimum exactly, and the script checks every KKT
condition in integers before it writes anything.

Rows = S + D; columns = D * arcs; nonzeros = 2 * columns.

Usage:
    python bench/runners/generate_transport_lp.py --sources 500000 --sinks 500000 \\
        --seed 7 --out transport-1m.mps
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PLAN_RANGE = 20          # a shipment x*[i,j] is drawn from [1, this] when it is used
USED_PROBABILITY = 0.6   # fraction of arcs that carry flow at x*
SLACK_RANGE = 9          # a slack supply row has this much [1, this] left over
MULTIPLIER_RANGE = 9     # |u_i| and r[i,j] drawn from [1, this] / [0, this]
TIGHT_PROBABILITY = 0.5  # fraction of supply rows tight at x*


def build(sources: int, sinks: int, arcs: int, window: int, seed: int) -> dict:
    """The instance as flat integer arrays; arcs are grouped by sink, in source order."""
    rng = random.Random(seed)
    arcs = min(arcs, window, sources)
    window = min(window, sources)
    arc_source: list[int] = []
    arc_flow: list[int] = []
    sink_start = [0] * (sinks + 1)
    for j in range(sinks):
        first = min(j * sources // sinks, sources - window)
        for i in sorted(rng.sample(range(first, first + window), arcs)):
            arc_source.append(i)
            arc_flow.append(rng.randint(1, PLAN_RANGE) if rng.random() < USED_PROBABILITY
                            else 0)
        sink_start[j + 1] = len(arc_source)

    shipped = [0] * sources
    for i, x in zip(arc_source, arc_flow):
        shipped[i] += x
    demand = [sum(arc_flow[sink_start[j]:sink_start[j + 1]]) for j in range(sinks)]
    tight = [rng.random() < TIGHT_PROBABILITY for _ in range(sources)]
    supply = [shipped[i] + (0 if tight[i] else rng.randint(1, SLACK_RANGE))
              for i in range(sources)]
    u = [-rng.randint(1, MULTIPLIER_RANGE) if tight[i] else 0 for i in range(sources)]
    v = [rng.randint(MULTIPLIER_RANGE + 1, 3 * MULTIPLIER_RANGE) for _ in range(sinks)]
    cost = [0] * len(arc_source)
    for j in range(sinks):
        for a in range(sink_start[j], sink_start[j + 1]):
            r = 0 if arc_flow[a] > 0 else rng.randint(0, MULTIPLIER_RANGE)
            cost[a] = r + u[arc_source[a]] + v[j]
    return {"sources": sources, "sinks": sinks, "arc_source": arc_source,
            "arc_flow": arc_flow, "sink_start": sink_start, "supply": supply,
            "demand": demand, "u": u, "v": v, "cost": cost}


def verify(inst: dict) -> int:
    """Every KKT condition, in integers. Returns the optimum c^T x*."""
    sources, sinks = inst["sources"], inst["sinks"]
    arc_source, flow, start = inst["arc_source"], inst["arc_flow"], inst["sink_start"]
    u, v, cost = inst["u"], inst["v"], inst["cost"]
    shipped = [0] * sources
    for j in range(sinks):
        received = 0
        for a in range(start[j], start[j + 1]):
            i = arc_source[a]
            assert flow[a] >= 0, f"arc {a}: negative shipment"
            shipped[i] += flow[a]
            received += flow[a]
            r = cost[a] - u[i] - v[j]
            assert r >= 0, f"arc {a}: negative reduced cost {r}"
            assert flow[a] == 0 or r == 0, f"arc {a}: basic arc with reduced cost {r}"
            assert cost[a] > 0, f"arc {a}: non-positive cost"
        assert received == inst["demand"][j], f"sink {j}: demand not met"
    for i in range(sources):
        assert shipped[i] <= inst["supply"][i], f"source {i}: supply exceeded"
        assert u[i] <= 0, f"source {i}: L row needs a non-positive multiplier"
        assert u[i] == 0 or shipped[i] == inst["supply"][i], f"source {i}: slack, priced"
    return sum(c * x for c, x in zip(cost, flow))


def write_mps(inst: dict, optimum: int, out: Path, seed: int) -> None:
    sources, sinks = inst["sources"], inst["sinks"]
    arc_source, start, cost = inst["arc_source"], inst["sink_start"], inst["cost"]
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"NAME          TRANSPORT_S{sources}_D{sinks}_S{seed}\n")
        f.write("* generator: bench/runners/generate_transport_lp.py (#751)\n")
        f.write(f"* structure: transport, {sources} sources, {sinks} sinks\n")
        f.write(f"* analytic optimum: {optimum}\n")
        f.write("ROWS\n N  COST\n")
        f.writelines(f" L  S{i}\n" for i in range(sources))
        f.writelines(f" E  D{j}\n" for j in range(sinks))
        f.write("COLUMNS\n")
        for j in range(sinks):
            for a in range(start[j], start[j + 1]):
                name = f"X{arc_source[a]}_{j}"
                f.write(f"    {name}  COST  {cost[a]}  S{arc_source[a]}  1\n")
                f.write(f"    {name}  D{j}  1\n")
        f.write("RHS\n")
        f.writelines(f"    RHS  S{i}  {s}\n" for i, s in enumerate(inst["supply"]) if s)
        f.writelines(f"    RHS  D{j}  {d}\n" for j, d in enumerate(inst["demand"]) if d)
        f.write("ENDATA\n")
    columns = len(arc_source)
    print(f"wrote {out}")
    print(f"  rows: {sources + sinks}  columns: {columns}  nonzeros: {2 * columns}")
    print(f"  analytic optimum: {optimum}")
    print(f"  structure: transport, {sources} sources, {sinks} sinks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=int, required=True)
    parser.add_argument("--sinks", type=int, required=True)
    parser.add_argument("--arcs", type=int, default=4, help="arcs into each sink")
    parser.add_argument("--window", type=int, default=8,
                        help="the sources a sink may be served from, a contiguous run; the "
                             "normal equations' bandwidth")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.sources < 1 or args.sinks < 1 or args.arcs < 1 or args.window < 1:
        parser.error("--sources, --sinks, --arcs and --window must be positive")
    inst = build(args.sources, args.sinks, args.arcs, args.window, args.seed)
    optimum = verify(inst)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_mps(inst, optimum, args.out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
