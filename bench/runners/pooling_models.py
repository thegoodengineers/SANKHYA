#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the P-, Q- and PQ-formulations of the standard pooling instances as QCQP MPS (#516).

INPUT. data/pooling/instances/<name>.json, the network transcribed by fetch_pooling.py from
Alfaki and Haugland's published data file: sources, pools, terminals, arcs with a unit cost,
source qualities, terminal quality upper bounds, node flow bounds.

THE THREE FORMULATIONS (minimize total arc cost; every flow is nonnegative).

  P  (Haverly 1978; the "p-formulation" of Tawarmalani and Sahinidis, *Convexification and
     Global Optimization in Continuous and Mixed-Integer Nonlinear Programming*, Kluwer 2002,
     ch. 9). Columns: one flow f_u_v per arc, and the pool quality p_l_k. Bilinear terms
     p_l_k * f_l_j appear in the pool quality balance
         sum_i lambda_ik f_il  -  sum_j p_lk f_lj  = 0
     and in each terminal's quality specification
         sum_l p_lk f_lj + sum_i lambda_ik f_ij - P_jk (inflow of j) <= 0.

  Q  (Ben-Tal, Eiger and Gershovitz, *Global minimization by reducing the duality gap*, Math.
     Programming 63, 1994). Columns: the proportion q_i_l of pool l's content that came from
     source i (sum_i q_il = 1), the pool-to-terminal and direct flows f, and the path flow
     v_i_l_j = q_il f_lj written as its own column with the bilinear defining row
         v_ilj - q_il f_lj = 0,
     so that every other row is linear in (q, f, v). Source supply, costs and terminal
     qualities are written over the path flows. A McCormick relaxation of q_il*f_lj is the
     same whether the product is named v_ilj or left inline, so this is the Q-formulation's
     relaxation exactly.

  PQ (Quesada and Grossmann, *Global optimization of bilinear process networks with
     multicomponent flows*, Comput. Chem. Eng. 19, 1995; Tawarmalani and Sahinidis 2002).
     Q plus the reformulation-linearization constraints obtained by multiplying
         sum_i q_il = 1            by f_lj:   sum_i v_ilj = f_lj
         sum_j f_lj <= cap_l       by q_il:   sum_j v_ilj <= cap_l q_il.
     Redundant for the nonconvex model, strictly tighter for its relaxation.

Every column carries the bound its network implies (a flow cannot exceed the capacity of
either end, a proportion lies in [0, 1], a pool quality between the extreme source
qualities that can reach it). These are consequences of the rows, stated so a relaxation
over the box is finite; they cut off no feasible point.

FILE FORMAT. Free MPS with the QCMATRIX extension (CPLEX and Gurobi) for quadratic rows:
QCMATRIX <row> lists the full symmetric matrix, with no factor 1/2, so a term c*x*y is the
two entries (x, y) and (y, x) of c/2 each. The row is  a'x + x'Qx  (sense)  rhs.

    python bench/runners/pooling_models.py            # (re)write data/pooling/*.mps
    python bench/runners/pooling_models.py --check    # fail if a committed file differs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "pooling"
INSTANCE_DIR = DATA_DIR / "instances"
FORMULATIONS = ("p", "q", "pq")
INF = math.inf


@dataclass
class Row:
    name: str
    sense: str  # "E", "L" or "G"
    rhs: float
    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)
    range_: float | None = None  # MPS RANGES value: an L row is then rhs - |R| <= . <= rhs

    def add(self, column: str, value: float) -> None:
        if value != 0.0:
            self.linear[column] = self.linear.get(column, 0.0) + value

    def add_product(self, a: str, b: str, value: float) -> None:
        key = (a, b) if a <= b else (b, a)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + value


@dataclass
class QcqpModel:
    name: str
    header: list[str]
    columns: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective: dict[str, float] = field(default_factory=dict)
    rows: list[Row] = field(default_factory=list)

    def column(self, name: str, lower: float, upper: float) -> str:
        self.columns[name] = (lower, upper)
        return name

    def row(self, name: str, sense: str, rhs: float) -> Row:
        self.rows.append(Row(name, sense, rhs))
        return self.rows[-1]


class Network:
    def __init__(self, data: dict) -> None:
        self.name = data["name"]
        self.data = data
        self.sources = list(data["sources"])
        self.pools = list(data["pools"])
        self.terminals = list(data["terminals"])
        self.K = int(data["num_qualities"])
        self.cost = {(u, v): c for u, v, c in data["arcs"]}
        self.quality = {int(n): q for n, q in data["quality"].items()}
        self.lower = {int(n): b for n, b in data["lower"].items()}
        self.upper = {int(n): b for n, b in data["upper"].items()}
        kinds = {**{n: "s" for n in self.sources}, **{n: "p" for n in self.pools},
                 **{n: "t" for n in self.terminals}}
        for u, v in self.cost:
            if (kinds.get(u), kinds.get(v)) not in (("s", "p"), ("s", "t"), ("p", "t")):
                raise ValueError(f"{self.name}: arc ({u},{v}) is not source-pool, "
                                 "source-terminal or pool-terminal; not a standard pooling "
                                 "network")
        for s in self.sources:
            if s not in self.quality:
                raise ValueError(f"{self.name}: source {s} has no quality")

    def arcs_into(self, node: int) -> list[int]:
        return sorted(u for u, v in self.cost if v == node)

    def arcs_out(self, node: int) -> list[int]:
        return sorted(v for u, v in self.cost if u == node)

    def cap(self, node: int) -> float:
        return self.upper.get(node, INF)

    def arc_cap(self, *nodes: int) -> float:
        return min(self.cap(n) for n in nodes)

    def pool_quality_range(self, pool: int, k: int) -> tuple[float, float]:
        values = [self.quality[i][k] for i in self.arcs_into(pool)]
        return min(values), max(values)


def flow(u: int, v: int) -> str:
    return f"f_{u}_{v}"


def _node_rows(model: QcqpModel, net: Network, out_flow: dict[int, dict[str, float]],
               in_flow: dict[int, dict[str, float]]) -> None:
    """Supply at every source and demand at every terminal, each as one row over the
    columns that carry its flow in this formulation."""
    for node, terms, kind in ([(s, out_flow[s], "src") for s in net.sources]
                              + [(t, in_flow[t], "dem") for t in net.terminals]):
        lower, upper = net.lower.get(node, 0.0), net.cap(node)
        if upper == INF and lower <= 0.0:
            continue
        if upper == INF:
            row = model.row(f"{kind}_{node}", "G", lower)
        else:
            row = model.row(f"{kind}_{node}", "L", upper)
            if lower > 0.0:
                row.range_ = upper - lower
        for column, coefficient in terms.items():
            row.add(column, coefficient)


def p_formulation(net: Network) -> QcqpModel:
    model = QcqpModel(f"{net.name}_p", ["P-formulation (Haverly 1978; Tawarmalani and "
                                        "Sahinidis 2002, ch. 9)"])
    for (u, v), cost in sorted(net.cost.items()):
        model.column(flow(u, v), 0.0, net.arc_cap(u, v))
        if cost:
            model.objective[flow(u, v)] = cost
    for pool in net.pools:
        for k in range(net.K):
            model.column(f"p_{pool}_{k + 1}", *net.pool_quality_range(pool, k))
    out_flow = {s: {flow(s, v): 1.0 for v in net.arcs_out(s)} for s in net.sources}
    in_flow = {t: {flow(u, t): 1.0 for u in net.arcs_into(t)} for t in net.terminals}
    _node_rows(model, net, out_flow, in_flow)
    for pool in net.pools:
        row = model.row(f"bal_{pool}", "E", 0.0)
        for i in net.arcs_into(pool):
            row.add(flow(i, pool), 1.0)
        for j in net.arcs_out(pool):
            row.add(flow(pool, j), -1.0)
        if net.cap(pool) < INF:
            row = model.row(f"cap_{pool}", "L", net.cap(pool))
            for i in net.arcs_into(pool):
                row.add(flow(i, pool), 1.0)
        for k in range(net.K):
            row = model.row(f"pq_{pool}_{k + 1}", "E", 0.0)
            for i in net.arcs_into(pool):
                row.add(flow(i, pool), net.quality[i][k])
            for j in net.arcs_out(pool):
                row.add_product(f"p_{pool}_{k + 1}", flow(pool, j), -1.0)
    for t in net.terminals:
        if t not in net.quality:
            continue
        for k in range(net.K):
            spec = net.quality[t][k]
            row = model.row(f"tq_{t}_{k + 1}", "L", 0.0)
            for u in net.arcs_into(t):
                if u in net.pools:
                    row.add_product(f"p_{u}_{k + 1}", flow(u, t), 1.0)
                    row.add(flow(u, t), -spec)
                else:
                    row.add(flow(u, t), net.quality[u][k] - spec)
    return model


def path(i: int, pool: int, j: int) -> str:
    return f"v_{i}_{pool}_{j}"


def q_formulation(net: Network, rlt: bool) -> QcqpModel:
    title = ("PQ-formulation (Quesada and Grossmann 1995; Tawarmalani and Sahinidis 2002)"
             if rlt else "Q-formulation (Ben-Tal, Eiger and Gershovitz 1994), path flows "
             "as columns")
    model = QcqpModel(f"{net.name}_{'pq' if rlt else 'q'}", [title])
    paths = [(i, pool, j) for pool in net.pools for i in net.arcs_into(pool)
             for j in net.arcs_out(pool)]
    for pool in net.pools:
        for i in net.arcs_into(pool):
            model.column(f"q_{i}_{pool}", 0.0, 1.0)
    for (u, v), cost in sorted(net.cost.items()):
        if u in net.sources and v in net.pools:
            continue  # source-to-pool flow is sum_j v_ilj, not a column of its own
        model.column(flow(u, v), 0.0, net.arc_cap(u, v))
        if cost:
            model.objective[flow(u, v)] = cost
    for i, pool, j in paths:
        model.column(path(i, pool, j), 0.0, net.arc_cap(i, pool, j))
        cost = net.cost[(i, pool)]
        if cost:
            model.objective[path(i, pool, j)] = cost
    out_flow = {s: {**{path(s, pool, j): 1.0 for (i, pool, j) in paths if i == s},
                    **{flow(s, v): 1.0 for v in net.arcs_out(s) if v in net.terminals}}
                for s in net.sources}
    in_flow = {t: {flow(u, t): 1.0 for u in net.arcs_into(t)} for t in net.terminals}
    _node_rows(model, net, out_flow, in_flow)
    for pool in net.pools:
        row = model.row(f"sumq_{pool}", "E", 1.0)
        for i in net.arcs_into(pool):
            row.add(f"q_{i}_{pool}", 1.0)
        if net.cap(pool) < INF:
            row = model.row(f"cap_{pool}", "L", net.cap(pool))
            for j in net.arcs_out(pool):
                row.add(flow(pool, j), 1.0)
    for t in net.terminals:
        if t not in net.quality:
            continue
        for k in range(net.K):
            spec = net.quality[t][k]
            row = model.row(f"tq_{t}_{k + 1}", "L", 0.0)
            for u in net.arcs_into(t):
                if u in net.pools:
                    row.add(flow(u, t), -spec)
                    for i in net.arcs_into(u):
                        row.add(path(i, u, t), net.quality[i][k])
                else:
                    row.add(flow(u, t), net.quality[u][k] - spec)
    for i, pool, j in paths:
        row = model.row(f"path_{i}_{pool}_{j}", "E", 0.0)
        row.add(path(i, pool, j), 1.0)
        row.add_product(f"q_{i}_{pool}", flow(pool, j), -1.0)
    if rlt:
        for pool in net.pools:
            for j in net.arcs_out(pool):
                row = model.row(f"rlt_{pool}_{j}", "E", 0.0)
                for i in net.arcs_into(pool):
                    row.add(path(i, pool, j), 1.0)
                row.add(flow(pool, j), -1.0)
            if net.cap(pool) < INF:
                for i in net.arcs_into(pool):
                    row = model.row(f"rltc_{i}_{pool}", "L", 0.0)
                    for j in net.arcs_out(pool):
                        row.add(path(i, pool, j), 1.0)
                    row.add(f"q_{i}_{pool}", -net.cap(pool))
    return model


def build(net: Network, formulation: str) -> QcqpModel:
    if formulation == "p":
        return p_formulation(net)
    return q_formulation(net, rlt=formulation == "pq")


def num(value: float) -> str:
    """Shortest text that reads back as the same double."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def write_mps(model: QcqpModel, preamble: list[str]) -> str:
    lines = [f"* {line}" if line else "*" for line in preamble + model.header]
    lines += [f"NAME {model.name}", "ROWS", " N cost"]
    lines += [f" {row.sense} {row.name}" for row in model.rows]
    lines.append("COLUMNS")
    by_column: dict[str, list[tuple[str, float]]] = {c: [] for c in model.columns}
    if model.objective:
        for column, value in model.objective.items():
            by_column[column].append(("cost", value))
    for row in model.rows:
        for column, value in row.linear.items():
            by_column[column].append((row.name, value))
    for column, entries in by_column.items():
        if not entries:
            # A column in no row and not in the objective still has to be declared.
            entries = [("cost", 0.0)]
        lines += [f" {column} {row} {num(value)}" for row, value in entries]
    lines.append("RHS")
    lines += [f" rhs {row.name} {num(row.rhs)}" for row in model.rows if row.rhs != 0.0]
    ranged = [row for row in model.rows if row.range_ is not None]
    if ranged:
        lines.append("RANGES")
        lines += [f" rng {row.name} {num(row.range_)}" for row in ranged]
    lines.append("BOUNDS")
    for column, (lower, upper) in model.columns.items():
        if lower != 0.0:
            lines.append(f" LO bnd {column} {num(lower)}")
        if upper < INF:
            lines.append(f" UP bnd {column} {num(upper)}")
    for row in model.rows:
        if not row.quadratic:
            continue
        lines.append(f"QCMATRIX {row.name}")
        for (a, b), value in row.quadratic.items():
            if a == b:
                lines.append(f" {a} {a} {num(value)}")
            else:
                lines.append(f" {a} {b} {num(value / 2)}")
                lines.append(f" {b} {a} {num(value / 2)}")
    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def preamble(net: Network, formulation: str) -> list[str]:
    data = net.data
    return [
        f"SANKHYA pooling instance {net.name}, {formulation.upper()}-formulation (#516).",
        "GENERATED by bench/runners/pooling_models.py from "
        f"data/pooling/instances/{net.name}.json; do not edit.",
        f"Data: Alfaki and Haugland, standard pooling instance {data['source_file']}",
        f"(sha256 {data['source_sha256']}).",
        "Source paper and published optimum: data/pooling/reference.json.",
        "Minimize total arc cost. Quadratic rows use QCMATRIX (full symmetric, no 1/2).",
        "",
    ]


def evaluate(model: QcqpModel, point: dict[str, float]) -> tuple[float, float]:
    """(objective, worst violation of a row or a bound) at `point`; for tests."""
    objective = sum(value * point.get(c, 0.0) for c, value in model.objective.items())
    worst = 0.0
    for column, (lower, upper) in model.columns.items():
        x = point.get(column, 0.0)
        worst = max(worst, lower - x, x - upper)
    for row in model.rows:
        activity = sum(v * point.get(c, 0.0) for c, v in row.linear.items())
        activity += sum(v * point.get(a, 0.0) * point.get(b, 0.0)
                        for (a, b), v in row.quadratic.items())
        if row.sense == "E":
            worst = max(worst, abs(activity - row.rhs))
        elif row.sense == "L":
            worst = max(worst, activity - row.rhs)
            if row.range_ is not None:
                worst = max(worst, row.rhs - row.range_ - activity)
        else:
            worst = max(worst, row.rhs - activity)
    return objective, worst


def point_from_flows(net: Network, formulation: str,
                     flows: dict[tuple[int, int], float]) -> dict[str, float]:
    """The columns of a formulation that describe the same arc flows; for tests."""
    arc = {key: flows.get(key, 0.0) for key in net.cost}
    point: dict[str, float] = {}
    for pool in net.pools:
        inflow = sum(arc[(i, pool)] for i in net.arcs_into(pool))
        sources = net.arcs_into(pool)
        shares = {i: (arc[(i, pool)] / inflow if inflow > 0 else 1.0 / len(sources))
                  for i in sources}
        if formulation == "p":
            for k in range(net.K):
                point[f"p_{pool}_{k + 1}"] = sum(shares[i] * net.quality[i][k] for i in sources)
        else:
            for i in sources:
                point[f"q_{i}_{pool}"] = shares[i]
                for j in net.arcs_out(pool):
                    point[path(i, pool, j)] = shares[i] * arc[(pool, j)]
    for (u, v), value in arc.items():
        if formulation == "p" or not (u in net.sources and v in net.pools):
            point[flow(u, v)] = value
    return point


def load(name: str) -> Network:
    return Network(json.loads((INSTANCE_DIR / f"{name}.json").read_text(encoding="utf-8")))


def instance_names() -> list[str]:
    return sorted(path.stem for path in INSTANCE_DIR.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed files instead of writing them")
    args = parser.parse_args()
    reference_path = DATA_DIR / "reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    differs = 0
    for name in instance_names():
        net = load(name)
        entry = reference["instances"].setdefault(name, {})
        models = {}
        for formulation in FORMULATIONS:
            text = write_mps(build(net, formulation), preamble(net, formulation))
            target = DATA_DIR / f"{name}_{formulation}.mps"
            digest = hashlib.sha256(text.encode("ascii")).hexdigest()
            models[formulation] = {"file": target.name, "sha256": digest}
            if args.check:
                same = target.exists() and target.read_bytes().replace(b"\r\n", b"\n") \
                    == text.encode("ascii")
                if not same:
                    differs += 1
                    print(f"{target.name}: DIFFERS from the generator's output")
            else:
                target.write_bytes(text.encode("ascii"))
        if args.check and entry.get("models") != models:
            differs += 1
            print(f"{name}: reference.json's model hashes are stale")
        entry["models"] = models
    if args.check:
        print("all pooling models match their generator" if not differs
              else f"{differs} mismatches")
        return 1 if differs else 0
    reference_path.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8",
                              newline="\n")
    print(f"wrote {3 * len(instance_names())} models and {reference_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
