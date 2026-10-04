#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a global-optimality certificate of the spatial branch and bound, exactly (#514).

Usage:
    python tools/verify_global_certificate.py model.mps certificate.json [--feas-tol T]

The model is a QCQP in MPS (QCMATRIX rows, QUADOBJ objective); the certificate is what
`sankhya solve model.mps --option nonconvex=global --option write_certificate=cert.json` writes.
This checker never links the solver and does not read the model with the solver's reader: the
model comes from tools/verify_solution_mps.py, the independent reader verify_solution.py checks
every answer against, and every number is an exact rational (fractions.Fraction).

WHAT IS CHECKED. The certificate is a tree of boxes and, per node, row multipliers. Nothing in it
is a claim the checker accepts:

  * THE LEAVES COVER THE ROOT BOX. Boxes are not in the file; the checker starts from the model's
    own column bounds and splits at each written point, so a missing child, a node with one child
    or a leaf with children is rejected, and nothing else can be uncovered.
  * EVERY BOUND IS RECOMPUTED. For a node, the McCormick relaxation of the products over THAT box
    is rebuilt here from the original rows (products replaced by one column each, then four
    envelope rows per bilinear product, or a secant and three tangents per square), and the
    Lagrangian at the written multipliers is evaluated:

        sum_i y_i (lower_i if y_i > 0, upper_i if y_i < 0) + sum_j min over the box of (c - A'y)_j z_j

    which is a lower bound on the relaxation, and so on the node, for ANY y (weak duality;
    Neumaier and Shcherbina, Math. Programming 99, 2004). A tampered multiplier can only weaken
    a bound. A multiplier that leans on a row side the row does not have is dropped, as the
    solver drops it. A node's proven bound is the larger of its parent's and its own.
  * A FARKAS NODE is empty only if the same expression with no cost, at y or -y, is positive.
  * THE CLAIM. The global lower bound is the least proven bound over the leaves. The certificate's
    claimed bound may not exceed it (to rounding), and for an `optimal` claim the incumbent -
    checked here against the original model, products evaluated exactly - must be within the
    claimed gap of it.

The relaxation's row layout is the one stated in the certificate's own "layout" field. That, the
products' numbering, and nothing else is shared with the solver; the products themselves are
re-derived from the model, and a product the model has that the certificate does not list is
rejected (it would let a relaxation drop a term).

Exit status: 0 the bound is proved and the claim holds; 1 the certificate is rejected; 2 every
step checks but the claim of optimality is not shown (the gap is wider than claimed, or an
`optimal` claim has no incumbent).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_solution_mps import parse_mps  # noqa: E402  (the independent reader)

ZERO = Fraction(0)
INF = math.inf


class Rejected(Exception):
    pass


def number(value):
    """A JSON number or one of the strings the writer uses for a non-finite one."""
    if isinstance(value, str):
        return {"inf": INF, "-inf": -INF, "nan": math.nan}[value]
    return float(value)


def exact(value: float):
    """The exact rational of a double; an infinity stays one."""
    return value if math.isinf(value) else Fraction(value)


class Problem:
    """The model as the checker reads it, in minimisation form, with its products named."""

    def __init__(self, path: Path, products: list[tuple[int, int]]):
        model = parse_mps(path)
        if any(model.col_integer):
            raise Rejected("the model has integer columns; the global method does not take them")
        self.n = model.num_cols
        self.m = model.num_rows
        self.sigma = Fraction(-1) if model.maximize else Fraction(1)
        self.col_lower = [exact(v) for v in model.col_lower]
        self.col_upper = [exact(v) for v in model.col_upper]
        self.row_lower = [exact(v) for v in model.row_lower]
        self.row_upper = [exact(v) for v in model.row_upper]
        self.linear: list[list[tuple[int, Fraction]]] = [[] for _ in range(self.m)]
        for j, entries in enumerate(model.entries):
            for row, value in entries:
                self.linear[row].append((j, Fraction(value)))
        self.products = [(min(a, b), max(a, b)) for a, b in products]
        self.index = {key: p for p, key in enumerate(self.products)}
        if len(self.index) != len(self.products):
            raise Rejected("the certificate lists a product twice")
        for a, b in self.products:
            if not (0 <= a <= b < self.n):
                raise Rejected(f"product ({a}, {b}) is outside the model's columns")
        # Quadratic rows: the row is a'x + sum of value * x_i * x_j over the listed entries.
        self.row_products: list[dict[int, Fraction]] = [{} for _ in range(self.m)]
        for (row, a, b), value in model.qc_entries.items():
            key = (min(a, b), max(a, b))
            if key not in self.index:
                raise Rejected(f"row {row} has the product x{key[0]} * x{key[1]}, which the "
                               "certificate does not list")
            p = self.index[key]
            self.row_products[row][p] = self.row_products[row].get(p, ZERO) + Fraction(value)
        # Objective: 0.5 x'Qx over the stored lower triangle, so an off-diagonal entry is the
        # product's whole coefficient and a diagonal one keeps its half.
        self.cost = [self.sigma * Fraction(c) for c in model.col_cost]
        self.offset = self.sigma * Fraction(model.objective_offset)
        self.objective_products: dict[int, Fraction] = {}
        for (r, c), value in model.hessian.items():
            key = (min(r, c), max(r, c))
            if key not in self.index:
                raise Rejected(f"the objective has the product x{key[0]} * x{key[1]}, which the "
                               "certificate does not list")
            weight = Fraction(1, 2) if r == c else Fraction(1)
            p = self.index[key]
            self.objective_products[p] = (self.objective_products.get(p, ZERO)
                                          + self.sigma * weight * Fraction(value))
        self.rows_total = self.m + 4 * len(self.products)

    # ---- the McCormick relaxation over a box, rebuilt from the definition --------------------

    def column_bounds(self, lo: list, hi: list):
        """Bounds of the x columns (the box) and of every product column."""
        lower, upper = list(lo), list(hi)
        for a, b in self.products:
            la, ua, lb, ub = lo[a], hi[a], lo[b], hi[b]
            if a == b:
                if la >= 0:
                    r = (la * la, ua * ua)
                elif ua <= 0:
                    r = (ua * ua, la * la)
                else:
                    r = (ZERO, max(la * la, ua * ua))
            else:
                corners = [la * lb, la * ub, ua * lb, ua * ub]
                r = (min(corners), max(corners))
            lower.append(r[0])
            upper.append(r[1])
        return lower, upper

    def row(self, r: int, lo: list, hi: list):
        """(entries {column: coefficient}, row lower, row upper) of relaxation row r."""
        if not 0 <= r < self.rows_total:
            raise Rejected(f"a multiplier names row {r}, outside the {self.rows_total} rows")
        if r < self.m:
            entries = {j: v for j, v in self.linear[r]}
            for p, coefficient in self.row_products[r].items():
                entries[self.n + p] = entries.get(self.n + p, ZERO) + coefficient
            return entries, self.row_lower[r], self.row_upper[r]
        p, k = divmod(r - self.m, 4)
        a, b = self.products[p]
        w = self.n + p
        la, ua, lb, ub = lo[a], hi[a], lo[b], hi[b]
        if a != b:
            table = [
                ({w: Fraction(1), a: -lb, b: -la}, -la * lb, INF),
                ({w: Fraction(1), a: -ub, b: -ua}, -ua * ub, INF),
                ({w: Fraction(1), a: -ub, b: -la}, -INF, -la * ub),
                ({w: Fraction(1), a: -lb, b: -ua}, -INF, -ua * lb),
            ]
            return table[k]
        if k == 0:
            return {w: Fraction(1), a: -(la + ua)}, -INF, -la * ua
        t = (la, ua, (la + ua) / 2)[k - 1]
        return {w: Fraction(1), a: -2 * t}, -t * t, INF

    def lagrangian(self, y: dict[int, Fraction], lo: list, hi: list, with_cost: bool):
        """The Lagrangian at y, or None when it needs a bound that is infinite."""
        col_lower, col_upper = self.column_bounds(lo, hi)
        d: dict[int, Fraction] = {}
        total = ZERO
        if with_cost:
            total = self.offset
            for j, c in enumerate(self.cost):
                if c != 0:
                    d[j] = c
            for p, c in self.objective_products.items():
                d[self.n + p] = d.get(self.n + p, ZERO) + c
        for r, yr in y.items():
            if yr == 0:
                continue
            entries, rl, ru = self.row(r, lo, hi)
            # A multiplier that leans on a side the row does not have is dropped (it costs a
            # weak-duality bound no validity), as the solver does.
            if (yr > 0 and rl == -INF) or (yr < 0 and ru == INF):
                continue
            total += yr * (rl if yr > 0 else ru)
            for column, a in entries.items():
                d[column] = d.get(column, ZERO) - a * yr
        for column, dj in d.items():
            if dj > 0:
                if col_lower[column] == -INF:
                    return None
                total += dj * col_lower[column]
            elif dj < 0:
                if col_upper[column] == INF:
                    return None
                total += dj * col_upper[column]
        return total


def read_certificate(path: Path) -> dict:
    try:
        cert = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as why:
        raise Rejected(f"cannot read the certificate: {why}")
    if cert.get("format") != "sankhya-global-certificate-1":
        raise Rejected(f"unknown certificate format {cert.get('format')!r}")
    return cert


def verify(problem_path: Path, cert: dict, feas_tol: float) -> dict:
    problem = Problem(problem_path, [tuple(p) for p in cert["products"]])
    if cert["columns"] != problem.n or cert["rows"] != problem.m:
        raise Rejected(f"the certificate is for a {cert['rows']} x {cert['columns']} model; "
                       f"this one is {problem.m} x {problem.n}")
    if Fraction(number(cert["sigma"])) != problem.sigma:
        raise Rejected("the certificate's objective sense is not the model's")

    nodes = {}
    for node in cert["nodes"]:
        if node["id"] in nodes:
            raise Rejected(f"node {node['id']} is listed twice")
        nodes[node["id"]] = node
    roots = [node for node in nodes.values() if node["parent"] == -1]
    if len(roots) != 1 or roots[0]["side"] != "root":
        raise Rejected(f"the tree has {len(roots)} roots")
    children: dict[int, dict[str, int]] = {}
    for node in nodes.values():
        if node["parent"] == -1:
            continue
        if node["parent"] not in nodes:
            raise Rejected(f"node {node['id']} has no parent {node['parent']}")
        slot = children.setdefault(node["parent"], {})
        if node["side"] in slot or node["side"] not in ("low", "high"):
            raise Rejected(f"node {node['parent']} has a bad or repeated child side")
        slot[node["side"]] = node["id"]

    # Walk the tree from the model's own box, splitting where the certificate says.
    root = roots[0]
    stack = [(root["id"], list(problem.col_lower), list(problem.col_upper), -INF)]
    seen = 0
    leaves = 0
    farkas_nodes = 0
    dual_nodes = 0
    global_bound = INF
    while stack:
        node_id, lo, hi, inherited = stack.pop()
        node = nodes[node_id]
        seen += 1
        proven = inherited
        if node.get("proof") in ("dual", "farkas"):
            y = {int(r): Fraction(number(v)) for r, v in node["y"]}
            if node["proof"] == "dual":
                value = problem.lagrangian(y, lo, hi, with_cost=True)
                if value is not None:
                    dual_nodes += 1
                    proven = max(proven, value)
            else:
                negated = {r: -v for r, v in y.items()}
                for candidate in (y, negated):
                    value = problem.lagrangian(candidate, lo, hi, with_cost=False)
                    if value is not None and value > 0:
                        proven = INF
                        farkas_nodes += 1
                        break
                else:
                    raise Rejected(f"node {node_id}: the Farkas vector does not prove its "
                                   "relaxation empty")
        branch = node.get("branch")
        kids = children.get(node_id, {})
        if branch is None:
            if kids:
                raise Rejected(f"node {node_id} has children but no branch")
            leaves += 1
            global_bound = min(global_bound, proven)
            continue
        column, point = int(branch[0]), exact(number(branch[1]))
        if not 0 <= column < problem.n or math.isinf(point):
            raise Rejected(f"node {node_id} branches on a bad column or point")
        if not lo[column] <= point <= hi[column]:
            raise Rejected(f"node {node_id} branches outside its box")
        if set(kids) != {"low", "high"}:
            raise Rejected(f"node {node_id} is split but does not have both children")
        below_hi = list(hi)
        below_hi[column] = point
        above_lo = list(lo)
        above_lo[column] = point
        stack.append((kids["low"], list(lo), below_hi, proven))
        stack.append((kids["high"], above_lo, list(hi), proven))
    if seen != len(nodes):
        raise Rejected(f"{len(nodes) - seen} nodes are not reachable from the root")

    result = {"nodes": len(nodes), "leaves": leaves, "dual_nodes": dual_nodes,
              "farkas_nodes": farkas_nodes, "bound": global_bound}

    # The incumbent, against the original model, products and all.
    incumbent = cert.get("incumbent")
    objective = None
    if incumbent is not None:
        x = [Fraction(number(v)) for v in incumbent]
        if len(x) != problem.n:
            raise Rejected("the incumbent has the wrong number of columns")
        worst = 0.0
        for j, v in enumerate(x):
            scale = max(1, abs(v))
            if not math.isinf(problem.col_lower[j]):
                worst = max(worst, float((problem.col_lower[j] - v) / scale))
            if not math.isinf(problem.col_upper[j]):
                worst = max(worst, float((v - problem.col_upper[j]) / scale))
        for i in range(problem.m):
            activity = ZERO
            scale = Fraction(1)
            terms = [v * x[j] for j, v in problem.linear[i]]
            terms += [c * x[problem.products[p][0]] * x[problem.products[p][1]]
                      for p, c in problem.row_products[i].items()]
            for t in terms:
                activity += t
                scale = max(scale, abs(t))
            if not math.isinf(problem.row_lower[i]):
                worst = max(worst, float((problem.row_lower[i] - activity) / scale))
            if not math.isinf(problem.row_upper[i]):
                worst = max(worst, float((activity - problem.row_upper[i]) / scale))
        if worst > feas_tol:
            raise Rejected(f"the incumbent violates the model by {worst:.3e} (tolerance "
                           f"{feas_tol:.1e})")
        objective = problem.offset + sum(c * x[j] for j, c in enumerate(problem.cost))
        for p, c in problem.objective_products.items():
            objective += c * x[problem.products[p][0]] * x[problem.products[p][1]]
        result["incumbent_violation"] = worst
        result["incumbent_objective"] = objective

    # The certificate may not claim a better bound than its own tree proves (to rounding).
    claimed = number(cert["lower_bound_min_form"])
    if global_bound == -INF:
        if claimed > -INF:
            raise Rejected(f"the certificate claims the bound {claimed:.12g} but the tree "
                           "proves none")
    elif global_bound != INF:
        slack = Fraction(1, 10**9) * max(1, abs(global_bound))
        if claimed > global_bound + slack:
            raise Rejected(f"the certificate claims the bound {claimed:.12g} but the tree "
                           f"proves only {float(global_bound):.12g}")
    result["claimed"] = claimed
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--feas-tol", type=float, default=1e-6,
                        help="the largest relative violation of the model the incumbent may "
                             "have (default 1e-6); the bound side is always exact")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        cert = read_certificate(args.certificate)
        verdict = verify(args.model, cert, args.feas_tol)
    except Rejected as why:
        print(f"REJECTED: {why}")
        return 1
    sigma = Fraction(number(cert["sigma"]))
    bound = verdict["bound"]
    print(f"nodes {verdict['nodes']}, leaves {verdict['leaves']}, bounds from "
          f"{verdict['dual_nodes']} dual and {verdict['farkas_nodes']} Farkas nodes")
    shown = "infinite (every leaf is proved empty)" if bound == INF else f"{float(sigma * bound):.12g}"
    print(f"proved global bound (model sense): {shown}")
    status = cert["status"]
    objective = verdict.get("incumbent_objective")
    if objective is not None:
        print(f"incumbent objective (model sense): {float(sigma * objective):.12g}, "
              f"violation {verdict['incumbent_violation']:.2e}")
    if status == "optimal":
        if objective is None or bound == -INF:
            print("NOT SHOWN: the certificate claims optimality but has no incumbent or no bound")
            return 2
        gap = float(objective - bound)
        relative = gap / max(1.0, abs(float(objective)))
        print(f"gap {gap:.3e} absolute, {relative:.3e} relative (claimed within "
              f"{number(cert['absolute_gap']):.1e} / {number(cert['relative_gap']):.1e})")
        slack = 1e-9
        if gap <= number(cert["absolute_gap"]) + slack or \
                relative <= number(cert["relative_gap"]) + slack:
            print(f"VERIFIED: global optimum within the claimed gap ({time.perf_counter() - started:.2f} s)")
            return 0
        print("NOT SHOWN: the proved gap is wider than the certificate claims")
        return 2
    if status == "infeasible":
        if bound == INF and objective is None:
            print("VERIFIED: every leaf is proved empty, so the model has no feasible point")
            return 0
        print("NOT SHOWN: the certificate claims infeasibility but a leaf is not proved empty")
        return 2
    print(f"VERIFIED: the bound is proved ({status}); no optimality is claimed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
