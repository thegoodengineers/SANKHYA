#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the refinery suite's generators (#1004). Hermetic: no solver runs.

*   Every model generates at small size, deterministically (the same bytes twice).
*   Every file is read back by the verifier's independent MPS reader with the dimensions
    the generator states, integer and semi-continuous columns included.
*   The KKT construction re-verifies, and rejects a plan moved off its optimum.
*   The rational simplex agrees with a hand-solved LP, reports infeasibility, and agrees with
    the KKT construction's optimum on a constructed LP.
*   The Wagner-Whitin recursion agrees with brute-force enumeration of order patterns.
*   The doc section renders from a two-row CSV.

    python bench/refinery/test_refinery.py
"""
from __future__ import annotations

import csv
import itertools
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "bench" / "runners"))
import blending  # noqa: E402
import generate  # noqa: E402
import planning  # noqa: E402
from exact_lp import exact_optimum  # noqa: E402
from kit import F, Lp  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


print("every model at small size: generates, deterministic, reads back")
with tempfile.TemporaryDirectory() as tmp:
    for name in generate.MODELS:
        first = generate.generate(name, "small", Path(tmp))
        text = (Path(tmp) / f"{name}_small.mps").read_text(encoding="utf-8")
        second = generate.generate(name, "small", Path(tmp))
        check(first["sha256"] == second["sha256"], f"{name}: deterministic")
        model = parse_mps(Path(tmp) / f"{name}_small.mps")
        check(model.num_rows == first["rows"] and model.num_cols == first["columns"],
              f"{name}: dimensions read back", f"{model.num_rows} x {model.num_cols}")
        integer = sum(model.col_integer)
        check(integer == first["integer"], f"{name}: integer columns", str(integer))
        check(first["kind"] == ("mip" if first["integer"] or first["semicontinuous"]
                                else "lp"), f"{name}: kind")
        if first["reference_source"] != "cross-check":
            check("reference_exact" in first and f"exact optimum: " in text,
                  f"{name}: states its exact optimum", first.get("reference_exact", ""))
        check("source: " in text, f"{name}: cites its source")
    sc = parse_mps(Path(tmp) / "unit_modes_small.mps")
    check(len(sc.semicontinuous) == 36, "unit_modes: semi-continuous columns read back",
          str(len(sc.semicontinuous)))

print("the KKT construction")
lp, value, _, _ = blending.crude_blend("small", 3)
lp.verify_kkt()
check(True, "crude_blend small re-verifies", str(float(value)))
moved = blending.crude_blend("small", 3)[0]
j = next(j for j in range(len(moved.cols)) if moved.x[j] > 0)
moved.x[j] -= 1
try:
    moved.verify_kkt()
    check(False, "a plan moved off the optimum is rejected")
except AssertionError as error:
    check(True, "a plan moved off the optimum is rejected", str(error)[:60])

print("the rational simplex")
tiny = Lp("TINY")
a, b = tiny.col("A", cost=-3), tiny.col("B", cost=-2, up=4)
tiny.row("R1", "L", {a: 1, b: 1}, 6)
tiny.row("R2", "L", {a: 2, b: 1}, 9)
check(exact_optimum(tiny) == F(-15), "hand-solved LP: -15 at (3, 3)", str(exact_optimum(tiny)))
check(exact_optimum(tiny, {a: F(1)}) == F(-11), "fixed column", str(exact_optimum(tiny, {a: F(1)})))
tiny.row("R3", "G", {a: 1}, 10)
check(exact_optimum(tiny) is None, "infeasible LP reported as None")
lp, value, _, _ = blending.distribution("small", 5)
check(exact_optimum(lp) == value, "agrees with the KKT construction on distribution small",
      f"{float(value)}")
lp, value, _, _ = planning.two_stage("small", 5)
check(exact_optimum(lp) == value, "agrees with the KKT construction on two_stage small",
      f"{float(value)}")

print("Wagner-Whitin against enumeration")
demand = [F(d) for d in (20, 0, 35, 10, 50)]
fixed = [F(k) for k in (100, 120, 90, 110, 130)]
unit = [F(c) for c in (10, 12, 11, 9, 13)]
hold = F(2)
best = None
for pattern in itertools.product((0, 1), repeat=len(demand)):
    if not pattern[0]:
        continue
    cost, stock = F(0), F(0)
    orders = [t for t in range(len(demand)) if pattern[t]]
    for n, t in enumerate(orders):
        end = orders[n + 1] if n + 1 < len(orders) else len(demand)
        lot = sum(demand[t:end], F(0))
        cost += fixed[t] + unit[t] * lot
        stock = lot
        for u in range(t, end):
            stock -= demand[u]
            cost += hold * stock
    best = cost if best is None or cost < best else best
check(planning.wagner_whitin(demand, fixed, unit, hold) == best, "recursion = enumeration",
      str(best))

print("the doc section")
import refinery_doc  # noqa: E402
# The section draws docs/img/profile-refinery.svg; a test must not rewrite a tracked figure.
refinery_doc.perf_profile.write_figure = lambda *args, **kwargs: None
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "head-to-head-refinery-abc1234.csv"
    import compare_suite  # noqa: E402
    base = {k: "" for k in compare_suite.CSV_COLUMNS}
    rows = [{**base, "instance": "crude_blend_small", "solver": "sankhya", "status": "optimal",
             "point_objective": "12.129", "relative_gap": "0.0", "matches_reference": "1",
             "independently_verified": "1", "verification": "primal+dual",
             "counted_for_time": "1", "solver_seconds": "0.01", "time_limit": "600",
             "git_commit": "abc1234", "machine": "laptop", "solver_version": "abc1234"},
            {**base, "instance": "crude_blend_small", "solver": "highs", "status": "optimal",
             "point_objective": "12.129", "relative_gap": "0.0", "matches_reference": "1",
             "independently_verified": "1", "verification": "primal+dual",
             "counted_for_time": "1", "solver_seconds": "0.02", "time_limit": "600",
             "git_commit": "abc1234", "machine": "laptop", "solver_version": "1.15.1"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compare_suite.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sgm = lambda v, s: math.exp(sum(math.log(x + s) for x in v) / len(v)) - s  # noqa: E731
    text = refinery_doc.section(path, sgm)
    check("| SANKHYA |" in text and "| HiGHS |" in text, "solver table rendered")
    check("`crude_blend_small`" in text and "verified" in text, "per-instance table rendered")
    check("shared laptop" in text, "the laptop caveat is printed")
    check("No `head-to-head-refinery-*.csv`" in refinery_doc.section(None, sgm),
          "no CSV: no numbers")


print("\nthe convex-cost refinery QP (refinery_qp.py)")
import refinery_qp  # noqa: E402

inst, q = refinery_qp.build(3)
refinery_qp.verify_qp(inst, q)
check(all(v > 0 for v, name in zip(q, inst.cols) if name.startswith(("BUY_", "RUN_")))
      and all(v == 0 for v, name in zip(q, inst.cols)
              if not name.startswith(("BUY_", "RUN_"))),
      "curvature on every purchase and run column and nowhere else")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "refinery_qp_t3.qps"
    value = refinery_qp.write_qps(inst, q, path, "t3", 3, refinery_qp.QP_SEED)
    again = Path(tmp) / "again.qps"
    inst2, q2 = refinery_qp.build(3)
    refinery_qp.write_qps(inst2, q2, again, "t3", 3, refinery_qp.QP_SEED)
    check(path.read_bytes() == again.read_bytes(), "the QP is deterministic in (periods, seed)")
    model = parse_mps(path)
    check(model.num_cols == len(inst.cols) and model.num_rows == len(inst.rows),
          "the verifier's reader sees the generator's dimensions")
    x = [float(v) for v in inst.x]
    stated = sum(c * v for c, v in zip(model.col_cost, x)) + model.quadratic_objective(x)
    check(abs(stated - float(value)) <= 1e-9 * max(1.0, abs(float(value))),
          "c'x* + x*'Qx*/2 read back from the file is the stated optimum",
          f"{stated} vs {float(value)}")
# A price moved off the derived one breaks dual feasibility at an interior column.
j = next(k for k, name in enumerate(inst.cols)
         if name.startswith("RUN_") and inst.lower[k] < inst.x[k]
         and (inst.upper[k] is None or inst.x[k] < inst.upper[k]))
inst.cost[j] += 1
try:
    refinery_qp.verify_qp(inst, q)
    check(False, "a moved price is rejected")
except AssertionError:
    check(True, "a moved price is rejected")

print(f"\n{FAILURES} failure(s)")
sys.exit(1 if FAILURES else 0)
