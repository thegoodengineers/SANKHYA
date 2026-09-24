#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the multi-seed MIPLIB harness (#504): the MPS permutation leaves the model
unchanged, the per-seed numbers and their aggregation are what the docstrings say, and the
tier-2 selection rule is the rule. Pure Python, no network, no solver.

    python bench/runners/test_miplib_seeds.py
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import fetch_miplib  # noqa: E402
import miplib_seeds  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402  (the independent reader)

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# Two N rows (the first is the objective, the second must be dropped wherever it lands), an
# objective constant, a range, integer blocks separated by a continuous column, and bounds of
# several kinds, so every section the permutation copies or reorders is exercised.
MPS = """\
NAME          SEEDTEST
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
 N  FREE
 L  R4
 G  R5
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         1.0   R1           2.0
    X1        R3           1.0   FREE         9.0
    X2        COST        -3.0   R2           1.5
    X2        R4           1.0
    MARKER                 'MARKER'                 'INTEND'
    Y1        COST         0.5   R1           1.0
    Y1        R5           2.0
    MARKER                 'MARKER'                 'INTORG'
    X3        R2           1.0   R3          -1.0
    X3        R5           1.0
    X4        COST         2.0   R4           3.0
    MARKER                 'MARKER'                 'INTEND'
    Y2        COST        -1.0   R1           1.0
    Y2        R2           1.0   R5          -1.0
RHS
    RHS       COST        -4.0   R1          10.0
    RHS       R2           1.0   R3           2.0
    RHS       R4           8.0   R5           0.5
RANGES
    RNG       R1           4.0   R3           1.0
BOUNDS
 UP BND       X1           5.0
 LI BND       X2           1.0
 UI BND       X2           7.0
 BV BND       X3
 UP BND       X4           3.0
 MI BND       Y1
 UP BND       Y1           6.0
 FR BND       Y2
ENDATA
"""


def canonical(path: Path) -> dict:
    """The model as the independent reader sees it, keyed by name so order is irrelevant."""
    model = parse_mps(path)
    columns = {}
    for j, name in enumerate(model.col_names):
        entries = sorted((model.row_names[i], v) for i, v in model.entries[j])
        columns[name] = (model.col_cost[j], model.col_lower[j], model.col_upper[j],
                         model.col_integer[j], tuple(entries))
    rows = {name: (model.row_lower[i], model.row_upper[i])
            for i, name in enumerate(model.row_names)}
    return {"columns": columns, "rows": rows, "offset": model.objective_offset,
            "maximize": model.maximize, "col_order": list(model.col_names),
            "row_order": list(model.row_names)}


def test_permutation_keeps_the_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "seedtest.mps.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(MPS)
        original = canonical(source)
        check(len(original["columns"]) == 6 and len(original["rows"]) == 5,
              "the fixture parses to 6 columns and 5 constraint rows")
        orders = set()
        for seed in range(1, 9):
            target = Path(tmp) / f"seed{seed}.mps"
            counts = miplib_seeds.permute_mps(source, target, seed)
            permuted = canonical(target)
            same = all(permuted[k] == original[k]
                       for k in ("columns", "rows", "offset", "maximize"))
            check(same, f"seed {seed}: same columns, rows, bounds, integrality, offset")
            check(counts == {"rows": 5, "columns": 6, "integer_columns": 4},
                  f"seed {seed}: counts", str(counts))
            orders.add((tuple(permuted["col_order"]), tuple(permuted["row_order"])))
            again = Path(tmp) / f"again{seed}.mps"
            miplib_seeds.permute_mps(source, again, seed)
            check(again.read_bytes() == target.read_bytes(), f"seed {seed} is deterministic")
            n_rows = [line.split()[1] for line in target.read_text().splitlines()
                      if line.split()[:1] == ["N"]]
            check(n_rows == ["COST", "FREE"],
                  f"seed {seed}: the objective row is still the first N row", str(n_rows))
        check(len(orders) >= 6, "eight seeds give (nearly) eight different orders",
              f"{len(orders)} distinct")
        check((tuple(original["col_order"]), tuple(original["row_order"])) not in orders
              or len(orders) > 1, "the permutations are not all the original order")


def test_permutation_refuses_a_split_column() -> None:
    split = MPS.replace("    X2        R4           1.0\n", "").replace(
        "    X4        COST         2.0   R4           3.0\n",
        "    X4        COST         2.0   R4           3.0\n    X2        R4           1.0\n")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "split.mps"
        source.write_text(split)
        try:
            miplib_seeds.permute_mps(source, Path(tmp) / "out.mps", 1)
            refused = False
        except ValueError:
            refused = True
        check(refused, "a column declared in two places is refused, not guessed at")


def test_shifted_geometric_mean() -> None:
    sgm = miplib_seeds.shifted_geometric_mean
    check(sgm([], 10.0) is None, "empty is None")
    check(abs(sgm([0.0, 0.0], 10.0)) < 1e-12, "all zero gives zero")
    check(abs(sgm([5.0], 10.0) - 5.0) < 1e-12, "one value is itself")
    expected = math.sqrt(20.0 * 40.0) - 10.0
    check(abs(sgm([10.0, 30.0], 10.0) - expected) < 1e-12, "sqrt(20*40) - 10",
          f"{sgm([10.0, 30.0], 10.0)!r}")


def test_primal_gap() -> None:
    gap = miplib_seeds.primal_gap
    check(gap(None, 5.0) == 1.0, "no point: 1")
    check(gap(math.inf, 5.0) == 1.0, "an infinite objective: 1")
    check(gap(0.0, 0.0) == 0.0, "both zero: 0")
    check(gap(-1.0, 2.0) == 1.0, "opposite signs: 1")
    check(abs(gap(110.0, 100.0) - 10.0 / 110.0) < 1e-15, "|110-100| / 110")
    check(abs(gap(-90.0, -100.0) - 0.1) < 1e-15, "|-90+100| / 100, a maximisation too")
    check(gap(0.0, 3.0) == 1.0, "zero against a nonzero optimum: 1")


def test_primal_integral() -> None:
    integral = miplib_seeds.primal_integral
    value = integral([(1.0, 110.0), (3.0, 100.0)], 100.0, 10.0)
    expected = 1.0 * 1.0 + (10.0 / 110.0) * 2.0
    check(abs(value - expected) < 1e-12, "gap 1 until the first point, then a step",
          f"{value!r} vs {expected!r}")
    check(integral([], 100.0, 7.0) == 7.0, "no point: the whole run at gap 1")
    check(integral([(0.0, 100.0)], 100.0, 7.0) == 0.0, "optimal at once: 0")
    out_of_order = integral([(3.0, 100.0), (1.0, 110.0)], 100.0, 10.0)
    check(abs(out_of_order - expected) < 1e-12, "events are read in time order")
    late = integral([(1.0, 110.0), (12.0, 100.0)], 100.0, 10.0)
    check(abs(late - (1.0 + 9.0 * 10.0 / 110.0)) < 1e-12, "an event after the end is ignored")


def test_incumbent_metrics() -> None:
    metrics = miplib_seeds.incumbent_metrics
    found = metrics([[0.5, 12.0], [2.0, 10.0]], 10.0, 4.0, True)
    check(found["time_to_first_feasible"] == 0.5 and found["incumbents"] == 2
          and found["trace_recorded"], "the first event is the time to first feasible")
    check(abs(found["primal_integral"] - (0.5 + 1.5 * 2.0 / 12.0)) < 1e-12,
          "and the integral runs to the end")
    none = metrics([], 10.0, 4.0, False)
    check(none["time_to_first_feasible"] is None and none["primal_integral"] == 4.0,
          "nothing found: no time, integral = the run")
    untraced = metrics([], 10.0, 4.0, True)
    check(untraced["time_to_first_feasible"] == 4.0 and not untraced["trace_recorded"],
          "a point without a trace is charged at the end and flagged")


def test_aggregate() -> None:
    rows = [
        {"instance": "a", "seed": 0, "matched": True, "proved": True, "seconds": 2.0,
         "time_to_first_feasible": 0.1, "primal_integral": 0.3},
        {"instance": "a", "seed": 1, "matched": True, "proved": False, "seconds": 20.5,
         "time_to_first_feasible": 0.2, "primal_integral": 0.5},
        {"instance": "a", "seed": 2, "matched": False, "proved": False, "seconds": 20.1,
         "time_to_first_feasible": None, "primal_integral": 20.1},
        {"instance": "b", "seed": 0, "matched": True, "proved": True, "seconds": 4.0,
         "time_to_first_feasible": 1.0, "primal_integral": 1.0},
    ]
    summary = {s["instance"]: s for s in miplib_seeds.aggregate(rows, 20.0)}
    a = summary["a"]
    check(a["seeds"] == 3 and a["seeds_matched"] == 2 and a["seeds_proved"] == 1,
          "a: 3 seeds, 2 matched, 1 proved")
    check(a["matched_seeds"] == [0, 1] and a["proved_seeds"] == [0], "a: which seeds")
    expected = math.exp((math.log(12.0) + 2 * math.log(30.0)) / 3) - 10.0
    check(abs(a["sgm_seconds"] - expected) < 1e-12,
          "a: unproved runs are charged the limit, proved ones their time",
          f"{a['sgm_seconds']!r}")
    expected_first = math.exp((math.log(1.1) + math.log(1.2) + math.log(21.0)) / 3) - 1.0
    check(abs(a["sgm_first_feasible_seconds"] - expected_first) < 1e-12,
          "a: a seed that found nothing is charged the limit for first feasible")
    check(abs(a["mean_primal_integral"] - (0.3 + 0.5 + 20.1) / 3) < 1e-12,
          "a: mean primal integral")
    check(summary["b"]["seeds"] == 1 and abs(summary["b"]["sgm_seconds"] - 4.0) < 1e-12,
          "b: one seed is its own mean")
    total = miplib_seeds.overall(rows, 20.0)
    check(total == {"runs": 4, "matched": 3, "proved": 2,
                    "sgm_seconds": total["sgm_seconds"]}, "overall counts")


def test_tier2_rule() -> None:
    optima = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    sizes = {"a": 300, "b": 100, "c": 100, "d": None, "e": 50, "f": 10}
    chosen = fetch_miplib.select_by_size(["a", "b", "c", "d", "e", "f"], optima, sizes, 3)
    check(chosen == [("e", 50), ("b", 100), ("c", 100)],
          "smallest first, ties by name, unsized and unproved left out", str(chosen))
    record = json.loads(fetch_miplib.TIER2_FILE.read_text())
    listed = record["instances"]
    check(record["rule"] == fetch_miplib.TIER2_RULE, "the committed list states the rule")
    check(len(listed) == fetch_miplib.TIER2_COUNT == 60, "sixty instances")
    keys = [(entry["gz_bytes"], entry["name"]) for entry in listed]
    check(keys == sorted(keys), "the committed list is in the rule's order")
    check(len({entry["name"] for entry in listed}) == 60, "no instance twice")
    check(all(math.isfinite(entry["published_optimal"]) for entry in listed),
          "every instance carries its proven optimum")


def main() -> int:
    for test in (test_permutation_keeps_the_model, test_permutation_refuses_a_split_column,
                 test_shifted_geometric_mean, test_primal_gap, test_primal_integral,
                 test_incumbent_metrics, test_aggregate, test_tier2_rule):
        print(test.__name__)
        test()
    print(f"{'OK' if FAILURES == 0 else 'FAILED'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
