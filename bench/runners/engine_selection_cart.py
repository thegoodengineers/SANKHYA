#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Learn the engine choice from our own runs as a shallow decision tree (#477).

    python bench/runners/engine_selection_cart.py bench/results/engine-selection-<sha>.csv
    python bench/runners/engine_selection_cart.py <csv> --emit src/core/engine_selection_learned.inc

The CSV is the one bench/runners/engine_selection_data.py writes: per instance, the model
features `sankhya info --features` prints (computed by src/core/engine_features.cpp, so the
tree reads exactly the numbers the solver would), the split the instance belongs to, and each
engine's status and solve time. An engine that did not return a verified optimum is charged
the time limit, as the MIPLIB summary charges an unproved run (miplib_seeds.py). The label of
an instance is its fastest engine.

The tree is CART (Breiman, Friedman, Olshen and Stone, Classification and Regression Trees,
1984): binary splits `feature <= threshold` at midpoints between observed values, chosen to
minimise the weighted Gini impurity of the two children, grown to MAX_DEPTH with at least
MIN_LEAF instances per leaf. Ties are broken by feature order and then by threshold, so the
same CSV always gives the same tree. Algorithm selection from instance features: Rice, The
algorithm selection problem, Advances in Computers 15 (1976); Kotthoff, Algorithm selection
for combinatorial search problems: a survey, AI Magazine 35(3) (2014).

WHAT IS REPORTED. On the held-out rows: the tree's accuracy (it picked the fastest engine),
its regret (time lost against the fastest engine, summed), and the same two numbers for the
rule table `algorithm=auto` runs today (the `rule_table` column), so the tree is judged
against what it would replace, not against nothing.

THE TRAINING-SIZE GATE. With fewer than MIN_TRAINING training rows or MIN_HELD_OUT held-out
rows, or with fewer than MIN_TRAINING_SHAPES / MIN_HELD_OUT_SHAPES distinct shapes (the
`family` column) on either side, the tree is printed but --emit refuses to write rules: a
depth-4 tree fitted to a dozen instances describes those instances, not the next model a
user brings, and eight seeds of one shape are closer to one instance than to eight. --emit
also refuses a CSV where a shape appears on both sides of the split, which would let the
held-out score reward recall of a shape the tree was trained on.
--emit also refuses when the tree does not beat the rule table's regret on the held-out rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ENGINES = ["dual-simplex", "ipm", "pdhg"]
# The features the tree may split on: every field of EngineFeatures (engine_features.hpp)
# except the raw bound it derives a ratio from.
FEATURES = [
    "rows", "columns", "nonzeros", "density", "max_column_count", "dense_columns",
    "rows_per_column", "equality_row_share", "integer_share", "boxed_column_share",
    "free_column_share", "fixed_column_share", "normal_equations_ratio",
]
MAX_DEPTH = 4
MIN_LEAF = 3
MIN_TRAINING = 60
MIN_HELD_OUT = 20
MIN_TRAINING_SHAPES = 8
MIN_HELD_OUT_SHAPES = 4


def engine_seconds(row: dict, time_limit: float) -> dict:
    """Each engine's time, the time limit for one that did not return a verified optimum."""
    out = {}
    for engine in ENGINES:
        ok = row.get(f"{engine}_status") == "optimal" and row.get(f"{engine}_verified") == "1"
        seconds = float(row.get(f"{engine}_seconds") or "nan")
        out[engine] = seconds if ok and math.isfinite(seconds) else time_limit
    return out


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            limit = float(raw["time_limit"])
            times = engine_seconds(raw, limit)
            best = min(ENGINES, key=lambda e: (times[e], ENGINES.index(e)))
            rows.append({
                "instance": raw["instance"],
                "shape": raw.get("family") or raw["instance"],
                "split": raw["split"],
                "x": [float(raw[f]) for f in FEATURES],
                "times": times,
                "label": best,
                "rule_table": raw.get("rule_table", ""),
                "any_solved": min(times.values()) < limit,
                "limit": limit,
            })
    return rows


def gini(labels: list[str]) -> float:
    if not labels:
        return 0.0
    n = len(labels)
    return 1.0 - sum((labels.count(e) / n) ** 2 for e in ENGINES)


def majority(labels: list[str]) -> str:
    return max(ENGINES, key=lambda e: (labels.count(e), -ENGINES.index(e)))


def best_split(rows: list[dict]):
    """(feature index, threshold) minimising the children's weighted Gini, or None."""
    labels = [r["label"] for r in rows]
    parent = gini(labels)
    best = None
    best_score = parent - 1e-12  # a split must reduce impurity
    for f in range(len(FEATURES)):
        values = sorted({r["x"][f] for r in rows})
        for lo, hi in zip(values, values[1:]):
            t = (lo + hi) / 2.0
            left = [r["label"] for r in rows if r["x"][f] <= t]
            right = [r["label"] for r in rows if r["x"][f] > t]
            if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
                continue
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(rows)
            if score < best_score:
                best, best_score = (f, t), score
    return best


def grow(rows: list[dict], depth: int = 0) -> dict:
    labels = [r["label"] for r in rows]
    node = {"engine": majority(labels), "counts": {e: labels.count(e) for e in ENGINES},
            "samples": len(rows)}
    if depth >= MAX_DEPTH or len(set(labels)) == 1:
        return node
    split = best_split(rows)
    if split is None:
        return node
    f, t = split
    node["feature"], node["threshold"] = f, t
    node["left"] = grow([r for r in rows if r["x"][f] <= t], depth + 1)
    node["right"] = grow([r for r in rows if r["x"][f] > t], depth + 1)
    return node


def predict(node: dict, x: list[float]) -> str:
    while "feature" in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["engine"]


def evaluate(rows: list[dict], choose) -> dict:
    """Accuracy and regret of a chooser over rows; rows no engine solved are left out. A
    choice that was not measured (an engine outside ENGINES) is charged the time limit."""
    judged = [r for r in rows if r["any_solved"]]
    hits = sum(choose(r) == r["label"] for r in judged)
    regret = sum(r["times"].get(choose(r), r["limit"]) - r["times"][r["label"]]
                 for r in judged)
    return {"rows": len(judged), "hits": hits, "regret": regret}


def describe(node: dict, indent: str = "") -> list[str]:
    counts = ", ".join(f"{e} {node['counts'][e]}" for e in ENGINES)
    if "feature" not in node:
        return [f"{indent}-> {node['engine']}   ({node['samples']} rows: {counts})"]
    name = FEATURES[node["feature"]]
    return ([f"{indent}if {name} <= {node['threshold']:.17g}:   ({node['samples']} rows: "
             f"{counts})"] + describe(node["left"], indent + "    ")
            + [f"{indent}else:"] + describe(node["right"], indent + "    "))


def emit_cpp(node: dict, indent: str = "  ") -> list[str]:
    """The tree as C++ statements over an EngineFeatures `f`, one comment per branch."""
    counts = ", ".join(f"{e} {node['counts'][e]}" for e in ENGINES)
    if "feature" not in node:
        return [f"{indent}// {node['samples']} training rows: {counts}",
                f"{indent}return \"{node['engine']}\";"]
    name = FEATURES[node["feature"]]
    value = f"static_cast<double>(f.{name})"
    return ([f"{indent}// {node['samples']} training rows: {counts}",
             f"{indent}if ({value} <= {node['threshold']!r}) {{"]
            + emit_cpp(node["left"], indent + "  ") + [f"{indent}}}"]
            + emit_cpp(node["right"], indent))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--emit", type=Path, default=None,
                        help="write the tree as C++ statements here, if the gates pass")
    args = parser.parse_args(argv)

    rows = load(args.csv)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "test"]
    tree = grow([r for r in train if r["any_solved"]])
    print(f"{len(train)} training rows ({len({r['shape'] for r in train})} shapes), "
          f"{len(held)} held-out rows ({len({r['shape'] for r in held})} shapes)")
    print("\n".join(describe(tree)))
    learned = evaluate(held, lambda r: predict(tree, r["x"]))
    table = evaluate(held, lambda r: r["rule_table"])
    for name, result in (("tree", learned), ("rule table", table)):
        rate = result["hits"] / result["rows"] if result["rows"] else float("nan")
        print(f"held out, {name}: {result['hits']}/{result['rows']} fastest engine "
              f"({rate:.1%}), regret {result['regret']:.3f} s")

    if args.emit is None:
        return 0
    if len(train) < MIN_TRAINING or len(held) < MIN_HELD_OUT:
        print(f"not emitted: {len(train)} training and {len(held)} held-out rows, below the "
              f"{MIN_TRAINING} and {MIN_HELD_OUT} the gate asks for", file=sys.stderr)
        return 2
    train_shapes = {r["shape"] for r in train}
    held_shapes = {r["shape"] for r in held}
    if train_shapes & held_shapes:
        print(f"not emitted: shape(s) on both sides of the split: "
              f"{' '.join(sorted(train_shapes & held_shapes))}", file=sys.stderr)
        return 2
    if len(train_shapes) < MIN_TRAINING_SHAPES or len(held_shapes) < MIN_HELD_OUT_SHAPES:
        print(f"not emitted: {len(train_shapes)} training and {len(held_shapes)} held-out "
              f"shapes, below the {MIN_TRAINING_SHAPES} and {MIN_HELD_OUT_SHAPES} the gate "
              f"asks for", file=sys.stderr)
        return 2
    if learned["regret"] >= table["regret"]:
        print("not emitted: the tree does not lose less time than the rule table on the "
              "held-out rows", file=sys.stderr)
        return 2
    lines = ["// Generated by bench/runners/engine_selection_cart.py from "
             f"{args.csv.name} (#477). Review before committing."] + emit_cpp(tree)
    args.emit.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
