#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Learn the engine choice from our own runs as a shallow decision tree (#477).

    python bench/runners/engine_selection_cart.py bench/results/engine-selection-<sha>.csv
    python bench/runners/engine_selection_cart.py <csv> --emit src/core/engine_selection_tree.cpp \\
        --emit-cases tests/unit/engine_selection_tree_cases.inc --lists bench/engine_selection

The CSV is the one bench/runners/engine_selection_data.py writes: per instance, the model
features `sankhya info --features` prints (computed by src/core/engine_features.cpp, so the
tree reads exactly the numbers the solver would), the split the instance belongs to, and each
engine's status and solve time. An engine that did not return a verified optimum is charged
the time limit, as the MIPLIB summary charges an unproved run (miplib_seeds.py). The label of
an instance is its fastest engine; its REGRET under a choice is the chosen engine's time minus
the fastest engine's.

The tree is CART (Breiman, Friedman, Olshen and Stone, Classification and Regression Trees,
1984): binary splits `feature <= threshold`, chosen to minimise the weighted Gini impurity of
the two children, grown to MAX_DEPTH with at least MIN_LEAF instances per leaf. Ties are
broken by feature order and then by threshold, so the same CSV always gives the same tree.
A threshold is any number between the two neighbouring training values (every one gives the
same partition of the training rows); the one written is the shortest decimal in the middle
half of that gap, nearest its midpoint, so a reviewer reads `rows <= 1500`, not
`rows <= 1499.5`. Algorithm selection from instance features: Rice, The algorithm selection
problem, Advances in Computers 15 (1976); Kotthoff, Algorithm selection for combinatorial
search problems: a survey, AI Magazine 35(3) (2014).

WHAT IS REPORTED. On the held-out rows: the tree's accuracy (it picked the fastest engine),
its regret (time lost against the fastest engine, summed), and the same two numbers for the
rule table `algorithm=auto` runs today (the `rule_table` column), so the tree is judged
against what it would replace, not against nothing; beside them, always choosing each one
engine, and the per-instance choices.

THE TRAINING-SIZE GATE. With fewer than MIN_TRAINING training rows or MIN_HELD_OUT held-out
rows, or with fewer than MIN_TRAINING_SHAPES / MIN_HELD_OUT_SHAPES distinct shapes (the
`family` column) on either side, the tree is printed but --emit refuses to write rules: a
depth-4 tree fitted to a dozen instances describes those instances, not the next model a
user brings, and eight seeds of one shape are closer to one instance than to eight. --emit
also refuses a CSV where a shape appears on both sides of the split, which would let the
held-out score reward recall of a shape the tree was trained on; a CSV where a TRAINING
instance has the sha256 of an instance any reported benchmark CSV in bench/results/ ran
(the `instance_sha256` column of the Netlib, Kennington, Mittelmann, MIPLIB and scale
tables), or the name of a file in a reported data set; and a tree that does not lose less
time than the rule table on the held-out rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
# The data sets docs/BENCHMARKS.md reports, by directory under data/.
REPORTED_DATA = ("netlib", "netlib-infeasible", "kennington", "mittelmann", "miplib3",
                 "maros-meszaros", "qplib", "scale", "stress", "pooling", "casestudies")
ENGINES = ["dual-simplex", "ipm", "pdhg"]
# The features the tree may split on: every field of EngineFeatures (engine_features.hpp)
# except the raw bound it derives a ratio from and the cap flag.
FEATURES = [
    "rows", "columns", "nonzeros", "density", "max_column_count", "dense_columns",
    "rows_per_column", "equality_row_share", "integer_share", "boxed_column_share",
    "free_column_share", "fixed_column_share", "normal_equations_ratio",
    "cholesky_nonzeros", "cholesky_fill_ratio",
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
        out[engine] = min(seconds, time_limit) if ok and math.isfinite(seconds) else time_limit
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
                "sha256": raw.get("sha256", ""),
                "shape": raw.get("family") or raw["instance"],
                "split": raw["split"],
                "x": [float(raw.get(f) or "nan") for f in FEATURES],
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


def nice_threshold(lo: float, hi: float) -> float:
    """The shortest decimal in the middle half of [lo, hi], nearest the midpoint. Every
    threshold in [lo, hi) splits the training rows alike; this one reads well and keeps a
    margin from both neighbours. The nearest d-digit decimal to the midpoint is the only
    d-digit candidate that can lie in the middle half, so the first d whose rounding lands
    there gives the answer."""
    mid = (lo + hi) / 2.0
    a, b = lo + (hi - lo) / 4.0, hi - (hi - lo) / 4.0
    for digits in range(1, 18):
        candidate = float(f"{mid:.{digits}g}")
        if a <= candidate <= b:
            return candidate
    return mid


def best_split(rows: list[dict]):
    """(feature index, threshold) minimising the children's weighted Gini, or None."""
    labels = [r["label"] for r in rows]
    parent = gini(labels)
    best = None
    best_score = parent - 1e-12  # a split must reduce impurity
    for f in range(len(FEATURES)):
        values = sorted({r["x"][f] for r in rows if math.isfinite(r["x"][f])})
        for lo, hi in zip(values, values[1:]):
            left = [r["label"] for r in rows if r["x"][f] <= lo]
            right = [r["label"] for r in rows if r["x"][f] > lo]
            if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
                continue
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(rows)
            if score < best_score:
                best, best_score = (f, nice_threshold(lo, hi)), score
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
    if "feature" not in node["left"] and "feature" not in node["right"] \
            and node["left"]["engine"] == node["right"]["engine"]:
        # Two leaves naming the same engine are one rule; a reviewer should not have to
        # read a test that changes nothing.
        for key in ("feature", "threshold", "left", "right"):
            del node[key]
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
    total = sum(r["times"].get(choose(r), r["limit"]) for r in judged)
    return {"rows": len(judged), "hits": hits, "regret": regret, "total": total}


def number(value: float) -> str:
    """A threshold as a reviewer reads it: an integer without a fraction, else shortest."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def literal(value: float) -> str:
    """The same number as a C++ double literal."""
    text = number(value)
    return text + ".0" if text.lstrip("-").isdigit() else text


def counts_text(node: dict) -> str:
    return ", ".join(f"{e} {node['counts'][e]}" for e in ENGINES)


def describe(node: dict, indent: str = "") -> list[str]:
    if "feature" not in node:
        return [f"{indent}-> {node['engine']}   ({node['samples']} rows: {counts_text(node)})"]
    name = FEATURES[node["feature"]]
    return ([f"{indent}if {name} <= {number(node['threshold'])}:   ({node['samples']} rows: "
             f"{counts_text(node)})"] + describe(node["left"], indent + "    ")
            + [f"{indent}else:"] + describe(node["right"], indent + "    "))


def emit_body(node: dict, indent: str = "  ") -> list[str]:
    """The tree as C++ statements over an EngineFeatures `f`, one comment per branch, each
    test recorded in the choice's path."""
    if "feature" not in node:
        return [f"{indent}// {node['samples']} training rows end here: {counts_text(node)}.",
                f"{indent}c.algorithm = \"{node['engine']}\";", f"{indent}return c;"]
    name = FEATURES[node["feature"]]
    shown = number(node["threshold"])
    return ([f"{indent}// {node['samples']} training rows reach this test: "
             f"{counts_text(node)}.",
             f"{indent}if (static_cast<double>(f.{name}) <= {literal(node['threshold'])}) {{",
             f"{indent}  take(&c, \"{name} <= {shown}\");"]
            + emit_body(node["left"], indent + "  ") + [f"{indent}}}",
                                                         f"{indent}take(&c, \"{name} > {shown}\");"]
            + emit_body(node["right"], indent))


def emit_cpp(tree: dict, source: str, train: list[dict], scores: dict) -> str:
    max_rows = int(max(r["x"][FEATURES.index("rows")] for r in train))
    max_nonzeros = int(max(r["x"][FEATURES.index("nonzeros")] for r in train))
    learned, table = scores["tree"], scores["rule table"]
    head = f"""// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the learned engine selection's decision tree (#477), behind
// algorithm_selection=learned. GENERATED by bench/runners/engine_selection_cart.py from
// bench/results/{source}; regenerate it rather than edit it, then run scripts/format.sh.
//
// CART (Breiman, Friedman, Olshen and Stone, Classification and Regression Trees, 1984):
// Gini impurity, depth at most {MAX_DEPTH}, at least {MIN_LEAF} training rows per leaf. Algorithm
// selection from instance features: Rice, "The algorithm selection problem", Advances in
// Computers 15 (1976); Kotthoff, "Algorithm selection for combinatorial search problems: a
// survey", AI Magazine 35(3) (2014).
//
// Trained on {len(train)} generated instances from {len({r['shape'] for r in train})} shapes, none of them an instance any
// reported benchmark runs (bench/engine_selection/train.txt). On the {learned['rows']} held-out instances
// of the other shapes (test.txt) it picks the fastest engine on {learned['hits']} and loses
// {learned['regret']:.1f} s against the fastest engine; the rule table picks it on {table['hits']} and loses
// {table['regret']:.1f} s. Each comment below says how many training rows reached that point and which
// engine was fastest on them.

#include "core/engine_selection_tree.hpp"

namespace sankhya {{
namespace {{

/// Append one test to the path the log prints.
void take(LearnedTreeChoice* c, const char* test) {{
  if (!c->path.empty()) c->path += " and ";
  c->path += test;
}}

}}  // namespace

LearnedTreeDomain learned_tree_domain() {{
  // The largest training instance: past either, the tree has seen nothing like the model.
  return LearnedTreeDomain{{{max_rows}, {max_nonzeros}}};
}}

LearnedTreeChoice learned_engine_tree(const EngineFeatures& f) {{
  LearnedTreeChoice c;
"""
    unused = "" if "feature" in tree else "  (void)f;\n  (void)&take;\n"
    return head + unused + "\n".join(emit_body(tree)) + "\n}\n\n}  // namespace sankhya\n"


def emit_cases(rows: list[dict], tree: dict, source: str) -> str:
    """Every row's features and the engine the Python tree picks, for the C++ test that the
    generated rules are the same tree (tests/unit/test_engine_selection_tree.cpp)."""
    lines = [f"// Generated by bench/runners/engine_selection_cart.py from {source} (#477):",
             "// each instance's features, in kTreeCaseFeatures order, and the engine the trained",
             "// tree picks for it. tests/unit/test_engine_selection_tree.cpp checks the generated",
             "// C++ picks the same.",
             "inline constexpr const char* kTreeCaseFeatures[] = {"
             + ", ".join(f'"{f}"' for f in FEATURES) + "};"
             , "inline constexpr TreeCase kTreeCases[] = {"]
    for r in rows:
        values = ", ".join(repr(v) for v in r["x"])
        lines.append(f'    {{"{r["instance"]}", "{predict(tree, r["x"])}", {{{values}}}}},')
    lines.append("};")
    return "\n".join(lines) + "\n"


def reported_instances() -> tuple[set[str], set[str]]:
    """(sha256s, names) of every instance a reported benchmark ran: the instance_sha256
    column of every CSV in bench/results/, and the file names in the reported data sets."""
    hashes: set[str] = set()
    for path in RESULTS_DIR.glob("*.csv"):
        if path.name.startswith(("engine-selection-", "engine-tree-ab-")):
            continue
        with path.open(newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "instance_sha256" not in reader.fieldnames:
                continue
            hashes.update(r["instance_sha256"] for r in reader if r.get("instance_sha256"))
    names = {p.stem.split(".")[0] for d in REPORTED_DATA
             for p in (REPO_ROOT / "data" / d).glob("**/*") if p.is_file()}
    return hashes, names


def write_lists(directory: Path, rows: list[dict], source: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train.txt"), ("test", "test.txt")):
        chosen = [r for r in rows if r["split"] == split]
        lines = [f"# {split} split of the learned engine selection (#477), {len(chosen)} instances "
                 f"from {len({r['shape'] for r in chosen})} shapes.",
                 f"# Written by bench/runners/engine_selection_cart.py from {source}; each "
                 "line is the instance, its shape and",
                 "# the sha256 of the generated MPS file. The generator and arguments of each "
                 "shape are in",
                 "# bench/runners/engine_selection_data.py (SHAPES); the seed is the -sN suffix."]
        lines += [f"{r['instance']} {r['shape']} {r['sha256']}" for r in chosen]
        (directory / name).write_text("\n".join(lines) + "\n", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--emit", type=Path, default=None,
                        help="write the tree as a C++ source file here, if the gates pass")
    parser.add_argument("--emit-cases", type=Path, default=None,
                        help="with --emit: every row and the tree's choice, for the C++ test")
    parser.add_argument("--lists", type=Path, default=None,
                        help="with --emit: write the split as train.txt and test.txt here")
    args = parser.parse_args(argv)

    rows = load(args.csv)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "test"]
    tree = grow([r for r in train if r["any_solved"]])
    print(f"{len(train)} training rows ({len({r['shape'] for r in train})} shapes), "
          f"{len(held)} held-out rows ({len({r['shape'] for r in held})} shapes)")
    print("\n".join(describe(tree)))
    choosers = {"tree": lambda r: predict(tree, r["x"]), "rule table": lambda r: r["rule_table"]}
    for engine in ENGINES:
        choosers[f"always {engine}"] = lambda r, e=engine: e
    choosers["fastest (oracle)"] = lambda r: r["label"]
    scores = {}
    for split_name, part in (("training", train), ("held out", held)):
        for name, choose in choosers.items():
            result = evaluate(part, choose)
            if split_name == "held out":
                scores[name] = result
            rate = result["hits"] / result["rows"] if result["rows"] else float("nan")
            print(f"{split_name}, {name}: {result['hits']}/{result['rows']} fastest engine "
                  f"({rate:.1%}), regret {result['regret']:.3f} s, total {result['total']:.3f} s")
    print("held-out instances: instance, fastest, tree, rule table, seconds per engine")
    for r in held:
        seconds = " ".join(f"{e}={r['times'][e]:.2f}" for e in ENGINES)
        print(f"  {r['instance']:<34} {r['label']:<13} {predict(tree, r['x']):<13} "
              f"{r['rule_table']:<13} {seconds}")

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
    hashes, names = reported_instances()
    leaked = [r["instance"] for r in train if r["sha256"] in hashes or r["instance"] in names]
    if leaked:
        print(f"not emitted: training instance(s) a reported benchmark also runs: "
              f"{' '.join(leaked)}", file=sys.stderr)
        return 2
    print(f"no training instance is among the {len(hashes)} instance hashes of the reported "
          f"benchmark CSVs or the {len(names)} files of the reported data sets")
    reported_held = [r["instance"] for r in held if r["sha256"] in hashes]
    if reported_held:
        print(f"held-out instances a reported benchmark also ran (allowed; only training must "
              f"be disjoint): {' '.join(reported_held)}")
    if scores["tree"]["regret"] >= scores["rule table"]["regret"]:
        print("not emitted: the tree does not lose less time than the rule table on the "
              "held-out rows", file=sys.stderr)
        return 2
    args.emit.write_text(emit_cpp(tree, args.csv.name, train, scores), newline="\n")
    print(f"wrote {args.emit}")
    if args.emit_cases is not None:
        args.emit_cases.write_text(emit_cases(rows, tree, args.csv.name), newline="\n")
        print(f"wrote {args.emit_cases}")
    if args.lists is not None:
        write_lists(args.lists, rows, args.csv.name)
        print(f"wrote {args.lists / 'train.txt'} and {args.lists / 'test.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
