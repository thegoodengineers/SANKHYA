#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Training data for the learned engine selection (#477): generated instances, every engine.

    python bench/runners/engine_selection_data.py --time-limit 30
    python bench/runners/engine_selection_data.py --list          # the split, nothing solved

WHY GENERATED INSTANCES. Every public set we report (Netlib, Kennington, Mittelmann, MIPLIB,
Maros-Meszaros) must stay out of training, or the accuracy the tree reports on them would be
an accuracy on its own training data. The instances here come from our four generators -
generate_large_lp.py (random and staircase sparsity), generate_refinery_lp.py,
make_scaling_instance.py and generate_transport_lp.py - over a grid of shapes and seeds that
is fixed in this file, so the split is committed as code; engine_selection_cart.py also
refuses to train when a training instance has the sha256 of an instance a reported benchmark
CSV ran, and writes the split out as instance lists (bench/engine_selection/). The split is by SHAPE, not by seed: every seed of a shape in
HELD_OUT_SHAPES is held out and no seed of it is trained on. A seed changes the numbers, not
the shape, so for most generators it leaves every feature the tree reads (rows, columns,
nonzeros, the bound shares) unchanged; holding out seeds would put an exact feature twin of
each held-out instance in the training set and measure recall, not generalization. One
shape of each generator is held out, sized between trained shapes of the same generator.
`--list` prints both lists.

Each instance is generated into a temporary directory, its features read with `sankhya info
--features` (the same numbers the solver computes, and the rule table's choice beside them),
then solved once per engine with presolve as the solver runs it; an engine's answer counts
only when it is optimal AND tools/verify_solution.py accepts it. One CSV row per instance,
carrying the sha256 of the generated MPS file and each engine's objective and iterations.
A solve that outlives twice its time limit plus a minute is killed and recorded as
`killed`.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = REPO_ROOT / "bench" / "runners"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
ENGINES = ["dual-simplex", "ipm", "pdhg"]
SEEDS = range(1, 6)

# (family, generator, arguments) with {seed} and {out} filled in per instance. The grid spans
# the regimes the probe runs for #477 found each engine winning on this machine: sparse
# models of a few thousand rows (the dual simplex), dense ones with tens of entries per
# column (PDHG, then the interior point on the staircase), multi-period refineries from
# about 4,000 rows (the interior point) and large transportation models (PDHG).
SHAPES = []
for rows, cols, per_col in ((300, 600, 4), (1000, 1000, 5), (2000, 4000, 3), (4000, 2000, 6),
                            (8000, 4000, 4), (1000, 4000, 40), (2000, 2000, 60)):
    for structure in ("random", "staircase"):
        SHAPES.append((f"{structure}-{rows}x{cols}k{per_col}", "generate_large_lp.py",
                       ["--rows", str(rows), "--cols", str(cols), "--nnz-per-col",
                        str(per_col), "--structure", structure, "--periods", "10"]))
# SPARSE AND LARGE (#477, second pass). Without these the grid had no sparse model above
# 20,000 nonzeros: every model past that was dense (tens of entries per column), a refinery or
# a transportation model, and the first tree split on `nonzeros <= 20000` alone, a boundary
# the grid drew rather than the engines. Here the dual simplex wins at 24,000 to 30,000.
for rows, cols, per_col in ((6000, 6000, 4), (5000, 10000, 3)):
    for structure in ("random", "staircase"):
        SHAPES.append((f"{structure}-{rows}x{cols}k{per_col}", "generate_large_lp.py",
                       ["--rows", str(rows), "--cols", str(cols), "--nnz-per-col",
                        str(per_col), "--structure", structure, "--periods", "10"]))
SHAPES.append(("random-1500x3000k50", "generate_large_lp.py",
               ["--rows", "1500", "--cols", "3000", "--nnz-per-col", "50", "--structure",
                "random", "--periods", "10"]))
for periods, crudes in ((4, 10), (12, 20), (26, 30), (52, 30), (100, 30)):
    SHAPES.append((f"refinery-t{periods}c{crudes}", "generate_refinery_lp.py",
                   ["--periods", str(periods), "--crudes", str(crudes)]))
for periods, goods in ((50, 2), (200, 3), (500, 3)):
    SHAPES.append((f"scaling-t{periods}g{goods}", "make_scaling_instance.py",
                   ["--periods", str(periods), "--goods", str(goods)]))
for sources, sinks, arcs, window in ((300, 600, 4, 10), (1000, 2000, 5, 20),
                                     (3000, 6000, 6, 30)):
    SHAPES.append((f"transport-s{sources}d{sinks}a{arcs}w{window}", "generate_transport_lp.py",
                   ["--sources", str(sources), "--sinks", str(sinks), "--arcs", str(arcs),
                    "--window", str(window)]))

# One shape per generator, and one dense one, each between trained shapes of its generator
# in size, so the held-out set asks the tree to interpolate to shapes it has not seen rather
# than to recall the ones it has.
HELD_OUT_SHAPES = frozenset({"random-1000x1000k5", "random-1500x3000k50",
                             "staircase-2000x4000k3", "refinery-t52c30", "scaling-t200g3",
                             "transport-s1000d2000a5w20"})


def instances():
    """Every (name, split, generator, arguments) of the grid."""
    for family, script, arguments in SHAPES:
        for seed in SEEDS:
            split = "test" if family in HELD_OUT_SHAPES else "train"
            yield f"{family}-s{seed}", split, script, arguments + ["--seed", str(seed)]


COLUMNS = (["instance", "sha256", "split", "family", "seed", "features_json", "rule_table",
            "rule"]
           + ["rows", "columns", "nonzeros", "density", "max_column_count", "dense_columns",
              "rows_per_column", "equality_row_share", "integer_share", "boxed_column_share",
              "free_column_share", "fixed_column_share", "normal_equations_nnz_bound",
              "normal_equations_ratio", "cholesky_nonzeros", "cholesky_fill_ratio",
              "cholesky_capped"]
           + [f"{e}_{k}" for e in ENGINES
              for k in ("status", "seconds", "verified", "objective", "iterations")]
           + ["time_limit", "git_commit", "machine", "timestamp_utc"])


def solve(binary: Path, model: Path, engine: str, time_limit: float, scratch: Path) -> dict:
    stats, sol = scratch / "stats.json", scratch / "solution.sol"
    stats.unlink(missing_ok=True)
    sol.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        subprocess.run([str(binary), "solve", str(model), "--time-limit", str(time_limit),
                        "--stats", str(stats), "--write-sol", str(sol),
                        "--option", "log_to_console=false", "--option", f"algorithm={engine}"],
                       capture_output=True, text=True, timeout=2.0 * time_limit + 60.0)
    except subprocess.TimeoutExpired:
        return {"status": "killed", "seconds": time.perf_counter() - started, "verified": 0,
                "objective": "", "iterations": ""}
    wall = time.perf_counter() - started
    if not stats.exists():
        return {"status": "no_output", "seconds": wall, "verified": 0, "objective": "",
                "iterations": ""}
    blob = json.loads(stats.read_text())
    result = blob.get("result", {})
    status = result.get("status", "unknown")
    verified = 0
    if status == "optimal" and sol.exists():
        check = subprocess.run([sys.executable, str(VERIFIER), str(model), str(sol), "--quiet"],
                               capture_output=True, text=True)
        verified = int(check.returncode == 0)
    objective = result.get("objective")
    return {"status": status, "seconds": wall, "verified": verified,
            "objective": "" if objective is None else repr(float(objective)),
            "iterations": blob.get("effort", {}).get("iterations", "")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "sankhya")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="print the split and stop")
    parser.add_argument("--only", nargs="+", default=None, metavar="SHAPE",
                        help="run only these shapes (the family names of SHAPES)")
    args = parser.parse_args()

    grid = list(instances())
    if args.only:
        unknown = set(args.only) - {family for family, _, _ in SHAPES}
        if unknown:
            print(f"unknown shape(s): {' '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        grid = [g for g in grid if g[0].rpartition("-s")[0] in args.only]
    if args.list:
        for split in ("train", "test"):
            names = [name for name, s, _, _ in grid if s == split]
            print(f"{split} ({len(names)}): {' '.join(names)}")
        return 0
    if not args.binary.exists():
        print("no solver binary; build first", file=sys.stderr)
        return 1

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    when = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = []
    with tempfile.TemporaryDirectory(prefix="sankhya-engine-selection-") as tmp:
        scratch = Path(tmp)
        model = scratch / "model.mps"
        for name, split, script, arguments in grid:
            subprocess.run([sys.executable, str(RUNNERS / script), *arguments, "--out",
                            str(model)], check=True, capture_output=True, text=True)
            info = subprocess.run([str(args.binary), "info", str(model), "--features"],
                                  check=True, capture_output=True, text=True)
            features = json.loads(info.stdout)
            family, _, seed = name.rpartition("-s")
            row = {"instance": name,
                   "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "split": split, "family": family, "seed": seed,
                   "features_json": json.dumps(features, sort_keys=True),
                   "rule_table": features.pop("rule_table"), "rule": features.pop("rule"),
                   "time_limit": f"{args.time_limit:g}", "git_commit": commit,
                   "machine": machine, "timestamp_utc": when}
            # The tree compiled into this binary is not what is being trained.
            for key in ("learned_tree", "learned_path", "learned_in_domain"):
                features.pop(key, None)
            features["cholesky_capped"] = int(bool(features["cholesky_capped"]))
            row.update(features)
            for engine in ENGINES:
                result = solve(args.binary, model, engine, args.time_limit, scratch)
                row[f"{engine}_status"] = result["status"]
                row[f"{engine}_seconds"] = f"{result['seconds']:.6f}"
                row[f"{engine}_verified"] = result["verified"]
                row[f"{engine}_objective"] = result["objective"]
                row[f"{engine}_iterations"] = result["iterations"]
            rows.append(row)
            times = "  ".join(f"{e} {row[f'{e}_status']} {float(row[f'{e}_seconds']):.2f}s"
                              for e in ENGINES)
            print(f"{name:<36}{split:<6}{row['rule_table']:<14}{times}", flush=True)

    out = args.out or (RESULTS_DIR / f"engine-selection-{commit}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
