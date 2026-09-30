#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""N scenarios in one run against N separate single solves (#752).

    python bench/runners/scenarios.py --binary build/sankhya --counts 10 100 1000

For each instance - the demo crude blend and a daily refinery year from the #517 generator
(365 periods, the medium dimensions) - this draws N scenarios from a fixed seed, runs
`sankhya scenarios MODEL SCENARIOS.csv --compare`, and writes one row per scenario to
bench/results/scenarios_<machine>.csv: our objective beside the separate single solve's (the
reference), the gap, the status, whether the in-process check verified it, the times, the
commit the binary was built from, the machine and the sha256 of the instance.

THE SCENARIOS. The blend: each crude's margin moved by a uniform +/-3 $/bbl (a price move).
The refinery, as two sets: price sets (every crude's purchase cost scaled by one factor in
[0.9, 1.1]) and demand forecasts (every delivery commitment scaled by one factor in
[0.95, 1.0]). The refinery year's single solves take about half a minute each on the
7.7 GB laptop, so its counts are given separately (--refinery-counts) and kept small.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench" / "runners"))
sys.path.insert(0, str(REPO / "tools"))

import stamp  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend_scenarios(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    names = model.col_names
    lines = ["scenario," + ",".join(f"cost:{n}" for n in names)]
    for k in range(count):
        costs = [model.col_cost[j] + rng.uniform(-3.0, 3.0) for j in range(len(names))]
        lines.append(f"p{k}," + ",".join(f"{c:.6f}" for c in costs))
    return lines


def refinery_prices(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    buys = [j for j, n in enumerate(model.col_names) if n.startswith("BUY_")]
    lines = ["scenario," + ",".join(f"cost:{model.col_names[j]}" for j in buys)]
    for k in range(count):
        price = rng.uniform(0.9, 1.1)
        lines.append(f"p{k}," + ",".join(f"{model.col_cost[j] * price:.9g}" for j in buys))
    return lines


def refinery_demand(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    commits = [i for i, n in enumerate(model.row_names) if n.startswith("COMMIT")]
    lines = ["scenario," + ",".join(f"rhs:{model.row_names[i]}" for i in commits)]
    for k in range(count):
        demand = rng.uniform(0.95, 1.0)
        lines.append(f"d{k}," + ",".join(f"{model.row_lower[i] * demand:.9g}" for i in commits))
    return lines


def run(binary: Path, model: Path, scenario_lines: list[str], work: Path) -> list[dict]:
    csv_in = work / "scenarios.csv"
    csv_in.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")
    out = work / "out.csv"
    completed = subprocess.run([str(binary), "scenarios", str(model), str(csv_in), "--compare",
                                "--out", str(out)], capture_output=True, text=True, check=False)
    tail = completed.stdout.strip().splitlines()[-4:]
    print("\n".join(tail))
    if not out.exists():
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"{binary} scenarios wrote no table for {model}")
    with open(out, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--seed", type=int, default=752)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--refinery-counts", type=int, nargs="*", default=[10])
    args = parser.parse_args()

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    out = args.out or REPO / "bench" / "results" / f"scenarios_{machine}.csv"
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        instances = [("crude_blend", REPO / "demo" / "crude_blend.mps", blend_scenarios,
                      args.counts)]
        if args.refinery_counts:
            year = work / "refinery_daily_year.mps"
            subprocess.run([sys.executable,
                            str(REPO / "bench" / "case_studies" / "refinery" / "generator.py"),
                            "--size", "medium", "--periods", "365", "--seed", "1",
                            "--out", str(year)], check=True)
            instances.append(("refinery_daily_year:prices", year, refinery_prices,
                              args.refinery_counts))
            instances.append(("refinery_daily_year:demand", year, refinery_demand,
                              args.refinery_counts))
        for name, model, make, counts in instances:
            digest = sha256(model)
            for count in counts:
                print(f"== {name}, N = {count}")
                lines = make(model, count, random.Random(args.seed + count))
                for r in run(args.binary, model, lines, work):
                    ours = float(r["objective"]) if r["objective"] not in ("", "-") else None
                    ref = (float(r["single_objective"])
                           if r["single_objective"] not in ("", "-") else None)
                    gap = abs(ours - ref) if ours is not None and ref is not None else ""
                    rows.append({
                        "instance": name, "n_scenarios": count, "scenario": r["scenario"],
                        "instance_sha256": digest, "objective": r["objective"],
                        "reference_objective": r["single_objective"],
                        "reference": "separate single solve",
                        "abs_gap": gap,
                        "rel_gap": (gap / max(1.0, abs(ref))) if gap != "" else "",
                        "status": r["status"], "verified": r["verified"],
                        "agrees": r["agrees"], "wall_seconds": r["seconds"],
                        "single_wall_seconds": r["single_seconds"],
                        "iterations": r["iterations"], "git_commit": commit,
                        "machine": machine})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if r["verified"] != "yes" or r["agrees"] != "yes"]
    print(f"wrote {len(rows)} rows to {out}; {len(bad)} unverified or disagreeing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
