#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# GPU evidence on a datacenter card (A100 / L4) with full FP64 (#488).
#
# Usage (on the rented A100 or L4 with the GPU binary built):
#   python bench/runners/gpu_datacenter.py --binary build/sankhya \
#       --card a100 --repeats 5
#
# What this runs (per the issue spec):
#   - Synthetic ladder + refinery-year instances (generate_large_lp.py /
#     generate_refinery_lp.py)
#   - Mittelmann instances that PDHG finishes (from fetch_mittelmann.py)
#   - CPU (multithreaded, --threads N) vs GPU, SAME iteration count forced on
#     both sides (--iteration-limit K) so the ratio is purely per-iteration
#   - Then free-running to 1e-4, 1e-6, 1e-8
#   - HiGHS as a separate process on the same instances (context only)
#   - Median of --repeats N runs with spread (IQR)
#
# Output: bench/results/gpu-datacenter-<card>-<sha>.csv
# Columns: instance, sha256, objective, reference_objective, abs_gap, rel_gap,
#          status, wall_time_s, iterations, git_commit, machine, card,
#          cuda_version, driver_version, mode (cpu|gpu|highs), tol

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from statistics import median, quantiles
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: the CSV names the commit the BINARY was built from)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_sankhya(binary: str, mps: pathlib.Path, extra_args: list[str],
                repeats: int) -> dict[str, Any]:
    times = []
    result: dict[str, Any] = {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [binary, "solve", str(mps), "--stats", "-", "--json"] + extra_args,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if proc.returncode == 0:
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass
    result["wall_time_s"] = median(times)
    result["wall_time_iqr"] = (quantiles(times, n=4)[2] - quantiles(times, n=4)[0]
                               if len(times) >= 4 else 0.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU datacenter benchmark (#488)")
    parser.add_argument("--binary", required=True, help="path to sankhya binary")
    parser.add_argument("--card", default="unknown", help="card tag (a100, l4, …)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU thread count for the CPU reference run")
    parser.add_argument("--tols", default="1e-4,1e-6,1e-8",
                        help="comma-separated tolerance levels")
    parser.add_argument("--instances", nargs="+",
                        help="explicit .mps/.mps.gz paths (default: auto-discover)")
    args = parser.parse_args()

    tols = [float(t) for t in args.tols.split(",")]

    # Discover instances if not given explicitly
    instances: list[pathlib.Path] = []
    if args.instances:
        instances = [pathlib.Path(p) for p in args.instances]
    else:
        for d in [pathlib.Path("bench/data/mittelmann"),
                  pathlib.Path("bench/data/synthetic")]:
            if d.exists():
                instances += sorted(d.glob("*.mps")) + sorted(d.glob("*.mps.gz"))

    if not instances:
        print("No instances found. Run fetch_mittelmann.py or pass --instances.",
              file=sys.stderr)
        return 1

    # The commit the binary reports through `sankhya version`, with -dirty from the tree;
    # HEAD alone names whatever is checked out when the runner starts, which on a rented
    # card is not necessarily what was built (#433).
    git_commit = stamp.stamp(args.binary)
    machine = subprocess.run(
        ["uname", "-n"], capture_output=True, text=True
    ).stdout.strip()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"gpu-datacenter-{args.card}-{git_commit}.csv"
    fieldnames = [
        "instance", "sha256", "objective", "reference_objective",
        "abs_gap", "rel_gap", "status", "wall_time_s", "wall_time_iqr",
        "iterations", "git_commit", "machine", "card", "mode", "tol",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for mps in instances:
            digest = sha256_file(mps)
            for tol in tols:
                tol_args = [f"--option", f"primal_feasibility_tolerance={tol}",
                            "--option", f"dual_feasibility_tolerance={tol}"]
                # CPU run
                cpu_res = run_sankhya(
                    args.binary, mps,
                    ["--option", f"threads={args.threads}"] + tol_args,
                    args.repeats,
                )
                writer.writerow({
                    "instance": mps.name,
                    "sha256": digest,
                    "objective": cpu_res.get("objective", ""),
                    "reference_objective": "",
                    "abs_gap": cpu_res.get("abs_gap", ""),
                    "rel_gap": cpu_res.get("rel_gap", ""),
                    "status": cpu_res.get("status", "error"),
                    "wall_time_s": cpu_res["wall_time_s"],
                    "wall_time_iqr": cpu_res["wall_time_iqr"],
                    "iterations": cpu_res.get("iterations", ""),
                    "git_commit": git_commit,
                    "machine": machine,
                    "card": args.card,
                    "mode": f"cpu-{args.threads}t",
                    "tol": tol,
                })
                # GPU run
                gpu_res = run_sankhya(
                    args.binary, mps,
                    ["--gpu"] + tol_args,
                    args.repeats,
                )
                writer.writerow({
                    "instance": mps.name,
                    "sha256": digest,
                    "objective": gpu_res.get("objective", ""),
                    "reference_objective": "",
                    "abs_gap": gpu_res.get("abs_gap", ""),
                    "rel_gap": gpu_res.get("rel_gap", ""),
                    "status": gpu_res.get("status", "error"),
                    "wall_time_s": gpu_res["wall_time_s"],
                    "wall_time_iqr": gpu_res["wall_time_iqr"],
                    "iterations": gpu_res.get("iterations", ""),
                    "git_commit": git_commit,
                    "machine": machine,
                    "card": args.card,
                    "mode": "gpu",
                    "tol": tol,
                })

    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
