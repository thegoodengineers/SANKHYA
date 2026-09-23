#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU PDHG on a datacenter card against the same machine's CPU PDHG (#488).

The crossover runner (gpu_report.py) and the real-instance runner (gpu_real_instances.py)
each measure one card against one host at two tolerances. This one is the protocol for a
RENTED card, where the question is per-card and the answer has to survive a different host
CPU: the card is named in the file, every wall time is the median of `--repeats` runs with
its spread, three tolerances are measured, and one extra pair of runs forces the SAME
iteration count on both sides so the per-iteration ratio is separated from the
tolerance-dependent iteration count.

    python bench/runners/gpu_datacenter.py --binary build_gpu/sankhya --card l4 --repeats 3

Output: bench/results/gpu-datacenter-<card>-<sha>.csv, the sha from the binary (#433).
Columns: instance, instance_sha256, mode (cpu-<N>t | gpu), tol, iteration_limit, status,
objective, iterations, seconds (the solver's own), wall_median_s, wall_spread_s (max-min of
the repeats), repeats, git_commit, machine, card, gpu, timestamp_utc.

The default instances are the Mittelmann ones CPU PDHG finishes inside the time limit
(#446: chromaticindex1024-7, brazil3), read from data/mittelmann after
fetch_mittelmann.py; pass --instances for anything else. No comparator solver runs here:
that is gpu_pdlp_compare.py's job.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import platform
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: the CSV names the commit the BINARY was built from)
from gpu_real_instances import find_mittelmann_mps, gpu_description, run_solve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

COLUMNS = [
    "instance", "instance_sha256", "mode", "tol", "iteration_limit", "status", "objective",
    "iterations", "seconds", "wall_median_s", "wall_spread_s", "repeats", "git_commit",
    "machine", "card", "gpu", "timestamp_utc",
]

# Mittelmann instances CPU PDHG already finishes (#446); the rest hit the limit on both sides
# and would only measure the limit.
DEFAULT_INSTANCES = ["chromaticindex1024-7", "brazil3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repeated(binary: Path, mps: Path, algorithm: str, tol: float, time_limit: float,
             repeats: int, iteration_limit: int | None, threads: int) -> dict:
    """`repeats` solves; the answer of the last one, the median and spread of all walls."""
    walls: list[float] = []
    last: dict = {}
    extra = [f"threads={threads}"] if algorithm == "pdhg-cpu" else []
    if iteration_limit is not None:
        extra.append(f"iteration_limit={iteration_limit}")
    for _ in range(repeats):
        last = run_solve(binary, mps, algorithm, tol, time_limit, extra_options=extra)
        walls.append(float(last["wall"]))
    last["wall_median_s"] = statistics.median(walls)
    last["wall_spread_s"] = max(walls) - min(walls)
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--card", required=True,
                        help="short card name for the filename: l4, a100, h100, ...")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU thread count for the CPU arm")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--tols", default="1e-4,1e-6,1e-8")
    parser.add_argument("--iteration-limit", type=int, default=2000,
                        help="the forced count for the per-iteration pair; 0 skips it")
    parser.add_argument("--instances", nargs="+", type=Path,
                        help="explicit .mps paths (default: the Mittelmann ones PDHG finishes)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.instances:
        instances = list(args.instances)
    else:
        instances = [p for p in (find_mittelmann_mps(n) for n in DEFAULT_INSTANCES) if p]
    if not instances:
        print("no instances: run bench/runners/fetch_mittelmann.py or pass --instances",
              file=sys.stderr)
        return 1
    tols = [float(t) for t in args.tols.split(",")]

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary)
    stamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    out = args.out or RESULTS_DIR / f"gpu-datacenter-{args.card}-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"card {args.card}: {gpu}; solver {commit}; {len(instances)} instance(s), "
          f"{args.repeats} repeat(s), tolerances {args.tols}")
    print(f"{'instance':>24} {'mode':>8} {'tol':>6} {'iters':>8} {'status':>12} "
          f"{'median s':>10} {'spread':>8}")

    def emit(mps: Path, digest: str, mode: str, tol: float, iteration_limit: int | None,
             r: dict) -> dict:
        row = {
            "instance": mps.name.split(".")[0],
            "instance_sha256": digest,
            "mode": mode,
            "tol": f"{tol:g}",
            "iteration_limit": "" if iteration_limit is None else iteration_limit,
            "status": r["status"],
            "objective": "" if r["objective"] is None else repr(r["objective"]),
            "iterations": r["iterations"],
            "seconds": f"{r['seconds']:.6f}",
            "wall_median_s": f"{r['wall_median_s']:.6f}",
            "wall_spread_s": f"{r['wall_spread_s']:.6f}",
            "repeats": args.repeats,
            "git_commit": commit,
            "machine": machine,
            "card": args.card,
            "gpu": gpu,
            "timestamp_utc": stamp_utc,
        }
        print(f"{row['instance']:>24} {mode:>8} {row['tol']:>6} {str(r['iterations']):>8} "
              f"{r['status']:>12} {r['wall_median_s']:>10.3f} {r['wall_spread_s']:>8.3f}")
        return row

    cpu_mode = f"cpu-{args.threads}t"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for mps in instances:
            digest = sha256_file(mps)
            # One untimed warm-up on the card so the first measured run is not paying for
            # context creation and module load (the same as gpu_real_instances.py).
            run_solve(args.binary, mps, "pdhg-cuda", tols[0], args.time_limit)
            plan: list[tuple[float, int | None]] = [(tol, None) for tol in tols]
            if args.iteration_limit > 0:
                plan.append((tols[-1], args.iteration_limit))
            for tol, iteration_limit in plan:
                for algorithm, mode in (("pdhg-cpu", cpu_mode), ("pdhg-cuda", "gpu")):
                    r = repeated(args.binary, mps, algorithm, tol, args.time_limit,
                                 args.repeats, iteration_limit, args.threads)
                    writer.writerow(emit(mps, digest, mode, tol, iteration_limit, r))
                    handle.flush()

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
