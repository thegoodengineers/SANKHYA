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

The CPU side has two arms (bench/runners/gpu_arms.py): `cpu-1t`, one thread and the serial
A x - the default configuration - and `cpu-<N>t`, `--cpu-threads N` workers with
`pdhg_parallel_spmv=true`, so A x is row-parallel as well as A^T y. The first file this
runner wrote (gpu-datacenter-l4-fdc89c5.csv) passed `threads=16` alone: its "cpu-16t" rows
parallelised A^T y and left A x serial. Every GPU row's speedup is printed against both
arms, and each instance carries one reference objective (HiGHS, separate process) with the
absolute and relative gap of every answer to it.

Output: bench/results/gpu-datacenter-<card>-<sha>.csv, the sha from the binary (#433).
Columns: instance, instance_sha256, mode (the arm: cpu-1t | cpu-<N>t | gpu), cpu_threads,
parallel_spmv, tol, iteration_limit, status, objective, reference_objective,
reference_source (highs | construction | none), abs_gap, rel_gap (|obj-ref|/max(1,|ref|)),
primal_residual, dual_residual (absolute, of the last repeat), iterations, seconds (the
solver's own, last repeat), solver_median_s (the solver's own, median of the repeats),
wall_median_s, wall_spread_s (max-min of the repeats), repeats, git_commit, machine, card,
gpu, timestamp_utc.

The default instances are the Mittelmann ones CPU PDHG finishes inside the time limit
(#446: chromaticindex1024-7, brazil3), read from data/mittelmann after
fetch_mittelmann.py; pass --instances for anything else. No comparator solver's TIME is
measured here: that is gpu_pdlp_compare.py's job.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402  (#488: the CPU arms, the reference and the gap arithmetic)
import stamp  # noqa: E402  (#433: the CSV names the commit the BINARY was built from)
from gpu_real_instances import find_mittelmann_mps, gpu_description, run_solve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

COLUMNS = [
    "instance", "instance_sha256", "mode", "cpu_threads", "parallel_spmv", "tol",
    "iteration_limit", "status", "objective", "reference_objective", "reference_source",
    "abs_gap", "rel_gap", "primal_residual", "dual_residual", "iterations", "seconds",
    "solver_median_s", "wall_median_s", "wall_spread_s", "repeats", "git_commit", "machine",
    "card", "gpu", "timestamp_utc",
]

# Mittelmann instances CPU PDHG already finishes (#446); the rest hit the limit on both sides
# and would only measure the limit.
DEFAULT_INSTANCES = ["chromaticindex1024-7", "brazil3"]

# Kept for callers that imported it from here before gpu_arms existed.
sha256_file = gpu_arms.sha256_file


def repeated(binary: Path, mps: Path, arm: tuple, tol: float, time_limit: float,
             repeats: int, iteration_limit: int | None) -> dict:
    """`repeats` solves of one arm; the answer of the last one, the median and spread of
    all walls, and the median of the solver's own clock."""
    walls: list[float] = []
    solver: list[float] = []
    last: dict = {}
    extra = list(arm[2])
    if iteration_limit is not None:
        extra.append(f"iteration_limit={iteration_limit}")
    for _ in range(repeats):
        last = run_solve(binary, mps, arm[1], tol, time_limit, extra_options=extra)
        walls.append(float(last["wall"]))
        solver.append(float(last["seconds"]))
    last["wall_median_s"] = statistics.median(walls)
    last["wall_spread_s"] = max(walls) - min(walls)
    last["solver_median_s"] = statistics.median(solver)
    return last


def build_row(name: str, digest: str, arm: tuple, tol: float, iteration_limit: int | None,
              r: dict, reference: float | None, reference_source: str, fixed: dict) -> dict:
    """One CSV row. Pure, so test_gpu_runners.py can pin it with a synthetic result."""
    cells = gpu_arms.fairness_cells(arm, digest, r["objective"], reference, reference_source)
    row = {
        "instance": name,
        "instance_sha256": digest,
        "mode": cells["arm"],
        "cpu_threads": cells["cpu_threads"],
        "parallel_spmv": cells["parallel_spmv"],
        "tol": f"{tol:g}",
        "iteration_limit": "" if iteration_limit is None else iteration_limit,
        "status": r["status"],
        "objective": "" if r["objective"] is None else repr(r["objective"]),
        "reference_objective": cells["reference_objective"],
        "reference_source": cells["reference_source"],
        "abs_gap": cells["abs_gap"],
        "rel_gap": cells["rel_gap"],
        "primal_residual": r.get("primal_residual", ""),
        "dual_residual": r.get("dual_residual", ""),
        "iterations": r["iterations"],
        "seconds": f"{r['seconds']:.6f}",
        "solver_median_s": f"{r['solver_median_s']:.6f}",
        "wall_median_s": f"{r['wall_median_s']:.6f}",
        "wall_spread_s": f"{r['wall_spread_s']:.6f}",
    }
    row.update(fixed)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--card", required=True,
                        help="short card name for the filename: l4, a100, h100, ...")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", "--threads", dest="cpu_threads", type=int, default=16,
                        help="threads of the parallel CPU arm, run with pdhg_parallel_spmv=true; "
                             "the serial one-thread arm always runs as well (1 drops the "
                             "parallel arm)")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--tols", default="1e-4,1e-6,1e-8")
    parser.add_argument("--iteration-limit", type=int, default=2000,
                        help="the forced count for the per-iteration pair; 0 skips it")
    parser.add_argument("--reference-time-limit", type=float, default=600.0,
                        help="HiGHS's own limit for the reference objective, once per instance")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip HiGHS; the reference is blank unless the file states one")
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
    fixed = {"repeats": args.repeats, "git_commit": commit, "machine": machine,
             "card": args.card, "gpu": gpu, "timestamp_utc": stamp_utc}
    arms = gpu_arms.all_arms(args.cpu_threads)

    print(f"card {args.card}: {gpu}; solver {commit}; {len(instances)} instance(s), "
          f"{args.repeats} repeat(s), tolerances {args.tols}; arms {', '.join(a[0] for a in arms)}")
    print(f"{'instance':>24} {'mode':>8} {'tol':>6} {'iters':>8} {'status':>15} "
          f"{'solver s':>10} {'median s':>10} {'spread':>8} {'rel gap':>10}  speedup (solver)")

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for mps in instances:
            name = mps.name.split(".")[0]
            digest = gpu_arms.sha256_file(mps)
            reference, source, note = gpu_arms.reference_objective(
                mps, args.reference_time_limit, use_highs=not args.no_reference)
            print(f"{name}: sha256 {digest[:16]}..., reference {reference!r} ({source}; {note})")
            # One untimed warm-up on the card so the first measured run is not paying for
            # context creation and module load (the same as gpu_real_instances.py).
            run_solve(args.binary, mps, "pdhg-cuda", tols[0], args.time_limit)
            plan: list[tuple[float, int | None]] = [(tol, None) for tol in tols]
            if args.iteration_limit > 0:
                plan.append((tols[-1], args.iteration_limit))
            for tol, iteration_limit in plan:
                cpu_solver: dict[str, float] = {}
                for arm in arms:
                    r = repeated(args.binary, mps, arm, tol, args.time_limit, args.repeats,
                                 iteration_limit)
                    row = build_row(name, digest, arm, tol, iteration_limit, r, reference,
                                    source, fixed)
                    if arm[1] == "pdhg-cpu":
                        cpu_solver[arm[0]] = r["solver_median_s"]
                        versus = "baseline"
                    else:
                        versus = "  ".join(
                            f"{gpu_arms.speedup(s, r['solver_median_s'])} vs {cpu}"
                            for cpu, s in cpu_solver.items())
                    print(f"{name:>24} {arm[0]:>8} {row['tol']:>6} {str(r['iterations']):>8} "
                          f"{r['status']:>15} {r['solver_median_s']:>10.3f} "
                          f"{r['wall_median_s']:>10.3f} {r['wall_spread_s']:>8.3f} "
                          f"{row['rel_gap'] or '-':>10}  {versus}")
                    writer.writerow(row)
                    handle.flush()

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
