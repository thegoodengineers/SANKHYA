#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One card against two for the row-partitioned multi-GPU PDHG (#295).

Every instance is solved in four configurations of the same binary:

    single-engine     gpu_devices=0                      the production single-GPU PDHG
    partitioned-1     gpu_devices=0, gpu_partitioned     the multi-GPU engine on one card:
                                                         the like-for-like baseline
    partitioned-2     gpu_devices=0,1                    two cards, peer-to-peer exchange
    partitioned-2-host gpu_devices=0,1, gpu_peer_access=false   two cards, host-staged

The scaling question is partitioned-2 against partitioned-1: the same arithmetic, split.
single-engine is there so the reader can see what the partitioned engine costs against the
engine a one-card user actually runs. PDHG alone is timed (pdhg_polish=false), from the
solver's own clock, and the partitioned engine's log line "Timing: ... us per attempt on
the cards; evaluation ..." gives the per-step device time with the residual
evaluation and the setup taken out; the single engine prints no such line, so its per-step
figure is its solve time over its iterations and includes its evaluation, its setup and the
CUDA context creation. On a short fixed-step budget that overhead dominates (a 200-step
check on a 2,000-row instance gave 1.6 ms a step against 82 us a step to tolerance), so
compare the single engine on the tolerance budget only.

Two budgets per instance: a fixed step count (--iterations; the per-step time is the number
that scales) and, when --tolerance is given, a solve to that tolerance with
pdhg_stop_at_request=true (the total). Synthetic instances come from generate_large_lp.py,
built backwards from a KKT pair, so their optimum is known exactly; --instance adds MPS files
with --reference objectives where known.

Writes a CSV to bench/results/ (or --out) with the instance's sha256, the objective against
the reference, status, times, iterations, the commit (stamp.py, #433) and the machine tag.
Since the #295 follow-up every cell is the median of --repeats runs (us_per_step_min and
_max beside it), and the partitioned engine's log gives, per row, the partition (rows and
nonzeros per device), the per-card peak device memory, and the exchange's time as a share
of the device time with the bytes it moved - the comm/compute split of a two-card step.

Usage:
    python bench/runners/multi_gpu_scaling.py --binary build/sankhya --machine a100x2 \\
        --sizes 100000,1000000 --nnz-per-col 8 --iterations 2000 --tolerance 1e-4
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_large_lp  # noqa: E402
import gpu_arms  # noqa: E402
import stamp  # noqa: E402
from gpu_real_instances import gpu_description, mps_dimensions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

CONFIGS = {
    "single-engine": ["gpu_devices=0"],
    "partitioned-1": ["gpu_devices=0", "gpu_partitioned=true"],
    "partitioned-2": ["gpu_devices=0,1"],
    "partitioned-2-host": ["gpu_devices=0,1", "gpu_peer_access=false"],
}

COMMON = ["algorithm=pdhg", "gpu=true", "pdhg_polish=false", "presolve=off"]

CSV_COLUMNS = [
    "instance", "sha256", "rows", "cols", "nnz", "config", "budget", "tolerance",
    "status", "objective", "reference_objective", "absolute_gap", "relative_gap",
    "iterations", "step_attempts", "us_per_step", "us_per_step_min", "us_per_step_max",
    "repeats", "host_evaluation_seconds", "exchange_seconds", "exchange_share",
    "exchange_us_per_step", "exchange_mib", "setup_seconds", "partition", "peak_mib",
    "solve_seconds", "wall_seconds", "algorithm", "transport", "git_commit", "machine", "gpu",
    "driver_version", "cuda_runtime", "timestamp_utc",
]

TIMING = re.compile(r"Timing: loop ([\d.]+)s, (\d+) step attempts, ([\d.]+) us per attempt "
                    r"on the cards; (?:host )?evaluation ([\d.]+)s")
EXCHANGE = re.compile(r"exchange ([\d.]+)s \(([\d.]+)% of the device time, ([\d.]+) us per "
                      r"attempt, ([\d.]+) MiB moved over [a-z-]+\); setup ([\d.]+)s")
PARTITION = re.compile(r"device (\d+): rows \[(\d+), (\d+)\), (\d+) nonzeros")
PEAK = re.compile(r"device (\d+) \(slot (\d+)\): peak (\d+) MiB allocated")
TRANSPORT = re.compile(r"\((\d+) devices?, ([a-z-]+) exchange\)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def solve(binary: Path, mps: Path, config: str, options: list[str], time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--time-limit", str(time_limit)]
        for option in COMMON + CONFIGS[config] + options:
            command += ["--option", option]
        started = time.perf_counter()
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - started
        row = {"wall_seconds": f"{wall:.3f}"}
        if not stats.exists():
            row["status"] = "no_output"
            return row
        blob = json.loads(stats.read_text())
        result, effort = blob.get("result", {}), blob.get("effort", {})
        row.update(status=result.get("status", ""), objective=result.get("objective", ""),
                   algorithm=result.get("algorithm", ""), iterations=effort.get("iterations", ""),
                   solve_seconds=effort.get("solve_seconds", ""))
        timing = TIMING.search(run.stdout)
        if timing:
            row.update(step_attempts=timing.group(2), us_per_step=timing.group(3),
                       host_evaluation_seconds=timing.group(4))
        else:
            try:
                row["us_per_step"] = f"{1e6 * float(row['solve_seconds']) / float(row['iterations']):.1f}"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        exchange = EXCHANGE.search(run.stdout)
        if exchange:
            row.update(exchange_seconds=exchange.group(1), exchange_share=exchange.group(2),
                       exchange_us_per_step=exchange.group(3), exchange_mib=exchange.group(4),
                       setup_seconds=exchange.group(5))
        # "0:250000r/1000000nz;1:250000r/1000000nz" and "0:1234;1:1230" (MiB per slot).
        row["partition"] = ";".join(f"{d}:{int(b) - int(a)}r/{nz}nz"
                                    for d, a, b, nz in PARTITION.findall(run.stdout))
        row["peak_mib"] = ";".join(f"{slot}:{mib}" for _, slot, mib in PEAK.findall(run.stdout))
        transport = TRANSPORT.search(result.get("message", ""))
        row["transport"] = transport.group(2) if transport else ""
        return row


def solve_repeated(binary: Path, mps: Path, config: str, options: list[str],
                   time_limit: float, repeats: int) -> dict:
    """The row of the run nearest the median us_per_step, with the min and max of all."""
    rows = [solve(binary, mps, config, options, time_limit) for _ in range(max(1, repeats))]
    steps = [float(r["us_per_step"]) for r in rows if r.get("us_per_step")]
    if not steps:
        return dict(rows[-1], repeats=len(rows))
    median = sorted(steps)[len(steps) // 2]
    row = next(r for r in rows if r.get("us_per_step") and float(r["us_per_step"]) == median)
    return dict(row, us_per_step_min=f"{min(steps):.1f}", us_per_step_max=f"{max(steps):.1f}",
                repeats=len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--machine", required=True, help="machine tag written to every row")
    parser.add_argument("--sizes", default="", help="comma-separated synthetic sizes (rows = cols)")
    parser.add_argument("--nnz-per-col", type=int, default=8)
    parser.add_argument("--seed", type=int, default=295)
    parser.add_argument("--instance", type=Path, action="append", default=[],
                        help="an MPS file to add (repeatable)")
    parser.add_argument("--reference", type=float, action="append", default=[],
                        help="reference objective for each --instance, in order (optional)")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="fixed step budget for the per-step timing")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="also solve to this tolerance (0: skip)")
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per cell; the row is the median us_per_step with min and max")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--workdir", type=Path, default=Path(tempfile.gettempdir()) / "sankhya-mgpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = stamp.stamp(args.binary)
    args.workdir.mkdir(parents=True, exist_ok=True)
    instances: list[tuple[str, Path, float | None, int, int, int]] = []
    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        path = args.workdir / f"kkt_{size}_{args.nnz_per_col}_{args.seed}.mps"
        if path.exists():
            # A file generate_large_lp.py wrote earlier: its header carries the optimum.
            with path.open(encoding="utf-8") as handle:
                header = [next(handle) for _ in range(8)]
            found = [ln.split("=", 1)[1] for ln in header if "analytic optimum" in ln]
            instances.append((path.stem, path, float(found[0]) if found else None, size, size, 0))
            continue
        built = generate_large_lp.build(size, size, args.nnz_per_col, args.seed)
        path.write_text("\n".join(built["lines"]) + "\n", encoding="utf-8", newline="\n")
        instances.append((path.stem, path, float(built["optimal_objective"]), size, size,
                          built["nonzeros"]))
    for k, path in enumerate(args.instance):
        ref = args.reference[k] if k < len(args.reference) else None
        instances.append((path.stem, path, ref, *mps_dimensions(path)))
    gpu = gpu_description(args.binary)
    platform_cells = gpu_arms.platform_cells(gpu)

    out = args.out or RESULTS_DIR / f"multi-gpu-{args.machine}-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    budgets = [("steps", [f"iteration_limit={args.iterations}"], "")]
    if args.tolerance > 0:
        budgets.append(("tolerance", [f"pdhg_tolerance={args.tolerance:g}",
                                      "pdhg_stop_at_request=true"], f"{args.tolerance:g}"))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for name, path, ref, rows, cols, nnz in instances:
            digest = sha256(path)
            for budget, options, tol in budgets:
                for config in [c for c in args.configs.split(",") if c]:
                    row = solve_repeated(args.binary, path, config, options, args.time_limit,
                                         args.repeats)
                    obj = row.get("objective", "")
                    gap_abs = gap_rel = ""
                    if ref is not None and obj not in ("", None):
                        gap_abs = abs(float(obj) - ref)
                        gap_rel = gap_abs / max(1.0, abs(ref))
                    row.update(instance=name, sha256=digest, rows=rows or "", cols=cols or "",
                               nnz=nnz or "", config=config, budget=budget, tolerance=tol,
                               reference_objective="" if ref is None else ref,
                               absolute_gap=gap_abs, relative_gap=gap_rel, git_commit=commit,
                               machine=args.machine, gpu=gpu, **platform_cells,
                               timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"))
                    writer.writerow(row)
                    handle.flush()
                    print(f"{name:>24} {budget:>9} {config:>18}: {row.get('status', ''):>16} "
                          f"obj {obj!s:>22} it {row.get('iterations', '')!s:>7} "
                          f"{row.get('us_per_step', '')!s:>8} us/step "
                          f"solve {row.get('solve_seconds', '')!s:>10}s "
                          f"{row.get('transport', '')}", flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
