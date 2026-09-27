#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The interior point's normal equations on cuDSS against the CPU factor, on #488's instances
(#489): the synthetic KKT ladder, the refinery planning year and the Mittelmann instances
PDHG finishes, each solved with `algorithm=ipm` to the project standard (relative gap 1e-8,
feasibility 1e-7; the interior point has no looser setting) with `ipm_linear_solver=cpu`
and `=cudss`.

Every row carries the instance's sha256, the reference objective (the generator's analytic
optimum for the synthetic models, HiGHS in its own process for the Mittelmann files) with the
absolute and relative gap to it, the verifier's verdict on the written solution, the solver's
own clock and the wall, the commit the BINARY was built from (#433), a machine tag that names
the card, the card as the binary describes it, and the driver and CUDA runtime (#488).

    python bench/runners/ipm_cudss.py --binary build/sankhya --card a100

writes bench/results/ipm-cudss-<card>-<sha>.csv, which docs/BENCHMARKS.md section 1g.7
reads. A build without cuDSS makes the cudss leg warn and run the CPU factor: the runner
refuses to write a CSV whose two legs ran the same factor, by reading the log for the line
the engine prints when the device is used.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402
import gpu_report  # noqa: E402  (the synthetic ladder)
import stamp  # noqa: E402
from gpu_real_instances import find_mittelmann_mps, gpu_description, mps_dimensions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

COLUMNS = [
    "instance", "instance_sha256", "rows", "cols", "nnz", "linear_solver", "device_used",
    "tolerance", "status", "objective", "reference_objective", "reference_source",
    "reference_seconds", "abs_gap", "rel_gap", "independently_verified", "verifier_message",
    "iterations", "seconds", "wall_seconds", "git_commit", "machine", "gpu", "driver_version",
    "cuda_runtime", "timestamp_utc", "solver_options",
]
LEGS = ["cpu", "cudss"]
# The interior point has no tolerance option: it stops at the project standard
# (include/sankhya/tolerances.hpp: relative gap 1e-8, primal and dual feasibility 1e-7), and
# that is what the tolerance column records.
TOLERANCE = 1e-8
MITTELMANN_INSTANCES = ["chromaticindex1024-7", "brazil3"]
DEVICE_LINE = re.compile(r"factored on the device by cuDSS")


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_solve(binary: Path, mps: Path, time_limit: float, options: list[str]) -> dict:
    """One solve, its solution verified by tools/verify_solution.py, and whether the log
    says the device factored the normal equations."""
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        sol = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit)]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - started
        out = {"status": "no_output", "objective": None, "iterations": "", "seconds": wall,
               "wall": wall, "verified": "", "verifier_message": "",
               "device_used": int(bool(DEVICE_LINE.search(done.stdout + done.stderr)))}
        if not stats.exists():
            return out
        blob = json.loads(stats.read_text())
        result, effort = blob.get("result", {}), blob.get("effort", {})
        solver = as_number(effort.get("solve_seconds"))
        out.update(status=result.get("status", "unknown"), objective=as_number(result.get("objective")),
                   iterations=effort.get("iterations", ""),
                   seconds=wall if solver is None else solver)
        if sol.exists() and out["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                                   capture_output=True, text=True, check=False)
            out["verified"] = 1 if check.returncode == 0 else 0
            if check.returncode != 0:
                failing = [ln.strip() for ln in check.stdout.splitlines() if "[FAIL]" in ln]
                out["verifier_message"] = "; ".join(failing)[:300]
        return out


def build_row(name: str, dims: tuple, digest: str, leg: str, tolerance: float, result: dict,
              reference: float | None, source: str, reference_seconds: float | None,
              fixed: dict, options: list[str]) -> dict:
    absolute, relative = gpu_arms.gaps(result["objective"], reference)
    row = {
        "instance": name, "instance_sha256": digest,
        "rows": dims[0], "cols": dims[1], "nnz": dims[2],
        "linear_solver": leg, "device_used": result["device_used"],
        "tolerance": f"{tolerance:.0e}", "status": result["status"],
        "objective": "" if result["objective"] is None else repr(result["objective"]),
        "reference_objective": "" if reference is None else repr(reference),
        "reference_source": source if reference is not None else "none",
        "reference_seconds": "" if reference_seconds is None else f"{reference_seconds:.6f}",
        "abs_gap": gpu_arms.fmt_gap(absolute), "rel_gap": gpu_arms.fmt_gap(relative),
        "independently_verified": result["verified"],
        "verifier_message": result["verifier_message"],
        "iterations": result["iterations"], "seconds": round(result["seconds"], 6),
        "wall_seconds": round(result["wall"], 6), "solver_options": " ".join(options),
    }
    row.update(fixed)
    return row


def instances(args) -> list[tuple[str, Path, float | None]]:
    """(name, path, analytic optimum or None), the synthetic ones written into args.workdir."""
    out = []
    if not args.skip_ladder:
        for rows, cols, nnz in gpu_report.SIZES:
            path = args.workdir / f"kkt_{rows}x{cols}.mps"
            out.append((f"kkt_{rows}x{cols}", path,
                        gpu_report.generate_lp(rows, cols, nnz, args.seed, path)))
    if not args.skip_refinery:
        path = args.workdir / "refinery_year.mps"
        gen = subprocess.run(
            [sys.executable, str(REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"),
             "--periods", "8760", "--seed", "42", "--out", str(path)],
            capture_output=True, text=True, check=False)
        if gen.returncode == 0 and path.exists():
            out.append(("refinery_year", path, None))  # its optimum is in the file
        else:
            print(f"refinery year not generated: {gen.stderr[:200]}", file=sys.stderr)
    for name in MITTELMANN_INSTANCES:
        path = find_mittelmann_mps(name)
        if path is None:
            print(f"SKIP {name}: run fetch_mittelmann.py first", file=sys.stderr)
            continue
        out.append((name, path, None))
    for path in args.instance:
        out.append((path.stem, path, None))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--card", required=True, help="short card name: a100, l4, ...")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-time-limit", type=float, default=600.0)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-refinery", action="store_true")
    parser.add_argument("--instance", type=Path, action="append", default=[])
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="added to both legs, e.g. crossover=false")
    parser.add_argument("--workdir", type=Path, default=Path(tempfile.gettempdir()) / "sankhya-cudss")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    commit = stamp.stamp(args.binary)
    gpu = gpu_description(args.binary)
    fixed = {"git_commit": commit,
             "machine": f"{platform.system()}-{platform.machine()}-{args.card}", "gpu": gpu,
             **gpu_arms.platform_cells(gpu),
             "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    out = args.out or RESULTS_DIR / f"ipm-cudss-{args.card}-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"card {args.card}: {gpu}; solver {commit}; tolerance {TOLERANCE:g} (project standard); "
          f"time limit {args.time_limit:g}s")
    print(f"{'instance':>22} {'factor':>6} {'status':>12} {'verified':>8} {'iters':>6} "
          f"{'solver s':>10} {'rel gap':>10}  speedup")

    rows: list[dict] = []
    device_rows = 0
    for name, mps, analytic in instances(args):
        digest = gpu_arms.sha256_file(mps)
        dims = mps_dimensions(mps)
        if analytic is not None:
            reference, source, ref_seconds = analytic, "construction", None
        else:
            reference, source, _, ref_seconds = gpu_arms.reference_objective(
                mps, args.reference_time_limit, use_highs=not args.no_reference)
        cpu_seconds = None
        for leg in LEGS:
            # The log stays on: the device_used column reads the engine's own line.
            options = ["algorithm=ipm", f"ipm_linear_solver={leg}", *args.solver_option]
            result = run_solve(args.binary, mps, args.time_limit, options)
            row = build_row(name, dims, digest, leg, TOLERANCE, result, reference, source,
                            ref_seconds, fixed, options)
            rows.append(row)
            device_rows += result["device_used"]
            if leg == "cpu":
                cpu_seconds = result["seconds"]
                versus = "baseline"
            else:
                versus = gpu_arms.speedup(cpu_seconds, result["seconds"])
            print(f"{name:>22} {leg:>6} {result['status']:>12} {str(result['verified']):>8} "
                  f"{str(result['iterations']):>6} {result['seconds']:>10.3f} "
                  f"{row['rel_gap'] or '-':>10}  {versus}", flush=True)
    if device_rows == 0:
        print("no solve used the device: this binary has no cuDSS, or no card answered; "
              "no CSV written", file=sys.stderr)
        return 1
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
