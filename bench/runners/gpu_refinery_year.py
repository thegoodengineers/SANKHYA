#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The refinery year on the card, to a verified vertex, end to end (#888).

THE QUESTION is whether the GPU solves the 779,640-row refinery year
(generate_refinery_lp.py --periods 8760 --seed 42, the hourly plan, optimum exact by
construction) to an answer tools/verify_solution.py accepts, and in how long against a
simplex on the same host. Every earlier GPU row on this model ended `time_limit` at 300 s on
both cards (gpu-real-a100-fdd1f35.csv, gpu-real-l4-58a8374.csv) although the card reached a
relative KKT error of 1e-4 in 3 s and 1e-8 in 97 s: PDHG's stopping test demands ABSOLUTE
primal feasibility to 1e-7 as well, and on this model's right-hand sides the iterates never
get there (the diagnosis is in the issue). So the route measured here is the one a first-
order answer takes to an exact one: PDHG on the card to the relative request, then the
crossover from that point (`pdhg_crossover`, #888), then the verifier on the vertex.

THE ARMS, each a separate `sankhya solve` with the .sol written and verified, repeated
`--repeats` times with the median (min, max) recorded:

    gpu-pdhg-1e4-xover   gpu=true pdhg_tolerance=1e-4 pdhg_stop_at_request=true pdhg_crossover=true
    gpu-pdhg-1e8-xover   the same from the 1e-8 point
    gpu-pdhg-1e8         the card alone at 1e-8, no crossover (what the earlier rows measured)
    cpu-pdhg-<N>t-1e4-xover  the CPU engine on N threads, same route, the host's own PDHG
    highs-ipm-1t, highs-ipm-<N>t, highs-simplex-1t, highs-simplex-<N>t
                         HiGHS (highspy, a separate process; PROVENANCE row highspy) with
                         its interior point plus crossover and its dual simplex, once each

The reference objective is the generator's analytic optimum. The CSV carries the sha256 of
the instance file, the commit the binary was built from (stamp.py), the card, driver and
CUDA runtime, and the host's CPU, cores and RAM. Rows are appended and flushed one at a time
and an (instance, arm) already in the file is not rerun, so a killed run resumes.

Usage:
    python bench/runners/gpu_refinery_year.py --binary build_gpu/sankhya --card a100 \\
        --keep /scratch/refinery --cpu-threads 16 --time-limit 1800
    # also the million-row refinery (--periods 8760 --crudes 56 --seed 7, million.py's):
    python bench/runners/gpu_refinery_year.py ... --instances refinery_year million-refinery
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402
import rivals  # noqa: E402  (write_sol for HiGHS's answer)
import stamp  # noqa: E402
from gpu_real_instances import gpu_description, mps_dimensions  # noqa: E402
from million import machine_tag  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
GENERATOR = REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"

INSTANCES = {
    "refinery_year": ["--periods", "8760", "--seed", "42"],
    "million-refinery": ["--periods", "8760", "--crudes", "56", "--seed", "7"],
    # a small stand-in for exercising the runner without a card or a long wait
    "refinery_365": ["--periods", "365", "--seed", "42"],
    "refinery_12": ["--periods", "12", "--seed", "42"],
}

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "cols", "nnz", "arm", "engine", "threads",
    "tolerance", "status", "algorithm_ran", "objective", "reference_objective", "abs_gap",
    "rel_gap", "iterations", "restarts", "crossover_pivots", "kkt_1e4_seconds",
    "kkt_1e6_seconds", "kkt_1e8_seconds", "seconds", "seconds_min", "seconds_max", "repeats",
    "wall_seconds", "independently_verified", "verifier_message", "message", "time_limit",
    "solver_options", "git_commit", "machine", "host", "gpu", "driver_version",
    "cuda_runtime", "timestamp_utc",
]

PDHG_COMMON = ["log_to_console=false", "algorithm=pdhg", "pdhg_polish=false",
               "pdhg_stop_at_request=true"]


def arms(cpu_threads: int, gpu: bool) -> list[tuple[str, str, list[str]]]:
    """(arm, engine, solver options). The HiGHS arms carry `solver=` and `threads=` in the
    same slot; run_highs reads them."""
    out = []
    if gpu:
        out += [("gpu-pdhg-1e4-xover", "sankhya",
                 PDHG_COMMON + ["gpu=true", "pdhg_tolerance=1e-4", "pdhg_crossover=true"]),
                ("gpu-pdhg-1e8-xover", "sankhya",
                 PDHG_COMMON + ["gpu=true", "pdhg_tolerance=1e-8", "pdhg_crossover=true"]),
                ("gpu-pdhg-1e8", "sankhya", PDHG_COMMON + ["gpu=true", "pdhg_tolerance=1e-8"])]
    cpu = [f"threads={cpu_threads}", "pdhg_parallel_spmv=true"] if cpu_threads > 1 else []
    out.append((f"cpu-pdhg-{cpu_threads}t-1e4-xover", "sankhya",
                PDHG_COMMON + cpu + ["pdhg_tolerance=1e-4", "pdhg_crossover=true"]))
    for threads in sorted({1, cpu_threads}):
        for solver in ("ipm", "simplex"):
            out.append((f"highs-{solver}-{threads}t", "highs",
                        [f"solver={solver}", f"threads={threads}"]))
    return out


def tolerance_of(options: list[str]) -> str:
    for option in options:
        if option.startswith("pdhg_tolerance="):
            return f"{float(option.split('=', 1)[1]):.0e}"
    return ""


def threads_of(options: list[str]) -> str:
    for option in options:
        if option.startswith("threads="):
            return option.split("=", 1)[1]
    return "1"


RESTARTS = re.compile(r"after (\d+) iterations and (\d+) restarts")
PIVOTS = re.compile(r"crossover: (\d+) pivots")


def verify(mps: Path, sol: Path) -> tuple[int, str]:
    check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                           capture_output=True, text=True, check=False)
    failing = [ln.strip() for ln in check.stdout.splitlines() if "[FAIL]" in ln]
    return (1 if check.returncode == 0 else 0), "; ".join(failing)[:300]


def run_sankhya(binary: Path, mps: Path, options: list[str], time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats, sol = Path(tmp) / "stats.json", Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit)]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "seconds": wall, "wall": wall,
                    "message": done.stderr.strip()[-300:]}
        blob = json.loads(stats.read_text())
        effort, result = blob.get("effort", {}), blob.get("result", {})
        message = result.get("message", "")
        restarts = RESTARTS.search(message)
        pivots = PIVOTS.search(message)
        out = {"status": result.get("status", "unknown"), "objective": result.get("objective"),
               "algorithm_ran": result.get("algorithm", ""),
               "iterations": effort.get("iterations", ""),
               "restarts": restarts.group(2) if restarts else "",
               "crossover_pivots": pivots.group(1) if pivots else "",
               "kkt_1e4_seconds": effort.get("kkt_1e4_seconds", ""),
               "kkt_1e6_seconds": effort.get("kkt_1e6_seconds", ""),
               "kkt_1e8_seconds": effort.get("kkt_1e8_seconds", ""),
               "seconds": float(effort.get("solve_seconds", wall)), "wall": wall,
               "message": message, "verified": "", "verifier_message": ""}
        if sol.exists() and out["status"] in ("optimal", "feasible"):
            out["verified"], out["verifier_message"] = verify(mps, sol)
        return out


def highs_worker(mps: Path, solver: str, threads: int, time_limit: float, sol: Path) -> dict:
    """Inside the child: highspy, the documented option names, its answer into our .sol."""
    import highspy  # noqa: PLC0415 - external solver, never a build dependency
    h = highspy.Highs()
    for key, value in (("output_flag", False), ("time_limit", float(time_limit)),
                       ("solver", solver), ("threads", int(threads)),
                       ("primal_feasibility_tolerance", rivals.FEASIBILITY_TOLERANCE),
                       ("dual_feasibility_tolerance", rivals.FEASIBILITY_TOLERANCE)):
        h.setOptionValue(key, value)
    h.readModel(str(mps))
    h.run()
    status = h.modelStatusToString(h.getModelStatus()).strip().lower()
    info, lp, solution = h.getInfo(), h.getLp(), h.getSolution()
    mapped = {"optimal": "optimal", "time limit reached": "time_limit",
              "iteration limit reached": "iteration_limit"}.get(status, status)
    out = {"status": mapped, "message": status, "seconds": h.getRunTime(),
           "iterations": info.simplex_iteration_count + max(info.ipm_iteration_count, 0),
           "crossover_pivots": (info.crossover_iteration_count
                                if hasattr(info, "crossover_iteration_count") else ""),
           "objective": info.objective_function_value if mapped == "optimal" else None,
           "algorithm_ran": f"highs-{solver}", "version": h.version()}
    if mapped == "optimal" and solution.value_valid:
        duals = bool(solution.dual_valid)
        col_value, col_dual = list(solution.col_value), list(solution.col_dual)
        row_value, row_dual = list(solution.row_value), list(solution.row_dual)
        rivals.write_sol(sol, solver="highs", status="optimal" if duals else "feasible",
                         objective=out["objective"],
                         columns={n: (col_value[j], col_dual[j] if duals else 0.0)
                                  for j, n in enumerate(lp.col_names_)},
                         rows={n: (row_value[i], row_dual[i] if duals else 0.0)
                               for i, n in enumerate(lp.row_names_)})
    return out


def run_highs(mps: Path, options: list[str], time_limit: float) -> dict:
    solver = options[0].split("=", 1)[1]
    threads = int(threads_of(options))
    with tempfile.TemporaryDirectory() as tmp:
        sol, report = Path(tmp) / "highs.sol", Path(tmp) / "report.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--highs-worker", str(mps),
                   solver, str(threads), repr(float(time_limit)), str(sol), str(report)]
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        started = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True, env=env,
                                  timeout=time_limit + rivals.HANG_MARGIN_SECONDS + 900.0)
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - started
            return {"status": "hung", "objective": None, "seconds": wall, "wall": wall,
                    "message": "no exit"}
        wall = time.perf_counter() - started
        if not report.exists():
            return {"status": "crashed", "objective": None, "seconds": wall, "wall": wall,
                    "message": (done.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        out = json.loads(report.read_text(encoding="utf-8"))
        out["wall"] = wall
        out["verified"], out["verifier_message"] = "", ""
        if sol.exists():
            out["verified"], out["verifier_message"] = verify(mps, sol)
        return out


def repeated(run, repeats: int) -> dict:
    results = [run() for _ in range(max(1, repeats))]
    seconds = [r["seconds"] for r in results]
    median = statistics.median(seconds)
    result = dict(min(results, key=lambda r: abs(r["seconds"] - median)))
    result.update(seconds=median, seconds_min=min(seconds), seconds_max=max(seconds),
                  repeats=len(results))
    return result


def build_row(name: str, dims: tuple, digest: str, arm: tuple, result: dict,
              reference: float | None, time_limit: float, stamp_fields: dict) -> dict:
    """One CSV row; pure, so test_mittelmann_cuopt.py pins it."""
    label, engine, options = arm
    absolute, relative = gpu_arms.gaps(result.get("objective"), reference)
    row = {
        "instance": name, "instance_sha256": digest, "rows": dims[0], "cols": dims[1],
        "nnz": dims[2], "arm": label, "engine": engine, "threads": threads_of(options),
        "tolerance": tolerance_of(options), "status": result.get("status", "unknown"),
        "algorithm_ran": result.get("algorithm_ran", ""),
        "objective": "" if result.get("objective") is None else repr(result["objective"]),
        "reference_objective": "" if reference is None else repr(reference),
        "abs_gap": gpu_arms.fmt_gap(absolute), "rel_gap": gpu_arms.fmt_gap(relative),
        "iterations": result.get("iterations", ""), "restarts": result.get("restarts", ""),
        "crossover_pivots": result.get("crossover_pivots", ""),
        "kkt_1e4_seconds": result.get("kkt_1e4_seconds", ""),
        "kkt_1e6_seconds": result.get("kkt_1e6_seconds", ""),
        "kkt_1e8_seconds": result.get("kkt_1e8_seconds", ""),
        "seconds": round(float(result.get("seconds", 0.0)), 6),
        "seconds_min": round(float(result.get("seconds_min", result.get("seconds", 0.0))), 6),
        "seconds_max": round(float(result.get("seconds_max", result.get("seconds", 0.0))), 6),
        "repeats": result.get("repeats", 1),
        "wall_seconds": round(float(result.get("wall", result.get("seconds", 0.0))), 6),
        "independently_verified": result.get("verified", ""),
        "verifier_message": result.get("verifier_message", ""),
        "message": str(result.get("message", ""))[:300], "time_limit": time_limit,
        "solver_options": " ".join(options),
    }
    row.update(stamp_fields)
    return row


def generate(name: str, directory: Path) -> Path:
    path = directory / f"{name}.mps"
    if not path.exists():
        print(f"generating {name} ({' '.join(INSTANCES[name])})...")
        done = subprocess.run([sys.executable, str(GENERATOR), *INSTANCES[name], "--out",
                               str(path)], capture_output=True, text=True, check=False)
        if done.returncode != 0 or not path.exists():
            raise SystemExit(f"generator failed on {name}: {done.stderr[-400:]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--highs-worker", nargs=6, metavar="ARG", help=argparse.SUPPRESS)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--card", default="")
    parser.add_argument("--keep", type=Path, default=None,
                        help="directory holding the generated instances (reused on a rerun)")
    parser.add_argument("--instances", nargs="+", default=["refinery_year"],
                        choices=sorted(INSTANCES))
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3,
                        help="solves per SANKHYA arm (median, min, max); HiGHS runs once")
    parser.add_argument("--no-gpu", action="store_true", help="CPU and HiGHS arms only")
    parser.add_argument("--no-highs", action="store_true")
    parser.add_argument("--host-label", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.highs_worker:
        mps, solver, threads, time_limit, sol, report = args.highs_worker
        out = highs_worker(Path(mps), solver, int(threads), float(time_limit), Path(sol))
        Path(report).write_text(json.dumps(out), encoding="utf-8")
        return 0
    if args.binary is None:
        parser.error("--binary is required")

    commit = stamp.stamp(args.binary)
    gpu = gpu_description(args.binary)
    stamp_fields = {"git_commit": commit,
                    "machine": f"{platform.system()}-{platform.machine()}",
                    "host": machine_tag(args.host_label), "gpu": gpu,
                    **gpu_arms.platform_cells(gpu),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds")}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"gpu-refinery-year-{args.card or 'unknown'}-{commit}.csv"
    done_keys: set[tuple] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as handle:
            done_keys = {(r["instance"], r["arm"]) for r in csv.DictReader(handle)}
        print(f"resuming {out}: {len(done_keys)} rows already there")
    handle = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    if not done_keys:
        writer.writeheader()
        handle.flush()
    print(f"refinery year on the card (#888): commit {commit}  gpu {gpu}  host "
          f"{stamp_fields['host']}  limit {args.time_limit:g} s  repeats {args.repeats}")
    print(f"{'instance':>18} {'arm':>24} {'status':>12} {'seconds':>9} {'rel gap':>10} "
          f"{'verified':>8}  message")

    keep = args.keep or Path(tempfile.mkdtemp(prefix="refinery-"))
    keep.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for name in args.instances:
        mps = generate(name, keep)
        digest, dims = gpu_arms.sha256_file(mps), mps_dimensions(mps)
        reference = gpu_arms.analytic_optimum(mps)
        for arm in arms(args.cpu_threads, not args.no_gpu):
            label, engine, options = arm
            if (name, label) in done_keys or (engine == "highs" and args.no_highs):
                continue
            if engine == "sankhya":
                result = repeated(lambda: run_sankhya(args.binary, mps, options,
                                                      args.time_limit), args.repeats)
            else:
                result = run_highs(mps, options, args.time_limit)
            row = build_row(name, dims, digest, arm, result, reference, args.time_limit,
                            stamp_fields)
            writer.writerow(row)
            handle.flush()
            wrote += 1
            print(f"{name:>18} {label:>24} {row['status']:>12} {row['seconds']:>9.2f} "
                  f"{row['rel_gap'] or '-':>10} {str(row['independently_verified']):>8}  "
                  f"{row['message'][:70]}")
    handle.close()
    print(f"\nwrote {out} ({wrote} new rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
