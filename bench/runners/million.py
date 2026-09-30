#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A million rows, solved and verified, or the reason why not (#751).

Three generated families at 1,000,000 rows or more, each with its optimum exact by
construction and each written deterministically from a seed:

    refinery   bench/runners/generate_refinery_lp.py, the hourly year (8,760 periods) with
               enough crudes that the rows pass a million - the industrial shape
    staircase  bench/runners/generate_large_lp.py --structure staircase, a multi-period
               planning band
    transport  bench/runners/generate_transport_lp.py, plants to depots over a delivery
               window, whose normal equations stay a band

Each instance is solved by each ARM - an engine and its route: the interior point then
crossover, PDHG then the interior-point polish - under one time limit (3,600 s by default),
with the solution written and handed to tools/verify_solution.py, which parses the model
itself and recomputes every residual. A row is `verified` only when the verifier's exit code
is 0; `reached_optimum` compares the objective with the analytic one to a relative 1e-6.

WHEN AN ARM DOES NOT FINISH, the row carries the numbers that say why: the iterations it
did, the primal and dual infeasibility it stopped at, the peak resident memory of the solver
process (from wait4, so the kernel's number, not an estimate), the solver's own message -
which names the factor's nonzeros when the interior point or the polish declines - and the
termination reason. `attribution` is one word read off those numbers (iterations, fill,
memory, polish); docs/BENCHMARKS.md renders the table from this CSV.

The machine tag is measured, not typed: CPU model, logical cores and RAM from /proc, and
whether this is a container. `--host-label` adds what /proc cannot know (the provider).

Usage:
    python bench/runners/million.py --binary build/sankhya --keep /scratch/million \\
        --host-label "cloud container"
    python bench/runners/million.py --families transport --arms ipm --time-limit 600
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
RUNNERS = REPO_ROOT / "bench" / "runners"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# The generator command for each family, all seeded. The refinery runs 56 crudes so the
# hourly year crosses a million rows: 115 rows a period (56 crude balances, the CDU, 16
# production and 16 product balances, 8 units, 10 specs, 8 commitments) times 8,760 hours is
# 1,007,400. The transportation model has 500,000 plants and 500,000 depots.
FAMILIES = {
    "refinery": ["generate_refinery_lp.py", "--periods", "8760", "--crudes", "56",
                 "--seed", "7"],
    "staircase": ["generate_large_lp.py", "--rows", "1000000", "--cols", "1000000",
                  "--nnz-per-col", "5", "--seed", "7", "--structure", "staircase"],
    "transport": ["generate_transport_lp.py", "--sources", "500000", "--sinks", "500000",
                  "--arcs", "4", "--window", "8", "--seed", "7"],
}

# Each arm is an engine and the route it takes. Options beyond the engine are recorded in
# the CSV. The PDHG arm lifts the polish's 30 s and 5e7-nonzero caps, which are sized for a
# polish that should cost a fraction of the solve, so that at this size the polish gets the
# 30% of the limit PDHG leaves it and the same factor cap the interior point sizes from the
# machine (an eighth of physical memory in 12-byte nonzeros, src/ipm/ipm.cpp, #576). The
# polish option has no "size from the machine" value - 0 there is a cap of zero - so the
# number is computed here and written into the row.
#
# `ipm-xover` is the interior point with crossover_from_nonoptimal (#474): when the iterate
# stalls a hair short of the optimality test and is reported `feasible`, the dual simplex
# finishes from it at a vertex, and only that vertex can be reported optimal. The PDHG arm
# runs the sparse products and the vector updates on every core (#587, #689), bitwise
# identical to one thread, so the arm is the same method with the machine's cores used.
ARMS = {
    "ipm": ["algorithm=ipm", "crossover=true"],
    "ipm-xover": ["algorithm=ipm", "crossover=true", "crossover_from_nonoptimal=true"],
    "pdhg": ["algorithm=pdhg", "pdhg_polish=true", "polish_max_seconds=3600",
             f"threads={os.cpu_count() or 1}", "pdhg_parallel_spmv=true",
             "pdhg_parallel_updates=true"],
}


def machine_factor_budget() -> int:
    """The interior point's own default factor cap on this machine, in nonzeros."""
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return max(100_000_000, memory // 8 // 12)

CSV_COLUMNS = [
    "instance", "family", "instance_sha256", "rows", "columns", "nonzeros",
    "analytic_optimum", "arm", "route", "status", "our_objective", "absolute_error",
    "relative_error", "reached_optimum", "verified", "verifier_verdict", "iterations",
    "polish_iterations", "primal_infeasibility", "dual_infeasibility", "wall_seconds",
    "solver_seconds", "verify_seconds", "peak_rss_mb", "termination_reason", "attribution",
    "message", "time_limit", "solver_options", "generator", "git_commit", "machine",
    "timestamp_utc",
]

MATCH_RELATIVE_TOLERANCE = 1e-6
STATUSES_WITH_A_POINT = ("optimal", "feasible", "iteration_limit", "time_limit")
# The solver checks its clock between iterations; one long factorization can overrun.
OVERRUN_FACTOR = 1.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_tag(host_label: str) -> str:
    """CPU model, logical cores, RAM and container-ness, read from the machine itself."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gib = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_gib = int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    container = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    try:
        container = container or "container" in Path("/proc/1/environ").read_text(
            errors="ignore")
    except OSError:
        pass
    parts = [host_label] if host_label else []
    if container and "container" not in host_label:
        parts.append("container")
    parts += [f"{platform.system()}-{platform.machine()}", cpu, f"{os.cpu_count()} cores",
              f"{ram_gib:.1f} GiB RAM"]
    return "; ".join(parts)


def generate(family: str, directory: Path) -> dict:
    """Write (or reuse) the family's instance; return its path, sizes, optimum and command.

    A sidecar JSON keeps what the generator printed, so a re-run reuses a 1 GB file without
    regenerating it; the sha256 is always recomputed from the file itself."""
    path = directory / f"million-{family}.mps"
    meta_path = path.with_suffix(".json")
    script, *arguments = FAMILIES[family]
    command = [sys.executable, str(RUNNERS / script), *arguments, "--out", str(path)]
    shown = " ".join(["bench/runners/" + script, *arguments])
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("generator") == shown:
            return {**meta, "path": path}
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    meta = {"generator": shown, "generate_seconds": time.perf_counter() - started}
    for line in result.stdout.splitlines():
        text = line.strip()
        if text.startswith("analytic optimum:"):
            meta["optimum"] = text.split(":", 1)[1].strip()
        elif text.startswith("rows:") and "columns:" in text:
            fields = text.replace(":", " ").split()
            meta["rows"] = int(fields[fields.index("rows") + 1])
            meta["columns"] = int(fields[fields.index("columns") + 1])
            meta["nonzeros"] = int(fields[fields.index("nonzeros") + 1])
        elif ("rows," in text and "columns," in text and "nonzeros" in text
              and "rows" not in meta):
            # generate_large_lp.py prints "N rows, M columns, K nonzeros (...)".
            fields = text.replace(",", " ").split()
            meta["rows"] = int(fields[fields.index("rows") - 1])
            meta["columns"] = int(fields[fields.index("columns") - 1])
            meta["nonzeros"] = int(fields[fields.index("nonzeros") - 1])
    if "optimum" not in meta or "rows" not in meta:
        raise SystemExit(f"{family}: the generator did not print its sizes and optimum:\n"
                         f"{result.stdout}")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {**meta, "path": path}


def run_measured(command: list[str], timeout: float, log: Path) -> tuple[int, float, float]:
    """Run to completion or `timeout`; return (exit code, wall seconds, peak RSS in MB)."""
    started = time.perf_counter()
    with log.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        deadline = started + timeout
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() > deadline:
                process.kill()
                pid, status, usage = os.wait4(process.pid, 0)
                break
            time.sleep(0.5)
    wall = time.perf_counter() - started
    code = os.waitstatus_to_exitcode(status)
    return code, wall, usage.ru_maxrss / 1024.0  # ru_maxrss is in KiB on Linux


def attribute(row: dict, ram_mb: float) -> str:
    """One word for why an arm did not end verified optimal, read off the row's numbers."""
    if row["verified"] == "1" and row["status"] == "optimal":
        return ""
    message = row["message"].lower()
    rss = float(row["peak_rss_mb"] or 0)
    if row["status"] in ("killed", "crashed") or "bad_alloc" in message or (
            ram_mb and rss > 0.9 * ram_mb):
        return "memory"
    if "polish" in message and ("declin" in message or "factor" in message):
        return "polish"
    if "factor" in message or "nonzeros" in message or "ordering" in message:
        return "fill"
    if row["status"] == "optimal":
        return "verifier"
    return "iterations"


def solve(binary: Path, instance: dict, arm: str, time_limit: float, work: Path,
          extra: list[str], ram_mb: float) -> dict:
    stem = f"{instance['path'].stem}-{arm}"
    sol, stats, log = work / f"{stem}.sol", work / f"{stem}.stats.json", work / f"{stem}.log"
    for stale in (sol, stats):
        stale.unlink(missing_ok=True)
    options = ARMS[arm] + extra
    if arm == "pdhg":
        options = options + [f"polish_max_factor_nonzeros={machine_factor_budget()}"]
    command = [str(binary), "solve", str(instance["path"]), "--stats", str(stats),
               "--write-sol", str(sol), "--time-limit", str(time_limit)]
    for option in options:
        command += ["--option", option]
    code, wall, rss = run_measured(command, time_limit * OVERRUN_FACTOR + 600, log)
    row = {"arm": arm, "wall_seconds": f"{wall:.3f}", "peak_rss_mb": f"{rss:.0f}",
           "solver_options": " ".join(options), "time_limit": f"{time_limit:g}",
           "status": "", "route": "", "our_objective": "", "iterations": "",
           "polish_iterations": "", "primal_infeasibility": "", "dual_infeasibility": "",
           "solver_seconds": "", "termination_reason": "", "message": "",
           "verified": "0", "verifier_verdict": "", "verify_seconds": ""}
    if not stats.exists():
        row["status"] = "killed" if code < 0 else "crashed"
        tail = log.read_text(errors="replace").strip().splitlines()[-3:] if log.exists() else []
        row["message"] = f"exit {code}; " + " | ".join(tail)
        return row
    blob = json.loads(stats.read_text())
    result, effort, quality = blob.get("result", {}), blob.get("effort", {}), blob.get(
        "quality", {})
    row.update({
        "status": result.get("status", "unknown"),
        "route": result.get("algorithm", ""),
        "our_objective": "" if result.get("objective") is None else repr(
            float(result["objective"])),
        "iterations": effort.get("iterations", ""),
        "polish_iterations": effort.get("polish_iterations", ""),
        "primal_infeasibility": quality.get("primal_infeasibility", ""),
        "dual_infeasibility": quality.get("dual_infeasibility", ""),
        "solver_seconds": effort.get("solve_seconds", ""),
        "termination_reason": blob.get("limits", {}).get("termination_reason", ""),
        "message": " ".join(str(result.get("message", "")).split()),
    })
    if sol.exists() and row["status"] in STATUSES_WITH_A_POINT:
        started = time.perf_counter()
        verdict = subprocess.run([sys.executable, str(VERIFIER), str(instance["path"]),
                                  str(sol), "--quiet"], capture_output=True, text=True)
        row["verify_seconds"] = f"{time.perf_counter() - started:.1f}"
        row["verified"] = "1" if verdict.returncode == 0 else "0"
        lines = (verdict.stdout + verdict.stderr).strip().splitlines()
        row["verifier_verdict"] = lines[-1].strip() if lines else f"exit {verdict.returncode}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES),
                        default=["transport", "staircase", "refinery"])
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=["ipm", "ipm-xover", "pdhg"])
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="added to every arm and recorded in solver_options")
    parser.add_argument("--keep", type=Path, required=True,
                        help="where the instances, solutions and logs go (several GB)")
    parser.add_argument("--host-label", default="",
                        help="what /proc cannot tell, e.g. 'cloud container'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = stamp.stamp(args.binary)
    machine = machine_tag(args.host_label)
    ram_mb = 0.0
    try:
        ram_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20
    except (ValueError, OSError):
        pass
    out = args.out or RESULTS_DIR / f"million-cpu-{commit}.csv"
    args.keep.mkdir(parents=True, exist_ok=True)
    print(f"A million rows (#751), commit {commit}, {args.time_limit:g}s per solve\n"
          f"machine: {machine}\n", flush=True)
    rows: list[dict] = []
    for family in args.families:
        instance = generate(family, args.keep)
        digest = sha256(instance["path"])
        optimum = float(instance["optimum"])
        print(f"{family}: {instance['rows']:,} rows, {instance['columns']:,} columns, "
              f"{instance['nonzeros']:,} nonzeros, optimum {optimum!r}, sha256 {digest}",
              flush=True)
        for arm in args.arms:
            row = solve(args.binary, instance, arm, args.time_limit, args.keep,
                        args.solver_option, ram_mb)
            objective = float(row["our_objective"]) if row["our_objective"] else math.nan
            has_point = row["status"] in STATUSES_WITH_A_POINT and math.isfinite(objective)
            absolute = abs(objective - optimum) if has_point else None
            relative = absolute / max(1.0, abs(optimum)) if has_point else None
            row.update({
                "instance": instance["path"].name, "family": family,
                "instance_sha256": digest, "rows": instance["rows"],
                "columns": instance["columns"], "nonzeros": instance["nonzeros"],
                "analytic_optimum": repr(optimum),
                "absolute_error": "" if absolute is None else repr(absolute),
                "relative_error": "" if relative is None else repr(relative),
                "reached_optimum": int(relative is not None
                                       and relative <= MATCH_RELATIVE_TOLERANCE),
                "generator": instance["generator"], "git_commit": commit,
                "machine": machine,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"),
            })
            row["attribution"] = attribute(row, ram_mb)
            rows.append(row)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {arm:<6}{row['status']:<16}route {row['route'] or '-':<9}"
                  f"rel err {row['relative_error'] or '-':<24}verified {row['verified']}  "
                  f"{float(row['wall_seconds']):.0f}s  {row['peak_rss_mb']} MB  "
                  f"{row['attribution'] or 'finished'}", flush=True)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
