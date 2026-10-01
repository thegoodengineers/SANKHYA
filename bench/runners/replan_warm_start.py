#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-plan from yesterday's basis through the CLI: `sankhya solve --warm-start` cold vs warm.

A refinery plan is re-run every morning with today's prices and demands, not rebuilt from
nothing. `rolling_warm_start.py` (#524) measured that restart through the Python bindings,
editing one cost at a time. This runner measures the judge-facing form of it: yesterday's
`.sol` file handed to today's model on the command line,

    sankhya solve today.mps --warm-start yesterday.sol

with BOTH crude prices and delivery commitments moved at once, which is what a real morning
looks like and is the case where the old basis is neither primal nor dual feasible any more.

THE DAYS. Day 0 is the refinery case study LP (bench/case_studies/refinery/generator.py,
#517) solved cold; its .sol is "yesterday". Each following day copies the previous day's
MPS with every crude purchase cost (`BUY_*`) scaled by a random factor in [1 - delta,
1 + delta], and every product's demand - its market (the `SELL_*` upper bound) and its
delivery commitment (the `COMMIT_*` right-hand side), together - by one in
[1 - demand_delta, 1 + demand_delta] (--demand-delta, default 0.05). The generator sets a
tight commitment equal to the market, so moving the commitment alone pushes it past the
market and makes every day infeasible by construction (the first run of this runner did
exactly that). The SAME edited file is then solved twice: cold (the slack
basis, what a fresh read-and-solve does) and warm (from the previous day's WARM .sol), so the
two arms see the same model and differ only in the starting basis.

A MORNING THE PLANT CANNOT MEET. Commitments that rise past capacity make a day
infeasible; that is a real outcome, recorded as such: both arms must say `infeasible` with a
certificate the verifier accepts, the day counts as agreeing, contributes nothing to the
iteration totals, and tomorrow starts from the last feasible day's basis and file.

WHAT COUNTS. Every .sol is checked by tools/verify_solution.py, which shares no code with
the solver; a day whose cold and warm objectives disagree beyond 1e-6 relative is a failure
of this runner, not a data point. The iteration counts are the headline: they do not depend
on what else the laptop was doing. The wall times are reported beside them and labelled by
the `machine` column, which says what the box was.

Usage:
    python bench/runners/replan_warm_start.py --size medium --days 10
    python bench/runners/replan_warm_start.py --size small --days 5 --delta 0.1 --out out.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
GENERATOR = REPO_ROOT / "bench" / "case_studies" / "refinery" / "generator.py"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

CSV_COLUMNS = [
    "instance", "instance_sha256", "day", "arm", "status", "our_objective",
    "published_objective", "absolute_gap", "relative_gap", "independently_verified",
    "iterations", "solver_seconds", "wall_seconds", "algorithm", "message", "time_limit",
    "git_commit", "machine", "timestamp_utc",
]


def perturb_mps(source: Path, out: Path, rng: random.Random, delta: float,
                demand_delta: float | None = None) -> tuple[int, int]:
    """Copy `source` with every crude price and every product's demand moved.

    Prices: each `BUY_*` cost (one `NAME  COST  value` line in COLUMNS) by a factor in
    [1 - delta, 1 + delta]. Demands: product j's market in period t is the upper bound of
    `SELL_j_t` (BOUNDS) and its delivery commitment the right-hand side of `COMMIT_j_t` (RHS);
    the generator sets a tight commitment equal to the market, so the two move by ONE factor
    in [1 - demand_delta, 1 + demand_delta] (default delta / 2) - moving the commitment alone
    raises it past the market and makes every morning infeasible by construction. Returns
    how many prices and how many demands moved.
    """
    spread = delta / 2 if demand_delta is None else demand_delta
    factors: dict[str, float] = {}

    def demand(key: str) -> float:
        if key not in factors:
            factors[key] = rng.uniform(1.0 - spread, 1.0 + spread)
        return factors[key]

    lines, section, prices = [], "", 0
    for line in source.read_text().splitlines():
        if line.startswith("* LP analytic optimum:"):
            line = "* re-planned by bench/runners/replan_warm_start.py: prices and demands moved"
        elif line and not line[0].isspace():
            section = line.split()[0]
        elif section == "COLUMNS":
            tokens = line.split()
            if len(tokens) == 3 and tokens[1] == "COST" and tokens[0].startswith("BUY_"):
                factor = rng.uniform(1.0 - delta, 1.0 + delta)
                line = f"    {tokens[0]}  COST  {float(tokens[2]) * factor:.9g}"
                prices += 1
        elif section == "RHS":
            tokens = line.split()
            if len(tokens) == 3 and tokens[1].startswith("COMMIT_"):
                key = tokens[1][len("COMMIT_"):]
                line = f"    RHS  {tokens[1]}  {float(tokens[2]) * demand(key):.9g}"
        elif section == "BOUNDS":
            tokens = line.split()
            if len(tokens) == 4 and tokens[0] == "UP" and tokens[2].startswith("SELL_"):
                key = tokens[2][len("SELL_"):]
                line = f" UP BND  {tokens[2]}  {float(tokens[3]) * demand(key):.9g}"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n")
    return prices, len(factors)


def solve(binary: Path, mps: Path, sol: Path, time_limit: float,
          warm_from: Path | None) -> dict:
    stats = sol.with_suffix(".json")
    command = [str(binary), "solve", str(mps), "--write-sol", str(sol), "--stats", str(stats),
               "--time-limit", str(time_limit), "--option", "log_to_console=false"]
    if warm_from is not None:
        command += ["--warm-start", str(warm_from)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if not stats.exists():
        return {"status": "crashed", "message": (completed.stderr or completed.stdout)[-300:],
                "wall_seconds": wall, "objective": None, "iterations": "",
                "solver_seconds": "", "algorithm": "", "verified": ""}
    blob = json.loads(stats.read_text())
    result, effort = blob["result"], blob["effort"]
    verified = ""
    if sol.exists():
        check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol), "--quiet"],
                               capture_output=True, text=True)
        verified = int(check.returncode == 0)
    return {"status": result["status"], "message": result.get("message", "")[:300],
            "wall_seconds": wall, "objective": result.get("objective"),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "algorithm": result.get("algorithm", ""), "verified": verified}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_binary() -> Path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError as error:
        raise SystemExit(str(error))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.10,
                        help="prices move by up to this fraction")
    parser.add_argument("--demand-delta", type=float, default=0.05,
                        help="each product's market and commitment move together by up to "
                             "this fraction")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--machine", default=None,
                        help="machine tag for every row: say what the box is and whether it "
                             "was shared; defaults to <system>-<arch>")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    commit = stamp.stamp(args.binary)
    machine = args.machine or f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        today = work / "day0.mps"
        subprocess.run([sys.executable, str(GENERATOR), "--size", args.size, "--seed",
                        str(args.seed), "--out", str(today)], check=True, capture_output=True)
        print(f"solver   {binary}\ncommit   {commit}   machine {machine}   "
              f"size {args.size}   days {args.days}   delta {args.delta:g}   "
              f"demand delta {args.demand_delta:g}")
        print(f"{'day':>3} {'arm':<5} {'status':<10} {'objective':>16} {'iters':>8} "
              f"{'solve s':>9} {'wall s':>8}  verified")
        rows: list[dict] = []
        yesterday = work / "day0.sol"
        first = solve(binary, today, yesterday, args.time_limit, None)
        rows.append(record(today, 0, "cold", first, first, args, commit, machine, timestamp))
        show(0, "cold", first)
        if first["status"] != "optimal" or first["verified"] != 1:
            print("day 0 did not solve to a verified optimum; stopping")
            return 1
        cold_total = warm_total = 0
        cold_secs = warm_secs = 0.0
        agree = infeasible_days = 0
        for day in range(1, args.days + 1):
            source, today = today, work / f"day{day}.mps"
            perturb_mps(source, today, rng, args.delta, args.demand_delta)
            cold = solve(binary, today, work / f"day{day}-cold.sol", args.time_limit, None)
            warm_sol = work / f"day{day}-warm.sol"
            warm = solve(binary, today, warm_sol, args.time_limit, yesterday)
            rows.append(record(today, day, "cold", cold, cold, args, commit, machine, timestamp))
            rows.append(record(today, day, "warm", warm, cold, args, commit, machine, timestamp))
            show(day, "cold", cold)
            show(day, "warm", warm)
            proved = cold["verified"] == 1 and warm["verified"] == 1
            if proved and cold["status"] == "infeasible" == warm["status"]:
                agree += 1
                infeasible_days += 1
                today = source  # tomorrow edits the last plan the plant could meet
                continue
            both = cold["status"] == "optimal" == warm["status"]
            if both and proved and \
                    abs(cold["objective"] - warm["objective"]) <= 1e-6 * max(1.0, abs(cold["objective"])):
                agree += 1
                cold_total += int(cold["iterations"])
                warm_total += int(warm["iterations"])
                cold_secs += float(cold["solver_seconds"])
                warm_secs += float(warm["solver_seconds"])
            if warm["status"] == "optimal" and warm_sol.exists():
                yesterday = warm_sol
        print(f"\n{agree}/{args.days} days: both arms verified and agreeing "
              f"({infeasible_days} of them infeasible in both, the rest the same optimum)")
        if agree:
            print(f"over those days: cold {cold_total} iterations / {cold_secs:.2f} s, "
                  f"warm {warm_total} iterations / {warm_secs:.2f} s; warm is "
                  f"{warm_total / max(cold_total, 1):.3f}x the iterations and "
                  f"{warm_secs / max(cold_secs, 1e-9):.3f}x the solver time")

    out = args.out or RESULTS_DIR / f"replan-warm-start-{args.size}-{commit}.csv"
    out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0 if agree == args.days else 1


def record(mps: Path, day: int, arm: str, run: dict, reference: dict, args, commit: str,
           machine: str, timestamp: str) -> dict:
    """One CSV row. The cold arm's objective is the reference the warm arm is held to."""
    ours, ref = run["objective"], reference["objective"]
    gap = abs(ours - ref) if ours is not None and ref is not None else ""
    return {
        "instance": mps.name, "instance_sha256": sha256_file(mps), "day": day, "arm": arm,
        "status": run["status"], "our_objective": "" if ours is None else ours,
        "published_objective": "" if ref is None else ref, "absolute_gap": gap,
        "relative_gap": "" if gap == "" else gap / max(1.0, abs(ref)),
        "independently_verified": run["verified"], "iterations": run["iterations"],
        "solver_seconds": run["solver_seconds"], "wall_seconds": round(run["wall_seconds"], 6),
        "algorithm": run["algorithm"], "message": run["message"], "time_limit": args.time_limit,
        "git_commit": commit, "machine": machine, "timestamp_utc": timestamp,
    }


def show(day: int, arm: str, run: dict) -> None:
    objective = "-" if run["objective"] is None else f"{run['objective']:.6f}"
    print(f"{day:>3} {arm:<5} {run['status']:<10} {objective:>16} {str(run['iterations']):>8} "
          f"{str(run['solver_seconds'])[:9]:>9} {run['wall_seconds']:>8.2f}  "
          f"{ {1: 'yes', 0: 'NO', '': '-'}[run['verified']] }")


if __name__ == "__main__":
    sys.exit(main())
