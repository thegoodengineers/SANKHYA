#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched MIPLIB subset and record what happened.

The MILP counterpart of netlib.py. Same CSV contract from ENGINEERING_RULES.md - instance, sha256, our
objective, published optimum, gaps, status, time, git commit, machine - plus the two columns
a MILP needs and an LP does not: NODES, and the bound.

THE DISTINCTION THIS FILE EXISTS TO KEEP. On an LP there is one question: is the objective
right. On a MILP there are two, and they come apart constantly:

    matched   - our objective equals the published optimum
    proved    - we also closed the bound against it and know it is optimal

`flugpl` is the case in point. We return exactly 1201500, which IS the published optimum, and
we report `feasible` rather than `optimal` because the search stopped on the relative gap
target instead of proving the bound. Collapsing those two into one "passed" column would
either throw away a correct answer or launder a tolerance stop into a proof. Both columns are
recorded, and the summary reports them separately.

A third column, `verified`, is the independent check: tools/verify_solution.py re-reads the
model with its own MPS reader and confirms the point is feasible AND integral. An incumbent
that is not integral is a relaxation, whatever the status says.

    python bench/runners/miplib.py --time-limit 600
    python bench/runners/miplib.py --instances flugpl gen-ip002
    python bench/runners/miplib.py --seeds 3 --time-limit 300      # #504
    python bench/runners/miplib.py --tier 2 --seeds 3              # the 60-instance tier

SEEDS (#504). `--seeds N` runs every instance N times: seed 0 is the published file, seeds
1..N-1 are row-and-column permutations of it written to a temporary directory by
miplib_seeds.permute_mps (the solver itself is untouched). Every run is one CSV row with its
seed; the per-instance summary - seeds matched, seeds proved, shifted geometric mean time,
time to first feasible, primal integral - is printed and written beside it as
`summary-<out name>`. The independent check always reads the ORIGINAL file: the .sol names
its columns, so a permutation that changed the model would fail it.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
import stamp  # noqa: E402  (#433: stamps from the binary)
import miplib_seeds  # noqa: E402  (#504: permutations and the per-seed summary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib"
# The second, larger tier (#504): selected by bench/runners/miplib_tier2.json's rule and
# fetched by `fetch_miplib.py --tier 2`, into its own directory so the 30-instance tier's
# manifest is never overwritten.
TIER_DIRS = {1: DATA_DIR, 2: REPO_ROOT / "data" / "miplib-tier2"}
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# An objective within this relative distance of the published optimum counts as MATCHED.
# Looser than the LP set's 1e-6 on purpose: MIPLIB objectives run to eight and nine figures,
# and the published values in the .solu file are themselves given to about ten.
MATCH_RELATIVE_TOLERANCE = 1e-6

CSV_COLUMNS = [
    "instance",
    "instance_sha256",
    "rows",
    "columns",
    "nonzeros",
    "integer_columns",
    "status",
    "message",
    "our_objective",
    "published_objective",
    "dual_bound",
    "absolute_gap",
    "relative_gap",
    "matched_published",
    "proved_optimal",
    "independently_verified",
    "nodes",
    "cuts_applied",
    "cut_filter",
    "root_bound",
    "root_bound_after_cuts",
    "root_gap_closed",
    "wall_seconds",
    "solver_seconds",
    "git_commit",
    "machine",
    "timestamp_utc",
    "solver_options",
    # Tree worker threads (#222): mip_threads from the options, 1 when it is not given. Last,
    # so a reader of the older CSVs that indexes columns by position still reads them.
    "threads",
    # Restarts and reduced-cost fixings (#418), after threads for the same reason; blank in
    # every CSV written before them.
    "restarts",
    "reduced_cost_fixings",
    # #504, after the rest for the same reason. seed 0 is the published file; the others are
    # permutations of it, identified by the sha256 of the permuted text. The time to first
    # feasible and the primal integral are on the solver's clock, from the stats JSON's
    # incumbent trace; incumbent_trace_recorded 0 means the solver kept none and the first
    # point was charged at the end of the run (an upper bound).
    "seed",
    "permuted_sha256",
    "time_to_first_feasible",
    "primal_integral",
    "incumbents",
    "incumbent_trace_recorded",
]

SUMMARY_COLUMNS = [
    "instance", "seeds", "seeds_matched", "seeds_proved", "matched_seeds", "proved_seeds",
    "sgm_seconds", "sgm_first_feasible_seconds", "mean_primal_integral", "time_limit",
    "git_commit", "machine",
]


def root_gap_closed(before, after, objective):
    """The share of the root integrality gap the root cuts closed (#221).

    (after - before) / (objective - before), measured against the objective the run ended
    with; None when any of the three is missing or the gap was already zero, so the column
    is blank rather than a made-up 0 or 1. Clamped to [0, 1] against rounding noise.
    """
    values = (before, after, objective)
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    gap = objective - before
    if abs(gap) <= 1e-9 * max(1.0, abs(objective)):
        return None
    return min(1.0, max(0.0, (after - before) / gap))


def as_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("inf", "+inf", "infinity"):
            return math.inf
        if text in ("-inf", "-infinity"):
            return -math.inf
        if text == "nan":
            return math.nan
        try:
            return float(text)
        except ValueError:
            return None
    return float(value)


def find_binary(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError:
        return None


def git_commit(binary=None) -> str:
    """The commit this CSV is stamped with: the binary's own, read from `sankhya
    version`, with `-dirty` from the tree; HEAD only when no binary answers (#433,
    bench/runners/stamp.py)."""
    return stamp.stamp(binary)

def solve(binary: Path, instance: Path, time_limit: float, verify: bool,
          solver_options: list[str] | None = None, verify_against: Path | None = None) -> dict:
    import time

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(instance),
                   "--time-limit", str(time_limit),
                   "--stats", str(stats_path),
                   "--write-sol", str(sol_path),
                   "--option", "log_to_console=false"]
        for option in solver_options or []:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started

        if not stats_path.exists():
            return {"status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                    "message": completed.stderr.strip()[:300],
                    "wall_seconds": wall, "verified": None}

        blob = json.loads(stats_path.read_text())
        result = blob.get("result", {})
        model = blob.get("model", {})
        effort = blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", "")[:300],
            "objective": as_number(result.get("objective")),
            "dual_bound": as_number(result.get("dual_bound")),
            "absolute_gap": as_number(result.get("absolute_gap")),
            "relative_gap": as_number(result.get("relative_gap")),
            "rows": model.get("rows", ""),
            "columns": model.get("columns", ""),
            "nonzeros": model.get("nonzeros", ""),
            "integer_columns": model.get("integer_columns", ""),
            "nodes": effort.get("nodes", ""),
            "cuts_applied": effort.get("cuts_applied", ""),
            "cut_filter": effort.get("cut_filter", ""),
            "restarts": effort.get("restarts", ""),
            "reduced_cost_fixings": effort.get("reduced_cost_fixings", ""),
            "root_bound": as_number(effort.get("root_bound")),
            "root_bound_after_cuts": as_number(effort.get("root_bound_after_cuts")),
            "solver_seconds": effort.get("solve_seconds", ""),
            "incumbent_trace": effort.get("incumbent_trace", []),
            "wall_seconds": wall,
            "verified": None,
        }

        # The independent check runs on any point we claim, proved or not. An incumbent that
        # is not integral is a relaxation whatever the status says, and that is precisely
        # what this catches.
        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run(
                [sys.executable, str(VERIFIER), str(verify_against or instance), str(sol_path),
                 "--quiet"],
                capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
        return flat


def run_seed(binary: Path, instance: Path, seed: int, scratch: Path, args,
             published: float) -> tuple[dict, str, dict]:
    """One run of one seed: the published file for seed 0, a permuted copy otherwise.
    Returns the solve blob, the permuted file's sha256 (blank for seed 0) and the incumbent
    metrics."""
    target, digest = instance, ""
    if seed != 0:
        target = scratch / f"{instance.name.replace('.mps.gz', '')}-seed{seed}.mps"
        miplib_seeds.permute_mps(instance, target, seed)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    blob = solve(binary, target, args.time_limit, not args.no_verify, args.solver_option,
                 verify_against=instance)
    if seed != 0:
        target.unlink(missing_ok=True)
    trace = blob.get("incumbent_trace") or []
    solver_seconds = as_number(blob.get("solver_seconds"))
    end = blob["wall_seconds"]
    if solver_seconds is not None and math.isfinite(solver_seconds):
        end = solver_seconds
    # The trace is on solve()'s clock, which also counts presolve; the MILP's reported solve
    # time is the search's own, so the end of the integral is whichever is later.
    end = max([end] + [float(t) for t, _ in trace])
    ours = blob.get("objective")
    has_point = (blob["status"] in ("optimal", "feasible") and ours is not None
                 and math.isfinite(ours))
    metrics = miplib_seeds.incumbent_metrics(trace, published, end, has_point)
    return blob, digest, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to the solver as --option KEY=VALUE; repeatable, and "
                             "recorded in the CSV so a run with a non-default option is "
                             "distinguishable from the default at the same commit")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="tree worker threads, the same as --solver-option mip_threads=N")
    parser.add_argument("--seeds", type=int, default=1,
                        help="runs per instance: seed 0 is the published file, seeds 1..N-1 "
                             "permute its rows and columns (#504); default 1")
    parser.add_argument("--tier", type=int, choices=sorted(TIER_DIRS), default=1,
                        help="1: the 30 smallest easy instances (data/miplib); 2: the "
                             "60-instance tier of bench/runners/miplib_tier2.json")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    binary = find_binary(args.binary)
    if binary is None:
        print("no solver binary; build first", file=sys.stderr)
        return 1

    data_dir = TIER_DIRS[args.tier]
    reference_path = data_dir / "reference.json"
    if not reference_path.exists():
        fetch = "bench/runners/fetch_miplib.py" + ("" if args.tier == 1 else " --tier 2")
        print(f"no {shown(reference_path)}; run {fetch} first", file=sys.stderr)
        return 1
    manifest = json.loads(reference_path.read_text())
    reference = manifest["instances"]

    names = args.instances or sorted(reference)
    commit = git_commit(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    if args.threads is not None:
        args.solver_option.append(f"mip_threads={args.threads}")
    solver_options = " ".join(args.solver_option)
    threads = 1
    for option in args.solver_option:
        key, _, value = option.partition("=")
        if key.strip() == "mip_threads":
            threads = int(value)
    print(f"commit   {commit}   machine {machine}   time limit {args.time_limit:g}s"
          + (f"   seeds {args.seeds}" if args.seeds > 1 else "")
          + (f"   options {solver_options}" if solver_options else ""))
    print()
    print(f"{'instance':<24}{'seed':>4} {'status':<14}{'our objective':>18}{'published':>18}"
          f"{'gap':>10}{'nodes':>9}{'time':>9}{'first':>8}  match proved ver")
    print("-" * 132)

    rows = []
    per_seed = []
    with tempfile.TemporaryDirectory(prefix="sankhya-seeds-") as scratch_dir:
        scratch = Path(scratch_dir)
        for name in names:
            entry = reference.get(name)
            instance = data_dir / f"{name}.mps.gz"
            if entry is None or not instance.exists():
                print(f"{name:<24}{'':>4} {'MISSING':<14}", flush=True)
                continue
            published = float(entry["published_optimal"])
            for seed in range(args.seeds):
                try:
                    blob, digest, metrics = run_seed(binary, instance, seed, scratch, args,
                                                     published)
                except ValueError as error:
                    # Named, never dropped: a seed that could not be permuted is a row missing
                    # from the table, and the reader has to be told which.
                    print(f"{name:<24}{seed:>4} {'NOT PERMUTED':<14}{error}", flush=True)
                    continue
                row = make_row(name, entry, published, blob, commit, solver_options, threads,
                               machine, stamp)
                first = metrics["time_to_first_feasible"]
                row.update({
                    "seed": seed,
                    "permuted_sha256": digest,
                    "time_to_first_feasible": "" if first is None else f"{first:.6f}",
                    "primal_integral": f"{metrics['primal_integral']:.6f}",
                    "incumbents": metrics["incumbents"],
                    "incumbent_trace_recorded": int(metrics["trace_recorded"]),
                })
                rows.append(row)
                per_seed.append({
                    "instance": name, "seed": seed,
                    "matched": bool(row["matched_published"]),
                    "proved": bool(row["proved_optimal"]),
                    "seconds": blob["wall_seconds"],
                    "time_to_first_feasible": first,
                    "primal_integral": metrics["primal_integral"],
                })
                print_row(row, blob, published, first)

    print("-" * 132, flush=True)
    matched_count = sum(r["matched_published"] for r in rows)
    proved_count = sum(r["proved_optimal"] for r in rows)
    print(f"{matched_count}/{len(rows)} reached the published optimum; "
          f"{proved_count}/{len(rows)} also PROVED it optimal")
    unproved = sorted({r["instance"] for r in rows if not r["proved_optimal"]})
    if unproved:
        # Naming them is not optional. A rate without its failures is a claim, not evidence.
        print(f"not proved (in at least one seed): {', '.join(unproved)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Default names that bench/runners/latest_result.py will NOT take for the 30-instance
    # tier's own per-commit run: a tier-2 or multi-seed table has a different shape.
    tier = "" if args.tier == 1 else f"tier{args.tier}-"
    seeds = "" if args.seeds == 1 else f"seeds{args.seeds}-"
    out_path = args.out or (RESULTS_DIR / f"miplib-{tier}{seeds}{commit}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    write_csv(out_path, CSV_COLUMNS, rows)
    print(f"wrote {shown(out_path)}")
    if args.seeds > 1:
        # A prefix, not a suffix: `miplib-<sha>-summary.csv` would read as a default run.
        summary_path = out_path.with_name("summary-" + out_path.name)
        write_summary(summary_path, per_seed, args.time_limit, commit, machine)
        print(f"wrote {shown(summary_path)}")
    return 0


def make_row(name, entry, published, blob, commit, solver_options, threads, machine,
             stamp) -> dict:
    ours = blob.get("objective")
    status = blob["status"]
    matched = (ours is not None and math.isfinite(ours)
               and abs(ours - published) <= MATCH_RELATIVE_TOLERANCE * max(1.0, abs(published)))
    # PROVED means the solver closed the bound itself - to within the gap target (1e-4
    # relative, 1e-6 absolute; since #188 that is reported optimal) or by exhausting the
    # tree - not that the number happens to be right. Only kOptimal asserts that, and
    # #29's guard has already re-measured it.
    proved = status == "optimal" and matched
    relative = blob.get("relative_gap")
    closed = root_gap_closed(blob.get("root_bound"), blob.get("root_bound_after_cuts"), ours)
    return {
        "instance": name,
        "instance_sha256": entry.get("gz_sha256", ""),
        "rows": blob.get("rows", ""),
        "columns": blob.get("columns", ""),
        "nonzeros": blob.get("nonzeros", ""),
        "integer_columns": blob.get("integer_columns", ""),
        "status": status,
        "message": blob.get("message", ""),
        "our_objective": "" if ours is None else repr(ours),
        "published_objective": repr(published),
        "dual_bound": "" if blob.get("dual_bound") is None else repr(blob["dual_bound"]),
        "absolute_gap": "" if blob.get("absolute_gap") is None else repr(blob["absolute_gap"]),
        "relative_gap": "" if relative is None else repr(relative),
        "matched_published": int(matched),
        "proved_optimal": int(proved),
        "independently_verified": "" if blob["verified"] is None else int(blob["verified"]),
        "nodes": blob.get("nodes", ""),
        "cuts_applied": blob.get("cuts_applied", ""),
        "cut_filter": blob.get("cut_filter", ""),
        "root_bound": "" if blob.get("root_bound") is None else repr(blob["root_bound"]),
        "root_bound_after_cuts": ("" if blob.get("root_bound_after_cuts") is None
                                  else repr(blob["root_bound_after_cuts"])),
        "root_gap_closed": "" if closed is None else f"{closed:.4f}",
        "wall_seconds": f"{blob['wall_seconds']:.6f}",
        "solver_seconds": blob.get("solver_seconds", ""),
        "git_commit": commit,
        "solver_options": solver_options,
        "threads": threads,
        "restarts": blob.get("restarts", ""),
        "reduced_cost_fixings": blob.get("reduced_cost_fixings", ""),
        "machine": machine,
        "timestamp_utc": stamp,
    }


def print_row(row: dict, blob: dict, published: float, first: float | None) -> None:
    relative = blob.get("relative_gap")
    ours = blob.get("objective")
    gap_text = "-" if relative is None or not math.isfinite(relative) else f"{relative:.2e}"
    ours_text = "-" if ours is None else f"{ours:.10g}"
    first_text = "-" if first is None else f"{first:.2f}s"
    mark = lambda flag: " yes " if flag else " NO  "  # noqa: E731
    ver = blob["verified"]
    print(f"{row['instance']:<24}{row['seed']:>4} {row['status']:<14}{ours_text:>18}"
          f"{published:>18.10g}{gap_text:>10}{str(blob.get('nodes', '')):>9}"
          f"{blob['wall_seconds']:>8.1f}s{first_text:>8}"
          f" {mark(row['matched_published'])}{mark(row['proved_optimal'])}"
          f"{'  -  ' if ver is None else mark(ver)}", flush=True)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def shown(path: Path):
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def write_summary(path: Path, per_seed: list[dict], time_limit: float, commit: str,
                  machine: str) -> None:
    """The per-instance table over seeds (#504), printed and written as a CSV."""
    summary = miplib_seeds.aggregate(per_seed, time_limit)
    fmt = lambda v: "" if v is None else f"{v:.4f}"  # noqa: E731
    print()
    print(f"per instance over seeds; time: shifted geometric mean, shift "
          f"{miplib_seeds.TIME_SHIFT_SECONDS:g} s, unproved runs charged {time_limit:g} s; "
          f"first feasible: shift {miplib_seeds.FIRST_FEASIBLE_SHIFT_SECONDS:g} s, none "
          f"found charged {time_limit:g} s; primal integral: mean, in seconds")
    print(f"{'instance':<24}{'matched':>9}{'proved':>8}{'sgm time':>10}{'sgm first':>11}"
          f"{'primal int':>12}")
    for item in summary:
        print(f"{item['instance']:<24}{item['seeds_matched']:>5}/{item['seeds']:<3}"
              f"{item['seeds_proved']:>4}/{item['seeds']:<3}{fmt(item['sgm_seconds']):>10}"
              f"{fmt(item['sgm_first_feasible_seconds']):>11}"
              f"{fmt(item['mean_primal_integral']):>12}")
    total = miplib_seeds.overall(per_seed, time_limit)
    print(f"all runs: {total['matched']}/{total['runs']} matched, {total['proved']}/"
          f"{total['runs']} proved, shifted geometric mean time {fmt(total['sgm_seconds'])} s")
    rows = []
    for item in summary:
        row = dict(item)
        row["matched_seeds"] = " ".join(str(s) for s in item["matched_seeds"])
        row["proved_seeds"] = " ".join(str(s) for s in item["proved_seeds"])
        for key in ("sgm_seconds", "sgm_first_feasible_seconds", "mean_primal_integral"):
            row[key] = fmt(item[key])
        row.update({"time_limit": f"{time_limit:g}", "git_commit": commit, "machine": machine})
        rows.append(row)
    write_csv(path, SUMMARY_COLUMNS, rows)


if __name__ == "__main__":
    sys.exit(main())
