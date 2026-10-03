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
    python bench/runners/miplib.py --certificate                   # #518: VIPR proofs
    python bench/runners/miplib.py --set miplib3 --seeds 3 --time-limit 60   # #761

MIPLIB 3 (#761). `--set miplib3` runs the classic set fetched by fetch_miplib3.py from
data/miplib3 and its manifest.json, with every other flag as for MIPLIB 2017. Its CSV is
named `miplib3-<limit>s-seeds<N>-<sha>.csv` (summary `summary-miplib3-...`), a name no
MIPLIB 2017 glob in make_benchmarks_doc.py or latest_result.py matches, so the two sets can
never be read as one. Its references carry `published_tolerance`, one unit in the last
decimal place the catalogue prints (fetch_miplib3.py says why: rgn is printed 82.1999 and
its optimum is 82.19999924); an objective matches when it is within
max(MATCH_RELATIVE_TOLERANCE * max(1, |published|), published_tolerance) of the published
value. MIPLIB 2017's manifest has no such field, so its rule is unchanged.

CERTIFICATES (#518). `--certificate` asks the solver for a VIPR proof of every answer
(option write_certificate, which also turns presolve off and runs the tree on one thread)
and checks it with tools/verify_certificate.py, in exact rational arithmetic against the
file that was solved, the incumbent up to the 1e-7 primal tolerance and the bound
exactly. The verdict is the `certificate` column - verified, rejected,
unproved (every step checks but the gap is not closed) or not_written - and the checker's
own runtime is `certificate_check_seconds`. Both are blank without the flag.

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
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import stamp  # noqa: E402  (#433: stamps from the binary)
import miplib_seeds  # noqa: E402  (#504: permutations and the per-seed summary)
from compare_suite import machine_tag  # noqa: E402  (kind, CPU model, cores, RAM, OS)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib"
# The second, larger tier (#504): selected by bench/runners/miplib_tier2.json's rule and
# fetched by `fetch_miplib.py --tier 2`, into its own directory so the 30-instance tier's
# manifest is never overwritten.
TIER_DIRS = {1: DATA_DIR, 2: REPO_ROOT / "data" / "miplib-tier2"}
# The classic MIPLIB 3 set (#761), fetched by fetch_miplib3.py: plain .mps files, exactly as
# archived, and a manifest of its own name, so nothing here ever reads it as MIPLIB 2017.
MIPLIB3_DIR = REPO_ROOT / "data" / "miplib3"
SETS = ("miplib2017", "miplib3")
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
CERTIFICATE_CHECKER = REPO_ROOT / "tools" / "verify_certificate.py"
# verify_certificate.py's exit status (see its docstring) -> the certificate column.
CERTIFICATE_VERDICTS = {0: "verified", 1: "rejected", 2: "unproved"}
# The incumbent's continuous values are doubles, so the SOL point is checked up to the
# project's primal feasibility tolerance (tolerances.hpp kPrimalFeasibility, the same 1e-7
# as verify_solution.py); the bound side is always checked exactly.
CERTIFICATE_FEAS_TOL = 1e-7

# An objective within this relative distance of the published optimum counts as MATCHED.
# Looser than the LP set's 1e-6 on purpose: MIPLIB objectives run to eight and nine figures,
# and the published values in the .solu file are themselves given to about ten.
MATCH_RELATIVE_TOLERANCE = 1e-6


def match_tolerance(published: float, entry: dict) -> float:
    """The absolute distance from `published` that still counts as MATCHED.

    MATCH_RELATIVE_TOLERANCE * max(1, |published|), or the manifest entry's
    `published_tolerance` when it has one and it is looser (#761: MIPLIB 3's catalogue
    prints its optima to limited precision, one of them truncated). Never tighter than the
    relative rule; MIPLIB 2017's entries have no such field."""
    rule = MATCH_RELATIVE_TOLERANCE * max(1.0, abs(published))
    extra = entry.get("published_tolerance")
    return rule if extra in (None, "") else max(rule, float(extra))


def matches_published(ours, published: float, entry: dict) -> bool:
    return (ours is not None and math.isfinite(ours)
            and abs(ours - published) <= match_tolerance(published, entry))


def default_out_name(set_name: str, tier: int, seeds: int, time_limit: float,
                     commit: str) -> str:
    """The CSV name a run writes when --out is not given.

    MIPLIB 2017: `miplib-<sha>.csv` for the 30-instance tier's own per-commit run, the only
    name bench/runners/latest_result.py takes for it; `miplib-tier2-` and `seedsN-` mark the
    other shapes. MIPLIB 3 (#761): `miplib3-<limit>s-seeds<N>-<sha>.csv`, the limit and the
    seed count always in the name. `miplib3-` does not begin `miplib-`, so no MIPLIB 2017
    glob can pick it up (test_miplib3.py checks every one)."""
    if set_name == "miplib3":
        return f"miplib3-{time_limit:g}s-seeds{seeds}-{commit}.csv"
    tier_part = "" if tier == 1 else f"tier{tier}-"
    seeds_part = "" if seeds == 1 else f"seeds{seeds}-"
    return f"miplib-{tier_part}{seeds_part}{commit}.csv"


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
    # #518, after the rest for the same reason; blank unless --certificate.
    "certificate",
    "certificate_check_seconds",
    # #756, after the rest: the proof file's size, its dual-bounded leaves, and how many of
    # those were bounded from the batched PDHG run's multipliers (gpu_batch_nodes), read from
    # the solver's own "leaves:" log line. Blank unless --certificate.
    "certificate_bytes",
    "certificate_dual_leaves",
    "certificate_batch_leaves",
    # #520, after the rest for the same reason; blank unless --profile. Inclusive seconds of
    # the solver's profile regions (profile=detailed): the branching decision with strong
    # branching's probe LPs inside it, and the two batched-PDHG calls.
    "branching_seconds",
    "batch_strong_branching_seconds",
    "batch_node_bounds_seconds",
    # Safe dual bounds (#519): filled only when safe_bounds is on, by the serial tree; last,
    # for the same reason as the columns above.
    "safe_bound_nodes",
    "safe_bound_infinite",
    "safe_bound_refusals",
    "safe_bound_max_gap",
    "safe_bound_max_rel_gap",
    # Conflict analysis (#503): filled only when conflict_analysis is on, by the serial tree.
    "conflicts_analysed",
    "conflicts_learned",
    "conflicts_learned_cutoff",
    "conflict_nodes_pruned",
    "conflict_tightenings",
    # The cut pool (#497), last for the same reason: cut rows aged out and put back in force
    # (any mode), and of those the rows deleted from the node LP and appended again
    # (mip_cut_pooling); the node LP's rows at each node's first solve, mean and most.
    "cut_rows_aged_out",
    "cuts_reactivated",
    "cut_rows_removed",
    "cut_rows_readded",
    "node_lp_rows_mean",
    "node_lp_rows_max",
    # Probing (#512): presolve's own counts from the stats JSON, and the clique cuts the
    # separator found over the search; last, for the same reason.
    "probing_fixings",
    "probing_tightenings",
    "probing_implications",
    "probing_cliques",
    "clique_cuts_generated",
]

PROBING_COLUMNS = ("probing_fixings", "probing_tightenings", "probing_implications",
                   "probing_cliques")

SAFE_BOUND_COLUMNS = ("safe_bound_nodes", "safe_bound_infinite", "safe_bound_refusals",
                      "safe_bound_max_gap", "safe_bound_max_rel_gap",
                      # The conflict counters (#503) travel the same way; the name is historical.
                      "conflicts_analysed", "conflicts_learned", "conflicts_learned_cutoff",
                      "conflict_nodes_pruned", "conflict_tightenings",
                      # So do the cut pool's (#497).
                      "cut_rows_aged_out", "cuts_reactivated", "cut_rows_removed",
                      "cut_rows_readded", "node_lp_rows_mean", "node_lp_rows_max")

# The certificate writer's log line (src/mip/certificate_writer.cpp, #756).
LEAVES_LINE = re.compile(r"leaves: (\d+) from their own LP duals, (\d+) from an ancestor's,.*?"
                         r"(\d+) of the dual-bounded leaves from the batched PDHG run")

# --profile's columns -> the profile region each one sums (src/mip/branch_and_bound*.cpp).
PROFILE_REGIONS = {"branching_seconds": "branching",
                   "batch_strong_branching_seconds": "batched strong branching",
                   "batch_node_bounds_seconds": "batched node bounds"}

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
          solver_options: list[str] | None = None, verify_against: Path | None = None,
          certificate: bool = False, profile: bool = False) -> dict:
    import time

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(instance),
                   "--time-limit", str(time_limit),
                   "--stats", str(stats_path),
                   "--write-sol", str(sol_path),
                   # Under a certificate the log carries the leaf counts (#756).
                   "--option", f"log_to_console={'true' if certificate else 'false'}"]
        for option in solver_options or []:
            command += ["--option", option]
        proof_path = Path(tmp) / "proof.vipr"
        if certificate:
            command += ["--option", f"write_certificate={proof_path}"]
        profile_path = Path(tmp) / "profile.json"
        if profile:
            command += ["--option", "profile=detailed", "--option", f"profile_out={profile_path}"]
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
            **{key: effort.get(key, "") for key in SAFE_BOUND_COLUMNS},
            "clique_cuts_generated": effort.get("clique_cuts_generated", ""),
            **{key: blob.get("presolve", {}).get("reductions", {}).get(key, "")
               for key in PROBING_COLUMNS},
            "root_bound": as_number(effort.get("root_bound")),
            "root_bound_after_cuts": as_number(effort.get("root_bound_after_cuts")),
            "solver_seconds": effort.get("solve_seconds", ""),
            "incumbent_trace": effort.get("incumbent_trace", []),
            "wall_seconds": wall,
            "verified": None,
            "certificate": None,
            "certificate_check_seconds": None,
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
        if certificate:
            flat.update(check_certificate(proof_path, instance))
            flat.update(leaf_counts(completed.stdout + completed.stderr))
            flat["certificate_bytes"] = (proof_path.stat().st_size if proof_path.exists()
                                         else None)
        if profile:
            flat.update(profile_seconds(profile_path))
        return flat


def leaf_counts(log: str) -> dict:
    """The dual-bounded leaves of the written proof and how many the batch bounded (#756)."""
    match = LEAVES_LINE.search(log)
    if match is None:
        return {"certificate_dual_leaves": None, "certificate_batch_leaves": None}
    return {"certificate_dual_leaves": int(match.group(1)) + int(match.group(2)),
            "certificate_batch_leaves": int(match.group(3))}


def profile_seconds(path: Path) -> dict:
    """PROFILE_REGIONS' inclusive seconds from a profile_out JSON; blank when none was written.
    A region's path is `parent/child`; one nested under a region of its own name is already
    inside that region's time, so only the outermost occurrence is summed."""
    if not path.exists():
        return {}
    regions = json.loads(path.read_text()).get("regions", [])
    out = {}
    for column, name in PROFILE_REGIONS.items():
        out[column] = sum(r["inclusive_seconds"] for r in regions
                          if r["path"].split("/")[-1] == name
                          and r["path"].split("/").count(name) == 1)
    return out


def check_certificate(proof: Path, instance: Path) -> dict:
    """The independent checker's verdict on the proof the solver wrote for `instance` (#518),
    and how long the check took. The model is the file that was solved - for a permuted seed
    the permuted copy, since the proof names its rows and columns."""
    import time

    if not proof.exists():
        return {"certificate": "not_written", "certificate_check_seconds": None}
    started = time.perf_counter()
    check = subprocess.run(
        [sys.executable, str(CERTIFICATE_CHECKER), str(proof), "--mps", str(instance),
         "--feas-tol", repr(CERTIFICATE_FEAS_TOL)],
        capture_output=True, text=True)
    seconds = time.perf_counter() - started
    return {"certificate": CERTIFICATE_VERDICTS.get(check.returncode, "checker_failed"),
            "certificate_check_seconds": seconds}


class PermutationError(ValueError):
    """A seed whose instance could not be permuted; the row is named, not dropped."""


def run_seed(binary: Path, instance: Path, seed: int, scratch: Path, args,
             published: float) -> tuple[dict, str, dict]:
    """One run of one seed: the published file for seed 0, a permuted copy otherwise.
    Returns the solve blob, the permuted file's sha256 (blank for seed 0) and the incumbent
    metrics."""
    stem = instance.name.removesuffix(".gz").removesuffix(".mps")
    target, digest = instance, ""
    if seed != 0:
        target = scratch / f"{stem}-seed{seed}.mps"
        try:
            miplib_seeds.permute_mps(instance, target, seed)
        except ValueError as error:
            # Only the permutation is caught as "not permuted"; a ValueError from the solve
            # or the stats parse below is a real failure and must not be relabelled (#627).
            raise PermutationError(str(error)) from error
        digest = hashlib.sha256(target.read_bytes()).hexdigest()

    run_options = []
    for opt in (args.solver_option or []):
        if "{instance}" in opt or "{seed}" in opt:
            run_options.append(opt.format(instance=stem, seed=seed))
        else:
            run_options.append(opt)

    blob = solve(binary, target, args.time_limit, not args.no_verify, run_options,
                 verify_against=instance, certificate=args.certificate,
                 profile=args.profile)
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
    parser.add_argument("--set", dest="set_name", choices=SETS, default="miplib2017",
                        help="miplib2017 (default): the tiers above; miplib3: the classic set "
                             "in data/miplib3, fetched by fetch_miplib3.py (#761)")
    parser.add_argument("--certificate", action="store_true",
                        help="write a VIPR proof of every answer and check it with "
                             "tools/verify_certificate.py (#518)")
    parser.add_argument("--machine-kind", default=None,
                        help="what kind of machine this is, for the machine tag, e.g. "
                             "'cloud container' (default: systemd-detect-virt's answer)")
    parser.add_argument("--profile", action="store_true",
                        help="run with profile=detailed and record the branching and batched-"
                             "PDHG seconds (#520)")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.set_name == "miplib3" and args.tier != 1:
        parser.error("--tier is a MIPLIB 2017 choice; MIPLIB 3 is one set")

    binary = find_binary(args.binary)
    if binary is None:
        print("no solver binary; build first", file=sys.stderr)
        return 1

    if args.set_name == "miplib3":
        data_dir, suffix = MIPLIB3_DIR, ".mps"
        reference_path = data_dir / "manifest.json"
        fetch = "bench/runners/fetch_miplib3.py"
    else:
        data_dir, suffix = TIER_DIRS[args.tier], ".mps.gz"
        reference_path = data_dir / "reference.json"
        fetch = "bench/runners/fetch_miplib.py" + ("" if args.tier == 1 else " --tier 2")
    if not reference_path.exists():
        print(f"no {shown(reference_path)}; run {fetch} first", file=sys.stderr)
        return 1
    manifest = json.loads(reference_path.read_text())
    reference = manifest["instances"]

    names = args.instances or sorted(reference)
    commit = git_commit(args.binary)
    machine = machine_tag(args.machine_kind)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    if args.threads is not None:
        args.solver_option.append(f"mip_threads={args.threads}")
    # The proof's path is a temporary file; the CSV records that the mode was on.
    solver_options = " ".join(args.solver_option
                              + (["write_certificate=on"] if args.certificate else []))
    threads = 1
    for option in args.solver_option:
        key, _, value = option.partition("=")
        if key.strip() == "mip_threads":
            threads = int(value)
    if args.certificate:
        threads = 1  # write_certificate runs the tree on one thread, whatever mip_threads says
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
            instance = data_dir / f"{name}{suffix}"
            if entry is None or not instance.exists():
                print(f"{name:<24}{'':>4} {'MISSING':<14}", flush=True)
                continue
            published = float(entry["published_optimal"])
            for seed in range(args.seeds):
                try:
                    blob, digest, metrics = run_seed(binary, instance, seed, scratch, args,
                                                     published)
                except PermutationError as error:
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
    if args.certificate:
        certified = [r for r in rows if r["proved_optimal"] and r["certificate"] == "verified"]
        print(f"{len(certified)}/{proved_count} proved optima ship a certificate the "
              f"independent checker accepts")
        uncertified = [f"{r['instance']} seed {r['seed']} ({r['certificate']})" for r in rows
                       if r["proved_optimal"] and r["certificate"] != "verified"]
        if uncertified:
            print("proved but not certified: " + ", ".join(uncertified))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Default names that bench/runners/latest_result.py will NOT take for the 30-instance
    # tier's own per-commit run: a tier-2, multi-seed or MIPLIB 3 table has a different shape.
    out_path = args.out or (RESULTS_DIR / default_out_name(args.set_name, args.tier,
                                                            args.seeds, args.time_limit,
                                                            commit))
    out_path = (REPO_ROOT / out_path).resolve()
    write_csv(out_path, CSV_COLUMNS, rows)
    print(f"wrote {shown(out_path)}")
    if args.seeds > 1 or args.set_name == "miplib3":
        # MIPLIB 3 writes its per-instance summary at any seed count, beside its run.
        # A prefix, not a suffix: `miplib-<sha>-summary.csv` would read as a default run.
        summary_path = out_path.with_name("summary-" + out_path.name)
        write_summary(summary_path, per_seed, args.time_limit, commit, machine)
        print(f"wrote {shown(summary_path)}")
    return 0


def make_row(name, entry, published, blob, commit, solver_options, threads, machine,
             stamp) -> dict:
    ours = blob.get("objective")
    status = blob["status"]
    matched = matches_published(ours, published, entry)
    # PROVED means the solver closed the bound itself - to within the gap target (1e-4
    # relative, 1e-6 absolute; since #188 that is reported optimal) or by exhausting the
    # tree - not that the number happens to be right. Only kOptimal asserts that, and
    # #29's guard has already re-measured it.
    proved = status == "optimal" and matched
    relative = blob.get("relative_gap")
    closed = root_gap_closed(blob.get("root_bound"), blob.get("root_bound_after_cuts"), ours)
    return {
        "instance": name,
        # MIPLIB 2017's digest is of the .mps.gz it ships; MIPLIB 3's of the plain file solved.
        "instance_sha256": entry.get("gz_sha256") or entry.get("sha256", ""),
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
        **{key: ("" if blob.get(key) is None else blob[key]) for key in SAFE_BOUND_COLUMNS},
        **{key: ("" if blob.get(key) is None else blob[key])
           for key in PROBING_COLUMNS + ("clique_cuts_generated",)},
        "machine": machine,
        "timestamp_utc": stamp,
        "certificate": blob.get("certificate") or "",
        **{column: ("" if blob.get(column) is None else f"{blob[column]:.6f}")
           for column in PROFILE_REGIONS},
        "certificate_check_seconds": ("" if blob.get("certificate_check_seconds") is None
                                      else f"{blob['certificate_check_seconds']:.6f}"),
        **{key: "" if blob.get(key) is None else blob[key]
           for key in ("certificate_bytes", "certificate_dual_leaves",
                       "certificate_batch_leaves")},
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
