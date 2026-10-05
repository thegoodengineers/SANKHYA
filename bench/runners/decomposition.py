#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decomposed against monolithic, on the multi-period refinery ladder (#525).

    python bench/runners/decomposition.py [--binary build/sankhya] [--sizes small medium large]
                                          [--blocks 2 4 8] [--time-limit 300] [--out CSV]

For each size of the refinery ladder (bench/case_studies/refinery/generator.py - T periods, each
joined to the next by the stocks it carries) the LP is generated, its exact analytic optimum is
read from the generator (it verifies every KKT condition in rational arithmetic before it writes
the file, so that number is the reference and shares nothing with the solver), and it is solved
once monolithically and once per block count with `--option decomposition=benders`. Every answer
is checked by tools/verify_solution.py, the independent verifier, and one CSV row per solve
carries the columns ENGINEERING_RULES.md asks for: the instance, the sha256 of the file solved,
our objective, the reference objective, the absolute and relative gap, status, wall time,
iterations, git commit (from the binary, stamp.py) and machine - plus what decomposition adds:
which method actually ran, whether it matched the monolithic objective, and the speed-up.

A decomposition that declined or whose answer was discarded falls back to the monolithic engines
inside solve(); the row's `algorithm` says so (it is not `benders`), and that is what is
reported - never the option that was asked for.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "bench" / "case_studies" / "refinery" / "generator.py"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
SIZES = ("small", "medium", "large")
SEED = 1

FIELDS = ["instance", "sha256", "mode", "blocks", "rows", "columns", "nonzeros", "status",
          "algorithm", "our_objective", "reference_objective", "absolute_gap", "relative_gap",
          "matches_monolithic", "verified", "wall_seconds", "solver_seconds", "iterations",
          "speedup_vs_monolithic", "message", "git_commit", "machine", "time_limit",
          "timestamp_utc", "solver_options"]


def parse_optimum(generator_output: str) -> float | None:
    """The `LP analytic optimum: <number>` line the generator prints, or None."""
    found = re.search(r"LP analytic optimum:\s*([-+0-9.eE]+)", generator_output)
    return float(found.group(1)) if found else None


def relative_gap(ours: float, reference: float) -> float:
    return abs(ours - reference) / max(1.0, abs(reference))


def run_solve(binary: Path, mps: Path, time_limit: float, options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        sol = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit), "--option", "log_to_console=false"]
        for option in options:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=time_limit + 120)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "crashed", "wall_seconds": wall,
                    "message": completed.stderr.strip()[:300]}
        blob = json.loads(stats.read_text(encoding="utf-8"))
        result, model, effort = blob.get("result", {}), blob.get("model", {}), blob.get("effort", {})
        row = {"status": result.get("status", "unknown"), "algorithm": result.get("algorithm", ""),
               "objective": result.get("objective"), "message": result.get("message", "")[:200],
               "rows": model.get("rows", ""), "columns": model.get("columns", ""),
               "nonzeros": model.get("nonzeros", ""), "iterations": effort.get("iterations", ""),
               "solver_seconds": effort.get("solve_seconds", ""), "wall_seconds": wall,
               "verified": "no point"}
        if sol.exists() and row["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol), "--quiet"],
                                   capture_output=True, text=True)
            row["verified"] = "yes" if check.returncode == 0 else "REJECTED"
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--sizes", nargs="*", default=list(SIZES), choices=SIZES)
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 4, 8])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    if args.binary is None:
        import sankhya
        args.binary = Path(sankhya.locate_executable())
    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = args.out or (REPO_ROOT / "bench" / "results" / f"decomposition-{commit}.csv")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            mps = Path(tmp) / f"refinery-{size}.mps"
            made = subprocess.run([sys.executable, str(GENERATOR), "--size", size, "--seed",
                                   str(SEED), "--out", str(mps)], capture_output=True, text=True)
            if made.returncode != 0:
                print(f"{size}: generator failed: {made.stderr.strip()}", file=sys.stderr)
                return 2
            reference = parse_optimum(made.stdout)
            digest = hashlib.sha256(mps.read_bytes()).hexdigest()
            modes = [("monolithic", 0, [])] + [
                (f"benders-{k}", k, ["decomposition=benders", f"decomposition_blocks={k}"])
                for k in args.blocks]
            monolithic_objective = None
            monolithic_wall = None
            for mode, blocks, options in modes:
                result = run_solve(args.binary, mps, args.time_limit, options)
                objective = result.get("objective")
                if mode == "monolithic":
                    monolithic_objective, monolithic_wall = objective, result["wall_seconds"]
                gap = (relative_gap(objective, reference)
                       if objective is not None and reference is not None else None)
                same = ""
                if objective is not None and monolithic_objective is not None:
                    same = "yes" if relative_gap(objective, monolithic_objective) <= 1e-6 else "NO"
                speedup = (monolithic_wall / result["wall_seconds"]
                           if monolithic_wall and result.get("wall_seconds") else "")
                rows.append({
                    "instance": f"refinery-{size}", "sha256": digest, "mode": mode,
                    "blocks": blocks or "", "rows": result.get("rows", ""),
                    "columns": result.get("columns", ""), "nonzeros": result.get("nonzeros", ""),
                    "status": result["status"], "algorithm": result.get("algorithm", ""),
                    "our_objective": "" if objective is None else repr(objective),
                    "reference_objective": "" if reference is None else repr(reference),
                    "absolute_gap": "" if gap is None else f"{abs(objective - reference):.6e}",
                    "relative_gap": "" if gap is None else f"{gap:.6e}",
                    "matches_monolithic": same, "verified": result["verified"],
                    "wall_seconds": f"{result['wall_seconds']:.3f}",
                    "solver_seconds": result.get("solver_seconds", ""),
                    "iterations": result.get("iterations", ""),
                    "speedup_vs_monolithic": "" if speedup == "" else f"{speedup:.3f}",
                    "message": result.get("message", ""), "git_commit": commit,
                    "machine": machine, "time_limit": args.time_limit, "timestamp_utc": now,
                    "solver_options": " ".join(options)})
                print(f"{size:<7} {mode:<12} {result['status']:<9} {result.get('algorithm', ''):<16}"
                      f"{'' if objective is None else f'{objective:>22.10g}'}  "
                      f"{result['wall_seconds']:8.2f}s  verified={result['verified']}  "
                      f"matches={same or '-'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    rejected = [r for r in rows if r["verified"] == "REJECTED"]
    mismatched = [r for r in rows if r["matches_monolithic"] == "NO"]
    decomposed = [r for r in rows if r["algorithm"] == "benders"]
    print(f"\n{len(decomposed)} of {len(rows) - len(args.sizes)} decomposed solves ran Benders; "
          f"{len(rejected)} rejected by the verifier; {len(mismatched)} disagree with the "
          f"monolithic objective; wrote {out}")
    return 1 if rejected or mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
