#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The share of a MILP solve spent in Feasibility Jump (#1030).

Runs the solver with default options on the demo crude blend MILP (demo/blend_milp.mps) and
the MILPs of the refinery suite (data/refinery, kind "mip" in its manifest), or with --set
miplib on the fetched MIPLIB tier 1 (seed 0, the published files), reads the
heuristic table the search prints at its exit, and records per instance the solve time, the
seconds Feasibility Jump took over its calls, and their ratio. The CSV contract of
ENGINEERING_RULES.md (sha256, objective against the reference, status, time, commit,
machine) is kept so the rows can be compared across commits.

    python bench/runners/fj_share.py --binary build/sankhya --out bench/results/fj-share-<sha>.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import stamp  # noqa: E402
from compare_suite import machine_tag  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HEURISTIC_LINE = re.compile(
    r"Heuristic feasibility jump\s+calls\s+(\d+)\s+proposed\s+(\d+)\s+improved\s+(\d+)"
    r"\s+work\s+(\d+)\s+([0-9.]+)s")
SOLVE_SECONDS = re.compile(r"^solve seconds\s+([0-9.]+)", re.MULTILINE)


def instances(which: str) -> list[tuple[str, Path, float | None]]:
    rows: list[tuple[str, Path, float | None]] = []
    if which == "refinery":
        rows.append(("blend_milp", REPO_ROOT / "demo" / "blend_milp.mps", None))
        manifest = json.loads((REPO_ROOT / "data" / "refinery" / "manifest.json").read_text())
        for name, entry in sorted(manifest["instances"].items()):
            if entry.get("kind") != "mip":
                continue
            rows.append((name, REPO_ROOT / "data" / "refinery" / f"{name}.mps",
                         entry.get("reference_objective")))
    else:  # the fetched MIPLIB tier 1 (fetch_miplib.py), as miplib.py reads it
        data_dir = REPO_ROOT / "data" / "miplib"
        reference = json.loads((data_dir / "reference.json").read_text())["instances"]
        for name, entry in sorted(reference.items()):
            path = data_dir / f"{name}.mps.gz"
            if path.exists():
                rows.append((name, path, float(entry["published_optimal"])))
    return rows


def run_one(binary: Path, path: Path, time_limit: float, threads: int,
            options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        cmd = [str(binary), "solve", str(path), "--stats", str(stats_path), "--time-limit",
               str(time_limit), "--option", "log_to_console=true", "--option", f"threads={threads}"]
        for option in options:
            cmd += ["--option", option]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    out = proc.stdout
    fj = HEURISTIC_LINE.search(out)
    seconds = SOLVE_SECONDS.search(out)
    result = stats.get("result", {})
    return {
        "status": result.get("status", f"exit {proc.returncode}"),
        "objective": result.get("objective"),
        "nodes": result.get("nodes"),
        "solve_seconds": float(seconds.group(1)) if seconds else None,
        "fj_calls": int(fj.group(1)) if fj else 0,
        "fj_proposed": int(fj.group(2)) if fj else 0,
        "fj_improved": int(fj.group(3)) if fj else 0,
        "fj_work": int(fj.group(4)) if fj else 0,
        "fj_seconds": float(fj.group(5)) if fj else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "sankhya")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--machine-kind", default=None,
                        help="what kind of machine, e.g. github-actions ubuntu-latest")
    parser.add_argument("--set", dest="which", choices=("refinery", "miplib"), default="refinery",
                        help="refinery: demo/blend_milp.mps and the refinery suite MILPs; "
                             "miplib: the fetched MIPLIB tier 1")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every solve; recorded in the solver_options column")
    parser.add_argument("--instances", nargs="*", help="only these (default: the whole set)")
    parser.add_argument("--sweep", default=None, metavar="KEY=V1,V2,...",
                        help="run every instance and repeat once per value of one option, in "
                             "one process on one machine, the order rotating so a drift lands "
                             "on every value; a 60 s objective depends on the CPU, and separate "
                             "jobs can land on different ones (#1030)")
    args = parser.parse_args()
    arms: list[list[str]] = [[]]
    if args.sweep:
        key, values = args.sweep.split("=", 1)
        arms = [[f"{key}={v}"] for v in values.split(",")]
    commit = stamp.stamp(args.binary)
    machine = machine_tag(args.machine_kind)
    fields = ["instance", "sha256", "objective", "reference_objective", "absolute_gap",
              "relative_gap", "status", "solve_seconds", "fj_seconds", "fj_share", "fj_calls",
              "fj_proposed", "fj_improved", "fj_work", "nodes", "repeat", "solver_options",
              "git_commit", "machine"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, path, reference in instances(args.which):
            if args.instances and name not in args.instances:
                continue
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            for repeat, arm in ((rep, arms[(rep + k) % len(arms)])
                                for rep in range(args.repeats) for k in range(len(arms))):
                options = args.solver_option + arm
                r = run_one(args.binary, path, args.time_limit, args.threads, options)
                objective = r["objective"]
                abs_gap = rel_gap = ""
                if reference is not None and isinstance(objective, (int, float)) and math.isfinite(objective):
                    abs_gap = abs(objective - reference)
                    rel_gap = abs_gap / max(1.0, abs(reference))
                share = (r["fj_seconds"] / r["solve_seconds"]
                         if r["solve_seconds"] else "")
                row = {"instance": name, "sha256": sha, "objective": objective,
                       "reference_objective": reference if reference is not None else "",
                       "absolute_gap": abs_gap, "relative_gap": rel_gap, "repeat": repeat,
                       "fj_share": share, "solver_options": " ".join(options),
                       "git_commit": commit, "machine": machine, **r}
                writer.writerow(row)
                handle.flush()
                share_text = "" if share == "" else f"{share:.3f}"
                print(f"{name:<28} {' '.join(options):<36} {r['status']:<10} "
                      f"{r['solve_seconds']!s:>9}s obj {objective}  "
                      f"fj {r['fj_seconds']:.3f}s  share {share_text}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
