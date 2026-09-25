#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Feasibility Jump, CPU (#506) against GPU (#508), on MIPLIB: feasible found at 10 s and 60 s.

The issue's A/B: for each instance, does the search find a feasible point, and when. Each
engine runs once per instance, as its own process (bench/tools/fj_ab.cpp, target
`sankhya-fj-ab`), from the box point closest to zero with an unbounded work budget, stopped
at --seconds (60 by default). The tool reports when the first point was handed over, from the
call (the GPU's context creation and transfers included); "found at 10 s" is a first point at
or before 10 s of that same run, not a second run. A point counts only when every point of
the run passed mip::feasibility_jump_point_is_feasible() against the model read from the file
(`verified=1`); the GPU side also verifies each point before handing it over.

THE INSTANCES are fixed by rule before anything is run: every third instance of the MIPLIB
2017 tier-2 list (bench/runners/miplib_tier2.json, the benchmark set's 60 smallest with a
proven optimum), starting with the first: 20 instances across the sizes. Fetch them with
`python bench/runners/fetch_miplib.py --tier 2`. Pass names to run others.

The CSV (bench/results/gpu-fj-ab-<commit>-<machine>.csv) carries, per instance and engine:
the sha256 of the file, the first-point time, found at 10 s and at the limit, the best
objective found against the published optimum (absolute and relative gap), the effort, the
commit the binary was built from and the machine. Only a run of a build from main is
evidence; a branch run is a smoke run and is quoted as one.

    python bench/runners/gpu_fj_ab.py --build build
    python bench/runners/gpu_fj_ab.py --build build --seconds 60 --engines cpu gpu
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433, #589: the CSV names the commit the BINARY was built from)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
TIER2 = REPO_ROOT / "bench" / "runners" / "miplib_tier2.json"
DATA = REPO_ROOT / "data" / "miplib-tier2"
EARLY_SECONDS = 10.0

CSV_COLUMNS = [
    "instance", "sha256", "engine", "status", "first_seconds", "found_at_10s",
    "found_at_limit", "limit_seconds", "our_objective", "reference_objective", "abs_gap",
    "rel_gap", "points", "verified", "wall_seconds", "work", "moves", "weight_updates",
    "restarts", "rejected", "git_commit", "machine", "gpu", "timestamp_utc",
]


def default_instances() -> list[str]:
    names = [entry["name"] for entry in json.loads(TIER2.read_text())["instances"]]
    return names[::3]


def references() -> dict[str, float]:
    return {entry["name"]: entry["published_optimal"]
            for entry in json.loads(TIER2.read_text())["instances"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def run_one(tool: Path, model: Path, engine: str, seconds: float, seed: int) -> dict[str, str]:
    command = [str(tool), str(model), engine, str(seconds), str(seed)]
    try:
        # The tool stops itself at the limit; the margin covers reading and setting up.
        out = subprocess.run(command, capture_output=True, text=True, check=False,
                             timeout=seconds * 3 + 120)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    fields = {k: v.strip('"') for k, v in fields.items()}
    if not fields:
        fields = {"status": f"crashed({out.returncode})"}
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("instances", nargs="*", help="default: every third tier-2 instance")
    parser.add_argument("--build", default="build", help="build directory with the tool")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--engines", nargs="+", default=["cpu", "gpu"])
    parser.add_argument("--machine", default=platform.node())
    parser.add_argument("--out", help="CSV path (default under bench/results/)")
    args = parser.parse_args()

    build = Path(args.build)
    tool = build / "sankhya-fj-ab"
    cli = build / "sankhya"
    if not tool.exists():
        print(f"{tool} not found: cmake --build {build} --target sankhya-fj-ab", file=sys.stderr)
        return 2
    commit = stamp.stamp(cli if cli.exists() else None)
    names = args.instances or default_instances()
    refs = references()
    card = gpu_name()
    out = Path(args.out) if args.out else RESULTS_DIR / f"gpu-fj-ab-{commit}-{args.machine}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        model = DATA / f"{name}.mps.gz"
        if not model.exists():
            print(f"{name}: not fetched (python bench/runners/fetch_miplib.py --tier 2)",
                  file=sys.stderr)
            continue
        digest = sha256(model)
        for engine in args.engines:
            got = run_one(tool, model, engine, args.seconds, args.seed)
            verified = got.get("verified") == "1"
            found = got.get("status") == "found" and verified
            first = float(got.get("first_seconds", "-1")) if found else -1.0
            ref = refs.get(name)
            ours = float(got["best_objective"]) if found else None
            abs_gap = abs(ours - ref) if ours is not None and ref is not None else None
            rel_gap = (abs_gap / max(1.0, abs(ref))) if abs_gap is not None else None
            row = {
                "instance": name, "sha256": digest, "engine": engine,
                "status": got.get("status", ""),
                "first_seconds": f"{first:.3f}" if found else "",
                "found_at_10s": int(found and first <= EARLY_SECONDS),
                "found_at_limit": int(found),
                "limit_seconds": args.seconds,
                "our_objective": "" if ours is None else repr(ours),
                "reference_objective": "" if ref is None else repr(ref),
                "abs_gap": "" if abs_gap is None else f"{abs_gap:.6g}",
                "rel_gap": "" if rel_gap is None else f"{rel_gap:.6g}",
                "points": got.get("points", ""), "verified": got.get("verified", ""),
                "wall_seconds": got.get("seconds", ""), "work": got.get("work", ""),
                "moves": got.get("moves", ""), "weight_updates": got.get("weight_updates", ""),
                "restarts": got.get("restarts", ""), "rejected": got.get("rejected", ""),
                "git_commit": commit, "machine": args.machine, "gpu": card,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"),
            }
            rows.append(row)
            print(f"{name:<24} {engine:<3} {row['status']:<8} first {row['first_seconds'] or '-':>8}"
                  f"  obj {row['our_objective'] or '-':>22}  ref {row['reference_objective']:>14}"
                  f"  restarts {row['restarts']}", flush=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({shlex.quote(commit)})")
    for engine in args.engines:
        mine = [r for r in rows if r["engine"] == engine]
        early = sum(r["found_at_10s"] for r in mine)
        late = sum(r["found_at_limit"] for r in mine)
        print(f"{engine}: feasible found at {EARLY_SECONDS:g} s: {early}/{len(mine)}, "
              f"at {args.seconds:g} s: {late}/{len(mine)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
