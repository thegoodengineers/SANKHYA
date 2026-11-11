#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time to the first incumbent on the models where the search has none, one arm per option.

The subset is first_incumbent.json: MIPLIB models on which a committed run ended without a
point, or with a dual bound at the optimum and an incumbent away from it, each with the CSV
that says so. For every arm (the defaults, then each --arm NAME=opt=val,opt=val) this calls
miplib.py once per set on those instances, so every row is graded, verified and stamped
exactly as a tier run's is, and joins the rows into one CSV with an `arm` column. It then
prints, per instance and arm, the status, the time to the first feasible point and the
objective, and per arm how many instances got a point.

With --logs DIR it also solves each instance once more under the defaults at
log_level=verbose and keeps the log there: what each heuristic did, in the solver's words.

    python bench/runners/first_incumbent.py --binary build/sankhya --time-limit 60 \\
        --arm heuristics=mip_heuristics=true --arm local-mip=mip_local_mip=true \\
        --out bench/results/first-incumbent-<sha>.csv --logs /tmp/logs
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET = Path(__file__).resolve().parent / "first_incumbent.json"
MIPLIB = Path(__file__).resolve().parent / "miplib.py"
DIRS = {"miplib2017-bench": ("data/miplib2017-bench", ".mps.gz"),
        "miplib3": ("data/miplib3", ".mps")}


def parse_arm(text: str) -> tuple[str, list[str]]:
    name, _, options = text.partition("=")
    if not name or not options:
        raise SystemExit(f"--arm wants NAME=opt=val[,opt=val], got {text!r}")
    return name, options.split(",")


def run_arm(binary: Path, set_name: str, names: list[str], options: list[str],
            time_limit: float, machine_kind: str | None) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "arm.csv"
        command = [sys.executable, str(MIPLIB), "--binary", str(binary), "--set", set_name,
                   "--threads", "1", "--time-limit", str(time_limit), "--out", str(out),
                   "--instances", *names]
        for option in options:
            command += ["--solver-option", option]
        if machine_kind:
            command += ["--machine-kind", machine_kind]
        subprocess.run(command, check=False)  # miplib.py exits 1 when a row does not pass
        if not out.exists():
            return []
        with out.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=opt=val[,..]")
    parser.add_argument("--instances", nargs="*", help="a part of the subset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--machine-kind", default=None)
    args = parser.parse_args()

    subset = json.loads(SUBSET.read_text(encoding="utf-8"))["instances"]
    if args.instances:
        subset = {name: entry for name, entry in subset.items() if name in args.instances}
    arms = [("defaults", [])] + [parse_arm(a) for a in args.arm]
    rows: list[dict] = []
    for arm, options in arms:
        for set_name in DIRS:
            names = sorted(n for n, e in subset.items() if e["set"] == set_name)
            if not names:
                continue
            for row in run_arm(args.binary, set_name, names, options, args.time_limit,
                               args.machine_kind):
                rows.append({"arm": arm, "set": set_name, **row})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with args.out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    by = {(r["arm"], r["instance"]): r for r in rows}
    print(f"\n{'instance':<24}" + "".join(f"{arm:>34}" for arm, _ in arms))
    for name in sorted(subset):
        cells = []
        for arm, _ in arms:
            r = by.get((arm, name))
            if r is None:
                cells.append("-")
                continue
            first = r.get("time_to_first_feasible") or ""
            point = f"first {float(first):.1f}s" if first else "no point"
            cells.append(f"{r['status']}, {point}, m{r['matched_published']}")
        print(f"{name:<24}" + "".join(f"{c:>34}" for c in cells))
    for arm, _ in arms:
        mine = [r for r in rows if r["arm"] == arm]
        print(f"{arm}: a point on {sum(bool(r.get('time_to_first_feasible')) for r in mine)} "
              f"of {len(mine)}, matched {sum(r['matched_published'] == '1' for r in mine)}, "
              f"proved {sum(r['proved_optimal'] == '1' for r in mine)}")
    print(f"wrote {args.out}")

    if args.logs is not None:
        args.logs.mkdir(parents=True, exist_ok=True)
        for name, entry in sorted(subset.items()):
            folder, suffix = DIRS[entry["set"]]
            model = REPO_ROOT / folder / f"{name}{suffix}"
            with (args.logs / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(args.binary), "solve", str(model), "--time-limit",
                                str(args.time_limit), "--option", "mip_threads=1",
                                "--option", "log_level=verbose"],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
