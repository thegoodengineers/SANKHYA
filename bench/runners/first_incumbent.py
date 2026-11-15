#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time to the first incumbent on the models where the search has none, one arm per option.

The subset is first_incumbent.json: MIPLIB models on which a committed run ended without a
point, or with a dual bound at the optimum and an incumbent away from it, each with the CSV
that says so. A case is a model and the options every arm of it runs with (the closed-root
group runs with presolve_probing), so one model can be two cases; the CSV's `case` column
names it and `group` says which of the three groups it belongs to. For every arm (the defaults, then each --arm NAME=opt=val,opt=val) this calls
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
    parser.add_argument("--instances", nargs="*", help="a part of the subset, by case name")
    parser.add_argument("--groups", nargs="*",
                        help="only these groups: no-incumbent, bound-at-optimum, closed-root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--machine-kind", default=None)
    args = parser.parse_args()

    subset = json.loads(SUBSET.read_text(encoding="utf-8"))["instances"]
    if args.instances:
        subset = {name: entry for name, entry in subset.items() if name in args.instances}
    if args.groups:
        subset = {name: entry for name, entry in subset.items()
                  if entry.get("group") in args.groups}
    model_of = {case: entry.get("instance", case) for case, entry in subset.items()}
    # One miplib.py call per (set, the case's own options): the cases that share both.
    batches: dict[tuple, list[str]] = {}
    for case, entry in subset.items():
        batches.setdefault((entry["set"], tuple(entry.get("options", []))), []).append(case)
    arms = [("defaults", [])] + [parse_arm(a) for a in args.arm]
    rows: list[dict] = []
    for arm, options in arms:
        for (set_name, case_options), cases in sorted(batches.items()):
            case_of = {model_of[case]: case for case in cases}
            for row in run_arm(args.binary, set_name, sorted(case_of),
                               list(case_options) + options, args.time_limit,
                               args.machine_kind):
                case = case_of[row["instance"]]
                rows.append({"arm": arm, "set": set_name, "case": case,
                             "group": subset[case].get("group", ""), **row})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with args.out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    by = {(r["arm"], r["case"]): r for r in rows}
    print(f"\n{'case':<24}" + "".join(f"{arm:>34}" for arm, _ in arms))
    for name in sorted(subset, key=lambda n: (subset[n].get("group", ""), n)):
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
    for group in sorted({e.get("group", "") for e in subset.values()}):
        for arm, _ in arms:
            mine = [r for r in rows if r["arm"] == arm and r["group"] == group]
            print(f"{group or 'all'} / {arm}: a point on "
                  f"{sum(bool(r.get('time_to_first_feasible')) for r in mine)} of {len(mine)}, "
                  f"matched {sum(r['matched_published'] == '1' for r in mine)}, "
                  f"proved {sum(r['proved_optimal'] == '1' for r in mine)}")
    print(f"wrote {args.out}")

    if args.logs is not None:
        args.logs.mkdir(parents=True, exist_ok=True)
        for name, entry in sorted(subset.items()):
            folder, suffix = DIRS[entry["set"]]
            model = REPO_ROOT / folder / f"{model_of[name]}{suffix}"
            extra = [part for option in entry.get("options", []) for part in ("--option", option)]
            with (args.logs / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(args.binary), "solve", str(model), "--time-limit",
                                str(args.time_limit), "--option", "mip_threads=1",
                                "--option", "log_level=verbose", *extra],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
