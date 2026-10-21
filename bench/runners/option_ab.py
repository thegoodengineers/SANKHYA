#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Option A/B of one binary on one machine, interleaved per instance, over any runner.

Two bench.yml runs of the same suite land on whatever CPU the pool hands out (an EPYC 9V45
for one leg and a 7763 for the other, on the RINS A/B), and the time comparison is spoiled.
This runs both arms in the same process on the same machine: for every instance, the runner
(netlib.py, kennington.py, maros_meszaros.py, mittelmann.py, miplib.py or pooling.py) is
called once for the `off` arm (the shared --solver-option only) and once for the `on` arm
(plus --option), the order flipping from one instance to the next so a drift in the machine
lands on both arms.
Each call writes the runner's own CSV, verifier and all; the rows are then put in one CSV
`<prefix>.csv` with an `arm` column, so every row keeps the runner's machine column (which
names the CPU model) and its git_commit.

Usage:
    python bench/runners/option_ab.py --runner kennington.py --binary build/sankhya \\
        --option ipm_supernodal=true --instances cre-a pds-02 --time-limit 600 \\
        --machine "..." --out-prefix bench/results/option-ab-kennington-supernodal-<sha>
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
# The manifest each runner reads its instance list from, for when --instances is not given.
MANIFESTS = {
    "netlib.py": "data/netlib/reference.json",
    "kennington.py": "data/kennington/reference.json",
    "maros_meszaros.py": "data/maros-meszaros/reference.json",
    "mittelmann.py": "data/mittelmann/reference.json",
    "miplib.py": "data/miplib/reference.json",
    "pooling.py": "data/pooling/reference.json",
}


def instances_of(runner: str) -> list[str]:
    path = REPO_ROOT / MANIFESTS[runner]
    return sorted(json.loads(path.read_text(encoding="utf-8"))["instances"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", required=True, choices=sorted(MANIFESTS))
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--option", action="append", required=True, metavar="KEY=VALUE",
                        help="the `on` arm's extra solver option; repeatable")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="for both arms")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    names = args.instances or instances_of(args.runner)
    arms = {"off": args.solver_option, "on": args.solver_option + args.option}
    work = Path(tempfile.mkdtemp(prefix="option-ab-"))
    rows: list[dict] = []
    fields: list[str] = []
    for repeat in range(1, args.repeats + 1):
        for index, name in enumerate(names):
            order = ("off", "on") if (index + repeat) % 2 else ("on", "off")
            for arm in order:
                out = work / f"{arm}-{repeat}-{name}.csv"
                cmd = [sys.executable, f"bench/runners/{args.runner}", "--binary",
                       str(args.binary), "--instances", name, "--out", str(out)]
                # miplib.py builds its own tag (CPU model included) around a machine kind.
                cmd += (["--machine-kind", args.machine] if args.runner == "miplib.py"
                        else ["--machine", args.machine])
                if args.time_limit is not None:
                    cmd += ["--time-limit", f"{args.time_limit:g}"]
                for option in arms[arm]:
                    cmd += ["--solver-option", option]
                print(f"== r{repeat} {name} {arm}", flush=True)
                subprocess.run(cmd, cwd=REPO_ROOT, check=False)
                if not out.exists():
                    rows.append({"arm": arm, "repeat": repeat, "instance": name,
                                 "status": "no_csv"})
                    continue
                with out.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for field in reader.fieldnames or []:
                        if field not in fields:
                            fields.append(field)
                    for row in reader:
                        rows.append({"arm": arm, "repeat": repeat, **row})

    path = Path(f"{args.out_prefix}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "repeat"] + fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"on arm: {' '.join(args.option)}; {len(names)} instances x {args.repeats} "
          f"repeat(s); wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
