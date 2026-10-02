#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Semi-continuous and SOS models solved two ways, native branching against the binary
reformulation (#754).

The branch and bound branches on a semi-continuous column (x = 0 or x >= l) and on a special
ordered set (Beale and Tomlin's split at the weighted centre) natively; `sos_reformulate=true`
writes the same conditions as binaries and big-M rows and solves that MILP instead. Both are
exact methods for the same model, so on every instance they must report the same optimum.
This runner generates the two model families the issue names, from a seed:

  pwl     a production plan whose cost curves are piecewise linear and NOT convex (marginal
          costs drawn in random order, economies of scale included): product k's output is
          sum_j b_kj l_kj with sum_j l_kj = 1 and l_k an SOS2 over its breakpoints b_kj,
          shared resource rows and a demand row; profit is maximised as cost minus revenue
          minimised
  onoff   units that are off or run between a minimum and a maximum rate, p_ut in {0} or
          [pmin_u, pmax_u] as semi-continuous columns, a demand row per period and one
          emissions budget across all periods that couples them

and solves each instance natively and with sos_reformulate=true, with the gap targets at
1e-9 so "optimal" means the tree closed in both arms. Every .sol is checked by
tools/verify_solution.py, which re-reads the MPS file with its own reader and checks both
conditions on the original model. A row agrees when both arms are optimal and verified and
their objectives match to 1e-6 relative.

The CSV is bench/results/sos-sc-ab-<commit>.csv: instance, sha256 of the MPS file, both
objectives, statuses, solve times and node counts, the verdicts, the git commit (from the
binary, bench/runners/stamp.py) and the machine.

Usage:
    python bench/runners/sos_sc_ab.py --binary build/sankhya.exe
    python bench/runners/sos_sc_ab.py --seeds 5 --time-limit 120 --keep sos-models/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# (products, breakpoints) and (units, periods) per size.
PWL_SIZES = [(8, 6), (16, 8), (32, 10)]
ONOFF_SIZES = [(10, 8), (20, 12), (30, 24)]


# ---------------------------------------------------------------------------------------
# The two families, written as free-format MPS by this script (not by the solver's writer)
# ---------------------------------------------------------------------------------------

class Mps:
    """Rows, columns, bounds, SC flags and sets, written in one pass."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[tuple[str, str, float]] = []  # (sense, name, rhs)
        self.cols: list[tuple[str, float, dict[str, float]]] = []  # (name, cost, {row: a})
        self.bounds: list[str] = []
        self.sets: list[tuple[int, str, list[tuple[str, float]]]] = []

    def text(self, relaxed: bool = False) -> str:
        """The model, or with `relaxed` its continuous relaxation: every semi-continuous
        column from 0 to its maximum, no sets."""
        out = [f"NAME          {self.name}", "ROWS", " N  COST"]
        out += [f" {sense}  {name}" for sense, name, _ in self.rows]
        out.append("COLUMNS")
        for name, cost, entries in self.cols:
            pairs = ([("COST", cost)] if cost != 0.0 else []) + list(entries.items())
            if not pairs:
                pairs = [("COST", 0.0)]
            out += [f"    {name}  {row}  {value!r}" for row, value in pairs]
        out.append("RHS")
        out += [f"    RHS  {name}  {rhs!r}" for _, name, rhs in self.rows if rhs != 0.0]
        out.append("BOUNDS")
        if relaxed:
            out += [line.replace(" SC BND", " UP BND") for line in self.bounds
                    if not line.startswith(" LO BND")]
        else:
            out += self.bounds
        if self.sets and not relaxed:
            out.append("SOS")
            for kind, name, members in self.sets:
                out.append(f" S{kind} SOS  {name}  0")
                out += [f"    {name}  {col}  {weight!r}" for col, weight in members]
        out.append("ENDATA")
        return "\n".join(out) + "\n"


def pwl_model(products: int, breakpoints: int, seed: int) -> Mps:
    rng = random.Random(f"pwl-{products}-{breakpoints}-{seed}")
    model = Mps(f"PWL_K{products}_B{breakpoints}_S{seed}")
    resources = max(2, products // 2)
    usage = [[rng.randint(1, 6) for _ in range(products)] for _ in range(resources)]
    top = [rng.randint(20, 60) for _ in range(products)]
    for k in range(products):
        model.rows.append(("E", f"CONV_{k}", 1.0))
    for i in range(resources):
        cap = 0.5 * sum(usage[i][k] * top[k] for k in range(products))
        model.rows.append(("L", f"RES_{i}", float(round(cap))))
    demand = 0.25 * sum(top)
    model.rows.append(("G", "DEMAND", float(round(demand))))
    # A minimum output per product, below the uniform 0.4 * top that meets every other row:
    # with it binding, the relaxation reaches the minimum by mixing breakpoints that are not
    # adjacent, which is what SOS2 forbids.
    floor = [rng.randint(0, 35) / 100.0 * top[k] for k in range(products)]
    for k in range(products):
        model.rows.append(("G", f"MIN_{k}", float(round(floor[k], 2))))
    for k in range(products):
        cuts = sorted(rng.sample(range(1, top[k]), breakpoints - 2))
        points = [0] + cuts + [top[k]]
        # Marginal costs in random order: the curve is neither convex nor concave, so the
        # LP relaxation mixes breakpoints that are not adjacent and SOS2 branching matters.
        price = rng.randint(6, 12)
        cost = 0.0
        members = []
        for j, b in enumerate(points):
            if j > 0:
                cost += rng.randint(2, 14) * (b - points[j - 1])
            name = f"L_{k}_{j}"
            entries = {f"CONV_{k}": 1.0}
            for i in range(resources):
                if b:
                    entries[f"RES_{i}"] = float(usage[i][k] * b)
            if b:
                entries["DEMAND"] = float(b)
                entries[f"MIN_{k}"] = float(b)
            model.cols.append((name, cost - price * b, entries))
            model.bounds.append(f" UP BND  {name}  1")
            members.append((name, float(b)))
        model.sets.append((2, f"CURVE_{k}", members))
    return model


def onoff_model(units: int, periods: int, seed: int) -> Mps:
    rng = random.Random(f"onoff-{units}-{periods}-{seed}")
    model = Mps(f"ONOFF_U{units}_T{periods}_S{seed}")
    pmax = [rng.randint(20, 100) for _ in range(units)]
    # Cheaper units carry larger minimum rates (base load), so the LP relaxation's marginal
    # unit tends to run below its minimum and the on/off decision matters.
    cost = sorted((rng.randint(10, 60) for _ in range(units)))
    share = sorted((rng.uniform(0.2, 0.7) for _ in range(units)), reverse=True)
    pmin = [max(1, round(share[u] * pmax[u])) for u in range(units)]
    emission = [rng.randint(1, 9) for _ in range(units)]
    demand = [round(rng.uniform(0.3, 0.75) * sum(pmax)) for _ in range(periods)]
    # An emissions budget a feasible schedule meets: units in cost order, each at its maximum,
    # the last one raised to its minimum when the demand left is below it.
    greedy = 0.0
    for t in range(periods):
        left = demand[t]
        for u in range(units):
            if left <= 0:
                break
            run = min(pmax[u], max(pmin[u], left))
            greedy += emission[u] * run
            left -= run
    for t in range(periods):
        model.rows.append(("G", f"DEMAND_{t}", float(demand[t])))
    model.rows.append(("L", "EMISSIONS", float(round(1.05 * greedy))))
    for t in range(periods):
        for u in range(units):
            name = f"P_{u}_{t}"
            model.cols.append((name, float(cost[u]),
                               {f"DEMAND_{t}": 1.0, "EMISSIONS": float(emission[u])}))
            model.bounds.append(f" LO BND  {name}  {pmin[u]}")
            model.bounds.append(f" SC BND  {name}  {pmax[u]}")
    return model


# ---------------------------------------------------------------------------------------
# Solving and recording
# ---------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(binary: Path, mps: Path, reformulate: bool, time_limit: float) -> dict:
    arm = "reformulated" if reformulate else "native"
    sol = mps.with_suffix(f".{arm}.sol")
    stats = mps.with_suffix(f".{arm}.json")
    cmd = [str(binary), "solve", str(mps), "--write-sol", str(sol), "--stats", str(stats),
           "--time-limit", str(time_limit), "--option", "mip_relative_gap=1e-9",
           "--option", "mip_absolute_gap=1e-9", "--option", "log_to_console=false"]
    if reformulate:
        cmd += ["--option", "sos_reformulate=true"]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not stats.exists():
        return {"status": "no_output", "objective": "", "nodes": "", "seconds": "",
                "verified": "0"}
    blob = json.loads(stats.read_text())
    check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol), "--quiet"],
                           capture_output=True, text=True, check=False)
    return {"status": blob["result"]["status"], "objective": blob["result"]["objective"],
            "nodes": blob["effort"]["nodes"], "seconds": blob["effort"]["solve_seconds"],
            "verified": "1" if check.returncode == 0 else "0"}


def relaxation_objective(binary: Path, mps: Path, model: Mps, time_limit: float) -> str:
    """The objective of the model with the conditions dropped (both families minimise), so
    the CSV shows on which instances they change the answer and the branching had work."""
    relaxed = mps.with_suffix(".relaxed.mps")
    relaxed.write_text(model.text(relaxed=True), newline="\n")
    stats = relaxed.with_suffix(".json")
    subprocess.run([str(binary), "solve", str(relaxed), "--stats", str(stats), "--time-limit",
                    str(time_limit), "--option", "log_to_console=false"],
                   capture_output=True, text=True, check=False)
    if not stats.exists():
        return ""
    blob = json.loads(stats.read_text())
    return blob["result"]["objective"] if blob["result"]["status"] == "optimal" else ""


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya"):
        path = REPO_ROOT / candidate
        if path.is_file():
            return path
    raise SystemExit("no build/sankhya; pass --binary")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=3, help="instances per size and family")
    parser.add_argument("--time-limit", type=float, default=60.0, help="per solve, seconds")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path; default bench/results/sos-sc-ab-<commit>.csv")
    parser.add_argument("--keep", type=Path, default=None, help="keep the MPS files here")
    parser.add_argument("--machine", default=None,
                        help="machine tag for every row; defaults to <system>-<arch>")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    commit = stamp.stamp(binary)
    machine = args.machine or f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = args.out or REPO_ROOT / "bench" / "results" / f"sos-sc-ab-{commit}.csv"

    jobs = [("pwl", k, b, s, pwl_model(k, b, s)) for k, b in PWL_SIZES
            for s in range(1, args.seeds + 1)]
    jobs += [("onoff", u, t, s, onoff_model(u, t, s)) for u, t in ONOFF_SIZES
             for s in range(1, args.seeds + 1)]

    print(f"solver   {binary}\ncommit   {commit}   machine {machine}   "
          f"{len(jobs)} instances   time limit {args.time_limit:g} s per solve")
    print(f"{'instance':<22} {'native':>16} {'nodes':>7} {'s':>7}   "
          f"{'reformulated':>16} {'nodes':>7} {'s':>7}  agree")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for family, a, b, seed, model in jobs:
            mps = work / f"{model.name.lower()}.mps"
            # LF on every platform, so the sha256 in the CSV is the same wherever it ran.
            mps.write_text(model.text(), newline="\n")
            native = solve(binary, mps, False, args.time_limit)
            reform = solve(binary, mps, True, args.time_limit)
            relaxation = relaxation_objective(binary, mps, model, args.time_limit)
            binds = ""
            if relaxation != "" and native["status"] == "optimal":
                gap = float(native["objective"]) - float(relaxation)
                binds = "1" if gap > 1e-6 * max(1.0, abs(float(native["objective"]))) else "0"
            agree = "0"
            abs_diff = rel_diff = ""
            if native["status"] == "optimal" and reform["status"] == "optimal":
                diff = abs(float(native["objective"]) - float(reform["objective"]))
                scale = max(1.0, abs(float(native["objective"])))
                abs_diff, rel_diff = f"{diff:.3e}", f"{diff / scale:.3e}"
                if diff <= 1e-6 * scale and native["verified"] == reform["verified"] == "1":
                    agree = "1"
            rows.append({
                "family": family, "instance": model.name, "seed": seed,
                "size": f"{a}x{b}", "sha256": sha256(mps),
                "status_native": native["status"], "objective_native": native["objective"],
                "nodes_native": native["nodes"], "time_native_s": native["seconds"],
                "verified_native": native["verified"],
                "status_reformulated": reform["status"],
                "objective_reformulated": reform["objective"],
                "nodes_reformulated": reform["nodes"],
                "time_reformulated_s": reform["seconds"],
                "verified_reformulated": reform["verified"],
                "abs_diff": abs_diff, "rel_diff": rel_diff, "agree": agree,
                "objective_relaxation": relaxation, "conditions_bind": binds,
                "time_limit_s": args.time_limit, "git_commit": commit, "machine": machine,
                "timestamp_utc": timestamp})
            print(f"{model.name:<22} {native['objective']!s:>16} {native['nodes']!s:>7} "
                  f"{float(native['seconds'] or 0):7.2f}   {reform['objective']!s:>16} "
                  f"{reform['nodes']!s:>7} {float(reform['seconds'] or 0):7.2f}  "
                  f"{'yes' if agree == '1' else 'NO'}", flush=True)
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                shutil.copy(mps, args.keep / mps.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agreed = sum(r["agree"] == "1" for r in rows)
    binding = sum(r["conditions_bind"] == "1" for r in rows)
    print(f"\n{agreed} of {len(rows)} instances agree (both optimal, both verified, "
          f"objectives within 1e-6 relative); on {binding} the relaxation without the "
          f"conditions is strictly better, so they changed the answer; wrote {out}")
    return 0 if agreed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
