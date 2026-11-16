#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""N scenarios in one run against N separate single solves (#752).

    python bench/runners/scenarios.py --binary build/sankhya --counts 10 100 1000 \
        --arm dual-previous:scenario_start=previous --arm primal-base:algorithm=simplex

For each instance - the demo crude blend and a daily refinery year from the #517 generator
(365 periods, the medium dimensions) - this draws N scenarios from a fixed seed, runs
`sankhya scenarios MODEL SCENARIOS.csv --compare`, and writes one row per scenario to
bench/results/scenarios-<commit>.csv: our objective beside the separate single solve's (the
reference), the gap, the status, whether the in-process check verified it, the times, the
commit the binary was built from, the machine and the sha256 of the instance.

THE SCENARIOS. The blend: each crude's margin moved by a uniform +/-3 $/bbl (a price move).
The refinery, as two sets: price sets (every crude's purchase cost scaled by one factor in
[0.9, 1.1]) and demand forecasts (every delivery commitment scaled by one factor in
[0.95, 1.0]). The refinery year's single solves take about half a minute each on the
7.7 GB laptop, so its counts are given separately (--refinery-counts) and kept small.

THE ARMS. The first arm, dual-base, is the shipped default (each scenario restarted by the
dual simplex from the base basis) and the one run with --compare, so the separate single
solves are timed once. Each --arm NAME:opt=val,opt=val runs the same scenarios again with
those solver options, e.g. scenario_start=previous (restart from the scenario before) or
algorithm=simplex (the primal restart a pure cost move keeps primal feasible); its rows are
held to the same single solves.

EVERY ANSWER IS VERIFIED OUTSIDE THE SOLVER. Each run writes every scenario's own model and
answer (--write-dir); every .sol goes through tools/verify_solution.py, which shares no code
with the solver, and every written model is checked here, with the verifier's own MPS reader,
to be the base model with exactly that scenario's overrides, so our writer cannot hand the
verifier a wrong model. The per-scenario CSV records both verdicts; a second CSV,
scenario-summary-<commit>.csv (not scenarios-*, which tools/make_report.py reads as
per-scenario rows), has one row per (instance, N, arm): the run's wall time with the base
solve, the single solves' total, and their ratio.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench" / "runners"))
sys.path.insert(0, str(REPO / "tools"))

import stamp  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402

TOTAL = re.compile(r"from one read, base solve \S+ in \d+ iteration\(s\): ([0-9.]+)s total")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend_scenarios(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    names = model.col_names
    lines = ["scenario," + ",".join(f"cost:{n}" for n in names)]
    for k in range(count):
        costs = [model.col_cost[j] + rng.uniform(-3.0, 3.0) for j in range(len(names))]
        lines.append(f"p{k}," + ",".join(f"{c:.6f}" for c in costs))
    return lines


def refinery_prices(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    buys = [j for j, n in enumerate(model.col_names) if n.startswith("BUY_")]
    lines = ["scenario," + ",".join(f"cost:{model.col_names[j]}" for j in buys)]
    for k in range(count):
        price = rng.uniform(0.9, 1.1)
        lines.append(f"p{k}," + ",".join(f"{model.col_cost[j] * price:.9g}" for j in buys))
    return lines


def refinery_demand(model_path: Path, count: int, rng: random.Random) -> list[str]:
    model = parse_mps(model_path)
    commits = [i for i, n in enumerate(model.row_names) if n.startswith("COMMIT")]
    lines = ["scenario," + ",".join(f"rhs:{model.row_names[i]}" for i in commits)]
    for k in range(count):
        demand = rng.uniform(0.95, 1.0)
        lines.append(f"d{k}," + ",".join(f"{model.row_lower[i] * demand:.9g}" for i in commits))
    return lines


def run(binary: Path, model: Path, scenario_lines: list[str], work: Path, options: list[str],
        compare: bool, sol_dir: Path) -> tuple[list[dict], float]:
    """The table, and the run's wall time with the base solve (the binary's own clock)."""
    csv_in = work / "scenarios.csv"
    csv_in.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")
    out = work / "out.csv"
    out.unlink(missing_ok=True)
    command = [str(binary), "scenarios", str(model), str(csv_in), "--out", str(out),
               "--write-dir", str(sol_dir)]
    command += ["--compare"] if compare else []
    for option in options:
        command += ["--option", option]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print("\n".join(completed.stdout.strip().splitlines()[-4:]))
    total = TOTAL.search(completed.stdout)
    if not out.exists() or total is None:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"{binary} scenarios wrote no table for {model}")
    with open(out, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), float(total.group(1))


def overrides_of(scenario_lines: list[str]) -> dict[str, list[tuple[str, str, float]]]:
    """Each scenario's (kind, name, value) overrides, read back from the scenario file."""
    rows = [line.split(",") for line in scenario_lines if line and not line.startswith("#")]
    header = [cell.split(":", 1) for cell in rows[0][1:]]
    return {row[0]: [(kind, name, float(cell)) for (kind, name), cell in zip(header, row[1:])
                     if cell.strip()] for row in rows[1:]}


def close(a: float, b: float) -> bool:
    """Equal up to the last digits of a 17-digit round trip; a model identity check, not a
    grading tolerance."""
    return a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def model_check(base_path: str, written_path: str,
                overrides: list[tuple[str, str, float]]) -> str:
    """'ok' when the written model is the base with exactly these overrides, else the first
    difference. Both files are read with the verifier's MPS reader, not ours."""
    base, got = parse_mps(Path(base_path)), parse_mps(Path(written_path))
    if base.col_names != got.col_names or base.row_names != got.row_names:
        return "names differ"
    want = {"col_cost": list(base.col_cost), "col_lower": list(base.col_lower),
            "col_upper": list(base.col_upper), "row_lower": list(base.row_lower),
            "row_upper": list(base.row_upper)}
    for kind, name, value in overrides:
        if kind == "rhs":
            i = base.row_index[name]
            for side in ("row_lower", "row_upper"):
                if math.isfinite(getattr(base, side)[i]):
                    want[side][i] = value
        elif kind == "cost":
            want["col_cost"][base.col_index[name]] = value
        else:
            want[kind][(base.col_index if kind.startswith("col") else base.row_index)[name]] = value
    for field, values in want.items():
        for i, (a, b) in enumerate(zip(values, getattr(got, field))):
            if not close(a, b):
                return f"{field}[{i}] is {b!r}, expected {a!r}"
    if base.maximize != got.maximize or not close(base.objective_offset, got.objective_offset):
        return "objective sense or offset differs"
    for j, (a, b) in enumerate(zip(base.entries, got.entries)):
        a, b = sorted(a), sorted(b)
        if len(a) != len(b) or any(ra != rb or not close(va, vb)
                                   for (ra, va), (rb, vb) in zip(a, b)):
            return f"column {base.col_names[j]}'s entries differ"
    return "ok"


def verify_solution(model: str, sol: str) -> str:
    completed = subprocess.run([sys.executable, str(REPO / "tools" / "verify_solution.py"),
                                "--quiet", model, sol], capture_output=True, text=True,
                               check=False)
    if completed.returncode == 0:
        return "pass"
    last = (completed.stdout.strip().splitlines() or ["(no output)"])[-1]
    return "fail: " + last.replace(",", ";")


def verify_all(base: Path, table: list[dict],
               overrides: dict[str, list[tuple[str, str, float]]]) -> list[tuple[str, str]]:
    """(verify_solution.py verdict, written-model check) per row, in parallel; not timed."""
    with concurrent.futures.ProcessPoolExecutor(os.cpu_count() or 1) as pool:
        models = [r["sol_file"][:-len(".sol")] + ".mps" for r in table]
        checks = [pool.submit(model_check, str(base), m, overrides[r["scenario"]])
                  for m, r in zip(models, table)]
        verdicts = [pool.submit(verify_solution, m, r["sol_file"])
                    for m, r in zip(models, table)]
        return [(v.result(), c.result()) for v, c in zip(verdicts, checks)]


def number(text: str) -> float | None:
    return float(text) if text not in ("", "-") else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--seed", type=int, default=752)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--refinery-counts", type=int, nargs="*", default=[10])
    parser.add_argument("--arm", action="append", default=[],
                        help="NAME:opt=val,opt=val, run after the default dual-base arm")
    args = parser.parse_args()

    commit = stamp.stamp(args.binary)
    machine = args.machine or f"{platform.system()}-{platform.machine()}"
    out = args.out or REPO / "bench" / "results" / f"scenarios-{commit}.csv"
    summary_out = args.summary or out.with_name(f"scenario-summary-{commit}.csv")
    arms: list[tuple[str, list[str]]] = [("dual-base", [])]
    for spec in args.arm:
        name, _, opts = spec.partition(":")
        arms.append((name, [o for o in opts.split(",") if o]))

    rows: list[dict] = []
    summary: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        instances = [("crude_blend", REPO / "demo" / "crude_blend.mps", blend_scenarios,
                      args.counts)]
        if args.refinery_counts:
            year = work / "refinery_daily_year.mps"
            subprocess.run([sys.executable,
                            str(REPO / "bench" / "case_studies" / "refinery" / "generator.py"),
                            "--size", "medium", "--periods", "365", "--seed", "1",
                            "--out", str(year)], check=True)
            instances.append(("refinery_daily_year:prices", year, refinery_prices,
                              args.refinery_counts))
            instances.append(("refinery_daily_year:demand", year, refinery_demand,
                              args.refinery_counts))
        for name, model, make, counts in instances:
            digest = sha256(model)
            for count in counts:
                lines = make(model, count, random.Random(args.seed + count))
                overrides = overrides_of(lines)
                singles: dict[str, dict] = {}
                for arm, options in arms:
                    print(f"== {name}, N = {count}, arm {arm} {' '.join(options)}", flush=True)
                    sol_dir = work / "answers"
                    shutil.rmtree(sol_dir, ignore_errors=True)
                    table, total = run(args.binary, model, lines, work, options,
                                       compare=not singles, sol_dir=sol_dir)
                    if not singles:
                        singles = {r["scenario"]: r for r in table}
                    checks = verify_all(model, table, overrides)
                    first = len(rows)
                    for r, (verdict, written) in zip(table, checks):
                        single = singles[r["scenario"]]
                        ours, ref = number(r["objective"]), number(single["single_objective"])
                        gap = abs(ours - ref) if ours is not None and ref is not None else ""
                        rel = (gap / max(1.0, abs(ref))) if gap != "" else ""
                        # The status as the single solve prints it: ours carries an
                        # "unverified-" prefix when the check failed, which then disagrees.
                        agrees = (r["status"] == single["single_status"]
                                  and (rel == "" or rel <= 1e-8))
                        rows.append({
                            "instance": name, "arm": arm, "options": " ".join(options),
                            "n_scenarios": count, "scenario": r["scenario"],
                            "instance_sha256": digest, "objective": r["objective"],
                            "reference_objective": single["single_objective"],
                            "reference": "separate single solve", "abs_gap": gap,
                            "rel_gap": rel, "status": r["status"], "verified": r["verified"],
                            "verify_solution": verdict, "written_model": written,
                            "agrees": "yes" if agrees else "no",
                            "wall_seconds": r["seconds"],
                            "single_wall_seconds": single["single_seconds"],
                            "iterations": r["iterations"], "git_commit": commit,
                            "machine": machine})
                    mine = rows[first:]
                    single_total = sum(float(singles[r["scenario"]]["single_seconds"])
                                       for r in table)
                    summary.append({
                        "instance": name, "arm": arm, "options": " ".join(options),
                        "n_scenarios": count, "instance_sha256": digest,
                        "run_seconds_with_base_solve": f"{total:.4f}",
                        "single_solves_seconds": f"{single_total:.4f}",
                        "speedup": f"{single_total / total:.2f}" if total > 0 else "",
                        "iterations": sum(int(r["iterations"]) for r in table),
                        "verified": sum(r["verified"] == "yes" for r in mine),
                        "verify_solution_pass": sum(r["verify_solution"] == "pass" for r in mine),
                        "written_model_ok": sum(r["written_model"] == "ok" for r in mine),
                        "agrees": sum(r["agrees"] == "yes" for r in mine),
                        "git_commit": commit, "machine": machine})
                    print(summary[-1], flush=True)
    for path, data in ((out, rows), (summary_out, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    bad = [r for r in rows if r["verified"] != "yes" or r["agrees"] != "yes"
           or r["verify_solution"] != "pass" or r["written_model"] != "ok"]
    print(f"wrote {len(rows)} rows to {out} and {len(summary)} to {summary_out}; "
          f"{len(bad)} unverified, rejected by verify_solution.py, misread, or disagreeing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
