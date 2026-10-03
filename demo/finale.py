#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The finale walk (#758): one refinery MILP, solved, proved, checked, offline.

    demo/finale.sh            (Linux, macOS, Git Bash)      demo\\finale.cmd   (Windows)
    python demo/finale.py [--dry] [--binary PATH] [--keep DIR]

Eight steps, each printing one line of result and the seconds it took. Every number printed
comes from a command this script runs in front of the audience; nothing is typed in.

  1. solve   the small refinery MILP (bench/case_studies/refinery, #517; the medium one does
             not finish its certified search within the walk's five minutes) on the GPU when the binary
             reports a device, on the CPU otherwise, writing a VIPR certificate (#518)
  2. prove   tools/verify_certificate.py checks the certificate in exact rational arithmetic,
             and tools/verify_solution.py checks the plan independently of the solver
  3. plan    the planner's view of the same plant as an LP, with sensitivity ranging: the
             binding limits, their shadow prices, and what relaxing each is worth over its
             allowable range (shadow price x allowable increase)
  4. repair  one delivery commitment raised past what the plant can make: the infeasibility
             is proved by a Farkas certificate and the smallest repair is computed
  5. bundle  the MILP run is packed into an evidence bundle (#526) and replayed
  6. prices  twenty crude price sets solved in one scenario run (#752), every answer
             verified
  7. replan  today's prices and demands moved, the plan re-solved cold and from
             yesterday's basis (`--warm-start`, #218): the same verified optimum, the
             pivot counts side by side
  8. units   the same plant with a minimum run rate on its crude unit (#754): a crude it runs
             in a period runs at least 1/6 of the unit's capacity or not at all, a
             semi-continuous column per run, branched on natively and solved again through
             the binary reformulation (`sos_reformulate`); both answers verified, and equal

Runs from a checkout or an unpacked release archive (#748): it needs the `sankhya` binary in
build/ (or --binary), the Python standard library, and nothing from the network. `--dry`
checks the machine and stops.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EXE = "sankhya.exe" if os.name == "nt" else "sankhya"


def find_binary(explicit: str | None) -> Path | None:
    for candidate in ([explicit] if explicit else []) + [os.environ.get("SANKHYA_BIN", "")] + [
            str(ROOT / d / EXE) for d in ("build", "build-release", "build-dbg")]:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT,
                          env=ENV, **kw)


def step(number: int, name: str, started: float, line: str) -> None:
    print(f"[{number}] {name:<7} {line}  ({time.perf_counter() - started:.2f} s)", flush=True)


def fail(number: int, name: str, what: str, proc: subprocess.CompletedProcess) -> None:
    print(f"[{number}] {name:<7} FAILED: {what}")
    print((proc.stdout + proc.stderr).strip()[-2000:])
    sys.exit(1)


def field(text: str, name: str) -> str:
    match = re.search(rf"^{name}\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "?"


def raise_rhs(mps: Path, row: str, value: float, out: Path) -> None:
    """Copy `mps` with row `row`'s right-hand side set to `value` (fixed or free MPS)."""
    lines, section = [], ""
    for line in mps.read_text().splitlines():
        if line and not line[0].isspace():
            section = line.split()[0]
        elif section == "RHS":
            tokens = line.split()
            for k in range(1, len(tokens) - 1, 2):
                if tokens[k] == row:
                    tokens[k + 1] = repr(value)
                    line = "    " + "  ".join(tokens)
        lines.append(line)
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true", help="check the machine and stop")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--keep", type=Path, default=None, help="keep the run's files here")
    args = parser.parse_args()

    total = time.perf_counter()
    binary = find_binary(args.binary)
    if binary is None:
        print(f"no {EXE} in build/; pass --binary or set SANKHYA_BIN")
        return 1
    version = run([binary, "version"])
    if version.returncode != 0:
        print(f"{binary} does not run: {version.stderr.strip()}")
        return 1
    banner = version.stdout.splitlines()[0]
    gpu_match = re.search(r"GPU (.*)\)$", banner)
    gpu = bool(gpu_match) and not re.match(r"(none|no device)", gpu_match.group(1))
    print(banner)
    print(f"binary  {binary}")
    print(f"python  {sys.version.split()[0]} at {PY}")
    print(f"engine  {'GPU: ' + gpu_match.group(1) if gpu else 'CPU (no GPU device reported)'}")
    needed = ["tools/verify_certificate.py", "tools/verify_solution.py", "tools/report.py",
              "tools/bundle.py", "tools/replay_bundle.py", "tools/repair_infeasibility.py",
              "bench/case_studies/refinery/generator.py", "bench/runners/replan_warm_start.py",
              "bindings/python/sankhya/__init__.py"]
    missing = [p for p in needed if not (ROOT / p).is_file()]
    if missing:
        print("missing: " + ", ".join(missing))
        return 1
    if args.dry:
        print("dry run: the machine is ready")
        return 0

    work = Path(tempfile.mkdtemp(prefix="sankhya-finale-"))
    gen = ROOT / "bench" / "case_studies" / "refinery" / "generator.py"
    milp, lp = work / "refinery_milp.mps", work / "refinery_lp.mps"
    for out, extra in ((milp, ["--milp"]), (lp, [])):
        proc = run([PY, gen, "--size", "small", "--seed", "1", "--out", out, *extra])
        if proc.returncode != 0:
            fail(0, "model", "the generator failed", proc)
    print("model   refinery (small, seed 1), synthetic data - see generator.py")
    print()

    # 1. solve the MILP, writing its proof
    t = time.perf_counter()
    cert, sol = work / "refinery.vipr", work / "refinery_milp.sol"
    stats = work / "refinery_milp.json"
    cmd = [binary, "solve", milp, "--write-sol", sol, "--stats", stats,
           "--option", f"write_certificate={cert}"]
    proc = run(cmd + (["--gpu"] if gpu else []))
    status = field(proc.stdout, "status")
    if proc.returncode != 0 or status != "optimal":
        fail(1, "solve", f"status {status}", proc)
    effort = json.loads(stats.read_text())["effort"]
    milp_objective = float(field(proc.stdout, "objective"))
    step(1, "solve", t, f"MILP {status}, objective {field(proc.stdout, 'objective')}, "
         f"{effort['nodes']} nodes, root bound {effort['root_bound']:.6g}, "
         f"on the {'GPU' if gpu else 'CPU'}")

    # 2. prove and check
    t = time.perf_counter()
    # The bound is checked exactly; the plan may violate a row by rounding, at most 1e-9, as
    # in demo/run_sih_demo.sh.
    proof = run([PY, "tools/verify_certificate.py", cert, "--mps", milp, "--feas-tol", "1e-9"])
    if proof.returncode != 0:
        fail(2, "prove", "the certificate did not verify", proof)
    check = run([PY, "tools/verify_solution.py", milp, sol, "--quiet"])
    if check.returncode != 0:
        fail(2, "prove", "the independent verifier rejected the plan", check)
    verdict_line = next((ln for ln in proof.stdout.splitlines() if ln.startswith("VERIFIED")),
                        "?")
    step(2, "prove", t, f"certificate {verdict_line} in exact arithmetic; "
         f"plan: {check.stdout.strip().splitlines()[-1]}")

    # 3. the planner's view on the LP
    t = time.perf_counter()
    lp_sol = work / "refinery_lp.sol"
    proc = run([binary, "solve", lp, "--ranging", "--write-sol", lp_sol])
    if field(proc.stdout, "status") != "optimal":
        fail(3, "plan", "the LP did not solve", proc)
    report = run([PY, "tools/report.py", lp, lp_sol, "--format", "json"])
    if report.returncode != 0:
        fail(3, "plan", "the report failed", report)
    binding = json.loads(report.stdout)["binding_constraints"]
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_solution_mps import parse_mps
    source = parse_mps(lp)
    # What one more unit of each limit is worth. The shadow price alone can overstate it: at
    # a degenerate vertex the price holds for no relaxation at all (its ranging interval is
    # 0 on the relaxing side), and on this plant the limits with the largest prices are all
    # of that kind. So the eight limits with the largest price are each re-solved one unit
    # looser, ranked by the gain those re-solves measured, and the step prints the top three
    # beside their prices. Balance rows (equalities) are not limits a planner can relax, and
    # are skipped.
    base = float(field(proc.stdout, "objective"))
    sense = -1.0 if source.maximize else 1.0
    probe, probe_sol = work / "refinery_probe.mps", work / "probe.sol"
    limits = [row for row in binding
              if source.row_lower[source.row_index[row["name"]]]
              != source.row_upper[source.row_index[row["name"]]]]
    measured = []
    for row in limits[:8]:
        looser = row["bound"] + (1.0 if row["at"] == "upper" else -1.0)
        raise_rhs(lp, row["name"], looser, probe)
        if probe.read_text() == lp.read_text():
            continue  # the limit sits at 0 with no RHS entry; nothing to loosen in place
        again = run([binary, "solve", probe, "--write-sol", probe_sol])
        if field(again.stdout, "status") != "optimal":
            continue
        measured.append((sense * (base - float(field(again.stdout, "objective"))), row))
    measured.sort(key=lambda pair: -pair[0])
    worth = [f"{row['name']} {gain:.4g} (price {abs(row['shadow_price']):.4g})"
             for gain, row in measured[:3] if gain > 0] or ["none of them gains"]
    overstated = sum(1 for gain, row in measured if gain < 0.5 * abs(row["shadow_price"]))
    degenerate = sum(1 for row in limits
                     if (row.get("ranging") or {}).get(
                         "allow_increase" if row["at"] == "upper" else "allow_decrease",
                         1.0) <= 1e-9 * max(1.0, abs(row["bound"])))
    step(3, "plan", t, f"LP optimum {field(proc.stdout, 'objective')}; {len(limits)} binding "
         f"limits, {degenerate} priced at a degenerate vertex; one more unit, re-solved, "
         f"is worth most on: " + "; ".join(worth)
         + (f"; the price overstates the next unit on {overstated} of the {len(measured)} "
            f"re-solved" if overstated else ""))

    # 4. an impossible demand, proved and repaired
    t = time.perf_counter()
    target = next(n for n in source.row_names if n.startswith("COMMIT_"))
    base = source.row_lower[source.row_index[target]]
    bad, bad_sol = work / "refinery_impossible.mps", work / "impossible.sol"
    for factor in (10, 100, 1000):
        raise_rhs(lp, target, base * factor, bad)
        proc = run([binary, "solve", bad, "--write-sol", bad_sol])
        status = field(proc.stdout, "status")
        if status == "infeasible":
            break
    verdict = run([PY, "tools/verify_solution.py", bad, bad_sol, "--quiet"])
    if status != "infeasible" or verdict.returncode != 0:
        fail(4, "repair", f"expected a verified infeasibility, got {status}", verdict)
    repair = run([PY, "tools/repair_infeasibility.py", bad, "--format", "json"])
    if repair.returncode != 0:
        fail(4, "repair", "the repair failed", repair)
    fix = json.loads(repair.stdout)
    moves = fix.get("relaxations", [])
    if not fix.get("repairable") or not moves:
        fail(4, "repair", "no repair found", repair)
    shown = "; ".join(f"{m['name']} {m['side']} by {m['amount']:,.6g}" for m in moves[:2])
    step(4, "repair", t, f"{target} raised {base:g} -> {base * factor:g}: infeasible, Farkas "
         f"proof verified; "
         f"smallest repair {fix['total_weighted_relaxation']:,.6g} over {len(moves)} "
         f"limit(s): {shown}")

    # 5. evidence bundle and replay
    t = time.perf_counter()
    bundle = work / "finale_bundle.zip"
    made = run([PY, "tools/bundle.py", milp, "--binary", binary, "--certificate", "--bundle",
                bundle])
    if made.returncode != 0:
        fail(5, "bundle", "bundling failed", made)
    replay = run([PY, "tools/replay_bundle.py", bundle])
    if replay.returncode != 0:
        fail(5, "bundle", "the replay failed", replay)
    step(5, "bundle", t, f"{bundle.name} ({bundle.stat().st_size // 1024} KiB) replayed: "
         f"manifest intact, verifier passes, proof checked exactly")

    # 6. crude price scenarios in one run
    t = time.perf_counter()
    rng = random.Random(758)
    costs = {n: source.col_cost[j] for j, n in enumerate(source.col_names)
             if n.startswith("BUY_")}
    scen = work / "prices.csv"
    with open(scen, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario"] + [f"cost:{n}" for n in costs])
        for k in range(20):
            factor = rng.uniform(0.9, 1.1)
            writer.writerow([f"p{k}"] + [f"{c * factor:.9g}" for c in costs.values()])
    table = work / "prices_out.csv"
    proc = run([binary, "scenarios", lp, scen, "--out", table])
    if proc.returncode != 0 or not table.exists():
        fail(6, "prices", "a scenario was not verified", proc)
    rows = list(csv.DictReader(open(table, newline="")))
    objectives = [float(r["objective"]) for r in rows if r["objective"] not in ("", "-")]
    step(6, "prices", t, f"{len(rows)} price sets, {sum(r['verified'] == 'yes' for r in rows)} "
         f"verified, objective {min(objectives):,.0f} to {max(objectives):,.0f}")

    # 7. re-plan from yesterday's basis
    t = time.perf_counter()
    sys.path.insert(0, str(ROOT / "bench" / "runners"))
    from replan_warm_start import perturb_mps
    today = work / "refinery_today.mps"
    # A morning whose demands the plant cannot meet is step 4's story, not this one's:
    # the small plant runs near its limits, so a draw that makes today infeasible is skipped
    # (the line says how many were) and the next seed's morning is re-planned.
    for skipped in range(10):
        prices, commits = perturb_mps(lp, today, random.Random(759 + skipped), 0.10)
        probe = run([binary, "solve", today])
        if field(probe.stdout, "status") == "optimal":
            break
    else:
        fail(7, "replan", "ten mornings in a row were infeasible", probe)
    pivots, objectives = {}, {}
    for arm, extra in (("cold", []), ("warm", ["--warm-start", lp_sol])):
        sol, stats = work / f"today_{arm}.sol", work / f"today_{arm}.json"
        proc = run([binary, "solve", today, "--write-sol", sol, "--stats", stats, *extra])
        if field(proc.stdout, "status") != "optimal":
            fail(7, "replan", f"the {arm} re-solve did not reach optimal", proc)
        check = run([PY, "tools/verify_solution.py", today, sol, "--quiet"])
        if check.returncode != 0:
            fail(7, "replan", f"the verifier rejected the {arm} re-solve", check)
        blob = json.loads(stats.read_text())
        pivots[arm], objectives[arm] = blob["effort"]["iterations"], blob["result"]["objective"]
    if abs(objectives["cold"] - objectives["warm"]) > 1e-6 * max(1.0, abs(objectives["cold"])):
        fail(7, "replan", f"cold {objectives['cold']} and warm {objectives['warm']} disagree",
             proc)
    step(7, "replan", t, f"{prices} prices and {commits} demands moved"
         f"{f' ({skipped} infeasible morning(s) skipped)' if skipped else ''}: cold {pivots['cold']} "
         f"pivots, from yesterday's basis {pivots['warm']}; same optimum "
         f"{objectives['warm']:,.2f}, both verified")

    # 8. the crude unit's minimum run rate, a semi-continuous condition (#754)
    t = time.perf_counter()
    minrun = work / "refinery_min_run.mps"
    proc = run([PY, gen, "--size", "small", "--seed", "1", "--milp", "--crude-min-run", "1/6",
                "--out", minrun])
    if proc.returncode != 0:
        fail(8, "units", "the generator failed", proc)
    answers = {}
    for arm, extra in (("native", []), ("binary", ["--option", "sos_reformulate=true"])):
        sol, stats = work / f"min_run_{arm}.sol", work / f"min_run_{arm}.json"
        proc = run([binary, "solve", minrun, "--write-sol", sol, "--stats", stats, *extra])
        if field(proc.stdout, "status") != "optimal":
            fail(8, "units", f"the {arm} solve did not reach optimal", proc)
        check = run([PY, "tools/verify_solution.py", minrun, sol, "--quiet"])
        if check.returncode != 0:
            fail(8, "units", f"the verifier rejected the {arm} plan", check)
        blob = json.loads(stats.read_text())
        answers[arm] = (blob["result"]["objective"], blob["effort"]["nodes"], sol)
    native, reformulated = answers["native"], answers["binary"]
    if abs(native[0] - reformulated[0]) > 1e-6 * max(1.0, abs(native[0])):
        fail(8, "units", f"native {native[0]} and reformulated {reformulated[0]} disagree", proc)
    runs = [line.split() for line in native[2].read_text().splitlines()
            if line.startswith("RUN_")]
    idle = sum(1 for fields in runs if abs(float(fields[1])) <= 1e-9)
    step(8, "units", t, f"each crude runs 0 or at least 1/6 of the crude unit: optimum "
         f"{native[0]:,.2f} against {milp_objective:,.2f} without the minimum, {idle} of "
         f"{len(runs)} runs idle; {native[1]} nodes branching on it natively, the binary "
         f"reformulation agrees; both verified")

    print(f"\nthe whole walk: {time.perf_counter() - total:.1f} s")
    if args.keep:
        shutil.copytree(work, args.keep, dirs_exist_ok=True)
        print(f"files kept in {args.keep}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(ROOT / "bindings" / "python") + os.pathsep + ENV.get("PYTHONPATH", "")

if __name__ == "__main__":
    sys.exit(main())
