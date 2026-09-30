#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The finale walk (#758): one refinery MILP, solved, proved, checked, offline.

    demo/finale.sh            (Linux, macOS, Git Bash)      demo\\finale.cmd   (Windows)
    python demo/finale.py [--dry] [--binary PATH] [--keep DIR]

Six steps, each printing one line of result and the seconds it took. Every number printed
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
              "bench/case_studies/refinery/generator.py", "bindings/python/sankhya/__init__.py"]
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
    # What one more unit of each limit is worth, and how far the price holds (the ranging
    # interval of the reported basis; 0 means the vertex is degenerate there).
    worth = []
    for row in binding[:3]:
        up = (row.get("ranging") or {}).get("allow_increase")
        holds = (f"holds for +{up:.4g}" if isinstance(up, (int, float)) and up < 1e29
                 else "holds without limit")
        worth.append(f"{row['name']} {abs(row['shadow_price']):.4g}/unit ({holds})")
    step(3, "plan", t, f"LP optimum {field(proc.stdout, 'objective')}; {len(binding)} binding "
         f"limits; worth relaxing most: " + "; ".join(worth))

    # 4. an impossible demand, proved and repaired
    t = time.perf_counter()
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_solution_mps import parse_mps
    source = parse_mps(lp)
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
    made = run([PY, "tools/bundle.py", milp, "--binary", binary, "--bundle", bundle])
    if made.returncode != 0:
        fail(5, "bundle", "bundling failed", made)
    replay = run([PY, "tools/replay_bundle.py", bundle])
    if replay.returncode != 0:
        fail(5, "bundle", "the replay failed", replay)
    step(5, "bundle", t, f"{bundle.name} ({bundle.stat().st_size // 1024} KiB) replayed: "
         f"manifest intact, verifier passes")

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
