#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""compare.py --suite: every solver over a whole test set, graded the same way (#766).

compare.py's original mode times SANKHYA against HiGHS on the Netlib tier in data/netlib and
stays as it was. This is its second mode: SANKHYA, HiGHS, SCIP, CBC/Clp and GLPK over one of
three suites, each solver a separate process on the same machine with the same time limit
and one thread, and every answer judged by the same three tests:

*   `solved`: the solver's own status is optimal.
*   `matches_reference`: solved, and the objective AT THE SOLVER'S POINT, recomputed from the
    model by the verifier's own reader, is within a relative 1e-6 of the published optimum.
    The point's objective and not the stated one, so a solver's convention for the objective
    constant cannot move the grade (GLPK's differs, rivals.reconcile). Netlib's and
    Kennington's readmes exclude the objective-row constant and are graded as c'x; the
    Maros-Meszaros readme includes it and is graded as c'x + 0.5 x'Qx + c0. The same rule
    is applied to every solver, SANKHYA included, which is why e226 is a match here and an
    OFFSET row in docs/BENCHMARKS.md section 1c.
*   `independently_verified`: tools/verify_solution.py accepts the .sol file - for SANKHYA
    the one it wrote, for a rival its output converted by rivals.py. `verification` says
    which conditions were checked: `primal+dual` when the solver gave usable duals,
    `primal-only` when it did not (SCIP).

A run counts toward the timing only when all three hold; any other outcome is charged the
full time limit, Mittelmann's rule. Times are each solver's own clock (`solver_seconds`);
`wall_seconds` is the child process around it, which for HiGHS and SCIP includes starting a
Python interpreter.

The CSV is written after every instance, so an interrupted run loses at most one instance,
and --resume picks it up again from the file.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import maros_meszaros  # noqa: E402
import netlib  # noqa: E402
import rivals  # noqa: E402
import stamp  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402
from verify_solution_sol import parse_sol  # noqa: E402

RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
MATCH_TOLERANCE = 1e-6  # netlib.PASS_RELATIVE_TOLERANCE and maros_meszaros.REFERENCE_TOLERANCE
ALL_SOLVERS = ("sankhya",) + rivals.SOLVERS

SUITES = {
    "netlib": {"dir": "netlib", "suffix": ".mps", "kind": "lp", "reference": "published_optimal",
               "with_constant": False, "time_limit": 120.0},
    "kennington": {"dir": "kennington", "suffix": ".mps", "kind": "lp",
                   "reference": "published_optimal", "with_constant": False,
                   "time_limit": 120.0},
    "maros-meszaros": {"dir": "maros-meszaros", "suffix": ".qps", "kind": "qp",
                       "reference": "reference_objective", "with_constant": True,
                       "time_limit": 60.0},
}

CSV_COLUMNS = [
    "suite", "instance", "instance_sha256", "rows", "columns", "solver", "solver_version",
    "status", "message", "objective", "point_objective", "reference_objective",
    "absolute_gap", "relative_gap", "matches_reference", "exact_objective", "matches_exact",
    "verification", "independently_verified", "verifier_message", "counted_for_time",
    "solver_seconds", "wall_seconds", "iterations", "time_limit", "threads",
    "git_commit", "machine", "timestamp_utc",
]


def _virtualisation() -> str:
    """`cloud container (docker)`, `virtual machine (kvm)` or `bare metal`, from
    systemd-detect-virt; `unknown` when the machine cannot say."""
    import subprocess  # noqa: PLC0415
    for flag, label in (("--container", "container"), ("--vm", "virtual machine")):
        try:
            found = subprocess.run(["systemd-detect-virt", flag], capture_output=True,
                                   text=True, check=False).stdout.strip()
        except OSError:
            return "unknown"
        if found and found != "none":
            return f"{label} ({found})"
    return "bare metal"


def machine_tag(kind: str | None = None) -> str:
    """Where the numbers came from, stated plainly: what kind of machine (`kind` when the
    caller knows better, e.g. `cloud container`), CPU model, cores visible to this process,
    RAM, OS and architecture."""
    cpu, ram = platform.processor() or "unknown CPU", "unknown RAM"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = " ".join(line.split(":", 1)[1].split())
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram = f"{int(line.split()[1]) / 1048576:.0f} GiB RAM"
                break
    except OSError:
        pass
    where = kind or _virtualisation()
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return (f"{where}; {cpu}; {cores} cores; {ram}; "
            f"{platform.system()}-{platform.machine()}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_objective(model, sol_path: Path, with_constant: bool) -> float | None:
    """c'x + 0.5 x'Qx (+ c0) at the point in the .sol file, from the verifier's reader."""
    solution = parse_sol(sol_path)
    if any(name not in solution.col_value for name in model.col_names):
        return None
    x = [solution.col_value[name] for name in model.col_names]
    value = sum(c * v for c, v in zip(model.col_cost, x)) + model.quadratic_objective(x)
    return value + (model.objective_offset if with_constant else 0.0)


def verify(model_path: Path, sol_path: Path) -> tuple[bool, str]:
    import subprocess  # noqa: PLC0415
    check = subprocess.run([sys.executable, str(VERIFIER), str(model_path), str(sol_path)],
                           capture_output=True, text=True)
    if check.returncode == 0:
        return True, ""
    failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
    return False, ("; ".join(failing) or
                                   (check.stdout + check.stderr).strip()[-300:])[:300]


def solve(solver: str, suite: dict, model_path: Path, time_limit: float, sol: Path,
          binary: Path) -> dict:
    """One solve, normalised: status, objective, seconds, wall, iterations, version,
    message, and whether the .sol file claims duals."""
    if solver != "sankhya":
        out = rivals.run(solver, model_path, suite["kind"], time_limit, sol)
        out.setdefault("objective", None)
        return out
    options = ["threads=1"]
    if suite["kind"] == "qp":
        blob = maros_meszaros.run_one(binary, model_path, time_limit, False, options,
                                      keep_sol=sol)
    else:
        blob = netlib.run_one(binary, model_path, time_limit, False, options, keep_sol=sol,
                              hang_margin=rivals.HANG_MARGIN_SECONDS)
    seconds = netlib.as_number(blob.get("solver_seconds"))
    return {"status": blob.get("status", "unknown"), "objective": blob.get("objective"),
            "message": str(blob.get("message") or blob.get("stderr") or "")[:200],
            "seconds": seconds if seconds is not None else blob.get("wall_seconds"),
            "wall_seconds": blob.get("wall_seconds"), "iterations": blob.get("iterations", ""),
            "version": stamp.stamp(binary), "duals": True}


def grade(solver: str, suite: dict, model, model_path: Path, sol: Path, out: dict,
          reference: float, exact: float | None, time_limit: float) -> dict:
    """The row's verdicts from one solve; see the module docstring for each rule."""
    status = out["status"]
    row = {"status": status, "objective": out.get("objective"), "point_objective": None,
           "verification": "not-run" if status != "unsupported" else "unsupported",
           "independently_verified": None, "verifier_message": ""}
    notes = []
    if status == "optimal" and sol.exists():
        if solver != "sankhya":
            notes = rivals.reconcile(sol, model, solver)
        row["point_objective"] = point_objective(model, sol, suite["with_constant"])
        row["verification"] = ("primal+dual" if parse_sol(sol).status == "optimal"
                               else "primal-only")
        row["independently_verified"], row["verifier_message"] = verify(model_path, sol)
    elif status == "optimal":
        row["verifier_message"] = "optimal, but no solution file was produced"
        row["independently_verified"] = False
    ours = row["point_objective"]
    gap = None if ours is None else abs(ours - reference) / max(1.0, abs(reference))
    row["absolute_gap"] = None if ours is None else abs(ours - reference)
    row["relative_gap"] = gap
    row["matches_reference"] = bool(status == "optimal" and gap is not None
                                    and gap <= MATCH_TOLERANCE)
    row["exact_objective"] = exact
    row["matches_exact"] = (None if exact is None else bool(
        status == "optimal" and ours is not None
        and abs(ours - exact) / max(1.0, abs(exact)) <= MATCH_TOLERANCE))
    row["counted_for_time"] = bool(row["matches_reference"]
                                   and row["independently_verified"] is True)
    message = "; ".join(filter(None, [str(out.get("message", "")), *notes]))
    row["message"] = message[:300]
    return row


def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def summarise(rows: list[dict], solvers: list[str], time_limit: float, sgm) -> None:
    print(f"\n{'solver':<10}{'runs':>6}{'solved':>8}{'matched':>9}{'verified':>10}"
          f"{'SGM s (shift 10)':>18}")
    for solver in solvers:
        mine = [r for r in rows if r["solver"] == solver]
        if not mine:
            continue
        count = lambda key: sum(r[key] == "1" for r in mine)  # noqa: E731
        solved = sum(r["status"] == "optimal" for r in mine)
        charged = [float(r["solver_seconds"]) if r["counted_for_time"] == "1" else time_limit
                   for r in mine]
        print(f"{rivals.LABELS[solver]:<10}{len(mine):>6}{solved:>8}"
              f"{count('matches_reference'):>9}{count('independently_verified'):>10}"
              f"{sgm(charged, 10.0):>18.3f}")


def run_suite(args) -> int:
    suite = SUITES[args.suite]
    data_dir = REPO_ROOT / "data" / suite["dir"]
    manifest_path = data_dir / "reference.json"
    if not manifest_path.exists():
        raise SystemExit(f"no {manifest_path.relative_to(REPO_ROOT)}; fetch the suite first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["instances"]
    exact_path = data_dir / "koch_exact.json"
    exact = (json.loads(exact_path.read_text())["instances"] if exact_path.exists() else {})
    names = sorted(args.instances or manifest)
    solvers = args.solvers.split(",") if args.solvers else list(ALL_SOLVERS)
    unknown = [s for s in solvers if s not in ALL_SOLVERS]
    if unknown:
        raise SystemExit(f"unknown solver(s): {', '.join(unknown)}")
    time_limit = args.time_limit if args.time_limit_given else suite["time_limit"]
    binary = args.sankhya_binary or netlib.default_binary()
    commit, machine = stamp.stamp(binary), machine_tag(args.machine_kind)
    full = not args.instances and set(solvers) == set(ALL_SOLVERS)
    default = (f"head-to-head-{args.suite}-{commit}.csv" if full
               else f"head-to-head-{args.suite}-partial-{commit}.csv")
    out_path = (REPO_ROOT / args.out).resolve() if args.out else RESULTS_DIR / default

    rows: list[dict] = []
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["git_commit"] == commit]
    done = {(r["instance"], r["solver"]) for r in rows}
    print(f"suite {args.suite}: {len(names)} instances, solvers {', '.join(solvers)}, "
          f"time limit {time_limit:g} s, 1 thread\ncommit {commit}\nmachine {machine}\n"
          f"writing {out_path}" + (f" (resuming, {len(done)} runs kept)" if done else ""))

    for name in names:
        path = data_dir / f"{name}{suite['suffix']}"
        if not path.exists():
            print(f"{name}: MISSING")
            continue
        todo = [s for s in solvers if (name, s) not in done]
        if not todo:
            continue
        model = parse_mps(path)
        digest = sha256_file(path)
        reference = float(manifest[name][suite["reference"]])
        exact_text = exact.get(name, {}).get("exact_objective")
        exact_value = None if exact_text is None else float(exact_text)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        line = [f"{name:<10}"]
        for solver in todo:
            with tempfile.TemporaryDirectory() as tmp:
                sol = Path(tmp) / "solution.sol"
                out = solve(solver, suite, path, time_limit, sol, binary)
                verdict = grade(solver, suite, model, path, sol, out, reference, exact_value,
                                time_limit)
            row = {"suite": args.suite, "instance": name, "instance_sha256": digest,
                   "rows": model.num_rows, "columns": model.num_cols, "solver": solver,
                   "solver_version": out.get("version", ""), **verdict,
                   "reference_objective": reference,
                   "solver_seconds": out.get("seconds"), "wall_seconds": out.get("wall_seconds"),
                   "iterations": out.get("iterations", ""), "time_limit": time_limit,
                   "threads": 1, "git_commit": commit, "machine": machine,
                   "timestamp_utc": timestamp}
            rows.append({key: text(row.get(key)) for key in CSV_COLUMNS})
            mark = ("ok" if verdict["counted_for_time"] else
                    "WRONG" if verdict["status"] == "optimal" and not verdict["matches_reference"]
                    else "REJ" if verdict["independently_verified"] is False
                    else verdict["status"])
            seconds = out.get("seconds")
            line.append(f"{solver}={mark}" + (f"/{seconds:.2f}s" if seconds is not None
                                              else ""))
        print(" ".join(line), flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    import make_benchmarks_doc  # noqa: PLC0415 - the one shifted geometric mean
    summarise(rows, solvers, time_limit, make_benchmarks_doc.shifted_geometric_mean)
    wrong = sorted({r["instance"] for r in rows if r["solver"] == "sankhya"
                    and r["status"] == "optimal" and r["matches_reference"] != "1"})
    if wrong:
        print(f"SANKHYA claimed optimal away from the reference on: {', '.join(wrong)}")
    print(f"wrote {out_path}")
    return 0
