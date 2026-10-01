#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the numerical stress set (#762) through SANKHYA and HiGHS, and grade every answer.

    python bench/runners/stress_instances.py                     # generate data/stress/
    python bench/runners/stress.py --machine <honest tag>        # solve, verify, grade

Each instance in data/stress/reference.json carries its sha256 and its known answer (see
stress_instances.py for how each answer is known). Every instance is solved by

  sankhya  the CLI, exactly as a judge would run it, writing its own .sol file;
  highs    highspy in a SEPARATE Python process (this script re-invoked with
           --highs-worker), its primal point, duals, basis and any Farkas or primal ray
           written into the same .sol layout. HiGHS is a comparator run as a released
           binary; no HiGHS source is read (docs/PROVENANCE.md, Judgement calls).

and both .sol files are checked by tools/verify_solution.py with the same tolerances.
The grade:

  correct  the known verdict, and the verifier accepts the file; for an optimal instance
           also the objective within 1e-6 relative of the known optimum (netlib.py's bar).
           On a thin-infeasible instance a point the verifier accepts - every row within
           its 1e-7 - is also correct, since at the stated tolerance it IS feasible; the
           reason column says so.
  wrong    a verdict the instance contradicts (optimal on an unbounded LP, infeasible on a
           feasible one), an objective off the known optimum, or a file the verifier rejects.
  failed   no verdict: a limit, a numerical error, a crash, or infeasible_or_unbounded.

HiGHS reports a Farkas vector or primal ray in its own sign convention. When the verifier
rejects one as written, it is re-checked negated, and the reason says `sign flipped`: the
certificate is judged, not the convention.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "stress"
RESULTS = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
PASS_RELATIVE_TOLERANCE = 1e-6

FIELDS = ["instance", "family", "parameter", "sha256", "rows", "cols", "solver",
          "solver_version", "expected_status", "status", "objective", "reference_objective",
          "absolute_gap", "relative_gap", "verified", "verdict", "reason", "wall_seconds",
          "iterations", "git_commit", "machine", "timestamp_utc"]

HIGHS_STATUS = {"kOptimal": "optimal", "kInfeasible": "infeasible", "kUnbounded": "unbounded",
                "kUnboundedOrInfeasible": "infeasible_or_unbounded",
                "kTimeLimit": "time_limit", "kIterationLimit": "iteration_limit"}
HIGHS_BASIS = {"kLower": "at_lower", "kUpper": "at_upper", "kBasic": "basic",
               "kZero": "free", "kNonbasic": "at_lower"}


# =========================================================================================
# HiGHS, in its own process
# =========================================================================================

def highs_worker(mps: str, sol: str, time_limit: float) -> int:
    import importlib.metadata
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.readModel(mps)
    h.run()
    status = HIGHS_STATUS.get(h.getModelStatus().name, h.getModelStatus().name)
    lp, solution, basis, info = h.getLp(), h.getSolution(), h.getBasis(), h.getInfo()
    cols = [lp.col_names_[j] if lp.col_names_ else f"C{j}" for j in range(lp.num_col_)]
    rows = [lp.row_names_[i] if lp.row_names_ else f"R{i}" for i in range(lp.num_row_)]
    out = ["# HiGHS solution in the SANKHYA .sol layout (bench/runners/stress.py)",
           f"status {status}"]
    certificate, extra = "none", []
    if status == "infeasible":
        _, has, ray = h.getDualRay()
        if has:
            certificate = "farkas"
            extra = [f"begin farkas {len(rows)}", *(f"{rows[i]} {float(v)!r}"
                                                   for i, v in enumerate(ray)), "end farkas"]
    elif status == "unbounded":
        _, has, ray = h.getPrimalRay()
        if has:
            certificate = "ray"
            extra = [f"begin ray {len(cols)}", *(f"{cols[j]} {float(v)!r}"
                                                for j, v in enumerate(ray)), "end ray"]
    point = status in ("optimal", "unbounded") and solution.value_valid
    if point:
        out.append(f"objective {h.getObjectiveValue()!r}")
    out.append(f"certificate {certificate}")
    if point:
        duals = solution.dual_valid
        def tag(s):  # noqa: E306
            return HIGHS_BASIS.get(s.name, "unknown") if basis.valid else "unknown"
        out.append(f"begin columns {len(cols)}")
        out += [f"{cols[j]} {solution.col_value[j]!r} "
                f"{solution.col_dual[j] if duals else 0.0!r} {tag(basis.col_status[j])}"
                for j in range(len(cols))]
        out += ["end columns", f"begin rows {len(rows)}"]
        out += [f"{rows[i]} {solution.row_value[i]!r} "
                f"{solution.row_dual[i] if duals else 0.0!r} {tag(basis.row_status[i])}"
                for i in range(len(rows))]
        out.append("end rows")
    Path(sol).write_text("\n".join(out + extra) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "objective": h.getObjectiveValue() if point else None,
                      "iterations": info.simplex_iteration_count + info.ipm_iteration_count,
                      "version": importlib.metadata.version("highspy")}))
    return 0


def negate_certificate(sol: Path) -> Path:
    out, block = [], ""
    for line in sol.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["begin"]:
            block = fields[1]
        elif fields[:1] == ["end"]:
            block = ""
        elif block in ("farkas", "ray") and len(fields) == 2:
            line = f"{fields[0]} {-float(fields[1])!r}"
        out.append(line)
    flipped = sol.with_suffix(".flipped.sol")
    flipped.write_text("\n".join(out) + "\n", encoding="utf-8")
    return flipped


# =========================================================================================
# Solving, verifying, grading
# =========================================================================================

def run_sankhya(binary: Path, mps: Path, sol: Path, time_limit: float) -> dict:
    stats = sol.with_suffix(".json")
    completed = subprocess.run([str(binary), "solve", str(mps), "--stats", str(stats),
                                "--write-sol", str(sol), "--time-limit", str(time_limit),
                                "--option", "log_to_console=false"],
                               capture_output=True, text=True, timeout=time_limit * 3 + 60)
    if not stats.exists():
        return {"status": f"crashed (exit {completed.returncode})",
                "reason": completed.stderr.strip()[:160]}
    blob = json.loads(stats.read_text())
    result = blob.get("result", {})
    # Koch's optima exclude the objective-row constant (e226: 7.113), so the objective is
    # graded without it, as netlib.py grades matches_exact (#783: scaled_e226 was never
    # optimal before, so the units never met).
    objective = result.get("objective")
    offset = blob.get("model", {}).get("objective_offset") or 0.0
    if isinstance(objective, (int, float)) and offset:
        objective -= offset
    return {"status": result.get("status", "unknown"), "objective": objective,
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "reason": result.get("message", "")}


def run_highs(mps: Path, sol: Path, time_limit: float) -> dict:
    completed = subprocess.run([sys.executable, __file__, "--highs-worker", str(mps), str(sol),
                                str(time_limit)], capture_output=True, text=True,
                               timeout=time_limit * 3 + 60)
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": f"crashed (exit {completed.returncode})",
                "reason": completed.stderr.strip()[-160:]}


def verify(mps: Path, sol: Path) -> tuple[int, str]:
    """(1 accepted, 0 rejected, '' not checkable) and the first failing check."""
    if not sol.exists():
        return "", "no .sol file"
    check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                           capture_output=True, text=True)
    if check.returncode == 2:
        return "", "verifier could not read the files: " + check.stderr.strip()[:120]
    failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
    return (1 if check.returncode == 0 else 0), "; ".join(failing)[:200]


def grade(expected: str, reference: float | None, status: str, objective,
          verified, failing: str) -> tuple[str, str]:
    if status not in ("optimal", "infeasible", "unbounded"):
        return "failed", f"no verdict: {status}"
    if verified == 0:
        return "wrong", f"{status}, verifier rejects: {failing}"
    if status == "optimal" and expected == "infeasible":
        return "correct", "tolerance-feasible point accepted by the verifier"
    if status != expected:
        return "wrong", f"{status} on an instance that is {expected}"
    if status == "optimal":
        rel = abs(objective - reference) / max(1.0, abs(reference))
        if rel > PASS_RELATIVE_TOLERANCE:
            return "wrong", f"objective off the known optimum by {rel:.2e} relative"
    return "correct", ""


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--highs-worker":
        return highs_worker(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--solvers", nargs="+", default=["sankhya", "highs"],
                        choices=["sankhya", "highs"])
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--machine", default=f"{platform.system()}-{platform.machine()}")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.binary is None:
        sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
        import sankhya
        args.binary = sankhya.locate_executable()
    commit = stamp.stamp(args.binary)
    manifest = json.loads((args.data / "reference.json").read_text())["instances"]
    names = [n for n in manifest if (args.instances is None or n in args.instances)
             and (args.families is None or manifest[n]["family"] in args.families)]
    rows = []
    print(f"sankhya {args.binary} at {commit}, machine {args.machine}, {len(names)} instances")
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            entry, mps = manifest[name], args.data / f"{name}.mps"
            digest = hashlib.sha256(mps.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise SystemExit(f"{mps}: sha256 {digest} is not the manifest's "
                                 f"{entry['sha256']}; regenerate with stress_instances.py")
            reference = (float(entry["expected_objective"]) if entry["expected_objective"]
                         else None)
            for solver in args.solvers:
                sol = Path(tmp) / f"{name}.{solver}.sol"
                start = time.perf_counter()
                try:
                    got = (run_sankhya(args.binary, mps, sol, args.time_limit)
                           if solver == "sankhya" else run_highs(mps, sol, args.time_limit))
                except subprocess.TimeoutExpired:
                    got = {"status": "killed", "reason": "no exit within 3x the time limit"}
                wall = time.perf_counter() - start
                verified, failing = verify(mps, sol)
                flipped = False
                if solver == "highs" and verified == 0 and got["status"] in ("infeasible",
                                                                              "unbounded"):
                    retry, retry_failing = verify(mps, negate_certificate(sol))
                    if retry == 1:
                        verified, failing, flipped = retry, retry_failing, True
                objective = got.get("objective")
                verdict, reason = grade(entry["expected_status"], reference, got["status"],
                                        objective, verified, failing)
                if flipped:
                    reason = (reason + "; " if reason else "") + "sign flipped"
                if verdict == "failed" and got.get("reason"):
                    reason += f" ({got['reason'][:120]})"
                gap = (abs(objective - reference) if isinstance(objective, (int, float))
                       and reference is not None else None)
                rows.append({
                    "instance": name, "family": entry["family"],
                    "parameter": entry["parameter"], "sha256": digest, "rows": entry["rows"],
                    "cols": entry["cols"], "solver": solver,
                    "solver_version": commit if solver == "sankhya" else got.get("version", ""),
                    "expected_status": entry["expected_status"], "status": got["status"],
                    "objective": "" if objective is None else repr(objective),
                    "reference_objective": entry["expected_objective"],
                    "absolute_gap": "" if gap is None else repr(gap),
                    "relative_gap": "" if gap is None else repr(gap / max(1.0, abs(reference))),
                    "verified": verified, "verdict": verdict, "reason": reason,
                    "wall_seconds": f"{wall:.3f}", "iterations": got.get("iterations", ""),
                    "git_commit": commit, "machine": args.machine,
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds"),
                })
                print(f"{name:28s} {solver:8s} {got['status']:24s} {verdict:8s} {wall:8.2f}s "
                      f"{reason[:90]}", flush=True)

    print("\nsolver    family            correct  wrong  failed")
    for solver in args.solvers:
        for family in sorted({r["family"] for r in rows}):
            picked = [r["verdict"] for r in rows if r["solver"] == solver and r["family"] == family]
            print(f"{solver:9s} {family:16s} {picked.count('correct'):>8d} "
                  f"{picked.count('wrong'):>6d} {picked.count('failed'):>7d}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS / f"stress-{commit}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
