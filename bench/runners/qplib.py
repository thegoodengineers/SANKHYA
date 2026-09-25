#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over QPLIB's convex continuous QPs and emit the evidence CSV (#492).

The instances, their sha256s and their reference objectives come from data/qplib/reference.json,
written by fetch_qplib.py from QPLIB's own pages; the solver reads the QPS file fetch_qplib.py
converted each .qplib file to (qplib_format.py), whose conversion was checked there at
QPLIB's published point. An instance whose check failed is not solved: its row says so.

Each instance is solved by each engine asked for, `auto` (the default QP engine) and `ipm`
(`qp_algorithm=ipm`, the proximal interior point of #490) unless --engines says otherwise,
one row per (instance, engine). A row reports, as maros_meszaros.py does:

*   `matches_reference`: status optimal and the objective within 1e-6 relative
    (|ours - ref| / max(1, |ref|)) of QPLIB's value. That value is a best known point, not a
    proven optimum, printed to about ten significant digits in qplib.solu; an objective BELOW
    it by more than the tolerance is a disagreement too, and is named, not credited.
*   the three QP optimality measures of qp_residuals.py, recomputed from the QPS file and the
    .sol file, relative, and `success_rel_1e-6` when all three are within 1e-6;
*   `verified`: tools/verify_solution.py's verdict on the QPS file and the .sol file.

`passed` = optimal, verified, matching the reference and successful at 1e-6 relative. An
instance with no published point (QPLIB_9002 at the time of writing) cannot pass; its row
leaves `passed` empty and the doc names it on its own line.

Output: bench/results/qplib-<commit>.csv for the full selection in the manifest (every
instance not skipped for size), qplib-small-<commit>.csv for `--tier small`, and
qplib-partial-<commit>.csv for any other selection, which docs/BENCHMARKS.md never reads.

    python bench/runners/fetch_qplib.py
    python bench/runners/qplib.py                      # the full selection, both engines
    python bench/runners/qplib.py --tier small --time-limit 60
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maros_meszaros as mm  # noqa: E402  (run_one, the solve-and-verify step, shared)
import stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "qplib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
REFERENCE_TOLERANCE = 1e-6
DEFAULT_TIME_LIMIT = 1000.0  # the Maros-Meszaros runner's, so the two QP sections compare
ENGINES = {"auto": [], "ipm": ["qp_algorithm=ipm"]}

CSV_COLUMNS = [
    "instance", "engine", "instance_sha256", "qps_sha256", "rows", "cols", "nnz", "q_nnz",
    "status", "message", "our_objective", "reference_objective", "reference_source",
    "abs_gap", "rel_gap", "matches_reference",
    "primal_residual_rel", "dual_residual_rel", "duality_gap_rel", "success_rel_1e-6",
    "verified", "verifier_message", "passed",
    "wall_seconds", "solver_seconds", "iterations", "algorithm", "time_limit",
    "git_commit", "machine", "timestamp_utc", "solver_options", "engine_options",
]


def select(instances: dict, tier: str | None, names: list[str] | None) -> list[str]:
    runnable = sorted(name for name, entry in instances.items() if "skipped" not in entry)
    if names:
        unknown = [name for name in names if name not in runnable]
        if unknown:
            raise SystemExit(f"not fetched (or skipped for size): {', '.join(unknown)}")
        return list(names)
    if tier == "small":
        return [name for name in runnable if instances[name]["tier"] == "small"]
    return runnable


def judge(status: str, ours: float | None, reference: float | None, residuals, verified,
          checked: bool) -> dict:
    optimal = status == "optimal"
    gap = None if ours is None or reference is None else abs(ours - reference)
    rel = None if gap is None else gap / max(1.0, abs(reference))
    matches = None if reference is None else bool(optimal and rel is not None
                                                  and rel <= REFERENCE_TOLERANCE)
    success = bool(optimal and residuals is not None
                   and residuals.meets(REFERENCE_TOLERANCE, relative=True))
    passed = None if reference is None else bool(
        checked and matches and success and verified is True)
    return {"abs_gap": gap, "rel_gap": rel, "matches_reference": matches,
            "success_rel_1e-6": success, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--tier", choices=("small", "full"), default="full")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--engines", nargs="+", choices=sorted(ENGINES), default=["auto", "ipm"])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        raise SystemExit("no data/qplib/reference.json; run bench/runners/fetch_qplib.py")
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))["instances"]
    names = select(instances, args.tier, args.instances)
    binary = args.binary or mm.default_binary()
    commit, machine = stamp.stamp(args.binary), f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)
    print(f"solver   {binary}\ncommit   {commit}   machine {machine}   time limit "
          f"{args.time_limit:g}s   {len(names)} instances x {len(args.engines)} engines"
          + (f"   options {solver_options}" if solver_options else ""))
    print(f"\n{'instance':<13}{'engine':<7}{'status':<16}{'our objective':>22}{'reference':>20}"
          f"{'rel gap':>9}{'resid':>9}{'time':>8}  ver  pass")
    print("-" * 111)

    rows = []
    for name in names:
        entry = instances[name]
        qps = DATA_DIR / f"{name}.qps"
        checked = entry.get("converter_check", {}).get("passed") is not False
        for engine in args.engines:
            engine_options = ENGINES[engine]
            if not checked:
                flat = {"status": "conversion_unchecked", "wall_seconds": 0.0,
                        "message": "fetch_qplib.py's check at QPLIB's point failed"}
            elif not qps.exists():
                flat = {"status": "missing", "wall_seconds": 0.0,
                        "message": "run bench/runners/fetch_qplib.py"}
            else:
                flat = mm.run_one(binary, qps, args.time_limit, not args.no_verify,
                                  engine_options + args.solver_option)
            residuals, ours = flat.get("residuals"), flat.get("objective")
            verdict = judge(flat["status"], ours, entry.get("reference_objective"), residuals,
                            flat.get("verified"), checked)
            row = {
                "instance": name, "engine": engine,
                "instance_sha256": entry.get("qplib", {}).get("sha256", ""),
                "qps_sha256": entry.get("qps", {}).get("sha256", ""),
                "rows": flat.get("rows", ""), "cols": flat.get("columns", ""),
                "nnz": flat.get("nonzeros", ""), "q_nnz": entry.get("q0_entries", ""),
                "status": flat["status"], "message": str(flat.get("message", ""))[:200],
                "our_objective": ours, "reference_objective": entry.get("reference_objective"),
                "reference_source": entry.get("reference_source", ""), **verdict,
                "primal_residual_rel": residuals.primal_rel if residuals else None,
                "dual_residual_rel": residuals.dual_rel if residuals else None,
                "duality_gap_rel": residuals.gap_rel if residuals else None,
                "verified": flat.get("verified"),
                "verifier_message": flat.get("verifier_message", ""),
                "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
                "solver_seconds": flat.get("solver_seconds", ""),
                "iterations": flat.get("iterations", ""), "algorithm": flat.get("algorithm", ""),
                "time_limit": args.time_limit, "git_commit": commit, "machine": machine,
                "timestamp_utc": timestamp, "solver_options": solver_options,
                "engine_options": " ".join(engine_options),
            }
            rows.append({key: mm.text(row[key]) for key in CSV_COLUMNS})
            worst = None if residuals is None else max(
                residuals.primal_rel, residuals.dual_rel, residuals.gap_rel)
            reference = entry.get("reference_objective")
            print(f"{name:<13}{engine:<7}{flat['status']:<16}"
                  f"{'-' if ours is None else format(ours, '.12e'):>22}"
                  f"{'-' if reference is None else format(reference, '.10e'):>20}"
                  f"{'-' if verdict['rel_gap'] is None else format(verdict['rel_gap'], '.1e'):>9}"
                  f"{'-' if worst is None else format(worst, '.1e'):>9}"
                  f"{flat.get('wall_seconds', 0.0):>7.1f}s  "
                  f"{ {True: 'yes', False: 'NO ', None: ' - '}[flat.get('verified')]}  "
                  f"{ {True: 'yes', False: 'no', None: '-'}[verdict['passed']]}")
            if flat.get("verifier_message"):
                print(f"{'':19}verifier: {flat['verifier_message'][:90]}")

    print("-" * 111)
    for engine in args.engines:
        mine = [r for r in rows if r["engine"] == engine]
        judged = [r for r in mine if r["passed"] != ""]
        failed = [r["instance"] for r in judged if r["passed"] != "1"]
        unreferenced = [r["instance"] for r in mine if r["passed"] == ""]
        print(f"{engine}: passed {len(judged) - len(failed)}/{len(judged)} with a published "
              f"reference; optimal {sum(r['status'] == 'optimal' for r in mine)}/{len(mine)}; "
              f"verified {sum(r['verified'] == '1' for r in mine)}/{len(mine)}"
              + (f"; not passed: {', '.join(failed)}" if failed else "")
              + (f"; no reference: {', '.join(unreferenced)}" if unreferenced else ""))

    if args.instances:
        default_name = f"qplib-partial-{commit}.csv"
    elif args.tier == "small":
        default_name = f"qplib-small-{commit}.csv"
    else:
        default_name = f"qplib-{commit}.csv"
    if sorted(args.engines) != sorted(ENGINES) and not args.instances:
        default_name = f"qplib-partial-{commit}.csv"
    out_path = (REPO_ROOT / args.out).resolve() if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0 if rows and all(r["passed"] != "0" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
