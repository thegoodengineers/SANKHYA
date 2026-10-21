#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over QPLIB's convex continuous QCQPs and emit the evidence CSV (#755).

The instances are the convex continuous ones with convex quadratic CONSTRAINTS (PROBTYPE
?CD), fetched by `fetch_qplib.py --qcqp` into data/qplib/reference_qcqp.json with QPLIB's own
reference objectives; the solver reads the QPS file with its QCMATRIX sections.

Each instance is solved by each engine asked for: `conic` (the default path: the quadratic
rows classified convex at read time and solved by the conic interior point) and `global`
(`nonconvex=global`, the spatial branch and bound of #514), one row per (instance, engine).
A row reports:

*   `matches_reference`: status optimal and the objective within 1e-6 relative of QPLIB's
    value (a best known point, not a proof; a lower objective is a disagreement too);
*   `verified`: tools/verify_solution.py on the QPS and the .sol file - for the conic engine
    the KKT conditions through every quadratic row's gradient, for the global method its bound;
*   `passed` = optimal, verified and matching the reference.

Output: bench/results/qplib-qcqp-<commit>.csv (qplib-qcqp-partial-<commit>.csv for a
selection), read by qplib_qcqp_doc.py for docs/BENCHMARKS.md.

    python bench/runners/fetch_qplib.py --qcqp
    python bench/runners/qplib_qcqp.py --time-limit 300
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
DEFAULT_TIME_LIMIT = 300.0  # the issue's limit
ENGINES = {"conic": [], "global": ["nonconvex=global"]}

CSV_COLUMNS = [
    "instance", "engine", "instance_sha256", "qps_sha256", "rows", "cols", "nnz",
    "status", "message", "our_objective", "reference_objective", "reference_source",
    "abs_gap", "rel_gap", "matches_reference", "verified", "verifier_message", "passed",
    "wall_seconds", "solver_seconds", "iterations", "algorithm", "time_limit",
    "git_commit", "machine", "timestamp_utc", "solver_options", "engine_options",
]


def judge(status: str, ours: float | None, reference: float | None, verified,
          checked: bool) -> dict:
    optimal = status == "optimal"
    gap = None if ours is None or reference is None else abs(ours - reference)
    rel = None if gap is None else gap / max(1.0, abs(reference))
    matches = None if reference is None else bool(optimal and rel is not None
                                                  and rel <= REFERENCE_TOLERANCE)
    passed = None if reference is None else bool(checked and matches and verified is True)
    return {"abs_gap": gap, "rel_gap": rel, "matches_reference": matches, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--engines", nargs="+", choices=sorted(ENGINES), default=["conic", "global"])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--machine", default=None)
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference_qcqp.json"
    if not manifest_path.exists():
        raise SystemExit("no data/qplib/reference_qcqp.json; run bench/runners/fetch_qplib.py --qcqp")
    instances = json.loads(manifest_path.read_text(encoding="utf-8"))["instances"]
    names = sorted(name for name, entry in instances.items() if "skipped" not in entry)
    if args.instances:
        unknown = [name for name in args.instances if name not in names]
        if unknown:
            raise SystemExit(f"not fetched (or skipped for size): {', '.join(unknown)}")
        names = list(args.instances)
    binary = args.binary or mm.default_binary()
    commit = stamp.stamp(args.binary)
    machine = args.machine or f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)
    print(f"solver   {binary}\ncommit   {commit}   machine {machine}   time limit "
          f"{args.time_limit:g}s   {len(names)} instances x {len(args.engines)} engines")
    print(f"\n{'instance':<13}{'engine':<8}{'status':<16}{'our objective':>22}{'reference':>20}"
          f"{'rel gap':>9}{'time':>8}  ver  pass")
    print("-" * 102)

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
                        "message": "run bench/runners/fetch_qplib.py --qcqp"}
            else:
                flat = mm.run_one(binary, qps, args.time_limit, not args.no_verify,
                                  engine_options + args.solver_option)
            ours = flat.get("objective")
            verdict = judge(flat["status"], ours, entry.get("reference_objective"),
                            flat.get("verified"), checked)
            row = {
                "instance": name, "engine": engine,
                "instance_sha256": entry.get("qplib", {}).get("sha256", ""),
                "qps_sha256": entry.get("qps", {}).get("sha256", ""),
                "rows": flat.get("rows", ""), "cols": flat.get("columns", ""),
                "nnz": flat.get("nonzeros", ""),
                "status": flat["status"], "message": str(flat.get("message", ""))[:200],
                "our_objective": ours, "reference_objective": entry.get("reference_objective"),
                "reference_source": entry.get("reference_source", ""), **verdict,
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
            reference = entry.get("reference_objective")
            print(f"{name:<13}{engine:<8}{flat['status']:<16}"
                  f"{'-' if ours is None else format(ours, '.12e'):>22}"
                  f"{'-' if reference is None else format(reference, '.10e'):>20}"
                  f"{'-' if verdict['rel_gap'] is None else format(verdict['rel_gap'], '.1e'):>9}"
                  f"{flat.get('wall_seconds', 0.0):>7.1f}s  "
                  f"{ {True: 'yes', False: 'NO ', None: ' - '}[flat.get('verified')]}  "
                  f"{ {True: 'yes', False: 'no', None: '-'}[verdict['passed']]}")
            if flat.get("verifier_message"):
                print(f"{'':21}verifier: {flat['verifier_message'][:90]}")
            sys.stdout.flush()

    print("-" * 102)
    for engine in args.engines:
        mine = [r for r in rows if r["engine"] == engine]
        judged = [r for r in mine if r["passed"] != ""]
        failed = [r["instance"] for r in judged if r["passed"] != "1"]
        print(f"{engine}: passed {len(judged) - len(failed)}/{len(judged)}; optimal "
              f"{sum(r['status'] == 'optimal' for r in mine)}/{len(mine)}; verified "
              f"{sum(r['verified'] == '1' for r in mine)}/{len(mine)}"
              + (f"; not passed: {', '.join(failed)}" if failed else ""))

    partial = bool(args.instances) or sorted(args.engines) != sorted(ENGINES)
    default_name = f"qplib-qcqp-{'partial-' if partial else ''}{commit}.csv"
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
