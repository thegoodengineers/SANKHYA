#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: agreement of objectives against size-limited commercial solver editions (#533).

WHY THIS EXISTS. The problem statement names commercial solvers as the thing to replace.
Their free size-limited editions can be run over the same MPS files as SANKHYA, on the
instances that fit their limits, and the question asked is AGREEMENT - the same status and
the same objective - not time: on models this small every solver is fast.

HOW THEY ARE RUN. Each edition is the vendor's own package from PyPI (`cplex`, `gurobipy`,
`xpress`), installed into a separate virtualenv whose interpreter is passed as `--python`.
bench/runners/commercial_child.py is started in that interpreter once per solve, calls the
package's documented public API, and prints one JSON line. No commercial code is imported
into this process, linked into SANKHYA or read.

LICENCES DECIDE WHAT MAY BE RUN AND PUBLISHED (docs/PROVENANCE.md, judgement call 16, read
from the LICENSE files inside the wheels on 2026-09-30):
  - cplex 22.2.0.1, IBM ILOG CPLEX Optimization Studio Community Edition: the licence
    information and the International License Agreement for Non-Warranted Programs contain
    no clause on benchmarking or on publishing results. Allowed.
  - gurobipy 13.0.3: the pip licence is an EVALUATION licence whose s2.2 reads "You will not
    publish any benchmark testing results on the Product." Refused unless
    `--gurobi-academic` confirms an academic licence is installed instead.
  - xpress 9.9.1: the community licence's s2.4(vii) forbids disclosing or publishing
    performance benchmark results without Fair Isaac's prior written consent. Refused
    unless `--xpress-consent` confirms that consent is held.

SIZE LIMITS. Each package enforces its own; a model it refuses comes back as status `error`
with the package's message and is named in the CSV, not dropped. The runner skips models
beyond the documented limits up front so a run does not spend its time on refusals:
CPLEX Community 1000 columns and 1000 rows, Gurobi's pip licence 2000 and 2000, Xpress
Community 5000 rows plus columns.

    python bench/runners/commercial_agreement.py --python <venv>/Scripts/python.exe --solver cplex
    python bench/runners/commercial_agreement.py --python ... --solver cplex --suite netlib
    python bench/runners/commercial_agreement.py --python ... --solver cplex afiro scrs8
    python bench/runners/commercial_agreement.py --dry-run
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp as _stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = {
    "netlib": REPO_ROOT / "data" / "netlib",
    "miplib": REPO_ROOT / "data" / "miplib",
    "maros_meszaros": REPO_ROOT / "data" / "maros-meszaros",
}
RESULTS_DIR = REPO_ROOT / "bench" / "results"
CHILD = Path(__file__).resolve().parent / "commercial_child.py"

# name -> (max columns, max rows, max columns + rows); None is no limit of that kind.
LIMITS = {
    "cplex": (1000, 1000, None),
    "gurobi": (2000, 2000, None),
    "xpress": (None, None, 5000),
}

# Relative, on the objective: the same 1e-6 the MIPLIB and Netlib runners use for a match.
AGREEMENT_TOL = 1e-6

FIELDS = [
    "instance", "instance_sha256", "suite", "rows", "columns", "solver", "solver_version",
    "our_objective", "our_status", "our_wall_s", "our_nodes", "our_iterations",
    "ref_objective", "ref_status", "ref_wall_s", "ref_nodes", "ref_iterations",
    "abs_gap", "rel_gap", "verdict", "message", "time_limit", "git_commit", "machine",
    "timestamp_utc",
]

POINT_STATUSES = ("optimal", "feasible")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover(suite: str) -> list[Path]:
    d = DATA_DIRS[suite]
    if not d.exists():
        return []
    # A set: on a case-insensitive filesystem *.mps and *.MPS find the same file.
    return sorted({p for pat in ("*.mps", "*.mps.gz", "*.MPS", "*.qps", "*.QPS")
                   for p in d.glob(pat)})


def fits(limit: tuple, rows: int, cols: int) -> bool:
    max_cols, max_rows, max_total = limit
    return ((max_cols is None or cols <= max_cols) and (max_rows is None or rows <= max_rows)
            and (max_total is None or rows + cols <= max_total))


def run_sankhya(binary: Path, path: Path, limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        command = [str(binary), "solve", str(path), "--time-limit", str(limit),
                   "--stats", str(stats), "--option", "log_to_console=false"]
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "wall": wall, "message": done.stderr.strip()[:200]}
        blob = json.loads(stats.read_text(encoding="utf-8"))
    result, model, effort = blob.get("result", {}), blob.get("model", {}), blob.get("effort", {})
    return {"status": result.get("status", "unknown"),
            "objective": as_number(result.get("objective")),
            "rows": int(model.get("rows") or 0), "columns": int(model.get("columns") or 0),
            "nodes": effort.get("nodes", ""), "iterations": effort.get("iterations", ""),
            "wall": wall}


def run_package(python: str, solver: str, path: Path, limit: float) -> dict:
    started = time.perf_counter()
    try:
        done = subprocess.run([python, str(CHILD), solver, str(path), str(limit)],
                              capture_output=True, text=True, timeout=limit + 120)
    except subprocess.TimeoutExpired:
        return {"status": "hung", "wall": time.perf_counter() - started}
    wall = time.perf_counter() - started
    try:
        out = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        out = {"status": "no_output", "message": done.stderr.strip()[:200]}
    out["wall"] = wall
    return out


def verdict(ours: dict, ref: dict) -> str:
    a, b = ours.get("objective"), ref.get("objective")
    if ours["status"] in POINT_STATUSES and ref["status"] in POINT_STATUSES and \
            a is not None and b is not None:
        gap = abs(a - b) / max(1.0, abs(b))
        return "agree" if gap <= AGREEMENT_TOL else "disagree"
    if ours["status"] == ref["status"]:
        return "status_agree"
    if ref["status"] in ("error", "not_installed", "no_output", "hung"):
        return "reference_failed"
    return "status_disagree"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", help="named instances (default: all)")
    ap.add_argument("--solver", action="append", choices=list(LIMITS), default=None)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter of the virtualenv holding the vendor packages")
    ap.add_argument("--suite", action="append", choices=list(DATA_DIRS), default=None)
    ap.add_argument("--binary", type=Path, default=None, help="sankhya binary")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gurobi-academic", action="store_true",
                    help="an academic Gurobi licence is installed (the pip evaluation "
                         "licence forbids publishing benchmark results)")
    ap.add_argument("--xpress-consent", action="store_true",
                    help="Fair Isaac's written consent to publish results is held")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    solvers = args.solver or ["cplex"]
    if "gurobi" in solvers and not args.gurobi_academic:
        print("refusing gurobi: the pip licence is an evaluation licence that forbids "
              "publishing benchmark results (docs/PROVENANCE.md, judgement call 16)",
              file=sys.stderr)
        solvers.remove("gurobi")
    if "xpress" in solvers and not args.xpress_consent:
        print("refusing xpress: the community licence forbids publishing benchmark results "
              "without Fair Isaac's written consent (docs/PROVENANCE.md, judgement call 16)",
              file=sys.stderr)
        solvers.remove("xpress")
    if not solvers:
        return 1

    suites = args.suite or list(DATA_DIRS)
    paths = [(s, p) for s in suites for p in discover(s)]
    if args.instances:
        wanted = set(args.instances)
        paths = [(s, p) for s, p in paths if p.name.split(".")[0] in wanted]
    if not paths:
        print("no instances found; fetch the data sets first", file=sys.stderr)
        return 1

    from miplib import find_binary  # the same lookup every runner uses
    binary = find_binary(args.binary)
    if binary is None:
        print("no solver binary; build first", file=sys.stderr)
        return 1
    commit = _stamp.stamp(binary)
    machine = f"{platform.system()}-{platform.machine()}"
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    rows = []
    for suite, path in paths:
        name = path.name.split(".")[0]
        # SANKHYA first: its stats give the size the limits are checked against, as read.
        ours = None
        for solver in solvers:
            if args.dry_run:
                print(f"{solver:<7} {suite:<15} {name}")
                continue
            if ours is None:
                ours = run_sankhya(binary, path, args.time_limit)
            if not fits(LIMITS[solver], ours.get("rows", 0), ours.get("columns", 0)):
                continue
            ref = run_package(args.python, solver, path, args.time_limit)
            a, b = ours.get("objective"), ref.get("objective")
            gap = abs(a - b) if a is not None and b is not None else None
            row = {
                "instance": name, "instance_sha256": sha256_file(path), "suite": suite,
                "rows": ours.get("rows", ""), "columns": ours.get("columns", ""),
                "solver": solver, "solver_version": ref.get("version", ""),
                "our_objective": "" if a is None else repr(a), "our_status": ours["status"],
                "our_wall_s": f"{ours['wall']:.3f}", "our_nodes": ours.get("nodes", ""),
                "our_iterations": ours.get("iterations", ""),
                "ref_objective": "" if b is None else repr(b), "ref_status": ref["status"],
                "ref_wall_s": f"{ref['wall']:.3f}", "ref_nodes": ref.get("nodes", ""),
                "ref_iterations": ref.get("iterations", ""),
                "abs_gap": "" if gap is None else f"{gap:.3g}",
                "rel_gap": "" if gap is None else f"{gap / max(1.0, abs(b)):.3g}",
                "verdict": verdict(ours, ref), "message": ref.get("message", ""),
                "time_limit": args.time_limit, "git_commit": commit, "machine": machine,
                "timestamp_utc": ts,
            }
            rows.append(row)
            print(f"{solver:<7} {name:<20} ours {ours['status']:<11} {row['our_objective']:<24}"
                  f" ref {ref['status']:<11} {row['ref_objective']:<24} {row['verdict']}",
                  flush=True)
    if args.dry_run:
        return 0

    out = args.out or RESULTS_DIR / f"commercial-agreement-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"wrote {out} ({len(rows)} rows): "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
