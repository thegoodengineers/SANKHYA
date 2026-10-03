#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The QPLIB convex MIQP subset, MIQP node QPs warm-started against cold (#494, #893).

qplib.py (#492) judges the convex continuous QPs and qplib_all.py (#835) every instance under
the defaults. This one takes the convex MIQPs with linear constraints and solves each twice
with the QP interior point as the node solver (`miqp_node_ipm=true`, #687):

*   `cold`: `miqp_node_ipm_warm_start=false`, every node QP from the engine's cold start;
*   `warm`: `miqp_node_ipm_warm_start=true`, every child's QP from its parent's save point
    (src/qp/qp_ipm_warm.cpp).

Nothing else differs between the two arms. THE SELECTION is read off the manifest
fetch_qplib_all.py writes (data/qplib/all.json), in QPLIB's own classification
(https://qplib.zib.de/doc.html): `Cvx` ticked, objective letter O in {C, D} (convex, or convex
with a diagonal Q), variable letter V not C (binary, mixed or integer columns), constraint
letter C in {N, B, L} (no quadratic constraints). 17 instances when #494 was finished.

A row per instance and arm, with what the issue asks for: `nodes`, `qp_node_iterations` (the
interior point's iterations summed over every node QP, an abandoned warm run included),
`iterations_per_node` (that over `qp_node_solves`), the wall and solver time, the status and
the objective; `verified` is tools/verify_solution.py's verdict on the written .sol file
against the QPS file, for every status that reports a point, and `matches_reference` the
objective against qplib.solu within kMipRelativeGap (1e-4) for an optimal answer, as
qplib_all.py judges it. The warm row also says whether it `agrees_with_cold`: the same
status, and for two optimal answers objectives within kMipRelativeGap of each other (the gap
both searches stop at). A time-limited pair is not judged on its incumbents, which depend on
when the clock stopped; its node and iteration counts are what it is read for.

The two arms of one instance run back to back, never at the same time as anything else the
runner starts (--jobs 1 is the only setting). Wall times mean something only on a machine
running nothing else, which is why a CSV of this runner made on a shared laptop is not
evidence and is not committed.

    python bench/runners/fetch_qplib_all.py
    python bench/runners/qplib_miqp.py --time-limit 300
    python bench/runners/qplib_miqp.py --instances QPLIB_10069 --time-limit 60 --out x.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_suite  # noqa: E402  (machine_tag)
import maros_meszaros as mm  # noqa: E402  (default_binary, as_number, text, VERIFIER)
import qplib_all  # noqa: E402  (judge, POINT_STATUSES)
import stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "qplib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
AGREEMENT_TOLERANCE = 1e-4  # kMipRelativeGap, include/sankhya/tolerances.hpp
ARMS = ("cold", "warm")
VERDICT_STATUSES = ("optimal", "infeasible", "unbounded")

CSV_COLUMNS = [
    "instance", "arm", "problem_type", "qps_sha256", "rows", "cols", "integer_columns",
    "status", "message", "our_objective", "reference_objective", "abs_gap", "rel_gap",
    "matches_reference", "verified", "verifier_message", "agrees_with_cold", "nodes",
    "qp_node_solves", "qp_node_iterations", "iterations_per_node", "qp_node_warm_starts",
    "qp_node_warm_fallbacks", "wall_seconds", "solver_seconds", "time_limit", "options",
    "git_commit", "machine", "timestamp_utc",
]


def is_convex_miqp(entry: dict) -> bool:
    """QPLIB's own letters (doc.html PROBTYPE and CONVEX): a convex objective, integer
    columns, linear constraints at most, and a conversion that exists."""
    kind = entry.get("problem_type", "")
    return (len(kind) == 3 and bool(entry.get("listed_convex")) and kind[0] in "CD"
            and kind[1] != "C" and kind[2] in "NBL" and "conversion_error" not in entry)


def select(instances: dict, names: list[str] | None) -> list[str]:
    """Explicit names (which must be in the manifest) or every convex MIQP, by name."""
    if names:
        unknown = [name for name in names if name not in instances]
        if unknown:
            raise SystemExit(f"not in data/qplib/all.json: {', '.join(unknown)}; fetch them "
                             "with bench/runners/fetch_qplib_all.py --instances ...")
        return list(names)
    return sorted(name for name, entry in instances.items() if is_convex_miqp(entry))


def arm_options(arm: str) -> list[str]:
    """The solver options that make an arm; the node solver is the same in both."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    warm = "true" if arm == "warm" else "false"
    return ["miqp_node_ipm=true", f"miqp_node_ipm_warm_start={warm}", "log_to_console=false"]


def command_for(binary: Path, qps: Path, stats: Path, sol: Path, time_limit: float,
                arm: str) -> list[str]:
    command = [str(binary), "solve", str(qps), "--stats", str(stats), "--write-sol", str(sol),
               "--time-limit", str(time_limit)]
    for option in arm_options(arm):
        command += ["--option", option]
    return command


def effort_of(blob: dict) -> dict:
    """The stats JSON's fields this runner reports, iterations per node derived."""
    result, model = blob.get("result", {}), blob.get("model", {})
    effort = blob.get("effort", {})
    status = result.get("status", "unknown")
    solves = effort.get("qp_node_solves")
    iterations = effort.get("qp_node_iterations")
    per_node = (iterations / solves if isinstance(solves, int) and isinstance(iterations, int)
                and solves > 0 else None)
    return {
        "status": status, "message": result.get("message", ""),
        "objective": mm.as_number(result.get("objective"))
        if status in qplib_all.POINT_STATUSES else None,
        "rows": model.get("rows", ""), "cols": model.get("columns", ""),
        "nodes": effort.get("nodes", ""), "qp_node_solves": solves,
        "qp_node_iterations": iterations, "iterations_per_node": per_node,
        "qp_node_warm_starts": effort.get("qp_node_warm_starts"),
        "qp_node_warm_fallbacks": effort.get("qp_node_warm_fallbacks"),
        "solver_seconds": effort.get("solve_seconds", ""),
    }


def agrees(cold: dict, warm: dict) -> bool | None:
    """Did the warm arm give the cold arm's answer? None when neither verdict is final (a
    limit stopped one of them), since two incumbents cut off by a clock need not agree."""
    if cold["status"] not in VERDICT_STATUSES or warm["status"] not in VERDICT_STATUSES:
        return None
    if cold["status"] != warm["status"]:
        return False
    if cold["status"] != "optimal":
        return True
    a, b = mm.as_number(cold["our_objective"]), mm.as_number(warm["our_objective"])
    if a is None or b is None:
        return False
    return abs(a - b) <= AGREEMENT_TOLERANCE * max(1.0, abs(a))


def verify(qps: Path, sol: Path) -> tuple[bool, str]:
    check = subprocess.run([sys.executable, str(mm.VERIFIER), str(qps), str(sol)],
                           capture_output=True, text=True)
    if check.returncode == 0:
        return True, ""
    failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
    return False, ("; ".join(failing) or (check.stdout + check.stderr).strip())[:300]


def solve(binary: Path, qps: Path, time_limit: float, arm: str, check: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats, sol = Path(tmp) / "stats.json", Path(tmp) / "solution.sol"
        started = time.perf_counter()
        try:
            done = subprocess.run(command_for(binary, qps, stats, sol, time_limit, arm),
                                  capture_output=True, text=True, timeout=time_limit + 120)
        except subprocess.TimeoutExpired:
            return {"status": "hung", "message": f"no exit within {time_limit + 120:g} s",
                    "wall_seconds": time.perf_counter() - started}
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "crashed" if done.returncode not in (0, 1) else "no_output",
                    "message": (done.stderr.strip() or f"exit code {done.returncode}")[:300],
                    "wall_seconds": wall}
        flat = effort_of(json.loads(stats.read_text(encoding="utf-8")))
        flat["wall_seconds"] = wall
        flat["verified"], flat["verifier_message"] = None, ""
        if check and sol.exists() and flat["status"] in qplib_all.POINT_STATUSES:
            flat["verified"], flat["verifier_message"] = verify(qps, sol)
        return flat


def row_for(name: str, entry: dict, arm: str, flat: dict, time_limit: float,
            stamp_fields: dict) -> dict:
    verdict = qplib_all.judge(flat["status"], flat.get("objective"),
                              entry.get("reference_objective"), True, entry.get("maximize"))
    per_node = flat.get("iterations_per_node")
    row = {
        "instance": name, "arm": arm, "problem_type": entry.get("problem_type", ""),
        "qps_sha256": entry.get("qps", {}).get("sha256", ""), "rows": flat.get("rows", ""),
        "cols": flat.get("cols", ""), "integer_columns": entry.get("integer_columns", ""),
        "status": flat["status"], "message": str(flat.get("message", ""))[:200],
        "our_objective": flat.get("objective"),
        "reference_objective": entry.get("reference_objective"),
        "abs_gap": verdict["abs_gap"], "rel_gap": verdict["rel_gap"],
        "matches_reference": verdict["matches_reference"], "verified": flat.get("verified"),
        "verifier_message": flat.get("verifier_message", ""), "agrees_with_cold": None,
        "nodes": flat.get("nodes", ""), "qp_node_solves": flat.get("qp_node_solves"),
        "qp_node_iterations": flat.get("qp_node_iterations"),
        "iterations_per_node": None if per_node is None else round(per_node, 4),
        "qp_node_warm_starts": flat.get("qp_node_warm_starts"),
        "qp_node_warm_fallbacks": flat.get("qp_node_warm_fallbacks"),
        "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
        "solver_seconds": flat.get("solver_seconds", ""), "time_limit": time_limit,
        "options": " ".join(arm_options(arm)), **stamp_fields,
    }
    return row


def shifted_geomean(values: list[float], shift: float = 1.0) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(max(v, 0.0) + shift) for v in values) / len(values)) - shift


def summary(rows: list[dict]) -> dict:
    """Per arm: instances, verified, optimal, total nodes and QP iterations, and the shifted
    geometric mean of iterations per node over the instances where BOTH arms report it; plus
    how many warm answers agree, disagree or cannot be judged."""
    by = {(r["instance"], r["arm"]): r for r in rows}
    instances = sorted({r["instance"] for r in rows})
    both = [n for n in instances
            if all(mm.as_number(by.get((n, a), {}).get("iterations_per_node")) is not None
                   for a in ARMS)]
    out = {"paired_instances": len(both)}
    for arm in ARMS:
        mine = [by[(n, arm)] for n in instances if (n, arm) in by]
        out[arm] = {
            "instances": len(mine),
            "optimal": sum(r["status"] == "optimal" for r in mine),
            "verified": sum(r["verified"] in (True, "1") for r in mine),
            "rejected": sum(r["verified"] in (False, "0") for r in mine),
            "nodes": sum(int(mm.as_number(r["nodes"]) or 0) for r in mine),
            "qp_node_iterations": sum(int(mm.as_number(r["qp_node_iterations"]) or 0)
                                      for r in mine),
            "iterations_per_node_geomean": shifted_geomean(
                [mm.as_number(by[(n, arm)]["iterations_per_node"]) for n in both]),
        }
    warm = [by[(n, "warm")] for n in instances if (n, "warm") in by]
    out["agree"] = sum(r["agrees_with_cold"] is True for r in warm)
    out["disagree"] = sum(r["agrees_with_cold"] is False for r in warm)
    out["not_judged"] = sum(r["agrees_with_cold"] is None for r in warm)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="seconds per solve, the same for both arms")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--machine-kind", default="cloud container",
                        help="the kind of machine the default tag starts with")
    parser.add_argument("--machine", default=None,
                        help="the whole machine tag; by default the kind, CPU model, cores "
                             "and RAM read from this box")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = DATA_DIR / "all.json"
    if not manifest_path.exists():
        raise SystemExit("no data/qplib/all.json; run bench/runners/fetch_qplib_all.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest["instances"]
    names = select(instances, args.instances)
    binary = args.binary or mm.default_binary()
    stamp_fields = {
        "git_commit": stamp.stamp(binary),
        "machine": args.machine or compare_suite.machine_tag(args.machine_kind),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"solver {binary}  commit {stamp_fields['git_commit']}\nmachine {stamp_fields['machine']}"
          f"\n{len(names)} instances, two arms each, at {args.time_limit:g} s")

    rows = []
    for name in names:
        entry = instances[name]
        qps = DATA_DIR / f"{name}.qps"
        pair = {}
        for arm in ARMS:
            if not qps.exists():
                flat = {"status": "missing", "message": "run bench/runners/fetch_qplib_all.py",
                        "wall_seconds": 0.0}
            else:
                flat = solve(binary, qps, args.time_limit, arm, not args.no_verify)
            pair[arm] = row_for(name, entry, arm, flat, args.time_limit, stamp_fields)
        pair["warm"]["agrees_with_cold"] = agrees(pair["cold"], pair["warm"])
        for arm in ARMS:
            r = pair[arm]
            rows.append(r)
            print(f"{name:<13}{arm:<6}{r['status']:<16}{mm.text(r['our_objective']):>24} "
                  f"nodes={r['nodes']} qp_its={r['qp_node_iterations']} "
                  f"its/node={mm.text(r['iterations_per_node'])} ver={mm.text(r['verified'])} "
                  f"agree={mm.text(r['agrees_with_cold'])} {r['wall_seconds']:.1f}s", flush=True)

    partial = bool(args.instances) or manifest.get("partial")
    default_name = (f"qplib-miqp-warm-ab{'-partial' if partial else ''}-"
                    f"{stamp_fields['git_commit']}.csv")
    out_path = (REPO_ROOT / args.out).resolve() if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({key: mm.text(row[key]) for key in CSV_COLUMNS} for row in rows)
    totals = summary(rows)
    for arm in ARMS:
        t = totals[arm]
        print(f"{arm}: {t['instances']} instances, {t['optimal']} optimal, {t['verified']} "
              f"verified, {t['rejected']} rejected by the verifier; {t['nodes']} nodes, "
              f"{t['qp_node_iterations']} QP iterations; iterations per node (shifted geomean "
              f"over the {totals['paired_instances']} paired) "
              f"{t['iterations_per_node_geomean']:.3f}")
    print(f"warm against cold: {totals['agree']} agree, {totals['disagree']} disagree, "
          f"{totals['not_judged']} not judged (a limit stopped an arm)")
    print(f"wrote {out_path}")
    return 1 if totals["disagree"] or totals["warm"]["rejected"] or totals["cold"]["rejected"] \
        else 0


if __name__ == "__main__":
    sys.exit(main())
