#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The convex QCQP section of docs/BENCHMARKS.md (#755), rendered from qplib_qcqp.py's CSV.

Kept in its own file for the reason qplib_doc.py is. Every number comes from a CSV row, every
instance that did not pass is named, and the two engines' objectives are compared row by row:
the issue asks that the global method and the conic engine agree wherever both finish.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

REPRODUCE = [
    "```",
    "python bench/runners/fetch_qplib.py --qcqp",
    "python bench/runners/qplib_qcqp.py --time-limit 300   # engines conic and global",
    "```",
]


def _float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def agreement(rows: list[dict], tolerance: float = 1e-4) -> list[str]:
    """One line per instance both engines finished `optimal` on: agree or not, and by how much.
    The tolerance is the global method's own gap target (mip_relative_gap)."""
    by_instance: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], {})[row.get("engine", "")] = row
    both, disagree = [], []
    for name, engines in sorted(by_instance.items()):
        conic, glob = engines.get("conic"), engines.get("global")
        if not conic or not glob or conic.get("status") != "optimal" \
                or glob.get("status") != "optimal":
            continue
        a, b = _float(conic, "our_objective"), _float(glob, "our_objective")
        if a is None or b is None:
            continue
        rel = abs(a - b) / max(1.0, abs(a))
        both.append(name)
        if rel > tolerance:
            disagree.append(f"`{name}` ({rel:.1e})")
    if not both:
        return ["No instance was finished `optimal` by both engines, so agreement is not "
                "measured."]
    return [f"Both engines `optimal` on {len(both)} instance(s); objectives agree within "
            f"{tolerance:g} relative on {len(both) - len(disagree)} of them"
            + (f"; **disagree on {', '.join(disagree)}**." if disagree else ".")]


def section(path: Path | None) -> str:
    if path is None:
        return "\n".join(["Not yet run on `main`. Reproduce with:", "", *REPRODUCE, ""])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return f"`{path.name}` holds no rows.\n"
    commit = rows[0].get("git_commit", "unknown")
    if "-dirty" in commit:
        return (f"`{path.name}` is stamped `{commit}`: produced from a modified tree. Re-run on "
                "a clean checkout of a main commit.\n")
    limit = _float(rows[0], "time_limit")
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{rows[0].get('machine', 'unknown')}` · time limit "
        f"{'-' if limit is None else f'{limit:g}'} s per instance",
        "",
    ]
    for engine in sorted({row.get("engine", "") for row in rows}):
        mine = [row for row in rows if row.get("engine") == engine]
        judged = [row for row in mine if row.get("passed") != ""]
        failed = sorted(row["instance"] for row in judged if row.get("passed") != "1")
        out.append(
            f"- `{engine}`: **passed {len(judged) - len(failed)} of {len(judged)}**; `optimal` "
            f"on {sum(r.get('status') == 'optimal' for r in mine)} of {len(mine)}, verified on "
            f"{sum(r.get('verified') == '1' for r in mine)}"
            + (f"; not passed: {', '.join(f'`{n}`' for n in failed)}" if failed else "") + ".")
    out += ["", *agreement(rows), "",
            "A pass is `optimal`, within 1e-6 relative of QPLIB's value (qplib.solu, a best "
            "known point, not a proof), and accepted by `tools/verify_solution.py`: for the "
            "conic engine stationarity through every quadratic row's gradient `a + 2Qx`, the "
            "multipliers' signs and complementarity; for the global method the bound it proved.",
            "",
            "| instance | engine | status | our objective | QPLIB value | rel. gap | iters | "
            "solver time (s) | verified | passed |",
            "|---|---|---|---:|---:|---:|---:|---:|:--:|:--:|"]

    def num(row: dict, key: str, spec: str) -> str:
        value = _float(row, key)
        return "-" if value is None else format(value, spec)

    def mark(row: dict, key: str) -> str:
        return {"1": "yes", "0": "**no**"}.get(row.get(key, ""), "-")

    for row in sorted(rows, key=lambda r: (r["instance"], r.get("engine", ""))):
        out.append(
            f"| `{row['instance']}` | {row.get('engine', '')} | {row.get('status', '')} | "
            f"{num(row, 'our_objective', '.10g')} | {num(row, 'reference_objective', '.10g')} | "
            f"{num(row, 'rel_gap', '.1e')} | {row.get('iterations', '') or '-'} | "
            f"{num(row, 'solver_seconds', '.2f')} | {mark(row, 'verified')} | "
            f"{mark(row, 'passed')} |")
    misses = [row for row in rows if row.get("passed") == "0" and row.get("message")]
    if misses:
        out += ["", "Why the rows that did not pass stopped where they did:", ""]
        for row in sorted(misses, key=lambda r: (r["instance"], r.get("engine", ""))):
            out.append(f"- `{row['instance']}` ({row.get('engine', '')}): {row['message']}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    # The smallest self-check: a synthetic CSV, both engines optimal and agreeing on one row.
    import tempfile
    columns = ["instance", "engine", "status", "our_objective", "reference_objective",
               "rel_gap", "verified", "passed", "git_commit", "machine", "time_limit",
               "iterations", "solver_seconds", "message"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qplib-qcqp-abc.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for engine, value in (("conic", -1.0), ("global", -1.00001)):
                writer.writerow({"instance": "Q1", "engine": engine, "status": "optimal",
                                 "our_objective": value, "reference_objective": -1.0,
                                 "rel_gap": 0.0, "verified": "1", "passed": "1",
                                 "git_commit": "abc", "machine": "m", "time_limit": 300,
                                 "iterations": 10, "solver_seconds": 0.1, "message": ""})
        text = section(path)
        assert "passed 1 of 1" in text, text
        assert "agree within 0.0001 relative on 1 of them." in text, text
    print("qplib_qcqp_doc self-check passed")
