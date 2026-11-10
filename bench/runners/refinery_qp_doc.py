#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""docs/BENCHMARKS.md section 4d: the refinery planning model with convex cost curves, at
about 13k, 110k and 1M columns, SANKHYA's two QP engines against HiGHS's QP solver.

Reads bench/results/head-to-head-refinery-qp-<commit>.csv (compare.py --suite refinery-qp).
Every number printed is a cell of that CSV: the instance's dimensions and sha256, and per
solver its status, its own time, the objective at its point, the relative gap to the
optimum the generator constructed (bench/refinery/refinery_qp.py), and the verifier's
verdict. A run that is not optimal, matched and verified is said to be so, by name.
"""
from __future__ import annotations

import csv
from pathlib import Path

SOLVERS = (("sankhya", "SANKHYA, interior point (default)"),
           ("sankhya-condat-vu", "SANKHYA, Condat-Vu (first-order)"),
           ("highs", "HiGHS QP"))
ORDER = ("refinery_qp_13k", "refinery_qp_110k", "refinery_qp_1m")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell(row: dict | None) -> str:
    if row is None:
        return "-"
    seconds = row.get("solver_seconds") or ""
    secs = f"{float(seconds):.2f} s" if seconds else "no time"
    if row["status"] == "optimal":
        verdict = "verified" if row["independently_verified"] == "1" else "**REJECTED**"
        gap = row.get("relative_gap") or ""
        gap_text = f"gap {float(gap):.1e}" if gap else "no gap"
        matched = "" if row["matches_reference"] == "1" else ", **not matched**"
        return f"optimal, {secs}, {gap_text}, {verdict}{matched}"
    return f"**{row['status']}**, {secs}"


def counted(row: dict | None) -> bool:
    return row is not None and row.get("counted_for_time") == "1"


def section(path: Path | None) -> str:
    out = ["**The model.** `bench/runners/generate_refinery_lp.py`'s T-period refinery (#211) - "
           "crude purchases, CDU throughput, crude and product tanks, yields, unit capacities, "
           "quality budgets and delivery commitments - with convex cost curves: every purchase "
           "column `BUY[k,t]` and throughput column `RUN[k,t]` carries `q_j x_j^2 / 2`, `q_j` a "
           "hundredth in [0.01, 0.09], so the Hessian is diagonal, positive on those columns "
           "and zero elsewhere. The optimum is exact by construction: the LP generator's plan, "
           "tight set and multipliers are kept and the linear prices re-derived as "
           "`c = d + A'y - Q x*`, every KKT condition of the QP re-checked in rational "
           "arithmetic before the file is written (`bench/refinery/refinery_qp.py`). "
           "Generated where the comparison runs by `python bench/refinery/generate.py --qp` "
           "(seed 1) and not committed - the largest file is about 120 MB - so the generator "
           "and its seed are the record and the CSV carries the sha256 of every file each "
           "solver read. Graded as section 4a: `matched` within 1e-6 relative of the "
           "constructed optimum at the solver's own point, `verified` by "
           "`tools/verify_solution.py`; 600 s, one thread, each solver a separate process.", ""]
    if path is None:
        out.append("No `head-to-head-refinery-qp-*.csv` in `bench/results/`, so **no numbers "
                   "are stated for this model**. Run `python bench/runners/compare.py --suite "
                   "refinery-qp` after `python bench/refinery/generate.py --qp`.")
        return "\n".join(out)
    rows = read_csv(path)
    if not rows:
        return "\n".join(out + [f"`{path.name}` is empty."])
    first = rows[0]
    by = {(r["instance"], r["solver"]): r for r in rows}
    names = [n for n in ORDER if any(r["instance"] == n for r in rows)]
    present = [(s, label) for s, label in SOLVERS if any(r["solver"] == s for r in rows)]
    out += [f"Source CSV: `bench/results/{path.name}`  ",
            f"Commit `{first['git_commit']}` · machine `{first['machine']}`  ",
            f"time limit {float(first['time_limit']):g} s · 1 thread per solver", "",
            "| instance | rows | columns | sha256 | optimum | " +
            " | ".join(label for _, label in present) + " |",
            "|---|---:|---:|---|---:|" + "---|" * len(present)]
    for name in names:
        any_row = next(r for r in rows if r["instance"] == name)
        ref = any_row.get("reference_objective") or ""
        ref_text = f"{float(ref):.12g}" if ref else "-"
        out.append(f"| `{name}` | {int(any_row['rows']):,} | {int(any_row['columns']):,} | "
                   f"`{any_row['instance_sha256'][:12]}` | {ref_text} | "
                   + " | ".join(cell(by.get((name, s))) for s, _ in present) + " |")
    out.append("")
    ours, theirs = "sankhya", "highs"
    lines = []
    for name in names:
        a, b = by.get((name, ours)), by.get((name, theirs))
        if counted(a) and counted(b):
            ratio = float(b["solver_seconds"]) / max(float(a["solver_seconds"]), 1e-9)
            lines.append(f"`{name}`: both optimal, matched and verified; HiGHS's time over "
                         f"SANKHYA's is {ratio:.1f}x.")
        elif counted(a):
            lines.append(f"`{name}`: SANKHYA optimal, matched and verified in "
                         f"{float(a['solver_seconds']):.2f} s; HiGHS's QP solver "
                         f"{b['status'] if b else 'did not run'}"
                         + (f" after {float(b['solver_seconds']):.0f} s" if b and
                            b.get("solver_seconds") else "") + ", so no ratio is stated.")
        elif counted(b):
            lines.append(f"`{name}`: HiGHS's QP solver optimal, matched and verified in "
                         f"{float(b['solver_seconds']):.2f} s; SANKHYA "
                         f"{a['status'] if a else 'did not run'} - **a loss**.")
        else:
            lines.append(f"`{name}`: neither SANKHYA's default engine nor HiGHS's QP solver "
                         "produced a matched, verified answer within the limit.")
    out += ["**Read per instance**, from the cells above:", ""] + [f"- {x}" for x in lines]
    out.append("")
    return "\n".join(out)
