#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Section 1f.6 of docs/BENCHMARKS.md: decomposed against monolithic on the refinery ladder (#525).

Rendered from the CSV bench/runners/decomposition.py writes. Every row is a solve, and every row is
shown: a decomposition that declined and fell back to the monolithic engines is a row whose
`algorithm` is not `benders`, named as such, and a time slower than the monolithic solve is
printed as slower - the table is the measurement, not an argument for the method.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPRODUCE = [
    "```",
    "python bench/runners/decomposition.py --blocks 2 4 8",
    "```",
]


def _float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def section(path: Path | None) -> str:
    intro = (
        "A multi-period planning model is independent periods joined by what each carries to the "
        "next. Where the joining part is a few LINKING COLUMNS - here the crude and product "
        "stocks - Benders decomposition (`decomposition=benders`, issue #525) fixes them and solves "
        "each group of periods as a small LP of its own, the groups in parallel, with a master LP "
        "over the stocks collecting a cut from each. The refinery ladder of #517 "
        "(`bench/case_studies/refinery/generator.py`) is that model at three sizes, and its exact "
        "analytic optimum, verified in rational arithmetic by the generator, is the reference.")
    if path is None:
        return "\n".join([
            intro, "",
            "**Not yet run at this commit.** The run is made from `main` after the decomposition "
            "merges, so that its stamp is a commit on `main`; the command is:", "", *REPRODUCE, ""])
    rows = read_rows(path)
    if not rows:
        return intro + "\n\nThe CSV is empty.\n"
    out = [intro, "", f"Source CSV: `bench/results/{path.name}`  ",
           f"Commit `{rows[0].get('git_commit', '?')}` · machine `{rows[0].get('machine', '?')}` · "
           f"{rows[0].get('time_limit', '?')} s per solve", "",
           "| instance | rows x cols | mode | method that ran | status | objective | vs exact optimum "
           "| wall (s) | x monolithic | rounds | same objective | verified |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|"]
    for row in rows:
        ours = _float(row, "our_objective")
        gap = _float(row, "relative_gap")
        wall = _float(row, "wall_seconds")
        speed = _float(row, "speedup_vs_monolithic")
        out.append(
            f"| `{row['instance']}` | {row.get('rows', '')} x {row.get('columns', '')} "
            f"| {row['mode']} | {row.get('algorithm') or '-'} | {row.get('status', '')} "
            f"| {'-' if ours is None else f'{ours:.10g}'} "
            f"| {'-' if gap is None else f'{gap:.1e}'} "
            f"| {'-' if wall is None else f'{wall:.2f}'} "
            f"| {'-' if speed is None else f'{speed:.2f}'} "
            f"| {row.get('iterations') or '-'} | {row.get('matches_monolithic') or '-'} "
            f"| {row.get('verified', '')} |")
    out.append("")
    decomposed = [r for r in rows if r["mode"] != "monolithic"]
    ran = [r for r in decomposed if r.get("algorithm") == "benders"]
    fell_back = [r for r in decomposed if r.get("algorithm") != "benders"]
    wrong = [r for r in rows if r.get("matches_monolithic") == "NO"]
    rejected = [r for r in rows if r.get("verified") == "REJECTED"]
    faster = [r for r in ran if (_float(r, "speedup_vs_monolithic") or 0.0) > 1.0]
    out.append(f"**{len(ran)} of {len(decomposed)}** decomposed solves ran Benders; "
               f"**{len(faster)}** of those were faster than the monolithic solve. "
               f"{len(wrong)} disagreed with the monolithic objective; {len(rejected)} answers were "
               "rejected by the independent verifier (`tools/verify_solution.py`).")
    out.append("")
    if fell_back:
        names = ", ".join(f"`{r['instance']}` ({r['mode']}: {r.get('algorithm') or r.get('status')})"
                          for r in fell_back)
        out.append(f"**Fell back to the monolithic engines**, named rather than dropped: {names}.")
        out.append("")
    out += ["Reproduce:", "", *REPRODUCE]
    return "\n".join(out)
