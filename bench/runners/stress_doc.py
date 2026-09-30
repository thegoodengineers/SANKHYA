#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The stress-set section of docs/BENCHMARKS.md (#762), rendered from stress-<sha>.csv.

Kept out of make_benchmarks_doc.py, which calls section(). Every count comes from a row of
the CSV, and every answer not graded correct is named with the CSV's own reason.
"""
from __future__ import annotations

import csv
from pathlib import Path

VERDICTS = ("correct", "wrong", "failed")


def section(path: Path | None) -> str:
    if path is None:
        return ("_No `stress-*.csv` in `bench/results/`. Run "
                "`python bench/runners/stress_instances.py` then "
                "`python bench/runners/stress.py`._\n")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    solvers = sorted({r["solver"] for r in rows}, key=lambda s: (s != "sankhya", s))
    families = sorted({r["family"] for r in rows})
    versions = {r["solver"]: r["solver_version"] for r in rows}
    first = rows[0]
    out = [f"Measured on SANKHYA commit `{first['git_commit']}` against HiGHS "
           f"{versions.get('highs', '-')} (highspy, separate process), machine "
           f"`{first['machine']}`, {len({r['instance'] for r in rows})} instances. "
           f"Source: `{path.name}`.", "",
           "| family | instances | " + " | ".join(
               f"{s} correct / wrong / failed" for s in solvers) + " |",
           "|---|---|" + "---|" * len(solvers)]
    for family in families + ["**all**"]:
        picked = [r for r in rows if family == "**all**" or r["family"] == family]
        cells = []
        for solver in solvers:
            verdicts = [r["verdict"] for r in picked if r["solver"] == solver]
            cells.append(" / ".join(str(verdicts.count(v)) for v in VERDICTS))
        out.append(f"| {family if family.startswith('*') else '`' + family + '`'} | "
                   f"{len({r['instance'] for r in picked})} | " + " | ".join(cells) + " |")
    for solver in solvers:
        for verdict in ("wrong", "failed"):
            named = [r for r in rows if r["solver"] == solver and r["verdict"] == verdict]
            if not named:
                continue
            out += ["", f"**{solver}, {verdict}** ({len(named)}):", ""]
            out += [f"- `{r['instance']}`: {r['reason'].replace('|', '/')[:220]}" for r in named]
    tolerance = [r["instance"] for r in rows
                 if r["verdict"] == "correct" and r["reason"].startswith("tolerance-feasible")]
    if tolerance:
        out += ["", f"{len(tolerance)} thin-infeasible answers are `optimal` at a point the "
                "verifier accepts with every row inside 1e-7: at the stated tolerance that "
                "point is feasible, so the answer is graded correct, and named here so the "
                "grade is not mistaken for a proof of infeasibility."]
    return "\n".join(out) + "\n"
