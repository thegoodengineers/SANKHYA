#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Section 1f.5 of docs/BENCHMARKS.md, a million rows (#751), rendered from the CSV that
bench/runners/million.py writes. Kept in its own file, as the GPU and QP sections are.

Two tables: every arm on every model, and the attribution of each arm that did not end
`optimal` and verified - the numbers that say why, not an adjective. `readme_scale_line`
is the sentence the README's scale row quotes, printed by this module so the row is copied
from the CSV rather than written from memory.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

WITH_A_POINT = ("optimal", "feasible", "iteration_limit", "time_limit")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def _finished(row: dict) -> bool:
    return row.get("status") == "optimal" and row.get("verified") == "1"


def _seconds(row: dict) -> str:
    value = _float(row, "wall_seconds")
    return "-" if value is None else f"{value:,.0f}"


def _sci(row: dict, key: str) -> str:
    value = _float(row, key)
    return "-" if value is None else f"{value:.1e}"


def million_section(path: Path | None) -> str:
    if path is None:
        return ("_No `million-cpu-*.csv` in `bench/results/`. Produce one with_ "
                "`python bench/runners/million.py --binary build/sankhya --keep DIR`.\n")
    rows = _rows(path)
    if not rows:
        return "No million-row results recorded yet.\n"
    first = rows[0]
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{first.get('git_commit', 'unknown')}` · machine `{first.get('machine', '')}`"
        f" · {first.get('time_limit', '?')}s per solve",
        "",
        "Three generated models at a million rows or more (#751), each with its optimum "
        "exact by construction and each written from a seed; the sha256 in the CSV names the "
        "file that was solved. `verified` is `tools/verify_solution.py`'s exit code on the "
        "written solution, which parses the model itself and recomputes every residual.",
        "",
        "| model | rows x cols | nonzeros | arm | route | status | relative error "
        "| verified | seconds | peak memory |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        answered = r.get("status") in WITH_A_POINT
        out.append(
            f"| {r['family']} | {int(r['rows']):,} x {int(r['columns']):,} "
            f"| {int(r['nonzeros']):,} | `{r['arm']}` | {r.get('route') or '-'} "
            f"| {r.get('status', '').replace('_', ' ')} "
            f"| {_sci(r, 'relative_error') if answered else '-'} "
            f"| {'yes' if r.get('verified') == '1' else 'no'} | {_seconds(r)} "
            f"| {int(float(r.get('peak_rss_mb') or 0)):,} MB |")
    out.append("")
    finished = [r for r in rows if _finished(r)]
    out.append(f"**{len(finished)} of {len(rows)}** arms end `optimal` and verified.")
    out.append("")
    unfinished = [r for r in rows if not _finished(r)]
    if unfinished:
        out += [
            "Every arm that did not finish, and the numbers that say why. `attribution` is "
            "read off the row by `bench/runners/million.py`: `iterations` when the method "
            "ran to the limit still converging, `fill` when the factor's size stopped it, "
            "`polish` when PDHG's point was not finished by the interior point, `memory` "
            "when the process was killed or its peak neared the machine's RAM.",
            "",
            "| model | arm | attribution | status | iterations | primal inf | dual inf "
            "| peak memory | solver message |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in unfinished:
            message = (r.get("message") or "-").replace("|", "/")
            out.append(
                f"| {r['family']} | `{r['arm']}` | **{r.get('attribution') or '-'}** "
                f"| {r.get('status', '').replace('_', ' ')} | {r.get('iterations') or '-'} "
                f"| {_sci(r, 'primal_infeasibility')} | {_sci(r, 'dual_infeasibility')} "
                f"| {int(float(r.get('peak_rss_mb') or 0)):,} MB | {message} |")
        out.append("")
    return "\n".join(out)


def readme_scale_line(path: Path) -> str:
    """The README's million-row sentence, from the CSV."""
    rows = _rows(path)
    parts = []
    for r in rows:
        if _finished(r):
            parts.append(f"{r['family']} ({int(r['rows']):,} rows) `optimal` and verified by "
                         f"`{r['arm']}` ({r.get('route')}) in {_seconds(r)} s, relative error "
                         f"{_sci(r, 'relative_error')}")
        else:
            parts.append(f"{r['family']} by `{r['arm']}` {r.get('status', '')} at "
                         f"{_seconds(r)} s ({r.get('attribution') or 'unattributed'})")
    return (f"At a million rows on the CPU (#751, `bench/results/{path.name}`, "
            f"{rows[0].get('machine', '')}): " + "; ".join(parts) + ".")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: million_doc.py bench/results/million-cpu-<commit>.csv", file=sys.stderr)
        sys.exit(2)
    print(readme_scale_line(Path(sys.argv[1])))
