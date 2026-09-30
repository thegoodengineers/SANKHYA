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


def latest_csv(results_dir: Path) -> Path | None:
    """The newest `million-cpu-<sha>.csv`, by git history then the CSV's own timestamp.

    Not latest_result.latest(): that skips every CSV with a non-empty `solver_options`
    column, because elsewhere such a file is an option experiment beside the tier's default
    run. Here every row carries its arm's options by design, so that filter would hide the
    only evidence there is and the section would print its placeholder. The ordering is the
    same one, taken from latest_result so the two cannot drift; the name must be
    `million-cpu-<sha>.csv`, so a named experiment beside it is never picked."""
    import latest_result

    candidates = [p for p in results_dir.glob("million-cpu-*.csv")
                  if latest_result.is_default_named(p, "million-cpu")]
    if not candidates:
        return None
    order = latest_result.commit_order()

    def rank(path: Path) -> tuple[int, float, str]:
        recorded = latest_result.commit_of(path)
        position = next((i for i, sha in enumerate(order)
                         if recorded and sha.startswith(recorded)), len(order))
        return (position, -latest_result.timestamp_of(path), path.name)

    return sorted(candidates, key=rank)[0]


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
            "`polish` when PDHG's point was not finished by the interior point, `stall` "
            "when the interior point's step collapsed short of its optimality test, "
            "`crossover` when the pivots from that point ran out of time, `overran` when "
            "the runner killed a solve still running past its backstop (1.5 times the "
            "limit plus 600 s), `memory` when the process was killed with its peak near "
            "the machine's RAM.",
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


def experiments_section(results_dir: Path) -> str:
    """The million-row runs made with an option set on purpose (#751), each in its own
    `million-<name>-<sha>.csv` beside the default run: the interior point with its factor
    cap raised, with the supernodal factor, and so on. Never mixed into the table above,
    whose rows are the arms as they ship; every row here shows the options it ran with."""
    paths = sorted(p for p in results_dir.glob("million-*.csv")
                   if not p.name.startswith("million-cpu-"))
    if not paths:
        return ""
    out = [
        "",
        "**Past the defaults.** The same models and verifier with an option changed on "
        "purpose, one CSV per experiment; `options` is what the solver was given.",
        "",
        "| CSV | model | arm | options | status | relative error | verified | iterations "
        "| seconds | peak memory | solver message |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for path in paths:
        for r in _rows(path):
            answered = r.get("status") in WITH_A_POINT
            message = (r.get("message") or "-").replace("|", "/")
            out.append(
                f"| `{path.name}` | {r['family']} | `{r['arm']}` "
                f"| `{r.get('solver_options', '')}` | {r.get('status', '').replace('_', ' ')} "
                f"| {_sci(r, 'relative_error') if answered else '-'} "
                f"| {'yes' if r.get('verified') == '1' else 'no'} "
                f"| {r.get('iterations') or '-'} | {_seconds(r)} "
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
