#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The nonlinear section of docs/BENCHMARKS.md, rendered from the nlp_bench.py CSVs.

Kept out of make_benchmarks_doc.py for the reason qplib_doc.py and maros_meszaros_doc.py
are: that file is already several times the size guideline. Every number comes from a CSV
row written by bench/runners/nlp_bench.py, and every instance that did not match, that the
independent checker rejected, or that returned no point is named under why.

THE STATUSES ARE THE POINT. A local method on a non-convex model can only claim a KKT point,
and the engine says so: `optimal` is a global claim made only when the model is proved
convex, `locally_optimal` is any other KKT point, `locally_infeasible` is a local minimizer
of the constraint violation, and a limit is a limit. `match` is whether the objective is at
the published one to the runner's tolerance; a `locally_optimal` row at another local
optimum is a legitimate answer recorded as a non-match, never hidden.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPRODUCE = [
    "```",
    "python bench/runners/nlp_bench.py --data data/nlp/hs                              # 70 Hock-Schittkowski",
    "python bench/runners/nlp_bench.py --data data/nlp/minlplib --match-tolerance 1e-4  # convex MINLPLib",
    "python bench/runners/nlp_bench.py --data data/nlp/minlplib --match-tolerance 1e-4 \\",
    "    --solver-option minlp_method=oa                                                # the same, by outer approximation",
    "python bench/runners/nlp_bench.py --data data/nlp/minlplib --match-tolerance 1e-4 \\",
    "    --solver-option minlp_method=bnb                                               # the tree as an option run, for the comparison",
    "```",
]


def _float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _names(rows: list[dict]) -> str:
    return ", ".join(f"`{row['instance']}`" for row in sorted(rows, key=lambda r: r["instance"]))


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_section(title: str, path: Path | None, what: str) -> str:
    """One set: the counts, the status breakdown, the table, and the named exceptions."""
    if path is None:
        return f"**{title}**: not yet run at this commit."
    rows = read_rows(path)
    if not rows:
        return f"**{title}**: the CSV is empty."

    matched = [r for r in rows if r.get("match") == "yes"]
    verified = [r for r in rows if r.get("verified") == "yes"]
    rejected = [r for r in rows if r.get("verified") == "REJECTED"]
    no_point = [r for r in rows if r.get("verified") == "no point"]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.get("status", "")] = statuses.get(row.get("status", ""), 0) + 1
    breakdown = ", ".join(f"`{status}` {count}" for status, count in sorted(statuses.items()))
    limit = rows[0].get("time_limit", "?")

    out = [
        f"**{title}** - {what}",
        "",
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{rows[0].get('git_commit', '?')}` · machine `{rows[0].get('machine', '?')}` · "
        f"{limit} s per problem",
        "",
        f"**{len(matched)} of {len(rows)}** reached the published objective, and "
        f"**{len(verified)} of {len(rows)}** answers were accepted by the independent checker "
        f"(`tools/verify_solution.py`, its own `.nl` reader and its own derivatives). "
        f"Statuses: {breakdown}.",
        "",
        "| problem | status | our objective | published | rel. gap | iters | time (s) | match | verified |",
        "|---|---|---:|---:|---:|---:|---:|:--:|:--:|",
    ]
    for row in sorted(rows, key=lambda r: r["instance"]):
        ours = _float(row, "objective")
        published = _float(row, "reference_objective")
        gap = _float(row, "rel_gap")
        seconds = _float(row, "seconds")
        out.append(
            f"| `{row['instance']}` | {row.get('status', '')} "
            f"| {'-' if ours is None else f'{ours:.10g}'} "
            f"| {'-' if published is None else f'{published:.10g}'} "
            f"| {'-' if gap is None else f'{gap:.2e}'} "
            f"| {row.get('iterations', '') or '-'} "
            f"| {'-' if seconds is None else f'{seconds:.2f}'} "
            f"| {'yes' if row.get('match') == 'yes' else '**NO**'} "
            f"| {row.get('verified', '')} |")
    out.append("")
    unmatched = [r for r in rows if r.get("match") != "yes"]
    if unmatched:
        out.append(f"**Not at the published objective**, named rather than dropped: "
                   f"{_names(unmatched)}.")
        out.append("")
    if rejected:
        out.append(f"**Rejected by the independent checker**: {_names(rejected)}.")
        out.append("")
    if no_point:
        out.append(f"**No point to check** (a limit or an infeasibility verdict): "
                   f"{_names(no_point)}.")
        out.append("")
    return "\n".join(out)


def comparison(default_path: Path | None, oa_path: Path) -> str:
    """The two MINLP methods side by side on the same instances (#528): one row per instance,
    each method's status, objective and wall time, and whether they reached the same value.
    Both CSVs are runs of the same binary's two methods on the same set, so a difference in
    a row is a difference between the methods, not between the inputs."""
    if default_path is None:
        return "No default-method run is committed to compare against."
    by_name = {r["instance"]: r for r in read_rows(default_path)}
    oa_rows = {r["instance"]: r for r in read_rows(oa_path)}
    names = sorted(set(by_name) & set(oa_rows))

    def cell(row: dict) -> tuple[str, str]:
        ours = _float(row, "objective")
        seconds = _float(row, "seconds")
        status = row.get("status", "")
        return (f"{status}" + ("" if ours is None else f" {ours:.10g}"),
                "-" if seconds is None else f"{seconds:.2f}")

    out = ["Outer approximation against NLP-based branch and bound, instance by instance "
           f"(`{oa_path.name}` against `{default_path.name}`):", "",
           "| problem | branch and bound | s | outer approximation | s | published | both optimal "
           "and equal |", "|---|---|---:|---|---:|---:|:--:|"]
    both = 0
    for name in names:
        b, o = by_name[name], oa_rows[name]
        (bs, bt), (os_, ot) = cell(b), cell(o)
        bo, oo = _float(b, "objective"), _float(o, "objective")
        same = (b.get("status") == "optimal" and o.get("status") == "optimal"
                and bo is not None and oo is not None
                and abs(bo - oo) <= 2e-4 * max(1.0, abs(bo)))
        both += same
        ref = _float(b, "reference_objective")
        out.append(f"| `{name}` | {bs} | {bt} | {os_} | {ot} "
                   f"| {'-' if ref is None else f'{ref:.10g}'} | {'yes' if same else '-'} |")
    out += ["", f"Both methods proved the same optimum on **{both} of {len(names)}** instances.",
            ""]
    return "\n".join(out)


def section(hs_path: Path | None, minlp_path: Path | None,
            minlp_oa_path: Path | None = None, minlp_bnb_path: Path | None = None) -> str:
    oa_part = [
        set_section("MINLPLib, convex, outer approximation", minlp_oa_path,
                    "the same instances and tolerance, solved with `--option minlp_method=oa`: "
                    "a MILP master over linearizations of the nonlinear rows (Duran and "
                    "Grossmann 1986) alternated with the convex NLP at the master's integers, "
                    "off by default (#528)."),
        "",
    ]
    if minlp_oa_path is not None:
        oa_part += [comparison(minlp_bnb_path or minlp_path, minlp_oa_path), ""]
    out = [
        set_section("Hock-Schittkowski", hs_path,
                    "the 70 problems of Hock and Schittkowski, *Test Examples for Nonlinear "
                    "Programming Codes* (1981), from Vanderbei's AMPL models converted by "
                    "`bench/runners/hs_mod_to_nl.py`, matched at a relative 1e-6. The "
                    "interior point with a filter line search after Wachter and Biegler "
                    "(2006); where the model is not proved convex, `locally_optimal` is the "
                    "status the engine can honestly give, and the breakdown says how often "
                    "that is."),
        "",
        set_section("MINLPLib, convex", minlp_path,
                    "convex mixed-integer instances of MINLPLib with a published primal "
                    "bound, matched at the MIP gap target 1e-4, solved by NLP-based branch "
                    "and bound - run only when the relaxation is proved convex, so `optimal` "
                    "here is a closed bound."),
        "",
        *oa_part,
        "Reproduce:",
        "",
        *REPRODUCE,
    ]
    return "\n".join(out)
