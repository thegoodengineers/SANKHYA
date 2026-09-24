#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The pooling section of docs/BENCHMARKS.md, rendered from its CSV (#516).

Kept out of make_benchmarks_doc.py, which calls section(), and importable on its own so
test_pooling.py can render a section from a synthetic CSV. Every number comes from a row of
the CSV; every run that did not pass is NAMED under the reason it did not.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

REPRODUCE = [
    "```",
    "python bench/runners/pooling_models.py --check",
    "python bench/runners/pooling.py --time-limit 60",
    "```",
]


def _float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _num(row: dict, key: str, spec: str = ".10g") -> str:
    value = _float(row, key)
    return "-" if value is None else format(value, spec)


def root_bound_table(rows: list[dict]) -> list[str]:
    """P against PQ at the root: the relaxation bound of each and how much of the P root gap
    the PQ root closes. Only instances where both formulations report a root bound."""
    by_instance: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], {})[row["formulation"]] = row
    out = []
    for name in sorted(by_instance):
        forms = by_instance[name]
        p_root = _float(forms.get("p", {}), "root_bound")
        pq_root = _float(forms.get("pq", {}), "root_bound")
        if p_root is None or pq_root is None:
            continue
        reference = _float(forms.get("pq", forms.get("p")), "reference_objective")
        closed = "-"
        if reference is not None and reference - p_root > 1e-9:
            closed = f"{100.0 * (pq_root - p_root) / (reference - p_root):.1f} %"
        q_root = _num(forms.get("q", {}), "root_bound", ".8g")
        out.append(f"| `{name}` | {p_root:.8g} | {q_root} | {pq_root:.8g} | "
                   f"{'-' if reference is None else f'{reference:.10g}'} | {closed} |")
    if not out:
        return ["No run reported a root relaxation bound for both the P- and the "
                "PQ-formulation, so the comparison #516 asks for is not available from this "
                "CSV.", ""]
    return ["| instance | P root bound | Q root bound | PQ root bound | global optimum | "
            "P root gap closed by PQ |", "|---|---:|---:|---:|---:|---:|", *out, ""]


def section(path: Path | None) -> str:
    if path is None:
        return "\n".join(["Not yet run on `main`. Reproduce with:", "", *REPRODUCE, ""])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return f"`{path.name}` holds no rows.\n"
    commit = rows[0].get("git_commit", "unknown")
    if "-dirty" in commit:
        return (f"`{path.name}` is stamped `{commit}`: produced from a modified tree. "
                "Re-run on a clean checkout of a main commit.\n")
    limit = _float(rows[0], "time_limit")
    options = (rows[0].get("solver_options") or "").strip()
    passed = sum(row.get("passed") == "1" for row in rows)
    instances = sorted({row["instance"] for row in rows})
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{rows[0].get('machine', 'unknown')}` · time limit "
        f"{'-' if limit is None else f'{limit:g}'} s per run"
        + (f" · solver options `{options}`" if options else " · solver defaults"),
        "",
        f"**{passed} of {len(rows)} runs** ({len(instances)} instances) reached the published "
        "global optimum within 1e-4 relative, claimed `optimal`, and were accepted by "
        "`tools/verify_solution.py` against the original non-convex model.",
        "",
        "| instance | form | status | our objective | global optimum | proven bound | "
        "root bound | nodes | time (s) | verified |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in sorted(rows, key=lambda r: (r["instance"], r["formulation"])):
        verified = {"1": "yes", "0": "**no**"}.get(row.get("independently_verified", ""), "-")
        out.append(
            f"| `{row['instance']}` | {row['formulation'].upper()} | {row.get('status', '')} "
            f"| {_num(row, 'our_objective')} | {_num(row, 'reference_objective')} "
            f"| {_num(row, 'dual_bound')} | {_num(row, 'root_bound', '.8g')} "
            f"| {row.get('nodes', '') or '-'} | {_num(row, 'wall_seconds', '.2f')} "
            f"| {verified} |")
    out += ["", "#### Root relaxation: PQ against P", ""] + root_bound_table(rows)
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row.get("failure_reason"):
            grouped.setdefault(row["failure_reason"], []).append(
                f"`{row['instance']}_{row['formulation']}`")
    if grouped:
        out += ["Every run that did not pass, named:", ""]
        for reason, names in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            out.append(f"- **{reason}** ({len(names)}): {', '.join(names)}.")
    else:
        out.append("Every run passed.")
    out.append("")
    return "\n".join(out)
