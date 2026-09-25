#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The QPLIB section of docs/BENCHMARKS.md, rendered from its CSV and the manifest (#492).

Kept out of make_benchmarks_doc.py for the reason maros_meszaros_doc.py is: that file is
already several times the size guideline, and test_qplib.py renders this section from a
synthetic CSV without generating the whole document. Every number comes from a CSV row or
from data/qplib/reference.json, and every instance that did not pass is named under why.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "qplib" / "reference.json"

REPRODUCE = [
    "```",
    "python bench/runners/fetch_qplib.py",
    "python bench/runners/qplib.py                # the full selection, engines auto and ipm",
    "python bench/runners/qplib.py --tier small   # the instances of at most 100,000 coefficients",
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


def load_manifest(path: Path = MANIFEST) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def selection_paragraph(manifest: dict | None) -> str:
    if not manifest:
        return "No `data/qplib/reference.json` is committed yet; `fetch_qplib.py` writes it."
    instances = manifest.get("instances", {})
    skipped = sorted(name for name, entry in instances.items() if "skipped" in entry)
    fetched = [entry for entry in instances.values() if "skipped" not in entry]
    small = sum(entry.get("tier") == "small" for entry in fetched)
    unreferenced = sorted(name for name, entry in instances.items()
                          if entry.get("reference_objective") is None)
    others = {code: count for code, count in manifest.get("convex_continuous_classes", {}).items()
              if not (code[0] in "CD" and code[2] in "NBL")}
    listed = manifest.get("listing_counts", {}).get("rows", "?")
    parts = [
        f"Selection, read from QPLIB's listing: {manifest.get('selection_rule', '?')}. "
        f"**{len(instances)} instances** of the {listed} listed; {len(fetched)} fetched "
        f"({small} in the small tier, at most "
        f"{manifest.get('tiers', {}).get('small_max_coefficients', '?'):,} stored coefficients)"
        + (f", {len(skipped)} over the size cap and not run: {', '.join(f'`{n}`' for n in skipped)}"
           if skipped else "") + ".",
    ]
    if unreferenced:
        parts.append("QPLIB publishes no solution point for "
                     + ", ".join(f"`{n}`" for n in unreferenced)
                     + ": run, named, and outside the pass count.")
    if others:
        parts.append("Convex continuous instances outside the selection, by type: "
                     + ", ".join(f"{code} {count}" for code, count in sorted(others.items()))
                     + " (quadratic constraints: #514's set).")
    licence = manifest.get("licence", {})
    parts.append(f"Licence, as the site states it: {licence.get('licence_statement', '?')} "
                 f"({licence.get('licence_url', '')})")
    return " ".join(parts)


def failure_lists(rows: list[dict]) -> list[str]:
    lines = []
    for engine in sorted({row.get("engine", "") for row in rows}):
        mine = [row for row in rows if row.get("engine") == engine]
        optimal = [r for r in mine if r.get("status") == "optimal"]
        groups = [
            ("did not reach `optimal`", [r for r in mine if r.get("status") != "optimal"]),
            ("`optimal` but more than 1e-6 relative from QPLIB's value",
             [r for r in optimal if r.get("matches_reference") == "0"]),
            ("rejected by `tools/verify_solution.py`",
             [r for r in mine if r.get("verified") == "0"]),
            ("`optimal` but a residual above 1e-6 relative",
             [r for r in optimal if r.get("success_rel_1e-6") != "1"]),
            ("no published reference", [r for r in mine if r.get("passed") == ""]),
        ]
        for title, members in groups:
            if not members:
                continue
            if title == "did not reach `optimal`":  # the status says which way
                named = ", ".join(f"`{r['instance']}` ({r.get('status', '?')})"
                                  for r in sorted(members, key=lambda r: r["instance"]))
            else:
                named = _names(members)
            lines.append(f"- `{engine}`, **{title}** ({len(members)}): {named}.")
    return lines


def section(path: Path | None, manifest: dict | None = None) -> str:
    manifest = manifest if manifest is not None else load_manifest()
    head = [selection_paragraph(manifest), ""]
    if path is None:
        return "\n".join(head + ["Not yet run on `main`. Reproduce with:", "", *REPRODUCE, ""])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "\n".join(head + [f"`{path.name}` holds no rows.", ""])
    commit = rows[0].get("git_commit", "unknown")
    if "-dirty" in commit:
        return "\n".join(head + [f"`{path.name}` is stamped `{commit}`: produced from a "
                                 "modified tree. Re-run on a clean checkout of a main commit.", ""])
    limit = _float(rows[0], "time_limit")
    options = (rows[0].get("solver_options") or "").strip()
    out = head + [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{rows[0].get('machine', 'unknown')}` · time limit "
        f"{'-' if limit is None else f'{limit:g}'} s per instance"
        + (f" · solver options `{options}`" if options else " · solver defaults"),
        "",
    ]
    for engine in sorted({row.get("engine", "") for row in rows}):
        mine = [row for row in rows if row.get("engine") == engine]
        judged = [row for row in mine if row.get("passed") != ""]
        engine_options = (mine[0].get("engine_options") or "").strip()
        optimal = sum(r.get("status") == "optimal" for r in mine)
        out.append(
            f"- `{engine}`" + (f" (`{engine_options}`)" if engine_options else "")
            + f": **passed {sum(r.get('passed') == '1' for r in judged)} of {len(judged)}** "
            f"with a published reference; `optimal` on {optimal} of {len(mine)}, within 1e-6 "
            f"of QPLIB's value on {sum(r.get('matches_reference') == '1' for r in mine)}, "
            f"accepted by the verifier on "
            f"{sum(r.get('verified') == '1' for r in mine)}.")
    out += [
        "",
        "A pass is `optimal`, within 1e-6 relative of QPLIB's value (qplib.solu, a best known "
        "point, not a proven optimum), all three QP residuals of `qp_residuals.py` within 1e-6 "
        "relative, and accepted by the independent verifier, all on the QPS file "
        "`qplib_format.py` converted the `.qplib` file to - a conversion checked at QPLIB's own "
        "published point for every instance when it was fetched.",
        "",
        "| instance | engine | rows | cols | status | our objective | QPLIB value | rel. gap | "
        "worst residual | iters | solver time (s) | verified | passed |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|:--:|",
    ]

    def num(row: dict, key: str, spec: str) -> str:
        value = _float(row, key)
        return "-" if value is None else format(value, spec)

    def mark(row: dict, key: str) -> str:
        return {"1": "yes", "0": "**no**"}.get(row.get(key, ""), "-")

    for row in sorted(rows, key=lambda r: (r["instance"], r.get("engine", ""))):
        residuals = [_float(row, k) for k in ("primal_residual_rel", "dual_residual_rel",
                                              "duality_gap_rel")]
        worst = "-" if None in residuals else f"{max(residuals):.1e}"
        out.append(
            f"| `{row['instance']}` | {row.get('engine', '')} | {row.get('rows', '')} "
            f"| {row.get('cols', '')} | {row.get('status', '')} "
            f"| {num(row, 'our_objective', '.10g')} | {num(row, 'reference_objective', '.10g')} "
            f"| {num(row, 'rel_gap', '.1e')} | {worst} | {row.get('iterations', '')} "
            f"| {num(row, 'solver_seconds', '.2f')} | {mark(row, 'verified')} "
            f"| {mark(row, 'passed')} |")
    failures = failure_lists(rows)
    out += [""] + (["Every failure, named:", "", *failures] if failures
                   else ["Every instance run passed under every engine."]) + [""]
    return "\n".join(out)
