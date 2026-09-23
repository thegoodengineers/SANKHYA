#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The Maros-Meszaros section of docs/BENCHMARKS.md, rendered from its CSV (#491).

Kept out of make_benchmarks_doc.py, which calls section() and is already several times the
project's per-file size, and importable on its own so test_maros_meszaros.py can render a
section from a synthetic CSV without generating the whole document.

The rule the section follows is the document's: every number comes from a row of the CSV,
and every instance that did not succeed is NAMED, under the reason it did not.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "maros-meszaros" / "reference.json"
SHIFT_SECONDS = 10.0

REPRODUCE = [
    "```",
    "python bench/runners/fetch_maros_meszaros.py",
    "python bench/runners/maros_meszaros.py",
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


def _set_size() -> int | None:
    try:
        return len(json.loads(MANIFEST.read_text(encoding="utf-8"))["instances"])
    except (OSError, ValueError, KeyError):
        return None


def _sgm(values: list[float], shift: float = SHIFT_SECONDS) -> float:
    return math.exp(sum(math.log(max(v, 0.0) + shift) for v in values) / len(values)) - shift


def failure_lists(rows: list[dict]) -> list[str]:
    """One line per way of not succeeding, each naming its instances. An instance can be on
    more than one line; none is left off every line unless it passed everything."""
    lines = []
    groups = [
        ("Did not reach `optimal`", [r for r in rows if r.get("status") != "optimal"]),
        ("`optimal` but more than 1e-6 relative from the reference objective",
         [r for r in rows if r.get("status") == "optimal" and r.get("matches_reference") != "1"]),
        ("Rejected by `tools/verify_solution.py`",
         [r for r in rows if r.get("independently_verified") == "0"]),
        ("`optimal` but not successful at 1e-6 relative",
         [r for r in rows if r.get("status") == "optimal" and r.get("success_rel_1e-6") != "1"]),
        ("`optimal` but not successful at 1e-9 relative",
         [r for r in rows if r.get("status") == "optimal" and r.get("success_rel_1e-9") != "1"]),
    ]
    for title, members in groups:
        if members:
            lines.append(f"- **{title}** ({len(members)}): {_names(members)}.")
    return lines


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
    machine = rows[0].get("machine", "unknown")
    limit = _float(rows[0], "time_limit")
    options = (rows[0].get("solver_options") or "").strip()
    total = len(rows)
    size = _set_size()

    def count(key: str) -> int:
        return sum(row.get(key) == "1" for row in rows)

    optimal = sum(row.get("status") == "optimal" for row in rows)
    charged = [(_float(r, "solver_seconds") or 0.0) if r.get("success_rel_1e-6") == "1"
               else (limit or 0.0) for r in rows]
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}` · time limit "
        f"{'-' if limit is None else f'{limit:g}'} s per instance"
        + (f" · solver options `{options}`" if options else " · solver defaults"),
        "",
        f"**{total} of {size if size else '?'} instances** run. "
        f"`optimal` on **{optimal}**; within 1e-6 relative of the published objective on "
        f"**{count('matches_reference')}**; accepted by the independent verifier on "
        f"**{count('independently_verified')}**.",
        "",
        "| level | success, relative measures (#491) | success, absolute measures "
        "(the published report's criterion) |",
        "|---|---:|---:|",
        f"| 1e-6 | {count('success_rel_1e-6')} / {total} | {count('success_abs_1e-6')} / {total} |",
        f"| 1e-9 | {count('success_rel_1e-9')} / {total} | {count('success_abs_1e-9')} / {total} |",
        "",
        "Success at a level means the status is `optimal` and the primal residual, the dual "
        "residual and the duality gap are all within it, each recomputed from the QPS file and "
        "the written `.sol` by `bench/runners/qp_residuals.py` through the verifier's own MPS "
        "reader. The reference objectives are the readme's OPT column (BPMPD at default "
        "settings, eight significant digits), parsed by `bench/runners/fetch_maros_meszaros.py` "
        "into `data/maros-meszaros/reference.json` with every file's sha256.",
        "",
        f"Shifted geometric mean of solver time, shift {SHIFT_SECONDS:g} s, an instance not "
        f"successful at 1e-6 charged the full limit as the published report does: "
        f"**{_sgm(charged):.2f} s**." if charged else "",
        "",
        "| instance | rows | cols | status | our objective | reference | rel. gap | "
        "primal rel. | dual rel. | gap rel. | iters | solver time (s) | verified | 1e-6 | 1e-9 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|",
    ]

    def num(row: dict, key: str, spec: str) -> str:
        value = _float(row, key)
        return "-" if value is None else format(value, spec)

    def mark(row: dict, key: str) -> str:
        return {"1": "yes", "0": "**no**"}.get(row.get(key, ""), "-")

    for row in sorted(rows, key=lambda r: r["instance"]):
        out.append(
            f"| `{row['instance']}` | {row.get('rows', '')} | {row.get('columns', '')} "
            f"| {row.get('status', '')} | {num(row, 'our_objective', '.10g')} "
            f"| {num(row, 'reference_objective', '.8g')} | {num(row, 'relative_gap', '.1e')} "
            f"| {num(row, 'primal_residual_rel', '.1e')} | {num(row, 'dual_residual_rel', '.1e')} "
            f"| {num(row, 'duality_gap_rel', '.1e')} | {row.get('iterations', '')} "
            f"| {num(row, 'solver_seconds', '.2f')} | {mark(row, 'independently_verified')} "
            f"| {mark(row, 'success_rel_1e-6')} | {mark(row, 'success_rel_1e-9')} |")
    failures = failure_lists(rows)
    out += [""] + (["Every failure, named:", "", *failures] if failures
                   else ["Every instance run succeeded at both levels."]) + [""]
    return "\n".join(out)
