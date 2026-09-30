#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The whole-QPLIB section of docs/BENCHMARKS.md, rendered from qplib-all-<sha>.csv (#835).

Every number is counted from the CSV's rows, and every instance appears in the per-instance
table: read or not, the engine it went to, the refusal and its reason, solved, matched to
QPLIB's published objective, verified. An instance whose answer disagrees with the reference
the wrong way, or whose point the verifier rejected, is also named on its own line.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPRODUCE = [
    "```",
    "python bench/runners/fetch_qplib_all.py",
    "python bench/runners/qplib_all.py --time-limit 60",
    "```",
]
REFUSALS = {
    "nonconvex": "nonconvex objective, refused by the engine's convexity test",
    "quadratic_constraints": "quadratic constraints, refused by the reader without "
                             "`--option nonconvex=global`",
    "conversion": "not converted to QPS by `qplib_format.py`",
    "reader_error": "reader error",
    "crashed": "crashed",
    "hung": "no exit within the limit plus 120 s",
}


def _yes(row: dict, key: str) -> bool:
    return row.get(key) == "1"


def _names(rows: list[dict]) -> str:
    return ", ".join(f"`{row['instance']}`" for row in rows)


def _cell(value: str) -> str:
    return value.replace("|", "/") if value else "-"


def _mark(row: dict, key: str) -> str:
    return {"1": "yes", "0": "no"}.get(row.get(key, ""), "-")


def section(path: Path | None) -> str:
    if path is None:
        return ("Not yet run: no `bench/results/qplib-all-<commit>.csv`. Reproduce with" + "\n\n"
                + "\n".join(REPRODUCE) + "\n")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: r["instance"])
    if not rows:
        return f"`{path.name}` has no rows.\n"
    first = rows[0]
    count = lambda key: sum(_yes(r, key) for r in rows)  # noqa: E731
    out = [
        f"From `bench/results/{path.name}`: commit `{first['git_commit']}`, "
        f"{first['machine']}, {float(first['time_limit']):g} s per instance, "
        f"{first['jobs']} instance(s) at a time. Default options: no engine chosen, no "
        "`nonconvex=global`, so this is what the reader and the dispatcher do on their own.",
        "",
        "| | instances |", "|---|---:|",
        f"| in the run | {len(rows)} |",
        f"| read OK | {count('read_ok')} |",
        f"| refused (any reason) | {sum(bool(r['refusal']) for r in rows)} |",
        f"| solved (status optimal) | {count('solved')} |",
        f"| matched QPLIB's objective | {count('matches_reference')} |",
        f"| verified by `tools/verify_solution.py` | {count('verified')} |",
        f"| better than QPLIB's objective by more than the tolerance | "
        f"{count('beats_reference')} |",
        f"| point rejected by the verifier | {sum(r['verified'] == '0' for r in rows)} |",
        "",
        "Matched means optimal and |ours - ref| / max(1, |ref|) within 1e-6 for a continuous "
        "model, and within 1e-4 (`kMipRelativeGap`, the gap branch and bound stops at) for a "
        "model with integer columns. QPLIB's value is a best known point (`qplib.solu`), not "
        "always a proven optimum.",
        "",
        "**Refusals, by reason.**", "",
        "| reason | instances |", "|---|---:|",
    ]
    reasons = Counter(r["refusal"] for r in rows if r["refusal"])
    out += [f"| {REFUSALS.get(k, k)} | {n} |" for k, n in sorted(reasons.items())]
    if not reasons:
        out.append("| none | 0 |")
    out += ["", "**By engine** (the dispatcher's `algorithm`, for every model it read).", "",
            "| engine | instances | refused | solved | matched | verified |",
            "|---|---:|---:|---:|---:|---:|"]
    engines: dict[str, list[dict]] = {}
    for row in rows:
        if _yes(row, "read_ok"):
            engines.setdefault(row["engine"] or "(none)", []).append(row)
    for engine, mine in sorted(engines.items()):
        out.append(f"| {engine} | {len(mine)} | {sum(bool(r['refusal']) for r in mine)} | "
                   f"{sum(_yes(r, 'solved') for r in mine)} | "
                   f"{sum(_yes(r, 'matches_reference') for r in mine)} | "
                   f"{sum(_yes(r, 'verified') for r in mine)} |")
    out += ["", "**By QPLIB class** (doc.html's O V C letters).", "",
            "| class | instances | read | refused | solved | matched |",
            "|---|---:|---:|---:|---:|---:|"]
    classes: dict[str, list[dict]] = {}
    for row in rows:
        classes.setdefault(row["problem_type"], []).append(row)
    for code, mine in sorted(classes.items()):
        out.append(f"| {code} | {len(mine)} | {sum(_yes(r, 'read_ok') for r in mine)} | "
                   f"{sum(bool(r['refusal']) for r in mine)} | "
                   f"{sum(_yes(r, 'solved') for r in mine)} | "
                   f"{sum(_yes(r, 'matches_reference') for r in mine)} |")
    beats = [r for r in rows if _yes(r, "beats_reference")]
    rejected = [r for r in rows if r["verified"] == "0"]
    broken = [r for r in rows if r["refusal"] in ("reader_error", "crashed", "hung")]
    wrong = [r for r in rows if _yes(r, "solved") and r["matches_reference"] == "0"]
    out += ["", "**Named, because each is a bug until shown otherwise.**", ""]
    out.append(f"- Better than the reference: {_names(beats) or 'none'}.")
    out.append(f"- Point rejected by the verifier: {_names(rejected) or 'none'}.")
    out.append(f"- Reader error, crash or hang: {_names(broken) or 'none'}.")
    out.append(f"- Optimal but not matching the reference: {_names(wrong) or 'none'}.")
    out += ["", "<details><summary>Every instance</summary>", "",
            "| instance | class | read | engine | refusal | status | solved | ours | "
            "reference | rel gap | matched | verified | time (s) |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---|---|---:|"]
    for r in rows:
        out.append(f"| {r['instance']} | {r['problem_type']} | {_mark(r, 'read_ok')} | "
                   f"{_cell(r['engine'])} | {_cell(r['refusal'])} | {r['status']} | "
                   f"{_mark(r, 'solved')} | {_cell(r['our_objective'])} | "
                   f"{_cell(r['reference_objective'])} | {_cell(r['rel_gap'])} | "
                   f"{_mark(r, 'matches_reference')} | {_mark(r, 'verified')} | "
                   f"{float(r['wall_seconds']):.2f} |")
    out += ["", "</details>", "", "Reproduce with", ""] + REPRODUCE
    return "\n".join(out) + "\n"
