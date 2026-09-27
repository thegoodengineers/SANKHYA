#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The GPU A/B sections of docs/BENCHMARKS.md, rendered from their CSVs.

Feasibility Jump on the device against the CPU (#508, bench/runners/gpu_fj_ab.py) and the
per-iteration cost of deterministic device reductions on the refinery year (#478,
bench/runners/pdhg_two_matvec_ab.py over one --mps file, one CSV per option leg). Kept out
of make_benchmarks_doc.py for the size reason the other *_doc.py modules are. Every number is
counted from a CSV row; nothing is asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fj_section(path: Path | None) -> str:
    if path is None:
        return "Not yet run at this commit (`python bench/runners/gpu_fj_ab.py --build build`)."
    rows = _rows(path)
    if not rows:
        return "The Feasibility Jump CSV is empty."
    engines = sorted({r["engine"] for r in rows})
    instances = sorted({r["instance"] for r in rows})
    first = rows[0]
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{first.get('git_commit', '?')}` · machine `{first.get('machine', '?')}` · "
        f"GPU {first.get('gpu', '?')} · {first.get('limit_seconds', '?')} s per run, one run "
        f"per instance and engine, every point checked against the model (`verified`)",
        "",
    ]
    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine]
        at10 = sum(r.get("found_at_10s") == "1" for r in mine)
        atlim = sum(r.get("found_at_limit") == "1" for r in mine)
        out.append(f"- `{engine}`: a feasible point on **{atlim} of {len(mine)}** instances by the "
                   f"limit, {at10} of {len(mine)} by 10 s.")
    out += ["", "| instance | " + " | ".join(f"{e} first (s) | {e} rel. gap" for e in engines) + " |",
            "|---|" + "---:|---:|" * len(engines)]
    by = {(r["instance"], r["engine"]): r for r in rows}
    for name in instances:
        cells = []
        for engine in engines:
            r = by.get((name, engine), {})
            found = r.get("found_at_limit") == "1"
            cells.append(r.get("first_seconds", "-") if found else "none")
            cells.append(r.get("rel_gap", "-") if found else "-")
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def refinery_478_section(paths: dict[str, Path | None]) -> str:
    """paths: leg label -> CSV (None when that leg was not run)."""
    present = {label: p for label, p in paths.items() if p is not None}
    if not present:
        return "Not yet run at this commit."
    out = ["| leg | solver options | three products (us/iter) | two products (us/iter) | "
           "two / three | source |",
           "|---|---|---:|---:|---:|---|"]
    header = None
    for label, path in present.items():
        rows = _rows(path)
        three = next((r for r in rows if r.get("two_matvec") == "false"), None)
        two = next((r for r in rows if r.get("two_matvec") == "true"), None)
        if three is None or two is None:
            continue
        header = header or three
        t3 = float(three["seconds_per_iteration"]) * 1e6
        t2 = float(two["seconds_per_iteration"]) * 1e6
        out.append(f"| {label} | `{three.get('solver_options') or 'defaults'}` | {t3:.1f} | "
                   f"{t2:.1f} | {t2 / t3:.3f} | `{path.name}` |")
    if header is None:
        return "The refinery-year CSVs carry no complete pair."
    out = [f"Instance `{header['instance']}` ({header['rows']} rows, {header['cols']} columns, "
           f"{header['nnz']} nonzeros), engine `{header['engine']}`, "
           f"{header['iterations_requested']} iterations per solve, median of "
           f"{header['repeats']} repeats; the per-iteration figure is the marginal time "
           f"between the full and the one-fifth run, as `pdhg_two_matvec_ab.py` defines it. "
           f"Commit `{header.get('git_commit', '?')}`, GPU {header.get('gpu', '?')}.", ""] + out
    return "\n".join(out) + "\n"


def _num(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _closer(a: float | None, b: float | None, target: float) -> int:
    """+1 when a is closer to target than b, -1 when further, 0 when level (1e-6 relative)."""
    tol = 1e-6 * max(1.0, abs(target))
    if a is not None and (b is None or abs(a - target) < abs(b - target) - tol):
        return 1
    if b is not None and (a is None or abs(a - target) > abs(b - target) + tol):
        return -1
    return 0


def tier2_legs_section(legs: dict[str, Path | None]) -> str:
    """The #509 / #520 legs on the MIPLIB tier-2 set, each against the first (baseline) leg.

    Per leg: feasible points found, published optimum matched and proved, node throughput
    (total nodes over total solver seconds), and per instance against the baseline whether
    the incumbent (primal) and the final dual bound are closer to or further from the
    published optimum. All counted from the rows.
    """
    present = {label: p for label, p in legs.items() if p is not None}
    if len(present) < 2:
        return "Not yet run at this commit."
    labels = list(present)
    base_label = labels[0]
    tables = {label: {r["instance"]: r for r in _rows(p)} for label, p in present.items()}
    base = tables[base_label]
    first = next(iter(base.values()))
    out = [f"Commit `{first.get('git_commit', '?')}` · machine `{first.get('machine', '?')}` · "
           f"{len(base)} instances, one seed, one thread; every leg against `{base_label}`, "
           f"the same binary with every GPU option off.", "",
           "| leg | solver options | feasible | matched | proved | nodes / s | primal closer / "
           "further | dual bound closer / further | source |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for label in labels:
        rows = tables[label]
        feasible = sum(_num(r, "our_objective") is not None for r in rows.values())
        matched = sum(r.get("matched_published") == "1" for r in rows.values())
        proved = sum(r.get("proved_optimal") == "1" for r in rows.values())
        nodes = sum(_num(r, "nodes") or 0.0 for r in rows.values())
        secs = sum(_num(r, "solver_seconds") or 0.0 for r in rows.values())
        primal = [0, 0]
        dual = [0, 0]
        if label != base_label:
            for name, r in rows.items():
                b = base.get(name)
                pub = _num(r, "published_objective")
                if b is None or pub is None:
                    continue
                p = _closer(_num(r, "our_objective"), _num(b, "our_objective"), pub)
                d = _closer(_num(r, "dual_bound"), _num(b, "dual_bound"), pub)
                primal[0] += p > 0
                primal[1] += p < 0
                dual[0] += d > 0
                dual[1] += d < 0
        opts = next(iter(rows.values())).get("solver_options") or "defaults"
        pc = "-" if label == base_label else f"{primal[0]} / {primal[1]}"
        dc = "-" if label == base_label else f"{dual[0]} / {dual[1]}"
        rate = f"{nodes / secs:.0f}" if secs else "-"
        out.append(f"| {label} | `{opts}` | {feasible} | {matched} | {proved} | {rate} | {pc} "
                   f"| {dc} | `{present[label].name}` |")
    return "\n".join(out) + "\n"
