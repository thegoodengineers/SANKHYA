#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The dual pricing A/B of docs/BENCHMARKS.md: Devex against dual steepest edge (#411).

Kept out of make_benchmarks_doc.py for the reason qplib_doc.py is: that file is already
several times the size guideline. Every number is read from the A/B's own CSVs, all at one
commit, and every one of them ran with `pricing=` set in its solver_options column, so none
of them is ever picked up as a tier's default evidence:

    netlib-medium-<sha>-devex-r<k>.csv, netlib-medium-<sha>-dse-r<k>.csv   repeats, alternated
    netlib-full-<sha>-dse.csv, against the default run at <sha>          (netlib-full-<sha>.csv
                                                                          or certified-gap-...)
    scale-<sha>-devex.csv, scale-<sha>-dse.csv                            the four scale sizes

Iterations are deterministic at one thread, so they are read from the first repeat; seconds
are the median of the repeats. The full set and the scale sizes ran once per rule.
"""
from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

import latest_result

RULES = (("devex", "pricing=devex"), ("dse", "pricing=dual-steepest-edge"))
_MEDIUM = re.compile(r"^netlib-medium-([0-9a-f]{7,40})-(devex|dse)-r(\d+)\.csv$")

REPRODUCE = [
    "```",
    "python bench/runners/fetch_data.py --set medium --offline-fallback",
    "python bench/runners/netlib.py --solver-option pricing=devex "
    "--out bench/results/netlib-medium-<sha>-devex-r1.csv",
    "python bench/runners/netlib.py --solver-option pricing=dual-steepest-edge "
    "--out bench/results/netlib-medium-<sha>-dse-r1.csv",
    "#   ... alternated, three repeats each",
    "python bench/runners/fetch_data.py --set full --offline-fallback",
    "python bench/runners/netlib.py --solver-option pricing=dual-steepest-edge "
    "--out bench/results/netlib-full-<sha>-dse.csv",
    "python bench/runners/scale.py --engines dual-simplex --solver-option pricing=devex "
    "--out bench/results/scale-<sha>-devex.csv",
    "python bench/runners/scale.py --engines dual-simplex "
    "--solver-option pricing=dual-steepest-edge --out bench/results/scale-<sha>-dse.csv",
    "```",
]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _sgm(values: list[float], shift: float = 1.0) -> float:
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def _certified(row: dict) -> bool:
    gap = _num(row, "certified_relative_gap")
    return row.get("status") == "optimal" and gap is not None and abs(gap) <= 1e-6


def _names(names) -> str:
    return ", ".join(f"`{n}`" for n in sorted(names)) or "none"


def newest_commit(results: Path) -> str | None:
    """The newest commit, in git order, with medium-tier repeats under both rules."""
    found: dict[str, set[str]] = {}
    for path in results.glob("netlib-medium-*-r*.csv"):
        match = _MEDIUM.match(path.name)
        if match:
            found.setdefault(match.group(1), set()).add(match.group(2))
    complete = [sha for sha, rules in found.items() if rules == {"devex", "dse"}]
    if not complete:
        return None
    order = latest_result.commit_order()

    def position(sha: str) -> int:
        return next((i for i, full in enumerate(order) if full.startswith(sha)), len(order))

    return sorted(complete, key=lambda sha: (position(sha), sha))[0]


def _medium(results: Path, sha: str) -> list[str]:
    runs: dict[str, list[dict[str, dict]]] = {"devex": [], "dse": []}
    for path in sorted(results.glob(f"netlib-medium-{sha}-*-r*.csv")):
        match = _MEDIUM.match(path.name)
        if match:
            runs[match.group(2)].append({r["instance"]: r for r in _read(path)})
    names = sorted(set.intersection(*(set(run) for rule in runs.values() for run in rule)))
    both = [n for n in names
            if all(run[n].get("passed") == "1" for rule in runs.values() for run in rule)]
    if not both:
        return ["No instance passed under both rules in every repeat; nothing to compare."]

    def iters(rule: str, name: str) -> int:
        return int(float(runs[rule][0][name]["iterations"]))

    def seconds(rule: str, name: str) -> float:
        return statistics.median(_num(run[name], "solver_seconds") or 0.0 for run in runs[rule])

    out = [f"**Netlib medium tier**, {len(runs['devex'])} repeats under Devex and "
           f"{len(runs['dse'])} under dual steepest edge, alternated, one thread.", "",
           "| rule | passed (every repeat) | not passed | iterations over the "
           f"{len(both)} passed by both | shifted geomean seconds (1 s) | total seconds |",
           "|---|---:|---|---:|---:|---:|"]
    for rule, label in (("devex", "Devex"), ("dse", "dual steepest edge")):
        passed = [n for n in names if all(run[n].get("passed") == "1" for run in runs[rule])]
        secs = [seconds(rule, n) for n in both]
        out.append(f"| {label} | {len(passed)} of {len(names)} | "
                   f"{_names(set(names) - set(passed))} | {sum(iters(rule, n) for n in both):,} | "
                   f"{_sgm(secs):.4f} | {sum(secs):.3f} |")
    ratios = [iters("dse", n) / iters("devex", n) for n in both if iters("devex", n) > 0]
    fewer = sum(r < 1 for r in ratios)
    more = sum(r > 1 for r in ratios)
    out += ["", f"Per instance, dual steepest edge took fewer iterations on {fewer}, more on "
            f"{more}, the same on {len(ratios) - fewer - more}. The eight slowest under Devex:",
            "", "| instance | Devex iterations | DSE iterations | Devex s | DSE s |",
            "|---|---:|---:|---:|---:|"]
    for n in sorted(both, key=lambda n: -seconds("devex", n))[:8]:
        out.append(f"| `{n}` | {iters('devex', n):,} | {iters('dse', n):,} | "
                   f"{seconds('devex', n):.3f} | {seconds('dse', n):.3f} |")
    return out


def _default_full(results: Path, sha: str) -> Path | None:
    for name in (f"netlib-full-{sha}.csv", f"certified-gap-netlib-full-{sha}.csv"):
        path = results / name
        if path.exists():
            options = (latest_result.first_row(path).get("solver_options") or "").split()
            if not options or options == ["pricing=devex"]:
                return path
    return None


def _full(results: Path, sha: str) -> list[str]:
    dse_path = results / f"netlib-full-{sha}-dse.csv"
    devex_path = _default_full(results, sha)
    if not dse_path.exists() or devex_path is None:
        return [f"**Netlib full set**: no pair of runs at `{sha}`, so no claim."]
    devex = {r["instance"]: r for r in _read(devex_path)}
    dse = {r["instance"]: r for r in _read(dse_path)}
    names = sorted(set(devex) & set(dse))
    common = [n for n in names
              if devex[n].get("status") == "optimal" and dse[n].get("status") == "optimal"]
    out = [f"**Netlib full set**, once per rule: Devex from `{devex_path.name}` (the default "
           f"run at this commit), dual steepest edge from `{dse_path.name}`.", "",
           f"| rule | passed | optimal | certified to 1e-6 | not optimal | iterations over "
           f"the {len(common)} optimal under both | shifted geomean seconds (1 s) |",
           "|---|---:|---:|---:|---|---:|---:|"]
    for label, rows in (("Devex", devex), ("dual steepest edge", dse)):
        not_optimal = [f"`{n}` ({rows[n].get('status')})" for n in names
                       if rows[n].get("status") != "optimal"]
        iters = sum(int(float(rows[n]["iterations"])) for n in common)
        secs = [_num(rows[n], "solver_seconds") or 0.0 for n in common]
        out.append(f"| {label} | {sum(rows[n].get('passed') == '1' for n in names)} of "
                   f"{len(names)} | {sum(rows[n].get('status') == 'optimal' for n in names)} | "
                   f"{sum(_certified(rows[n]) for n in names)} | "
                   f"{', '.join(not_optimal) or 'none'} | {iters:,} | {_sgm(secs):.4f} |")
    out += ["",
            "- passed under Devex only: " + _names(
                n for n in names if devex[n].get("passed") == "1" and dse[n].get("passed") != "1"),
            "- passed under dual steepest edge only: " + _names(
                n for n in names if dse[n].get("passed") == "1" and devex[n].get("passed") != "1"),
            "- certified under Devex only: " + _names(
                n for n in names if _certified(devex[n]) and not _certified(dse[n])),
            "- certified under dual steepest edge only: " + _names(
                n for n in names if _certified(dse[n]) and not _certified(devex[n]))]
    return out


def _scale(results: Path, sha: str) -> list[str]:
    paths = {rule: results / f"scale-{sha}-{rule}.csv" for rule, _ in RULES}
    if not all(p.exists() for p in paths.values()):
        return [f"**Scale sizes**: no pair of runs at `{sha}`, so no claim."]
    rows = {rule: {int(r["rows"]): r for r in _read(p)} for rule, p in paths.items()}
    first = next(iter(rows["devex"].values()), {})
    out = [f"**The four scale sizes** (`scale.py`, random structure, dual simplex, "
           f"{_num(first, 'time_limit') or 0:g} s per solve, once per rule):", "",
           "| rows x cols | rule | status | relative error | iterations | seconds |",
           "|---:|---|---|---:|---:|---:|"]
    for size in sorted(set(rows["devex"]) & set(rows["dse"])):
        for rule, label in (("devex", "Devex"), ("dse", "dual steepest edge")):
            r = rows[rule][size]
            err = _num(r, "relative_error")
            out.append(f"| {size:,} | {label} | {r.get('status', '')} | "
                       f"{'-' if err is None else f'{err:.1e}'} | "
                       f"{int(float(r.get('iterations') or 0)):,} | "
                       f"{_num(r, 'solver_seconds') or 0:.1f} |")
    return out


def section(results: Path) -> str:
    sha = newest_commit(results)
    if sha is None:
        return ("Not yet measured: no `netlib-medium-<sha>-devex-r<k>.csv` and "
                "`-dse-r<k>.csv` pair in `bench/results/`.\n")
    machine = latest_result.first_row(
        sorted(results.glob(f"netlib-medium-{sha}-devex-r*.csv"))[0]).get("machine", "")
    out = [f"All at commit `{sha}` on `{machine}`, nothing else running.", ""]
    out += _medium(results, sha) + [""] + _full(results, sha) + [""] + _scale(results, sha)
    out += ["", "To reproduce:", ""] + REPRODUCE
    return "\n".join(out) + "\n"
