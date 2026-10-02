#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The MIPLIB 3 section of docs/BENCHMARKS.md (#761), rendered from its CSVs.

Kept out of make_benchmarks_doc.py, which calls section() below with the newest CSV of each
kind at each limit. The classic set sits beside the MIPLIB 2017 tables and is never merged
into them: its CSVs have names of their own (`miplib3-<limit>s-seeds<N>-<sha>.csv` from
`miplib.py --set miplib3`, `highs-miplib3-<limit>s-<sha>.csv` from `compare.py --suite
miplib3`) that no MIPLIB 2017 glob matches, and its table is its own.

Per instance and limit: the seeds that reached the published optimum, the seeds that proved
it, the runs whose point tools/verify_solution.py accepted, and HiGHS's result on the same
file. Every number is counted from the rows; every run that fails a check is NAMED, and the
one #761 forbids outright, a run reported optimal whose point the verifier rejects, gets a
line of its own even when there is none.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import miplib_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "miplib3" / "manifest.json"
LIMITS = (60, 300)
SEEDS = 3


def sankhya_pattern(limit: int) -> str:
    """The glob for miplib.py's CSV at `limit`. `60s` cannot match `600s`: the `s` is literal."""
    return f"miplib3-{limit}s-seeds*-*.csv"


def highs_pattern(limit: int) -> str:
    """compare_suite.py's HiGHS CSV at `limit`; `[0-9a-f]` leaves out a `-partial-` run."""
    return f"highs-miplib3-{limit}s-[0-9a-f]*.csv"


def commands(limit: int) -> list[str]:
    return [f"python bench/runners/miplib.py --set miplib3 --seeds {SEEDS} --time-limit {limit}",
            f"python bench/runners/compare.py --suite miplib3 --time-limit {limit}"]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def _names(names) -> str:
    return ", ".join(f"`{n}`" for n in names) or "none"


def load_manifest(path: Path = MANIFEST) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def references(manifest: dict) -> list[str]:
    """Where every reference comes from, read from the manifest: the count, each exclusion
    by reason, each override of the catalogue and each integer print a proven value confirms."""
    instances, excluded = manifest.get("instances", {}), manifest.get("excluded", {})
    groups: dict[str, list[str]] = {}
    for name, entry in sorted(excluded.items()):
        reason = entry.get("reason", "")
        key = ("marked \"(not opt)\"" if "(not opt)" in reason
               else "no catalogue row" if reason.startswith("no row")
               else "printed as an integer while the objective can be fractional, with no "
                    "proven value with more digits to confirm it")
        groups.setdefault(key, []).append(name)
    overrides = {n: e for n, e in instances.items() if e.get("reference_source") == "override"}
    confirmed = {n: e for n, e in instances.items()
                 if e.get("reference_source") != "override" and e.get("reference_note")}
    lines = [
        f"The archive holds {manifest.get('models_in_archive', '?')} models; **{len(instances)}** "
        f"carry a reference and are the instances below. Excluded, with no reference: "
        + "; ".join(f"{_names(names)} ({key})" for key, names in groups.items()) + ".",
        "",
    ]
    if overrides:
        lines += ["**References that override the catalogue** (`reference_source: override` in "
                  "the manifest), each a later PROVEN optimum of the same model: "
                  + "; ".join(f"`{n}` {e.get('catalogue_int_soln', '?')} -> "
                              f"{e['published_text']}: {e['reference_note']}"
                              for n, e in sorted(overrides.items())) + ".", ""]
    if confirmed:
        lines += ["**Integer prints kept because a proven optimum confirms them**: "
                  + "; ".join(f"`{n}` {e['published_text']}: {e['reference_note']}"
                              for n, e in sorted(confirmed.items())) + ".", ""]
    return lines


def intro(manifest: dict | None) -> str:
    lines = [
        "The 1998 set much of the published record and most teaching material still report "
        "on, run beside the MIPLIB 2017 tables above and never merged into them: its own "
        "fetcher, manifest, CSVs and table. Source: Bixby, Ceria, McZeal and Savelsbergh, "
        "*An updated mixed integer programming library: MIPLIB 3.0*, Optima 58 (1998); the "
        "archive `https://miplib2010.zib.de/miplib3/miplib3.tar.gz`, pinned by size and "
        "sha256 in `bench/runners/fetch_miplib3.py`, whose manifest "
        "(`data/miplib3/manifest.json`) records the sha256 of every model as solved.",
        "",
    ]
    if manifest:
        lines += references(manifest)
    lines += [
        "**Matching rule.** The catalogue prints its optima to limited precision, and one is "
        "truncated: `rgn` is printed 82.1999 and its optimum is 82.19999924. A run REACHES "
        "the published optimum when its objective is within max(1e-6 x max(1, |published|), "
        "one unit in the last printed decimal place after trailing zeros are stripped) of "
        "it, so 82.1999 allows 1e-4 and a value printed as an integer (21166.000, 1201500) "
        "gets the 1e-6 rule alone; that is why an integer print under 1e6 of an objective "
        "that can be fractional is kept only when a proven value confirms it. PROVED means "
        "the run also reported optimal, its gap target (1e-4 relative) met; VERIFIED means "
        "`tools/verify_solution.py` accepted the point as feasible and integral. HiGHS runs "
        "as a separate process on the same file, same machine, same limit, one thread, and "
        "its point is converted by "
        "`bench/runners/rivals.py`, checked by the same verifier and graded by the same rule; "
        "its result counts only an optimal status (its default gap target is also 1e-4 "
        "relative).",
        "",
    ]
    return "\n".join(lines)


def _per_seed(rows: list[dict]) -> list[dict]:
    return [{
        "instance": r["instance"], "seed": int(r.get("seed") or 0),
        "matched": r.get("matched_published") == "1", "proved": r.get("proved_optimal") == "1",
        "seconds": _float(r.get("wall_seconds")) or 0.0,
        "time_to_first_feasible": _float(r.get("time_to_first_feasible")),
        "primal_integral": _float(r.get("primal_integral")),
    } for r in rows]


def _highs_cell(row: dict | None) -> str:
    if row is None:
        return "not run"
    status = row.get("status") or "unknown"
    seconds = _float(row.get("solver_seconds"))
    if status != "optimal":
        return f"{status}" + (f", {seconds:.1f} s" if seconds is not None else "")
    parts = ["optimal",
             "matched" if row.get("matches_reference") == "1" else "NOT matched",
             {"1": "verified", "0": "REJECTED"}.get(row.get("independently_verified", ""),
                                                    "not verified")]
    return ", ".join(parts) + (f", {seconds:.1f} s" if seconds is not None else "")


def limit_section(limit: int, sankhya: Path | None, highs: Path | None,
                  manifest: dict | None) -> str:
    """One limit's table, or "Not yet run" with the exact commands."""
    if sankhya is None and highs is None:
        return "\n".join(["Not yet run. Reproduce with:", "", "```",
                          "python bench/runners/fetch_miplib3.py", *commands(limit), "```", ""])
    reference = (manifest or {}).get("instances", {})
    rows = _read(sankhya) if sankhya else []
    highs_rows = {r["instance"]: r for r in _read(highs)} if highs else {}
    out: list[str] = []
    if sankhya:
        first = rows[0] if rows else {}
        out.append(f"SANKHYA: `bench/results/{sankhya.name}`, commit "
                   f"`{first.get('git_commit', '?')}`, machine `{first.get('machine', '?')}`, "
                   f"{first.get('threads', '?')} thread(s), shipped defaults.  ")
    else:
        out.append(f"SANKHYA: not yet run (`{commands(limit)[0]}`).  ")
    if highs:
        first = next(iter(highs_rows.values()), {})
        out.append(f"HiGHS: `bench/results/{highs.name}`, HiGHS "
                   f"{first.get('solver_version', '?')}, machine `{first.get('machine', '?')}`, "
                   f"one thread, {first.get('time_limit', '?')} s.")
        if sankhya and rows and first.get("machine") != rows[0].get("machine"):
            out.append("")
            out.append("**The two CSVs name different machines**, so their times are not "
                       "comparable; the verdicts still are.")
    else:
        out.append(f"HiGHS: not yet run (`{commands(limit)[1]}`).")
    out.append("")

    per = _per_seed(rows)
    summary = {s["instance"]: s for s in miplib_seeds.aggregate(per, float(limit))}
    by_instance: dict[str, list[dict]] = {}
    for r in rows:
        by_instance.setdefault(r["instance"], []).append(r)
    names = sorted(set(reference) | set(by_instance) | set(highs_rows))

    if rows:
        total = miplib_seeds.overall(per, float(limit))
        claimed = [r for r in rows if r.get("independently_verified", "") != ""]
        accepted = sum(1 for r in claimed if r["independently_verified"] == "1")
        seeds = max((s["seeds"] for s in summary.values()), default=0)
        every_reached = sum(1 for s in summary.values() if s["seeds_matched"] == s["seeds"])
        every_proved = sum(1 for s in summary.values() if s["seeds_proved"] == s["seeds"])
        out.append(f"**SANKHYA**, {total['runs']} runs ({len(summary)} instances x {seeds} "
                   f"seeds): **{total['matched']}** reached the published optimum, "
                   f"**{total['proved']}** proved it, **{accepted} of {len(claimed)}** points "
                   f"verified; reached in every seed on {every_reached} of {len(summary)} "
                   f"instances, proved in every seed on {every_proved}; shifted geometric mean "
                   f"time {_fmt(total['sgm_seconds'])} s (shift 10 s, every run that did not "
                   f"prove charged {limit} s).")
    if highs_rows:
        optimal = [r for r in highs_rows.values() if r.get("status") == "optimal"]
        matched = [r for r in optimal if r.get("matches_reference") == "1"]
        verified = [r for r in optimal if r.get("independently_verified") == "1"]
        out.append(f"**HiGHS**, {len(highs_rows)} runs: **{len(optimal)}** optimal, "
                   f"**{len(matched)}** of them at the published optimum, **{len(verified)}** "
                   f"verified.")
    out += ["",
            "| instance | published | reached (seeds) | proved (seeds) | verified | "
            "sgm time (s) | HiGHS |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for n in names:
        s = summary.get(n)
        mine = by_instance.get(n, [])
        published = reference.get(n, {}).get("published_text") or (
            mine[0].get("published_objective") if mine else "?")
        if s is None:
            cells = ["no run", "-", "-", "-"]
        else:
            claimed = [r for r in mine if r.get("independently_verified", "") != ""]
            ok = sum(1 for r in claimed if r["independently_verified"] == "1")
            seeds_text = lambda ids: f" ({' '.join(map(str, ids))})" if ids else ""  # noqa: E731
            cells = [f"{s['seeds_matched']}/{s['seeds']}{seeds_text(s['matched_seeds'])}",
                     f"{s['seeds_proved']}/{s['seeds']}{seeds_text(s['proved_seeds'])}",
                     f"{ok}/{len(claimed)}" if claimed else "-",
                     _fmt(s["sgm_seconds"])]
        out.append(f"| `{n}` | {published} | " + " | ".join(cells) + f" | "
                   f"{_highs_cell(highs_rows.get(n)) if highs else 'not run'} |")

    out.append("")
    if rows:
        missed = [f"`{r['instance']}` seed {r.get('seed', 0)} ({r['status']})" for r in rows
                  if r.get("matched_published") != "1"]
        rejected = [f"`{r['instance']}` seed {r.get('seed', 0)}" for r in rows
                    if r.get("status") == "optimal" and r.get("independently_verified") == "0"]
        wrong = [f"`{r['instance']}` seed {r.get('seed', 0)}" for r in rows
                 if r.get("status") == "optimal" and r.get("matched_published") != "1"]
        absent = sorted(set(reference) - set(by_instance))
        out += [f"SANKHYA runs that did not reach the published optimum, named: "
                f"{', '.join(missed) or 'none'}.",
                "",
                f"Reported optimal but rejected by the verifier: {', '.join(rejected) or 'none'}."
                f" Reported optimal away from the published optimum: "
                f"{', '.join(wrong) or 'none'}.",
                ""]
        if absent:
            out += [f"Instances with a reference but no SANKHYA run in the CSV: "
                    f"{_names(absent)}.", ""]
    if highs_rows:
        failed = [f"`{n}` ({_highs_cell(r)})" for n, r in sorted(highs_rows.items())
                  if not (r.get("status") == "optimal" and r.get("matches_reference") == "1"
                          and r.get("independently_verified") == "1")]
        out += [f"HiGHS runs that did not end optimal, matched and verified, named: "
                f"{', '.join(failed) or 'none'}.", ""]
    return "\n".join(out)


def section(runs: dict[int, tuple[Path | None, Path | None]],
            manifest: dict | None = None) -> str:
    """The whole section: the intro, then one sub-section per limit in LIMITS. `runs` maps a
    limit to its (SANKHYA CSV, HiGHS CSV), either None when not yet run."""
    out = [intro(manifest)]
    for limit in LIMITS:
        sankhya, highs = runs.get(limit, (None, None))
        out += [f"#### At {limit} s", "", limit_section(limit, sankhya, highs, manifest)]
    return "\n".join(out)
