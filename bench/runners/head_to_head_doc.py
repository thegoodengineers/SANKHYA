#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""docs/BENCHMARKS.md section 4a: the head-to-head of compare.py --suite (#766).

Reads bench/results/head-to-head-<suite>-<commit>.csv for each suite and renders, per
solver: runs, solved (optimal status), matched (the point's objective at the published
optimum), verified (tools/verify_solution.py accepted it, and on which conditions), and the
shifted geometric mean of the solver's own time with a 10 s shift, a run that did not count
charged the full time limit (Mittelmann's convention). Every failure is named, and each
suite gets a Dolan-More profile drawn by perf_profile.py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_profile  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIFT_SECONDS = 10.0  # Mittelmann's shift for these tables
ORDER = ("sankhya", "highs", "scip", "cbc-clp", "glpk")
TITLES = {"netlib": "Netlib LP", "kennington": "Kennington LP",
          "maros-meszaros": "Maros-Meszaros QP"}
# Netlib's two models that netlib.org ships as Fortran generators, not EMPS files (#745):
# the headline row is the 92 EMPS files, and these two are reported beside it.
GENERATED_NETLIB = ("truss", "stocfor3")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count(rows: list[dict], key: str) -> int:
    return sum(r.get(key) == "1" for r in rows)


def solver_table(rows: list[dict], sgm) -> list[str]:
    time_limit = float(rows[0]["time_limit"])
    exact = any(r.get("matches_exact", "") != "" for r in rows)
    out = ["| solver | version | runs | solved | matched | "
           + ("matched exact | " if exact else "") + "verified | checked on | "
           f"SGM time, shift {SHIFT_SECONDS:g} s |",
           "|---|---|---:|---:|---:|" + ("---:|" if exact else "") + "---:|---|---:|"]
    for solver in ORDER:
        mine = [r for r in rows if r["solver"] == solver]
        if not mine:
            continue
        if all(r["status"] == "unsupported" for r in mine):
            out.append(f"| {perf_profile.LABELS[solver]} | {mine[0]['solver_version'] or '-'} "
                       f"| {len(mine)} | unsupported | - | " + ("- | " if exact else "")
                       + "- | - | - |")
            continue
        charged = [float(r["solver_seconds"]) if r["counted_for_time"] == "1" else time_limit
                   for r in mine]
        kinds = sorted({r["verification"] for r in mine
                        if r["verification"] in ("primal+dual", "primal-only")})
        version = next((r["solver_version"] for r in mine if r["solver_version"]), "-")
        out.append(f"| {perf_profile.LABELS[solver]} | {version} | {len(mine)} "
                   f"| {sum(r['status'] == 'optimal' for r in mine)} "
                   f"| {_count(mine, 'matches_reference')} "
                   + (f"| {_count(mine, 'matches_exact')} " if exact else "")
                   + f"| {_count(mine, 'independently_verified')} | {', '.join(kinds) or '-'} "
                   f"| {sgm(charged, SHIFT_SECONDS):.3f} s |")
    return out


def failures(rows: list[dict]) -> list[str]:
    """Every run that did not count, named with what happened, per solver."""
    out = []
    for solver in ORDER:
        bad = [r for r in rows if r["solver"] == solver and r["counted_for_time"] != "1"
               and r["status"] != "unsupported"]
        if not bad:
            continue
        items = []
        for r in sorted(bad, key=lambda r: r["instance"]):
            if r["status"] != "optimal":
                what = r["status"]
            elif r["independently_verified"] == "0":
                what = "optimal, rejected by the verifier"
            elif r.get("matches_exact") == "0":
                what = "optimal, away from the exact optimum"
            else:
                what = f"optimal at relative gap {float(r['relative_gap'] or 'nan'):.1e}"
            items.append(f"`{r['instance']}` ({what})")
        out.append(f"- **{perf_profile.LABELS[solver]}**, {len(bad)}: " + ", ".join(items))
    return out


def rejections(rows: list[dict]) -> list[str]:
    """Why the verifier said no, one line per rejected run."""
    out = []
    for r in sorted(rows, key=lambda r: (ORDER.index(r["solver"]), r["instance"])):
        if r["independently_verified"] == "0":
            out.append(f"- {perf_profile.LABELS[r['solver']]} on `{r['instance']}`: "
                       f"{r['verifier_message'][:220]}")
    return out


def suite_section(suite: str, path: Path, sgm) -> str:
    rows = read_csv(path)
    if not rows:
        return f"`{path.name}` is empty.\n"
    first = rows[0]
    instances = sorted({r["instance"] for r in rows})
    out = [f"Source CSV: `bench/results/{path.name}`  ",
           f"Commit `{first['git_commit']}` · machine `{first['machine']}`  ",
           f"{len(instances)} instances · time limit {float(first['time_limit']):g} s · "
           "1 thread per solver · every solver a separate process", ""]
    headline = rows
    if suite == "netlib":
        headline = [r for r in rows if r["instance"] not in GENERATED_NETLIB]
        shipped = len({r["instance"] for r in headline})
        out += [f"**The {shipped} LPs netlib.org ships as EMPS files:**", ""]
    out += solver_table(headline, sgm)
    out.append("")
    if suite == "netlib":
        extra = [r for r in rows if r["instance"] in GENERATED_NETLIB]
        if extra:
            out += ["**The two generated from netlib.org's Fortran bundles (#745), "
                    "`truss` and `stocfor3`:**", ""]
            out += solver_table(extra, sgm)
            out.append("")
        if any(r["matches_exact"] != "" for r in headline):
            out.append("`matched` grades against the readme's optimum and `matched exact` "
                       "against Koch's exact rational optimum (`data/netlib/koch_exact.json`, "
                       "#747); the timing counts the exact grade. Where the two differ the "
                       "readme is the one that is wrong: on 80bau3b, ganges, greenbea, "
                       "greenbeb, nesm, pilot, pilot.we, scrs8 and stocfor3 every solver here "
                       "agrees with Koch.")
            out.append("")
    solvers = [s for s in ORDER if any(r["solver"] == s and r["status"] != "unsupported"
                                       for r in headline)]
    figure = perf_profile.write_figure(
        suite, f"{TITLES[suite]}: Dolan-More performance profile", headline, solvers)
    if figure is not None:
        out += [f"![Dolan-More performance profile, {TITLES[suite]}]"
                f"(img/{figure.name})", "",
                f"*`docs/img/{figure.name}`, drawn from the CSV by "
                "`bench/runners/perf_profile.py` each time this document is generated. A "
                "curve's height at tau is the fraction of the instances above on which that "
                "solver's run counted and took at most tau times the fastest counted run; "
                f"times under {perf_profile.TIME_FLOOR_SECONDS:g} s are floored there "
                "(GLPK's clock resolution), so the fast end is a tie.*", ""]
    named = failures(headline)
    if named:
        out += ["Every run that did not count, by solver:", "", *named, ""]
    reasons = rejections(rows)
    if reasons:
        out += ["What the verifier rejected:", "", *reasons, ""]
    return "\n".join(out)


def section(paths: dict[str, Path | None], sgm) -> str:
    """The whole of 4a; `sgm(values, shift)` is make_benchmarks_doc's shifted geometric
    mean, passed in so there is one implementation of it."""
    out = []
    for number, suite in enumerate(("netlib", "kennington", "maros-meszaros"), start=1):
        path = paths.get(suite)
        out.append(f"### 4a.{number} {TITLES[suite]}\n")
        if path is None:
            out.append(f"No `head-to-head-{suite}-*.csv` in `bench/results/`, so **no numbers "
                       f"are stated for this suite**. Run `python bench/runners/compare.py "
                       f"--suite {suite}`.\n")
            continue
        out.append(suite_section(suite, path, sgm))
    return "\n".join(out)
