#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MIPLIB A/B and multi-seed sections of docs/BENCHMARKS.md, rendered from their CSVs.

Kept out of make_benchmarks_doc.py, which calls the functions below. Every number is
counted from the rows of a CSV in bench/results/; every instance that fails a check is NAMED.

  node_rate_ab   #501: the node LP factor cache, deterministic node-limited legs, where the
                 node counts must be identical and only the time may differ.
  seeds_ab       #506 (and any on/off A/B on the #504 harness): per-seed CSVs of two legs,
                 matched and proved over seeds, shifted geometric means, time to first
                 feasible and the primal integral (miplib_seeds.py).
  plateau_ab     #418: seeds_ab, then the dual bound of the four plateau instances seed by
                 seed in both legs, and whether it moved, which is #418's (and #221's)
                 acceptance criterion.
  tier2_section  #504: the 60-instance tier over three seeds.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import miplib_seeds


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


def _per_seed(rows: list[dict]) -> list[dict]:
    """The rows miplib_seeds.aggregate() takes, from a per-seed CSV."""
    return [{
        "instance": r["instance"], "seed": int(r.get("seed") or 0),
        "matched": r.get("matched_published") == "1", "proved": r.get("proved_optimal") == "1",
        "seconds": _float(r.get("wall_seconds")) or 0.0,
        "time_to_first_feasible": _float(r.get("time_to_first_feasible")),
        "primal_integral": _float(r.get("primal_integral")),
    } for r in rows]


def _limit(path: Path) -> float:
    """The run's time limit, from the summary CSV miplib.py writes beside a multi-seed run."""
    summary = path.with_name("summary-" + path.name)
    rows = _read(summary) if summary.exists() else []
    value = _float(rows[0].get("time_limit")) if rows else None
    if value is None:
        raise ValueError(f"no time limit for {path.name}: {summary.name} is missing")
    return value


def node_rate_ab(off: Path | None, on: Path | None, option: str) -> str:
    """Two deterministic, node-limited legs that differ in `option` only (#501)."""
    if off is None or on is None:
        return ""
    a = {r["instance"]: r for r in _read(off)}
    b = {r["instance"]: r for r in _read(on)}
    names = sorted(set(a) & set(b))
    same_nodes, ratios, lines, differ, proof_changed = [], [], [], [], []
    for n in names:
        na, nb = a[n].get("nodes", ""), b[n].get("nodes", "")
        ta, tb = _float(a[n].get("solver_seconds")), _float(b[n].get("solver_seconds"))
        if na != nb or a[n]["status"] != b[n]["status"] or \
                a[n].get("our_objective") != b[n].get("our_objective"):
            differ.append(n)
        else:
            same_nodes.append(n)
        if a[n].get("proved_optimal") != b[n].get("proved_optimal"):
            proof_changed.append(n)
        ok = na == nb and na not in ("", "0") and ta and tb
        if ok:
            ratios.append(ta / tb)
        rate = lambda nodes, t: _fmt(int(nodes) / t, ".0f") if ok else "-"  # noqa: E731
        lines.append(f"| `{n}` | {a[n]['status']} | {na} | {nb} | {rate(na, ta)} | "
                     f"{rate(nb, tb)} | {_fmt(ta / tb if ok else None)} |")
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    commit = next(iter(a.values()))["git_commit"]
    return "\n".join([
        f"Source CSVs: `bench/results/{off.name}` (`{option}` off) and "
        f"`bench/results/{on.name}` (on), commit `{commit}`, both deterministic and "
        f"node-limited, one thread.",
        "",
        f"**{len(same_nodes)} of {len(names)}** instances end with the same node count, status "
        f"and objective in both legs; differ: {', '.join(f'`{n}`' for n in differ) or 'none'}. "
        f"Proved verdict changed: {', '.join(f'`{n}`' for n in proof_changed) or 'none'}. "
        f"Geometric mean speedup in nodes per second over the {len(ratios)} instances with "
        f"equal, non-zero node counts: **{_fmt(geo, '.3f')}x** (above 1 is faster with the "
        f"option on).",
        "",
        "| instance | status | nodes off | nodes on | nodes/s off | nodes/s on | speedup |",
        "|---|---|---:|---:|---:|---:|---:|",
        *lines,
        "",
    ])


def _leg(path: Path) -> dict:
    rows = _read(path)
    per = _per_seed(rows)
    limit = _limit(path)
    summary = miplib_seeds.aggregate(per, limit)
    total = miplib_seeds.overall(per, limit)
    firsts = [limit if r["time_to_first_feasible"] is None
              else min(r["time_to_first_feasible"], limit) for r in per]
    integrals = [r["primal_integral"] for r in per if r["primal_integral"] is not None]
    return {
        "summary": {s["instance"]: s for s in summary}, "total": total, "limit": limit,
        "seeds": max((s["seeds"] for s in summary), default=0),
        "sgm_first": miplib_seeds.shifted_geometric_mean(
            firsts, miplib_seeds.FIRST_FEASIBLE_SHIFT_SECONDS),
        "found": sum(1 for r in per if r["time_to_first_feasible"] is not None),
        "mean_integral": sum(integrals) / len(integrals) if integrals else None,
    }


def seeds_ab(off: Path | None, on: Path | None, option: str) -> str:
    """Two multi-seed legs on the #504 harness that differ in `option` only."""
    if off is None or on is None:
        return ""
    legs = {"off": _leg(off), "on": _leg(on)}
    out = [
        f"Source CSVs: `bench/results/{off.name}` and `bench/results/{on.name}`, "
        f"`{option}` off and on, {legs['off']['seeds']} seeds each (the published file and "
        f"row and column permutations of it), {legs['off']['limit']:g} s per run, one thread.",
        "",
        "| leg | runs | matched | proved | sgm time (s, shift 10) | first feasible found | "
        "sgm first feasible (s, shift 1) | mean primal integral (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, leg in legs.items():
        t = leg["total"]
        out.append(f"| {name} | {t['runs']} | {t['matched']} | {t['proved']} | "
                   f"{_fmt(t['sgm_seconds'])} | {leg['found']} | {_fmt(leg['sgm_first'], '.3f')}"
                   f" | {_fmt(leg['mean_integral'])} |")
    a, b = legs["off"]["summary"], legs["on"]["summary"]
    moved = []
    for n in sorted(set(a) & set(b)):
        if (a[n]["seeds_matched"], a[n]["seeds_proved"]) != \
                (b[n]["seeds_matched"], b[n]["seeds_proved"]):
            moved.append(f"`{n}` matched {a[n]['seeds_matched']}->{b[n]['seeds_matched']}, "
                         f"proved {a[n]['seeds_proved']}->{b[n]['seeds_proved']}")
    out += ["", "Instances whose matched or proved seed count moved between the legs: "
            + ("; ".join(moved) if moved else "none") + ".", ""]
    return "\n".join(out)


# The four MIPLIB instances that hold the published optimum without proving it, with every
# open node on one bound (#221, #418): more nodes did not move that bound, nor did cuts or
# objective integrality, so a bound that moves here is the evidence #418 asks for.
PLATEAU_INSTANCES = ("b-ball", "opt1217", "rlp1", "noswot")


def _bound_distance(row: dict) -> float | None:
    """How far the reported dual bound sits from the published optimum, relative.

    The bound is on the optimum's far side from every incumbent, whatever the sense, so a
    bound that moved towards the optimum is one whose distance shrank; no sense is needed."""
    bound = _float(row.get("dual_bound"))
    published = _float(row.get("published_objective"))
    if bound is None or published is None:
        return None
    return abs(published - bound) / max(1.0, abs(published))


def plateau_ab(off: Path | None, on: Path | None, option: str,
               instances: tuple[str, ...] = PLATEAU_INSTANCES) -> str:
    """Two multi-seed legs that differ in `option` (#418), read for bound movement.

    The whole-set comparison is seeds_ab's. Below it, every seed of each plateau instance
    in both legs: the dual bound, the nodes, and in the on leg the restarts and the objective
    branches it took, so a leg in which the option never fired is visible as such rather than
    read as a measurement. The legs must carry one commit, clean, or nothing is compared."""
    if off is None or on is None:
        return ""
    rows_off, rows_on = _read(off), _read(on)
    commits = {r.get("git_commit", "") for r in rows_off + rows_on}
    if len(commits) != 1 or any(c.endswith("-dirty") or not c for c in commits):
        return (f"Not compared: `bench/results/{off.name}` and `bench/results/{on.name}` must "
                f"carry one clean commit, and they carry "
                f"{', '.join(f'`{c}`' for c in sorted(commits)) or 'none'}.")
    out = [seeds_ab(off, on, option)]
    a = {(r["instance"], int(r.get("seed") or 0)): r for r in rows_off}
    b = {(r["instance"], int(r.get("seed") or 0)): r for r in rows_on}

    def count(rows: list[dict], column: str) -> int:
        return sum(1 for r in rows if (_float(r.get(column)) or 0) > 0)

    fired = count(rows_on, "restarts")
    branched = count(rows_on, "objective_branches")
    out += [f"Runs in the on leg that restarted at least once: **{fired} of {len(rows_on)}**; "
            f"that split a node on the objective row: **{branched} of {len(rows_on)}**.", "",
            "| instance | seed | status off | status on | bound off | bound on | bound moved "
            "| nodes off | nodes on | restarts on | objective branches on |",
            "|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|"]
    moved, worse, missing = [], [], []
    for name in instances:
        seeds = sorted(s for (n, s) in set(a) & set(b) if n == name)
        if not seeds:
            missing.append(name)
            continue
        for seed in seeds:
            x, y = a[(name, seed)], b[(name, seed)]
            d0, d1 = _bound_distance(x), _bound_distance(y)
            scale = 1e-9
            if d0 is None or d1 is None:
                verdict = "-"
            elif d1 < d0 - scale:
                verdict = "**closer**"
                moved.append(f"`{name}` seed {seed}")
            elif d1 > d0 + scale:
                verdict = "further"
                worse.append(f"`{name}` seed {seed}")
            else:
                verdict = "no"
            out.append(f"| `{name}` | {seed} | {x['status']} | {y['status']} | "
                       f"{_fmt(_float(x.get('dual_bound')), '.6g')} | "
                       f"{_fmt(_float(y.get('dual_bound')), '.6g')} | {verdict} | "
                       f"{x.get('nodes') or '-'} | {y.get('nodes') or '-'} | "
                       f"{y.get('restarts') or '-'} | {y.get('objective_branches') or '-'} |")
    met = bool(moved)
    out += ["", f"Bound moved towards the published optimum: "
            f"{', '.join(moved) if moved else 'none'}. Moved away: "
            f"{', '.join(worse) if worse else 'none'}."
            + (f" Not in both legs: {', '.join(f'`{n}`' for n in missing)}." if missing else "")
            + f" #418's acceptance, bound movement on at least one of "
            f"{', '.join(f'`{n}`' for n in instances)}, is "
            + ("**met** by this run." if met else "**not met** by this run."), ""]
    return "\n".join(out)


def tier2_section(path: Path | None) -> str:
    """The 60-instance tier (#504) over its seeds, every instance listed."""
    if path is None:
        return "\n".join([
            "Not yet run. Reproduce with:", "", "```",
            "python bench/runners/fetch_miplib.py --tier 2",
            "python bench/runners/miplib.py --tier 2 --seeds 3 --time-limit 300", "```", ""])
    rows = _read(path)
    leg = _leg(path)
    t = leg["total"]
    commit, machine = rows[0]["git_commit"], rows[0]["machine"]
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}` · {leg['seeds']} seeds · "
        f"{leg['limit']:g} s per run · one thread",
        "",
        f"The tier is chosen by the rule in `bench/runners/miplib_tier2.json`, written before "
        f"it was run: the first 60 MIPLIB 2017 benchmark instances with a proven optimum, by "
        f"file size. Over all {t['runs']} runs: **{t['matched']}** reached the published "
        f"optimum and **{t['proved']}** proved it; shifted geometric mean time "
        f"**{_fmt(t['sgm_seconds'])} s** (shift 10 s, every run that did not prove charged "
        f"the limit).",
        "",
        "| instance | seeds matched | seeds proved | sgm time (s) | sgm first feasible (s) | "
        "mean primal integral (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n, s in sorted(leg["summary"].items()):
        out.append(f"| `{n}` | {s['seeds_matched']}/{s['seeds']} | "
                   f"{s['seeds_proved']}/{s['seeds']} | {_fmt(s['sgm_seconds'])} | "
                   f"{_fmt(s['sgm_first_feasible_seconds'], '.3f')} | "
                   f"{_fmt(s['mean_primal_integral'])} |")
    failed = [f"`{r['instance']}` seed {r.get('seed', 0)} ({r['status']})" for r in rows
              if r.get("matched_published") != "1"]
    out += ["", "Runs that did not reach the published optimum, named: "
            + (", ".join(failed) if failed else "none") + ".", ""]
    # A proof claim the independent verifier refused is counted above as the solver
    # reported it, and named here so the count is never read as checked.
    rejected = [r for r in rows if r["status"] == "optimal"
                and str(r.get("independently_verified", "")) == "0"]
    if rejected:
        out += ["Runs reported optimal whose proof the independent verifier rejected, named "
                "(counted as proved above, as the solver reported them): " + ", ".join(
                    f"`{r['instance']}` seed {r.get('seed', 0)} (relative gap "
                    f"{float(r['relative_gap']):.3g})" if r.get("relative_gap") else
                    f"`{r['instance']}` seed {r.get('seed', 0)}" for r in rejected) + ".", ""]
    return "\n".join(out)
