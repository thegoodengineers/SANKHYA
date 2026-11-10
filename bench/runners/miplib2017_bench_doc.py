#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The whole MIPLIB 2017 benchmark set in docs/BENCHMARKS.md (#1199), rendered from its CSVs.

SANKHYA's rows come from `miplib.py --set miplib2017-bench`, HiGHS's from `compare.py --suite
miplib2017-bench`, both run shard by shard in bench.yml and joined by merge_shards.py into
`miplib2017-bench-<limit>s-<sha>.csv` and `highs-miplib2017-bench-<limit>s-<sha>.csv`. Neither
name matches a MIPLIB 2017 subset glob (`miplib-*`) or MIPLIB 3's (test_miplib2017_bench.py
checks every glob), and a shard's name (`...-shard1of6-...`) matches neither of these.

THE THREE COUNTS, the same rule for both solvers, every number counted from the rows:
- PROVED: the solver said optimal at the published optimum (its own gap target, 1e-4
  relative for both) with a point the verifier accepted, or said infeasible on one of the
  seven models the solution file proves infeasible;
- MATCHED: a verified point at the published optimum, or infeasible on an `=inf=` model, the
  status aside (SANKHYA can reach the optimum and not prove it; HiGHS reports an objective at
  the optimum only when it proves it);
- FEASIBLE AND VERIFIED: any point the verifier accepted, at the optimum or not.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "miplib2017-bench" / "manifest.json"
LIMIT = 60


def sankhya_pattern(limit: int = LIMIT) -> str:
    return f"miplib2017-bench-{limit}s-[0-9a-f]*.csv"


def highs_pattern(limit: int = LIMIT) -> str:
    return f"highs-miplib2017-bench-{limit}s-[0-9a-f]*.csv"


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sankhya_verdicts(r: dict) -> dict[str, bool]:
    verified = r.get("independently_verified") == "1"
    infeasible = r.get("status") == "infeasible" and r.get("matched_published") == "1"
    return {"proved": r.get("proved_optimal") == "1" and (verified or infeasible),
            "matched": r.get("matched_published") == "1" and (verified or infeasible),
            "feasible": verified}


def highs_verdicts(r: dict) -> dict[str, bool]:
    verified = r.get("independently_verified") == "1"
    matches = r.get("matches_reference") == "1"
    infeasible = r.get("status") == "infeasible" and matches
    proved = (r.get("status") == "optimal" and matches and verified) or infeasible
    return {"proved": proved, "matched": proved, "feasible": verified}


def counts(rows: list[dict], verdicts) -> dict[str, int]:
    out = {"instances": len({r["instance"] for r in rows}), "proved": 0, "matched": 0,
           "feasible": 0}
    for r in rows:
        for key, value in verdicts(r).items():
            out[key] += int(value)
    return out


def load_manifest(path: Path = MANIFEST) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cell(status: str, verdict: dict[str, bool], seconds: str) -> str:
    marks = [status or "?"] + [word for word, key in (("proved", "proved"),
                                                      ("matched", "matched"),
                                                      ("verified", "feasible"))
                               if verdict[key]]
    try:
        marks.append(f"{float(seconds):.1f} s")
    except (TypeError, ValueError):
        pass
    return ", ".join(marks)


def section(sankhya: Path | None, highs: Path | None, manifest: dict | None) -> str:
    instances = (manifest or {}).get("instances", {})
    statuses: dict[str, int] = {}
    for entry in instances.values():
        statuses[entry.get("solu_status", "?")] = statuses.get(entry.get("solu_status", "?"), 0) + 1
    out = [
        "Every one of the 240 instances of the MIPLIB 2017 benchmark set (`benchmark-v2.test`), "
        "none chosen or left out, at 60 s, one thread, one seed (the published file), SANKHYA "
        "and HiGHS in the same bench.yml job on the same files, shard by shard. References are "
        "what `miplib2017-v28.solu` proves (pinned by sha256 in `bench/runners/fetch_miplib.py`; "
        "`data/miplib2017-bench/manifest.json` records the sha256 of every model and the line "
        "each reference was read from): "
        + ", ".join(f"{n} `{s}`" for s, n in sorted(statuses.items()))
        + ". An `=inf=` model is matched and proved by an infeasible status.",
        "",
        "PROVED: optimal at the published optimum with a verified point, or infeasible on an "
        "`=inf=` model. MATCHED: a verified point at the published optimum (or infeasible on an "
        "`=inf=` model), proved or not. FEASIBLE AND VERIFIED: any point "
        "`tools/verify_solution.py` accepted. A solve still running at the limit plus 60 s is "
        "killed and recorded `hang`.",
        "",
    ]
    if sankhya is None and highs is None:
        return "\n".join(out + ["Not yet run. Reproduce, one shard per job:", "", "```",
                                "python bench/runners/fetch_miplib.py --benchmark --shard 1/6",
                                "# bench.yml suite=miplib2017-bench options=1/6 ... 6/6", "```",
                                ""])
    ours = read(sankhya) if sankhya else []
    theirs = read(highs) if highs else []
    machines = sorted({r.get("machine", "") for r in ours + theirs} - {""})
    first = (ours or theirs)[0]
    out += [f"Commit `{first.get('git_commit', '?')}`; SANKHYA `bench/results/"
            f"{sankhya.name if sankhya else '-'}`, HiGHS `bench/results/"
            f"{highs.name if highs else '-'}`"
            + (f" (HiGHS {theirs[0].get('solver_version', '?')})" if theirs else "")
            + ". The jobs ran on: " + "; ".join(f"`{m}`" for m in machines) + ".", ""]
    rows = [("SANKHYA", counts(ours, sankhya_verdicts))] if ours else []
    rows += [("HiGHS", counts(theirs, highs_verdicts))] if theirs else []
    out += ["| solver | instances | proved | matched | feasible and verified |",
            "|---|---:|---:|---:|---:|"]
    out += [f"| {label} | {c['instances']} | **{c['proved']}** | {c['matched']} | "
            f"{c['feasible']} |" for label, c in rows]
    rejected = [f"`{r['instance']}`" for r in ours if r.get("independently_verified") == "0"]
    rejected += [f"`{r['instance']}` (HiGHS)" for r in theirs
                 if r.get("independently_verified") == "0"]
    wrong = [f"`{r['instance']}`" for r in ours
             if r.get("status") == "optimal" and r.get("matched_published") != "1"]
    out += ["", f"Points the verifier rejected: {', '.join(rejected) or 'none'}. SANKHYA "
            f"reported optimal away from the reference: {', '.join(wrong) or 'none'}.", ""]
    by_ours = {r["instance"]: r for r in ours}
    by_theirs = {r["instance"]: r for r in theirs}
    out += ["| instance | reference | SANKHYA | HiGHS |", "|---|---:|---|---|"]
    for name in sorted(set(instances) | set(by_ours) | set(by_theirs)):
        entry = instances.get(name, {})
        reference = ("infeasible" if entry.get("solu_status") == "=inf="
                     else entry.get("published_text") or "?")
        mine, rival = by_ours.get(name), by_theirs.get(name)
        out.append(f"| `{name}` | {reference} | "
                   + (_cell(mine["status"], sankhya_verdicts(mine), mine.get("wall_seconds", ""))
                      if mine else "no run") + " | "
                   + (_cell(rival["status"], highs_verdicts(rival),
                            rival.get("solver_seconds", "")) if rival else "no run") + " |")
    out.append("")
    return "\n".join(out)
