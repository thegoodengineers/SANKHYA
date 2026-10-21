#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Warm against cold after rows and columns are added and removed (#913).

A planner adds a unit (columns), a limit (a row), or drops a period (rows and columns),
and re-solves from yesterday's answer. Through the bindings this is `Model.add_column`,
`add_row`, `delete_rows`, `delete_cols` and `Model.solve(start=previous)`, which places the
previous basis on the edited model by name (a new row basic on its slack, a new column at a
bound, removals repaired with row logicals; Maros, "Computational Techniques of the Simplex
Method", sec. 9.4) and restarts the dual simplex from it.

THE WALK. On each Netlib model, `--steps` steps of one to three random structural edits,
each step applied to the previous accepted model. Every step is solved twice on the same
edited model: cold (`start=None`) and warm (`start=` the previous accepted warm result),
both with the dual simplex, so the pivot counts compare one engine with itself. A step
whose cold solve has no optimum (an edit can make the model infeasible or unbounded) is
recorded and then undone - the model is read again and the accepted edits replayed - so
the walk continues from a model that has an optimum. The reference objective of an edited
model is the cold solve's; there is no published one.

The same walk, with each point checked against the model for feasibility, dual sign,
complementarity and a zero duality gap, is the test
`CApiStructuralWarm.ManyRandomEditsThroughTheCApiGiveTheVerifiedColdAnswer`.

Usage:
    python bench/runners/structural_warm_start.py
    python bench/runners/structural_warm_start.py --steps 20 --seed 5 --out out.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
NETLIB = REPO_ROOT / "data" / "netlib"
MODELS = ["afiro", "adlittle", "sc50a", "sc105", "share2b", "blend", "kb2", "scagr7",
          "stocfor1", "recipe"]
# Objectives agree when within this, relative to max(1, |cold|): the bar the test holds.
AGREE = 1e-9

sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402


def apply(model, edit) -> None:
    kind = edit[0]
    if kind == "-c":
        model.delete_cols([edit[1]])
    elif kind == "-r":
        model.delete_rows([edit[1]])
    elif kind == "+c":
        _, name, cost, upper, entries = edit
        j = model.add_column(cost=cost, lower=0.0, upper=upper, name=name)
        for i, value in entries:
            model.set_coefficient(i, j, value)
    else:
        _, name, lower, upper, entries = edit
        model.add_row(dict(entries), lower=lower, upper=upper, name=name)


def random_edit(rng: random.Random, cols: int, rows: int, x: list[float], cost_scale: float,
                fresh: int):
    """One edit of a model with `cols` columns and `rows` rows whose last point is `x`."""
    kind = rng.random()
    if kind < 0.25 and cols > 2:
        return ("-c", rng.randrange(cols))
    if kind < 0.5 and rows > 2:
        return ("-r", rng.randrange(rows))
    if kind < 0.75:
        picked = sorted({rng.randrange(rows) for _ in range(1 + rng.randrange(4))})
        entries = [(i, rng.choice((-1.0, 1.0)) * (0.5 + rng.random())) for i in picked]
        return ("+c", f"NEWC{fresh}", (rng.random() - 0.3) * cost_scale,
                1.0 + 100.0 * rng.random(), entries)
    picked = sorted({rng.randrange(cols) for _ in range(2 + rng.randrange(4))})
    entries = [(j, 0.5 + rng.random()) for j in picked]
    activity = sum(v * x[j] for j, v in entries)
    bound = activity * (0.9 + 0.2 * rng.random())
    slack = 1e-3 * (1.0 + abs(activity))
    if rng.random() < 0.5:
        return ("+r", f"NEWR{fresh}", None, bound + slack, entries)
    return ("+r", f"NEWR{fresh}", bound - slack, None, entries)


def track(x: list[float], edit) -> None:
    """Keep `x`, the last point, aligned with the columns after `edit`."""
    if edit[0] == "-c":
        del x[edit[1]]
    elif edit[0] == "+c":
        x.append(0.0)


def costs_of(path: Path) -> float:
    """The largest |objective coefficient| in a free-or-fixed MPS file, at least 1."""
    objective = None
    largest = 1.0
    section = ""
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.strip() or line.startswith("*"):
            continue
        if not line[0].isspace():
            section = line.split()[0]
            continue
        fields = line.split()
        if section == "ROWS" and fields[0] == "N" and objective is None:
            objective = fields[1]
        elif section == "COLUMNS" and objective is not None and "'MARKER'" not in line:
            for k in range(1, len(fields) - 1, 2):
                if fields[k] == objective:
                    largest = max(largest, abs(float(fields[k + 1].replace("D", "E"))))
    return largest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=12, help="steps per model")
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--machine", default=None,
                        help="machine tag for the CSV (default: OS-architecture)")
    args = parser.parse_args()

    import sankhya  # after sys.path is set up

    rng = random.Random(args.seed)
    rows: list[dict] = []
    fresh = 0
    options = {"log_to_console": False, "algorithm": "dual-simplex"}
    for name in args.models:
        path = NETLIB / f"{name}.mps"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cost_scale = costs_of(path)
        model = sankhya.Model.read(str(path))
        accepted: list = []
        previous = model.solve(**options)
        if previous.status != "optimal":
            print(f"{name}: the unedited model is {previous.status}; skipped", file=sys.stderr)
            continue
        x = list(previous.x)
        for step in range(1, args.steps + 1):
            edits = []
            x_step = list(x)
            for _ in range(1 + rng.randrange(3)):
                edit = random_edit(rng, model.num_cols, model.num_rows, x_step, cost_scale,
                                   fresh)
                fresh += 1
                apply(model, edit)
                track(x_step, edit)
                edits.append(edit)
            cold = model.solve(**options)
            warm = model.solve(start=previous, **options)
            both = cold.status == "optimal" and warm.status == "optimal"
            gap = abs(warm.objective - cold.objective) if both else float("nan")
            rel = gap / max(1.0, abs(cold.objective)) if both else float("nan")
            agree = cold.status == warm.status and (not both or rel <= AGREE)
            rows.append({
                "instance": name,
                "step": step,
                "edits": " ".join(e[0] for e in edits),
                "cols": model.num_cols,
                "rows": model.num_rows,
                "cold_status": cold.status,
                "warm_status": warm.status,
                "cold_objective": f"{cold.objective:.17g}" if both else "",
                "warm_objective": f"{warm.objective:.17g}" if both else "",
                "abs_gap": f"{gap:.3g}" if both else "",
                "rel_gap": f"{rel:.3g}" if both else "",
                "answers_agree": agree,
                "warm_route": "presolve bypassed" in warm.message
                              and "#883" not in warm.message,
                "cold_iterations": cold.iterations,
                "warm_iterations": warm.iterations,
                "cold_primal_infeasibility": f"{cold.primal_infeasibility:.3g}",
                "warm_primal_infeasibility": f"{warm.primal_infeasibility:.3g}",
                "cold_dual_infeasibility": f"{cold.dual_infeasibility:.3g}",
                "warm_dual_infeasibility": f"{warm.dual_infeasibility:.3g}",
                "cold_seconds": f"{cold.seconds:.6f}",
                "warm_seconds": f"{warm.seconds:.6f}",
                "accepted": cold.status == "optimal",
                "instance_sha256": sha,
            })
            if not agree:
                print(f"{name} step {step}: WARM AND COLD DISAGREE - cold {cold.status} "
                      f"{cold.objective}, warm {warm.status} {warm.objective}", file=sys.stderr)
            if cold.status == "optimal":
                accepted.extend(edits)
                previous = warm
                x = list(warm.x)
            else:
                model = sankhya.Model.read(str(path))
                for edit in accepted:
                    apply(model, edit)

    pairs = [r for r in rows if r["accepted"]]
    cold_total = sum(r["cold_iterations"] for r in pairs)
    warm_total = sum(r["warm_iterations"] for r in pairs)
    edits_total = sum(len(r["edits"].split()) for r in pairs)
    disagree = sum(1 for r in rows if not r["answers_agree"])
    print(f"{len(rows)} steps, {len(pairs)} optimal pairs over {edits_total} edits; "
          f"pivots warm {warm_total} against cold {cold_total} "
          f"({warm_total / max(1, cold_total):.3f}x); warm fewer on "
          f"{sum(1 for r in pairs if r['warm_iterations'] < r['cold_iterations'])}; "
          f"warm route on {sum(1 for r in pairs if r['warm_route'])}; "
          f"answers disagree on {disagree}")

    commit = stamp.stamp(sankhya.locate_executable())
    out = args.out or (RESULTS_DIR / f"structural-warm-start-{commit}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    machine = args.machine or f"{platform.system()}-{platform.machine()}"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())
                                + ["git_commit", "machine"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "git_commit": commit, "machine": machine})
    print(f"wrote {out}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
