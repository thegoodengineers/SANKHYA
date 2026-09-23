#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rolling-horizon re-planning: warm against cold, one edit at a time (#524).

Refinery planning is re-run daily and weekly with a small change - a price, a tank level -
not rebuilt from nothing. #218 already gives the primitive this needs (`Model.set_cost` /
`set_col_bounds` / `set_row_bounds` editing a model in place, `Model.solve(start=result)`
resuming from the previous basis) and `tests/unit/test_warm_start.cpp` already proves a
SINGLE warm re-solve matches a cold one on four Netlib instances. What #524 asks for beyond
that is the SEQUENCE: many edits in a row, on an instance shaped like the thing PS26119
actually names - a T-period refinery plan (`generate_refinery_lp.py`, #211) - with the
warm-vs-cold cost of each step written down, not asserted.

THE EDIT. Each "week" perturbs ONE crude's purchase price (`BUY_<crude>_<week>`'s cost) by a
random +/-10% - "a price changes" is the issue's own first example, and a COST-only edit can
never make a feasible model infeasible (only bounds and rows can), so every step's status is
determined by the model alone and cannot differ between the warm and cold arm for a reason
unrelated to warm-starting - the pivot count and the wall clock are the only things being
measured, not two different feasibility questions.

THE COMPARISON, PER STEP: the SAME edit applied to the SAME model, solved twice - once with
`start=None` (cold: the slack basis, exactly what a fresh read-and-solve would do) and once
with `start=<the previous WARM result>` (warm: this step's basis restart). Reusing one model
object rather than two is deliberate: solving does not mutate the model, so both arms see
identical bounds/costs/rows at every step, and the only thing that differs between them is
which basis (if any) the simplex started from.

Usage:
    python bench/runners/rolling_warm_start.py --periods 52 --seed 11
    python bench/runners/rolling_warm_start.py --periods 52 --seed 11 --delta 0.05 --out out.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
REFINERY_GENERATOR = REPO_ROOT / "bench" / "runners" / "generate_refinery_lp.py"

sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402


def generate(periods: int, seed: int, directory: Path) -> Path:
    out = directory / f"refinery-t{periods}-s{seed}.mps"
    subprocess.run(
        [sys.executable, str(REFINERY_GENERATOR), "--periods", str(periods), "--seed",
         str(seed), "--out", str(out)],
        check=True, capture_output=True, text=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--periods", type=int, default=52)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--crude", type=int, default=0, help="which crude's price to roll")
    parser.add_argument("--delta", type=float, default=0.10,
                        help="fractional price perturbation per step")
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import sankhya  # after sys.path is set up

    with tempfile.TemporaryDirectory() as tmp:
        instance = generate(args.periods, args.seed, Path(tmp))

        # BUY_<crude>_<t>'s column index, read off the MPS file's own COLUMNS section
        # rather than assumed: the generator's layout is documented, not part of this
        # script's contract with it, so this is checked against the file every run.
        target_name = [f"BUY_{args.crude}_{t}" for t in range(args.periods)]
        name_to_index: dict[str, int] = {}
        in_columns = False
        seen = 0
        with open(instance, "r", encoding="ascii") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped == "COLUMNS":
                    in_columns = True
                    continue
                if in_columns and stripped and not stripped.startswith("*"):
                    if stripped in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                        break
                    # Free-format MPS (the generator's own description): fields are
                    # whitespace-separated, not fixed-width columns.
                    name = stripped.split()[0]
                    if name not in name_to_index:
                        name_to_index[name] = seen
                        seen += 1
        missing = [n for n in target_name if n not in name_to_index]
        if missing:
            print(f"error: {len(missing)} target column(s) not found, e.g. {missing[0]}; "
                 "the generator's naming may have changed", file=sys.stderr)
            return 2
        columns = [name_to_index[n] for n in target_name]

        model = sankhya.Model.read(str(instance))
        check = model.num_cols
        if columns[-1] >= check:
            print(f"error: computed column index {columns[-1]} is out of range for a "
                 f"{check}-column model", file=sys.stderr)
            return 2

        rng = random.Random(args.rng_seed)
        rows: list[dict] = []
        warm_previous = None
        base_costs = {}

        for week in range(args.periods):
            column = columns[week]
            # The cost this column already carries - read once and perturbed relative to
            # ITSELF each visit, so ten runs of this script at the same --rng-seed move the
            # same column by the same sequence of factors regardless of solve order.
            if column not in base_costs:
                # Model has no "read one coefficient back" accessor either (by the same
                # write-only design as column lookup above); the ORIGINAL cost is instead
                # read once from Result.reduced_costs's algebra is not it - simplest correct
                # source is the file itself, parsed once per column on first use.
                base_costs[column] = _read_cost_from_mps(instance, target_name[week])
            factor = 1.0 + rng.uniform(-args.delta, args.delta)
            model.set_cost(column, base_costs[column] * factor)

            cold = model.solve(log_to_console=False)
            warm = model.solve(log_to_console=False, start=warm_previous)
            warm_previous = warm

            check_match = cold.status == warm.status and (
                not cold.claims_a_point or abs(cold.objective - warm.objective)
                <= 1e-6 * max(1.0, abs(cold.objective)))
            rows.append({
                "week": week + 1,
                "column": target_name[week],
                "cost_factor": f"{factor:.6f}",
                "cold_status": cold.status,
                "warm_status": warm.status,
                "cold_objective": f"{cold.objective:.10g}" if cold.claims_a_point else "",
                "warm_objective": f"{warm.objective:.10g}" if warm.claims_a_point else "",
                "answers_agree": check_match,
                "cold_iterations": cold.iterations,
                "warm_iterations": warm.iterations,
                "cold_seconds": f"{cold.seconds:.6f}",
                "warm_seconds": f"{warm.seconds:.6f}",
            })
            if not check_match:
                print(f"week {week + 1}: WARM AND COLD DISAGREE - "
                     f"cold={cold.status}/{cold.objective if cold.claims_a_point else 'n/a'} "
                     f"warm={warm.status}/{warm.objective if warm.claims_a_point else 'n/a'}",
                     file=sys.stderr)

        total_cold_iter = sum(r["cold_iterations"] for r in rows)
        total_warm_iter = sum(r["warm_iterations"] for r in rows)
        total_cold_s = sum(float(r["cold_seconds"]) for r in rows)
        total_warm_s = sum(float(r["warm_seconds"]) for r in rows)
        disagreements = sum(1 for r in rows if not r["answers_agree"])

        print(f"{args.periods} weeks, crude {args.crude}, delta +/-{args.delta:.0%}:")
        print(f"  iterations: cold {total_cold_iter}, warm {total_warm_iter} "
             f"({total_warm_iter / max(1, total_cold_iter):.3f}x)")
        print(f"  seconds:    cold {total_cold_s:.4f}, warm {total_warm_s:.4f} "
             f"({total_warm_s / max(1e-9, total_cold_s):.3f}x)")
        print(f"  answers agree on {len(rows) - disagreements}/{len(rows)} weeks")

        # The commit the BINARY was built from (#433), read from the CLI built beside the
        # library the bindings load - the library itself reports no commit. HEAD would
        # name whatever branch is checked out, which is how a branch sha reached a CSV.
        commit = stamp.stamp(sankhya.locate_executable())
        out = args.out or (RESULTS_DIR / f"rolling-warm-start-{commit}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        machine = f"{platform.system()}-{platform.machine()}"
        with open(out, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys()) + ["git_commit", "machine", "instance_sha256"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["git_commit"] = commit
                row["machine"] = machine
                row["instance_sha256"] = _sha256(instance)
                writer.writerow(row)
        print(f"wrote {out}")

        return 0 if disagreements == 0 else 1


def _read_cost_from_mps(path: Path, column_name: str) -> float:
    with open(path, "r", encoding="ascii") as handle:
        in_columns = False
        for line in handle:
            stripped = line.strip()
            if stripped == "COLUMNS":
                in_columns = True
                continue
            if not in_columns:
                continue
            if stripped in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                break
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == column_name and parts[1] == "COST":
                return float(parts[2])
    raise ValueError(f"no COST entry found for column {column_name!r} in {path}")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
