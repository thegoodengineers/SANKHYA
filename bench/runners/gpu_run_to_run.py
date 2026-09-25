#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The same GPU PDHG solve repeated, to show whether it is reproducible run to run (#478).

Until #478 the device reductions summed in whatever order the atomics landed, so three
identical brazil3 solves at 1e-8 took different iteration counts and landed at different
distances from the optimum (#488's investigation). `deterministic=true` now fixes the order.
This runner repeats one solve `--repeats` times for each option set given with `--leg`
(an empty leg is the default configuration) and writes every repeat as its own row, so the
spread of iterations, objective and gap is read straight from the file.

The solve protocol is gpu_real_instances.py's (`run_solve`: PDHG alone, pdhg_polish=false,
the solver's own clock) and the reference is gpu_arms.reference_objective (HiGHS as a
separate process, once per instance).

    python bench/runners/gpu_run_to_run.py --binary build/sankhya \\
        --mps data/mittelmann/brazil3.mps --tolerance 1e-8 --repeats 3 \\
        --leg "" --leg "deterministic=true"

Output: bench/results/gpu-run-to-run-<sha>.csv (the sha from the binary, #433).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_arms  # noqa: E402
import stamp  # noqa: E402
from gpu_real_instances import gpu_description, run_solve  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "bench" / "results"
COLUMNS = ["instance", "instance_sha256", "leg", "repeat", "tolerance", "status", "objective",
           "reference_objective", "reference_source", "abs_gap", "rel_gap", "iterations",
           "seconds", "wall_seconds", "primal_residual", "dual_residual", "git_commit",
           "machine", "gpu", "timestamp_utc", "solver_options"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mps", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--leg", action="append", default=[],
                        help="space-separated KEY=VALUE options for one leg; '' is the default")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--reference-time-limit", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    legs = args.leg or [""]

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    gpu = gpu_description(args.binary)
    name = args.mps.name.split(".")[0]
    digest = gpu_arms.sha256_file(args.mps)
    reference, source, note = gpu_arms.reference_objective(args.mps, args.reference_time_limit)
    print(f"{name}: sha256 {digest[:16]}..., reference {reference!r} ({source}; {note})")
    print(f"commit {commit}  gpu {gpu}")

    rows = []
    for leg in legs:
        options = leg.split()
        for repeat in range(1, args.repeats + 1):
            result = run_solve(args.binary, args.mps, "pdhg-cuda", args.tolerance,
                               args.time_limit, options)
            absolute, relative = gpu_arms.gaps(result["objective"], reference)
            rows.append({
                "instance": name, "instance_sha256": digest, "leg": leg or "default",
                "repeat": repeat, "tolerance": f"{args.tolerance:.0e}",
                "status": result["status"],
                "objective": "" if result["objective"] is None else repr(result["objective"]),
                "reference_objective": "" if reference is None else repr(reference),
                "reference_source": source, "abs_gap": gpu_arms.fmt_gap(absolute),
                "rel_gap": gpu_arms.fmt_gap(relative), "iterations": result["iterations"],
                "seconds": round(result["seconds"], 6), "wall_seconds": round(result["wall"], 6),
                "primal_residual": result.get("primal_residual", ""),
                "dual_residual": result.get("dual_residual", ""),
                "git_commit": commit, "machine": machine, "gpu": gpu,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"),
                "solver_options": leg,
            })
            row = rows[-1]
            print(f"{row['leg']:>22}  #{repeat}  {row['status']:>10}  it {row['iterations']:>8}"
                  f"  {row['seconds']:>9.3f} s  rel gap {row['rel_gap'] or '-'}")

    out = args.out or RESULTS_DIR / f"gpu-run-to-run-{commit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
