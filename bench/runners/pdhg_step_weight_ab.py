#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PDHG step and primal weight: the adaptive rule against a constant step on a proved norm
bound, and PDLP's primal weight against a PID controller (#482).

One command runs the four legs on every instance, each on the same binary:

    default    the adaptive step of PDLP section 3.1 and the theta = 0.5 primal weight
    constant   pdhg_constant_step=true: eta = 0.998 / U, U a proved bound on ||A||_2
    pid        pdhg_primal_weight_pid=true: the PID controller at the option's gains
    both       both of the above

under each iteration scheme asked for (`--scheme`): `averaged`, the PDLP restarts on the
running average (the engine's default), and `halpern`, the restarted Halpern iteration
(pdhg_halpern=true, pdhg_restart=false), which is the pairing Lu, Peng & Yang (cuPDLPx,
arXiv:2507.14051) use the constant step with. Every solve is PDHG alone: pdhg_polish=false,
so a point the first-order method leaves short is not finished by the interior point and the
row measures the step and the weight, nothing else.

Instances: the nine committed Netlib instances (#481's set: data/netlib, published optimum
from data/netlib/reference.json), or `--instances`. Per row: the columns
ENGINEERING_RULES.md requires (instance, its sha256, objective, published objective,
absolute and relative error, status, wall and solver seconds, iterations, git commit,
machine), the relative-KKT crossings of the run as iterations and seconds (#486,
kkt_crossings.py), and tools/verify_solution.py's verdict on the written point.

Iteration counts are deterministic for a given binary, option set and seed; seconds are not,
and a CSV with seconds in it is evidence only from a machine that ran nothing else. ONE SEED
IS NOT ENOUGH even for the counts: random_seed moves nothing but the start of the power
iteration behind the norm estimate, and that alone moves the default path from 12,560 to
14,880 iterations on sc50a and from 62,640 to 83,160 on blend over seeds 0 to 3, because a
restart decision is a threshold that the last bits can tip. So every leg runs at every seed
of `--seeds` (0 1 2 by default), the ratio to the default leg is taken at the same seed, and
the summary is the geometric mean over instances and seeds.

    python bench/runners/pdhg_step_weight_ab.py --binary build/sankhya
    python bench/runners/pdhg_step_weight_ab.py --binary build/sankhya --engine cuda
    python bench/runners/pdhg_step_weight_ab.py --binary build/sankhya --engine cuda \\
        --solver-option gpu_on_device_loop=true

Writes bench/results/pdhg-step-weight-ab-<engine>-<sha>.csv (the sha from the binary,
#433), a name no glob of make_benchmarks_doc.py or latest_result.py reads, or --out.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kkt_crossings  # noqa: E402
import netlib  # noqa: E402  (run_one, sha256_file, machine_tag, stamp, default_binary)

REPO_ROOT = netlib.REPO_ROOT
DATA_DIR = netlib.DATA_DIR
RESULTS_DIR = netlib.RESULTS_DIR

NETLIB_NINE = ["afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1",
               "israel"]

LEGS = {
    "default": ("pdhg_constant_step=false", "pdhg_primal_weight_pid=false"),
    "constant": ("pdhg_constant_step=true", "pdhg_primal_weight_pid=false"),
    "pid": ("pdhg_constant_step=false", "pdhg_primal_weight_pid=true"),
    "both": ("pdhg_constant_step=true", "pdhg_primal_weight_pid=true"),
}
SCHEMES = {
    "averaged": ("pdhg_halpern=false", "pdhg_restart=true"),
    "halpern": ("pdhg_halpern=true", "pdhg_restart=false"),
}

COLUMNS = [
    "instance", "instance_sha256", "rows", "cols", "nnz", "engine", "scheme", "leg", "seed",
    "status", "objective", "published_objective", "absolute_error", "relative_error",
    "iterations", "iteration_ratio_to_default", *kkt_crossings.ITERATION_COLUMNS,
    *kkt_crossings.COLUMNS, "solver_seconds", "wall_seconds", "independently_verified",
    "verifier_message", "algorithm_ran", "iteration_limit", "time_limit", "solver_options",
    "git_commit", "machine", "gpu", "timestamp_utc",
]


def leg_options(engine: str, scheme: str, leg: str, iteration_limit: int,
                extra: list[str], seed: int = 0) -> list[str]:
    """The --option list of one solve. The leg's two switches and the seed come after
    `extra`, so a --solver-option cannot quietly turn one leg into another; the gains can be
    set there."""
    return ["algorithm=pdhg", "pdhg_polish=false", "pdhg_tolerance=1e-8",
            f"iteration_limit={iteration_limit}",
            f"gpu={'true' if engine == 'cuda' else 'false'}",
            *extra, *SCHEMES[scheme], *LEGS[leg], f"random_seed={seed}"]


def gpu_description(binary: Path) -> str:
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                            check=False)
    match = re.search(r"GPU ([^)]*\))", result.stdout)
    return match.group(1).strip() if match else ""


def references() -> dict[str, float]:
    blob = json.loads((DATA_DIR / "reference.json").read_text())
    return {name: float(entry["published_optimal"])
            for name, entry in blob["instances"].items()
            if entry.get("published_optimal") is not None}


def geometric_mean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def make_row(name: str, digest: str, reference: float | None, args, scheme: str, leg: str,
             seed: int, options: list[str], result: dict) -> dict:
    """One CSV row from one solve; the stamp columns are filled by the caller."""
    objective = result.get("objective")
    absolute = relative = ""
    if objective is not None and reference is not None:
        absolute = abs(objective - reference)
        relative = absolute / max(1.0, abs(reference))
    verified = result.get("verified")
    return {
        "instance": name, "instance_sha256": digest,
        "rows": result.get("rows", ""), "cols": result.get("columns", ""),
        "nnz": result.get("nonzeros", ""), "engine": args.engine,
        "scheme": scheme, "leg": leg, "seed": seed, "status": result.get("status", ""),
        "objective": "" if objective is None else repr(objective),
        "published_objective": "" if reference is None else repr(reference),
        "absolute_error": absolute, "relative_error": relative,
        "iterations": result.get("iterations", ""), "iteration_ratio_to_default": "",
        **{k: result.get(k, "") for k in kkt_crossings.ALL_COLUMNS},
        "solver_seconds": result.get("solver_seconds", ""),
        "wall_seconds": round(result.get("wall_seconds", 0.0), 6),
        "independently_verified": "" if verified is None else int(bool(verified)),
        "verifier_message": result.get("verifier_output", "")[:300],
        "algorithm_ran": result.get("algorithm", ""),
        "iteration_limit": args.iteration_limit, "time_limit": args.time_limit,
        "solver_options": " ".join(options),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--engine", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scheme", choices=("averaged", "halpern", "both"), default="both")
    parser.add_argument("--legs", nargs="*", choices=tuple(LEGS), default=list(LEGS))
    parser.add_argument("--instances", nargs="*", metavar="NAME", default=NETLIB_NINE,
                        help="Netlib names under data/netlib (default: the nine committed)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                        help="random_seed values; every leg runs at each (default 0 1 2)")
    parser.add_argument("--iteration-limit", type=int, default=1000000,
                        help="per solve; the default is the engine's own ceiling")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every solve of every leg and recorded, e.g. "
                             "pdhg_pid_ki=0.1 or gpu_on_device_loop=true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = args.binary or netlib.default_binary()
    commit = netlib.git_commit(binary)
    gpu = gpu_description(binary) if args.engine == "cuda" else ""
    machine = netlib.machine_tag()
    stamp_time = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    published = references()
    schemes = ["averaged", "halpern"] if args.scheme == "both" else [args.scheme]
    legs = [leg for leg in LEGS if leg in args.legs]
    print(f"binary {binary}  commit {commit}  engine {args.engine} {gpu}  schemes "
          f"{' '.join(schemes)}  legs {' '.join(legs)}  options "
          f"{' '.join(args.solver_option) or 'none'}", flush=True)

    rows: list[dict] = []
    for scheme in schemes:
        print(f"\n{scheme}: iterations (status), and the ratio to the default leg at the same "
              f"seed\n")
        print(f"{'instance':<11}{'seed':>5}" + "".join(f"{leg:>24}" for leg in legs))
        for name in args.instances:
            mps = DATA_DIR / f"{name}.mps"
            if not mps.exists():
                print(f"{name:<11}  missing {mps}")
                continue
            digest = netlib.sha256_file(mps)
            reference = published.get(name)
            for seed in args.seeds:
                mine = []
                for leg in legs:
                    options = leg_options(args.engine, scheme, leg, args.iteration_limit,
                                          args.solver_option, seed)
                    result = netlib.run_one(binary, mps, args.time_limit, not args.no_verify,
                                            options)
                    mine.append(make_row(name, digest, reference, args, scheme, leg, seed,
                                         options, result))
                base = next((r["iterations"] for r in mine if r["leg"] == "default"), None)
                cells = []
                for row in mine:
                    text = f"{row['iterations']} ({row['status'][:4]})"
                    if (isinstance(base, int) and base > 0
                            and isinstance(row["iterations"], int)):
                        ratio = row["iterations"] / base
                        row["iteration_ratio_to_default"] = f"{ratio:.4f}"
                        text += f" {ratio:.2f}x"
                    cells.append(f"{text:>24}")
                print(f"{name:<11}{seed:>5}" + "".join(cells), flush=True)
                for row in mine:
                    row.update(git_commit=commit, machine=machine, gpu=gpu,
                               timestamp_utc=stamp_time)
                rows += mine

    # ---- Summary: verdicts, and the iteration ratio where both legs reached optimal --------
    print("\nper scheme and leg: optimal / verified of the runs, and the geometric mean over "
          "instances and seeds of iterations over the default leg's at the same seed, where "
          "both are optimal\n")
    for scheme in schemes:
        for leg in legs:
            mine = [r for r in rows if r["scheme"] == scheme and r["leg"] == leg]
            optimal = sum(r["status"] == "optimal" for r in mine)
            verified = sum(r["independently_verified"] == 1 for r in mine)
            ratios = []
            for r in mine:
                base = next((b for b in rows if b["scheme"] == scheme and b["leg"] == "default"
                             and b["instance"] == r["instance"] and b["seed"] == r["seed"]),
                            None)
                if (base and base["status"] == "optimal" and r["status"] == "optimal"
                        and isinstance(base["iterations"], int) and base["iterations"] > 0
                        and isinstance(r["iterations"], int) and r["iterations"] > 0):
                    ratios.append(r["iterations"] / base["iterations"])
            mean = geometric_mean(ratios)
            print(f"  {scheme:<9}{leg:<10} optimal {optimal}/{len(mine)}  verified "
                  f"{verified}/{len(mine)}  iterations vs default "
                  f"{'-' if mean is None else f'{mean:.3f}x'} over {len(ratios)}")

    tag = re.sub(r"[^A-Za-z0-9]+", "-", " ".join(args.solver_option)).strip("-")
    out = args.out or (RESULTS_DIR / f"pdhg-step-weight-ab-{args.engine}-{commit}"
                       f"{'-' + tag if tag else ''}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {netlib.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
