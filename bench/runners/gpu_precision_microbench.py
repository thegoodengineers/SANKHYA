#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device product micro-benchmark for GPU PDHG mixed precision, #982 step 1.

"Measure before building." Before any mixed-precision iteration code is trusted, this
measures the thing #982's "Why" is actually betting on: that A x and A^T y, run with single
precision VALUES on the same sparsity pattern, are faster on the device than the same
products in double. It does NOT run the solver and makes NO claim about end-to-end PDHG
time, iteration counts or convergence - that is bench/runners/gpu_datacenter.py's and a
follow-up runner's job, once (and only if) this step says the product pair is worth it.

What it does, on a machine with a CUDA device and both tools built
(`--target sankhya-precision-microbench sankhya-dump-scaled-csr`, CUDA on):
  1. For each --model (the generated 10,000 x 10,000 LP from generate_large_lp.py, brazil3
     after fetch_mittelmann.py, one refinery ladder step from
     bench/case_studies/refinery/generator.py), runs sankhya-dump-scaled-csr, which writes the
     SCALED CSR PDHG multiplies by (build_scaling with the engine's ten Ruiz rounds and one
     Pock-Chambolle pass, then CsrView, exactly as src/gpu/pdhg_gpu.cu uploads it) in the flat
     format bench/tools/gpu_precision_microbench.cu reads. A ready dump can be given with
     --csr-dump instead.
  2. Runs sankhya-precision-microbench on each dump and parses its two CSV lines (double,
     float).
  3. Writes bench/results/gpu-precision-<card>-<commit>.csv in the column order #982 asks for.
     Only the product-pair columns are filled; the iteration, residual and end-to-end columns
     belong to the solver A/B that follows only if this step says the pair is worth it.

    python bench/runners/gpu_precision_microbench.py --binary build_gpu/sankhya \\
        --microbench build_gpu/sankhya-precision-microbench \\
        --dump-tool build_gpu/sankhya-dump-scaled-csr --card "RTX 5050 Laptop GPU" \\
        --model lp_10000x10000.mps --model-name kkt_10000x10000 \\
        --model brazil3.mps --model-name brazil3 \\
        --model refinery_large.mps --model-name refinery_large
"""
from __future__ import annotations

import argparse
import datetime as dt
import csv
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: the CSV names the commit the BINARY was built from)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# The exact column order #982's acceptance item asks for.
CSV_COLUMNS = [
    "model", "rows", "nonzeros", "precision_arm", "iterations", "restarts",
    "refinement_rounds", "seconds_per_iteration", "seconds_end_to_end_median",
    "seconds_end_to_end_min", "seconds_end_to_end_max", "final_residual_primal_double",
    "final_residual_dual_double", "final_residual_gap_double", "status", "verifier_verdict",
    "peak_device_memory_bytes", "git_commit", "card", "timestamp_utc",
]


def run_microbench(microbench: Path, dump_path: Path, warmup: int, timed: int) -> dict:
    out = subprocess.run([str(microbench), str(dump_path), str(warmup), str(timed)],
                         capture_output=True, text=True, timeout=600, check=True)
    rows = list(csv.DictReader(out.stdout.splitlines()))
    return {row["precision"]: row for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, help="sankhya-cli binary, for the commit stamp "
                    "(#433); not required to run the product-pair measurement itself")
    ap.add_argument("--microbench", type=Path,
                    help="path to the built sankhya-precision-microbench tool")
    ap.add_argument("--card", required=True, help="the card's name, for the output filename "
                    "and the CSV's card column, e.g. 'RTX 5050 Laptop GPU'")
    ap.add_argument("--csr-dump", type=Path, action="append", default=[],
                    help="a pre-built CSR dump (bench/tools/gpu_precision_microbench.cu's "
                    "format) to measure; repeatable. Without this, the script cannot source "
                    "a real matrix today (see the module docstring's NOTE) and exits "
                    "without writing a CSV.")
    ap.add_argument("--model", type=Path, action="append", default=[],
                    help="an MPS/LP model to dump with --dump-tool and measure; repeatable")
    ap.add_argument("--dump-tool", type=Path,
                    help="path to the built sankhya-dump-scaled-csr tool (for --model)")
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="where --model dumps are written (default: a temporary directory)")
    ap.add_argument("--model-name", action="append", default=[],
                    help="label for each --model, then each --csr-dump, in that order")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=21)
    args = ap.parse_args()

    if not args.csr_dump and not args.model:
        print("Nothing to measure: give --model (with --dump-tool) or --csr-dump.",
              file=sys.stderr)
        return 1
    if args.model and (not args.dump_tool or not args.dump_tool.exists()):
        print(f"--dump-tool not found: {args.dump_tool}", file=sys.stderr)
        return 1
    if not args.microbench or not args.microbench.exists():
        print(f"--microbench not found: {args.microbench}", file=sys.stderr)
        return 1

    git_commit = stamp.stamp(args.binary)
    card_slug = args.card.lower().replace(" ", "-")
    out_path = RESULTS_DIR / f"gpu-precision-{card_slug}-{git_commit}.csv"

    dump_dir = args.dump_dir or Path(tempfile.mkdtemp(prefix="sankhya-csr-"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    dumps = []
    for model in args.model:
        dump = dump_dir / (model.name.split(".")[0] + ".csr")
        made = subprocess.run([str(args.dump_tool), str(model), str(dump)],
                              capture_output=True, text=True, timeout=3600, check=True)
        print(f"{model.name}: {made.stdout.strip()}")
        dumps.append(dump)
    dumps += args.csr_dump

    rows_out = []
    for i, dump in enumerate(dumps):
        label = args.model_name[i] if i < len(args.model_name) else dump.stem
        parsed = run_microbench(args.microbench, dump, args.warmup, args.timed)
        with open(dump, "rb") as f:
            rows, cols, nnz = struct.unpack("<qqq", f.read(24))
        for arm, key in (("double-product", "double"), ("float-product", "float")):
            r = parsed[key]
            rows_out.append({
                "model": label, "rows": rows, "nonzeros": nnz, "precision_arm": arm,
                "iterations": "", "restarts": "", "refinement_rounds": "",
                "seconds_per_iteration": float(r["ax_seconds_median"]) +
                float(r["atx_seconds_median"]),
                "seconds_end_to_end_median": "", "seconds_end_to_end_min": "",
                "seconds_end_to_end_max": "", "final_residual_primal_double": "",
                "final_residual_dual_double": "", "final_residual_gap_double": "",
                "status": "product-pair-only", "verifier_verdict": "",
                "peak_device_memory_bytes": r["peak_bytes"], "git_commit": git_commit,
                "card": args.card,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
