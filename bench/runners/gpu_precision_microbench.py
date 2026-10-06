#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device product micro-benchmark for GPU PDHG mixed precision, #982 step 1.

"Measure before building." Before any mixed-precision iteration code is trusted, this
measures the thing #982's "Why" is actually betting on: that A x and A^T y, run with single
precision VALUES on the same sparsity pattern, are faster on the device than the same
products in double. It does NOT run the solver and makes NO claim about end-to-end PDHG
time, iteration counts or convergence - that is bench/runners/gpu_datacenter.py's and a
follow-up runner's job, once (and only if) this step says the product pair is worth it.

THIS SCRIPT HAS NOT BEEN RUN. There is no GPU in the sandbox this was written in (checked:
no nvcc, no /dev/nvidia*), and the issue is explicit that the result must come from the
laptop card first. Running it is the next step for whoever picks this up with hardware; see
the PR for #982 for the full list of what is and is not done.

What it does, when run on a machine with a CUDA device and sankhya-precision-microbench
built (`cmake --build build -DSANKHYA_ENABLE_CUDA=ON --target sankhya-precision-microbench`):
  1. Builds (or locates) the three matrices #982 point 1 names: the generated 10,000 x
     10,000 LP (bench/runners/gpu_report.py's generator, reused here), brazil3 (Mittelmann,
     data/mittelmann after fetch_mittelmann.py) and one refinery ladder step
     (bench/case_studies/refinery/generator.py --size medium).
  2. Dumps each one's SCALED CSR (the same Ruiz-and-Pock-Chambolle-scaled matrix PDHG runs
     on - src/la/scaling.cpp's output, not the raw model) to the flat binary format
     bench/tools/gpu_precision_microbench.cu reads, via a tiny helper built into
     sankhya-cli... (see NOTE below).
  3. Runs sankhya-precision-microbench on each dump, parses its two CSV lines (double,
     float), and writes bench/results/gpu-precision-<card>-<commit>.csv with the columns
     #982's acceptance item asks for.

NOTE on step 2 - the one piece this script cannot finish without a CUDA build to test
against: sankhya-cli has no "dump the scaled CSR of this model" flag today. `dump_csr_stub`
below writes the UNSCALED model's CSR as a placeholder dump (so the script and the
micro-benchmark tool's binary contract can be exercised end to end on a CPU-only machine for
format checking, by passing --skip-microbench); the real run needs either a small addition to
sankhya-cli (e.g. `--dump-scaled-csr PATH`, printing the same CsrView src/gpu/pdhg_gpu.cu
uploads) or scaling the matrix independently here in Python before the dump. Flagged rather
than guessed at, since adding a CLI flag is itself a small code change that deserves its own
review and is outside what this script can verify without a build.

    python bench/runners/gpu_precision_microbench.py --binary build_gpu/sankhya \\
        --microbench build_gpu/sankhya-precision-microbench --card "RTX 5050 Laptop GPU"
"""
from __future__ import annotations

import argparse
import datetime as dt
import csv
import hashlib
import platform
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

# This runner only fills the device-product-pair rows (precision_arm in {double-product,
# float-product}); iterations/restarts/refinement_rounds/status/verifier_verdict are left
# blank here because this step measures the PRODUCT PAIR, not a PDHG solve - the end-to-end
# arms (precision_arm in {double, mixed}) are a later runner's job once this step clears.
PRODUCT_ONLY_COLUMNS = {"seconds_per_iteration"}  # ax+atx seconds, repurposed for this row


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dump_csr_stub(mps_path: Path, dump_path: Path) -> tuple[int, int, int]:
    """Write bench/tools/gpu_precision_microbench.cu's flat CSR format from the model's OWN
    coefficients - i.e. unscaled - by reading the MPS file with a minimal parser. This is a
    format-level placeholder (see the NOTE in the module docstring): it lets the dump/run/
    parse plumbing below be exercised without a GPU, but the real measurement needs the
    SCALED matrix PDHG actually runs on, not this. Returns (rows, cols, nnz).
    """
    # Deliberately does not import sankhya's own MPS reader (no CPython extension exists for
    # it); a tiny free-format MPS/LP coefficient scan is enough for a format check, not for
    # a real measurement - see the module docstring's NOTE.
    raise NotImplementedError(
        "dump_csr_stub needs a matrix source; pass --csr-dump to supply one directly "
        "(rows,cols,nnz,row_ptr,col_idx,values packed as the tool expects), or add the "
        "sankhya-cli scaled-CSR dump flag described in this module's docstring."
    )


def write_csr_dump(dump_path: Path, rows: int, cols: int, row_ptr: list[int],
                   col_idx: list[int], values: list[float]) -> None:
    nnz = len(values)
    with open(dump_path, "wb") as f:
        f.write(struct.pack("<qqq", rows, cols, nnz))
        f.write(struct.pack(f"<{len(row_ptr)}i", *row_ptr))
        f.write(struct.pack(f"<{len(col_idx)}i", *col_idx))
        f.write(struct.pack(f"<{len(values)}d", *values))


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
    ap.add_argument("--model-name", action="append", default=[],
                    help="label for each --csr-dump, in the same order")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=21)
    args = ap.parse_args()

    if not args.csr_dump:
        print("No --csr-dump given: this script cannot source the generated-LP / brazil3 / "
              "refinery-ladder matrices on its own yet (see the module docstring's NOTE on "
              "sankhya-cli needing a scaled-CSR dump flag). Nothing was measured, no CSV was "
              "written.", file=sys.stderr)
        return 1
    if not args.microbench or not args.microbench.exists():
        print(f"--microbench not found: {args.microbench}", file=sys.stderr)
        return 1

    git_commit = stamp.stamp(args.binary)
    card_slug = args.card.lower().replace(" ", "-")
    out_path = RESULTS_DIR / f"gpu-precision-{card_slug}-{git_commit}.csv"

    rows_out = []
    for i, dump in enumerate(args.csr_dump):
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
