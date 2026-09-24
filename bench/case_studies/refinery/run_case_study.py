#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the multi-period refinery case study at three sizes and collect results (#517).

For each size (small / medium / large) this script:
  1. Generates an LP .mps file using generator.py.
  2. Generates a MILP .mps file (with binary crude-purchase decisions).
  3. Solves both with the sankhya CLI (if --sankhya is given or `sankhya` is on PATH).
  4. Records rows / cols / solve time / objective in a CSV.

SYNTHETIC DATA NOTICE
---------------------
All .mps files produced here contain synthetic data only.
See bench/case_studies/refinery/generator.py for the disclaimer.

Usage
-----
    # Generate files only (no solver needed):
    python bench/case_studies/refinery/run_case_study.py --generate-only

    # Generate and solve (requires a built sankhya binary):
    python bench/case_studies/refinery/run_case_study.py --sankhya build/sankhya

    # Write results to a specific CSV:
    python bench/case_studies/refinery/run_case_study.py \\
        --sankhya build/sankhya --out bench/case_studies/refinery/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generator  # noqa: E402 (local import after sys.path setup)

# Default seed for reproducibility
SEED = 42

# Column order for the results CSV
CSV_FIELDS = ["size", "type", "periods", "crudes", "products", "units",
              "rows", "cols", "nonzeros", "binary",
              "status", "objective", "solve_s"]

# Size configurations: (label, size_key, lp/milp)
RUNS: list[tuple[str, str]] = [
    ("small",  "small"),
    ("medium", "medium"),
    ("large",  "large"),
]


def _find_sankhya(hint: str | None) -> str | None:
    if hint:
        return hint
    # Relative to the repository, not the working directory, so the driver finds the
    # build from anywhere (review of #633); SANKHYA_BIN wins, as for the other runners.
    env = os.environ.get("SANKHYA_BIN")
    if env and Path(env).exists():
        return env
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "build" / "sankhya.exe",
        root / "build" / "sankhya",
        root / "build" / "Release" / "sankhya.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    found = shutil.which("sankhya")
    return found


def _mps_stats(mps_path: Path) -> dict[str, int]:
    """Count rows, cols, nonzeros and binary columns in the MPS file."""
    rows = cols = nnz = binary = 0
    in_rows = in_cols = in_int = False
    with mps_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("ROWS"):
                in_rows, in_cols, in_int = True, False, False
            elif stripped.startswith("COLUMNS"):
                in_rows, in_cols, in_int = False, True, False
            elif stripped.startswith(("RHS", "BOUNDS", "RANGES", "ENDATA")):
                in_rows = in_cols = False
            elif in_rows and stripped and stripped[0] not in ("N", "*"):
                rows += 1
            elif in_cols and stripped:
                if "INTORG" in stripped:
                    in_int = True
                elif "INTEND" in stripped:
                    in_int = False
                elif not stripped.startswith("*"):
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[1] not in ("COST",) or True:
                        # Each data line in COLUMNS contributes one (col, row) nonzero
                        # and a new column name when it first appears
                        pass
    # Re-parse properly
    rows = cols = nnz = binary = 0
    seen_cols: set[str] = set()
    seen_int: set[str] = set()
    in_rows = in_cols = in_int_block = False
    with mps_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            if s.startswith("ROWS"):
                in_rows, in_cols = True, False
            elif s.startswith("COLUMNS"):
                in_rows, in_cols = False, True
            elif s.startswith(("RHS", "BOUNDS", "RANGES", "ENDATA")):
                in_rows = in_cols = False
            elif in_rows:
                sense = s.split()[0]
                if sense in ("E", "L", "G"):
                    rows += 1
            elif in_cols:
                if "INTORG" in s:
                    in_int_block = True
                elif "INTEND" in s:
                    in_int_block = False
                else:
                    parts = s.split()
                    if len(parts) >= 3:
                        col_name = parts[0]
                        if col_name not in seen_cols:
                            seen_cols.add(col_name)
                            cols += 1
                            if in_int_block:
                                seen_int.add(col_name)
                        nnz += 1  # one entry per line
    binary = len(seen_int)
    return {"rows": rows, "cols": cols, "nonzeros": nnz, "binary": binary}


def _solve(sankhya: str, mps_path: Path) -> dict[str, object]:
    """Run sankhya on the MPS file; return status, objective and elapsed time."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        stats_path = Path(tf.name)
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            [sankhya, "solve", str(mps_path), "--stats", str(stats_path)],
            capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter() - t0
        status = "unknown"
        objective: float | str = ""
        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text(encoding="utf-8"))
                status = data.get("status", "unknown")
                objective = data.get("objective", "")
            except (json.JSONDecodeError, KeyError):
                pass
        if not status or status == "unknown":
            # Fall back to parsing stdout
            for line in result.stdout.splitlines():
                if "optimal" in line.lower():
                    status = "optimal"
                elif "infeasible" in line.lower():
                    status = "infeasible"
                elif "objective" in line.lower():
                    try:
                        objective = float(line.split()[-1])
                    except ValueError:
                        pass
        return {"status": status, "objective": objective, "solve_s": round(elapsed, 3)}
    finally:
        stats_path.unlink(missing_ok=True)


def run(sankhya: str | None, out_csv: Path | None, generate_only: bool) -> int:
    rows_out: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="refinery_case_") as tmpdir:
        tmp = Path(tmpdir)
        for label, size_key in RUNS:
            for milp_flag in (False, True):
                suffix = "milp" if milp_flag else "lp"
                mps = tmp / f"refinery_{label}_{suffix}.mps"
                dims = dict(generator.SIZE_PRESETS[size_key])
                dims["seed"] = SEED

                print(f"\n--- {label} {suffix.upper()} ---")
                inst = generator.build_lp(**dims)
                generator.verify_lp(inst)
                if milp_flag:
                    generator.extend_to_milp(inst, dims["periods"], dims["crudes"])
                generator.write_mps(
                    inst, mps, milp=milp_flag,
                    periods=dims["periods"], crudes=dims["crudes"],
                    products=dims["products"], units=dims["units"],
                    seed=dims["seed"])

                stats = _mps_stats(mps)
                row: dict = {
                    "size": label, "type": suffix,
                    "periods": dims["periods"], "crudes": dims["crudes"],
                    "products": dims["products"], "units": dims["units"],
                    **stats,
                    "status": "", "objective": "", "solve_s": "",
                }

                if not generate_only and sankhya:
                    print(f"  solving with {sankhya} ...")
                    sol = _solve(sankhya, mps)
                    row.update(sol)
                    print(f"  status={sol['status']}  obj={sol['objective']}"
                          f"  time={sol['solve_s']}s")
                elif not generate_only:
                    print("  (no sankhya binary; skipping solve)")

                rows_out.append(row)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nresults written to {out_csv}")
    else:
        # Print a simple table to stdout
        print("\n" + "  ".join(f"{h:>12}" for h in CSV_FIELDS))
        for r in rows_out:
            print("  ".join(f"{str(r.get(h, '')):>12}" for h in CSV_FIELDS))

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sankhya", metavar="PATH",
                        help="path to the sankhya binary (default: auto-detect)")
    parser.add_argument("--generate-only", action="store_true",
                        help="generate .mps files without solving")
    parser.add_argument("--out", type=Path, metavar="CSV",
                        help="write results table to this CSV file")
    args = parser.parse_args(argv)

    sankhya = _find_sankhya(args.sankhya)
    if not args.generate_only and sankhya is None:
        print("note: no sankhya binary found; generating files only", file=sys.stderr)

    return run(sankhya=sankhya, out_csv=args.out, generate_only=args.generate_only)


if __name__ == "__main__":
    sys.exit(main())
