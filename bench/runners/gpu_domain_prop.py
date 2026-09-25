#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Root domain propagation, CPU reference against the CUDA propagator, over model size (#510).

The issue's second acceptance item: "time against model size, CPU against GPU". Both
backends run the same synchronous rounds and return the same bounds (tests/unit/
test_domain_propagation.cpp holds them to that); this measures what each costs.

Models are generated here, integer knapsack-style rows over integer columns in [0, 9] with a
few continuous columns, so that propagation has work to do (several rounds, many bounds
tightened) at every size. For each size the solver is run with `gpu_domain_prop=true`,
presolve off, node limit 0 (the propagation happens before the first node), once with
`domain_prop_backend=cpu` and once with `auto` (the device), `--repeats` times each; the
median of the propagation time the solver itself logs is reported, which includes, on the
device, building the row-major copy and the transfers to and from the card.

A row of the CSV names the instance by its generator parameters and the sha256 of the MPS
file, the backend, the rounds and bounds tightened (which must agree between backends: a
row where they do not is marked), the median seconds, the commit the binary was built from
and the machine.

    python bench/runners/gpu_domain_prop.py --binary build_gpu/sankhya
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433, #589: the CSV names the commit the BINARY was built from)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

CSV_COLUMNS = [
    "instance", "sha256", "rows", "cols", "nnz", "backend", "status", "rounds", "tightened",
    "seconds", "repeats", "agrees_with_cpu", "git_commit", "machine", "gpu", "timestamp_utc",
]

SIZES = [1_000, 10_000, 100_000, 300_000, 1_000_000]

LINE = re.compile(r"Domain propagation \(#510, (GPU|CPU)\): (\d+) round\(s\), (\d+) bound\(s\) "
                  r"tightened in ([0-9.eE+-]+)s")


def write_model(path: Path, rows: int, seed: int) -> tuple[int, int, int]:
    """Knapsack-style rows over integer columns in [0, 9]: sum_j a_j x_j <= b with a_j in
    [1, 9] on 5 random columns, b a quarter of the row's maximum, so every row tightens its
    columns' upper bounds and the tightened bounds feed the next round through shared
    columns. One column in ten is continuous in [0, 9]. cols = rows."""
    rng = random.Random(seed)
    cols = rows
    entries: list[list[tuple[int, int]]] = [[] for _ in range(cols)]
    rhs = []
    for i in range(rows):
        chosen = rng.sample(range(cols), 5)
        total = 0
        for j in chosen:
            a = rng.randint(1, 9)
            entries[j].append((i, a))
            total += 9 * a
        rhs.append(max(1, total // 4))
    nnz = sum(len(e) for e in entries)
    with path.open("w", encoding="ascii") as f:
        f.write(f"NAME PROP{rows}\nROWS\n N obj\n")
        for i in range(rows):
            f.write(f" L r{i}\n")
        f.write("COLUMNS\n")
        f.write("    MARKER 'MARKER' 'INTORG'\n")
        for j in range(cols):
            if j % 10 == 9:
                continue
            f.write(f"    x{j} obj -1\n")
            for i, a in entries[j]:
                f.write(f"    x{j} r{i} {a}\n")
        f.write("    MARKER 'MARKER' 'INTEND'\n")
        for j in range(9, cols, 10):
            f.write(f"    x{j} obj -1\n")
            for i, a in entries[j]:
                f.write(f"    x{j} r{i} {a}\n")
        f.write("RHS\n")
        for i in range(rows):
            f.write(f"    rhs r{i} {rhs[i]}\n")
        f.write("BOUNDS\n")
        for j in range(cols):
            f.write(f" UP bnd x{j} 9\n")
        f.write("ENDATA\n")
    return rows, cols, nnz


def run_once(binary: Path, mps: Path, backend: str, time_limit: float):
    command = [str(binary), "solve", str(mps), "--time-limit", str(time_limit)]
    for option in ("log_to_console=true", "presolve=false", "gpu_domain_prop=true",
                   "node_limit=0", f"domain_prop_backend={backend}"):
        command += ["--option", option]
    out = subprocess.run(command, capture_output=True, text=True, check=False)
    match = LINE.search(out.stdout + out.stderr)
    if match is None:
        return None
    return {"engine": match.group(1), "rounds": int(match.group(2)),
            "tightened": int(match.group(3)), "seconds": float(match.group(4))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    version = subprocess.run([str(args.binary), "--version"], capture_output=True, text=True,
                             check=False).stdout
    gpu_match = re.search(r"GPU ([^)]*\))", version)
    gpu = gpu_match.group(1).strip() if gpu_match else "none"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows_out: list[dict] = []
    print(f"commit {commit}  machine {machine}  gpu {gpu}")
    print(f"{'rows':>9} {'backend':>7} {'rounds':>6} {'tightened':>10} {'median s':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            mps = Path(tmp) / f"prop{size}.mps"
            rows, cols, nnz = write_model(mps, size, seed=size)
            digest = hashlib.sha256(mps.read_bytes()).hexdigest()
            cpu_result = None
            for backend in ("cpu", "auto"):
                runs = [run_once(args.binary, mps, backend, args.time_limit)
                        for _ in range(args.repeats)]
                runs = [r for r in runs if r is not None]
                if not runs:
                    print(f"{rows:>9} {backend:>7}  no propagation line in the log")
                    continue
                engine = runs[0]["engine"]
                label = "cpu" if engine == "CPU" else "gpu"
                median = statistics.median(r["seconds"] for r in runs)
                first = runs[0]
                agrees = ""
                if label == "cpu":
                    cpu_result = first
                elif cpu_result is not None:
                    agrees = "yes" if (first["rounds"], first["tightened"]) == (
                        cpu_result["rounds"], cpu_result["tightened"]) else "NO"
                print(f"{rows:>9} {label:>7} {first['rounds']:>6} {first['tightened']:>10} "
                      f"{median:>10.6f} {agrees}")
                rows_out.append({
                    "instance": f"prop-knapsack-{size}", "sha256": digest, "rows": rows,
                    "cols": cols, "nnz": nnz, "backend": label, "status": "propagated",
                    "rounds": first["rounds"], "tightened": first["tightened"],
                    "seconds": f"{median:.6f}", "repeats": len(runs), "agrees_with_cpu": agrees,
                    "git_commit": commit, "machine": machine, "gpu": gpu,
                    "timestamp_utc": timestamp,
                })
    if not rows_out:
        print("no results")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"gpu-domain-prop-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
