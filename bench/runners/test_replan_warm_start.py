#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A warm start on a morning the plant cannot meet reaches the cold verdict (#883).

The medium refinery LP (bench/case_studies/refinery/generator.py, seed 1) is solved cold and
its .sol kept as "yesterday". Each test morning moves every crude price by up to 10 % and
every delivery commitment by up to 5 %, the commitments WITHOUT their market caps, so the
morning is infeasible. `sankhya solve --warm-start yesterday.sol` must then say `infeasible`
with a Farkas certificate `tools/verify_solution.py` accepts. At 5a39fb2, 13 of 40 such
mornings ended `numerical_error` warm while the cold solve proved infeasibility in presolve;
the four seeds below were among them on a Windows laptop.

Needs a built sankhya-cli: SANKHYA_BINARY, or build/sankhya[.exe]. Skips without one.

    SANKHYA_BINARY=build/sankhya python3 bench/runners/test_replan_warm_start.py
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "bench" / "case_studies" / "refinery" / "generator.py"
VERIFY = REPO_ROOT / "tools" / "verify_solution.py"
SEEDS = (2, 3, 6, 12)


def binary() -> Path | None:
    named = os.environ.get("SANKHYA_BINARY")
    candidates = [Path(named)] if named else []
    candidates += [REPO_ROOT / "build" / "sankhya", REPO_ROOT / "build" / "sankhya.exe"]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if path.is_file():
            return path
    return None


def infeasible_morning(source: Path, out: Path, seed: int) -> None:
    """Prices by a factor in [0.9, 1.1], commitments (`COMMIT_*` right-hand sides) in
    [0.95, 1.05] with the markets left where they were - the edit #883 was found with."""
    rng = random.Random(seed)
    lines, section = [], ""
    for line in source.read_text().splitlines():
        if line and not line[0].isspace():
            section = line.split()[0]
        elif section == "COLUMNS":
            tokens = line.split()
            if len(tokens) == 3 and tokens[1] == "COST" and tokens[0].startswith("BUY_"):
                line = f"    {tokens[0]}  COST  {float(tokens[2]) * rng.uniform(0.9, 1.1):.9g}"
        elif section == "RHS":
            tokens = line.split()
            if len(tokens) == 3 and tokens[1].startswith("COMMIT_"):
                line = f"    RHS  {tokens[1]}  {float(tokens[2]) * rng.uniform(0.95, 1.05):.9g}"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n")


def status_of(stdout: str) -> str:
    match = re.search(r"^status\s+(\S+)", stdout, re.M)
    return match.group(1) if match else "?"


def main() -> int:
    exe = binary()
    if exe is None:
        print("SKIPPED: no sankhya-cli binary (set SANKHYA_BINARY or build into build/)")
        return 0
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        day0, day0_sol = work / "day0.mps", work / "day0.sol"
        subprocess.run([sys.executable, str(GENERATOR), "--size", "medium", "--seed", "1",
                        "--out", str(day0)], check=True, capture_output=True)
        cold = subprocess.run([str(exe), "solve", str(day0), "--write-sol", str(day0_sol),
                               "--option", "log_to_console=false"],
                              capture_output=True, text=True)
        if status_of(cold.stdout) != "optimal":
            print(f"[FAIL] yesterday's plan is not optimal:\n{cold.stdout}{cold.stderr}")
            return 1
        for seed in SEEDS:
            today, today_sol = work / f"day_{seed}.mps", work / f"day_{seed}.sol"
            infeasible_morning(day0, today, seed)
            warm = subprocess.run([str(exe), "solve", str(today), "--warm-start", str(day0_sol),
                                   "--write-sol", str(today_sol), "--option",
                                   "log_to_console=false"], capture_output=True, text=True)
            status = status_of(warm.stdout)
            check = subprocess.run([sys.executable, str(VERIFY), str(today), str(today_sol),
                                    "--quiet"], capture_output=True, text=True)
            route = "cold fallback" if "#883" in warm.stdout else "warm"
            ok = status == "infeasible" and check.returncode == 0
            failures += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] morning {seed}: {status} ({route}), verifier "
                  f"{'accepts' if check.returncode == 0 else 'rejects'} the certificate")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} morning(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
