#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the #482 A/B runner, pdhg_step_weight_ab.py. Pure Python: no solve - the
solver call is replaced by a recorder that returns synthetic results.

Pinned: each leg passes exactly its two switches and its seed, after any --solver-option so
one cannot turn a leg into another; each scheme its two; every solve is PDHG alone (pdhg_polish=false);
the CUDA engine asks for the device; and one run writes one row per instance, scheme and
leg and seed, with the iteration ratio to the default leg at the same seed.

    python bench/runners/test_pdhg_step_weight_ab.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netlib  # noqa: E402
import pdhg_step_weight_ab as ab  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def value_of(options: list[str], key: str) -> str | None:
    """The value the solver would see: the LAST --option for a key wins."""
    found = None
    for option in options:
        name, _, value = option.partition("=")
        if name == key:
            found = value
    return found


def test_leg_options() -> None:
    expected = {"default": ("false", "false"), "constant": ("true", "false"),
                "pid": ("false", "true"), "both": ("true", "true")}
    hostile = ["pdhg_constant_step=true", "pdhg_primal_weight_pid=true", "pdhg_pid_ki=0.1"]
    for leg, (constant, pid) in expected.items():
        for scheme, halpern in (("averaged", "false"), ("halpern", "true")):
            options = ab.leg_options("cpu", scheme, leg, 500, hostile)
            check(value_of(options, "pdhg_constant_step") == constant
                  and value_of(options, "pdhg_primal_weight_pid") == pid,
                  f"{scheme} {leg}: the leg's switches win over --solver-option")
            check(value_of(options, "pdhg_halpern") == halpern
                  and value_of(options, "pdhg_restart") == ("false" if halpern == "true"
                                                             else "true"),
                  f"{scheme} {leg}: the scheme's switches")
            check(value_of(options, "pdhg_polish") == "false"
                  and value_of(options, "algorithm") == "pdhg"
                  and value_of(options, "iteration_limit") == "500"
                  and value_of(options, "pdhg_pid_ki") == "0.1",
                  f"{scheme} {leg}: PDHG alone, the limit and the gains passed through")
            check(value_of(ab.leg_options("cpu", scheme, leg, 1, ["random_seed=9"], 4),
                           "random_seed") == "4", f"{scheme} {leg}: the seed wins")
    check(value_of(ab.leg_options("cuda", "averaged", "default", 1, []), "gpu") == "true",
          "the CUDA engine asks for the device")


def test_rows() -> None:
    iterations = {("averaged", "default"): 1000, ("averaged", "constant"): 1500,
                  ("averaged", "pid"): 800, ("averaged", "both"): 1200,
                  ("halpern", "default"): 900, ("halpern", "constant"): 600,
                  ("halpern", "pid"): 900, ("halpern", "both"): 450}

    def fake_run_one(binary, mps, time_limit, verify, options):
        scheme = "halpern" if value_of(options, "pdhg_halpern") == "true" else "averaged"
        constant = value_of(options, "pdhg_constant_step") == "true"
        pid = value_of(options, "pdhg_primal_weight_pid") == "true"
        leg = {(False, False): "default", (True, False): "constant",
               (False, True): "pid", (True, True): "both"}[(constant, pid)]
        seed = int(value_of(options, "random_seed"))
        return {"status": "optimal", "objective": -464.7531428571, "iterations":
                iterations[(scheme, leg)] + seed, "rows": 27, "columns": 32, "nonzeros": 83,
                "algorithm": "pdhg-cpu", "solver_seconds": 0.01, "wall_seconds": 0.02,
                "verified": True}

    original = netlib.run_one
    netlib.run_one = fake_run_one
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ab.csv"
            argv = sys.argv
            sys.argv = ["pdhg_step_weight_ab.py", "--binary", "sankhya", "--instances",
                        "afiro", "--seeds", "0", "3", "--out", str(out)]
            gpu = ab.gpu_description
            ab.gpu_description = lambda binary: ""
            stamp = netlib.git_commit
            netlib.git_commit = lambda binary=None: "abc1234"
            try:
                ab.main()
            finally:
                sys.argv = argv
                ab.gpu_description = gpu
                netlib.git_commit = stamp
            with out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
    finally:
        netlib.run_one = original
    check(len(rows) == 16, "one row per scheme, leg and seed", f"{len(rows)} rows")
    ratios = {(r["scheme"], r["leg"], r["seed"]): r["iteration_ratio_to_default"]
              for r in rows}
    check(ratios[("averaged", "constant", "0")] == "1.5000"
          and ratios[("halpern", "both", "0")] == "0.5000"
          and ratios[("averaged", "default", "3")] == "1.0000"
          and ratios[("averaged", "constant", "3")] == f"{1503 / 1003:.4f}",
          "the iteration ratio to the default leg at the same seed", str(ratios))
    check(all(r["git_commit"] == "abc1234" and r["independently_verified"] == "1"
              and r["instance_sha256"] for r in rows),
          "stamp, verdict and instance digest on every row")


def main() -> int:
    test_leg_options()
    test_rows()
    print(f"\n{'all passed' if FAILURES == 0 else f'{FAILURES} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
