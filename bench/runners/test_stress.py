#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The stress set's generator and grade (#762). Hermetic: no solver, no network.

1. The adversarial instances regenerate to the sha256 values committed in
   data/stress/reference.json (the scaled Netlib files need data/netlib/, which CI has; they
   are checked too).
2. Each adversarial answer is what the generator says: the known optimum is feasible and
   attains the stated objective, checked with the independent verifier's own MPS reader.
3. grade() calls a verdict wrong, correct or failed by the rule stress.py documents.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stress  # noqa: E402
import stress_doc  # noqa: E402
import stress_instances as gen  # noqa: E402
from verify_solution_mps import parse_mps  # noqa: E402  (stress_instances put tools/ on the path)

FAILURES = 0


def check(ok: bool, name: str, detail: str = "") -> None:
    global FAILURES
    FAILURES += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))


def objective_at(model, x: list[float]) -> float:
    return sum(c * v for c, v in zip(model.col_cost, x))


def feasible(model, x: list[float]) -> bool:
    activity = [0.0] * model.num_rows
    for j, entries in enumerate(model.entries):
        for i, v in entries:
            activity[i] += v * x[j]
    return all(model.row_lower[i] - 1e-9 <= a <= model.row_upper[i] + 1e-9
               for i, a in enumerate(activity))


def main() -> int:
    committed = json.loads((gen.OUT / "reference.json").read_text())["instances"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name, family, _, model, status, objective in gen.adversarial():
            path = out / f"{name}.mps"
            gen.write_mps(model, path)
            check(committed.get(name, {}).get("sha256") == gen.sha256(path),
                  f"{name} regenerates to its committed sha256")
            check(committed[name]["expected_status"] == status, f"{name} status is {status}")
        # Klee-Minty: x = b_n e_n attains -b_n, read back through the verifier's reader.
        km = parse_mps(out / "klee_minty_40.mps")
        x = [0.0] * 39 + [float(5 ** 40)]
        check(feasible(km, x) and objective_at(km, x) == float(committed["klee_minty_40"]
                                                               ["expected_objective"]),
              "klee_minty_40's stated optimum is attained at x = 5^40 e_40")
        beale = parse_mps(out / "beale.mps")
        check(feasible(beale, [0.04, 0.0, 1.0, 0.0])
              and abs(objective_at(beale, [0.04, 0.0, 1.0, 0.0]) + 0.05) < 1e-15,
              "Beale's optimum -1/20 is attained at x4 = 1/25, x6 = 1")
        sb = parse_mps(out / "singular_basis_50.mps")
        check(feasible(sb, [1.0, 1.0]) and objective_at(sb, [1.0, 1.0]) ==
              float(committed["singular_basis_50"]["expected_objective"]),
              "singular_basis_50's optimum is attained at (1, 1)")
        if (gen.NETLIB / "afiro.mps").exists():
            for name, _, _, model, _, _ in gen.scaled_netlib(["afiro"]):
                gen.write_mps(model, out / "s.mps")
                check(gen.sha256(out / "s.mps") == committed[name]["sha256"],
                      "scaled_afiro regenerates to its committed sha256")

    g = stress.grade
    check(g("optimal", -5.0, "optimal", -5.0 + 1e-7, 1, "")[0] == "correct", "a match is correct")
    check(g("optimal", -5.0, "optimal", -4.0, 1, "")[0] == "wrong", "an objective off is wrong")
    check(g("optimal", -5.0, "optimal", -5.0, 0, "row")[0] == "wrong", "a rejected file is wrong")
    check(g("optimal", -5.0, "infeasible", None, 1, "")[0] == "wrong",
          "infeasible on a feasible instance is wrong, certificate or not")
    check(g("infeasible", None, "optimal", 3.0, 1, "")[0] == "correct",
          "a verified point on a thin-infeasible instance is correct at the tolerance")
    check(g("unbounded", None, "optimal", 3.0, 1, "")[0] == "wrong", "optimal when unbounded")
    check(g("optimal", -5.0, "numerical_error", None, "", "")[0] == "failed", "no verdict fails")
    check(g("infeasible", None, "infeasible_or_unbounded", None, 1, "")[0] == "failed",
          "infeasible_or_unbounded is not a verdict")
    check("_No `stress-*.csv`" in stress_doc.section(None), "the doc section without a CSV")
    print("ALL TESTS PASSED" if FAILURES == 0 else f"{FAILURES} check(s) FAILED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
