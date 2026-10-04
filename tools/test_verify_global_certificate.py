#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for verify_global_certificate.py (#514).

Standalone, no test framework dependency - consistent with the rest of tools/. Run directly:

    python tools/test_verify_global_certificate.py

Two layers. The first tests the checker on its own, with no solver: its McCormick relaxation
holds at every point of a box (so it IS a relaxation of the model, whatever the solver builds),
and its Lagrangian is below the objective at every feasible point for ARBITRARY multipliers (so
a wrong multiplier cannot make it accept a bound that is too good). The second, when a built
solver is found (build/sankhya, or SANKHYA_BIN), checks certificates the solver writes for
Haverly's pooling models end to end and then tampers with them: a better bound claimed, a
child removed, an incumbent moved off the feasible set, a product dropped, every multiplier
erased. Each tampered file must be rejected.

Exit codes: 0 all tests passed, 1 at least one failed.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_global_certificate as vg  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


# min  x*y + z^2 + 0.5 x - z   s.t.  x + y + z <= 5 (cap),  x*y <= 3 (q),  x, y, z in [0, 2].
# QUADOBJ holds the lower triangle of Q in 0.5 x'Qx: (y, x) is the product x*y, and (z, z) = 2
# is z^2.
MODEL = """NAME small
ROWS
 N cost
 L cap
 L q
COLUMNS
 x cost 0.5
 x cap 1
 y cap 1
 z cost -1
 z cap 1
RHS
 rhs cap 5
 rhs q 3
BOUNDS
 UP bnd x 2
 UP bnd y 2
 UP bnd z 2
QUADOBJ
 y x 1
 z z 2
QCMATRIX q
 x y 0.5
 y x 0.5
ENDATA
"""
PRODUCTS = [(0, 1), (2, 2)]


def small_problem(tmp: Path) -> vg.Problem:
    path = tmp / "small.mps"
    path.write_text(MODEL)
    return vg.Problem(path, PRODUCTS)


def layer_one() -> None:
    print("the checker on its own")
    rng = random.Random(514)
    with tempfile.TemporaryDirectory() as tmp:
        problem = small_problem(Path(tmp))
        check(problem.rows_total == problem.m + 8, "four relaxation rows per product",
              f"{problem.rows_total} rows for {problem.m} model rows and 2 products")

        def point_in(lo, hi):
            return [lo[j] + (hi[j] - lo[j]) * Fraction(rng.randint(0, 1000), 1000)
                    for j in range(problem.n)]

        def values(x):
            return x + [x[a] * x[b] for a, b in problem.products]

        # 1. The relaxation holds at every point of every box it is built over.
        violated = 0
        for _ in range(300):
            lo = [Fraction(rng.randint(0, 100), 100) for _ in range(problem.n)]
            hi = [l + Fraction(rng.randint(0, 100), 100) for l in lo]
            col = values(point_in(lo, hi))
            for r in range(problem.rows_total):
                entries, rl, ru = problem.row(r, lo, hi)
                activity = sum(a * col[c] for c, a in entries.items())
                if activity < rl or activity > ru:
                    violated += 1
            clo, chi = problem.column_bounds(lo, hi)
            for c, v in enumerate(col):
                if v < clo[c] or v > chi[c]:
                    violated += 1
        check(violated == 0, "the McCormick rows and product bounds hold at every sampled point",
              f"{violated} violations in 300 boxes")

        # 2. The Lagrangian is below the objective at every point the relaxation allows, for any
        #    multipliers whatever (weak duality), so a tampered y cannot make a bound too good.
        too_good = 0
        usable = 0
        for _ in range(300):
            lo = [Fraction(rng.randint(0, 100), 100) for _ in range(problem.n)]
            hi = [l + Fraction(rng.randint(1, 100), 100) for l in lo]
            y = {r: Fraction(rng.randint(-20, 20), 4) for r in range(problem.rows_total)
                 if rng.random() < 0.6}
            bound = problem.lagrangian(y, lo, hi, with_cost=True)
            if bound is None:
                continue
            usable += 1
            for _ in range(20):
                x = point_in(lo, hi)
                col = values(x)
                feasible = all(
                    problem.row(r, lo, hi)[1] <= sum(a * col[c] for c, a in
                                                     problem.row(r, lo, hi)[0].items())
                    <= problem.row(r, lo, hi)[2] for r in range(problem.m))
                if not feasible:
                    continue
                objective = problem.offset + sum(c * col[j] for j, c in enumerate(problem.cost))
                for p, c in problem.objective_products.items():
                    objective += c * col[problem.n + p]
                if bound > objective:
                    too_good += 1
        check(usable > 100 and too_good == 0,
              "a Lagrangian at arbitrary multipliers never exceeds the objective",
              f"{too_good} violations over {usable} multiplier sets")

        # 3. A product the model has and the certificate does not list is refused.
        path = Path(tmp) / "small.mps"
        try:
            vg.Problem(path, [(0, 1)])
            check(False, "an unlisted product is rejected")
        except vg.Rejected as why:
            check("does not list" in str(why), "an unlisted product is rejected", str(why)[:70])


def find_binary() -> Path | None:
    env = os.environ.get("SANKHYA_BIN")
    if env and Path(env).exists():
        return Path(env)
    root = HERE.parent
    for candidate in ("build/sankhya", "build/sankhya.exe", "build-release/sankhya",
                      "build-release/sankhya.exe"):
        if (root / candidate).exists():
            return root / candidate
    return None


def run_checker(model: Path, cert: Path) -> tuple[int, str]:
    result = subprocess.run([sys.executable, str(HERE / "verify_global_certificate.py"),
                             str(model), str(cert)], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def layer_two() -> None:
    binary = find_binary()
    if binary is None:
        print("end to end: no built solver (build/sankhya or SANKHYA_BIN); skipped")
        return
    root = HERE.parent
    # haverly1_q needs Farkas proofs (empty boxes), so every kind of node is checked and tampered.
    models = [root / "data" / "pooling" / name
              for name in ("haverly1_p.mps", "haverly2_p.mps", "haverly1_q.mps")]
    for model in models:
        if not model.exists():
            continue
        print(f"end to end: {model.name}")
        with tempfile.TemporaryDirectory() as tmp:
            cert_path = Path(tmp) / "cert.json"
            subprocess.run([str(binary), "solve", str(model), "--option", "nonconvex=global",
                            "--option", f"write_certificate={cert_path}",
                            "--option", "log_to_console=false"], capture_output=True, text=True)
            if not cert_path.exists():
                check(False, "the solver wrote a certificate")
                continue
            code, out = run_checker(model, cert_path)
            check(code == 0 and "VERIFIED" in out, "the certificate is verified", out.strip()[-120:])
            cert = json.loads(cert_path.read_text())

            def tampered(name: str, change, expect_code: int = 1) -> None:
                copy = json.loads(json.dumps(cert))
                change(copy)
                path = Path(tmp) / "tampered.json"
                path.write_text(json.dumps(copy))
                code, out = run_checker(model, path)
                check(code == expect_code, name, f"exit {code}: {out.strip().splitlines()[0][:90]}")

            def claim_better(c):
                c["lower_bound_min_form"] = float(c["lower_bound_min_form"]) + 1000.0

            def drop_a_leaf(c):
                split = {n["id"] for n in c["nodes"] if "branch" in n}
                leaf = next(n for n in c["nodes"] if n["parent"] in split and "branch" not in n)
                c["nodes"].remove(leaf)

            def move_incumbent(c):
                c["incumbent"][0] = float(c["incumbent"][0]) + 1e3

            def drop_a_product(c):
                c["products"] = c["products"][1:]

            def erase_multipliers(c):
                for n in c["nodes"]:
                    n.pop("proof", None)
                    n.pop("y", None)

            tampered("a better bound claimed than the tree proves", claim_better)
            tampered("a child removed from a split node", drop_a_leaf)
            tampered("an incumbent moved off the feasible set", move_incumbent)
            tampered("a product dropped from the list", drop_a_product)
            tampered("every multiplier erased", erase_multipliers)
            farkas = [n for n in cert["nodes"] if n.get("proof") == "farkas"]
            if farkas:
                def break_farkas(c):
                    target = next(n for n in c["nodes"] if n.get("proof") == "farkas")
                    target["y"] = [[row, 0.0] for row, _ in target["y"]]
                tampered("a Farkas vector zeroed", break_farkas)
            else:
                print("  [skip] no Farkas node in this certificate")


def main() -> int:
    layer_one()
    layer_two()
    print(f"{'ALL PASSED' if FAILURES == 0 else str(FAILURES) + ' FAILED'}")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
