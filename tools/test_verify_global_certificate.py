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
ZERO_ = Fraction(0)


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


# The shape of a pooling Q-formulation, cut down: proportions q1 + q2 = 1 (sumq), a flow f
# under a capacity (cap), and the path flows v_i = q_i * f as quadratic rows (path1, path2).
RLT_MODEL = """NAME rlt
ROWS
 N cost
 E sumq
 L cap
 E path1
 E path2
COLUMNS
 q1 sumq 1
 q2 sumq 1
 f cost -1
 f cap 1
 v1 cost 2
 v1 path1 1
 v2 cost 1
 v2 path2 1
RHS
 rhs sumq 1
 rhs cap 8
BOUNDS
 UP bnd q1 1
 UP bnd q2 1
 UP bnd f 10
 UP bnd v1 10
 UP bnd v2 10
QCMATRIX path1
 q1 f -0.5
 f q1 -0.5
QCMATRIX path2
 q2 f -0.5
 f q2 -0.5
ENDATA
"""
# Columns q1 0, q2 1, f 2, v1 3, v2 4; rows sumq 0, cap 1, path1 2, path2 3.
RLT_PRODUCTS = [(0, 2), (1, 2)]
RLT_ROWS = [[0, 2, "eq"], [1, 0, "ul"], [1, 0, "uu"], [1, 1, "ul"], [1, 1, "uu"]]


def layer_rlt() -> None:
    print("the checker's RLT rows")
    rng = random.Random(1995)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rlt.mps"
        path.write_text(RLT_MODEL)
        problem = vg.Problem(path, RLT_PRODUCTS, RLT_ROWS)
        first = problem.m + 4 * len(RLT_PRODUCTS)
        check(problem.rows_total == first + len(RLT_ROWS), "one relaxation row per rlt entry",
              f"{problem.rows_total} rows")
        # The PQ-formulation's rows, read back: sum_i w_i = f, and w_1 <= 8 q1 (q1's lower
        # bound is 0).
        entries, rl, ru = problem.rlt_row(0, problem.col_lower, problem.col_upper)
        check(entries == {5: 1, 6: 1, 2: -1} and rl == ru == 0, "sumq times f is w1 + w2 = f",
              str(entries))
        entries, rl, ru = problem.rlt_row(1, problem.col_lower, problem.col_upper)
        check(entries == {5: 1, 0: -8} and rl == -vg.INF and ru == 0, "cap times q1 is w1 <= 8 q1",
              str(entries))

        # Every RLT row holds at every point of the model (rows sumq and cap) in every box.
        violated = 0
        points = 0
        for _ in range(300):
            lo = [Fraction(rng.randint(0, 50), 100) for _ in range(2)]
            lo.append(Fraction(rng.randint(0, 9)))
            hi = [lo[0] + Fraction(rng.randint(0, 50), 100),
                  lo[1] + Fraction(rng.randint(0, 50), 100),
                  lo[2] + Fraction(rng.randint(0, 10 - int(lo[2])))]
            lo += [ZERO_, ZERO_]
            hi += [Fraction(10), Fraction(10)]
            for _ in range(20):
                q1 = lo[0] + (hi[0] - lo[0]) * Fraction(rng.randint(0, 100), 100)
                q2 = 1 - q1
                f = lo[2] + (hi[2] - lo[2]) * Fraction(rng.randint(0, 100), 100)
                if not (lo[1] <= q2 <= hi[1]) or f > 8:
                    continue
                points += 1
                col = [q1, q2, f, q1 * f, q2 * f, q1 * f, q2 * f]
                for r in range(first, problem.rows_total):
                    entries, rl, ru = problem.row(r, lo, hi)
                    activity = sum(a * col[c] for c, a in entries.items())
                    if activity < rl or activity > ru:
                        violated += 1
        check(points > 100 and violated == 0, "every RLT row holds at every sampled model point",
              f"{violated} violations at {points} points")

        def rejected(entry, phrase: str, name: str) -> None:
            try:
                vg.Problem(path, RLT_PRODUCTS, [entry])
                check(False, name)
            except vg.Rejected as why:
                check(phrase in str(why), name, str(why)[:80])

        rejected([2, 0, "ll"], "not a linear row", "an RLT row over a quadratic row is rejected")
        rejected([1, 2, "ul"], "does not list",
                 "an RLT row needing an unlisted product is rejected")
        rejected([1, 0, "eq"], "not an equality", "an inequality used as an equality is rejected")
        rejected([1, 0, "lu"], "no such side", "a row side the row does not have is rejected")


# haverly1 in the P-formulation (data/pooling/haverly1_p.mps): columns f_1_4 f_2_4 f_3_5 f_3_6
# f_4_5 f_4_6 p_4_1 (0..6), rows src_1 src_2 src_3 dem_5 dem_6 bal_4 cap_4 pq_4_1 tq_5_1 tq_6_1
# (0..9). The lift adds q_1 = 7, q_2 = 8 and six rows (10..15).
HAVERLY1_LIFT = [{"balance": 5, "qualities": [7], "capacities": [6]}]
HAVERLY1_LIFTED_PRODUCTS = [(4, 6), (5, 6), (4, 7), (5, 7), (4, 8), (5, 8)]


def layer_lift() -> None:
    print("the checker's pool lift (global_pool_lift)")
    path = HERE.parent / "data" / "pooling" / "haverly1_p.mps"
    if not path.exists():
        print("  [skip] data/pooling/haverly1_p.mps not found")
        return
    problem = vg.Problem(path, HAVERLY1_LIFTED_PRODUCTS, [], HAVERLY1_LIFT)
    check(problem.n == 9 and problem.m == 16, "two proportion columns and six rows appended",
          f"{problem.m} x {problem.n}")
    check(problem.col_lower[7:] == [0, 0] and problem.col_upper[7:] == [1, 1],
          "each proportion is in [0, 1]")
    # Every added row holds, exactly, at random points of the pool's own rows (q_i = f_i / Y),
    # and at Y = 0 with any q on the simplex.
    rng = random.Random(514)
    violated = 0
    for trial in range(200):
        empty = trial % 10 == 0
        f14 = ZERO_ if empty else Fraction(rng.randint(0, 1000), 10)
        f24 = ZERO_ if empty else Fraction(rng.randint(0, 1000), 10)
        y = f14 + f24
        f45 = y * Fraction(rng.randint(0, 100), 100)
        x = [f14, f24, ZERO_, ZERO_, f45, y - f45,
             (3 * f14 + f24) / y if y else Fraction(rng.randint(100, 300), 100)]
        q1 = f14 / y if y else Fraction(rng.randint(0, 100), 100)
        x += [q1, 1 - q1]
        for r in range(10, 16):
            activity = sum(a * x[j] for j, a in problem.linear[r])
            activity += sum(c * x[problem.products[p][0]] * x[problem.products[p][1]]
                            for p, c in problem.row_products[r].items())
            if activity < problem.row_lower[r] or activity > problem.row_upper[r]:
                violated += 1
    check(violated == 0, "every lifted row holds at every sampled point of the pool",
          f"{violated} violations")

    def rejected(pool, phrase: str, name: str) -> None:
        try:
            vg.Problem(path, HAVERLY1_LIFTED_PRODUCTS, [], [pool])
            check(False, name)
        except vg.Rejected as why:
            check(phrase in str(why), name, str(why)[:80])

    rejected({"balance": 6, "qualities": [7], "capacities": []}, "not a balance row",
             "a capacity row named as the balance row is rejected")
    rejected({"balance": 5, "qualities": [8], "capacities": []}, "not a quality row",
             "a terminal row named as a quality row is rejected")
    rejected({"balance": 5, "qualities": [7], "capacities": [0]}, "not a capacity",
             "a source row named as a capacity row is rejected")
    rejected({"balance": 5, "qualities": [7], "capacities": [16]}, "does not have",
             "a row past the model file is rejected")


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
            # global_rlt=false: the plain McCormick tree, so haverly1_q still needs Farkas proofs
            # (with the RLT rows it closes at the root); the RLT rows are checked below.
            subprocess.run([str(binary), "solve", str(model), "--option", "nonconvex=global",
                            "--option", "global_rlt=false",
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

        if model.name == "haverly1_p.mps":
            # global_pool_lift: the search ran on the lifted model, the certificate names the
            # pool's rows, and the checker rebuilds the lift from the model file.
            with tempfile.TemporaryDirectory() as tmp:
                cert_path = Path(tmp) / "cert.json"
                subprocess.run([str(binary), "solve", str(model), "--option", "nonconvex=global",
                                "--option", "global_pool_lift=true",
                                "--option", f"write_certificate={cert_path}",
                                "--option", "log_to_console=false"], capture_output=True, text=True)
                if not cert_path.exists():
                    check(False, "the solver wrote a certificate under global_pool_lift")
                    continue
                code, out = run_checker(model, cert_path)
                cert = json.loads(cert_path.read_text())
                check(code == 0 and "VERIFIED" in out and cert.get("lift") == HAVERLY1_LIFT,
                      "a global_pool_lift certificate is verified with the lift rebuilt",
                      f"lift {cert.get('lift')}; {out.strip()[-80:]}")
                cert["lift"][0]["balance"] = 6
                cert_path.write_text(json.dumps(cert))
                code, out = run_checker(model, cert_path)
                check(code == 1, "a lift naming a row of the wrong shape is rejected end to end",
                      f"exit {code}: {out.strip().splitlines()[0][:90]}")

        if model.name == "haverly1_q.mps":
            # global_rlt: the PQ rows the search adds to every relaxation are in the certificate,
            # rebuilt by the checker, and an RLT row it cannot derive is rejected.
            with tempfile.TemporaryDirectory() as tmp:
                cert_path = Path(tmp) / "cert.json"
                subprocess.run([str(binary), "solve", str(model), "--option", "nonconvex=global",
                                "--option", "global_rlt=true",
                                "--option", f"write_certificate={cert_path}",
                                "--option", "log_to_console=false"], capture_output=True, text=True)
                if not cert_path.exists():
                    check(False, "the solver wrote a certificate under global_rlt")
                    continue
                code, out = run_checker(model, cert_path)
                cert = json.loads(cert_path.read_text())
                check(code == 0 and "VERIFIED" in out and len(cert.get("rlt", [])) > 0,
                      "a global_rlt certificate with its RLT rows is verified",
                      f"{len(cert.get('rlt', []))} rlt rows; {out.strip()[-80:]}")
                names = vg.parse_mps(model).row_names
                quadratic = next(i for i, name in enumerate(names) if name.startswith("path_"))
                cert["rlt"][0][0] = quadratic
                cert_path.write_text(json.dumps(cert))
                code, out = run_checker(model, cert_path)
                check(code == 1, "an RLT row over a quadratic row is rejected end to end",
                      f"exit {code}: {out.strip().splitlines()[0][:90]}")

            # A search stopped after three nodes has a wide gap. Calling it optimal, with a gap
            # tolerance in the file wide enough to cover it, must not verify: the gap accepted
            # is the checker's own, not the certificate's.
            with tempfile.TemporaryDirectory() as tmp:
                cert_path = Path(tmp) / "cert.json"
                subprocess.run([str(binary), "solve", str(model), "--option", "nonconvex=global",
                                "--option", "global_rlt=false",
                                "--option", f"write_certificate={cert_path}",
                                "--option", "node_limit=3", "--option", "log_to_console=false"],
                               capture_output=True, text=True)
                if not cert_path.exists():
                    check(False, "the solver wrote a certificate at the node limit")
                    continue
                code, out = run_checker(model, cert_path)
                check(code == 0 and "no optimality is claimed" in out,
                      "a node-limited certificate proves its bound", out.strip()[-80:])
                cert = json.loads(cert_path.read_text())
                cert["status"] = "optimal"
                cert["absolute_gap"] = 1e30
                cert["relative_gap"] = 1e30
                cert_path.write_text(json.dumps(cert))
                code, out = run_checker(model, cert_path)
                check(code == 2 and "NOT SHOWN" in out,
                      "an open gap called optimal with a wide stated tolerance is not shown",
                      f"exit {code}: {out.strip().splitlines()[-1][:90]}")


def main() -> int:
    layer_one()
    layer_rlt()
    layer_lift()
    layer_two()
    print(f"{'ALL PASSED' if FAILURES == 0 else str(FAILURES) + ' FAILED'}")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
