#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the numerical stress set (#762): badly scaled Netlib plus adversarial LPs, every
one with its answer known before anything is solved.

    python bench/runners/stress_instances.py            # writes data/stress/*.mps
    python bench/runners/stress_instances.py --check    # regenerate, compare every sha256

The MPS files are not committed; data/stress/reference.json is, and it records each file's
sha256, so a run on another machine either solves the identical bytes or says it did not.
Everything below is deterministic: seeded generators, exact arithmetic where it matters,
and `repr` for every number written.

1. SCALED NETLIB. For each Netlib LP, row i is multiplied by r_i = 2^a_i and column j is
   replaced by x_j = s_j x'_j with s_j = 2^b_j, a_i and b_j drawn uniformly from -20..20
   (2^20 = 1.05e6, so every factor lies in 9.5e-7 .. 1.05e6 and a matrix entry moves by up
   to 2^40 = 1.1e12). Then A' = R A S, c' = S c, row bounds R [lo, hi], column bounds
   [l/s, u/s]. Powers of two make the transform EXACT in binary floating point (only the
   exponent changes, barring overflow, which Netlib's magnitudes are far from), so the
   scaled model is the original model and its optimum is the original optimum - Koch's
   exact value (data/netlib/koch_exact.json, Koch 2004, Oper. Res. Lett. 32:138-142).

2. ADVERSARIAL.
   klee_minty_n   Klee and Minty (1972), "How good is the simplex algorithm?": max
                  sum 2^(n-j) x_j s.t. sum_{j<i} 2^(i-j+1) x_j + x_i <= 5^i, x >= 0; written
                  as a minimisation. y = e_n is dual feasible for any b >= 0, so the optimum
                  is -b_n with b_n = 5^n as stored in a double - exact, not approximate.
   near_parallel  robustness.py's family (#71): every active row gets a twin differing in
                  one coefficient by a relative 10^-k, a KKT-built LP with a known optimum.
   singular_basis the optimal basis is [[1, 1], [1, 1 + e]] with e = 2^-k exact: min
                  -(x1 + (1 + e/2) x2) s.t. x1 + x2 <= 2, x1 + (1 + e) x2 <= 2 + e. The
                  optimum is x = (1, 1), objective -(2 + e/2), both multipliers 1/2.
   thin_infeas    a KKT-built LP with optimum z*, plus the row c'x <= z* - delta: empty,
                  and made feasible by relaxing that one right-hand side by delta, no less.
   beale          Beale (1955), "Cycling in the dual simplex algorithm", the classic
                  example on which the textbook rule cycles; optimum -1/20 at x4 = 1/25,
                  x6 = 1. chvatal_cycle is Chvatal's (Linear Programming, 1983, ch. 3)
                  cycling example, optimum -1 (maximum 1) at x1 = 1, x3 = 1.
   degenerate     robustness.py's degeneracy family: k * n rows all tight at the optimum.
   unbounded      a KKT-built LP plus two new columns u, v with a_v = -a_u + w, w >= 0, and
                  c_u = -2, c_v = 1: neither column alone improves without limit, the
                  direction e_u + e_v does (A d = w >= 0, c'd = -1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from verify_solution_mps import Model, parse_mps  # noqa: E402  (the independent reader)
import robustness  # noqa: E402  (the KKT construction, #71)

NETLIB = REPO_ROOT / "data" / "netlib"
OUT = REPO_ROOT / "data" / "stress"
INF = float("inf")
SCALE_EXPONENT = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mps(model: Model, path: Path) -> None:
    """Free MPS with generated names (Netlib has names with spaces), every number by repr,
    and both column bounds written whenever the upper one is finite, so no reader's
    convention for a lone negative UP can change the model."""
    out = [f"NAME {model.name or 'STRESS'}", "ROWS", " N OBJ"]
    for i in range(model.num_rows):
        lo, hi = model.row_lower[i], model.row_upper[i]
        kind = "E" if lo == hi else ("L" if lo == -INF else "G")
        out.append(f" {kind} R{i}")
    out.append("COLUMNS")
    for j in range(model.num_cols):
        merged: dict[int, float] = {}
        for i, v in model.entries[j]:
            merged[i] = merged.get(i, 0.0) + v
        out.append(f" C{j} OBJ {model.col_cost[j]!r}")
        out.extend(f" C{j} R{i} {v!r}" for i, v in sorted(merged.items()) if v != 0.0)
    out.append("RHS")
    if model.objective_offset != 0.0:
        out.append(f" RHS OBJ {-model.objective_offset!r}")
    ranges = []
    for i in range(model.num_rows):
        lo, hi = model.row_lower[i], model.row_upper[i]
        rhs = hi if lo == -INF else lo
        if rhs != 0.0:
            out.append(f" RHS R{i} {rhs!r}")
        if lo != hi and lo != -INF and hi != INF:
            ranges.append(f" RNG R{i} {hi - lo!r}")
    if ranges:
        out += ["RANGES", *ranges]
    out.append("BOUNDS")
    for j in range(model.num_cols):
        lo, hi = model.col_lower[j], model.col_upper[j]
        if lo == -INF and hi == INF:
            out.append(f" FR BND C{j}")
            continue
        if lo == hi:
            out.append(f" FX BND C{j} {lo!r}")
            continue
        out.append(f" MI BND C{j}" if lo == -INF else f" LO BND C{j} {lo!r}")
        if hi != INF:
            out.append(f" UP BND C{j} {hi!r}")
    out.append("ENDATA")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def from_instance(instance: robustness.Instance) -> Model:
    """robustness.py's row-wise Instance as a column-wise Model."""
    model = Model()
    model.name = instance.name
    for j in range(instance.num_cols()):
        model.add_column(f"C{j}", False)
        model.col_cost[j] = instance.cost[j]
        model.col_lower[j], model.col_upper[j] = instance.lower[j], instance.upper[j]
    for i, (entries, lo, hi) in enumerate(instance.rows):
        model.row_names.append(f"R{i}")
        model.row_lower.append(lo)
        model.row_upper.append(hi)
        for j, v in entries.items():
            model.entries[j].append((i, v))
    return model


def dense(name: str, cost: list[float], rows: list[tuple[list[float], float, float]],
          upper: list[float] | None = None) -> Model:
    return from_instance(_instance(name, cost, rows, upper))


def _instance(name, cost, rows, upper):
    instance = robustness.Instance(name)
    instance.cost = list(cost)
    instance.lower = [0.0] * len(cost)
    instance.upper = list(upper) if upper else [INF] * len(cost)
    instance.rows = [({j: v for j, v in enumerate(a) if v != 0.0}, lo, hi) for a, lo, hi in rows]
    return instance


# =========================================================================================
# The families. Each yields (name, family, parameter, model, expected status, objective).
# =========================================================================================

def scaled_netlib(names: list[str] | None):
    koch = json.loads((NETLIB / "koch_exact.json").read_text())["instances"]
    for name in sorted(names or koch):
        model = parse_mps(NETLIB / f"{name}.mps")
        rng = random.Random(f"stress-762:{name}")
        r = [2.0 ** rng.randint(-SCALE_EXPONENT, SCALE_EXPONENT) for _ in range(model.num_rows)]
        s = [2.0 ** rng.randint(-SCALE_EXPONENT, SCALE_EXPONENT) for _ in range(model.num_cols)]
        for j in range(model.num_cols):
            model.col_cost[j] *= s[j]
            model.col_lower[j] /= s[j]
            model.col_upper[j] /= s[j]
            model.entries[j] = [(i, v * r[i] * s[j]) for i, v in model.entries[j]]
        model.row_lower = [lo * r[i] for i, lo in enumerate(model.row_lower)]
        model.row_upper = [hi * r[i] for i, hi in enumerate(model.row_upper)]
        yield (f"scaled_{name}", "scaled_netlib", f"2^+-{SCALE_EXPONENT}", model, "optimal",
               koch[name]["exact_objective"])


def klee_minty(n: int) -> Model:
    rows = []
    for i in range(1, n + 1):
        a = [2.0 ** (i - j + 1) for j in range(1, i)] + [1.0] + [0.0] * (n - i)
        rows.append((a, -INF, float(5 ** i)))
    return dense(f"KM{n}", [-(2.0 ** (n - j)) for j in range(1, n + 1)], rows)


def singular_basis(k: int) -> Model:
    e = 2.0 ** -k
    return dense(f"SB{k}", [-1.0, -(1.0 + e / 2)],
                 [([1.0, 1.0], -INF, 2.0), ([1.0, 1.0 + e], -INF, 2.0 + e)])


def thin_infeasible(delta: float, seed: int) -> Model:
    base = robustness.kkt_instance(random.Random(seed), cols=12, rows=10, active_fraction=0.6,
                                   name="THIN")
    base.rows.append(({j: c for j, c in enumerate(base.cost) if c != 0.0}, -INF,
                      base.expected - delta))
    return from_instance(base)


def unbounded(seed: int) -> Model:
    rng = random.Random(seed)
    base = robustness.kkt_instance(rng, cols=10, rows=8, active_fraction=0.75, name="UNBD")
    base.cost += [-2.0, 1.0]
    base.lower += [0.0, 0.0]
    base.upper += [INF, INF]
    u, v = base.num_cols() - 2, base.num_cols() - 1
    for entries, _, _ in base.rows:
        au = float(rng.randint(-5, 5))
        entries[u], entries[v] = au, -au + rng.randint(0, 3)
    return from_instance(base)


def adversarial():
    for n in (10, 15, 20, 25, 30, 35, 40):
        yield (f"klee_minty_{n}", "klee_minty", f"n={n}", klee_minty(n), "optimal",
               repr(-float(5 ** n)))
    for k in (6, 9, 12, 15):
        inst = robustness.near_parallel(random.Random(76200 + k), k)
        yield (f"near_parallel_{k}", "near_singular", f"twin rows 1e-{k} apart",
               from_instance(inst), "optimal", repr(inst.expected))
    for k in (20, 30, 40, 50):
        yield (f"singular_basis_{k}", "near_singular", f"e=2^-{k}", singular_basis(k),
               "optimal", repr(-(2.0 + 2.0 ** -k / 2)))
    for exponent in (9, 8, 7, 6):
        for seed in (1, 2):
            yield (f"thin_infeasible_1e-{exponent}_{seed}", "thin_infeasible",
                   f"delta=1e-{exponent}", thin_infeasible(10.0 ** -exponent, 76210 + seed),
                   "infeasible", "")
    yield ("beale", "degenerate", "Beale 1955", dense(
        "BEALE", [-0.75, 150.0, -0.02, 6.0],
        [([0.25, -60.0, -0.04, 9.0], -INF, 0.0), ([0.5, -90.0, -0.02, 3.0], -INF, 0.0),
         ([0.0, 0.0, 1.0, 0.0], -INF, 1.0)]), "optimal", "-0.05")
    yield ("chvatal_cycle", "degenerate", "Chvatal 1983", dense(
        "CHVATAL", [-10.0, 57.0, 9.0, 24.0],
        [([0.5, -5.5, -2.5, 9.0], -INF, 0.0), ([0.5, -1.5, -0.5, 1.0], -INF, 0.0),
         ([1.0, 0.0, 0.0, 0.0], -INF, 1.0)]), "optimal", "-1")
    for k in (8, 32, 64):
        inst = robustness.degeneracy(random.Random(76230 + k), k)
        yield (f"degenerate_{k}", "degenerate", f"{k}n tight rows", from_instance(inst),
               "optimal", repr(inst.expected))
    for seed in (1, 2, 3, 4):
        yield (f"unbounded_{seed}", "unbounded", "ray e_u + e_v", unbounded(76240 + seed),
               "unbounded", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--netlib", nargs="*", default=None, help="Netlib names (default all)")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare with the committed sha256 values")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "reference.json"
    committed = (json.loads(manifest_path.read_text())["instances"]
                 if manifest_path.exists() else {})
    instances, mismatches = {}, []
    families = [adversarial()] + ([] if args.netlib == [] else [scaled_netlib(args.netlib)])
    for family in families:
        for name, fam, parameter, model, status, objective in family:
            path = args.out / f"{name}.mps"
            write_mps(model, path)
            digest = sha256(path)
            instances[name] = {"family": fam, "parameter": parameter, "expected_status": status,
                               "expected_objective": objective, "sha256": digest,
                               "rows": model.num_rows, "cols": model.num_cols}
            if name in committed and committed[name]["sha256"] != digest:
                mismatches.append(name)
            print(f"{name:28s} {fam:16s} {model.num_rows:>6d} x {model.num_cols:<6d} {digest[:12]}")
    if args.check:
        print(f"{len(mismatches)} sha256 mismatch(es)" + (": " + ", ".join(mismatches)
                                                           if mismatches else ""))
        return 1 if mismatches else 0
    manifest = {"generator": "bench/runners/stress_instances.py", "issue": 762,
                "instances": dict(sorted(instances.items()))}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                             newline="\n")
    print(f"wrote {len(instances)} instances and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
