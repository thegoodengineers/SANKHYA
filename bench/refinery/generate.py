#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the refinery suite (#1004): ten models at three sizes, and its manifest.

    python bench/refinery/generate.py                 # write data/refinery/*.mps + manifest
    python bench/refinery/generate.py --check         # regenerate to a temp dir; fail on any
                                                      # file or exact optimum that differs
    python bench/refinery/generate.py --cross-check   # fill the references of the two
                                                      # scheduling MILPs from HiGHS and SCIP
    python bench/refinery/generate.py --qp 13k 110k 1m # the convex-cost refinery QPs
                                                      # (refinery_qp.py) into data/refinery-qp/,
                                                      # not committed: generated where run

Every generator is deterministic in (size, seed), so `--check` is what CI runs: a committed
file is the file the generator writes today, byte for byte, and the manifest's exact optimum
is the one it computes. The manifest carries, per instance, the model and its citation, the
dimensions, the sha256, the kind (`lp` or `mip`, which compare_suite.py reads to pick each
solver's path), the reference objective and how it is known:

*   `kkt-construction`: chosen plan, derived costs, every KKT condition re-checked exactly
    (kit.Lp.derive_costs / verify_kkt); `rational-simplex`: integer pivoting over the
    written LP (exact_lp.py); `enumeration`: every discrete choice an LP, each by the
    rational simplex; `wagner-whitin`: the recursion in rational arithmetic. These four are
    exact, and the manifest states them as decimal strings of the exact value.
*   `cross-check`: the two scheduling MILPs, whose optimum no construction gives. The value
    is recorded only when HiGHS and SCIP, each run to a zero gap on one thread, both report
    optimal within CROSS_CHECK_TOLERANCE of each other; both values and both versions are
    kept beside it. An instance on which they do not agree, or one does not finish, gets no
    reference and compare_suite.py skips it.

A MIP row carries `published_tolerance`, the absolute slack the grade allows beyond the
relative 1e-6: kMipRelativeGap (include/sankhya/tolerances.hpp) times max(1, |reference|),
since every solver here, SANKHYA included, stops at that relative gap by default.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blending  # noqa: E402
import planning  # noqa: E402
import pooling  # noqa: E402
import refinery_qp  # noqa: E402
import scheduling  # noqa: E402
from kit import sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "refinery"
QP_DIR = REPO_ROOT / "data" / "refinery-qp"
MANIFEST = DATA_DIR / "manifest.json"
SEED = 1
SIZES = ("small", "medium", "large")
MIP_RELATIVE_GAP = 1e-4        # kMipRelativeGap, include/sankhya/tolerances.hpp
CROSS_CHECK_TOLERANCE = 1e-7   # relative, between HiGHS and SCIP at a zero gap
CROSS_CHECK_LIMIT = 3600.0     # seconds per solver per instance

# The ten, in the issue's order: name -> (module, builder, how the optimum is known)
MODELS = {
    "crude_blend": (blending.crude_blend, "kkt-construction"),
    "product_blend": (blending.product_blend, "kkt-construction"),
    "haverly_mccormick": (lambda s, seed: pooling.build("haverly_mccormick", s, seed),
                          "rational-simplex"),
    "haverly_discrete": (lambda s, seed: pooling.build("haverly_discrete", s, seed),
                         "enumeration"),
    "unloading": (scheduling.unloading, "cross-check"),
    "unit_modes": (scheduling.unit_modes, "cross-check"),
    "multi_period": (planning.multi_period, "kkt-construction"),
    "two_stage": (planning.two_stage, "kkt-construction"),
    "distribution": (blending.distribution, "kkt-construction"),
    "lot_sizing": (planning.lot_sizing, "wagner-whitin"),
}


def generate(name: str, size: str, out_dir: Path) -> dict:
    """Write one instance and return its manifest entry (reference included when exact)."""
    builder, how = MODELS[name]
    lp, value, _, header = builder(size, SEED)
    if how == "kkt-construction":
        lp.verify_kkt()
    path = out_dir / f"{name}_{size}.mps"
    stated = ([] if value is None or any(h.startswith("exact optimum") for h in header)
              else [f"exact optimum: {float(value)!r} ({how})"])
    lp.write(path, header + stated + [f"suite: bench/refinery/generate.py, seed {SEED}"])
    kind = "mip" if lp.integer or lp.semicontinuous else "lp"
    entry = {"model": name, "size": size, "kind": kind, "source": header[1][len("source: "):],
             "rows": len(lp.rows), "columns": len(lp.cols), "nonzeros": lp.nonzeros,
             "integer": len(lp.integer), "semicontinuous": len(lp.semicontinuous),
             "sha256": sha256_file(path), "reference_source": how}
    if value is not None:
        entry["reference_exact"] = str(value)  # the Fraction, e.g. -29550/7
        entry["reference_objective"] = float(value)
    if kind == "mip" and value is not None:
        entry["published_tolerance"] = MIP_RELATIVE_GAP * max(1.0, abs(float(value)))
    return entry


def cross_check(path: Path) -> dict:
    """HiGHS and SCIP on `path`, each to a zero gap on one thread; a reference when they
    agree. External solvers, imported here only, never a build dependency."""
    import highspy  # noqa: PLC0415
    import pyscipopt  # noqa: PLC0415
    h = highspy.Highs()
    for key, value in (("output_flag", False), ("threads", 1), ("mip_rel_gap", 0.0),
                       ("mip_abs_gap", 0.0), ("time_limit", CROSS_CHECK_LIMIT)):
        h.setOptionValue(key, value)
    h.readModel(str(path))
    h.run()
    highs = {"status": h.modelStatusToString(h.getModelStatus()).lower(),
             "objective": h.getInfo().objective_function_value, "version": h.version()}
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(str(path), extension="mps")
    for key, value in (("limits/time", CROSS_CHECK_LIMIT), ("lp/threads", 1),
                       ("parallel/maxnthreads", 1), ("limits/gap", 0.0)):
        m.setParam(key, value)
    m.optimize()
    scip = {"status": m.getStatus(), "objective": m.getObjVal() if m.getStatus() == "optimal"
            else None, "version": f"{m.getMajorVersion()}.{m.getMinorVersion()}."
            f"{m.getTechVersion()}"}
    out = {"cross_check": {"highs": highs, "scip": scip}}
    if highs["status"] == "optimal" and scip["status"] == "optimal":
        a, b = highs["objective"], scip["objective"]
        if abs(a - b) <= CROSS_CHECK_TOLERANCE * max(1.0, abs(a)):
            out["reference_objective"] = a
            out["published_tolerance"] = MIP_RELATIVE_GAP * max(1.0, abs(a))
    return out


def generate_qp(labels: list[str], out_dir: Path) -> int:
    """The convex-cost refinery QPs (refinery_qp.py) and their manifest, in the shape
    bench/runners/compare_suite.py's `refinery-qp` suite reads. The files are generated where
    the comparison runs, not committed (the largest is about 120 MB); the generator, its seed
    and the sha256 of every file it writes are the record, and the run's CSV carries the
    sha256 of the file each solver read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = {}
    for label in labels:
        periods = refinery_qp.QP_SIZES[label]
        inst, q = refinery_qp.build(periods, refinery_qp.QP_SEED)
        refinery_qp.verify_qp(inst, q)
        path = out_dir / f"refinery_qp_{label}.qps"
        value = refinery_qp.write_qps(inst, q, path, label, periods, refinery_qp.QP_SEED)
        instances[path.stem] = {
            "kind": "qp", "periods": periods, "rows": len(inst.rows),
            "columns": len(inst.cols), "quadratic_columns": sum(1 for v in q if v),
            "nonzeros": sum(len(e) for e in inst.row_entries), "sha256": sha256_file(path),
            "reference_exact": str(value), "reference_objective": float(value),
            "reference_source": "kkt-construction"}
        print(f"{path.stem:<22} qp {len(inst.rows):>7} x {len(inst.cols):<8} "
              f"reference {float(value)!r}")
    (out_dir / "manifest.json").write_text(json.dumps({
        "generator": "bench/refinery/generate.py --qp (bench/refinery/refinery_qp.py)",
        "seed": refinery_qp.QP_SEED,
        "reference_kinds": {"kkt-construction": "chosen plan, derived linear prices "
                            "c = d + A'y - Qx, every KKT condition of the QP re-checked "
                            "exactly (refinery_qp.verify_qp)"},
        "instances": instances}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cross-check", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--sizes", nargs="*", default=list(SIZES))
    parser.add_argument("--qp", nargs="*", choices=list(refinery_qp.QP_SIZES), default=None,
                        help="write the convex-cost refinery QPs instead (13k, 110k, 1m)")
    parser.add_argument("--qp-dir", type=Path, default=QP_DIR)
    args = parser.parse_args()
    if args.qp is not None:
        return generate_qp(args.qp or list(refinery_qp.QP_SIZES), args.qp_dir)
    old = (json.loads(MANIFEST.read_text(encoding="utf-8"))["instances"]
           if MANIFEST.exists() else {})
    if args.check:
        failures = 0
        with tempfile.TemporaryDirectory() as tmp:
            for name in args.models:
                for size in args.sizes:
                    entry = generate(name, size, Path(tmp))
                    key = f"{name}_{size}"
                    have = old.get(key, {})
                    same = (have.get("sha256") == entry["sha256"]
                            and have.get("reference_exact") == entry.get("reference_exact"))
                    print(f"  [{'PASS' if same else 'FAIL'}] {key}")
                    failures += not same
        return 1 if failures else 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    instances = dict(old)
    for name in args.models:
        for size in args.sizes:
            key = f"{name}_{size}"
            entry = generate(name, size, DATA_DIR)
            # Keep an earlier cross-check when the file it was computed on is unchanged.
            if entry["reference_source"] == "cross-check":
                kept = old.get(key, {})
                if kept.get("sha256") == entry["sha256"]:
                    for field in ("cross_check", "reference_objective", "published_tolerance"):
                        if field in kept:
                            entry[field] = kept[field]
                if args.cross_check:
                    entry.update(cross_check(DATA_DIR / f"{key}.mps"))
            instances[key] = entry
            ref = entry.get("reference_objective")
            print(f"{key:<28} {entry['kind']} {entry['rows']:>6} x {entry['columns']:<6} "
                  f"{'reference ' + repr(ref) if ref is not None else 'no reference'} "
                  f"({entry['reference_source']})")
    MANIFEST.write_text(json.dumps({
        "issue": 1004, "seed": SEED,
        "generator": "bench/refinery/generate.py",
        "reference_kinds": {
            "kkt-construction": "chosen plan, derived costs, every KKT condition re-checked "
                                "exactly (bench/refinery/kit.py)",
            "rational-simplex": "integer pivoting over the written LP "
                                "(bench/refinery/exact_lp.py)",
            "enumeration": "every discrete choice an LP solved by the rational simplex",
            "wagner-whitin": "the lot-sizing recursion in rational arithmetic",
            "cross-check": "HiGHS and SCIP at a zero gap, one thread, agreeing within "
                           f"{CROSS_CHECK_TOLERANCE:g} relative; both values kept"},
        "instances": dict(sorted(instances.items()))}, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
