#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evidence for the first-order engine: PDHG against the simplex, at two tolerances.

The plan for this phase asks for four things, and this script produces all four from live
runs rather than from anybody's recollection:

1.  PDHG's objective next to the simplex's, on every Netlib instance we solve.
2.  Results reported at 1e-4 and 1e-8 SEPARATELY. A first-order method's cost depends
    enormously on the requested accuracy, so a single blended number would be meaningless.
3.  Iteration counts with restarts on and off, so the claim that restarts help is measured
    rather than asserted.
4.  An honest list of the instances PDHG cannot drive to 1e-8.

Every row also carries the independent verifier's verdict on the written solution
(tools/verify_solution.py, which never links our C++) and, for a PDHG row, the solver
clock at which the relative KKT error first crossed 1e-4, 1e-6 and 1e-8 in that run (#486,
kkt_crossings.py has the formula). The crossings are the PDLP measurement; the verdict is
the project's absolute standard. They are side by side because they are not the same claim.

Usage:
    python bench/runners/pdhg_report.py --binary build/sankhya
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import kkt_crossings  # noqa: E402  (#486: the relative-KKT crossing columns)
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

CSV_COLUMNS = [
    "instance", "algorithm", "tolerance", "restarts_enabled", "status",
    "objective", "published_objective", "relative_error", "iterations", "seconds",
    "reached_tolerance", "git_commit", "machine", "timestamp_utc", "solver_options",
    # Appended (#486), so a reader of an older CSV by name is unaffected.
    "instance_sha256", "absolute_error", "independently_verified", "verifier_message",
    *kkt_crossings.ALL_COLUMNS,
]


def git_commit(binary=None) -> str:
    """The commit this CSV is stamped with: the binary's own, read from `sankhya
    version`, with `-dirty` from the tree; HEAD only when no binary answers (#433,
    bench/runners/stamp.py)."""
    return stamp.stamp(binary)

def as_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verdict(mps: Path, sol: Path, status: str) -> tuple[str, str]:
    """The independent verifier on a written point: ("1" | "0", why) for a status that
    claims a point, ("", "") otherwise - a limit hands back no claim to check."""
    if status not in ("optimal", "feasible") or not sol.exists():
        return "", ""
    check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                           capture_output=True, text=True)
    if check.returncode == 0:
        return "1", ""
    failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
    return "0", "; ".join(failing)[:300]


def default_binary() -> Path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError as error:
        raise SystemExit(str(error))


def run(binary: Path, mps: Path, algorithm: str, tolerance: float | None,
        restarts: bool, time_limit: float, extra_options: list[str] | None = None,
        verify: bool = True) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        sol = Path(tmp) / "s.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--write-sol", str(sol),
                   "--time-limit", str(time_limit),
                   "--option", "log_to_console=false",
                   "--option", f"algorithm={algorithm}"]
        if tolerance is not None:
            command += ["--option", f"pdhg_tolerance={tolerance:g}"]
        if not restarts:
            command += ["--option", "pdhg_restart=false"]
        for option in extra_options or []:
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "seconds": seconds, "verified": "", "verifier_message": "",
                    **{k: "" for k in kkt_crossings.ALL_COLUMNS}}
        blob = json.loads(stats.read_text())
        status = blob.get("result", {}).get("status", "unknown")
        verified, why = verdict(mps, sol, status) if verify else ("", "")
        return {
            "status": status,
            "objective": as_number(blob.get("result", {}).get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": seconds,
            "verified": verified,
            "verifier_message": why,
            **kkt_crossings.crossings(blob),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*", metavar="NAME",
                        help="run only these instances. Without it every instance in "
                             "data/netlib/reference.json is run, which is the whole tier the "
                             "last fetch left there - 89 instances at four settings each, "
                             "hours of solving for a report whose point is the committed nine.")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="pass --option KEY=VALUE to every PDHG solve and record it in the "
                             "CSV's solver_options column. A run made with one is a measurement "
                             "OF that option, not the engine's evidence, and latest_result.py "
                             "skips it for that reason.")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the CSV here instead of bench/results/pdhg-<commit>.csv")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the independent verifier (the verdict column stays blank)")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    reference = json.loads((DATA_DIR / "reference.json").read_text())["instances"]
    names = sorted(args.instances) if args.instances else sorted(reference)
    unknown = [n for n in names if n not in reference]
    if unknown:
        raise SystemExit("not in data/netlib/reference.json: " + ", ".join(unknown))
    commit = git_commit(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    digests = {n: sha256_file(DATA_DIR / f"{n}.mps") for n in names
               if (DATA_DIR / f"{n}.mps").exists()}
    verify = not args.no_verify

    def record(name, algorithm, tolerance, restarts, result, published):
        error = (None if result["objective"] is None
                 else abs(result["objective"] - published) / max(1.0, abs(published)))
        absolute = (None if result["objective"] is None
                    else abs(result["objective"] - published))
        rows.append({
            "instance": name, "algorithm": algorithm,
            "tolerance": "" if tolerance is None else f"{tolerance:g}",
            "restarts_enabled": "" if restarts is None else int(restarts),
            "status": result["status"],
            "objective": "" if result["objective"] is None else repr(result["objective"]),
            "published_objective": repr(published),
            "relative_error": "" if error is None else repr(error),
            "iterations": result["iterations"], "seconds": round(result["seconds"], 6),
            # `optimal` and `feasible` both mean the loop stopped because the requested
            # relative tolerance was met; `feasible` is the point that met it without also
            # meeting the project's absolute standard (see #179, #180). A limit means it
            # was not met.
            "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
            "git_commit": commit, "machine": machine, "timestamp_utc": timestamp,
            "solver_options": " ".join(args.solver_option),
            "instance_sha256": digests[name],
            "absolute_error": "" if absolute is None else repr(absolute),
            "independently_verified": result.get("verified", ""),
            "verifier_message": result.get("verifier_message", ""),
            **{k: result.get(k, "") for k in kkt_crossings.ALL_COLUMNS},
        })
        return error

    # ---- 1 & 2: agreement with the simplex, at both tolerances -------------------------
    print("PDHG vs the simplex. Relative error is against the PUBLISHED optimum.\n")
    print(f"{'instance':<11}{'simplex obj':>21}{'simplex err':>12}"
          f"{'PDHG 1e-4 obj':>21}{'err':>10}{'iters':>8}"
          f"{'PDHG 1e-8 obj':>21}{'err':>10}{'iters':>8}  1e-8?")
    print("-" * 124)

    missed_tight: list[str] = []
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]

        simplex = run(binary, mps, "simplex", None, True, args.time_limit, verify=verify)
        simplex_error = record(name, "simplex", None, None, simplex, published)

        loose = run(binary, mps, "pdhg", 1e-4, True, args.time_limit,
                  args.solver_option, verify=verify)
        loose_error = record(name, "pdhg", 1e-4, True, loose, published)

        tight = run(binary, mps, "pdhg", 1e-8, True, args.time_limit,
                  args.solver_option, verify=verify)
        tight_error = record(name, "pdhg", 1e-8, True, tight, published)
        if tight["status"] not in ("optimal", "feasible"):
            missed_tight.append(name)

        def fmt(value, width, digits):
            return "-".rjust(width) if value is None else f"{value:>{width}.{digits}e}"

        print(f"{name:<11}{fmt(simplex['objective'], 21, 12)}{fmt(simplex_error, 12, 1)}"
              f"{fmt(loose['objective'], 21, 12)}{fmt(loose_error, 10, 1)}"
              f"{str(loose['iterations']):>8}"
              f"{fmt(tight['objective'], 21, 12)}{fmt(tight_error, 10, 1)}"
              f"{str(tight['iterations']):>8}"
              f"  {'yes' if tight['status'] in ('optimal', 'feasible') else 'NO'}")

    print("-" * 124)
    if missed_tight:
        print(f"did NOT reach 1e-8 within the time limit: {', '.join(missed_tight)}")
    else:
        print("every instance reached the 1e-8 relative tolerance")

    # ---- the relative-KKT crossings of the 1e-8 run, beside the verifier (#486) ---------
    print("\nRelative KKT error (PDLP definition) first at or under each level, in the 1e-8 "
          "run,\nbeside the independent verifier's verdict on the point it returned.\n")
    print(f"{'instance':<11}{'1e-4 (s)':>12}{'1e-6 (s)':>12}{'1e-8 (s)':>12}"
          f"  {'status':<10}verified")
    print("-" * 70)
    for row in rows:
        if row["algorithm"] != "pdhg" or row["tolerance"] != "1e-08":
            continue
        cells = [kkt_crossings.cell_text(row[k]) for k in kkt_crossings.COLUMNS]
        mark = {"1": "yes", "0": "NO"}.get(str(row["independently_verified"]), "-")
        print(f"{row['instance']:<11}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}"
              f"  {row['status']:<10}{mark}")

    # ---- 3: restarts on versus off ------------------------------------------------------
    print("\nRestarts on vs off, at 1e-8. This is the measurement behind the claim that "
          "restarts help.\n")
    print(f"{'instance':<11}{'restarts on':>14}{'restarts off':>15}{'speedup':>10}  status off")
    print("-" * 62)
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]
        on = run(binary, mps, "pdhg", 1e-8, True, args.time_limit,
                  args.solver_option, verify=verify)
        off = run(binary, mps, "pdhg", 1e-8, False, args.time_limit,
                  args.solver_option, verify=verify)
        record(name, "pdhg", 1e-8, True, on, published)
        record(name, "pdhg", 1e-8, False, off, published)
        on_iters = on["iterations"] if isinstance(on["iterations"], int) else 0
        off_iters = off["iterations"] if isinstance(off["iterations"], int) else 0
        speedup = (off_iters / on_iters) if on_iters else 0.0
        print(f"{name:<11}{on_iters:>14}{off_iters:>15}{speedup:>9.2f}x  {off['status']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"pdhg-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # An --out outside the repository (a scratch run) is printed as given.
    shown = out.resolve().relative_to(REPO_ROOT) if out.resolve().is_relative_to(REPO_ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
