#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched Kennington LPs and emit the evidence CSV (#530).

The same rules as the Netlib full-set runner, and the same CSV: bench/runners/netlib.py's
CSV_COLUMNS, written by the same solve-and-verify step (netlib.run_one), so the two tables
can be read with one pair of eyes and one script. A row PASSES when the status is optimal,
the objective matches the PUBLISHED optimum to a relative 1e-6, and tools/verify_solution.py
re-derives feasibility, the objective and strong duality from the MPS without touching our
C++. The published optima come from data/kennington/reference.json, which
fetch_kennington.py parsed out of the directory's readme; none was typed in.

The readme prints its optima to eight significant figures, computed by an interior-point
code (Vanderbei's ALPO). Rounding to eight figures moves a value by at most 5e-8 relative,
twenty times inside the 1e-6 pass tolerance, so the tolerance is not loosened for this set
and a gap reported here is ours, not the table's.

Usage:
    python bench/runners/fetch_kennington.py --set small
    python bench/runners/kennington.py --time-limit 600
    python bench/runners/kennington.py --instances pds-02 ken-07 --time-limit 60
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netlib  # noqa: E402  - run_one, CSV_COLUMNS, sha256_file, machine_tag, ...

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "kennington"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

PASS_RELATIVE_TOLERANCE = netlib.PASS_RELATIVE_TOLERANCE


def relative_gap(ours: float | None, published: float) -> float | None:
    """|ours - published| / max(1, |published|), netlib.py's definition."""
    if ours is None:
        return None
    return abs(ours - published) / max(1.0, abs(published))


def verdict(status: str, gap: float | None, verified: bool | None) -> bool:
    """The pass rule: optimal, within 1e-6 of the published value, and not rejected by the
    independent verifier. Either of the last two alone can be satisfied by a wrong solver."""
    matches = status == "optimal" and gap is not None and gap <= PASS_RELATIVE_TOLERANCE
    return bool(matches and verified is not False)


def matches_manifest(mps: Path, entry: dict) -> bool:
    """Whether `mps` is the instance the manifest recorded, blind to CRLF versus LF (emps
    writes CRLF on Windows)."""
    expected = entry.get("mps_lf_sha256")
    if not expected:
        return netlib.sha256_file(mps) == entry.get("mps_sha256")
    data = mps.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest() == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the independent verifier (not recommended)")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed as --option KEY=VALUE; repeatable, recorded in the CSV. "
                             "Use algorithm=dual-simplex / simplex / pdhg / ipm for the "
                             "per-engine runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths resolve against the repo root")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        raise SystemExit("no data/kennington/reference.json; run "
                         "bench/runners/fetch_kennington.py first")
    # Read once, for the reason netlib.py gives: the set tag goes in the filename and must be
    # the set that was run, not whatever a concurrent fetch left in the file.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest["instances"]
    tier = manifest.get("instance_set", "")
    names = sorted(args.instances or reference)
    # A subset named on the command line is not the manifest's set.
    if args.instances:
        tier = "explicit"
    binary = args.binary or netlib.default_binary()
    commit, machine = netlib.git_commit(args.binary), netlib.machine_tag()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)

    print(f"solver   {binary}")
    print(f"commit   {commit}   machine {machine}   time limit {args.time_limit:g}s"
          + (f"   options {solver_options}" if solver_options else ""))
    print()
    print(f"{'instance':<9}{'rows':>7}{'cols':>8}  {'status':<16}{'our objective':>20}"
          f"{'published':>16}{'rel gap':>10}{'iters':>8}{'time':>9}  verified  result")
    print("-" * 124)

    rows: list[dict] = []
    for name in names:
        entry = reference.get(name)
        mps = DATA_DIR / f"{name}.mps"
        if entry is None or not mps.exists():
            print(f"{name:<9} MISSING (run fetch_kennington.py)")
            continue
        published = entry["published_optimal"]
        blob = netlib.run_one(binary, mps, args.time_limit, not args.no_verify,
                              args.solver_option)
        status = blob["status"]
        ours = blob.get("objective")
        verified = blob.get("verified")
        gap = relative_gap(ours, published)
        matches = bool(status == "optimal" and gap is not None
                       and gap <= PASS_RELATIVE_TOLERANCE)
        passed = verdict(status, gap, verified)
        # The objective-row constant, as netlib.py handles it: reported, never forgiven.
        offset = blob.get("objective_offset") or 0.0
        offset_gap = (None if ours is None or not offset
                      else relative_gap(ours - offset, published))
        explained_by_offset = bool(not matches and status == "optimal"
                                   and offset_gap is not None
                                   and offset_gap <= PASS_RELATIVE_TOLERANCE
                                   and verified is not False)
        rows.append({
            "instance": name,
            "instance_sha256": netlib.sha256_file(mps),
            "rows": blob.get("rows", ""),
            "columns": blob.get("columns", ""),
            "nonzeros": blob.get("nonzeros", ""),
            "status": status,
            "message": blob.get("message", ""),
            "verifier_message": blob.get("verifier_output", ""),
            "our_objective": "" if ours is None else repr(ours),
            "published_objective": repr(published),
            "absolute_gap": "" if ours is None else repr(abs(ours - published)),
            "relative_gap": "" if gap is None else repr(gap),
            "matches_published": int(matches),
            "independently_verified": "" if verified is None else int(verified),
            "passed": int(passed),
            "objective_offset": repr(offset),
            "differs_by_objective_constant": int(explained_by_offset),
            "wall_seconds": round(blob.get("wall_seconds", 0.0), 6),
            "solver_seconds": blob.get("solver_seconds", ""),
            "iterations": blob.get("iterations", ""),
            "algorithm": blob.get("algorithm", ""),
            "git_commit": commit,
            "solver_options": solver_options,
            "machine": machine,
            "timestamp_utc": timestamp,
        })
        ours_text = "-" if ours is None else f"{ours:.10e}"
        gap_text = "-" if gap is None else f"{gap:.1e}"
        mark = {True: "  yes   ", False: "  NO    ", None: "  -     "}[verified]
        print(f"{name:<9}{str(blob.get('rows', '')):>7}{str(blob.get('columns', '')):>8}  "
              f"{status:<16}{ours_text:>20}{entry.get('published_optimal_text', published):>16}"
              f"{gap_text:>10}{str(blob.get('iterations', '-')):>8}"
              f"{blob.get('wall_seconds', 0.0):>8.1f}s{mark}  "
              f"{'PASS' if passed else ('OFFSET' if explained_by_offset else 'FAIL')}")
        if not matches_manifest(mps, entry):
            print(f"{'':9}the file on disk is not the one the manifest recorded")
        if not passed:
            if blob.get("message"):
                print(f"{'':9}message: {str(blob['message'])[:200]}")
            if blob.get("verifier_output"):
                print(f"{'':9}verifier: {blob['verifier_output'][:200]}")

    passes = sum(row["passed"] for row in rows)
    print("-" * 124)
    print(f"{passes}/{len(rows)} matched the published optimum to a relative "
          f"{PASS_RELATIVE_TOLERANCE:g} AND passed independent verification")
    failed = [row["instance"] for row in rows if not row["passed"]]
    # Named, not counted. A pass rate without its failures is a claim.
    if failed:
        print(f"failed: {', '.join(failed)}")

    tier_tag = f"{tier}-" if tier and tier != "explicit" else ""
    out_path = args.out or (RESULTS_DIR / f"kennington-{tier_tag}{commit}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=netlib.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {netlib.display_path(out_path)}")
    return 0 if rows and passes == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
