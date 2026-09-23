#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over Netlib's infeasible LP collection and emit the evidence CSV (#529).

Every instance in lp/infeas is infeasible, so the only right answer is "infeasible" - and a
status string is not evidence of it. A solver that gives up and prints `infeasible` looks
exactly like one that proved it. So a row here PASSES only when all of these hold:

  1. the solver's status is `infeasible`;
  2. the .sol file it wrote says `certificate farkas` and carries a farkas section with at
     least one nonzero multiplier (src/core/certificate.cpp, #192);
  3. tools/verify_solution.py, which re-reads the MPS with its own parser and shares no code
     with the solver, accepts the file AND its report contains a passed
     `infeasibility proof` check.

Condition 3 is two checks rather than one on purpose. The verifier exits 0 on an `infeasible`
file with `certificate none`, because a verdict that offers no proof claims nothing it could
reject (#191); that is right for a verifier and wrong for a pass criterion. Presolve proves
some of these instances from bound arithmetic and does not keep a Farkas vector; such a row
is named here as `infeasible without a certificate`, not counted as a pass.

The CSV carries the columns ENGINEERING_RULES.md requires of every runner. `our_objective`,
`published_objective` and the two gaps are present and EMPTY: Netlib publishes no objective
for an infeasible model and a solver that reports one has already failed.

Usage:
    python bench/runners/fetch_netlib_infeasible.py
    python bench/runners/netlib_infeasible.py --time-limit 60
    python bench/runners/netlib_infeasible.py --solver-option algorithm=simplex
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402  (#433: stamps from the binary)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib-infeasible"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# The line tools/verify_solution.py prints when the aggregate contradicts the column box.
# The detail text is part of the match, not decoration: the verifier renders a NOTE with the
# same `[PASS] infeasibility proof` prefix when no certificate was offered ("nothing is
# claimed and nothing is checked"), so the prefix alone matches a file that proved nothing.
PROOF_PASSED = re.compile(r"\[PASS\] infeasibility proof\s+the rows aggregate to at least")

CSV_COLUMNS = [
    "instance",
    "instance_sha256",
    # 0 when the file on disk is not the one fetch_netlib_infeasible.py recorded, compared
    # line-ending-blind (the manifest's mps_lf_sha256), since emps writes CRLF on Windows.
    "matches_manifest",
    "rows",
    "columns",
    "nonzeros",
    "expected_status",
    "status",
    "message",
    # What the .sol header says (`farkas`, `none`, ...) and how many multipliers are nonzero.
    "certificate",
    "certificate_multipliers",
    "our_objective",
    "published_objective",
    "absolute_gap",
    "relative_gap",
    "independently_verified",
    "verifier_message",
    "passed",
    # Why a row is not a pass, in one phrase; empty on a pass. The doc groups by it.
    "failure_reason",
    "wall_seconds",
    "solver_seconds",
    "iterations",
    "algorithm",
    "time_limit",
    "git_commit",
    "machine",
    "timestamp_utc",
    "solver_options",
]


def read_certificate(sol_text: str) -> tuple[str, int]:
    """(the header's `certificate` value, the number of nonzero farkas multipliers).

    Parsed here rather than trusted from the stats blob because the .sol file is what the
    verifier reads; a header saying `farkas` over an empty or all-zero section is exactly the
    inconsistency a pass criterion must not wave through. Returns ("", 0) for a file with no
    certificate line at all.
    """
    kind = ""
    nonzero = 0
    in_farkas = False
    in_header = True
    for raw in sol_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("begin "):
            in_header = False
            in_farkas = line.split()[1] == "farkas"
            continue
        if line.startswith("end "):
            in_farkas = False
            continue
        fields = line.split()
        if in_header and fields[0] == "certificate" and len(fields) > 1:
            kind = fields[1]
        elif in_farkas and len(fields) >= 2:
            try:
                if float(fields[-1]) != 0.0:
                    nonzero += 1
            except ValueError:
                pass
    return kind, nonzero


def classify(status: str, certificate: str, multipliers: int, verifier_rc: int | None,
             verifier_text: str, algorithm: str = "") -> str:
    """Why a row is not a pass, or "" when it is. Order is the order of evidence.

    A missing certificate carries the engine that reached the verdict, because the causes
    differ: presolve proves infeasibility from bound arithmetic and keeps no Farkas vector,
    while a simplex route that ends `infeasible` with none has a ray it did not turn into one.
    """
    if status in ("crashed", "no_output"):
        return f"solver {status.replace('_', ' ')}"
    if status != "infeasible":
        # `optimal` or `feasible` on an infeasible model is a wrong answer, not a limit.
        if status in ("optimal", "feasible", "unbounded"):
            return f"wrong verdict: {status}"
        return f"no verdict: {status}"
    if certificate != "farkas" or multipliers == 0:
        return "infeasible without a certificate" + (f" ({algorithm})" if algorithm else "")
    if verifier_rc is None:
        return "not verified"
    if verifier_rc == 2 or "cannot read the model" in verifier_text:
        return "verifier cannot parse the model"
    if verifier_rc != 0 or not PROOF_PASSED.search(verifier_text):
        return "certificate rejected by the verifier"
    return ""


def independently_verified(status: str, certificate: str, multipliers: int,
                           verifier_rc: int | None, verifier_text: str) -> int | str:
    """The CSV's `independently_verified`: 1 or 0 where something was claimed, else "".

    An `infeasible` with no certificate claims nothing the verifier can check, and the
    verifier (rightly) exits 0 on it; writing 1 there would read as a verified proof.
    """
    if verifier_rc is None:
        return ""
    if status == "infeasible":
        if certificate != "farkas" or multipliers == 0:
            return ""
        return int(verifier_rc == 0 and bool(PROOF_PASSED.search(verifier_text)))
    if status in ("optimal", "feasible", "unbounded"):
        return int(verifier_rc == 0)
    return ""


def verifier_failures(text: str) -> str:
    """The verifier's failing lines, or its last line, for the CSV."""
    failing = [line.strip() for line in text.splitlines() if "[FAIL]" in line]
    if failing:
        return "; ".join(failing)[:400]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:400] if lines else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches_manifest(mps: Path, entry: dict) -> bool:
    """Whether `mps` is the instance the manifest recorded, blind to CRLF versus LF."""
    expected = entry.get("mps_lf_sha256")
    if not expected:
        return sha256_file(mps) == entry.get("mps_sha256")
    data = mps.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest() == expected


def default_binary() -> Path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
    import sankhya
    try:
        return sankhya.locate_executable()
    except sankhya.SankhyaError as error:
        raise SystemExit(str(error))


def run_one(binary: Path, mps: Path, time_limit: float, verify: bool,
            solver_options: list[str]) -> dict:
    """Solve one instance, then check the certificate it wrote, independently."""
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats_path), "--write-sol",
                   str(sol_path), "--time-limit", str(time_limit), "--option",
                   "log_to_console=false"]
        for option in solver_options:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats_path.exists():
            status = "crashed" if completed.returncode not in (0, 1) else "no_output"
            return {"status": status, "wall_seconds": wall, "certificate": "",
                    "multipliers": 0, "verifier_rc": None, "verifier_text": "",
                    "message": completed.stderr.strip()[:300]}

        blob = json.loads(stats_path.read_text())
        result = blob.get("result", {})
        model = blob.get("model", {})
        effort = blob.get("effort", {})
        certificate, multipliers = ("", 0)
        if sol_path.exists():
            certificate, multipliers = read_certificate(sol_path.read_text(errors="replace"))
        flat = {
            "status": result.get("status", "unknown"), "message": result.get("message", ""),
            "algorithm": result.get("algorithm", ""), "rows": model.get("rows", ""),
            "columns": model.get("columns", ""), "nonzeros": model.get("nonzeros", ""),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""), "wall_seconds": wall,
            "certificate": certificate, "multipliers": multipliers, "verifier_rc": None,
            "verifier_text": "",
        }
        # Verified whatever the status: a wrong `optimal` deserves the verifier's words too.
        if verify and sol_path.exists():
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol_path)],
                                   capture_output=True, text=True)
            flat["verifier_rc"] = check.returncode
            flat["verifier_text"] = check.stdout + check.stderr
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the verifier; every row then fails as `not verified`")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="passed as --option KEY=VALUE; repeatable, recorded in the CSV. "
                             "Use algorithm=simplex / dual-simplex / pdhg for the per-engine "
                             "runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths resolve against the repo root")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        raise SystemExit("no data/netlib-infeasible/reference.json; run "
                         "bench/runners/fetch_netlib_infeasible.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest["instances"]
    names = sorted(args.instances or instances)
    binary = args.binary or default_binary()
    commit = stamp.stamp(args.binary)
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)

    print(f"solver   {binary}")
    print(f"commit   {commit}   machine {machine}   time limit {args.time_limit:g}s"
          + (f"   options {solver_options}" if solver_options else ""))
    print()
    print(f"{'instance':<10}{'rows':>6}{'cols':>7}  {'status':<16}{'algorithm':<22}"
          f"{'cert':<8}{'mult':>5}{'time':>9}  verified  result")
    print("-" * 106)

    rows: list[dict] = []
    for name in names:
        entry = instances.get(name)
        mps = DATA_DIR / f"{name}.mps"
        if entry is None or not mps.exists():
            print(f"{name:<10} MISSING (run fetch_netlib_infeasible.py)")
            continue
        sha = sha256_file(mps)
        same = matches_manifest(mps, entry)
        flat = run_one(binary, mps, args.time_limit, not args.no_verify, args.solver_option)
        reason = classify(flat["status"], flat["certificate"], flat["multipliers"],
                          flat["verifier_rc"], flat["verifier_text"], flat.get("algorithm", ""))
        verified = independently_verified(flat["status"], flat["certificate"],
                                          flat["multipliers"], flat["verifier_rc"],
                                          flat["verifier_text"])
        rows.append({
            "instance": name, "instance_sha256": sha,
            "matches_manifest": int(same),
            "rows": flat.get("rows", ""), "columns": flat.get("columns", ""),
            "nonzeros": flat.get("nonzeros", ""),
            "expected_status": entry.get("expected_status", "infeasible"),
            "status": flat["status"], "message": flat.get("message", "")[:300],
            "certificate": flat["certificate"], "certificate_multipliers": flat["multipliers"],
            "our_objective": "", "published_objective": "", "absolute_gap": "",
            "relative_gap": "", "independently_verified": verified,
            "verifier_message": verifier_failures(flat["verifier_text"]),
            "passed": int(reason == ""), "failure_reason": reason,
            "wall_seconds": round(flat.get("wall_seconds", 0.0), 6),
            "solver_seconds": flat.get("solver_seconds", ""),
            "iterations": flat.get("iterations", ""), "algorithm": flat.get("algorithm", ""),
            "time_limit": args.time_limit, "git_commit": commit, "machine": machine,
            "timestamp_utc": timestamp, "solver_options": solver_options,
        })
        mark = {"": "  -     ", 0: "  NO    ", 1: "  yes   "}[verified]
        print(f"{name:<10}{str(flat.get('rows', '')):>6}{str(flat.get('columns', '')):>7}  "
              f"{flat['status']:<16}{flat.get('algorithm', ''):<22}"
              f"{flat['certificate'] or '-':<8}{flat['multipliers']:>5}"
              f"{flat.get('wall_seconds', 0.0):>8.2f}s{mark}  "
              f"{'PASS' if not reason else 'FAIL: ' + reason}")
        if not same:
            print(f"{'':10}the file on disk is not the one the manifest recorded")

    passes = sum(row["passed"] for row in rows)
    print("-" * 106)
    print(f"{passes}/{len(rows)} reported infeasible with a Farkas certificate that "
          f"tools/verify_solution.py accepted")
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["failure_reason"]:
            grouped.setdefault(row["failure_reason"], []).append(row["instance"])
    # Every failure named, with its cause. A pass count without them is a claim.
    for reason, failed in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {reason}: {', '.join(failed)}")

    engine = next((option.split("=", 1)[1] for option in args.solver_option
                   if option.startswith("algorithm=")), "")
    out_path = args.out or (RESULTS_DIR / f"netlib-infeasible-{commit}"
                            f"{'-' + engine if engine else ''}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0 if rows and passes == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
