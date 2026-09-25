#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: agreement of objectives against size-limited commercial solver editions (#533).

WHY THIS EXISTS.

The problem statement names commercial solvers as the thing to replace. A direct head-to-head
on the instances that fit each edition's size limits is the clearest statement of where we
stand. This runner invokes each solver as a separate process over the same MPS files and
records objective, status, wall time and node count in a CSV for side-by-side comparison.

LICENCE TERMS (checked 2026-09-24, recorded here per the issue's acceptance criterion):
  - Gurobi Academic/Trial (free, <=2000 variables + constraints): benchmarking permitted under
    academic/evaluation licence terms as of 2024; see docs/PROVENANCE.md judgement call.
  - CPLEX Community Edition (free, <=1000 variables + constraints): benchmarking is permitted
    under IBM's community licence for non-commercial academic use.
  - HiGHS (open source, MIT): always permitted.
  - GLPK (GPL-3.0, no size limit): always permitted.

WHAT THIS IS NOT.

This does not grade SANKHYA against itself. Agreement is what is being measured, not timing:
on Netlib-scale instances every modern solver is fast. A timing comparison on larger instances
is a separate measurement that belongs in a different runner, on rented hardware, with the
results in a separate CSV.

SIZE LIMITS (conservative; each solver's own docs are authoritative):
  GUROBI_LIMIT_VARS  = 2000   # variables
  GUROBI_LIMIT_CONS  = 2000   # linear constraints
  CPLEX_LIMIT_VARS   = 1000
  CPLEX_LIMIT_CONS   = 1000

Usage:
    python bench/runners/commercial_agreement.py                  # all available solvers, all fetched instances
    python bench/runners/commercial_agreement.py --solver gurobi  # one solver only
    python bench/runners/commercial_agreement.py --suite netlib   # one instance set
    python bench/runners/commercial_agreement.py afiro scrs8      # named instances
    python bench/runners/commercial_agreement.py --dry-run        # show plan, solve nothing
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import stamp as _stamp

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = {
    "netlib": REPO_ROOT / "data" / "netlib",
    "miplib": REPO_ROOT / "data" / "miplib",
    "maros_meszaros": REPO_ROOT / "data" / "maros_meszaros",
}
RESULTS_DIR = REPO_ROOT / "bench" / "results"
SANKHYA_BIN = REPO_ROOT / "build" / "sankhya"

GUROBI_LIMIT_VARS = 2000
GUROBI_LIMIT_CONS = 2000
CPLEX_LIMIT_VARS = 1000
CPLEX_LIMIT_CONS = 1000

AGREEMENT_TOL = 1e-6

FIELDS = [
    "instance",
    "instance_sha256",
    "solver",
    "our_objective",
    "our_status",
    "our_wall_s",
    "our_nodes",
    "ref_objective",
    "ref_status",
    "ref_wall_s",
    "ref_nodes",
    "abs_gap",
    "rel_gap",
    "verdict",
    "git_commit",
    "machine",
    "timestamp_utc",
]


@dataclass
class Result:
    instance: str = ""
    path: Path = field(default_factory=Path)
    solver: str = ""
    our_objective: Optional[float] = None
    our_status: str = ""
    our_wall_s: float = 0.0
    our_nodes: int = 0
    ref_objective: Optional[float] = None
    ref_status: str = ""
    ref_wall_s: float = 0.0
    ref_nodes: int = 0

    @property
    def abs_gap(self) -> str:
        if self.our_objective is None or self.ref_objective is None:
            return "n/a"
        return f"{abs(self.our_objective - self.ref_objective):.6g}"

    @property
    def rel_gap(self) -> str:
        if self.our_objective is None or self.ref_objective is None:
            return "n/a"
        denom = max(abs(self.ref_objective), 1e-10)
        return f"{abs(self.our_objective - self.ref_objective) / denom:.6g}"

    @property
    def verdict(self) -> str:
        if self.our_objective is None or self.ref_objective is None:
            if self.our_status == self.ref_status:
                return "status_agree"
            return f"status_disagree(ours={self.our_status},ref={self.ref_status})"
        denom = max(abs(self.ref_objective), 1e-10)
        if abs(self.our_objective - self.ref_objective) / denom <= AGREEMENT_TOL:
            return "agree"
        sign = "+" if self.our_objective > self.ref_objective else "-"
        return f"disagree({sign}{self.rel_gap})"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(binary: Optional[Path] = None) -> str:
    return _stamp.stamp(str(binary) if binary else None)


def machine_tag() -> str:
    return platform.node() or platform.machine() or "unknown"


# ---------------------------------------------------------------------------
# Instance discovery and size probing
# ---------------------------------------------------------------------------

def mps_dimensions(path: Path) -> tuple[int, int]:
    """Return (num_vars, num_cons) by scanning the MPS file's ROWS/COLUMNS sections."""
    rows: set[str] = set()
    cols: set[str] = set()
    section = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                upper = stripped.upper()
                for sec in ("NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "RANGES",
                            "ENDATA", "OBJSENSE", "SETS", "SOS", "INDICATORS"):
                    if upper.startswith(sec) and (len(upper) == len(sec) or not upper[len(sec)].isalpha()):
                        section = sec
                        break
                else:
                    parts = stripped.split()
                    if section == "ROWS" and len(parts) >= 2 and parts[0] != "N":
                        rows.add(parts[1])
                    elif section == "COLUMNS" and len(parts) >= 2:
                        cols.add(parts[0])
    except OSError:
        pass
    return len(cols), len(rows)


def fits_limit(path: Path, max_vars: int, max_cons: int) -> bool:
    v, c = mps_dimensions(path)
    return v <= max_vars and c <= max_cons


def discover_instances(suite: str) -> list[Path]:
    data_dir = DATA_DIRS.get(suite)
    if data_dir is None or not data_dir.exists():
        return []
    patterns = ["*.mps", "*.mps.gz", "*.MPS"]
    instances: list[Path] = []
    for pat in patterns:
        instances.extend(sorted(data_dir.glob(pat)))
    return instances


# ---------------------------------------------------------------------------
# Solving with SANKHYA
# ---------------------------------------------------------------------------

def run_sankhya(path: Path, binary: Path, time_limit: float = 120.0) -> tuple[Optional[float], str, float, int]:
    """Returns (objective, status_str, wall_s, nodes)."""
    if not binary.exists():
        return None, "binary_not_found", 0.0, 0
    cmd = [str(binary), str(path), "--json", "--time-limit", str(time_limit)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 10)
        wall_s = time.monotonic() - t0
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None, "parse_error", wall_s, 0
        status = data.get("status", "unknown")
        obj = data.get("objective")
        nodes = data.get("nodes", 0)
        return (float(obj) if obj is not None else None), status, wall_s, int(nodes)
    except subprocess.TimeoutExpired:
        return None, "timeout", time.monotonic() - t0, 0
    except Exception as e:
        return None, f"error:{e}", time.monotonic() - t0, 0


# ---------------------------------------------------------------------------
# Solving with reference solvers (separate processes, no linkage)
# ---------------------------------------------------------------------------

def _parse_gurobi_log(stdout: str, stderr: str) -> tuple[Optional[float], str, int]:
    """Extract objective, status and node count from Gurobi command-line output."""
    obj: Optional[float] = None
    status = "unknown"
    nodes = 0
    for line in (stdout + stderr).splitlines():
        m = re.search(r"Best objective\s+([-\d.e+]+)", line, re.I)
        if m:
            try:
                obj = float(m.group(1))
            except ValueError:
                pass
        m2 = re.search(r"Explored\s+(\d+)\s+nodes", line, re.I)
        if m2:
            nodes = int(m2.group(1))
        if "Optimal solution found" in line:
            status = "optimal"
        elif "Infeasible" in line:
            status = "infeasible"
        elif "Time limit" in line:
            status = "time_limit"
    return obj, status, nodes


def run_gurobi(path: Path, time_limit: float = 120.0) -> tuple[Optional[float], str, float, int]:
    gurobi = shutil.which("gurobi_cl") or shutil.which("gurobi")
    if gurobi is None:
        return None, "not_installed", 0.0, 0
    cmd = [gurobi, f"TimeLimit={time_limit}", str(path)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 10)
        wall_s = time.monotonic() - t0
        obj, status, nodes = _parse_gurobi_log(proc.stdout, proc.stderr)
        return obj, status, wall_s, nodes
    except subprocess.TimeoutExpired:
        return None, "timeout", time.monotonic() - t0, 0
    except Exception as e:
        return None, f"error:{e}", time.monotonic() - t0, 0


def _parse_cplex_log(stdout: str, stderr: str) -> tuple[Optional[float], str, int]:
    obj: Optional[float] = None
    status = "unknown"
    nodes = 0
    for line in (stdout + stderr).splitlines():
        m = re.search(r"Objective\s*=\s*([-\d.e+]+)", line, re.I)
        if m:
            try:
                obj = float(m.group(1))
            except ValueError:
                pass
        m2 = re.search(r"(\d+)\s+nodes", line, re.I)
        if m2:
            nodes = int(m2.group(1))
        if "MIP solution" in line or "optimal" in line.lower():
            status = "optimal"
        elif "infeasible" in line.lower():
            status = "infeasible"
    return obj, status, nodes


def run_cplex(path: Path, time_limit: float = 120.0) -> tuple[Optional[float], str, float, int]:
    cplex = shutil.which("cplex") or shutil.which("cpoptimizer")
    if cplex is None:
        return None, "not_installed", 0.0, 0
    # CPLEX interactive batch via -c flag or concert.
    script = f"read {path}\nmip\nset timelimit {time_limit}\noptimize\nquit\n"
    cmd = [cplex]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=time_limit + 10)
        wall_s = time.monotonic() - t0
        obj, status, nodes = _parse_cplex_log(proc.stdout, proc.stderr)
        return obj, status, wall_s, nodes
    except subprocess.TimeoutExpired:
        return None, "timeout", time.monotonic() - t0, 0
    except Exception as e:
        return None, f"error:{e}", time.monotonic() - t0, 0


def run_glpk(path: Path, time_limit: float = 120.0) -> tuple[Optional[float], str, float, int]:
    glpsol = shutil.which("glpsol")
    if glpsol is None:
        return None, "not_installed", 0.0, 0
    cmd = [glpsol, "--mps", str(path), "--tmlim", str(int(time_limit)), "-o", "/dev/null"]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 10)
        wall_s = time.monotonic() - t0
        obj: Optional[float] = None
        status = "unknown"
        for line in (proc.stdout + proc.stderr).splitlines():
            m = re.search(r"obj =\s*([-\d.e+]+)", line, re.I)
            if m:
                try:
                    obj = float(m.group(1))
                except ValueError:
                    pass
            if "INTEGER OPTIMAL" in line.upper() or "OPTIMAL LP SOLUTION" in line.upper():
                status = "optimal"
            elif "INFEASIBLE" in line.upper():
                status = "infeasible"
            elif "TIME LIMIT" in line.upper():
                status = "time_limit"
        return obj, status, wall_s, 0
    except subprocess.TimeoutExpired:
        return None, "timeout", time.monotonic() - t0, 0
    except Exception as e:
        return None, f"error:{e}", time.monotonic() - t0, 0


SOLVERS = {
    "gurobi": (run_gurobi, GUROBI_LIMIT_VARS, GUROBI_LIMIT_CONS),
    "cplex": (run_cplex, CPLEX_LIMIT_VARS, CPLEX_LIMIT_CONS),
    "glpk": (run_glpk, 10_000_000, 10_000_000),  # no size limit for GLPK
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    binary = Path(args.binary) if args.binary else SANKHYA_BIN
    commit = git_commit(binary)
    machine = machine_tag()

    # Choose instance files.
    paths: list[Path] = []
    if args.instances:
        for suite_dir in DATA_DIRS.values():
            if suite_dir.exists():
                for p in suite_dir.iterdir():
                    if any(p.stem == inst or p.name == inst for inst in args.instances):
                        paths.append(p)
    elif args.suite:
        paths = discover_instances(args.suite)
    else:
        for suite in DATA_DIRS:
            paths.extend(discover_instances(suite))
    if not paths:
        print("No instances found.", file=sys.stderr)
        return 1

    # Choose solvers — Gurobi requires --gurobi-academic (see PROVENANCE.md row 16).
    solver_names = args.solver if args.solver else list(SOLVERS.keys())
    if "gurobi" in solver_names and not getattr(args, "gurobi_academic", False):
        print(
            "error: Gurobi's standard EULA forbids publishing benchmark results.\n"
            "Only an academic licence waives that restriction.\n"
            "Pass --gurobi-academic to confirm you hold one (see docs/PROVENANCE.md row 16).",
            file=sys.stderr,
        )
        solver_names = [s for s in solver_names if s != "gurobi"]
        if not solver_names:
            return 1

    if args.dry_run:
        for solver in solver_names:
            fn, max_v, max_c = SOLVERS[solver]
            eligible = [p for p in paths if fits_limit(p, max_v, max_c)]
            print(f"[{solver}] {len(eligible)}/{len(paths)} instances within size limit "
                  f"({max_v} vars, {max_c} cons)")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    csv_path = RESULTS_DIR / f"commercial_agreement-{commit}-{ts}.csv"
    rows = []

    for solver in solver_names:
        ref_fn, max_v, max_c = SOLVERS[solver]
        eligible = [p for p in paths if fits_limit(p, max_v, max_c)]
        print(f"[{solver}] {len(eligible)} instances within size limit", flush=True)
        for path in sorted(eligible):
            print(f"  {path.stem} ...", end=" ", flush=True)
            sha = sha256_file(path)
            our_obj, our_status, our_wall, our_nodes = run_sankhya(path, binary, args.time_limit)
            ref_obj, ref_status, ref_wall, ref_nodes = ref_fn(path, args.time_limit)
            r = Result(
                instance=path.stem,
                path=path,
                solver=solver,
                our_objective=our_obj,
                our_status=our_status,
                our_wall_s=our_wall,
                our_nodes=our_nodes,
                ref_objective=ref_obj,
                ref_status=ref_status,
                ref_wall_s=ref_wall,
                ref_nodes=ref_nodes,
            )
            print(r.verdict, flush=True)
            rows.append({
                "instance": r.instance,
                "instance_sha256": sha,
                "solver": solver,
                "our_objective": "" if our_obj is None else f"{our_obj:.15g}",
                "our_status": our_status,
                "our_wall_s": f"{our_wall:.3f}",
                "our_nodes": our_nodes,
                "ref_objective": "" if ref_obj is None else f"{ref_obj:.15g}",
                "ref_status": ref_status,
                "ref_wall_s": f"{ref_wall:.3f}",
                "ref_nodes": ref_nodes,
                "abs_gap": r.abs_gap,
                "rel_gap": r.rel_gap,
                "verdict": r.verdict,
                "git_commit": commit,
                "machine": machine,
                "timestamp_utc": ts,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Summary.
    agree = sum(1 for r in rows if r.get("verdict", "") == "agree" or r.get("verdict", "") == "status_agree")
    disagree = sum(1 for r in rows if r.get("verdict", "").startswith("disagree"))
    other = len(rows) - agree - disagree
    print(f"agree={agree}  disagree={disagree}  other={other}")
    return 0 if disagree == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", help="Named instances (default: all)")
    ap.add_argument("--solver", action="append", choices=list(SOLVERS.keys()),
                    help="Which solver(s) to compare against (default: all detected)")
    ap.add_argument("--gurobi-academic", action="store_true",
                    help="Confirm an academic Gurobi licence is in use (required to run "
                         "Gurobi; the standard EULA forbids publishing benchmark results "
                         "and only the academic licence waives that restriction — see "
                         "docs/PROVENANCE.md row 16)")
    ap.add_argument("--suite", choices=list(DATA_DIRS.keys()),
                    help="Instance suite (default: all)")
    ap.add_argument("--binary", help="Path to the sankhya binary (default: build/sankhya)")
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="Per-instance time limit in seconds (default: 120)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without solving")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
