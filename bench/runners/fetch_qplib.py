#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the convex continuous QPs of QPLIB (#492), with every reference parsed, never typed.

    F. Furini et al., "QPLIB: a library of quadratic programming instances", Mathematical
    Programming Computation 11 (2019), DOI 10.1007/s12532-018-0147-4; https://qplib.zib.de

THE SELECTION is read off the site's instance listing, https://qplib.zib.de/instances.html,
whose columns are defined on https://qplib.zib.de/doc.html:

*   `Cvx` ("Continuous Relaxation convex", doc.html CONVEX) is a tick: the objective is convex
    (concave for a maximization) and the constraints are convex;
*   `O`, `V`, `C` are the three letters of PROBTYPE. O is L (linear), D (convex for min /
    concave for max with a symmetric - in the site's triangular convention, DIAGONAL - Q^0),
    C (convex / concave) or Q (otherwise). V is C when no variable is integer. C is N (no
    constraints, no bounds), B (bounds only), L (linear constraints), or D, C, Q for
    quadratic constraints.

A convex continuous QP with linear constraints is therefore: Cvx ticked, O in {C, D}, V = C,
C in {N, B, L}. Convex continuous instances outside it are named in the manifest, not dropped:
type L*D (a linear objective under convex quadratic constraints) is a QCQP, #514's test set.

THE REFERENCE for each selected instance comes from two of QPLIB's own publications, which
must agree: `qplib.solu` (linked from instances.html, "the objective value of all solution
points", one `=tag= QPLIB_xxxx value` line each, sixteen decimals) and the solobjvalue row
of the instance's own page (eight printed decimals). The solu value is the one compared
against, because the page's fixed eight decimals leave QPLIB_8790's -0.0001562421091 with
five significant digits. QPLIB calls these SOLUTION POINTS, not optima: the tag is `=best=`
and the site's history describes them as feasible points found by a variety of solvers under
a 12-hour limit, some improved since. Every entry records that status beside its value.

THE CONVERSION to QPS (qplib_format.py) is checked here on every instance: QPLIB's published
point (`sol/QPLIB_xxxx.sol`) is read against the converted file by the independent verifier's
MPS reader, and the objective there must reproduce the point's own objective to 1e-9
relative, and its constraint and bound violation must not exceed what the page reports
(solinfeasibility) by more than 1e-7. An instance whose check fails is marked so and the
runner does not solve it.

SIZE. The tiers are by stored coefficients (the listing's NZ, the nonzeros of the objective
gradient and Jacobian, plus the page's nobjquadnz): `small` is at most 100,000, the default
fetch takes everything up to --max-coefficients (2,000,000) and records the larger ones as
skipped without downloading them; --max-coefficients 0 lifts the cap.

Instance files, solutions and pages go to data/qplib/ (ignored by git); data/qplib/reference.json
is the committed manifest, with the URL and sha256 of every file read.

    python bench/runners/fetch_qplib.py
    python bench/runners/fetch_qplib.py --max-coefficients 0   # also the two largest
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import qplib_format  # noqa: E402
from fetch_mittelmann import download  # noqa: E402  (retries, a .part file, curl fallback)
from verify_solution_mps import parse_mps  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "qplib"
BASE_URL = "https://qplib.zib.de/"
SITE_FILES = ["index.html", "instances.html", "doc.html", "qplib.solu"]

CITATION = ("F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould, "
            "L. Liberti, A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis, S. Vigerske, "
            "A. Wiegele, QPLIB: a library of quadratic programming instances, Mathematical "
            "Programming Computation 11 (2019), DOI 10.1007/s12532-018-0147-4")
SELECTION_RULE = ("instances.html: Cvx ticked, O in {C, D}, V = C, C in {N, B, L} "
                  "(doc.html PROBTYPE and CONVEX)")
SMALL_COEFFICIENTS = 100_000
DEFAULT_MAX_COEFFICIENTS = 2_000_000
OBJECTIVE_CHECK = 1e-9  # relative, converted QPS at QPLIB's point vs the point's objvar
VIOLATION_SLACK = 1e-7  # absolute, over the page's solinfeasibility


# ---- the site's pages --------------------------------------------------------------------

def _cell_text(cell: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).replace("\xa0", " ").strip()


def _integer(text: str) -> int:
    return int(text) if text.strip() else 0


def parse_listing(page: str) -> list[dict]:
    """Every row of instances.html's table, in the column order its header declares."""
    header = re.search(r"<THEAD.*?</THEAD>", page, re.S | re.I)
    body = re.search(r"<TBODY>(.*?)</TBODY>", page, re.S | re.I)
    if not header or not body:
        raise ValueError("instances.html has no instance table")
    titles = [_cell_text(cell) for cell in re.findall(r"<TH[^>]*>(.*?)</TH>", header.group(0),
                                                      re.S | re.I)]
    expected = ["Instance", "Cvx", "O", "Q0density", "Q0probl.ev", "V", "TotalVars.",
                "BinaryVars.", "IntegerVars.", "C", "TotalCons.", "Quad.Cons.", "Non-zeros"]
    if [re.sub(r"\s+", "", t) for t in titles] != expected:
        raise ValueError(f"instances.html columns changed: {titles}")
    rows = []
    for raw in re.findall(r"<TR[^>]*>(.*?)</TR>", body.group(1), re.S | re.I):
        cells = re.findall(r"<TD[^>]*>(.*?)</TD>", raw, re.S | re.I)
        if len(cells) != len(expected):
            raise ValueError(f"a listing row has {len(cells)} cells, not {len(expected)}")
        page_link = re.search(r'href="?(QPLIB_\d+)\.html', cells[0])
        qplib_link = re.search(r'href="(qplib/QPLIB_\d+\.qplib)"', cells[0])
        if not page_link:
            raise ValueError(f"a listing row names no instance page: {cells[0][:80]}")
        text = [_cell_text(cell) for cell in cells]
        rows.append({
            "name": page_link.group(1), "convex": text[1] != "-", "cvx_cell": text[1],
            "o": text[2], "v": text[5], "c": text[9], "nvars": _integer(text[6]),
            "ncons": _integer(text[10]), "nz": _integer(text[12]),
            "qplib_path": qplib_link.group(1) if qplib_link else None,
        })
    return rows


def selected(row: dict) -> bool:
    return row["convex"] and row["o"] in ("C", "D") and row["v"] == "C" and row["c"] in (
        "N", "B", "L")


def parse_instance_page(page: str) -> dict:
    """The page's identifier -> value table (the identifiers are doc.html's anchors), plus
    the link to the solution point file."""
    fields = {}
    for key, cell in re.findall(r'doc\.html#(\w+)"><SUP>\w+</SUP></a></TD>\s*<TD>(.*?)</TD>',
                                page, re.S | re.I):
        fields[key] = _cell_text(cell)
    sol = re.search(r'href="(sol/QPLIB_\d+\.sol)"', page)
    fields["sol_path"] = sol.group(1) if sol else None
    value = fields.get("solobjvalue", "").split()
    fields["solobjvalue_text"] = value[0] if value else None
    return fields


def parse_solu(text: str) -> dict[str, tuple[str, float | None, str | None]]:
    """qplib.solu: `=tag= QPLIB_xxxx [value]` per line -> {name: (tag, value, text)}."""
    out = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if not re.fullmatch(r"=\w+=", tokens[0]) or len(tokens) < 2:
            raise ValueError(f"qplib.solu line not understood: {line!r}")
        text_value = tokens[2] if len(tokens) > 2 else None
        if tokens[1] in out:
            raise ValueError(f"qplib.solu lists {tokens[1]} twice")
        out[tokens[1]] = (tokens[0].strip("="), None if text_value is None else float(text_value),
                          text_value)
    return out


def parse_home(page: str) -> dict:
    counts = re.search(r"QPLIB contains\s+(\d+)(?:\s|&nbsp;)+discrete\s+and\s+(\d+)"
                       r"(?:\s|&nbsp;)+continuous", page)
    licence = re.findall(r'QPLIB is licensed under <a href="([^"]+)">([^<]+)</a>', page)
    if not counts or not licence:
        raise ValueError("index.html no longer states the instance counts and the licence")
    return {"discrete": int(counts.group(1)), "continuous": int(counts.group(2)),
            "licence": licence[-1][1], "licence_url": licence[-1][0],
            "licence_statement": f"QPLIB is licensed under {licence[-1][1]}."}


def page_consistent(row: dict, fields: dict) -> list[str]:
    """What the instance page says against what the listing said; empty when they agree."""
    problems = []
    if fields.get("probtype") != row["o"] + row["v"] + row["c"]:
        problems.append(f"probtype {fields.get('probtype')} vs listing "
                        f"{row['o'] + row['v'] + row['c']}")
    sense, curvature = fields.get("objsense"), fields.get("objcurvature")
    if (sense, curvature) not in (("min", "convex"), ("max", "concave")):
        problems.append(f"objective {sense} {curvature} is not convex")
    if fields.get("conscurvature") != "linear":
        problems.append(f"constraints {fields.get('conscurvature')}")
    for key, listed in (("nvars", row["nvars"]), ("ncons", row["ncons"])):
        if _integer(fields.get(key, "")) != listed:
            problems.append(f"{key} {fields.get(key)} vs listing {listed}")
    return problems


def references_agree(page_text: str | None, solu_value: float | None) -> bool:
    """The page prints eight decimals; the solu value must round to what it prints."""
    if page_text is None or solu_value is None:
        return page_text is None and solu_value is None
    return abs(float(page_text) - solu_value) <= 0.5e-8 + 1e-15 * abs(solu_value)


# ---- the conversion check ----------------------------------------------------------------

def converter_check(qps: Path, model: qplib_format.QplibModel, sol_text: str,
                    solinfeasibility: float | None) -> dict:
    """QPLIB's own point, evaluated on the converted QPS by the verifier's reader."""
    x, objvar = qplib_format.read_solution(sol_text, model)
    reread = parse_mps(qps)
    point = [0.0] * reread.num_cols
    for j in range(model.n):
        point[reread.col_index[qplib_format.column_name(j)]] = x[j]
    objective = (reread.objective_offset + sum(c * v for c, v in zip(reread.col_cost, point))
                 + reread.quadratic_objective(point))
    activity = [0.0] * reread.num_rows
    for j, column in enumerate(reread.entries):
        for i, value in column:
            activity[i] += value * point[j]
    violation = 0.0
    for i in range(reread.num_rows):
        violation = max(violation, reread.row_lower[i] - activity[i],
                        activity[i] - reread.row_upper[i])
    for j in range(reread.num_cols):
        violation = max(violation, reread.col_lower[j] - point[j], point[j] - reread.col_upper[j])
    objective_ok = objvar is not None and math.isfinite(objective) and (
        abs(objective - objvar) <= OBJECTIVE_CHECK * max(1.0, abs(objvar)))
    violation_ok = violation <= (solinfeasibility or 0.0) + VIOLATION_SLACK
    return {"published_point_objective": objvar, "objective_at_point_via_qps": objective,
            "violation_at_point_via_qps": violation, "passed": bool(objective_ok and violation_ok)}


# ---- the fetch ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(relative: str, force: bool) -> tuple[Path, dict]:
    target = DATA_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + relative
    if force or not target.exists():
        print(f"fetching {url}")
        download(url, target)
    return target, {"url": url, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download present files")
    parser.add_argument("--max-coefficients", type=int, default=DEFAULT_MAX_COEFFICIENTS,
                        help="skip instances with more stored coefficients; 0 for no cap")
    args = parser.parse_args()

    files = {}
    for name in SITE_FILES:
        # The listing, the solu file and the home page change as QPLIB is updated, so they
        # are re-read every time; the manifest records the sha256 of what was read.
        _, files[name] = fetch(name, force=True)
    home = parse_home((DATA_DIR / "index.html").read_text(encoding="utf-8"))
    rows = parse_listing((DATA_DIR / "instances.html").read_text(encoding="utf-8"))
    continuous = [row for row in rows if row["v"] == "C"]
    if len(rows) != home["discrete"] + home["continuous"] or \
            len(continuous) != home["continuous"]:
        raise SystemExit(f"instances.html lists {len(rows)} rows, {len(continuous)} continuous; "
                         f"index.html says {home['discrete']} discrete and {home['continuous']} "
                         "continuous. Refusing to select from a listing that was misread.")
    solu = parse_solu((DATA_DIR / "qplib.solu").read_text(encoding="utf-8"))

    classes: dict[str, int] = {}
    for row in continuous:
        if row["convex"]:
            code = row["o"] + row["v"] + row["c"]
            classes[code] = classes.get(code, 0) + 1
    picked = [row for row in rows if selected(row)]
    instances, problems = {}, []
    for row in picked:
        name = row["name"]
        page_path, page_file = fetch(f"{name}.html", args.force)
        fields = parse_instance_page(page_path.read_text(encoding="utf-8"))
        mismatch = page_consistent(row, fields)
        tag, solu_value, solu_text = solu.get(name, (None, None, None))
        if not references_agree(fields["solobjvalue_text"], solu_value):
            mismatch.append(f"solobjvalue {fields['solobjvalue_text']} vs qplib.solu {solu_text}")
        if mismatch:
            problems.append(f"{name}: " + "; ".join(mismatch))
            continue
        coefficients = row["nz"] + _integer(fields.get("nobjquadnz", ""))
        solinf = fields.get("solinfeasibility") or None
        entry = {
            "name": name, "problem_type": fields["probtype"], "nvars": row["nvars"],
            "ncons": row["ncons"], "nz": row["nz"],
            "nobjquadnz": _integer(fields.get("nobjquadnz", "")), "coefficients": coefficients,
            "tier": "small" if coefficients <= SMALL_COEFFICIENTS else "full",
            "page": page_file,
            "reference_objective": solu_value, "reference_text": solu_text,
            "reference_status": None if tag is None else f"qplib.solu tag ={tag}=",
            "page_solobjvalue_text": fields["solobjvalue_text"],
            "solinfeasibility": None if solinf is None else float(solinf),
            "reference_source": (f"{BASE_URL}qplib.solu (={tag}=, a best known solution point, "
                                 f"not a proof of optimality); agrees with solobjvalue on "
                                 f"{BASE_URL}{name}.html" if solu_value is not None else
                                 f"none: {BASE_URL}{name}.html publishes no solution point"),
        }
        if args.max_coefficients and coefficients > args.max_coefficients:
            entry["skipped"] = (f"{coefficients} coefficients > --max-coefficients "
                                f"{args.max_coefficients}; not downloaded")
            instances[name] = entry
            continue
        qplib_path, entry["qplib"] = fetch(row["qplib_path"], args.force)
        model = qplib_format.read(qplib_path)
        qps_path = DATA_DIR / f"{name}.qps"
        qps_path.write_text(qplib_format.to_qps(model), encoding="utf-8", newline="\n")
        entry["qps"] = {"converted_by": "bench/runners/qplib_format.py",
                        "bytes": qps_path.stat().st_size, "sha256": sha256_file(qps_path)}
        entry["q0_entries"] = len(model.q0)
        if fields["sol_path"]:
            sol_path, entry["sol"] = fetch(fields["sol_path"], args.force)
            entry["converter_check"] = converter_check(
                qps_path, model, sol_path.read_text(encoding="latin-1"), entry["solinfeasibility"])
        else:
            entry["converter_check"] = {"passed": None, "note": "no published point to check at"}
        instances[name] = entry
        check = entry["converter_check"]
        print(f"  {name} {entry['problem_type']} {coefficients:>9} coeffs  reference "
              f"{solu_text}  check {check.get('passed')}  "
              f"objective at QPLIB's point via QPS {check.get('objective_at_point_via_qps')}")
    if problems:
        print("THE LISTING AND THE INSTANCE PAGES DISAGREE:")
        for line in problems:
            print(f"  {line}")
        return 1

    manifest = {
        "source": BASE_URL, "citation": CITATION,
        "licence": {key: home[key] for key in ("licence", "licence_url", "licence_statement")},
        "selection_rule": SELECTION_RULE,
        "convex_continuous_classes": dict(sorted(classes.items())),
        "listing_counts": {"rows": len(rows), "continuous": len(continuous),
                           "selected": len(picked)},
        "tiers": {"small_max_coefficients": SMALL_COEFFICIENTS,
                  "max_coefficients": args.max_coefficients or None},
        "objective_convention": ("0.5 x'Q0 x + b0'x + q0 with Q0 the lower triangle as the "
                                 ".qplib file lists it (doc.html); see qplib_format.py"),
        "files": files, "instances": dict(sorted(instances.items())),
    }
    (DATA_DIR / "reference.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8", newline="\n")
    fetched = [e for e in instances.values() if "skipped" not in e]
    failed = [e["name"] for e in fetched if e["converter_check"].get("passed") is False]
    print(f"wrote data/qplib/reference.json: {len(picked)} selected ({SELECTION_RULE}); "
          f"{len(fetched)} fetched, {sum(e['tier'] == 'small' for e in fetched)} in the small "
          f"tier, {len(instances) - len(fetched)} over the size cap; converter check failed on "
          f"{len(failed)}" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
