#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kennington set's parsing and pass rule (#530): the readme table, the
published optimum's precision, the set selection, and the pass rule. Pure Python, no network.

    python bench/runners/test_kennington.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_kennington  # noqa: E402
import kennington  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# Shaped like lp/data/kennington/readme: prose with numbers in it, the header, rows with a
# negative optimum and one with a large MPS size, and prose after the table.
README = """\
The "Kennington" problems, sixteen problems described in "An Empirical
Evaluation of the KORBX Algorithms" (Operations Research vol. 38, no. 2 (1990), pp. 240-248)
some of the cost coefficients are in columns 26-37 (or 26-38)

Name       rows  columns  nonzeros  bounds      mpc      MPS     optimal value

CRE-A      3517    4067     19054        0    152726    659682   2.3595407e+07
KEN-07     2427    3602     11981     7204    150525    718748  -6.7952044e+08
OSA-60    10281  232966   1630758        0  10377094  52402461   4.0440725e+06
PDS-02     2954    7535     21252     2134    197821    801690   2.8857862e+10


Thanks go to Irv Lustig for transmitting these problems.
"""


def test_summary_table() -> None:
    table = fetch_kennington.parse_summary_table(README)
    check(sorted(table) == ["cre-a", "ken-07", "osa-60", "pds-02"],
          "exactly the table's rows are read, keyed lower-case", str(sorted(table)))
    ken = table["ken-07"]
    check(ken["published_optimal"] == -6.7952044e+08, "a negative optimum is parsed",
          repr(ken["published_optimal"]))
    check(ken["published_optimal_text"] == "-6.7952044e+08", "the printed text is kept")
    check(ken["published_rows"] == 2427 and ken["published_cols"] == 3602
          and ken["published_nonzeros"] == 11981 and ken["published_bounds"] == 7204,
          "counts are parsed")
    check(table["osa-60"]["published_mps_bytes"] == 52402461, "the MPS size is parsed")
    check(all(e["published_significant_digits"] == 8 for e in table.values()),
          "the optima carry eight significant figures")
    check(fetch_kennington.parse_summary_table("CRE-A 1 2 3\nno table") == {},
          "a short row and prose yield nothing")


def test_significant_digits() -> None:
    digits = fetch_kennington.significant_digits
    check(digits("2.3595407e+07") == 8, "2.3595407e+07 has 8")
    check(digits("-6.7952044e+08") == 8, "the sign is not a digit")
    check(digits("1.5E+02") == 2, "upper-case exponent")


def test_select() -> None:
    table = fetch_kennington.parse_summary_table(README)
    check(fetch_kennington.select(table, "small") == ["cre-a", "ken-07", "pds-02"],
          "small is published MPS size <= 1 MB", str(fetch_kennington.select(table, "small")))
    check(fetch_kennington.select(table, "full") == sorted(table), "full is everything")


def test_pass_rule() -> None:
    gap = kennington.relative_gap
    check(gap(None, 1.0) is None, "no objective, no gap")
    check(abs(gap(2.3595407e+07 * (1 + 4e-7), 2.3595407e+07) - 4e-7) < 1e-12,
          "the gap is relative to the published value")
    check(gap(0.5, 0.0) == 0.5, "an optimum of zero is measured absolutely")
    # Eight printed figures move a value by at most 5e-8 relative; that is a pass.
    rounded = gap(2.35954074999e+07, 2.3595407e+07)
    check(rounded is not None and rounded < 5e-8 and kennington.verdict("optimal", rounded, True),
          "rounding in the eighth figure passes", f"{rounded:.1e}")
    check(not kennington.verdict("optimal", 2e-6, True), "a gap above 1e-6 fails")
    check(not kennington.verdict("optimal", 0.0, False), "a verifier rejection fails")
    check(kennington.verdict("optimal", 0.0, None), "an unverified match passes, as in netlib.py")
    check(not kennington.verdict("time_limit", 0.0, None), "a limit is not a pass")


def test_manifest_hash() -> None:
    lf = b"NAME X\nROWS\n N COST\nENDATA\n"
    crlf = lf.replace(b"\n", b"\r\n")
    check(fetch_kennington.lf_sha256(lf) == fetch_kennington.lf_sha256(crlf),
          "CRLF and LF give the same manifest hash")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.mps"
        path.write_bytes(crlf)
        entry = {"mps_sha256": "not this", "mps_lf_sha256": fetch_kennington.lf_sha256(lf)}
        check(kennington.matches_manifest(path, entry),
              "a Windows-decoded file matches a manifest fetched on Linux")
        check(not kennington.matches_manifest(path, {"mps_lf_sha256": "0" * 64}),
              "a different file does not")


if __name__ == "__main__":
    print("fetch_kennington.py / kennington.py (#530)")
    test_summary_table()
    test_significant_digits()
    test_select()
    test_pass_rule()
    test_manifest_hash()
    print(f"{FAILURES} check(s) FAILED" if FAILURES else "all checks passed")
    sys.exit(1 if FAILURES else 0)
