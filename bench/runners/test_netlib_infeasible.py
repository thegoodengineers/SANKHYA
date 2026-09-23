#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Netlib infeasible set's parsing (#529): the readme's summary table, the
certificate read out of a .sol file, and the pass criterion. Pure Python, no network.

    python bench/runners/test_netlib_infeasible.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_netlib_infeasible  # noqa: E402
import netlib_infeasible  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


# Shaped like lp/infeas/readme: prose with a number-bearing line before the table, the
# table with flags and free-text notes, and prose after REFERENCES that must not be read.
README = """\
SUMMARY OF INFEASIBLE LPs
telephone 613 788 5733

PROBLEM SUMMARY TABLE
---------------------

Name       Rows   Cols   Nonzeros Bounds      Notes

bgdbg1      349    407     1485   B
bgprtr       21     34       90
box1        232    261      912   B            all cols are LO bounded
ceria3d    3577    824    17604   B FR         dense col (> 967)
gosh       3793  10733    97257   B FR         242 free cols
greenbea   2505   5405    35159   B FR FX


REFERENCES
----------
chinneck    1991   3    2
"""


def test_summary_table() -> None:
    table = fetch_netlib_infeasible.parse_summary_table(README)
    check(sorted(table) == ["bgdbg1", "bgprtr", "box1", "ceria3d", "gosh", "greenbea"],
          "exactly the table's rows are read", str(sorted(table)))
    check("telephone" not in table and "chinneck" not in table,
          "number-shaped prose before and after the table is ignored")
    check(table["bgprtr"]["published_rows"] == 21 and table["bgprtr"]["published_cols"] == 34
          and table["bgprtr"]["published_nonzeros"] == 90, "counts are parsed")
    check(table["bgprtr"]["published_flags"] == "", "a row with no flags has none")
    check(table["greenbea"]["published_flags"] == "B FR FX", "every flag is kept",
          table["greenbea"]["published_flags"])
    check(table["box1"]["published_notes"] == "all cols are LO bounded",
          "a note is not mistaken for flags", table["box1"]["published_notes"])
    check(table["gosh"]["published_flags"] == "B FR"
          and table["gosh"]["published_notes"] == "242 free cols",
          "a note starting with a number stays a note")
    check(all(e["expected_status"] == "infeasible" for e in table.values()),
          "every entry is expected infeasible")
    check(fetch_netlib_infeasible.parse_summary_table("no table here") == {},
          "a readme without the table yields nothing")


def test_line_ending_blind_hash() -> None:
    import tempfile
    lf = b"NAME X\nROWS\n N COST\nENDATA\n"
    crlf = lf.replace(b"\n", b"\r\n")
    digest = fetch_netlib_infeasible.lf_sha256
    check(digest(lf) == digest(crlf), "CRLF and LF decode to the same manifest hash")
    check(digest(lf) != digest(lf + b"\n"), "and a real difference still shows")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.mps"
        path.write_bytes(crlf)
        entry = {"mps_sha256": "not this", "mps_lf_sha256": digest(lf)}
        check(netlib_infeasible.matches_manifest(path, entry),
              "a Windows-decoded file matches a manifest fetched on Linux")
        check(not netlib_infeasible.matches_manifest(path, {"mps_lf_sha256": digest(b"x")}),
              "a different instance does not")


SOL_FARKAS = """\
# SANKHYA solution file
status infeasible
certificate farkas
rows 3

# Farkas certificate
begin farkas 3
ROW1 0
ROW2 -1
ROW3 2.5
end farkas
"""

SOL_NONE = "status infeasible\ncertificate none\nmessage proved by presolve\n"

SOL_EMPTY_FARKAS = "status infeasible\ncertificate farkas\nbegin farkas 2\nR1 0\nR2 0\nend farkas\n"

# A farkas-looking row outside the farkas section must not be counted.
SOL_OTHER_SECTION = ("status optimal\ncertificate none\nbegin rows 1\nR1 5 1\nend rows\n")


def test_read_certificate() -> None:
    read = netlib_infeasible.read_certificate
    check(read(SOL_FARKAS) == ("farkas", 2), "farkas with two nonzero multipliers",
          str(read(SOL_FARKAS)))
    check(read(SOL_NONE) == ("none", 0), "presolve's certificate none", str(read(SOL_NONE)))
    check(read(SOL_EMPTY_FARKAS) == ("farkas", 0), "an all-zero farkas section counts zero")
    check(read(SOL_OTHER_SECTION) == ("none", 0), "other sections are not multipliers")
    check(read("") == ("", 0), "an empty file has no certificate")


def test_classify() -> None:
    ok = ("  [PASS] infeasibility proof                the rows aggregate to at least "
          "1.1e+01, the column bounds allow at most 0\nVERIFIED: 4 checks passed")
    # What the verifier prints for `certificate none`: a NOTE, rendered with the same prefix.
    note = ("  [PASS] infeasibility proof  no certificate offered, so nothing is claimed and "
            "nothing is checked\nVERIFIED: 1 checks passed")
    classify = netlib_infeasible.classify
    check(classify("infeasible", "farkas", 3, 0, ok) == "", "a verified farkas passes")
    check(classify("infeasible", "none", 0, 0, "VERIFIED: 0 checks passed")
          == "infeasible without a certificate",
          "the verifier's exit 0 on certificate none is not a pass")
    check(classify("infeasible", "farkas", 0, 0, ok) == "infeasible without a certificate",
          "an all-zero farkas section is not a certificate")
    check(classify("infeasible", "farkas", 3, 0, "VERIFIED: 1 checks passed")
          == "certificate rejected by the verifier",
          "exit 0 without a passed infeasibility proof is not a pass")
    check(classify("infeasible", "farkas", 3, 0, note)
          == "certificate rejected by the verifier",
          "the verifier's no-certificate note is not a passed proof")
    check(classify("infeasible", "farkas", 3, 1, "[FAIL] infeasibility proof")
          == "certificate rejected by the verifier", "a rejected certificate fails")
    check(classify("infeasible", "farkas", 3, 2, "cannot read the model: bad")
          == "verifier cannot parse the model", "a verifier parse error is named as such")
    check(classify("infeasible", "farkas", 3, None, "") == "not verified",
          "no verifier run is not a pass")
    check(classify("optimal", "none", 0, 1, "") == "wrong verdict: optimal",
          "optimal on an infeasible model is a wrong verdict")
    check(classify("infeasible", "none", 0, 0, note, "presolve")
          == "infeasible without a certificate (presolve)",
          "a missing certificate names the engine that reached the verdict")
    check(classify("time_limit", "none", 0, 0, "") == "no verdict: time_limit",
          "a limit is no verdict")
    check(classify("crashed", "", 0, None, "") == "solver crashed", "a crash is named")

    verified = netlib_infeasible.independently_verified
    check(verified("infeasible", "farkas", 3, 0, ok) == 1, "a checked proof is verified")
    check(verified("infeasible", "none", 0, 0, note) == "",
          "no certificate is not 'verified', whatever the exit code")
    check(verified("infeasible", "farkas", 3, 1, "[FAIL] x") == 0, "a rejection is 0")
    check(verified("optimal", "none", 0, 1, "") == 0, "a rejected wrong point is 0")
    check(verified("numerical_error", "none", 0, 0, "") == "", "no claim, no verdict")
    check(verified("infeasible", "farkas", 3, None, "") == "", "not run is empty")


if __name__ == "__main__":
    print("fetch_netlib_infeasible.py / netlib_infeasible.py (#529)")
    test_summary_table()
    test_line_ending_blind_hash()
    test_read_certificate()
    test_classify()
    print(f"{FAILURES} check(s) FAILED" if FAILURES else "all checks passed")
    sys.exit(1 if FAILURES else 0)
