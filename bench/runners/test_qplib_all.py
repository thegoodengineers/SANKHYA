#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the whole-QPLIB runner's verdicts and its doc section (#835).

Hermetic: nothing here downloads a file or runs the solver.

    python bench/runners/test_qplib_all.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qplib_all as runner  # noqa: E402
import qplib_all_doc as doc  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def test_classify() -> None:
    nonconvex = ("the quadratic objective is not convex: sparse LDL^T reached -17.4694 at "
                 "column 48 (x49) in minimization sense")
    check(runner.classify("model_error", nonconvex) == "nonconvex", "a convexity refusal")
    check(runner.classify("numerical_error", f"node LP returned model_error ({nonconvex})")
          == "nonconvex", "the same refusal raised from inside branch and bound")
    check(runner.classify("read_error", "error: x.qps:15: QCMATRIX section: quadratic "
                          "constraints are not supported.") == "quadratic_constraints",
          "a QCMATRIX refusal by the reader")
    check(runner.classify("read_error", "error: x.qps:3: bad bound") == "reader_error",
          "any other read failure is a reader error")
    check(runner.classify("crashed", "") == "crashed", "a crash")
    check(runner.classify("optimal", "non-convex words in a message") == "",
          "a status with a point is never a refusal")


def test_judge() -> None:
    v = runner.judge("optimal", 1.0 + 5e-7, 1.0, integer=False, maximize=False)
    check(v["matches_reference"] and not v["beats_reference"], "within 1e-6, continuous")
    v = runner.judge("optimal", 1.0 + 5e-5, 1.0, integer=False, maximize=False)
    check(not v["matches_reference"], "5e-5 off is not a match for a continuous model")
    v = runner.judge("optimal", 1.0 + 5e-5, 1.0, integer=True, maximize=False)
    check(v["matches_reference"] and v["tolerance"] == 1e-4,
          "but is within kMipRelativeGap for an integer one")
    v = runner.judge("optimal", 0.5, 1.0, integer=False, maximize=False)
    check(v["beats_reference"] and not v["matches_reference"], "below a minimum is flagged")
    v = runner.judge("optimal", 1.5, 1.0, integer=False, maximize=True)
    check(v["beats_reference"], "above a maximum is flagged")
    v = runner.judge("time_limit", 1.0, 1.0, integer=True, maximize=False)
    check(not v["solved"] and not v["matches_reference"], "an unproven point is not solved")
    v = runner.judge("model_error", None, 1.0, integer=False, maximize=False)
    check(v["matches_reference"] is False and v["rel_gap"] is None, "a refusal has no gap")
    v = runner.judge("optimal", 1.0, None, integer=False, maximize=False)
    check(v["matches_reference"] is None, "no reference, no verdict")


def _row(name: str, **fields) -> dict:
    row = {key: "" for key in runner.CSV_COLUMNS}
    row.update({"instance": name, "problem_type": "QCL", "read_ok": "1", "engine": "qp-ipm",
                "status": "optimal", "wall_seconds": "0.5", "time_limit": "60.0", "jobs": "1",
                "git_commit": "abc1234", "machine": "cloud container; test CPU"})
    row.update(fields)
    return row


def test_doc() -> None:
    check("Not yet run" in doc.section(None), "no CSV says so")
    rows = [
        _row("QPLIB_0001", solved="1", matches_reference="1", verified="1"),
        _row("QPLIB_0002", status="model_error", refusal="nonconvex", solved="0"),
        _row("QPLIB_0003", read_ok="0", engine="", status="read_error",
             refusal="quadratic_constraints", problem_type="QCQ"),
        _row("QPLIB_0004", solved="1", matches_reference="0", beats_reference="1",
             verified="0"),
        _row("QPLIB_0005", read_ok="0", engine="", status="read_error", refusal="reader_error"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qplib-all-abc1234.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        text = doc.section(path)
    check("| in the run | 5 |" in text and "| read OK | 3 |" in text, "counts from the rows")
    check("| matched QPLIB's objective | 1 |" in text, "matched count")
    check("Better than the reference: `QPLIB_0004`." in text, "a better-than-reference named")
    check("Reader error, crash or hang: `QPLIB_0005`." in text, "a reader error named")
    check("Optimal but not matching the reference: `QPLIB_0004`." in text,
          "an optimal mismatch named")
    check(text.count("| QPLIB_000") == 5, "every instance in the per-instance table")
    check("| qp-ipm | 3 | 1 | 2 | 1 | 1 |" in text, "the engine table", text[:0])


def main() -> int:
    for test in (test_classify, test_judge, test_doc):
        print(test.__name__)
        test()
    print(f"\n{'OK' if FAILURES == 0 else 'FAILED'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
