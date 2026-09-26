# SPDX-License-Identifier: Apache-2.0
"""Tests for the parsing the board does on the solver's output.

The board shows what the solver wrote, so the one thing it can get wrong on its own is
reading it: the iteration table, the .sol blocks, the info table. Each expected value below
is copied from a real solver output quoted in a comment, not captured from a previous run.

    python apps/web/test_server.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402


def test_iteration_line():
    # From `sankhya solve demo/crude_blend.mps --option algorithm=ipm`:
    m = server.ITER_LINE.match("         4    2.14145147e+02     6.56e-08     1.47e-07      0.00s")
    assert m and int(m[1]) == 4 and float(m[2]) == 214.145147
    assert float(m[3]) == 6.56e-8 and float(m[4]) == 1.47e-7 and float(m[5]) == 0.0
    # The dual simplex prints "-" for a dual infeasibility it does not track:
    m = server.ITER_LINE.match("         0    1.48968789e+06     3.90e+05            -      0.00s")
    assert m and m[4] == "-"
    # Not iteration lines: the header and a text line that happens to hold numbers.
    assert server.ITER_LINE.match(" Iteration         Objective   Primal Inf     Dual Inf       Time") is None
    assert server.ITER_LINE.match("refactorized at iteration 2: smallest pivot 3.676e-01") is None


def test_parse_sol():
    # The .sol demo/crude_blend_infeasible.mps produces, abridged to the parsed blocks.
    text = """# SANKHYA solution file
model CRUDEBLEND_INFEASIBLE
status infeasible
certificate farkas
iis_irreducible yes

begin farkas 3
THRUPUT -0.57735026918962584
DIESEL 1.9245008972987536
SULPHUR 0
end farkas

begin iis 4
row THRUPUT
row DIESEL
col_hi BN
col_hi MU
end iis

begin iis_witness row THRUPUT 3
AL 1
end iis_witness

begin columns 1
AL 51.489189189189197 0 basic
end columns

begin rows 1
DIESEL 40.000000000000014 4.9729729729729746 fixed
end rows
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.sol"
        p.write_text(text)
        sol = server.parse_sol(p)
    assert sol["header"]["status"] == "infeasible"
    assert sol["header"]["iis_irreducible"] == "yes"
    assert [f["row"] for f in sol["farkas"]] == ["THRUPUT", "DIESEL", "SULPHUR"]
    assert sol["farkas"][1]["multiplier"] == 1.9245008972987536
    assert sol["iis"] == [{"kind": "row", "name": "THRUPUT"}, {"kind": "row", "name": "DIESEL"},
                          {"kind": "col_hi", "name": "BN"}, {"kind": "col_hi", "name": "MU"}]
    # The witness block is neither columns nor rows and must not leak into either.
    assert sol["columns"] == [{"name": "AL", "value": 51.489189189189197, "reduced_cost": 0.0,
                               "basis": "basic"}]
    assert sol["rows"] == [{"name": "DIESEL", "activity": 40.000000000000014,
                            "dual": 4.9729729729729746, "basis": "fixed"}]


def test_models_exist():
    # Every built-in model must be a file the function can see, or the picker lies.
    for m in server.BUILTIN:
        assert (server.DEMO / m["file"]).is_file(), m["file"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
