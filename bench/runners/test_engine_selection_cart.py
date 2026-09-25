#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine-selection CART (#477): labels charge unsolved engines the time limit,
the tree recovers a separable rule, it is deterministic, the gate refuses a small training
set, and the emitted C++ is the tree. Pure Python, no solver.

    python bench/runners/test_engine_selection_cart.py
"""
from __future__ import annotations

import csv
import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine_selection_cart as cart  # noqa: E402
import engine_selection_data as data  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


def synthetic(count: int, test_every: int = 4, per_shape: int = 1) -> list[dict]:
    """Rows whose fastest engine is decided by `rows` alone: the dual simplex below 1000,
    the interior point from there to 5000, PDHG above; the rule table always says ipm.
    Consecutive runs of `per_shape` rows share a shape, and the split is by shape."""
    out = []
    for k in range(count):
        rows = 100 + 97 * k
        shape = k // per_shape
        fastest = "dual-simplex" if rows < 1000 else ("ipm" if rows < 5000 else "pdhg")
        row = {"instance": f"i{k}", "family": f"shape{shape}",
               "split": "test" if shape % test_every == 0 else "train",
               "time_limit": "60", "rule_table": "ipm"}
        for f in cart.FEATURES:
            row[f] = "0"
        row["rows"] = str(rows)
        row["density"] = str((k * 7919) % 101 / 100.0)  # noise the tree must ignore
        for e in cart.ENGINES:
            row[f"{e}_status"] = "optimal"
            row[f"{e}_verified"] = "1"
            row[f"{e}_seconds"] = "1.0" if e == fastest else "5.0"
        out.append(row)
    return out


def write(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    return Path(handle.name)


def run(argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cart.main(argv)
    return code, out.getvalue() + err.getvalue()


def main() -> int:
    print("labels")
    row = synthetic(1)[0]
    row["dual-simplex_status"] = "time_limit"
    times = cart.engine_seconds(row, 60.0)
    check(times["dual-simplex"] == 60.0, "an engine without an optimum is charged the limit")
    row["ipm_verified"] = "0"
    check(cart.engine_seconds(row, 60.0)["ipm"] == 60.0,
          "an optimum the verifier rejected is charged the limit")

    print("tree")
    path = write(synthetic(120))
    rows = cart.load(path)
    tree = cart.grow([r for r in rows if r["split"] == "train"])
    check(cart.FEATURES[tree["feature"]] == "rows", "the root splits on the feature that decides",
          cart.FEATURES[tree["feature"]])
    held = [r for r in rows if r["split"] == "test"]
    result = cart.evaluate(held, lambda r: cart.predict(tree, r["x"]))
    check(result["hits"] >= result["rows"] - 2, "held-out accuracy on a separable rule",
          f"{result['hits']}/{result['rows']}")
    again = cart.grow([r for r in rows if r["split"] == "train"])
    check(cart.describe(tree) == cart.describe(again), "the same data gives the same tree")

    print("gate and emission")
    small = write(synthetic(30))
    code, text = run([str(small), "--emit", str(small.with_suffix(".inc"))])
    check(code == 2 and "not emitted" in text, "a small training set is refused")
    few = write(synthetic(120, per_shape=20))  # 120 rows, but 6 shapes
    code, text = run([str(few), "--emit", str(few.with_suffix(".inc"))])
    check(code == 2 and "shapes, below" in text, "enough rows from too few shapes is refused",
          text.strip().splitlines()[-1])
    leaky = synthetic(120)
    leaky[1]["family"] = leaky[0]["family"]  # a training row shares a held-out row's shape
    leak = write(leaky)
    code, text = run([str(leak), "--emit", str(leak.with_suffix(".inc"))])
    check(code == 2 and "both sides of the split" in text,
          "a shape on both sides of the split is refused", text.strip().splitlines()[-1])
    target = path.with_suffix(".inc")
    code, text = run([str(path), "--emit", str(target)])
    check(code == 0 and target.exists(), "a large enough set that beats the rule table emits",
          text.strip().splitlines()[-1])
    cpp = target.read_text() if target.exists() else ""
    check("f.rows" in cpp and "return \"pdhg\";" in cpp and "training rows" in cpp,
          "the emitted rules name the feature, the engines and each branch's rows")

    print("data grid")
    grid = list(data.instances())
    names = [name for name, _, _, _ in grid]
    check(len(names) == len(set(names)), "instance names are unique", str(len(names)))
    test = [name for name, split, _, _ in grid if split == "test"]
    check(len(test) >= cart.MIN_HELD_OUT and len(grid) - len(test) >= cart.MIN_TRAINING,
          "the committed grid clears the gate", f"{len(grid) - len(test)} / {len(test)}")
    check(all((split == "test") == (name.rpartition("-s")[0] in data.HELD_OUT_SHAPES)
              for name, split, _, _ in grid),
          "held out exactly when the shape is in HELD_OUT_SHAPES")
    shapes = {family for family, _, _ in data.SHAPES}
    check(data.HELD_OUT_SHAPES <= shapes, "every held-out shape is in the grid")
    check(len(shapes - data.HELD_OUT_SHAPES) >= cart.MIN_TRAINING_SHAPES
          and len(data.HELD_OUT_SHAPES) >= cart.MIN_HELD_OUT_SHAPES,
          "the committed grid clears the shape gate",
          f"{len(shapes - data.HELD_OUT_SHAPES)} / {len(data.HELD_OUT_SHAPES)}")
    generators = {script for _, script, _ in data.SHAPES}
    check({script for family, script, _ in data.SHAPES if family in data.HELD_OUT_SHAPES}
          == generators, "every generator has a held-out shape")

    print("all passed" if FAILURES == 0 else f"{FAILURES} failure(s)")
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
