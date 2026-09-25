#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nonlinear Python API (sankhya.nonlinear, NLP stage 1).

Hand-rolled like test_bindings.py, and like it every expected value is derived by hand in a
comment: the binding's failure mode is a right number from a wrong field.

    PYTHONPATH=bindings/python python bindings/python/test_nonlinear.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sankhya  # noqa: E402
from sankhya.nonlinear import cos, exp, log, quicksum, sin, sqrt  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def raises(action) -> str | None:
    try:
        action()
    except (sankhya.SankhyaError, TypeError) as error:
        return str(error)
    return None


def test_hs071_is_built_and_evaluated() -> None:
    # HS071: min x0 x3 (x0 + x1 + x2) + x2, x0 x1 x2 x3 >= 25, sum x^2 = 40, 1 <= x <= 5.
    m = sankhya.Model()
    x = [m.var(m.add_column(cost=1.0 if j == 2 else 0.0, lower=1, upper=5)) for j in range(4)]
    objective = x[0] * x[3] * quicksum(x[:3])
    m.set_nonlinear_objective(objective)
    first = m.add_nonlinear_row(x[0] * x[1] * x[2] * x[3], lower=25, name="prod")
    second = m.add_nonlinear_row(quicksum(v ** 2 for v in x), lower=40, upper=40)
    check((first, second) == (0, 1), "nonlinear rows are numbered from 0")
    check(m.num_nonlinear_rows == 2 and m.num_rows == 0, "two nonlinear rows, no linear ones")
    m.validate()
    # At (1, 5, 5, 1): 1 * 1 * (1 + 5 + 5) = 11.
    check(objective.value([1, 5, 5, 1]) == 11.0, "objective expression at the start", "11")
    m.set_start([1, 5, 5, 1])
    result = m.solve(log_to_console=False)
    check(result.status == "not_solved" and not result.claims_a_point,
          "stage 1 says it has no engine instead of solving the linear part", result.message)


def test_numbers_mix_and_functions_are_exact() -> None:
    m = sankhya.Model()
    a = m.var(m.add_column(lower=-10, upper=10))
    b = m.var(m.add_column(lower=-10, upper=10))
    # f = 2 - a/4 + 3^b + sqrt(a*a + 1) + sin(a) cos(b) + exp(log(2 + b))
    f = 2 - a / 4 + 3 ** b + sqrt(a * a + 1) + sin(a) * cos(b) + exp(log(2 + b))
    at = [0.5, 1.5]
    want = (2 - 0.5 / 4 + 3 ** 1.5 + math.sqrt(1.25) + math.sin(0.5) * math.cos(1.5)
            + math.exp(math.log(3.5)))
    got = f.value(at)
    check(abs(got - want) <= 1e-14 * abs(want), "a mixed expression by hand", f"{got} vs {want}")
    check(exp(0.0) == 1.0 and log(1.0) == 0.0, "the functions accept plain numbers too")


def test_misuse_is_refused_with_the_reason() -> None:
    m = sankhya.Model()
    x = m.var(m.add_column(lower=-1, upper=1))
    message = raises(lambda: m.var(5))
    check(message is not None and "column 5" in message, "a missing column is named", str(message))
    message = raises(lambda: x ** x)
    check(message is not None and "exponent" in message, "a variable exponent is refused",
          str(message))
    other = sankhya.Model()
    y = other.var(other.add_column())
    message = raises(lambda: x + y)
    check(message is not None and "same model" in message, "models do not mix", str(message))
    message = raises(lambda: log(x).value([-0.5]))
    check(message is not None and "log" in message, "a domain error is an error, not NaN",
          str(message))


def test_a_nl_file_is_read() -> None:
    # The unconstrained .nl model f = (x0 - 1)^2 read back: two nonlinear objective terms.
    import tempfile
    text = ("g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0 0 0 0\n 0 1\n 0 0\n"
            " 0 0 0 0 0\nO0 0\no5\no0\nv0\nn-1\nn2\nb\n3\nG0 1\n0 0\n")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "square.nl"
        path.write_text(text)
        m = sankhya.Model.read(str(path))
        check(m.num_cols == 1 and m.num_nonlinear_rows == 0, "a .nl file reads through the C API")


def main() -> int:
    for test in (test_hs071_is_built_and_evaluated, test_numbers_mix_and_functions_are_exact,
                 test_misuse_is_refused_with_the_reason, test_a_nl_file_is_read):
        print(test.__name__)
        test()
    print(f"\n{'FAILED' if FAILURES else 'PASSED'}: {FAILURES} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
