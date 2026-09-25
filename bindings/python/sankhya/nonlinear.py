# SPDX-License-Identifier: Apache-2.0
"""Nonlinear expressions over a Model's columns (NLP stage 1).

    >>> import sankhya
    >>> from sankhya.nonlinear import exp, log
    >>> m = sankhya.Model()
    >>> x = [m.var(m.add_column(lower=1, upper=5)) for _ in range(2)]
    >>> m.set_nonlinear_objective(x[0] * x[1] + exp(x[0]) / 2)
    >>> m.add_nonlinear_row(x[0] ** 2 + x[1] ** 2, upper=10)
    0

An :class:`Expr` is a handle to a node of the model's expression graph, built through
``include/sankhya/sankhya_nonlinear.h`` - the same C boundary a third-party caller uses, so
the graph that is solved is the graph the C API built, not a Python copy of it. Numbers mix
freely with expressions (``2 * x``, ``x + 1``); an exponent must be a NUMBER, because the
solver differentiates ``x ** p`` with a constant ``p`` exactly and a variable exponent is a
different function (write ``exp(y * log(x))`` for it, whose domain is then explicit).

Failures raise :class:`SankhyaError` with the solver's own message: a column that does not
exist, a non-finite constant, an expression from another model.
"""

from __future__ import annotations

import ctypes
import math
from typing import Iterable, Sequence

from ._library import SankhyaError, load

__all__ = ["Expr", "exp", "log", "sqrt", "sin", "cos", "quicksum", "NonlinearModelMixin"]

_NEG, _EXP, _LOG, _SQRT, _SIN, _COS = range(6)
_ADD, _SUB, _MUL, _DIV = range(4)


def _check(status: int, what: str) -> None:
    if status != 0:
        detail = load().sankhya_last_error().decode()
        raise SankhyaError(f"{what}: {detail}" if detail else what)


def _built(handle: int, what: str) -> int:
    if handle < 0:
        detail = load().sankhya_last_error().decode()
        raise SankhyaError(f"{what}: {detail}" if detail else what)
    return handle


class Expr:
    """A nonlinear expression of one :class:`sankhya.Model`."""

    __slots__ = ("_model", "_id")

    def __init__(self, model, handle: int) -> None:
        self._model = model
        self._id = handle

    @property
    def handle(self) -> int:
        """The expression's handle in the C API."""
        return self._id

    # ---- Construction -----------------------------------------------------------------------

    def _lift(self, other) -> int:
        if isinstance(other, Expr):
            if other._model is not self._model:
                raise SankhyaError("an expression may only combine with one of the same model")
            return other._id
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _built(load().sankhya_expr_constant(self._model._handle, float(other)),
                          "a constant")
        return NotImplemented  # type: ignore[return-value]

    def _binary(self, op: int, left: int, right: int) -> "Expr":
        return Expr(self._model, _built(
            load().sankhya_expr_binary(self._model._handle, op, left, right), "an operation"))

    def _unary(self, op: int) -> "Expr":
        return Expr(self._model, _built(
            load().sankhya_expr_unary(self._model._handle, op, self._id), "a function"))

    def __add__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_ADD, self._id, o)

    def __radd__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_ADD, o, self._id)

    def __sub__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_SUB, self._id, o)

    def __rsub__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_SUB, o, self._id)

    def __mul__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_MUL, self._id, o)

    def __rmul__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_MUL, o, self._id)

    def __truediv__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_DIV, self._id, o)

    def __rtruediv__(self, other):
        o = self._lift(other)
        return NotImplemented if o is NotImplemented else self._binary(_DIV, o, self._id)

    def __neg__(self) -> "Expr":
        return self._unary(_NEG)

    def __pos__(self) -> "Expr":
        return self

    def __pow__(self, exponent) -> "Expr":
        if isinstance(exponent, Expr) or not isinstance(exponent, (int, float)):
            raise TypeError("the exponent must be a number; for a variable exponent write "
                            "exp(y * log(x)), which is defined where x > 0")
        return Expr(self._model, _built(
            load().sankhya_expr_power(self._model._handle, self._id, float(exponent)),
            "a power"))

    def __rpow__(self, base) -> "Expr":
        # c ** e with c > 0 is exp(e log c) everywhere; any other constant base is refused.
        if not isinstance(base, (int, float)) or base <= 0:
            raise TypeError("a number raised to an expression needs a positive base")
        return exp(self * math.log(base))

    # ---- Evaluation -------------------------------------------------------------------------

    def value(self, x: Sequence[float]) -> float:
        """The expression at ``x`` (one value per column). A point outside its domain - log
        of a non-positive number, say - raises SankhyaError with the reason."""
        n = len(x)
        array = (ctypes.c_double * n)(*[float(v) for v in x])
        out = ctypes.c_double(0.0)
        status = load().sankhya_expr_evaluate(self._model._handle, self._id, array, n,
                                              ctypes.byref(out))
        if status != 0:
            raise SankhyaError(f"evaluating: {load().sankhya_last_error().decode()}")
        return out.value

    def __repr__(self) -> str:
        return f"<sankhya.nonlinear.Expr {self._id}>"


def _apply(op: int, argument, fallback):
    if isinstance(argument, Expr):
        return argument._unary(op)
    return fallback(argument)


def exp(e):
    """exp of an expression (or of a number)."""
    return _apply(_EXP, e, math.exp)


def log(e):
    """Natural logarithm of an expression (or of a number)."""
    return _apply(_LOG, e, math.log)


def sqrt(e):
    """Square root of an expression (or of a number)."""
    return _apply(_SQRT, e, math.sqrt)


def sin(e):
    """sin of an expression, in radians (or of a number)."""
    return _apply(_SIN, e, math.sin)


def cos(e):
    """cos of an expression, in radians (or of a number)."""
    return _apply(_COS, e, math.cos)


def quicksum(terms: Iterable) -> "Expr | float":
    """The sum of many terms as ONE node, rather than a chain of pairwise additions."""
    items = list(terms)
    exprs = [t for t in items if isinstance(t, Expr)]
    if not exprs:
        return float(sum(items))
    model = exprs[0]._model
    handles = [exprs[0]._lift(t) for t in items]
    array = (ctypes.c_int * len(handles))(*handles)
    return Expr(model, _built(load().sankhya_expr_sum(model._handle, len(handles), array),
                              "a sum"))


class NonlinearModelMixin:
    """The nonlinear methods of :class:`sankhya.Model` (kept here so the model class's own
    file stays the linear API)."""

    _handle: int

    def var(self, column: int) -> Expr:
        """Column ``column`` as an :class:`Expr`."""
        return Expr(self, _built(load().sankhya_expr_variable(self._handle, int(column)),
                                 "a column expression"))

    def set_nonlinear_objective(self, expression: "Expr | None") -> None:
        """Add ``expression`` to the objective, in the model's sense (None removes it)."""
        handle = -1 if expression is None else expression.handle
        _check(load().sankhya_model_set_nonlinear_objective(self._handle, handle),
               "setting the nonlinear objective")

    def add_nonlinear_row(self, expression: Expr, lower: float | None = None,
                          upper: float | None = None, name: str | None = None) -> int:
        """Add ``lower <= expression <= upper`` (None: no bound on that side) and return its
        index among the nonlinear rows, which follow the linear rows in every per-row
        solution vector."""
        infinity = load().sankhya_infinity()
        index = ctypes.c_int(-1)
        _check(load().sankhya_model_add_nonlinear_row(
            self._handle, expression.handle,
            -infinity if lower is None else float(lower),
            infinity if upper is None else float(upper),
            name.encode() if name else None, ctypes.byref(index)), "adding a nonlinear row")
        return index.value

    def set_start(self, x: Sequence[float]) -> None:
        """A starting point for a local nonlinear method, one value per column."""
        array = (ctypes.c_double * len(x))(*[float(v) for v in x])
        _check(load().sankhya_model_set_start(self._handle, array, len(x)),
               "setting the starting point")

    @property
    def num_nonlinear_rows(self) -> int:
        """The number of nonlinear rows."""
        return load().sankhya_model_num_nonlinear_rows(self._handle)
