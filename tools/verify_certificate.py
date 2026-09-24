#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a MILP optimality certificate in exact rational arithmetic (#518).

Usage:
    python tools/verify_certificate.py proof.vipr [--mps model.mps] [--abs-gap A] [--rel-gap R]

The certificate format is VIPR, from Cheung, Gleixner and Steffy, "Verifying integer
programming results", IPCO 2017 (LNCS 10328): the model (VAR, INT, OBJ, CON), the claim
(RTP), the solutions that establish the primal side (SOL) and a list of derived
constraints (DER), each with the reason it holds. This checker was written from that paper
alone. It shares no code with the solver and never links it: every number is read as an
exact rational (fractions.Fraction) and every step is re-derived here.

Reasons understood (all indices are 0-based positions in the combined CON + DER list):

  { asm }                  an assumption: holds under itself only
  { sol }                  objective <= the best SOL value (minimise; >= for maximise)
  { lin k i1 m1 ... }      the combination sum m_t * C_it, signs as the derived sense needs
  { rnd k i1 m1 ... }      the same, then Chvatal-Gomory rounding of the right-hand side
  { uns i1 l1 i2 l2 }      C_i1 holds under assumption l1, C_i2 under l2, and l1, l2 are the
                           two sides of an integer disjunction a.x <= d | a.x >= d + 1

ONE EXTENSION, stated rather than hidden: a lin or rnd combination need not reproduce the
derived coefficients exactly. The difference e = d - g between the derived coefficients and
the combination's is priced over the tightest column bounds available - every
single-variable constraint of CON, and every single-variable constraint the step references
(an assumption on the path, listed with multiplier 0) - exactly as the Neumaier-Shcherbina
safe bound prices reduced costs. A strict VIPR 1.0 checker would reject such a step; the
completion is what lets the solver write only the row multipliers of each leaf LP.

A derived constraint may give OBJ in place of its coefficient list, meaning the objective's.

Exit status: 0 the claim is proved and the solution is exactly feasible (or within
--feas-tol, when given) with the gap within tolerance; 1 the certificate is rejected; 2 every
step checks but optimality is not shown (the gap exceeds the tolerance, or the solution is
feasible only up to rounding). The bound side is always checked exactly: --feas-tol only
concerns the SOL point, whose continuous values are floating-point numbers.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_solution_io import open_text  # noqa: E402

ZERO = Fraction(0)


class Rejected(Exception):
    """The certificate does not prove what it claims; the message says where."""


class Constraint:
    __slots__ = ("name", "sense", "rhs", "coefs", "assumptions", "is_asm")

    def __init__(self, name: str, sense: str, rhs: Fraction, coefs: dict[int, Fraction]):
        self.name = name
        self.sense = sense
        self.rhs = rhs
        self.coefs = {j: v for j, v in coefs.items() if v != 0}
        self.assumptions: frozenset[int] = frozenset()
        self.is_asm = False

    def absurd(self) -> bool:
        """0 >= positive, 0 <= negative, or 0 = nonzero: no point satisfies it."""
        if self.coefs:
            return False
        return ((self.sense == "G" and self.rhs > 0) or (self.sense == "L" and self.rhs < 0)
                or (self.sense == "E" and self.rhs != 0))

    def implies(self, other: "Constraint") -> bool:
        """Does this constraint, on its own, imply `other`?"""
        if self.absurd():
            return True
        if self.coefs != other.coefs:
            return not other.coefs and not other.absurd() and _trivially_true(other)
        if other.sense == "G":
            return self.sense in ("G", "E") and self.rhs >= other.rhs
        if other.sense == "L":
            return self.sense in ("L", "E") and self.rhs <= other.rhs
        return self.sense == "E" and self.rhs == other.rhs


def _trivially_true(c: Constraint) -> bool:
    return ((c.sense == "G" and c.rhs <= 0) or (c.sense == "L" and c.rhs >= 0)
            or (c.sense == "E" and c.rhs == 0))


class Certificate:
    def __init__(self) -> None:
        self.var_names: list[str] = []
        self.integer: list[bool] = []
        self.maximize = False
        self.objective: dict[int, Fraction] = {}
        self.constraints: list[Constraint] = []
        self.num_con = 0
        self.rtp = "range"
        self.rtp_lower: Fraction | None = None  # None means -infinity
        self.rtp_upper: Fraction | None = None  # None means +infinity
        self.solutions: list[tuple[str, dict[int, Fraction]]] = []
        self.num_der = 0


# =========================================================================================
# Reading
# =========================================================================================


def _tokens(path: Path):
    with open_text(path) as handle:
        for line in handle:
            line = line.split("%", 1)[0]
            for token in line.replace("{", " { ").replace("}", " } ").split():
                yield token


class _Stream:
    def __init__(self, path: Path) -> None:
        self._it = _tokens(path)

    def next(self) -> str:
        try:
            return next(self._it)
        except StopIteration:
            raise Rejected("the file ends early") from None

    def expect(self, keyword: str) -> None:
        token = self.next()
        if token != keyword:
            raise Rejected(f"expected {keyword}, found {token!r}")

    def integer(self) -> int:
        token = self.next()
        try:
            return int(token)
        except ValueError:
            raise Rejected(f"expected an integer, found {token!r}") from None

    def number(self) -> Fraction:
        token = self.next()
        try:
            return Fraction(token)
        except (ValueError, ZeroDivisionError):
            raise Rejected(f"expected a number, found {token!r}") from None

    def bound(self) -> Fraction | None:
        token = self.next()
        if token.lower() in ("inf", "-inf", "+inf"):
            return None
        try:
            return Fraction(token)
        except (ValueError, ZeroDivisionError):
            raise Rejected(f"expected a bound, found {token!r}") from None


def _read_coefficients(s: _Stream, cert: Certificate, allow_obj: bool) -> dict[int, Fraction]:
    first = s.next()
    if first == "OBJ":
        if not allow_obj:
            raise Rejected("OBJ is not allowed here")
        return dict(cert.objective)
    try:
        k = int(first)
    except ValueError:
        raise Rejected(f"expected a coefficient count, found {first!r}") from None
    coefs: dict[int, Fraction] = {}
    for _ in range(k):
        j = s.integer()
        if not 0 <= j < len(cert.var_names):
            raise Rejected(f"variable index {j} out of range")
        coefs[j] = coefs.get(j, ZERO) + s.number()
    return coefs


def read_certificate(path: Path) -> tuple[Certificate, list[tuple[Constraint, list[str]]]]:
    """Parse everything; return the certificate and the DER entries with their reasons."""
    s = _Stream(path)
    cert = Certificate()
    s.expect("VER")
    version = s.next()
    if not version.startswith("1."):
        raise Rejected(f"unsupported VIPR version {version}")
    s.expect("VAR")
    n = s.integer()
    cert.var_names = [s.next() for _ in range(n)]
    cert.integer = [False] * n
    s.expect("INT")
    for _ in range(s.integer()):
        j = s.integer()
        if not 0 <= j < n:
            raise Rejected(f"integer index {j} out of range")
        cert.integer[j] = True
    s.expect("OBJ")
    sense = s.next()
    if sense not in ("min", "max"):
        raise Rejected(f"objective sense must be min or max, found {sense!r}")
    cert.maximize = sense == "max"
    cert.objective = {j: v for j, v in _read_coefficients(s, cert, False).items() if v != 0}
    s.expect("CON")
    m = s.integer()
    s.integer()  # the number of bound constraints: informational
    for _ in range(m):
        name = s.next()
        csense = s.next()
        if csense not in ("E", "L", "G"):
            raise Rejected(f"constraint {name}: sense {csense!r}")
        rhs = s.number()
        cert.constraints.append(Constraint(name, csense, rhs, _read_coefficients(s, cert, True)))
    cert.num_con = m
    s.expect("RTP")
    cert.rtp = s.next()
    if cert.rtp == "range":
        cert.rtp_lower = s.bound()
        cert.rtp_upper = s.bound()
    elif cert.rtp != "infeas":
        raise Rejected(f"RTP must be range or infeas, found {cert.rtp!r}")
    s.expect("SOL")
    for _ in range(s.integer()):
        name = s.next()
        cert.solutions.append((name, _read_coefficients(s, cert, False)))
    s.expect("DER")
    cert.num_der = s.integer()
    derived: list[tuple[Constraint, list[str]]] = []
    for _ in range(cert.num_der):
        name = s.next()
        dsense = s.next()
        if dsense not in ("E", "L", "G"):
            raise Rejected(f"derivation {name}: sense {dsense!r}")
        rhs = s.number()
        constraint = Constraint(name, dsense, rhs, _read_coefficients(s, cert, True))
        s.expect("{")
        reason: list[str] = []
        while True:
            token = s.next()
            if token == "}":
                break
            reason.append(token)
        s.next()  # the index of the last constraint that uses this one: a hint only
        derived.append((constraint, reason))
    return cert, derived


# =========================================================================================
# Checking
# =========================================================================================


def _single_variable_bound(c: Constraint):
    """(j, lower, upper) when c bounds one variable, else None. Bounds may be None."""
    if len(c.coefs) != 1:
        return None
    (j, a), = c.coefs.items()
    value = c.rhs / a
    lower = upper = None
    if c.sense in ("G", "E"):
        if a > 0:
            lower = value
        else:
            upper = value
    if c.sense in ("L", "E"):
        if a > 0:
            upper = value
        else:
            lower = value
    return j, lower, upper


def _tighten(box: dict[int, list], j: int, lower, upper) -> None:
    entry = box.setdefault(j, [None, None])
    if lower is not None and (entry[0] is None or lower > entry[0]):
        entry[0] = lower
    if upper is not None and (entry[1] is None or upper < entry[1]):
        entry[1] = upper


class Checker:
    def __init__(self, cert: Certificate) -> None:
        self.cert = cert
        self.all: list[Constraint] = list(cert.constraints)
        # The global box: every single-variable constraint of CON.
        self.global_box: dict[int, list] = {}
        for c in cert.constraints:
            bound = _single_variable_bound(c)
            if bound is not None:
                _tighten(self.global_box, *bound)

    def _combine(self, target: Constraint, pairs: list[tuple[int, Fraction]], where: str):
        """The combination's coefficients and right-hand side, completed over the bounds.

        Returns the rhs the combination proves for target's coefficients (before any
        rounding) and the union of the referenced constraints' assumptions."""
        if target.sense == "E":
            raise Rejected(f"{where}: an equality cannot be derived by a combination")
        g: dict[int, Fraction] = {}
        rhs = ZERO
        assumptions: set[int] = set()
        box = None
        for index, mult in pairs:
            if not 0 <= index < len(self.all):
                raise Rejected(f"{where}: reference {index} is not an earlier constraint")
            c = self.all[index]
            assumptions |= c.assumptions
            bound = _single_variable_bound(c)
            if bound is not None:
                if box is None:
                    box = {j: list(v) for j, v in self.global_box.items()}
                _tighten(box, *bound)
            if mult == 0:
                continue
            # For a >= target every term must be a >= after scaling, for a <= target a <=.
            if c.sense != "E":
                keeps = (mult > 0) == (c.sense == target.sense)
                if not keeps:
                    raise Rejected(f"{where}: multiplier {mult} on {c.name} ({c.sense}) has "
                                   f"the wrong sign for a {target.sense} derivation")
            rhs += mult * c.rhs
            for j, a in c.coefs.items():
                g[j] = g.get(j, ZERO) + mult * a
        if box is None:
            box = self.global_box
        # Completion: target.coefs . x = g . x + e . x, e priced over the box.
        for j in set(target.coefs) | set(g):
            e = target.coefs.get(j, ZERO) - g.get(j, ZERO)
            if e == 0:
                continue
            lower, upper = box.get(j, (None, None))
            # For >=, the smallest e_j x_j; for <=, the largest.
            use_lower = (e > 0) == (target.sense == "G")
            bound = lower if use_lower else upper
            if bound is None:
                name = self.cert.var_names[j]
                raise Rejected(f"{where}: coefficient of {name} differs by {float(e):.3e} and "
                               f"no bound on it is available to absorb the difference")
            rhs += e * bound
        return rhs, assumptions

    def check_step(self, position: int, c: Constraint, reason: list[str]) -> None:
        where = f"derivation {c.name} (index {position})"
        if not reason:
            raise Rejected(f"{where}: empty reason")
        kind = reason[0]
        if kind == "asm":
            c.assumptions = frozenset({position})
            c.is_asm = True
        elif kind == "sol":
            self._check_sol(c, where)
        elif kind in ("lin", "rnd"):
            k = int(reason[1])
            if len(reason) != 2 + 2 * k:
                raise Rejected(f"{where}: {kind} lists {k} pairs but has "
                               f"{len(reason) - 2} numbers")
            pairs = [(int(reason[2 + 2 * t]), Fraction(reason[3 + 2 * t])) for t in range(k)]
            if any(index >= position for index, _ in pairs):
                raise Rejected(f"{where}: refers forward")
            proved, assumptions = self._combine(c, pairs, where)
            if kind == "rnd":
                for j, a in c.coefs.items():
                    if not self.cert.integer[j] or a.denominator != 1:
                        raise Rejected(f"{where}: rounding needs integer coefficients on "
                                       f"integer variables ({self.cert.var_names[j]})")
                proved = Fraction(math.ceil(proved) if c.sense == "G" else math.floor(proved))
            ok = proved >= c.rhs if c.sense == "G" else proved <= c.rhs
            if not ok:
                raise Rejected(f"{where}: the combination proves {float(proved):.17g}, not "
                               f"{float(c.rhs):.17g}")
            c.assumptions = frozenset(assumptions)
        elif kind == "uns":
            if len(reason) != 5:
                raise Rejected(f"{where}: uns takes four indices")
            i1, l1, i2, l2 = (int(t) for t in reason[1:])
            if max(i1, l1, i2, l2) >= position or min(i1, l1, i2, l2) < 0:
                raise Rejected(f"{where}: uns refers forward")
            self._check_disjunction(self.all[l1], self.all[l2], where)
            for i in (i1, i2):
                if not self.all[i].implies(c):
                    raise Rejected(f"{where}: {self.all[i].name} does not imply it")
            c.assumptions = ((self.all[i1].assumptions - {l1}) |
                             (self.all[i2].assumptions - {l2}))
        else:
            raise Rejected(f"{where}: unknown reason {kind!r}")
        self.all.append(c)

    def _check_disjunction(self, a: Constraint, b: Constraint, where: str) -> None:
        if not (a.is_asm and b.is_asm):
            raise Rejected(f"{where}: uns must discharge two assumptions")
        if a.coefs != b.coefs or not a.coefs:
            raise Rejected(f"{where}: the two assumptions are not on the same expression")
        for j, v in a.coefs.items():
            if not self.cert.integer[j] or v.denominator != 1:
                raise Rejected(f"{where}: the disjunction is not on an integer expression")
        low, high = (a, b) if a.sense == "L" else (b, a)
        if not (low.sense == "L" and high.sense == "G" and low.rhs.denominator == 1
                and high.rhs == low.rhs + 1):
            raise Rejected(f"{where}: {a.name} and {b.name} are not a <= d | >= d + 1 split")

    def _check_sol(self, c: Constraint, where: str) -> None:
        # Only an EXACTLY feasible point bounds the optimum. An infeasible one would let a sol
        # step contradict a true bound, and from a contradiction anything follows, including
        # a claimed bound the model does not have (review of #639).
        feasible = [self.objective_of(x) for _, x in self.cert.solutions
                    if self.solution_violation(x) == 0]
        best = (max(feasible) if self.cert.maximize else min(feasible)) if feasible else None
        if best is None or c.coefs != self.cert.objective:
            raise Rejected(f"{where}: sol needs an exactly feasible solution and the "
                           "objective's coefficients")
        ok = (c.sense == "G" and c.rhs <= best) if self.cert.maximize else \
            (c.sense == "L" and c.rhs >= best)
        if not ok:
            raise Rejected(f"{where}: no solution attains {float(c.rhs):.17g}")

    # ---- Solutions ---------------------------------------------------------------------

    def solution_violation(self, x: dict[int, Fraction]) -> Fraction:
        """The largest violation of any CON constraint or integrality; 0 when exact."""
        worst = ZERO
        for j, value in x.items():
            if self.cert.integer[j] and value.denominator != 1:
                worst = max(worst, abs(value - round(value)))
        for c in self.cert.constraints:
            activity = sum((a * x.get(j, ZERO) for j, a in c.coefs.items()), ZERO)
            if c.sense in ("G", "E"):
                worst = max(worst, c.rhs - activity)
            if c.sense in ("L", "E"):
                worst = max(worst, activity - c.rhs)
        return worst

    def objective_of(self, x: dict[int, Fraction]) -> Fraction:
        return sum((a * x.get(j, ZERO) for j, a in self.cert.objective.items()), ZERO)

    def best_solution_value(self) -> Fraction | None:
        values = [self.objective_of(x) for _, x in self.cert.solutions]
        if not values:
            return None
        return max(values) if self.cert.maximize else min(values)


# =========================================================================================
# The claim
# =========================================================================================


class Verdict:
    def __init__(self) -> None:
        self.proved_bound: Fraction | None = None  # on the objective, in the model's sense
        self.solution_value: Fraction | None = None
        self.solution_violation = ZERO
        self.infeasible = False
        self.derivations = 0


def verify(cert: Certificate, derived: list[tuple[Constraint, list[str]]]) -> Verdict:
    """Check every step, then the claim. Raises Rejected on the first failure."""
    checker = Checker(cert)
    verdict = Verdict()
    for offset, (constraint, reason) in enumerate(derived):
        checker.check_step(cert.num_con + offset, constraint, reason)
    verdict.derivations = len(derived)
    # Solutions: each must be feasible; the best one's value is the primal side.
    worst = ZERO
    for name, x in cert.solutions:
        worst = max(worst, checker.solution_violation(x))
    verdict.solution_violation = worst
    verdict.solution_value = checker.best_solution_value()
    # The dual side: an assumption-free derived constraint on the objective (or an absurd one).
    objective = cert.objective
    best: Fraction | None = None
    for c in checker.all[cert.num_con:]:
        if c.assumptions:
            continue
        if c.absurd():
            verdict.infeasible = True
            continue
        if c.coefs != objective:
            continue
        if not cert.maximize and c.sense in ("G", "E"):
            best = c.rhs if best is None else max(best, c.rhs)
        if cert.maximize and c.sense in ("L", "E"):
            best = c.rhs if best is None else min(best, c.rhs)
    verdict.proved_bound = best
    if cert.rtp == "infeas":
        if not verdict.infeasible:
            raise Rejected("RTP infeas, but no assumption-free contradiction is derived")
        if cert.solutions:
            raise Rejected("RTP infeas, but the certificate lists a solution")
        return verdict
    # RTP range lower upper: the proof side and the solution side, by sense.
    proof_side, primal_side = ((cert.rtp_upper, cert.rtp_lower) if cert.maximize
                               else (cert.rtp_lower, cert.rtp_upper))
    if proof_side is not None and not verdict.infeasible:
        if best is None:
            raise Rejected("no assumption-free derived constraint bounds the objective")
        ok = best <= proof_side if cert.maximize else best >= proof_side
        if not ok:
            raise Rejected(f"the derivation proves {float(best):.17g}, the claim is "
                           f"{float(proof_side):.17g}")
    if primal_side is not None:
        value = verdict.solution_value
        if value is None:
            raise Rejected("the claim needs a solution and SOL is empty")
        ok = value >= primal_side if cert.maximize else value <= primal_side
        if not ok:
            raise Rejected(f"the best solution is worth {float(value):.17g}, the claim is "
                           f"{float(primal_side):.17g}")
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--mps", type=Path, help="check the certificate's model is this file")
    parser.add_argument("--abs-gap", type=float, default=1e-6,
                        help="absolute gap accepted as optimal (default: mip_absolute_gap)")
    parser.add_argument("--rel-gap", type=float, default=1e-4,
                        help="relative gap accepted as optimal (default: mip_relative_gap)")
    parser.add_argument("--model-as-written", action="store_true",
                        help="accept the model the certificate states without a source file "
                             "to compare it with; without this or --mps nothing exits 0")
    parser.add_argument("--feas-tol", type=float, default=0.0,
                        help="accept a solution violating the model by at most this much "
                             "(default 0: exact); the bound is always checked exactly")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        cert, derived = read_certificate(args.certificate)
        if args.mps is not None:
            from verify_certificate_model import compare_with_mps
            compare_with_mps(cert, args.mps)
        verdict = verify(cert, derived)
    except Rejected as why:
        print(f"REJECTED: {why}")
        print(f"checker seconds {time.perf_counter() - started:.3f}")
        return 1
    seconds = time.perf_counter() - started
    print(f"derivations checked {verdict.derivations}")
    if verdict.infeasible and cert.rtp == "infeas":
        print("the model is proved infeasible")
        if args.mps is None:
            print("model NOT checked against a source file (no --mps)")
        print(f"checker seconds {seconds:.3f}")
        return 0 if (args.mps is not None or args.model_as_written) else 2
    bound = verdict.proved_bound
    value = verdict.solution_value
    print(f"proved bound {'none' if bound is None else f'{float(bound):.17g}'}")
    print(f"solution objective {'none' if value is None else f'{float(value):.17g}'}")
    exact = verdict.solution_violation == 0
    print(f"solution {'exactly feasible' if exact else 'violates the model by'}"
          f"{'' if exact else f' {float(verdict.solution_violation):.3e}'}")
    status = 0
    if bound is None or value is None:
        status = 2
    else:
        gap = abs(value - bound)
        relative = gap / max(Fraction(1), abs(value))
        within = gap <= Fraction(args.abs_gap) or relative <= Fraction(args.rel_gap)
        print(f"gap {float(gap):.6e} absolute, {float(relative):.6e} relative: "
              f"{'within' if within else 'OUTSIDE'} the tolerance")
        if not within:
            status = 2
    tolerated = not exact and verdict.solution_violation <= Fraction(args.feas_tol)
    if not exact and not tolerated:
        status = 2
    # Without --mps the proof is checked against whatever VAR/OBJ/CON the file states, which
    # proves nothing about the model the user meant. Say so, and never exit 0 on it unless
    # the caller asked for exactly that (review of #639).
    unchecked = args.mps is None and not args.model_as_written
    if args.mps is None:
        print("model NOT checked against a source file (no --mps)")
    if unchecked and status == 0:
        status = 2
    if status != 0:
        print("BOUND VERIFIED, OPTIMALITY NOT SHOWN")
    elif tolerated:
        print(f"VERIFIED (the solution only within --feas-tol {args.feas_tol:g})")
    else:
        print("VERIFIED")
    print(f"checker seconds {seconds:.3f}")
    return status


if __name__ == "__main__":
    sys.exit(main())
