#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Semi-continuous columns and special ordered sets (#754) for the independent verifier.

Two halves, both written from the format documentation and Beale and Tomlin (1970), not from
src/io/mps_sos.cpp or src/core/sc_sos.cpp - the verifier shares no code with the solver:

  SosReader   the MPS spellings: the SC bound (handled by the caller), the SOS / SETS
              section ("S1 SOS name [priority]" headers, members "[set] column weight" or
              "[set] column:weight"), and the S1/S2 'MARKER' 'SOSORG' ... 'SOSEND' form in
              COLUMNS, whose members are weighted 1, 2, 3, ... in order of appearance
  check()     a point against the model's conditions, on the ORIGINAL model:
                semi-continuous   x_j = 0 or col_lower[j] <= x_j (the upper bound and the
                                  rest of [l, u] are the column-bounds check's)
                SOS1              at most one member nonzero
                SOS2              at most two members nonzero, and those adjacent in weight
                                  order
              Each is measured as the smallest change that repairs it, against the integrality
              tolerance: a fractional integer and a value inside a semi-continuous gap are the
              same kind of defect.
"""
from __future__ import annotations


class SosReader:
    """Collects the sets of one MPS file while _parse_mps reads it."""

    def __init__(self) -> None:
        # [type, name, priority, columns, weights], in the order the file declares them.
        self.sets: list[list] = []
        self.current: int | None = None
        self.marker_open = False

    def _find(self, name: str) -> int | None:
        for k, entry in enumerate(self.sets):
            if entry[1] == name:
                return k
        return None

    def section_line(self, fields: list[str], col_index: dict[str, int], where: str) -> None:
        kind = fields[0].upper()
        is_type = kind in ("S1", "S2")
        names_a_set = self._find(fields[0]) is not None
        if is_type and ((len(fields) >= 2 and fields[1].upper() == "SOS") or not names_a_set):
            rest = fields[1:]
            if rest and rest[0].upper() == "SOS":
                rest = rest[1:]
            if len(rest) > 2:
                raise ValueError(f"{where}: SOS header has too many fields")
            name = rest[0] if rest else f"SOS{len(self.sets) + 1}"
            priority = float(rest[1]) if len(rest) == 2 else 0.0
            if self._find(name) is not None:
                raise ValueError(f"{where}: duplicate SOS name {name}")
            self.sets.append([1 if kind == "S1" else 2, name, priority, [], []])
            self.current = len(self.sets) - 1
            return
        if self.current is None:
            raise ValueError(f"{where}: SOS member before any S1 or S2 header")
        body = fields
        if len(fields) == 3 or (len(fields) == 2 and ":" in fields[1]):
            named = self._find(fields[0])
            if named is None:
                raise ValueError(f"{where}: SOS member names unknown set {fields[0]}")
            self.current = named
            body = fields[1:]
        if len(body) == 1 and ":" in body[0]:
            column, weight = body[0].rsplit(":", 1)
        elif len(body) == 2:
            column, weight = body
        else:
            raise ValueError(f"{where}: malformed SOS member line")
        if column not in col_index:
            raise ValueError(f"{where}: SOS member {column} is not a column")
        entry = self.sets[self.current]
        entry[3].append(col_index[column])
        entry[4].append(float(weight))

    def marker_line(self, fields: list[str], where: str) -> bool:
        """A COLUMNS MARKER record: True when it opened or closed a set."""
        bare = [f.replace("'", "").replace('"', "") for f in fields]
        upper = [f.upper() for f in bare]
        if "SOSORG" in upper:
            if self.marker_open:
                raise ValueError(f"{where}: SOSORG inside an open set")
            kinds = [u for u in upper if u in ("S1", "S2")]
            if not kinds:
                raise ValueError(f"{where}: SOSORG names no set type")
            names = [b for b, u in zip(bare, upper)
                     if u not in ("S1", "S2", "MARKER", "SOSORG")]
            name = names[0] if names else f"SOS{len(self.sets) + 1}"
            self.sets.append([1 if kinds[0] == "S1" else 2, name, 0.0, [], []])
            self.current = len(self.sets) - 1
            self.marker_open = True
            return True
        if "SOSEND" in upper:
            if not self.marker_open:
                raise ValueError(f"{where}: SOSEND with no set open")
            self.marker_open = False
            self.current = None
            return True
        return False

    def marker_member(self, column: int) -> None:
        if not self.marker_open:
            return
        entry = self.sets[self.current]
        if column not in entry[3]:
            entry[3].append(column)
            entry[4].append(float(len(entry[3])))

    def finish(self) -> list[tuple[int, str, list[int], list[float]]]:
        out = []
        for kind, name, _priority, columns, weights in self.sets:
            order = sorted(range(len(columns)), key=lambda k: weights[k])
            cols = [columns[k] for k in order]
            ws = [weights[k] for k in order]
            if any(b <= a for a, b in zip(ws, ws[1:])):
                raise ValueError(f"SOS {name}: two members share a weight, so their order is "
                                 "undefined")
            if len(set(cols)) != len(cols):
                raise ValueError(f"SOS {name}: a column is listed twice")
            out.append((kind, name, cols, ws))
        return out


def semicontinuous_gap(value: float, lower: float) -> float:
    """How far a value strictly between 0 and the run range's lower end is from either."""
    if 0.0 < value < lower:
        return min(value, lower - value)
    return 0.0


def sos_gap(kind: int, magnitudes: list[float]) -> float:
    """The largest |x| that must vanish for the set to hold, after the best choice of the
    one member (SOS1) or the two adjacent members (SOS2) allowed to stay."""
    n = len(magnitudes)
    if kind == 1:
        ordered = sorted(magnitudes, reverse=True)
        return ordered[1] if n >= 2 else 0.0
    if n <= 2:
        return 0.0
    best = float("inf")
    for k in range(n - 1):
        outside = magnitudes[:k] + magnitudes[k + 2:]
        best = min(best, max(outside, default=0.0))
    return best


def check(model, x: list[float], report, integer_tol: float) -> None:
    """Add the semi-continuous and SOS checks to `report`; nothing for a model with neither."""
    if model.semicontinuous:
        worst, where = 0.0, ""
        for j in sorted(model.semicontinuous):
            gap = semicontinuous_gap(x[j], model.col_lower[j])
            if gap > worst:
                worst, where = gap, model.col_names[j]
        report.check(worst <= integer_tol, "semi-continuous",
                     f"{len(model.semicontinuous)} columns, each 0 or at least its lower bound; "
                     f"worst {worst:.3e}" + (f" on {where}" if where else ""))
    if model.sos:
        worst, where = 0.0, ""
        for kind, name, columns, _weights in model.sos:
            gap = sos_gap(kind, [abs(x[j]) for j in columns])
            if gap > worst:
                worst, where = gap, name
        report.check(worst <= integer_tol, "special ordered sets",
                     f"{len(model.sos)} sets, at most one member (SOS1) or two adjacent "
                     f"members (SOS2) nonzero; worst {worst:.3e}" + (f" in {where}" if where else ""))
