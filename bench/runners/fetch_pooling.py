#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the standard pooling instances' published data and transcribe it to JSON (#516).

WHERE THE NUMBERS COME FROM. The thirteen instances (Haverly 1-3, Ben-Tal 4-5, Foulds 2-5,
Adhya 1-4) are the "Standard Pooling Problem Instances" Alfaki and Haugland published with
*Strong formulations for the pooling problem*, J. Global Optimization 56:3 (2013) 897-916:
one GAMS DATA file per instance (sets, arc costs, adjacency, qualities, node capacities; no
model and no solver code), formerly at www.ii.uib.no/~mohammeda/spooling/ and now served by
the Internet Archive. MINLPLib's pooling_*pq instances name the same files as their source.
The original papers are cited per instance in data/pooling/reference.json.

WHAT THIS WRITES. data/pooling/instances/<name>.json: the network, transcribed by a parser
from the GAMS file, with the archived URL and the sha256 of the bytes it parsed. Nothing is
typed by hand. The models themselves (P-, Q- and PQ-formulations) are generated from these
JSON files by bench/runners/pooling_models.py.

    python bench/runners/fetch_pooling.py            # download, parse, write the JSON
    python bench/runners/fetch_pooling.py --check    # download and compare with the JSON
    python bench/runners/fetch_pooling.py --from-dir DIR   # parse files already on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "pooling" / "instances"
ORIGINAL_URL = "http://www.ii.uib.no/~mohammeda/spooling/"
ARCHIVE_URL = "http://web.archive.org/web/2020id_/" + ORIGINAL_URL

# Our instance name -> Alfaki and Haugland's file.
FILES = {
    "haverly1": "Haverly1.gms", "haverly2": "Haverly2.gms", "haverly3": "Haverly3.gms",
    "bental4": "Bental4.gms", "bental5": "Bental5.gms",
    "foulds2": "Foulds2.gms", "foulds3": "Foulds3.gms", "foulds4": "Foulds4.gms",
    "foulds5": "Foulds5.gms",
    "adhya1": "Adhya1.gms", "adhya2": "Adhya2.gms", "adhya3": "Adhya3.gms",
    "adhya4": "Adhya4.gms",
}


def _range(text: str) -> list[int]:
    """`1*6` -> [1..6]; `3` -> [3]."""
    text = text.strip()
    if "*" in text:
        first, last = (int(part) for part in text.split("*"))
        return list(range(first, last + 1))
    return [int(text)]


def _strip_comments(text: str) -> str:
    text = re.sub(r"\$ontext.*?\$offtext", "", text, flags=re.S | re.I)
    lines = []
    for line in text.splitlines():
        # `$eolcom #` makes `#` an end-of-line comment in these files.
        line = line.split("#", 1)[0]
        if line.strip().startswith("$"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _table(body: str) -> dict[tuple[int, int], float]:
    rows = [line.split() for line in body.strip().splitlines() if line.strip()]
    header = [int(label) for label in rows[0]]
    out: dict[tuple[int, int], float] = {}
    for fields in rows[1:]:
        if len(fields) != len(header) + 1:
            raise ValueError(f"table row {fields!r} does not match header {header!r}")
        row = int(fields[0])
        for column, value in zip(header, fields[1:]):
            out[(row, column)] = float(value)
    return out


def _parameter(body: str) -> dict[int, float]:
    fields = body.split()
    if len(fields) % 2:
        raise ValueError(f"parameter body has an odd number of fields: {body!r}")
    return {int(fields[i]): float(fields[i + 1]) for i in range(0, len(fields), 2)}


def parse_gams_data(text: str) -> dict:
    """The network of one Alfaki-Haugland data file.

    Nodes are the integers of set i; s(i) are the sources, t(i) the terminals, the rest the
    pools. An arc (u, v) exists where a(u, v) = 1 and costs c(u, v) per unit of flow. q(u, k)
    is a source's quality k, or a terminal's upper bound on it; bl/bu bound the flow through
    a node (supply, pool capacity, demand)."""
    text = _strip_comments(text)
    sets = {name: _range(body) for name, body in
            re.findall(r"set\s+(\w+)(?:\(\w+\))?\s*/([^/]*)/\s*;", text, flags=re.I)}
    tables = {name: _table(body) for name, body in
              re.findall(r"table\s+(\w+)\(\w+,\s*\w+\)\s*\n(.*?);", text, flags=re.I | re.S)}
    params = {name: _parameter(body) for name, body in
              re.findall(r"parameter\s+(\w+)\(\w+\)\s*/(.*?)/\s*;", text, flags=re.I | re.S)}
    for key in ("i", "s", "t", "k"):
        if key not in sets:
            raise ValueError(f"set {key} missing")
    for key in ("c", "a", "q"):
        if key not in tables:
            raise ValueError(f"table {key} missing")
    for key in ("bl", "bu"):
        if key not in params:
            raise ValueError(f"parameter {key} missing")
    sources, terminals = sets["s"], sets["t"]
    pools = [n for n in sets["i"] if n not in sources and n not in terminals]
    arcs = []
    for (u, v), flag in sorted(tables["a"].items()):
        if flag == 0.0:
            continue
        if flag != 1.0:
            raise ValueError(f"adjacency a({u},{v}) = {flag}, expected 0 or 1")
        arcs.append([u, v, tables["c"].get((u, v), 0.0)])
    for (u, v), cost in tables["c"].items():
        if cost != 0.0 and tables["a"].get((u, v), 0.0) == 0.0:
            raise ValueError(f"cost on ({u},{v}) but no arc")
    qualities = len(sets["k"])
    quality: dict[str, list[float]] = {}
    for (node, k), value in sorted(tables["q"].items()):
        quality.setdefault(str(node), [0.0] * qualities)[sets["k"].index(k)] = value
    return {
        "sources": sources, "pools": pools, "terminals": terminals,
        "num_qualities": qualities, "arcs": arcs, "quality": quality,
        "lower": {str(n): v for n, v in sorted(params["bl"].items())},
        "upper": {str(n): v for n, v in sorted(params["bu"].items())},
    }


def download(filename: str, attempts: int = 6) -> bytes:
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(ARCHIVE_URL + filename, timeout=90) as response:
                data = response.read()
            if data:
                return data
        except OSError as error:  # the archive is slow and sometimes resets; retry
            last = error
        time.sleep(2 + 3 * attempt)
    raise SystemExit(f"could not download {filename} from the Internet Archive: {last}")


def transcribe(name: str, raw: bytes) -> dict:
    network = parse_gams_data(raw.decode("ascii"))
    return {
        "name": name,
        "source_file": FILES[name],
        "source_url": ORIGINAL_URL + FILES[name],
        "archived_url": ARCHIVE_URL + FILES[name],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "cost_convention": "cost per unit of flow on every arc; a pool-to-terminal or "
                           "source-to-terminal arc's cost includes minus the terminal's price",
        **network,
    }


def dump(record: dict) -> str:
    """JSON with one arc, one node's qualities or one node's bound per line: a reviewer diffs
    it against the GAMS file row by row."""
    out = ["{"]
    items = list(record.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position + 1 < len(items) else ""
        if key == "arcs":
            body = [f"  {json.dumps(arc)}" for arc in value]
            out += [f" {json.dumps(key)}: [", ",\n".join(body), f" ]{comma}"]
        elif isinstance(value, dict):
            body = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()]
            out += [f" {json.dumps(key)}: {{", ",\n".join(body), f" }}{comma}"]
        else:
            out.append(f" {json.dumps(key)}: {json.dumps(value)}{comma}")
    return "\n".join(out) + "\n}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed JSON instead of writing it")
    parser.add_argument("--from-dir", type=Path, default=None,
                        help="parse the .gms files from this directory instead of downloading")
    parser.add_argument("--instances", nargs="*", default=sorted(FILES))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    for name in args.instances:
        raw = ((args.from_dir / FILES[name]).read_bytes() if args.from_dir
               else download(FILES[name]))
        record = transcribe(name, raw)
        target = OUT_DIR / f"{name}.json"
        text = dump(record)
        if args.check:
            same = target.exists() and json.loads(target.read_text()) == record
            mismatches += not same
            print(f"{name:<10} {'same' if same else 'DIFFERS'}  sha256 {record['source_sha256']}")
        else:
            target.write_text(text, encoding="utf-8", newline="\n")
            print(f"{name:<10} {len(record['arcs']):>4} arcs  sha256 {record['source_sha256']}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
