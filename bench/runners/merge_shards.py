#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Put the shard CSVs of one sharded run into the one CSV the docs read (#1199).

The 240-model MIPLIB 2017 benchmark set does not fit one bench.yml job at 60 s for two
solvers, so it runs as N jobs of one shard each (`fetch_miplib.py --shard K/N`). Each job
writes its own CSVs; this joins them, row for row and byte for byte as the runners wrote
them, and refuses anything that would make the joined file say more than the shards did:

- shards with different headers, or rows of different commits (one run, one binary);
- an instance (per solver, for a head-to-head CSV) in two shards;
- with --manifest, any instance of the manifest in no shard: a set count over a partial
  set is not the set's count.

Each row keeps its machine column, so a reader sees which CPU every instance ran on.

    python bench/runners/merge_shards.py --manifest data/miplib2017-bench/manifest.json \\
        --out bench/results/miplib2017-bench-60s-<sha>.csv shard1.csv shard2.csv ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def merge(paths: list[Path], manifest: Path | None) -> tuple[list[str], list[dict]]:
    """The header and rows of the shards joined, or ValueError naming what is wrong."""
    header: list[str] | None = None
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if header is None:
                header = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != header:
                raise ValueError(f"{path.name}: a header unlike the first shard's")
            rows += list(reader)
    if header is None:
        raise ValueError("no shard given")
    commits = sorted({r.get("git_commit", "") for r in rows})
    if len(commits) != 1:
        raise ValueError(f"rows of more than one commit: {', '.join(commits)}")
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["instance"], r.get("solver", ""))
        if key in seen:
            raise ValueError(f"{r['instance']} is in more than one shard")
        seen.add(key)
    if manifest is not None:
        wanted = set(json.loads(manifest.read_text(encoding="utf-8"))["instances"])
        missing = sorted(wanted - {r["instance"] for r in rows})
        if missing:
            raise ValueError(f"{len(missing)} instance(s) of the manifest in no shard: "
                             f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    rows.sort(key=lambda r: (r["instance"], r.get("solver", "")))
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="refuse unless every instance of this manifest is in a shard")
    args = parser.parse_args()
    try:
        header, rows = merge(args.shards, args.manifest)
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    machines = sorted({r.get("machine", "") for r in rows})
    print(f"wrote {args.out}: {len(rows)} rows from {len(args.shards)} shard(s), "
          f"{len(machines)} machine(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
