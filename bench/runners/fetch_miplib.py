#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch a MIPLIB 2017 subset and its PUBLISHED optimal objective values.

The MILP counterpart of fetch_data.py, and careful about the same things for the same
reasons: a benchmark table is only evidence if a judge can regenerate it, and a reference
value we typed in ourselves is not a reference value.

THREE THINGS THIS IS DELIBERATE ABOUT.

1.  ONLY INSTANCES MARKED `=opt=`. MIPLIB's solution file marks each instance `=opt=` (a
    PROVEN optimum), `=best=` (the best value anyone has found, not proven), `=unkn=`,
    `=inf=` or `=unbd=`. Only `=opt=` gives a value we can asserted equality against. Scoring
    ourselves against a `=best=` bound would mean a run could "beat" the reference and look
    like a triumph while actually being wrong, which is the failure mode this project exists
    to avoid.

2.  THE SUBSET IS CHOSEN BY SIZE, AND SIZE COMES FROM AN HTTP HEAD. MIPLIB tags instances
    easy/hard/open, but "easy" there means easy for a mature solver with cuts, presolve and
    a dual simplex. Ours has none of the first two, so the tag alone selects models we cannot
    finish. Compressed file size is a crude proxy for difficulty and an honest one: it needs
    no endpoint beyond the instance URL itself, it is identical for everyone, and the exact
    list ends up recorded in the manifest either way.

3.  NOTHING IS COMMITTED. `data/miplib/` is gitignored. The manifest records the sha256 of
    every instance downloaded, so a later run can prove it solved the same bytes.

    python bench/runners/fetch_miplib.py                 # 30 smallest easy instances
    python bench/runners/fetch_miplib.py --count 10
    python bench/runners/fetch_miplib.py flugpl gen-ip002
    python bench/runners/fetch_miplib.py --select-tier2      # rewrite miplib_tier2.json (#504)
    python bench/runners/fetch_miplib.py --tier 2            # download the tier-2 instances
    python bench/runners/fetch_miplib.py --tier 2 neos5      # ... or some of them

THE SECOND TIER (#504). TIER2_RULE below is the whole selection rule, written before the tier
was ever run: MIPLIB 2017's BENCHMARK set (not the easy list), those with a proven optimum,
the 60 smallest by compressed size. Size is the only criterion - never whether we solve an
instance - and small files are what fit this laptop's memory. `--select-tier2` applies the
rule (HTTP HEAD only, no instance downloaded) and records the result, with the sizes it saw,
in bench/runners/miplib_tier2.json, which is committed; `--tier 2` downloads exactly that
list into data/miplib-tier2/ (gitignored) with the same manifest shape as the first tier.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib"

MIPLIB = "https://miplib.zib.de"
SOLUTION_FILE = f"{MIPLIB}/downloads/miplib2017-v28.solu"
EASY_LIST = f"{MIPLIB}/downloads/easy-v15.test"
BENCHMARK_LIST = f"{MIPLIB}/downloads/benchmark-v2.test"
TIER2_FILE = REPO_ROOT / "bench" / "runners" / "miplib_tier2.json"
TIER2_DIR = REPO_ROOT / "data" / "miplib-tier2"
TIER2_COUNT = 60
TIER2_RULE = (
    "The instances of the MIPLIB 2017 benchmark set (benchmark-v2.test) that miplib2017-v28.solu "
    "marks =opt=, ordered by the byte size of their .mps.gz file as the server reports it "
    "(Content-Length), ties broken by name, the first 60. No instance is added or removed for "
    "any other reason, in particular not for how SANKHYA does on it.")
INSTANCE_BASE = f"{MIPLIB}/WebData/instances"


def ssl_context() -> ssl.SSLContext:
    """certifi's bundle when it is installed, the system default otherwise.

    This box has no default CA file at all (`ssl.get_default_verify_paths().cafile` is None),
    so miplib.zib.de fails certificate verification while netlib.org happens to succeed.
    Falling back rather than requiring certifi keeps the script working where the system
    store is fine.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout, context=ssl_context()) as response:
        return response.read()


def content_length(url: str, timeout: int = 30) -> int | None:
    """Size without downloading the body. Returns None when the server will not say."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as r:
            value = r.headers.get("Content-Length")
            return int(value) if value else None
    except OSError:
        return None


def published_optima() -> dict[str, float]:
    """Parse MIPLIB's own solution file. `=opt=` lines only - see the module docstring."""
    text = get(SOLUTION_FILE).decode(errors="replace")
    optima: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "=opt=":
            try:
                optima[parts[1]] = float(parts[2])
            except ValueError:
                continue
    return optima


def easy_instances() -> list[str]:
    text = get(EASY_LIST).decode(errors="replace")
    return [entry.replace(".mps.gz", "") for entry in text.split() if entry.endswith(".mps.gz")]


def benchmark_instances() -> list[str]:
    text = get(BENCHMARK_LIST).decode(errors="replace")
    return [entry.replace(".mps.gz", "") for entry in text.split() if entry.endswith(".mps.gz")]


def select_by_size(names: list[str], optima: dict[str, float], sizes: dict[str, int | None],
                   count: int) -> list[tuple[str, int]]:
    """TIER2_RULE as a function: those of `names` with a proven optimum and a known size,
    smallest first, ties by name, the first `count`. Pure, so the rule is unit-tested."""
    known = [(name, sizes[name]) for name in names
             if name in optima and sizes.get(name) is not None]
    known.sort(key=lambda pair: (pair[1], pair[0]))
    return known[:count]


def sizes_by_head(names: list[str]) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(content_length, f"{INSTANCE_BASE}/{n}.mps.gz"): n for n in names}
        for future in concurrent.futures.as_completed(futures):
            sizes[futures[future]] = future.result()
    return sizes


def select_tier2() -> int:
    """Apply TIER2_RULE and record the list; downloads no instance."""
    optima = published_optima()
    names = benchmark_instances()
    sizes = sizes_by_head(names)
    unsized = sorted(n for n in names if n in optima and sizes.get(n) is None)
    chosen = select_by_size(names, optima, sizes, TIER2_COUNT)
    record = {
        "rule": TIER2_RULE,
        "benchmark_list": BENCHMARK_LIST,
        "solution_file": SOLUTION_FILE,
        "benchmark_instances": len(names),
        "with_proven_optimum": sum(1 for n in names if n in optima),
        "not_sized": unsized,
        "instances": [{"name": n, "gz_bytes": b, "published_optimal": optima[n]}
                      for n, b in chosen],
    }
    TIER2_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  {len(names)} benchmark instances, {record['with_proven_optimum']} with a proven "
          f"optimum, {len(unsized)} the server would not size")
    print(f"  kept {len(chosen)}: {chosen[0][1]} to {chosen[-1][1]} bytes compressed")
    print(f"wrote {TIER2_FILE.relative_to(REPO_ROOT)}")
    return 0


def choose(count: int) -> list[tuple[str, int]]:
    """The `count` smallest easy instances that have a proven optimum, with their sizes."""
    optima = published_optima()
    candidates = [name for name in easy_instances() if name in optima]
    print(f"  {len(optima)} instances have a proven optimum", flush=True)
    print(f"  {len(candidates)} of the easy-tagged set are among them", flush=True)
    print(f"  sizing them by HTTP HEAD (no bodies downloaded)...", flush=True)

    # Concurrently, and flushed as it goes. Sequentially this is roughly one request per
    # instance per second - twelve minutes of a script printing nothing, because Python
    # buffers stdout when it is not a terminal. Both halves of that were worth fixing: a
    # fetch that looks hung is a fetch people interrupt.
    sized: list[tuple[str, int]] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(content_length, f"{INSTANCE_BASE}/{n}.mps.gz"): n
                   for n in candidates}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            size = future.result()
            if size is not None:
                sized.append((futures[future], size))
            if done % 100 == 0 or done == len(candidates):
                print(f"    {done}/{len(candidates)}", flush=True)
    sized.sort(key=lambda pair: (pair[1], pair[0]))
    return sized[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instances", nargs="*",
                        help="explicit instance names; omit to take the smallest easy ones")
    parser.add_argument("--count", type=int, default=30,
                        help="how many of the smallest easy instances to take (default 30)")
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1,
                        help="2: download the list in bench/runners/miplib_tier2.json into "
                             "data/miplib-tier2 (all of it, or the names given)")
    parser.add_argument("--select-tier2", action="store_true",
                        help="apply the tier-2 rule and rewrite miplib_tier2.json; downloads "
                             "no instance")
    args = parser.parse_args()

    if args.select_tier2:
        return select_tier2()
    data_dir = DATA_DIR if args.tier == 1 else TIER2_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    optima = published_optima()

    if args.tier == 2:
        listed = [entry["name"] for entry in json.loads(TIER2_FILE.read_text())["instances"]]
        unlisted = [n for n in args.instances if n not in listed]
        if unlisted:
            print(f"not in the tier-2 list: {', '.join(unlisted)}", file=sys.stderr)
            return 1
        chosen = [(name, 0) for name in (args.instances or listed)]
        selection = "tier2" if not args.instances else "tier2-explicit"
    elif args.instances:
        missing = [n for n in args.instances if n not in optima]
        if missing:
            print(f"no PROVEN optimum published for: {', '.join(missing)}", file=sys.stderr)
            print("MIPLIB marks those =best=, =unkn=, =inf= or =unbd=; see the module "
                  "docstring for why only =opt= is usable here", file=sys.stderr)
            return 1
        chosen = [(name, 0) for name in args.instances]
        selection = "explicit"
    else:
        chosen = choose(args.count)
        selection = f"smallest-{args.count}-easy"

    manifest = {
        "source": MIPLIB,
        "solution_file": SOLUTION_FILE,
        "easy_list": EASY_LIST,
        "selection": selection,
        "available_with_proven_optimum": len(optima),
        "instances": {},
    }

    print()
    print(f"  {'instance':<28}{'bytes':>10}  published optimum")
    for name, _size in chosen:
        url = f"{INSTANCE_BASE}/{name}.mps.gz"
        blob = get(url)
        destination = data_dir / f"{name}.mps.gz"
        # Written to a scratch path and renamed, so an interrupted fetch cannot leave a
        # truncated instance behind that looks like a real one. Same reasoning as #87.
        scratch = destination.with_suffix(destination.suffix + ".partial")
        try:
            scratch.write_bytes(blob)
            scratch.replace(destination)
        finally:
            scratch.unlink(missing_ok=True)

        manifest["instances"][name] = {
            "name": name,
            "url": url,
            "gz_bytes": len(blob),
            "gz_sha256": hashlib.sha256(blob).hexdigest(),
            "published_optimal": optima[name],
        }
        print(f"  {name:<28}{len(blob):>10}  {optima[name]:.10g}", flush=True)

    reference = data_dir / "reference.json"
    reference.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                         newline="\n")
    print()
    print(f"wrote {reference.relative_to(REPO_ROOT)}")
    print(f"{len(manifest['instances'])} instance(s) in {data_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
