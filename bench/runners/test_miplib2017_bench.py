#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the whole MIPLIB 2017 benchmark set (#1199): the .solu parse, the shards, the
`=inf=` grading, the hang timeout, HiGHS on a .mps.gz, the CSV names against every other glob,
the shard merge and the doc counts. Hermetic: nothing is downloaded and no solver runs.

    python bench/runners/test_miplib2017_bench.py
"""
from __future__ import annotations

import csv
import fnmatch
import gzip
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import compare_suite  # noqa: E402
import fetch_miplib  # noqa: E402
import merge_shards  # noqa: E402
import miplib  # noqa: E402
import miplib2017_bench_doc as doc  # noqa: E402
import rivals  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES += 1


print(".solu records")
SOLU = "=opt=  30n20b8  302\n=inf=  bnatt500\n=best= dano3mip  664.02\n=opt=  30n20b8  999\n"
records = fetch_miplib.solu_records(SOLU)
check(records["30n20b8"] == {"status": "=opt=", "text": "302", "line": 1},
      "an =opt= record, the first of a name wins", str(records["30n20b8"]))
check(records["bnatt500"] == {"status": "=inf=", "text": "", "line": 2}, "=inf= has no value")
check(records["dano3mip"]["status"] == "=best=", "=best= kept as its marker")

print("shards")
entries = {f"m{i:03d}": {"gz_bytes": (i * 7919) % 1000} for i in range(240)}
shards = [fetch_miplib.shard(entries, k, 6) for k in range(1, 7)]
union = [name for part in shards for name in part]
check(sorted(union) == sorted(entries) and len(union) == len(set(union)),
      "six shards are disjoint and cover the set")
check({len(part) for part in shards} == {40}, "each holds 40")
largest = sorted(entries, key=lambda n: (entries[n]["gz_bytes"], n))[-6:]
check(all(sum(1 for n in largest if n in part) == 1 for part in shards),
      "the six largest models land in six different shards")
for bad in ((0, 6), (7, 6)):
    try:
        fetch_miplib.shard(entries, *bad)
        check(False, f"shard {bad} refused")
    except ValueError:
        check(True, f"shard {bad} refused")

print("grading an =inf= model")
blob = {"status": "infeasible", "objective": None, "verified": None, "wall_seconds": 1.0}
row = miplib.make_row("bnatt500", {"solu_status": "=inf=", "published_optimal": None}, None,
                      blob, "abc1234", "", 1, "box", "now")
check(row["matched_published"] == 1 and row["proved_optimal"] == 1,
      "infeasible on an =inf= model is matched and proved")
blob = {"status": "optimal", "objective": 5.0, "verified": True, "wall_seconds": 1.0}
row = miplib.make_row("bnatt500", {"solu_status": "=inf=", "published_optimal": None}, None,
                      blob, "abc1234", "", 1, "box", "now")
check(row["matched_published"] == 0 and row["proved_optimal"] == 0,
      "optimal on an =inf= model is neither")
blob = {"status": "infeasible", "objective": None, "verified": None, "wall_seconds": 1.0}
row = miplib.make_row("p", {"solu_status": "=opt=", "published_optimal": 3.0}, 3.0, blob,
                      "abc1234", "", 1, "box", "now")
check(row["matched_published"] == 0 and row["proved_optimal"] == 0,
      "infeasible on an =opt= model is neither")

print("the hang timeout")
with tempfile.TemporaryDirectory() as tmp:
    sleeper = Path(tmp) / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    saved = miplib.HANG_MARGIN_SECONDS
    miplib.HANG_MARGIN_SECONDS = 0.5
    try:
        import subprocess  # noqa: PLC0415
        real_run = subprocess.run

        def fake_run(command, **kwargs):
            return real_run([sys.executable, str(sleeper)], **kwargs)

        subprocess.run = fake_run
        result = miplib.solve(Path("unused"), Path("unused.mps"), 0.5, False)
    finally:
        subprocess.run = real_run
        miplib.HANG_MARGIN_SECONDS = saved
    check(result["status"] == "hang" and result["wall_seconds"] < 10,
          "a solve past the limit plus the margin is killed and recorded hang", str(result))

print("the overrun log")
with tempfile.TemporaryDirectory() as tmp:
    talker = Path(tmp) / "talker.py"
    talker.write_text("import sys, time\nprint('Presolve: done', flush=True)\n"
                      "print('Symmetry: searching', flush=True)\ntime.sleep(30)\n",
                      encoding="utf-8")
    logs = Path(tmp) / "logs"
    saved_margin, saved_dir = miplib.HANG_MARGIN_SECONDS, miplib.OVERRUN_LOG_DIR
    miplib.HANG_MARGIN_SECONDS, miplib.OVERRUN_LOG_DIR = 1.5, logs
    try:
        import subprocess  # noqa: PLC0415
        real_run = subprocess.run
        seen = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            return real_run([sys.executable, str(talker)], **kwargs)

        subprocess.run = fake_run
        result = miplib.solve(Path("unused"), Path("big-model.mps.gz"), 0.5, False)
    finally:
        subprocess.run = real_run
        miplib.HANG_MARGIN_SECONDS, miplib.OVERRUN_LOG_DIR = saved_margin, saved_dir
    kept = logs / "big-model.log"
    text = kept.read_text(encoding="utf-8") if kept.exists() else ""
    check(result["status"] == "hang" and "killed at the hang margin" in text
          and text.rstrip().endswith("Symmetry: searching"),
          "a killed run leaves its log, and the last line names the phase it was in", text)
    check("log_to_console=true" in seen["command"], "the solver logs to its console then")
    miplib.keep_overrun_log(Path("quick.mps"), 60.0, 61.0, False, "all fine")
    saved_dir = miplib.OVERRUN_LOG_DIR
    miplib.OVERRUN_LOG_DIR = logs
    try:
        miplib.keep_overrun_log(Path("quick.mps"), 60.0, 61.0, False, "all fine")
        miplib.keep_overrun_log(Path("late.mps"), 60.0, 66.0, False, b"Root cuts: filtering")
    finally:
        miplib.OVERRUN_LOG_DIR = saved_dir
    check(not (logs / "quick.log").exists() and "returned late" in
          (logs / "late.log").read_text(encoding="utf-8"),
          "a run within 5 s of its limit leaves none; one past it does, bytes or text")

print("HiGHS reads a .mps.gz")
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp) / "tiny.mps.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(b"NAME TINY\nROWS\n N obj\nCOLUMNS\nENDATA\n")
    out_dir = Path(tmp) / "work"
    out_dir.mkdir()
    plain = rivals._mps_path(source, out_dir)
    check(plain.name == "tiny.mps" and plain.read_bytes().startswith(b"NAME TINY"),
          "decompressed to a plain .mps beside the work", str(plain))
check(compare_suite.SUITES["miplib2017-bench"]["solvers"] == ("highs",)
      and compare_suite.SUITES["miplib2017-bench"]["suffix"] == ".mps.gz", "the suite")

print("CSV names")
names = {"ours": miplib.default_out_name("miplib2017-bench", 1, 1, 60.0, "abc1234"),
         "highs": compare_suite.default_out_name("miplib2017-bench", True, 60.0, "abc1234"),
         "shard": "miplib2017-bench-60s-shard1of6-abc1234.csv",
         "hshard": "highs-miplib2017-bench-60s-shard1of6-abc1234.csv"}
check(names["ours"] == "miplib2017-bench-60s-abc1234.csv"
      and names["highs"] == "highs-miplib2017-bench-60s-abc1234.csv", "the joined names",
      str(names))
SOURCES = [*sorted((REPO_ROOT / "bench" / "runners").glob("*.py")),
           *sorted((REPO_ROOT / "tools").glob("*.py"))]
globs = set()
for source in SOURCES:
    if source.name in ("miplib2017_bench_doc.py", "test_miplib2017_bench.py"):
        continue
    for literal in re.findall(r"[\"']([^\"'\s]*[*?\[][^\"'\s]*\.csv)[\"']",
                              source.read_text(encoding="utf-8", errors="replace")):
        globs.add(re.sub(r"\{[^}]*\}", "*", literal))
globs.discard("*.csv")
hits = sorted((g, n) for g in globs for n in names.values() if fnmatch.fnmatch(n, g))
check(not hits, "no other glob takes these names", str(hits))
check(fnmatch.fnmatch(names["ours"], doc.sankhya_pattern())
      and fnmatch.fnmatch(names["highs"], doc.highs_pattern())
      and not fnmatch.fnmatch(names["shard"], doc.sankhya_pattern())
      and not fnmatch.fnmatch(names["hshard"], doc.highs_pattern()),
      "the doc's globs take the joined files and not a shard")

print("merging shards")
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)

    def write(name, rows):
        path = d / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def r(instance, commit="abc1234", machine="cpu A"):
        return {"instance": instance, "git_commit": commit, "machine": machine}

    one = write("s1.csv", [r("b"), r("a")])
    two = write("s2.csv", [r("c", machine="cpu B")])
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"instances": {"a": {}, "b": {}, "c": {}}}), encoding="utf-8")
    header, rows = merge_shards.merge([one, two], manifest)
    check([x["instance"] for x in rows] == ["a", "b", "c"]
          and {x["machine"] for x in rows} == {"cpu A", "cpu B"},
          "joined, sorted, each row keeps its machine")
    for label, paths, why in (
            ("a duplicate", [one, write("s3.csv", [r("a")])], "more than one shard"),
            ("two commits", [one, write("s4.csv", [r("c", commit="def5678")])], "commit"),
            ("a missing instance", [one], "in no shard")):
        try:
            merge_shards.merge(paths, manifest)
            check(False, f"{label} is refused")
        except ValueError as error:
            check(why in str(error), f"{label} is refused", str(error))

print("the doc's counts")
ours = [{"instance": "a", "status": "optimal", "matched_published": "1", "proved_optimal": "1",
         "independently_verified": "1"},
        {"instance": "b", "status": "feasible", "matched_published": "1", "proved_optimal": "0",
         "independently_verified": "1"},
        {"instance": "c", "status": "feasible", "matched_published": "0", "proved_optimal": "0",
         "independently_verified": "1"},
        {"instance": "d", "status": "infeasible", "matched_published": "1",
         "proved_optimal": "1", "independently_verified": ""},
        {"instance": "e", "status": "optimal", "matched_published": "1", "proved_optimal": "1",
         "independently_verified": "0"}]
c = doc.counts(ours, doc.sankhya_verdicts)
check(c == {"instances": 5, "proved": 2, "matched": 3, "feasible": 3},
      "SANKHYA: proved needs the proof, matched does not, a rejected point counts nowhere", str(c))
theirs = [{"instance": "a", "status": "optimal", "matches_reference": "1",
           "independently_verified": "1"},
          {"instance": "b", "status": "time_limit", "matches_reference": "0",
           "independently_verified": "1"},
          {"instance": "d", "status": "infeasible", "matches_reference": "1",
           "independently_verified": ""}]
c = doc.counts(theirs, doc.highs_verdicts)
check(c == {"instances": 3, "proved": 2, "matched": 2, "feasible": 2},
      "HiGHS: an incumbent at the limit is feasible and verified, not matched", str(c))
empty = doc.section(None, None, {"instances": {"a": {"solu_status": "=opt="}}})
check("Not yet run" in empty and "--shard 1/6" in empty, "with no CSV: how to reproduce")

COMMITTED = REPO_ROOT / "data" / "miplib2017-bench" / "manifest.json"
if COMMITTED.exists():
    real = json.loads(COMMITTED.read_text(encoding="utf-8"))
    entries = real["instances"]
    check(len(entries) == 240, "the committed manifest holds all 240", str(len(entries)))
    check(real["solution_file_sha256"] == fetch_miplib.SOLUTION_FILE_SHA256,
          "read from the pinned solution file")
    check(all(e["solu_status"] in ("=opt=", "=inf=") for e in entries.values())
          and all((e["published_optimal"] is None) == (e["solu_status"] == "=inf=")
                  for e in entries.values()),
          "every reference is proven: an =opt= value, or =inf= with no value")
    check(all(re.fullmatch(r"[0-9a-f]{64}", e["gz_sha256"]) for e in entries.values()),
          "a sha256 for every model")

print(f"\n{'all passed' if FAILURES == 0 else f'{FAILURES} FAILED'}")
sys.exit(1 if FAILURES else 0)
