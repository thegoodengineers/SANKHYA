#!/usr/bin/env python3
"""Measurement only (never merged): per-instance dual simplex counters and HiGHS's presolve size."""
import json, re, subprocess, sys, tempfile, time
from pathlib import Path

binary = sys.argv[1]
import os
dump = []
for a in sys.argv[2:]:
    if a.startswith("thr="):
        os.environ["A527_THR"] = a[4:]
    if a.startswith("minfrac="):
        os.environ["A527_MINFRAC"] = a[8:]
    if a.startswith("dump="):
        dump = a[5:].split(",")
extra = [a for a in sys.argv[2:] if not a.startswith(("names=", "thr=", "minfrac=", "dump="))]
namearg = [a[6:] for a in sys.argv[2:] if a.startswith("names=")]
import os
names = os.environ.get("A527_NAMES") or "stocfor3 dfl001 pilot87 pilot greenbea greenbeb d2q06c stocfor2 woodw ganges bnl2 fit2p bnl1 pilot.we 25fv47 degen3 cycle"
names = (namearg[0].replace(",", " ") if namearg else names).split()
try:
    import highspy
except ImportError:
    highspy = None
for name in names:
    mps = Path("data/netlib") / f"{name}.mps"
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        cmd = [binary, "solve", str(mps), "--time-limit", "600", "--stats", str(stats),
               "--option", "log_level=verbose", "--option", "certified_gap=false",
               "--option", "threads=1"]
        for e in extra:
            cmd += ["--option", e]
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t
        log = p.stdout + p.stderr
        blob = json.loads(stats.read_text()) if stats.exists() else {}
    res = blob.get("result", {}); eff = blob.get("effort", {})
    meas = re.findall(r"MEASURE (.*)", log)
    pre = re.findall(r"Presolve: .*", log)
    hand = re.findall(r"Dual simplex: (.*?) at iteration (\d+); the primal", log)
    print(f"OURS {name} status={res.get('status')} obj={res.get('objective')} iters={eff.get('iterations')} "
          f"alg={res.get('algorithm')} wall={wall:.2f}")
    if not blob:
        print("  NOBLOB rc=%s %s" % (p.returncode, " | ".join(log[-800:].splitlines())))
    for m in meas:
        print(f"  MEASURE {m}")
    for h in hand:
        print(f"  HANDOVER at {h[1]}: {h[0][:200]}")
    for x in re.findall(r".*(?:MEASURE-SUB|Dual reductions|Substitution|route|unscaled|guard|violat|refut|withdr|numerical|retry|postsolve|Status|status).*", log)[:40]:
        print("  LOG " + x[:300])
    for x in pre[:1]:
        print(f"  {x[:240]}")
    if name in dump:
        for x in log.splitlines()[-4000:]:
            print("  FULL " + x[:400])
    for x in re.findall(r"presolve: .*", log):
        print(f"  OURPRE {x}")
    if highspy is not None:
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        h.readModel(str(mps))
        lp = h.getLp()
        h.setOptionValue("output_flag", True); h.setOptionValue("presolve_rule_logging", True)
        h.setOptionValue("log_file", "/tmp/hp.log"); h.setOptionValue("log_to_console", False)
        h.presolve()
        try:
            for line in open("/tmp/hp.log"):
                print("  HPRE " + line.rstrip())
        except OSError:
            pass
        try:
            plp = h.getPresolvedLp()
            pdims = f"{plp.num_row_} x {plp.num_col_} nnz {plp.a_matrix_.value_.__len__()}"
        except Exception as ex:  # noqa: BLE001
            pdims = f"n/a ({ex})"
        h2 = highspy.Highs(); h2.setOptionValue("output_flag", False); h2.setOptionValue("threads", 1)
        h2.readModel(str(mps)); t = time.perf_counter(); h2.run(); hw = time.perf_counter() - t
        info = h2.getInfo()
        print(f"  HIGHS {lp.num_row_} x {lp.num_col_} -> presolved {pdims}; iters={info.simplex_iteration_count} "
              f"obj={info.objective_function_value} wall={hw:.2f}")
    sys.stdout.flush()
