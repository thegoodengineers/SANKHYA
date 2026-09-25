# SPDX-License-Identifier: Apache-2.0
"""The figures section 6 of demo/run_sih_demo.sh states, read from committed CSVs.

Prints one KEY=value line per figure; the demo turns each into a @KEY@ substitution in its
closing ledger. Each figure comes with a *_CSV line naming the file it was read from, and
every CSV is named here by its full file name, so a figure always belongs to the run that
produced it. A missing file
raises, and the demo stops rather than print a sentence without its number.

    python demo/evidence.py bench/results
"""

import csv
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "bench/results"


def rows(name):
    with open(os.path.join(ROOT, name), newline="") as f:
        return list(csv.DictReader(f))


def count(rs, column, value="1"):
    return sum(1 for r in rs if r.get(column) == value)


def cite(name):
    return "bench/results/" + name


def gpu_ratios(name, tolerance):
    """CPU seconds over GPU seconds per synthetic size, and the card's name."""
    table = {}
    for r in rows(name):
        if r["tolerance"] == tolerance:
            table.setdefault(r["instance"], {})[r["algorithm"]] = float(r["seconds"])
    parts = []
    for instance, t in table.items():
        if "pdhg-cpu" in t and "pdhg-cuda" in t:
            size = instance.split("_")[1].split("x")[0]
            parts.append(f"{size} {t['pdhg-cpu'] / t['pdhg-cuda']:.2f}x")
    return ", ".join(parts)


def main():
    out = {}

    name = "kennington-full-65eecbc.csv"
    r = rows(name)
    out["KENN"] = (f"{count(r, 'passed')} of {len(r)} match the published optimum and verify, "
                   f"{count(r, 'independently_verified')} of {len(r)} verified")
    out["KENN_CSV"] = cite(name)

    name = "netlib-infeasible-65eecbc.csv"
    r = rows(name)
    claimed = sum(1 for x in r if x["status"] in ("optimal", "feasible", "unbounded"))
    out["INFEAS"] = (f"{count(r, 'status', 'infeasible')} of {len(r)} declared infeasible, "
                     f"{count(r, 'passed')} with a Farkas certificate the verifier accepts, "
                     f"{claimed} given a solution")
    out["INFEAS_CSV"] = cite(name)

    name = "maros-meszaros-65eecbc.csv"
    r = rows(name)
    out["MM"] = (f"{count(r, 'passed')} of {len(r)} pass, {count(r, 'status', 'optimal')} "
                 f"reach optimal, at a {float(r[0]['time_limit']):g} s limit")
    out["MM_CSV"] = cite(name)

    off, on = "mittelmann-65eecbc.csv", "mittelmann-race-65eecbc.csv"
    r_off, r_on = rows(off), rows(on)
    out["RACE_MM"] = (f"{count(r_on, 'passed')} of {len(r_on)} with the race, "
                      f"{count(r_off, 'passed')} of {len(r_off)} without, at a "
                      f"{float(r_on[0]['time_limit']):g} s limit")
    out["RACE_MM_CSV"] = f"{cite(on)} against {cite(off)}"
    name = "netlib-full-race-65eecbc.csv"
    r = rows(name)
    out["RACE_NETLIB"] = (f"{count(r, 'independently_verified')} of {len(r)} verified, "
                          f"{count(r, 'matches_published')} matching Netlib's table")
    out["RACE_NETLIB_CSV"] = cite(name)

    for key, name in (("GPU_LAPTOP", "gpu-fb72ab4.csv"), ("GPU_L4", "gpu-l4-fdc89c5.csv")):
        out[key + "_CARD"] = rows(name)[0]["gpu"].split(" (")[0]
        out[key + "_CSV"] = cite(name)
        out[key + "_4"] = gpu_ratios(name, "1e-04")
        out[key + "_8"] = gpu_ratios(name, "1e-08")

    name = "gpu-real-l4-fdc89c5.csv"
    runs = {}
    for x in rows(name):
        runs.setdefault(x["instance"], {}).setdefault(x["tolerance"], {})[x["algorithm"]] = x
    lines = []
    for instance, by_tolerance in sorted(runs.items()):
        parts = []
        for tolerance, t in by_tolerance.items():
            cpu, gpu = t.get("pdhg-cpu"), t.get("pdhg-cuda")
            if not cpu or not gpu:
                continue
            if cpu["reached_tolerance"] == "1" and gpu["reached_tolerance"] == "1":
                parts.append(f"{float(cpu['seconds']):.1f} s against "
                             f"{float(gpu['seconds']):.1f} s at {tolerance}")
            else:
                parts.append(f"{cpu['status']} on both at {tolerance}")
        lines.append(f"{instance}: " + ", ".join(parts))
    for k, line in enumerate(lines, start=1):
        out[f"GPU_REAL_{k}"] = line
    out["GPU_REAL_CSV"] = cite(name)

    stamps = set()
    for leg, label in (("off", "no cuts:              "), ("on", "the root round:       "),
                       ("tree", "root plus tree rounds:")):
        r = rows(f"miplib-cuts-{leg}.csv")
        stamps |= {x["git_commit"] for x in r}
        out["CUTS_" + leg.upper()] = (f"{label} {count(r, 'matched_published')} matched, "
                                      f"{count(r, 'proved_optimal')} proved")
    out["CUTS_AB_CSV"] = ("bench/results/miplib-cuts-{off,on,tree}.csv, at "
                          + "/".join(sorted(stamps)))

    for key, name in (("MITT_SMALL", "mittelmann-65eecbc.csv"),
                      ("MITT_MEDIUM", "mittelmann-72123ff.csv")):
        r = rows(name)
        out[key] = (f"{count(r, 'passed')} of {len(r)} solved and verified at a "
                    f"{float(r[0]['time_limit']):g} s limit")
        out[key + "_CSV"] = cite(name)

    for key, value in out.items():
        # The demo substitutes these with sed: none may carry its delimiter or an ampersand.
        if "|" in value or "&" in value or "\\" in value:
            raise SystemExit(f"{key}: a value the demo's sed cannot carry: {value!r}")
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
