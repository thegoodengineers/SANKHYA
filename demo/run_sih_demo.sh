#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SANKHYA - the SIH PS26119 demonstration.
#
# Walks the problem statement in its own order and shows, for each thing it asks for, either
# a live result or an explicit admission that we do not have it yet. Every number printed
# below is either produced by a command run during this script or read, by a script and
# with its commit named beside it, from a results CSV committed under bench/results/.
# Nothing is quoted from memory or typed in by hand - see ENGINEERING_RULES.md, "Evidence rules".
#
#   demo/run_sih_demo.sh              # the full walk
#   demo/run_sih_demo.sh --quick      # skip the HiGHS comparison
#
# demo/run_demo.sh is the other script: it goes deep on ONE instance. This one goes wide.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIN="${SANKHYA_BIN:-}"
if [ -z "$BIN" ]; then
  # build-dbg is searched too, and deliberately last. On Windows 11, Smart App Control
  # blocks a freshly linked unsigned executable by hash and reputation, and a Release build
  # is the one that usually gets blocked; a Debug build has different bytes and runs. That
  # is the escape hatch scripts/preflight.sh documents and scripts/reproduce.sh offers as
  # --debug, and without it the demo stops at its first command on a machine that can
  # nevertheless run the solver perfectly well.
  for candidate in build/sankhya build/sankhya.exe build/Release/sankhya.exe                    build-dbg/sankhya build-dbg/sankhya.exe; do
    [ -x "$REPO/$candidate" ] && BIN="$REPO/$candidate" && break
  done
fi
if [ -z "$BIN" ]; then
  echo "No solver binary found. Build first:" >&2
  echo "    scripts/configure.sh build Release && cmake --build build -j" >&2
  exit 1
fi
# CHECKED, NOT ASSUMED. `[ -x ]` above answers a question Windows answers wrongly: the file
# carries the executable bit and is still refused at exec time by Smart App Control, so the
# first thing this script does would fail with "Permission denied" and nothing would say
# why. One cheap invocation settles it here, where the message can name the cause and the
# way out.
if ! "$BIN" version >/dev/null 2>&1; then
  echo "The solver binary at $BIN cannot be executed." >&2
  echo "" >&2
  echo "On Windows 11 this is usually Smart App Control refusing a freshly linked unsigned" >&2
  echo "executable. A Debug build has different bytes and is generally allowed:" >&2
  echo "" >&2
  echo "    scripts/configure.sh build-dbg Debug && cmake --build build-dbg -j" >&2
  echo "    SANKHYA_BIN=\$PWD/build-dbg/sankhya.exe demo/run_sih_demo.sh" >&2
  echo "" >&2
  echo "scripts/preflight.sh reports on this before anything else runs." >&2
  exit 1
fi

# IS IT THIS TREE'S BINARY? The loop above takes the first binary it finds, and a `build/`
# from an older commit runs perfectly while answering as that commit's solver. Said here,
# once, above the first result, rather than left for the audience to infer from a screen of
# failures - see scripts/binary_provenance.sh for the run that cost six of nine instances.
# Not fatal: a judge who pointed SANKHYA_BIN at a binary on purpose is entitled to run it.
# shellcheck source=../scripts/binary_provenance.sh
[ -f "$REPO/scripts/binary_provenance.sh" ] && . "$REPO/scripts/binary_provenance.sh"
if command -v sankhya_binary_staleness >/dev/null 2>&1; then
  STALE="$(sankhya_binary_staleness "$BIN")"
  if [ -n "$STALE" ]; then
    printf '\n\033[1;33m%s\033[0m\n' "The binary is not built from this checkout."
    printf '  %s was linked at %s; this tree is at %s.\n' \
      "$BIN" "${STALE% *}" "${STALE#* }"
    printf '  Every number below is that commit answering, not this one. Rebuild first:\n'
    printf '    scripts/configure.sh build Release && cmake --build build -j\n\n'
  fi
fi

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CASES="data/casestudies"

rule() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 86))"; }
ask()  { printf '\n\033[2mPS26119 asks:\033[0m %s\n\n' "$1"; }

# Solve one model quietly and leave stats in $WORK/<tag>.json and the point in <tag>.sol.
# Extra args, if any, are forwarded to `solve` after the standard ones - e.g. a tighter
# --option for a case where the default tolerance is not the point being demonstrated.
solve_case() {
  local tag="$1" path="$2"
  shift 2
  "$BIN" solve "$path" --option log_to_console=false \
    --write-sol "$WORK/$tag.sol" --stats "$WORK/$tag.json" "$@" >/dev/null
}

field() {  # field <tag> <section> <key>
  "$PYTHON" -c "
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" "$WORK/$1.json" "$2" "$3"
}

# ===========================================================================================
rule "0. What this is"
# ===========================================================================================
"$BIN" version
cat <<'INTRO'

SANKHYA is a mathematical optimization solver core written from scratch in C++20 for
SIH PS26119, issued by Mangalore Refinery and Petrochemicals. LP and MILP engines are
implemented and benchmarked, and so is convex QP - end to end from an MPS file with a
QUADOBJ section through to an independently verified answer, demonstrated twice below: a
small hand-checkable QP right after this intro, and a price-impact variant of the crude
blend in section 3. What remains open on QP: no QPLIB-format reader or benchmark harness
yet (only the generic QPS convention any MPS file can carry). MIQP - a quadratic objective
WITH integer variables - is now solved too, by branch and bound over QP relaxations, with
the bound caveat set out in section 6. This script walks the problem statement in its own
order. Where we do not have something, it says so rather than changing the subject.
INTRO

echo
echo "--- Convex QP, solved for real: demo/qp_blend.mps -----------------------------------"
echo
cat <<'QPINTRO'
    Two streams blending to a fixed 100 kbbl/day pool, with a strictly convex processing
    cost - illustrative, not a claim about real MRPL economics, but a genuine quadratic
    objective read from an MPS QUADOBJ section, dispatched by solve() to the Condat-Vu
    primal-dual QP engine (src/qp/qp_condat_vu.cpp), not an LP relaxation of one.
QPINTRO
# qp_tolerance tighter than default: this is a 2-variable, 1-row QP, so the extra iterations
# cost nothing measurable, and the default 1e-8 leaves a strong-duality relative gap of
# ~3.5e-09 - just over the independent verifier's 1e-9 threshold below. Per the same
# reasoning as tolerances.hpp's kDualityGap note: make the solver converge tighter, not the
# checker looser.
solve_case qp_blend demo/qp_blend.mps --option qp_tolerance=1e-12
echo "    SANKHYA: status $(field qp_blend result status), objective $(field qp_blend result objective)"
echo
echo "    Checked independently by tools/verify_solution.py, which now reads QUADOBJ itself"
echo "    and evaluates c'x + 0.5 x'Qx directly - no code shared with the solver:"
echo
"$PYTHON" tools/verify_solution.py demo/qp_blend.mps "$WORK/qp_blend.sol" --quiet | sed 's/^/        /'
echo
"$PYTHON" - "$WORK/qp_blend.json" <<'PYQP'
import json, sys
obj = json.load(open(sys.argv[1]))["result"]["objective"]
analytic = 200.0 / 3.0
rel = abs(obj - analytic) / analytic
print("    analytic optimum (KKT, by hand): {:.6f}".format(analytic))
print("    relative difference:             {:.3e}".format(rel))
print("    The two agree." if rel <= 1e-6 else "    THEY DISAGREE - see demo/qp_blend.mps.")
sys.exit(0 if rel <= 1e-6 else 1)
PYQP

echo
echo "--- Convex MIQP, solved for real: demo/miqp_blend.mps --------------------------------"
echo
cat <<'MIQPINTRO'
    The same blend, scheduled in WHOLE units. A refinery does not run a stream at 66.667
    kbbl/day because the arithmetic says so - it runs an integer number of batches, tanks or
    campaign days - and the moment that is written down the problem is neither a QP nor a
    MILP but both at once, which is the class PS26119 names as MIQP.

    Nothing new solves it. It is branch and bound (src/mip/branch_and_bound.cpp) with the
    convex QP engine as the node relaxation instead of the simplex.

    The gap targets are set to ZERO below, so the run has to EXHAUST the tree rather than
    stop once it is close enough. At the default 1e-4 the search meets the target at the
    root and reports optimal within it (#188), which is the ordinary industrial answer; a
    zero target is the harder thing and the only one worth demonstrating here: finding
    66.67 is easy, PROVING nothing better exists is the search.
MIQPINTRO
solve_case miqp_blend demo/miqp_blend.mps --option mip_relative_gap=0 --option mip_absolute_gap=0
# The GAP is the point, not the objective: a MIP reports an objective the moment it finds any
# incumbent, and 66.67 would print just the same if the search had given up right after. A
# relative gap of 0 is the part that says nothing better exists.
echo "    SANKHYA: status $(field miqp_blend result status), objective $(field miqp_blend result objective), bound $(field miqp_blend result dual_bound), relative gap $(field miqp_blend result relative_gap)"
echo
echo "    Checked independently by tools/verify_solution.py - no code shared with the solver:"
echo
"$PYTHON" tools/verify_solution.py demo/miqp_blend.mps "$WORK/miqp_blend.sol" --quiet | sed 's/^/        /'
echo
"$PYTHON" - "$WORK/miqp_blend.json" <<'PYMIQP'
import json, sys
obj = json.load(open(sys.argv[1]))["result"]["objective"]

# The hand check from the header of demo/miqp_blend.mps, recomputed here rather than quoted,
# so the demo cannot drift away from the instance it is describing. The equality leaves one
# degree of freedom, so the whole problem is a parabola over the integers and the optimum is
# found by ENUMERATION - the one method that needs no solver and no theory to trust.
def f(t):
    return 0.01 * t * t + 0.02 * (100 - t) ** 2

best = min(range(101), key=f)
print("    integer optimum by enumeration:  LN = {}, HN = {}, objective {:.5f}"
      .format(best, 100 - best, f(best)))
print("    continuous relaxation (200/3):   {:.5f}".format(200.0 / 3.0))
print("    SANKHYA:                         {:.5f}".format(obj))

rel = abs(obj - f(best)) / abs(f(best))
print("    relative difference:             {:.3e}".format(rel))
# The relaxation is a DIFFERENT number, 66.66667 against 66.67, which is what makes this
# check meaningful: a solver that quietly dropped integrality would land on the relaxation
# and this line would catch it rather than agreeing with it.
print("    The two agree, and they are not the relaxation."
      if rel <= 1e-6 else "    THEY DISAGREE - see demo/miqp_blend.mps.")
sys.exit(0 if rel <= 1e-6 else 1)
PYMIQP

# ===========================================================================================
rule "1. Sovereignty: not built on an existing solver"
# ===========================================================================================
ask "'It shall not be built upon any existing open source solver library but shall be built
              from scratch from mathematical foundation.'"

echo "Everything the binary links against, right now:"
if command -v ldd >/dev/null 2>&1; then
  ldd "$BIN" | sed 's/^/    /'
elif command -v objdump >/dev/null 2>&1; then
  objdump -p "$BIN" | grep "DLL Name" | sed 's/^/    /'
fi
cat <<'PROV'

CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP, OR-Tools, Gurobi, CPLEX and
Xpress appear nowhere in that list - not their simplex, not their cuts, not their MPS
reader. The simplex, the LU factorization, the branch and bound, the MPS parser and the
first-order method are all ours. docs/PROVENANCE.md carries the dependency table, the
algorithm-to-citation table and the full CMake link line; CI regenerates an SPDX SBOM.
PROV

# ===========================================================================================
rule "2. The LP core, on a recognised benchmark library"
# ===========================================================================================
ask "'The solver should successfully solve standard benchmark problems from recognised
              optimization libraries such as MIPLIB, Netlib or Mittelmann benchmark sets.'"

echo "Solving the committed Netlib set live, and checking every answer against the optimum"
echo "published by netlib.org - which bench/runners/fetch_data.py parses from their readme."
echo
"$PYTHON" bench/runners/netlib.py --binary "$BIN" --time-limit 60 \
  --out "$WORK/netlib.csv" 2>&1 | tail -14

# ===========================================================================================
rule "2.5. Scale: an instance orders of magnitude larger than anything committed"
# ===========================================================================================
cat <<'SCALE'
    Section 6 below is honest that the committed Netlib set is small and settles nothing
    about the "thousands to millions of variables"; docs/BENCHMARKS.md sections 1f-1f.3 are where that
    is measured, on generated instances only, and this run does not repeat them.
    This does not close that gap - one instance is not a benchmark suite - but it is a live
    demonstration at a size none of the case studies above reach, on an LP whose optimum is
    known before the solver ever sees the file (bench/runners/generate_large_lp.py builds the
    instance BACKWARDS from a chosen primal-dual pair that already satisfies the KKT
    conditions, the same construction tests/oracles/lp_generator.cpp uses for the fuzz gate),
    solved by the GPU-story engine (restarted PDHG, CPU path) with --progress-out streaming
    live iteration progress to a file, exactly as an operator watching a long solve would use.
SCALE
echo
LARGE="$WORK/large.mps"
GEN_OUT="$("$PYTHON" bench/runners/generate_large_lp.py --rows 5000 --cols 5000 \
  --nnz-per-col 5 --seed 42 --out "$LARGE")"
echo "$GEN_OUT" | sed 's/^/    /'
ANALYTIC="$(echo "$GEN_OUT" | sed -n 's/.*analytic optimum: //p')"

echo
"$BIN" solve "$LARGE" --option log_to_console=false --option algorithm=pdhg \
  --progress-out "$WORK/large_progress.jsonl" --stats "$WORK/large.json" >/dev/null || true
echo "    SANKHYA: status $(field large result status), objective $(field large result objective),"
echo "             $(field large effort iterations) PDHG iterations, $(field large effort solve_seconds)s"
echo
echo "    --progress-out, tail of $WORK/large_progress.jsonl (one line per residual"
echo "    evaluation, every 40 iterations, flushed after every write - what 'tail -f' would"
echo "    show live during the solve above):"
echo
tail -3 "$WORK/large_progress.jsonl" | sed 's/^/        /'
echo
"$PYTHON" - "$WORK/large.json" "$ANALYTIC" <<'PYSCALE'
import json, sys
obj = json.load(open(sys.argv[1]))["result"]["objective"]
analytic = float(sys.argv[2])
rel = abs(obj - analytic) / max(1.0, abs(analytic))
print("    analytic optimum (exact by construction, not measured): {}".format(sys.argv[2]))
print("    relative difference:                                    {:.3e}".format(rel))
# 1e-4 is PDHG's OWN default relative tolerance (kPdhgLoose, tolerances.hpp) - the bar this
# check applies is the one the algorithm targets, not a tighter one picked after the fact.
if rel <= 1e-4:
    print("    Within PDHG's own default tolerance. The two agree.")
else:
    print("    OUTSIDE PDHG's own default tolerance. That is a bug in our solver.")
sys.exit(0 if rel <= 1e-4 else 1)
PYSCALE

# ===========================================================================================
rule "3. The industrial scope PS26119 names"
# ===========================================================================================
ask "'refinery scheduling, crude blending, process optimization, production planning,
              logistics, power system dispatch, transportation and supply chain management.'"

printf '    %-22s %-6s %6s %6s  %-10s %16s\n' \
  "case study" "class" "rows" "cols" "status" "objective"
printf '    %s\n' "$(printf -- '-%.0s' $(seq 1 76))"

for entry in \
  "crude_blend:demo/crude_blend.mps:crude blending" \
  "crude_blend_qp:demo/crude_blend_qp.mps:blending, price impact" \
  "blend_milp:demo/blend_milp.mps:refinery scheduling" \
  "power_dispatch:$CASES/power_dispatch.mps:power dispatch" \
  "supply_chain:$CASES/supply_chain.mps:supply chain" \
  "lot_sizing:$CASES/lot_sizing.mps:production planning" ; do
  tag="${entry%%:*}"; rest="${entry#*:}"; path="${rest%%:*}"; label="${rest#*:}"
  solve_case "$tag" "$path"
  ints=$(field "$tag" model integer_columns)
  # Read the class off the ENGINE the dispatcher picked rather than guessing here.
  # solve() classifies from the model itself, so this cannot drift from what ran.
  algo=$(field "$tag" result algorithm)
  klass="LP"; [ "$ints" != "0" ] && klass="MILP"
  case "$algo" in qp-*) klass="QP" ;; esac
  printf '    %-22s %-6s %6s %6s  %-10s %16.6f\n' \
    "$label" "$klass" "$(field "$tag" model rows)" "$(field "$tag" model columns)" \
    "$(field "$tag" result status)" "$(field "$tag" result objective)"
done

echo
echo "Each of those was then handed to tools/verify_solution.py, which has its OWN MPS reader"
echo "and links no part of our C++. It re-derives the row activities, the objective, the"
echo "reduced costs and the duality gap independently:"
echo
for entry in \
  "crude_blend:demo/crude_blend.mps" \
  "crude_blend_qp:demo/crude_blend_qp.mps" \
  "blend_milp:demo/blend_milp.mps" \
  "power_dispatch:$CASES/power_dispatch.mps" \
  "supply_chain:$CASES/supply_chain.mps" \
  "lot_sizing:$CASES/lot_sizing.mps" ; do
  tag="${entry%%:*}"; path="${entry#*:}"
  printf '    %-34s %s\n' "$(basename "$path")" \
    "$("$PYTHON" tools/verify_solution.py "$path" "$WORK/$tag.sol" --quiet 2>&1 | tail -1)"
done

# -------------------------------------------------------------------------------------------
echo
echo "--- The QP we refuse: demo/crude_blend_discount_qp.mps ------------------------------"
echo
cat <<"DISCOUNT"
    The price-impact QP above is convex because every price RISES with the volume lifted.
    Change one word of the story - Bonny Light comes with a volume DISCOUNT, its price
    falling as more is taken - and the same file with one sign flipped in QUADOBJ is no
    longer convex: the margin curves upward along BN, the objective has local optima at
    corners of the feasible set, and an iteration that stops at one of them would print it
    as the optimum. Nothing in the printed answer distinguishes that point from the global
    one. Stated position: SANKHYA does not report a local optimum as a global one. The
    Hessian is factorized (LDL^T, src/qp/convexity.cpp) before any iteration runs, and a
    negative pivot is a certificate of non-convexity, not a warning to be overridden:
DISCOUNT
echo
rc=0
"$BIN" solve demo/crude_blend_discount_qp.mps --option log_to_console=false \
  --write-sol "$WORK/discount.sol" >"$WORK/discount.out" 2>&1 || rc=$?
echo "    status   $(sed -n "s/^status  *//p" "$WORK/discount.out")   (exit code $rc)"
echo "    message"
sed -n "s/^message  *//p" "$WORK/discount.out" | fold -s -w 84 | sed "s/^/             /"
echo
echo "    The .sol file the run wrote says the same, so nothing downstream can mistake it:"
grep -E "^(status|certificate) " "$WORK/discount.sol" | sed "s/^/        /"
if [ "$rc" -ne 5 ] || ! grep -q "^status  *model_error" "$WORK/discount.out"; then
  echo "    THE REFUSAL DID NOT HAPPEN - a non-convex QP was solved. See src/qp/convexity.cpp."
  exit 1
fi
echo
cat <<"POOLING"
    The pooling problem itself - Haverly (1978), demo/pooling_haverly.mps, where the pool
    quality is a variable and the sulphur it carries into a product is quality TIMES flow -
    is bilinear in its CONSTRAINTS. By default the solver does not read that class at all:
    its QCMATRIX sections are refused at read time, by name, rather than read with the
    bilinear terms dropped, which would solve a linear model and report it as this one:
POOLING
echo
rc=0
"$BIN" solve demo/pooling_haverly.mps --option log_to_console=false >"$WORK/pooling.out" 2>&1 || rc=$?
sed "s|$REPO/||" "$WORK/pooling.out" | fold -s -w 84 | sed "s/^/        /"
echo "        (exit code $rc)"
if [ "$rc" -ne 3 ] || ! grep -q "QCMATRIX" "$WORK/pooling.out"; then
  echo "    THE REFUSAL DID NOT HAPPEN - a pooling model was read. See src/io/mps_reader.cpp."
  exit 1
fi
echo
cat <<"GLOBAL"
    Asked for by name, --option nonconvex=global (#514), the same file is read with its
    quadratic rows and solved by spatial branch and bound (src/global/): McCormick envelopes
    of every product over the current box, bound tightening at every node, a node bound from
    the LP multipliers by weak duality, and an incumbent accepted only once it is checked
    against the ORIGINAL bilinear rows. "optimal" means the proven bound met the incumbent,
    the test a MILP answer is held to. Recursion from a poor start stops at a local optimum,
    which is the trap Haverly's paper is about; this is checked against the global one:
GLOBAL
echo
solve_case pooling_global demo/pooling_haverly.mps --option nonconvex=global
echo "    SANKHYA: status $(field pooling_global result status), objective $(field pooling_global result objective), bound $(field pooling_global result dual_bound), $(field pooling_global effort nodes) nodes"
echo
echo "    Checked independently by tools/verify_solution.py, which reads the QCMATRIX rows"
echo "    itself and adds each product to its row's activity - no code shared with the solver:"
echo
"$PYTHON" tools/verify_solution.py demo/pooling_haverly.mps "$WORK/pooling_global.sol" --quiet | sed 's/^/        /'
echo
"$PYTHON" - "$WORK/pooling_global.json" data/pooling/reference.json <<'PYPOOL'
import json, sys
obj = json.load(open(sys.argv[1]))["result"]["objective"]
ref = json.load(open(sys.argv[2]))["instances"]["haverly1"]
# The reference file states the optimum as a MINIMUM COST (-400, MINLPLib's pooling_haverly1pq);
# this demo file maximizes profit, so the same optimum is +400. Read, not typed.
published = -ref["reference_objective"]
rel = abs(obj - published) / max(1.0, abs(published))
source = ref["paper"].split(",")[0] + " 1978, via data/pooling/reference.json"
print("    published global optimum:        {:g}  ({})".format(published, source))
print("    relative difference:             {:.3e}".format(rel))
print("    The two agree." if rel <= 1e-6 else "    THEY DISAGREE - see demo/pooling_haverly.mps.")
sys.exit(0 if rel <= 1e-6 else 1)
PYPOOL

# -------------------------------------------------------------------------------------------
echo
echo "--- What a planner actually reads: the shadow prices on the blend ------------------"
echo
echo "The dual value on a constraint is the marginal worth of relaxing it by one unit. It is"
echo "the output a refinery planner acts on, and it is why an LP is worth solving exactly."
echo
sed -n '/begin rows/,/end rows/p' "$WORK/crude_blend.sol" | grep -v '^begin\|^end' | \
  awk '{printf "    %-12s activity %14.6f    shadow price %14.6f\n", $1, $2, $3}'

# -------------------------------------------------------------------------------------------
echo
echo "--- Sensitivity ranging: how far each price and capacity can move (#220) -----------"
echo
cat <<'RANGING_INTRO'
The LP already gave the optimal blend and the shadow prices. It also answers "how sure is it":
for each crude's margin, how far it can fall and how far it can rise before the blend changes;
for each constraint, how far its bound can move before the plan changes. The planner reads the
ten most sensitive of each before anything else.

RANGING_INTRO
solve_case "crude_blend_ranging" "demo/crude_blend.mps" --ranging
echo "Ten most sensitive objective (cost) ranges:"
printf "    %-30s %14s %14s\n" "crude / column" "allow_decrease" "allow_increase"
sed -n '/begin ranging_columns/,/end ranging_columns/p' "$WORK/crude_blend_ranging.sol" \
  | grep -v '^begin\|^end' \
  | awk '{lo = ($2 == "inf") ? 1e300 : $2 + 0; hi = ($3 == "inf") ? 1e300 : $3 + 0;
          print (lo < hi ? lo : hi), $1, $2, $3}' \
  | sort -g \
  | head -10 \
  | awk '{printf "    %-30s %14s %14s\n", $2, ($3 == "inf" ? "inf" : sprintf("%.6g", $3)),
                 ($4 == "inf" ? "inf" : sprintf("%.6g", $4))}'
echo
echo "Ten most sensitive RHS (capacity) ranges:"
printf "    %-30s %14s %14s\n" "row / constraint" "allow_decrease" "allow_increase"
sed -n '/begin ranging_rows/,/end ranging_rows/p' "$WORK/crude_blend_ranging.sol" \
  | grep -v '^begin\|^end' \
  | awk '{lo = ($2 == "inf") ? 1e300 : $2 + 0; hi = ($3 == "inf") ? 1e300 : $3 + 0;
          print (lo < hi ? lo : hi), $1, $2, $3}' \
  | sort -g \
  | head -10 \
  | awk '{printf "    %-30s %14s %14s\n", $2, ($3 == "inf" ? "inf" : sprintf("%.6g", $3)),
                 ($4 == "inf" ? "inf" : sprintf("%.6g", $4))}'
# "Most sensitive" above is the smaller of the two sides, with inf sorted last.
echo
echo "    The ranges are in the model's own sense (this blend maximizes margin), and they are"
echo "    those of the reported basis: $(grep '^ranging_basis' "$WORK/crude_blend_ranging.sol" | cut -d' ' -f2)."
echo "    Checked by the definition when this was reviewed: a price moved 0.9x of its reported"
echo "    side keeps the blend, moved 1.1x of it changes the blend."

# -------------------------------------------------------------------------------------------
echo
echo "--- The second-best plan: a solution pool on production planning (#225) -----------"
echo
cat <<'POOL_INTRO'
A model never knows everything. If the set-up the optimal plan needs cannot happen - the line
is down that month - the planner wants the next plan, already costed, not a re-run. The branch
and bound keeps the integer plans it finds; with pool_complete it keeps searching until the
pool provably holds the best ones. Top three plans for the lot-sizing model, and what each
changes against the best:

POOL_INTRO
solve_case "lot_sizing_pool" "$CASES/lot_sizing.mps" \
  --option pool_complete=true --option pool_size=3 --option pool_write_all_columns=true
"$PYTHON" - "$WORK/lot_sizing_pool.sol" <<'POOL'
import sys
plans, block = [], False
for line in open(sys.argv[1]):
    fields = line.split()
    if fields[:2] == ["begin", "pool"]:
        block = True
    elif fields[:2] == ["end", "pool"]:
        block = False
    elif block and len(fields) == 3 and fields[0] == "solution":
        plans.append((float(fields[2]), {}))
    elif block and len(fields) == 2 and plans:
        plans[-1][1][fields[0]] = round(float(fields[1]), 6)
best = plans[0][1]
for rank, (cost, plan) in enumerate(plans, start=1):
    # Y1..Y4 are this model's set-up binaries; X (production) and S (stock) follow from them.
    setups = " ".join(n for n, v in plan.items() if n.startswith("Y") and v) or "(none)"
    changed = [f"{name} {best[name]:g}->{value:g}" for name, value in plan.items()
               if value != best[name]]
    print(f"    plan {rank}: cost {cost:10.2f}   set up in {setups:<24}"
          + ("  (the optimum)" if rank == 1 else f"  +{cost - plans[0][0]:.2f}; changes " +
             ", ".join(changed)))
POOL
echo
echo "    Independently checked (every plan integral, within bounds, every row, ordered, distinct):"
echo "    $("$PYTHON" tools/verify_solution.py "$CASES/lot_sizing.mps" \
        "$WORK/lot_sizing_pool.sol" --quiet 2>&1 | tail -1)"

# -------------------------------------------------------------------------------------------
echo
echo "--- When the plan cannot be met: which lines of the model fight each other ---------"
echo
echo "The same blend with the diesel commitment raised from 40 to 60 kbbl/day: no plan exists."
echo "'Infeasible' alone sends a planner hunting through the model. The solver names an"
echo "irreducible infeasible subsystem (#217) - a set of constraints that cannot hold together"
echo "and from which no single one can be dropped: here the throughput window, the commitment"
echo "and the caps on the two high-yield crudes. Lift any one of them and a plan exists. The"
echo "solution file carries a proof of both halves of that claim, and the sulphur limit is not"
echo "named because it plays no part:"
echo
# The CLI exits non-zero for an infeasible model - that is its contract, and under
# `set -e` it would end the demo here - so the status is captured and checked, not ignored.
set +e
infeasible_report="$("$BIN" solve demo/crude_blend_infeasible.mps --option log_to_console=false \
  --write-sol "$WORK/crude_blend_infeasible.sol" 2>&1)"
infeasible_rc=$?
set -e
printf '%s\n' "$infeasible_report" | grep -E '^(status|IIS) ' | sed 's/^/    /'
if [ "$infeasible_rc" -eq 0 ]; then
  echo "    the over-constrained blend was reported solvable; that is a bug" >&2
  exit 1
fi
echo
echo "    tools/verify_solution.py checks the certificate and the IIS with its own arithmetic:"
"$PYTHON" tools/verify_solution.py demo/crude_blend_infeasible.mps \
  "$WORK/crude_blend_infeasible.sol" 2>&1 | grep -E 'IIS|VERIFIED|REJECTED' | sed 's/^/    /'

# -------------------------------------------------------------------------------------------
echo
echo "--- A MILP answer corroborated by exhaustion, not by itself ------------------------"
echo
cat <<'ORACLE'
verify_solution.py can confirm a MILP answer is feasible, integral and that the bound closed.
It CANNOT confirm no better answer exists - that is the whole content of the search, and a
branch and bound which wrongly fathoms a node returns a point that passes every such check.
The dispatch instance is small enough to settle by exhaustion, so we settle it:

ORACLE
ORACLE_OUT="$("$PYTHON" "$CASES/dispatch_oracle.py")"
echo "$ORACLE_OUT" | sed 's/^/    /'
# The oracle's own printed total is the reference, parsed from its output rather than typed
# here: a literal would have to be kept in step with the instance by hand.
ORACLE_COST="$(echo "$ORACLE_OUT" | sed -n 's/.*total cost *//p' | tail -1)"
echo
echo "    SANKHYA returned:  $(field power_dispatch result objective)  in $(field power_dispatch effort nodes) nodes"
"$PYTHON" - "$WORK/power_dispatch.json" "$ORACLE_COST" <<'PYCHK'
import json, sys
got = json.load(open(sys.argv[1]))["result"]["objective"]
want = float(sys.argv[2])
ok = abs(got - want) <= 1e-9 * max(1.0, abs(want))
print("    " + ("AGREES with the exhaustive oracle ({:.6f}) to 1e-9.".format(want)
                if ok else "DISAGREES with the oracle ({:.6f}) - got {!r}".format(want, got)))
sys.exit(0 if ok else 1)
PYCHK

# -------------------------------------------------------------------------------------------
echo
echo "--- A MILP optimum with a proof a separate program checks in exact arithmetic ------"
echo
cat <<'VIPR'
Exhaustion settles a model this small; a real one needs a proof. With write_certificate
(#518) the branch and bound writes its whole tree as a VIPR certificate: every leaf's bound
derived from its LP duals as a safe bound (#519), every branching step, and the rounding
that closes the gap. tools/verify_certificate.py re-reads the MPS file itself and checks
every derivation in exact rational arithmetic. Root cuts are switched off here because the
writer does not yet derive cut rows, and the point is checked to 1e-9 because its values are
decimals of doubles: the BOUND is checked exactly.

VIPR
solve_case lot_cert "$CASES/lot_sizing.mps" --option enable_root_cuts=false \
  --option write_certificate="$WORK/lot_sizing.vipr"
echo "    SANKHYA: status $(field lot_cert result status), objective $(field lot_cert result objective), $(field lot_cert effort nodes) nodes"
# The checker's verdict is captured and TESTED, not only printed: a certificate that fails
# must stop the demo, and under `set -e` a failing checker would stop it with no word why.
CERT_OUT="$("$PYTHON" tools/verify_certificate.py "$WORK/lot_sizing.vipr" \
  --mps "$CASES/lot_sizing.mps" --feas-tol 1e-9 2>&1)" || true
echo "$CERT_OUT" | grep -E "^(derivations checked|proved bound|solution objective|VERIFIED)" \
  | sed 's/^/    /' || true
if ! echo "$CERT_OUT" | grep -q "^VERIFIED"; then
  echo "$CERT_OUT" | sed 's/^/    /'
  echo "    THE CERTIFICATE WAS NOT VERIFIED. See src/mip/certificate_writer.cpp."
  exit 1
fi

# ===========================================================================================
rule "4. Numerical robustness, on the three hazards PS26119 names"
# ===========================================================================================
ask "'...challenging large-scale optimization problems involving degeneracy, weak LP
              relaxations or ill-conditioned constraint matrices, where simpler
              implementations struggle to achieve reliable convergence.'"

# --- degeneracy ---------------------------------------------------------------------------
echo "(a) DEGENERACY - $CASES/supply_chain.mps"
echo
cat <<'DEGEN'
    Balanced transportation: total supply equals total demand, so the seven equality rows
    are linearly DEPENDENT and the constraint matrix has rank 6, not 7. A basis needs
    m + n - 1 = 6 basic variables against 7 rows, so a basic variable sits at zero at every
    iteration and ratio-test ties are the norm. This is the classical cycling test bed - the
    structure that makes a naive simplex stall forever on a sequence of zero-length steps.
DEGEN
echo
echo "    Measured, not asserted. The matrix is read by verify_solution.py's INDEPENDENT MPS"
echo "    reader, and the rank is computed EXACTLY by elimination over the rationals - no"
echo "    singular-value threshold to pick, and no SANKHYA code involved:"
echo
PYTHONWARNINGS=ignore "$PYTHON" "$CASES/matrix_stats.py" "$CASES/supply_chain.mps" --rank | sed 's/^/        /'
echo
echo "    SANKHYA: status $(field supply_chain result status), objective $(field supply_chain result objective), $(field supply_chain effort iterations) iterations, no stall."
echo "    Anti-cycling is Bland's rule, cited in src/simplex/primal_simplex.cpp."

# --- ill conditioning ----------------------------------------------------------------------
echo
echo "(b) ILL-CONDITIONING - $CASES/ill_conditioned.mps"
echo
cat <<'ILLC'
    The SAME model, with rows scaled by 10^-6 ... 10^+6 and columns re-parameterised by
    10^-6 ... 10^+4. Both are exact changes of variable, so the optimal objective MUST be
    unchanged. That makes the robustness claim falsifiable instead of decorative: we know
    the right answer, so a solver that drifts here cannot hide it.
ILLC
solve_case ill_conditioned "$CASES/ill_conditioned.mps"
echo
echo "    what the scaling did to the matrix, measured the same independent way:"
echo
PYTHONWARNINGS=ignore "$PYTHON" "$CASES/matrix_stats.py" "$CASES/ill_conditioned.mps" --conditioning | sed 's/^/        /'
echo
printf '    %-28s %s\n' "well-scaled supply_chain:" "$(field supply_chain result objective)"
printf '    %-28s %s\n' "ill-conditioned re-scaling:" "$(field ill_conditioned result objective)"
"$PYTHON" - "$WORK/supply_chain.json" "$WORK/ill_conditioned.json" <<'PYAGREE'
import json, sys
a = json.load(open(sys.argv[1]))["result"]["objective"]
b = json.load(open(sys.argv[2]))["result"]["objective"]
rel = abs(a - b) / max(1.0, abs(a))
print("    relative difference:         {:.3e}".format(rel))
if rel <= 1e-9:
    print("    The two agree. The scaling did not move the answer.")
else:
    print("    THEY DISAGREE. That is a bug in our solver, and this demo just found it.")
sys.exit(0 if rel <= 1e-9 else 1)
PYAGREE

# --- weak relaxation -----------------------------------------------------------------------
echo
echo "(c) WEAK LP RELAXATION - $CASES/lot_sizing.mps"
echo
cat <<'WEAK'
    Lot sizing with a big-M set-up link, x_t <= M y_t. In the relaxation y_t is free to take
    x_t / M, so a period producing one unit pays 1/180 of a set-up rather than a whole one.
    The bound therefore sits far below the integer optimum and branch and bound has to close
    the difference by search. This is the standard weak-formulation failure mode.
WEAK
solve_case lot_relaxed "$CASES/lot_sizing_relaxed.mps"
echo
printf '    %-34s %s\n' "LP relaxation bound:" "$(field lot_relaxed result objective)"
printf '    %-34s %s\n' "proven integer optimum:" "$(field lot_sizing result objective)"
printf '    %-34s %s\n' "nodes explored to close it:" "$(field lot_sizing effort nodes)"
"$PYTHON" - "$WORK/lot_relaxed.json" "$WORK/lot_sizing.json" <<'PYGAP'
import json, sys
lp = json.load(open(sys.argv[1]))["result"]["objective"]
ip = json.load(open(sys.argv[2]))["result"]["objective"]
print("    {:<34} {:.2f}%".format("integrality gap the search closed:",
                                  100.0 * (ip - lp) / abs(ip)))
PYGAP

# ===========================================================================================
rule "5. Compared against an established solver"
# ===========================================================================================
ask "'...with solution quality and computational performance compared against at least one
              established commercial or open-source solver.'"

if [ "$QUICK" = "1" ]; then
  echo "Skipped (--quick). Run without --quick to execute the comparison live."
elif "$PYTHON" -c "import highspy" >/dev/null 2>&1; then
  echo "HiGHS is the reference. It is invoked as a SEPARATE PROCESS over the same MPS files;"
  echo "no HiGHS code is linked into, or read by, SANKHYA. Both sides are timed on solver"
  echo "time only, so neither is charged for interpreter start-up."
  echo
  # `tail -n +3`, NOT `tail -18`. The intent is to drop compare.py's first two lines -
  # the local path to our binary and the HiGHS backend string - because the paragraph
  # above already says what they say, and a demo should not print someone's home
  # directory. A fixed tail expresses that as "keep the last 18", which silently
  # depends on the instance count: at eight instances the output was 20 lines and 18
  # was right; adding a ninth made it 21, so the header's second line was cut and its
  # continuation was left dangling under nothing. Counting from the FRONT does not
  # care how many instances run.
  "$PYTHON" bench/runners/compare.py --sankhya-binary "$BIN" --time-limit 60 \
    --out "$WORK/compare.csv" 2>&1 | tail -n +3
else
  echo "highspy is not importable, so the comparison cannot run here."
  echo "    pip install highspy    then re-run this script."
fi

# ===========================================================================================
# Read the medium-tier result out of the newest committed CSV. Falls back to naming the
# reproduction command if none is present, rather than printing a number from nowhere.
MEDIUM_SUMMARY="$("$PYTHON" bench/runners/latest_result.py "netlib-medium-*.csv" --summary)"

# MIPLIB, read from its committed CSV for the same reason. --status-counts rather than
# --summary because the MIPLIB runner writes no `passed` column: there is no published
# optimum to compare against for most of the set, so the status IS the result.
MIPLIB_SUMMARY="$("$PYTHON" bench/runners/latest_result.py "miplib-*.csv" --status-counts)"

# The FULL tier, same treatment. This is the figure section 6's own text calls the one the
# Phase 6 exit criterion is measured against, so it belongs in the demo rather than in a CSV
# nobody opens - and it is the LOWER number of the two, because `medium` is defined by a row
# cap and is therefore the easier half by construction. Reporting the easier half as the
# headline is the kind of thing this section exists to not do.
FULL_SUMMARY="$("$PYTHON" bench/runners/latest_result.py "netlib-full-*.csv" --summary)"

# And the full tier's failures, BY NAME AND REASON, from the same CSV. This paragraph was
# hand-written ("15 numerical_error, 5 wrong objective, 3 time limit, 1 feasible only") and
# described a run three solver generations old by the time anyone reread it. Same reasoning
# as FULL_SUMMARY: read, not typed.
FULL_FAILURES="$("$PYTHON" bench/runners/latest_result.py "netlib-full-*.csv" --failures)"
FULL_EXTREMES="$("$PYTHON" bench/runners/latest_result.py "netlib-full-*.csv" --extremes)"

# Everything else section 6 states is READ as well, each figure from the committed CSV
# named beside it, by a script, when this runs: Kennington, Netlib's infeasible set,
# Maros-Meszaros, the engine race, both GPU cards, the cut A/B and Mittelmann. A CSV that
# is missing ends the demo here rather than letting a sentence print without its number.
EVIDENCE="$WORK/evidence.txt"
"$PYTHON" demo/evidence.py bench/results >"$EVIDENCE"

# The instance count in section 6 is READ, not typed. It said "eight" until someone
# fetched a ninth instance, at which point the closing paragraph contradicted the table
# printed directly above it. Same reasoning as MEDIUM_SUMMARY.
NETLIB_COUNT="$("$PYTHON" -c "import csv,sys;print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=''))))" "$WORK/netlib.csv" 2>/dev/null || echo "the committed")"

rule "6. What PS26119 asks for that we do NOT yet have"
# THE MEDIUM-TIER FIGURE BELOW IS READ FROM THE COMMITTED CSV, not typed here. The demo does
# not run that tier - fetching 50 instances takes minutes - so it was hand-written, and it
# went stale three separate times in two days: 26 after #49 made it 37, then 37 after #86
# made it 40. Each time it UNDERSTATED the solver, which is the safe direction and still
# wrong. Deriving it from bench/results/netlib-medium-*.csv removes the failure mode rather
# than asking the next person to remember. Same reasoning as #53.
# ===========================================================================================
# The `g` flags matter: @NCOUNT@ appears twice on one line ("not the 9/9 above"), and
# without them sed substitutes only the first occurrence per line.
# Every KEY=value line of $EVIDENCE becomes one more @KEY@ substitution (demo/evidence.py
# guarantees no value carries a | or an & that sed would read as syntax).
sed 's/^\([A-Z0-9_]*\)=\(.*\)$/s|@\1@|\2|g/' "$EVIDENCE" >"$WORK/evidence.sed"
fill_gaps() {
  sed -e "s|@FULL@|${FULL_SUMMARY}|g" -e "s|@MEDIUM@|${MEDIUM_SUMMARY}|g" \
      -e "s|@NCOUNT@|${NETLIB_COUNT}|g" -e "s|@MIPLIB@|${MIPLIB_SUMMARY}|g" \
      -e "s|@FAILURES@|${FULL_FAILURES}|g" -f "$WORK/evidence.sed"
}
cat <<'GAPS' | fill_gaps
    Stating these is the point. A solver that is vague about its limits is not one an
    industrial user can plan around. Every figure below is read from the committed CSV
    named beside it when this script runs; none is typed here.

    MIQP                Implemented: branch and bound with the convex QP engine as the node
                        solver. The caveat is the BOUND, and it is worth stating because it
                        is what limits the class. A simplex node bound is a vertex objective,
                        exact to rounding; a first-order QP node bound is only accurate to
                        the tolerance it converged to. Since branch and bound PRUNES on that
                        bound, and an optimistic bound can fathom the subtree holding the
                        optimum, the node tolerance is tightened to 1e-10 and the pruning
                        margin widened by the same amount rather than pruning on the
                        optimistic side. That is the safe direction and it costs nodes.
                        Expect MIQP to show the same weakness the MIPLIB line below reports:
                        incumbents found, optimality proved on fewer.
    Non-convex QP       REFUSED BY DEFAULT, deliberately. src/qp/convexity.cpp decides
                        semidefiniteness of sense * Q by LDL^T before any arithmetic starts,
                        and returns a negative pivot as a certificate; a file with bilinear
                        rows is refused at read time. A local optimum reported as a global
                        one is the failure mode we will not ship. The GLOBAL method exists
                        behind --option nonconvex=global (#514): spatial branch and bound
                        over McCormick relaxations for bilinear and quadratic rows, which
                        section 3 runs on Haverly's pooling problem. It is a first slice: it
                        refuses integer columns and any column left unbounded in a product,
                        and on the harder pooling formulations it stops with a gap rather
                        than claim an optimum. No committed CSV measures it yet.
    Interior point      LP: implemented and OPT-IN (--option algorithm=ipm, issue #56):
                        Mehrotra predictor-corrector over a from-scratch sparse LDL^T, with
                        crossover to a vertex. On the full Netlib set it verifies fewer
                        instances than the dual simplex, so the default continuous engine
                        stays the dual simplex (exact, gives a basis); restarted PDHG is the
                        first-order one.
                        QP: a proximal interior point on the quasi-definite augmented system
                        (#614, --option qp_algorithm=ipm) is on main and OFF by default. It
                        has no committed CSV. The Maros-Meszaros CSV that exists measures the
                        default first-order QP engine, and it is weak:
                            @MM@
                            (@MM_CSV@)
    Engine race         --option engine_race=true (#563) runs the dual simplex, the interior
                        point and PDHG at once and keeps the first answer that passes an
                        independent check. On Mittelmann's smallest LPs, solved and
                        verified:
                            @RACE_MM@
                            (@RACE_MM_CSV@)
                        On the full Netlib set:
                            @RACE_NETLIB@
                            (@RACE_NETLIB_CSV@)
                        Off by default: where the dual simplex wins anyway, the other two
                        engines only cost time.
    Cutting planes      ON BY DEFAULT AT THE ROOT since 0018254: Gomory mixed-integer, lifted
                        knapsack cover, MIR, clique and {0,1/2}-Chvatal-Gomory cuts, each
                        validity-gated against the exact rational oracle, with cut selection
                        by efficacy and parallelism (#415). The A/B that decided it, at 60 s
                        on the 30 MIPLIB instances:
                            @CUTS_OFF@
                            @CUTS_ON@
                            @CUTS_TREE@
                            (@CUTS_AB_CSV@)
                        OFF BY DEFAULT, each for a reason, each until its A/B on main: c-MIR
                        (#625), flow cover (#419) and implied-bound cuts (#630), which no A/B
                        has measured yet; cut rounds below the root (tree_cut_depth), because
                        the tree leg above lost the proof the root round gained; and the root
                        separation loop (#621), which repeats the root round until the bound
                        stalls, because every row it adds is paid for at every node LP and
                        no A/B on main has weighed the two yet.
    GPU acceleration    MEASURED, on two cards, PDHG alone, CPU time over GPU time on
                        synthetic LPs with a known optimum, by size (rows):
                            @GPU_LAPTOP_CARD@ (@GPU_LAPTOP_CSV@)
                                1e-4: @GPU_LAPTOP_4@
                                1e-8: @GPU_LAPTOP_8@
                            @GPU_L4_CARD@ (@GPU_L4_CSV@)
                                1e-4: @GPU_L4_4@
                                1e-8: @GPU_L4_8@
                        The GPU loses below the crossover, where launch and transfer costs
                        are not amortised, and wins above it. On real LPs on the L4, CPU
                        seconds against GPU seconds (@GPU_REAL_CSV@):
                            @GPU_REAL_1@
                            @GPU_REAL_2@
                            @GPU_REAL_3@
                        Each of those is PDHG alone reaching the requested tolerance, not a
                        certified optimum. What is not measured: the real instances on the
                        laptop card, a comparison with another GPU solver, and a multi-GPU
                        number.
    Scale               Section 2.5 above solves one 5000 x 5000 instance, which is the
                        largest thing here by two orders of magnitude and is checked against
                        an optimum known by construction - but ONE generated instance is a
                        demonstration, not a benchmark, and it says nothing about the sparse
                        industrial structure real models have. Everything else here is small.
                        Nothing in this run supports a claim about the "millions of
                        variables" end of what the problem statement asks for; the measured
                        scale evidence is docs/BENCHMARKS.md sections 1f-1f.3, generated
                        instances with exact optima, and no real industrial model at that size.

                        THE HONEST HEADLINE IS THE FULL NETLIB SET - not the @NCOUNT@ solved
                        live above, and not the medium tier either:

                            full set, 89 instances:    @FULL@
                            medium tier, 50 instances: @MEDIUM@

                        Both counts are instances we solve AND independently verify, so they
                        are answers checked by something that shares no code with the solver,
                        not just runs that exited zero. As a RATE the medium tier looks the
                        better of the two, and that is exactly why it is not the headline:
                        the tier is defined by a 500-row cap, so it is the easier half by
                        construction, and quoting it would be choosing the denominator that
                        flatters us.

                        What the rest are is worth stating by name, read from the same CSV:

                            @FAILURES@

                        "published value differs" is, on this set, mostly Netlib's readme
                        being older than the instance files: on every such instance our
                        objective agrees with HiGHS (bench/runners/cross_check_highs.py), and
                        HiGHS disagrees with the readme by the same amount.

                        On size, read from the same CSV as the counts above:
GAPS
# Several lines, so they are printed between the two halves of the ledger rather than
# substituted into one line of it.
echo "$FULL_EXTREMES" | sed 's/^/                            /'
cat <<'GAPS' | fill_gaps
                        Mittelmann's LP benchmark, its smallest instances:
                            @MITT_SMALL@
                            (@MITT_SMALL_CSV@)
                        and the medium set, which adds the next size class to them:
                            @MITT_MEDIUM@
                            (@MITT_MEDIUM_CSV@)
                        (bench/runners/mittelmann.py) - which is the honest shape of it:
                        correct wherever we finish, and both curves bend well before
                        "millions of variables". That is issue #198. Reproduce with:
                            python bench/runners/fetch_data.py --set full
                            python bench/runners/netlib.py --time-limit 120
    Other LP sets       Kennington's large LPs:
                            @KENN@
                            (@KENN_CSV@)
                        Netlib's infeasible set, where the right answer is a proof of
                        infeasibility, not a point:
                            @INFEAS@
                            (@INFEAS_CSV@)
                        The rest stop on a numerical error or without a certificate the
                        verifier accepts, which is what the count separates.
    MIPLIB              PS26119 names MIPLIB before Netlib, and this demo does not run it.
                        We do have results:
                            @MIPLIB@
                        They are the weakest numbers in the project: branch and bound reaches
                        a feasible incumbent on most of the set but PROVES optimality on few.
                        Root cuts are on (above); what the instances that hold the optimum
                        without proving it need is the tree bound (#221). Stated here rather
                        than left out - a reader who opens bench/results/ finds it either
                        way, and #54 is the tracker.
    MILP extras         On main, each behind its own option and OFF by default, none with a
                        committed CSV yet, so none is claimed as a gain here: the Feasibility
                        Jump primal heuristic (#635, mip_heur_fj), safe dual bounds that hold
                        in exact arithmetic whatever the LP's rounding (#636, safe_bounds),
                        and a MILP optimality certificate in the VIPR format, checked by an
                        independent exact-arithmetic checker (#639, write_certificate),
                        which section 3 runs live on the lot-sizing model.
    MINLP               NOT ATTEMPTED. The global method refuses integer columns, and the
                        convex NLP layer (src/nlp/) has no branch and bound over it. A model
                        with both integers and nonlinear terms has no engine here.
    Parallelism         Single-threaded by default. --option threads=N runs an iteration's
                        column loops under OpenMP, deterministically (bit-identical results at
                        1 and 8 threads), and at Netlib scale it is measured to buy nothing:
                        an iteration is too short to amortize the fork (#57). The branch and
                        bound tree runs on --option mip_threads=N workers (#403), default 1
                        until its A/B on main. A switch that preserves correctness, not a
                        speed claim.

    On speed against HiGHS, section 5 above prints the measured ratio for this run rather
    than repeating a number here that would go stale - and it is a narrow comparison either
    way: @NCOUNT@ small instances settle nothing about large models. HiGHS is a decade of
    specialist work, and its dual simplex is far more mature than ours. Presolve is no
    longer part of that gap: ours is in src/presolve and runs by default (#43).
    The claim we do make is narrower and checkable: on every instance we report as solved,
    the answer matches the published optimum AND survives an independent verifier that
    shares no code with the solver.
GAPS
echo
