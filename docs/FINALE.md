# The finale walk

One refinery MILP, solved, proved, checked and re-planned from yesterday's basis in front of the jury, with nothing
on the stage machine that can fail for want of a network or a compiler (#758).

    demo/finale.sh --dry      # check the machine first: binary, GPU, Python, the tools
    demo/finale.sh            # the walk          (Windows: demo\finale.cmd)

It runs the same from a checkout with `build/` and from an unpacked release archive (#748),
which keeps the repository's layout. It needs the `sankhya` binary, Python 3.9 or later, and
nothing else: the model is generated on the spot by `bench/case_studies/refinery/generator.py`
from a fixed seed (synthetic data), and every checker is in `tools/`.

## The eight steps

Each prints one line of result and the seconds it took. Every number comes from a command the
script runs; the script stops at the first step whose check fails.

| Step | What runs | What the line shows |
|---|---|---|
| 1 solve | `sankhya solve` on the small refinery MILP with `write_certificate`, `--gpu` when the binary reports a device | status, objective, nodes, root bound, CPU or GPU |
| 2 prove | `tools/verify_certificate.py` (exact rational arithmetic, VIPR) and `tools/verify_solution.py` | both verdicts |
| 3 plan | the same plant as an LP with `--ranging`, read by `tools/report.py`; then the eight binding limits with the largest shadow price, each re-solved one unit looser | the LP optimum, how many limits bind and how many are priced at a degenerate vertex, the limits whose next unit is worth most as the re-solves measured it, beside their prices, and on how many the price overstates the next unit |
| 4 repair | one delivery commitment raised tenfold (then 100x, 1000x) until the plant cannot meet it; `tools/verify_solution.py` checks the Farkas certificate; `tools/repair_infeasibility.py` finds the smallest repair | the proof, the repair's size and the limits it moves |
| 5 bundle | `tools/bundle.py --certificate` then `tools/replay_bundle.py` on the MILP run | the bundle replays: manifest intact, verifier passes, the bundled VIPR proof checked by `tools/verify_certificate.py` |
| 6 prices | twenty crude price sets through `sankhya scenarios` (#752) | how many verified, the objective's range |
| 7 replan | today's crude prices and product demands (market and commitment together) moved (`bench/runners/replan_warm_start.py`'s edit), the LP solved cold and with `--warm-start` from step 3's `.sol` (#218), both checked by `tools/verify_solution.py` | the pivot counts side by side, the shared optimum |
| 8 units | the step 1 plant with a minimum run rate on its crude unit (#754): `generator.py --crude-min-run 1/6` makes every crude run semi-continuous, 0 or at least 1/6 of the unit's capacity (an MPS SC bound); solved by the native semi-continuous branching and again with `--option sos_reformulate=true` (binaries and big-M rows), both checked by `tools/verify_solution.py` | the optimum beside step 1's, how many runs are idle, the nodes the native branching took, and that the two routes agree |

## Why step 3 re-solves

A shadow price is the objective's derivative on one side of the limit, and at a degenerate
vertex it can hold for no relaxation at all: its ranging interval is 0 on the side that
relaxes the limit. The small refinery LP is degenerate that way. Its three largest prices
are 29.07 (`SPEC_1_2`), 26.36 (`UNIT_2_2`) and 21.98 (`UNIT_0_3`), and one more unit of each,
re-solved, gains 0, 2.80 and 0. The limit worth most is `UNIT_2_1`: 6.60 for the next unit,
at a price of 9.12 that ranks it seventh. Reading the price alone would send the planner to
the wrong limit, so the walk prints what the re-solves measured and says on how many limits
the price overstates the next unit. Certified two-sided prices are #757.

## Timing

On the 7.7 GB Windows laptop (CPU, no GPU), at the commit this document was merged with, the
whole walk took 4.1 s; the CI Release leg runs it on every pull request. Record a stage machine's
own time with `demo/finale.sh` before the finale, not this figure. With step 8 (#754) added,
one run on the same laptop, with another build running on it, took 9.0 s, step 8 1.37 s of it.

**From a fresh release archive, Windows.** `sankhya-0.1.0-windows-x64-cpu.zip` from the
`v1.0.0-rc1` release, its SHA-256 matching the release's `SHA256SUMS`, unpacked outside any
checkout on a Windows 11 laptop (Intel i7-1355U, 16 GB, CPU only), with nothing built and no
compiler called: `demoinale.cmd --dry` reports the machine ready, and `demoinale.cmd` runs all
six steps in 2.6 s with the archive's own `sankhya.exe` (step 3 as it was before the re-solves
above). `demo/finale.py` imports no networking module, and neither do the two checkers it
calls. The archive's binary reports version 0.1.0 (`893634b`), older than the release's tag:
it was built before the version bump in #822.

## What it does not do yet

- **The GPU leg.** Step 1 passes `--gpu` when the binary reports a device, but the walk has not
  been timed on a card: the GPU instances are stopped. On a CUDA build with a device the solve
  runs the device paths the MILP search already has; the certified GPU tree (#756) and a timed
  GPU run remain.
- **Certified shadow prices (#757).** Step 3 reports the simplex's own duals and ranging on
  the LP, not prices re-derived in exact arithmetic; the one-unit re-solves are floating point.
- **Linux from a release archive.** Only the Windows archive has been run fresh (above).
- **The medium refinery** does not finish its certified search inside five minutes on the
  laptop, so the walk uses the small one.
