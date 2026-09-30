# The finale walk

One refinery MILP, solved, proved, checked and re-planned in front of the jury, with nothing
on the stage machine that can fail for want of a network or a compiler (#758).

    demo/finale.sh --dry      # check the machine first: binary, GPU, Python, the tools
    demo/finale.sh            # the walk          (Windows: demo\finale.cmd)

It runs the same from a checkout with `build/` and from an unpacked release archive (#748),
which keeps the repository's layout. It needs the `sankhya` binary, Python 3.9 or later, and
nothing else: the model is generated on the spot by `bench/case_studies/refinery/generator.py`
from a fixed seed (synthetic data), and every checker is in `tools/`.

## The six steps

Each prints one line of result and the seconds it took. Every number comes from a command the
script runs; the script stops at the first step whose check fails.

| Step | What runs | What the line shows |
|---|---|---|
| 1 solve | `sankhya solve` on the small refinery MILP with `write_certificate`, `--gpu` when the binary reports a device | status, objective, nodes, root bound, CPU or GPU |
| 2 prove | `tools/verify_certificate.py` (exact rational arithmetic, VIPR) and `tools/verify_solution.py` | both verdicts |
| 3 plan | the same plant as an LP with `--ranging`, read by `tools/report.py` | the LP optimum, how many limits bind, the three worth most per unit and how far each price holds |
| 4 repair | one delivery commitment raised tenfold (then 100x, 1000x) until the plant cannot meet it; `tools/verify_solution.py` checks the Farkas certificate; `tools/repair_infeasibility.py` finds the smallest repair | the proof, the repair's size and the limits it moves |
| 5 bundle | `tools/bundle.py` then `tools/replay_bundle.py` on the MILP run | the bundle replays: manifest intact, verifier passes |
| 6 prices | twenty crude price sets through `sankhya scenarios` (#752) | how many verified, the objective's range |

## Timing

On the 7.7 GB Windows laptop (CPU, no GPU), at the commit this document was merged with, the
whole walk took 4.1 s; the CI Release leg runs it on every pull request. Record a stage machine's
own time with `demo/finale.sh` before the finale, not this figure.

## What it does not do yet

- **The GPU leg.** Step 1 passes `--gpu` when the binary reports a device, but the walk has not
  been timed on a card: the GPU instances are stopped. On a CUDA build with a device the solve
  runs the device paths the MILP search already has; the certified GPU tree (#756) and a timed
  GPU run remain.
- **Semi-continuous units (#754)** are not in the generator's MILP yet; step 1 solves the
  ordering-cost MILP (binary purchase decisions with big-M links).
- **Certified shadow prices (#757).** Step 3 reports the simplex's own duals and ranging on
  the LP, not prices re-derived in exact arithmetic.
- **The medium refinery** does not finish its certified search inside five minutes on the
  laptop, so the walk uses the small one.
