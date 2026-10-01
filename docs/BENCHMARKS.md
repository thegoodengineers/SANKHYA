# SANKHYA — benchmarks

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate with: python bench/runners/make_benchmarks_doc.py -->

This file is generated from the CSVs in `bench/results/`, so it cannot drift from the
evidence. Every number below came out of a run that recorded the instance sha256, the git
commit and the machine tag alongside it. CI checks that every CSV's commit is on `main`
(`bench/runners/check_result_stamps.py`, #433); a run made on a branch and carried to `main`
by a squash merge keeps the commit it was produced on and is listed with that squash merge
in `bench/results/squash-stamps.txt`, so the number still ties back to a build.

Times in sections 1a-1c, 1f and 2 are wall-clock, measured around the whole process, so they
include reading the model and writing the outputs. That makes them slightly pessimistic and
honest; it is not the figure to quote for algorithmic speed, and no attempt is made to
dress it up. Section 1d prints solver-internal seconds (its instances take minutes, and the
read is not what is being measured) and section 4 is solver-internal on both sides, as it
says.

Reporting follows Mittelmann's conventions: shifted geometric means with a
1-second shift, an explicit time limit, and failures counted and named
rather than dropped.

---

## 1. Netlib LP — accuracy against published optima

The reference optimum for each instance is parsed by `bench/runners/fetch_data.py` from
Netlib's own `readme`. None of these values was typed from memory.

### 1a. The small set — what the demo runs

Nine instances, committed to the repository so a fresh clone can reproduce this with no
network. **This is the set `demo/run_demo.sh` lets a judge pick from, and it is the easy end
of Netlib.** Its pass rate is not the headline; section 1c is.

Source CSV: `bench/results/netlib-small-2b4eb6b.csv`  
Commit `2b4eb6b` · machine `Windows-AMD64` · generated 2026-09-09T03:41:33+00:00

**9 of 9 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

The same run graded against the **exact** optimum, Koch's rational values (*The final NETLIB-LP results*, Oper. Res. Lett. 32, 2004; `data/netlib/koch_exact.json`): **9 of 9** within a relative 1e-6 **and** verified. Not within it: none.

Coverage: this run used **9 of the 97 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

Every instance in this set passed.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 73 | 0.054 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.024 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 97 | 0.030 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 176 | 0.044 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 88 | 0.023 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.019 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.029 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.034 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.027 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.031s**
- slowest solved instance: 0.054s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

### 1b. The medium tier — instances up to 500 rows

Source CSV: `bench/results/netlib-medium-2b4eb6b.csv`  
Commit `2b4eb6b` · machine `Windows-AMD64` · generated 2026-09-09T03:36:04+00:00

**48 of 50 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

The same run graded against the **exact** optimum, Koch's rational values (*The final NETLIB-LP results*, Oper. Res. Lett. 32, 2004; `data/netlib/koch_exact.json`): **50 of 50** within a relative 1e-6 **and** verified. Not within it: none.

Coverage: this run used **50 of the 97 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

**2 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| disagrees with the published optimum (#75) | 2 | e226, scrs8 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 73 | 0.397 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.040 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 147 | 0.132 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 452 | 0.406 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 115 | 0.165 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 97 | 0.281 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 357 | 0.223 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 133 | 0.205 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 161 | 0.145 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 282 | 0.244 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 236 | 0.112 | yes |
| `d6cube` | 415 | 6184 | optimal | 3.1549166667e+02 | 3.1549166667e+02 | 1.1e-11 | 1013 | 2.035 | yes |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 639 | 0.653 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 507 | 0.911 | yes |
| `etamacro` | 400 | 688 | optimal | -7.5571523312e+02 | -7.5571521774e+02 | 2.0e-08 | 705 | 0.420 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 403 | 0.307 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 56 | 0.292 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.5e-12 | 224 | 2.146 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 309 | 0.669 | yes |
| `grow15` | 300 | 645 | optimal | -1.0687094129e+08 | -1.0687094129e+08 | 3.3e-11 | 3630 | 2.191 | yes |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 3853 | 4.883 | yes |
| `grow7` | 140 | 301 | optimal | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 1932 | 0.785 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 176 | 0.152 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 51 | 0.109 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 243 | 0.107 | yes |
| `pilot4` | 410 | 1000 | optimal | -2.5811392589e+03 | -2.5811392641e+03 | 2.0e-09 | 1060 | 1.036 | yes |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 41 | 0.068 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 88 | 0.112 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 209 | 0.061 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.098 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.107 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 465 | 0.146 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 111 | 0.095 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 460 | 0.174 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 242 | 0.207 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 610 | 0.241 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 155 | 0.179 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000077e+01 | 5.0500000078e+01 | 1.7e-11 | 367 | 0.165 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 1785 | 1.129 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 269 | 0.109 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 136 | 0.206 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.160 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 449 | 0.441 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 300 | 0.336 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 448 | 0.551 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 58 | 0.083 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 200 | 0.100 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.095 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 239 | 0.133 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 406 | 1.046 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.382s**
- slowest solved instance: 4.883s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `e226`, `scrs8`** — kept in the table on purpose

#### 1b.1 Dual pricing: Devex against dual steepest edge (#411)

The dual simplex picks its leaving row by primal infeasibility over a weight. Devex
approximates that weight with a reference framework; dual steepest edge keeps it exact as the
norm of the row of the basis inverse, through the Forrest-Goldfarb update (Math. Programming
57, 1992), at one extra solve per pivot. The question is whether the iterations it saves pay
for that solve.

All at commit `0134c92` on `cloud-container-IntelXeon-2.10GHz-4vCPU-15GB-quiet`, nothing else running.

**Netlib medium tier**, 3 repeats under Devex and 3 under dual steepest edge, alternated, one thread.

| rule | passed (every repeat) | not passed | iterations over the 49 passed by both | shifted geomean seconds (1 s) | total seconds |
|---|---:|---|---:|---:|---:|
| Devex | 49 of 51 | `e226`, `scrs8` | 16,339 | 0.0143 | 0.718 |
| dual steepest edge | 49 of 51 | `e226`, `scrs8` | 13,665 | 0.0108 | 0.536 |

Per instance, dual steepest edge took fewer iterations on 33, more on 13, the same on 3. The eight slowest under Devex:

| instance | Devex iterations | DSE iterations | Devex s | DSE s |
|---|---:|---:|---:|---:|
| `grow22` | 2,264 | 1,639 | 0.120 | 0.097 |
| `d6cube` | 868 | 353 | 0.114 | 0.049 |
| `fit2d` | 226 | 126 | 0.093 | 0.060 |
| `scsd8` | 1,563 | 786 | 0.069 | 0.033 |
| `pilot4` | 4 | 641 | 0.049 | 0.028 |
| `grow15` | 1,238 | 983 | 0.039 | 0.039 |
| `wood1p` | 219 | 223 | 0.023 | 0.026 |
| `degen2` | 611 | 434 | 0.022 | 0.019 |

**Netlib full set**, once per rule: Devex from `certified-gap-netlib-full-0134c92.csv` (the default run at this commit), dual steepest edge from `netlib-full-0134c92-dse.csv`.

| rule | passed | optimal | certified to 1e-6 | not optimal | iterations over the 92 optimal under both | shifted geomean seconds (1 s) |
|---|---:|---:|---:|---|---:|---:|
| Devex | 82 of 94 | 93 | 73 | `dfl001` (time_limit) | 150,029 | 0.1820 |
| dual steepest edge | 83 of 94 | 93 | 72 | `pilot87` (time_limit) | 111,726 | 0.1569 |

- passed under Devex only: none
- passed under dual steepest edge only: `dfl001`
- certified under Devex only: `25fv47`
- certified under dual steepest edge only: none

**The four scale sizes** (`scale.py`, random structure, dual simplex, 120 s per solve, once per rule):

| rows x cols | rule | status | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|
| 1,000 | Devex | optimal | 1.3e-15 | 3,241 | 0.4 |
| 1,000 | dual steepest edge | optimal | 5.0e-15 | 2,077 | 0.2 |
| 5,000 | Devex | optimal | 3.7e-16 | 76,253 | 48.1 |
| 5,000 | dual steepest edge | optimal | 5.5e-16 | 27,449 | 19.1 |
| 20,000 | Devex | time_limit | 4.0e+00 | 30,897 | 120.0 |
| 20,000 | dual steepest edge | time_limit | 8.5e-01 | 22,595 | 120.0 |
| 100,000 | Devex | time_limit | 2.7e+01 | 17,961 | 120.1 |
| 100,000 | dual steepest edge | time_limit | 2.8e+01 | 15,458 | 120.1 |

To reproduce:

```
python bench/runners/fetch_data.py --set medium --offline-fallback
python bench/runners/netlib.py --solver-option pricing=devex --out bench/results/netlib-medium-<sha>-devex-r1.csv
python bench/runners/netlib.py --solver-option pricing=dual-steepest-edge --out bench/results/netlib-medium-<sha>-dse-r1.csv
#   ... alternated, three repeats each
python bench/runners/fetch_data.py --set full --offline-fallback
python bench/runners/netlib.py --solver-option pricing=dual-steepest-edge --out bench/results/netlib-full-<sha>-dse.csv
python bench/runners/scale.py --engines dual-simplex --solver-option pricing=devex --out bench/results/scale-<sha>-devex.csv
python bench/runners/scale.py --engines dual-simplex --solver-option pricing=dual-steepest-edge --out bench/results/scale-<sha>-dse.csv
```

### 1c. The full set — the honest headline

Every instance in Netlib's summary table. Both tiers above are defined by a row cap, which
makes them the easier half of the library by construction; this is the number Phase 6's
">= 95% of Netlib" exit criterion is measured against, and the one the README quotes.

Source CSV: `bench/results/netlib-full-9094e1c.csv`  
Commit `9094e1c` · machine `Windows-AMD64` · generated 2026-09-29T12:26:26+00:00

**82 of 92 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

The same run graded against the **exact** optimum, Koch's rational values (*The final NETLIB-LP results*, Oper. Res. Lett. 32, 2004; `data/netlib/koch_exact.json`): **90 of 92** within a relative 1e-6 **and** verified. Not within it: `pilot.ja`, `pilot87`.

Coverage: this run used **92 of the 97 instances** Netlib publishes an optimal value for (set `full`, selected by `fetch_data.py --set full`). Phase 6's "full Netlib >= 95%" exit criterion is measured against this set.

**10 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| disagrees with the published optimum (#75) | 9 | 80bau3b, e226, ganges, greenbea, greenbeb, nesm, pilot, pilot.we, scrs8 |
| duals miss feasibility (#52) | 1 | pilot.ja |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `25fv47` | 821 | 1571 | optimal | 5.5018458883e+03 | 5.5018458883e+03 | 2.4e-12 | 4586 | 0.310 | yes |
| `80bau3b` | 2262 | 9799 | optimal | 9.8722419241e+05 | 9.8723216072e+05 | 8.1e-06 | 3887 | 0.339 | yes |
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 86 | 0.016 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.015 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 140 | 0.023 | yes |
| `agg2` | 516 | 302 | optimal | -2.0239252356e+07 | -2.0239252356e+07 | 1.1e-12 | 165 | 0.026 | yes |
| `agg3` | 516 | 302 | optimal | 1.0312115935e+07 | 1.0312115935e+07 | 8.7e-12 | 168 | 0.025 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 520 | 0.027 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 115 | 0.022 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 99 | 0.019 | yes |
| `bnl1` | 643 | 1175 | optimal | 1.9776295615e+03 | 1.9776292856e+03 | 1.4e-07 | 1674 | 0.078 | yes |
| `bnl2` | 2324 | 3489 | optimal | 1.8112365404e+03 | 1.8112365404e+03 | 2.3e-11 | 2377 | 0.171 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 465 | 0.030 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 223 | 0.021 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 162 | 0.019 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 299 | 0.022 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 259 | 0.020 | yes |
| `cycle` | 1903 | 2857 | optimal | -5.2263930249e+00 | -5.2263930249e+00 | 1.1e-12 | 335 | 0.058 | yes |
| `czprob` | 929 | 3523 | optimal | 2.1851966989e+06 | 2.1851966989e+06 | 2.0e-11 | 981 | 0.059 | yes |
| `d2q06c` | 2171 | 5167 | optimal | 1.2278421081e+05 | 1.2278423615e+05 | 2.1e-07 | 14998 | 2.815 | yes |
| `d6cube` | 415 | 6184 | optimal | 3.1549166667e+02 | 3.1549166667e+02 | 1.1e-11 | 868 | 0.170 | yes |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 611 | 0.040 | yes |
| `degen3` | 1503 | 1818 | optimal | -9.8729400000e+02 | -9.8729400000e+02 | 1.2e-16 | 3164 | 0.351 | yes |
| `dfl001` | 6071 | 12230 | optimal | 1.1266396047e+07 | 1.1266400000e+07 | 3.5e-07 | 57076 | 100.551 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 435 | 0.032 | yes |
| `etamacro` | 400 | 688 | optimal | -7.5571523335e+02 | -7.5571521774e+02 | 2.1e-08 | 779 | 0.039 | yes |
| `fffff800` | 524 | 854 | optimal | 5.5567956482e+05 | 5.5567961165e+05 | 8.4e-08 | 706 | 0.036 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 398 | 0.029 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 56 | 0.030 | yes |
| `fit1p` | 627 | 1677 | optimal | 9.1463780924e+03 | 9.1463780924e+03 | 2.3e-12 | 1088 | 0.079 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.5e-12 | 226 | 0.161 | yes |
| `fit2p` | 3000 | 13525 | optimal | 6.8464293294e+04 | 6.8464293232e+04 | 9.0e-10 | 10207 | 2.611 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 298 | 0.033 | yes |
| `ganges` | 1309 | 1681 | optimal | -1.0958573613e+05 | -1.0958636356e+05 | 5.7e-06 | 1183 | 0.054 | yes |
| `gfrd-pnc` | 616 | 1092 | optimal | 6.9022359995e+06 | 6.9022359995e+06 | 7.1e-12 | 448 | 0.033 | yes |
| `greenbea` | 2392 | 5405 | optimal | -7.2555248130e+07 | -7.2462405908e+07 | 1.3e-03 | 8526 | 1.107 | yes |
| `greenbeb` | 2392 | 5405 | optimal | -4.3022602612e+06 | -4.3021476065e+06 | 2.6e-05 | 10966 | 1.363 | yes |
| `grow15` | 300 | 645 | optimal | -1.0687094129e+08 | -1.0687094129e+08 | 3.3e-11 | 1238 | 0.063 | yes |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 2264 | 0.148 | yes |
| `grow7` | 140 | 301 | optimal | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 337 | 0.032 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 170 | 0.027 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 51 | 0.019 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 200 | 0.023 | yes |
| `maros` | 846 | 1443 | optimal | -5.8063743701e+04 | -5.8063743701e+04 | 2.2e-12 | 1802 | 0.110 | yes |
| `maros-r7` | 3136 | 9408 | optimal | 1.4971851665e+06 | 1.4971851665e+06 | 1.2e-11 | 28 | 10.134 | yes |
| `modszk1` | 687 | 1620 | optimal | 3.2061972906e+02 | 3.2061972906e+02 | 1.3e-11 | 662 | 0.043 | yes |
| `nesm` | 662 | 2923 | optimal | 1.4076036488e+07 | 1.4076073035e+07 | 2.6e-06 | 3321 | 0.145 | yes |
| `perold` | 625 | 1376 | optimal | -9.3807552782e+03 | -9.3807580773e+03 | 3.0e-07 | 2492 | 0.174 | yes |
| `pilot` | 1441 | 3652 | optimal | -5.5748972927e+02 | -5.5740430007e+02 | 1.5e-04 | 9900 | 4.066 | yes |
| `pilot.ja` | 940 | 1988 | feasible | -6.1131364656e+03 | -6.1131344111e+03 | 3.4e-07 | 3220 | 0.245 | yes |
| `pilot.we` | 722 | 2789 | optimal | -2.7201075328e+06 | -2.7201027439e+06 | 1.8e-06 | 5254 | 0.330 | yes |
| `pilot4` | 410 | 1000 | optimal | -2.5811392589e+03 | -2.5811392641e+03 | 2.0e-09 | 4 | 0.062 | yes |
| `pilot87` | 2030 | 4883 | optimal | 3.0171069277e+02 | 3.0171072827e+02 | 1.2e-07 | 21106 | 31.753 | yes |
| `pilotnov` | 975 | 2172 | optimal | -4.4972761882e+03 | -4.4972761882e+03 | 4.2e-12 | 2311 | 0.181 | yes |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 40 | 0.020 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 98 | 0.016 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 232 | 0.021 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.017 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.017 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 457 | 0.025 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 111 | 0.021 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 433 | 0.027 | yes |
| `scfxm2` | 660 | 914 | optimal | 3.6660261565e+04 | 3.6660261565e+04 | 3.3e-14 | 937 | 0.049 | yes |
| `scfxm3` | 990 | 1371 | optimal | 5.4901254550e+04 | 5.4901254550e+04 | 4.5e-12 | 1641 | 0.074 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 242 | 0.026 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 610 | 0.035 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 98 | 0.023 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000078e+01 | 5.0500000078e+01 | 7.1e-12 | 399 | 0.033 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 1563 | 0.092 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 1.6e-16 | 278 | 0.022 | yes |
| `sctap2` | 1090 | 1880 | optimal | 1.7248071429e+03 | 1.7248071429e+03 | 2.5e-11 | 775 | 0.049 | yes |
| `sctap3` | 1480 | 2480 | optimal | 1.4240000000e+03 | 1.4240000000e+03 | 0.0e+00 | 1111 | 0.057 | yes |
| `seba` | 515 | 1028 | optimal | 1.5711600000e+04 | 1.5711600000e+04 | 1.2e-16 | 439 | 0.029 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 136 | 0.048 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 124 | 0.019 | yes |
| `shell` | 536 | 1775 | optimal | 1.2088253460e+09 | 1.2088253460e+09 | 0.0e+00 | 461 | 0.033 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 447 | 0.030 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 296 | 0.026 | yes |
| `ship08l` | 778 | 4283 | optimal | 1.9090552114e+06 | 1.9090552114e+06 | 5.7e-12 | 743 | 0.047 | yes |
| `ship08s` | 778 | 2387 | optimal | 1.9200982105e+06 | 1.9200982105e+06 | 1.8e-11 | 422 | 0.036 | yes |
| `ship12l` | 1151 | 5427 | optimal | 1.4701879193e+06 | 1.4701879193e+06 | 2.0e-11 | 1063 | 0.061 | yes |
| `ship12s` | 1151 | 2763 | optimal | 1.4892361344e+06 | 1.4892361344e+06 | 4.1e-12 | 590 | 0.037 | yes |
| `sierra` | 1227 | 2036 | optimal | 1.5394362184e+07 | 1.5394362184e+07 | 2.4e-11 | 550 | 0.044 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 445 | 0.039 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 51 | 0.022 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 197 | 0.033 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 104 | 0.019 | yes |
| `stocfor2` | 2157 | 2031 | optimal | -3.9024408538e+04 | -3.9024408538e+04 | 3.0e-12 | 1877 | 0.132 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 242 | 0.030 | yes |
| `vtp.base` | 198 | 203 | optimal | 1.2983146246e+05 | 1.2983146246e+05 | 1.0e-11 | 162 | 0.021 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 219 | 0.099 | yes |
| `woodw` | 1098 | 8405 | optimal | 1.3044763331e+00 | 1.3044763331e+00 | 1.2e-11 | 2495 | 0.273 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.236s**
- slowest solved instance: 100.551s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `80bau3b`, `e226`, `ganges`, `greenbea`, `greenbeb`, `nesm`, `pilot`, `pilot.ja`, `pilot.we`, `scrs8`** — kept in the table on purpose

**`pilot87` passes this table and is 1.1e-6 from the exact optimum (#548).** The table
grades against Netlib's own readme value, 301.71072827; Koch's exact rational
recomputation (ZIB-Report 03-05, 2003) gives 301.710347333, and ours, 301.71069146, is
1.2e-7 from the readme and 1.1e-6 from the exact value. Reproduced on `main` at
`d709281` (`sankhya solve data/netlib/pilot87.mps`, default options, `simplex-dual+primal`)
and re-checked by `tools/verify_solution.py`, which shares no code with the solver: primal
infeasibility 3.8e-12, complementarity 6.8e-15, and one column, `CPCSU04`, with a
reduced-cost violation of 6.786e-08 - under the 1e-7 dual feasibility tolerance, so the
status is `optimal` by policy - which the verifier's gap accounting names as the source of
essentially the whole 3.2e-4 absolute duality gap. So the attribution is a dual that is
not quite feasible on a badly scaled model, not an infeasible primal basis and not a
relaxed unscaled retry (none ran). Not fixed here: tightening the dual tolerance to chase
one instance is the move the evidence rules forbid without a numerical justification,
and #548 stays open for a scale-aware reduced-cost test.

### 1c.1 The Kennington set - the next rung

The sixteen Kennington LPs (the `cre`, `ken`, `osa` and `pds` families, Carolan et al.,
Operations Research 38(2), 1990), larger and sparser than the core Netlib set. Fetched and
hashed by `bench/runners/fetch_kennington.py`, which parses the published optima from the
directory's own readme; run by `bench/runners/kennington.py` under the full-set rules.

Source CSV: `bench/results/kennington-full-65eecbc.csv`  
Commit `65eecbc` · machine `Windows-AMD64` · generated 2026-09-23T22:38:22+00:00

**15 of 16** matched the readme's published optimum to a relative 1e-6 **and** passed independent verification. The published values are Vanderbei's ALPO results printed to eight significant figures, so rounding moves them by at most 5e-8 relative, well inside the tolerance.

**1 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| optimality claim withdrawn by our own check (#157) | 1 | pds-20 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `cre-a` | 3516 | 4067 | optimal | 2.3595407061e+07 | 2.3595407e+07 | 2.6e-09 | 2945 | 0.262 | yes |
| `cre-b` | 9648 | 72447 | optimal | 2.3129639887e+07 | 2.3129640e+07 | 4.9e-09 | 8560 | 19.293 | yes |
| `cre-c` | 3068 | 3678 | optimal | 2.5275116141e+07 | 2.5275116e+07 | 5.6e-09 | 2937 | 0.314 | yes |
| `cre-d` | 8926 | 69980 | optimal | 2.4454969765e+07 | 2.4454970e+07 | 9.6e-09 | 7356 | 14.159 | yes |
| `ken-07` | 2426 | 3602 | optimal | -6.7952044338e+08 | -6.7952044e+08 | 5.0e-09 | 1330 | 0.088 | yes |
| `ken-11` | 14694 | 21349 | optimal | -6.9723822625e+09 | -6.9723823e+09 | 5.4e-09 | 9051 | 2.514 | yes |
| `ken-13` | 28632 | 42659 | optimal | -1.0257394789e+10 | -1.0257395e+10 | 2.1e-08 | 6117 | 5.041 | yes |
| `ken-18` | 105127 | 154699 | optimal | -5.2217025287e+10 | -5.2217025e+10 | 5.5e-09 | 187081 | 423.351 | yes |
| `osa-07` | 1118 | 23949 | optimal | 5.3572251730e+05 | 5.3572252e+05 | 5.0e-09 | 48 | 0.392 | yes |
| `osa-14` | 2337 | 52460 | optimal | 1.1064628447e+06 | 1.1064628e+06 | 4.0e-08 | 50 | 0.800 | yes |
| `osa-30` | 4350 | 100024 | optimal | 2.1421398732e+06 | 2.1421399e+06 | 1.3e-08 | 76 | 1.636 | yes |
| `osa-60` | 10280 | 232966 | optimal | 4.0440725032e+06 | 4.0440725e+06 | 7.8e-10 | 200 | 4.953 | yes |
| `pds-02` | 2953 | 7535 | optimal | 2.8857862010e+10 | 2.8857862e+10 | 3.5e-10 | 964 | 0.112 | yes |
| `pds-06` | 9881 | 28655 | optimal | 2.7761037600e+10 | 2.7761038e+10 | 1.4e-08 | 9478 | 2.608 | yes |
| `pds-10` | 16558 | 48763 | optimal | 2.6727094976e+10 | 2.6727095e+10 | 9.0e-10 | 6183 | 26.015 | yes |
| `pds-20` | 33874 | 105728 | feasible | 2.3821658640e+10 | 2.3821659e+10 | 1.5e-08 | 77 | 373.403 | yes |

**Summary**

- shifted geometric mean solve time over the passed instances (shift 1s): **3.881s**
- worst relative error against a published optimum: **4.04e-08**
- **failed: `pds-20`**, kept in the table on purpose

No per-engine option run is committed yet (`--solver-option algorithm=dual-simplex`, `simplex`, `pdhg`, `ipm`).

### 1c.2 Certified gaps on the full Netlib set — a bound valid by construction (#763)

Every LP reported `optimal` carries the Neumaier-Shcherbina safe bound of its own duals,
computed with outward rounding (Math. Programming 99, 2004), and the gap from the objective
to it. `tools/verify_solution.py` re-derives each stated bound from the model and the .sol
file in exact rational arithmetic. A certified relative gap at or under 1e-6 means the
reported objective is within 1e-6 of a proven bound on the true optimum, so that closeness
rests on a proof and not on a solver tolerance. The timings of this run are not used anywhere.

Run at `0134c92` on `cloud-container-IntelXeon-2.10GHz-4vCPU-15GB-quiet` (`certified-gap-netlib-full-0134c92.csv`).

| optimal answers | certified to 1e-6 relative | finite bound, looser | no finite bound | stated bound re-derived exactly by the verifier |
|---:|---:|---:|---:|---:|
| 93 | **73** (worst 5.1e-08) | 2 | 18 | 93 of 93 |

Finite but looser than 1e-6: modszk1 (1.09e-04), pilot87 (4.10e-06).

No finite bound from the reported duals, even after bound propagation and the basis shift (the file says `safe_lower_bound -inf` and claims nothing): bnl2, brandy, d2q06c, finnis, greenbea, greenbeb, lotfi, maros, maros-r7, perold, pilot.ja, pilot4, scfxm1, scfxm2, scfxm3, scorpion, scrs8, stair.

### 1d. Beyond Netlib — Mittelmann's LP set

Netlib's largest instance has about 6,000 rows. PS26119 asks about "thousands to millions
of variables", and the only honest way to say where this solver stands on that is to run
instances of that size and name what happens. These are the eight smallest archives in
Mittelmann's LP test set (`bench/runners/fetch_mittelmann.py`, provenance in
`data/mittelmann/reference.json`).

Source CSV: `bench/results/mittelmann-72123ff.csv`  
Commit `72123ff` · machine `Windows-AMD64` · time limit 300 s per instance, both solvers

**5 of 12** instances reached `optimal` inside the limit; **5 of 12** also passed the independent verifier and agree with HiGHS. HiGHS, run as a separate process under the same limit, finished **4 of 12**.

These are the smallest archives in Mittelmann's LP directory; against Netlib's largest instance (dfl001, 6,071 rows, 35,632 nonzeros) they range from the same row count with 2.7x the nonzeros (qap15) to 62x the rows and 42x the nonzeros (bdry2). No published optimum exists for them, so there is no pass-against-a-number column: the outcome is the status, the verifier's verdict where a solution was written, and HiGHS's objective where HiGHS finished. `our objective` on a `time_limit` row is the last iterate's value, not a bound, and is printed only so that a later run can be compared with it.

| instance | rows | cols | nonzeros | status | our objective | HiGHS objective | rel. diff | iters | solver time (s) | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `Linf_520c` | 93326 | 69004 | 566193 | time_limit | 0.7756970849 | Time limit reached | - | 24039 | 300.1 | - |
| `bdry2` | 376500 | 250998 | 1500003 | time_limit | 0.001999991456 | Time limit reached | - | 7019 | 253.4 | - |
| `brazil3` | 14646 | 23968 | 133184 | optimal | 2 | 2 | 3.7e-11 | 15 | 5.9 | yes |
| `chromaticindex1024-7` | 67583 | 73728 | 270324 | optimal | 3 | 3 | 2.6e-12 | 560 | 57.9 | yes |
| `datt256_lp` | 11077 | 262144 | 1503732 | optimal | 256 | Time limit reached | - | 10 | 153.0 | yes |
| `ex10` | 69608 | 17680 | 1162000 | optimal | 100 | 100 | 3.0e-12 | 1480 | 77.6 | yes |
| `irish-electricity` | 104259 | 61728 | 523257 | time_limit | 2454378.022 | Time limit reached | - | 36902 | 240.2 | - |
| `physiciansched3-3` | 266227 | 79555 | 1062479 | time_limit | 1431579.932 | Time limit reached | - | 21399 | 216.8 | - |
| `qap15` | 6330 | 22275 | 94950 | time_limit | 172.7930772 | Time limit reached | - | 23875 | 300.1 | - |
| `rmine15` | 358395 | 42438 | 879732 | time_limit | -5042.486721 | Time limit reached | - | 20597 | 216.7 | - |
| `s250r10` | 10962 | 273142 | 1318607 | optimal | -0.1726767184 | -0.1726770419 | 3.2e-07 | 116 | 185.0 | yes |
| `supportcase10` | 165684 | 14770 | 555082 | time_limit | 3.383924801 | Time limit reached | - | 22080 | 220.7 | - |

**Not solved inside the limit**, named rather than dropped: `Linf_520c`, `bdry2`, `irish-electricity`, `physiciansched3-3`, `qap15`, `rmine15`, `supportcase10`.

#### The same eight under each engine

Source CSVs: `bench/results/mittelmann-72123ff.csv` (dual simplex), `bench/results/mittelmann-pdhg-5c7efbc.csv` (PDHG), `bench/results/mittelmann-brazil3-ipm-xover-65eecbc.csv` (interior point)  
Same 300 s limit per instance and engine; HiGHS is not re-run here.

| instance | dual simplex: status · verified · time (s) | PDHG: status · verified · time (s) | interior point: status · verified · time (s) |
|---|---|---|---|
| `Linf_520c` | time_limit · - · 300.1 | time_limit · - · 231.4 | not run |
| `bdry2` | time_limit · - · 253.4 | time_limit · - · 241.1 | not run |
| `brazil3` | optimal · yes · 5.9 | optimal · yes · 58.0 | optimal · yes · 2.7 |
| `chromaticindex1024-7` | optimal · yes · 57.9 | optimal · yes · 0.7 | not run |
| `datt256_lp` | optimal · yes · 153.0 | not run | not run |
| `ex10` | optimal · yes · 77.6 | not run | not run |
| `irish-electricity` | time_limit · - · 240.2 | time_limit · - · 240.0 | not run |
| `physiciansched3-3` | time_limit · - · 216.8 | not run | not run |
| `qap15` | time_limit · - · 300.1 | optimal · yes · 93.4 | not run |
| `rmine15` | time_limit · - · 216.7 | time_limit · - · 216.2 | not run |
| `s250r10` | optimal · yes · 185.0 | not run | not run |
| `supportcase10` | time_limit · - · 220.7 | time_limit · - · 220.9 | not run |

Finished and verified inside the limit: dual simplex **5 of 12**, PDHG **3 of 12**, interior point **1 of 12**.

### 1e. The first-order engine — PDHG

The simplex is not the only continuous engine. Restarted PDHG (`--option algorithm=pdhg`) is
a first-order method: no basis, no factorization, and a cost that depends enormously on the
accuracy asked of it - which is why this section reports two tolerances separately rather
than one blended number. It is also the engine the GPU work targets, so its CPU behaviour is
the baseline every GPU claim will be measured against.

Source CSV: `bench/results/pdhg-4177ae6.csv`  
Commit `4177ae6` · machine `Windows-AMD64` · 9 instances, the ones committed to the repository

- **9 of 9** reach `optimal` at a requested 0.0001 with restarts on.
- **9 of 9** reach `optimal` at a requested 1e-08 with restarts on, **9 of 9** with restarts off.

`optimal` here means what it means everywhere else in this document: the point also survives the project's absolute tolerances, not merely the relative ones the first-order loop converges on. That distinction is the whole of #179 - the loop used to stop on the relative measure and the report then downgraded the point it stopped on, so the engine gave up early and handed back the weaker answer.

**Read the two tolerance columns together, because they are the same run.** Since #179 the loop stops only where the absolute standard is met, so a request looser than that standard no longer stops the solve any earlier - ask for 1e-4 and you get the 1e-8 point, at the 1e-8 cost. That is the honest reading of the identical columns below, and it is a real trade: the old behaviour honoured a loose request and returned a point it then had to label `feasible`. #180 made that the opt-in: `--option pdhg_stop_at_request=true` waives the dual, gap and complementarity halves of the standard - absolute primal feasibility is kept, so `feasible` still means a feasible point - and reports the point as `feasible` unless it meets the full standard anyway. Measured on these instances at 1e-4 it costs 0.85x the iterations (`bench/results/pdhg-stop-at-request-02cd92c.csv`) and turns `share2b` from an iteration limit into a usable point at 807,760. The two tolerance columns stay identical on `adlittle`, `israel` and `sc50b` even with the switch on, because on those the kept primal clause is what binds.

| instance | simplex | PDHG 0.0001: objective / iterations | PDHG 1e-08: objective / iterations |
|---|---:|---:|---:|
| `adlittle` | 225494.9632 | 225494.9632 / 192080 | 225494.9632 / 192080 |
| `afiro` | -464.7531429 | -464.7531428 / 1040 | -464.7531428 / 1040 |
| `blend` | -30.81214985 | -30.81214988 / 44520 | -30.81214988 / 44520 |
| `israel` | -896644.8219 | -896644.8219 / 368720 | -896644.8219 / 368720 |
| `sc105` | -52.20206121 | -52.20206122 / 61440 | -52.20206122 / 61440 |
| `sc50a` | -64.57507706 | -64.57507705 / 7680 | -64.57507705 / 7680 |
| `sc50b` | -70 | -69.99999999 / 8360 | -69.99999999 / 8360 |
| `share2b` | -415.7322407 | -415.7322407 / 1000009 | -415.7322407 / 1000009 |
| `stocfor1` | -41131.97622 | -41131.97619 / 321000 | -41131.97619 / 321000 |

**Restarts, measured at 1e-08.** The claim that restarting the averaging helps is checked rather than repeated:

| instance | restarts on | restarts off | ratio |
|---|---:|---:|---:|
| `adlittle` | 192080 | 207040 | 1.08x |
| `afiro` | 1040 | 2800 | 2.69x |
| `blend` | 44520 | 76400 | 1.72x |
| `israel` | 368720 | 1000005 | 2.71x |
| `sc105` | 61440 | 291200 | 4.74x |
| `sc50a` | 7680 | 28280 | 3.68x |
| `sc50b` | 8360 | 33200 | 3.97x |
| `share2b` | 1000009 | 1000009 | 1.00x |
| `stocfor1` | 321000 | 496320 | 1.55x |

A ratio above 1 means restarts saved iterations on that instance.


**The relative KKT error, and the three crossing times (#486).** Every PDHG run records
`kkt_1e4_seconds`, `kkt_1e6_seconds` and `kkt_1e8_seconds` in the stats JSON and, where a
runner carries them, in its CSV: the solver clock at which the relative KKT error first
came at or under that level, measured at the engine's evaluation interval. The error is
the PDLP definition, `max(||Ax - proj(Ax)|| / (1 + ||b||), ||c - A'y - z|| / (1 + ||c||),
|c'x - b'y| / (1 + |c'x| + |b'y|))` in the 2-norm on the unscaled model
(`src/pdhg/pdhg_evaluate.hpp`), which is the measurement published PDLP and cuPDLP figures
use and the one Mittelmann's LP feasibility page is quoted at (1e-6, no basis). A crossing
is what a first-order method reaches at a relative tolerance; it is NOT an optimal basis and
NOT the project's absolute standard, which is what `optimal` in the status column means and
what the verifier checks. `nan` in a crossing column means the level was never reached in
that run.
---

### 1g. GPU PDHG crossover — when the GPU wins

The GPU backend (`algorithm=pdhg gpu=true`) offloads the matrix-vector products to CUDA.
Small problems spend more time on data transfer than on computation; the crossover point
below is where the GPU overtakes the CPU.

Source CSV: `bench/results/gpu-fb72ab4.csv`  
Commit `fb72ab4` · machine `Windows-AMD64`

Both columns time PDHG alone (`pdhg_polish=false`) on the solver's own clock, to the tolerance named, the CPU side on one thread (#487); a warm-up GPU solve absorbed CUDA's context creation before the timed ones. The GPU pays a per-iteration launch and transfer cost that a small model cannot amortise; the crossover is where the parallel products start to pay for it.

> **GPU iteration counts differ from the CPU's (#448).** The device sums its reductions in a fixed order since #478, so a GPU run repeats itself, but a one-ulp difference from the CPU's summation order can flip a restart decision and shift the whole trajectory. Speedup figures here are the median of repeated solves, and a ratio below 1x is a loss, printed in the same type as a win. Do not compare a GPU iteration count against a CPU count for the same instance: the two engines take different trajectories and any comparison is meaningless. `tests/unit/test_pdhg_cuda_regression.cpp` (#451) holds both engines to the same stopping tolerance rather than to identical iterates. See also `docs/ARCHITECTURE.md` § 7.

| rows×cols | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 200×200 | 0.027 | 2.495 | 0.01× | 0.068 | 3.365 | 0.02× |
| 500×500 | 0.105 | 0.597 | 0.18× | 0.137 | 7.505 | 0.02× |
| 1000×1000 | 0.040 | 0.738 | 0.05× | 0.041 | 0.602 | 0.07× |
| 2000×2000 | 0.698 | 5.081 | 0.14× | 0.715 | 7.467 | 0.10× |
| 5000×5000 | 0.784 | 1.436 | 0.55× | 0.770 | 1.635 | 0.47× |
| 10000×10000 | 2.855 | 2.082 | 1.37× | 2.929 | 1.903 | 1.54× |

GPU: NVIDIA GeForce RTX 5050 Laptop GPU (compute 12.0, 8151 MiB VRAM); driver and CUDA runtime not recorded (the CSV predates #488's columns).  
Instances are synthetic KKT LPs with ~5 nonzeros per column (seed 42).

#### 1e-8 ceiling — sizes PDHG does not drive to project standard

Project standard: absolute primal ≤ 1e-7, dual ≤ 1e-7, gap ≤ 1e-8.  
`feasible` means the solver met the *requested* 1e-8 tolerance but not all three project-standard thresholds. These are not dropped from the table.

| rows×cols | engine | achieved primal | achieved dual |
|----------:|--------|----------------:|--------------:|
| 200×200 | CPU | 9.583e-08 | 0.000e+00 |
| 200×200 | GPU | 5.652e-08 | 5.194e-17 |
| 500×500 | CPU | 7.220e-08 | 0.000e+00 |
| 500×500 | GPU | 3.208e-08 | 8.831e-17 |
| 1000×1000 | CPU | 8.748e-08 | 0.000e+00 |
| 1000×1000 | GPU | 8.565e-08 | 8.532e-16 |
| 2000×2000 | CPU | 9.897e-08 | 0.000e+00 |

#### 1g.1 GPU on non-synthetic instances

Not yet run on this tier (the datacenter card's run is in 1g.3). Reproduce with:

```
python bench/runners/fetch_mittelmann.py
python bench/runners/gpu_real_instances.py --binary build_gpu/sankhya
```

> **Needs a CUDA-capable card and a CUDA build** (`-DSANKHYA_ENABLE_CUDA=ON`, the CUDA runtime installed). On a build without CUDA, or a machine whose card fails the device checks, `gpu=true` warns and runs on the CPU, so both arms of the runner would be CPU solves and the GPU column would mean nothing: do not run it there.

#### 1g.2 GPU PDHG vs OR-Tools PDLP

Not yet run. Reproduce with (needs GPU card and `pip install ortools`):

```
python bench/runners/fetch_mittelmann.py
python bench/runners/gpu_pdlp_compare.py --binary build_gpu/sankhya
```

See `docs/PROVENANCE.md` row 21 for the OR-Tools provenance judgement call.

#### 1g.3 The same measurements on a datacenter card

Everything above 1g.3 is one laptop card. A rented card (E2E Networks TIR) runs the same
runners from a fresh clone at a `main` commit; the CPU column in each table is that
machine's own CPU, so a ratio here is card against host, not card against the laptop.

**L4**

![GPU speedup against nonzeros on the L4](img/gpu-speedup-l4.svg)

The figure is regenerated from the two CSVs by `bench/runners/gpu_plot.py` each time this document is; each point is one instance and tolerance, the speedup against the faster CPU arm on the solver's clock (the synthetic ladder against one thread; the real instances against the faster of 1 thread and 16 threads); the dashed line is 1x, the crossover, and everything below it is a loss.

The crossover, the same protocol as 1g (`bench/runners/gpu_report.py`, medians of repeats with their min-max):

Source CSV: `bench/results/gpu-l4-fdc89c5.csv`  
Commit `fdc89c5` · machine `Linux-x86_64`

Both columns time PDHG alone (`pdhg_polish=false`) on the solver's own clock, to the tolerance named, the CPU side on one thread (#487); a warm-up GPU solve absorbed CUDA's context creation before the timed ones. The GPU pays a per-iteration launch and transfer cost that a small model cannot amortise; the crossover is where the parallel products start to pay for it.

> **GPU iteration counts differ from the CPU's (#448).** The device sums its reductions in a fixed order since #478, so a GPU run repeats itself, but a one-ulp difference from the CPU's summation order can flip a restart decision and shift the whole trajectory. Speedup figures here are the median of repeated solves (5 per cell), and a ratio below 1x is a loss, printed in the same type as a win. Do not compare a GPU iteration count against a CPU count for the same instance: the two engines take different trajectories and any comparison is meaningless. `tests/unit/test_pdhg_cuda_regression.cpp` (#451) holds both engines to the same stopping tolerance rather than to identical iterates. See also `docs/ARCHITECTURE.md` § 7.

Each cell is the median of 5 solves; `[min–max]` shows the spread from run-to-run variance (thermal state, clock boost on the laptop GPU).

| rows×cols | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 200×200 | 0.043 [0.043–0.044] | 0.663 [0.407–1.738] | 0.07× | 0.091 [0.090–0.092] | 0.723 [0.622–0.796] | 0.13× |
| 500×500 | 0.178 [0.171–0.182] | 0.832 [0.542–3.330] | 0.21× | 0.228 [0.226–0.229] | 1.327 [1.287–2.707] | 0.17× |
| 1000×1000 | 0.064 [0.063–0.064] | 0.267 [0.240–0.373] | 0.24× | 0.064 [0.063–0.066] | 0.270 [0.244–0.364] | 0.24× |
| 2000×2000 | 1.025 [1.014–1.034] | 1.449 [1.264–1.795] | 0.71× | 1.028 [1.019–1.063] | 1.302 [1.260–2.244] | 0.79× |
| 5000×5000 | 0.890 [0.879–0.905] | 0.434 [0.429–0.562] | 2.05× | 0.884 [0.870–0.904] | 0.501 [0.457–0.540] | 1.77× |
| 10000×10000 | 2.854 [2.829–3.001] | 0.678 [0.610–0.799] | 4.21× | 2.906 [2.856–2.928] | 0.745 [0.644–0.779] | 3.90× |

GPU: NVIDIA L4 (compute 8.9, 22478 MiB VRAM); driver and CUDA runtime not recorded (the CSV predates #488's columns).  
Instances are synthetic KKT LPs with ~5 nonzeros per column (seed 42).

#### 1e-8 ceiling — sizes PDHG does not drive to project standard

Project standard: absolute primal ≤ 1e-7, dual ≤ 1e-7, gap ≤ 1e-8.  
`feasible` means the solver met the *requested* 1e-8 tolerance but not all three project-standard thresholds. These are not dropped from the table.

| rows×cols | engine | achieved primal | achieved dual |
|----------:|--------|----------------:|--------------:|
| 200×200 | GPU | 3.378e-08 | 2.481e-16 |
| 500×500 | CPU | 7.220e-08 | 0.000e+00 |
| 500×500 | GPU | 9.665e-08 | 7.241e-15 |
| 1000×1000 | CPU | 8.748e-08 | 0.000e+00 |
| 2000×2000 | CPU | 9.897e-08 | 0.000e+00 |
| 2000×2000 | GPU | 6.291e-08 | 2.167e-17 |

On non-synthetic instances (`bench/runners/gpu_real_instances.py`):

Source CSV: `bench/results/gpu-real-l4-58a8374.csv`  
Commit `58a8374` · machine `Linux-x86_64`  
GPU: NVIDIA L4 (compute 8.9, 22478 MiB VRAM); driver and CUDA runtime not recorded (the CSV predates #488's columns)

Same protocol as §1g: PDHG alone, solver clock, warm-up GPU solve per instance, one solve per cell. Report the result whichever way it goes: a speedup below 1x is a loss and is printed in the same type as a win. The card is compared with two CPU arms: one thread with the serial A x (the default configuration), and 16 threads with `pdhg_parallel_spmv=true`, so A x is row-parallel as well as A^T y (#487, #488). The gap is |obj - ref| / max(1, |ref|) against the reference in the last column, which also carries HiGHS's own run time on the same file in its own process when HiGHS is the reference; `feasible` means the requested relative tolerance was met but not the project's absolute standard.

| instance | rows | tol | CPU 1 thread (s) | CPU 16 threads (s) | GPU (s) | GPU vs 1 thread | GPU vs 16 threads | GPU status | rel gap CPU 1t / CPU 16t / GPU | reference |
|----------|-----:|----:|-------------:|-------------:|--------:|--------:|--------:|--------|--------|--------|
| `refinery_year` | 779640 | 1e-04 | 300.434 | 300.519 | 300.426 | 1.00x | 1.00x | time_limit | 2.620e-08 / 1.686e-08 / 8.482e-12 | -38289180.41914 (construction) |
| `refinery_year` | 779640 | 1e-08 | 300.415 | 300.406 | 300.443 | 1.00x | 1.00x | time_limit | 2.620e-08 / 3.578e-10 / 3.656e-11 | -38289180.41914 (construction) |
| `chromaticindex1024-7` | 67583 | 1e-04 | 1.095 | 1.366 | 0.374 | 2.92x | 3.65x | feasible | 7.666e-10 / 7.666e-10 / 3.514e-10 | 3.0 (highs) |
| `chromaticindex1024-7` | 67583 | 1e-08 | 1.093 | 1.371 | 0.380 | 2.88x | 3.61x | feasible | 7.666e-10 / 7.666e-10 / 2.722e-10 | 3.0 (highs) |
| `brazil3` | 14646 | 1e-04 | 37.615 | 73.734 | 6.537 | 5.75x | 11.28x | feasible | 5.942e-07 / 5.942e-07 / 4.583e-07 | 2.0000000000012172 (highs) |
| `brazil3` | 14646 | 1e-08 | 90.802 | 205.149 | 11.206 | 8.10x | 18.31x | feasible | 1.228e-09 / 1.228e-09 / 2.371e-07 | 2.0000000000012172 (highs) |

The datacenter runner (`bench/runners/gpu_datacenter.py`, #488):

`gpu-datacenter-l4-58a8374.csv` - NVIDIA L4 (compute 8.9, 22478 MiB VRAM), driver and CUDA runtime not recorded (the CSV predates #488's columns), solver at `58a8374`, Linux-x86_64, 3 repeats per cell. HiGHS's own time on the same file, once per instance in its own process: `chromaticindex1024-7` -; `brazil3` -.

| instance | mode | threads | parallel A x | tol | forced iterations | status | objective | rel gap | primal res | dual res | iterations | solver (s) | solver median (s) | median wall (s) | spread (s) |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chromaticindex1024-7` | cpu-1t | 1 | false | 0.0001 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.090639 | 1.090639 | 1.474255 | 0.045129 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 0.0001 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.620675 | 1.521952 | 1.945741 | 0.310240 |
| `chromaticindex1024-7` | gpu | - | - | 0.0001 | - | feasible | 3.0000000002465512 | 8.218e-11 | 6.860e-09 | 5.735e-20 | 480 | 0.382011 | 0.382011 | 0.830584 | 0.101041 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-06 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.088841 | 1.088841 | 1.520321 | 0.105772 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-06 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.372105 | 1.609226 | 2.034045 | 0.461847 |
| `chromaticindex1024-7` | gpu | - | - | 1e-06 | - | feasible | 2.9999999999389435 | 2.035e-11 | 6.496e-09 | 1.612e-20 | 440 | 0.494805 | 0.409369 | 0.891146 | 0.076106 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-08 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.077835 | 1.089177 | 1.504897 | 0.057781 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-08 | - | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.399473 | 1.376116 | 1.831740 | 0.084701 |
| `chromaticindex1024-7` | gpu | - | - | 1e-08 | - | optimal | 3.000000000464075 | 1.547e-10 | 1.400e-08 | 1.037e-19 | 480 | 0.422733 | 0.394340 | 0.872333 | 0.043593 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-08 | 2000 | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.084394 | 1.087431 | 1.577028 | 0.164928 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-08 | 2000 | feasible | 3.0000000022999487 | 7.666e-10 | 7.735e-08 | 0.000e+00 | 480 | 1.426865 | 1.426865 | 2.084692 | 0.562197 |
| `chromaticindex1024-7` | gpu | - | - | 1e-08 | 2000 | optimal | 3.0000000005869634 | 1.957e-10 | 1.771e-08 | 2.689e-20 | 480 | 0.399453 | 0.399453 | 0.875116 | 0.096236 |
| `brazil3` | cpu-1t | 1 | false | 0.0001 | - | feasible | 1.999998811677631 | 5.942e-07 | 9.810e-08 | 4.616e-06 | 80800 | 37.960764 | 37.960764 | 38.344607 | 0.237166 |
| `brazil3` | cpu-16t | 16 | true | 0.0001 | - | feasible | 1.999998811677631 | 5.942e-07 | 9.810e-08 | 4.616e-06 | 80800 | 74.711291 | 85.362817 | 85.830594 | 16.782420 |
| `brazil3` | gpu | - | - | 0.0001 | - | feasible | 2.000001032349996 | 5.162e-07 | 9.870e-08 | 2.212e-15 | 62480 | 5.466302 | 7.404682 | 7.756653 | 4.042350 |
| `brazil3` | cpu-1t | 1 | false | 1e-06 | - | feasible | 2.0000000851361506 | 4.257e-08 | 8.140e-09 | 8.089e-08 | 101600 | 47.516754 | 47.566027 | 47.895255 | 0.126173 |
| `brazil3` | cpu-16t | 16 | true | 1e-06 | - | feasible | 2.0000000851361506 | 4.257e-08 | 8.140e-09 | 8.089e-08 | 101600 | 101.554162 | 101.554162 | 101.893775 | 17.412368 |
| `brazil3` | gpu | - | - | 1e-06 | - | feasible | 2.00000001864988 | 9.324e-09 | 1.820e-09 | 9.282e-08 | 121600 | 10.515429 | 11.849320 | 12.250285 | 2.249318 |
| `brazil3` | cpu-1t | 1 | false | 1e-08 | - | feasible | 1.9999999975461507 | 1.228e-09 | 2.028e-10 | 3.687e-10 | 194240 | 90.565692 | 90.756010 | 91.120279 | 1.197783 |
| `brazil3` | cpu-16t | 16 | true | 1e-08 | - | feasible | 1.9999999975461507 | 1.228e-09 | 2.028e-10 | 3.687e-10 | 194240 | 184.212826 | 193.741678 | 194.096287 | 40.233682 |
| `brazil3` | gpu | - | - | 1e-08 | - | feasible | 1.9999999998182212 | 9.150e-11 | 1.502e-11 | 2.038e-11 | 215920 | 18.045897 | 15.171753 | 15.812082 | 5.445201 |
| `brazil3` | cpu-1t | 1 | false | 1e-08 | 2000 | iteration_limit | 0.0 | 1.000e+00 | 2.212e-02 | 0.000e+00 | 2000 | 1.003801 | 1.003801 | 1.341848 | 0.102726 |
| `brazil3` | cpu-16t | 16 | true | 1e-08 | 2000 | iteration_limit | 0.0 | 1.000e+00 | 2.212e-02 | 0.000e+00 | 2000 | 1.621168 | 1.621168 | 1.966050 | 0.208349 |
| `brazil3` | gpu | - | - | 1e-08 | 2000 | iteration_limit | 0.0 | 1.000e+00 | 2.211e-02 | 1.155e-17 | 2000 | 0.371867 | 0.463225 | 0.791387 | 0.197338 |

GPU speedup on the solver's own clock (median of the repeats):

| instance | tol | forced iterations | GPU vs 1 thread | GPU vs 16 threads, parallel A x | GPU rel gap |
|---|---:|---:|---:|---:|---:|
| `chromaticindex1024-7` | 0.0001 | - | 2.85x | 3.98x | 8.218e-11 |
| `chromaticindex1024-7` | 1e-06 | - | 2.66x | 3.93x | 2.035e-11 |
| `chromaticindex1024-7` | 1e-08 | - | 2.76x | 3.49x | 1.547e-10 |
| `chromaticindex1024-7` | 1e-08 | 2000 | 2.72x | 3.57x | 1.957e-10 |
| `brazil3` | 0.0001 | - | 5.13x | 11.53x | 5.162e-07 |
| `brazil3` | 1e-06 | - | 4.01x | 8.57x | 9.324e-09 |
| `brazil3` | 1e-08 | - | 5.98x | 12.77x | 9.150e-11 |
| `brazil3` | 1e-08 | 2000 | 2.17x | 3.50x | 1.000e+00 |

**A100**

![GPU speedup against nonzeros on the A100](img/gpu-speedup-a100.svg)

The figure is regenerated from the two CSVs by `bench/runners/gpu_plot.py` each time this document is; each point is one instance and tolerance, the speedup against the faster CPU arm on the solver's clock (the synthetic ladder against one thread; the real instances against the faster of 1 thread and 16 threads); the dashed line is 1x, the crossover, and everything below it is a loss.

The crossover, the same protocol as 1g (`bench/runners/gpu_report.py`, medians of repeats with their min-max):

Source CSV: `bench/results/gpu-a100-fdd1f35.csv`  
Commit `fdd1f35` · machine `Linux-x86_64`

Both columns time PDHG alone (`pdhg_polish=false`) on the solver's own clock, to the tolerance named, the CPU side on one thread (#487); a warm-up GPU solve absorbed CUDA's context creation before the timed ones. The GPU pays a per-iteration launch and transfer cost that a small model cannot amortise; the crossover is where the parallel products start to pay for it.

> **GPU iteration counts differ from the CPU's (#448).** The device sums its reductions in a fixed order since #478, so a GPU run repeats itself, but a one-ulp difference from the CPU's summation order can flip a restart decision and shift the whole trajectory. Speedup figures here are the median of repeated solves (5 per cell), and a ratio below 1x is a loss, printed in the same type as a win. Do not compare a GPU iteration count against a CPU count for the same instance: the two engines take different trajectories and any comparison is meaningless. `tests/unit/test_pdhg_cuda_regression.cpp` (#451) holds both engines to the same stopping tolerance rather than to identical iterates. See also `docs/ARCHITECTURE.md` § 7.

Each cell is the median of 5 solves; `[min–max]` shows the spread from run-to-run variance (thermal state, clock boost on the laptop GPU).

| rows×cols | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 200×200 | 0.089 [0.088–0.108] | 1.061 [0.915–1.594] | 0.08× | 0.090 [0.089–0.091] | 1.296 [0.875–1.799] | 0.07× |
| 500×500 | 0.271 [0.271–0.281] | 1.444 [0.377–1.806] | 0.19× | 0.368 [0.367–0.382] | 1.903 [1.779–2.735] | 0.19× |
| 1000×1000 | 0.075 [0.075–0.076] | 0.439 [0.328–0.512] | 0.17× | 0.075 [0.075–0.081] | 0.479 [0.400–0.612] | 0.16× |
| 2000×2000 | 1.575 [1.555–1.579] | 1.997 [1.393–2.154] | 0.79× | 1.565 [1.562–1.579] | 1.986 [1.540–2.346] | 0.79× |
| 5000×5000 | 1.082 [1.077–1.102] | 0.648 [0.444–0.706] | 1.67× | 1.086 [1.078–1.091] | 0.596 [0.463–0.684] | 1.82× |
| 10000×10000 | 3.156 [3.148–3.183] | 0.773 [0.693–0.895] | 4.08× | 3.151 [3.147–3.152] | 0.960 [0.710–1.615] | 3.28× |

GPU: NVIDIA A100-SXM4-40GB (compute 8.0, 40326 MiB VRAM, CUDA runtime 12.4, driver API 12.4); driver 550.127.08, CUDA runtime 12.4.  
Instances are synthetic KKT LPs with ~5 nonzeros per column (seed 42).

#### 1e-8 ceiling — sizes PDHG does not drive to project standard

Project standard: absolute primal ≤ 1e-7, dual ≤ 1e-7, gap ≤ 1e-8.  
`feasible` means the solver met the *requested* 1e-8 tolerance but not all three project-standard thresholds. These are not dropped from the table.

| rows×cols | engine | achieved primal | achieved dual |
|----------:|--------|----------------:|--------------:|
| 200×200 | CPU | 7.112e-08 | 0.000e+00 |
| 200×200 | GPU | 5.606e-08 | 9.947e-16 |
| 500×500 | CPU | 8.676e-08 | 0.000e+00 |
| 500×500 | GPU | 7.995e-08 | 1.616e-15 |

On non-synthetic instances (`bench/runners/gpu_real_instances.py`):

Source CSV: `bench/results/gpu-real-a100-fdd1f35.csv`  
Commit `fdd1f35` · machine `Linux-x86_64`  
GPU: NVIDIA A100-SXM4-40GB (compute 8.0, 40326 MiB VRAM, CUDA runtime 12.4, driver API 12.4); driver 550.127.08, CUDA runtime 12.4

Same protocol as §1g: PDHG alone, solver clock, warm-up GPU solve per instance, each cell the median of 3 solves. Report the result whichever way it goes: a speedup below 1x is a loss and is printed in the same type as a win. The card is compared with two CPU arms: one thread with the serial A x (the default configuration), and 16 threads with `pdhg_parallel_spmv=true`, so A x is row-parallel as well as A^T y (#487, #488). The gap is |obj - ref| / max(1, |ref|) against the reference in the last column, which also carries HiGHS's own run time on the same file in its own process when HiGHS is the reference; `feasible` means the requested relative tolerance was met but not the project's absolute standard.

| instance | rows | tol | CPU 1 thread (s) | CPU 16 threads (s) | GPU (s) | GPU vs 1 thread | GPU vs 16 threads | GPU status | rel gap CPU 1t / CPU 16t / GPU | reference |
|----------|-----:|----:|-------------:|-------------:|--------:|--------:|--------:|--------|--------|--------|
| `refinery_year` | 779640 | 1e-04 | 300.486 | 300.580 | 300.537 | 1.00x | 1.00x | time_limit | 1.410e-08 / 2.347e-09 / 1.786e-12 | -38289180.41914 (construction) |
| `refinery_year` | 779640 | 1e-08 | 300.515 | 300.516 | 300.550 | 1.00x | 1.00x | time_limit | 1.410e-08 / 2.653e-08 / 2.148e-11 | -38289180.41914 (construction) |
| `chromaticindex1024-7` | 67583 | 1e-04 | 1.058 | 0.886 | 0.439 | 2.41x | 2.02x | feasible | 3.655e-10 / 3.655e-10 / 1.881e-10 | 3.0 (highs, 134.339 s) |
| `chromaticindex1024-7` | 67583 | 1e-08 | 1.150 | 0.868 | 0.526 | 2.19x | 1.65x | feasible | 1.468e-10 / 1.468e-10 / 1.083e-10 | 3.0 (highs, 134.339 s) |
| `brazil3` | 14646 | 1e-04 | 40.005 | 38.658 | 5.154 | 7.76x | 7.50x | feasible | 2.973e-07 / 2.973e-07 / 9.721e-08 | 2.0000000000012172 (highs, 9.285 s) |
| `brazil3` | 14646 | 1e-08 | 92.221 | 91.841 | 10.535 | 8.75x | 8.72x | feasible | 4.341e-10 / 4.341e-10 / 3.947e-09 | 2.0000000000012172 (highs, 9.285 s) |

The datacenter runner (`bench/runners/gpu_datacenter.py`, #488):

`gpu-datacenter-a100-fdd1f35.csv` - NVIDIA A100-SXM4-40GB (compute 8.0, 40326 MiB VRAM, CUDA runtime 12.4, driver API 12.4), driver 550.127.08, CUDA runtime 12.4, solver at `fdd1f35`, Linux-x86_64, 3 repeats per cell. HiGHS's own time on the same file, once per instance in its own process: `chromaticindex1024-7` 132.257 s; `brazil3` 9.158 s; `refinery_year` -.

| instance | mode | threads | parallel A x | tol | forced iterations | status | objective | rel gap | primal res | dual res | iterations | solver (s) | solver median (s) | median wall (s) | spread (s) |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chromaticindex1024-7` | cpu-1t | 1 | false | 0.0001 | - | feasible | 3.0000000010966152 | 3.655e-10 | 4.298e-08 | 0.000e+00 | 440 | 1.044568 | 1.045359 | 1.699264 | 0.094034 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 0.0001 | - | feasible | 3.0000000010966152 | 3.655e-10 | 4.298e-08 | 0.000e+00 | 440 | 0.934624 | 0.934624 | 1.630916 | 0.073677 |
| `chromaticindex1024-7` | gpu | - | - | 0.0001 | - | feasible | 3.000000001790135 | 5.967e-10 | 5.078e-08 | 1.440e-20 | 440 | 0.444030 | 0.527923 | 1.363874 | 0.305273 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-06 | - | feasible | 3.0000000010966152 | 3.655e-10 | 4.298e-08 | 0.000e+00 | 440 | 1.048594 | 1.046543 | 1.779338 | 0.180825 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-06 | - | feasible | 3.0000000010966152 | 3.655e-10 | 4.298e-08 | 0.000e+00 | 440 | 0.944112 | 0.944112 | 1.695665 | 0.165481 |
| `chromaticindex1024-7` | gpu | - | - | 1e-06 | - | optimal | 2.999999999933687 | 2.210e-11 | 2.647e-09 | 6.386e-21 | 480 | 0.441226 | 0.474954 | 1.348346 | 0.069336 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-08 | - | feasible | 3.0000000004404526 | 1.468e-10 | 1.413e-08 | 0.000e+00 | 480 | 1.140325 | 1.134743 | 1.748526 | 0.156432 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-08 | - | feasible | 3.0000000004404526 | 1.468e-10 | 1.413e-08 | 0.000e+00 | 480 | 1.134410 | 1.023737 | 1.770078 | 0.071265 |
| `chromaticindex1024-7` | gpu | - | - | 1e-08 | - | optimal | 2.9999999999201497 | 2.662e-11 | 4.337e-09 | 8.637e-21 | 480 | 0.514749 | 0.587042 | 1.410876 | 0.130916 |
| `chromaticindex1024-7` | cpu-1t | 1 | false | 1e-08 | 2000 | feasible | 3.0000000004404526 | 1.468e-10 | 1.413e-08 | 0.000e+00 | 480 | 1.140050 | 1.141037 | 1.806845 | 0.110548 |
| `chromaticindex1024-7` | cpu-16t | 16 | true | 1e-08 | 2000 | feasible | 3.0000000004404526 | 1.468e-10 | 1.413e-08 | 0.000e+00 | 480 | 1.118039 | 1.089569 | 1.863584 | 0.250995 |
| `chromaticindex1024-7` | gpu | - | - | 1e-08 | 2000 | feasible | 3.0000000003387575 | 1.129e-10 | 1.126e-08 | 3.651e-20 | 480 | 0.492554 | 0.512776 | 1.477552 | 0.549593 |
| `brazil3` | cpu-1t | 1 | false | 0.0001 | - | feasible | 2.0000005945624593 | 2.973e-07 | 5.838e-08 | 0.000e+00 | 80400 | 39.673795 | 39.673795 | 40.187951 | 0.860591 |
| `brazil3` | cpu-16t | 16 | true | 0.0001 | - | feasible | 2.0000005945624593 | 2.973e-07 | 5.838e-08 | 0.000e+00 | 80400 | 45.219215 | 39.959953 | 40.563899 | 7.264595 |
| `brazil3` | gpu | - | - | 0.0001 | - | feasible | 2.000001059866077 | 5.299e-07 | 9.924e-08 | 8.602e-06 | 76000 | 5.173633 | 5.173633 | 5.774609 | 1.267470 |
| `brazil3` | cpu-1t | 1 | false | 1e-06 | - | feasible | 2.000000046366026 | 2.318e-08 | 4.445e-09 | 3.274e-09 | 100720 | 49.791687 | 49.757710 | 50.423300 | 0.229131 |
| `brazil3` | cpu-16t | 16 | true | 1e-06 | - | feasible | 2.000000046366026 | 2.318e-08 | 4.445e-09 | 3.274e-09 | 100720 | 53.048559 | 48.922255 | 49.480363 | 5.037091 |
| `brazil3` | gpu | - | - | 1e-06 | - | feasible | 2.0000003248963667 | 1.624e-07 | 3.107e-08 | 3.952e-08 | 109520 | 7.508020 | 7.508020 | 8.135186 | 3.005358 |
| `brazil3` | cpu-1t | 1 | false | 1e-08 | - | feasible | 2.0000000008694463 | 4.341e-10 | 8.318e-11 | 9.559e-10 | 186480 | 91.624233 | 91.624233 | 92.173520 | 0.075755 |
| `brazil3` | cpu-16t | 16 | true | 1e-08 | - | feasible | 2.0000000008694463 | 4.341e-10 | 8.318e-11 | 9.559e-10 | 186480 | 86.064040 | 86.064040 | 86.742761 | 6.218757 |
| `brazil3` | gpu | - | - | 1e-08 | - | feasible | 2.000000000359578 | 1.792e-10 | 3.445e-11 | 4.879e-10 | 188320 | 12.552638 | 12.246187 | 12.851702 | 1.638437 |
| `brazil3` | cpu-1t | 1 | false | 1e-08 | 2000 | iteration_limit | 0.015343213351677185 | 9.923e-01 | 2.029e-02 | 9.577e-04 | 2000 | 1.060653 | 1.063369 | 1.559908 | 0.203006 |
| `brazil3` | cpu-16t | 16 | true | 1e-08 | 2000 | iteration_limit | 0.015343213351677185 | 9.923e-01 | 2.029e-02 | 9.577e-04 | 2000 | 0.887245 | 0.868460 | 1.711694 | 0.586000 |
| `brazil3` | gpu | - | - | 1e-08 | 2000 | iteration_limit | 0.0 | 1.000e+00 | 2.153e-02 | 1.909e-17 | 2000 | 0.351621 | 0.466897 | 1.079123 | 0.171720 |
| `refinery_year` | cpu-1t | 1 | false | 0.0001 | - | time_limit | -38289179.8791777 | 1.410e-08 | 3.513e-02 | 3.548e-04 | 6484 | 300.502737 | 300.483279 | 312.530795 | 0.162886 |
| `refinery_year` | cpu-16t | 16 | true | 0.0001 | - | time_limit | -38289180.32925697 | 2.347e-09 | 8.827e-03 | 1.834e-04 | 12360 | 300.745930 | 300.690346 | 312.785082 | 0.070754 |
| `refinery_year` | gpu | - | - | 0.0001 | - | time_limit | -38289180.41991669 | 2.028e-11 | 3.065e-05 | 1.187e-11 | 526610 | 300.580993 | 300.580993 | 312.948826 | 0.112449 |
| `refinery_year` | cpu-1t | 1 | false | 1e-06 | - | time_limit | -38289179.8791777 | 1.410e-08 | 3.513e-02 | 3.548e-04 | 6404 | 300.500946 | 300.500946 | 312.829517 | 0.162958 |
| `refinery_year` | cpu-16t | 16 | true | 1e-06 | - | time_limit | -38289180.32925697 | 2.347e-09 | 8.827e-03 | 1.834e-04 | 12626 | 300.696524 | 300.602121 | 312.769317 | 0.179278 |
| `refinery_year` | gpu | - | - | 1e-06 | - | time_limit | -38289180.420124196 | 2.570e-11 | 8.175e-05 | 1.853e-09 | 527335 | 300.557498 | 300.561976 | 312.729397 | 0.249979 |
| `refinery_year` | cpu-1t | 1 | false | 1e-08 | - | time_limit | -38289179.8791777 | 1.410e-08 | 3.513e-02 | 3.548e-04 | 6462 | 300.495616 | 300.495616 | 312.697036 | 0.226163 |
| `refinery_year` | cpu-16t | 16 | true | 1e-08 | - | time_limit | -38289181.43510503 | 2.653e-08 | 1.662e-02 | 3.066e-04 | 12969 | 300.618292 | 300.678841 | 313.097623 | 0.715307 |
| `refinery_year` | gpu | - | - | 1e-08 | - | time_limit | -38289180.41914109 | 2.841e-14 | 1.274e-05 | 3.233e-08 | 527654 | 300.541985 | 300.554939 | 312.713069 | 0.705166 |
| `refinery_year` | cpu-1t | 1 | false | 1e-08 | 2000 | iteration_limit | -38289178.233224735 | 5.709e-08 | 1.351e-01 | 5.119e-03 | 2000 | 95.325896 | 95.249205 | 107.188652 | 0.602393 |
| `refinery_year` | cpu-16t | 16 | true | 1e-08 | 2000 | iteration_limit | -38289178.233224735 | 5.709e-08 | 1.351e-01 | 5.119e-03 | 2000 | 51.905765 | 51.905765 | 64.141963 | 3.874404 |
| `refinery_year` | gpu | - | - | 1e-08 | 2000 | iteration_limit | -38289179.325400755 | 2.857e-08 | 1.383e-01 | 2.162e-03 | 2000 | 4.522373 | 4.510435 | 16.555003 | 0.209577 |

GPU speedup on the solver's own clock (median of the repeats):

| instance | tol | forced iterations | GPU vs 1 thread | GPU vs 16 threads, parallel A x | GPU rel gap |
|---|---:|---:|---:|---:|---:|
| `chromaticindex1024-7` | 0.0001 | - | 1.98x | 1.77x | 5.967e-10 |
| `chromaticindex1024-7` | 1e-06 | - | 2.20x | 1.99x | 2.210e-11 |
| `chromaticindex1024-7` | 1e-08 | - | 1.93x | 1.74x | 2.662e-11 |
| `chromaticindex1024-7` | 1e-08 | 2000 | 2.23x | 2.12x | 1.129e-10 |
| `brazil3` | 0.0001 | - | 7.67x | 7.72x | 5.299e-07 |
| `brazil3` | 1e-06 | - | 6.63x | 6.52x | 1.624e-07 |
| `brazil3` | 1e-08 | - | 7.48x | 7.03x | 1.792e-10 |
| `brazil3` | 1e-08 | 2000 | 2.28x | 1.86x | 1.000e+00 |
| `refinery_year` | 0.0001 | - | 1.00x | 1.00x | 2.028e-11 |
| `refinery_year` | 1e-06 | - | 1.00x | 1.00x | 2.570e-11 |
| `refinery_year` | 1e-08 | - | 1.00x | 1.00x | 2.841e-14 |
| `refinery_year` | 1e-08 | 2000 | 21.12x | 11.51x | 2.857e-08 |

#### 1g.4 What the CPU side does with its cores

The CPU column of every crossover table in 1g and 1g.3 is one thread: `threads` defaults
to 1 and the crossover runner does not set it. The real-instance and datacenter tables name
their CPU arms in their own columns, one thread with the serial A x and N threads with the
row-parallel A x (`pdhg_parallel_spmv=true`), and a speed-up is printed against each arm.
`pdhg_parallel_spmv` (#487) computes A x row-parallel
over the `threads` workers, bitwise the same at any thread count (the test holds it to the
bit); this is what it buys, per instance, at a fixed iteration count:

Source CSV: `pdhg-threads-58a8374.csv`  
Commit `58a8374` · machine `Linux-x86_64` · 2000 iterations per solve, PDHG alone, `pdhg_parallel_spmv=true` except the `serial` rows; `+ updates` rows also run the vector updates and the step rule's sums over the threads (`pdhg_parallel_updates=true`).

| instance | rows | threads | A x | solver (s) | speed-up over 1 thread |
|---|---:|---:|---|---:|---:|
| `kkt_1000x1000` | 1000 | 1 | serial | 0.049 | - |
| `kkt_1000x1000` | 1000 | 1 | parallel + updates | 0.054 | 1.000x |
| `kkt_1000x1000` | 1000 | 2 | parallel + updates | 0.134 | 0.401x |
| `kkt_1000x1000` | 1000 | 4 | parallel + updates | 0.162 | 0.332x |
| `kkt_1000x1000` | 1000 | 8 | parallel + updates | 0.177 | 0.304x |
| `kkt_1000x1000` | 1000 | 16 | parallel + updates | 0.184 | 0.293x |
| `kkt_2000x2000` | 2000 | 1 | serial | 0.099 | - |
| `kkt_2000x2000` | 2000 | 1 | parallel + updates | 0.102 | 1.000x |
| `kkt_2000x2000` | 2000 | 2 | parallel + updates | 0.165 | 0.619x |
| `kkt_2000x2000` | 2000 | 4 | parallel + updates | 0.179 | 0.573x |
| `kkt_2000x2000` | 2000 | 8 | parallel + updates | 0.204 | 0.502x |
| `kkt_2000x2000` | 2000 | 16 | parallel + updates | 0.230 | 0.445x |
| `kkt_5000x5000` | 5000 | 1 | serial | 0.320 | - |
| `kkt_5000x5000` | 5000 | 1 | parallel + updates | 0.330 | 1.000x |
| `kkt_5000x5000` | 5000 | 2 | parallel + updates | 0.483 | 0.682x |
| `kkt_5000x5000` | 5000 | 4 | parallel + updates | 0.390 | 0.846x |
| `kkt_5000x5000` | 5000 | 8 | parallel + updates | 0.349 | 0.945x |
| `kkt_5000x5000` | 5000 | 16 | parallel + updates | 0.360 | 0.915x |
| `kkt_10000x10000` | 10000 | 1 | serial | 0.652 | - |
| `kkt_10000x10000` | 10000 | 1 | parallel + updates | 0.677 | 1.000x |
| `kkt_10000x10000` | 10000 | 2 | parallel + updates | 0.697 | 0.971x |
| `kkt_10000x10000` | 10000 | 4 | parallel + updates | 0.519 | 1.305x |
| `kkt_10000x10000` | 10000 | 8 | parallel + updates | 0.562 | 1.204x |
| `kkt_10000x10000` | 10000 | 16 | parallel + updates | 0.531 | 1.274x |
| `refinery_year` | 779640 | 1 | serial | 91.743 | - |
| `refinery_year` | 779640 | 1 | parallel + updates | 82.422 | 1.000x |
| `refinery_year` | 779640 | 2 | parallel + updates | 49.096 | 1.679x |
| `refinery_year` | 779640 | 4 | parallel + updates | 32.964 | 2.500x |
| `refinery_year` | 779640 | 8 | parallel + updates | 28.346 | 2.908x |
| `refinery_year` | 779640 | 16 | parallel + updates | 23.879 | 3.452x |

#### 1g.5 Root domain propagation, CPU against the device (#510)

Activity-based bound propagation at the MIP root (`gpu_domain_prop=true`), the CPU
reference against the CUDA propagator (one thread per row, one per column; Sofranac,
Gleixner and Pokutta, arXiv:2009.07785). The two return the same bounds bit for bit
(`tests/unit/test_domain_propagation.cpp`, on random models, Netlib and the fetched MIPLIB
sets); this is what each costs, on generated knapsack-row models from 1,000 to 1,000,000
rows (`bench/runners/gpu_domain_prop.py`).

Source CSV: `gpu-domain-prop-58a8374.csv`  
Commit `58a8374` · machine `Linux-x86_64` · GPU NVIDIA L4 (compute 8.9, 22478 MiB VRAM) · median of 5 run(s) per cell, the propagation time the solver logs (on the device: the row-major copy, the transfers and the rounds).

| rows | columns | nonzeros | rounds | bounds tightened | CPU, 1 thread (s) | GPU (s) | GPU / CPU | GPU context (s) | GPU without context (s) | same rounds and count |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 1,000 | 5,000 | 2 | 797 | 0.000078 | 0.110972 | 1422.72x | 0.110305 | 0.000669 | yes |
| 10,000 | 10,000 | 50,000 | 2 | 7,708 | 0.000802 | 0.114150 | 142.33x | 0.112876 | 0.001274 | yes |
| 100,000 | 100,000 | 500,000 | 2 | 77,630 | 0.011564 | 0.118193 | 10.22x | 0.111768 | 0.006318 | yes |
| 300,000 | 300,000 | 1,500,000 | 2 | 233,257 | 0.042749 | 0.162775 | 3.81x | 0.140816 | 0.021959 | yes |
| 1,000,000 | 1,000,000 | 5,000,000 | 2 | 777,545 | 0.287839 | 0.325613 | 1.13x | 0.221346 | 0.108063 | yes |

**The device loses at every measured size (5 of 5).** The CPU reference is faster from the smallest model to the largest; the GPU backend stays off by default (`gpu_domain_prop=false`, `domain_prop_backend=auto`).
Without the context (a process that has already touched the card), the device is faster at 3 of 5.

#### 1g.6 Two cards against one: the row-partitioned engine (#295)

Rows of A split across the cards, balanced by nonzeros plus rows; the primal iterate
replicated, the dual partitioned, A^T y summed across the cards every iteration in slot
order over P2P (NVLink or PCIe) or, without it, staged through the host
(`src/gpu/pdhg_multi_gpu.cu`). Dispatch is explicit (`gpu_devices=0,1`); nothing chooses
several cards by itself, for the reason the table gives (`docs/ARCHITECTURE.md` section 7).

Source CSV: `bench/results/multi-gpu-e2e-2xA100-SXM4-40GB-NV12-58a8374.csv`  
Commit `58a8374` · machine `e2e-2xA100-SXM4-40GB-NV12` · GPU not recorded; driver and CUDA runtime not recorded (the CSV predates #488's columns)  
1 run(s) per cell; the row is the median per-step time with its min and max where recorded.

`partitioned-1` is the multi-card engine on one card, the like-for-like baseline; `single-engine` is the production single-card engine. The exchange share is the time the cards spend in the cross-card sum of A^T y (waiting for the slowest partial included) as a share of the device time per step; `peak MiB` is what each slot allocated. A ratio below 1x is a loss and is printed in the same type as a win.

| instance | rows | nnz | budget | config | status | rel gap | us/step [min-max] | exchange share | exchange us/step | MiB moved | peak MiB per slot | partition | solve (s) |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| `kkt_250000_8_295` | - | - | steps | single-engine | iteration_limit | 2.0365770667639497e-06 | 1532.4 | - | - | - | - | - | 3.064838692 |
| `kkt_250000_8_295` | - | - | steps | partitioned-1 | iteration_limit | 2.0417248172018695e-06 | 301.8 | - | - | - | - | - | 3.89680004 |
| `kkt_250000_8_295` | - | - | steps | partitioned-2 | iteration_limit | 2.05556322707943e-06 | 281.8 | - | - | - | - | - | 4.033234973 |
| `kkt_250000_8_295` | - | - | steps | partitioned-2-host | iteration_limit | 2.05556322707943e-06 | 561.7 | - | - | - | - | - | 5.919624766 |
| `kkt_1000000_8_295` | - | - | steps | single-engine | iteration_limit | 7.828675480288482e-07 | 7589.2 | - | - | - | - | - | 15.178445209 |
| `kkt_1000000_8_295` | - | - | steps | partitioned-1 | iteration_limit | 7.50299129631909e-07 | 780.2 | - | - | - | - | - | 14.744972242 |
| `kkt_1000000_8_295` | - | - | steps | partitioned-2 | iteration_limit | 8.851956530360737e-07 | 658.6 | - | - | - | - | - | 14.804839477 |
| `kkt_1000000_8_295` | - | - | steps | partitioned-2-host | iteration_limit | 8.851956530360737e-07 | 1844.8 | - | - | - | - | - | 18.013142631 |
| `kkt_4000000_8_295` | - | - | steps | single-engine | iteration_limit | 1.0458861020341966e-06 | 53819.2 | - | - | - | - | - | 107.638338313 |
| `kkt_4000000_8_295` | - | - | steps | partitioned-1 | iteration_limit | 1.0619466278583482e-06 | 4465.8 | - | - | - | - | - | 108.389042159 |
| `kkt_4000000_8_295` | - | - | steps | partitioned-2 | iteration_limit | 1.100689788744084e-06 | 2722.3 | - | - | - | - | - | 106.526108326 |
| `kkt_4000000_8_295` | - | - | steps | partitioned-2-host | iteration_limit | 1.100689788744084e-06 | 7919.3 | - | - | - | - | - | 115.886434061 |
| `Linf_520c` | - | - | steps | single-engine | iteration_limit | - | 764.5 | - | - | - | - | - | 1.528970668 |
| `Linf_520c` | - | - | steps | partitioned-1 | iteration_limit | - | 143.2 | - | - | - | - | - | 1.259586643 |
| `Linf_520c` | - | - | steps | partitioned-2 | iteration_limit | - | 167.7 | - | - | - | - | - | 1.910138542 |
| `Linf_520c` | - | - | steps | partitioned-2-host | iteration_limit | - | 238.7 | - | - | - | - | - | 1.547371667 |
| `bdry2` | - | - | steps | single-engine | iteration_limit | - | 1623.4 | - | - | - | - | - | 3.246786156 |
| `bdry2` | - | - | steps | partitioned-1 | iteration_limit | - | 260.3 | - | - | - | - | - | 2.468190109 |
| `bdry2` | - | - | steps | partitioned-2 | iteration_limit | - | 592.2 | - | - | - | - | - | 4.256273472 |
| `bdry2` | - | - | steps | partitioned-2-host | iteration_limit | - | 830.5 | - | - | - | - | - | 3.764644477 |

Two cards against one, per step (partitioned-2 / partitioned-1, the same arithmetic split; `single-engine`'s per-step figure includes its setup and evaluation, so it is not a like-for-like denominator and is not divided here):

| instance | budget | two cards vs partitioned-1 | host-staged vs partitioned-1 |
|---|---|---:|---:|
| `kkt_250000_8_295` | steps | 1.07x | 0.54x |
| `kkt_1000000_8_295` | steps | 1.18x | 0.42x |
| `kkt_4000000_8_295` | steps | 1.64x | 0.56x |
| `Linf_520c` | steps | 0.85x | 0.60x |
| `bdry2` | steps | 0.44x | 0.31x |

#### 1g.7 The interior point's normal equations on cuDSS (#489)

`ipm_linear_solver=cudss` factors the normal equations on the device (NVIDIA cuDSS, a
vendor library and not a solver, judgement call 28 in `docs/PROVENANCE.md`); the Newton
iteration and the refinement stay on the host. The instances are section 1g.3's: the
synthetic ladder, the refinery year and the Mittelmann pair.

Not yet run. Needs a build with `-DSANKHYA_ENABLE_CUDSS=ON` and a card:

```
python bench/runners/ipm_cudss.py --binary build/sankhya --card a100
```


#### 1g.8 Feasibility Jump on the device against the CPU (#508)

The MIP primal heuristic of Luteberget and Sartor (Feasibility Jump, Mathematical Programming
Computation 15, 2023) run once per instance on each engine from the box point closest to zero,
every third instance of the MIPLIB tier-2 list, `bench/runners/gpu_fj_ab.py`. The question is
only whether and when a feasible point appears; neither engine is in the default solve path.

Source CSV: `bench/results/gpu-fj-ab-ad57c03-e2e-V100.csv`  
Commit `ad57c03` · machine `e2e-V100` · GPU Tesla V100-PCIE-32GB · 60.0 s per run, one run per instance and engine, every point checked against the model (`verified`)

- `cpu`: a feasible point on **16 of 20** instances by the limit, 14 of 20 by 10 s.
- `gpu`: a feasible point on **15 of 20** instances by the limit, 14 of 20 by 10 s.

| instance | cpu first (s) | cpu rel. gap | gpu first (s) | gpu rel. gap |
|---|---:|---:|---:|---:|
| `csched007` | none | - | none | - |
| `glass4` | 1.902 | 0.733728 | 2.019 | 0.472213 |
| `mad` | 0.002 | 0.6322 | 0.172 | 0.31 |
| `markshare_4_0` | 0.000 | 162 | 0.153 | 46 |
| `mas76` | 0.000 | 0.0971636 | 0.158 | 0.0411497 |
| `mc11` | 0.030 | 0.621011 | 0.653 | 1.16109 |
| `mik-250-20-75-4` | 0.000 | 0.026902 | 0.156 | 0.021491 |
| `n5-3` | 0.296 | 0.559284 | 2.483 | 0.467366 |
| `neos-3024952-loue` | 52.911 | 7.86418 | 31.681 | 7.55042 |
| `neos-3627168-kasai` | 15.132 | 0.0160906 | none | - |
| `neos-3754480-nidda` | 0.000 | 0.840704 | 0.189 | 0.46102 |
| `neos-4338804-snowy` | 0.018 | 0.0509857 | 0.281 | 0.0135962 |
| `pk1` | 0.001 | 8.63636 | 0.156 | 2.81818 |
| `qap10` | 0.001 | 0.376471 | 0.141 | 0.111765 |
| `reblock115` | 0.000 | 1 | 0.130 | 0.921877 |
| `rococoC10-001000` | none | - | none | - |
| `roll3000` | none | - | none | - |
| `timtab1` | none | - | none | - |
| `tr12-30` | 0.008 | 0.145533 | 0.242 | 0.0990153 |
| `uct-subprob` | 0.010 | 0.207006 | 0.172 | 0.10828 |


#### 1g.9 Deterministic device reductions on the refinery year, per iteration (#478)

Since #478 the device sums its reductions in a fixed order (`deterministic=true`) so a GPU
solve repeats itself. This measures what that costs per PDHG iteration on the 779,640-row
refinery year (`generate_refinery_lp.py --periods 8760 --seed 42`), with three and with two
sparse products per iteration, and with the whole iteration loop kept on the device
(`gpu_on_device_loop=true`).

Instance `refinery_year.mps` (779640 rows, 1208880 columns, 9968834 nonzeros), engine `cuda`, 10000 iterations per solve, median of 3 repeats; the per-iteration figure is the marginal time between the full and the one-fifth run, as `pdhg_two_matvec_ab.py` defines it. Commit `fdd1f35`, GPU NVIDIA A100-SXM4-40GB (compute 8.0, 40326 MiB VRAM, CUDA runtime 12.4, driver API 12.4).

| leg | solver options | three products (us/iter) | two products (us/iter) | two / three | source |
|---|---|---:|---:|---:|---|
| default | `defaults` | 598.1 | 489.5 | 0.818 | `pdhg-478-refinery-default-fdd1f35.csv` |
| deterministic | `deterministic=true` | 569.0 | 495.8 | 0.871 | `pdhg-478-refinery-deterministic-fdd1f35.csv` |
| deterministic-loop | `deterministic=true gpu_on_device_loop=true` | 575.7 | 448.1 | 0.778 | `pdhg-478-refinery-deterministic-loop-fdd1f35.csv` |
| loop | `gpu_on_device_loop=true` | 553.3 | 479.1 | 0.866 | `pdhg-478-refinery-loop-fdd1f35.csv` |


#### 1g.10 GPU MIP heuristics and batched node bounds on the full MIPLIB tier 2 (#509, #520)

All 60 instances of the tier-2 list (`bench/runners/miplib_tier2.json`), 60 s each, one run
per leg on one binary: the feasibility pump and fix-and-propagate on PDHG relaxations (#509,
on the device and with `gpu_heur_backend=cpu`), and batched PDHG for node bounds and for
strong-branching scores (#520, on the device and with `gpu_batch_backend=cpu`). All of
these are off by default; each row says what turning one on does against `off`. "Closer" and
"further" compare the incumbent, and the final dual bound, with the published optimum,
instance by instance. Every leg runs with `miplib.py --profile` (profile=detailed), which
records the seconds spent choosing the branching column (strong branching's probe LPs are
inside them) and the batched-PDHG calls. The legs ran on a 64-core host with one V100 as two
streams side by side, one single-threaded process each: the CPU-only legs (`off` and the
`*-cpu` legs) in one, the device legs in the other, so exactly one process used the card.
The CPU stream has four legs and the device stream six, so every leg except the last two
device legs (`520-sb-filter`, `520-both`, which ran alone) shared the host with exactly one
other single-threaded solve.

Commit `a0cc065` · machine `Linux-x86_64` · 60 instances, one seed, one thread; every leg against `off`, the same binary with every GPU option off.

| leg | solver options | feasible (verified) | matched | proved | nodes / s | branching s | batched SB s | primal closer / further | dual bound closer / further | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| off | `defaults` | 40 (40) | 4 | 0 | 1276 | 960.5 | 0.0 | - | - | `miplib-t2-full-off-a0cc065.csv` |
| 509-fixprop-gpu | `gpu_fix_and_prop=true` | 42 (42) | 4 | 0 | 1270 | 949.1 | 0.0 | 9 / 4 | 14 / 14 | `miplib-t2-full-509-fixprop-gpu-a0cc065.csv` |
| 509-fixprop-cpu | `gpu_fix_and_prop=true gpu_heur_backend=cpu` | 42 (42) | 4 | 0 | 1274 | 952.7 | 0.0 | 9 / 4 | 11 / 12 | `miplib-t2-full-509-fixprop-cpu-a0cc065.csv` |
| 509-pump-gpu | `gpu_pump=true` | 44 (44) | 4 | 0 | 1241 | 922.2 | 0.0 | 5 / 5 | 3 / 28 | `miplib-t2-full-509-pump-gpu-a0cc065.csv` |
| 509-pump-cpu | `gpu_pump=true gpu_heur_backend=cpu` | 44 (44) | 4 | 0 | 1260 | 924.8 | 0.0 | 5 / 6 | 6 / 22 | `miplib-t2-full-509-pump-cpu-a0cc065.csv` |
| 520-nodes | `gpu_batch_nodes=true` | 40 (40) | 2 | 0 | 424 | 938.2 | 0.0 | 1 / 15 | 4 / 28 | `miplib-t2-full-520-nodes-a0cc065.csv` |
| 520-sb-score | `gpu_batch_strong_branching=score` | 36 (36) | 2 | 0 | 844 | 1965.7 | 1958.8 | 9 / 29 | 5 / 34 | `miplib-t2-full-520-sb-score-a0cc065.csv` |
| 520-sb-filter | `gpu_batch_strong_branching=filter` | 37 (37) | 2 | 0 | 1094 | 1865.0 | 1225.2 | 3 / 18 | 0 / 36 | `miplib-t2-full-520-sb-filter-a0cc065.csv` |
| 520-both | `gpu_batch_nodes=true gpu_batch_strong_branching=score` | 36 (36) | 2 | 0 | 255 | 1914.4 | 1910.3 | 6 / 32 | 5 / 34 | `miplib-t2-full-520-both-a0cc065.csv` |
| 520-both-cpu | `gpu_batch_nodes=true gpu_batch_strong_branching=score gpu_batch_backend=cpu` | 33 (33) | 1 | 0 | 185 | 2157.4 | 2155.2 | 5 / 32 | 1 / 38 | `miplib-t2-full-520-both-cpu-a0cc065.csv` |

---

### 1f. Scale — how far up this goes

Every tier above is Netlib-sized: the largest instance in the full set has 12,230 columns, and
most have a few hundred, so none of them speaks to the size PS26119 asks about.

Source CSV: `bench/results/scale-e134aeb.csv`  
Commit `e134aeb` · machine `Windows-AMD64` · 120.0s per solve

PS26119 asks for **thousands to millions of variables**, and this is the section that answers it with a file rather than an adjective. The instances are generated backwards from a primal-dual pair that already satisfies the KKT conditions, from integer data, so the optimum is known EXACTLY before the solver sees the model (`bench/runners/generate_large_lp.py`). A large random instance would prove nothing: nobody would know whether the answer was right.

**Read the error column before the clock.** Whether an objective is right is a property of the solver. How long it took, and therefore whether the run ended at the limit, is a property of this laptop on this day.

| size (rows x cols) | engine | status | objective | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|---:|
| 1,000 | `dual-simplex` | optimal | -362 | 9.4e-15 | 4271 | 0.4 |
| 1,000 | `ipm` | optimal | -361.9999994 | 1.6e-09 | 19 | 0.4 |
| 1,000 | `pdhg` | optimal | -362 | 6.0e-13 | 71200 | 1.2 |
| 5,000 | `dual-simplex` | time limit | 27163.63491 | 1.7e+00 | 33019 | 120.0 |
| 5,000 | `ipm` | feasible | 9945.000027 | 2.7e-09 | 24 | 70.9 |
| 5,000 | `pdhg` | optimal | 9945 | 1.1e-11 | 410840 | 53.7 |
| 20,000 | `dual-simplex` | time limit | -115703.4636 | 3.4e+00 | 34163 | 120.2 |
| 20,000 | `ipm` | time limit | 27091.64405 | 2.0e+00 | 0 | 120.2 |
| 20,000 | `pdhg` | time limit | -26592.00005 | 2.0e-09 | 122309 | 86.2 |
| 100,000 | `dual-simplex` | time limit | -2148996.41 | 2.5e+01 | 19497 | 121.0 |
| 100,000 | `ipm` | no output | - | - | - | 202.6 |
| 100,000 | `pdhg` | time limit | -83109.99102 | 1.1e-07 | 20400 | 114.9 |

**7 of 12** solves reached the analytic optimum to a relative 1e-06.

- `dual-simplex` reached it at **1,000** rows and columns (optimal, 0.4 s).
- `ipm` reached it at **5,000** rows and columns (feasible, 70.9 s).
- `pdhg` reached it at **100,000** rows and columns (time limit, 114.9 s).

**Reaching the answer and proving it are different things, and at this scale they come apart.** `ipm` at 5,000, `pdhg` at 20,000, `pdhg` at 100,000 landed on the analytic optimum and still stopped at the limit, because the convergence test had not been satisfied when the clock ran out. Reported as what it is - not `optimal` - and worth knowing: a first-order method is useful long before it can certify itself.

**What this does NOT say.** The largest instance here is 100,000 rows and columns. That is the thousands end of what the problem statement asks for and the low end of the millions; nothing above is evidence about a million-variable model. The instances in this table are also one shape - square, 5000 nonzeros at the smallest size, with nonzeros placed at random - which is an expander graph, the worst case for anything that factorizes, and nothing like a refinery. Section 1f.2 runs the same sizes on a shape a planning model has; section 5 is where the structural hazards are pushed.

#### 1f.1 The same question without the clock

Source CSV: `bench/results/scale-iterations-bf3df02.csv`  
Commit `bf3df02` · machine `Windows-AMD64` · **1000 iterations per solve**, not a clock

**This is the one measurement on this page another machine reproduces exactly.** Every other timing here is partly a property of this laptop: a machine at half speed does half the iterations inside a time limit and lands further from the optimum, so the same solver looks worse. Fixing the iteration count removes the machine, and what is left is a property of the algorithm.

| size (rows x cols) | engine | objective | analytic optimum | relative error | polish iterations | seconds |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | `pdhg-raw` | -361.7805711 | -362 | 6.1e-04 | - | 0.1 |
| 1,000 | `pdhg` | -361.9999999 | -362 | 1.9e-10 | 7 | 0.3 |
| 5,000 | `pdhg-raw` | 9944.617973 | 9945 | 3.8e-05 | - | 0.3 |
| 5,000 | `pdhg` | 9944.617973 | 9945 | 3.8e-05 | 7 | 30.3 |
| 20,000 | `pdhg-raw` | -26591.57153 | -26592 | 1.6e-05 | - | 1.3 |
| 20,000 | `pdhg` | -26591.57153 | -26592 | 1.6e-05 | - | 4.9 |
| 100,000 | `pdhg-raw` | -83110.56935 | -83110 | 6.9e-06 | - | 7.6 |
| 100,000 | `pdhg` | -83110.56935 | -83110 | 6.9e-06 | - | 37.7 |
| 500,000 | `pdhg-raw` | 87564.13601 | 87565 | 9.9e-06 | - | 50.4 |
| 500,000 | `pdhg` | 87564.13601 | 87565 | 9.9e-06 | - | 80.0 |
| 1,000,000 | `pdhg-raw` | 208841.722 | 208845 | 1.6e-05 | - | 122.2 |
| 1,000,000 | `pdhg` | 208841.722 | 208845 | 1.6e-05 | - | 151.7 |

| size | unpolished error | polished error | polish iterations | polished status |
|---:|---:|---:|---:|---|
| 1,000 | 6.1e-04 | 1.9e-10 | 7 | optimal |
| 5,000 | 3.8e-05 | 3.8e-05 | 7 | iteration limit |
| 20,000 | 1.6e-05 | 1.6e-05 | 0 | iteration limit |
| 100,000 | 6.9e-06 | 6.9e-06 | 0 | iteration limit |
| 500,000 | 9.9e-06 | 9.9e-06 | 0 | iteration limit |
| 1,000,000 | 1.6e-05 | 1.6e-05 | 0 | iteration limit |

**The polish is where the accuracy comes from, where it can run.** On the size where it ran (1,000 rows) the same 1000 first-order iterations, finished by the interior point from the point they reached, land 3,166,592x closer to the optimum, at the cost of the polish-iteration column - each of those is a factorization, and the answer's iteration count is the sum of both phases. A first-order method converges linearly with a rate that flattens near the optimum; a second-order method started there converges quadratically. At 5,000, 20,000, 100,000, 500,000, 1,000,000 rows the polish did not improve the answer - its factor, or the ordering that sizes it, did not fit `polish_max_factor_nonzeros` / `polish_max_seconds` (a nonzero polish-iteration count there is what it managed before the clock) - and the first-order answer stands unchanged, which is what the table shows. The seconds column carries the cost of finding that out. The polish is on by default (`pdhg_polish`) and is measured here so that the unpolished number stays on the page beside it.

**The accuracy does not degrade with the model.** Across sizes from 1,000 to 1,000,000 rows and columns, the same 1000 iterations land between 1.9e-10 and 6.1e-04 of an optimum known exactly by construction. The number of iterations a first-order method needs is a property of the problem's conditioning, not of its size, and on this family that shows: what grows with the model is the cost of one iteration, not how many are required.

Only the first-order method is measured here, and deliberately. The simplex and the interior point are not iterative in the same sense - a simplex iteration is a pivot and an interior-point iteration is a factorization, so the same count means something different for each, and section 1f already shows both running out of time well below these sizes.

#### 1f.2 The same sizes on a second shape

Source CSV: `bench/results/scale-staircase-e134aeb.csv`  
Commit `e134aeb` · machine `Windows-AMD64` · 120.0s per solve · staircase structure

**Same construction, same sizes, same nonzeros per column, different pattern.** The random family above draws each column's rows uniformly, which makes an expander graph: no small separators, so every elimination ordering fills catastrophically. That is the worst case for a method that factorizes and it looks nothing like an industrial model. This family is a staircase, each column in its own period with one coupling into the next - a multi-period planning model, which is the shape PS26119's own domain produces. The optimum is exact by construction either way.

| size (rows x cols) | engine | status | objective | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|---:|
| 1,000 | `dual-simplex` | optimal | -5821 | 0.0e+00 | 5338 | 0.5 |
| 1,000 | `ipm` | optimal | -5820.999999 | 1.3e-10 | 18 | 0.3 |
| 1,000 | `pdhg` | optimal | -5821 | 4.1e-11 | 76560 | 1.6 |
| 5,000 | `dual-simplex` | time limit | -2622.001205 | 8.1e-01 | 42448 | 120.1 |
| 5,000 | `ipm` | optimal | -14008 | 1.1e-10 | 24 | 2.6 |
| 5,000 | `pdhg` | optimal | -14008 | 7.8e-12 | 155280 | 21.7 |
| 20,000 | `dual-simplex` | time limit | -255228.4684 | 3.5e+01 | 40108 | 120.2 |
| 20,000 | `ipm` | feasible | 7459.000101 | 1.4e-08 | 26 | 19.1 |
| 20,000 | `pdhg` | feasible | 7459.000382 | 5.1e-08 | 98578 (of which 19 polish) | 99.6 |
| 100,000 | `dual-simplex` | time limit | -2005644.115 | 2.1e+01 | 20655 | 121.1 |
| 100,000 | `ipm` | time limit | 98414.11935 | 1.2e-06 | 36 | 120.8 |
| 100,000 | `pdhg` | time limit | 98413.9697 | 3.1e-07 | 21250 (of which 9 polish) | 114.9 |

**8 of 12** solves reached the analytic optimum to a relative 1e-06 on this shape.

| engine | largest size reached, random | largest size reached, staircase |
|---|---:|---:|
| `dual-simplex` | 1,000 | 1,000 |
| `ipm` | 5,000 | 20,000 |
| `pdhg` | 100,000 | 100,000 |

**Structure is what a direct method needs, and the table shows it:** `ipm` from 5,000 to 20,000. Nothing about the solver changed between the two families. What changed between the shapes is whether the matrix has small separators, and the measurements behind that - the ordering time and the fill in the factor at the same size on both shapes - are in #193. The lesson for the section above is that its random family is a fair test of the first-order engine and an unfair one of the other two.

The 20,000-row `ipm` row is the one to read against the first measurement of this family, `bench/results/scale-staircase-81596a3.csv`, where it was a numerical failure after 300 iterations. Running that row is what found the defect fixed in #205 - the method had converged to a relative gap of 1e-7 and then iterated on a NaN that overwrote the answer - and here it reports feasible at a relative error of 1.4e-08 in 26 iterations. Same instance, same hash; the stopping rule is what changed.

#### 1f.3 A refinery planning model, by the year

Source CSV: `bench/results/scale-refinery-e134aeb.csv`  
Commit `e134aeb` · machine `Windows-AMD64` · 120.0s per solve · refinery structure

**A refinery planning model, rolled out over T periods** (`bench/runners/generate_refinery_lp.py`, #211): crude purchases, distillation throughput and crude tanks per crude; production by yields, sales and product tanks per product; distillation and unit capacities, quality budgets and delivery commitments per period; inventory balances coupling each period to the next. The operating plan is chosen first and the prices derived from the KKT conditions, so the optimum is exact by construction, as for the other two families. Rows are what the generator built - T = 12 is a monthly year, 365 a daily one, 8,760 hourly.

| periods | rows x cols | engine | status | objective | relative error | iterations | seconds |
|---:|---:|---|---|---:|---:|---:|---:|
| 12 | 1,068 x 1,656 | `dual-simplex` | optimal | -61203.88791 | 8.3e-16 | 5853 | 0.5 |
| 12 | 1,068 x 1,656 | `ipm` | optimal | -61203.88791 | 1.4e-13 | 36 | 0.3 |
| 12 | 1,068 x 1,656 | `pdhg` | optimal | -61203.88791 | 5.5e-13 | 17720 | 0.9 |
| 365 | 32,485 x 50,370 | `dual-simplex` | time limit | -2167772.561 | 3.8e-01 | 41540 | 120.7 |
| 365 | 32,485 x 50,370 | `ipm` | optimal | -1571173.846 | 6.8e-12 | 45 | 67.7 |
| 365 | 32,485 x 50,370 | `pdhg` | optimal | -1571173.846 | 1.4e-14 | 43944 (of which 5 polish) | 92.3 |
| 8760 | 779,640 x 1,208,880 | `dual-simplex` | time limit | -1217992956 | 3.2e+01 | 2413 | 136.4 |
| 8760 | 779,640 x 1,208,880 | `ipm` | time limit | -475650568.5 | 1.2e+01 | 2 | 140.0 |
| 8760 | 779,640 x 1,208,880 | `pdhg` | time limit | -36893748.68 | 1.0e-08 | 1522 | 111.8 |

**6 of 9** solves reached the analytic optimum to a relative 1e-06 on this model; the largest solved is 779,640 rows.

#### 1f.4 The same families under `algorithm=auto`


**random** (`bench/results/auto-scale-random-078cb24.csv`):

| size | engine that ran | status | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|
| 1000 | `simplex-dual+primal` | optimal | 9.4e-15 | 4271 | 0.5 |
| 5000 | `simplex-dual` | time_limit | 1.5e-06 | 111239 | 120.0 |
| 20000 | `pdhg-cpu` | time_limit | 8.8e-10 | 136302 | 120.1 |
| 100000 | `pdhg-cpu` | time_limit | 1.1e-07 | 20033 | 90.9 |

**staircase** (`bench/results/auto-scale-staircase-078cb24.csv`):

| size | engine that ran | status | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|
| 1000 | `simplex-dual+primal` | optimal | 1.6e-16 | 5608 | 0.6 |
| 5000 | `simplex-dual+primal` | optimal | 1.4e-15 | 34071 | 22.5 |
| 20000 | `ipm+crossover` | optimal | 1.8e-15 | 3904 | 23.4 |
| 100000 | `pdhg-cpu` | time_limit | 5.4e-07 | 18717 | 114.6 |

**refinery** (`bench/results/auto-scale-refinery-078cb24.csv`):

| size | engine that ran | status | relative error | iterations | seconds |
|---:|---|---|---:|---:|---:|
| 1068 x 1656 | `simplex-dual+primal` | optimal | 8.3e-16 | 5853 | 0.5 |
| 32485 x 50370 | `ipm` | optimal | 8.6e-12 | 16 | 30.0 |
| 779640 x 1208880 | `pdhg-cpu` | time_limit | 1.0e-08 | 1510 | 106.6 |

**10 of 11** solves under `auto` reached the analytic optimum to a relative 1e-06 (commit 078cb24). The engine column is what ran, which after a decline is the fallback: `pdhg-cpu` on a row that the rule table sent to the interior point means the set-up passed `ipm_setup_share` of the limit and the first-order method took the rest (#357).

#### 1f.5 A million rows on the CPU

Source CSV: `bench/results/million-cpu-b561fba.csv`  
Commit `b561fba` · machine `cloud container; Linux-x86_64; Intel(R) Xeon(R) Processor @ 2.10GHz; 4 cores; 15.7 GiB RAM` · 3600s per solve

Three generated models at a million rows or more (#751), each with its optimum exact by construction and each written from a seed; the sha256 in the CSV names the file that was solved. `verified` is `tools/verify_solution.py`'s exit code on the written solution, which parses the model itself and recomputes every residual.

| model | rows x cols | nonzeros | arm | route | status | relative error | verified | seconds | peak memory |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| transport | 1,000,000 x 2,000,000 | 4,000,000 | `ipm` | ipm | feasible | 1.9e-11 | yes | 34 | 1,865 MB |
| transport | 1,000,000 x 2,000,000 | 4,000,000 | `ipm-xover` | ipm | feasible | 1.9e-11 | yes | 3,611 | 1,870 MB |
| transport | 1,000,000 x 2,000,000 | 4,000,000 | `pdhg` | pdhg-cpu | optimal | 4.1e-14 | yes | 119 | 1,091 MB |
| staircase | 1,000,000 x 1,000,000 | 5,000,000 | `ipm` | ipm | numerical error | - | no | 23 | 1,534 MB |
| staircase | 1,000,000 x 1,000,000 | 5,000,000 | `ipm-xover` | ipm | numerical error | - | no | 23 | 1,534 MB |
| staircase | 1,000,000 x 1,000,000 | 5,000,000 | `pdhg` | pdhg-cpu | time limit | 2.0e-08 | yes | 2,542 | 1,604 MB |
| refinery | 1,007,400 x 1,892,160 | 17,940,408 | `ipm` | ipm | numerical error | - | no | 44 | 3,221 MB |
| refinery | 1,007,400 x 1,892,160 | 17,940,408 | `ipm-xover` | ipm | numerical error | - | no | 44 | 3,221 MB |
| refinery | 1,007,400 x 1,892,160 | 17,940,408 | `pdhg` | pdhg-cpu | optimal | 3.1e-14 | yes | 2,336 | 2,312 MB |

**2 of 9** arms end `optimal` and verified.

Every arm that did not finish, and the numbers that say why. `attribution` is read off the row by `bench/runners/million.py`: `iterations` when the method ran to the limit still converging, `fill` when the factor's size stopped it, `polish` when PDHG's point was not finished by the interior point, `stall` when the interior point's step collapsed short of its optimality test, `crossover` when the pivots from that point ran out of time, `overran` when the runner killed a solve still running past its backstop (1.5 times the limit plus 600 s), `memory` when the process was killed with its peak near the machine's RAM.

| model | arm | attribution | status | iterations | primal inf | dual inf | peak memory | solver message |
|---|---|---|---|---:|---:|---:|---:|---|
| transport | `ipm` | **stall** | feasible | 21 | 1.8e-08 | 5.3e-09 | 1,865 MB | the interior-point iteration stalled after 21 iterations (steps 3.4e-11 / 1.4e-11); the best iterate is reported as a feasible point |
| transport | `ipm-xover` | **crossover** | feasible | 21 | 1.8e-08 | 5.3e-09 | 1,870 MB | the interior-point iteration stalled after 21 iterations (steps 3.4e-11 / 1.4e-11); the best iterate is reported as a feasible point; crossover did not reach a vertex (time_limit after 198900 pivots, 3576.36s), the interior point's answer stands |
| staircase | `ipm` | **fill** | numerical error | 0 | 3.7e+02 | 3.0e+02 | 1,534 MB | declined: the factor of the normal equations would hold more than 175797034 nonzeros, above ipm_max_factor_nonzeros = 175797034; raise the option or use another engine |
| staircase | `ipm-xover` | **fill** | numerical error | 0 | 3.7e+02 | 3.0e+02 | 1,534 MB | declined: the factor of the normal equations would hold more than 175797034 nonzeros, above ipm_max_factor_nonzeros = 175797034; raise the option or use another engine; the best iterate (merit 5.0e+03) is attached for crossover_from_nonoptimal, and is not claimed as a point; crossover not attempted from this numerical_error answer: its scaled infeasibility 4.7e+00 / 1.0e+00 is above 1e-04, or it holds no finite point |
| staircase | `pdhg` | **polish** | time limit | 229633 | 2.6e-05 | 2.1e-12 | 1,604 MB | stopped at relative primal 1.466e-09, dual 0.000e+00, gap 1.349e-07 after 229633 iterations and 18 restarts (target 1.0e-04); the interior-point polish did not improve it (not_solved after 0 iterations: declined: the factor of the normal equations would hold more than 175797034 nonzeros, above polish_max_factor_nonzeros = 175797034; raise the option or use another engine) |
| refinery | `ipm` | **fill** | numerical error | 0 | 5.9e+02 | 3.6e+01 | 3,221 MB | declined: the factor of the normal equations would hold more than 175797034 nonzeros, above ipm_max_factor_nonzeros = 175797034; raise the option or use another engine |
| refinery | `ipm-xover` | **fill** | numerical error | 0 | 5.9e+02 | 3.6e+01 | 3,221 MB | declined: the factor of the normal equations would hold more than 175797034 nonzeros, above ipm_max_factor_nonzeros = 175797034; raise the option or use another engine; the best iterate (merit 1.5e+05) is attached for crossover_from_nonoptimal, and is not claimed as a point; crossover not attempted from this numerical_error answer: its scaled infeasibility 5.4e+01 / 4.2e+00 is above 1e-04, or it holds no finite point |

**Past the defaults.** The same models and verifier with an option changed on purpose, one CSV per experiment; `options` is what the solver was given.

| CSV | model | arm | options | status | relative error | verified | iterations | seconds | peak memory | solver message |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| `million-factor-cap-b561fba.csv` | refinery | `ipm` | `algorithm=ipm crossover=true ipm_max_factor_nonzeros=600000000` | time limit | 7.0e-03 | no | 12 | 3,628 | 8,068 MB | time limit 3598.06s reached inside the factorization, which was abandoned |
| `million-factor-cap-b561fba.csv` | staircase | `ipm` | `algorithm=ipm crossover=true ipm_max_factor_nonzeros=600000000` | time limit | 1.5e-04 | no | 65 | 3,614 | 4,347 MB | time limit 3598.86s reached |
| `million-supernodal-b561fba.csv` | staircase | `ipm` | `algorithm=ipm crossover=true ipm_max_factor_nonzeros=600000000 ipm_supernodal=true` | numerical error | - | no | 86 | 2,106 | 6,550 MB | the interior-point iteration stalled after 86 iterations (steps 4.1e-19 / 4.8e-17); the best iterate is reported as a feasible point; engine reported feasible but the returned point violates primal feasibility by 9.318e-06 (2.588e-07 relative to the scale it was measured on), above the 1.0e-07 tolerance; it is not a feasible point |
| `million-supernodal-b561fba.csv` | refinery | `ipm` | `algorithm=ipm crossover=true ipm_max_factor_nonzeros=600000000 ipm_supernodal=true` | killed | - | no | - | 6,000 | 11,806 MB | exit -9;  |

---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

Source CSV: `bench/results/miplib-ad57c03.csv`  
Commit `ad57c03` · machine `Linux-x86_64`

**14 of 30** instances reached the published optimum. **11 of 30** also PROVED it - closed the bound to within the requested gap target rather than stopping at a node or time limit.

Every row above was counted under the #188 convention: a search that meets the requested gap target reports `optimal`, because the incumbent is within the tolerance that was asked for. Only a node or time limit leaves a row unproved.
Those are different claims and are kept apart deliberately. Branch and bound here finds good incumbents far more often than it finishes the proof: reliability branching (#69) and warm-started dual node LPs (#65) do the searching, and the root cutting planes that exist (#159: Gomory mixed-integer and lifted knapsack cover, selected by score since #415) have their default decided by the measurement below, not asserted here. Collapsing the two columns would hide exactly the thing cuts are meant to improve.

**Root cuts, on versus off** (`bench/results/miplib-cuts-off.csv` and `miplib-cuts-on.csv`, both at `0018254`, 30 instances, the same time limit): with cuts on, 13 of 30 reach the published optimum and 10 prove it, against 15 and 9 with them off. Over the 29 instances that end the same way either way, the cuts take the total node count to 1.015x (per instance from 0.308x to 1.207x). The outcome changed on 1: `neos-3611689-kaihu` feasible -> optimal. Both runs were recorded under #188, where a search meeting its gap target is optimal, so the counts are the CSVs' own. Instances whose matched or proved verdict differs between the two runs: `enlight8`: objective 27.0 without cuts and 22.5 with them (matched yes -> no, proved no -> no); `neos-3611689-kaihu`: objective 119.0 without cuts and 119.0 with them (matched yes -> yes, proved no -> yes); `noswot`: objective -41.0 without cuts and -40.0 with them (matched yes -> no, proved no -> no). Cuts make every node LP dearer, because each cut is a row; on this measurement they prove +1 and match -2 against a node count of 1.015x. A proof is a closed bound and a match at the limit is an incumbent that moves between identical runs, so this is the measurement that earns `enable_root_cuts` its default of on (#415): a proof gained at that node cost, with matches lost only where the clock decides. **With cut rounds below the root as well** (`miplib-cuts-tree.csv`, `tree_cut_depth=4`, same commit): 12 of 30 reach the published optimum and 9 prove it, +0 proved and -3 matched against cuts off, node count 0.850x over the 30 instances that end the same way; verdicts that moved: `enlight8`: matched yes -> no, proved no -> no; `neos-3611689-kaihu`: matched yes -> no, proved no -> no; `noswot`: matched yes -> no, proved no -> no.

| instance | root cuts | root gap closed | root + tree cuts |
|---|---:|---:|---:|
| `b-ball` | 7 | 26.7% | 7 |
| `ej` | 0 | 0.0% | 0 |
| `enlight8` | 14 | 2.3% | 418 |
| `enlight_hard` | 30 | 4.0% | 490 |
| `f2gap40400` | 30 | 90.4% | 44 |
| `flugpl` | 1 | 2.0% | 2 |
| `gen-ip016` | 0 | 0.0% | 0 |
| `gen-ip054` | 1 | 1.0% | 2 |
| `gr4x6` | 4 | 26.2% | 49 |
| `gt2` | 13 | 91.9% | 115 |
| `k16x240b` | 20 | 6.7% | 192 |
| `markshare1` | 0 | 0.0% | 0 |
| `markshare_4_0` | 0 | 0.0% | 0 |
| `markshare_5_0` | 0 | 0.0% | 0 |
| `neos-1425699` | 0 | 0.0% | 0 |
| `neos-3072252-nete` | 30 | 3.3% | 630 |
| `neos-3611689-kaihu` | 30 | 46.7% | 581 |
| `neos-5140963-mincio` | 26 | 0.0% | 30 |
| `neos-5192052-neckar` | 2 | 0.0% | 2 |
| `neos5` | 0 | 0.0% | 0 |
| `noswot` | 17 | 0.0% | 48 |
| `opt1217` | 0 | 0.0% | 0 |
| `p0201` | 9 | 37.8% | 58 |
| `pk1` | 0 | 0.0% | 0 |
| `ran12x21` | 23 | 3.7% | 196 |
| `ran13x13` | 22 | 15.0% | 196 |
| `rlp1` | 0 | 0.0% | 0 |
| `supportcase14` | 27 | 9.4% | 107 |
| `supportcase16` | 30 | 18.8% | 113 |
| `timtab1` | 30 | 15.3% | 508 |

Root gap closed is (bound after cuts - bound before) / (final objective - bound before) on the cuts-on run, for the 30 of 30 instances whose CSV row carries the column and whose root gap was not already zero.

**Flow cover cuts (#419).** `ran12x21`, `ran13x13` and `k16x240b` share a fixed-charge / single-node flow row structure - a continuous flow variable switched on by a binary through a variable upper bound - that none of the other cut families separates. Three strengthenings of the family that separates it were tried in turn, each measured against its own same-commit baseline:

**one-row separation** (`bench/results/miplib-419-cuts-baseline.csv` and `bench/results/miplib-419-flow-cover-on.csv`, both at `91b6c08`, 30 instances, `enable_root_cuts=true` in both legs, 60s, `mip_threads=1`): 13 of 30 reach the published optimum and 9 prove it, against 13 and 9 with the family off. On the three instances #419 names: `ran12x21` 3832.0 -> 3853.0 (published 3663.999999980964); `ran13x13` 3319.0 -> 3297.0 (published 3252.0); `k16x240b` 12271.0 -> 12271.0 (published 11392.99999884458).

**a single-node flow relaxation (small multi-row aggregation)** (`bench/results/miplib-419-aggregation-baseline.csv` and `bench/results/miplib-419-aggregation-on.csv`, both at `ebafccf`, 30 instances, `enable_root_cuts=true` in both legs, 60s, `mip_threads=1`): 13 of 30 reach the published optimum and 9 prove it, against 13 and 9 with the family off. On the three instances #419 names: `ran12x21` 3812.0 -> 3853.0 (published 3663.999999980964); `ran13x13` 3319.0 -> 3297.0 (published 3252.0); `k16x240b` 12271.0 -> 12271.0 (published 11392.99999884458).

**a bounded local search over the cover choice** (`bench/results/miplib-419-localsearch-baseline.csv` and `bench/results/miplib-419-localsearch-on.csv`, both at `8358f75`, 30 instances, `enable_root_cuts=true` in both legs, 60s, `mip_threads=1`): 15 of 30 reach the published optimum and 9 prove it, against 13 and 9 with the family off. On the three instances #419 names: `ran12x21` 3839.0 -> 3724.0 (published 3663.999999980964); `ran13x13` 3385.0 -> 3319.0 (published 3252.0); `k16x240b` 12297.0 -> 12297.0 (published 11392.99999884458). The baselines are not perfectly repeatable at this 60s budget: `neos-3611689-kaihu` matched in 1 of 3 baseline legs; `noswot` matched in 2 of 3 baseline legs - ordinary run-to-run timing variance (reliability branching and the primal heuristics both make timing-sensitive choices), not a code effect. Any single round's apparent gain or loss of a MATCHED instance OTHER than the three #419 names should be read against this. **None of the three reached its published optimum in any leg run for #419** - the acceptance criterion is not met, so `enable_flow_cover_cuts` stays off by default, the same "measurement, not caution" reasoning `enable_root_cuts` already carries.

**Per-heuristic A/B (#414):** not measured on this checkout (no `bench/results/miplib-heur-*.csv`).

**Over 3 seeds** (`bench/results/summary-miplib-seeds3-ad57c03.csv`, the published file and 2 permutations of it, same commit and limit): **14 of 30** instances reach the published optimum on every seed and 15 on at least one; **9 of 30** prove it on every seed and 11 on at least one. Instances whose verdict moves with the seed, which is the noise floor a single run carries: `b-ball`, `enlight8`, `neos-3611689-kaihu`.

**The time limit decides some of these, not the solver.** A row that stops at the limit with a small gap says "needs more time than we gave it", not "cannot"; which side of the limit such a row lands on moves with the machine's speed rather than with anything about the search. The remedy is a longer limit, and the reason this table does not already use one is that the set already adds up to 20 minutes of solve time per run at this one.

Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone has found, not a proof, and scoring against one would let a wrong answer look like a record.

| instance | rows | cols | int | status | our objective | published | rel. gap | nodes | time (s) | matched | proved | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `b-ball` | 30 | 100 | 88 | optimal | -1.5 | -1.5 | 0.00e+00 | 7 | 0.1 | yes | yes | yes |
| `ej` | 1 | 3 | 3 | feasible | 41013 | 25508 | 1.00e+00 | 246049 | 60.1 | **NO** | **NO** | yes |
| `enlight8` | 64 | 128 | 128 | time_limit | inf | 27 | - | 310481 | 60.1 | **NO** | **NO** | **NO** |
| `enlight_hard` | 100 | 200 | 200 | time_limit | inf | 37 | - | 188430 | 60.1 | **NO** | **NO** | **NO** |
| `f2gap40400` | 40 | 400 | 400 | optimal | 20772 | 20772 | 9.63e-05 | 336 | 2.3 | yes | yes | yes |
| `flugpl` | 18 | 18 | 11 | optimal | 1201500 | 1201500 | 0.00e+00 | 701 | 0.0 | yes | yes | yes |
| `gen-ip016` | 24 | 28 | 28 | feasible | -9439.769515 | -9476.155197 | 6.25e-03 | 452771 | 60.2 | **NO** | **NO** | yes |
| `gen-ip054` | 27 | 30 | 30 | feasible | 6859.084635 | 6840.965642 | 9.29e-03 | 389308 | 60.1 | **NO** | **NO** | yes |
| `gr4x6` | 34 | 48 | 24 | optimal | 202.35 | 202.35 | 0.00e+00 | 44 | 0.0 | yes | yes | yes |
| `gt2` | 29 | 188 | 188 | optimal | 21166 | 21166 | 0.00e+00 | 1327 | 0.3 | yes | yes | yes |
| `k16x240b` | 256 | 480 | 240 | feasible | 12506 | 11393 | 3.94e-01 | 119921 | 60.1 | **NO** | **NO** | yes |
| `markshare1` | 6 | 62 | 50 | feasible | 24 | 1 | 1.00e+00 | 742917 | 60.2 | **NO** | **NO** | yes |
| `markshare_4_0` | 4 | 34 | 30 | feasible | 1 | 1 | 1.00e+00 | 1185510 | 60.1 | yes | **NO** | yes |
| `markshare_5_0` | 5 | 45 | 40 | feasible | 14 | 1 | 1.00e+00 | 862411 | 60.2 | **NO** | **NO** | yes |
| `neos-1425699` | 89 | 105 | 85 | optimal | 3179698977 | 3179698977 | 0.00e+00 | 3 | 0.0 | yes | yes | yes |
| `neos-3072252-nete` | 432 | 576 | 144 | feasible | 12046823 | 11807698 | 1.12e-01 | 42685 | 60.0 | **NO** | **NO** | yes |
| `neos-3611689-kaihu` | 323 | 421 | 88 | optimal | 119 | 119 | 0.00e+00 | 70614 | 51.3 | yes | yes | yes |
| `neos-5140963-mincio` | 184 | 196 | 183 | feasible | 14535 | 14393 | 1.89e-01 | 135624 | 60.0 | **NO** | **NO** | yes |
| `neos-5192052-neckar` | 57 | 180 | 24 | optimal | -11670000 | -11670000 | 0.00e+00 | 9 | 0.0 | yes | yes | yes |
| `neos5` | 63 | 63 | 53 | feasible | 15.5 | 15 | 9.68e-02 | 65418 | 60.0 | **NO** | **NO** | yes |
| `noswot` | 182 | 128 | 100 | feasible | -39 | -41.00000885 | 1.03e-01 | 169146 | 60.1 | **NO** | **NO** | yes |
| `opt1217` | 64 | 769 | 768 | feasible | -16 | -16 | 2.50e-01 | 157511 | 60.1 | yes | **NO** | yes |
| `p0201` | 133 | 201 | 201 | optimal | 7615 | 7615 | 0.00e+00 | 389 | 1.3 | yes | yes | yes |
| `pk1` | 45 | 86 | 55 | feasible | 17 | 11 | 5.27e-01 | 272062 | 60.1 | **NO** | **NO** | yes |
| `ran12x21` | 285 | 504 | 252 | feasible | 3681 | 3664 | 5.24e-02 | 66979 | 60.0 | **NO** | **NO** | yes |
| `ran13x13` | 195 | 338 | 169 | feasible | 3319 | 3252 | 4.93e-02 | 99981 | 60.0 | **NO** | **NO** | yes |
| `rlp1` | 68 | 461 | 450 | feasible | 15 | 15 | 6.67e-02 | 197074 | 60.1 | yes | **NO** | yes |
| `supportcase14` | 234 | 304 | 304 | optimal | 288 | 288 | 0.00e+00 | 47 | 0.3 | yes | yes | yes |
| `supportcase16` | 130 | 319 | 319 | optimal | 288 | 288 | 0.00e+00 | 82 | 0.5 | yes | yes | yes |
| `timtab1` | 171 | 397 | 171 | feasible | 1172343 | 764772 | 7.18e-01 | 126693 | 60.1 | **NO** | **NO** | yes |

**Not proved optimal**, named rather than dropped: `ej`, `enlight8`, `enlight_hard`, `gen-ip016`, `gen-ip054`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-3072252-nete`, `neos-5140963-mincio`, `neos5`, `noswot`, `opt1217`, `pk1`, `ran12x21`, `ran13x13`, `rlp1`, `timtab1`.

#### The same set at 600 s

Source CSV: `bench/results/miplib-600s-cca77e0.csv` (600 s per instance), beside `bench/results/miplib-ad57c03.csv` (60 s)  
Commit `54e561b`

At 600 s: **15 of 30** reach the published optimum, **9 of 30** prove it.

| instance | 60 s: status · matched · proved | 600 s: status · matched · proved · gap | verdict |
|---|---|---|---|
| `b-ball` | optimal · yes · yes | feasible · yes · no · 2.1e-01 | needs a bound (#221) |
| `ej` | feasible · no · no | feasible · no · no · 1.0e+00 | needs an incumbent (#290) |
| `enlight8` | time_limit · no · no | time_limit · no · no · 1.8e-01 | needs an incumbent (#290) |
| `enlight_hard` | time_limit · no · no | time_limit · no · no · 0.0e+00 | needs an incumbent (#290) |
| `f2gap40400` | optimal · yes · yes | optimal · yes · yes · 4.9e-07 | proved |
| `flugpl` | optimal · yes · yes | optimal · yes · yes · 1.9e-05 | proved |
| `gen-ip016` | feasible · no · no | feasible · no · no · 5.6e-03 | needs an incumbent (#290) |
| `gen-ip054` | feasible · no · no | feasible · no · no · 8.7e-03 | needs an incumbent (#290) |
| `gr4x6` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `gt2` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `k16x240b` | feasible · no · no | feasible · no · no · 3.7e-01 | needs an incumbent (#290) |
| `markshare1` | feasible · no · no | feasible · no · no · 1.0e+00 | needs an incumbent (#290) |
| `markshare_4_0` | feasible · yes · no | feasible · no · no · 1.0e+00 | needs an incumbent (#290) |
| `markshare_5_0` | feasible · no · no | feasible · no · no · 1.0e+00 | needs an incumbent (#290) |
| `neos-1425699` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `neos-3072252-nete` | feasible · no · no | feasible · no · no · 1.1e-01 | needs an incumbent (#290) |
| `neos-3611689-kaihu` | optimal · yes · yes | feasible · yes · no · 8.9e-03 | needs a bound (#221) |
| `neos-5140963-mincio` | feasible · no · no | feasible · no · no · 1.8e-01 | needs an incumbent (#290) |
| `neos-5192052-neckar` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `neos5` | feasible · no · no | feasible · yes · no · 5.0e-02 | needs a bound (#221) |
| `noswot` | feasible · no · no | feasible · yes · no · 4.9e-02 | needs a bound (#221) |
| `opt1217` | feasible · yes · no | feasible · yes · no · 2.5e-01 | needs a bound (#221) |
| `p0201` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `pk1` | feasible · no · no | feasible · no · no · 5.3e-01 | needs an incumbent (#290) |
| `ran12x21` | feasible · no · no | feasible · no · no · 4.5e-02 | needs an incumbent (#290) |
| `ran13x13` | feasible · no · no | feasible · no · no · 3.8e-02 | needs an incumbent (#290) |
| `rlp1` | feasible · yes · no | feasible · yes · no · 6.7e-02 | needs a bound (#221) |
| `supportcase14` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `supportcase16` | optimal · yes · yes | optimal · yes · yes · 0.0e+00 | proved |
| `timtab1` | feasible · no · no | feasible · no · no · 6.8e-01 | needs an incumbent (#290) |

**Needed time** (proved at 600 s, not at 60 s): none. **Needs a bound** (optimum reached at both limits, proved at neither): `b-ball`, `neos-3611689-kaihu`, `neos5`, `noswot`, `opt1217`, `rlp1`. **Needs an incumbent** (wrong answer even at 600 s): `ej`, `enlight8`, `enlight_hard`, `gen-ip016`, `gen-ip054`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-3072252-nete`, `neos-5140963-mincio`, `pk1`, `ran12x21`, `ran13x13`, `timtab1`.

#### The 60-instance tier over three seeds (#504)

Not yet run. Reproduce with:

```
python bench/runners/fetch_miplib.py --tier 2
python bench/runners/miplib.py --tier 2 --seeds 3 --time-limit 300
```

#### A/B: the node LP factor cache (#501)

Not yet run on an idle machine.

#### A/B: Feasibility Jump on the seed harness (#506)

Not yet run on an idle machine.

---

## 2b. Maros-Meszaros, the convex QP set

The 138 convex QPs of Maros and Meszaros, *A repository of convex quadratic programming
problems*, Optimization Methods and Software 11-12 (1999): the set every convex QP paper
reports. Solved by whichever QP engine was the default at the run's commit (the CSV's
`algorithm` column names it per instance: the proximal interior point `qp-ipm` since #740,
Condat-Vu `qp-condat-vu` before it and as the interior point's fallback on a numerical
error) and judged the way the published QP benchmark judges it, on primal residual, dual residual and duality gap at 1e-6
and at 1e-9, as well as against the published objective and by the independent verifier.

Source CSV: `bench/results/maros-meszaros-9094e1c.csv`  
Commit `9094e1c` · machine `Windows-AMD64` · time limit 60 s per instance · solver defaults

**138 of 138 instances** run. `optimal` on **106**; within 1e-6 relative of the published objective on **106**; accepted by the independent verifier on **120**.

| level | success, relative measures (#491) | success, absolute measures (the published report's criterion) |
|---|---:|---:|
| 1e-6 | 106 / 138 | 90 / 138 |
| 1e-9 | 96 / 138 | 33 / 138 |

Success at a level means the status is `optimal` and the primal residual, the dual residual and the duality gap are all within it, each recomputed from the QPS file and the written `.sol` by `bench/runners/qp_residuals.py` through the verifier's own MPS reader. The reference objectives are the readme's OPT column (BPMPD at default settings, eight significant digits), parsed by `bench/runners/fetch_maros_meszaros.py` into `data/maros-meszaros/reference.json` with every file's sha256.

Shifted geometric mean of solver time, shift 10 s, an instance not successful at 1e-6 charged the full limit as the published report does: **6.36 s**.

| instance | rows | cols | status | our objective | reference | rel. gap | primal rel. | dual rel. | gap rel. | iters | solver time (s) | verified | 1e-6 | 1e-9 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `aug2d` | 10000 | 20200 | optimal | 1687411.753 | 1687411.8 | 2.8e-08 | 1.1e-12 | 3.9e-14 | 4.7e-11 | 2 | 0.08 | yes | yes | yes |
| `aug2dc` | 10000 | 20200 | optimal | 1818368.066 | 1818368.1 | 1.9e-08 | 1.2e-12 | 4.0e-14 | 5.1e-11 | 2 | 0.11 | yes | yes | yes |
| `aug2dcqp` | 10000 | 20200 | optimal | 6498134.739 | 6498134.8 | 9.3e-09 | 3.6e-14 | 8.4e-16 | 9.7e-15 | 16 | 0.38 | yes | yes | yes |
| `aug2dqp` | 10000 | 20200 | optimal | 6237012.026 | 6237012.1 | 1.2e-08 | 2.2e-16 | 0.0e+00 | 7.0e-13 | 18 | 0.48 | yes | yes | yes |
| `aug3d` | 1000 | 3873 | optimal | 554.0677258 | 554.06773 | 7.6e-09 | 4.4e-16 | 2.9e-16 | 8.2e-16 | 2 | 0.01 | yes | yes | yes |
| `aug3dc` | 1000 | 3873 | optimal | 771.2624387 | 771.26244 | 1.7e-09 | 1.0e-15 | 3.2e-16 | 3.5e-14 | 2 | 0.02 | yes | yes | yes |
| `aug3dcqp` | 1000 | 3873 | optimal | 993.3621467 | 993.36215 | 3.4e-09 | 2.4e-13 | 0.0e+00 | 1.7e-10 | 11 | 0.05 | yes | yes | yes |
| `aug3dqp` | 1000 | 3873 | optimal | 675.2376715 | 675.23767 | 2.2e-09 | 7.7e-16 | 0.0e+00 | 6.4e-10 | 13 | 0.06 | yes | yes | yes |
| `boyd1` | 18 | 93261 | feasible | -61735219.64 | -61735220 | 5.8e-09 | 1.5e-14 | 1.7e-15 | 1.5e-12 | 35 | 18.82 | yes | **no** | **no** |
| `boyd2` | 186531 | 93263 | optimal | 21.256767 | 21.256767 | 2.2e-10 | 1.8e-14 | 4.1e-15 | 8.5e-10 | 69 | 35.46 | yes | yes | yes |
| `cont-050` | 2401 | 2597 | optimal | -4.563850904 | -4.5638509 | 9.5e-10 | 1.3e-15 | 0.0e+00 | 9.6e-12 | 11 | 0.09 | yes | yes | yes |
| `cont-100` | 9801 | 10197 | optimal | -4.644397869 | -4.6443979 | 6.7e-09 | 2.1e-15 | 0.0e+00 | 2.6e-11 | 12 | 0.82 | yes | yes | yes |
| `cont-101` | 10098 | 10197 | optimal | 0.1955273247 | 0.19552733 | 5.3e-09 | 7.5e-16 | 0.0e+00 | 4.5e-11 | 11 | 0.76 | yes | yes | yes |
| `cont-200` | 39601 | 40397 | optimal | -4.68487592 | -4.6848759 | 4.3e-09 | 1.0e-14 | 0.0e+00 | 1.9e-11 | 13 | 6.52 | yes | yes | yes |
| `cont-201` | 40198 | 40397 | optimal | 0.192483373 | 0.19248337 | 3.0e-09 | 9.5e-16 | 0.0e+00 | 1.0e-11 | 13 | 6.71 | yes | yes | yes |
| `cont-300` | 90298 | 90597 | optimal | 0.1915122861 | 0.19151232 | 3.4e-08 | 1.1e-15 | 0.0e+00 | 1.5e-10 | 14 | 30.15 | yes | yes | yes |
| `cvxqp1l` | 5000 | 10000 | model_error | 0 | 1.087048e+08 | 1.0e+00 | - | - | - | 0 | 0.05 | - | **no** | **no** |
| `cvxqp1m` | 500 | 1000 | optimal | 1087511.567 | 1087511.6 | 3.0e-08 | 9.4e-12 | 0.0e+00 | 3.1e-11 | 14 | 0.10 | yes | yes | yes |
| `cvxqp1s` | 50 | 100 | optimal | 11590.71812 | 11590.718 | 1.0e-08 | 3.4e-14 | 0.0e+00 | 2.0e-12 | 10 | 0.00 | yes | yes | yes |
| `cvxqp2l` | 2500 | 10000 | model_error | 0 | 81842458 | 1.0e+00 | - | - | - | 0 | 0.05 | - | **no** | **no** |
| `cvxqp2m` | 250 | 1000 | optimal | 820155.431 | 820155.43 | 1.2e-09 | 2.2e-13 | 0.0e+00 | 7.7e-14 | 13 | 0.06 | yes | yes | yes |
| `cvxqp2s` | 25 | 100 | optimal | 8120.940477 | 8120.9405 | 2.8e-09 | 2.9e-13 | 0.0e+00 | 6.4e-12 | 12 | 0.00 | yes | yes | yes |
| `cvxqp3l` | 7500 | 10000 | model_error | 0 | 1.157111e+08 | 1.0e+00 | - | - | - | 0 | 0.05 | - | **no** | **no** |
| `cvxqp3m` | 750 | 1000 | optimal | 1362828.742 | 1362828.7 | 3.1e-08 | 6.5e-15 | 0.0e+00 | 3.0e-12 | 284100 | 5.92 | yes | yes | yes |
| `cvxqp3s` | 75 | 100 | optimal | 11943.4322 | 11943.432 | 1.7e-08 | 1.2e-12 | 0.0e+00 | 1.6e-12 | 10 | 0.00 | yes | yes | yes |
| `dpklo1` | 77 | 133 | optimal | 0.3700962171 | 0.37009622 | 2.9e-09 | 3.9e-16 | 3.5e-16 | 1.3e-15 | 2 | 0.00 | yes | yes | yes |
| `dtoc3` | 9998 | 14999 | optimal | 235.262481 | 235.26248 | 4.5e-09 | 1.6e-14 | 2.2e-14 | 2.0e-10 | 3 | 0.04 | yes | yes | yes |
| `dual1` | 1 | 85 | optimal | 0.03501296614 | 0.035012966 | 1.4e-10 | 1.0e-15 | 0.0e+00 | 4.3e-10 | 16 | 0.01 | yes | yes | yes |
| `dual2` | 1 | 96 | optimal | 0.03373367613 | 0.033733676 | 1.3e-10 | 1.8e-15 | 0.0e+00 | 1.4e-10 | 14 | 0.01 | yes | yes | yes |
| `dual3` | 1 | 111 | optimal | 0.1357558371 | 0.13575584 | 2.9e-09 | 1.9e-15 | 0.0e+00 | 4.5e-10 | 18 | 0.02 | yes | yes | yes |
| `dual4` | 1 | 75 | optimal | 0.7460908418 | 0.74609084 | 1.8e-09 | 2.2e-15 | 0.0e+00 | 1.1e-11 | 16 | 0.01 | yes | yes | yes |
| `dualc1` | 215 | 9 | optimal | 6155.250829 | 6155.2508 | 4.8e-09 | 2.1e-17 | 0.0e+00 | 1.3e-11 | 14 | 0.00 | yes | yes | yes |
| `dualc2` | 229 | 7 | optimal | 3551.307693 | 3551.3077 | 2.1e-09 | 1.0e-15 | 0.0e+00 | 2.9e-11 | 10 | 0.00 | yes | yes | yes |
| `dualc5` | 278 | 8 | optimal | 427.2323268 | 427.23233 | 7.4e-09 | 2.7e-16 | 0.0e+00 | 1.4e-10 | 9 | 0.00 | yes | yes | yes |
| `dualc8` | 503 | 8 | optimal | 18309.35883 | 18309.359 | 9.1e-09 | 9.2e-16 | 0.0e+00 | 2.9e-11 | 10 | 0.00 | yes | yes | yes |
| `exdata` | 3001 | 3000 | optimal | -141.8434322 | -141.84343 | 1.5e-08 | 9.9e-15 | 3.2e-12 | 1.8e-10 | 14 | 9.43 | yes | yes | yes |
| `genhs28` | 8 | 10 | optimal | 0.9271736944 | 0.92717369 | 4.4e-09 | 1.0e-09 | 6.7e-10 | 3.1e-10 | 1 | 0.00 | yes | yes | **no** |
| `gouldqp2` | 349 | 699 | optimal | 0.0001842745168 | 0.00018427534 | 8.2e-10 | 1.3e-16 | 0.0e+00 | 5.7e-11 | 14 | 0.00 | yes | yes | yes |
| `gouldqp3` | 349 | 699 | optimal | 2.062784038 | 2.062784 | 1.8e-08 | 1.5e-13 | 0.0e+00 | 8.6e-07 | 7 | 0.00 | yes | yes | **no** |
| `hs118` | 17 | 15 | optimal | 664.82045 | 664.82045 | 1.9e-12 | 0.0e+00 | 0.0e+00 | 3.1e-12 | 10 | 0.00 | yes | yes | yes |
| `hs21` | 1 | 2 | optimal | -99.96 | -99.96 | 4.0e-13 | 0.0e+00 | 0.0e+00 | 4.2e-13 | 9 | 0.00 | yes | yes | yes |
| `hs268` | 5 | 5 | optimal | 2.704291546e-08 | 5.7310705e-07 | 5.5e-07 | 0.0e+00 | 1.5e-15 | 5.6e-08 | 11 | 0.00 | yes | yes | **no** |
| `hs35` | 1 | 3 | optimal | 0.1111111118 | 0.11111111 | 1.8e-09 | 0.0e+00 | 0.0e+00 | 9.9e-10 | 6 | 0.00 | yes | yes | yes |
| `hs35mod` | 1 | 3 | optimal | 0.2500000013 | 0.25 | 1.3e-09 | 0.0e+00 | 0.0e+00 | 2.8e-09 | 11 | 0.00 | yes | yes | **no** |
| `hs51` | 3 | 5 | optimal | 0 | 8.8817842e-16 | 8.9e-16 | 3.9e-16 | 2.3e-16 | 1.8e-15 | 2 | 0.00 | yes | yes | yes |
| `hs52` | 3 | 5 | optimal | 5.326647564 | 5.3266476 | 6.7e-09 | 4.9e-17 | 1.1e-16 | 0.0e+00 | 2 | 0.00 | yes | yes | yes |
| `hs53` | 3 | 5 | optimal | 4.093023256 | 4.0930233 | 1.1e-08 | 0.0e+00 | 0.0e+00 | 2.5e-11 | 5 | 0.00 | yes | yes | yes |
| `hs76` | 3 | 4 | optimal | -4.681818182 | -4.6818182 | 3.9e-09 | 0.0e+00 | 0.0e+00 | 3.3e-11 | 7 | 0.00 | yes | yes | yes |
| `hues-mod` | 2 | 10000 | feasible | 34824463.87 | 34824690 | 6.5e-06 | 3.6e-13 | 8.2e-16 | 2.1e-14 | 15 | 0.13 | yes | **no** | **no** |
| `huestis` | 2 | 10000 | feasible | 3.482446387e+11 | 3.482469e+11 | 6.5e-06 | 3.6e-13 | 2.0e-07 | 2.2e-14 | 16 | 0.12 | yes | **no** | **no** |
| `ksip` | 1001 | 20 | optimal | 0.5757979412 | 0.57579794 | 1.2e-09 | 3.2e-12 | 1.9e-13 | 4.0e-11 | 16 | 0.04 | yes | yes | yes |
| `laser` | 1000 | 1002 | optimal | 2409601.346 | 2409601.4 | 2.3e-08 | 3.9e-14 | 5.2e-14 | 6.1e-14 | 17 | 0.01 | yes | yes | yes |
| `liswet1` | 10000 | 10002 | iteration_limit | 25.16085239 | 36.122402 | 3.0e-01 | 1.0e-07 | 1.0e-12 | 8.1e-03 | 200 | 1.22 | - | **no** | **no** |
| `liswet10` | 10000 | 10002 | iteration_limit | 25.01344513 | 49.485785 | 4.9e-01 | 8.9e-08 | 5.1e-13 | 9.7e-03 | 200 | 1.16 | - | **no** | **no** |
| `liswet11` | 10000 | 10002 | iteration_limit | 25.21156157 | 49.523957 | 4.9e-01 | 2.9e-07 | 2.0e-12 | 7.3e-02 | 200 | 1.13 | - | **no** | **no** |
| `liswet12` | 10000 | 10002 | iteration_limit | 26.66778075 | 1736.9274 | 9.8e-01 | 1.4e-06 | 8.6e-12 | 4.1e-01 | 200 | 1.18 | - | **no** | **no** |
| `liswet2` | 10000 | 10002 | iteration_limit | 24.99803852 | 24.998076 | 1.5e-06 | 6.8e-10 | 1.4e-14 | 4.6e-07 | 200 | 1.18 | - | **no** | **no** |
| `liswet3` | 10000 | 10002 | optimal | 25.00122049 | 25.00122 | 2.0e-08 | 8.6e-10 | 4.9e-13 | 3.4e-08 | 13 | 0.11 | yes | yes | **no** |
| `liswet4` | 10000 | 10002 | optimal | 25.00011139 | 25.000112 | 2.4e-08 | 2.5e-10 | 8.3e-15 | 2.7e-08 | 58 | 0.36 | yes | yes | **no** |
| `liswet5` | 10000 | 10002 | optimal | 25.03425248 | 25.034253 | 2.1e-08 | 3.1e-10 | 2.6e-14 | 3.6e-08 | 18 | 0.13 | yes | yes | **no** |
| `liswet6` | 10000 | 10002 | optimal | 24.99574548 | 24.995748 | 1.0e-07 | 8.3e-10 | 3.7e-14 | 7.8e-08 | 37 | 0.26 | yes | yes | **no** |
| `liswet7` | 10000 | 10002 | time_limit | 25.00281237 | 498.84089 | 9.5e-01 | 9.9e-08 | 1.4e-03 | 5.9e-04 | 310597 | 59.07 | - | **no** | **no** |
| `liswet8` | 10000 | 10002 | iteration_limit | 25.02591285 | 7144.7006 | 1.0e+00 | 3.9e-07 | 2.5e-12 | 1.5e-01 | 200 | 3.62 | - | **no** | **no** |
| `liswet9` | 10000 | 10002 | iteration_limit | 25.63110186 | 1963.2513 | 9.9e-01 | 1.6e-06 | 7.7e-12 | 5.5e-01 | 200 | 1.60 | - | **no** | **no** |
| `lotschd` | 7 | 12 | optimal | 2398.415891 | 2398.4159 | 3.6e-09 | 4.5e-13 | 2.6e-13 | 1.4e-12 | 8 | 0.00 | yes | yes | yes |
| `mosarqp1` | 700 | 2500 | optimal | -952.875443 | -952.87544 | 3.2e-09 | 3.6e-13 | 7.9e-13 | 9.1e-12 | 10 | 0.02 | yes | yes | yes |
| `mosarqp2` | 600 | 900 | optimal | -1597.482118 | -1597.4821 | 1.1e-08 | 5.3e-13 | 1.0e-13 | 5.6e-12 | 10 | 0.02 | yes | yes | yes |
| `powell20` | 10000 | 10000 | feasible | 5.208958281e+10 | 5.2089583e+10 | 3.7e-09 | 3.6e-14 | 1.2e-12 | 8.5e-11 | 195650 | 44.84 | yes | **no** | **no** |
| `primal1` | 85 | 325 | optimal | -0.03501296558 | -0.035012965 | 5.8e-10 | 0.0e+00 | 1.3e-13 | 7.7e-10 | 13 | 0.01 | yes | yes | yes |
| `primal2` | 96 | 649 | optimal | -0.0337336761 | -0.033733676 | 1.0e-10 | 0.0e+00 | 7.6e-15 | 2.0e-11 | 12 | 0.02 | yes | yes | yes |
| `primal3` | 111 | 745 | optimal | -0.1357558365 | -0.13575584 | 3.5e-09 | 0.0e+00 | 6.2e-14 | 6.0e-10 | 13 | 0.05 | yes | yes | yes |
| `primal4` | 75 | 1489 | optimal | -0.7460908417 | -0.74609083 | 1.2e-08 | 0.0e+00 | 6.3e-14 | 5.9e-11 | 11 | 0.03 | yes | yes | yes |
| `primalc1` | 9 | 230 | feasible | -6155.250829 | -6155.2508 | 4.8e-09 | 5.7e-16 | 1.3e-16 | 3.2e-10 | 19 | 0.00 | yes | **no** | **no** |
| `primalc2` | 7 | 231 | optimal | -3551.307693 | -3551.3077 | 2.1e-09 | 1.3e-16 | 1.6e-18 | 4.2e-14 | 14 | 0.00 | yes | yes | yes |
| `primalc5` | 8 | 287 | optimal | -427.2323266 | -427.23233 | 7.9e-09 | 0.0e+00 | 1.9e-15 | 5.3e-10 | 19 | 0.00 | yes | yes | yes |
| `primalc8` | 8 | 520 | optimal | -18309.42979 | -18309.43 | 1.2e-08 | 4.5e-16 | 1.8e-17 | 1.4e-13 | 19 | 0.01 | yes | yes | yes |
| `q25fv47` | 820 | 1571 | optimal | 13744447.89 | 13744448 | 7.6e-09 | 2.7e-16 | 4.1e-17 | 2.7e-16 | 36 | 0.56 | yes | yes | yes |
| `qadlittl` | 56 | 97 | optimal | 480318.8585 | 480318.86 | 3.0e-09 | 1.9e-16 | 3.4e-13 | 2.5e-13 | 15 | 0.00 | yes | yes | yes |
| `qafiro` | 27 | 32 | optimal | -1.590781794 | -1.5907818 | 3.9e-09 | 3.2e-16 | 0.0e+00 | 2.3e-10 | 11 | 0.00 | yes | yes | yes |
| `qbandm` | 305 | 472 | optimal | 16352.34204 | 16352.342 | 2.2e-09 | 7.8e-14 | 1.2e-11 | 1.3e-11 | 21 | 0.01 | yes | yes | yes |
| `qbeaconf` | 173 | 262 | optimal | 164712.0601 | 164712.06 | 9.1e-10 | 1.6e-15 | 8.6e-19 | 6.0e-15 | 18 | 0.01 | yes | yes | yes |
| `qbore3d` | 233 | 315 | optimal | 3100.200802 | 3100.2008 | 5.7e-10 | 3.4e-15 | 1.6e-16 | 8.9e-12 | 20 | 0.01 | yes | yes | yes |
| `qbrandy` | 220 | 249 | optimal | 28375.11486 | 28375.115 | 5.1e-09 | 1.4e-14 | 1.3e-14 | 2.4e-12 | 19 | 0.01 | yes | yes | yes |
| `qcapri` | 271 | 353 | feasible | 66793293.26 | 66793293 | 3.9e-09 | 6.6e-15 | 2.1e-17 | 9.6e-11 | 46 | 0.02 | yes | **no** | **no** |
| `qe226` | 223 | 282 | optimal | 212.6534329 | 212.65343 | 1.4e-08 | 3.0e-17 | 1.3e-13 | 2.8e-10 | 23 | 0.02 | yes | yes | yes |
| `qetamacr` | 400 | 688 | optimal | 86760.36963 | 86760.37 | 4.3e-09 | 6.1e-15 | 6.9e-18 | 4.5e-12 | 36 | 0.05 | yes | yes | yes |
| `qfffff80` | 524 | 854 | feasible | 873147.4605 | 873147.47 | 1.1e-08 | 6.2e-17 | 1.9e-11 | 4.5e-11 | 188 | 0.39 | yes | **no** | **no** |
| `qforplan` | 161 | 421 | feasible | 7456631461 | 7.4566315e+09 | 5.3e-09 | 2.5e-16 | 2.9e-15 | 8.8e-15 | 29 | 0.02 | yes | **no** | **no** |
| `qgfrdxpn` | 616 | 1092 | feasible | 1.007905849e+11 | 1.0079059e+11 | 5.1e-08 | 3.3e-15 | 1.6e-18 | 7.9e-15 | 36 | 0.01 | yes | **no** | **no** |
| `qgrow15` | 300 | 645 | feasible | -101693640.5 | -1.0169364e+08 | 4.6e-09 | 5.7e-16 | 6.3e-14 | 5.5e-14 | 28 | 0.02 | yes | **no** | **no** |
| `qgrow22` | 440 | 946 | optimal | -149628953.5 | -1.4962895e+08 | 2.3e-08 | 2.9e-16 | 5.8e-16 | 1.6e-15 | 34 | 0.05 | yes | yes | yes |
| `qgrow7` | 140 | 301 | optimal | -42798713.87 | -42798714 | 3.0e-09 | 2.7e-16 | 9.3e-16 | 1.9e-14 | 27 | 0.01 | yes | yes | yes |
| `qisrael` | 174 | 142 | optimal | 25347837.79 | 25347838 | 8.3e-09 | 0.0e+00 | 3.1e-17 | 7.1e-15 | 30 | 0.02 | yes | yes | yes |
| `qpcblend` | 74 | 83 | optimal | -0.007842543019 | -0.0078425409 | 2.1e-09 | 2.0e-11 | 0.0e+00 | 7.4e-11 | 16 | 0.00 | yes | yes | yes |
| `qpcboei1` | 351 | 384 | optimal | 11503914.01 | 11503914 | 8.5e-10 | 3.3e-18 | 7.0e-18 | 3.4e-14 | 29 | 0.02 | yes | yes | yes |
| `qpcboei2` | 166 | 143 | optimal | 8171962.244 | 8171962.3 | 6.8e-09 | 5.8e-16 | 1.8e-21 | 1.6e-13 | 30 | 0.00 | yes | yes | yes |
| `qpcstair` | 356 | 467 | optimal | 6204387.476 | 6204387.5 | 3.9e-09 | 1.4e-13 | 1.5e-16 | 2.0e-13 | 25 | 0.02 | yes | yes | yes |
| `qpilotno` | 975 | 2172 | iteration_limit | 3178009.673 | 4728586.9 | 3.3e-01 | 1.3e-01 | 2.3e-04 | 1.0e+00 | 1000000 | 38.56 | - | **no** | **no** |
| `qptest` | 2 | 2 | optimal | 4.371875 | 4.371875 | 3.1e-12 | 0.0e+00 | 0.0e+00 | 9.1e-12 | 8 | 0.00 | yes | yes | yes |
| `qrecipe` | 91 | 180 | optimal | -266.616 | -266.616 | 1.7e-10 | 5.8e-15 | 1.7e-14 | 3.3e-10 | 18 | 0.00 | yes | yes | yes |
| `qsc205` | 205 | 203 | optimal | -0.005813953469 | -0.0058139518 | 1.7e-09 | 3.3e-16 | 4.2e-15 | 2.1e-11 | 14 | 0.00 | yes | yes | yes |
| `qscagr25` | 471 | 500 | optimal | 201737938.4 | 2.0173794e+08 | 8.1e-09 | 1.6e-16 | 3.2e-15 | 1.8e-15 | 20 | 0.00 | yes | yes | yes |
| `qscagr7` | 129 | 140 | optimal | 26865948.59 | 26865949 | 1.5e-08 | 2.0e-16 | 9.1e-16 | 1.6e-14 | 21 | 0.00 | yes | yes | yes |
| `qscfxm1` | 330 | 457 | optimal | 16882691.64 | 16882692 | 2.1e-08 | 4.1e-17 | 1.3e-14 | 3.1e-13 | 29 | 0.02 | yes | yes | yes |
| `qscfxm2` | 660 | 914 | optimal | 27776161.58 | 27776162 | 1.5e-08 | 7.4e-15 | 1.8e-16 | 5.4e-16 | 37 | 0.04 | yes | yes | yes |
| `qscfxm3` | 990 | 1371 | optimal | 30816354.47 | 30816355 | 1.7e-08 | 6.8e-15 | 1.3e-14 | 3.0e-14 | 39 | 0.06 | yes | yes | yes |
| `qscorpio` | 388 | 358 | optimal | 1880.509553 | 1880.5096 | 2.5e-08 | 3.8e-12 | 4.0e-17 | 6.1e-11 | 17 | 0.00 | yes | yes | yes |
| `qscrs8` | 490 | 1169 | optimal | 904.5600139 | 904.56001 | 4.3e-09 | 1.3e-15 | 2.0e-16 | 1.0e-10 | 30 | 0.01 | yes | yes | yes |
| `qscsd1` | 77 | 760 | optimal | 8.666666675 | 8.6666667 | 2.9e-09 | 7.4e-16 | 0.0e+00 | 1.1e-10 | 13 | 0.01 | yes | yes | yes |
| `qscsd6` | 147 | 1350 | optimal | 50.80821391 | 50.808214 | 1.7e-09 | 5.1e-13 | 0.0e+00 | 3.5e-10 | 16 | 0.01 | yes | yes | yes |
| `qscsd8` | 397 | 2750 | optimal | 940.7635742 | 940.76357 | 4.5e-09 | 1.8e-13 | 0.0e+00 | 6.3e-11 | 14 | 0.02 | yes | yes | yes |
| `qsctap1` | 300 | 480 | optimal | 1415.861111 | 1415.8611 | 7.9e-09 | 3.1e-15 | 0.0e+00 | 1.3e-11 | 23 | 0.01 | yes | yes | yes |
| `qsctap2` | 1090 | 1880 | optimal | 1735.026498 | 1735.0265 | 9.7e-10 | 1.4e-14 | 0.0e+00 | 6.0e-10 | 21 | 0.03 | yes | yes | yes |
| `qsctap3` | 1480 | 2480 | optimal | 1438.754681 | 1438.7547 | 1.3e-08 | 4.1e-15 | 0.0e+00 | 5.4e-12 | 23 | 0.05 | yes | yes | yes |
| `qseba` | 515 | 1028 | feasible | 81481800.36 | 81481801 | 7.9e-09 | 9.0e-15 | 3.4e-14 | 1.1e-12 | 37 | 0.02 | yes | **no** | **no** |
| `qshare1b` | 117 | 225 | feasible | 720078.3182 | 720078.32 | 2.6e-09 | 3.2e-16 | 2.5e-15 | 2.0e-12 | 33 | 0.00 | yes | **no** | **no** |
| `qshare2b` | 96 | 79 | optimal | 11703.69172 | 11703.692 | 2.4e-08 | 3.6e-15 | 3.9e-13 | 5.7e-12 | 19 | 0.00 | yes | yes | yes |
| `qshell` | 536 | 1775 | feasible | 1.572636843e+12 | 1.5726368e+12 | 2.7e-08 | 4.1e-16 | 5.8e-17 | 2.2e-15 | 35 | 0.04 | yes | **no** | **no** |
| `qship04l` | 402 | 2118 | optimal | 2420015.534 | 2420015.5 | 1.4e-08 | 3.1e-12 | 5.8e-17 | 1.6e-12 | 17 | 0.02 | yes | yes | yes |
| `qship04s` | 402 | 1458 | optimal | 2424993.673 | 2424993.7 | 1.1e-08 | 1.1e-13 | 1.8e-16 | 1.4e-11 | 17 | 0.01 | yes | yes | yes |
| `qship08l` | 778 | 4283 | optimal | 2376040.617 | 2376040.6 | 7.0e-09 | 2.4e-13 | 1.0e-17 | 4.7e-13 | 23 | 0.34 | yes | yes | yes |
| `qship08s` | 778 | 2387 | optimal | 2385728.851 | 2385728.9 | 2.0e-08 | 8.9e-12 | 2.6e-16 | 7.5e-12 | 22 | 0.07 | yes | yes | yes |
| `qship12l` | 1151 | 5427 | time_limit | 3018876.63 | 3018876.6 | 1.0e-08 | 9.3e-11 | 3.2e-07 | 3.3e-07 | 225002 | 59.28 | - | **no** | **no** |
| `qship12s` | 1151 | 2763 | feasible | 3056962.249 | 3056962.3 | 1.7e-08 | 1.3e-13 | 1.4e-17 | 5.8e-11 | 28 | 0.07 | yes | **no** | **no** |
| `qsierra` | 1227 | 2036 | iteration_limit | 122699497.3 | 23750458 | 4.2e+00 | 4.1e-02 | 0.0e+00 | 1.0e+00 | 1000000 | 44.45 | - | **no** | **no** |
| `qstair` | 356 | 467 | optimal | 7985452.756 | 7985452.8 | 5.5e-09 | 1.9e-17 | 1.5e-17 | 8.0e-14 | 28 | 0.02 | yes | yes | yes |
| `qstandat` | 359 | 1075 | optimal | 6411.838389 | 6411.8384 | 1.7e-09 | 1.8e-16 | 2.2e-16 | 8.1e-11 | 24 | 0.01 | yes | yes | yes |
| `s268` | 5 | 5 | optimal | 2.704291546e-08 | 5.7310705e-07 | 5.5e-07 | 0.0e+00 | 1.5e-15 | 5.6e-08 | 11 | 0.00 | yes | yes | **no** |
| `stadat1` | 3999 | 2001 | time_limit | -28708039.02 | -28526864 | 6.4e-03 | 2.5e-04 | 1.6e-05 | 1.6e-04 | 883547 | 59.86 | - | **no** | **no** |
| `stadat2` | 3999 | 2001 | optimal | -32.62666486 | -32.626665 | 4.2e-09 | 0.0e+00 | 8.2e-10 | 1.4e-09 | 17 | 0.03 | yes | yes | **no** |
| `stadat3` | 7999 | 4001 | optimal | -35.77945294 | -35.779453 | 1.6e-09 | 0.0e+00 | 3.0e-10 | 5.7e-10 | 19 | 0.08 | yes | yes | yes |
| `stcqp1` | 2052 | 4097 | optimal | 155143.5547 | 155143.56 | 3.4e-08 | 4.1e-15 | 0.0e+00 | 3.6e-14 | 10 | 0.05 | yes | yes | yes |
| `stcqp2` | 2052 | 4097 | optimal | 22327.31327 | 22327.313 | 1.2e-08 | 3.0e-13 | 0.0e+00 | 6.5e-13 | 10 | 0.19 | yes | yes | yes |
| `tame` | 1 | 2 | optimal | 0 | 0 | 0.0e+00 | 3.1e-12 | 0.0e+00 | 1.6e-10 | 5 | 0.00 | yes | yes | yes |
| `ubh1` | 12000 | 18009 | iteration_limit | 1.11600082 | 1.1160008 | 1.8e-08 | 1.9e-16 | 6.2e-11 | 1.2e-04 | 200 | 1.76 | - | **no** | **no** |
| `values` | 1 | 202 | model_error | 0 | -1.3966211 | 1.0e+00 | - | - | - | 0 | 0.00 | - | **no** | **no** |
| `yao` | 2000 | 2002 | iteration_limit | 59.63179688 | 197.70426 | 7.0e-01 | 3.0e-06 | 2.0e-13 | 3.5e-01 | 200 | 0.22 | - | **no** | **no** |
| `zecevic2` | 2 | 2 | optimal | -4.125 | -4.125 | 1.1e-11 | 8.0e-12 | 0.0e+00 | 8.4e-10 | 6 | 0.00 | yes | yes | yes |

Every failure, named:

- **Did not reach `optimal`** (32): `boyd1`, `cvxqp1l`, `cvxqp2l`, `cvxqp3l`, `hues-mod`, `huestis`, `liswet1`, `liswet10`, `liswet11`, `liswet12`, `liswet2`, `liswet7`, `liswet8`, `liswet9`, `powell20`, `primalc1`, `qcapri`, `qfffff80`, `qforplan`, `qgfrdxpn`, `qgrow15`, `qpilotno`, `qseba`, `qshare1b`, `qshell`, `qship12l`, `qship12s`, `qsierra`, `stadat1`, `ubh1`, `values`, `yao`.
- **`optimal` but not successful at 1e-9 relative** (10): `genhs28`, `gouldqp3`, `hs268`, `hs35mod`, `liswet3`, `liswet4`, `liswet5`, `liswet6`, `s268`, `stadat2`.

---

## 2c. The standard pooling instances, non-convex

Haverly 1-3, Ben-Tal 4-5, Foulds 2-5 and Adhya 1-4, each in the P-, Q- and PQ-formulation
(`data/pooling/`, generated by `bench/runners/pooling_models.py` from Alfaki and Haugland's
published data; sources and the proven global optima in `data/pooling/reference.json`).
Bilinear quality terms make every one a non-convex QCQP. A run passes only on `optimal` at
the published optimum with a solution the independent verifier accepts; a model the solver
refuses is listed as refused, not dropped (`bench/runners/pooling.py`).

Source CSV: `bench/results/pooling-5a39fb2.csv`  
Commit `5a39fb2` · machine `Windows-AMD64` · time limit 60 s per run · `nonconvex=global` (set by the runner, #516), otherwise solver defaults

**32 of 39 runs** (13 instances) reached the published global optimum within 1e-4 relative, claimed `optimal`, and were accepted by `tools/verify_solution.py` against the original non-convex model.

| instance | form | status | our objective | global optimum | proven bound | root bound | nodes | time (s) | verified |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `adhya1` | P | optimal | -549.7948331 | -549.8030502 | -549.8487943 | -989.18928 | 3601 | 5.50 | yes |
| `adhya1` | PQ | optimal | -549.765337 | -549.8030502 | -549.8199626 | -840.27056 | 181 | 0.28 | yes |
| `adhya1` | Q | optimal | -549.7938835 | -549.8030502 | -549.847225 | -1335 | 5699 | 5.30 | yes |
| `adhya2` | P | optimal | -549.7707537 | -549.8030502 | -549.8256722 | -847.00386 | 13359 | 24.88 | yes |
| `adhya2` | PQ | optimal | -549.7936477 | -549.8030502 | -549.8479255 | -574.78261 | 149 | 0.20 | yes |
| `adhya2` | Q | optimal | -549.7841738 | -549.8030502 | -549.8387989 | -1335 | 5347 | 4.54 | yes |
| `adhya3` | P | feasible | -558.1999612 | -561.0446875 | -578.0302274 | -876.20685 | 20917 | 60.04 | yes |
| `adhya3` | PQ | optimal | -561.0446804 | -561.0446875 | -561.0446909 | -574.78261 | 130 | 0.33 | yes |
| `adhya3` | Q | feasible | -559.2204679 | -561.0446875 | -662.3196974 | -1335 | 21838 | 60.03 | yes |
| `adhya4` | P | optimal | -877.6417921 | -877.6457399 | -877.6767281 | -992.66728 | 890 | 1.48 | yes |
| `adhya4` | PQ | optimal | -877.6457443 | -877.6457399 | -877.6462669 | -961.93218 | 40 | 0.12 | yes |
| `adhya4` | Q | feasible | -857.8808973 | -877.6457399 | -1023.391537 | -1345 | 27903 | 60.04 | yes |
| `bental4` | P | optimal | -450 | -450 | -450 | -550 | 3 | 0.02 | yes |
| `bental4` | PQ | optimal | -450 | -450 | -450 | -550 | 3 | 0.03 | yes |
| `bental4` | Q | optimal | -450.0000003 | -450 | -450.0132749 | -2933.3333 | 197 | 0.11 | yes |
| `bental5` | P | optimal | -3500 | -3500 | -3500 | -3500 | 1 | 0.03 | yes |
| `bental5` | PQ | optimal | -3500 | -3500 | -3500 | -3500 | 1 | 0.03 | yes |
| `bental5` | Q | feasible | -3500 | -3500 | -7035.831399 | -9700 | 28251 | 60.04 | yes |
| `foulds2` | P | optimal | -1100 | -1100 | -1100 | -1100 | 1 | 0.03 | yes |
| `foulds2` | PQ | optimal | -1100 | -1100 | -1100 | -1100 | 1 | 0.03 | yes |
| `foulds2` | Q | optimal | -1100.000001 | -1100 | -1100.107363 | -6900 | 15523 | 12.40 | yes |
| `foulds3` | P | optimal | -8 | -8 | -8 | -8 | 1 | 0.04 | yes |
| `foulds3` | PQ | optimal | -8 | -8 | -8 | -8 | 1 | 0.08 | yes |
| `foulds3` | Q | feasible | -8 | -8 | -260 | -260 | 9499 | 60.05 | yes |
| `foulds4` | P | optimal | -8 | -8 | -8 | -8 | 2 | 0.06 | yes |
| `foulds4` | PQ | optimal | -8 | -8 | -8 | -8 | 1 | 0.09 | yes |
| `foulds4` | Q | feasible | -8 | -8 | -260 | -260 | 9543 | 60.04 | yes |
| `foulds5` | P | optimal | -8 | -8 | -8 | -8 | 1 | 0.03 | yes |
| `foulds5` | PQ | optimal | -8 | -8 | -8 | -8 | 2 | 0.12 | yes |
| `foulds5` | Q | feasible | -8 | -8 | -258.5134801 | -260 | 6580 | 60.04 | yes |
| `haverly1` | P | optimal | -400 | -400 | -400 | -500 | 3 | 0.02 | yes |
| `haverly1` | PQ | optimal | -400 | -400 | -400 | -500 | 3 | 0.03 | yes |
| `haverly1` | Q | optimal | -400.0000011 | -400 | -400.0000011 | -2450 | 41 | 0.05 | yes |
| `haverly2` | P | optimal | -600 | -600 | -600 | -1000 | 7 | 0.03 | yes |
| `haverly2` | PQ | optimal | -600 | -600 | -600 | -1000 | 3 | 0.03 | yes |
| `haverly2` | Q | optimal | -600 | -600 | -600.0000002 | -4700 | 33 | 0.04 | yes |
| `haverly3` | P | optimal | -750 | -750 | -750 | -800 | 7 | 0.03 | yes |
| `haverly3` | PQ | optimal | -750 | -750 | -750 | -800 | 3 | 0.03 | yes |
| `haverly3` | Q | optimal | -750.0000011 | -750 | -750.0000011 | -2450 | 29 | 0.05 | yes |

#### Root relaxation: PQ against P

| instance | P root bound | Q root bound | PQ root bound | global optimum | P root gap closed by PQ |
|---|---:|---:|---:|---:|---:|
| `adhya1` | -989.18928 | -1335 | -840.27056 | -549.8030502 | 33.9 % |
| `adhya2` | -847.00386 | -1335 | -574.78261 | -549.8030502 | 91.6 % |
| `adhya3` | -876.20685 | -1335 | -574.78261 | -561.0446875 | 95.6 % |
| `adhya4` | -992.66728 | -1345 | -961.93218 | -877.6457399 | 26.7 % |
| `bental4` | -550 | -2933.3333 | -550 | -450 | 0.0 % |
| `bental5` | -3500 | -9700 | -3500 | -3500 | - |
| `foulds2` | -1100 | -6900 | -1100 | -1100 | - |
| `foulds3` | -8 | -260 | -8 | -8 | - |
| `foulds4` | -8 | -260 | -8 | -8 | - |
| `foulds5` | -8 | -260 | -8 | -8 | - |
| `haverly1` | -500 | -2450 | -500 | -400 | 0.0 % |
| `haverly2` | -1000 | -4700 | -1000 | -600 | 0.0 % |
| `haverly3` | -800 | -2450 | -800 | -750 | 0.0 % |

Every run that did not pass, named:

- **gap not closed** (4): `bental5_q`, `foulds3_q`, `foulds4_q`, `foulds5_q`.
- **gap not closed, incumbent worse than the optimum** (3): `adhya3_p`, `adhya3_q`, `adhya4_q`.

---

## 2d. QPLIB, the convex continuous QPs

The convex, continuous, linearly constrained (or box- or un-constrained) instances of QPLIB
(Furini et al., *QPLIB: a library of quadratic programming instances*, Mathematical
Programming Computation 11, 2019), selected from the site's own listing by
`bench/runners/fetch_qplib.py`, which also reads each reference value from QPLIB's
`qplib.solu` and the instance's page and records the sha256 of every file in
`data/qplib/reference.json`. The `.qplib` files are converted to QPS by
`bench/runners/qplib_format.py` and solved by `bench/runners/qplib.py` under the default QP
engine and the interior point.

Selection, read from QPLIB's listing: instances.html: Cvx ticked, O in {C, D}, V = C, C in {N, B, L} (doc.html PROBTYPE and CONVEX). **19 instances** of the 453 listed; 17 fetched (12 in the small tier, at most 100,000 stored coefficients), 2 over the size cap and not run: `QPLIB_8547`, `QPLIB_9008`. QPLIB publishes no solution point for `QPLIB_9002`: run, named, and outside the pass count. Convex continuous instances outside the selection, by type: LCD 13 (quadratic constraints: #514's set). Licence, as the site states it: QPLIB is licensed under CC-BY 4.0. (https://creativecommons.org/licenses/by/4.0/)

Source CSV: `bench/results/qplib-5a39fb2.csv`  
Commit `5a39fb2` · machine `laptop, Intel i7-1355U 10 cores 12 threads, 16 GB, on mains, overnight with no other jobs` · time limit 1000 s per instance · solver defaults

- `auto`: **passed 9 of 16** with a published reference; `optimal` on 9 of 17, within 1e-6 of QPLIB's value on 9, accepted by the verifier on 10.
- `ipm` (`qp_algorithm=ipm`): **passed 9 of 16** with a published reference; `optimal` on 9 of 17, within 1e-6 of QPLIB's value on 9, accepted by the verifier on 10.

A pass is `optimal`, within 1e-6 relative of QPLIB's value (qplib.solu, a best known point, not a proven optimum), all three QP residuals of `qp_residuals.py` within 1e-6 relative, and accepted by the independent verifier, all on the QPS file `qplib_format.py` converted the `.qplib` file to - a conversion checked at QPLIB's own published point for every instance when it was fetched.

| instance | engine | rows | cols | status | our objective | QPLIB value | rel. gap | worst residual | iters | solver time (s) | verified | passed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| `QPLIB_10034` | auto | 40200 | 40400 | iteration_limit | -0.0665712768 | -0.06601017605 | 5.6e-04 | 1.0e+00 | 1000000 | 618.82 | - | **no** |
| `QPLIB_10034` | ipm | 40200 | 40400 | iteration_limit | -0.0665712768 | -0.06601017605 | 5.6e-04 | 1.0e+00 | 1000000 | 597.40 | - | **no** |
| `QPLIB_10038` | auto | 160400 | 160800 | time_limit | -0.06666457598 | 0 | 6.7e-02 | 1.4e-01 | 286160 | 866.77 | - | **no** |
| `QPLIB_10038` | ipm | 160400 | 160800 | time_limit | -0.06666444289 | 0 | 6.7e-02 | 1.4e-01 | 272740 | 867.90 | - | **no** |
| `QPLIB_8495` | auto | 8000 | 27543 | optimal | 42857.4964 | 42857.49639 | 3.3e-10 | 7.7e-10 | 14 | 2.16 | yes | yes |
| `QPLIB_8495` | ipm | 8000 | 27543 | optimal | 42857.4964 | 42857.49639 | 3.3e-10 | 7.7e-10 | 14 | 2.24 | yes | yes |
| `QPLIB_8500` | auto | 250498 | 250997 | time_limit | -0.08904792676 | 0.01792173763 | 1.1e-01 | 1.0e+00 | 142650 | 700.88 | - | **no** |
| `QPLIB_8500` | ipm | 250498 | 250997 | time_limit | -0.08775810513 | 0.01792173763 | 1.1e-01 | 1.0e+00 | 140531 | 690.75 | - | **no** |
| `QPLIB_8515` | auto | 8002 | 16002 | model_error | 0 | 319.9999887 | 1.0e+00 | - | 0 | 0.01 | - | **no** |
| `QPLIB_8515` | ipm | 8002 | 16002 | model_error | 0 | 319.9999887 | 1.0e+00 | - | 0 | 0.01 | - | **no** |
| `QPLIB_8559` | auto | 5000 | 10000 | iteration_limit | 74222438.31 | 74223239.83 | 1.1e-05 | 5.0e-06 | 1000000 | 209.41 | - | **no** |
| `QPLIB_8559` | ipm | 5000 | 10000 | iteration_limit | 74222438.31 | 74223239.83 | 1.1e-05 | 5.0e-06 | 1000000 | 201.23 | - | **no** |
| `QPLIB_8567` | auto | 7500 | 10000 | iteration_limit | 78962705.39 | 78965987.95 | 4.2e-05 | 9.7e-06 | 1000000 | 240.67 | - | **no** |
| `QPLIB_8567` | ipm | 7500 | 10000 | iteration_limit | 78962705.39 | 78965987.95 | 4.2e-05 | 9.7e-06 | 1000000 | 242.82 | - | **no** |
| `QPLIB_8602` | auto | 52983 | 34552 | optimal | 27353.3872 | 27353.3872 | 3.4e-12 | 4.8e-09 | 16 | 566.45 | yes | yes |
| `QPLIB_8602` | ipm | 52983 | 34552 | optimal | 27353.3872 | 27353.3872 | 3.4e-12 | 4.8e-09 | 16 | 547.67 | yes | yes |
| `QPLIB_8616` | auto | 10404 | 13870 | optimal | 245.0685978 | 245.0685978 | 1.6e-10 | 3.4e-10 | 11 | 0.10 | yes | yes |
| `QPLIB_8616` | ipm | 10404 | 13870 | optimal | 245.0685978 | 245.0685978 | 1.6e-10 | 3.4e-10 | 11 | 0.10 | yes | yes |
| `QPLIB_8785` | auto | 11362 | 10399 | optimal | 7867.491149 | 7867.491149 | 1.8e-11 | 2.9e-08 | 14 | 12.31 | yes | yes |
| `QPLIB_8785` | ipm | 11362 | 10399 | optimal | 7867.491149 | 7867.491149 | 1.8e-11 | 2.9e-08 | 14 | 11.99 | yes | yes |
| `QPLIB_8790` | auto | 0 | 39204 | optimal | -0.0001562421091 | -0.0001562421091 | 5.0e-14 | 1.8e-12 | 6 | 0.63 | yes | yes |
| `QPLIB_8790` | ipm | 0 | 39204 | optimal | -0.0001562421091 | -0.0001562421091 | 5.0e-14 | 1.8e-12 | 6 | 0.63 | yes | yes |
| `QPLIB_8792` | auto | 0 | 15129 | optimal | 3593.516294 | 3593.518355 | 5.7e-07 | 1.2e-10 | 10 | 0.30 | yes | yes |
| `QPLIB_8792` | ipm | 0 | 15129 | optimal | 3593.516294 | 3593.518355 | 5.7e-07 | 1.2e-10 | 10 | 0.30 | yes | yes |
| `QPLIB_8845` | auto | 777 | 1546 | optimal | 10907992.49 | 10907992.49 | 3.7e-10 | 8.4e-15 | 32 | 0.41 | yes | yes |
| `QPLIB_8845` | ipm | 777 | 1546 | optimal | 10907992.49 | 10907992.49 | 3.7e-10 | 8.4e-15 | 32 | 0.41 | yes | yes |
| `QPLIB_8906` | auto | 838 | 5223 | feasible | 2699111.515 | 2699111.513 | 7.9e-10 | 6.2e-10 | 25 | 0.52 | yes | **no** |
| `QPLIB_8906` | ipm | 838 | 5223 | feasible | 2699111.515 | 2699111.513 | 7.9e-10 | 6.2e-10 | 25 | 0.51 | yes | **no** |
| `QPLIB_8938` | auto | 11999 | 4001 | optimal | -35.77945295 | -35.77945295 | 6.5e-11 | 8.3e-12 | 20 | 0.10 | yes | yes |
| `QPLIB_8938` | ipm | 11999 | 4001 | optimal | -35.77945295 | -35.77945295 | 6.5e-11 | 8.3e-12 | 20 | 0.10 | yes | yes |
| `QPLIB_8991` | auto | 0 | 14400 | optimal | -0.001667867378 | -0.001667867378 | 8.6e-14 | 9.5e-12 | 6 | 0.15 | yes | yes |
| `QPLIB_8991` | ipm | 0 | 14400 | optimal | -0.001667867378 | -0.001667867378 | 8.6e-14 | 9.5e-12 | 6 | 0.15 | yes | yes |
| `QPLIB_9002` | auto | 1649 | 2890 | iteration_limit | 1.729346531e+11 | - | - | 1.2e+00 | 200 | 3.56 | - | - |
| `QPLIB_9002` | ipm | 1649 | 2890 | iteration_limit | 1.729346531e+11 | - | - | 1.2e+00 | 200 | 3.56 | - | - |

Every failure, named:

- `auto`, **did not reach `optimal`** (8): `QPLIB_10034` (iteration_limit), `QPLIB_10038` (time_limit), `QPLIB_8500` (time_limit), `QPLIB_8515` (model_error), `QPLIB_8559` (iteration_limit), `QPLIB_8567` (iteration_limit), `QPLIB_8906` (feasible), `QPLIB_9002` (iteration_limit).
- `auto`, **no published reference** (1): `QPLIB_9002`.
- `ipm`, **did not reach `optimal`** (8): `QPLIB_10034` (iteration_limit), `QPLIB_10038` (time_limit), `QPLIB_8500` (time_limit), `QPLIB_8515` (model_error), `QPLIB_8559` (iteration_limit), `QPLIB_8567` (iteration_limit), `QPLIB_8906` (feasible), `QPLIB_9002` (iteration_limit).
- `ipm`, **no published reference** (1): `QPLIB_9002`.

### Every QPLIB instance through the reader and the dispatcher

All of QPLIB, not only the convex continuous selection above: integer, nonconvex and
quadratically constrained instances too, fetched and converted to QPS (integer markers and
`QCMATRIX` rows included) by `bench/runners/fetch_qplib_all.py`, each conversion checked at
QPLIB's published point, then solved by `bench/runners/qplib_all.py` under the CLI's defaults.

Not yet run: no `bench/results/qplib-all-<commit>.csv`. Reproduce with

```
python bench/runners/fetch_qplib_all.py
python bench/runners/qplib_all.py --time-limit 60
```

---

## 2e. Nonlinear programs - Hock-Schittkowski, and convex MINLPLib

The nonlinear engine behind the `solve()` seam (NLP stages 1-3): a model read from AMPL's
`.nl` format, exact first and second derivatives by automatic differentiation, a primal-dual
interior point with a filter line search, and NLP-based branch and bound for integer columns.
Both sets ship in `data/nlp/` with their published objectives in each set's `REFERENCE.csv`,
and every answer is checked by `tools/verify_solution.py`, which reads the `.nl` file with
its own reader and evaluates the constraints with its own derivatives. The statuses are the
finding: `optimal` is claimed only where the model is proved convex, `locally_optimal` is any
other KKT point, and `locally_infeasible` is a local minimizer of the violation.

**Hock-Schittkowski** - the 70 problems of Hock and Schittkowski, *Test Examples for Nonlinear Programming Codes* (1981), from Vanderbei's AMPL models converted by `bench/runners/hs_mod_to_nl.py`, matched at a relative 1e-6. The interior point with a filter line search after Wachter and Biegler (2006); where the model is not proved convex, `locally_optimal` is the status the engine can honestly give, and the breakdown says how often that is.

Source CSV: `bench/results/nlp-hs-ad57c03.csv`  
Commit `ad57c03` · machine `Linux-x86_64` · 60.0 s per problem

**64 of 70** reached the published objective, and **70 of 70** answers were accepted by the independent checker (`tools/verify_solution.py`, its own `.nl` reader and its own derivatives). Statuses: `locally_optimal` 51, `optimal` 19.

| problem | status | our objective | published | rel. gap | iters | time (s) | match | verified |
|---|---|---:|---:|---:|---:|---:|:--:|:--:|
| `hs001` | locally_optimal | 1.604114528e-17 | -0 | 1.60e-17 | 26 | 0.00 | yes | yes |
| `hs002` | locally_optimal | 4.941229328 | 0.0504261879 | 4.89e+00 | 11 | 0.00 | **NO** | yes |
| `hs003` | optimal | 1e-08 | -0 | 1.00e-08 | 4 | 0.00 | yes | yes |
| `hs004` | locally_optimal | 2.666666687 | 2.666666667 | 7.50e-09 | 6 | 0.00 | yes | yes |
| `hs005` | locally_optimal | -1.913222955 | -1.91322207 | 4.62e-07 | 9 | 0.00 | yes | yes |
| `hs006` | locally_optimal | 3.405159426e-16 | -0 | 3.41e-16 | 9 | 0.00 | yes | yes |
| `hs007` | locally_optimal | -1.732050808 | -1.732050808 | 6.48e-13 | 27 | 0.00 | yes | yes |
| `hs008` | locally_optimal | -1 | -1 | 0.00e+00 | 6 | 0.00 | yes | yes |
| `hs010` | locally_optimal | -0.99999999 | -1 | 1.00e-08 | 12 | 0.00 | yes | yes |
| `hs011` | optimal | -8.498464213 | -8.498464223 | 1.16e-09 | 8 | 0.00 | yes | yes |
| `hs012` | locally_optimal | -29.99999999 | -30 | 3.33e-10 | 8 | 0.00 | yes | yes |
| `hs013` | locally_optimal | 0.9976580967 | 1 | 2.34e-03 | 1123 | 0.01 | **NO** | yes |
| `hs014` | optimal | 1.393464991 | 1.393464981 | 7.18e-09 | 7 | 0.00 | yes | yes |
| `hs015` | locally_optimal | 306.5 | 306.5 | 6.53e-11 | 15 | 0.00 | yes | yes |
| `hs016` | locally_optimal | 23.14466096 | 0.25 | 2.29e+01 | 10 | 0.00 | **NO** | yes |
| `hs017` | locally_optimal | 1.000000426 | 1 | 4.26e-07 | 20 | 0.00 | yes | yes |
| `hs018` | locally_optimal | 5.00000001 | 5 | 2.00e-09 | 12 | 0.00 | yes | yes |
| `hs019` | locally_optimal | -6961.813876 | -6961.81381 | 9.42e-09 | 16 | 0.00 | yes | yes |
| `hs020` | locally_optimal | 40.19872983 | 38.19872981 | 5.24e-02 | 12 | 0.00 | **NO** | yes |
| `hs021` | optimal | -99.95999999 | -99.96 | 9.90e-11 | 9 | 0.00 | yes | yes |
| `hs022` | optimal | 1.00000002 | 1 | 2.00e-08 | 6 | 0.00 | yes | yes |
| `hs023` | locally_optimal | 2.00000002 | 2 | 9.98e-09 | 10 | 0.00 | yes | yes |
| `hs024` | locally_optimal | -0.99999998 | -1 | 2.00e-08 | 14 | 0.00 | yes | yes |
| `hs026` | locally_optimal | 8.482336173e-14 | -0 | 8.48e-14 | 21 | 0.00 | yes | yes |
| `hs027` | locally_optimal | 0.04 | 0.04 | 0.00e+00 | 17 | 0.00 | yes | yes |
| `hs028` | optimal | 0 | -0 | 0.00e+00 | 2 | 0.00 | yes | yes |
| `hs029` | locally_optimal | -22.62741699 | -22.627417 | 4.41e-10 | 8 | 0.00 | yes | yes |
| `hs030` | optimal | 1 | 1 | 4.00e-12 | 16 | 0.00 | yes | yes |
| `hs031` | locally_optimal | 6.00000001 | 6 | 1.67e-09 | 7 | 0.00 | yes | yes |
| `hs032` | optimal | 1.000000361 | 1 | 3.61e-07 | 12 | 0.00 | yes | yes |
| `hs033` | locally_optimal | -4.585786408 | -4.585786438 | 6.54e-09 | 48 | 0.00 | yes | yes |
| `hs034` | optimal | -0.8340324152 | -0.8340324452 | 3.01e-08 | 9 | 0.00 | yes | yes |
| `hs035` | locally_optimal | 0.1111111212 | 0.1111111111 | 1.00e-08 | 7 | 0.00 | yes | yes |
| `hs036` | locally_optimal | -3300 | -3300 | 9.09e-12 | 13 | 0.00 | yes | yes |
| `hs037` | locally_optimal | -3456 | -3456 | 2.89e-12 | 11 | 0.00 | yes | yes |
| `hs038` | locally_optimal | 2.317148421e-21 | -0 | 2.32e-21 | 40 | 0.00 | yes | yes |
| `hs039` | locally_optimal | -1 | -1 | 0.00e+00 | 9 | 0.00 | yes | yes |
| `hs040` | locally_optimal | -0.25 | -0.25 | 0.00e+00 | 8 | 0.00 | yes | yes |
| `hs041` | locally_optimal | 1.925925936 | 1.925925926 | 5.18e-09 | 10 | 0.00 | yes | yes |
| `hs042` | locally_optimal | 13.85786438 | 13.85786438 | 2.23e-12 | 6 | 0.00 | yes | yes |
| `hs043` | optimal | -43.99999998 | -44 | 4.55e-10 | 9 | 0.00 | yes | yes |
| `hs044` | locally_optimal | -12.99999996 | -15 | 1.33e-01 | 18 | 0.00 | **NO** | yes |
| `hs045` | locally_optimal | 1.00000005 | 1 | 5.00e-08 | 23 | 0.00 | yes | yes |
| `hs046` | locally_optimal | 4.870714159e-13 | -0 | 4.87e-13 | 16 | 0.00 | yes | yes |
| `hs047` | locally_optimal | 1.902239821e-12 | -0 | 1.90e-12 | 20 | 0.00 | yes | yes |
| `hs048` | optimal | 2.888956546e-28 | -0 | 2.89e-28 | 3 | 0.00 | yes | yes |
| `hs049` | optimal | 5.366698265e-11 | -0 | 5.37e-11 | 18 | 0.00 | yes | yes |
| `hs050` | optimal | 2.095411779e-31 | -0 | 2.10e-31 | 9 | 0.00 | yes | yes |
| `hs051` | optimal | 2.465190329e-32 | -0 | 2.47e-32 | 2 | 0.00 | yes | yes |
| `hs052` | optimal | 5.326647564 | 5.326647564 | 1.62e-14 | 2 | 0.00 | yes | yes |
| `hs053` | optimal | 4.093023256 | 4.093023256 | 9.66e-13 | 6 | 0.00 | yes | yes |
| `hs060` | locally_optimal | 0.03256820026 | 0.03256820025 | 5.10e-12 | 7 | 0.00 | yes | yes |
| `hs061` | locally_optimal | -143.6461422 | -143.6461422 | 1.39e-11 | 16 | 0.00 | yes | yes |
| `hs062` | locally_optimal | -26272.51449 | -26272.51448 | 2.78e-10 | 8 | 0.00 | yes | yes |
| `hs063` | locally_optimal | 961.7151721 | 961.7151721 | 3.12e-11 | 14 | 0.00 | yes | yes |
| `hs064` | optimal | 6299.842428 | 6299.842428 | 1.11e-11 | 18 | 0.00 | yes | yes |
| `hs065` | optimal | 0.9535288668 | 0.9535288567 | 1.01e-08 | 12 | 0.00 | yes | yes |
| `hs066` | optimal | 0.5181632941 | 0.5181632741 | 2.00e-08 | 7 | 0.00 | yes | yes |
| `hs071` | locally_optimal | 17.01401731 | 17.0140173 | 5.35e-10 | 8 | 0.00 | yes | yes |
| `hs073` | locally_optimal | 29.89437819 | 29.894378 | 6.33e-09 | 8 | 0.00 | yes | yes |
| `hs076` | locally_optimal | -4.681818162 | -4.681818181 | 4.10e-09 | 7 | 0.00 | yes | yes |
| `hs077` | locally_optimal | 0.2415051288 | 0.24150513 | 1.21e-09 | 9 | 0.00 | yes | yes |
| `hs078` | locally_optimal | -2.919700409 | -2.91970041 | 3.56e-10 | 8 | 0.00 | yes | yes |
| `hs079` | locally_optimal | 0.07877682087 | 0.0787768209 | 2.89e-11 | 5 | 0.00 | yes | yes |
| `hs093` | locally_optimal | 135.0759628 | 135.075961 | 1.37e-08 | 9 | 0.00 | yes | yes |
| `hs100` | locally_optimal | 680.6300574 | 680.6300573 | 1.38e-10 | 10 | 0.00 | yes | yes |
| `hs104` | locally_optimal | 3.95116348 | 3.95116344 | 1.03e-08 | 9 | 0.00 | yes | yes |
| `hs108` | locally_optimal | -0.6749813833 | -0.8660254038 | 1.91e-01 | 15 | 0.01 | **NO** | yes |
| `hs110` | locally_optimal | -45.77846971 | -45.77846971 | 5.68e-11 | 6 | 0.00 | yes | yes |
| `hs113` | locally_optimal | 24.30620913 | 24.3062091 | 1.16e-09 | 19 | 0.01 | yes | yes |

**Not at the published objective**, named rather than dropped: `hs002`, `hs013`, `hs016`, `hs020`, `hs044`, `hs108`.


**MINLPLib, convex** - convex mixed-integer instances of MINLPLib with a published primal bound, matched at the MIP gap target 1e-4, solved by NLP-based branch and bound - run only when the relaxation is proved convex, so `optimal` here is a closed bound.

Source CSV: `bench/results/nlp-minlplib-ad57c03.csv`  
Commit `ad57c03` · machine `Linux-x86_64` · 60.0 s per problem

**14 of 17** reached the published objective, and **16 of 17** answers were accepted by the independent checker (`tools/verify_solution.py`, its own `.nl` reader and its own derivatives). Statuses: `feasible` 2, `numerical_error` 1, `optimal` 14.

| problem | status | our objective | published | rel. gap | iters | time (s) | match | verified |
|---|---|---:|---:|---:|---:|---:|:--:|:--:|
| `ball_mk2_10` | optimal | 0 | 0 | 0.00e+00 | 24115 | 0.54 | yes | yes |
| `batchdes` | optimal | 167427.6571 | 167427.6571 | 8.96e-11 | 175 | 0.01 | yes | yes |
| `clay0203m` | feasible | 41573.26247 | 41573.26252 | 1.23e-09 | 237083 | 34.69 | **NO** | yes |
| `cvxnonsep_pcon20` | optimal | -21.51230115 | -21.5123012 | 2.22e-09 | 837 | 0.06 | yes | yes |
| `ex1223` | optimal | 4.579582452 | 4.579582402 | 1.10e-08 | 140 | 0.01 | yes | yes |
| `ex1223a` | optimal | 4.579582452 | 4.579582402 | 1.10e-08 | 39 | 0.01 | yes | yes |
| `ex1223b` | optimal | 4.579582452 | 4.579582402 | 1.10e-08 | 118 | 0.01 | yes | yes |
| `fac1` | optimal | 160912612.3 | 160912612.4 | 3.11e-10 | 195 | 0.02 | yes | yes |
| `flay02m` | optimal | 37.94733201 | 37.94733192 | 2.42e-09 | 101 | 0.01 | yes | yes |
| `gbd` | optimal | 2.20000001 | 2.2 | 4.55e-09 | 34 | 0.00 | yes | yes |
| `jit1` | numerical_error | - | 173983.33 | - | 3000 | 0.21 | **NO** | no point |
| `m3` | feasible | 37.80000029 | 37.8 | 7.67e-09 | 13078 | 1.53 | **NO** | yes |
| `nvs03` | optimal | 16 | 16 | 0.00e+00 | 110 | 0.00 | yes | yes |
| `st_e14` | optimal | 4.579582452 | 4.579582402 | 1.10e-08 | 140 | 0.01 | yes | yes |
| `syn05m` | optimal | 837.7324008 | 837.7324009 | 6.92e-11 | 197 | 0.01 | yes | yes |
| `synthes2` | optimal | 73.03531256 | 73.03531253 | 3.44e-10 | 189 | 0.01 | yes | yes |
| `synthes3` | optimal | 68.00974056 | 68.00974052 | 6.50e-10 | 330 | 0.02 | yes | yes |

**Not at the published objective**, named rather than dropped: `clay0203m`, `jit1`, `m3`.

**No point to check** (a limit or an infeasibility verdict): `jit1`.


Reproduce:

```
python bench/runners/nlp_bench.py --data data/nlp/hs                              # 70 Hock-Schittkowski
python bench/runners/nlp_bench.py --data data/nlp/minlplib --match-tolerance 1e-4  # convex MINLPLib
```
---

## 3. Correctness beyond the objective value

An objective that matches a published number is necessary, not sufficient — it says nothing
about whether the reported solution is internally consistent. Two independent checks cover
that, and both run in CI:

- **`tools/verify_solution.py`** re-parses the model with its own MPS reader, recomputes the
  row activities, the objective, the reduced costs and the dual objective, and checks primal
  feasibility, dual feasibility, complementary slackness and strong duality. It shares no
  code with the solver, so a reader bug shows up as a disagreement rather than as agreement.
  The `verified` column above is its verdict.

- **The exact rational oracle** (`tests/oracles/`) solves generated instances in exact
  arithmetic with no rounding error anywhere, and the floating-point simplex is compared
  against it. See `docs/PROVENANCE.md` for the citations.

### 3a. Netlib's infeasible set - a verdict is only as good as its certificate

Chinneck's collection of infeasible LPs (`netlib.org/lp/infeas`, fetched and hashed by
`bench/runners/fetch_netlib_infeasible.py`). Every instance is infeasible, so the status
alone proves nothing; a pass needs the Farkas certificate the solver wrote to survive
`tools/verify_solution.py` (`bench/runners/netlib_infeasible.py`).

Source CSV: `bench/results/netlib-infeasible-5a39fb2.csv`  
Commit `5a39fb2` · machine `laptop, Intel i7-1355U 10 cores 12 threads, 16 GB, on mains, overnight with no other jobs` · time limit 60 s per instance

**28 of 29** reported `infeasible` **and** wrote a Farkas certificate that `tools/verify_solution.py` accepted. A status of `infeasible` without a certificate is not counted: the verifier has nothing to check, so the verdict is unproven.

**12 of 29** also named an irreducible infeasible subsystem (#217, the Chinneck-Dravnieks deletion filter) that the verifier proved on its own arithmetic: infeasible using nothing outside the set, and irreducible by one witness per element, a point that satisfies every other element and violates that one. 18 named one; not proved on klein1, klein2, klein3, qual, refinery, vol1.

**1 not passed**, every one named with its cause:

| why | count | instances |
|---|---:|---|
| no verdict: numerical_error | 1 | cplex2 |

| instance | rows | cols | status | engine | certificate | multipliers | time (s) | verified | IIS | the solver's message |
|---|---:|---:|---|---|---|---:|---:|:--:|---|---|
| `bgdbg1` | 348 | 407 | infeasible | presolve | farkas | 2 | 0.026 | yes | - | row 163 allows activity of at most 24 but the column bounds force at least 54; proved by presolve; proof: the rows aggre |
| `bgetam` | 400 | 688 | infeasible | simplex-dual | farkas | 7 | 0.062 | yes | 23 proved | dual simplex: basic variable 833 is outside its bounds by 5.949e+02, far above the 1.0e-07 feasibility tolerance, and no |
| `bgindy` | 2671 | 10116 | infeasible | simplex-dual | farkas | 3 | 1.068 | yes | 154 proved | dual simplex: basic variable 12627 is outside its bounds by 7.134e+03, far above the 1.0e-07 feasibility tolerance, and  |
| `bgprtr` | 20 | 34 | infeasible | simplex-dual+primal | farkas | 6 | 0.022 | yes | 12 proved | phase 1 terminated with max bound violation 2.343e+01, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `box1` | 231 | 261 | infeasible | simplex-dual | farkas | 8 | 0.030 | yes | 9 proved | dual simplex: basic variable 271 is outside its bounds by 7.071e-01, far above the 1.0e-07 feasibility tolerance, and no |
| `ceria3d` | 3576 | 824 | infeasible | presolve | farkas | 236 | 0.228 | yes | - | row 292 needs activity of at least 0.75 but the column bounds cap it at 0.5; proved by presolve; certificate from the el |
| `chemcom` | 288 | 720 | infeasible | simplex-dual | farkas | 7 | 0.039 | yes | 42 proved | dual simplex: basic variable 724 is outside its bounds by 2.849e+03, far above the 1.0e-07 feasibility tolerance, and no |
| `cplex1` | 3005 | 3221 | infeasible | simplex-dual+primal | farkas | 5 | 0.350 | yes | 6 proved | phase 1 terminated with max bound violation 6.834e+06, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `cplex2` | 224 | 221 | numerical_error | simplex-dual+primal | none | 0 | 0.035 | - | - | phase 1 stalled at max bound violation 4.100e-06, only just above the 1.0e-07 feasibility tolerance; no column prices as |
| `ex72a` | 197 | 215 | infeasible | simplex-dual | farkas | 58 | 0.039 | yes | 59 proved | dual simplex: basic variable 300 is outside its bounds by 7.071e-01, far above the 1.0e-07 feasibility tolerance, and no |
| `ex73a` | 193 | 211 | infeasible | simplex-dual | farkas | 24 | 0.045 | yes | 25 proved | dual simplex: basic variable 288 is outside its bounds by 7.071e-01, far above the 1.0e-07 feasibility tolerance, and no |
| `forest6` | 66 | 95 | infeasible | simplex-dual | farkas | 66 | 0.072 | yes | 94 proved | dual simplex: basic variable 0 is outside its bounds by 3.221e+05, far above the 1.0e-07 feasibility tolerance, and no n |
| `galenet` | 8 | 8 | infeasible | simplex-dual | farkas | 2 | 0.023 | yes | - | dual simplex: basic variable 12 is outside its bounds by 1.500e+01, far above the 1.0e-07 feasibility tolerance, and no  |
| `gosh` | 3792 | 10733 | infeasible | simplex-dual+primal | farkas | 9 | 5.633 | yes | 9 proved | phase 1 terminated with max bound violation 1.263e+01, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `gran` | 2658 | 2520 | infeasible | simplex-dual+primal | farkas | 731 | 0.219 | yes | - | phase 1 terminated with max bound violation 5.249e+06, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `greenbea` | 2393 | 5405 | infeasible | presolve | farkas | 1 | 0.048 | yes | - | row 1491 allows activity of at most 0 but the column bounds force at least 320; proved by presolve; proof: the rows aggr |
| `itest2` | 9 | 4 | infeasible | presolve | farkas | 3 | 0.027 | yes | - | row 4 allows activity of at most 2 but the column bounds force at least 13; proved by presolve; proof: the rows aggregat |
| `itest6` | 11 | 8 | infeasible | presolve | farkas | 3 | 0.025 | yes | - | row 3 needs activity of at least 50000 but the column bounds cap it at -30000; proved by presolve; proof: the rows aggre |
| `klein1` | 54 | 54 | infeasible | simplex-dual | farkas | 51 | 0.069 | yes | 55 **not proved** | dual simplex: basic variable 89 is outside its bounds by 7.734e+04, far above the 1.0e-07 feasibility tolerance, and no  |
| `klein2` | 477 | 54 | infeasible | simplex-dual | farkas | 54 | 0.226 | yes | 55 **not proved** | dual simplex: basic variable 282 is outside its bounds by 5.350e+04, far above the 1.0e-07 feasibility tolerance, and no |
| `klein3` | 994 | 88 | infeasible | simplex-dual | farkas | 87 | 1.235 | yes | 89 **not proved** | dual simplex: basic variable 414 is outside its bounds by 6.342e+04, far above the 1.0e-07 feasibility tolerance, and no |
| `mondou2` | 312 | 604 | infeasible | simplex-dual | farkas | 36 | 0.060 | yes | 65 proved | dual simplex: basic variable 474 is outside its bounds by 9.130e+03, far above the 1.0e-07 feasibility tolerance, and no |
| `pang` | 361 | 459 | infeasible | simplex-dual+primal | farkas | 13 | 0.075 | yes | 35 proved | phase 1 terminated with max bound violation 7.814e+04, far above the 1.0e-07 feasibility tolerance; route: the scaled at |
| `pilot4i` | 410 | 1000 | infeasible | presolve | farkas | 1 | 0.022 | yes | - | row 390 needs activity of at least 15.17 but the column bounds cap it at 0; proved by presolve; proof: the rows aggregat |
| `qual` | 323 | 464 | infeasible | simplex-dual+primal | farkas | 120 | 3.531 | yes | 229 **not proved** | phase 1 terminated with max bound violation 3.971e+04, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `reactor` | 318 | 637 | infeasible | presolve | farkas | 1 | 0.020 | yes | - | row 116 needs activity of at least 0 but the column bounds cap it at -1; proved by presolve; proof: the rows aggregate t |
| `refinery` | 323 | 464 | infeasible | simplex-dual+primal | farkas | 85 | 1.716 | yes | 167 **not proved** | phase 1 terminated with max bound violation 9.715e+04, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `vol1` | 323 | 464 | infeasible | simplex-dual+primal | farkas | 209 | 32.433 | yes | 394 **not proved** | phase 1 terminated with max bound violation 9.546e+05, far above the 1.0e-07 feasibility tolerance; proof: the rows aggr |
| `woodinfe` | 35 | 89 | infeasible | presolve | farkas | 1 | 0.022 | yes | - | row 17 needs activity of at least 0 but the column bounds cap it at -5; proved by presolve; proof: the rows aggregate to |

No per-engine option run is committed yet (`--solver-option algorithm=simplex`, `dual-simplex`, `pdhg`).

### 3b. Parametric LP - the optimal value as a function of one coefficient

`tools/parametric.py` walks one cost or one right-hand side across a range: from the optimal
basis at the start it reads the ranging interval (#220) to find where that basis stops being
optimal, jumps there, warm-starts (#218) from the basis it is leaving, and records the
objective and the basis change at every breakpoint. Each reported point is a fresh optimum
re-solved and checked by `tools/test_parametric.py` through the verifier's own MPS reader,
not a value extrapolated from ranging. It re-solves at each breakpoint rather than pivoting
once as a dedicated parametric simplex would; the tool's docstring says what that costs.

**`crude-blend-AL`** - `bench/results/parametric-crude-blend-AL-887b173.csv`, solver at `887b173`, Windows-AMD64, 3 breakpoint(s):

| parameter | objective | status | what changed at this point |
|---:|---:|---|---|
| 0.0 | 148.88888888888894 | optimal | initial basis |
| 1.0666666666666669 | 159.5555555555556 | optimal | no basis change (objective still moves linearly) |
| 5.0 | 348.017837837838 | optimal | entered: AL, MU; left: BN |

**`crude-blend-THRUPUT`** - `bench/results/parametric-crude-blend-THRUPUT-887b173.csv`, solver at `887b173`, Windows-AMD64, 3 breakpoint(s):

| parameter | objective | status | what changed at this point |
|---:|---:|---|---|
| 100.0 | 173.12499999999983 | optimal | initial basis |
| 107.81351351351357 | 214.14594594594607 | optimal | no basis change (objective still moves linearly) |
| 200.0 | 214.14594594594604 | optimal | no basis change (objective still moves linearly) |


---

## 4. Comparison against an established solver

HiGHS is the reference. It runs as a SEPARATE PROCESS over the same MPS files; no HiGHS code
is linked into, or read by, SANKHYA - see `docs/PROVENANCE.md`. Both sides are timed on
solver-internal time only.

The comparison below is run on **the same tier as section 1b**, not on the nine-instance
demo set. Comparing only where we pass would be the easy version of this table and would say
nothing: the instances we fail are exactly the ones a reader should want to see against a
mature solver.

Source CSV: `bench/results/compare-highs-medium-ad57c03.csv`  
Commit `ad57c03` · machine `Linux-x86_64`

**50 of 50** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.001 | 0.002 | 0.74x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.000 | 0.000 | 0.58x |
| `agg` | -3.59917673e+07 | -3.59917673e+07 | yes | 0.004 | 0.004 | 1.15x |
| `bandm` | -1.58628018e+02 | -1.58628018e+02 | yes | 0.016 | 0.009 | 1.79x |
| `beaconfd` | 3.35924858e+04 | 3.35924858e+04 | yes | 0.003 | 0.003 | 0.98x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.001 | 0.001 | 0.72x |
| `boeing1` | -3.35213568e+02 | -3.35213568e+02 | yes | 0.014 | 0.011 | 1.23x |
| `boeing2` | -3.15018728e+02 | -3.15018728e+02 | yes | 0.002 | 0.003 | 0.90x |
| `bore3d` | 1.37308039e+03 | 1.37308039e+03 | yes | 0.003 | 0.003 | 0.88x |
| `brandy` | 1.51850990e+03 | 1.51850990e+03 | yes | 0.008 | 0.005 | 1.47x |
| `capri` | 2.69001291e+03 | 2.69001291e+03 | yes | 0.005 | 0.004 | 1.51x |
| `d6cube` | 3.15491667e+02 | 3.15491667e+02 | yes | 0.104 | 0.132 | 0.79x |
| `degen2` | -1.43517800e+03 | -1.43517800e+03 | yes | 0.021 | 0.014 | 1.53x |
| `e226` | -1.16389291e+01 | -1.16389291e+01 | yes | 0.012 | 0.007 | 1.60x |
| `etamacro` | -7.55715233e+02 | -7.55715233e+02 | yes | 0.027 | 0.009 | 2.91x |
| `finnis` | 1.72791066e+05 | 1.72791066e+05 | yes | 0.014 | 0.005 | 2.85x |
| `fit1d` | -9.14637809e+03 | -9.14637809e+03 | yes | 0.005 | 0.011 | 0.43x |
| `fit2d` | -6.84642933e+04 | -6.84642933e+04 | yes | 0.070 | 0.144 | 0.49x |
| `forplan` | -6.64218961e+02 | -6.64218961e+02 | yes | 0.007 | 0.006 | 1.24x |
| `grow15` | -1.06870941e+08 | -1.06870941e+08 | yes | 0.042 | 0.042 | 1.01x |
| `grow22` | -1.60834336e+08 | -1.60834336e+08 | yes | 0.111 | 0.088 | 1.26x |
| `grow7` | -4.77878118e+07 | -4.77878118e+07 | yes | 0.010 | 0.012 | 0.88x |
| `israel` | -8.96644822e+05 | -8.96644822e+05 | yes | 0.005 | 0.003 | 1.39x |
| `kb2` | -1.74990013e+03 | -1.74990013e+03 | yes | 0.001 | 0.001 | 0.97x |
| `lotfi` | -2.52647061e+01 | -2.52647061e+01 | yes | 0.003 | 0.004 | 0.77x |
| `pilot4` | -2.58113926e+03 | -2.58113926e+03 | yes | 0.052 | 0.030 | 1.77x |
| `recipe` | -2.66616000e+02 | -2.66616000e+02 | yes | 0.001 | 0.001 | 0.66x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.001 | 0.001 | 1.11x |
| `sc205` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.003 | 0.003 | 1.04x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.000 | 0.001 | 0.86x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.001 | 0.001 | 0.78x |
| `scagr25` | -1.47534331e+07 | -1.47534331e+07 | yes | 0.009 | 0.007 | 1.44x |
| `scagr7` | -2.33138982e+06 | -2.33138982e+06 | yes | 0.001 | 0.003 | 0.56x |
| `scfxm1` | 1.84167590e+04 | 1.84167590e+04 | yes | 0.012 | 0.008 | 1.53x |
| `scorpion` | 1.87812482e+03 | 1.87812482e+03 | yes | 0.005 | 0.004 | 1.30x |
| `scrs8` | 9.04296954e+02 | 9.04296954e+02 | yes | 0.014 | 0.009 | 1.49x |
| `scsd1` | 8.66666667e+00 | 8.66666667e+00 | yes | 0.002 | 0.002 | 0.79x |
| `scsd6` | 5.05000001e+01 | 5.05000001e+01 | yes | 0.014 | 0.007 | 1.85x |
| `scsd8` | 9.05000000e+02 | 9.05000000e+02 | yes | 0.070 | 0.040 | 1.74x |
| `sctap1` | 1.41225000e+03 | 1.41225000e+03 | yes | 0.006 | 0.005 | 1.11x |
| `share1b` | -7.65893186e+04 | -7.65893186e+04 | yes | 0.002 | 0.003 | 0.90x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.001 | 0.002 | 0.55x |
| `ship04l` | 1.79332454e+06 | 1.79332454e+06 | yes | 0.010 | 0.008 | 1.32x |
| `ship04s` | 1.79871470e+06 | 1.79871470e+06 | yes | 0.006 | 0.006 | 0.99x |
| `stair` | -2.51266951e+02 | -2.51266951e+02 | yes | 0.016 | 0.014 | 1.12x |
| `standata` | 1.25769950e+03 | 1.25769950e+03 | yes | 0.002 | 0.004 | 0.39x |
| `standmps` | 1.40601750e+03 | 1.40601750e+03 | yes | 0.006 | 0.005 | 1.19x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.001 | 0.001 | 1.00x |
| `tuff` | 2.92147765e-01 | 2.92147765e-01 | yes | 0.005 | 0.008 | 0.64x |
| `wood1p` | 1.44290241e+00 | 1.44290241e+00 | yes | 0.021 | 0.067 | 0.32x |

**Summary**

- SANKHYA shifted geometric mean: **0.015s**
- HiGHS shifted geometric mean: **0.015s**
- SANKHYA is **1.0x** the HiGHS time by that measure

- per-instance ratio: median **1.02x**, worst **2.91x**, faster than HiGHS on **23 of 50** instances

The two are **within noise of each other** here, at 1.00x. A narrow claim: eight small instances settle nothing about large models. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme. This solver now has a presolve (#43, #92) and a dual simplex (#65) of its own, both defaults, so what remains between the two is the pricing and the years. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

---

## 4a. Head-to-head: HiGHS, SCIP, CBC/Clp and GLPK (#766)

**The instance counts first, because they differ between reports.** netlib.org's `lp/data`
ships **92** feasible LPs as EMPS files, and those 92 are the headline row below. Two more,
`truss` and `stocfor3`, are shipped only as Fortran generators; they are generated and pinned
by sha256 (#745) and reported in their own row. `qap8`, `qap12` and `qap15` are in the
readme's table but not in `lp/data`. Published comparisons count differently: 93 or 94 when
they add the generated or `qap` models, 114 when the sixteen Kennington LPs are folded in.
Kennington is reported separately here (4a.2), and so is the 138-problem Maros-Meszaros set
(4a.3).

Every solver runs as a SEPARATE PROCESS on the same machine, with the same time limit and one
thread: HiGHS through `highspy`, SCIP through `pyscipopt`, Clp (CBC's LP engine) and GLPK as
the distribution's binaries. No rival code is linked into or read by SANKHYA; see judgement
call 35 in `docs/PROVENANCE.md`. Every answer, a rival's included, is converted into the
`.sol` layout and checked by `tools/verify_solution.py`: on primal and dual conditions where
the solver gives usable duals, on primal feasibility and the objective only where it does not
(SCIP; the `checked on` column says which). `solved` is the solver's own optimal status;
`matched` is the objective recomputed at its point within a relative 1e-6 of the published
optimum; `verified` is the verifier's acceptance. Only a run that is solved, verified and
matched counts for the time - matched against the exact optimum where one is published, which
for Netlib is Koch's (the readme is wrong on nine instances) - and any other is charged the
full limit, and the shifted geometric mean uses a
10-second shift, Mittelmann's, not the 1 s of section 1.
Times are each solver's own clock. Produced by `python bench/runners/compare.py --suite
<suite>` (`bench/runners/compare_suite.py`, `bench/runners/rivals.py`).

**Where the Kennington and Maros-Meszaros files came from.** netlib.org and www.doc.ic.ac.uk
refuse connections from the cloud container these suites ran on, so the files were taken from
public mirrors and accepted only because every one has the sha256 pinned in
`data/kennington/reference.json` (`mps_lf_sha256`, all 16) and
`data/maros-meszaros/reference.json` (the readme, the three archives and all 138 QPS files).
The mirrors are named in judgement call 37 of `docs/PROVENANCE.md`.

**Reading the Maros-Meszaros table.** GLPK has no quadratic objective, so it is listed as
unsupported rather than left out. `values` has a non-convex objective; SANKHYA refuses it as a
model error (its LDL^T of Q meets a negative pivot), and HiGHS and Clp return a local optimum at
the published value. SANKHYA also refuses `cvxqp1l`, `cvxqp2l` and `cvxqp3l` as non-convex (its LDL^T meets a pivot of -3.5e-3), which HiGHS and Clp solve as convex; that is a miss of SANKHYA's, present in every Maros-Meszaros CSV since `9094e1c`, not a wrong answer. On `hues-mod` SANKHYA and Clp agree with each other to 1e-13 relative and
SANKHYA's answer passes the verifier on primal and dual conditions, yet both sit 6.5e-6 from the
readme's eight-figure OPT: the readme is the likely outlier there, as it is for nine Netlib
instances, but with no exact value published the row is graded against it and not counted. On
`dpklo1` HiGHS's MPS reader takes the RHS section, whose set is named `1` beside numeric row
names, differently from our reader and the other three solvers (50 right-hand sides differ;
matrix, costs and bounds are identical), so it solves another model and the verifier rejects
its point. Most rival rejections below are not wrong answers but conditions missed by more
than the verifier allows: complementary slackness and dual feasibility at 1e-7 (HiGHS), or a
stated objective that differs from the one recomputed at the solver's own point by more than
1e-9 relative (Clp, SCIP). The verifier and its tolerances are the same for every solver,
SANKHYA included.

### 4a.1 Netlib LP

Source CSV: `bench/results/head-to-head-netlib-5e490ef.csv`  
Commit `5e490ef` · machine `cloud container (docker); Intel(R) Xeon(R) Processor @ 2.10GHz; 4 cores; 16 GiB RAM; Linux-x86_64`  
94 instances · time limit 120 s · 1 thread per solver · every solver a separate process

**The 92 LPs netlib.org ships as EMPS files:**

| solver | version | runs | solved | matched | matched exact | verified | checked on | SGM time, shift 10 s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| SANKHYA | 5e490ef | 92 | 92 | 83 | 92 | 92 | primal+dual | 0.575 s |
| HiGHS | 1.15.1 | 92 | 92 | 83 | 92 | 92 | primal+dual | 0.163 s |
| SCIP | 10.0.2 | 92 | 92 | 83 | 92 | 92 | primal-only | 0.398 s |
| CBC/Clp | 1.17.9 | 92 | 92 | 83 | 92 | 88 | primal+dual | 1.355 s |
| GLPK | 5.0 | 92 | 92 | 83 | 92 | 91 | primal+dual | 0.486 s |

**Instance by instance, SANKHYA against each rival:**

| against | instances compared | SANKHYA faster | tie | rival faster | median of SANKHYA's time over the rival's, floored | the same, clocks as recorded |
|---|---:|---:|---:|---:|---:|---:|
| HiGHS | 92 | 3 | 72 | 17 | 1.00x | 1.08x |
| SCIP | 92 | 20 | 65 | 7 | 1.00x | 0.42x |
| CBC/Clp | 92 | 6 | 71 | 15 | 1.00x | 0.72x |
| GLPK | 92 | 7 | 72 | 13 | 1.00x | 1.86x |

*A tie is two times within 10% of each other, or both under 0.1 s (the coarsest clock here cannot tell those apart). A run that did not count loses to one that did, and an instance on which neither counted is left out. Both medians are over the instances both solvers counted on, and above 1x SANKHYA is the slower one: the first floors each time at 0.1 s as the profile below does, the second takes each solver's clock as recorded, which on instances of a few milliseconds is a ratio of start-up costs and, for a solver that prints 0.0, is left out.*

**The two generated from netlib.org's Fortran bundles (#745), `truss` and `stocfor3`:**

| solver | version | runs | solved | matched | matched exact | verified | checked on | SGM time, shift 10 s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| SANKHYA | 5e490ef | 2 | 2 | 1 | 2 | 2 | primal+dual | 3.874 s |
| HiGHS | 1.15.1 | 2 | 2 | 1 | 2 | 2 | primal+dual | 1.397 s |
| SCIP | 10.0.2 | 2 | 2 | 1 | 2 | 2 | primal-only | 2.314 s |
| CBC/Clp | 1.17.9 | 2 | 2 | 1 | 2 | 2 | primal+dual | 1.279 s |
| GLPK | 5.0 | 2 | 2 | 1 | 2 | 2 | primal+dual | 2.164 s |

`matched` grades against the readme's optimum and `matched exact` against Koch's exact rational optimum (`data/netlib/koch_exact.json`, #747); the timing counts the exact grade. Where the two differ the readme is the one that is wrong: on 80bau3b, ganges, greenbea, greenbeb, nesm, pilot, pilot.we, scrs8 and stocfor3 every solver here agrees with Koch.

![Dolan-More performance profile, Netlib LP](img/profile-netlib.svg)

*`docs/img/profile-netlib.svg`, drawn from the CSV by `bench/runners/perf_profile.py` each time this document is generated. A curve's height at tau is the fraction of the instances above on which that solver's run counted and took at most tau times the fastest counted run; times under 0.1 s are floored there (GLPK's clock resolution), so the fast end is a tie.*

Every run that did not count, by solver:

- **CBC/Clp**, 4: `pilot` (optimal, rejected by the verifier), `pilot.ja` (optimal, rejected by the verifier), `pilot.we` (optimal, rejected by the verifier), `pilotnov` (optimal, rejected by the verifier)
- **GLPK**, 1: `dfl001` (optimal, rejected by the verifier)

What the verifier rejected:

- CBC/Clp on `pilot`: [FAIL] column bounds               worst violation 5.392e-07 (5.392e-07 relative) on UFO006
- CBC/Clp on `pilot.ja`: [FAIL] column bounds               worst violation 1.538e-05 (1.538e-05 relative) on USLB06
- CBC/Clp on `pilot.we`: [FAIL] column bounds               worst violation 1.000e-05 (1.000e-05 relative) on UOSE03
- CBC/Clp on `pilotnov`: [FAIL] column bounds               worst violation 1.538e-05 (1.538e-05 relative) on USLB06
- GLPK on `dfl001`: [FAIL] dual feasibility (columns)  worst 2.851e-07 (2.851e-07 relative) on C11729

### 4a.2 Kennington LP

Source CSV: `bench/results/head-to-head-kennington-1860864.csv`  
Commit `1860864` · machine `cloud container (docker); Intel(R) Xeon(R) Processor @ 2.10GHz; 4 cores; 16 GiB RAM; Linux-x86_64`  
16 instances · time limit 120 s · 1 thread per solver · every solver a separate process

| solver | version | runs | solved | matched | verified | checked on | SGM time, shift 10 s |
|---|---|---:|---:|---:|---:|---|---:|
| SANKHYA | 1860864 | 16 | 15 | 15 | 15 | primal+dual | 9.430 s |
| HiGHS | 1.15.1 | 16 | 16 | 16 | 16 | primal+dual | 1.855 s |
| SCIP | 10.0.2 | 16 | 15 | 15 | 15 | primal-only | 10.899 s |
| CBC/Clp | 1.17.9 | 16 | 16 | 16 | 16 | primal+dual | 0.795 s |
| GLPK | 5.0 | 16 | 15 | 15 | 15 | primal+dual | 7.345 s |

**Instance by instance, SANKHYA against each rival:**

| against | instances compared | SANKHYA faster | tie | rival faster | median of SANKHYA's time over the rival's, floored | the same, clocks as recorded |
|---|---:|---:|---:|---:|---:|---:|
| HiGHS | 16 | 3 | 2 | 11 | 2.83x | 3.12x |
| SCIP | 16 | 6 | 1 | 9 | 1.25x | 1.25x |
| CBC/Clp | 16 | 1 | 2 | 13 | 2.86x | 4.39x |
| GLPK | 16 | 9 | 1 | 6 | 0.98x | 0.74x |

*A tie is two times within 10% of each other, or both under 0.1 s (the coarsest clock here cannot tell those apart). A run that did not count loses to one that did, and an instance on which neither counted is left out. Both medians are over the instances both solvers counted on, and above 1x SANKHYA is the slower one: the first floors each time at 0.1 s as the profile below does, the second takes each solver's clock as recorded, which on instances of a few milliseconds is a ratio of start-up costs and, for a solver that prints 0.0, is left out.*

![Dolan-More performance profile, Kennington LP](img/profile-kennington.svg)

*`docs/img/profile-kennington.svg`, drawn from the CSV by `bench/runners/perf_profile.py` each time this document is generated. A curve's height at tau is the fraction of the instances above on which that solver's run counted and took at most tau times the fastest counted run; times under 0.1 s are floored there (GLPK's clock resolution), so the fast end is a tie.*

Every run that did not count, by solver:

- **SANKHYA**, 1: `pds-20` (time_limit)
- **SCIP**, 1: `osa-60` (time_limit)
- **GLPK**, 1: `ken-18` (time_limit)

### 4a.3 Maros-Meszaros QP

Source CSV: `bench/results/head-to-head-maros-meszaros-1860864.csv`  
Commit `1860864` · machine `cloud container (docker); Intel(R) Xeon(R) Processor @ 2.10GHz; 4 cores; 16 GiB RAM; Linux-x86_64`  
138 instances · time limit 60 s · 1 thread per solver · every solver a separate process

| solver | version | runs | solved | matched | verified | checked on | SGM time, shift 10 s |
|---|---|---:|---:|---:|---:|---|---:|
| SANKHYA | 1860864 | 138 | 107 | 106 | 107 | primal+dual | 6.437 s |
| HiGHS | 1.15.1 | 138 | 105 | 99 | 32 | primal+dual | 35.781 s |
| SCIP | 10.0.2 | 138 | 78 | 76 | 53 | primal-only | 27.436 s |
| CBC/Clp | 1.17.9 | 138 | 127 | 113 | 62 | primal+dual | 20.974 s |
| GLPK | - | 138 | unsupported | - | - | - | - |

**Instance by instance, SANKHYA against each rival:**

| against | instances compared | SANKHYA faster | tie | rival faster | median of SANKHYA's time over the rival's, floored | the same, clocks as recorded |
|---|---:|---:|---:|---:|---:|---:|
| HiGHS | 109 | 82 | 22 | 5 | 1.00x | 0.26x |
| SCIP | 114 | 87 | 19 | 8 | 0.52x | 0.03x |
| CBC/Clp | 118 | 75 | 28 | 15 | 1.00x | 0.24x |

*A tie is two times within 10% of each other, or both under 0.1 s (the coarsest clock here cannot tell those apart). A run that did not count loses to one that did, and an instance on which neither counted is left out. Both medians are over the instances both solvers counted on, and above 1x SANKHYA is the slower one: the first floors each time at 0.1 s as the profile below does, the second takes each solver's clock as recorded, which on instances of a few milliseconds is a ratio of start-up costs and, for a solver that prints 0.0, is left out.*

![Dolan-More performance profile, Maros-Meszaros QP](img/profile-maros-meszaros.svg)

*`docs/img/profile-maros-meszaros.svg`, drawn from the CSV by `bench/runners/perf_profile.py` each time this document is generated. A curve's height at tau is the fraction of the instances above on which that solver's run counted and took at most tau times the fastest counted run; times under 0.1 s are floored there (GLPK's clock resolution), so the fast end is a tie.*

Every run that did not count, by solver:

- **SANKHYA**, 32: `boyd1` (feasible), `cvxqp1l` (model_error), `cvxqp2l` (model_error), `cvxqp3l` (model_error), `hues-mod` (optimal at relative gap 6.5e-06), `huestis` (feasible), `liswet1` (iteration_limit), `liswet10` (iteration_limit), `liswet11` (iteration_limit), `liswet12` (iteration_limit), `liswet2` (iteration_limit), `liswet7` (time_limit), `liswet8` (iteration_limit), `liswet9` (iteration_limit), `powell20` (feasible), `primalc1` (feasible), `qcapri` (feasible), `qfffff80` (feasible), `qforplan` (feasible), `qgfrdxpn` (feasible), `qgrow15` (feasible), `qpilotno` (iteration_limit), `qseba` (feasible), `qshare1b` (feasible), `qshell` (feasible), `qship12l` (time_limit), `qship12s` (feasible), `qsierra` (iteration_limit), `stadat1` (iteration_limit), `ubh1` (iteration_limit), `values` (model_error), `yao` (iteration_limit)
- **HiGHS**, 106: `aug2d` (solve error), `aug2dc` (solve error), `aug2dcqp` (time_limit), `aug2dqp` (time_limit), `aug3d` (optimal, rejected by the verifier), `aug3dc` (optimal, rejected by the verifier), `aug3dcqp` (optimal, rejected by the verifier), `aug3dqp` (optimal, rejected by the verifier), `boyd1` (time_limit), `boyd2` (time_limit), `cont-050` (optimal, rejected by the verifier), `cont-100` (optimal, rejected by the verifier), `cont-101` (time_limit), `cont-200` (time_limit), `cont-201` (time_limit), `cont-300` (time_limit), `cvxqp1l` (optimal, rejected by the verifier), `cvxqp2l` (optimal, rejected by the verifier), `cvxqp2m` (optimal, rejected by the verifier), `cvxqp2s` (optimal, rejected by the verifier), `dpklo1` (optimal, rejected by the verifier), `dtoc3` (solve error), `exdata` (optimal, rejected by the verifier), `gouldqp2` (optimal, rejected by the verifier), `gouldqp3` (optimal, rejected by the verifier), `hs118` (optimal, rejected by the verifier), `hs268` (optimal, rejected by the verifier), `hs35` (optimal, rejected by the verifier), `hs35mod` (optimal, rejected by the verifier), `hs51` (optimal, rejected by the verifier), `hs76` (optimal, rejected by the verifier), `hues-mod` (time_limit), `huestis` (time_limit), `ksip` (not set), `laser` (not set), `liswet1` (optimal, rejected by the verifier), `liswet10` (optimal, rejected by the verifier), `liswet11` (optimal, rejected by the verifier), `liswet12` (optimal, rejected by the verifier), `liswet2` (optimal, rejected by the verifier), `liswet3` (optimal, rejected by the verifier), `liswet4` (optimal, rejected by the verifier), `liswet5` (optimal, rejected by the verifier), `liswet6` (optimal, rejected by the verifier), `liswet7` (optimal, rejected by the verifier), `liswet8` (optimal, rejected by the verifier), `liswet9` (optimal, rejected by the verifier), `lotschd` (optimal, rejected by the verifier), `mosarqp1` (solve error), `mosarqp2` (time_limit), `powell20` (optimal, rejected by the verifier), `primalc1` (optimal, rejected by the verifier), `primalc2` (optimal, rejected by the verifier), `primalc5` (optimal, rejected by the verifier), `primalc8` (optimal, rejected by the verifier), `q25fv47` (time_limit), `qadlittl` (optimal, rejected by the verifier), `qafiro` (optimal, rejected by the verifier), `qbandm` (optimal, rejected by the verifier), `qbeaconf` (optimal, rejected by the verifier), `qbore3d` (optimal, rejected by the verifier), `qbrandy` (optimal, rejected by the verifier), `qcapri` (solve error), `qe226` (not set), `qetamacr` (optimal, rejected by the verifier), `qfffff80` (optimal, rejected by the verifier), `qforplan` (optimal, rejected by the verifier), `qgfrdxpn` (optimal, rejected by the verifier), `qgrow15` (solve error), `qgrow22` (time_limit), `qgrow7` (time_limit), `qisrael` (optimal, rejected by the verifier), `qpcboei1` (optimal, rejected by the verifier), `qpcboei2` (optimal, rejected by the verifier), `qpcstair` (not set), `qpilotno` (not set), `qrecipe` (optimal, rejected by the verifier), `qscagr25` (optimal, rejected by the verifier), `qscagr7` (optimal, rejected by the verifier), `qscfxm1` (optimal, rejected by the verifier), `qscfxm2` (optimal, rejected by the verifier), `qscfxm3` (optimal, rejected by the verifier), `qscorpio` (optimal, rejected by the verifier), `qscrs8` (optimal, rejected by the verifier), `qscsd8` (time_limit), `qsctap1` (optimal, rejected by the verifier), `qsctap2` (not set), `qsctap3` (not set), `qseba` (optimal, rejected by the verifier), `qshare1b` (solve error), `qshare2b` (optimal, rejected by the verifier), `qshell` (optimal, rejected by the verifier), `qship04l` (optimal, rejected by the verifier), `qship04s` (optimal, rejected by the verifier), `qship08l` (optimal, rejected by the verifier), `qship08s` (optimal, rejected by the verifier), `qship12l` (optimal, rejected by the verifier), `qship12s` (optimal, rejected by the verifier), `qsierra` (solve error), `qstair` (time_limit), `qstandat` (optimal, rejected by the verifier), `s268` (optimal, rejected by the verifier), `stadat1` (optimal, rejected by the verifier), `stadat2` (optimal, rejected by the verifier), `stadat3` (solve error), `ubh1` (time_limit)
- **SCIP**, 87: `aug2d` (hung), `aug2dc` (hung), `aug2dcqp` (hung), `aug2dqp` (hung), `aug3d` (time_limit), `aug3dc` (time_limit), `boyd1` (time_limit), `boyd2` (time_limit), `cont-050` (optimal, rejected by the verifier), `cont-100` (hung), `cont-101` (time_limit), `cont-200` (time_limit), `cont-201` (hung), `cont-300` (time_limit), `cvxqp1l` (hung), `cvxqp1m` (time_limit), `cvxqp2l` (time_limit), `cvxqp2m` (time_limit), `cvxqp3l` (hung), `cvxqp3m` (time_limit), `dpklo1` (optimal, rejected by the verifier), `dtoc3` (hung), `dual1` (optimal, rejected by the verifier), `dual2` (time_limit), `dual3` (time_limit), `dual4` (optimal, rejected by the verifier), `dualc1` (optimal at relative gap 6.8e-06), `exdata` (crashed), `genhs28` (optimal, rejected by the verifier), `gouldqp2` (time_limit), `gouldqp3` (time_limit), `hs268` (time_limit), `hs35` (optimal, rejected by the verifier), `hs35mod` (optimal, rejected by the verifier), `hs51` (optimal, rejected by the verifier), `hs52` (time_limit), `hs53` (optimal, rejected by the verifier), `hs76` (optimal, rejected by the verifier), `hues-mod` (hung), `huestis` (hung), `ksip` (optimal, rejected by the verifier), `laser` (crashed), `liswet1` (hung), `liswet10` (hung), `liswet11` (hung), `liswet12` (hung), `liswet2` (hung), `liswet3` (hung), `liswet4` (hung), `liswet5` (hung), `liswet6` (hung), `liswet7` (hung), `liswet8` (hung), `liswet9` (hung), `powell20` (hung), `primal1` (optimal, rejected by the verifier), `primal2` (optimal, rejected by the verifier), `primal3` (time_limit), `primal4` (time_limit), `q25fv47` (time_limit), `qafiro` (optimal, rejected by the verifier), `qe226` (optimal, rejected by the verifier), `qetamacr` (time_limit), `qforplan` (time_limit), `qgfrdxpn` (crashed), `qisrael` (optimal, rejected by the verifier), `qpcblend` (optimal, rejected by the verifier), `qpcboei2` (optimal, rejected by the verifier), `qptest` (optimal, rejected by the verifier), `qsc205` (optimal, rejected by the verifier), `qsctap1` (optimal, rejected by the verifier), `qsctap2` (optimal, rejected by the verifier), `qsctap3` (optimal, rejected by the verifier), `qseba` (crashed), `qshell` (crashed), `qship12l` (time_limit), `s268` (time_limit), `stadat1` (crashed), `stadat2` (crashed), `stadat3` (crashed), `stcqp1` (time_limit), `stcqp2` (time_limit), `tame` (optimal, rejected by the verifier), `ubh1` (time_limit), `values` (time_limit), `yao` (optimal at relative gap 7.7e-03), `zecevic2` (optimal, rejected by the verifier)
- **CBC/Clp**, 78: `aug2dcqp` (optimal, rejected by the verifier), `aug2dqp` (optimal, rejected by the verifier), `boyd1` (unparsed), `boyd2` (hung), `cont-050` (optimal, rejected by the verifier), `cont-100` (optimal, rejected by the verifier), `cont-101` (optimal, rejected by the verifier), `cont-200` (hung), `cont-201` (unparsed), `cont-300` (hung), `cvxqp1l` (unparsed), `cvxqp2l` (unparsed), `cvxqp3l` (unparsed), `dpklo1` (optimal, rejected by the verifier), `dual1` (optimal, rejected by the verifier), `dual2` (optimal, rejected by the verifier), `dual3` (optimal, rejected by the verifier), `dual4` (optimal, rejected by the verifier), `gouldqp2` (optimal, rejected by the verifier), `hs268` (optimal, rejected by the verifier), `hs35` (optimal, rejected by the verifier), `hs35mod` (optimal, rejected by the verifier), `hs53` (optimal, rejected by the verifier), `hs76` (optimal, rejected by the verifier), `hues-mod` (optimal at relative gap 6.5e-06), `huestis` (optimal, rejected by the verifier), `ksip` (optimal, rejected by the verifier), `liswet8` (optimal at relative gap 9.0e-01), `mosarqp2` (optimal, rejected by the verifier), `powell20` (optimal, rejected by the verifier), `primal1` (optimal, rejected by the verifier), `primal2` (optimal, rejected by the verifier), `primal3` (optimal, rejected by the verifier), `primalc2` (optimal, rejected by the verifier), `q25fv47` (optimal, rejected by the verifier), `qadlittl` (optimal, rejected by the verifier), `qafiro` (optimal, rejected by the verifier), `qbeaconf` (optimal, rejected by the verifier), `qbore3d` (optimal, rejected by the verifier), `qbrandy` (optimal, rejected by the verifier), `qcapri` (unparsed), `qe226` (optimal, rejected by the verifier), `qfffff80` (unparsed), `qforplan` (optimal, rejected by the verifier), `qgfrdxpn` (optimal, rejected by the verifier), `qgrow15` (optimal, rejected by the verifier), `qgrow22` (optimal, rejected by the verifier), `qgrow7` (optimal, rejected by the verifier), `qisrael` (optimal, rejected by the verifier), `qpcblend` (optimal, rejected by the verifier), `qpcboei1` (optimal, rejected by the verifier), `qpcboei2` (optimal, rejected by the verifier), `qpcstair` (optimal, rejected by the verifier), `qpilotno` (hung), `qptest` (optimal, rejected by the verifier), `qsc205` (optimal, rejected by the verifier), `qscagr25` (optimal, rejected by the verifier), `qscagr7` (optimal, rejected by the verifier), `qscfxm1` (optimal, rejected by the verifier), `qscfxm2` (optimal, rejected by the verifier), `qscfxm3` (optimal, rejected by the verifier), `qscsd8` (optimal, rejected by the verifier), `qsctap1` (optimal, rejected by the verifier), `qsctap3` (optimal, rejected by the verifier), `qseba` (optimal, rejected by the verifier), `qshare1b` (optimal, rejected by the verifier), `qshell` (optimal, rejected by the verifier), `qship08l` (optimal, rejected by the verifier), `qship08s` (optimal, rejected by the verifier), `qsierra` (optimal, rejected by the verifier), `qstair` (optimal, rejected by the verifier), `qstandat` (optimal, rejected by the verifier), `s268` (optimal, rejected by the verifier), `stadat1` (optimal, rejected by the verifier), `stadat3` (optimal, rejected by the verifier), `tame` (optimal, rejected by the verifier), `ubh1` (optimal, rejected by the verifier), `zecevic2` (optimal, rejected by the verifier)

What the verifier rejected:

- HiGHS on `aug3d`: [FAIL] dual feasibility (columns)  worst 2.006e-07 (2.006e-07 relative) on C---3696; [FAIL] strong duality              primal 5.540677257925e+02  dual 5.540682388161e+02  gap 5.130e-04 (relative 9.259e-07), 2.789e-04 of
- HiGHS on `aug3dc`: [FAIL] dual feasibility (columns)  worst 1.828e-07 (1.828e-07 relative) on C---3697; [FAIL] strong duality              primal 7.712624386889e+02  dual 7.712628998921e+02  gap 4.612e-04 (relative 5.980e-07), 2.248e-04 of
- HiGHS on `aug3dcqp`: [FAIL] dual feasibility (columns)  worst 1.991e-07 (1.991e-07 relative) on C---3576; [FAIL] complementary slackness     worst |multiplier| * slack = 1.096e-06 on C---1105; [FAIL] strong duality              primal 9.9336
- HiGHS on `aug3dqp`: [FAIL] dual feasibility (columns)  worst 7.805e-07 (7.550e-07 relative) on C---2179; [FAIL] complementary slackness     worst |multiplier| * slack = 3.160e-06 on C---1691
- HiGHS on `cont-050`: [FAIL] strong duality              primal -4.563850868314e+00  dual -4.564606144916e+00  gap 7.553e-04 (relative 1.655e-04), 5.197e-04 of it from per-item violations accepted above
- HiGHS on `cont-100`: [FAIL] strong duality              primal -4.644397375958e+00  dual -4.646823585593e+00  gap 2.426e-03 (relative 5.224e-04), 1.950e-03 of it from per-item violations accepted above
- HiGHS on `cvxqp1l`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.689e-05 on C---6304
- HiGHS on `cvxqp2l`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.500e-06 on C---1023
- HiGHS on `cvxqp2m`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.263e-06 on C----111
- HiGHS on `cvxqp2s`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.168e-06 on C------9
- HiGHS on `dpklo1`: [FAIL] row activity                worst violation 3.653e+01 (3.653e+01 relative to the row's terms) on 53
- HiGHS on `exdata`: [FAIL] dual feasibility (columns)  worst 1.000e-06 (1.000e-06 relative) on bs47; [FAIL] complementary slackness     worst |multiplier| * slack = 1.000e-05 on bs249
- HiGHS on `gouldqp2`: [FAIL] strong duality              primal 1.882025172792e-04  dual 1.596021559707e-04  gap 2.860e-05 (relative 2.860e-05), 8.530e-06 of it from per-item violations accepted above
- HiGHS on `gouldqp3`: [FAIL] strong duality              primal 2.062783971713e+00  dual 2.062776572639e+00  gap 7.399e-06 (relative 3.587e-06), 6.220e-06 of it from per-item violations accepted above
- HiGHS on `hs118`: [FAIL] complementary slackness     worst |multiplier| * slack = 3.591e-04 on C------8; [FAIL] strong duality              primal 6.648204500000e+02  dual 6.648187602000e+02  gap 1.690e-03 (relative 2.542e-06), 1.455e-03 
- HiGHS on `hs268`: [FAIL] dual feasibility (columns)  worst 4.000e-07 (4.000e-07 relative) on C------5; [FAIL] strong duality              primal 1.818989403546e-12  dual 3.099996320088e-06  gap 3.100e-06 (relative 3.100e-06), 1.999e-07 of
- HiGHS on `hs35`: [FAIL] dual feasibility (columns)  worst 1.333e-07 (1.333e-07 relative) on C------1
- HiGHS on `hs35mod`: [FAIL] dual feasibility (columns)  worst 1.500e-07 (1.500e-07 relative) on C------1
- HiGHS on `hs51`: [FAIL] dual feasibility (columns)  worst 1.000e-07 (1.000e-07 relative) on C------2; [FAIL] strong duality              primal 0.000000000000e+00  dual 4.999999934086e-07  gap 5.000e-07 (relative 5.000e-07), 4.000e-07 of
- HiGHS on `hs76`: [FAIL] dual feasibility (columns)  worst 2.091e-07 (2.091e-07 relative) on C------2
- HiGHS on `liswet1`: [FAIL] dual feasibility (columns)  worst 7.482e-06 (7.482e-06 relative) on C--10002; [FAIL] complementary slackness     worst |multiplier| * slack = 7.482e-06 on C--10002; [FAIL] strong duality              primal 3.6122
- HiGHS on `liswet10`: [FAIL] dual feasibility (columns)  worst 4.814e-05 (4.814e-05 relative) on C---9530; [FAIL] complementary slackness     worst |multiplier| * slack = 4.814e-05 on C---9530; [FAIL] strong duality              primal 4.9485
- HiGHS on `liswet11`: [FAIL] dual feasibility (columns)  worst 8.005e-04 (8.005e-04 relative) on C--10001; [FAIL] complementary slackness     worst |multiplier| * slack = 8.005e-04 on C--10001; [FAIL] strong duality              primal 4.9523
- HiGHS on `liswet12`: [FAIL] dual feasibility (columns)  worst 3.274e-05 (3.274e-05 relative) on C---8482; [FAIL] complementary slackness     worst |multiplier| * slack = 3.274e-05 on C---8482
- HiGHS on `liswet2`: [FAIL] dual feasibility (columns)  worst 1.452e-07 (1.452e-07 relative) on C--10001; [FAIL] strong duality              primal 2.499807610289e+01  dual 2.499840954945e+01  gap 3.334e-04 (relative 1.334e-05), 3.332e-04 of
- HiGHS on `liswet3`: [FAIL] dual feasibility (columns)  worst 6.224e-04 (6.224e-04 relative) on C---9530; [FAIL] complementary slackness     worst |multiplier| * slack = 6.224e-04 on C---9530; [FAIL] strong duality              primal 2.5001
- HiGHS on `liswet4`: [FAIL] dual feasibility (columns)  worst 8.156e-04 (8.156e-04 relative) on C---9662; [FAIL] complementary slackness     worst |multiplier| * slack = 8.156e-04 on C---9662; [FAIL] strong duality              primal 2.5000
- HiGHS on `liswet5`: [FAIL] dual feasibility (columns)  worst 5.453e-03 (5.453e-03 relative) on C---9555; [FAIL] complementary slackness     worst |multiplier| * slack = 5.453e-03 on C---9555; [FAIL] strong duality              primal 2.5034
- HiGHS on `liswet6`: [FAIL] dual feasibility (columns)  worst 4.394e-03 (4.394e-03 relative) on C---9153; [FAIL] complementary slackness     worst |multiplier| * slack = 4.394e-03 on C---9153; [FAIL] strong duality              primal 2.4995
- HiGHS on `liswet7`: [FAIL] dual feasibility (columns)  worst 7.272e-07 (7.272e-07 relative) on C--10002
- HiGHS on `liswet8`: [FAIL] dual feasibility (columns)  worst 3.877e-05 (3.877e-05 relative) on C---9549; [FAIL] complementary slackness     worst |multiplier| * slack = 3.877e-05 on C---9549; [FAIL] strong duality              primal 7.1447
- HiGHS on `liswet9`: [FAIL] dual feasibility (columns)  worst 1.141e-05 (1.141e-05 relative) on C---9794; [FAIL] complementary slackness     worst |multiplier| * slack = 1.141e-05 on C---9794
- HiGHS on `lotschd`: [FAIL] complementary slackness     worst |multiplier| * slack = 7.516e-05 on C------5
- HiGHS on `powell20`: [FAIL] dual feasibility (columns)  worst 2.501e-04 (1.000e-07 relative) on C---5002; [FAIL] complementary slackness     worst |multiplier| * slack = 7.499e-04 on C--10000
- HiGHS on `primalc1`: [FAIL] dual feasibility (columns)  worst 1.032e-03 (1.032e-03 relative) on C------1; [FAIL] complementary slackness     worst |multiplier| * slack = 1.066e+01 on C------1; [FAIL] strong duality              primal -6.155
- HiGHS on `primalc2`: [FAIL] dual feasibility (columns)  worst 4.725e-04 (4.725e-04 relative) on C------1; [FAIL] complementary slackness     worst |multiplier| * slack = 2.233e+00 on C------1
- HiGHS on `primalc5`: [FAIL] dual feasibility (columns)  worst 4.601e-05 (4.601e-05 relative) on C------1; [FAIL] complementary slackness     worst |multiplier| * slack = 2.117e-02 on C------1
- HiGHS on `primalc8`: [FAIL] dual feasibility (columns)  worst 3.311e-03 (3.311e-03 relative) on C------1; [FAIL] complementary slackness     worst |multiplier| * slack = 1.096e+02 on C------1
- HiGHS on `qadlittl`: [FAIL] dual feasibility (columns)  worst 5.276e-06 (5.276e-06 relative) on ...142; [FAIL] complementary slackness     worst |multiplier| * slack = 1.484e-02 on ...175
- HiGHS on `qafiro`: [FAIL] dual feasibility (columns)  worst 2.552e-06 (2.552e-06 relative) on X37; [FAIL] complementary slackness     worst |multiplier| * slack = 6.515e-05 on X37
- HiGHS on `qbandm`: [FAIL] dual feasibility (columns)  worst 8.433e-05 (8.433e-05 relative) on MC90PT; [FAIL] complementary slackness     worst |multiplier| * slack = 7.111e-02 on MC90PT
- HiGHS on `qbeaconf`: [FAIL] dual feasibility (columns)  worst 2.792e-04 (2.792e-04 relative) on 10144; [FAIL] complementary slackness     worst |multiplier| * slack = 7.795e-01 on 10144
- HiGHS on `qbore3d`: [FAIL] dual feasibility (columns)  worst 8.437e-04 (8.437e-04 relative) on IUT.KWXI; [FAIL] complementary slackness     worst |multiplier| * slack = 7.118e+00 on IUT.KWXI
- HiGHS on `qbrandy`: [FAIL] dual feasibility (columns)  worst 3.532e-04 (3.531e-04 relative) on 100002; [FAIL] complementary slackness     worst |multiplier| * slack = 1.248e+00 on 100002
- HiGHS on `qetamacr`: [FAIL] dual feasibility (columns)  worst 1.393e-06 (1.393e-06 relative) on KAPSTK65; [FAIL] complementary slackness     worst |multiplier| * slack = 2.144e-04 on PCPETG05
- HiGHS on `qfffff80`: [FAIL] dual feasibility (columns)  worst 2.469e-02 (2.469e-02 relative) on YP.DWLGS; [FAIL] complementary slackness     worst |multiplier| * slack = 6.098e+03 on YP.DWLGS
- HiGHS on `qforplan`: [FAIL] dual feasibility (columns)  worst 1.604e-03 (1.604e-03 relative) on DEDO3 82; [FAIL] complementary slackness     worst |multiplier| * slack = 2.573e+01 on DEDO3 82; [FAIL] strong duality              primal 7.4566
- HiGHS on `qgfrdxpn`: [FAIL] dual feasibility (columns)  worst 4.381e-04 (3.678e-04 relative) on KB1LA1; [FAIL] complementary slackness     worst |multiplier| * slack = 3.510e+02 on XJ2DZ2
- HiGHS on `qisrael`: [FAIL] dual feasibility (columns)  worst 6.064e-04 (6.064e-04 relative) on A373; [FAIL] complementary slackness     worst |multiplier| * slack = 3.677e+00 on A373
- HiGHS on `qpcboei1`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.254e-02 on C-----69
- HiGHS on `qpcboei2`: [FAIL] complementary slackness     worst |multiplier| * slack = 7.700e-02 on C------9
- HiGHS on `qrecipe`: [FAIL] complementary slackness     worst |multiplier| * slack = 4.000e-05 on JAL1IOBE; [FAIL] strong duality              primal -2.666160000000e+02  dual -2.666175010000e+02  gap 1.501e-03 (relative 5.630e-06), 4.612e-0
- HiGHS on `qscagr25`: [FAIL] dual feasibility (columns)  worst 2.298e-03 (1.531e-04 relative) on COL00491; [FAIL] complementary slackness     worst |multiplier| * slack = 5.279e+01 on COL00491
- HiGHS on `qscagr7`: [FAIL] dual feasibility (columns)  worst 1.600e-04 (1.600e-04 relative) on COL00073; [FAIL] complementary slackness     worst |multiplier| * slack = 1.529e+00 on COL00131
- HiGHS on `qscfxm1`: [FAIL] dual feasibility (columns)  worst 1.547e-03 (1.545e-03 relative) on 1RMCST; [FAIL] complementary slackness     worst |multiplier| * slack = 2.395e+01 on 1RMCST
- HiGHS on `qscfxm2`: [FAIL] dual feasibility (columns)  worst 1.564e-03 (1.561e-03 relative) on 1RMCST; [FAIL] complementary slackness     worst |multiplier| * slack = 2.445e+01 on 1RMCST
- HiGHS on `qscfxm3`: [FAIL] dual feasibility (columns)  worst 1.564e-03 (1.561e-03 relative) on 1RMCST; [FAIL] complementary slackness     worst |multiplier| * slack = 2.445e+01 on 1RMCST
- HiGHS on `qscorpio`: [FAIL] dual feasibility (columns)  worst 2.677e-07 (1.834e-07 relative) on X0337
- HiGHS on `qscrs8`: [FAIL] dual feasibility (columns)  worst 2.103e-05 (2.103e-05 relative) on NELEDM75; [FAIL] complementary slackness     worst |multiplier| * slack = 4.424e-03 on NELEDM75
- HiGHS on `qsctap1`: [FAIL] dual feasibility (columns)  worst 1.000e-06 (1.818e-07 relative) on Z4ZZ7ZZ5; [FAIL] complementary slackness     worst |multiplier| * slack = 1.000e-05 on Z4ZZ7ZZ4
- HiGHS on `qseba`: [FAIL] dual feasibility (columns)  worst 7.896e-04 (7.896e-04 relative) on C0172000; [FAIL] complementary slackness     worst |multiplier| * slack = 6.235e+00 on C0172000
- HiGHS on `qshare2b`: [FAIL] dual feasibility (columns)  worst 6.000e-06 (6.000e-06 relative) on 010520; [FAIL] complementary slackness     worst |multiplier| * slack = 3.600e-04 on 010520
- HiGHS on `qshell`: [FAIL] complementary slackness     worst |multiplier| * slack = 6.132e+03 on C10010
- HiGHS on `qship04l`: [FAIL] complementary slackness     worst |multiplier| * slack = 6.357e-04 on PREG0101
- HiGHS on `qship04s`: [FAIL] complementary slackness     worst |multiplier| * slack = 6.438e-04 on PREG0101
- HiGHS on `qship08l`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.585e-04 on PREG0203
- HiGHS on `qship08s`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.589e-04 on PREG0203
- HiGHS on `qship12l`: [FAIL] complementary slackness     worst |multiplier| * slack = 8.495e-04 on PREG0407
- HiGHS on `qship12s`: [FAIL] complementary slackness     worst |multiplier| * slack = 8.517e-04 on PREG0407
- HiGHS on `qstandat`: [FAIL] dual feasibility (columns)  worst 9.830e-05 (9.829e-05 relative) on FTR.....; [FAIL] complementary slackness     worst |multiplier| * slack = 9.663e-02 on FTR.....
- HiGHS on `s268`: [FAIL] dual feasibility (columns)  worst 4.000e-07 (4.000e-07 relative) on C------5; [FAIL] strong duality              primal 1.818989403546e-12  dual 3.099996320088e-06  gap 3.100e-06 (relative 3.100e-06), 1.999e-07 of
- HiGHS on `stadat1`: [FAIL] dual feasibility (columns)  worst 1.930e+06 (1.901e+00 relative) on C----340; [FAIL] complementary slackness     worst |multiplier| * slack = 1.930e+06 on C----340
- HiGHS on `stadat2`: [FAIL] dual feasibility (columns)  worst 1.026e-06 (1.026e-06 relative) on C------1; [FAIL] complementary slackness     worst |multiplier| * slack = 1.414e-06 on C---2001; [FAIL] strong duality              primal -3.262
- SCIP on `cont-050`: [FAIL] objective           recomputed -4.563850884524e+00, solver said -4.563850981085e+00, difference 9.656e-08
- SCIP on `dpklo1`: [FAIL] objective           recomputed 3.700962516886e-01, solver said 3.700961596973e-01, difference 9.199e-08
- SCIP on `dual1`: [FAIL] objective           recomputed 3.501300191822e-02, solver said 3.501290376899e-02, difference 9.815e-08
- SCIP on `dual4`: [FAIL] objective           recomputed 7.460908862231e-01, solver said 7.460907878984e-01, difference 9.832e-08
- SCIP on `genhs28`: [FAIL] objective           recomputed 9.271737444168e-01, solver said 9.271736500756e-01, difference 9.434e-08
- SCIP on `hs35`: [FAIL] objective           recomputed 1.111111478922e-01, solver said 1.111110651898e-01, difference 8.270e-08
- SCIP on `hs35mod`: [FAIL] objective           recomputed 2.500000312530e-01, solver said 2.499999782114e-01, difference 5.304e-08
- SCIP on `hs51`: [FAIL] objective           recomputed 0.000000000000e+00, solver said -9.075448303975e-08, difference 9.075e-08
- SCIP on `hs53`: [FAIL] objective           recomputed 4.093023275720e+00, solver said 4.093023219721e+00, difference 5.600e-08
- SCIP on `hs76`: [FAIL] objective           recomputed -4.681818221798e+00, solver said -4.681818231789e+00, difference 9.990e-09
- SCIP on `ksip`: [FAIL] objective           recomputed 5.757979899730e-01, solver said 5.757978923372e-01, difference 9.764e-08
- SCIP on `primal1`: [FAIL] objective           recomputed -3.501293585563e-02, solver said -3.501301643532e-02, difference 8.058e-08
- SCIP on `primal2`: [FAIL] objective           recomputed -3.373363901253e-02, solver said -3.373373819399e-02, difference 9.918e-08
- SCIP on `qafiro`: [FAIL] objective           recomputed -1.590782842719e+00, solver said -1.590782851654e+00, difference 8.936e-09
- SCIP on `qe226`: [FAIL] row activity        worst violation 1.105e-07 (1.105e-07 relative to the row's terms) on ...054
- SCIP on `qisrael`: [FAIL] row activity        worst violation 9.990e-07 (9.990e-07 relative to the row's terms) on B173
- SCIP on `qpcblend`: [FAIL] objective           recomputed -7.842855006777e-03, solver said -7.842864304416e-03, difference 9.298e-09
- SCIP on `qpcboei2`: [FAIL] row activity        worst violation 4.114e-07 (4.114e-07 relative to the row's terms) on R-----49
- SCIP on `qptest`: [FAIL] objective           recomputed 4.371874914510e+00, solver said 4.371874904520e+00, difference 9.990e-09
- SCIP on `qsc205`: [FAIL] objective           recomputed -5.813959148495e-03, solver said -5.813968995113e-03, difference 9.847e-09
- SCIP on `qsctap1`: [FAIL] row activity        worst violation 7.073e-07 (7.073e-07 relative to the row's terms) on ACZZ9ZZ2
- SCIP on `qsctap2`: [FAIL] row activity        worst violation 7.097e-07 (7.097e-07 relative to the row's terms) on ACZ39ZZ2
- SCIP on `qsctap3`: [FAIL] row activity        worst violation 7.088e-07 (7.088e-07 relative to the row's terms) on ACZ32ZZ2
- SCIP on `tame`: [FAIL] objective           recomputed 5.187246390146e-23, solver said -9.985476434923e-09, difference 9.985e-09
- SCIP on `zecevic2`: [FAIL] objective           recomputed -4.124999992549e+00, solver said -4.125000059605e+00, difference 6.706e-08
- CBC/Clp on `aug2dcqp`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.583e-06 on C--19611
- CBC/Clp on `aug2dqp`: [FAIL] dual feasibility (columns)  worst 9.312e-07 (9.312e-07 relative) on C--19804; [FAIL] complementary slackness     worst |multiplier| * slack = 5.991e-05 on C--19804
- CBC/Clp on `cont-050`: [FAIL] objective                   recomputed -4.563850904326e+00, solver said -4.563850885233e+00, difference 1.909e-08
- CBC/Clp on `cont-100`: [FAIL] objective                   recomputed -4.644397868764e+00, solver said -4.644397861851e+00, difference 6.912e-09
- CBC/Clp on `cont-101`: [FAIL] objective                   recomputed 1.955273194356e-01, solver said 1.955273050684e-01, difference 1.437e-08
- CBC/Clp on `dpklo1`: [FAIL] objective                   recomputed 3.700962196558e-01, solver said 3.700962172567e-01, difference 2.399e-09
- CBC/Clp on `dual1`: [FAIL] objective                   recomputed 3.501297324012e-02, solver said 3.501296699663e-02, difference 6.243e-09
- CBC/Clp on `dual2`: [FAIL] objective                   recomputed 3.373368025771e-02, solver said 3.373363953785e-02, difference 4.072e-08
- CBC/Clp on `dual3`: [FAIL] objective                   recomputed 1.357558445771e-01, solver said 1.357558322585e-01, difference 1.232e-08
- CBC/Clp on `dual4`: [FAIL] objective                   recomputed 7.460908491974e-01, solver said 7.460908249191e-01, difference 2.428e-08
- CBC/Clp on `gouldqp2`: [FAIL] objective                   recomputed 1.842745033925e-04, solver said 1.837133764805e-04, difference 5.611e-07
- CBC/Clp on `hs268`: [FAIL] objective                   recomputed 3.510467649903e-08, solver said -5.036781658418e-09, difference 4.014e-08
- CBC/Clp on `hs35`: [FAIL] objective                   recomputed 1.111111202701e-01, solver said 1.111111019518e-01, difference 1.832e-08
- CBC/Clp on `hs35mod`: [FAIL] objective                   recomputed 2.500000118478e-01, solver said 2.499999719249e-01, difference 3.992e-08
- CBC/Clp on `hs53`: [FAIL] objective                   recomputed 4.093023250375e+00, solver said 4.093023255325e+00, difference 4.951e-09
- CBC/Clp on `hs76`: [FAIL] objective                   recomputed -4.681818176881e+00, solver said -4.681818185640e+00, difference 8.758e-09
- CBC/Clp on `huestis`: [FAIL] complementary slackness     worst |multiplier| * slack = 7.720e-03 on C------2
- CBC/Clp on `ksip`: [FAIL] objective                   recomputed 5.757979412412e-01, solver said 5.757978958171e-01, difference 4.542e-08
- CBC/Clp on `mosarqp2`: [FAIL] objective                   recomputed -1.576416353373e+03, solver said -1.530244759330e+03, difference 4.617e+01; [FAIL] dual feasibility (columns)  worst 9.750e-01 (9.750e-01 relative) on C----772; [FAIL] dual f
- CBC/Clp on `powell20`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.366e-05 on R---9976
- CBC/Clp on `primal1`: [FAIL] objective                   recomputed -3.501290856655e-02, solver said -3.501295335878e-02, difference 4.479e-08
- CBC/Clp on `primal2`: [FAIL] objective                   recomputed -3.373365616287e-02, solver said -3.373366498038e-02, difference 8.818e-09
- CBC/Clp on `primal3`: [FAIL] objective                   recomputed -1.357558254128e-01, solver said -1.357558342512e-01, difference 8.838e-09
- CBC/Clp on `primalc2`: [FAIL] objective                   recomputed -3.548649440958e+03, solver said -3.551305919049e+03, difference 2.656e+00; [FAIL] complementary slackness     worst |multiplier| * slack = 3.049e+00 on C------1
- CBC/Clp on `q25fv47`: [FAIL] dual feasibility (columns)  worst 3.520e+01 (1.833e+00 relative) on 1C1013; [FAIL] complementary slackness     worst |multiplier| * slack = 9.139e+01 on 1C1015
- CBC/Clp on `qadlittl`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.275e-05 on ...149
- CBC/Clp on `qafiro`: [FAIL] objective                   recomputed -1.590776580120e+00, solver said -1.590781146719e+00, difference 4.567e-06
- CBC/Clp on `qbeaconf`: [FAIL] objective                   recomputed 1.647120634900e+05, solver said 1.647120952022e+05, difference 3.171e-02; [FAIL] complementary slackness     worst |multiplier| * slack = 2.031e-03 on 10059S
- CBC/Clp on `qbore3d`: [FAIL] complementary slackness     worst |multiplier| * slack = 8.292e-05 on ION.DHXI
- CBC/Clp on `qbrandy`: [FAIL] row activity                worst violation 1.159e-06 (1.159e-06 relative to the row's terms) on 10131A; [FAIL] complementary slackness     worst |multiplier| * slack = 2.128e-04 on 102500
- CBC/Clp on `qe226`: [FAIL] objective                   recomputed 2.126738145550e+02, solver said 2.126748103066e+02, difference 9.958e-04; [FAIL] dual feasibility (columns)  worst 2.320e+00 (1.262e+00 relative) on .P0LYG; [FAIL] dual feasi
- CBC/Clp on `qforplan`: [FAIL] dual feasibility (columns)  worst 1.539e-03 (1.539e-03 relative) on DEDO3 82; [FAIL] complementary slackness     worst |multiplier| * slack = 4.271e+02 on DEDO5 12; [FAIL] strong duality              primal 7.4566
- CBC/Clp on `qgfrdxpn`: [FAIL] column bounds               worst violation 1.552e-06 (1.552e-06 relative) on FI1FJ1; [FAIL] complementary slackness     worst |multiplier| * slack = 2.946e-04 on P2UA
- CBC/Clp on `qgrow15`: [FAIL] complementary slackness     worst |multiplier| * slack = 3.573e-04 on XI0201
- CBC/Clp on `qgrow22`: [FAIL] complementary slackness     worst |multiplier| * slack = 8.964e-05 on SI1701
- CBC/Clp on `qgrow7`: [FAIL] objective                   recomputed -4.279830180767e+07, solver said -4.279879214731e+07, difference 4.903e+02; [FAIL] complementary slackness     worst |multiplier| * slack = 1.918e+01 on XI1505
- CBC/Clp on `qisrael`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.376e-03 on A340
- CBC/Clp on `qpcblend`: [FAIL] objective                   recomputed -7.842513105032e-03, solver said -7.842541175848e-03, difference 2.807e-08
- CBC/Clp on `qpcboei1`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.228e-03 on C----219
- CBC/Clp on `qpcboei2`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.783e-02 on C-----95
- CBC/Clp on `qpcstair`: [FAIL] complementary slackness     worst |multiplier| * slack = 6.706e-06 on R----281
- CBC/Clp on `qptest`: [FAIL] objective                   recomputed 4.371875013913e+00, solver said 4.371874993540e+00, difference 2.037e-08
- CBC/Clp on `qsc205`: [FAIL] objective                   recomputed -5.771121400589e-03, solver said 1.419142235815e+01, difference 1.420e+01; [FAIL] dual feasibility (columns)  worst 4.930e-04 (4.930e-04 relative) on COL00177; [FAIL] complem
- CBC/Clp on `qscagr25`: [FAIL] complementary slackness     worst |multiplier| * slack = 3.764e-05 on COL00003
- CBC/Clp on `qscagr7`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.167e-04 on ROW00041
- CBC/Clp on `qscfxm1`: [FAIL] row activity                worst violation 9.026e-07 (9.026e-07 relative to the row's terms) on 1DT071; [FAIL] objective                   recomputed 1.688284130238e+07, solver said 1.688268772313e+07, difference
- CBC/Clp on `qscfxm2`: [FAIL] row activity                worst violation 1.179e-06 (1.179e-06 relative to the row's terms) on 1DT071; [FAIL] objective                   recomputed 2.777663484293e+07, solver said 1.688531309338e+08, difference
- CBC/Clp on `qscfxm3`: [FAIL] row activity                worst violation 5.371e-07 (5.371e-07 relative to the row's terms) on 2DT071; [FAIL] objective                   recomputed 3.081671307074e+07, solver said 3.081623528605e+07, difference
- CBC/Clp on `qscsd8`: [FAIL] dual feasibility (columns)  worst 3.710e-06 (9.739e-07 relative) on 30006008
- CBC/Clp on `qsctap1`: [FAIL] objective                   recomputed 1.415861134552e+03, solver said 1.415860215188e+03, difference 9.194e-04; [FAIL] complementary slackness     worst |multiplier| * slack = 2.924e-06 on Z2ZZ5ZZ1
- CBC/Clp on `qsctap3`: [FAIL] objective                   recomputed 1.438754681177e+03, solver said 1.438754688970e+03, difference 7.793e-06
- CBC/Clp on `qseba`: [FAIL] complementary slackness     worst |multiplier| * slack = 4.065e-03 on M5245008
- CBC/Clp on `qshare1b`: [FAIL] complementary slackness     worst |multiplier| * slack = 7.847e-06 on CCC017
- CBC/Clp on `qshell`: [FAIL] complementary slackness     worst |multiplier| * slack = 1.648e-03 on C10550
- CBC/Clp on `qship08l`: [FAIL] objective                   recomputed 2.376040616519e+06, solver said 2.376040571576e+06, difference 4.494e-02; [FAIL] complementary slackness     worst |multiplier| * slack = 7.040e-06 on SH020244
- CBC/Clp on `qship08s`: [FAIL] complementary slackness     worst |multiplier| * slack = 2.845e-06 on OVRMIN02
- CBC/Clp on `qsierra`: [FAIL] objective                   recomputed 2.375118103720e+07, solver said 2.375046884817e+07, difference 7.122e+02; [FAIL] complementary slackness     worst |multiplier| * slack = 1.524e+00 on RBNBO3; [FAIL] strong d
- CBC/Clp on `qstair`: [FAIL] dual feasibility (columns)  worst 7.261e-06 (7.261e-06 relative) on ZM5; [FAIL] complementary slackness     worst |multiplier| * slack = 1.714e-06 on ZM5
- CBC/Clp on `qstandat`: [FAIL] row activity                worst violation 1.189e-07 (1.189e-07 relative to the row's terms) on TM.3S2T4; [FAIL] complementary slackness     worst |multiplier| * slack = 3.079e-06 on ZP12T408
- CBC/Clp on `s268`: [FAIL] objective                   recomputed 3.510467649903e-08, solver said -5.036781658418e-09, difference 4.014e-08
- CBC/Clp on `stadat1`: [FAIL] row activity                worst violation 2.263e+04 (4.230e-03 relative to the row's terms) on R---1339; [FAIL] objective                   recomputed -2.814939018558e+07, solver said -2.833845629765e+07, differ
- CBC/Clp on `stadat3`: [FAIL] objective                   recomputed -3.577945267267e+01, solver said -3.577945091082e+01, difference 1.762e-06
- CBC/Clp on `tame`: [FAIL] objective                   recomputed 0.000000000000e+00, solver said -3.712602037948e-08, difference 3.713e-08
- CBC/Clp on `ubh1`: [FAIL] objective                   recomputed 1.116000815695e+00, solver said 1.116000830621e+00, difference 1.493e-08
- CBC/Clp on `zecevic2`: [FAIL] objective                   recomputed -4.124999936391e+00, solver said -4.124999954272e+00, difference 1.788e-08

---

## 4b. Agreement against size-limited commercial editions (#533)

Gurobi Academic/Trial (≤2000 variables + constraints), CPLEX Community Edition (≤1000), and
GLPK (no size limit), each invoked as a **separate process** over the same MPS files. No
commercial code is linked into SANKHYA. Licence terms are recorded in `docs/PROVENANCE.md`.
The point of this section is not timing: on Netlib-scale instances every modern solver is
fast. What is being measured is **agreement of objective values and statuses** — because a
solver that claims the same answer as three independent implementations is more credible than
one that does not, regardless of how it compares on a wall clock.

No commercial-agreement run has been committed yet. Reproduce with:

```bash
python -m venv editions && editions/bin/pip install cplex   # CPLEX Community
python bench/runners/commercial_agreement.py --python editions/bin/python
```

Licence terms for each edition are in `docs/PROVENANCE.md`, judgement call 16: the pip editions of Gurobi and Xpress forbid publishing benchmark results, so only CPLEX Community is run by default.

---

## 5. Robustness — where the solver stops working

PS26119 asks for "a clear demonstration of numerical robustness ... involving degeneracy,
weak LP relaxations or ill-conditioned constraint matrices". `data/casestudies/` demonstrates
each hazard on one chosen instance; this section is the sweep that finds the case we do not
handle. Every instance is built from a chosen primal-dual pair, so its optimum is known
before it is solved (the construction is `tests/oracles/lp_generator.hpp`'s, in
`bench/runners/robustness.py`), and each family pushes one hazard until the answer, or the
certificate, moves. The reduced version runs in CI (`tests/robustness/`), together with the
adversarial families judged by the exact rational oracle and the classic cycling examples
of Beale and Kuhn.

Measured on commit `2b4eb6b` (Windows-AMD64), 192 solves, 5 families. Source: `robustness-2b4eb6b.csv`.

| family | parameter | last k that passed on every instance | first k that failed | what failed |
|---|---|---|---|---|
| `conditioning` | entry spread 10^k | 10 | 12 | infeasible, relative error 1.0e+00, verified no: row 7 needs activity of at least 8e-06 but the column bounds cap it at 0 |
| `near_parallel` | twin rows differing by a relative 10^-k | 16 | passes the whole sweep (k up to 16) | - |
| `cost_ratio` | costs spanning 10^k | 9 | 10 | optimal, relative error 5.7e-16, verified 0: [FAIL] complementary slackness     worst /multiplier/ * slack = 1.599e-05 on R4 |
| `redundancy` | k times the row count of implied rows | 32 | passes the whole sweep (k up to 32) | - |
| `degeneracy` | k times the column count of rows, all active | 64 | passes the whole sweep (k up to 64) | - |

Reading the table: the `conditioning` cliff is `kZeroDrop` (`tolerances.hpp`), the threshold below which a coefficient is treated as zero everywhere in the solver. At an entry spread of 1e12 the smallest coefficients fall under 1e-11, the model that gets solved is not the model that was written, and presolve then reports - correctly, about the truncated model - that a row cannot reach its bound. A model whose answer depends on a coefficient below 1e-11 is outside this solver's range; lowering the threshold would move the cliff, not remove it. The other limits are limits of the CERTIFICATE, not the answer: where the objective is right to 1e-15 and the verifier still rejects, the reduced costs or multipliers carry more rounding than its tolerances allow, which is worth knowing exactly because those tolerances are what a downstream consumer of the duals gets.

---

## 5b. The stress set - badly scaled Netlib and adversarial LPs, against HiGHS (#762)

Every Netlib LP with each row and column multiplied by a seeded power of two between 2^-20
and 2^20 (about 1e-6 to 1e6, so a matrix entry moves by up to 1e12), which is an exact
change of variables: the optimum is Koch's exact optimum of the original. Beside it, Klee-Minty
cubes n = 10..40, near-singular optimal bases, LPs infeasible by 1e-9..1e-6, Beale's and
Chvatal's cycling examples and heavily degenerate vertices, and unbounded LPs whose ray needs
two columns together. Generated by `bench/runners/stress_instances.py` (each file's sha256 in
`data/stress/reference.json`), solved by SANKHYA and by HiGHS in a separate process, and both
solvers' `.sol` files judged by the same `tools/verify_solution.py`. **correct** is the known
verdict with a file the verifier accepts (and, when optimal, the objective within 1e-6
relative); **wrong** is a contradicted verdict, an objective off the known optimum, or a file
the verifier rejects; **failed** is no verdict (a limit, a numerical error, `feasible`).

Reading it: SANKHYA's scaled-Netlib failures are the status guard refusing a claim it cannot
support, or a time limit, not answers. Three causes were fixed under #792: presolve reading
real coefficients below 1e-11 as zero (#824), complementary slackness judged absolutely
(#857, #806), and a numerical failure is now retried once on the model equilibrated by powers
of two, an exact change of variables (#886). What remains is tracked as #792 and #783.
HiGHS runs with its defaults, under which a matrix entry below `small_matrix_value` = 1e-9 is
dropped and a bound at or above `infinite_bound` = 1e20 is infinite. The scaled files carry
entries far below 1e-9, which is consistent with most of its scaled-Netlib points violating a
row of the file as written and most of its certificates failing to prove the file infeasible
(each failing check is named below); Klee-Minty from n = 30, whose last right-hand sides
reach 5^29 > 1e20, comes back unbounded.

Measured on SANKHYA commit `348d60ff` against HiGHS 1.15.1 (highspy, separate process), machine `laptop-i5-1135G7-7.7GB-Windows-AMD64-shared`, 126 instances. Source: `stress-348d60ff.csv`.

| family | instances | sankhya correct / wrong / failed | highs correct / wrong / failed |
|---|---|---|---|
| `degenerate` | 5 | 5 / 0 / 0 | 5 / 0 / 0 |
| `klee_minty` | 7 | 7 / 0 / 0 | 4 / 3 / 0 |
| `near_singular` | 8 | 8 / 0 / 0 | 8 / 0 / 0 |
| `scaled_netlib` | 94 | 73 / 0 / 21 | 0 / 87 / 7 |
| `thin_infeasible` | 8 | 8 / 0 / 0 | 8 / 0 / 0 |
| `unbounded` | 4 | 4 / 0 / 0 | 4 / 0 / 0 |
| **all** | 126 | 105 / 0 / 21 | 29 / 90 / 7 |

**sankhya, failed** (21):

- `scaled_80bau3b`: no verdict: time_limit (stopped at the time limit of 29.9981s after 30.00s, 61479 iterations, 0 nodes; route: the scaled attempt returned time_l)
- `scaled_bnl2`: no verdict: time_limit (time limit 29.9988s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_cycle`: no verdict: time_limit (stopped at the time limit of 29.9987s after 30.00s, 166058 iterations, 0 nodes; route: the scaled attempt returned time_)
- `scaled_d2q06c`: no verdict: time_limit (time limit 50.6978s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned nume)
- `scaled_dfl001`: no verdict: numerical_error (dual simplex: basic variable 15943 is outside its bounds by 3.146e+06, far above the 1.0e-07 feasibility tolerance, and )
- `scaled_etamacro`: no verdict: feasible (engine reported optimal but the largest /multiplier/ * slack is 1.122e-02 (1.122e-02 relative to its magnitudes), above )
- `scaled_fit2p`: no verdict: time_limit (stopped at the time limit of 29.9976s after 30.00s, 33175 iterations, 0 nodes; route: the scaled attempt returned time_l)
- `scaled_ganges`: no verdict: numerical_error (route: the scaled attempt returned optimal after 0.0 s of its 30 s share; neither attempt produced a usable point; the s)
- `scaled_modszk1`: no verdict: numerical_error (phase 1 diverged: the largest bound violation grew to 3.765e+08 from a least of 6.093e-08, which cannot happen on faithf)
- `scaled_nesm`: no verdict: time_limit (stopped at the time limit of 29.9989s after 30.00s, 130406 iterations, 0 nodes; route: the scaled attempt returned time_)
- `scaled_perold`: no verdict: time_limit (stopped at the time limit of 29.9995s after 30.00s, 7119 iterations, 0 nodes; route: the scaled attempt returned time_li)
- `scaled_pilot`: no verdict: numerical_error (basis became singular at iteration 896; route: the scaled attempt returned time_limit after 30.0 s of its 30 s share; ne)
- `scaled_pilot.ja`: no verdict: time_limit (time limit 29.9991s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_pilot.we`: no verdict: numerical_error (phase 1 diverged: the largest bound violation grew to 1.765e+06 from a least of 1.388e-17, which cannot happen on faithf)
- `scaled_pilot4`: no verdict: time_limit (time limit 29.9996s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_pilot87`: no verdict: time_limit (time limit 29.9913s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_pilotnov`: no verdict: time_limit (time limit 29.9976s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_stair`: no verdict: time_limit (time limit 29.9997s reached inside the basis factorization, which was abandoned; route: the scaled attempt returned time)
- `scaled_stocfor2`: no verdict: time_limit (stopped at the time limit of 29.9982s after 30.00s, 122773 iterations, 0 nodes; route: the scaled attempt returned time_)
- `scaled_stocfor3`: no verdict: time_limit (stopped at the time limit of 29.9937s after 29.99s, 19964 iterations, 0 nodes; route: the scaled attempt returned time_l)
- `scaled_woodw`: no verdict: numerical_error (route: the scaled attempt returned optimal after 0.2 s of its 30 s share; neither attempt produced a usable point; the s)

**highs, wrong** (90):

- `klee_minty_30`: unbounded, verifier rejects: [FAIL] ray respects the row bounds     17 would be crossed: R13, R14, R15, R16, R17
- `klee_minty_35`: unbounded, verifier rejects: [FAIL] ray respects the row bounds     22 would be crossed: R13, R14, R15, R16, R17
- `klee_minty_40`: unbounded, verifier rejects: [FAIL] ray respects the row bounds     27 would be crossed: R13, R14, R15, R16, R17
- `scaled_25fv47`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C474
- `scaled_80bau3b`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C1807
- `scaled_adlittle`: optimal, verifier rejects: [FAIL] row activity                      worst violation 4.385e-02 (4.385e-02 relative to the row's terms) on R27; [FAIL] activity agreement                max /ours - solver's/ = 4.385e-02; [F
- `scaled_afiro`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.043e-02 (2.043e-02 relative to the row's terms) on R4; [FAIL] activity agreement                max /ours - solver's/ = 2.043e-02; [FA
- `scaled_agg`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C119
- `scaled_agg2`: optimal, verifier rejects: [FAIL] row activity                      worst violation 7.601e+01 (1.977e+00 relative to the row's terms) on R438; [FAIL] activity agreement                max /ours - solver's/ = 1.850e+02; [
- `scaled_agg3`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.556e+00 (1.094e+00 relative to the row's terms) on R503; [FAIL] activity agreement                max /ours - solver's/ = 2.279e+02; [
- `scaled_bandm`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         11 unbounded in the direction used, so the aggregate proves nothing: C8, C88, C240, C278, C280
- `scaled_beaconfd`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C51
- `scaled_blend`: unbounded on an instance that is optimal
- `scaled_bnl1`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C4
- `scaled_bnl2`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C935
- `scaled_boeing1`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C33
- `scaled_boeing2`: optimal, verifier rejects: [FAIL] row activity                      worst violation 8.099e-03 (8.099e-03 relative to the row's terms) on R88; [FAIL] activity agreement                max /ours - solver's/ = 9.445e-03; [F
- `scaled_bore3d`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.766e-02 (1.766e-02 relative to the row's terms) on R190; [FAIL] activity agreement                max /ours - solver's/ = 1.766e-02; [
- `scaled_brandy`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C5
- `scaled_capri`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C260
- `scaled_cycle`: infeasible on an instance that is optimal
- `scaled_czprob`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C3267
- `scaled_d2q06c`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C3752
- `scaled_d6cube`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.441e-04 (2.441e-04 relative to the row's terms) on R406; [FAIL] activity agreement                max /ours - solver's/ = 2.441e-04; [
- `scaled_degen2`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C58, C59
- `scaled_degen3`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         3 unbounded in the direction used, so the aggregate proves nothing: C389, C390, C391
- `scaled_dfl001`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C4912
- `scaled_e226`: infeasible on an instance that is optimal
- `scaled_etamacro`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C226
- `scaled_fffff800`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C820
- `scaled_finnis`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C413, C414
- `scaled_fit1d`: optimal, verifier rejects: [FAIL] row activity                      worst violation 4.925e-04 (4.925e-04 relative to the row's terms) on R17; [FAIL] activity agreement                max /ours - solver's/ = 4.925e-04; [F
- `scaled_fit2d`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.282e-02 (1.282e-02 relative to the row's terms) on R13; [FAIL] activity agreement                max /ours - solver's/ = 1.282e-02; [F
- `scaled_forplan`: optimal, verifier rejects: [FAIL] row activity                      worst violation 6.199e-03 (6.199e-03 relative to the row's terms) on R10; [FAIL] activity agreement                max /ours - solver's/ = 6.199e-03; [F
- `scaled_ganges`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C211
- `scaled_gfrd-pnc`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.670e-01 (2.670e-01 relative to the row's terms) on R7; [FAIL] activity agreement                max /ours - solver's/ = 2.670e-01; [FA
- `scaled_greenbea`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C2678
- `scaled_greenbeb`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C1952
- `scaled_grow15`: optimal, verifier rejects: [FAIL] row activity                      worst violation 4.212e+00 (1.847e+00 relative to the row's terms) on R265; [FAIL] activity agreement                max /ours - solver's/ = 1.961e+02; [
- `scaled_grow22`: optimal, verifier rejects: [FAIL] row activity                      worst violation 6.851e+00 (1.626e+00 relative to the row's terms) on R139; [FAIL] activity agreement                max /ours - solver's/ = 1.451e+03; [
- `scaled_grow7`: optimal, verifier rejects: [FAIL] row activity                      worst violation 3.708e+00 (1.277e+00 relative to the row's terms) on R16; [FAIL] activity agreement                max /ours - solver's/ = 1.771e+02; [F
- `scaled_israel`: optimal, verifier rejects: [FAIL] row activity                      worst violation 9.693e+00 (1.000e+00 relative to the row's terms) on R0; [FAIL] activity agreement                max /ours - solver's/ = 9.693e+00; [FA
- `scaled_kb2`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.274e-03 (1.274e-03 relative to the row's terms) on R11; [FAIL] activity agreement                max /ours - solver's/ = 1.274e-03; [F
- `scaled_lotfi`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.753e-01 (1.901e-01 relative to the row's terms) on R138; [FAIL] activity agreement                max /ours - solver's/ = 2.753e-01; [
- `scaled_maros`: unbounded, verifier rejects: [FAIL] row activity                    worst violation 3.015e+00 (1.015e+00 relative to the row's terms) on R570; [FAIL] activity agreement              max /ours - solver's/ = 3.956e+01
- `scaled_maros-r7`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C5945
- `scaled_modszk1`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.454e+00 (1.000e+00 relative to the row's terms) on R685; [FAIL] activity agreement                max /ours - solver's/ = 1.454e+00; [
- `scaled_nesm`: infeasible, verifier rejects: [FAIL] infeasibility proof                the rows aggregate to at least 7.379150390625e-02, the column bounds allow at most 1.020935039062e-01, a contradiction of -2.830e-02
- `scaled_pilot`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C82, C2742
- `scaled_pilot.we`: infeasible, verifier rejects: [FAIL] certificate uses only real bounds  2 lean on an infinite bound: R60, R79
- `scaled_pilot4`: infeasible on an instance that is optimal
- `scaled_pilot87`: infeasible on an instance that is optimal
- `scaled_recipe`: infeasible, verifier rejects: [FAIL] infeasibility proof                the rows aggregate to at least 0.000000000000e+00, the column bounds allow at most 6.675720214844e-05, a contradiction of -6.676e-05
- `scaled_sc105`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.876e-02 (1.876e-02 relative to the row's terms) on R25; [FAIL] activity agreement                max /ours - solver's/ = 1.876e-02; [F
- `scaled_sc205`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.882e-02 (2.882e-02 relative to the row's terms) on R116; [FAIL] activity agreement                max /ours - solver's/ = 2.882e-02; [
- `scaled_sc50a`: optimal, verifier rejects: [FAIL] reduced costs                     max /c - A^T y - d/ = 1.112e-04 (1.112e-04 relative to its terms) on C44
- `scaled_sc50b`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.150e-04 (1.150e-04 relative to the row's terms) on R35; [FAIL] activity agreement                max /ours - solver's/ = 1.150e-04
- `scaled_scagr25`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C117, C126
- `scaled_scagr7`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C96
- `scaled_scfxm1`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C116
- `scaled_scfxm2`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C690
- `scaled_scfxm3`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C795
- `scaled_scorpion`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C249, C263
- `scaled_scrs8`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C1132
- `scaled_scsd1`: optimal, verifier rejects: [FAIL] row activity                      worst violation 2.271e-05 (2.271e-05 relative to the row's terms) on R7; [FAIL] activity agreement                max /ours - solver's/ = 2.271e-05; [FA
- `scaled_scsd6`: optimal, verifier rejects: [FAIL] row activity                      worst violation 3.052e-05 (3.052e-05 relative to the row's terms) on R101; [FAIL] activity agreement                max /ours - solver's/ = 3.052e-05; [
- `scaled_scsd8`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.659e-04 (1.659e-04 relative to the row's terms) on R173; [FAIL] activity agreement                max /ours - solver's/ = 1.659e-04; [
- `scaled_sctap1`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.831e-03 (1.831e-03 relative to the row's terms) on R150; [FAIL] activity agreement                max /ours - solver's/ = 1.831e-03; [
- `scaled_sctap2`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         3 unbounded in the direction used, so the aggregate proves nothing: C280, C281, C282
- `scaled_sctap3`: optimal, verifier rejects: [FAIL] row activity                      worst violation 9.766e-04 (9.766e-04 relative to the row's terms) on R171; [FAIL] activity agreement                max /ours - solver's/ = 9.766e-04; [
- `scaled_seba`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C76
- `scaled_share1b`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         2 unbounded in the direction used, so the aggregate proves nothing: C34, C133
- `scaled_share2b`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.908e-04 (1.908e-04 relative to the row's terms) on R74; [FAIL] activity agreement                max /ours - solver's/ = 4.232e-03; [F
- `scaled_shell`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C516
- `scaled_ship04l`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.314e-02 (1.314e-02 relative to the row's terms) on R59; [FAIL] activity agreement                max /ours - solver's/ = 1.314e-02; [F
- `scaled_ship04s`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C333
- `scaled_ship08l`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C177
- `scaled_ship08s`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C198
- `scaled_ship12l`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C5332
- `scaled_ship12s`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C2737
- `scaled_stair`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C406
- `scaled_standata`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C235
- `scaled_standmps`: infeasible, verifier rejects: [FAIL] infeasibility proof                the rows aggregate to at least 0.000000000000e+00, the column bounds allow at most 9.918212890625e-05, a contradiction of -9.918e-05
- `scaled_stocfor1`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C86
- `scaled_stocfor2`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         3 unbounded in the direction used, so the aggregate proves nothing: C28, C742, C1253
- `scaled_stocfor3`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         4 unbounded in the direction used, so the aggregate proves nothing: C102, C566, C1782, C1836
- `scaled_truss`: optimal, verifier rejects: [FAIL] row activity                      worst violation 1.562e-02 (1.562e-02 relative to the row's terms) on R458; [FAIL] activity agreement                max /ours - solver's/ = 1.562e-02; [
- `scaled_tuff`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C399
- `scaled_vtp.base`: infeasible, verifier rejects: [FAIL] aggregate is bounded above         1 unbounded in the direction used, so the aggregate proves nothing: C114
- `scaled_wood1p`: optimal, verifier rejects: [FAIL] reduced costs                     max /c - A^T y - d/ = 9.080e-06 (9.080e-06 relative to its terms) on C305

**highs, failed** (7):

- `scaled_fit1p`: no verdict: kSolveError
- `scaled_fit2p`: no verdict: time_limit
- `scaled_perold`: no verdict: kNotset
- `scaled_pilot.ja`: no verdict: kNotset
- `scaled_pilotnov`: no verdict: kNotset
- `scaled_sierra`: no verdict: kNotset
- `scaled_woodw`: no verdict: kNotset

14 thin-infeasible answers are `optimal` at a point the verifier accepts with every row inside 1e-7: at the stated tolerance that point is feasible, so the answer is graded correct, and named here so the grade is not mistaken for a proof of infeasibility.

---

## 6. What these numbers do not say

- **The large-model evidence is sections 1d and 1f to 1f.3, and none of it is a real
  million-variable industrial model.** 1d is Mittelmann's set, a table of named time limits.
  1f to 1f.3 are generated instances whose optimum is exact by construction: under a clock
  the first-order engine reaches 100,000 rows and columns, the interior point 5,000 on the
  random shape and 20,000 on the staircase, and the dual simplex 1,000; under a fixed
  iteration budget the random shape goes to 1,000,000; the largest refinery-shaped model
  solved exactly is 32,485 rows. No pass rate in the Netlib sections above substitutes for
  any of it. Tracked as #198 and as part of #54.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
  The comparison in section 4 uses solver-internal time on both sides for that reason.
- The failures in section 1b are real and are not going to be quietly dropped from a later
  edition of this file. Each one carries the issue tracking it.
- The GPU PDHG backend is measured in section 1g. The interior-point method (`algorithm=ipm`, #56) is opt-in and produces no
  basis, so it is not the engine behind any Netlib or MIPLIB table above - sections 1f to
  1f.3 are the exception, where it appears beside the others: since the AMD ordering (#193)
  it reaches 5,000 rows on the random shape and 20,000 on the staircase, and solves the
  32,485-row refinery year exactly (#206, #211). Its own Netlib run is committed as `netlib-full-*-ipm.csv` and quoted in
  `docs/PS26119_COVERAGE.md`, not here, because a run made with a non-default option is a
  measurement of that option rather than the tier's evidence. `docs/PROVENANCE.md` and issue #54 carry the full accounting.
