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

Coverage: this run used **9 of the 89 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

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

Coverage: this run used **50 of the 89 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

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

### 1c. The full set — the honest headline

Every instance in Netlib's summary table. Both tiers above are defined by a row cap, which
makes them the easier half of the library by construction; this is the number Phase 6's
">= 95% of Netlib" exit criterion is measured against, and the one the README quotes.

Source CSV: `bench/results/netlib-full-5b2bd80.csv`  
Commit `5b2bd80` · machine `Windows-AMD64` · generated 2026-09-23T22:11:43+00:00

**81 of 89 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **89 of the 89 instances** Netlib publishes an optimal value for (set `full`, selected by `fetch_data.py --set full`). Phase 6's "full Netlib >= 95%" exit criterion is measured against this set.

**8 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| disagrees with the published optimum (#75) | 8 | 80bau3b, e226, ganges, greenbea, greenbeb, nesm, pilot, scrs8 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `25fv47` | 821 | 1571 | optimal | 5.5018458883e+03 | 5.5018458883e+03 | 2.4e-12 | 4107 | 1.139 | yes |
| `80bau3b` | 2262 | 9799 | optimal | 9.8722419241e+05 | 9.8723216072e+05 | 8.1e-06 | 3868 | 0.469 | yes |
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 100 | 0.017 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.015 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 150 | 0.022 | yes |
| `agg2` | 516 | 302 | optimal | -2.0239252356e+07 | -2.0239252356e+07 | 1.1e-12 | 159 | 0.031 | yes |
| `agg3` | 516 | 302 | optimal | 1.0312115935e+07 | 1.0312115935e+07 | 8.7e-12 | 164 | 0.026 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 484 | 0.028 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 115 | 0.019 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 99 | 0.017 | yes |
| `bnl1` | 643 | 1175 | optimal | 1.9776295615e+03 | 1.9776292856e+03 | 1.4e-07 | 1576 | 0.086 | yes |
| `bnl2` | 2324 | 3489 | optimal | 1.8112365404e+03 | 1.8112365404e+03 | 2.3e-11 | 2279 | 0.224 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 357 | 0.025 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 122 | 0.020 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 161 | 0.024 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 269 | 0.025 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 235 | 0.020 | yes |
| `cycle` | 1903 | 2857 | optimal | -5.2263930249e+00 | -5.2263930249e+00 | 1.1e-12 | 339 | 0.063 | yes |
| `czprob` | 929 | 3523 | optimal | 2.1851966989e+06 | 2.1851966989e+06 | 2.0e-11 | 991 | 0.089 | yes |
| `d2q06c` | 2171 | 5167 | optimal | 1.2278421081e+05 | 1.2278423615e+05 | 2.1e-07 | 14357 | 3.752 | yes |
| `d6cube` | 415 | 6184 | optimal | 3.1549166667e+02 | 3.1549166667e+02 | 1.1e-11 | 868 | 0.298 | yes |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 649 | 0.042 | yes |
| `degen3` | 1503 | 1818 | optimal | -9.8729400000e+02 | -9.8729400000e+02 | 1.2e-16 | 2930 | 0.385 | yes |
| `dfl001` | 6071 | 12230 | optimal | 1.1266396047e+07 | 1.1266400000e+07 | 3.5e-07 | 72217 | 50.639 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 508 | 0.031 | yes |
| `etamacro` | 400 | 688 | optimal | -7.5571523337e+02 | -7.5571521774e+02 | 2.1e-08 | 795 | 0.039 | yes |
| `fffff800` | 524 | 854 | optimal | 5.5567956482e+05 | 5.5567961165e+05 | 8.4e-08 | 797 | 0.047 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 377 | 0.032 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 56 | 0.033 | yes |
| `fit1p` | 627 | 1677 | optimal | 9.1463780924e+03 | 9.1463780924e+03 | 2.3e-12 | 1092 | 0.086 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.5e-12 | 32 | 0.230 | yes |
| `fit2p` | 3000 | 13525 | optimal | 6.8464293294e+04 | 6.8464293232e+04 | 9.0e-10 | 10150 | 3.248 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 322 | 0.029 | yes |
| `ganges` | 1309 | 1681 | optimal | -1.0958573613e+05 | -1.0958636356e+05 | 5.7e-06 | 1092 | 0.056 | yes |
| `gfrd-pnc` | 616 | 1092 | optimal | 6.9022359995e+06 | 6.9022359995e+06 | 7.1e-12 | 448 | 0.034 | yes |
| `greenbea` | 2392 | 5405 | optimal | -7.2555248130e+07 | -7.2462405908e+07 | 1.3e-03 | 41768 | 12.107 | yes |
| `greenbeb` | 2392 | 5405 | optimal | -4.3022602612e+06 | -4.3021476065e+06 | 2.6e-05 | 10256 | 1.667 | yes |
| `grow15` | 300 | 645 | optimal | -1.0687094129e+08 | -1.0687094129e+08 | 3.3e-11 | 3782 | 0.154 | yes |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 7041 | 0.362 | yes |
| `grow7` | 140 | 301 | optimal | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 1144 | 0.046 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 169 | 0.023 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 51 | 0.017 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 248 | 0.020 | yes |
| `maros` | 846 | 1443 | optimal | -5.8063743701e+04 | -5.8063743701e+04 | 2.2e-12 | 2014 | 0.147 | yes |
| `maros-r7` | 3136 | 9408 | optimal | 1.4971851665e+06 | 1.4971851665e+06 | 1.2e-11 | 28 | 6.948 | yes |
| `modszk1` | 687 | 1620 | optimal | 3.2061972906e+02 | 3.2061972906e+02 | 1.3e-11 | 664 | 0.049 | yes |
| `nesm` | 662 | 2923 | optimal | 1.4076036488e+07 | 1.4076073035e+07 | 2.6e-06 | 2560 | 0.177 | yes |
| `perold` | 625 | 1376 | optimal | -9.3807552782e+03 | -9.3807580773e+03 | 3.0e-07 | 2823 | 0.215 | yes |
| `pilot` | 1441 | 3652 | optimal | -5.5748972929e+02 | -5.5740430007e+02 | 1.5e-04 | 5793 | 1.661 | yes |
| `pilot4` | 410 | 1000 | optimal | -2.5811392589e+03 | -2.5811392641e+03 | 2.0e-09 | 1616 | 0.172 | yes |
| `pilot87` | 2030 | 4883 | optimal | 3.0171069146e+02 | 3.0171072827e+02 | 1.2e-07 | 33985 | 33.397 | yes |
| `pilotnov` | 975 | 2172 | optimal | -4.4972761882e+03 | -4.4972761882e+03 | 4.2e-12 | 11429 | 1.199 | yes |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 40 | 0.017 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 87 | 0.017 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 188 | 0.019 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.015 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.017 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 457 | 0.025 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 111 | 0.019 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 451 | 0.029 | yes |
| `scfxm2` | 660 | 914 | optimal | 3.6660261565e+04 | 3.6660261565e+04 | 3.3e-14 | 922 | 0.053 | yes |
| `scfxm3` | 990 | 1371 | optimal | 5.4901254550e+04 | 5.4901254550e+04 | 4.5e-12 | 1478 | 0.079 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 242 | 0.023 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 610 | 0.038 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 144 | 0.024 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000077e+01 | 5.0500000078e+01 | 1.7e-11 | 363 | 0.036 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 1302 | 0.102 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 253 | 0.024 | yes |
| `sctap2` | 1090 | 1880 | optimal | 1.7248071429e+03 | 1.7248071429e+03 | 2.5e-11 | 738 | 0.045 | yes |
| `sctap3` | 1480 | 2480 | optimal | 1.4240000000e+03 | 1.4240000000e+03 | 0.0e+00 | 1111 | 0.066 | yes |
| `seba` | 515 | 1028 | optimal | 1.5711600000e+04 | 1.5711600000e+04 | 1.2e-16 | 439 | 0.027 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 136 | 0.019 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.020 | yes |
| `shell` | 536 | 1775 | optimal | 1.2088253460e+09 | 1.2088253460e+09 | 0.0e+00 | 463 | 0.033 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 449 | 0.032 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 297 | 0.028 | yes |
| `ship08l` | 778 | 4283 | optimal | 1.9090552114e+06 | 1.9090552114e+06 | 5.7e-12 | 743 | 0.060 | yes |
| `ship08s` | 778 | 2387 | optimal | 1.9200982105e+06 | 1.9200982105e+06 | 1.8e-11 | 422 | 0.034 | yes |
| `ship12l` | 1151 | 5427 | optimal | 1.4701879193e+06 | 1.4701879193e+06 | 2.0e-11 | 1064 | 0.074 | yes |
| `ship12s` | 1151 | 2763 | optimal | 1.4892361344e+06 | 1.4892361344e+06 | 4.1e-12 | 589 | 0.043 | yes |
| `sierra` | 1227 | 2036 | optimal | 1.5394362184e+07 | 1.5394362184e+07 | 2.4e-11 | 552 | 0.046 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 446 | 0.038 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 58 | 0.024 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 203 | 0.026 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.017 | yes |
| `stocfor2` | 2157 | 2031 | optimal | -3.9024408538e+04 | -3.9024408538e+04 | 3.0e-12 | 1836 | 0.129 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 241 | 0.031 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 494 | 0.160 | yes |
| `woodw` | 1098 | 8405 | optimal | 1.3044763331e+00 | 1.3044763331e+00 | 1.2e-11 | 3012 | 0.632 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.263s**
- slowest solved instance: 50.639s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `80bau3b`, `e226`, `ganges`, `greenbea`, `greenbeb`, `nesm`, `pilot`, `scrs8`** — kept in the table on purpose

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

Not yet run at this commit. Reproduce with:

```
python bench/runners/fetch_kennington.py
python bench/runners/kennington.py --time-limit 600
python bench/runners/kennington.py --time-limit 600 --solver-option algorithm=pdhg   # and simplex, dual-simplex, ipm
```

### 1d. Beyond Netlib — Mittelmann's LP set

Netlib's largest instance has about 6,000 rows. PS26119 asks about "thousands to millions
of variables", and the only honest way to say where this solver stands on that is to run
instances of that size and name what happens. These are the eight smallest archives in
Mittelmann's LP test set (`bench/runners/fetch_mittelmann.py`, provenance in
`data/mittelmann/reference.json`).

Source CSV: `bench/results/mittelmann-5c7efbc.csv`  
Commit `5c7efbc` · machine `Linux-x86_64` · time limit 300 s per instance, both solvers

**2 of 8** instances reached `optimal` inside the limit; **2 of 8** also passed the independent verifier and agree with HiGHS. HiGHS, run as a separate process under the same limit, finished **4 of 8**.

These are the smallest archives in Mittelmann's LP directory; against Netlib's largest instance (dfl001, 6,071 rows, 35,632 nonzeros) they range from the same row count with 2.7x the nonzeros (qap15) to 62x the rows and 42x the nonzeros (bdry2). No published optimum exists for them, so there is no pass-against-a-number column: the outcome is the status, the verifier's verdict where a solution was written, and HiGHS's objective where HiGHS finished. `our objective` on a `time_limit` row is the last iterate's value, not a bound, and is printed only so that a later run can be compared with it.

| instance | rows | cols | nonzeros | status | our objective | HiGHS objective | rel. diff | iters | solver time (s) | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `Linf_520c` | 93326 | 69004 | 566193 | time_limit | 0.1864263131 | Time limit reached | - | 171831 | 300.0 | - |
| `bdry2` | 376500 | 250998 | 1500003 | time_limit | 0.001999991456 | Time limit reached | - | 39422 | 241.0 | - |
| `brazil3` | 14646 | 23968 | 133184 | optimal | 2 | 2 | 3.7e-11 | 15 | 1.8 | yes |
| `chromaticindex1024-7` | 67583 | 73728 | 270324 | optimal | 3 | 3 | 2.6e-12 | 560 | 21.2 | yes |
| `irish-electricity` | 104259 | 61728 | 523257 | time_limit | 2454384.171 | 2546254.563 | - | 228067 | 240.0 | - |
| `qap15` | 6330 | 22275 | 94950 | time_limit | 1277.50546 | Time limit reached | - | 84064 | 300.0 | - |
| `rmine15` | 358395 | 42438 | 879732 | time_limit | -5042.482374 | Time limit reached | - | 76058 | 216.2 | - |
| `supportcase10` | 165684 | 14770 | 555082 | time_limit | 3.383923731 | 3.383923666 | - | 183732 | 220.9 | - |

**Not solved inside the limit**, named rather than dropped: `Linf_520c`, `bdry2`, `irish-electricity`, `qap15`, `rmine15`, `supportcase10`.

#### The same eight under each engine

Source CSVs: `bench/results/mittelmann-5c7efbc.csv` (dual simplex), `bench/results/mittelmann-pdhg-5c7efbc.csv` (PDHG), `bench/results/mittelmann-ipm-5c7efbc.csv` (interior point)  
Same 300 s limit per instance and engine; HiGHS is not re-run here.

| instance | dual simplex: status · verified · time (s) | PDHG: status · verified · time (s) | interior point: status · verified · time (s) |
|---|---|---|---|
| `Linf_520c` | time_limit · - · 300.0 | time_limit · - · 231.4 | numerical_error · - · 21.5 |
| `bdry2` | time_limit · - · 241.0 | time_limit · - · 241.1 | not_solved · - · 86.5 |
| `brazil3` | optimal · yes · 1.8 | optimal · yes · 58.0 | optimal · yes · 1.8 |
| `chromaticindex1024-7` | optimal · yes · 21.2 | optimal · yes · 0.7 | numerical_error · - · 20.4 |
| `irish-electricity` | time_limit · - · 240.0 | time_limit · - · 240.0 | numerical_error · - · 170.1 |
| `qap15` | time_limit · - · 300.0 | optimal · yes · 93.4 | optimal · yes · 200.3 |
| `rmine15` | time_limit · - · 216.2 | time_limit · - · 216.2 | not_solved · - · 60.2 |
| `supportcase10` | time_limit · - · 220.9 | time_limit · - · 220.9 | numerical_error · - · 11.0 |

Finished and verified inside the limit: dual simplex **2 of 8**, PDHG **3 of 8**, interior point **2 of 8**.

### 1e. The first-order engine — PDHG

The simplex is not the only continuous engine. Restarted PDHG (`--option algorithm=pdhg`) is
a first-order method: no basis, no factorization, and a cost that depends enormously on the
accuracy asked of it - which is why this section reports two tolerances separately rather
than one blended number. It is also the engine the GPU work targets, so its CPU behaviour is
the baseline every GPU claim will be measured against.

Source CSV: `bench/results/pdhg-threads-5daee10.csv`  
Commit `5daee10` · machine `Windows-AMD64` · 0 instances, the ones committed to the repository


`optimal` here means what it means everywhere else in this document: the point also survives the project's absolute tolerances, not merely the relative ones the first-order loop converges on. That distinction is the whole of #179 - the loop used to stop on the relative measure and the report then downgraded the point it stopped on, so the engine gave up early and handed back the weaker answer.

**Read the two tolerance columns together, because they are the same run.** Since #179 the loop stops only where the absolute standard is met, so a request looser than that standard no longer stops the solve any earlier - ask for 1e-4 and you get the 1e-8 point, at the 1e-8 cost. That is the honest reading of the identical columns below, and it is a real trade: the old behaviour honoured a loose request and returned a point it then had to label `feasible`. #180 made that the opt-in: `--option pdhg_stop_at_request=true` waives the dual, gap and complementarity halves of the standard - absolute primal feasibility is kept, so `feasible` still means a feasible point - and reports the point as `feasible` unless it meets the full standard anyway. Measured on these instances at 1e-4 it costs 0.85x the iterations (`bench/results/pdhg-stop-at-request-02cd92c.csv`) and turns `share2b` from an iteration limit into a usable point at 807,760. The two tolerance columns stay identical on `adlittle`, `israel` and `sc50b` even with the switch on, because on those the kept primal clause is what binds.

| instance | simplex |  |
|---|---:|


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

| rows×cols | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 200×200 | 0.027 | 2.495 | 0.01× | 0.068 | 3.365 | 0.02× |
| 500×500 | 0.105 | 0.597 | 0.18× | 0.137 | 7.505 | 0.02× |
| 1000×1000 | 0.040 | 0.738 | 0.05× | 0.041 | 0.602 | 0.07× |
| 2000×2000 | 0.698 | 5.081 | 0.14× | 0.715 | 7.467 | 0.10× |
| 5000×5000 | 0.784 | 1.436 | 0.55× | 0.770 | 1.635 | 0.47× |
| 10000×10000 | 2.855 | 2.082 | **1.37×** | 2.929 | 1.903 | **1.54×** |

GPU: NVIDIA GeForce RTX 5050 Laptop GPU (compute 12.0, 8151 MiB VRAM).  
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

Not yet run. Reproduce with:

```
python bench/runners/fetch_mittelmann.py
python bench/runners/gpu_real_instances.py --binary build_gpu/sankhya
```

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

The crossover, the same protocol as 1g (`bench/runners/gpu_report.py`, medians of repeats with their min-max):

Source CSV: `bench/results/gpu-l4-fdc89c5.csv`  
Commit `fdc89c5` · machine `Linux-x86_64`

Both columns time PDHG alone (`pdhg_polish=false`) on the solver's own clock, to the tolerance named, the CPU side on one thread (#487); a warm-up GPU solve absorbed CUDA's context creation before the timed ones. The GPU pays a per-iteration launch and transfer cost that a small model cannot amortise; the crossover is where the parallel products start to pay for it.

Each cell is the median of 5 solves; `[min–max]` shows the spread from run-to-run variance (thermal state, clock boost on the laptop GPU).

| rows×cols | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 200×200 | 0.043 [0.043–0.044] | 0.663 [0.407–1.738] | 0.07× | 0.091 [0.090–0.092] | 0.723 [0.622–0.796] | 0.13× |
| 500×500 | 0.178 [0.171–0.182] | 0.832 [0.542–3.330] | 0.21× | 0.228 [0.226–0.229] | 1.327 [1.287–2.707] | 0.17× |
| 1000×1000 | 0.064 [0.063–0.064] | 0.267 [0.240–0.373] | 0.24× | 0.064 [0.063–0.066] | 0.270 [0.244–0.364] | 0.24× |
| 2000×2000 | 1.025 [1.014–1.034] | 1.449 [1.264–1.795] | 0.71× | 1.028 [1.019–1.063] | 1.302 [1.260–2.244] | 0.79× |
| 5000×5000 | 0.890 [0.879–0.905] | 0.434 [0.429–0.562] | **2.05×** | 0.884 [0.870–0.904] | 0.501 [0.457–0.540] | **1.77×** |
| 10000×10000 | 2.854 [2.829–3.001] | 0.678 [0.610–0.799] | **4.21×** | 2.906 [2.856–2.928] | 0.745 [0.644–0.779] | **3.90×** |

GPU: NVIDIA L4 (compute 8.9, 22478 MiB VRAM).  
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

Source CSV: `bench/results/gpu-real-l4-fdc89c5.csv`  
Commit `fdc89c5` · machine `Linux-x86_64`  
GPU: NVIDIA L4 (compute 8.9, 22478 MiB VRAM)

Same protocol as §1g: PDHG alone, solver clock, warm-up GPU solve per instance. Report the result whichever way it goes.

| instance | rows | CPU 1e-4 (s) | GPU 1e-4 (s) | speedup | CPU 1e-8 (s) | GPU 1e-8 (s) | speedup |
|----------|-----:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| `brazil3` | 14646 | 37.829 | 5.843 | 6.47× | 91.143 | 10.987 | 8.30× |
| `chromaticindex1024-7` | 67583 | 1.114 | 0.395 | 2.82× | 1.109 | 0.382 | 2.91× |
| `refinery_year` | 779640 | 300.411 | 300.477 | 1.00× | 300.441 | 300.458 | 1.00× |

The datacenter runner (`bench/runners/gpu_datacenter.py`, #488):

`gpu-datacenter-l4-fdc89c5.csv` - NVIDIA L4 (compute 8.9, 22478 MiB VRAM), solver at `fdc89c5`, Linux-x86_64, 3 repeats per cell:

| instance | mode | tol | forced iterations | status | objective | iterations | solver (s) | median wall (s) | spread (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `chromaticindex1024-7` | cpu-16t | 0.0001 | - | feasible | 3.0000000022999487 | 480 | 1.103508 | 1.445666 | 0.129435 |
| `chromaticindex1024-7` | gpu | 0.0001 | - | feasible | 3.0000000012276438 | 440 | 0.438591 | 0.853233 | 0.022590 |
| `chromaticindex1024-7` | cpu-16t | 1e-06 | - | feasible | 3.0000000022999487 | 480 | 1.130150 | 1.417274 | 0.266043 |
| `chromaticindex1024-7` | gpu | 1e-06 | - | feasible | 3.000000000165927 | 480 | 0.369944 | 0.825440 | 0.117898 |
| `chromaticindex1024-7` | cpu-16t | 1e-08 | - | feasible | 3.0000000022999487 | 480 | 1.037373 | 1.502796 | 0.054597 |
| `chromaticindex1024-7` | gpu | 1e-08 | - | feasible | 3.0000000005021317 | 440 | 0.379603 | 0.834822 | 0.111375 |
| `chromaticindex1024-7` | cpu-16t | 1e-08 | 2000 | feasible | 3.0000000022999487 | 480 | 1.034538 | 1.445967 | 0.082200 |
| `chromaticindex1024-7` | gpu | 1e-08 | 2000 | feasible | 3.0000000006988623 | 440 | 0.370927 | 0.847714 | 0.072507 |
| `brazil3` | cpu-16t | 0.0001 | - | feasible | 1.999998811677631 | 80800 | 33.762208 | 34.172888 | 0.326525 |
| `brazil3` | gpu | 0.0001 | - | feasible | 2.0000010287786005 | 58000 | 4.610044 | 5.553975 | 3.634450 |
| `brazil3` | cpu-16t | 1e-06 | - | feasible | 2.0000000851361506 | 101600 | 42.906782 | 43.094890 | 0.681074 |
| `brazil3` | gpu | 1e-06 | - | feasible | 1.9999995978437894 | 91320 | 7.212001 | 10.508593 | 4.449151 |
| `brazil3` | cpu-16t | 1e-08 | - | feasible | 1.9999999975461507 | 194240 | 80.629251 | 79.984344 | 1.851811 |
| `brazil3` | gpu | 1e-08 | - | feasible | 1.9999996940528124 | 129000 | 10.044387 | 10.431059 | 3.828625 |
| `brazil3` | cpu-16t | 1e-08 | 2000 | iteration_limit | 0.0 | 2000 | 0.846638 | 1.165236 | 0.048787 |
| `brazil3` | gpu | 1e-08 | 2000 | iteration_limit | 0.0 | 2000 | 0.315572 | 0.688317 | 0.193310 |

#### 1g.4 What the CPU side does with its cores

Every CPU column above is one thread. `pdhg_parallel_spmv` (#487) computes A x row-parallel
over the `threads` workers, bitwise the same at any thread count (the test holds it to the
bit); this is what it buys, per instance, at a fixed iteration count:

Source CSV: `pdhg-threads-5daee10.csv`  
Commit `5daee10` · machine `Windows-AMD64` · 2000 iterations per solve, PDHG alone, `pdhg_parallel_spmv=true` except the `serial` rows.

| instance | rows | threads | A x | solver (s) | speed-up over 1 thread |
|---|---:|---:|---|---:|---:|
| `kkt_1000x1000` | 1000 | 1 | serial | 0.054 | - |
| `kkt_1000x1000` | 1000 | 1 | parallel | 0.064 | 1.000x |
| `kkt_1000x1000` | 1000 | 2 | parallel | 0.162 | 0.396x |
| `kkt_1000x1000` | 1000 | 4 | parallel | 0.272 | 0.236x |
| `kkt_1000x1000` | 1000 | 8 | parallel | 0.413 | 0.156x |
| `kkt_2000x2000` | 2000 | 1 | serial | 0.117 | - |
| `kkt_2000x2000` | 2000 | 1 | parallel | 0.121 | 1.000x |
| `kkt_2000x2000` | 2000 | 2 | parallel | 0.192 | 0.633x |
| `kkt_2000x2000` | 2000 | 4 | parallel | 0.298 | 0.408x |
| `kkt_2000x2000` | 2000 | 8 | parallel | 0.484 | 0.251x |
| `kkt_5000x5000` | 5000 | 1 | serial | 0.367 | - |
| `kkt_5000x5000` | 5000 | 1 | parallel | 0.414 | 1.000x |
| `kkt_5000x5000` | 5000 | 2 | parallel | 0.508 | 0.816x |
| `kkt_5000x5000` | 5000 | 4 | parallel | 0.509 | 0.814x |
| `kkt_5000x5000` | 5000 | 8 | parallel | 0.703 | 0.589x |
| `kkt_10000x10000` | 10000 | 1 | serial | 0.796 | - |
| `kkt_10000x10000` | 10000 | 1 | parallel | 0.973 | 1.000x |
| `kkt_10000x10000` | 10000 | 2 | parallel | 0.970 | 1.003x |
| `kkt_10000x10000` | 10000 | 4 | parallel | 0.918 | 1.060x |
| `kkt_10000x10000` | 10000 | 8 | parallel | 1.026 | 0.948x |
| `refinery_year` | 779640 | 1 | serial | 112.059 | - |
| `refinery_year` | 779640 | 1 | parallel | 102.912 | 1.000x |
| `refinery_year` | 779640 | 2 | parallel | 85.705 | 1.201x |
| `refinery_year` | 779640 | 4 | parallel | 81.997 | 1.255x |
| `refinery_year` | 779640 | 8 | parallel | 88.361 | 1.165x |

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
---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

Source CSV: `bench/results/miplib-5daee10.csv`  
Commit `5daee10` · machine `Windows-AMD64`

**13 of 30** instances reached the published optimum. **10 of 30** also PROVED it - closed the bound to within the requested gap target rather than stopping at a node or time limit.

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

**The time limit decides some of these, not the solver.** A row that stops at the limit with a small gap says "needs more time than we gave it", not "cannot"; which side of the limit such a row lands on moves with the machine's speed rather than with anything about the search. The remedy is a longer limit, and the reason this table does not already use one is that the set already adds up to 21 minutes of solve time per run at this one.

Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone has found, not a proof, and scoring against one would let a wrong answer look like a record.

| instance | rows | cols | int | status | our objective | published | rel. gap | nodes | time (s) | matched | proved | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `b-ball` | 30 | 100 | 88 | feasible | -1.5 | -1.5 | 1.85e-01 | 111312 | 60.1 | yes | **NO** | yes |
| `ej` | 1 | 3 | 3 | feasible | 51015 | 25508 | 1.00e+00 | 72054 | 60.0 | **NO** | **NO** | yes |
| `enlight8` | 64 | 128 | 128 | time_limit | inf | 27 | - | 120473 | 60.0 | **NO** | **NO** | **NO** |
| `enlight_hard` | 100 | 200 | 200 | time_limit | inf | 37 | - | 110779 | 60.0 | **NO** | **NO** | **NO** |
| `f2gap40400` | 40 | 400 | 400 | optimal | 20772 | 20772 | 9.63e-05 | 336 | 3.4 | yes | yes | yes |
| `flugpl` | 18 | 18 | 11 | optimal | 1201500 | 1201500 | 0.00e+00 | 701 | 0.1 | yes | yes | yes |
| `gen-ip016` | 24 | 28 | 28 | feasible | -9438.111136 | -9476.155197 | 6.59e-03 | 113511 | 60.0 | **NO** | **NO** | yes |
| `gen-ip054` | 27 | 30 | 30 | feasible | 6859.084635 | 6840.965642 | 1.02e-02 | 118506 | 60.0 | **NO** | **NO** | yes |
| `gr4x6` | 34 | 48 | 24 | optimal | 202.35 | 202.35 | 0.00e+00 | 44 | 0.0 | yes | yes | yes |
| `gt2` | 29 | 188 | 188 | optimal | 21166 | 21166 | 0.00e+00 | 78 | 0.1 | yes | yes | yes |
| `k16x240b` | 256 | 480 | 240 | feasible | 12506 | 11393 | 4.04e-01 | 85923 | 60.0 | **NO** | **NO** | yes |
| `markshare1` | 6 | 62 | 50 | feasible | 29 | 1 | 1.00e+00 | 129267 | 60.0 | **NO** | **NO** | yes |
| `markshare_4_0` | 4 | 34 | 30 | feasible | 7 | 1 | 1.00e+00 | 167528 | 60.0 | **NO** | **NO** | yes |
| `markshare_5_0` | 5 | 45 | 40 | feasible | 19 | 1 | 1.00e+00 | 138434 | 60.0 | **NO** | **NO** | yes |
| `neos-1425699` | 89 | 105 | 85 | optimal | 3179698977 | 3179698977 | 0.00e+00 | 3 | 0.0 | yes | yes | yes |
| `neos-3072252-nete` | 432 | 576 | 144 | feasible | 12009820 | 11807698 | 1.10e-01 | 41900 | 60.0 | **NO** | **NO** | yes |
| `neos-3611689-kaihu` | 323 | 421 | 88 | optimal | 119 | 119 | 0.00e+00 | 57576 | 44.0 | yes | yes | yes |
| `neos-5140963-mincio` | 184 | 196 | 183 | feasible | 14818 | 14393 | 2.16e-01 | 80762 | 60.0 | **NO** | **NO** | yes |
| `neos-5192052-neckar` | 57 | 180 | 24 | optimal | -11670000 | -11670000 | 0.00e+00 | 9 | 0.0 | yes | yes | yes |
| `neos5` | 63 | 63 | 53 | feasible | 15.5 | 15 | 9.68e-02 | 57540 | 60.0 | **NO** | **NO** | yes |
| `noswot` | 182 | 128 | 100 | feasible | -40 | -41.00000885 | 7.50e-02 | 88724 | 60.0 | **NO** | **NO** | yes |
| `opt1217` | 64 | 769 | 768 | feasible | -16 | -16 | 2.50e-01 | 74396 | 60.0 | yes | **NO** | yes |
| `p0201` | 133 | 201 | 201 | optimal | 7615 | 7615 | 0.00e+00 | 396 | 1.3 | yes | yes | yes |
| `pk1` | 45 | 86 | 55 | feasible | 14 | 11 | 6.32e-01 | 108237 | 60.0 | **NO** | **NO** | yes |
| `ran12x21` | 285 | 504 | 252 | feasible | 3681 | 3664 | 5.64e-02 | 56653 | 60.0 | **NO** | **NO** | yes |
| `ran13x13` | 195 | 338 | 169 | feasible | 3319 | 3252 | 5.47e-02 | 74286 | 60.0 | **NO** | **NO** | yes |
| `rlp1` | 68 | 461 | 450 | feasible | 15 | 15 | 6.67e-02 | 82611 | 60.0 | yes | **NO** | yes |
| `supportcase14` | 234 | 304 | 304 | optimal | 288 | 288 | 0.00e+00 | 46 | 0.3 | yes | yes | yes |
| `supportcase16` | 130 | 319 | 319 | optimal | 288 | 288 | 0.00e+00 | 41 | 0.3 | yes | yes | yes |
| `timtab1` | 171 | 397 | 171 | feasible | 1106773 | 764772 | 7.33e-01 | 82922 | 60.0 | **NO** | **NO** | yes |

**Not proved optimal**, named rather than dropped: `b-ball`, `ej`, `enlight8`, `enlight_hard`, `gen-ip016`, `gen-ip054`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-3072252-nete`, `neos-5140963-mincio`, `neos5`, `noswot`, `opt1217`, `pk1`, `ran12x21`, `ran13x13`, `rlp1`, `timtab1`.

#### The same set at 600 s

Source CSV: `bench/results/miplib-600s-cca77e0.csv` (600 s per instance), beside `bench/results/miplib-5daee10.csv` (60 s)  
Commit `54e561b`

At 600 s: **15 of 30** reach the published optimum, **9 of 30** prove it.

| instance | 60 s: status · matched · proved | 600 s: status · matched · proved · gap | verdict |
|---|---|---|---|
| `b-ball` | feasible · yes · no | feasible · yes · no · 2.1e-01 | needs a bound (#221) |
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
| `markshare_4_0` | feasible · no · no | feasible · no · no · 1.0e+00 | needs an incumbent (#290) |
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

---

## 2b. Maros-Meszaros, the convex QP set

The 138 convex QPs of Maros and Meszaros, *A repository of convex quadratic programming
problems*, Optimization Methods and Software 11-12 (1999): the set every convex QP paper
reports. Solved by the default QP engine (Condat-Vu, `src/qp/`) and judged the way the
published QP benchmark judges it, on primal residual, dual residual and duality gap at 1e-6
and at 1e-9, as well as against the published objective and by the independent verifier.

Not yet run on `main`. Reproduce with:

```
python bench/runners/fetch_maros_meszaros.py
python bench/runners/maros_meszaros.py
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

Not yet run at this commit. Reproduce with:

```
python bench/runners/fetch_netlib_infeasible.py
python bench/runners/netlib_infeasible.py --time-limit 60
python bench/runners/netlib_infeasible.py --time-limit 60 --solver-option algorithm=simplex   # and dual-simplex, pdhg
```

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

Source CSV: `bench/results/compare-highs-medium-bf3df02-third.csv`  
Commit `bf3df02` · machine `Windows-AMD64`

**50 of 50** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.002 | 0.002 | 1.15x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.001 | 0.001 | 1.02x |
| `agg` | -3.59917673e+07 | -3.59917673e+07 | yes | 0.014 | 0.006 | 2.52x |
| `bandm` | -1.58628018e+02 | -1.58628018e+02 | yes | 0.030 | 0.014 | 2.16x |
| `beaconfd` | 3.35924858e+04 | 3.35924858e+04 | yes | 0.006 | 0.004 | 1.66x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.003 | 0.002 | 1.39x |
| `boeing1` | -3.35213568e+02 | -3.35213568e+02 | yes | 0.023 | 0.015 | 1.51x |
| `boeing2` | -3.15018728e+02 | -3.15018728e+02 | yes | 0.005 | 0.004 | 1.40x |
| `bore3d` | 1.37308039e+03 | 1.37308039e+03 | yes | 0.007 | 0.004 | 1.75x |
| `brandy` | 1.51850990e+03 | 1.51850990e+03 | yes | 0.014 | 0.007 | 1.95x |
| `capri` | 2.69001291e+03 | 2.69001291e+03 | yes | 0.014 | 0.005 | 2.64x |
| `d6cube` | 3.15491667e+02 | 3.15491667e+02 | yes | 0.548 | 0.195 | 2.81x |
| `degen2` | -1.43517800e+03 | -1.43517800e+03 | yes | 0.065 | 0.020 | 3.19x |
| `e226` | -1.16389291e+01 | -1.16389291e+01 | yes | 0.031 | 0.011 | 2.84x |
| `etamacro` | -7.55715233e+02 | -7.55715233e+02 | yes | 0.045 | 0.015 | 2.94x |
| `finnis` | 1.72791066e+05 | 1.72791066e+05 | yes | 0.023 | 0.008 | 2.89x |
| `fit1d` | -9.14637809e+03 | -9.14637809e+03 | yes | 0.015 | 0.016 | 0.96x |
| `fit2d` | -6.84642933e+04 | -6.84642933e+04 | yes | 0.343 | 0.230 | 1.50x |
| `forplan` | -6.64218961e+02 | -6.64218961e+02 | yes | 0.020 | 0.009 | 2.25x |
| `grow15` | -1.06870941e+08 | -1.06870941e+08 | yes | 0.357 | 0.059 | 6.08x |
| `grow22` | -1.60834336e+08 | -1.60834336e+08 | yes | 0.514 | 0.119 | 4.32x |
| `grow7` | -4.77878118e+07 | -4.77878118e+07 | yes | 0.123 | 0.016 | 7.55x |
| `israel` | -8.96644822e+05 | -8.96644822e+05 | yes | 0.013 | 0.006 | 2.26x |
| `kb2` | -1.74990013e+03 | -1.74990013e+03 | yes | 0.002 | 0.001 | 1.64x |
| `lotfi` | -2.52647061e+01 | -2.52647061e+01 | yes | 0.010 | 0.004 | 2.95x |
| `pilot4` | -2.58113926e+03 | -2.58113926e+03 | yes | 0.146 | 0.044 | 3.36x |
| `recipe` | -2.66616000e+02 | -2.66616000e+02 | yes | 0.002 | 0.002 | 0.92x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.003 | 0.001 | 1.84x |
| `sc205` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.007 | 0.003 | 2.62x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.001 | 0.001 | 1.55x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.001 | 0.001 | 1.53x |
| `scagr25` | -1.47534331e+07 | -1.47534331e+07 | yes | 0.024 | 0.010 | 2.44x |
| `scagr7` | -2.33138982e+06 | -2.33138982e+06 | yes | 0.003 | 0.003 | 0.93x |
| `scfxm1` | 1.84167590e+04 | 1.84167590e+04 | yes | 0.028 | 0.013 | 2.08x |
| `scorpion` | 1.87812482e+03 | 1.87812482e+03 | yes | 0.012 | 0.004 | 2.92x |
| `scrs8` | 9.04296954e+02 | 9.04296954e+02 | yes | 0.050 | 0.015 | 3.43x |
| `scsd1` | 8.66666667e+00 | 8.66666667e+00 | yes | 0.011 | 0.003 | 3.29x |
| `scsd6` | 5.05000001e+01 | 5.05000001e+01 | yes | 0.031 | 0.011 | 2.85x |
| `scsd8` | 9.05000000e+02 | 9.05000000e+02 | yes | 0.253 | 0.059 | 4.28x |
| `sctap1` | 1.41225000e+03 | 1.41225000e+03 | yes | 0.013 | 0.010 | 1.36x |
| `share1b` | -7.65893186e+04 | -7.65893186e+04 | yes | 0.006 | 0.004 | 1.40x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.005 | 0.003 | 1.71x |
| `ship04l` | 1.79332454e+06 | 1.79332454e+06 | yes | 0.040 | 0.013 | 3.09x |
| `ship04s` | 1.79871470e+06 | 1.79871470e+06 | yes | 0.022 | 0.009 | 2.45x |
| `stair` | -2.51266951e+02 | -2.51266951e+02 | yes | 0.047 | 0.021 | 2.24x |
| `standata` | 1.25769950e+03 | 1.25769950e+03 | yes | 0.006 | 0.006 | 0.96x |
| `standmps` | 1.40601750e+03 | 1.40601750e+03 | yes | 0.016 | 0.009 | 1.75x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.003 | 0.002 | 1.72x |
| `tuff` | 2.92147765e-01 | 2.92147765e-01 | yes | 0.016 | 0.010 | 1.58x |
| `wood1p` | 1.44290241e+00 | 1.44290241e+00 | yes | 0.160 | 0.100 | 1.61x |

**Summary**

- SANKHYA shifted geometric mean: **0.057s**
- HiGHS shifted geometric mean: **0.022s**
- SANKHYA is **2.6x** the HiGHS time by that measure

- per-instance ratio: median **2.12x**, worst **7.55x**, faster than HiGHS on **4 of 50** instances

We are **2.62x slower** than HiGHS by this measure, and publish that rather than bury it. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme. This solver now has a presolve (#43, #92) and a dual simplex (#65) of its own, both defaults, so what remains between the two is the pricing and the years. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

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
