# Negative results

Features that were built, measured, and then demoted, left opt-in, or withdrawn — with the number that did it and the file the number lives in. This document exists because the decision *not* to ship something is evidence too, and it is the evidence a reader cannot recover from the code: an opt-in flag says nothing about why it is opt-in.

Every entry states which of three things it is:

- **Demoted by a committed measurement.** The CSV is in `bench/results/`, stamped with a commit on `main`, and `docs/BENCHMARKS.md` is generated from it. This is the standard the rest of the project holds itself to.
- **Demoted by a measurement made on a branch.** The number was produced and acted on, but the CSV was never re-run on `main` and is not committed. It is reported here as what it is: a decision on evidence that a reader cannot regenerate until the re-run happens.
- **Built opt-in, awaiting its measurement.** Nothing has demoted it; nothing has earned it a default either. Listed so that "off by default" is never mistaken for "measured and found wanting".

Withdrawn claims — numbers this project published and then retracted — are at the end. A withdrawn number stays named, with the run that replaced it.

---

## 1. Demoted by a committed measurement

### Cut rounds below the root (`tree_cut_depth`)

Rounds of MIR, clique and {0,1/2} cuts at every node down to a depth, with an aged pool (#351). Off: `tree_cut_depth = 0`.

| leg (30 MIPLIB instances, 60 s, `main` at `0018254`, alone on the machine) | reached | proved | nodes | wall |
|---|---:|---:|---:|---:|
| no cuts | 15 | 9 | 1.000× | 1.000× |
| root round (the default since `0018254`) | 13 | **10** | 1.015× | 0.986× |
| root + tree rounds, depth 4 | 12 | 9 | 0.850× | 0.999× |

The root round proves `neos-3611689-kaihu` (119.0, 57,576 nodes) that no cuts leaves `feasible` at the limit. Tree rounds lose it again: 120.0 `feasible` in 35,996 nodes — fewer nodes in the same wall time, because every tree round re-filters the whole waiting pool (#415). A node-count reduction that comes from dearer nodes is not a saving.

Files: `bench/results/miplib-cuts-off.csv`, `miplib-cuts-on.csv`, `miplib-cuts-tree.csv`. Section 2 of `docs/BENCHMARKS.md`.
What would earn the default: the per-node cost of the waiting pool coming down enough that depth 4 keeps the root's proof.

### Flow cover cut strengthening, three layers (#419)

One-row flow cover separation for the fixed-charge shape of `ran12x21`, `ran13x13` and `k16x240b` (Padberg, Van Roy & Wolsey 1985), then a single-node aggregation across rows, then a bounded local search over the cover. Each layer measured against its own same-commit baseline, `enable_root_cuts=true` in both legs, 60 s. Off: `enable_flow_cover_cuts = false`.

| layer | baseline | with the layer | on the three target instances |
|---|---:|---:|---|
| one-row separation (`91b6c08`) | 13 / 9 | 13 / 9 | `ran12x21` 3832 → 3853 (worse), `ran13x13` 3319 → 3297, `k16x240b` unchanged; none reaches its optimum |
| single-node aggregation (`ebafccf`) | 13 / 9 | 13 / 9 | the layer's own counters read **zero** on all three — their rows were already clean single-node rows, so it never fired |
| bounded local search (`8358f75`) | 13 / 9 | 15 / 9 | the +2 did not survive a same-commit re-check: `neos-3611689-kaihu` matched in 1 of 3 baseline legs and `noswot` in 2 of 3 |

Eight legs, every one negative on the instances the issue named. The family fires — 14 candidates on `ran12x21` alone, moving its root bound from 3157.38 to 3228.32 — and does not close enough gap in 60 s to change an outcome.

Files: `bench/results/miplib-419-{cuts-baseline,flow-cover-on,aggregation-baseline,aggregation-on,localsearch-baseline,localsearch-on}.csv`; the three stamps are branch commits carried to `main` by #454, recorded in `bench/results/squash-stamps.txt`. Section 2 of `docs/BENCHMARKS.md`.
What would earn the default: a target instance reaching its published optimum under the family. The next lever is separation strength — exact sequential lifting (Gu, Nemhauser & Savelsbergh 1999) — not the inequality or the filter.

### The Harris two-pass ratio test (`ratio_test=harris`)

Selectable beside the default ratio test (#67). Off.

| Netlib medium tier, 50 instances, `dba3a65` | optimal | total solve time | shifted geomean |
|---|---:|---:|---:|
| devex (default) | 49 / 50 | 8.0 s | 0.073 s |
| Harris alone | 49 / 50 | 154.8 s | 0.338 s |
| devex + Harris | 49 / 50 | 8.5 s | 0.078 s |

It changes no status and costs time: nineteen times slower alone, a rounding error beside devex. The stability it buys is not one this tier needed.

Files: `bench/results/netlib-medium-dba3a65-{devex,harris,devex-harris}.csv`.
What would earn the default: an instance class where devex goes singular and Harris does not — none in the committed sets does.

### OpenMP threads in the LP engines

Thread-parallel loops in the simplex and PDHG kernels, selectable by thread count. The default is one thread.

| Netlib subset, 8 instances, `0cd08cf` | optimal | total | shifted geomean |
|---|---:|---:|---:|
| 1 thread | 6 / 8 | 90.1 s | 7.83 s |
| 8 threads | 6 / 8 | 92.2 s | 8.70 s |

Eight threads are slower. At Netlib scale the per-iteration work is too small to amortise the fork, and the memory traffic of the BTRAN/FTRAN pair, not the arithmetic, is what an iteration costs.

Files: `bench/results/netlib-0cd08cf-threads{1,8}.csv`.
What would earn a default above one: a measured win at the sizes where the gather over `A` dominates — the 20,000-row staircase and above — which has not been run with threads.

### The GPU crossover as a single-run figure

Not a feature demoted but a *number* demoted: the CPU-vs-GPU PDHG crossover at 10,000×10,000 was published as one run. Two runs of the same protocol on the same card:

| run | CSV | 1e-4 crossover | 1e-8 crossover |
|---|---|---:|---:|
| #390 | `gpu-a4f02b1.csv` | 1.63× | 1.51× |
| #424 | `gpu-fb72ab4.csv` | 1.37× | 1.54× |

Same instances, same card, opposite movements at the two tolerances. The spread is the honest width of a laptop-GPU measurement, and a single two-decimal figure hides it. The runner now takes N repeats per cell and reports the median with its min–max (#449); a committed repeats CSV is #445.

### The Forrest–Tomlin basis update (`basis_update=forrest-tomlin`)

Landed in #279: the entering column folded into U with a sparse spike and one row eta, against the product-form default; #396 added Tomlin's loss-of-significance test (`kFtCancellationThreshold`), a row-eta bound and the pivot test after the fold was measured drifting 2.5e-3. Measured against the product form on the full Netlib set (94 instances), both sides the same binary built at `462fa39e`, alternated, 3 runs a side, `threads=1`, 600 s wall limit, on the shared Windows laptop at 100% CPU:

| | product form | Forrest–Tomlin |
|---|---:|---:|
| `pilot` | optimal, 9,900 iterations, 3 of 3 | **feasible, not optimal** (strong duality fails), 7,525 iterations, 3 of 3 |
| `pilot87` | optimal, 21,163 iterations, 2 of 3 (the third stopped by the limit at 21,044) | **time limit** 2 of 3 at 23,727 and 26,150 iterations, **phase 1 diverged** in the third |
| `dfl001` | optimal, 85,012 iterations | optimal, 70,654 iterations |
| exact-grade matches (Koch), per run | 92, 92, 93 | 90, 91, 91 |
| shifted geomean, 10 s, per-instance median | 1.000× | 1.025× |

It takes fewer iterations and less time on many mid-size instances (`d2q06c`, `truss`, `greenbea`, `dfl001`), but it loses `pilot` outright and `pilot87` in every run, and a wrong answer scores zero. The product form stays the default.

Files: `bench/results/lu-ft-ab-netlib-{product-form,forrest-tomlin}-462fa39e-r{1,2,3}.csv` and `lu-ft-ab-netlib-462fa39e-summary.csv`.
What would earn the default: `pilot` and `pilot87` optimal under it in every run — the accuracy of the row-eta file on the pilot family, not its speed.

### The primal heuristics that never earned a default (#414)

Each optional heuristic behind its own `mip_heur_*` switch, measured on alone against the defaults (rounding, the fractional dive, Feasibility Jump and parity): MIPLIB tier 1, 3 seeds, `node_limit=2000`, one thread, `main` at `729112d8`. Off by default (`auto`, which follows the `mip_heuristics` master switch, itself off); kept in the tree, since off they cost nothing.

| leg | matched / 90 | proved / 90 | moved |
|---|---:|---:|---|
| defaults | 44 | 28 | - |
| `mip_heur_lock_rounding`, `mip_heur_repair`, `mip_heur_pump`, `mip_heur_dive_coefficient`, `mip_heur_dive_vector_length`, `mip_heur_dive_guided`, `mip_heur_prop_dive` | 44 | 28 | nothing |
| `mip_heur_rens` | 43 | 28 | `neos-3611689-kaihu` one matched seed lost |
| `mip_heur_local_branching` | 43 | 28 | `neos-3611689-kaihu` one matched seed lost |

No leg moves the proved count, every leg finds a point on the same 84 of 90 runs, and the node ratio is 1.000x for every leg but local branching (0.998x). The tenth leg, `mip_heur_rins`, matched 45 and proved 28 (`kaihu` one seed gained), so #1076 proposed it as a default and gated it on a 60 s A/B at `main` `f6740af` (bench.yml, MIPLIB tier 1, 3 seeds, one thread, `mip_heur_rins=off` against `on`):

| leg | matched / 90 | proved / 90 | sgm time (s, shift 10) |
|---|---:|---:|---:|
| `mip_heur_rins=off` | 55 | 35 | 27.36 |
| `mip_heur_rins=on` | 57 | 29 | 28.45 |

RINS gains two matches (`neos-5140963-mincio`, `noswot`, `ran12x21` one seed each, `markshare_4_0` one lost) and loses six proofs: `enlight8` on all three seeds (proved in 35 to 60 s off, at the limit on), `neos-3611689-kaihu` seeds 0 and 1, `markshare_4_0` seed 1. #1076 was closed on that. The two legs landed on different runner CPUs (AMD EPYC 9V45 off, EPYC 7763 on), so part of the lost node throughput (`enlight8` seed 0: 692,437 against 408,841 nodes in the same 60 s) is the machine, not RINS; a same-machine re-run is what would reopen it. Files: `bench/results/rins-ab-miplib-60s-seeds3-{off,on}-f6740af9.csv` and their `summary-` files.

Files: `bench/results/miplib-414-n2000-<leg>-seeds3-729112d8.csv` and their `summary-` files, rendered under "every optional heuristic alone" in `docs/BENCHMARKS.md`.
What would earn a default: a leg that gains a matched or proved seed and loses neither, on this harness and again under a time limit.

### The multi-round root cut loop, with tailing off (`root_cut_loop`, #956, #1116)

Rounds of the root cut separators until the bound stalls, demoted in #956 at `f7cfd57`. #1116 added a tailing-off stop: a round gaining at most 0.1 of round 1's gain ends the loop (Padberg and Rinaldi 1991). It also cut the root's time share from 0.2 to 0.05. Root only, on the ten worst tier-1 instances, the tailed loop keeps most of the extra closure with about half the rows: k16x240b closes 40.1% of the root gap with 148 rows, against 48.6% with 256 untailed and 13.5% with one round. The 60 s A/B still loses.

| arm (`ab-option`, `main` at `e01ae9c`, tier 1, 3 seeds, 60 s, both arms in one job on one EPYC 7763 runner) | matched / 90 | proved / 90 | sgm time (s, shift 10) |
|---|---:|---:|---:|
| `root_cut_loop=false` (the default) | 53 | 30 | 28.33 |
| `root_cut_loop=true` | 49 | 28 | 28.49 |

- **Proofs lost:** both on `neos-3611689-kaihu`, seeds 0 and 2. The default proves them in 50 s and 32 s; with the loop the root adds 120 cut rows instead of 30, and the run stops at the limit.
- **Matches lost:** `ej` seeds 1 and 2, `markshare_4_0` seed 1, `neos5` seeds 0 and 1, `noswot` seed 2 and `kaihu` seed 2.
- **Matches gained:** `gen-ip054` seed 2, `markshare_4_0` seed 0 and `neos5` seed 2.
- **Node throughput:** lower on every run that moved; `ej` explored 416k nodes against 361k in the same 60 s.

The cost is still the rows every node LP carries, not the root's time.

Files: `bench/results/option-ab-miplib-rootloop-tailing-e01ae9c.csv` (the `arm` column), and the root-only legs `bench/results/rootgap-<leg>-24e289d9.csv`.
What would earn a default: a loop whose extra rows do not ride along in the tree. For example, keep only the rows still tight at the last root LP, or move the rest to the cut pool, then repeat this A/B.

---

## 2. Demoted by a measurement made on a branch, not re-run on `main`

These decisions were made on numbers that are not in `bench/results/`. They are recorded as such, and each names the re-run that would move it to section 1.

### Parallel tree search (`mip_threads`)

Landed in #403: N workers, each on its own subtree with its own node LPs, sharing the incumbent, the pool and the pseudocosts. The objective and status do not depend on N; the tree explored does. Default 1.

On the branch, 30 MIPLIB instances at 60 s: four threads reach the same 14 and prove the same 9 as one, explore 2× to 4× the nodes, and are slower on the small proofs — geometric mean 0.70× on the nine that finish. More nodes in the same time was not more proofs.

Re-run that would settle it: the A/B on `main` with the runner's `--threads` column.

### Conflict analysis (`mip_conflict_analysis`)

Landed in #381/#292: the search learns from every node it proves infeasible which branching decisions were to blame, and prunes or tightens later nodes that repeat them; a conflict is stored only after re-proof from the global bounds. Off by default.

On the branch at `6c405f8` (#405), 30 MIPLIB instances at 60 s: the same 14 reached and 9 proved either way, no verdict moved. On the nine instances that finish, conflicts take the node count to 0.965× (2,204 against 2,285). On the 21 stopped by the limit the count is 0.938× in the same wall clock — nodes not reached, not search saved — with `enlight8` at 0.373× and 37.4 s of its 60 s budget spent inside the analysis.

Re-run that would settle it: the same A/B at a `main` commit, with the analysis time per instance in the CSV.

### An infeasibility neighbourhood for the QP interior point (#981, branch `qp-ipm-pmm`, not merged)

The neighbourhood of the interior point-proximal method of multipliers (Pougkakiotis & Gondzio, Comput. Optim. Appl. 78, 2021; Wright 1997, ch. 6): the starting ratio mu / r is recorded and, whenever mu falls below a tenth of that ratio times the relative infeasibility left, the centring parameter is raised until the target sigma mu is not. The aim was the LISWET stalls, where the primal residual stays near 1e-7 while mu falls to 1e-33.

Measured at `45f31f8c` (branch) under WSL on the laptop with other jobs running, 60 s, the eleven instances the interior point stalls on, judged by iteration counts and the stall test's own measure (load changes how many iterations fit in 60 s, not what an iteration does):

| instance | off | on |
|---|---|---|
| liswet8 | stall at 210, worst measure 2.6e-3, gap 0.9965 | stall at 210, worst 3.8e-3, gap 0.9965 |
| liswet12 | stall at 210, worst 1.0e-2, gap 0.984 | stall at 210, worst 3.4e-2, gap 0.979 |
| yao | stall at 210, worst 0.21, gap 0.616 | stall at 210, worst 0.24, gap 0.578 |
| qpilotno | stall at 210, worst 1.0 | stall at 210, worst 1.0 |
| liswet1, 2, 7, 9, 10, 11 | gaps 0.30, 1.26e-6, 0.950, 0.987, 0.494, 0.487 | the same to three digits |

Where both arms reached the same iteration the stalled measure was no better with the floor on, and no instance moved toward its reference: keeping mu up does not unfreeze the residual, so the barrier collapsing is not what holds LISWET. Not merged, because an off-by-default option with no measured benefit is code without a reason. The CSVs were not committed (the branch stamp is not on `main`).

### An outer proximal method of multipliers for the QP interior point (#981, branch `qp-ipm-outer-pmm`, not merged)

The interior point moves both proximal centres to the current iterate at every iteration. This held the dual centre across inner iterations instead: each Newton step was the proximal subproblem's, M dv + delta dy = b - M v - delta (y - y_centre), refined against the matrix with delta kept and rho removed; the centre moved to y once the subproblem's residual was a tenth of the true one; delta was held at least 1e-6 (Rockafellar, Math. Oper. Res. 1, 1976; Pougkakiotis & Gondzio, Comput. Optim. Appl. 78, 2021). The aim was the LISWET stalls.

Measured at `6a31c05` (branch) in CI, `bench.yml`'s `maros-meszaros` suite on GitHub Actions `ubuntu-latest`, 60 s, both arms on the same binary (runs 39230376494 off and 39230384791 on): the eleven stall instances and twelve controls.

| | passed | verified | optimal | rejected optimal |
|---|---:|---:|---:|---:|
| off | 13 | 14 | 14 | 0 |
| on | 8 | 9 | 8 | 0 |

Nothing in the stall set improved: liswet2 lost its verified optimum (reached through the hand-off when off) to the 210-iteration stall, yao's gap went from 0.616 to 0.918, and the other LISWET instances ended where they were. Five controls that were optimal were lost - qseba, qship12s, qgrow15, powell20 and qcapri - most of them handed to the first-order engine after the interior point gave up. Holding the dual centre makes the step worse on well-posed models and does not help the degenerate ones. Not merged. The CSVs were not committed (the branch stamp is not on `main`).

### A derived reduced cost within the rounding of its own terms read as zero (#1079, #1130, branch `qp-rounding-reduced-cost`, not merged)

A QP's derived reduced cost c + Qx - A'y is a sum whose computed value carries rounding up to gamma_n * sum |terms| (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., eq. 3.5). The change read one at or below 1000 * eps * sum |terms| as zero, in the in-process gate and in the QP finish's complementarity product. The aim was qgfrdxpn and qshell, refused on such a reduced cost times a slack of thousands.

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, both arms interleaved per instance on one machine: base `76d880c`, new `524db7a`. The CSVs are `bench/results/option-ab-maros-meszaros-reduced-cost-rounding-524db7a.csv` (run 39637860103, 60 s) and `option-ab-netlib-reduced-cost-rounding-524db7a.csv` (run 39637869165).

| Maros-Meszaros 138 | optimal | matched | verified |
|---|---:|---:|---:|
| base | 119 | 117 | 124 |
| new | 122 | 119 | 123 |

Netlib was identical on all 94 instances in both arms (83 matched, 94 verified).

On Maros-Meszaros:
- **qgfrdxpn** became optimal, matched and verified. This is the gain.
- **qshell** became optimal and matched, but `tools/verify_solution.py` refuses it: "complementary slackness worst |multiplier| * slack = 1.986e-05 relative to its magnitudes on C11760". A false optimal.
- **huestis** became optimal and verified, still unmatched against the published value (the #1078 erratum).

The verifier is the independent judge and stays as it is. On qshell a 2e-5 relative product is a real defect in the point, not rounding. A rule that lets the gate pass it claims an optimum the judge rejects. Not merged.

### A size-scaled round budget for the QP active-set finish (#1079, #1140, branch `qp-finish-round-budget`, not merged)

The finish makes one working-set change per round, at most 10 rounds. A diagnostic build on CI (run 39642658047) logged qforplan pinning one new column in each of its 10 rounds without repeating one, and qfffff80 doing the same with violations up to 3.1e4. The change raised the budget to max(10, ceil(0.05 * columns)), giving qforplan 24 rounds and qfffff80 50.

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, both arms interleaved per instance on one machine: base `81b812c`, new `fc28df5`. The CSVs are `bench/results/option-ab-maros-meszaros-finish-round-budget-fc28df5.csv` (run 39647578284, 60 s) and `option-ab-netlib-finish-round-budget-fc28df5.csv` (run 39647586270).

- **Maros-Meszaros 138:** 119 optimal, 117 matched and 124 verified in both arms. The only row that differs is stadat1, iteration_limit -> time_limit, unsolved in both.
- **Netlib:** identical on all 94 instances.

The targets did not move:
- **qforplan:** still feasible with the same gap, 0.59 s -> 1.25 s.
- **qfffff80:** still feasible with the same gap, 26.8 s -> 61.5 s (now at the time limit).

More one-at-a-time rounds do not finish these working sets; they start too far from the optimal one. The interior point has to hand the finish a better point. Not merged: no gain and a measured cost.

---

### A flat lower QP regularization floor, Gondzio correctors, and Krylov refinement on every step (#981, PR #1150, kept off)

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, base `320fa9a1`, Maros-Meszaros 138 at 60 s, beside the adaptive dual floor that was adopted (`option-ab-maros-meszaros-qp-ipm-adaptive-dual-950712b.csv`). Each of these raised the verified count from 124 but lost an instance that was optimal:

- `qp_ipm_regularization=1e-12` (run 39736572856): 130 verified. It lost qforplan (feasible to the iteration ceiling) and cont-201 (to the time limit).
- That floor with `qp_ipm_krylov_refinement` and three `qp_ipm_centrality_correctors` (run 39736560914): 130 verified. The correctors end qisrael, qpcboei2 and qscagr25 with one product the KKT gate refuses. boyd2 and cont-300 ran out of time, because they paid for GMRES on every step.
- The adaptive floor with Krylov on every step (run 39752483470): 131 verified, boyd2 lost to time.
- The adaptive floor without Krylov (run 39752496156): 126 verified, cont-300 lost to time.

The options remain, off by default. The CSVs were not committed; the branch stamps are not on `main`.

## 3. Built opt-in, awaiting the measurement

Nothing has demoted these. Each is off by default until its A/B on `main` says what it changes, per the rule that a default is earned by a number.

| feature | option | what it is | the measurement it waits for |
|---|---|---|---|
| Dual steepest-edge pricing (#411, #434) | `pricing=dual-steepest-edge` | exact row norms of B⁻¹ maintained through the Forrest–Goldfarb update, one extra FTRAN per pivot | Netlib medium and the scale models, iterations and seconds under each rule |
| Objective branching (#418, #438) | `mip_objective_branching` | on an integral objective, branch on a split of the objective row instead of a column | the four plateau instances on MIPLIB |
| Dual fixing (#412, #429) | `presolve_dual_fixing` | a column whose entries only push rows away from a finite bound one way, and whose cost never rewards it, fixed at the other bound | Netlib and MIPLIB re-runs on `main` |
| Parallel rows (#412, #436) | `presolve_parallel_rows` | a row that is a scalar multiple of an earlier one folds its bounds into that row | same |
| Dominated columns (#412, #440) | `presolve_dominated_columns` | activity moves from a dominated column onto a dominating one; one of the two is fixed | same |
| Implied-free substitution (#412, #443) | `presolve_implied_free` | a singleton column whose row already implies its box is substituted away with its row | same |
| Coefficient tightening (#511) | `presolve_coefficient_tightening` | an integer column's coefficient larger than the row's slack at its maximum is cut to that slack (big-M to the column's implied bound), alternating with bound propagation | MIPLIB root bounds and times on `main` |
| Binary probing and clique table (#512) | `presolve_probing` | each binary fixed to 0 and to 1 and propagated: fixings, tightenings, and literal conflicts fed to the clique separator | MIPLIB root bounds, clique cuts and times on `main` |

---

## 4. Withdrawn claims

Numbers this project published and then retracted, kept here by name.

- **"Root plus tree rounds reach 14 and prove 10."** Published from the three-way at `078cb24`. The clean re-run at `0018254` — alone on the machine, apps closed — gave root + tree 12 / 9 and root alone 13 / 10. The `078cb24` tree figure did not reproduce; the default was set on the leg that did. Withdrawn in #459.
- **"1.63× faster at 10,000×10,000."** Published from `gpu-a4f02b1.csv`. The re-run `gpu-fb72ab4.csv` gave 1.37× at the same tolerance. Both are now named wherever the figure is stated, and the repeats runner (#449) replaces the single-run figure. Withdrawn in #435.
- **"0 of 8 on Mittelmann."** Published from `mittelmann-e134aeb.csv` for the default engine. The clean re-run at `0018254` on all three engines gave 2 of 8 under `auto` and 3 of 8 across engines; two of the eight are the interior point running out of memory on a 7.7 GB machine, and two are engine-selection misses (#417). Replaced in #459. The out-of-memory reading of that sentence is withdrawn in the next entry.
- **"The interior point is limited by memory on Mittelmann."** Read from the two out-of-memory rows of the `0018254` re-run on the 7.7 GB laptop. The same eight under `algorithm=ipm` on a 96 GB node with 86 GB free (`mittelmann-ipm-5c7efbc.csv`, #575) finish the same 2 of 8, and none of the six misses is memory: three abandon the ordering in 11–22 s at `ipm_max_ordering_entries = 1e8`, a budget sized for the laptop; two decline at the first factorization's share of the limit; `irish-electricity` is `numerical_error` because its point violates a row by 1.0e-4, which the status guard measures (objective 2454405 against HiGHS's 2546254). Memory was the 7.7 GB box's limit, not the engine's. The ordering and factor budgets are now sized from the machine's physical memory, with the laptop constants as the floor (#581); whether that moves the three abandoned orderings is the next ipm leg on the node, not yet run (#576).

---

## Adding an entry

An entry is a CSV first. Run the A/B at one `main` commit, alone on the machine, on AC, both legs in one sitting; commit the CSVs; regenerate `docs/BENCHMARKS.md`; then write the paragraph here with the file names and the number. A decision made on a branch run goes in section 2 with the re-run that would move it, not in section 1. A number that later fails to reproduce moves to section 4 with the run that replaced it — it is not deleted.
