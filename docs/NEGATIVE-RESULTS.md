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

### The primal heuristics, one switch each (#414)

Rounding, fractional diving, RENS and the diving family each behind `mip_heur_*`, with a counted budget. The master switch is off because, measured together, they cost proofs; the per-heuristic A/B that would say which one is not on `main` (`docs/BENCHMARKS.md` reports "not measured on this checkout").

Re-run that would settle it: `bench/runners/miplib_heuristics_ab.py` at one `main` commit, all legs in one sitting.

### The Forrest–Tomlin basis update (`basis_update=forrest-tomlin`)

Landed in #279: the entering column folded into U with a sparse spike and one row eta, against the product-form default. The fold was measured drifting 2.5e-3 where the product form stayed at 1e-9 over a full eta file, which #396 closed with Tomlin's loss-of-significance test (`kFtCancellationThreshold`), a row-eta bound and the pivot test — the unit test now holds it to the product form's drift over six seeds and 128 updates. Still opt-in: on the branch it won iterations on `greenbea` and `d2q06c` and lost them on `perold` and `pilotnov`, and no committed run puts a number on the trade.

Re-run that would settle it: the Netlib full set under both updates at one `main` commit.

---

## 3. Built opt-in, awaiting the measurement

Nothing has demoted these. Each is off by default until its A/B on `main` says what it changes, per the rule that a default is earned by a number.

| feature | option | what it is | the measurement it waits for |
|---|---|---|---|
| Dual steepest-edge pricing (#411, #434) | `pricing=dual-steepest-edge` | exact row norms of B⁻¹ maintained through the Forrest–Goldfarb update, one extra FTRAN per pivot | Netlib medium and the scale models, iterations and seconds under each rule |
| Objective branching (#418, #438) | `mip_objective_branching` | on an integral objective, branch on a split of the objective row instead of a column | the four plateau instances on MIPLIB |
| Formulation symmetry (#413, #441) | `mip_symmetry` | colour refinement and individualisation; one ordering row per verified generator | `enlight8`, `enlight_hard`, `markshare1` |
| Dual fixing (#412, #429) | `presolve_dual_fixing` | a column whose entries only push rows away from a finite bound one way, and whose cost never rewards it, fixed at the other bound | Netlib and MIPLIB re-runs on `main` |
| Parallel rows (#412, #436) | `presolve_parallel_rows` | a row that is a scalar multiple of an earlier one folds its bounds into that row | same |
| Dominated columns (#412, #440) | `presolve_dominated_columns` | activity moves from a dominated column onto a dominating one; one of the two is fixed | same |
| Implied-free substitution (#412, #443) | `presolve_implied_free` | a singleton column whose row already implies its box is substituted away with its row | same |

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
