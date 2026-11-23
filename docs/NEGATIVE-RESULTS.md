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

**The lifted form, on tier 1 (#1131).** With the sequence-independent lifting of Gu, Nemhauser and Savelsbergh (1999) in place, the family measured again as `enable_flow_cover_cuts=true` against the defaults on MIPLIB tier 1: `ab-option`, `main` at `1f468a6`, 3 seeds, 60 s, one thread, both arms in one job on one runner. The two arms are identical run for run: 53 matched, 33 proved and 90 verified of 90 either way, shifted geometric mean 28.00 s both. On tier 1 the family recognises a switched flow on k16x240b only (root only, it closes 24.2% of that root gap lifted against 23.9% unlifted), and the single round's 30-row selection never takes a flow cover over the Gomory and MIR cuts there. File: `bench/results/option-ab-miplib-flowcover-lifted-1f468a6.csv` (the `arm` column). The set the family is built for is MIPLIB 3's fixed-charge and lot-sizing models (#1158), where its reading is separate.

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

Landed in #279; #396 added Tomlin's loss-of-significance test (`kFtCancellationThreshold`), a row-eta bound and the product form's own pivot test. Since then, under #1042: contiguous row and column files for U (#1043), the update over alpha's support (#1077), the spike kept from the entering column's FTRAN (#1101), hyper-sparse solves (#1134), and, in #1282, the dual iteration's index lists carried through every Forrest–Tomlin solve, the update's partial BTRAN over its reach and an O(1) move of the leaving step. #1282 took ken-18 under Forrest–Tomlin from 289.7 G to 53.7 G instructions on the same 51,462-iteration path (callgrind, bench run 43796138692).

Measured against the product form after all of that, one binary, both arms in one job, GitHub-hosted runner:

| | product form | Forrest–Tomlin |
|---|---:|---:|
| Netlib, 94 x 3 repeats: optimal / exact (Koch) / verified / certified | 94 / 94 / 94 / 93 every repeat | 94 / 94 / 94 / 93 every repeat |
| Netlib solver-seconds SGM (medians, 1 ms floor) | 1.000 | **1.103**, faster in 44 of 282 pairs |
| `pilot` | 1.12 s, 5,694 iterations | 8.33 s, 6,255 (the scaled attempt ends `numerical_error`, the unscaled retry answers) |
| `pilot87` | 8.62 s, 18,887 iterations | 105.9 s, 11,100 (the first attempt ends `numerical_error`, the equilibrated retry answers) |
| `fit2p` / `truss` (wall) | 1.204 s / 1.331 s | 1.007 s / 1.200 s |
| Kennington, 16 x 3 repeats: optimal and verified | 16 every repeat | 16 every repeat |
| Kennington solver-seconds SGM | 1.000 | **1.107**, faster in 7 of 48 pairs |
| `ken-18` | 5.47 s | 5.85 s |

The pilot family no longer ends wrong or unsolved, as it did at `462fa39e` (feasible-not-optimal on `pilot`, time limits and a diverged phase 1 on `pilot87`), but it ends right only through the retry ladder, and the per-iteration gains do not carry the rest of the set past the product form.

Where the pilot family's time goes (#1042 step 4, a verbose solve log under callgrind, bench run 43878383727, main `1659c8b5`). On `pilot87` the scaled attempt runs 100,724 iterations with 2,245 refactorizations, 1,794 of them for an update declined as unsafe (1.8% of iterations) and 234 forced by the accuracy check; it passes over 1,526 rows whose pivot disagrees between the row and the column on **fresh** factors, repairs 41 singular bases, stalls in dual degeneracy at iteration 54,103, and ends in the primal at a primal infeasibility of 3.2e3. `pilot` does the same at a smaller scale (6,999 dual iterations, then 23 singular repairs in the primal). The decline rate is about the product form's on `pilot87` (366 declines in 18,887 iterations, the figure in #1042's step-4 brief, not re-measured in that run). Disagreement on fresh factors and singular repairs are properties of the bases the path visits, not of how the updates are represented. One run does not say why the Forrest–Tomlin path reaches those bases and the product form's does not, so no stability change was made and `kUpdatePivotThreshold` is unchanged.

Files: `bench/results/option-ab-netlib-ft-vs-pf-c0e28b8.csv` and `option-ab-kennington-ft-vs-pf-55abeac.csv` (#1282; both stamps are that PR's branch commits, whose library code `main` carries as `1659c8b5` after the squash merge), and the earlier `lu-ft-ab-netlib-{product-form,forrest-tomlin}-462fa39e-r{1,2,3}.csv` with `lu-ft-ab-netlib-462fa39e-summary.csv`. The profile and the verbose logs are run artifacts, not CSVs: a diagnosis, not a benchmark claim.
What would earn the default: the scaled attempt on `pilot` and `pilot87` ending optimal under it, then the Netlib and Kennington SGMs at or below 1.0. The next measurement is the split of the declines by which test refused them (the product form's pivot test, Tomlin's cancellation test, the row-eta bound), and the iteration at which the Forrest–Tomlin path first leaves the product form's.

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

**On MIPLIB 3 (#1158), alone and with the pool and the aggregated flow covers.** The 65-instance set is where the loop moves the root: set1ch, pp08a, fixnet6 and vpm2 close 0.3 to 0.6 of their root gap under it (root only, `bench/results/rootgap3-<cfg>-b69a3dc7.csv`). Two A/Bs at 60 s, 3 seeds, one thread, both arms in one bench.yml job:

| A/B | arm | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---|---:|
| (a) `main` d178213, run 41613095648 | defaults | 39/34/63 | 37/34/63 | 36/34/62 | 112/102/188 | 21.33 |
| | `root_cut_loop=true` | 40/36/63 | 38/36/62 | 38/34/63 | 116/106/188 | 20.22 |
| (b) `main` 94cdcc0 against #1183's 99dfa36, run 41755022388 | `main`'s defaults | 39/35/63 | 37/35/63 | 38/35/63 | 114/105/189 | 20.42 |
| | loop + pool + aggregated flow covers | 41/39/62 | 40/37/63 | 40/38/63 | 121/114/188 | 17.38 |

Proved is above the defaults on every seed in both, and both lose runs, so neither earns the default under the rule (proved >= on every seed and nothing lost). (a) loses 3 proofs and 6 matches; (b) loses 8 proofs (10teams s1, air05 s2, bell5 s1 and s2, gen s1, gesa2_o s0, misc06 s2, qiu s0), 8 matches and arki001 seed 0's feasible point. Two kinds of loss: gen, misc06, gesa2 and gesa2_o end `optimal within the relative gap target` at 1.8e-5 to 9.9e-5 with a verified incumbent a few 1e-5 from the published value (the tighter root meets the 1e-4 target before the published optimum is found - a gap-target stop the 1e-6 match rule scores as a miss, not a wrong answer); bell5, 10teams, air05, qiu and danoint are trees the extra rows do not finish in 60 s (bell5 proves at 11k nodes in 1.5 s with 23 rows and runs 343k nodes with 57). The gains are the lot-sizing and fixed-charge models: in (b) fixnet6, pp08a, vpm2 and modglob prove on all three seeds (60 s unproved to 0.8-25 s), set1ch and pp08aCUTS on the seeds the defaults miss.

**(c), the recipe gated on variable upper bounds** (`root_cut_loop_when=switched-flows`, #1158): `main` a806109f's defaults against the gate branch 26204e00, run 41776346452: 121/112/189 against 127/119/187 matched/proved/verified (proved 38/37/37 -> 41/38/40 per seed). Lost: bell5 seeds 1 and 2, qiu seed 0, air04 seed 1 and dsbmip seed 2 (a `numerical_error` the status guard caught: the point violated primal feasibility by 2e-4) as proofs, arki001 seed 0 and dsbmip seed 2 as points; four gap-target stops (gen, gesa2, gesa2_o, qnet1) beside them. The gate held 10teams, air05 and misc06 out but not bell5, a fixed-charge network that has the variable bounds it tests for; what separated the (b) losers was that no flow cover was found on them, so the next gate to try is round 1 finding a cover, not the rows existing. Not a default.

**(d), the recipe gated on a flow cover being taken** (`root_cut_loop_when=flow-covers`): `main` 4596fd66's defaults against 391ff686, run 41966026845: 112/102/188 against 124/115/188 matched/proved/verified, proved 34/34/34 -> 37/38/40 per seed, +13 proved, +12 matched, 1.19x faster on the shifted mean. The gate holds every tree the loop lost in (b) and (c) (bell5, 10teams, air05, qiu, gen, misc06, danoint, arki001 all stay at one round and lose nothing). What is left: dsbmip seed 2's `numerical_error` (#1187, which the loop does not cause - it never ran on that file - so the pool or the flow cover round does), and two gap-target stops (gesa2_o seed 0 at 1.8e-5, qnet1 seed 1 at 8.1e-5) the runner's proved column counts as lost. By the rule (proved >= on every seed and no lost proof or point) it fails on dsbmip alone, so the recipe stays off and stays available as an option for fixed-charge and lot-sizing models (`--option root_cut_loop=true --option mip_cut_pooling=true --option enable_flow_cover_cuts=true --option root_cut_loop_when=flow-covers`); the default question reopens with #1187 fixed and this A/B rerun.

Files: `bench/results/option-ab-miplib-rootloop-tailing-e01ae9c.csv` (tier 1), `option-ab-miplib3-rootloop-d178213.csv`, `option-ab-miplib3-recipe-99dfa36.csv`, `option-ab-miplib3-gated-recipe-26204e0.csv` and `option-ab-miplib3-covers-gated-recipe-391ff68.csv` (MIPLIB 3; the `arm` column), and the root-only legs `bench/results/rootgap-<leg>-24e289d9.csv` and `rootgap3-<cfg>-b69a3dc7.csv`.
What would earn a default: a loop whose extra rows do not ride along in the tree. For example, keep only the rows still tight at the last root LP, or move the rest to the cut pool, then repeat this A/B.

### Presolve coefficient tightening as a default (`presolve_coefficient_tightening`, #511, #1158)

Savelsbergh's (1994) coefficient reduction on integer rows, tried as the fix for p2756's root (506 coefficients tightened there). Off: `presolve_coefficient_tightening = false`.

All 65 MIPLIB 3 models x 3 seeds, 60 s, `mip_threads=1`, both arms the `d4bb701` binary in one job (bench.yml run 42953273975):

| arm | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---:|
| off | 41/37/63 | 40/39/63 | 43/41/62 | 124/117/188 | 17.34 |
| on | 39/38/63 | 40/38/62 | 38/35/61 | 117/111/186 | 18.50 |

- **Fails the rule:** fewer proofs on seeds 1 and 2, and two verified points lost.
- **Lost:** modglob on every seed (157-896 nodes become 3,028-8,552, and the run no longer matches the optimum), egout on two seeds, rentacar on two (a proof in 11-12 s becomes no feasible point in 60 s), 10teams seed 2, gesa2_o seed 1, set1ch seed 2.
- **Gained:** gesa2 on two seeds, dcmulti seed 1, set1ch seed 0, qiu seed 0.
- **p2756,** the target, is proved on no seed in either arm.

On the models it hurts, the root closes less gap after the tightening (modglob 0.89-0.95 to 0.79-0.84, egout 0.90-0.96 to 0.45-0.51): the cut rounds take fewer cuts from the tightened rows. The MIPLIB tier-1 gate was not run, since this one already fails.

File: `bench/results/option-ab-miplib3-coef-tightening-d4bb701.csv` (the `arm` column).
What would earn a default: tightening that leaves the rows the flow-cover and MIR separators read as they were, for example applied only to all-binary rows, then this A/B again.

### Reduced-cost fixing as a default (`mip_reduced_cost_fixing`, #418, #1158)

Bound tightening from the root's reduced costs and the incumbent (Nemhauser and Wolsey 1988; Achterberg 2007, ch. 7), expected to help nw04 and the set-partitioning models. Off: `mip_reduced_cost_fixing = false`.

Both arms one binary in one job, 3 seeds, 60 s, `mip_threads=1`.

MIPLIB 3, 65 models, main `3e7436d` (bench.yml run 43062830840):

| arm | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---:|
| off | 44/41/63 | 42/41/63 | 44/42/63 | 130/124/189 | 9.22 |
| on | 45/41/63 | 43/41/63 | 44/42/63 | 132/124/189 | 8.98 |

MIPLIB tier 1, 30 models, main `c73be5e` (bench.yml run 43135684987):

| arm | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---:|
| off | 18/11/30 | 19/11/30 | 17/10/30 | 54/32/90 | 17.06 |
| on | 18/11/30 | 19/10/30 | 17/11/30 | 54/32/90 | 16.72 |

- **Fails the rule on tier 1:** seed 1 proves one fewer. enlight8 seed 1 goes from a proof in 55.1 s (412,295 nodes) to no proof in 60 s (424,537 nodes). The fixing acted on that run (one bound tightened), and the search explored more nodes without closing, so this is the option and not the clock.
- **Gained on tier 1:** enlight8 seed 2, unproved at 60 s (445,306 nodes) to proved in 43.4 s (316,595 nodes). One tightened bound moves that tree either way.
- **MIPLIB 3 passes and proves nothing new:** nw04 seeds 0 and 1 reach the optimum 16862 (from 16970 and 16976) and stay unproved. No proof or verified point changes.
- The models it was expected to prove did not move: nw04 and air05 are proved on no seed more than before, and irp and neos-3083819-nubu are in neither set.

Files: `bench/results/option-ab-miplib3-reduced-cost-fixing-3e7436d.csv`, `option-ab-tier1-reduced-cost-fixing-c73be5e.csv` (the `arm` column).
What would earn a default: fixed columns that leave the node LP, so the fixing buys node speed on nw04 and not only a smaller box (after the fixing, 79,000 of nw04's 87,482 columns are fixed and every pivot still prices them), then this A/B again.

### More cuts a root round (`cut_max_per_round` above 30, #1158)

A root round takes the 30 best candidates, far fewer than the filter accepts on models such as mitre. Tried at 100 and 300, all 65 MIPLIB 3 models x 3 seeds, 60 s, `mip_threads=1`, both arms one binary in one job (m/p/v is matched / proved / verified):

| cap | commit, run | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---|---:|
| 30 | `3e7436d`, 43062815542 | 44/40/63 | 42/41/62 | 43/41/62 | 129/122/187 | 16.63 |
| 100 | the same | 44/40/63 | 43/41/62 | 44/42/62 | 131/123/187 | 16.77 |
| 30 | `66b38df`, 43135732275 | 44/41/63 | 43/42/63 | 44/43/62 | 131/126/188 | 15.70 |
| 300 | the same | 43/40/62 | 44/41/62 | 44/42/62 | 131/123/186 | 16.10 |

- **100 fails the rule:** it loses the proofs of 10teams seed 1 (no feasible point in 60 s) and misc07 seed 0. Its gain was gesa2 on every seed, which `root_cut_loop_when=auto` (#1214) has since delivered in about 3 s against 29-37 s here.
- **300 fails on every seed:** it loses the proofs of misc07 seed 0 and stein45 seeds 1 and 2, and the feasible points of 10teams seed 0 and arki001 seed 1; nothing is newly proved.
- **mitre,** the model it was aimed at: at 100 the root bound moves from 114740.52 to 114740.63 against an optimum of 115155, and no seed finds a feasible point.

Files: `bench/results/option-ab-miplib3-cut-cap-100-3e7436d.csv`, `option-ab-miplib3-cut-cap-300-66b38df.csv`. The MIPLIB 2017 models named for it (neos-3381206-awhea, pg) were not run, the default being ruled out already.
What would earn a default: nothing at a flat cap; a cap that grows only where the extra rows are tight at the next root LP.

### Probing in presolve as a default (`presolve_probing`, #512, #1158)

Binary probing (Savelsbergh 1994): fixings and bounds that hold in both branches of a binary, written into the reduced model. Off: `presolve_probing = false`.

All 65 MIPLIB 3 models x 3 seeds, 60 s, `mip_threads=1`, both arms the `66b38df` binary in one job (run 43149854825):

| arm | seed 0 m/p/v | seed 1 m/p/v | seed 2 m/p/v | all m/p/v | sgm time (s) |
|---|---|---|---|---|---:|
| off | 44/41/63 | 43/42/63 | 44/43/62 | 131/126/188 | 15.70 |
| on | 45/42/63 | 42/40/62 | 40/38/60 | 127/120/185 | 16.71 |

- **Fails the rule:** fewer proofs on seeds 1 and 2, and verified points lost.
- **Lost:** modglob on every seed; rentacar, 10teams and bell5 on seed 2; gesa2_o seed 1; air05 and qiu on seed 2. arki001 finds no feasible point on any seed, and rentacar seed 2 none.
- **Gained:** mitre is proved on seed 0 in 44 s, its root after cuts at 115141.0 against 114740.5 without probing (optimum 115155); misc07 seed 2 and qiu seed 0.

It is not a dead end, which is why this is recorded with its other half:
- **The roots it closes.** On the two drayage models of the MIPLIB 2017 benchmark set (`9d479e5`, run 43170308061, 3 seeds) the root bound goes from 11714.14 to 103333.87, the published optimum of drayage-100-23, and to 99718.79 against 101282.65 on drayage-25-23. Neither is proved: no seed finds the optimal point in 60 s.
- **The loss shares a cause with coefficient tightening's** (the entry above). On modglob both leave the Gomory and MIR candidates and the root LP bound as they were and cut the round-1 flow covers from 58 to 48, on every seed. What the two have in common is the tightened column bounds they write, not the coefficients.

Files: `bench/results/option-ab-miplib3-probing-66b38df.csv`, `option-ab-miplib2017-drayage-probing-9d479e5.csv` (the `arm` column).
What would earn a default: the flow cover separator keeping its covers under the tightened bounds, then this A/B again.

**What did become a default (#1264):** probing's integer bounds written only where probing fixes at least a fifth of the integer columns (`presolve_probing_fixed_share = 0.2`), the clique conflicts left off. On MIPLIB 3 only mitre reaches that share; the same-job A/B (run 43689623563, `option-ab-miplib3-probing-share-9605693.csv`) proved mitre on seed 0 and lost nothing, and tier 1 (run 43694331338, `option-ab-miplib-tier1-probing-share-9605693.csv`) was identical in its counts. Full probing as a default stays where this entry leaves it.

### PDHG feasibility polishing as a default (`pdhg_feasibility_polish`, #483, #1177)

First-order feasibility polishing after Applegate, Hinder, Lu & Lubin (arXiv:2501.07018): PDHG on the zero-objective and the zero-bound problem, so a first-order answer is primal feasible to 1e-8 with its gap measured. Off: `pdhg_feasibility_polish = false`.

| suite (`algorithm=pdhg`, defaults otherwise, `main` at `d4bb701`, both arms in one job) | arm | optimal | verified | passed | wall sum |
|---|---|---:|---:|---:|---:|
| Netlib full, 94, 60 s (run 42954273061, EPYC 9V74) | off | 83 | **88** | 77 | 1,448 s |
| | on | 83 | 87 | 77 | 1,437 s |
| refinery LPs, 18, 120 s (run 42954305874, EPYC 9V45) | off | 18 | 18 | 18 | 30 s |
| | on | 18 | 18 | 18 | 31 s |
| Mittelmann LPfeas-14, PDHG alone at 1e-4, 300 s (run 42814436237, EPYC 7763, `e494f47`) | off | 1 | 6 | - | - |
| | on | 0 | **8** | - | - |

On Netlib the polish loses fffff800 (`feasible` and verified off, `iteration_limit` on) and turns bnl1 and perold from `optimal` to `feasible`; it gains `optimal` on d2q06c and pilot4 and nothing else moves. On the refinery LPs it never changes an answer. Where it pays is the first-order engine alone on Mittelmann's set at a loose request: rmine15 and supportcase10 go from stopped at the absolute primal clause to verified, nothing lost. A default is flipped only when nothing is lost, so it stays off, and the LPfeas runner asks for it explicitly.

Files: `bench/results/option-ab-netlib-feasibility-polish-d4bb701.csv`, `option-ab-refinery-feasibility-polish-d4bb701.csv`, `option-ab-lpfeas14-feasibility-polish-e494f47.csv`.
What would earn the default: fffff800, bnl1 and perold keeping their answers with it on - the polished pair replacing a point only when it is at least as good on the dual side as well, not on primal feasibility alone.

---

### The first-order QP route as the default (`qp_algorithm=condat-vu` with `qp_relative_stop`, #1204)

On the refinery planning QP with convex cost curves, Condat-Vu stopped on the relative KKT request and finished by the interior point is the fastest verified arm at 13k and 110k columns: 4.05 s and 44.43 s against the default interior point's 10.63 s and 104.22 s (`bench/results/head-to-head-refinery-qp-90d952a.csv`, `docs/BENCHMARKS.md` section 4d). The question was whether that route should be the default QP engine.

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, both arms on `main` at `90d952a` in one job, Maros-Meszaros 138 at 60 s (run 42967892076, `bench/results/option-ab-maros-meszaros-qp-first-order-route-90d952a.csv`):

| arm | optimal | matched | verified | optimal rejected by the verifier | wall s |
|---|---:|---:|---:|---:|---:|
| today's default (interior point) | 128 | 127 | 132 | 0 | 474 |
| first-order route | 88 | 86 | 91 | 0 | 2,165 |

Forty-one instances lose their verified point, 21 to the time limit and 20 to the first-order engine's iteration ceiling before the relative request is met (the cont, boyd and LISWET families, most of the Netlib-derived q* models), and none gains one; huestis goes from feasible to optimal and is verified in both arms. The route pays on a large, well-scaled, staircase QP and not in general, so the default stays the interior point and `qp_relative_stop` stays an option, off.

### The optional primal heuristics always on, with deeper and repeated dives (PR #1227's first candidate)

On the 13 MIPLIB models where a committed run ended without a point or away from the optimum (`bench/runners/first_incumbent.json`), `mip_heuristics=true` with dives of 1,000 LP re-solves every 50 nodes and the one-level backtrack finds a point on 6 where the defaults find one on 3; deeper or repeated fractional dives without the other heuristics find nothing new (bench.yml runs 43063155433 and 43081881428). As an always-on default it fails the gate. Same job, `--base-binary`, `main` at `66b38df`, 3 seeds, 60 s, one thread:

| set | proved, best of 3, off / on | matched, off / on | proofs lost | matches gained |
|---|---|---|---|---|
| MIPLIB 3 (`option-ab-miplib3-heuristics-all-66b38df.csv`) | 45 / 44 | 46 / 49 | qiu | mas76, pk1, rout |
| MIPLIB 2017 tier 1 (`option-ab-miplib-tier1-heuristics-all-66b38df.csv`) | 12 / 11 | 20 / 22 | enlight8 | neos-5140963-mincio, pk1 |

The heuristics pay where the search has no point and cost proving time where it has one. What was adopted is the same schedule only while there is no incumbent (`mip_heuristics=auto`, the default since), which loses no proof on either set; `mip_heuristics=true` stays a choice, not the default. The 66b38df and c77bc4f CSVs record the options under the names they had then (`mip_heuristics=true` with the dive options; `mip_heuristics_until_incumbent=true`, since folded into `mip_heuristics=auto`).

### RINS kept running under `mip_heuristics=auto` (`mip_heur_rins=on`, #1158)

Under `auto` every optional heuristic stands down at the first incumbent, however poor: on mod011 the first fractional dive's -50.8M (optimum -54.56M) is still the incumbent after 13,000 nodes, and on p2756 the incumbents are 3587, 11588 and 9122 against 3124. RINS (Danna, Rothberg and Le Pape 2005) kept running while the relative gap is above a threshold was the candidate. The threshold comes from the mechanism, not a fit: a RINS sub-MIP can at best take the incumbent to the node bound, so the one gap below which it has nothing left to do is the gap at which the solve stops anyway, `mip_relative_gap`. That is what the existing switch `mip_heur_rins=on` does under `auto`, so it was measured as it stands. Option A/B on one `main` binary (`f60c6ea`), both arms in one job, 3 seeds, 60 s, one thread:

| set (run) | arm | seed 0 m / p / v | seed 1 | seed 2 | proved, best of 3 |
|---|---|---|---|---|---|
| MIPLIB 3, 65 (43885721444) | defaults | 46 / 44 / 64 | 46 / 45 / 64 | 45 / 45 / 64 | 46 |
| | RINS on | 47 / 44 / 64 | 47 / 45 / 64 | 46 / 44 / 64 | 46 |
| 25 MIPLIB 2017 targets (43885730120) | defaults | 3 / 0 / 18 | 4 / 1 / 18 | 3 / 1 / 18 | 1 |
| | RINS on | 3 / 0 / 18 | 4 / 1 / 18 | 4 / 1 / 18 | 1 |
| MIPLIB 2017 tier 1, 30 (43915160597) | defaults | 18 / 11 / 30 | 19 / 11 / 30 | 17 / 11 / 29 | 12 |
| | RINS on | 18 / 11 / 30 | 20 / 11 / 30 | 18 / 11 / 29 | 12 |

(m / p / v: matched the published optimum / proved it / verified point; the verifier rejected nothing in any arm.)

- **Fails the rule:** gesa2 seed 2 loses its proof and its match. It stops at the 1e-4 gap target in 1.75 s on an incumbent RINS found, 25780672.3, 3.2e-5 above the optimum, where the defaults prove 25779856.4. Matches are lost on irp seed 0 and neos-3083819-nubu seed 1 (targets) and markshare_4_0 seed 1 (tier 1); no verified point is lost.
- **Matches gained:** mas76 on all three seeds and noswot seed 2 (MIPLIB 3); p200x1188c on all three seeds (targets); k16x240b seed 1, neos5 seed 2 and noswot seed 1 (tier 1).
- **Better incumbents, no proof:** mod011 -50.8M to -53.1M (seed 0) and -52.7M to -53.4M (seed 2); p2756 3587 / 11588 / 9122 to 3514 / 3214 / 3487; rout seed 1 1104.73 to 1079.73 (optimum 1077.56); harp2, mas74, danoint and arki001 closer too. Over the runs whose incumbent changed, RINS on ends closer to the published optimum on 29 and further on 18 (MIPLIB 3), 13 and 4 (targets), 21 and 10 (tier 1); markshare1, mkc, swath seed 1 and air05 seed 1 are the clear losses.
- **Why it cannot prove them:** RINS moves the incumbent, not the bound. What stops mod011, p2756 and rout at 60 s is the dual bound (-56.0M to -56.5M against -54.56M; 2728 to 2782 against 3124; 1049.0 to 1052.9 against 1077.56, both arms, every seed), and none of them is proved with the better point.

Files: `bench/results/option-ab-miplib3-rins-on-f60c6ea.csv`, `option-ab-miplib2017-targets-rins-on-f60c6ea.csv`, `option-ab-miplib-tier1-rins-on-f60c6ea.csv` (the `arm` column; both arms are the `f60c6ea` binary on `main`). What would earn a default: a stronger root bound on those models, so a better incumbent closes a tree, then this A/B again.

### Four LP-engine settings for the root LPs of ex9, ex10 and ns1116954 (MIPLIB 2017 benchmark set)

HiGHS proves ex9, ex10 and ns1116954 in 3.4, 9.9 and 3.9 s in the same 60 s jobs (`highs-miplib2017-bench-60s-c8544ed.csv`); our search never leaves the root. Diagnosis run 44010017090 (profile=detailed with nested profiles, verbose logs, 3 seeds, the branch of #1316):

- **Presolve is not the gap on ex9 and ex10.** Ours takes ex9 from 40962 rows, 10404 columns to 33846 and 8560, and ex10 from 69608 and 17680 to 63016 and 15936: exactly the sizes the MIPLIB 2017 instance pages publish after trivial presolving. On ns1116954, a zero-objective (feasibility) model, ours removes nothing and trivial presolving removes 120 of 131991 rows and 720 of 12648 columns, which cannot matter at that size.
- **The root dual simplex does not finish in 57 s.** ex9: 24995 iterations, 446684 bound flips, 888 refactorizations (one every 28 iterations), and 26.7 s of 57 in the basic-values update (the FTRAN carrying each iteration's bound flips into x_B, and the recomputation after each refactorization), 10.2 s in FTRAN, 6.9 s refactorizing; 58711 of the FTRAN results were 30% dense or more. ex10: 19953 iterations, 27.6 s in basic values. ns1116954: 27775 iterations, 30.7 s in basic values, 10.7 s refactorizing. Symmetry detection spends 1.8-2.6 s on each before stopping at its budget.

Four settings were tried on one `main` binary (`c1e5290`), all arms in one job, seed 0, 60 s, one thread (run 44081344529): `dual_perturb_costs_at_start=true`, `dual_ratio_test=harris`, `basis_update=forrest-tomlin`, and `mip_node_engine=primal`. None finishes the root LP of ex9 or ex10 (57.3-57.6 s of node LP in every arm). On ns1116954 Harris finishes the root in 15.4 s and the primal simplex in 38.0 s, against 58.0 s unfinished for the default, and bound 0 is then proved; but no feasible point follows (24.0 s of root cuts and 18.6 s of heuristics under Harris), so the status is the same. On neos-960392 the arms change the node count and the incumbent (-23 default, -55 Harris, -118 perturbed), never the proof.

What would move these models: a root dual simplex whose flip update and refactorization rate do not grow with the density of B^-1 on degenerate 0/1 LPs; for ns1116954, a feasible point. Files: `bench/results/option-ab-miplib2017-root-lp-arms-c1e5290.csv` (the `arm` column).

### neos859080 is proved on some seeds and not others: conflict analysis is off (`conflict_analysis`)

neos859080 (164 rows, 80 binary and 80 general integer columns, infeasible) is proved infeasible in 3.2 s on seeds 1 and 2 (95 and 117 nodes) and runs out of time on seed 0 (141,000+ nodes, the open list growing by about one node per node from node 60). The row and column permutation decides the branching order, and on seed 0 the order leaves open nodes that fail for the same reason. With `conflict_analysis=true` every seed is proved, seed 0 in 2.7 s and 93 nodes; `mip_heuristics=true` changes nothing (run 44081402766, `main` `c1e5290`, 3 seeds, 60 s, one thread). Of each 3.2 s proof, about 3.0 s is the propagation dive at the root.

The default stays off, now on a `main` measurement (the same `c1e5290` binary, both arms in one job, 3 seeds, 60 s, one thread; this replaces the branch measurement in section 2):

| set (run) | arm | seed 0 m / p / v | seed 1 | seed 2 | proved, best of 3 |
|---|---|---|---|---|---|
| MIPLIB 3, 65 (44088198542) | off | 47 / 45 / 64 | 46 / 45 / 64 | 45 / 45 / 64 | 46 |
| | `conflict_analysis=true` | 47 / 44 / 64 | 46 / 45 / 64 | 45 / 45 / 64 | 46 |
| MIPLIB 2017 tier 1, 30 (44088204926) | off | 18 / 11 / 30 | 19 / 11 / 30 | 17 / 11 / 29 | 12 |
| | `conflict_analysis=true` | 18 / 11 / 30 | 20 / 10 / 30 | 17 / 11 / 29 | 12 |

- **Fails the rule:** enlight8 seed 1 loses its proof (optimal in 40.1 s and 412,295 nodes off; 37,936 nodes in 60 s on, the analysis taking most of each node), and air05 seed 0 loses a proof made at 59.9 s (523 against 532 nodes).
- **Gained:** neos859080 on every seed (above), and the noswot seed 1 match (-41).

Files: `bench/results/option-ab-miplib2017-heur-conflict-arms-c1e5290.csv`, `option-ab-miplib3-conflict-on-c1e5290.csv`, `option-ab-miplib-tier1-conflict-on-c1e5290.csv` (the `arm` column). What would earn a default: an analysis whose cost per node is bounded (enlight8), then this A/B again.

### The LU rebuilt every 128 updates again inside branch and bound (#1275 measured on MIP, default not changed)

#1275 raised `lu_update_limit` from 128 to 512 on an LP A/B (Netlib, Kennington). On the MIPLIB 2017 benchmark set, 9 proved at `c8544ed` became 7 at `3dba456`. A same-job bisect (both arms on one runner, 3 seeds, 60 s, one thread) finds #1275 to be the only commit in that range that changes the search path of nu25-pr12 and neos-957323. nu25-pr12's root bound after cuts moves from 53259.33 to 53269.67, and its seed 0 goes from proved in 46.3 s to not proved in 60 s. neos-957323's seed 0 goes from 32.3 s and 30 nodes to 60.5 s and 166 (run 44469021991). Every other step leaves the paths unchanged (runs 44463479230, 44463490379, 44469031925, 44469044197, 44469054720, 44469069852).

The default stays 512, on a `main` measurement at `854a328`: one binary, `lu_update_limit=128` against the default in the same job, 60 s, one thread.

| set (runs) | arm | proved | matched | feasible and verified |
|---|---|---:|---:|---:|
| benchmark set, 240, seed 0 (44473575740, 44473582769, 44473590520, 44473597442, 44473604002, 44473610259) | 512 | 7 | 15 | 132 |
| | 128 | 7 | 14 | 130 |
| SAMADHAN's 50, 3 seeds, 150 solves (44473616807, 44473623921) | 512 | 11 | 21 | 132 |
| | 128 | 13 | 22 | 132 |

- **Flips both ways:** on the 240, 128 gains nu25-pr12 (35.0 s) and loses neos-827175 (7.0 s at 512, not proved at 128). On the 50, 128 gains nu25-pr12 seed 0 (44.1 s) and qap10 seed 1 (45.2 s; at 512 that solve stopped at the 60 s limit) and loses none.
- **Time** where both arms prove: geometric mean ratio 128/512 of 0.878 on the 240 (6 models) and 0.945 on the 50 (11 solves).

The search path is sensitive to the factorization interval, and neither interval wins. Files: `bench/results/option-ab-miplib2017-bench-lu-update-limit-128-854a328.csv`, `option-ab-miplib2017-samadhan50-lu-update-limit-128-854a328.csv`, `option-ab-miplib2017-9b0ec5d-vs-ddf9913.csv`, `option-ab-miplib2017-c8544ed-vs-3dba456.csv` (the `arm` column). What would earn a change: a node-LP interval that wins on a whole set, not on a model.

## 2. Demoted by a measurement made on a branch, not re-run on `main`

These decisions were made on numbers that are not in `bench/results/`. They are recorded as such, and each names the re-run that would move it to section 1.

### Parallel tree search (`mip_threads`)

Landed in #403: N workers, each on its own subtree with its own node LPs, sharing the incumbent, the pool and the pseudocosts. The objective and status do not depend on N; the tree explored does. Default 1.

On the branch, 30 MIPLIB instances at 60 s: four threads reach the same 14 and prove the same 9 as one, explore 2× to 4× the nodes, and are slower on the small proofs — geometric mean 0.70× on the nine that finish. More nodes in the same time was not more proofs.

Re-run that would settle it: the A/B on `main` with the runner's `--threads` column.

### Conflict analysis (`mip_conflict_analysis`)

Landed in #381/#292: the search learns from every node it proves infeasible which branching decisions were to blame, and prunes or tightens later nodes that repeat them; a conflict is stored only after re-proof from the global bounds. Off by default.

On the branch at `6c405f8` (#405), 30 MIPLIB instances at 60 s: the same 14 reached and 9 proved either way, no verdict moved. On the nine instances that finish, conflicts take the node count to 0.965× (2,204 against 2,285). On the 21 stopped by the limit the count is 0.938× in the same wall clock — nodes not reached, not search saved — with `enlight8` at 0.373× and 37.4 s of its 60 s budget spent inside the analysis.

Re-run on `main` at `c1e5290`: section 1, "neos859080 is proved on some seeds and not others". It still fails the rule there (enlight8 and air05 each lose a proof), so the default stays off.

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

### Re-perturbing the dual at a larger scale on a stall (#1006, #1165, branch `a527/dual-reperturb`, not merged)

pilot87's path is deterministic, but it flips on any small change to the model: with 3% of its rows substituted away (#1125 at a looser gate) the solve goes from 14 s to 47 s through a numerical error and the unscaled retry. The full verbose log (bench.yml `a527-measure` run 39827844043) names the decision: 1,001 consecutive dual-degenerate iterations at iteration 9,291 that the cost perturbation at 500 did not break, so the basis goes to the primal loop, which starts a phase 1 from a point 3.2e3 outside its bounds (columns were still parked on the 1e6 artificial bounds), meets three singular bases and ends in a numerical error 39,000 pivots later. The change (`dual_stall_reperturb`) refactorized and perturbed the costs again at ten times the scale, up to 1e-3, before handing over (Koberstein 2005, sec. 6.2.3).

On the target it rescues nothing (run 39833979586): the re-perturbation fires at scales 100 and 1,000 and the counter runs out again each time, 12,170 of 17,575 dual iterations degenerate and 7,793 of them Harris cost shifts. The stalling pivots are not ties the perturbation can separate: the ratio test keeps picking a column whose reduced cost sits a hair on the wrong side, the shift sets it to exactly zero, the step is zero, and a column at zero is the first breakpoint of every later row that touches it. Under the textbook ratio test (run 39833987959) the option never engages: that path does not stall, it refactorizes 40,818 times in 151,727 iterations on row/column pivot disagreements (`kPivotAgreement` 1e-8 on a basis whose pivots reach 1e-9) and goes singular at 591,323. The unscaled model solves cleanly in every variant (smallest factorization pivot 1.1e-7 against 3.3e-9 scaled), so both failures are the scaled model's conditioning.

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite, one binary at `d658aaf`, both arms interleaved per instance on one machine: `bench/results/option-ab-netlib-stall-reperturb-d658aaf.csv` (run 39833941084, 94 instances, 2 repeats), `option-ab-kennington-stall-reperturb-d658aaf.csv` (39833948859) and `option-ab-refinery-stall-reperturb-d658aaf.csv` (39833956256, 3 repeats). The option engages on no instance of the three sets: Netlib and the refinery LPs are path-identical in both arms (94/94 and 18/18 verified, the same iteration counts), Kennington's counts differ by a few crossover pivots (16/16 verified), SGM times 0.2168 against 0.2170 s, 9.51 against 9.60 s, 0.2383 against 0.2392 s. Closed unmerged: an option that fires nowhere and does not rescue its one case. What the logs point at instead, open on #1006: the Harris-shift re-entry under the artificial bounds and the hand-over from a point on them, and the pivot-agreement rejections on the scaled pilot family.

---

### A flat lower QP regularization floor, Gondzio correctors, and Krylov refinement on every step (#981, PR #1150, kept off)

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, base `320fa9a1`, Maros-Meszaros 138 at 60 s, beside the adaptive dual floor that was adopted (`option-ab-maros-meszaros-qp-ipm-adaptive-dual-950712b.csv`). Each of these raised the verified count from 124 but lost an instance that was optimal:

- `qp_ipm_regularization=1e-12` (run 39736572856): 130 verified. It lost qforplan (feasible to the iteration ceiling) and cont-201 (to the time limit).
- That floor with `qp_ipm_krylov_refinement` and three `qp_ipm_centrality_correctors` (run 39736560914): 130 verified. The correctors end qisrael, qpcboei2 and qscagr25 with one product the KKT gate refuses. boyd2 and cont-300 ran out of time, because they paid for GMRES on every step.
- The adaptive floor with Krylov on every step (run 39752483470): 131 verified, boyd2 lost to time.
- The adaptive floor without Krylov (run 39752496156): 126 verified, cont-300 lost to time.

The options remain, off by default. The CSVs were not committed; the branch stamps are not on `main`.

### Artificial bounds sized from the model and released early, with Harris shift removal (#1006, #1169, branch `a527/dual-boxing`, not merged)

The re-perturbation entry above left the pilot87 stall at the Harris-shift re-entry under the artificial bounds and the hand-over from a point parked on them. The change (`dual_adaptive_boxing`) sized the box a dual infeasible column is parked on from the model (1e3 times the largest finite bound, capped at the flat 1e6) and gave the column its real bounds back as soon as it left the box (a flip to its other side, or entering the basis), and once 1,000 Harris cost shifts had accumulated removed them, refactorized and resumed (Koberstein 2005, sec. 4.5 and 6.2.2; Maros ch. 9).

On the target it is worse on both models (`a527-measure` runs 39841147240 and 39841154577, counters from the instrumented build). The model as presolve leaves it, optimal in 18,887 iterations and 14 s with the option off, goes through a scaled attempt that ends in a numerical error at 21.7 s and an unscaled retry that hits the 600 s limit at 331,920 iterations with 17,741 accuracy-forced refactorizations; the stall-case model (3% of rows substituted) goes from 12,314 iterations and 57 s on the unscaled retry to 21,406 and 164 s. The shift removal fired once per attempt and the shifts came straight back (3,130 in the first 10,656 iterations against 1,373 in 18,754 with the option off); the box release did not shorten the time on artificial bounds (2,542 and 3,633 iterations against 4,203). And the scaled attempt still hands over at a primal infeasibility of 3.200e+03 with the box at 1e3 times the largest bound, the same figure as with the flat 1e6: that number is not the box width, it is the scaled model's own conditioning (smallest factorization pivots 1.6e-9 scaled against 2.2e-7 unscaled, the unscaled retry solving every variant).

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite, one binary at `6b31e51`, both arms interleaved per instance on one machine: `bench/results/option-ab-netlib-adaptive-boxing-6b31e51.csv` (run 39854167447, 94 instances, 2 repeats; the first dispatch, 39841027101, failed after 54 minutes with its log purged and is not evidence either way), `option-ab-kennington-adaptive-boxing-6b31e51.csv` (41310882885; the first dispatch, 39841033606, was cancelled) and `option-ab-refinery-adaptive-boxing-6b31e51.csv` (39841040064, 3 repeats). Netlib loses pilot87 (optimal in 18,887 iterations and 14.3 s off, time_limit at 268,064 on), the other 93 verified in both arms with the iteration SGM flat, 410.9 against 411.4, and the time SGM 0.284 s against 0.646 s with the lost instance charged the limit; Kennington 16/16 both arms, iterations 2,039.1 against 2,039.4, 10.29 s against 10.36 s; the refinery LPs 18/18, iterations 257.5 against 256.8, 0.192 s against 0.210 s. Closed unmerged. What the two experiments leave as the item: the equilibration on the pilot family, not the dual loop.

### Dropping the QP dual regularization to its cut floor at once (#1178, PR #1180, not merged)

With `qp_ipm_adaptive_dual_regularization` (on since #1156) a step that misses a tenth of the primal residual cuts the floor of delta, but delta itself falls only through `min(delta, 0.1 mu)`, and on liswet8, liswet9 and liswet12 mu rises after the cut, so the cut floor is never reached and the step keeps its 1e-8 proximal term. The change dropped delta to the new floor for the next factorization (option `qp_ipm_adaptive_dual_drop`), in two forms: unconditional, and keyed on `delta |dy|` outrunning the residual and judged by the first step factored with it (undone, with no further drop in the run, when that step missed no less of the residual, when its length collapsed, or when the trial iterate was not finite).

Measured on GitHub Actions with `bench.yml`'s `ab-option` suite and `--base-binary`, both arms in one job, Maros-Meszaros 138 at 60 s:

| version | base | new | off | on | gained | lost |
|---|---|---|---|---|---|---|
| unconditional (run 41425495840) | `d178213a` | `87fa381e` | 128 optimal / 127 matched / 132 verified | 129 / 127 / 133 | liswet8, liswet9, liswet12 | qgfrdxpn (feasible to Condat-Vu's ceiling), qshare1b (optimal to the same), qsierra (optimal to feasible, one product at 3e-4, 63 s) |
| keyed and judged (run 41702161915) | `94cdcc02` | `58762237` | 128 / 127 / 132 | 129 / 127 / 133 | the same three | the same three |

Netlib was identical per instance in both (runs 41425504073 and 41702169175). On the laptop build the keyed version held all three (qgfrdxpn feasible in 69 iterations, qshare1b optimal in 80, qsierra optimal in 27) and still gained the LISWET three, so the judgement passes the drop on one machine's rounding and not another's: on a rank-deficient row block `|dy|` grows as 1/delta after a drop (qfffff80: 5.4e7 to 5.4e9 to 2.4e11 over 1e-8, 1e-10, 1e-12) and whether the run recovers is a knife edge. Not merged, in either form. What did ship from the same PR is the step rejection it was built on - a non-finite trial iterate restores the last finite one and raises the regularization, so no engine is ever handed a non-finite point - which changed no row on the 138 or on Netlib (run 41702177032). The three LISWET instances converge with a flat `qp_ipm_regularization=1e-12`, which loses qforplan and cont-201 (#1150); they stay open on #1178. The A/B CSVs are not committed (branch stamps).

What shipped later (#1178): the same drop restricted to a standard form in which every row carries a slack column. There M = [A, -I] has full row rank, the Schur complement on y stays positive definite as delta falls, and the 1/delta growth of dy above cannot occur; all three losses have equality rows, and in run 41425495840 every all-inequality model the drop touched (the LISWET family and yao) stayed optimal. Measured with `--base-binary`, base `ddf9913e`, new `53c580d8`, run 43788506503 (`bench/results/option-ab-maros-meszaros-qp-ipm-full-rank-drop-53c580d.csv`): 128 optimal / 127 matched / 132 verified to 131 / 129 / 135, liswet8, liswet9 and liswet12 gained (liswet8 at 714.47006, the erratum value), nothing lost, only LISWET and yao paths changed.

- **Complemented knapsack covers do not move the MIPLIB 3 root bounds they were built for (#1158).** The lifted cover separator reads only `<=` rows with positive coefficients; on p2756 its 357 such rows give no violated cover at the root, and the knapsacks are the 375 mixed-sign rows. A `knapsack_cover_complement` option, which negated `>=` rows and complemented negative-coefficient binaries (Crowder, Johnson & Padberg 1983; it passed an exhaustive 0/1 validity check over 400 random rows), separated 65 violated covers on p2756 and 363 on mitre where there were none. The bound did not follow.
  - p2756, with the loop on and every candidate taken each round: 2702.75 with the covers and 2702.75 without; published optimum 3124.
  - mitre, the same setting: 114929 with them and 114984 without; published 115155. Neither arm found a feasible point in 60 s.

  These are WSL seed-0 logs at 60 s, one thread, not a CSV, and the change was not merged; the branch was never pushed. What does move p2756's root is #511's coefficient tightening (506 coefficients, 2698.95 to 2730.14 after cuts in the same setting), whose two-gate A/B is the re-run that decides it.

### The 1e-4 PDHG hand-off as the default, with the polish budget scaled (#1166, PR #1280, default not flipped)

#1163 left the default hand-off at `pdhg_tolerance` 1e-8. At 1e-4 the 50-iteration interior-point polish fell short on agg and dfl001. PR #1280 scales the polish budget with the hand-off: `polish_iteration_limit` from 1e-8 or tighter, plus `tol::kPolishIterationsPerHandoffDecade` (25) for each decade looser, so 150 at 1e-4. At the default 1e-8 the budget is unchanged. The A/B then put the 1e-4 hand-off against the current default.

Setup: `bench.yml` `ab-option`, both arms on the `93bf260` binary in one job, `algorithm=pdhg`. The `off` arm is the default 1e-8 hand-off and the `on` arm is `pdhg_tolerance=1e-4`. Machine: 4 vCPU Xeon 6975P-C. CSV `option-ab-93bf260.csv` in each run's artifact.

| set | run | arm | verified | optimal | passed | SGM iterations (shift 10) | solver s |
|---|---|---|---:|---:|---:|---:|---:|
| Netlib full, 94, 60 s, 100,000 PDHG iterations | 43788380408 | 1e-8 | 86 | 85 | 79 | 62,251 | 301.9 |
| | | 1e-4 | **90** | 88 | 80 | 12,711 | 190.5 |
| refinery LPs, 18, 600 s, same | 43788380408 | 1e-8 | 18 | 18 | 18 | 6,486 | 15.9 |
| | | 1e-4 | 18 | 18 | 18 | 840 | 2.0 |
| Mittelmann small, 8, 300 s, no HiGHS cross-check | 43788401903 | 1e-8 | **3** | 2 | 2 | 56,945 | 1,337.1 |
| | | 1e-4 | 2 | 2 | 2 | 13,014 | 657.2 |

The `off` rows reproduce #1163's 1e-8 arm exactly (86 / 85, SGM 62,251; 18 / 18, SGM 6,486), so the default path did not move.

The larger budget is what finished four Netlib instances at 1e-4. Each was polished in more than the old 50 iterations:

- agg: 60 iterations.
- fffff800: 53 iterations.
- forplan: 74 iterations.
- pilot87: 57 iterations.

The issue's acceptance holds on its two sets: verified counts are at least the 1e-8 hand-off's and total time is lower. The default still stays at 1e-8, for three reasons:

- **dfl001 is lost.** At 1e-8 it is optimal and verified. At 1e-4 the polish reaches 68 iterations and then hits its own 30 s clock (`polish_max_seconds`), a limit the iteration budget does not touch.
- **finnis drops from optimal to feasible.** It is still verified, but the reduced costs miss dual feasibility after the polish.
- **The Mittelmann small set loses qap15.** At 1e-8 PDHG runs 242,120 iterations and qap15 ends feasible and verified at 178 s. At 1e-4 PDHG stops after 1,840 iterations at 31 s, the point is not primal feasible, and the polish does not finish it. irish-electricity, rmine15 and supportcase10 also stop at 31 to 79 s with most of their 300 s unused. Their 1e-8 arm runs to the time limit, so none of the three is verified in either arm.

A default flips only when nothing verified is lost, so the default hand-off stays at 1e-8, and 1e-4 remains an explicit request that now finishes more instances.

What would earn the flip: when the polish declines or cannot finish a 1e-4 point, PDHG resumes toward 1e-8 on the time that is left, and the polish's clock scales with the hand-off the same way its iteration budget does.

The CSVs are not committed because they carry branch stamps. Each is in its run's artifact.

### A global strong-branching iteration budget (PR #1289, closed)

On seven MIPLIB 2017 models HiGHS proves in 1 to 26 s, the profiled runs at `1b8af4e` put 33 to 53 s of 60 into `branching`: decomp2, drayage-100-23, drayage-25-23, irp, neos-3083819-nubu, neos-860300 and nw04. PR #1289 capped strong branching over the whole search: once the probes had spent more than 100,000 + 0.5 x the node LPs' own iterations, a node branched on its pseudocosts and probed nothing (Achterberg 2007, ch. 5).

Setup: `bench.yml` `ab-option`, run 43847282121. The `off` arm is `1b8af4e` and the `on` arm `968b50e`, both in one job. 25 models x 3 seeds, 60 s, one thread.

It changed no count: 2 proved, 10 matched and 54 verified in both arms, nothing gained or lost. On the drayage pair the budget bit: nodes went 1171 -> 3447 and 1238 -> 3978, and branching fell from 51.5 s to about 40 s, but the bound barely moved. On decomp2 and nw04 nothing changed: branching took 51.9 s and 36.6 s in both arms, at 328 and 184 nodes. decomp2's node LPs take 2.75 s in all, so the time inside `branching` is not the probes' simplex iterations, which is the only thing this budget limits. Finding what it is needs the nested profile per solve (`miplib.py --profile-dir`).

The CSV is not committed because it carries branch stamps. It is in the run's artifact.

### irish-electricity: no strict interior, and the two presolve reductions aimed at it are not enough (PR #1302, diagnosis only)

Every one of our runs on Mittelmann's irish-electricity settles at objective about 2,454,39x with a model-space primal violation near 1e-4. That covers the interior point with presolve on and off, scaling on and off, and both the proximal and normal-equations paths, plus PDHG. HiGHS reports 2,546,254.56 optimal. `bench/runners/interior_check.py` (PR #1302) answers two questions without our solver:

- **HiGHS's point is primal feasible by our standard** (run 43901416734). `tools/verify_solution.py` passes its bounds (7.9e-08), rows (8.1e-08 relative), objective and strong duality. Only the reduced-cost identity misses, at 2.9e-06 relative. Our point is therefore infeasible, not a better optimum: a 1e-4 relaxation of a few rows is worth 3.6% of the objective here.
- **The feasible set has no strict interior** (run 43919907943). The max-min-slack LP (Boyd & Vandenberghe 2004, 11.4.1), solved by HiGHS's interior point, gives t* = 0. Its 42 inequalities with a nonzero dual are pairs `C_initstart_k` / `C_aStart_k_1`. For k = 47 they read `cstart_47_1 - v_47_1 >= 0` and `cstart_47_1 - v_47_1 + hstart_47_1 <= 0`, with `hstart_47_1` in [0, 1] and in no other row. Together they force `hstart = 0` and `cstart = v` at every feasible point: implicit equalities, so there is no interior for the interior point to follow.

**Tried:** `presolve_dual_fixing=true presolve_parallel_rows=true`, the two reductions that remove exactly that pattern (fix `hstart` at 0, then merge the two identical rows into an equality). Both fired: 88 dual-fixed columns and 1,859 parallel rows merged (run 43995441320, verbose log in its artifact). Nothing changed:

| engine | run | off | on |
|---|---|---|---|
| `algorithm=ipm` | 43921068158 | iteration_limit, 2,454,396.5 | iteration_limit, 2,454,394.0 |
| `auto` (PDHG) | 43921075853 | time_limit, 2,454,384.17 | time_limit, 2,454,384.17 |

Both A/Bs were on main at `c288ba1`. Other implicit equalities remain after them.

**Next step, not taken:** the same slack LP on the presolved model, which needs the presolved model written out, to name what remains. The CSVs are not committed; they are in the runs' artifacts.

### c-MIR aggregation by variable-bound distance (PR #1296, draft)

PR #1296 changed how the MIR row aggregation picks the column to eliminate. Before, it measured a continuous column's distance to its simple bounds only; with `mir_variable_bound_aggregation` it measures the distance to the closest simple or variable bound at the LP point (Marchand and Wolsey 2001, sec. 2). A flow sitting on its x <= u y is then left for the c-MIR bound substitution. The option also lets a network's flow balance rows start an aggregation.

The target was the fixed-charge networks among the weak-root MIPLIB 2017 models. In run 43876593093 (#1297), p200x1188c's root closed 0.238 of the gap and beasleyC3's 0.637, even with `root_cut_loop_when=always` and 200 cuts a round; HiGHS's root closes 1.000 and 0.968.

Setup: `bench.yml` `ab-option` at `d80f5cb`. Both arms on one runner, 60 s, one thread, 3 seeds.

| gate | run | off: proved / matched / verified | on: proved / matched / verified |
|---|---|---|---|
| 25 MIPLIB 2017 targets | 43879990262 | 3 / 13 / 54 | 3 / 13 / 54 |
| MIPLIB 3, 65 models | 43879997189 | 137 / - / 192 | 134 / - / 191 |

- **Targets:** nothing gained. Only p200x1188c's root moved, from 0.224 to 0.318 of the gap closed.
- **MIPLIB 3:** bell5 lost its proof on all three seeds. The cuts closed more of its root gap (0.144 to 0.867), but the root loop then kept 6 rounds and 103 cut rows instead of 1 round and 23. The tree went from about 19,000 nodes and a proof in 1.6-3.5 s to about 500,000 nodes at the 60 s limit.
- **Tier-1:** not run.

The CSVs are not committed because they carry branch stamps (`d80f5cb`). They are in the runs' artifacts (`option-ab-d80f5cb.csv` in each) and on PR #1296's branch.

What would earn it a place: a separator that closes p200x1188c's root, and a loop gate that keeps bell5's rounds from costing its tree. The second is the root-loop gate's call (#1278), not this option's.

### Flow path cuts by a walk along switched outflows (PR #1303, closed)

`flow_cover_path_walk` extends the flow cover aggregation, which eliminates only unswitched columns. When none is left, the walk continues along the switched outflow carrying the most LP flow into the next node's balance row, and separates the flow cover of the summed path balance. These are the path inequalities of Van Roy and Wolsey (1985). The target was p200x1188c: HiGHS's root closes it, ours closes about 0.22 of the gap.

Setup: `bench.yml` `ab-option` at `31860ca`. Both arms on one runner, 60 s, one thread, 3 seeds.

| gate | run | off: proved / matched / verified | on: proved / matched / verified |
|---|---|---|---|
| 25 MIPLIB 2017 targets | 43912240097 | 0 / 10 / 54 | 0 / 10 / 54 |
| MIPLIB 3, 65 models | 43912248011 | 139 / 140 / 192 | 139 / 140 / 192 |

- **Counts:** nothing gained and nothing lost in either gate. This targets runner (an EPYC 7763) proved none of the 25 in either arm.
- **Shifted geometric mean** where both arms prove MIPLIB 3 (139 runs): 2.28 s off, 2.25 s on.
- **p200x1188c's root did not move:** 0.217-0.224 of the gap closed off, 0.215-0.218 on.
- **beasleyC3** gained 0.005-0.007 on every seed.
- **Root gap closed moved by up to 0.08 either way on** h80x6320d, vpm2, mod011, fixnet6, egout and modglob. Their round counts changed too.

The CSVs are not committed because they carry branch stamps (`31860ca`). They are in the runs' artifacts (`option-ab-31860ca.csv` in each) and on PR #1303's branch.

What would earn it a place: a gain on a fixed-charge network's root. p200x1188c's weakness is not missing path aggregates. It is not big-M arc capacities either: `presolve_coefficient_tightening` leaves its root LP and root bound unchanged (run 43995480555). That option also gives wrong `optimal` answers on egout and modglob (#1312).

### No incumbent on five MIPLIB 2017 models: where the 60 s go, and fix-and-propagate before the root LP (branch `exp/prelp-fixprop`, not merged)

HiGHS proves neos-950242, neos-3381206-awhea, hypothyroid-k1, neos-960392 and ns1116954 in 10.0, 1.6, 9.1, 7.2 and 3.9 s in the same 60 s jobs (`highs-miplib2017-bench-60s-c8544ed.csv`). We find no point on four of them and only -23 against -238 on neos-960392. Diagnosis run 44348500088 (`ab-option` on `main` at `4fc850f`, 60 s, one thread, 3 seeds, `profile=detailed` with nested profiles, verbose logs; a second arm with `mip_heuristics=true`) and run 44348916362 (the same with `gpu_fix_and_prop` or `gpu_pump` on the CPU):

- **neos-950242** (5520 binaries, 240 integers, 34224 rows): the root LP takes 0.9 s. One root dive then takes the rest: the coefficient dive spends 51.4, 34.3 and 38.7 s on 15, 50 and 5 LP re-solves (seeds 0-2), mostly in the dual simplex's basic-values update. Feasibility Jump stalls with no point after about 9200 moves. RENS and the pump do nothing. The root bound is 1 and the optimum is 4, so a point alone would not prove it.
- **hypothyroid-k1** (2602 binaries, 433884 nonzeros): the parity heuristic spends 11.3 s on two sub-MIPs, each fixing 3 of the 2602 binaries (#1323). The fractional and coefficient dives then spend 33 s on 25-31 re-solves (13.4 s refactorizing). With `mip_heuristics=true` the vector-length dive found the optimum, -2851, on seed 1 at 51 s.
- **neos-3381206-awhea** (475 binaries, 1900 general integers): the tree runs 16,500-23,000 nodes. The dives repeated every 50 nodes take 27 s over 459 calls per rule and find nothing, and the propagation dive takes 3-4 s. Fix-and-propagate spends its 5 back-ups in 0.01 s. The feasibility pump leaves general integers out of its distance (`heuristics.hpp`), and 1900 of the 2375 columns are general integers here.
- **neos-960392** (59376 binaries): the heuristics take 2 s. Feasibility Jump finds -23, -27 or -179 before the root LP. The root bound is already -238, the optimum. The 31 nodes take 39.6 s of node LP (about 8000 dual simplex iterations each) and 14.7 s of branching. The cost is the node LP on a degenerate 0/1 model, which belongs with the root dual simplex work.
- **ns1116954** (feasibility, zero objective): the root LP does not finish (57.9 s, 31.1 s in basic values; see the ex9/ex10/ns1116954 entry above). Feasibility Jump before the LP stalls after 5205 moves and 0.1 s. So only an LP-free heuristic before the root LP can find a point.

**Tried: fix-and-propagate before the root LP** (`mip_fix_and_prop_before_lp`, branch `exp/prelp-fixprop` at `ca303dd`). This is `gpu_fix_and_prop` on the CPU, from the box point closest to zero, once before the root LP, for the ns1116954 case. Run 44453732125, 3 seeds, 60 s, with `gpu_fix_backtrack` at 5 and at 1000. **No point on any of the five models in either arm.** With 5 back-ups it gives up in 0.03-1.0 s ("back-up budget spent; the repair found nothing"). With 1000 back-ups it runs into its time share (4.8-5.9 s) on neos-950242, hypothyroid-k1 and ns1116954, and spends the 1000 back-ups in 0.2-0.3 s on neos-3381206-awhea. Not merged. The CSV is in the run's artifact (`option-ab-ca303dd.csv`, branch stamp).

What these runs led to instead: a time budget per dive (#1324), which gives hypothyroid-k1 a point on every seed, and parity fixings only when they fix half the binaries (#1323). What would move the rest: for ns1116954 and neos-960392, the root and node dual simplex on degenerate 0/1 LPs. For neos-3381206-awhea, a pump distance that covers general integers (Bertacco, Fischetti and Lodi 2007).

**Tried: that pump distance** (`mip_pump_general_integers`, #1329, closed, branch kept). Every general integer gets an auxiliary column and two rows in the projection LP. On the 25 MIPLIB 2017 targets (run 44495592865, binary `e64302e`, 3 seeds, 60 s, one thread) no row changed status or objective in either arm: 5 proved, 16 matched, 57 verified. On neos-3381206-awhea the pump now really projects, taking 2.7-3.0 s for its 20 rounds where it took 0.06 s, and still proposes nothing. Run 44524975889 (binary `283b4cc`) gave it 200 rounds and gave the PDHG pump (#509, which already had this distance) 500. Both stopped on their time share after 55-95 rounds, and neither found a point on any seed. So the missing distance is not what stops us on this model. CSVs are in the runs' artifacts (`option-ab-e64302e.csv`, `option-ab-283b4cc.csv`, branch stamps).

### Probation for every model in the root cut loop (`root_cut_loop_when=probation`, PR #1332, opt-in)

`root_cut_loop_when=auto` lets a model into the root cut loop only when round 1 took a flow cover or the model has a variable upper bound. `probation` lets every model in, on the probation `auto` already applies: the rounds after round 1 stay only if together they gain at least what round 1 gained, and the rows slack at the last root LP go to the cut pool.

The target was SAMADHAN's 50, where we prove 6 on seed 0 against their claim of 8. On the near-misses HiGHS proves in 3 to 21 s (enlight_hard, nu25-pr12, neos17, mik-250-20-75-4, binkar10_1, neos-911970), diagnosis run 45114877618 (`main` at `cb5ef02`, `profile=detailed`, 3 seeds) shows trees of 14k to 330k nodes under a root bound far below HiGHS's: neos17 0.0039 against 0.136, neos-911970 23.26 against 52.2, enlight_hard 16.5 against 23.6. Each takes 30 of up to 377 cut candidates in one round of under 0.07 s, and the loop is not entered.

Setup: `bench.yml` `ab-option` on one binary, `8fd937f`. `off` is the defaults, `on` adds `root_cut_loop_when=probation`. Both arms in one job per gate, 60 s, one thread, 3 seeds. Counts are matched / proved / verified.

| gate | run | off, per seed | on, per seed | off, all | on, all |
|---|---|---|---|---|---|
| SAMADHAN's 50 | 45137785259, 45137799681 | 9/6/45, 7/4/45, 5/4/43 | 9/7/45, 8/5/45, 6/4/43 | 21/14/133 | 23/16/133 |
| 25 MIPLIB 2017 targets | 45137861197 | 5/1/19, 6/2/19, 5/2/19 | 6/2/19, 5/1/19, 7/3/19 | 16/5/57 | 18/6/57 |
| MIPLIB 3, 65 models | 45137875637 | 46/45/64, 47/46/65, 46/45/64 | 46/45/64, 47/46/65, 45/43/64 | 139/136/193 | 138/134/193 |
| tier 1, 30 models | 45137888938 | 18/11/30, 19/11/30, 17/11/29 | 17/12/30, 19/10/30, 17/10/29 | 54/33/89 | 53/32/89 |

It fails the rule: five proofs are lost.

- **neos859080 seed 1** (targets). `off` proves it infeasible in 3.2 s at 95 nodes. With 3 rounds kept (18 rows) the search hits the limit at 168,797 nodes. Seed 0 hits the limit in both arms.
- **misc07 seed 2** (MIPLIB 3). `off` proves it in 22.8 s. Round 1 gained nothing, so any later gain passes probation: 20 rounds and 557 rows move the root from 1415 to 1516 against an optimum of 2810, and the run ends at the limit with bound 2805.
- **misc06 seed 2** (MIPLIB 3). Six rounds, then `optimal within the relative gap target` at 4.0e-5 with 12851.076 against 12850.861: a gap-target stop the match rule scores as a miss, not a wrong answer. The earlier gates recorded in #1278's body lost the same run the same way.
- **enlight8 seeds 1 and 2** (tier 1). `off` proves them in 49.5 and 56.6 s at 412k and 478k nodes. With 20 rounds and about 580 rows the root goes from 1.5 to 10.1 and 13.9 against an optimum of 27, and the tree then runs 52k and 43k nodes in 60 s. The earlier gates in #1278's body lost seed 1.
- **Match lost:** neos5 seed 0 (15 to 15.5), in both runs that hold it.

What it gained:

- **enlight_hard seed 0 proved** in 17.7 to 25.0 s in the three runs that hold it: 20 rounds, root 15.5 to 30.4 against 37, 3,089 nodes against 370k unproved. Seeds 1 and 2 end at gaps of 5.4% and 2.7% against 32% and 24%.
- **swath1 seed 1 proved** in 38.4 s, and **irp seed 2** in 38.1 s.
- **mik-250-20-75-4 matched** on seeds 0 and 2, gap 7.4% to 1.3-1.8%.
- **Gaps:** binkar10_1 3.8-4.6% to 0.3-1.2%, pg 33% to 10%, pg5_34 11-13% to 6-7%.
- **Worse:** neos17's gap goes from 4.5-4.9% to 5.5-8.8%, at 23k-42k nodes against 76k-85k, and swath3's from 3.7% and 4.3% to 4.3% and 12% on seeds 0 and 1.

On SAMADHAN's 50 that is 7 proved on seed 0 against 6, and 7 against 6 best of 3 seeds in this job. It does not reach 8.

No reading of the root separates the wins from the losses. The share of the root gap the rounds close is 0.64-0.70 on enlight_hard and 0.35-0.50 on enlight8, but 0.09 on swath1 seed 1 (proof gained) and 0.07 on misc07 seed 2 (proof lost). #1278 charged each round for the nonzeros it adds and lost gesa2 and ej instead.

The CSVs are not committed because they carry branch stamps. Each is `option-ab-8fd937f.csv` in its run's artifact.

What would earn the default: a tree that does not pay for the rounds' rows. On enlight8 the search runs 38k-52k nodes in 60 s with the rounds kept, against 412k-509k without, and that is the whole loss.

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
