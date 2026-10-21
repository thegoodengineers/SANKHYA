# The finale roadmap, as it stands

The plan from issue #540, restated against `main` at bc5f7053 (21 October 2026). Each
line is an issue with the measurement that motivates it, the paper it comes from and its
acceptance checklist; the estimate is days for one person as planned. A ticked line is
an issue closed with its acceptance met and its evidence on `main` (a merged PR, and a
CSV in `bench/results/` where a number was promised). An unticked line is open, and the
reason is in the table at the end. This file is the record; the issue tracker is the
authority if they differ.

**Where it stands:** of the 77 planned items, 72 are closed and 5 open. Of the 18 items
that were already open when the plan was written, 15 are closed and 3 open. Every P0 item is
closed.

## Order of work

**Before anything else:** #539, #538, #504, #486 - all closed.

**Week 1, cheap and high payoff:** #463, #468, #474, #476, #496, #495, #479, #480, #505,
#529, #530, #491, #532 - all closed.

**Weeks 2 and 3, the big levers:** #467, #470, #490, #478, #481, #498, #506, #519,
#518 - all closed.

**Week 4, the freeze:** #488 closed; `docs/BENCHMARKS.md` is generated from the CSVs; the
A/Bs of the off-by-default features are recorded per issue, not as one pass.

## LP: simplex and linear algebra

- [x] #463 P0, 1d
- [x] #466 P0, 1d
- [x] #465 P1, 3d
- [x] #464 P1, 4d

## LP: interior point

- [x] #468 P0, 1d
- [x] #467 P0, 3d
- [x] #474 P0, 1d
- [x] #470 P0, 7d
- [x] #469 P1, 2d
- [x] #471 P1, 2d (re-scoped: the AMD supervariables landed in #979; METIS dropped)
- [x] #472 P1, 2d
- [ ] #473 P1, 2d
- [x] #475 P2, 4d

## Engine orchestration

- [x] #476 P0, 2d (novel)
- [ ] #477 P2, 3d (novel)

## First-order LP and GPU

- [x] #479 P0, 1d
- [x] #480 P0, 1d
- [x] #478 P0, 7d
- [x] #481 P0, 4d
- [x] #486 P0, 1d
- [x] #488 P0, 2d
- [x] #482 P1, 3d (the GPU leg of its A/B moved to #1110)
- [x] #483 P1, 3d
- [x] #484 P1, 3d
- [x] #485 P1, 4d
- [x] #487 P1, 2d
- [x] #489 P2, 8d

## QP and MIQP

- [x] #490 P0, 7d
- [x] #491 P0, 1d
- [x] #492 P2, 1d
- [x] #493 P2, 3d
- [x] #494 P2, 3d

## MILP: cuts

- [x] #496 P0, 1d
- [x] #495 P0, 2d
- [x] #498 P0, 3d
- [x] #500 P0, 2d
- [x] #497 P1, 2d
- [x] #499 P1, 1d

## MILP: search and measurement

- [x] #504 P0, 2d
- [x] #505 P0, 1d
- [x] #501 P0, 3d
- [x] #502 P1, 2d
- [x] #503 P2, 3d

## MILP: primal heuristics, CPU and GPU

- [x] #506 P0, 3d
- [x] #507 P1, 3d
- [ ] #508 P1, 5d (novel)
- [x] #509 P1, 4d (novel)
- [x] #510 P2, 4d

## MILP: presolve (slices of #412)

- [x] #511 P0, 1d
- [x] #512 P1, 3d
- [x] #513 P2, 1d

## Proofs and exactness

- [x] #519 P0, 2d (novel)
- [x] #518 P0, 5d (novel)
- [x] #521 P1, 5d (novel)
- [x] #520 P1, 6d (novel)

## Refinery: global optimization and the case study

- [ ] #514 P1, 6d (novel)
- [x] #515 P1, 2d
- [x] #516 P1, 1d
- [x] #517 P1, 3d

## Planner-facing analysis and API

- [x] #522 P1, 3d (novel)
- [x] #523 P1, 2d
- [x] #524 P1, 3d (novel)
- [x] #527 P2, 2d
- [x] #526 P1, 2d (novel)
- [x] #536 P1, 3d
- [x] #537 P2, 1d
- [x] #525 P2, 8d (novel) (Dantzig-Wolfe and the early give-up moved to #1048)
- [x] #528 P2, 5d

## Evidence and documentation

- [x] #532 P0, 1d
- [x] #529 P0, 1d
- [x] #530 P0, 1d
- [x] #531 P1, 1d
- [x] #533 P2, 2d
- [x] #538 P0, 1d
- [x] #539 P0, 1d

## Engineering

- [x] #534 P1, 1d
- [ ] #535 P1, 1d

## Already open and part of this milestone

Closed: #411, #412, #413, #415, #417, #419, #221, #358, #446, #447, #448, #456, #460,
#461, #462. Open: #414, #418, #295.

## What is open, and why

| issue | what landed | what keeps it open |
|---|---|---|
| #473 | proximal regularization behind `ipm_proximal_regularization` (#657), its A/B on `main` (#1060, #1067) | pilot.ja is lost on Netlib with the option on; irish-electricity unsolved either way |
| #477 | the learned tree behind `algorithm_selection=learned` (#941), worse than the rules on Netlib | a person's review of the four branches |
| #508 | Feasibility Jump on the device, off by default (#679), measured on a V100 (#734) | the A/B that would earn it a default |
| #514 | spatial branch and bound (#637) and its bound certificate (#978); 32 of 39 pooling runs optimal and verified | 7 runs end `feasible`; no pooling CSV after #978 records a checked certificate |
| #535 | the TSan job covers the race, the KKT check, the SpMV and the decomposition (#659) and the threaded Cholesky (#1035), green on `main` | no required check is enforced |
| #414 | a switch and a counted budget per heuristic (#425); FJ and parity on by default after their A/Bs | the per-heuristic A/B of the #378 four and the dives; 0 of the 5 named instances at the optimum |
| #418 | restarts and objective branching, off, with their A/Bs (#970) | no bound moved on b-ball, opt1217, rlp1 or noswot |
| #295 | multi-GPU PDHG with a fixed-order exchange (#684, #713) | the measurement on more than one card |

## Deliberately not planned, revisited

- Single-precision PDHG stays out of the reported path. #984 added `pdhg_precision=mixed`
  for the GPU engine (single-precision inner products, residuals and refinement in
  double), default `double`; every number in `docs/BENCHMARKS.md` is from the double path
  unless its row says otherwise.
- A GUI: still none; #537 is the notebook over the Python API.
- Symmetry slice 2: #413 is closed.
