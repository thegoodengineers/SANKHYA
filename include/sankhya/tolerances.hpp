// SPDX-License-Identifier: Apache-2.0
// SANKHYA — every numerical tolerance in the solver, in one place.
//
// ENGINEERING_RULES.md rule: no magic numbers in the numerical core. If a comparison against a
// small constant appears anywhere in src/, the constant is declared here with a justification.
// Anything tunable at runtime is ALSO an entry in the option table (src/util/options.cpp)
// whose default is one of these constants; these are the defaults, not the law.
#pragma once

#include "sankhya/types.hpp"

namespace sankhya::tol {

// ---------------------------------------------------------------------------------------
// Feasibility and optimality
// ---------------------------------------------------------------------------------------

/// Max allowed violation of a row or column bound before a point is called infeasible.
inline constexpr double kPrimalFeasibility = 1e-7;

/// Max allowed violation of dual feasibility (sign conditions on reduced costs).
///
/// STATED FOR COSTS OF ORDER ONE (#783). An LP is unchanged by multiplying c by a positive
/// constant; its reduced costs scale with it and an absolute 1e-7 does not. On scaled Netlib
/// sc205 (one cost, -3.05e-05) a wrong-signed reduced cost of -3.9e-08 passed and the answer
/// was objective 0 against -52.2. So the value is not changed; what it is measured against
/// is: the simplex solves with c scaled by a power of two to max|c| in [1, 2)
/// (cost_scale_factor, src/la/scaling.hpp), and the quality measure and the verifier divide
/// each dual violation by max(min(1, max|c|), its own terms) instead of max(1, its terms).
/// For max|c| >= 1 nothing moves; of the Netlib LPs four are below 1 (ganges 0.5, pilot
/// 0.026, pilotnov 0.89, tuff 0.005), and the A/B in the #783 pull request covers them.
inline constexpr double kDualFeasibility = 1e-7;

/// The least feasibility tolerance the simplex is handed when a scaled optimum is re-solved
/// at tighter tolerances because it did not hold in original units (#792). The tolerance is
/// divided by twice the measured amplification; this floor keeps it above the level the
/// basic-system residual reaches after refinement on badly conditioned bases, where a
/// tighter tolerance only chases rounding.
inline constexpr double kTightenedFeasibilityFloor = 1e-10;

/// Max allowed distance from an integer before a value is called fractional.
inline constexpr double kIntegrality = 1e-6;

/// MIP termination: stop when (incumbent - dual bound) / |incumbent| falls below this.
inline constexpr double kMipRelativeGap = 1e-4;

/// MIP termination: stop when (incumbent - dual bound) falls below this in absolute terms.
inline constexpr double kMipAbsoluteGap = 1e-6;

/// Diving heuristics (#25, #414): maximum integer columns fixed in one dive before it gives
/// up. Bounds the heuristic's own cost independently of instance size - Achterberg,
/// "Constraint Integer Programming" (thesis, 2007), ch. 6, notes a dive's payoff is
/// concentrated in its first handful of fixes, so capping it well short of the full integer
/// column count keeps a dive on a large MILP from itself becoming the expensive part of
/// solving the node it runs at.
inline constexpr int kDivingMaxDepth = 50;

/// Diving heuristics: maximum LP re-solves in one dive, the default of mip_dive_lp_resolves.
/// Without a backtrack every fixed column costs exactly one re-solve, so this moves together
/// with kDivingMaxDepth - kept as its own constant because the two bound different things
/// (how much of the box the dive may fix vs. how much simplex work it may spend doing so),
/// and a dive that backtracks (mip_dive_backtrack, #414) re-solves without fixing a new
/// column.
inline constexpr int kDivingMaxLpResolves = 50;

/// Sub-MIP heuristics (RINS #290, RENS #414): the least fraction of the integer columns the
/// sub-model must fix for the neighbourhood to be one. Below it the sub-MIP is nearly the
/// whole model, and the search would be paying for a second search; Danna, Rothberg and
/// Le Pape (2005) and Berthold (2014) both stop at a half.
inline constexpr double kSubMipMinFixedFraction = 0.5;

/// Sub-MIP heuristics: the share of the main search's nodes so far that its sub-MIPs may
/// have spent in total, and, when the solve is not deterministic, the share of the remaining
/// time one sub-MIP may take. A heuristic that out-spends the search it serves is not one.
inline constexpr double kSubMipBudgetShare = 0.1;

/// Feasibility Jump (#506; Luteberget and Sartor, Math. Programming Computation 15, 2023):
/// the candidate jumps sampled per move, the paper's 25; the default work budget of one run
/// in nonzero visits, the default of mip_fj_work; and the least decrease of the weighted
/// violation a jump must bring to be taken, so rounding noise in the scores cannot keep the
/// search cycling between two values of one column. The stall stop (#1030): a run ends
/// after kFeasibilityJumpStallSteps steps (moves or weight updates), or
/// kFeasibilityJumpStallSweeps steps per column when that is more, without the number of
/// violated rows falling below its best or a better point being found; the paper restarts
/// on no progress under a wall-clock cutoff, and this is the step-counted equivalent.
inline constexpr int kFeasibilityJumpSample = 25;
inline constexpr Count kFeasibilityJumpWork = 10'000'000;
inline constexpr double kFeasibilityJumpMinScore = 1e-9;
inline constexpr Count kFeasibilityJumpStallSteps = 1000;
inline constexpr Count kFeasibilityJumpStallSweeps = 2;
/// The completion LPs a run on a mixed model may spend (#1030), in simplex iterations per
/// row of the model (mip_fj_completion_iterations), with a floor for small models. The
/// default is 0, no limit: on GitHub runner 39564590275 every finite budget tried (10, 30
/// and 100 per row) ended the refinery lot_sizing medium and large models at a worse 60 s
/// objective than no limit (88045.4 against 87064.0, and 473108.6 to 489285.2 against
/// 454941.1), bench/results/fj-share-1030-sweep-d1a87e6.csv. The option stays for a caller
/// who prefers a bounded heuristic to its last improving points.
inline constexpr Count kFeasibilityJumpCompletionIterationsPerRow = 0;
inline constexpr Count kFeasibilityJumpCompletionIterationsFloor = 10'000;

/// The parity heuristic (#841, src/mip/parity.hpp): the most binary columns its dense GF(2)
/// elimination takes (rows x 2048^2 / 64 word operations at most, well under a second); the
/// fixings it tries, 2^4, so a null space of dimension four is enumerated whole; and the
/// node limit of the sub-MIP that completes each fixing.
inline constexpr int kParityMaxColumns = 2048;
inline constexpr int kParityCandidates = 16;
inline constexpr Count kParitySubMipNodes = 1000;

/// The propagation dive (#841): its budget in nonzero visits, spent as whole propagation
/// passes over the matrix (a pass reads every nonzero up to three times).
inline constexpr Count kPropagationDiveWork = 200'000'000;

/// Local branching (#841; Fischetti and Lodi 2003): the Hamming radius k on the binary
/// columns (the paper's range is 10 to 20), the node limit of one neighbourhood's sub-MIP,
/// and the recentrings after an improvement.
inline constexpr int kLocalBranchingRadius = 20;
inline constexpr Count kLocalBranchingNodes = 1000;
inline constexpr int kLocalBranchingRounds = 5;

/// Reliability branching (#69; Achterberg, Koch & Martin, "Branching rules revisited",
/// Operations Research Letters 33 (2005), 42-54). A column's pseudocost in a direction is
/// trusted once it has been observed this many times; until then the column is a
/// candidate for strong branching, which measures the two children directly.
/// eta_rel = 8 is the paper's recommendation, and the value SCIP ships.
inline constexpr int kPseudocostReliability = 8;

/// Strong branching evaluates at most this many unreliable candidates per node, the most
/// fractional first. Each costs two warm-started dual simplex solves; the paper's
/// "lookahead" bound of 8 stops after that many candidates fail to improve the best score,
/// which this simpler cap approximates.
inline constexpr int kStrongBranchingCandidates = 10;

/// Iteration cap for one strong-branching child LP. A dual simplex stopped at this cap
/// still reports a valid bound (its objective is dual feasible throughout), so a capped
/// probe measures a lower estimate of the gain rather than nothing.
inline constexpr int kStrongBranchingIterations = 50;

/// Strong-branch fixing (#502, option mip_strong_branch_fix): a probe that proves one side of
/// a column infeasible fixes the column to the other side, the node LP is re-solved, and the
/// branching decision is taken again on the new point. At most this many such rounds per
/// node; each can cost up to 2 * kStrongBranchingCandidates probes, so the cap bounds the
/// node's strong-branching work at a small multiple of what one decision already costs.
inline constexpr int kStrongBranchFixRounds = 3;

/// Node bound propagation: an implied column bound is recorded only when it tightens the
/// current one by more than this. Keeps the propagation loops from chasing changes at the
/// level of rounding error, which on continuous columns would never stop.
inline constexpr double kPropagationMinChange = 1e-9;

/// Incremental propagation to a fixpoint (#502, option mip_incremental_propagation): the
/// worklist may process at most max(kPropagationWorkFloor, kPropagationWorkPerRow * rows)
/// rows per node. A fixpoint on continuous columns can be approached geometrically and never
/// reached; stopping early is always sound, since propagation only tightens implied bounds.
inline constexpr int kPropagationWorkFloor = 1000;
inline constexpr int kPropagationWorkPerRow = 20;

/// LP optimality check used by the independent verifier: primal objective must equal dual
/// objective to this relative accuracy. Tighter than feasibility on purpose - a converged
/// simplex basis should reproduce strong duality far better than it satisfies bounds.
///
/// THIS CONSTANT IS NOT INDEPENDENT OF kDualFeasibility, and the two were originally chosen
/// as though it were. The duality gap at a primal-feasible point is bounded by the
/// complementarity residual, which is bounded in turn by the dual infeasibility times the
/// size of the primal solution:
///
///     relative gap  <=  kDualFeasibility * ||x||_1 / |objective|
///
/// So a fixed 1e-9 is only attainable when ||x||_1 / |objective| is small. Netlib `etamacro`
/// is where that surfaced (issue #52). Measured there:
///
///     dual infeasibility     1.345e-07      (just over kDualFeasibility)
///     sum |x_j|, 688 columns 2721
///     |objective|            755.7
///     => implied bound on the relative gap    4.84e-07
///
/// which is roughly FIVE HUNDRED TIMES looser than the 1e-9 promised here. The observed gap
/// was 3.25e-09 - far better than the bound, but still over this constant, and no amount of
/// tightening the gap test can fix a point whose duals have not converged.
///
/// DECISION (#52): neither number moves.
///
///   * The verifier is NOT loosened. It is the independent check the whole evidence story
///     rests on, and tuning it so we pass inverts its purpose. `ENGINEERING_RULES.md` says so
///     directly.
///   * kDualityGap is NOT relaxed to cover the worst case either. As an expectation for a
///     genuinely converged basis, 1e-9 is right; the models that miss it are models whose
///     duals are not converged, and hiding that behind a looser constant is the same
///     mistake in the other direction.
///
/// The honest outcome for such a point is the status kFeasible - a usable primal point,
/// optimality not proven - which solve() now assigns automatically when the measured dual
/// infeasibility exceeds tolerance (issue #27). `etamacro` takes that path today and the
/// verifier accepts it, skipping the strong-duality test because no optimality is claimed.
///
/// What is actually needed for `etamacro` to reach kOptimal is better dual convergence, not
/// a different threshold: Devex pricing (#66) and the Harris ratio test (#67). This constant
/// should be revisited only if those land and instances still miss it.
inline constexpr double kDualityGap = 1e-9;

/// Complementary slackness: the largest |multiplier| * slack over every row and column that
/// an optimality claim may carry. This is the test tools/verify_solution.py applies (its
/// "complementary slackness" check), mirrored in kkt_check.cpp and the status guard in
/// solve() so neither can let out a claim the verifier will reject. A product passes when it
/// is at most
///
///     max(kComplementarity, kComplementarityRounding * |multiplier| * magnitude)
///
/// where magnitude is the size of the numbers the slack is a difference of: for a row the
/// larger of its largest term |a_ij x_j| and its bound, for a column the larger of |x_j| and
/// its bound. The first term is the absolute test of #209, and it is what decides on every
/// model whose |multiplier| * magnitude is under 1e6: a row priced at 1e3 still may not sit
/// 3e-9 inside a bound of 1, and the interior point's row priced at 3.3e+03 sitting 3e-10
/// inside its bound (a product of 1.1e-06) still fails.
inline constexpr double kComplementarity = 1e-6;

/// The second term (#806): a slack is computed as bound - activity, and rounding leaves an
/// error of order eps * magnitude in it (times the number of terms summed), so a correct
/// multiplier times that error is a product no solver can avoid. 1e-12 is about 4500 eps: room
/// for the summation of a long row, and four orders of magnitude below the primal
/// feasibility tolerance, so a slack that is a real fraction of its magnitude is still read
/// as one. Judged by kComplementarity alone, Klee-Minty n = 20 (right-hand sides up to
/// 5^20 = 9.5e13) carries 1.6e-02, which is 1.6e-16 of its terms, and n = 30 carries 1.3e+05
/// (1.4e-16); both were downgraded from a correct optimum. A genuine violation stays one:
/// the scaled Netlib ship models of the stress set carry products equal to |d_j| * |x_j|, a
/// column interior by its own size. A slack that cannot close keeps the absolute test, since
/// there the condition is "the multiplier is zero" and has no magnitude.
inline constexpr double kComplementarityRounding = 1e-12;

/// The in-process KKT check of the engine race (#476, src/core/kkt_check.cpp) makes the
/// independent verifier's checks, and these are the two of its thresholds that are not
/// already above: the reduced costs must match c - A^T y, and the reported row activities
/// the recomputed ones, to kVerifierConsistency (the reduced costs relative to their terms,
/// the activities absolutely, as tools/verify_solution.py does); the reported objective must
/// match the recomputed one to kVerifierObjective relative. Copied, not tuned: an answer the
/// race accepts has to be one the verifier would accept.
inline constexpr double kVerifierConsistency = 1e-6;
inline constexpr double kVerifierObjective = 1e-9;

// ---------------------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------------------

/// Structural zero threshold. Values below this in magnitude are dropped when a sparse
/// matrix is finalised, and are never accepted as pivots.
inline constexpr double kZeroDrop = 1e-11;

/// Presolve fill-in cancellation, RELATIVE (#792). A doubleton substitution adds -a'b/a to
/// an existing coefficient; when the two nearly cancel, what is left is rounding and is set to
/// an exact zero. The test is |sum| <= kFillInCancellation * max(|before|, |added|): the
/// same 1e-11 that kZeroDrop applied absolutely, so on a model whose entries are of order one
/// nothing changes, but a coefficient of 1e-12 that is DATA (a row or column scaled by 2^-20,
/// Netlib under the stress set of #762) is no longer read as zero. Every other coefficient
/// presolve reads from the model is data too, and is zero only when it is exactly zero, the
/// rule the MPS reader follows since #590.
inline constexpr double kFillInCancellation = 1e-11;

/// Markowitz threshold for sparse LU pivoting (Phase 6). A candidate pivot must be at
/// least this fraction of the largest magnitude in its column to be numerically eligible.
/// 0.01 is the standard simplex compromise between sparsity and stability (Suhl & Suhl).
inline constexpr double kMarkowitzThreshold = 0.01;

/// Column substitution from equality rows (presolve_substitution, src/presolve/
/// substitute.cpp). The eliminated column's coefficient must be at least this fraction of
/// the largest entry of its row and of its column: the same stability argument as a
/// Markowitz pivot, since the substitution IS one elimination step, and it bounds the
/// multipliers a_ie / a_re that fill-in applies by 1 / threshold.
inline constexpr double kSubstitutionPivotThreshold = kMarkowitzThreshold;
/// Fill-in limits on one substitution (Gondzio 1997): the row may hold at most this many
/// entries and the column at most this many, and their Markowitz count (len - 1)(count - 1),
/// the most entries the elimination can create, at most kSubstitutionMaxFill.
inline constexpr Index kSubstitutionMaxRowLength = 16;
inline constexpr Index kSubstitutionMaxColumnCount = 16;
inline constexpr Index kSubstitutionMaxFill = 32;
/// The stage is applied only when it removes at least this fraction of the rows. Below it
/// the gain in iterations is small and the dual simplex takes a different path for it,
/// which on the pilot family cost more than it saved in every arm measured: pilot87
/// (2.8% of rows removable) went from 14.3 s to 46.7 s through a numerical_error and the
/// unscaled retry at the 0.01 pivot threshold, to 64.6 s at 0.1 and to the 600 s limit at
/// 0.5; pilot (4.7%) took 3,720 to 4,569 iterations against 3,487. Every Netlib instance
/// with a larger reduction measured gained (perold 7.7%, 25fv47 7.5%, d2q06c 9.6% up to
/// ganges 50%); bench.yml a527-measure runs 39558248485, 39579347698, 39579359625.
inline constexpr double kSubstitutionMinRowFraction = 0.05;

/// Hyper-sparse FTRAN and BTRAN (#464, lu_hyper_sparse): the symbolic reach is tried when
/// the right-hand side has at most this fraction of m nonzeros, and abandoned for the full
/// loops once the steps it reaches pass the same fraction. Hall & McKinnon, "Hyper-sparsity
/// in the revised simplex method and how to exploit it", Comput. Optim. Appl. 32 (2005),
/// call a result hyper-sparse below 10% and find the full loops cheaper above it; the
/// density bins the LU logs (LuSolveStats) are there to re-set it from our own data.
inline constexpr double kHyperSparseDensity = 0.10;

/// The weight of the newest result in a solve kind's predicted density
/// (SparseLu::DensityHistory): predicted += w * (latest - predicted), a memory of roughly
/// 1/w solves. From a sparse history, one fully dense result lifts the prediction to the
/// kHyperSparseDensity switch and the next solve skips the symbolic search; a kind that
/// stays dense stays skipped, and one that turns sparse again comes back under the switch
/// within about twenty sparse results (0.9^22 < 0.1). A heuristic weight, not a tolerance: it
/// decides only which of two bitwise-identical solves runs.
inline constexpr double kHyperSparseHistoryWeight = 0.1;

/// Below this, a computed pivot element is treated as a singular basis rather than a pivot.
inline constexpr double kPivotTolerance = 1e-9;

/// A floating-point dual, reduced cost or range is reported as CERTIFIED (#757) when it is
/// within this relative distance, max(1, |exact|), of the value the basis gives in exact
/// rational arithmetic, and as corrected otherwise. The float values come from one LU solve
/// of a basis the simplex has already accepted, so a well-conditioned basis lands within a
/// few hundred ulps (~1e-13 relative); 1e-9 leaves four orders of magnitude for conditioning
/// and still flags anything a planner would read as a different number.
inline constexpr double kSensitivityAgreement = 1e-9;

/// The dual ratio test's pivot floor RELATIVE to the row it is choosing from (#244):
/// a candidate whose |alpha_rj| is below this fraction of the row's largest |alpha_rj| is
/// skipped even when it clears kPivotTolerance. On an unscaled row whose entries are O(1e+3)
/// an absolute 1e-9 floor lets a pivot of 1e-8 through, and that is how a basis went
/// singular a few iterations later on every generated scale model (five repairs in 1,200
/// iterations on the 5,000-row random one). Koberstein, "The dual simplex method,
/// techniques for a fast and stable implementation" (thesis, 2005), sec. 6.2.2.2, uses
/// this value.
inline constexpr double kDualPivotRelativeFloor = 1e-7;

// ---------------------------------------------------------------------------------------
// Simplex
// ---------------------------------------------------------------------------------------

/// Reduced cost must beat this in magnitude to be an eligible entering candidate.
inline constexpr double kDualPricing = 1e-7;

/// Dual steepest edge (#411; Forrest & Goldfarb 1992): a warm basis with at most this many
/// rows starts from the exact row norms of B^-1, one BTRAN per row; a larger one starts at
/// 1, as Devex does, and the update takes it from there. The floor keeps a weight that
/// rounding drove to zero or below from turning a pricing score infinite.
inline constexpr int kDualSteepestEdgeExactInitRows = 2000;
inline constexpr double kDualSteepestEdgeWeightFloor = 1e-4;

/// Number of consecutive degenerate iterations after which the primal simplex switches to
/// Bland's rule. Bland's rule is provably non-cycling but prices badly, so it is a fallback
/// and not the default (Chvatal, "Linear Programming", ch. 3).
inline constexpr int kBlandSwitchIterations = 50;

/// The primal simplex's optimality cleanup (#548). "No reduced cost beyond kDualFeasibility"
/// bounds each dual violation, not what it costs: a column at its lower bound priced at
/// -6.9e-08 with no upper bound still lowered pilot87's objective by 3.4e-04 when entered,
/// 1.1e-06 relative, so the tolerance-optimal basis stopped away from the exact optimum. At the
/// optimal exit, on fresh factors, a nonbasic column whose reduced cost has the improving sign
/// and a magnitude above kPolishReducedCost is pivoted in when the ratio test shows the pivot
/// lowers the objective by more than kPolishObjectiveGain relative, which is kDualityGap: the
/// accuracy an optimality claim promises. Every such pivot strictly lowers the objective by a
/// finite amount, so the pass cannot cycle; kPolishPivotLimit bounds it anyway, and at most
/// kPolishCandidates columns are ratio-tested per round before the claim is made as before.
inline constexpr double kPolishReducedCost = kZeroDrop;
inline constexpr double kPolishObjectiveGain = kDualityGap;
inline constexpr int kPolishPivotLimit = 1000;
inline constexpr int kPolishCandidates = 8;

/// Feasibility tolerance used inside the ratio test, deliberately looser than
/// kPrimalFeasibility so that a marginally infeasible basic variable does not block a pivot.
inline constexpr double kRatioTestFeasibility = 1e-9;

/// Bound relaxation for the Harris two-pass ratio test (issue #67). Pass one finds the
/// tightest step allowed if every candidate row's bound were loosened by this much; pass two
/// then picks, among the rows whose EXACT (unrelaxed) step still fits under that limit, the
/// one with the largest pivot magnitude - buying numerical stability at the cost of a
/// controlled amount of new infeasibility.
///
/// PROVABLY TIGHTER THAN kPrimalFeasibility, which is what makes the trade safe: the
/// realised step is capped at the relaxed limit (see ratio_test()), so a Harris-relaxed pivot
/// alone can move a basic variable at most kHarrisRelaxation + kRatioTestFeasibility past its
/// bound - an order of magnitude inside kPrimalFeasibility, so it can never by itself turn a
/// point recompute_quality() would call feasible into one it calls infeasible.
inline constexpr double kHarrisRelaxation = 0.1 * kPrimalFeasibility;

/// The dual simplex's Harris relaxation (#465, dual_ratio_test=harris): pass one of the dual
/// ratio test loosens every candidate's reduced cost by this much. The same tenth of the
/// tolerance as the primal's kHarrisRelaxation, for the same reason: a column the step
/// passes by at most this much ends with a wrong-signed reduced cost an order of magnitude
/// inside kDualFeasibility, and the entering column's own wrong sign is removed by a cost
/// shift rather than by a backward step (Koberstein 2005, ch. 6).
inline constexpr double kDualHarrisRelaxation = 0.1 * kDualFeasibility;

/// Cost perturbation at the start of the dual simplex (#465, dual_perturb_costs_at_start;
/// Koberstein 2005, ch. 6). Applied only when the structural costs take fewer than
/// kDualStartPerturbationDistinctFraction * n distinct values, the shape of a model whose
/// ties the dual ratio test cannot break (brazil3, mostly zero costs). Each nonbasic
/// structural cost moves by xi_j = kDualStartPerturbationAbsolute +
/// kDualStartPerturbationRelative * |c_j|, times a per-column factor in [0.5, 1], in the
/// direction that keeps its reduced cost dual feasible. The absolute part is 100 times the
/// dual tolerance, so the shifts are distinct at the resolution the ratio test compares
/// reduced costs at.
inline constexpr double kDualStartPerturbationDistinctFraction = 0.25;
inline constexpr double kDualStartPerturbationAbsolute = 100.0 * kDualFeasibility;
inline constexpr double kDualStartPerturbationRelative = 1e-5;

// ---------------------------------------------------------------------------------------
// First-order method (PDHG)
// ---------------------------------------------------------------------------------------

/// PDHG results are reported at BOTH of these, separately, never blended
/// (ENGINEERING_RULES.md).
inline constexpr double kPdhgLoose = 1e-4;

/// Crossover from an interior point that is NOT optimal (#474, crossover_from_nonoptimal):
/// the largest scaled primal and dual infeasibility (Solution::*_infeasibility_scaled) a
/// feasible, time-limited, iteration-limited or numerically failed interior point may carry
/// and still be handed to the simplex as a starting point. The push itself refuses a point
/// whose guessed basis reproduces it worse than 1e3 times the primal tolerance
/// (simplex_push.cpp), which is this same 1e-4; the dual side is held to the same number so
/// "small primal and dual infeasibility" is one figure, the loose level PDHG reports at and
/// the verifier's ceiling for a stated infeasibility (#461). It only chooses where the
/// simplex starts: the vertex it returns is judged at kPrimalFeasibility and
/// kDualFeasibility like any other.
inline constexpr double kCrossoverStartInfeasibility = 1e-4;

/// The crossover's pivot budget, per row and column of the model it runs on (#1072): its
/// simplex stops with the interior point's answer standing after this many pivots times
/// rows + columns, whatever the clock says, so where it gives up is the same on every
/// machine. Over every successful crossover recorded in bench/results on 19 Oct 2026 (99
/// Netlib and Kennington instances, push pivots included, per row and column of the
/// original model) the largest was grow7's 1,179 pivots on 441, 2.67; Kennington's largest
/// was cre-a's 0.27. 10 leaves almost four times the worst. cycle's failing push (221,027
/// pivots, 57 per row and column of its presolved model) is stopped at 38,920.
inline constexpr double kCrossoverPivotsPerDimension = 10.0;
inline constexpr double kPdhgTight = 1e-8;

/// The LP PDHG's restart criteria on the relative KKT error of the restart candidate
/// (src/pdhg/pdhg.cpp; Applegate et al., PDLP, NeurIPS 2021, section 3.3): the candidate
/// at or under kPdhgRestartSufficient times the error at the last restart restarts at once;
/// at or under kPdhgRestartNecessary times it, having grown since the previous check (the
/// "no local progress" clause), restarts under option pdhg_restart_necessary; and a period
/// that reaches kPdhgRestartArtificial of all iterations so far restarts regardless. The
/// same three numbers the batch engine uses (kBatchPdhgRestart*).
inline constexpr double kPdhgRestartSufficient = 0.2;
inline constexpr double kPdhgRestartNecessary = 0.8;
inline constexpr double kPdhgRestartArtificial = 0.36;

/// PDHG feasibility polishing (#483, option pdhg_feasibility_polish; Applegate, Hinder, Lu &
/// Lubin, arXiv:2501.07018, the feasibility polishing section). Tried at the first
/// convergence evaluation at or after kPdhgPolishFirstIteration and again each time the
/// iteration count doubles, only while the main run's relative gap is at or under
/// kPdhgPolishGap; each of the primal and the dual phase is given iterations /
/// kPdhgPolishBudgetDivisor steps. Its target is kPdhgTight relative, together with the
/// absolute kPrimalFeasibility / kDualFeasibility. The three numbers are the issue's.
inline constexpr Count kPdhgPolishFirstIteration = 100;
inline constexpr double kPdhgPolishGap = 1e-2;
inline constexpr Count kPdhgPolishBudgetDivisor = 8;
/// The last polish, of the point a run stopped short is about to report, gets this many
/// steps per phase for each main-run iteration (#1000). It is the promise that a stopped run
/// hands back a feasible point, so it is not [AHLL25]'s fraction. On share2b stopped at 16
/// limits from 1,000 to 5,000 (g++ 11.4) the primal phase needed more than one step per
/// iteration from 3 of them - those whose stopped point was at relative primal 3e-2 to 6e-2 -
/// and reached from all 16 at two; MSVC stopped at 2,500 leaves such a point too. The phase
/// returns the moment it reaches, so the multiplier costs only where it fails.
inline constexpr Count kPdhgFinalPolishBudgetMultiplier = 2;

/// PDHG constant step (#482, option pdhg_constant_step): eta = kPdhgConstantStepShare / U,
/// U the certified upper bound on ||A||_2 of src/la/operator_norm.hpp, so tau sigma ||A||^2 =
/// eta^2 ||A||^2 <= 0.996 < 1 holds strictly (Chambolle & Pock 2011, Theorem 1) even where U
/// is exact. 0.998 is the share Lu, Peng & Yang use (cuPDLPx, arXiv:2507.14051), there on an
/// estimate of the norm. The bound is built from kPdhgConstantStepPowerIterations power
/// steps on A^T A (its lower side, for the log) and as many on |A|^T |A| (the
/// Collatz-Wielandt part of its upper side).
inline constexpr double kPdhgConstantStepShare = 0.998;
inline constexpr int kPdhgConstantStepPowerIterations = 30;

// ---------------------------------------------------------------------------------------
// Cuts
// ---------------------------------------------------------------------------------------

/// Maximum density (nonzero structural coefficients / original structural columns) for a cut
/// to be accepted. A performance/robustness filter to keep the LP relaxation sparse. With
/// cut_support_floor set, the cap is max(floor, this fraction of n) nonzeros (#496).
inline constexpr double kCutMaxDensity = 0.2;

/// Minimum efficacy (violation over the Euclidean norm of the cut) under cut_efficacy_test
/// (#496): the distance from the LP point to the cut's hyperplane, which is scale-free
/// where the absolute violation is not. Wesselmann & Suhl, *Implementing cutting plane
/// management and selection techniques*, 2012.
inline constexpr double kCutMinEfficacy = 1e-4;

/// Maximum ratio of max(abs(coeff)) / min(abs(coeff)) for materially nonzero coefficients.
/// Prevents extreme coefficient scaling from ruining the numerical stability of the LP.
inline constexpr double kCutMaxCoefficientRatio = 1e6;

/// The root's bounded admission of dense cuts (#496, `cut_dense_max`). Refusing every cut
/// above kCutMaxDensity refused every Gomory cut on opt1217, rlp1, neos5, pk1 and gen-ip016,
/// whose root gap then closed by nothing; admitting all of them (cut_support_floor=100000)
/// measured slower on the whole set. A dense cut is admitted at the root only when it
/// passed every other test, most efficacious first, while the nonzeros of all admitted
/// dense cuts stay within this share of the model's own nonzeros at the root: at 1.0 the
/// node LP at most doubles in nonzeros, whatever n is. The default of
/// `cut_dense_nonzero_share`.
inline constexpr double kCutDenseNonzeroShare = 1.0;

/// A dense cut is held to a tighter dynamism than kCutMaxCoefficientRatio. A Gomory row's
/// rounding error is relative to its LARGEST entry, so an entry 1e6 below it can be mostly
/// error, and a dense row has many such entries that every node LP then carries. At 1e4
/// every kept entry is at least 1e-4 of the largest, so an error of 1e-12 of the largest
/// is at most 1e-8 of the entry. A conservative choice of ours, in the spirit of the
/// dynamism limit Cornuejols, Margot and Nannicini (2013) study, not tuned from it.
inline constexpr double kCutDenseMaxCoefficientRatio = 1e4;

/// A cut coefficient this far below the cut's own largest coefficient is the ROUNDING of
/// the arithmetic that produced it rather than a quantity. The Gomory derivation is a few
/// dozen multiply-adds over the tableau row, so its relative error is a small multiple of the
/// machine epsilon (2.2e-16); 1e-14 leaves two orders of margin. Anything at or below this is
/// treated as the zero it is. Anything ABOVE it that is nevertheless dropped - because the
/// solver treats coefficients under kZeroDrop as zero everywhere - is paid for by weakening
/// the cut's constant, which is what keeps the cut valid; see src/mip/cuts.cpp.
inline constexpr double kCutNoiseRelative = 1e-14;

/// GMI safety, under `gmi_safety` (#496 item 4). Cornuejols, Margot and Nannicini, "On the
/// safety of Gomory cut generators", Math. Programming Computation 5 (2013), measure what a
/// GMI generator's parameters do to cut validity: among them the minimum fractionality of
/// the source row's basic variable, the maximum dynamism of the cut and a relaxation of its
/// right-hand side. The values here are conservative choices of ours, not tuned from that
/// study. A source row whose basic value is within kGmiMinFractionality of an integer gives
/// no cut: its f0 is the divisor of every coefficient, so a small f0 magnifies the tableau
/// row's rounding by 1/f0. The emitted rhs is loosened by kGmiRhsRelaxAbsolute plus
/// kGmiRhsRelaxRelative * |rhs|, which only weakens the cut. Dynamism is already capped for
/// every family by kCutMaxCoefficientRatio in the filter.
inline constexpr double kGmiMinFractionality = 0.01;
inline constexpr double kGmiRhsRelaxAbsolute = 1e-9;
inline constexpr double kGmiRhsRelaxRelative = 1e-9;

/// Minimum root-LP violation for a cut to be accepted. Valid cuts that are not violated
/// or barely violated are safely rejected to save LP solves.
inline constexpr double kCutViolationTolerance = 1e-5;

/// Presolve, parallel rows (#412; Andersen & Andersen 1995, sec. 5). Two rows are
/// proportional when every coefficient matches its scaled counterpart to this relative
/// tolerance, measured against max(1, |a|, |scaled|). Tighter than the pivot drop because a
/// merge that is even slightly wrong changes the feasible set silently; looser than
/// machine epsilon because the coefficients arrive through a reader and a scaling pass.
inline constexpr double kPresolveParallelRowTolerance = 1e-9;

/// Presolve, coefficient tightening (#511; Savelsbergh 1994; Achterberg et al. 2020, sec.
/// 3). When a row's activity is not an exact integer sum, the gap g = M - b it tightens to
/// is rounded UP by this much relative to max(1, the row's total magnitude): a g that comes
/// out a hair too small would cut off an integer point, a hair too large only leaves the
/// row a hair weaker. Also the relative margin a propagated bound is loosened by, for the
/// same reason. A few hundred ulps over a long row, far below every feasibility tolerance.
inline constexpr double kPresolveCoefficientSafety = 1e-9;

/// A coefficient is only tightened when it drops by more than this relative to max(1, |a|),
/// and only towards a g at least this large relative to the row's largest coefficient: a
/// change below it moves nothing the LP can see, and a coefficient driven to near zero by a
/// row that is barely binding is noise, not a reduction.
inline constexpr double kPresolveCoefficientMinStep = 1e-6;

/// A continuous column's bound propagated from a row is only written when it improves the
/// old one by more than this relative to max(1, |old bound|) (Achterberg et al. 2020 use a
/// rule of the same order): below it propagation converges geometrically and forever on a
/// cycle of rows, and the LP gains nothing from the last digits. An integer bound moves by
/// whole units and has no such floor.
inline constexpr double kPresolveBoundMinStep = 1e-3;

/// A propagated bound larger in magnitude than this is not written: it is finite in name
/// only and would put a coefficient range of 1e9 into the LP for no gain.
inline constexpr double kPresolveMaxPropagatedBound = 1e9;

/// Rounds of bound propagation followed by coefficient tightening, each one pass over the
/// rows. A tightened row propagates tighter bounds and a tighter bound tightens more rows,
/// so the two alternate; the cap is a work limit, not a convergence test.
inline constexpr int kPresolveCoefficientRounds = 20;

/// Postsolve of propagated bounds (#485): how many rows the search for the basis entry that
/// leaves for a column on a propagated bound visits before it gives up and leaves the
/// statuses as the engine had them. The degenerate chains it walks (a column held by a
/// singleton row, that row's logical basic) are a few rows long; the cap is a work limit.
inline constexpr int kImpliedBoundBasisSearchRows = 64;

/// Cut selection (#415; Wesselmann & Suhl, "Implementing cutting plane management and
/// selection techniques", 2012; Achterberg 2007, ch. 8). A cut's score is the weighted sum
/// of its efficacy (the Euclidean distance from the LP point to its hyperplane), its
/// parallelism to the objective and the share of its support on integer columns; the
/// weights are Wesselmann and Suhl's defaults, efficacy first and the other two as
/// tie-breakers.
inline constexpr double kCutSelectionEfficacyWeight = 1.0;
inline constexpr double kCutSelectionObjectiveWeight = 0.1;
inline constexpr double kCutSelectionIntegerSupportWeight = 0.1;

/// Two cuts whose normals have a cosine above this say nearly the same thing; the second
/// adds a row and no information and is deferred to a later round instead.
inline constexpr double kCutMaxParallelism = 0.9;

/// Cuts a round may keep waiting for a later round after selection; the best by score stay.
inline constexpr int kCutWaitingLimit = 500;

/// Node solves a cut row may sit slack (its logical basic) before it is freed, the default
/// of `mip_cut_age_limit` (Achterberg 2007, sec. 8.10 uses a comparable age). With
/// `mip_cut_pooling` a freed row is re-imposed when a later node LP point violates it (#497).
inline constexpr int kCutRowAgeLimit = 50;

/// Debug-solution check (#500): how far a known feasible point may sit outside a cut, a row
/// or a bound, relative to max(1, |rhs|, the largest term of the activity), and still count
/// as inside it. A valid cut computed in floating point can exclude a point on its boundary
/// by rounding: the Gomory derivation is a few dozen multiply-adds, each with relative error
/// near 1e-16, and presolve folds fixed columns into row bounds the same way. 1e-6 is ten
/// times the simplex's own row tolerance (kPrimalFeasibility), so rounding never trips it,
/// and a cut that really removes an integer point misses it by a whole fraction of a unit,
/// which is five or more orders of magnitude larger.
inline constexpr double kDebugSolutionTolerance = 1e-6;

// ---- The proximal interior point for convex QP (#490, qp_algorithm=ipm) -------------------

/// The complementarity products the QP interior point closes itself before handing the
/// point to the in-process KKT gate: a tenth of the gate's own kComplementarity, so a point
/// it calls optimal on the products passes the gate with room to spare.
inline constexpr double kQpIpmComplementarityShare = 0.1;

/// The pivot threshold of the quasi-definite factorization, as a fraction of the smaller
/// proximal parameter: a pivot below it (or of the wrong sign) is replaced and counted.
inline constexpr double kQpIpmPivotShare = 0.1;

/// When a pivot comes out wrong the proximal parameters are multiplied by this and the
/// matrix refactorized, at most kQpIpmRegularizationAttempts times.
inline constexpr double kQpIpmRegularizationRaise = 100.0;
inline constexpr int kQpIpmRegularizationAttempts = 8;

/// Iterative refinement of each Newton step against the UNREGULARIZED KKT matrix (rho =
/// delta = 0), the regularized factors as the preconditioner (Altman & Gondzio 1999, see
/// src/qp/qp_ipm_system.cpp): at most this many corrections per solve, each kept only when it
/// lowers the unregularized residual, and the refinement stops after one that gains less than
/// 10% (it has stalled at what the factors allow). The LP path's kIpmProximalRefinementSteps
/// and kIpmProximalRefinementProgress, tuned apart.
inline constexpr int kQpIpmRefinementSteps = 5;
inline constexpr double kQpIpmRefinementProgress = 0.9;
/// qp_ipm_krylov_refinement (#981): the Krylov space the GMRES refinement of each Newton step
/// against the unregularized KKT matrix may build, preconditioned by the regularized factors.
inline constexpr int kQpIpmKrylovDimension = 30;
/// qp_ipm_adaptive_dual_regularization (#981): a Newton step that leaves more than this share
/// of the primal residual b - M v in place (infinity norm) cuts the floor of the dual
/// regularization delta by kQpIpmDualFloorCut, never below kQpIpmDualFloorMin (the option
/// range's own lower end for qp_ipm_regularization).
inline constexpr double kQpIpmStepPrimalShare = 0.1;
inline constexpr double kQpIpmDualFloorCut = 0.01;
inline constexpr double kQpIpmDualFloorMin = 1e-14;

/// The active-set finish after the QP interior point (#980, qp_ipm_finish, off by default):
/// a bound counts as active when the point is within this distance of it, scaled by
/// max(1, |bound|). Loose on purpose - pinning a bound that should have stayed inactive only
/// costs a rejected finish (the acceptance guard in src/qp/qp_ipm_finish.cpp), never a wrong
/// answer; chosen by hand, not tuned on a benchmark.
inline constexpr double kQpIpmFinishActiveDistance = 1e-4;

/// The finish's rank decision (#980): an active row of the reduced system is dropped as
/// dependent when the QR with column pivoting of the active rows (restricted to the free
/// columns) finds its remaining norm at or below this share of the first pivot's. 1e-10 is
/// the prototype's value on the Maros-Meszaros holdouts (qseba: 68 of 972 rows dependent,
/// the first dropped norm 13 orders below the first pivot); chosen by hand, a row kept that
/// should have been dropped only costs a rejected finish.
inline constexpr double kQpIpmFinishRankTolerance = 1e-10;

/// Passes of symmetric Ruiz equilibration on the finish's dense reduced KKT matrix before
/// its rank decision (#1079). Each pass halves every row's distance from unit size in
/// orders of magnitude, so 10 bring a row of 1e-21 coefficients (huestis) to within a
/// factor 1.05 of 1. Chosen from that arithmetic, not tuned on a benchmark.
inline constexpr int kQpIpmFinishEquilibrationPasses = 10;

/// At most this many re-solves of the finish, each after one change to the active set - a
/// free column moved onto the bound it violates or whose reduced cost it cannot zero, or a
/// pinned bound whose multiplier came out with the wrong sign released (the primal
/// active-set rule, one change per step). 10 covers qgrow15 (one release), chosen by hand.
inline constexpr int kQpIpmFinishRounds = 10;

/// #893: a certificate candidate read off the QP interior point's iterates is offered to the
/// checker first with every entry at or below this share of its largest set to zero, then
/// as it came. The step that runs away along a ray still carries the part of the iterate
/// that is converging (a ray (2.9e-07, 1) where the exact one is (0, 1), measured on a
/// two-column model), and the rounded vector is the cleaner proof when it holds. Either one
/// is reported only if the checker accepts it.
inline constexpr double kQpIpmCertificateRounding = 1e-6;

/// #494, #893 (miqp_node_ipm_warm_start): the iterate a QP interior point hands on as a warm
/// start is the first one whose relative primal residual, dual residual and gap are all at or
/// below this, not the optimum: an advanced point that is still well inside the bounds
/// (Gondzio 1998, see src/qp/qp_ipm_warm.cpp). Our choice, not the paper's number: on the 80
/// random MIQPs of tests/unit/test_miqp_node_ipm.cpp, 1e-2 took 2,088 node IPM iterations
/// against 2,102 at 1e-1 and 2,100 at 3e-1, and the cold start 2,617.
inline constexpr double kQpIpmWarmSaveLevel = 1e-2;

/// The shift of a warm start into the interior: every column at least this far from each of
/// its finite bounds (half its width when narrower). 1e-3 left 9 of 242 warm starts on the
/// 80 random MIQPs to fail or stall, 1e-1 none.
inline constexpr double kQpIpmWarmShiftDistance = 0.1;

/// The least centring target of the shift, so the centrality box below is never empty.
inline constexpr double kQpIpmWarmCentringFloor = 1e-6;

/// ...and every complementarity product s z then moved into [low mu, high mu] by its
/// multiplier, mu the average product at the shifted point, the centrality box of Gondzio's
/// multiple centrality correctors (Gondzio 1996, beta_min = 0.1 and beta_max = 10).
inline constexpr double kQpIpmWarmCentringLow = 0.1;
inline constexpr double kQpIpmWarmCentringHigh = 10.0;

/// A warm run is abandoned for the cold start when its worst relative measure (primal
/// residual, dual residual, gap) has not fallen by kQpIpmWarmStallFactor over the last
/// kQpIpmWarmStallWindow iterations. Mehrotra's method gains an order of magnitude every two
/// or three iterations when it is converging, so ten iterations without one is a stall.
inline constexpr int kQpIpmWarmStallWindow = 10;
inline constexpr double kQpIpmWarmStallFactor = 0.1;

/// Gondzio's multiple centrality correctors (Comput. Optim. Appl. 6, 1996), #981: the trial
/// step aims kQpIpmGondzioStepGain further than the current one; trial products are pulled
/// into [kQpIpmGondzioBetaMin, kQpIpmGondzioBetaMax] times sigma mu; a corrector is kept
/// only when it lengthens the step by kQpIpmGondzioAcceptShare of the aimed gain. The
/// values are the paper's (sec. 4).
inline constexpr double kQpIpmGondzioStepGain = 0.1;
inline constexpr double kQpIpmGondzioBetaMin = 0.1;
inline constexpr double kQpIpmGondzioBetaMax = 10.0;
inline constexpr double kQpIpmGondzioAcceptShare = 0.1;

// ---- Proximal regularization of the LP interior point (#473, ipm_proximal_regularization) --

/// rho = delta = max(floor, min(previous, kIpmProximalShare * mu)), starting from
/// kIpmProximalStart: the regularization follows mu down and rises only when a pivot comes
/// out wrong (then by kIpmProximalRaise, at most kIpmProximalAttempts factorizations per
/// iteration, with the pivot threshold kIpmProximalPivotShare of it). The floor is 1e-8, the
/// default path's primal regularization, so this path never regularizes less than the
/// default one does; a lower floor has not been measured.
inline constexpr double kIpmProximalStart = 1e-6;
inline constexpr double kIpmProximalFloor = 1e-8;
inline constexpr double kIpmProximalShare = 1e-2;
inline constexpr int kIpmProximalAttempts = 3;
/// The LP path's own pivot rule. The values are the QP interior point's (kQpIpmPivotShare,
/// kQpIpmRegularizationRaise) today, but the two paths are tuned apart: a pivot is lifted
/// below kIpmProximalPivotShare * rho, and a lifted pivot raises rho by kIpmProximalRaise.
inline constexpr double kIpmProximalPivotShare = 0.1;
inline constexpr double kIpmProximalRaise = 100.0;
/// Iterative-refinement corrections on the unregularized Newton system per solve, at most.
inline constexpr int kIpmProximalRefinementSteps = 5;
/// A correction is kept whenever it lowers the unregularized residual; the refinement stops
/// after one that gains less than 10% (it has stalled at what the regularization allows).
inline constexpr double kIpmProximalRefinementProgress = 0.9;
/// The refinement's target, on the unregularized residual relative to max(1, |g|, |r_b|)
/// (infinity norms): the interior point's own 1e-8 convergence tolerance. A solve left above
/// it is counted and reported, and the next factorization's rho is shrunk by
/// kIpmProximalRefinementShrink toward the floor: a smaller rho is a K_reg nearer K_0, so the
/// refinement's contraction improves.
inline constexpr double kIpmProximalRefinementTarget = 1e-8;
inline constexpr double kIpmProximalRefinementShrink = 0.1;
/// GMRES iterations (one restart cycle) on the same factors after a refinement that ended
/// above kIpmProximalRefinementTarget: the stationary refinement stalls where Theta^-1 << rho.
inline constexpr int kIpmProximalKrylovSteps = 20;
/// After a non-finite Newton direction (#209) the rho floor rises by this factor per raise,
/// capped at kIpmProximalRecoveryCap. The default path's x1e4 on its 1e-10 dual
/// regularization, applied to the 1e-8 rho floor, took rho to 1 after two raises: a proximal
/// term as large as the matrix, whose refinement then has nothing to converge to.
inline constexpr double kIpmProximalRecoveryRaise = 100.0;
inline constexpr double kIpmProximalRecoveryCap = 1e-4;

// ---- The homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd) ----------

/// Convergence to an optimum, measured on the point x/tau, y/tau, z/tau: relative primal and
/// dual residuals, relative gap and the worst relative complementarity product, each at or
/// below this. The default path's kIpmTolerance, kIpmGap and kIpmComplementarity (1e-8, set in
/// ipm.cpp for the reasons written there), so the two paths converge to the same standard.
inline constexpr double kIpmHsdTolerance = 1e-8;
/// A certificate candidate (the dual ray y for infeasibility, the primal ray x for
/// unboundedness) is offered to the checker first with every entry at or below this share of
/// its largest set to zero, then as it came: the QP interior point's kQpIpmCertificateRounding
/// and for the same reason, the part of the iterate that converges to zero along the ray.
inline constexpr double kIpmHsdCertificateRounding = 1e-6;
/// The iteration stops without a verdict once mu has fallen this far below its starting
/// value while tau and kappa still have not decided: every residual has fallen as far with
/// it (they shrink at one rate in the embedding), so nothing is left for another step to show.
inline constexpr double kIpmHsdMuFloor = 1e-15;

// ---- Gondzio's centrality correctors in the LP interior point (#472) ----------------------

/// The products a corrector aims for: [kIpmCentralityBetaMin, kIpmCentralityBetaMax] times
/// the target sigma mu, Gondzio's (1996) 0.1 and 10 (the issue's gamma = 0.1, and 1 / gamma).
inline constexpr double kIpmCentralityBetaMin = 0.1;
inline constexpr double kIpmCentralityBetaMax = 10.0;
/// A corrector is kept only when alpha_p + alpha_d grows by this factor: 1% (the issue).
inline constexpr double kIpmCentralityAcceptance = 1.01;
/// The trial point is at the aspiration step min(scale * alpha + shift, 1) of Colombo and
/// Gondzio (2008): alpha + delta_alpha with delta_alpha growing as the step does.
inline constexpr double kIpmCentralityAspirationScale = 1.5;
inline constexpr double kIpmCentralityAspirationShift = 0.3;
/// The per-iteration budget from the factor's shape (ipm/centrality.hpp): the ratio of a
/// factorization's work (about sum_j c_j^2) to a solve's (about kIpmCentralitySolveWork *
/// sum_j c_j, a forward and a backward sweep over L, each a multiply and an add per entry).
/// One corrector while that ratio is at most kIpmCentralityBudgetStart, then one more each
/// time it grows by the factor kIpmCentralityBudgetGrowth.
inline constexpr double kIpmCentralitySolveWork = 4.0;
inline constexpr double kIpmCentralityBudgetStart = 2.0;
inline constexpr double kIpmCentralityBudgetGrowth = 2.0;
/// Where a corrector's solve is not one back-solve the budget above assumes (review of
/// #668). On the dense-column path (#467) and the n x n side (#469) every solve is a
/// preconditioned conjugate gradient of up to kIpmPcgMaxIterations steps: at most this many
/// correctors per iteration there.
inline constexpr int kIpmCentralityConjugateGradientCorrectors = 1;
/// On the proximal path (#473) a solve is up to 1 + kIpmProximalRefinementSteps back-solves
/// plus as many products with the unregularized system; the factor's ratio is divided by
/// that, and at most this many correctors run per iteration.
inline constexpr int kIpmCentralityProximalCorrectors = 2;

/// Binary probing (#512; Savelsbergh 1994; Achterberg et al. 2020). A probe x_j = v is
/// declared infeasible only when a row misses its bound by more than this, relative to
/// max(1, |bound|), on top of the rounding the activity sum can carry: a probe wrongly called
/// infeasible fixes a binary to the wrong value, so the test is ten times looser than the
/// feasibility tolerance the solver accepts a point at.
inline constexpr double kProbingInfeasibility = 1e-6;

/// The rounding margin of a propagated bound, relative to max(1, the row's total magnitude)
/// and divided by the column's coefficient: a continuous bound is loosened by it and an
/// integer one rounded past it, so no deduction is stronger than the arithmetic supports.
inline constexpr double kProbingSafety = 1e-9;

/// A continuous bound moves only when it improves by more than this relative to max(1,
/// |old bound|): below it propagation creeps geometrically along a cycle of rows forever.
inline constexpr double kProbingBoundMinStep = 1e-3;

/// A propagated bound larger in magnitude than this is not written: finite in name only.
inline constexpr double kProbingMaxBound = 1e9;

/// Matrix entries probing may visit in total, and in one probe. Probing is the most
/// expensive MIP presolve reduction (Achterberg et al. 2020 cap it by work too); past the
/// total the remaining binaries are left unprobed, past the per-probe cap a probe stops
/// propagating, which only means it deduces less.
inline constexpr std::int64_t kProbingWorkLimit = 20000000;
inline constexpr std::int64_t kProbingProbeWorkLimit = 1000000;

// ---------------------------------------------------------------------------------------
// Spatial branch and bound for products of columns (#514, src/global/)
// ---------------------------------------------------------------------------------------

/// FBBT (Belotti et al. 2009, sec. 3): a derived bound is relaxed outward by this relative
/// amount before it is applied, so rounding in the interval arithmetic cannot cut off a
/// feasible point; a bound moves only when it improves by more than kFbbtMinImprovement
/// relative, which also ends the passes on rows that would otherwise creep forever; and no
/// more than kFbbtMaxPasses sweeps over the rows are made per box.
inline constexpr double kFbbtSafety = 1e-9;
inline constexpr double kFbbtMinImprovement = 1e-6;
inline constexpr int kFbbtMaxPasses = 10;

/// Branching point: this weight on the relaxation's value of the branching column, the rest
/// on the box midpoint (the convex combination Belotti et al. 2009 discuss, sec. 5), kept at
/// least kGlobalBranchPointMargin of the width from either end so both children are real
/// parts of the box.
inline constexpr double kGlobalBranchPointLpWeight = 0.5;
inline constexpr double kGlobalBranchPointMargin = 0.1;

/// A column narrower than this, relative to max(1, |bound|), is not branched on again: at
/// that width the McCormick envelope is exact to the LP's own tolerance.
inline constexpr double kGlobalMinBranchWidth = 1e-9;

/// A product whose relaxation value is within this of the product of its factors, relative
/// to max(1, |x_a x_b|), is treated as satisfied when choosing where to branch.
inline constexpr double kGlobalProductTolerance = 1e-9;

/// The alternating LP (src/global/local_search.cpp): at most this many LP solves per call,
/// and it stops once a cycle through the sides improves the objective by less than
/// kGlobalLocalSearchProgress relative.
inline constexpr int kGlobalLocalSearchRounds = 8;
inline constexpr double kGlobalLocalSearchProgress = 1e-9;

/// A factor's implied value w / x_other is used only when the sum of the other factors'
/// squares exceeds this; below it the relaxation's own value is kept.
inline constexpr double kGlobalImpliedValueFloor = 1e-12;

/// The alternating LP runs at each of the first kGlobalHeuristicAlwaysNodes nodes and then
/// at every kGlobalHeuristicInterval-th node.
inline constexpr std::int64_t kGlobalHeuristicAlwaysNodes = 10;
inline constexpr std::int64_t kGlobalHeuristicInterval = 10;

/// Safe dual bounds (#519): the largest factor y is ever scaled down by in the (1 - e) y
/// retry, and how far past the first-order estimate of e the scaling goes so the rounding of
/// the second evaluation cannot undo it. Neumaier & Shcherbina, Math. Program. 99 (2004).
inline constexpr double kSafeBoundMaxShrink = 1e-6;
inline constexpr double kSafeBoundShrinkMargin = 4.0;

/// The certified gap of an optimal LP (#763): the relative gap below which the bound from
/// the reported duals is kept as it is, and the shifts along the basis direction tried
/// otherwise, relative to the largest term of c and A'y (src/core/lp_safe_bound.cpp): the
/// first is some 10^4 ulps of that term, past the rounding width of a reduced cost; each
/// next is kSafeBoundShiftGrowth times the last, up to kSafeBoundShiftLast.
inline constexpr double kCertifiedGapTarget = 1e-6;
inline constexpr double kSafeBoundShiftFirst = 1e-12;
inline constexpr double kSafeBoundShiftLast = 1e-6;
inline constexpr double kSafeBoundShiftGrowth = 100.0;
/// Rounds of bound propagation before the certified-gap bound gives up on a missing column
/// bound; on Netlib most instances stop finding new bounds well before ten (#763).
inline constexpr int kSafeBoundPropagationPasses = 10;
/// Stage 4 of the certified-gap bound (src/core/lp_safe_bound.cpp): the relative cost moves
/// tried on one-sided columns, from First up by Growth to Last. First sits two orders above
/// the 1e-11 rounding floor of a reduced cost; Last is the 1e-6 target itself, since the
/// bound gives up about delta per unit of the moved columns' values.
inline constexpr double kSafeBoundCostPerturbationFirst = 1e-9;
inline constexpr double kSafeBoundCostPerturbationLast = 1e-6;
inline constexpr double kSafeBoundCostPerturbationGrowth = 10.0;
/// Stage 4b of the safe bound (#1082, src/core/lp_safe_bound.cpp): rounds of floating-point
/// pivots on the columns the exact duals leave dual infeasible, each followed by the exact
/// duals of the new basis, all inside one exact_seconds. An effort cap, not a tolerance:
/// every round's bound is exact or absent, whatever the count.
inline constexpr int kSafeBoundPivotRounds = 8;
/// Stage 4b's dual phase (#1082): warm-started dual simplex solves from a basis whose exact
/// duals are dual feasible but whose bound is above the target, each followed by its exact
/// duals and more pivot rounds. An effort cap, like the rounds.
inline constexpr int kSafeBoundDualPhases = 2;
/// Bits the exact-dual refinement must have gained before its running approximation is
/// handed back as a polished floating-point y (ExactDualBound::refined, #1082): a double
/// carries 53, and the margin covers the error estimate's slack.
inline constexpr int kExactDualRefinedBits = 80;
/// The exact basis duals of the certified-gap bound (src/core/lp_exact_dual.cpp): leading bits
/// of each floating-point correction kept exactly (at most 62, so it fits a long long);
/// residual bits gained before the first rational reconstruction is tried (each later try
/// waits for 25% more); the iteration cap; a step that removes fewer bits than
/// kExactDualMinProgress counts as a stall, and kExactDualMaxStalls in a row give up; and
/// the bits of slack allowed on the estimate |x - y| <~ |z| 2^top(r).
inline constexpr int kExactDualStepBits = 50;
inline constexpr int kExactDualFirstAttemptBits = 128;
// 20,000 since #1082: a step now costs the same at 20,000 as at 1 (StepHistory), and
// pilot87 needs 5,297 at about 20 bits each; the wall-clock budget is the real limit.
inline constexpr int kExactDualMaxIterations = 20000;
inline constexpr int kExactDualMinProgress = 4;
inline constexpr int kExactDualMaxStalls = 5;
inline constexpr int kExactDualErrorMargin = 16;

/// MILP certificates (#518): how many ancestors' duals a leaf tries before it gives up on a
/// bound, and the largest magnitude an objective step or its rounded bound may have for the
/// Chvatal-Gomory rounding step to be written: below 2^53 every integer is a double, so the
/// ceiling and the product are exact (9e15 < 2^53 = 9.007e15).
inline constexpr Count kCertificateAncestorsTried = 8;
inline constexpr double kCertificateExactInteger = 9e15;
/// A cut whose certified derivation (#518, src/mip/cut_derivation.cpp) proves a right-hand
/// side above the cut's own by at most this, relative to max(1, |rhs|), is kept with the
/// proved right-hand side: that difference is the rounding of the generator's floating-point
/// arithmetic, which accumulates over the terms it summed and so is far more than one ulp of
/// the result. 1e-9 relative is loose on purpose; soundness does not rest on it, because the
/// row kept is the one with the PROVED right-hand side. More than this means the cut is not
/// the one the derivation describes, and it is dropped.
inline constexpr double kCertificateCutSlack = 1e-9;

/// The root separation loop (#495, `root_cut_loop`, off by default): separate, add, re-solve,
/// repeat, and stop at the first of these. Achterberg, "Constraint Integer Programming"
/// (thesis, 2007), ch. 8 describes the loop and a stall rule of this shape; the numbers are
/// the ones #495 proposes, chosen to bound the root's cost rather than tuned.
///
/// At most this many rounds, the first included.
inline constexpr int kRootCutMaxRounds = 20;
/// The bound has stalled when the last kRootCutStallRounds rounds together moved it by at
/// most kRootCutStallFraction of the reference: the gap to the incumbent when there is one,
/// max(1, |bound|) when there is not.
inline constexpr int kRootCutStallRounds = 3;
inline constexpr double kRootCutStallFraction = 1e-3;
/// Tailing off (Padberg and Rinaldi, "A branch-and-cut algorithm for the resolution of
/// large-scale symmetric traveling salesman problems", SIAM Review 33, 1991; Achterberg 2007,
/// ch. 8): the loop also stops after a round whose bound gain is at most this share of round
/// 1's. #956 measured the loop without it: the rounds that only crept added rows that every
/// node LP then paid for (enlight8 14 -> 100 rows, 0.45x the nodes in 60 s), while the
/// proofs it won came from rounds that moved the bound a lot (kaihu, b-ball). A round 1 that
/// moved nothing makes this "stop at the next round that moves nothing".
inline constexpr double kRootCutTailingOffFraction = 0.1;
/// And the loop stops once the solve has used this share of time_limit, so a root whose
/// every round is slow leaves the tree most of the time it was given. 0.2 until the
/// tailing-off rule above; 0.05 since (3 s of a 60 s limit).
inline constexpr double kRootCutTimeShare = 0.05;
/// And the loop adds at most max(kRootCutRowFloor, kRootCutRowShare * m) cut rows in all,
/// round 1 included, m the rows of the LP the root separates on before its first cut.
inline constexpr int kRootCutRowFloor = 100;
inline constexpr double kRootCutRowShare = 1.0;

/// GPU PDHG device loop (#478): iterations replayed on the device per host synchronisation.
/// Larger amortises the synchronisation further but checks the iteration and time limits
/// and the evaluation stride less often; 32 is under the evaluation interval of 40.
inline constexpr Count kPdhgDeviceLoopBlock = 32;

/// The least work, in nonzeros plus columns read, for which a column loop of the simplex
/// (reduced costs, the dual's pivot row), the sparse transpose product or a PDHG vector update
/// forks an OpenMP team (#514). Below it the loop runs on the calling thread: the fork and
/// join of a team cost microseconds on an idle machine and far more on a loaded one, where
/// every barrier waits for the scheduler, and a node LP of a few hundred nonzeros does its
/// whole loop in less. Default of the `omp_min_work` option; 0 restores a team on every loop.
/// A choice, set by the A/B in the PR that introduced it, not a measured crossover.
inline constexpr Count kOmpMinParallelWork = 50'000;

/// CPU PDHG parallel vector updates (#487): the fixed chunk the movement and interaction
/// sums are taken in. A constant, not a function of the thread count, which is what makes
/// the sums the same bits at any thread count. The chunk is also the unit of work a thread
/// takes, so it bounds the parallelism from above (a 10,000-column model has ten chunks);
/// 1024 doubles (8 KiB) keeps a chunk's working set of three or four vectors in L1, and a
/// vector shorter than one chunk runs on one thread. A choice, not a measurement.
inline constexpr Count kPdhgParallelChunk = 1024;

/// Feasibility Jump polls the caller's stop (interrupt, time limit) every this many work
/// units: about a millisecond of work, so a stop is seen promptly and the poll costs nothing.
inline constexpr Count kFeasibilityJumpPollWork = 65536;

/// GPU Feasibility Jump (#508; Corduk, Sielski, Boucher & Aatish, arXiv:2510.20499), one
/// search per CUDA block of 8 warps: each warp scores kGpuFeasibilityJumpSamplesPerWarp
/// sampled columns per move, 32 in all against the CPU's 25, the extra ones costing no wall
/// time since the warps run at once; a launch runs kGpuFeasibilityJumpLaunchSteps moves or
/// weight updates of every search before the host polls for points and for the stop; the
/// default is kGpuFeasibilityJumpRestartsPerSm searches per multiprocessor, as many as
/// kGpuFeasibilityJumpMemoryShare of the card's free memory holds.
inline constexpr int kGpuFeasibilityJumpSamplesPerWarp = 4;
inline constexpr int kGpuFeasibilityJumpLaunchSteps = 128;
inline constexpr int kGpuFeasibilityJumpRestartsPerSm = 2;
inline constexpr double kGpuFeasibilityJumpMemoryShare = 0.5;

/// Implied-integer detection (#513): every coefficient and right-hand side it reasons about
/// must be an integer of magnitude below 2^53, where a double holds every integer exactly
/// and fmod is exact, so the divisibility test adds no rounding of its own.
inline constexpr double kImpliedIntegerDataLimit = 9007199254740992.0;

/// OBBT at the root (#515): a column whose range is below kObbtFixedRange is not probed; a
/// probe's bound is written only when it tightens by at least kObbtMinimumTightening; the
/// objective cutoff keeps points better than the incumbent by kObbtCutoffEpsilon.
inline constexpr double kObbtFixedRange = 1e-6;
inline constexpr double kObbtMinimumTightening = 1e-8;
inline constexpr double kObbtCutoffEpsilon = 1e-6;

/// Root domain propagation (#510): synchronous rounds at most. Each round reads every row once;
/// most tightening happens in the first few (Savelsbergh 1994), and a cap keeps a slowly
/// converging chain (bounds creeping by a small amount each round) from running long.
inline constexpr int kDomainPropagationRounds = 50;

/// The PDHG heuristics (#509): the PDHG iteration cap of one projection (or relaxation) solve,
/// solved to kPdhgLoose - a projection only has to say which way the rounding should move;
/// the feasibility pump's perturbation, Fischetti, Glover & Lodi (2005) section 3 - a
/// rounding repeated from the round before flips a random number of columns in [T/2, 3T/2]
/// with T = kPumpFlips, and one repeated within the last kPumpCycleWindow rounds is restarted
/// by adding a random amount in [kPumpRestartLow, kPumpRestartHigh] to each column's distance
/// from its rounding and flipping those that pass one half; the LPs at most that complete the
/// continuous columns of one pump; and the Feasibility Jump budget, in nonzero visits, of the
/// fix-and-propagate repair.
inline constexpr Count kPdhgHeuristicIterations = 5000;
/// The share of the search's remaining time one run of either PDHG heuristic may spend, all
/// its solves included, when the search has a time limit and is not deterministic.
inline constexpr double kPdhgHeuristicTimeShare = 0.1;
inline constexpr int kPumpFlips = 20;
inline constexpr int kPumpCycleWindow = 3;
inline constexpr double kPumpRestartLow = -0.3;
inline constexpr double kPumpRestartHigh = 0.7;
inline constexpr int kPumpCompletionLimit = 10;
inline constexpr Count kFixPropRepairWork = 1'000'000;

/// PDHG infeasibility detection (#484; Applegate, Lubin & Hinder 2024): a restart difference is
/// tested only from the kPdhgDetectionMinRestarts-th restart on - the first "difference" is a
/// single projected gradient step, not the converging ray the method relies on - and only
/// when its norm exceeds kPdhgDetectionMinNorm (a converged LP's differences go to zero).
inline constexpr int kPdhgDetectionMinRestarts = 2;
inline constexpr double kPdhgDetectionMinNorm = 1e-12;

/// Mixed-precision GPU PDHG (#982, pdhg_precision=mixed, off by default): a restart period
/// run on single-precision products counts as useful only when it cuts the double-precision
/// KKT residual to at most this share of the previous restart's; otherwise the run falls
/// back to double products for the rest of the solve (src/pdhg/pdhg_refinement.hpp; Higham &
/// Mary, Acta Numerica 31 (2022), sec. 4, on stagnation detection). 0.5: half or better.
inline constexpr double kPdhgMixedMinUsefulShrink = 0.5;

// ---- Restarted Halpern Condat-Vu and the PID primal weight for QP (#493) -------------------
//
/// A restart when the fixed-point residual ||T z - z|| has fallen to this share of its value at
/// the period's first step; the sufficient-decay ratio of [PDLP] section 4.3 and of the LP
/// engine's Halpern path (#481), carried over unchanged.
inline constexpr double kQpRestartSufficientDecay = 0.2;
/// ... or when the period has lasted this share of every iteration so far, [PDLP]'s
/// artificial restart, as the LP engine uses it.
inline constexpr double kQpRestartArtificialShare = 0.36;
/// The reflection actually used, as a share of Condat's bound delta - 1 ([C13] Theorem 3.1
/// states convergence for rho strictly below delta). The bound is computed from norms already
/// inflated 5% over their power-iteration estimates; this keeps the step strictly inside it.
inline constexpr double kQpHalpernReflectionShare = 0.99;
/// The primal weight is held in this range, the LP engine's clamp.
inline constexpr double kQpPrimalWeightMin = 1e-6;
inline constexpr double kQpPrimalWeightMax = 1e6;
/// Below this primal or dual movement over a period the ratio is rounding, and the weight is
/// left alone (the LP engine's threshold).
inline constexpr double kQpPrimalWeightMinMovement = 1e-12;
/// Anti-windup on the controller's integral of log-errors: e^5 is a factor of about 150 in
/// the weight, beyond which an integral term would only be remembering a transient.
inline constexpr double kQpPidIntegralLimit = 5.0;

// ---- Dense columns in the LP interior point's normal equations (#467, ipm_dense_columns) ---

/// A column is dense when it has more than this times sqrt(rows) entries: the default of
/// `ipm_dense_column_factor`, the figure Andersen, Gondzio, Meszaros & Xu (1996, sec. 5)
/// and the issue use.
inline constexpr double kIpmDenseColumnFactor = 10.0;
/// At most this many columns go to the correction, densest first: each costs one solve with
/// the sparse factor per factorization to build the k x k Schur complement, which is then
/// factored densely.
inline constexpr Index kIpmMaxDenseColumns = 100;
/// Conjugate gradients aim at a normwise backward error (Higham 2002, sec. 7.1) of this: the
/// residual against the size of the terms M x is summed from, i.e. the rounding floor.
inline constexpr double kIpmPcgTargetBackwardError = 1e-16;
/// A solve whose backward error is at most this is converged, two decades above the target.
/// A solve above it is NOT used as a Newton direction: the interior point treats it as it
/// treats a non-finite direction (raise the regularization, refactorize, recompute).
inline constexpr double kIpmPcgAcceptedBackwardError = 1e-12;
/// Steps per solve with the Woodbury preconditioner; the sparse-factor fallback gets this
/// plus the number of dense columns.
inline constexpr int kIpmPcgMaxIterations = 50;
/// The iteration stops when the residual has not halved in this many steps. Conjugate
/// gradients minimize the M-norm of the error, not the residual, whose infinity norm can
/// rise for several steps before it falls: on israel (33 dense columns at factor 2) the
/// Woodbury-preconditioned residual went 27, 47, 41, 22, 15 and then 5e-3, and at 3 steps
/// the iteration stopped at a backward error of 1.3e-4; with no early stop every solve of
/// that run converged (worst 2.5e-15, longest run without halving 6 steps). 10 leaves room
/// over the longest run seen.
inline constexpr int kIpmPcgStagnationSteps = 10;
/// A row whose diagonal in the sparse part is below this fraction of the dense columns'
/// contribution gets the dense diagonal in the factor (see preconditioner_shift). Measured
/// on israel's normal equations: at 1e-2 CG stalled with a direction 34% off, at 1e-6 not.
inline constexpr double kIpmDenseSupportRatio = 1e-6;
/// Every eigenvalue of the scaled Schur complement I + V^T M_s^-1 V is at least 1 in exact
/// arithmetic; a Cholesky pivot below this means the solves it was built from were not
/// accurate, and the Woodbury preconditioner is not used for that factorization.
inline constexpr double kIpmDenseSchurMinPivot = 0.5;

// ---- The LP interior point's normal equations on the n x n side (#469, ipm_normal_side) ---
//
// The column side's conjugate gradients on M = A Theta A^T + D aim at and accept the same
// backward errors as the dense-column path (kIpmPcgTargetBackwardError,
// kIpmPcgAcceptedBackwardError) within the same kIpmPcgMaxIterations steps and the same
// kIpmPcgStagnationSteps without halving; an unaccepted solve is handled as a non-finite
// direction is. #469 stopped after 3 steps without halving; once an unaccepted solve was no
// longer used silently, share2b on the column side stopped three solves at backward errors
// of 1.4e-9, 6.3e-12 and 3.1e-12 after 3 to 6 steps - the non-monotone residual that
// kIpmPcgStagnationSteps was measured against on israel - and its regularization raises
// ended the solve as a numerical error at iteration 153 (row side: optimal in 43).

/// D in the preconditioner (and in N) is at least this times the row's diagonal of
/// A Theta A^T. An equality row's D is the 1e-10 regularization alone and would put 1e10
/// A_E^T A_E into N. Measured by #469 over nine Netlib instances (afiro, adlittle, sc50a,
/// sc50b, blend, share2b, scagr7, stocfor1, israel): at 1e-8 all end optimal at about two
/// conjugate-gradient steps per solve; 1e-6 takes up to three; at 1e-4 share2b runs to the
/// iteration limit; with no floor none converged.
inline constexpr double kIpmColumnSideDiagonalFloor = 1e-8;

// ---- Batched PDHG node bounds (#520, src/pdhg/batch_pdhg.hpp) -----------------------------
//
// The batch is only ever used through the Neumaier-Shcherbina bound of its duals, which is
// valid for ANY dual vector, so none of these decides whether a bound is correct - only how
// good it is for the iterations spent.

/// Iterations between the restart checks. Each check costs about three extra products per
/// LP (A x and A^T y at the average and A x at the current point), so 64 keeps them near 5%.
inline constexpr Count kBatchPdhgCheckInterval = 64;
/// Rows or columns per partial sum in the restart statistics: each is summed in index order
/// and the partials in chunk order, the same on the CPU and on the device, so the two take
/// the same restart decisions and return the same duals bit for bit.
inline constexpr Index kBatchPdhgChunk = 256;
/// Step eta = kBatchPdhgStepFraction / ||A||_2 (power-iteration estimate, rounded up), so
/// tau * sigma * ||A||^2 < 1 as Chambolle & Pock (2011) Theorem 1 requires.
inline constexpr double kBatchPdhgStepFraction = 0.9;
/// Restart criteria of Applegate et al. (2021) sec. 3.2 on the KKT error, as cuPDLP.jl (Lu &
/// Yang 2023) states them: sufficient decay, necessary decay with no local progress, and an
/// artificial restart once the epoch is this share of all iterations so far.
inline constexpr double kBatchPdhgRestartSufficient = 0.2;
inline constexpr double kBatchPdhgRestartNecessary = 0.8;
inline constexpr double kBatchPdhgRestartArtificial = 0.36;
/// Primal-weight smoothing theta of Applegate et al. (2021) sec. 3.3.
inline constexpr double kBatchPdhgWeightSmoothing = 0.5;
/// A restart moves the primal weight only when both iterates moved by more than this.
inline constexpr double kBatchPdhgMinMove = 1e-10;

// ---- The interior-point filter line-search method for general NLP (NLP stage 2) -----------
//
// Every value is the default of Wachter & Biegler, "On the implementation of an interior-
// point filter line-search algorithm for large-scale nonlinear programming", Math.
// Programming 106(1) (2006), named by the paper's symbol and section. None of them decides
// whether an answer is accepted: that is the KKT check at the project tolerances above
// (kPrimalFeasibility, kDualFeasibility, kComplementarity), made at the returned point.

/// mu_0, the initial barrier parameter (sec. 3.1).
inline constexpr double kNlpMuInit = 0.1;
/// kappa_epsilon: the barrier subproblem is solved to kappa_epsilon * mu (eq. 7).
inline constexpr double kNlpKappaEpsilon = 10.0;
/// kappa_mu and theta_mu: mu+ = max(eps/10, min(kappa_mu mu, mu^theta_mu)) (eq. 7).
inline constexpr double kNlpKappaMu = 0.2;
inline constexpr double kNlpThetaMu = 1.5;
/// tau_min: the fraction-to-the-boundary parameter is max(tau_min, 1 - mu) (eq. 8).
inline constexpr double kNlpTauMin = 0.99;
/// kappa_Sigma: the bound multipliers are kept within [mu / (kappa s), kappa mu / s] (eq. 16).
inline constexpr double kNlpKappaSigma = 1e10;
/// s_max: the scaling threshold of the optimality error (eq. 5).
inline constexpr double kNlpScalingMax = 100.0;
/// The filter's margins gamma_theta, gamma_phi (eq. 18), the switching condition's delta,
/// s_theta, s_phi (eq. 19), the Armijo constant eta_phi (eq. 20) and gamma_alpha, the safety
/// factor of the smallest step (eq. 23), all sec. 2.3 / 3.
inline constexpr double kNlpGammaTheta = 1e-5;
inline constexpr double kNlpGammaPhi = 1e-8;
inline constexpr double kNlpSwitchDelta = 1.0;
inline constexpr double kNlpSwitchSTheta = 1.1;
inline constexpr double kNlpSwitchSPhi = 2.3;
inline constexpr double kNlpEtaPhi = 1e-8;
inline constexpr double kNlpGammaAlpha = 0.05;
/// theta_max = kNlpThetaMaxFactor * max(1, theta(x_0)) and theta_min likewise (sec. 3.2).
inline constexpr double kNlpThetaMaxFactor = 1e4;
inline constexpr double kNlpThetaMinFactor = 1e-4;
/// Pushing the starting point into the bounds' interior: kappa_1, kappa_2 (sec. 3.6).
inline constexpr double kNlpBoundPush = 1e-2;
inline constexpr double kNlpBoundFraction = 1e-2;
/// lambda_max: a least-squares multiplier estimate larger than this is discarded (sec. 3.6).
inline constexpr double kNlpLambdaMax = 1e3;
/// Inertia correction, Algorithm IC (sec. 3.1): delta_w^min, delta_w^0, delta_w^max, the
/// growth factors kappa_w^+ (after a first success), kappa_w^+-bar (the first time) and the
/// decrease kappa_w^-, and delta_c-bar mu^kappa_c for a singular Jacobian.
inline constexpr double kNlpDeltaWMin = 1e-20;
inline constexpr double kNlpDeltaWInit = 1e-4;
inline constexpr double kNlpDeltaWMax = 1e40;
inline constexpr double kNlpKappaWPlus = 8.0;
inline constexpr double kNlpKappaWPlusFirst = 100.0;
inline constexpr double kNlpKappaWMinus = 1.0 / 3.0;
inline constexpr double kNlpDeltaC = 1e-8;
inline constexpr double kNlpKappaC = 0.25;
/// The feasibility restoration phase ends once theta <= kappa_resto theta(x_R) and the point
/// is acceptable to the filter (sec. 3.3); rho weighs the violation in its objective.
inline constexpr double kNlpKappaResto = 0.9;
inline constexpr double kNlpRestorationRho = 1000.0;
/// The backtracking line search halves the step: alpha_{k,l} = 2^-l alpha_max (sec. 2.3).
inline constexpr double kNlpBacktrack = 0.5;
/// A step no larger than this many machine epsilons relative to every entry is taken whole,
/// without the line search, which could only reject it for rounding. A project safeguard,
/// not a parameter of the paper.
inline constexpr double kNlpTinyStep = 10.0 * std::numeric_limits<double>::epsilon();

// Not from the paper - this project's choices, argued where they are used (src/nlp/).

/// A pivot of |d| below this, or of the wrong sign, makes the quasidefinite LDL^T report the
/// factorization inexact, so its inertia is NOT certified and Algorithm IC regularizes further.
/// The project's pivot tolerance (kZeroDrop).
inline constexpr double kNlpPivotFloor = kZeroDrop;
/// The (2,2) block always carries at least -kNlpDualRegularization, so the matrix is
/// quasidefinite and needs no numerical pivoting (Vanderbei 1995); the unregularized system
/// is then recovered by iterative refinement.
inline constexpr double kNlpDualRegularization = 1e-10;
/// Iterative refinement steps on the unregularized KKT system, and the relative residual that
/// ends it.
inline constexpr int kNlpRefinementSteps = 10;
inline constexpr double kNlpRefinementTolerance = 1e-12;
/// The default iteration budget of one NLP solve when iteration_limit is not set. A project
/// choice: a safety cap well above what the Hock-Schittkowski tests need, not a tuned value.
inline constexpr Count kNlpMaxIterations = 3000;
/// A restoration phase that has not ended in this many of its own iterations has failed.
inline constexpr Count kNlpMaxRestorationIterations = 500;
/// NLP-based branch and bound (NLP stage 3): a node is certified EMPTY when the least total
/// violation sum(p + n) over its box, solved to a global optimum of the convex minimum-
/// violation problem, exceeds this. Ten times kPrimalFeasibility's worth per unit, so that a
/// feasible node whose minimum sits at the solver's tolerance is never declared empty.
inline constexpr double kMinlpEmptyViolation = 1e-6;
/// Benders decomposition (#525): the master's bound and the value of the blocks at its point
/// agree - and the decomposition has converged - when they differ by no more than this,
/// relative to max(1, |value|). An LP is solved to about 1e-9 relative, and the two numbers are
/// sums of several such solves, so the figure sits a decade above that; the answer is then
/// measured by solve() against kPrimalFeasibility and kDualFeasibility like any other, which is
/// the test that decides whether it is believed.
inline constexpr double kBendersGap = 1e-8;
/// A block's cut is generated only when the master's epigraph value falls short of the block's
/// value by more than this, relative to max(1, |value|): a decade inside kBendersGap, so that a
/// round which is not converged always has a cut to add.
/// A shortfall below it would give a cut the master cannot tell from rounding.
inline constexpr double kBendersCutViolation = 1e-9;
/// A block that the LP engine reports infeasible must also show a violation in its elastic LP
/// above this; below it the verdicts disagree, which is a numerical failure and not a cut.
inline constexpr double kBendersFeasibilityCut = 1e-9;
/// Dantzig-Wolfe decomposition (#1048): the restricted master's value and the Lagrangian bound
/// (that value plus every block's negative reduced cost) agree - and column generation has
/// converged - when they differ by no more than this, relative to max(1, |value|). The same
/// target as kBendersGap, for the same reason: the answer is then measured by solve() against
/// kPrimalFeasibility and kDualFeasibility, which decides whether it is believed.
inline constexpr double kDantzigWolfeGap = 1e-8;
/// A block's column enters the restricted master only when its reduced cost is below minus
/// this, relative to max(1, |master value|): a decade inside kDantzigWolfeGap, so that a round
/// which has not converged always has a column to add, and no column enters whose reduced cost
/// the master could not tell from rounding.
inline constexpr double kDantzigWolfeReducedCost = 1e-9;
/// Outer approximation (#528): a point of the master is cut off by the linearization of a
/// nonlinear row only when the row is violated there by more than this, relative to
/// max(1, |bound|). Ten times kPrimalFeasibility, the MILP master's own feasibility tolerance,
/// so that a cut generated at the master's point always separates it by more than the master
/// can mistake for rounding; below it the point is feasible for the nonlinear row as far as
/// any engine here can tell.
inline constexpr double kMinlpOaViolation = 1e-6;
/// Outer approximation (#528): the master's bound on the objective's epigraph variable is the
/// continuous relaxation's value less this fraction of max(1, |value|) - the relaxation is
/// solved to nlp_tolerance (1e-6, relative), so its value is a lower bound only to that
/// accuracy, and the bound must never cut off the true optimum.
inline constexpr double kMinlpOaEpigraphSlack = 1e-6;

/// Scenario runs (#752): a scenario's objective and that of a separate single solve of the
/// same edited model must agree to this relative difference, measured against max(1, |obj|).
/// The acceptance figure of #752; both answers already pass the KKT check at
/// kPrimalFeasibility and kDualFeasibility, so this is the bar on their agreement.
inline constexpr double kScenarioAgreement = 1e-8;

// ---- The interior point's device factor under ipm_linear_solver = auto (#489) -------------

/// Under `ipm_linear_solver = auto` in a cuDSS build with a device, the normal equations go to
/// the device when they hold at least this many nonzeros, or when the CPU ordering says their
/// factor would; below it they stay on the CPU factor. Measured with `algorithm=ipm
/// crossover=false` on an A100 (#489): the Netlib factors at or above it are dfl001 (1.7e6),
/// maros-r7 (1.3e6) and fit2p (4.5e6), optimal on both factors in 52 s, 11.5 s and 144 s on
/// the CPU against 4.0 s, 2.8 s and 6.0 s on the device
/// (bench/results/ipm-cudss-ab-netlib-{cpu,cudss}-58a8374.csv); every other Netlib factor is
/// below 5e5, where forcing the device changed five statuses (agg, greenbeb, lotfi, ship08l,
/// tuff, all factors under 1e5); the one large saving left below it is pilot87 (factor 4.5e5,
/// 39 s against 6.9 s), feasible on both. Of the Mittelmann eight, qap15 (factor 9.2e6) and
/// rmine15 (7.8e6 assembled, its CPU ordering unfinished inside the set-up share) are over
/// it. A system this large skips the CPU ordering: the device orders it.
inline constexpr std::int64_t kIpmDeviceFactorFloor = 1'000'000;
/// Under `ipm_linear_solver = auto` the device is not started with less than this many
/// seconds left of the time limit: creating the cuDSS handle and running its analysis do not
/// consult the deadline, and the CPU ordering does. Measured on the A100: a 3,000-row system
/// whose normal equations are dense (4.5e6 nonzeros) returned time_limit 0.4 s later with the
/// device than on the CPU factor under a 0.5 s limit (InteriorPointTimeLimit.
/// ADenseColumnDoesNotCarryTheSolvePastItsTimeLimit, 0.9 s against 0.5 s), and 1.5 s later
/// under a loaded ctest; two seconds covers both with room.
inline constexpr double kIpmDeviceStartSeconds = 2.0;
/// Under `ipm_linear_solver = auto`, normal equations holding this many nonzeros or more
/// are not handed to the device before the CPU ordering has sized their factor. cuDSS's
/// analysis does not consult the deadline, and on the three Mittelmann systems this large
/// it ran past the set-up share to a factor no budget admits: chromaticindex1024-7 (3.0e8
/// nonzeros, factor 1.0e9, declined at 78 s of a 60 s share), supportcase10 (2.2e8, 7.8e9,
/// 86 s) and Linf_520c (4.7e8, 1.0e9, 158 s), where the CPU ordering declines at the share
/// (bench runs in #489). The largest system the device solved was rmine15's, 7.8e6.
inline constexpr double kIpmDeviceSystemCeiling = 1e8;

// ---- The conic interior point: second-order cones and convex quadratic rows (#755) ---------

/// Convergence of the homogeneous self-dual embedding to an optimum, measured on the point
/// x/tau, y/tau, s/tau: relative primal residual, relative dual residual and relative gap each
/// at or below this. The LP embedding's kIpmHsdTolerance, so the two converge alike.
inline constexpr double kConicTolerance = 1e-8;
/// Past kConicTolerance the interior point keeps going (#1146): an optimum's value converges
/// with the gap but its point and multipliers only with about the gap's square root on a
/// second-order cone, and at 1e-8 the model's multipliers missed kDualFeasibility on 12 of 13
/// QPLIB convex QCQPs. It aims for kConicTargetTolerance, at most kConicRefineIterations
/// iterations past the first one that met kConicTolerance, and returns the latest iterate
/// that met kConicTolerance when it stalls or a limit ends it - never a worse one.
inline constexpr double kConicTargetTolerance = 1e-12;
inline constexpr int kConicRefineIterations = 10;
/// A cone's point is inside it when its margin exceeds this relative to its own scale, and
/// at the apex when every member is within it of zero; in between it is on the boundary.
/// The INTERIOR_SHARE of tools/verify_solution_conic.py, so the gate splits a cone's
/// reduced costs as the verifier does (#1146).
inline constexpr double kConeInteriorShare = 1e-6;
/// A certificate (a dual ray y, s with A'y + s = 0, s in the dual cone, b'y = 1; or a primal
/// ray x with A x = 0, x in the cone, c'x = -1) is accepted when its defining equation holds
/// to this, relative to the terms it is made of.
inline constexpr double kConicCertificateTolerance = 1e-8;
/// The step keeps this fraction of the distance to the cone's boundary (Mehrotra 1992).
inline constexpr double kConicFractionToBoundary = 0.99;
/// The iteration stops without a verdict once mu has fallen this far below its starting
/// value while tau and kappa still have not decided, as kIpmHsdMuFloor does for the LP.
inline constexpr double kConicMuFloor = 1e-15;
/// The primal (rho, on the cone block) and dual (delta, on the rows) regularization that
/// keeps the augmented system quasi-definite with free columns in it (Altman & Gondzio 1999);
/// iterative refinement against the unregularized matrix takes it back out of the step.
inline constexpr double kConicRegularization = 1e-9;
/// Iterations after which the embedding is taken not to converge.
inline constexpr int kConicIterationCeiling = 200;
/// A convex quadratic row's matrix Q_r, and the objective's Hessian, are factored as F F'
/// by a dense pivoted Cholesky on their support (Higham 1990), which needs the support's
/// square in memory; a support wider than this is refused rather than factored.
inline constexpr Index kConicDenseFactorLimit = 4000;
/// The pivoted Cholesky stops at a pivot below this share of the largest diagonal: the rest
/// of the matrix is rounding of a rank-deficient but positive semidefinite Q (the convexity
/// test already accepted it at kCurvatureTolerance, 1e-12, of the same scale).
inline constexpr double kConicCholeskyPivotShare = 1e-12;
/// Dual recovery at the conic optimum (conic_duals.cpp): a row or a column bound is taken as
/// active when its slack is at most this share of max(1, the magnitude of its terms). The
/// interior point's point is right to about kConicTolerance relative, so a window a hundred
/// times wider identifies every tight row while staying far below any slack that is genuinely
/// open; a misidentified row only costs the recovery, which is kept when it measures better.
inline constexpr double kConicActiveShare = 1e-6;
/// The Tikhonov term of the recovery's normal equations (M'M + eps I), relative to the largest
/// diagonal of M'M, which keeps them positive definite when the active gradients are linearly
/// dependent (a degenerate optimum).
inline constexpr double kConicRecoveryRegularization = 1e-12;
/// Refinement passes of the recovery's normal equations on the residual r - M lambda; each
/// gains roughly the factor the squared condition number cost.
inline constexpr int kConicRecoveryRefinement = 3;
/// Newton's method on the active set's KKT system (conic_duals.cpp): at most this many steps,
/// stopping when the scaled residual is at or below kConicPolishResidual or stops falling.
/// Convergence from the interior point's exit is quadratic, so a handful of steps is all a
/// correct active set ever needs; the cap bounds what a wrong one can cost.
inline constexpr int kConicPolishIterations = 8;
inline constexpr double kConicPolishResidual = 1e-14;
/// The regularization of that KKT system, [H_L + rho I, -M; -M', -rho I], relative to the
/// largest gradient entry; refinement against the unregularized matrix takes it back out.
inline constexpr double kConicPolishRegularization = 1e-10;

}  // namespace sankhya::tol
