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
inline constexpr double kDualFeasibility = 1e-7;

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
/// search cycling between two values of one column.
inline constexpr int kFeasibilityJumpSample = 25;
inline constexpr Count kFeasibilityJumpWork = 10'000'000;
inline constexpr double kFeasibilityJumpMinScore = 1e-9;

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

/// Complementary slackness, ABSOLUTE: the largest |multiplier| * slack over every row and
/// column, in the model's own units, that an optimality claim may carry. This is the test
/// tools/verify_solution.py applies (its "complementary slackness" check), copied here so the
/// status guard in solve() cannot let out a claim the verifier will reject. The guard's other
/// dual measure is relative to each term's scale; this one is deliberately not, because the
/// verifier's is not (#52: the verifier is not loosened). Found by #209: the interior point's
/// duals can be purified to 1e-11 while a row priced at 3.3e+03 still sits 3e-10 inside its
/// bound, a product of 1.1e-06 that only crossover (#219) removes.
inline constexpr double kComplementarity = 1e-6;

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

/// Markowitz threshold for sparse LU pivoting (Phase 6). A candidate pivot must be at
/// least this fraction of the largest magnitude in its column to be numerically eligible.
/// 0.01 is the standard simplex compromise between sparsity and stability (Suhl & Suhl).
inline constexpr double kMarkowitzThreshold = 0.01;

/// Hyper-sparse FTRAN and BTRAN (#464, lu_hyper_sparse): the symbolic reach is tried when
/// the right-hand side has at most this fraction of m nonzeros, and abandoned for the full
/// loops once the steps it reaches pass the same fraction. Hall & McKinnon, "Hyper-sparsity
/// in the revised simplex method and how to exploit it", Comput. Optim. Appl. 32 (2005),
/// call a result hyper-sparse below 10% and find the full loops cheaper above it; the
/// density bins the LU logs (LuSolveStats) are there to re-set it from our own data.
inline constexpr double kHyperSparseDensity = 0.10;

/// Below this, a computed pivot element is treated as a singular basis rather than a pivot.
inline constexpr double kPivotTolerance = 1e-9;

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
inline constexpr double kPdhgTight = 1e-8;

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
/// After a non-finite Newton direction (#209) the rho floor rises by this factor per raise,
/// capped at kIpmProximalRecoveryCap. The default path's x1e4 on its 1e-10 dual
/// regularization, applied to the 1e-8 rho floor, took rho to 1 after two raises: a proximal
/// term as large as the matrix, whose refinement then has nothing to converge to.
inline constexpr double kIpmProximalRecoveryRaise = 100.0;
inline constexpr double kIpmProximalRecoveryCap = 1e-4;

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
/// And the loop stops once the solve has used this share of time_limit, so a root whose
/// every round is slow leaves the tree most of the time it was given.
inline constexpr double kRootCutTimeShare = 0.2;
/// And the loop adds at most max(kRootCutRowFloor, kRootCutRowShare * m) cut rows in all,
/// round 1 included, m the rows of the LP the root separates on before its first cut.
inline constexpr int kRootCutRowFloor = 100;
inline constexpr double kRootCutRowShare = 1.0;

/// GPU PDHG device loop (#478): iterations replayed on the device per host synchronisation.
/// Larger amortises the synchronisation further but checks the iteration and time limits
/// and the evaluation stride less often; 32 is under the evaluation interval of 40.
inline constexpr Count kPdhgDeviceLoopBlock = 32;

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

/// PDHG infeasibility detection (#484; Applegate, Lubin & Hinder 2024): a restart difference is
/// tested only from the kPdhgDetectionMinRestarts-th restart on - the first "difference" is a
/// single projected gradient step, not the converging ray the method relies on - and only
/// when its norm exceeds kPdhgDetectionMinNorm (a converged LP's differences go to zero).
inline constexpr int kPdhgDetectionMinRestarts = 2;
inline constexpr double kPdhgDetectionMinNorm = 1e-12;

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

}  // namespace sankhya::tol
