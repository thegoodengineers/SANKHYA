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

/// Minimum root-LP violation for a cut to be accepted. Valid cuts that are not violated
/// or barely violated are safely rejected to save LP solves.
inline constexpr double kCutViolationTolerance = 1e-5;

/// Presolve, parallel rows (#412; Andersen & Andersen 1995, sec. 5). Two rows are
/// proportional when every coefficient matches its scaled counterpart to this relative
/// tolerance, measured against max(1, |a|, |scaled|). Tighter than the pivot drop because a
/// merge that is even slightly wrong changes the feasible set silently; looser than
/// machine epsilon because the coefficients arrive through a reader and a scaling pass.
inline constexpr double kPresolveParallelRowTolerance = 1e-9;

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

}  // namespace sankhya::tol
