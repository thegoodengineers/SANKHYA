// SPDX-License-Identifier: Apache-2.0
// SANKHYA - iterative-refinement control for mixed-precision GPU PDHG (#982).
//
// Pure host-side arithmetic: no CUDA, no device state. This is the decision logic that the
// GPU engine (src/gpu/pdhg_gpu.cu, `pdhg_precision = mixed`) is meant to drive its refinement
// rounds with, split out so it can be exercised in a unit test without a GPU (#982 point 4,
// "falling back to double iterations when a round stops gaining").
//
// References:
//   [CH18]  Carson & Higham, "Accelerating the solution of linear systems by iterative
//           refinement in three precisions", SIAM J. Sci. Comput. 40 (2018). The round
//           structure here is theirs: compute a residual in the working (higher) precision,
//           solve for a correction in the lower precision, apply it, and stop refining once
//           the residual stops shrinking by a useful factor rather than count a fixed number
//           of rounds.
//   [GSW16] Gleixner, Steffy & Wolter, "Iterative refinement for linear programming",
//           INFORMS J. Comput. 28 (2016). The LP-specific point taken here: refinement is
//           judged on the SAME double-precision KKT measure the caller already reports
//           against (src/pdhg/pdhg_evaluate.hpp), never on a measure private to the
//           refinement loop, so a mixed run cannot claim a tolerance the double engine would
//           not also accept on that residual.
//   [HM22]  Higham & Mary, "Mixed precision algorithms in numerical linear algebra",
//           Acta Numerica 31 (2022) - survey; sec. 4 on stagnation detection is the basis for
//           `tol::kPdhgMixedMinUsefulShrink`.
#pragma once

#include <cstdint>

#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {

/// What the caller should do next, decided from the double-precision residual trend alone.
enum class RefinementAction {
  /// The residual is already at or below the target; stop.
  kDone,
  /// Run another refinement round: compute the residual in double from the current single
  /// precision iterate, solve the correction in single, apply it.
  kRefineAgain,
  /// The last round did not shrink the residual by a useful factor (or made it worse): single
  /// precision has stalled. Fall back to running further iterations fully in double
  /// ([CH18] sec. 2.3, the three-precision scheme's fallback when the inner solver cannot
  /// reduce the residual further at its precision).
  kFallBackToDouble,
  /// A round limit was hit without reaching the target and without stalling; the caller
  /// should treat this exactly as the double engine treats an iteration limit, never as a
  /// looser status ([GSW16]; issue #982 point 5, "the same gate at the end").
  kRoundLimitReached,
};

/// A round's residual history, double precision throughout, as the double-precision KKT
/// measure the rest of the engine already computes (`pdhg::Residuals::worst()`).
struct RefinementState {
  double previous_residual = 0.0;  ///< residual before the most recent round (0 = no round yet)
  double current_residual = 0.0;   ///< residual after the most recent round
  std::int64_t rounds_run = 0;
};

/// Decide the next action from the residual trend alone. `target` is the caller's requested
/// relative tolerance (the same `pdhg_tolerance` the double engine is held to); `max_rounds`
/// bounds the refinement loop so a stalled-but-not-yet-detected case cannot run forever.
[[nodiscard]] inline RefinementAction decide_refinement_action(const RefinementState& state,
                                                               double target,
                                                               std::int64_t max_rounds) {
  if (state.current_residual <= target) return RefinementAction::kDone;
  if (state.rounds_run == 0) {
    // First round has not been judged yet; the caller always gets at least one attempt
    // before refinement can be declared stalled.
    return state.rounds_run < max_rounds ? RefinementAction::kRefineAgain
                                         : RefinementAction::kRoundLimitReached;
  }
  if (state.previous_residual <= 0.0) {
    // No earlier residual to compare against (shouldn't normally happen once rounds_run > 0,
    // but fail toward the double fallback rather than divide by zero).
    return RefinementAction::kFallBackToDouble;
  }
  const double shrink = state.current_residual / state.previous_residual;
  if (shrink > tol::kPdhgMixedMinUsefulShrink) {
    // The round did not buy enough: [CH18]'s stalling condition.
    return RefinementAction::kFallBackToDouble;
  }
  if (state.rounds_run >= max_rounds) return RefinementAction::kRoundLimitReached;
  return RefinementAction::kRefineAgain;
}

}  // namespace sankhya::pdhg
