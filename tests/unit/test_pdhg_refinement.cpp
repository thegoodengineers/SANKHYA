// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for the mixed-precision refinement control (#982).
//
// Pure arithmetic: no CUDA device or GPU needed. See src/pdhg/pdhg_refinement.hpp for the
// citations behind the decision rule exercised here.

#include <gtest/gtest.h>

#include "pdhg/pdhg_refinement.hpp"
#include "sankhya/options.hpp"

namespace sankhya::pdhg {
namespace {

TEST(PdhgRefinement, AlreadyAtTargetIsDone) {
  RefinementState state;
  state.rounds_run = 0;
  state.current_residual = 1e-9;
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kDone);
}

TEST(PdhgRefinement, FirstRoundAboveTargetAlwaysRefines) {
  RefinementState state;
  state.rounds_run = 0;
  state.current_residual = 1e-4;
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kRefineAgain);
}

TEST(PdhgRefinement, FirstRoundRespectsZeroRoundBudget) {
  RefinementState state;
  state.rounds_run = 0;
  state.current_residual = 1e-4;
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 0), RefinementAction::kRoundLimitReached);
}

TEST(PdhgRefinement, GoodShrinkKeepsRefining) {
  RefinementState state;
  state.rounds_run = 1;
  state.previous_residual = 1e-4;
  state.current_residual = 1e-6;  // shrink = 1e-2, well under tol::kPdhgMixedMinUsefulShrink
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kRefineAgain);
}

TEST(PdhgRefinement, StalledRoundFallsBackToDouble) {
  RefinementState state;
  state.rounds_run = 1;
  state.previous_residual = 1e-4;
  state.current_residual =
      0.9e-4;  // shrink = 0.9 > tol::kPdhgMixedMinUsefulShrink: not worth another round
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kFallBackToDouble);
}

TEST(PdhgRefinement, WorseningRoundFallsBackToDouble) {
  RefinementState state;
  state.rounds_run = 1;
  state.previous_residual = 1e-6;
  state.current_residual = 2e-6;  // got worse
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kFallBackToDouble);
}

TEST(PdhgRefinement, RoundLimitReachedWithoutStalling) {
  RefinementState state;
  state.rounds_run = 10;
  state.previous_residual = 1e-4;
  state.current_residual = 1e-6;  // still shrinking well, but the budget is spent
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kRoundLimitReached);
}

TEST(PdhgRefinement, ExactThresholdShrinkCountsAsNotUseful) {
  RefinementState state;
  state.rounds_run = 1;
  state.previous_residual = 1e-4;
  state.current_residual =
      0.5e-4;  // shrink == tol::kPdhgMixedMinUsefulShrink exactly: boundary, not `<=`
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kRefineAgain);
}

TEST(PdhgRefinement, MissingPreviousResidualFallsBackSafely) {
  RefinementState state;
  state.rounds_run = 3;
  state.previous_residual = 0.0;  // should not happen once rounds_run > 0, but must not divide
                                  // by zero or crash
  state.current_residual = 1e-6;
  EXPECT_EQ(decide_refinement_action(state, 1e-8, 10), RefinementAction::kFallBackToDouble);
}

// The default is double precision: #982's mixed path is off unless a caller opts in, so no
// default solve, benchmark or reported number goes near a float.
TEST(PdhgRefinement, PrecisionDefaultsToDouble) {
  EXPECT_EQ(Options().get_string("pdhg_precision"), "double");
}

}  // namespace
}  // namespace sankhya::pdhg
