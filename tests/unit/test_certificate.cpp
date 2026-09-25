// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the two certificates, and the proof that they are proofs (#191).
//
// THE BUG THIS FILE EXISTS BECAUSE OF. `infeasible` and `unbounded` used to be written to a
// .sol file as a full all-zero point, indistinguishable from a claimed solution. The
// project's own independent checker read that point, found it violated the rows, and printed
// REJECTED - at a correct answer. A checker that calls our correct verdicts wrong is worse
// than no checker, because it is the one artefact a judge is most likely to trust.
//
// So each verdict now carries a proof, and the controlling tests here are the NEGATIVE
// controls: a certificate that does not prove the claim must be rejected. A validity check
// that has never failed is not evidence that it can fail, which is the same discipline
// test_cuts.cpp is built on.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

Options quiet(bool presolve) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  return options;
}

/// x >= 5 and x <= 2. The contradiction needs BOTH rows, which is why it is the right
/// smallest case: a one-row certificate cannot express it.
Model contradictory_pair() {
  return make_lp({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 2.0}, {1.0}, {0.0},
                 {kInfinity});
}

// =========================================================================================
// Farkas: what a proof of infeasibility is, and what it is not
// =========================================================================================

TEST(Certificate, AggregatingTheRowsProvesTheContradiction) {
  const Model model = contradictory_pair();
  // Add the first row and subtract the second: x - x = 0 on the left, 5 - 2 = 3 on the
  // right. Every feasible point would have to satisfy 0 >= 3.
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, {1.0, -1.0}, &why)) << why;
  EXPECT_NE(why.find("at least 3"), std::string::npos) << why;
}

TEST(Certificate, ScalingTheMultipliersDoesNotChangeTheProof) {
  // A Farkas vector is a ray: any positive multiple of it proves the same thing. If the
  // check were sensitive to the scale it would be testing the engine's normalisation rather
  // than the mathematics.
  const Model model = contradictory_pair();
  EXPECT_TRUE(farkas_proves_infeasible(model, {1e6, -1e6}));
  EXPECT_TRUE(farkas_proves_infeasible(model, {1e-6, -1e-6}));
}

TEST(Certificate, TheOppositeSignIsNotAProof) {
  // Negating a Farkas vector aggregates the rows the other way round, which yields a true
  // but useless inequality. This is the control on solve()'s try-both-signs step: if both
  // signs passed, that step would be picking arbitrarily rather than deciding.
  const Model model = contradictory_pair();
  EXPECT_FALSE(farkas_proves_infeasible(model, {-1.0, 1.0}));
}

TEST(Certificate, AMultiplierMayNotLeanOnABoundTheRowDoesNotHave) {
  // Row 0 is `x >= 5` with no upper bound. A negative multiplier on it claims to use an
  // upper bound of infinity, which would let any conclusion be drawn.
  const Model model = contradictory_pair();
  std::string why;
  EXPECT_FALSE(farkas_proves_infeasible(model, {-1.0, -1.0}, &why));
  EXPECT_NE(why.find("does not have"), std::string::npos) << why;
}

TEST(Certificate, AnAggregateOverAFreeColumnProvesNothing) {
  // x free, one row `x >= 5`. The aggregate `x >= 5` is perfectly satisfiable, and the
  // column bounds put no ceiling on x, so there is nothing to contradict. A checker that
  // silently treated the missing bound as zero would call this a proof.
  const Model model = make_lp({{1.0}}, {5.0}, {kInfinity}, {1.0}, {-kInfinity}, {kInfinity});
  std::string why;
  EXPECT_FALSE(farkas_proves_infeasible(model, {1.0}, &why));
  EXPECT_NE(why.find("free upward"), std::string::npos) << why;
}

TEST(Certificate, AllZeroMultipliersAreNotAProof) {
  const Model model = contradictory_pair();
  std::string why;
  EXPECT_FALSE(farkas_proves_infeasible(model, {0.0, 0.0}, &why));
  EXPECT_NE(why.find("every multiplier is zero"), std::string::npos) << why;
}

TEST(Certificate, AMarginalContradictionIsNotAProof) {
  // x >= 5 and x <= 5 - 1e-12. The two rows do contradict each other in exact arithmetic,
  // by a margin far below the feasibility tolerance. Calling that a proof of infeasibility
  // would tell a planner their model has no solution because of the twelfth decimal place.
  const Model model = make_lp({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 5.0 - 1e-12},
                              {1.0}, {0.0}, {kInfinity});
  std::string why;
  EXPECT_FALSE(farkas_proves_infeasible(model, {1.0, -1.0}, &why));
  EXPECT_NE(why.find("not a contradiction beyond rounding"), std::string::npos) << why;
}

// =========================================================================================
// Rays: what a proof of unboundedness is
// =========================================================================================

/// min -x subject to x >= 1, x free above.
Model open_below() {
  return make_lp({{1.0}}, {1.0}, {kInfinity}, {-1.0}, {-kInfinity}, {kInfinity});
}

TEST(Certificate, ADirectionNoBoundBlocksProvesUnboundedness) {
  std::string why;
  EXPECT_TRUE(ray_proves_unbounded(open_below(), {1.0}, &why)) << why;
  EXPECT_NE(why.find("improves"), std::string::npos) << why;
}

TEST(Certificate, ARayThatWorsensTheObjectiveIsNotAProof) {
  std::string why;
  EXPECT_FALSE(ray_proves_unbounded(open_below(), {-1.0}, &why));
}

TEST(Certificate, ARayABoundBlocksIsNotAProof) {
  // The same objective, but x is capped at 10. The direction improves the objective and is
  // still not a ray of this feasible region.
  const Model model = make_lp({{1.0}}, {1.0}, {kInfinity}, {-1.0}, {-kInfinity}, {10.0});
  std::string why;
  EXPECT_FALSE(ray_proves_unbounded(model, {1.0}, &why));
  EXPECT_NE(why.find("upper bound"), std::string::npos) << why;
}

TEST(Certificate, ARayARowBlocksIsNotAProof) {
  // x + y <= 10 blocks the direction (1, 1) even though both columns are free above.
  const Model model = make_lp({{1.0, 1.0}}, {-kInfinity}, {10.0}, {-1.0, -1.0},
                              {-kInfinity, -kInfinity}, {kInfinity, kInfinity});
  std::string why;
  EXPECT_FALSE(ray_proves_unbounded(model, {1.0, 1.0}, &why));
  EXPECT_NE(why.find("row"), std::string::npos) << why;
}

TEST(Certificate, TheZeroDirectionIsNotARay) {
  std::string why;
  EXPECT_FALSE(ray_proves_unbounded(open_below(), {0.0}, &why));
  EXPECT_NE(why.find("not a direction"), std::string::npos) << why;
}

// =========================================================================================
// End to end: what the solver actually hands over
// =========================================================================================

TEST(Certificate, TheSolverProvesTheInfeasibilityItReports) {
  // Presolve off, so the simplex reaches the conclusion and can hand over the dual ray it
  // used to get there. This is the path the .sol file's `certificate farkas` comes from.
  const Model model = contradictory_pair();
  const Solution solution = solve(model, quiet(/*presolve=*/false));
  ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  ASSERT_FALSE(solution.farkas_dual.empty())
      << "the simplex concluded infeasible but produced no certificate: " << solution.message;
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual, &why)) << why;
  EXPECT_NE(solution.message.find("proof:"), std::string::npos) << solution.message;
}

TEST(Certificate, TheSolverProvesTheUnboundednessItReports) {
  const Model model = open_below();
  const Solution solution = solve(model, quiet(/*presolve=*/false));
  ASSERT_EQ(solution.status, SolveStatus::kUnbounded) << solution.message;
  ASSERT_FALSE(solution.primal_ray.empty())
      << "the simplex concluded unbounded but produced no ray: " << solution.message;
  std::string why;
  EXPECT_TRUE(ray_proves_unbounded(model, solution.primal_ray, &why)) << why;

  // The other half of the claim: the ray has to start somewhere the model allows, so an
  // unbounded verdict now carries a feasible point as well.
  ASSERT_EQ(static_cast<Index>(solution.col_value.size()), model.num_cols());
  EXPECT_LE(solution.primal_infeasibility, tol::kPrimalFeasibility) << solution.message;
}

TEST(Certificate, ACertificateThatSurvivesPresolveStillProvesTheOriginalModel) {
  // Presolve rewrites the model the engine sees, so a certificate for the reduced model is
  // in the wrong numbering and may be in the wrong space entirely. Whatever comes back must
  // prove the ORIGINAL model or must not be there at all - there is no third option that is
  // honest.
  const Model model = open_below();
  const Solution solution = solve(model, quiet(/*presolve=*/true));
  ASSERT_EQ(solution.status, SolveStatus::kUnbounded) << solution.message;
  if (!solution.primal_ray.empty()) {
    std::string why;
    EXPECT_TRUE(ray_proves_unbounded(model, solution.primal_ray, &why)) << why;
  }
}

TEST(Certificate, AnUnprovableCertificateIsDroppedRatherThanPublished) {
  // Presolve proves this one by bound arithmetic and keeps no Farkas vector, so the verdict
  // arrives with no certificate and says so. The failure this pins is the opposite one: a
  // certificate field that is populated but does not prove the claim.
  const Model model = contradictory_pair();
  const Solution solution = solve(model, quiet(/*presolve=*/true));
  ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  if (!solution.farkas_dual.empty()) {
    std::string why;
    EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual, &why))
        << "a certificate was published that does not prove the claim: " << why;
  } else {
    EXPECT_NE(solution.message.find("presolve"), std::string::npos) << solution.message;
  }
}

// GALENET from Netlib's infeasible LP collection (J. W. Chinneck, 1993,
// https://netlib.org/lp/infeas/): a transportation network with three supplies, two transit
// nodes and three demands. Presolve proves it infeasible by bound arithmetic on row D8 and
// keeps no Farkas vector, so before #559 the verdict came back with no certificate at all.
Model galenet() {
  // Columns T14 T24 T25 T35 T46 T47 T57 T58; rows S1 S2 S3 NODE4 NODE5 D6 D7 D8.
  Model model;
  model.resize_columns(8);
  model.col_upper = {30.0, 20.0, 10.0, 10.0, 10.0, 2.0, 20.0, 30.0};
  model.resize_rows(8);
  model.row_lower = {-kInfinity, -kInfinity, -kInfinity, 0.0, 0.0, 10.0, 20.0, 30.0};
  model.row_upper = {20.0, 20.0, 20.0, 0.0, 0.0, kInfinity, kInfinity, kInfinity};
  model.matrix.reset(8, 8);
  const struct {
    Index row, col;
    double value;
  } entries[] = {{0, 0, 1.0}, {3, 0, 1.0},  {1, 1, 1.0}, {3, 1, 1.0},
                 {1, 2, 1.0}, {4, 2, 1.0},  {2, 3, 1.0}, {4, 3, 1.0},
                 {5, 4, 1.0}, {3, 4, -1.0}, {6, 5, 1.0}, {3, 5, -1.0},
                 {6, 6, 1.0}, {4, 6, -1.0}, {7, 7, 1.0}, {4, 7, -1.0}};
  for (const auto& e : entries) model.matrix.add_entry(e.row, e.col, e.value);
  model.matrix.finalize();
  return model;
}

TEST(Certificate, AnInfeasibilityPresolveProvesWithoutAProofIsRetriedForOne) {
  // #559: with presolve on (the default), an infeasible verdict whose certificate presolve
  // could not supply is retried once against the original model, and the retry's verified
  // certificate is what comes back. Without the retry this model has no certificate.
  const Model model = galenet();
  const Solution solution = solve(model, quiet(/*presolve=*/true));
  ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  ASSERT_FALSE(solution.farkas_dual.empty()) << solution.message;
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual, &why)) << why;
  EXPECT_NE(solution.message.find("retried directly against the original model"),
            std::string::npos)
      << solution.message;
}

TEST(Certificate, ATinyFarkasVectorOnAFeasibleModelIsNotAProof) {
  // Review of #652. x >= 1e6 with x in [0, 2e6] is feasible. The multiplier 1e-12 aggregates
  // to d = 1e-12, under the absolute zero floor, so the aggregate read as 0 >= 1e-6 and was
  // accepted. Scaled to unit size first, it is x >= 1e6 against a reachable 2e6.
  const Model feasible = make_lp({{1.0}}, {1e6}, {kInfinity}, {0.0}, {0.0}, {2e6});
  std::string why;
  EXPECT_FALSE(farkas_proves_infeasible(feasible, {1e-12}, &why));
  // And a real certificate still proves at any scale.
  EXPECT_TRUE(farkas_proves_infeasible(contradictory_pair(), {1e-9, -1e-9}, &why)) << why;
}

TEST(Certificate, ATinyRayOnABoundedModelIsNotAProof) {
  // min -x with x >= 0 and the row x <= 10: bounded. The ray 1e-8 moved the row by less than
  // the absolute 1e-7 and was accepted; at unit size it crosses the row's upper bound.
  const Model bounded = make_lp({{1.0}}, {-kInfinity}, {10.0}, {-1.0}, {0.0}, {kInfinity});
  std::string why;
  EXPECT_FALSE(ray_proves_unbounded(bounded, {1e-8}, &why));
}

}  // namespace
}  // namespace sankhya
