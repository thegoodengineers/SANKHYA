// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bound and objective rescaling and the initial primal weight ||c|| / ||b|| of the
// LP PDHG (#482 item 3, options pdhg_bound_objective_rescaling and
// pdhg_initial_weight_from_norms, src/pdhg/pdhg_norm_rescale.hpp).
//
// Pinned: both options off by default; the rescaling divides exactly what it says by exactly
// ||.||_2 + 1, leaves the matrix alone and folds itself into the multipliers so that
// x = column * z and y = row * w still give the scaled point (beta z, gamma w); the initial
// weight is ||c|| / ||b||, 1 when either is zero, clamped; and each option, and both, still
// reach a verified optimum that agrees with the default path on committed Netlib instances
// under both iteration schemes.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "la/scaling.hpp"
#include "pdhg/pdhg_norm_rescale.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

double norm(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s);
}

TEST(PdhgNormRescale, BothOptionsAreOffByDefault) {
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("pdhg_bound_objective_rescaling"));
  EXPECT_FALSE(defaults.get_bool("pdhg_initial_weight_from_norms"));
}

TEST(PdhgNormRescale, RescalesTheVectorsAndFoldsIntoTheMultipliers) {
  for (const char* name : {"afiro", "adlittle", "israel"}) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Scaling before = build_scaling(model, model.col_cost, 10);
    Scaling after = before;
    const pdhg::BoundObjectiveRescale r = pdhg::rescale_bounds_and_objective(&after);
    const double beta = pdhg::scaled_bound_norm(before) + 1.0;
    const double gamma = norm(before.cost) + 1.0;
    EXPECT_EQ(r.bound_divisor, beta) << name;
    EXPECT_EQ(r.objective_divisor, gamma) << name;
    EXPECT_GT(beta, 1.0) << name << ": a model with no right-hand side tests nothing here";
    EXPECT_EQ(after.matrix.values(), before.matrix.values()) << name;
    for (std::size_t j = 0; j < before.cost.size(); ++j) {
      EXPECT_EQ(after.cost[j], before.cost[j] / gamma);
      EXPECT_EQ(after.column[j], before.column[j] * beta);
      for (const auto& [b, a] : {std::pair{before.col_lower[j], after.col_lower[j]},
                                 std::pair{before.col_upper[j], after.col_upper[j]}}) {
        if (is_finite_bound(b)) {
          EXPECT_EQ(a, b / beta);
        } else {
          EXPECT_EQ(a, b);
        }
      }
    }
    for (std::size_t i = 0; i < before.row_lower.size(); ++i) {
      EXPECT_EQ(after.row[i], before.row[i] * gamma);
      for (const auto& [b, a] : {std::pair{before.row_lower[i], after.row_lower[i]},
                                 std::pair{before.row_upper[i], after.row_upper[i]}}) {
        if (is_finite_bound(b)) {
          EXPECT_EQ(a, b / beta);
        } else {
          EXPECT_EQ(a, b);
        }
      }
    }
    // Both rescaled norms are now below one: ||v|| / (||v|| + 1).
    EXPECT_NEAR(norm(after.cost), (gamma - 1.0) / gamma, 1e-14) << name;
    EXPECT_NEAR(pdhg::scaled_bound_norm(after), (beta - 1.0) / beta, 1e-14) << name;
  }
}

TEST(PdhgNormRescale, InitialWeightIsTheNormRatio) {
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("blend"), &model).ok);
  Scaling scaling = build_scaling(model, model.col_cost, 10);
  const double expected = norm(scaling.cost) / pdhg::scaled_bound_norm(scaling);
  EXPECT_NEAR(pdhg::initial_primal_weight_from_norms(scaling), expected, 1e-15 * expected);

  Scaling no_cost = scaling;
  std::fill(no_cost.cost.begin(), no_cost.cost.end(), 0.0);
  EXPECT_EQ(pdhg::initial_primal_weight_from_norms(no_cost), 1.0);
  Scaling no_rhs = scaling;
  std::fill(no_rhs.row_lower.begin(), no_rhs.row_lower.end(), 0.0);
  std::fill(no_rhs.row_upper.begin(), no_rhs.row_upper.end(), 0.0);
  EXPECT_EQ(pdhg::initial_primal_weight_from_norms(no_rhs), 1.0);
  Scaling huge = scaling;
  for (double& c : huge.cost) c *= 1e12;
  EXPECT_EQ(pdhg::initial_primal_weight_from_norms(huge), 1e6);  // the engine's clamp
}

Options pdhg_options(bool rescale, bool weight, bool halpern) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_bound_objective_rescaling", rescale);
  options.set_bool("pdhg_initial_weight_from_norms", weight);
  options.set_bool("pdhg_halpern", halpern);
  if (halpern) options.set_bool("pdhg_restart", false);
  return options;
}

TEST(PdhgNormRescale, EveryArmReachesTheDefaultPathsOptimum) {
  // The same instances as test_pdhg_step_weight.cpp, for the same reason.
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const bool halpern : {false, true}) {
    for (const char* name : names) {
      Model model;
      ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
      const Solution base = solve(model, pdhg_options(false, false, halpern));
      ASSERT_EQ(base.status, SolveStatus::kOptimal) << name << ": " << base.message;
      for (const auto& arm :
           {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
        const Solution s = solve(model, pdhg_options(arm.first, arm.second, halpern));
        const std::string label = std::string(name) + (halpern ? " halpern" : " averaged") +
                                  (arm.first ? " rescaled" : "") +
                                  (arm.second ? " norm-weight" : "");
        EXPECT_EQ(s.status, SolveStatus::kOptimal) << label << ": " << s.message;
        EXPECT_NEAR(s.objective, base.objective,
                    1e-6 * std::max(1.0, std::fabs(base.objective)))
            << label << ": " << s.iterations << " iterations against " << base.iterations;
        // The reported duals are mapped back through the folded multipliers: they must be the
        // original model's, so the dual objective the engine certified matches the primal.
        EXPECT_NEAR(s.dual_bound, s.objective, 1e-5 * std::max(1.0, std::fabs(s.objective)))
            << label;
      }
    }
  }
}

}  // namespace
}  // namespace sankhya
