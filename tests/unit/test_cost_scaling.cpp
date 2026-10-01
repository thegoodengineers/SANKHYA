// SPDX-License-Identifier: Apache-2.0
// SANKHYA - objective scaling (#783).
//
// Netlib sc205 with its rows and columns scaled by powers of two in 2^-20..2^20 - an exact
// change of variables - has a single cost, -1 * 2^-15 = -3.05e-05. Every reduced cost is that
// small, so an absolute dual tolerance of 1e-7 let a wrong-signed -3.9e-08 through and the
// solver returned x = 0, objective 0, against a true optimum of -52.202061211707248 (Koch).
// The exponents below are the stress set's own (bench/runners/stress_instances.py,
// random.Random("stress-762:sc205")), so this is the instance #783 was found on.
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "la/scaling.hpp"

namespace sankhya {
namespace {

constexpr int kRowExponents[] = {
    12,  5,   0,   -15, 6,   -6,  4,   9,   -16, -19, -8,  -9,  -10, -10, -4,  -4,  19,  17,
    -9,  -1,  -18, 4,   1,   -12, 12,  -12, 0,   -4,  8,   -11, -7,  9,   -10, -12, -13, 20,
    13,  18,  -1,  3,   20,  20,  10,  -1,  -5,  6,   -3,  -18, -3,  1,   -7,  -13, 12,  16,
    12,  -14, -17, -4,  9,   -11, 7,   -5,  -16, -12, -9,  -4,  -15, 3,   8,   -2,  -18, 11,
    -7,  -14, -20, 13,  -17, -4,  11,  -4,  -11, 20,  -18, 15,  19,  7,   -8,  13,  12,  -17,
    -13, 11,  -3,  16,  16,  -13, 1,   -14, -4,  4,   -19, 19,  18,  -7,  -6,  -4,  0,   17,
    -14, 1,   -9,  6,   -2,  -10, -3,  -10, -13, -10, 19,  10,  15,  -17, -5,  13,  5,   -11,
    6,   -1,  4,   5,   -2,  -10, -7,  5,   15,  2,   -3,  -12, 18,  1,   -11, 20,  7,   -6,
    -6,  20,  -17, 9,   -1,  8,   11,  19,  -14, 15,  -20, -5,  6,   -8,  5,   -12, -18, 2,
    13,  1,   -19, -5,  16,  -6,  -16, 10,  -18, -19, -17, 19,  -4,  -18, 17,  -6,  -11, -20,
    -14, -20, 3,   18,  -5,  16,  12,  19,  -19, 9,   2,   -8,  -19, -10, 5,   0,   9,   3,
    -1,  -5,  9,   -5,  16,  1,   19};

constexpr int kColumnExponents[] = {
    -15, 19,  -11, -15, 11,  -14, 5,   7,  0,   20,  -2,  -13, 12,  -3,  -12, 1,   17,
    10,  19,  -5,  0,   -7,  -12, 17,  9,  -16, -13, 6,   -9,  -16, -5,  14,  1,   1,
    13,  -10, 10,  -5,  2,   10,  3,   7,  6,   11,  13,  6,   -2,  -15, 18,  -5,  -1,
    -14, 18,  4,   -2,  9,   -1,  -10, 5,  -17, 14,  -6,  -2,  10,  -15, -2,  18,  7,
    12,  -1,  18,  7,   -17, -17, 3,   17, 20,  -6,  -12, 14,  1,   19,  13,  14,  14,
    10,  -4,  17,  19,  1,   -15, 6,   19, -14, -5,  -14, -9,  3,   1,   -2,  -2,  18,
    17,  14,  -18, 19,  20,  2,   13,  19, 4,   -1,  -18, -15, 3,   -18, -1,  -19, 0,
    -7,  -16, -6,  9,   1,   -14, -8,  -3, 9,   19,  15,  6,   -14, -18, -2,  11,  6,
    -4,  -10, 20,  -16, 5,   -2,  17,  16, 16,  15,  -14, -3,  -3,  -19, -14, -6,  11,
    -8,  -14, 8,   0,   -2,  -17, -20, -9, -15, 13,  1,   -12, 18,  -14, -2,  2,   20,
    -10, -17, -11, -2,  4,   -5,  -12, 14, -12, -6,  -9,  -4,  -1,  -2,  -11, 19,  20,
    12,  -1,  9,   -16, -16, 9,   12,  -4, 6,   15,  2,   0,   6,   -12, -4,  0};

constexpr double kKochOptimum = -52.202061211707248;

Model scaled_sc205() {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/sc205.mps")
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  EXPECT_EQ(model.num_rows(), static_cast<Index>(std::size(kRowExponents)));
  EXPECT_EQ(model.num_cols(), static_cast<Index>(std::size(kColumnExponents)));
  std::vector<double> r;
  std::vector<double> s;
  for (const int e : kRowExponents) r.push_back(std::ldexp(1.0, e));
  for (const int e : kColumnExponents) s.push_back(std::ldexp(1.0, e));
  model.matrix.scale(r, s);
  for (std::size_t j = 0; j < s.size(); ++j) {
    model.col_cost[j] *= s[j];
    model.col_lower[j] /= s[j];
    model.col_upper[j] /= s[j];
  }
  for (std::size_t i = 0; i < r.size(); ++i) {
    model.row_lower[i] *= r[i];
    model.row_upper[i] *= r[i];
  }
  return model;
}

TEST(CostScaling, FactorIsAPowerOfTwoThatOnlyScalesUp) {
  EXPECT_EQ(cost_scale_factor({}), 1.0);
  EXPECT_EQ(cost_scale_factor({0.0, 0.0}), 1.0);
  EXPECT_EQ(cost_scale_factor({1.0, -3.0}), 1.0);
  EXPECT_EQ(cost_scale_factor({0.0, -std::ldexp(1.0, -15)}), std::ldexp(1.0, 15));
  const double factor = cost_scale_factor({0.3, -0.026239});
  EXPECT_GE(0.3 * factor, 1.0);
  EXPECT_LT(0.3 * factor, 2.0);
}

TEST(CostScaling, ScaledSc205ReachesTheTrueOptimum) {
  const Model model = scaled_sc205();
  ASSERT_EQ(model.col_cost[3], -std::ldexp(1.0, -15));  // the one cost, as #783 reports it
  for (const char* algorithm : {"auto", "simplex", "dual-simplex"}) {
    Options options;
    options.set_bool("log_to_console", false);
    options.set_string("algorithm", algorithm);
    const Solution solution = solve(model, options);
    EXPECT_EQ(solution.status, SolveStatus::kOptimal) << algorithm << ": " << solution.message;
    EXPECT_NEAR(solution.objective, kKochOptimum, 1e-6 * std::fabs(kKochOptimum)) << algorithm;
  }
}

TEST(CostScaling, TheObjectiveZeroVertexIsNotCalledDualFeasible) {
  // The shape of the point #783 returned: a reduced cost wrong in sign by 1.3e-3 of the only
  // cost, 3.9e-08 absolute. It must count as dual infeasible in the quality measure the
  // status guard in solve.cpp uses, as tools/verify_solution.py judges it; against a floor
  // of 1 it read as 3.9e-08 and passed.
  Model model;
  model.col_cost = {-3.05e-5, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInfinity};
  model.row_upper = {0.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.28e-3);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  ASSERT_TRUE(model.validate().empty()) << model.validate();

  Solution solution;
  solution.col_value = {0.0, 0.0};
  solution.row_dual = {-3.05e-5};
  solution.col_dual = {0.0, -1.28e-3 * 3.05e-5};
  solution.recompute_quality(model);
  EXPECT_GT(solution.dual_infeasibility_scaled, 1e-4);
}

}  // namespace
}  // namespace sankhya
