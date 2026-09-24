// SPDX-License-Identifier: Apache-2.0
// SANKHYA - c-MIR with variable-bound substitution (#498).
//
// The family's reason to exist is the fixed-charge structure: a continuous flow x_j switched
// by an integer y_j through a variable upper bound x_j <= u_j y_j. The tests show that the
// variable bounds are read correctly, that c-MIR finds the textbook cut plain MIR cannot, and
// that no cut it returns removes a feasible point: every switch assignment of small random
// fixed-charge models is checked by maximising the cut over the flows the assignment allows,
// the exact optimum of each model must satisfy every cut, and a search with the family on
// must reach that exact optimum. A deliberately weakened gate input is the negative control.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "mip/mir_cmir.hpp"
#include "mip/mir_cuts.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

using mip::Cut;

constexpr double kInf = std::numeric_limits<double>::infinity();

/// A fixed-charge network in the oracle's form (A x >= b, 0 <= x <= u): columns
/// x_0..x_{k-1} continuous flows, y_0..y_{k-1} integer switches in [0, switch_max]; the
/// variable upper bounds u_j y_j - x_j >= 0, then demand rows sum_{j in S_i} x_j >= d_i.
struct FixedCharge {
  oracle::GeneratedLp lp;
  Index arcs = 0;
};

FixedCharge random_network(std::mt19937_64& rng, Index arcs, Index demands,
                           std::int64_t switch_max) {
  std::uniform_int_distribution<std::int64_t> capacity(2, 7);
  std::uniform_int_distribution<std::int64_t> flow_cost(1, 6);
  std::uniform_int_distribution<std::int64_t> fixed_cost(4, 30);
  std::uniform_int_distribution<int> percent(0, 99);
  FixedCharge fc;
  fc.arcs = arcs;
  oracle::GeneratedLp& lp = fc.lp;
  const auto k = static_cast<std::size_t>(arcs);
  lp.num_cols = 2 * arcs;
  lp.integral.assign(2 * k, 0);
  lp.upper.assign(2 * k, oracle::kNoUpperBound);
  lp.c.assign(2 * k, 0);
  std::vector<std::int64_t> u(k);
  for (std::size_t j = 0; j < k; ++j) {
    u[j] = capacity(rng);
    lp.integral[k + j] = 1;
    lp.upper[k + j] = switch_max;
    lp.upper[j] = u[j] * switch_max;  // the simple bound the plain MIR would use
    lp.c[j] = flow_cost(rng);
    lp.c[k + j] = fixed_cost(rng);
  }
  for (std::size_t j = 0; j < k; ++j) {
    std::vector<std::int64_t> row(2 * k, 0);
    row[j] = -1;
    row[k + j] = u[j];
    lp.a.push_back(row);
    lp.b.push_back(0);
  }
  for (Index i = 0; i < demands; ++i) {
    std::vector<std::int64_t> row(2 * k, 0);
    std::int64_t reach = 0;
    for (std::size_t j = 0; j < k; ++j) {
      if (percent(rng) < 60) {
        row[j] = 1;
        reach += u[j] * switch_max;
      }
    }
    if (reach == 0) {
      row[0] = 1;
      reach = u[0] * switch_max;
    }
    lp.a.push_back(row);
    lp.b.push_back(std::max<std::int64_t>(1, reach / 2 + static_cast<std::int64_t>(i)));
  }
  lp.num_rows = static_cast<Index>(lp.a.size());
  return fc;
}

Model model_of(const oracle::GeneratedLp& lp) {
  Model model = oracle::to_model(lp);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (lp.integral[u] != 0) model.col_type[u] = VarType::kInteger;
  }
  return model;
}

Solution lp_relaxation(const Model& model) {
  Model relaxed = model;
  relaxed.col_type.assign(static_cast<std::size_t>(model.num_cols()), VarType::kContinuous);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "dual-simplex");
  return solve(relaxed, options);
}

std::vector<Cut> cmir_cuts(const Model& model, const Solution& point, mip::MirStats* stats) {
  mip::MirOptions options;
  options.cmir = true;
  return mip::generate_mir_cuts(model, point, model.col_lower, model.col_upper, stats, options);
}

/// The largest the cut's left side gets over every flow a switch assignment allows, minus its
/// right side, maximised over every assignment: positive means a feasible point is cut off.
/// Each assignment's flow LP is solved by the simplex with the switches fixed.
double worst_violation(const FixedCharge& fc, const Model& model, const Cut& cut,
                       int* assignments) {
  const auto k = static_cast<std::size_t>(fc.arcs);
  double worst = -kInf;
  std::vector<std::int64_t> y(k, 0);
  while (true) {
    Model fixed = model;
    for (std::size_t j = 0; j < 2 * k; ++j) {
      fixed.col_type[j] = VarType::kContinuous;
      fixed.col_cost[j] = -cut.coeff[j];  // minimise minus the cut
    }
    for (std::size_t j = 0; j < k; ++j) {
      fixed.col_lower[k + j] = static_cast<double>(y[j]);
      fixed.col_upper[k + j] = static_cast<double>(y[j]);
    }
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("presolve", false);
    const Solution s = solve(fixed, options);
    if (s.status == SolveStatus::kOptimal) {
      ++*assignments;
      worst = std::max(worst, -s.objective - cut.rhs);
    }
    std::size_t j = 0;
    for (; j < k; ++j) {
      if (y[j] < fc.lp.upper[k + j]) {
        ++y[j];
        break;
      }
      y[j] = 0;
    }
    if (j == k) break;
  }
  return worst;
}

TEST(CmirCuts, VariableBoundsAreReadOffTwoNonzeroRows) {
  // Columns x (continuous), y (integer), z (continuous); rows:
  //   x - 5 y <= 0        x <= 5 y
  //   2 x + 4 y >= 6      x >= 3 - 2 y
  //   -z + 3 y <= 1       z >= 3 y - 1
  //   x + z <= 9          two continuous: not a variable bound
  Model m;
  m.col_cost = {0.0, 0.0, 0.0};
  m.col_lower = {0.0, 0.0, 0.0};
  m.col_upper = {kInf, 1.0, kInf};
  m.col_type = {VarType::kContinuous, VarType::kInteger, VarType::kContinuous};
  m.matrix.reset(4, 3);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, -5.0);
  m.matrix.add_entry(1, 0, 2.0);
  m.matrix.add_entry(1, 1, 4.0);
  m.matrix.add_entry(2, 2, -1.0);
  m.matrix.add_entry(2, 1, 3.0);
  m.matrix.add_entry(3, 0, 1.0);
  m.matrix.add_entry(3, 2, 1.0);
  m.matrix.finalize();
  m.row_lower = {-kInf, 6.0, -kInf, -kInf};
  m.row_upper = {0.0, kInf, 1.0, 9.0};
  const mip::VariableBounds vb = mip::find_variable_bounds(m);
  EXPECT_EQ(vb.rows, 3);
  ASSERT_EQ(vb.upper[0].size(), 1u);
  EXPECT_EQ(vb.upper[0][0].indicator, 1);
  EXPECT_DOUBLE_EQ(vb.upper[0][0].slope, 5.0);
  EXPECT_DOUBLE_EQ(vb.upper[0][0].constant, 0.0);
  ASSERT_EQ(vb.lower[0].size(), 1u);
  EXPECT_DOUBLE_EQ(vb.lower[0][0].slope, -2.0);
  EXPECT_DOUBLE_EQ(vb.lower[0][0].constant, 3.0);
  ASSERT_EQ(vb.lower[2].size(), 1u);  // -z + 3 y <= 1 with a = -1 < 0: a lower bound on z
  EXPECT_DOUBLE_EQ(vb.lower[2][0].slope, 3.0);
  EXPECT_DOUBLE_EQ(vb.lower[2][0].constant, -1.0);
  EXPECT_TRUE(vb.upper[2].empty());
  EXPECT_TRUE(vb.upper[1].empty());
  EXPECT_TRUE(vb.lower[1].empty());
}

/// A positive multiple of -y1 - y2 <= -2 (columns x1, x2, y1, y2), nothing on the flows.
bool is_two_switch_cut(const Cut& cut) {
  const double scale = -cut.coeff[2];
  if (scale <= 0.0) return false;
  return std::fabs(cut.coeff[3] / scale + 1.0) < 1e-9 && std::fabs(cut.coeff[0]) < 1e-9 &&
         std::fabs(cut.coeff[1]) < 1e-9 && std::fabs(cut.rhs / scale + 2.0) < 1e-9;
}

TEST(CmirCuts, TheTwoSwitchCutPlainMirCannotSee) {
  // x1 + x2 >= 5 with x_j <= 4 y_j, y binary: both switches must be on, y1 + y2 >= 2. At
  // the LP point y = (1, 0.25), x = (4, 1) the demand row has only continuous columns, so
  // the plain MIR cannot round it alone (it can aggregate a variable-bound row in and find
  // a one-switch cut, y2 >= 1, but not this one); substituting x_j = 4 y_j - s_j gives
  // -4 y1 - 4 y2 + s1 + s2 <= -5, and with delta = 4 and y1 complemented the rounding gives
  // -y1 - y2 <= -2.
  Model m;
  m.col_cost = {1.0, 1.0, 10.0, 10.0};
  m.col_lower = {0.0, 0.0, 0.0, 0.0};
  m.col_upper = {4.0, 4.0, 1.0, 1.0};
  m.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kInteger,
                VarType::kInteger};
  m.matrix.reset(3, 4);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 2, -4.0);
  m.matrix.add_entry(1, 1, 1.0);
  m.matrix.add_entry(1, 3, -4.0);
  m.matrix.add_entry(2, 0, 1.0);
  m.matrix.add_entry(2, 1, 1.0);
  m.matrix.finalize();
  m.row_lower = {-kInf, -kInf, 5.0};
  m.row_upper = {0.0, 0.0, kInf};
  Solution point;
  point.col_value = {4.0, 1.0, 1.0, 0.25};
  for (const Cut& cut : mip::generate_mir_cuts(m, point)) {
    EXPECT_FALSE(is_two_switch_cut(cut)) << "the plain MIR found the two-switch cut";
  }
  mip::MirStats stats;
  const std::vector<Cut> cuts = cmir_cuts(m, point, &stats);
  EXPECT_EQ(stats.variable_bound_rows, 2);
  ASSERT_FALSE(cuts.empty());
  bool found = false;
  for (const Cut& cut : cuts) found = found || is_two_switch_cut(cut);
  EXPECT_TRUE(found) << "c-MIR did not find y1 + y2 >= 2";
}

TEST(CmirCuts, NoCutRemovesAFeasiblePointOfRandomNetworks) {
  std::mt19937_64 rng(4980);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int cuts_checked = 0;
  int assignments = 0;
  int models = 0;
  for (int trial = 0; trial < 160; ++trial) {
    const FixedCharge fc =
        random_network(rng, 2 + trial % 3, 1 + trial % 2, trial % 4 == 3 ? 2 : 1);
    const Model model = model_of(fc.lp);
    // The LP relaxation's point, and random points in the box, so the separator is asked
    // at places the simplex would never stop.
    std::vector<Solution> points;
    const Solution relaxed = lp_relaxation(model);
    if (relaxed.status == SolveStatus::kOptimal) points.push_back(relaxed);
    for (int probe = 0; probe < 2; ++probe) {
      Solution p;
      p.col_value.assign(static_cast<std::size_t>(model.num_cols()), 0.0);
      const auto k = static_cast<std::size_t>(fc.arcs);
      for (std::size_t j = 0; j < k; ++j) {
        p.col_value[k + j] = unit(rng) * model.col_upper[k + j];
        p.col_value[j] = unit(rng) * model.col_upper[j] * p.col_value[k + j] /
                         std::max(1.0, model.col_upper[k + j]);
      }
      points.push_back(p);
    }
    ++models;
    for (const Solution& point : points) {
      for (const Cut& cut : cmir_cuts(model, point, nullptr)) {
        const double worst = worst_violation(fc, model, cut, &assignments);
        ASSERT_LE(worst, 1e-7 * std::max(1.0, std::fabs(cut.rhs)))
            << "trial " << trial << ": a switch assignment's flow violates the cut by " << worst
            << "\n"
            << fc.lp.to_text();
        ++cuts_checked;
      }
    }
  }
  EXPECT_GT(cuts_checked, 100) << "the sweep should exercise the separator";
  std::printf(
      "[  INFO    ] c-MIR: %d cuts on %d random networks, checked against %d switch "
      "assignments, none removes a feasible flow\n",
      cuts_checked, models, assignments);
}

TEST(CmirCuts, TheGateCatchesADeliberatelyInvalidCut) {
  // y1 + y2 >= 2 is exact for the two-switch model above; y1 + y2 >= 3 is not, and the
  // gate must say so.
  FixedCharge fc;
  fc.arcs = 2;
  fc.lp.num_cols = 4;
  fc.lp.integral = {0, 0, 1, 1};
  fc.lp.upper = {4, 4, 1, 1};
  fc.lp.c = {1, 1, 10, 10};
  fc.lp.a = {{-1, 0, 4, 0}, {0, -1, 0, 4}, {1, 1, 0, 0}};
  fc.lp.b = {0, 0, 5};
  fc.lp.num_rows = 3;
  const Model model = model_of(fc.lp);
  Cut exact;
  exact.coeff = {0.0, 0.0, -1.0, -1.0};
  exact.rhs = -2.0;
  int assignments = 0;
  EXPECT_LE(worst_violation(fc, model, exact, &assignments), 1e-9);
  Cut invalid = exact;
  invalid.rhs = -3.0;
  EXPECT_GT(worst_violation(fc, model, invalid, &assignments), 0.5);
}

TEST(CmirCuts, TheExactOptimumSatisfiesEveryCutAndTheSearchFindsIt) {
  std::mt19937_64 rng(4981);
  int compared = 0;
  int with_cuts = 0;
  std::int64_t cuts_applied = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const FixedCharge fc =
        random_network(rng, 3 + trial % 3, 1 + trial % 3, trial % 5 == 4 ? 2 : 1);
    const oracle::OracleResult exact = oracle::solve_exact_milp(fc.lp, 50000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    const Model model = model_of(fc.lp);
    const Solution relaxed = lp_relaxation(model);
    if (relaxed.status != SolveStatus::kOptimal) continue;
    const std::vector<Cut> cuts = cmir_cuts(model, relaxed, nullptr);
    if (!cuts.empty()) ++with_cuts;
    for (const Cut& cut : cuts) {
      double at_optimum = 0.0;
      for (std::size_t j = 0; j < cut.coeff.size(); ++j) {
        at_optimum += cut.coeff[j] * exact.x[j].to_double();
      }
      EXPECT_LE(at_optimum, cut.rhs + 1e-9 * std::max(1.0, std::fabs(cut.rhs)))
          << "the exact optimum was cut off\n"
          << fc.lp.to_text();
    }
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("enable_root_cuts", true);
    options.set_bool("mir_cmir", true);
    options.set_int("tree_cut_depth", 3);
    options.set_int("cut_support_floor", 100);  // the models are small (#608)
    const Solution s = solve(model, options);
    const double expected = exact.objective.to_double();
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << fc.lp.to_text();
    EXPECT_NEAR(s.objective, expected, 1e-6 * std::max(1.0, std::fabs(expected)))
        << fc.lp.to_text();
    cuts_applied += s.cuts_applied;
    ++compared;
  }
  EXPECT_GE(compared, 50);
  EXPECT_GE(with_cuts, 10) << "too few models produced a c-MIR cut for the gate to mean much";
  std::printf(
      "[  INFO    ] c-MIR search: %d fixed-charge models at the exact optimum, %d with a root "
      "c-MIR cut, %lld cut rows applied\n",
      compared, with_cuts, static_cast<long long>(cuts_applied));
}

}  // namespace
}  // namespace sankhya
