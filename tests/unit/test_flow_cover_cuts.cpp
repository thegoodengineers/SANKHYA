// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts (#419).
//
// A cut with continuous columns cannot be checked by enumerating integer points: for every
// setting of the switches the feasible flows are a polytope, and the cut is valid only if the
// most violating flow in each of those polytopes still satisfies it. So the gate here is
// exact: every switch assignment of small random fixed-charge models is fixed, the cut's
// violation is maximised over the flows by the rational simplex oracle, and the maximum must
// be non-positive with no tolerance anywhere. A negative control tightens a real cut by one
// unit and checks the gate catches it. The textbook cover is checked coefficient by
// coefficient, and a search with the family on is checked against the oracle's own optimum.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/flow_cover_cuts.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
using oracle::Rational;

/// A single-node fixed-charge model: arcs with flow x_j >= 0, switch y_j binary, the variable
/// upper bound x_j - u_j y_j <= 0 as a row, and one flow row
/// sum_j sign_j x_j + sum_k plain_sign_k p_k <= b over the arcs and some always-on columns
/// p_k in [0, plain_cap_k]. Columns: x_0..x_{k-1}, y_0..y_{k-1}, p_0..p_{r-1}; rows: the k
/// variable upper bounds, then the flow row.
struct FixedCharge {
  std::vector<std::int64_t> u;
  std::vector<int> sign;
  std::vector<std::int64_t> plain_cap;
  std::vector<int> plain_sign;
  std::int64_t b = 0;
  std::vector<double> flow_cost;   ///< on x_j; empty means zero
  std::vector<double> fixed_cost;  ///< on y_j; empty means zero
  bool demand_row = false;         ///< the flow row is an equality (demand met exactly)
};

Model build(const FixedCharge& fc) {
  const auto k = fc.u.size();
  const auto r = fc.plain_cap.size();
  const auto n = static_cast<Index>(2 * k + r);
  Model m;
  m.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), kInf);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (std::size_t j = 0; j < k; ++j) {
    m.col_type[k + j] = VarType::kInteger;
    m.col_upper[k + j] = 1.0;
    if (!fc.flow_cost.empty()) m.col_cost[j] = fc.flow_cost[j];
    if (!fc.fixed_cost.empty()) m.col_cost[k + j] = fc.fixed_cost[j];
  }
  for (std::size_t q = 0; q < r; ++q)
    m.col_upper[2 * k + q] = static_cast<double>(fc.plain_cap[q]);
  const auto rows = static_cast<Index>(k + 1);
  m.matrix.reset(rows, n);
  for (std::size_t j = 0; j < k; ++j) {
    m.matrix.add_entry(static_cast<Index>(j), static_cast<Index>(j), 1.0);
    m.matrix.add_entry(static_cast<Index>(j), static_cast<Index>(k + j),
                       -static_cast<double>(fc.u[j]));
  }
  for (std::size_t j = 0; j < k; ++j) {
    m.matrix.add_entry(static_cast<Index>(k), static_cast<Index>(j), fc.sign[j]);
  }
  for (std::size_t q = 0; q < r; ++q) {
    m.matrix.add_entry(static_cast<Index>(k), static_cast<Index>(2 * k + q), fc.plain_sign[q]);
  }
  m.matrix.finalize();
  m.row_lower.assign(static_cast<std::size_t>(rows), -kInf);
  m.row_upper.assign(static_cast<std::size_t>(rows), 0.0);
  m.row_upper[k] = static_cast<double>(fc.b);
  if (fc.demand_row) m.row_lower[k] = static_cast<double>(fc.b);
  m.hessian.reset(n, n);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

Solution at(const std::vector<double>& x) {
  Solution s;
  s.col_value = x;
  return s;
}

double lhs_at(const Cut& cut, const std::vector<double>& x) {
  double total = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) total += cut.coeff[j] * x[j];
  return total;
}

/// The exact gate: for every switch assignment, the most violating flow. Returns the number
/// of assignments checked, and sets `why` on the first assignment whose maximum violation
/// is positive. Overflow or an iteration limit in the oracle skips that assignment and is
/// counted in `skipped`; it is never taken as agreement.
int exact_gate(const FixedCharge& fc, const Cut& cut, int* skipped, std::string* why) {
  const auto k = fc.u.size();
  const auto r = fc.plain_cap.size();
  // The cut must have integer data for the exact comparison to mean anything; every input
  // here is an integer and lambda is a difference of integers, so it does.
  for (const double c : cut.coeff) EXPECT_EQ(c, std::round(c));
  EXPECT_EQ(cut.rhs, std::round(cut.rhs));
  int checked = 0;
  for (std::uint32_t mask = 0; mask < (1u << k); ++mask) {
    oracle::GeneratedLp lp;
    lp.num_rows = 1;
    lp.num_cols = static_cast<Index>(k + r);
    lp.a.assign(1, std::vector<std::int64_t>(k + r, 0));
    for (std::size_t j = 0; j < k; ++j) lp.a[0][j] = -fc.sign[j];  // A x >= b form
    for (std::size_t q = 0; q < r; ++q) lp.a[0][k + q] = -fc.plain_sign[q];
    lp.b = {-fc.b};
    lp.upper.resize(k + r);
    double switch_terms = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      const bool on = ((mask >> j) & 1u) != 0;
      lp.upper[j] = on ? fc.u[j] : 0;
      if (on) switch_terms += cut.coeff[k + j];
    }
    for (std::size_t q = 0; q < r; ++q) lp.upper[k + q] = fc.plain_cap[q];
    // Minimise minus the cut's flow part: the optimum is minus the largest left side.
    lp.c.resize(k + r);
    for (std::size_t j = 0; j < k; ++j) lp.c[j] = -static_cast<std::int64_t>(cut.coeff[j]);
    for (std::size_t q = 0; q < r; ++q) {
      lp.c[k + q] = -static_cast<std::int64_t>(cut.coeff[2 * k + q]);
    }
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kInfeasible) continue;  // no flow at all
    if (exact.status != oracle::OracleStatus::kOptimal) {
      ++*skipped;
      continue;
    }
    ++checked;
    const Rational largest_left =
        Rational(0) - exact.objective + Rational(static_cast<std::int64_t>(switch_terms));
    const Rational right(static_cast<std::int64_t>(cut.rhs));
    if (largest_left > right) {
      *why = "switch mask " + std::to_string(mask) + ": the largest left side " +
             std::to_string(largest_left.to_double()) + " exceeds " +
             std::to_string(right.to_double());
      return checked;
    }
  }
  return checked;
}

TEST(FlowCoverCuts, TheTextbookCoverIsFound) {
  // Three inflows of capacity 5, 4 and 3 into a node that takes 7. At x = (5, 2, 0),
  // y = (1, 1/2, 0) the cover is the two switched arcs, lambda = 9 - 7 = 2, and the flow
  // cover inequality is  x0 + x1 + 3 (1 - y0) + 2 (1 - y1) <= 7, that is
  // x0 + x1 - 3 y0 - 2 y1 <= 2, violated at the point by one unit.
  FixedCharge fc;
  fc.u = {5, 4, 3};
  fc.sign = {1, 1, 1};
  fc.b = 7;
  const Model m = build(fc);
  FlowCoverStats stats;
  const std::vector<Cut> cuts = generate_flow_cover_cuts(m, at({5.0, 2.0, 0.0, 1.0, 0.5, 0.0}),
                                                         m.col_lower, m.col_upper, &stats);
  EXPECT_EQ(stats.vub_rows, 3);
  ASSERT_EQ(cuts.size(), 1u) << "covers " << stats.covers << ", flow rows " << stats.flow_rows;
  const Cut& cut = cuts[0];
  EXPECT_EQ(cut.coeff, (std::vector<double>{1.0, 1.0, 0.0, -3.0, -2.0, 0.0}));
  EXPECT_EQ(cut.rhs, 2.0);
  EXPECT_NEAR(lhs_at(cut, {5.0, 2.0, 0.0, 1.0, 0.5, 0.0}) - cut.rhs, 1.0, 1e-12);
  int skipped = 0;
  std::string why;
  EXPECT_GT(exact_gate(fc, cut, &skipped, &why), 0);
  EXPECT_TRUE(why.empty()) << why;
}

TEST(FlowCoverCuts, AnOutflowSitsOnTheRightInItsSmallerForm) {
  // Inflows of 5 and 4, an outflow of capacity 3, the node takes 5. At x = (4.5, 1.5, 1),
  // y = (0.9, 0.5, 1/3) the cover is both inflows, lambda = 4, the first arc keeps
  // (5 - 4) (1 - y0), and the outflow's smaller form is its flow (1 against lambda y2 =
  // 4/3):  x0 + x1 - y0 - x2 <= 4, violated by 0.1.
  FixedCharge fc;
  fc.u = {5, 4, 3};
  fc.sign = {1, 1, -1};
  fc.b = 5;
  const Model m = build(fc);
  const std::vector<double> point{4.5, 1.5, 1.0, 0.9, 0.5, 1.0 / 3.0};
  const std::vector<Cut> cuts =
      generate_flow_cover_cuts(m, at(point), m.col_lower, m.col_upper);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_EQ(cuts[0].coeff, (std::vector<double>{1.0, 1.0, -1.0, -1.0, 0.0, 0.0}));
  EXPECT_EQ(cuts[0].rhs, 4.0);
  EXPECT_NEAR(lhs_at(cuts[0], point) - cuts[0].rhs, 0.1, 1e-12);
  int skipped = 0;
  std::string why;
  EXPECT_GT(exact_gate(fc, cuts[0], &skipped, &why), 0);
  EXPECT_TRUE(why.empty()) << why;
}

TEST(FlowCoverCuts, NothingWithoutASwitchedInflow) {
  // A knapsack row over binaries and a bounded continuous column has no variable upper
  // bound anywhere, so this family stays out of it (the cover cuts own that row).
  Model m;
  m.col_cost = {0, 0, 0};
  m.col_lower = {0, 0, 0};
  m.col_upper = {1, 1, 4};
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 3.0);
  m.matrix.add_entry(0, 1, 3.0);
  m.matrix.add_entry(0, 2, 1.0);
  m.matrix.finalize();
  m.row_lower = {-kInf};
  m.row_upper = {4.0};
  m.hessian.reset(3, 3);
  m.hessian.finalize();
  FlowCoverStats stats;
  EXPECT_TRUE(generate_flow_cover_cuts(m, at({0.5, 0.5, 1.0}), m.col_lower, m.col_upper, &stats)
                  .empty());
  EXPECT_EQ(stats.vub_rows, 0);
}

/// Random fixed-charge models for the sweeps: two to four arcs, mostly inflows, capacities
/// 1 to 6, up to two always-on columns, and a node that takes about half the inflow.
FixedCharge random_fixed_charge(std::mt19937& rng, int trial) {
  std::uniform_int_distribution<int> capacity(1, 6);
  std::uniform_int_distribution<int> coin(0, 3);
  FixedCharge fc;
  const int arcs = 2 + trial % 3;
  std::int64_t inflow = 0;
  for (int j = 0; j < arcs; ++j) {
    fc.u.push_back(capacity(rng));
    fc.sign.push_back(coin(rng) == 0 && j > 0 ? -1 : 1);
    if (fc.sign.back() > 0) inflow += fc.u.back();
  }
  const int plain = trial % 4 == 3 ? 2 : trial % 2;
  for (int q = 0; q < plain; ++q) {
    fc.plain_cap.push_back(capacity(rng));
    fc.plain_sign.push_back(coin(rng) == 0 ? -1 : 1);
  }
  fc.b = std::max<std::int64_t>(1, inflow / 2 + (trial % 5) - 2);
  return fc;
}

/// A point for the separator: each switch at a random value, each flow at a random share of
/// what its switch allows, each always-on column anywhere in its box.
std::vector<double> random_point(std::mt19937& rng, const FixedCharge& fc) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const auto k = fc.u.size();
  std::vector<double> x(2 * k + fc.plain_cap.size(), 0.0);
  for (std::size_t j = 0; j < k; ++j) {
    x[k + j] = unit(rng);
    x[j] = unit(rng) * static_cast<double>(fc.u[j]) * x[k + j];
  }
  for (std::size_t q = 0; q < fc.plain_cap.size(); ++q) {
    x[2 * k + q] = unit(rng) * static_cast<double>(fc.plain_cap[q]);
  }
  return x;
}

TEST(FlowCoverCuts, NoCutRemovesAnyFlowOfRandomModelsExactly) {
  std::mt19937 rng(419);
  int cuts_checked = 0;
  int assignments = 0;
  int skipped = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const FixedCharge fc = random_fixed_charge(rng, trial);
    const Model m = build(fc);
    for (int probe = 0; probe < 2; ++probe) {
      const std::vector<double> x = random_point(rng, fc);
      for (const Cut& cut : generate_flow_cover_cuts(m, at(x), m.col_lower, m.col_upper)) {
        std::string why;
        assignments += exact_gate(fc, cut, &skipped, &why);
        ASSERT_TRUE(why.empty()) << "trial " << trial << ": " << why;
        ++cuts_checked;
      }
    }
  }
  EXPECT_GT(cuts_checked, 100) << "the sweep should exercise the separator";
  EXPECT_GT(assignments, 500);
  EXPECT_LT(skipped, assignments / 10)
      << "too many oracle overflows for the sweep to mean much";
}

TEST(FlowCoverCuts, TheGateCatchesADeliberatelyInvalidCut) {
  // The textbook cut with its right-hand side lowered by one is violated by a feasible
  // flow (x = (5, 2, 0) with both switches on satisfies the model and gives 5 + 2 - 3 - 2
  // = 2 > 1), and the gate must say so: a harness that cannot fail proves nothing.
  FixedCharge fc;
  fc.u = {5, 4, 3};
  fc.sign = {1, 1, 1};
  fc.b = 7;
  const Model m = build(fc);
  std::vector<Cut> cuts =
      generate_flow_cover_cuts(m, at({5.0, 2.0, 0.0, 1.0, 0.5, 0.0}), m.col_lower, m.col_upper);
  ASSERT_EQ(cuts.size(), 1u);
  cuts[0].rhs -= 1.0;
  int skipped = 0;
  std::string why;
  exact_gate(fc, cuts[0], &skipped, &why);
  EXPECT_FALSE(why.empty()) << "the gate accepted a cut that removes a feasible flow";
}

TEST(FlowCoverCuts, TheSearchWithTheFamilyOnAgreesWithTheExactOptimum) {
  // Fixed-charge models with a demand to meet exactly, a fixed cost per switch and a unit
  // cost per flow, solved with the root cut round and this family on, against the exact
  // optimum: the oracle's LP for every switch assignment, the best kept. A cut that removed
  // the optimum would show up here as a wrong answer with a confident proof.
  std::mt19937 rng(4190);
  std::uniform_int_distribution<int> cost(1, 9);
  int compared = 0;
  for (int trial = 0; trial < 60; ++trial) {
    FixedCharge fc = random_fixed_charge(rng, trial);
    fc.demand_row = true;
    fc.plain_cap.clear();
    fc.plain_sign.clear();
    for (std::size_t j = 0; j < fc.u.size(); ++j) {
      fc.flow_cost.push_back(cost(rng));
      fc.fixed_cost.push_back(3 * cost(rng));
    }
    const Model m = build(fc);
    // The exact optimum over the switches.
    const auto k = fc.u.size();
    bool any = false;
    Rational best(0);
    for (std::uint32_t mask = 0; mask < (1u << k); ++mask) {
      oracle::GeneratedLp lp;
      lp.num_rows = 2;
      lp.num_cols = static_cast<Index>(k);
      lp.a.assign(2, std::vector<std::int64_t>(k, 0));
      for (std::size_t j = 0; j < k; ++j) {
        lp.a[0][j] = fc.sign[j];   // sum s x >= b
        lp.a[1][j] = -fc.sign[j];  // sum s x <= b
      }
      lp.b = {fc.b, -fc.b};
      lp.upper.resize(k);
      lp.c.resize(k);
      std::int64_t fixed = 0;
      for (std::size_t j = 0; j < k; ++j) {
        const bool on = ((mask >> j) & 1u) != 0;
        lp.upper[j] = on ? fc.u[j] : 0;
        lp.c[j] = static_cast<std::int64_t>(fc.flow_cost[j]);
        if (on) fixed += static_cast<std::int64_t>(fc.fixed_cost[j]);
      }
      const oracle::OracleResult exact = oracle::solve_exact(lp);
      if (exact.status != oracle::OracleStatus::kOptimal) continue;
      const Rational value = exact.objective + Rational(fixed);
      if (!any || value < best) best = value;
      any = true;
    }
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("enable_root_cuts", true);
    options.set_bool("enable_flow_cover_cuts", true);
    const Solution solved = solve(m, options);
    ++compared;
    if (!any) {
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
          << "trial " << trial << ": " << solved.message;
      continue;
    }
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best.to_double(), 1e-6) << "trial " << trial;
  }
  EXPECT_EQ(compared, 60);
}

}  // namespace
}  // namespace sankhya::mip
