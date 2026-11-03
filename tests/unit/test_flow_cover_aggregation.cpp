// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts on aggregated rows (#1158): the (l, S) path cuts of lot sizing.
//
// The separator in flow_cover_cuts.cpp walks the balance rows of a lot-sizing chain along
// its stocks (row_aggregation.hpp) and separates a flow cover on each aggregate. Three
// things: the two-period example gives the path cut by hand; every cut of random chains is
// gated exactly (each switch assignment fixed, the cut's violation maximised over the
// continuous columns by the rational simplex, the maximum non-positive); and a search with
// the family on reaches the exact optimum over the assignments.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

Solution at(const std::vector<double>& x) {
  Solution s;
  s.col_value = x;
  return s;
}

// ---- AGGREGATION (#1158): flow covers on aggregated balance rows ------------------------

/// A single-item lot-sizing chain of T periods (Barany, Van Roy and Wolsey 1984): production
/// x_t in [0, u_t y_t] with y_t binary, stock s_t >= 0 carried into the next period (s_0 and
/// s_T are zero and not columns), demand d_t. Columns x_0..x_{T-1}, y_0..y_{T-1},
/// s_0..s_{T-2}; rows the T variable upper bounds x_t - u_t y_t <= 0, then the T balance
/// equalities s_{t-1} + x_t - s_t = d_t. All data integer, so the gate below is exact.
struct LotSizing {
  std::vector<std::int64_t> u;
  std::vector<std::int64_t> d;
  std::vector<std::int64_t> fixed_cost;    ///< on y_t; empty means zero
  std::vector<std::int64_t> holding_cost;  ///< on s_t; empty means zero
};

Model build(const LotSizing& ls) {
  const auto T = ls.u.size();
  const auto n = static_cast<Index>(3 * T - 1);
  Model m;
  m.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), kInf);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (std::size_t t = 0; t < T; ++t) {
    m.col_type[T + t] = VarType::kInteger;
    m.col_upper[T + t] = 1.0;
    if (!ls.fixed_cost.empty()) m.col_cost[T + t] = static_cast<double>(ls.fixed_cost[t]);
  }
  for (std::size_t t = 0; t + 1 < T; ++t) {
    if (!ls.holding_cost.empty()) {
      m.col_cost[2 * T + t] = static_cast<double>(ls.holding_cost[t]);
    }
  }
  const auto rows = static_cast<Index>(2 * T);
  m.matrix.reset(rows, n);
  for (std::size_t t = 0; t < T; ++t) {
    m.matrix.add_entry(static_cast<Index>(t), static_cast<Index>(t), 1.0);
    m.matrix.add_entry(static_cast<Index>(t), static_cast<Index>(T + t),
                       -static_cast<double>(ls.u[t]));
  }
  for (std::size_t t = 0; t < T; ++t) {
    const auto i = static_cast<Index>(T + t);
    m.matrix.add_entry(i, static_cast<Index>(t), 1.0);
    if (t > 0) m.matrix.add_entry(i, static_cast<Index>(2 * T + t - 1), 1.0);
    if (t + 1 < T) m.matrix.add_entry(i, static_cast<Index>(2 * T + t), -1.0);
  }
  m.matrix.finalize();
  m.row_lower.assign(static_cast<std::size_t>(rows), -kInf);
  m.row_upper.assign(static_cast<std::size_t>(rows), 0.0);
  for (std::size_t t = 0; t < T; ++t) {
    m.row_lower[T + t] = static_cast<double>(ls.d[t]);
    m.row_upper[T + t] = static_cast<double>(ls.d[t]);
  }
  m.hessian.reset(n, n);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

/// The model's LP over its continuous columns with every binary fixed by `mask` (bit j of
/// the j-th binary column), in the oracle's A x >= b form with integer data, minimising
/// `objective` over the continuous columns. The binary part of each row moves into b.
oracle::GeneratedLp continuous_lp(const Model& m, std::uint32_t mask,
                                  const std::vector<std::int64_t>& objective,
                                  std::vector<Index>* continuous) {
  std::vector<Index> binaries;
  continuous->clear();
  for (Index j = 0; j < m.num_cols(); ++j) {
    (m.col_type[static_cast<std::size_t>(j)] == VarType::kInteger ? binaries : *continuous)
        .push_back(j);
  }
  std::vector<double> y(static_cast<std::size_t>(m.num_cols()), 0.0);
  for (std::size_t k = 0; k < binaries.size(); ++k) {
    y[static_cast<std::size_t>(binaries[k])] = ((mask >> k) & 1u) != 0 ? 1.0 : 0.0;
  }
  oracle::GeneratedLp lp;
  lp.num_cols = static_cast<Index>(continuous->size());
  for (Index i = 0; i < m.num_rows(); ++i) {
    const auto ui = static_cast<std::size_t>(i);
    double fixed = 0.0;
    std::vector<std::int64_t> row(continuous->size(), 0);
    for (std::size_t k = 0; k < continuous->size(); ++k) {
      row[k] = static_cast<std::int64_t>(m.matrix.at(i, (*continuous)[k]));
    }
    for (const Index j : binaries) fixed += m.matrix.at(i, j) * y[static_cast<std::size_t>(j)];
    if (std::isfinite(m.row_lower[ui])) {  // a x >= L - fixed
      lp.a.push_back(row);
      lp.b.push_back(static_cast<std::int64_t>(m.row_lower[ui] - fixed));
    }
    if (std::isfinite(m.row_upper[ui])) {  // -a x >= -(U - fixed)
      std::vector<std::int64_t> negated(row.size());
      for (std::size_t k = 0; k < row.size(); ++k) negated[k] = -row[k];
      lp.a.push_back(negated);
      lp.b.push_back(-static_cast<std::int64_t>(m.row_upper[ui] - fixed));
    }
  }
  lp.num_rows = static_cast<Index>(lp.a.size());
  lp.upper.resize(continuous->size());
  for (std::size_t k = 0; k < continuous->size(); ++k) {
    const double hi = m.col_upper[static_cast<std::size_t>((*continuous)[k])];
    lp.upper[k] = std::isfinite(hi) ? static_cast<std::int64_t>(hi) : oracle::kNoUpperBound;
  }
  lp.c = objective;
  return lp;
}

/// The exact gate on any model whose integer columns are binaries: for every assignment,
/// the largest left side of the cut over the continuous columns, which must not exceed the
/// right-hand side. Returns the assignments checked; `why` names the first that fails.
int exact_gate(const Model& m, const Cut& cut, int* skipped, std::string* why) {
  for (const double c : cut.coeff) EXPECT_EQ(c, std::round(c));
  EXPECT_EQ(cut.rhs, std::round(cut.rhs));
  int binaries = 0;
  for (Index j = 0; j < m.num_cols(); ++j) {
    if (m.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) ++binaries;
  }
  int checked = 0;
  for (std::uint32_t mask = 0; mask < (1u << binaries); ++mask) {
    std::vector<Index> continuous;
    std::vector<std::int64_t> objective;
    {
      std::vector<Index> probe;
      for (Index j = 0; j < m.num_cols(); ++j) {
        if (m.col_type[static_cast<std::size_t>(j)] != VarType::kInteger) probe.push_back(j);
      }
      for (const Index j : probe) {
        objective.push_back(-static_cast<std::int64_t>(cut.coeff[static_cast<std::size_t>(j)]));
      }
    }
    const oracle::GeneratedLp lp = continuous_lp(m, mask, objective, &continuous);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kInfeasible) continue;
    if (exact.status != oracle::OracleStatus::kOptimal) {
      ++*skipped;
      continue;
    }
    ++checked;
    std::int64_t switch_terms = 0;
    int k = 0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      if (m.col_type[static_cast<std::size_t>(j)] != VarType::kInteger) continue;
      if (((mask >> k) & 1u) != 0) {
        switch_terms += static_cast<std::int64_t>(cut.coeff[static_cast<std::size_t>(j)]);
      }
      ++k;
    }
    const Rational largest_left = Rational(0) - exact.objective + Rational(switch_terms);
    const Rational right(static_cast<std::int64_t>(cut.rhs));
    if (largest_left > right) {
      *why = "assignment " + std::to_string(mask) + ": the largest left side " +
             std::to_string(largest_left.to_double()) + " exceeds " +
             std::to_string(right.to_double());
      return checked;
    }
  }
  return checked;
}

TEST(FlowCoverCuts, TheTwoPeriodPathCutComesFromTheAggregatedBalanceRows) {
  // Capacities 6 and 5, demand 2 each. At y = (0.5, 0.2), x = (3, 1), s_1 = 1 both balance
  // rows and both variable upper bounds hold. Period 1 alone gives x_1 - 2 y_1 - s_1 <= 0
  // (cover {1}, lambda 4), violated by 1.0. Eliminating s_1 with the second balance row
  // gives x_1 + x_2 <= 4: cover {1} with lambda 2, and x_2 lifted with g(5) = max(0, 5 -
  // (6 - 2)) = 1, so its switch coefficient is 5 - 1 = 4: x_1 + x_2 - 4 y_1 - 4 y_2 <= 0,
  // violated by 1.2. The more violated aggregate is what comes back, and the gate must
  // pass it: with y_1 = 1 alone, x_1 is at most 4 (x_2 = 0 forces s_1 = 2).
  LotSizing ls;
  ls.u = {6, 5};
  ls.d = {2, 2};
  const Model m = build(ls);
  FlowCoverStats stats;
  const std::vector<Cut> cuts = generate_flow_cover_cuts(m, at({3.0, 1.0, 0.5, 0.2, 1.0}),
                                                         m.col_lower, m.col_upper, &stats);
  ASSERT_FALSE(cuts.empty());
  EXPECT_GE(stats.aggregated, 1) << "no cut came from an aggregate";
  bool path = false;
  for (const Cut& cut : cuts) {
    int skipped = 0;
    std::string why;
    EXPECT_GT(exact_gate(m, cut, &skipped, &why), 0);
    EXPECT_TRUE(why.empty()) << why;
    if (cut.coeff[0] != 0.0 && cut.coeff[1] != 0.0) path = true;
  }
  EXPECT_TRUE(path) << "no cut spans both periods";
}

LotSizing random_lot_sizing(std::mt19937& rng, int periods) {
  std::uniform_int_distribution<int> capacity(3, 9);
  std::uniform_int_distribution<int> demand(1, 4);
  LotSizing ls;
  for (int t = 0; t < periods; ++t) {
    ls.u.push_back(capacity(rng));
    ls.d.push_back(demand(rng));
  }
  return ls;
}

/// A point of the LP relaxation's neighbourhood: switches anywhere in [0, 1], production
/// at a share of what the switch allows, stocks following the balance rows and clamped at
/// zero (a clamped period's balance row is then violated, which is fine for a validity
/// check: the separator's cuts must be valid at every feasible point whatever x* is).
std::vector<double> random_point(std::mt19937& rng, const LotSizing& ls) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const auto T = ls.u.size();
  std::vector<double> x(3 * T - 1, 0.0);
  double stock = 0.0;
  for (std::size_t t = 0; t < T; ++t) {
    x[T + t] = unit(rng);
    x[t] = unit(rng) * static_cast<double>(ls.u[t]) * x[T + t];
    stock = std::max(0.0, stock + x[t] - static_cast<double>(ls.d[t]));
    if (t + 1 < T) x[2 * T + t] = stock;
  }
  return x;
}

TEST(FlowCoverCuts, NoAggregatedCutRemovesAnyProductionPlanExactly) {
  std::mt19937 rng(1158);
  int cuts_checked = 0;
  int aggregated = 0;
  int assignments = 0;
  int skipped = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const LotSizing ls = random_lot_sizing(rng, 2 + trial % 4);
    const Model m = build(ls);
    for (int probe = 0; probe < 2; ++probe) {
      FlowCoverStats stats;
      const std::vector<Cut> cuts = generate_flow_cover_cuts(m, at(random_point(rng, ls)),
                                                             m.col_lower, m.col_upper, &stats);
      aggregated += stats.aggregated;
      for (const Cut& cut : cuts) {
        std::string why;
        assignments += exact_gate(m, cut, &skipped, &why);
        ASSERT_TRUE(why.empty()) << "trial " << trial << ": " << why;
        ++cuts_checked;
      }
    }
  }
  EXPECT_GT(cuts_checked, 200) << "the sweep should exercise the separator";
  EXPECT_GT(aggregated, 50) << "the sweep should exercise the aggregation";
  EXPECT_GT(assignments, 1000);
  EXPECT_LT(skipped, assignments / 10)
      << "too many oracle overflows for the sweep to mean much";
  std::printf(
      "[  INFO    ] flow covers on lot-sizing chains: %d cuts, %d from aggregates, "
      "%d assignments gated\n",
      cuts_checked, aggregated, assignments);
}

TEST(FlowCoverCuts, TheSearchOnLotSizingChainsAgreesWithTheExactOptimum) {
  // Fixed and holding costs, the family on with the root loop so the aggregates are
  // separated at several points, against the exact optimum over every switch assignment.
  std::mt19937 rng(11580);
  std::uniform_int_distribution<int> cost(1, 9);
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    LotSizing ls = random_lot_sizing(rng, 3 + trial % 3);
    for (std::size_t t = 0; t < ls.u.size(); ++t) ls.fixed_cost.push_back(5 * cost(rng));
    for (std::size_t t = 0; t + 1 < ls.u.size(); ++t) ls.holding_cost.push_back(cost(rng));
    const Model m = build(ls);
    const auto T = ls.u.size();
    bool any = false;
    Rational best(0);
    for (std::uint32_t mask = 0; mask < (1u << T); ++mask) {
      std::vector<Index> continuous;
      std::vector<std::int64_t> objective;
      for (Index j = 0; j < m.num_cols(); ++j) {
        if (m.col_type[static_cast<std::size_t>(j)] != VarType::kInteger) {
          objective.push_back(
              static_cast<std::int64_t>(m.col_cost[static_cast<std::size_t>(j)]));
        }
      }
      const oracle::GeneratedLp lp = continuous_lp(m, mask, objective, &continuous);
      const oracle::OracleResult exact = oracle::solve_exact(lp);
      if (exact.status != oracle::OracleStatus::kOptimal) continue;
      std::int64_t fixed = 0;
      for (std::size_t t = 0; t < T; ++t) {
        if (((mask >> t) & 1u) != 0) fixed += ls.fixed_cost[t];
      }
      const Rational value = exact.objective + Rational(fixed);
      if (!any || value < best) best = value;
      any = true;
    }
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("enable_root_cuts", true);
    options.set_bool("enable_flow_cover_cuts", true);
    options.set_bool("root_cut_loop", true);
    const Solution solved = solve(m, options);
    ++compared;
    if (!any) {  // the capacities cannot meet the demand: the search must say so
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
          << "trial " << trial << ": " << solved.message;
      continue;
    }
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best.to_double(), 1e-6 * std::max(1.0, best.to_double()))
        << "trial " << trial;
  }
  EXPECT_EQ(compared, 40);
}

}  // namespace
}  // namespace sankhya::mip
