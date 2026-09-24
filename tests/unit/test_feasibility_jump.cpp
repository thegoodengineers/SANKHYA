// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Feasibility Jump (#506).
//
// The heuristic's own claims are checked here independently of it: every point it returns is
// re-measured against every row, bound and integrality in this file, on hand models where a
// point must be found, on models with no integer point where none may be, and over random
// MILPs. Then the search with mip_heur_fj=on is judged by the exact rational oracle, which is
// the property that matters: a heuristic may waste time, never change an answer.

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/feasibility_jump.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

/// Rows given densely; every column in [lower, upper], integer where `integer` says so.
Model make_model(const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                 const std::vector<double>& cost, const std::vector<double>& lower,
                 const std::vector<double>& upper, const std::vector<bool>& integer) {
  Model m;
  const auto n = cost.size();
  m.col_cost = cost;
  m.col_lower = lower;
  m.col_upper = upper;
  for (std::size_t j = 0; j < n; ++j) {
    m.col_type.push_back(integer[j] ? VarType::kInteger : VarType::kContinuous);
  }
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

/// Independent of the code under test: integral, inside the box, every row satisfied to the
/// same absolute tolerance offer_incumbent() applies.
bool feasible(const Model& m, const std::vector<double>& x, std::string* why) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) {
    *why = "wrong length";
    return false;
  }
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger && x[j] != std::round(x[j])) {
      *why = "column " + std::to_string(j) + " fractional";
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) {
      *why = "column " + std::to_string(j) + " outside its box";
      return false;
    }
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-7 || activity > m.row_upper[u] + 1e-7) {
      *why = "row " + std::to_string(i) + " violated, activity " + std::to_string(activity);
      return false;
    }
  }
  return true;
}

double objective(const Model& m, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) value += m.col_cost[j] * x[j];
  return value;
}

FeasibilityJumpResult run(const Model& m, Count work = 200000, std::uint64_t seed = 1) {
  FeasibilityJumpSettings settings;
  settings.work_limit = work;
  settings.seed = seed;
  return feasibility_jump(m, feasibility_jump_zero_start(m), settings);
}

void expect_all_feasible(const Model& m, const FeasibilityJumpResult& r) {
  for (std::size_t k = 0; k < r.points.size(); ++k) {
    std::string why;
    EXPECT_TRUE(feasible(m, r.points[k], &why)) << "point " << k << ": " << why;
  }
}

TEST(FeasibilityJump, FindsACoverFromTheZeroPoint) {
  // Six rows each needing one of its binaries, costs one: zero violates every row.
  const Model m = make_model({{1, 1, 0, 0, 0, 0, 0, 0},
                              {0, 1, 1, 0, 0, 0, 0, 1},
                              {0, 0, 1, 1, 0, 0, 0, 0},
                              {1, 0, 0, 0, 1, 1, 0, 0},
                              {0, 0, 0, 0, 0, 1, 1, 0},
                              {0, 0, 0, 1, 0, 0, 1, 1}},
                             std::vector<double>(6, 1.0), std::vector<double>(6, kInfinity),
                             std::vector<double>(8, 1.0), std::vector<double>(8, 0.0),
                             std::vector<double>(8, 1.0), std::vector<bool>(8, true));
  const FeasibilityJumpResult r = run(m);
  ASSERT_FALSE(r.points.empty()) << r.moves << " moves, " << r.weight_updates << " updates";
  expect_all_feasible(m, r);
  for (std::size_t k = 1; k < r.points.size(); ++k) {
    EXPECT_LT(objective(m, r.points[k]), objective(m, r.points[k - 1]))
        << "each point returned improves on the last";
  }
  EXPECT_GE(objective(m, r.points.back()), 3.0) << "better than the optimum of 3";
}

TEST(FeasibilityJump, SolvesEqualitiesOverGeneralIntegers) {
  // x + y + z = 7, x - y = 1 and y + z >= 3 over [0, 20]: every solution has values strictly
  // inside the boxes, so only jumps to breakpoints, not to bounds, can reach one.
  //
  // What it does NOT do, and this test does not ask: 3x + 5y + 7z = 41 with x + y + z <= 9
  // (solutions (2, 0, 5) and (1, 2, 4)) was not solved within this budget. A point such as
  // (0, 0, 6), one unit off the equality, is a trap of this kind: every one-column move makes
  // the only violated row worse, so raising that row's weight changes nothing. The budget
  // runs out and nothing is claimed.
  const Model m =
      make_model({{1, 1, 1}, {1, -1, 0}, {0, 1, 1}}, {7.0, 1.0, 3.0}, {7.0, 1.0, kInfinity},
                 {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {20.0, 20.0, 20.0}, {true, true, true});
  const FeasibilityJumpResult r = run(m);
  ASSERT_FALSE(r.points.empty());
  expect_all_feasible(m, r);
}

TEST(FeasibilityJump, MixesIntegerAndContinuousColumns) {
  // x integer in [0, 10], y continuous in [0, 0.3]: x + y >= 2.5 needs x >= 3, and the
  // equality 2x - 5y = 5.5 then fixes y = 0.1 at x = 3.
  const Model m = make_model({{1, 1}, {2, -5}}, {2.5, 5.5}, {kInfinity, 5.5}, {1.0, 1.0},
                             {0.0, 0.0}, {10.0, 0.3}, {true, false});
  const FeasibilityJumpResult r = run(m);
  ASSERT_FALSE(r.points.empty());
  expect_all_feasible(m, r);
  EXPECT_EQ(r.points.front()[0], 3.0);
  EXPECT_NEAR(r.points.front()[1], 0.1, 1e-12);
}

TEST(FeasibilityJump, FindsAnAssignment) {
  // A 4 x 4 assignment: every row and every column of the matrix of binaries sums to one.
  constexpr int k = 4;
  std::vector<std::vector<double>> rows;
  for (int r = 0; r < k; ++r) {
    std::vector<double> row(k * k, 0.0);
    for (int c = 0; c < k; ++c) row[static_cast<std::size_t>(r * k + c)] = 1.0;
    rows.push_back(row);
  }
  for (int c = 0; c < k; ++c) {
    std::vector<double> row(k * k, 0.0);
    for (int r = 0; r < k; ++r) row[static_cast<std::size_t>(r * k + c)] = 1.0;
    rows.push_back(row);
  }
  std::vector<double> cost;
  for (int e = 0; e < k * k; ++e) cost.push_back(static_cast<double>((e * 7) % 11));
  const Model m =
      make_model(rows, std::vector<double>(2 * k, 1.0), std::vector<double>(2 * k, 1.0), cost,
                 std::vector<double>(k * k, 0.0), std::vector<double>(k * k, 1.0),
                 std::vector<bool>(k * k, true));
  const FeasibilityJumpResult r = run(m);
  ASSERT_FALSE(r.points.empty());
  expect_all_feasible(m, r);
}

TEST(FeasibilityJump, ReturnsNothingWhenNoIntegerPointExists) {
  // 2x + 2y = 3 has real solutions and no integer one; x + y >= 3 over two binaries has
  // neither. The budget runs out and nothing is claimed.
  const Model parity =
      make_model({{2, 2}}, {3.0}, {3.0}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  EXPECT_TRUE(run(parity, 50000).points.empty());
  const Model cover = make_model({{1, 1}}, {3.0}, {kInfinity}, {1.0, 1.0}, {0.0, 0.0},
                                 {1.0, 1.0}, {true, true});
  EXPECT_TRUE(run(cover, 50000).points.empty());
  // Integer bounds with no integer inside them: nothing, at once.
  const Model empty_box = make_model({{1}}, {0.0}, {kInfinity}, {1.0}, {0.2}, {0.8}, {true});
  const FeasibilityJumpResult r = run(empty_box);
  EXPECT_TRUE(r.points.empty());
  EXPECT_EQ(r.work, 0);
}

TEST(FeasibilityJump, ImprovesTheObjectiveOfAMaximisation) {
  // max 10a + 13b + 7c + 8d s.t. 5a + 7b + 4c + 4d <= 12, binaries. Zero is feasible and
  // worth nothing; the objective phase must find something better.
  Model m = make_model({{5, 7, 4, 4}}, {-kInfinity}, {12.0}, {10.0, 13.0, 7.0, 8.0},
                       {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0}, {true, true, true, true});
  m.sense = ObjSense::kMaximize;
  const FeasibilityJumpResult r = run(m);
  ASSERT_GE(r.points.size(), 2U);
  expect_all_feasible(m, r);
  EXPECT_EQ(objective(m, r.points.front()), 0.0);
  for (std::size_t k = 1; k < r.points.size(); ++k) {
    EXPECT_GT(objective(m, r.points[k]), objective(m, r.points[k - 1]));
  }
  EXPECT_LE(objective(m, r.points.back()), 23.0) << "the optimum is 23, a and b";
}

TEST(FeasibilityJump, IsReproducibleForASeed) {
  const Model m =
      make_model({{1, 1, 1}, {1, -1, 0}, {0, 1, 1}}, {7.0, 1.0, 3.0}, {7.0, 1.0, kInfinity},
                 {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {20.0, 20.0, 20.0}, {true, true, true});
  const FeasibilityJumpResult a = run(m, 100000, 7);
  const FeasibilityJumpResult b = run(m, 100000, 7);
  EXPECT_EQ(a.points, b.points);
  EXPECT_EQ(a.moves, b.moves);
  EXPECT_EQ(a.work, b.work);
}

oracle::GeneratedLp random_milp(std::mt19937_64& rng, int trial) {
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  oracle::GeneratedLp lp = oracle::random_lp(rng, config);
  lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
  for (Index j = 0; j < lp.num_cols; ++j) {
    lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 3 == 0 ? 0 : 1;
  }
  return lp;
}

Model as_milp(const oracle::GeneratedLp& lp) {
  Model model = oracle::to_model(lp);
  for (Index j = 0; j < lp.num_cols; ++j) {
    if (lp.integral[static_cast<std::size_t>(j)] != 0) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
  return model;
}

TEST(FeasibilityJump, EveryPointOnRandomMilpsIsFeasible) {
  // The fuzz #506 asks for: whatever FJ returns on random MILPs is feasible, and on a MILP
  // the exact oracle proves infeasible it returns nothing.
  std::mt19937_64 rng(5060);
  int with_points = 0;
  int infeasible_checked = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const Model model = as_milp(lp);
    const FeasibilityJumpResult r = run(model, 100000, static_cast<std::uint64_t>(trial));
    for (std::size_t k = 0; k < r.points.size(); ++k) {
      std::string why;
      EXPECT_TRUE(feasible(model, r.points[k], &why))
          << "trial " << trial << " point " << k << ": " << why << "\n"
          << lp.to_text();
    }
    if (!r.points.empty()) ++with_points;
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      ++infeasible_checked;
      EXPECT_TRUE(r.points.empty()) << "trial " << trial << ": a point on an infeasible MILP";
    }
  }
  EXPECT_GT(with_points, 50) << "the generator should give FJ feasible models to solve";
  EXPECT_GT(infeasible_checked, 20);
}

TEST(FeasibilityJump, TheSearchWithItOnAgreesWithTheExactOracle) {
  // End to end: FJ before the root LP and from the root relaxation, its points through
  // offer_incumbent(), and the answer judged by exact arithmetic.
  std::mt19937_64 rng(50600);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("mip_heur_fj", "on");
  // A closed tree, not the default 1e-4 gap: FJ hands the search good incumbents early, and
  // an answer inside the gap target is not the exact optimum this compares against.
  options.set_double("mip_relative_gap", 0.0);
  options.set_double("mip_absolute_gap", 0.0);
  int compared = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    const Model model = as_milp(lp);
    const Solution got = solve(model, options);
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      EXPECT_EQ(got.status, SolveStatus::kInfeasible) << "trial " << trial << "\n"
                                                      << lp.to_text();
    } else {
      ASSERT_EQ(got.status, SolveStatus::kOptimal)
          << "trial " << trial << ": " << got.message << "\n"
          << lp.to_text();
      const double want = exact.objective.to_double();
      EXPECT_NEAR(got.objective, want, 1e-6 * std::max(1.0, std::fabs(want)))
          << "trial " << trial << "\n"
          << lp.to_text();
      std::string why;
      EXPECT_TRUE(feasible(model, got.col_value, &why)) << "trial " << trial << ": " << why;
    }
    ++compared;
  }
  EXPECT_GT(compared, 120) << "most generated MILPs should reach a verdict in the oracle";
}

}  // namespace
}  // namespace sankhya::mip
