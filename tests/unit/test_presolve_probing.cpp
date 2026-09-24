// SPDX-License-Identifier: Apache-2.0
// SANKHYA - binary probing and the clique table (#512).
//
// Every deduction probing makes - a fixing, a tightened bound, a conflict between two
// literals, a merged clique - must hold at every feasible integer point, or the search is
// handed a model with a point missing and proves the second-best answer optimal. So the
// hand models check each kind of deduction fires where it should, an enumeration of every
// integer point of random small models checks every deduction against every point, the clique
// separator is checked the same way with the probed conflicts fed in, and random MILPs are
// solved with probing on against the exact rational branch and bound.

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/combinatorial_cuts.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "presolve/presolve.hpp"
#include "presolve/probing.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr std::int64_t kWork = 1000000;

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& col_lower,
            const std::vector<double>& col_upper, const std::vector<bool>& integer) {
  Model model;
  const auto n = static_cast<Index>(col_lower.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (std::size_t j = 0; j < integer.size(); ++j) {
    if (integer[j]) model.col_type[j] = VarType::kInteger;
  }
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
  EXPECT_EQ(model.validate(), "");
  return model;
}

presolve::ProbingResult probe(const Model& model) {
  return presolve::probe_binaries(model, model.col_lower, model.col_upper, kWork);
}

bool has_conflict(const presolve::ProbingResult& result, Index a, Index b) {
  for (const auto& [p, q] : result.conflicts) {
    if ((p == a && q == b) || (p == b && q == a)) return true;
  }
  return false;
}

TEST(PresolveProbing, ASetPackingRowGivesPairwiseConflictsAndOneClique) {
  // x0 + x1 + x2 <= 1: x_j = 1 fixes the other two to 0, so every pair of positive literals
  // conflicts, and the three merge into one clique.
  const Model model =
      build({{1, 1, 1}}, {-kInf}, {1}, {0, 0, 0}, {1, 1, 1}, {true, true, true});
  const presolve::ProbingResult result = probe(model);
  EXPECT_EQ(result.probed, 3);
  EXPECT_TRUE(has_conflict(result, 0, 1));
  EXPECT_TRUE(has_conflict(result, 0, 2));
  EXPECT_TRUE(has_conflict(result, 1, 2));
  ASSERT_EQ(result.cliques.size(), 1u);
  EXPECT_EQ(result.cliques[0], (std::vector<Index>{0, 1, 2}));
  EXPECT_EQ(result.largest_clique, 3);
  EXPECT_EQ(result.fixings, 0);
}

TEST(PresolveProbing, AnImplicationIsAConflictWithAComplementedLiteral) {
  // x0 <= x1: x0 = 1 forces x1 = 1, so "x0 = 1" and "x1 = 0" never hold together. Literal
  // n + 1 is "x1 = 0".
  const Model model = build({{1, -1}}, {-kInf}, {0}, {0, 0}, {1, 1}, {true, true});
  const presolve::ProbingResult result = probe(model);
  EXPECT_TRUE(has_conflict(result, 0, 2 + 1));
  EXPECT_FALSE(has_conflict(result, 0, 1)) << "both at 1 is feasible";
  EXPECT_GT(result.implications, 0);
}

TEST(PresolveProbing, OneInfeasibleValueFixesTheOther) {
  // 2 x0 + y >= 1.5 with y in [0, 1]: x0 = 0 leaves at most 1, so x0 = 1.
  const Model model = build({{2, 1}}, {1.5}, {kInf}, {0, 0}, {1, 1}, {true, false});
  const presolve::ProbingResult result = probe(model);
  EXPECT_EQ(result.fixings, 1);
  EXPECT_EQ(result.col_lower[0], 1.0);
  EXPECT_EQ(result.col_upper[0], 1.0);
  EXPECT_FALSE(result.infeasible);
}

TEST(PresolveProbing, BothValuesAgreeingTightensAColumn) {
  // y - 3 x0 <= 2 and y + 3 x0 <= 5, y in [0, 10]: x0 = 0 gives y <= 2, x0 = 1 gives y <= 2,
  // so y <= 2 in every feasible point, though the LP alone only says y <= 3.5.
  const Model model =
      build({{-3, 1}, {3, 1}}, {-kInf, -kInf}, {2, 5}, {0, 0}, {1, 10}, {true, false});
  const presolve::ProbingResult result = probe(model);
  EXPECT_GE(result.tightenings, 1);
  // Loosened by the rounding margin (kProbingSafety times the row's magnitude over |a|, about
  // 1.1e-8 here), never tightened past 2.
  EXPECT_NEAR(result.col_upper[1], 2.0, 1e-7);
  EXPECT_GE(result.col_upper[1], 2.0) << "a propagated continuous bound is never tighter";
}

TEST(PresolveProbing, BothValuesInfeasibleIsAnInfeasibleModel) {
  // 2 x0 = 1 has no integer point.
  const Model model = build({{2, 1}}, {1}, {1}, {0, 0}, {1, 0}, {true, false});
  EXPECT_TRUE(probe(model).infeasible);
}

TEST(PresolveProbing, TheWorkLimitStopsProbingAndSaysSo) {
  const Model model =
      build({{1, 1, 1}}, {-kInf}, {1}, {0, 0, 0}, {1, 1, 1}, {true, true, true});
  const presolve::ProbingResult result =
      presolve::probe_binaries(model, model.col_lower, model.col_upper, 1);
  EXPECT_TRUE(result.work_limit_reached);
  EXPECT_LT(result.probed, 3);
}

// ---- Enumeration --------------------------------------------------------------------------

void for_each_point(const std::vector<double>& lo, const std::vector<double>& hi,
                    const std::function<void(const std::vector<double>&)>& visit) {
  std::vector<double> x(lo);
  std::function<void(std::size_t)> walk = [&](std::size_t j) {
    if (j == x.size()) {
      visit(x);
      return;
    }
    for (double v = lo[j]; v <= hi[j]; v += 1.0) {
      x[j] = v;
      walk(j + 1);
    }
  };
  walk(0);
}

bool feasible(const Model& model, const std::vector<double>& x) {
  std::vector<double> activity(static_cast<std::size_t>(model.num_rows()), 0.0);
  model.matrix.multiply(x.data(), activity.data());
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto r = static_cast<std::size_t>(i);
    if (activity[r] < model.row_lower[r] - 1e-9 || activity[r] > model.row_upper[r] + 1e-9) {
      return false;
    }
  }
  return true;
}

double literal_value(const std::vector<double>& x, Index literal, Index n) {
  return literal < n ? x[static_cast<std::size_t>(literal)]
                     : 1.0 - x[static_cast<std::size_t>(literal - n)];
}

/// Mostly binaries with a few general integers, rows of every sense with mixed signs:
/// packing, covering, implications and knapsacks all appear.
Model random_model(std::mt19937_64& rng) {
  std::uniform_int_distribution<int> cols(3, 7);
  std::uniform_int_distribution<int> nrows(1, 5);
  std::uniform_int_distribution<int> coef(-5, 5);
  std::uniform_int_distribution<int> coin(0, 3);
  const int n = cols(rng);
  const int m = nrows(rng);
  std::vector<std::vector<double>> rows(static_cast<std::size_t>(m));
  std::vector<double> row_lower, row_upper;
  std::vector<double> lo(static_cast<std::size_t>(n), 0.0),
      hi(static_cast<std::size_t>(n), 1.0);
  for (int j = 0; j < n; ++j) {
    if (coin(rng) == 0) hi[static_cast<std::size_t>(j)] = 2.0;
  }
  for (auto& row : rows) {
    double reach = 0.0;
    for (int j = 0; j < n; ++j) {
      const int v = coin(rng) == 0 ? 0 : coef(rng);
      row.push_back(v);
      reach += std::abs(v) * hi[static_cast<std::size_t>(j)];
    }
    std::uniform_int_distribution<int> rhs(-static_cast<int>(reach / 3),
                                           static_cast<int>(reach / 3));
    const double b = rhs(rng);
    switch (coin(rng)) {
      case 0:
        row_lower.push_back(-kInf);
        row_upper.push_back(b);
        break;
      case 1:
        row_lower.push_back(b);
        row_upper.push_back(kInf);
        break;
      case 2:
        row_lower.push_back(b);
        row_upper.push_back(b + 2);
        break;
      default:
        row_lower.push_back(-kInf);
        row_upper.push_back(b + 1);
        break;
    }
  }
  return build(rows, row_lower, row_upper, lo, hi,
               std::vector<bool>(static_cast<std::size_t>(n), true));
}

TEST(PresolveProbing, EveryDeductionHoldsAtEveryIntegerPoint) {
  std::mt19937_64 rng(5120);
  Count fixings = 0, tightenings = 0, conflicts = 0, cliques = 0;
  int infeasible_models = 0;
  for (int trial = 0; trial < 5000; ++trial) {
    const Model model = random_model(rng);
    const Index n = model.num_cols();
    const presolve::ProbingResult result = probe(model);
    fixings += result.fixings;
    tightenings += result.tightenings;
    conflicts += static_cast<Count>(result.conflicts.size());
    cliques += static_cast<Count>(result.cliques.size());
    if (result.infeasible) ++infeasible_models;
    for_each_point(model.col_lower, model.col_upper, [&](const std::vector<double>& x) {
      if (!feasible(model, x)) return;
      ASSERT_FALSE(result.infeasible)
          << "trial " << trial << ": a feasible model called infeasible";
      for (Index j = 0; j < n; ++j) {
        const auto u = static_cast<std::size_t>(j);
        ASSERT_GE(x[u], result.col_lower[u]) << "trial " << trial << ": column " << j;
        ASSERT_LE(x[u], result.col_upper[u]) << "trial " << trial << ": column " << j;
      }
      for (const auto& [a, b] : result.conflicts) {
        ASSERT_LE(literal_value(x, a, n) + literal_value(x, b, n), 1.0)
            << "trial " << trial << ": conflict " << a << ", " << b;
      }
      for (const auto& clique : result.cliques) {
        double sum = 0.0;
        for (const Index l : clique) sum += literal_value(x, l, n);
        ASSERT_LE(sum, 1.0) << "trial " << trial;
      }
    });
    if (::testing::Test::HasFatalFailure()) return;
  }
  std::printf("  %lld fixings, %lld tightenings, %lld conflicts, %lld cliques, %d infeasible\n",
              static_cast<long long>(fixings), static_cast<long long>(tightenings),
              static_cast<long long>(conflicts), static_cast<long long>(cliques),
              infeasible_models);
  EXPECT_GT(fixings, 100);
  EXPECT_GT(tightenings, 100);
  EXPECT_GT(conflicts, 1000);
  EXPECT_GT(cliques, 500);
  EXPECT_GT(infeasible_models, 10);
}

TEST(PresolveProbing, CliqueCutsFromProbedConflictsNeverCutOffAnIntegerPoint) {
  // The separator with the probed conflicts fed in, including complemented literals, at a
  // random fractional point: every cut it returns must hold at every feasible integer point.
  std::mt19937_64 rng(5121);
  std::uniform_real_distribution<double> fraction(0.0, 1.0);
  Count cuts_checked = 0;
  Count complemented = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const Model model = random_model(rng);
    const presolve::ProbingResult result = probe(model);
    if (result.infeasible || result.conflicts.empty()) continue;
    Solution point;
    for (Index j = 0; j < model.num_cols(); ++j) {
      point.col_value.push_back(fraction(rng) * model.col_upper[static_cast<std::size_t>(j)]);
    }
    const std::vector<mip::Cut> cuts = mip::generate_clique_cuts(
        model, point, model.col_lower, model.col_upper, nullptr, &result.conflicts);
    for (const mip::Cut& cut : cuts) {
      bool has_negative = false;
      for (const double c : cut.coeff) has_negative = has_negative || c < 0.0;
      if (has_negative) ++complemented;
      ++cuts_checked;
      for_each_point(model.col_lower, model.col_upper, [&](const std::vector<double>& x) {
        if (!feasible(model, x)) return;
        double lhs = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) lhs += cut.coeff[j] * x[j];
        ASSERT_LE(lhs, cut.rhs + 1e-9) << "trial " << trial << ": a clique cut removed a point";
      });
      if (::testing::Test::HasFatalFailure()) return;
    }
  }
  std::printf("  %lld cuts checked, %lld with a complemented literal\n",
              static_cast<long long>(cuts_checked), static_cast<long long>(complemented));
  EXPECT_GT(cuts_checked, 300);
  EXPECT_GT(complemented, 50);
}

// ---- Through presolve and the search -------------------------------------------------------

Options with_probing(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_probing", on);
  options.set_int("node_limit", 100000);
  return options;
}

TEST(PresolveProbing, TheOptionReachesPresolveAndOffMeansOff) {
  Model model = build({{2, 1, 0}, {1, 1, 1}}, {1.5, -kInf}, {kInf, 1.5}, {0, 0, 0}, {1, 1, 1},
                      {true, false, true});
  model.col_cost = {1, 1, -1};
  Logger logger(nullptr);
  const presolve::Result on = presolve::presolve(model, with_probing(true), logger);
  const presolve::Result off = presolve::presolve(model, with_probing(false), logger);
  EXPECT_GT(on.report.probing_fixings, 0);
  EXPECT_EQ(off.report.probing_fixings, 0);
  EXPECT_EQ(off.report.probing_conflicts, 0);
  const Solution with = solve(model, with_probing(true));
  const Solution without = solve(model, with_probing(false));
  ASSERT_EQ(with.status, SolveStatus::kOptimal) << with.message;
  ASSERT_EQ(without.status, SolveStatus::kOptimal) << without.message;
  EXPECT_NEAR(with.objective, without.objective, 1e-9);
  EXPECT_LE(with.primal_infeasibility, 1e-9);
}

bool agree(const oracle::OracleResult& exact, const Solution& got, std::string* why) {
  if (exact.status == oracle::OracleStatus::kInfeasible) {
    if (got.status == SolveStatus::kInfeasible) return true;
    *why = "exactly infeasible, the solver said " + std::string(to_string(got.status));
    return false;
  }
  if (got.status != SolveStatus::kOptimal) {
    *why = "the exact optimum exists but the solver said " +
           std::string(to_string(got.status)) + ": " + got.message;
    return false;
  }
  const double expected = exact.objective.to_double();
  if (std::fabs(got.objective - expected) > 1e-6 * std::max(1.0, std::fabs(expected))) {
    *why = "objectives differ: " + std::to_string(got.objective) + " against " +
           std::to_string(expected);
    return false;
  }
  return true;
}

TEST(PresolveProbing, RandomMilpsAgreeWithTheExactOracle) {
  // Binaries (integer columns bounded by 1) mixed with continuous columns, solved with
  // probing on - in presolve and in the clique separator - against the exact branch and
  // bound; the exact optimum must also satisfy every probed conflict and bound.
  std::mt19937_64 rng(51200);
  oracle::GeneratorConfig config;
  config.max_rows = 6;
  config.max_cols = 7;
  config.bounded_column_probability = 0.8;
  int compared = 0;
  Count conflicts = 0;
  for (int trial = 0; trial < 300; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if ((j + trial) % 3 != 0) {
        lp.integral[u] = 1;
        lp.upper[u] = 1;
      }
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    const presolve::ProbingResult result = probe(model);
    conflicts += static_cast<Count>(result.conflicts.size());
    if (exact.status == oracle::OracleStatus::kOptimal) {
      std::vector<double> x;
      for (const auto& v : exact.x) x.push_back(v.to_double());
      EXPECT_FALSE(result.infeasible) << "trial " << trial;
      const Index n = model.num_cols();
      for (const auto& [a, b] : result.conflicts) {
        EXPECT_LE(literal_value(x, a, n) + literal_value(x, b, n), 1.0 + 1e-9)
            << "trial " << trial << ": the exact optimum breaks a probed conflict";
      }
      for (Index j = 0; j < n && !result.infeasible; ++j) {
        const auto u = static_cast<std::size_t>(j);
        EXPECT_GE(x[u], result.col_lower[u] - 1e-9) << "trial " << trial;
        EXPECT_LE(x[u], result.col_upper[u] + 1e-9) << "trial " << trial;
      }
    }
    const Solution got = solve(model, with_probing(true));
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 200);
  EXPECT_GT(conflicts, 50);
}

}  // namespace
}  // namespace sankhya
