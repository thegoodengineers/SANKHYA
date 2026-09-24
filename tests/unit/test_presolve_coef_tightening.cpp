// SPDX-License-Identifier: Apache-2.0
// SANKHYA - coefficient tightening and big-M strengthening in presolve (#511).
//
// The reduction's whole claim is that the integer points of the model are unchanged while
// the LP relaxation gets tighter. A tightening that is slightly too strong removes an
// integer point and the search then proves a worse answer optimal with no symptom, so the
// claim is checked three ways: on hand models whose tightened rows can be read off, by
// enumerating every integer point of random small models before and after, and by solving
// random MILPs with the reduction on against the exact rational branch and bound, whose
// optimal point must also satisfy the tightened model (the debug-solution check).

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "presolve/coef_tightening.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integer) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
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

Options with_tightening(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_coefficient_tightening", on);
  options.set_int("node_limit", 100000);
  return options;
}

TEST(PresolveCoefficientTightening, AKnapsackRowBecomesASetPackingRow) {
  // 5 x0 + 3 x1 <= 6, both binary. M = 8, g = 2: each coefficient drops to 2 and the bound
  // by 3 and then by 1, so the row is 2 x0 + 2 x1 <= 2 - at most one of the two, which is
  // exactly what the integer points say. The LP point (1, 1/3) is gone.
  Model model = build({{5, 3}}, {-kInf}, {6}, {-1, -1}, {0, 0}, {1, 1}, {true, true});
  const presolve::CoefficientTighteningStats stats = presolve::tighten_coefficients(&model);
  EXPECT_EQ(stats.coefficients_tightened, 2);
  EXPECT_EQ(stats.rows_tightened, 1);
  EXPECT_EQ(model.matrix.at(0, 0), 2.0);
  EXPECT_EQ(model.matrix.at(0, 1), 2.0);
  EXPECT_EQ(model.row_upper[0], 2.0);
  EXPECT_TRUE(std::isinf(model.row_lower[0]));
}

TEST(PresolveCoefficientTightening, BigMBecomesTheColumnsOwnBound) {
  // x - 100 y <= 0, x continuous in [0, 10], y binary: the textbook loose big-M. M = 10,
  // b = 0, g = 10, and y's coefficient drops to -10: x <= 10 y. x is never tightened.
  Model model = build({{1, -100}}, {-kInf}, {0}, {1, 5}, {0, 0}, {10, 1}, {false, true});
  const auto stats = presolve::tighten_coefficients(&model);
  EXPECT_EQ(stats.coefficients_tightened, 1);
  EXPECT_EQ(model.matrix.at(0, 0), 1.0);
  EXPECT_EQ(model.matrix.at(0, 1), -10.0);
  EXPECT_EQ(model.row_upper[0], 0.0);
  // The same row written the other way round, as a >= row.
  Model ge = build({{-1, 100}}, {0}, {kInf}, {1, 5}, {0, 0}, {10, 1}, {false, true});
  EXPECT_EQ(presolve::tighten_coefficients(&ge).coefficients_tightened, 1);
  EXPECT_EQ(ge.matrix.at(0, 1), 10.0);
  EXPECT_EQ(ge.row_lower[0], 0.0);
}

TEST(PresolveCoefficientTightening, PropagationFindsTheBoundTheBigMNeeds) {
  // x has no upper bound of its own; x + z <= 10 with z >= 0 implies x <= 10, which is
  // written into the model, and then x - 100 y <= 0 tightens to x - 10 y <= 0.
  Model model = build({{1, 0, 1}, {1, -100, 0}}, {-kInf, -kInf}, {10, 0}, {1, 5, 0}, {0, 0, 0},
                      {kInf, 1, kInf}, {false, true, false});
  const auto stats = presolve::tighten_coefficients(&model);
  EXPECT_EQ(model.col_upper[0], 10.0) << "the bound the tightened row relies on is kept";
  EXPECT_GE(stats.bounds_tightened, 1);
  EXPECT_EQ(stats.coefficients_tightened, 1);
  EXPECT_EQ(model.matrix.at(1, 1), -10.0);
}

TEST(PresolveCoefficientTightening, AGeneralIntegerStepsByWholeUnits) {
  // 10 x + y <= 23, x integer in [0, 2], y in [0, 5]. M = 25, g = 2: 2 x + y <= 7. At x = 2
  // both say y <= 3; at x = 1 and x = 0 both are slack on the whole box.
  Model model = build({{10, 1}}, {-kInf}, {23}, {0, 0}, {0, 0}, {2, 5}, {true, false});
  EXPECT_EQ(presolve::tighten_coefficients(&model).coefficients_tightened, 1);
  EXPECT_EQ(model.matrix.at(0, 0), 2.0);
  EXPECT_EQ(model.row_upper[0], 7.0);
  // A negative coefficient is maximised at the LOWER bound: -10 x + y <= -7 with x in
  // [1, 3]. M = -10 + 5 = -5, g = 2: -2 x + y <= 1. At x = 1 both say y <= 3.
  Model negative = build({{-10, 1}}, {-kInf}, {-7}, {0, 0}, {1, 0}, {3, 5}, {true, false});
  EXPECT_EQ(presolve::tighten_coefficients(&negative).coefficients_tightened, 1);
  EXPECT_EQ(negative.matrix.at(0, 0), -2.0);
  EXPECT_EQ(negative.row_upper[0], 1.0);
}

TEST(PresolveCoefficientTightening, StaysOutWhereTheArgumentDoesNotHold) {
  // An equality: tightening one side would loosen the other.
  Model equality = build({{5, 3}}, {3}, {3}, {0, 0}, {0, 0}, {1, 1}, {true, true});
  EXPECT_EQ(presolve::tighten_coefficients(&equality).coefficients_tightened, 0);
  // A ranged row whose other side binds on the box: 1 <= 5 x0 + 3 x1 <= 6.
  Model ranged = build({{5, 3}}, {1}, {6}, {0, 0}, {0, 0}, {1, 1}, {true, true});
  EXPECT_EQ(presolve::tighten_coefficients(&ranged).coefficients_tightened, 0);
  // ... but when the box already implies the other side it is dropped and the row tightened.
  Model implied = build({{5, 3}}, {0}, {6}, {0, 0}, {0, 0}, {1, 1}, {true, true});
  EXPECT_EQ(presolve::tighten_coefficients(&implied).coefficients_tightened, 2);
  EXPECT_TRUE(std::isinf(implied.row_lower[0]));
  // A continuous column's coefficient is never touched: 5 x0 + 3 x1 <= 6 with x0 continuous.
  Model mixed = build({{5, 3}}, {-kInf}, {6}, {0, 0}, {0, 0}, {1, 1}, {false, true});
  EXPECT_EQ(presolve::tighten_coefficients(&mixed).coefficients_tightened, 1);
  EXPECT_EQ(mixed.matrix.at(0, 0), 5.0);
  // An unbounded column in the row: no maximum activity, nothing to tighten against.
  Model open = build({{5, 3}}, {-kInf}, {6}, {0, 0}, {0, 0}, {kInf, 1}, {false, true});
  EXPECT_EQ(presolve::tighten_coefficients(&open).coefficients_tightened, 0);
  // A pure LP is left alone entirely: its duals would belong to a different matrix.
  Model lp = build({{1, -100}}, {-kInf}, {0}, {1, 5}, {0, 0}, {10, 1}, {false, false});
  const auto lp_stats = presolve::tighten_coefficients(&lp);
  EXPECT_EQ(lp_stats.coefficients_tightened, 0);
  EXPECT_EQ(lp_stats.bounds_tightened, 0);
}

TEST(PresolveCoefficientTightening, TheOptionReachesPresolveAndOffMeansOff) {
  // The lot-sizing link: x_t <= 180 y_t where demand bounds x_t by 50.
  const Model model = build({{1, 0}, {1, -180}}, {-kInf, -kInf}, {50, 0}, {2, 150}, {0, 0},
                            {kInf, 1}, {false, true});
  Logger logger(nullptr);
  const presolve::Result on = presolve::presolve(model, with_tightening(true), logger);
  const presolve::Result off = presolve::presolve(model, with_tightening(false), logger);
  EXPECT_GT(on.report.coefficients_tightened, 0);
  EXPECT_EQ(off.report.coefficients_tightened, 0);
  EXPECT_EQ(off.report.propagated_bounds, 0);
  // Maximise 3 x - 150 y over it: the optimum is x = 50, y = 1, worth 0, with or without.
  Model max_model = model;
  max_model.sense = ObjSense::kMaximize;
  max_model.col_cost = {3, -150};
  const Solution with = solve(max_model, with_tightening(true));
  const Solution without = solve(max_model, with_tightening(false));
  ASSERT_EQ(with.status, SolveStatus::kOptimal) << with.message;
  ASSERT_EQ(without.status, SolveStatus::kOptimal) << without.message;
  EXPECT_NEAR(with.objective, without.objective, 1e-9);
  EXPECT_LE(with.primal_infeasibility, 1e-9) << "measured against the ORIGINAL model";
}

/// Every integer point of the box [lo, hi]^n, depth first.
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

bool feasible(const Model& model, const std::vector<double>& x, double tolerance) {
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (x[u] < model.col_lower[u] - tolerance || x[u] > model.col_upper[u] + tolerance) {
      return false;
    }
  }
  std::vector<double> activity(static_cast<std::size_t>(model.num_rows()), 0.0);
  model.matrix.multiply(x.data(), activity.data());
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto r = static_cast<std::size_t>(i);
    const double scale = tolerance * std::max(1.0, std::fabs(activity[r]));
    if (activity[r] < model.row_lower[r] - scale || activity[r] > model.row_upper[r] + scale) {
      return false;
    }
  }
  return true;
}

/// A random pure-integer model in a small box, with rows of every sense and coefficients
/// drawn wide enough (some large, big-M like) that the reduction fires on most instances.
/// `thirds` divides every coefficient by 3, so no activity is an exact integer and the
/// floating-point margin path is the one exercised.
Model random_integer_model(std::mt19937_64& rng, bool thirds) {
  std::uniform_int_distribution<int> cols(2, 5);
  std::uniform_int_distribution<int> nrows(1, 4);
  std::uniform_int_distribution<int> small(-4, 4);
  std::uniform_int_distribution<int> big(-30, 30);
  std::uniform_int_distribution<int> coin(0, 3);
  std::uniform_int_distribution<int> lower(-2, 1);
  std::uniform_int_distribution<int> width(0, 3);
  const int n = cols(rng);
  const int m = nrows(rng);
  const double divisor = thirds ? 3.0 : 1.0;
  std::vector<std::vector<double>> rows(static_cast<std::size_t>(m));
  std::vector<double> row_lower, row_upper;
  for (auto& row : rows) {
    double reach = 0.0;
    for (int j = 0; j < n; ++j) {
      const int v = coin(rng) == 0 ? big(rng) : small(rng);
      row.push_back(v / divisor);
      reach += std::fabs(v / divisor) * 3.0;
    }
    std::uniform_int_distribution<int> rhs(-static_cast<int>(reach / 2),
                                           static_cast<int>(reach / 2));
    const double b = rhs(rng) / divisor;
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
        row_upper.push_back(b + width(rng));
        break;
      default:
        row_lower.push_back(b - reach);
        row_upper.push_back(b);
        break;
    }
  }
  std::vector<double> lo, hi;
  for (int j = 0; j < n; ++j) {
    lo.push_back(lower(rng));
    hi.push_back(lo.back() + width(rng));
  }
  return build(rows, row_lower, row_upper,
               std::vector<double>(static_cast<std::size_t>(n), 0.0), lo, hi,
               std::vector<bool>(static_cast<std::size_t>(n), true));
}

void enumeration_sweep(std::uint64_t seed, bool thirds) {
  std::mt19937_64 rng(seed);
  Count tightened = 0;
  Count points = 0;
  // Trials where the reduction fired on a model with integer points both inside and outside
  // its feasible set: the ones where a wrong tightening would have something to remove.
  int telling = 0;
  for (int trial = 0; trial < 6000; ++trial) {
    const Model original = random_integer_model(rng, thirds);
    Model reduced = original;
    const Count fired = presolve::tighten_coefficients(&reduced).coefficients_tightened;
    tightened += fired;
    ASSERT_EQ(reduced.validate(), "");
    Count inside = 0;
    Count outside = 0;
    for_each_point(original.col_lower, original.col_upper, [&](const std::vector<double>& x) {
      ++points;
      const bool before = feasible(original, x, 1e-9);
      const bool after = feasible(reduced, x, 1e-9);
      (before ? inside : outside) += 1;
      EXPECT_EQ(before, after) << "trial " << trial << ": the integer point set changed";
    });
    if (fired > 0 && inside > 0 && outside > 0) ++telling;
    if (::testing::Test::HasFailure()) return;
  }
  EXPECT_GT(tightened, 1000) << "the generator should give the reduction work";
  EXPECT_GT(telling, 300);
  EXPECT_GT(points, 100000);
  std::printf("  %lld coefficients tightened, %d telling trials, %lld points compared\n",
              static_cast<long long>(tightened), telling, static_cast<long long>(points));
}

TEST(PresolveCoefficientTightening, EveryIntegerPointSurvivesOnRandomModels) {
  enumeration_sweep(5110, /*thirds=*/false);
}

TEST(PresolveCoefficientTightening, EveryIntegerPointSurvivesWhenTheArithmeticIsInexact) {
  enumeration_sweep(5111, /*thirds=*/true);
}

/// The fuzz harness's comparison for a MILP: the oracle's status is the truth, its optimum
/// to a relative 1e-6.
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
  const double difference = std::fabs(got.objective - expected);
  if (difference > 1e-6 * std::max(1.0, std::fabs(expected))) {
    *why = "objectives differ: " + std::to_string(got.objective) + " against " +
           std::to_string(expected);
    return false;
  }
  return true;
}

/// Add x_c <= M y_b on a random continuous column and a new binary, with M loose: the shape
/// the reduction exists for, which the generator's own rows rarely have.
void add_a_big_m_link(oracle::GeneratedLp* lp, std::mt19937_64& rng) {
  std::uniform_int_distribution<Index> pick(0, lp->num_cols - 1);
  std::uniform_int_distribution<std::int64_t> big(20, 60);
  std::uniform_int_distribution<std::int64_t> cost(-3, 6);
  const auto c = static_cast<std::size_t>(pick(rng));
  if (lp->upper[c] == oracle::kNoUpperBound) lp->upper[c] = 4;
  for (auto& row : lp->a) row.push_back(0);
  std::vector<std::int64_t> link(static_cast<std::size_t>(lp->num_cols) + 1, 0);
  link[c] = -1;  // A x >= b form: M y - x >= 0
  link.back() = big(rng);
  lp->a.push_back(link);
  lp->b.push_back(0);
  lp->c.push_back(cost(rng));
  lp->upper.push_back(1);
  lp->integral.push_back(1);
  if (!lp->lower.empty()) lp->lower.push_back(0);
  ++lp->num_cols;
  ++lp->num_rows;
}

TEST(PresolveCoefficientTightening, RandomMilpsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(51100);
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  config.bounded_column_probability = 0.7;
  int compared = 0;
  Count tightened = 0;
  for (int trial = 0; trial < 300; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 2 == 0 ? 1 : 0;
    }
    add_a_big_m_link(&lp, rng);
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
    // The debug-solution check: the exact optimum must satisfy the tightened model.
    if (exact.status == oracle::OracleStatus::kOptimal) {
      Model tightened_model = model;
      tightened += presolve::tighten_coefficients(&tightened_model).coefficients_tightened;
      std::vector<double> x;
      for (const auto& v : exact.x) x.push_back(v.to_double());
      EXPECT_TRUE(feasible(tightened_model, x, 1e-9))
          << "trial " << trial << ": the exact optimum was cut off\n"
          << lp.to_text();
    }
    const Solution got = solve(model, with_tightening(true));
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 200);
  EXPECT_GT(tightened, 50) << "the big-M links should give the reduction work";
}

}  // namespace
}  // namespace sankhya
