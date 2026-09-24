// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the dual simplex's Harris ratio test with cost shifting and its cost
// perturbation at the start (#465).
//
// Both are off by default, and both change the path the dual takes, never the answer it may
// report: every cost change is undone before an answer leaves the dual loop, and the primal
// simplex finishes from the basis. So the property pinned here is the same one
// test_dual_simplex.cpp pins for the textbook rule - the same status and the same optimum as
// the primal, a certificate that passes the dispatcher's KKT measurement, and the published
// Netlib optimum - under each option alone and under both together, on random instances
// built to be dual degenerate (costs from a set of two values) and on Netlib.
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Config {
  const char* name;
  bool harris;
  bool perturb_at_start;
};
constexpr Config kConfigs[] = {
    {"harris", true, false}, {"perturb-at-start", false, true}, {"both", true, true}};

Options dual_options(const Config& config) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "dual-simplex");
  options.set_string("dual_ratio_test", config.harris ? "harris" : "textbook");
  options.set_bool("dual_perturb_costs_at_start", config.perturb_at_start);
  return options;
}

Solution run_primal(const Model& model) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "simplex");
  return solve(model, options);
}

Model make_model(const std::vector<double>& cost, const std::vector<double>& col_lower,
                 const std::vector<double>& col_upper,
                 const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper) {
  Model model;
  model.name = "dual-harris-test";
  model.sense = ObjSense::kMinimize;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
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

/// A random LP feasible around a known point, with every cost drawn from {0, 1} (or from a
/// continuum when `distinct_costs`), so the dual ratio test ties from the first pivot.
Model random_degenerate_lp(std::mt19937* rng, bool distinct_costs) {
  std::uniform_real_distribution<double> coefficient(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_int_distribution<int> rows_count(4, 12);
  std::uniform_int_distribution<int> cols_count(12, 24);
  const auto m = static_cast<std::size_t>(rows_count(*rng));
  const auto n = static_cast<std::size_t>(cols_count(*rng));
  std::vector<double> point(n);
  std::vector<double> cost(n);
  std::vector<double> lower(n);
  std::vector<double> upper(n);
  for (std::size_t j = 0; j < n; ++j) {
    point[j] = 4.0 * unit(*rng);
    cost[j] = distinct_costs ? coefficient(*rng) : (unit(*rng) < 0.6 ? 0.0 : 1.0);
    const double roll = unit(*rng);
    if (roll < 0.6) {
      lower[j] = 0.0;
      upper[j] = 4.0 + 4.0 * unit(*rng);
    } else if (roll < 0.9) {
      lower[j] = 0.0;
      upper[j] = kInf;
    } else {
      lower[j] = -kInf;
      upper[j] = kInf;
    }
  }
  std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
  std::vector<double> row_lower(m);
  std::vector<double> row_upper(m);
  for (std::size_t i = 0; i < m; ++i) {
    double activity = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (unit(*rng) < 0.5) rows[i][j] = coefficient(*rng);
      activity += rows[i][j] * point[j];
    }
    const double roll = unit(*rng);
    if (roll < 0.35) {
      row_lower[i] = -kInf;
      row_upper[i] = activity + 2.0 * unit(*rng);
    } else if (roll < 0.7) {
      row_lower[i] = activity - 2.0 * unit(*rng);
      row_upper[i] = kInf;
    } else {
      row_lower[i] = activity;
      row_upper[i] = activity;
    }
  }
  return make_model(cost, lower, upper, rows, row_lower, row_upper);
}

TEST(DualHarris, AgreesWithThePrimalOnDualDegenerateRandomInstances) {
  for (const Config& config : kConfigs) {
    std::mt19937 rng(465);
    int optimal = 0;
    for (int trial = 0; trial < 200; ++trial) {
      const Model model = random_degenerate_lp(&rng, trial % 4 == 3);
      if (!model.validate().empty()) continue;
      const Solution dual = solve(model, dual_options(config));
      const Solution primal = run_primal(model);
      ASSERT_EQ(dual.status, primal.status) << config.name << " trial " << trial << ": dual "
                                            << dual.message << " / primal " << primal.message;
      if (primal.status != SolveStatus::kOptimal) continue;
      ++optimal;
      EXPECT_LE(dual.primal_infeasibility_scaled, tol::kPrimalFeasibility)
          << config.name << " trial " << trial;
      EXPECT_LE(dual.dual_infeasibility_scaled, tol::kDualFeasibility)
          << config.name << " trial " << trial;
      EXPECT_NEAR(dual.objective, primal.objective,
                  1e-7 * std::max(1.0, std::fabs(primal.objective)))
          << config.name << " trial " << trial;
    }
    EXPECT_GT(optimal, 100) << config.name << ": too few optimal instances to prove anything";
  }
}

TEST(DualHarris, ReachesThePublishedNetlibOptima) {
  struct Instance {
    const char* name;
    double published;
  };
  // Published values from data/netlib/reference.json (Netlib's readme). degen2 and
  // etamacro are here because they are the degenerate and the tolerance-sensitive cases:
  // etamacro is where a Harris window without cost shifting failed its duality check (#244).
  const Instance instances[] = {{"afiro", -464.75314286},   {"sc50b", -70.0},
                                {"share2b", -415.73224074}, {"stocfor1", -41131.976219},
                                {"degen2", -1435.178},      {"etamacro", -755.71521774}};
  for (const Instance& instance : instances) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(instance.name) + ".mps"))
            .string();
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;
    for (const Config& config : kConfigs) {
      const Solution s = solve(model, dual_options(config));
      ASSERT_EQ(s.status, SolveStatus::kOptimal)
          << instance.name << " " << config.name << ": " << s.message;
      EXPECT_NEAR(s.objective, instance.published,
                  1e-6 * std::max(1.0, std::fabs(instance.published)))
          << instance.name << " " << config.name;
      EXPECT_LE(s.primal_infeasibility_scaled, tol::kPrimalFeasibility)
          << instance.name << " " << config.name;
      EXPECT_LE(s.dual_infeasibility_scaled, tol::kDualFeasibility)
          << instance.name << " " << config.name;
    }
  }
}

TEST(DualHarris, PerturbationAtTheStartIsGatedOnDistinctCostsAndUndoneByThePrimal) {
  // Few distinct costs: the perturbation is applied, the dual finishes on perturbed costs,
  // and the primal simplex restores the exact optimum from that basis - the engine string
  // says so. Distinct costs: the gate refuses and the dual finishes alone.
  std::mt19937 rng(4651);
  int handed_over = 0;
  int alone = 0;
  const Config perturb_only = kConfigs[1];
  for (int trial = 0; trial < 40; ++trial) {
    const bool distinct = trial % 2 == 1;
    const Model model = random_degenerate_lp(&rng, distinct);
    if (!model.validate().empty()) continue;
    const Solution dual = solve(model, dual_options(perturb_only));
    const Solution primal = run_primal(model);
    ASSERT_EQ(dual.status, primal.status) << "trial " << trial << ": " << dual.message;
    if (dual.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(dual.objective, primal.objective,
                1e-7 * std::max(1.0, std::fabs(primal.objective)))
        << "trial " << trial;
    if (distinct) {
      // No perturbation; an artificial bound or a gap can still hand over, but the
      // perturbation never does.
      if (dual.algorithm == "simplex-dual") ++alone;
    } else {
      EXPECT_EQ(dual.algorithm, "simplex-dual+primal") << "trial " << trial;
      if (dual.algorithm == "simplex-dual+primal") ++handed_over;
    }
  }
  EXPECT_GT(handed_over, 5);
  EXPECT_GT(alone, 5);
}

}  // namespace
}  // namespace sankhya
