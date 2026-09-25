// SPDX-License-Identifier: Apache-2.0
// SANKHYA - conflict analysis from nodes pruned by bound (#503), against enumeration.
//
// A conflict learned from an infeasible node says "no point at all satisfies these literals".
// A conflict learned from a node pruned by BOUND says less: "no point satisfying these
// literals has an objective better than the cutoff the search held when it was learned". The
// search writes that cutoff out with the conflict (conflict_out), and this test checks it
// against every integer-feasible point of small programs found by enumeration: a cutoff
// conflict that held at a point strictly better than its cutoff would let the search prune
// an improving solution and then prove the wrong answer. The optimum itself is compared
// with enumeration too, which is what catches a search that loses its bounds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

/// A small pure-integer program with inequality rows (knapsack-like, where an incumbent is
/// found early and many nodes are then pruned by bound) and some equalities.
Model random_program(std::mt19937& rng, bool maximize) {
  std::uniform_int_distribution<int> columns(5, 8);
  std::uniform_int_distribution<int> rows(2, 4);
  std::uniform_int_distribution<int> upper(1, 3);
  std::uniform_int_distribution<int> coefficient(-2, 7);
  std::uniform_int_distribution<int> cost(-9, 9);
  const int n = columns(rng);
  const int m = rows(rng);
  Model model;
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  std::vector<double> point(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    model.col_upper.push_back(static_cast<double>(upper(rng)));
    model.col_cost.push_back(static_cast<double>(cost(rng)));
    point[static_cast<std::size_t>(j)] =
        std::uniform_int_distribution<int>(0, static_cast<int>(model.col_upper.back()))(rng);
  }
  model.matrix.reset(m, n);
  for (int i = 0; i < m; ++i) {
    double activity = 0.0;
    for (int j = 0; j < n; ++j) {
      const int a = coefficient(rng);
      if (a == 0 || a == 1) continue;
      model.matrix.add_entry(i, j, static_cast<double>(a));
      activity += a * point[static_cast<std::size_t>(j)];
    }
    const int kind = std::uniform_int_distribution<int>(0, 4)(rng);
    if (kind == 0) {
      model.row_lower.push_back(activity);
      model.row_upper.push_back(activity);
    } else if (kind <= 2) {
      model.row_lower.push_back(-kInfinity);
      model.row_upper.push_back(activity + std::uniform_int_distribution<int>(0, 4)(rng));
    } else {
      model.row_lower.push_back(activity - std::uniform_int_distribution<int>(0, 4)(rng));
      model.row_upper.push_back(kInfinity);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

/// Every integer point in the box that meets every row exactly.
std::vector<std::vector<double>> feasible_points(const Model& model) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<std::vector<double>> dense(m, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      dense[static_cast<std::size_t>(column.rows[k])][j] += column.values[k];
    }
  }
  std::vector<std::vector<double>> points;
  std::vector<double> x(model.col_lower);
  while (true) {
    bool ok = true;
    for (std::size_t i = 0; i < m && ok; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) activity += dense[i][j] * x[j];
      ok = activity >= model.row_lower[i] - 1e-9 && activity <= model.row_upper[i] + 1e-9;
    }
    if (ok) points.push_back(x);
    std::size_t j = 0;
    while (j < n && x[j] >= model.col_upper[j]) {
      x[j] = model.col_lower[j];
      ++j;
    }
    if (j == n) break;
    x[j] += 1.0;
  }
  return points;
}

/// The objective in the search's own space: minimise, no offset.
double internal_objective(const Model& model, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) value += model.col_cost[j] * x[j];
  return model.sense == ObjSense::kMaximize ? -value : value;
}

bool literals_hold(const nlohmann::json& conflict, const std::vector<double>& x) {
  for (const nlohmann::json& literal : conflict["literals"]) {
    const double value = x[literal[0].get<std::size_t>()];
    const double bound = literal[2].get<double>();
    const bool holds =
        literal[1].get<std::string>() == "<=" ? value <= bound + 1e-9 : value >= bound - 1e-9;
    if (!holds) return false;
  }
  return true;
}

Options searching(bool cutoff, const std::string& out) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);  // conflict indices are then the model's own
  options.set_bool("conflict_analysis", true);
  options.set_bool("conflict_cutoff", cutoff);
  if (!out.empty()) options.set_string("conflict_out", out);
  return options;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return nlohmann::json::parse(text.str());
}

TEST(ConflictCutoff, NoCutoffConflictHoldsAtAPointBetterThanItsCutoffAndTheOptimumIsKept) {
  std::mt19937 rng(503);
  std::int64_t cutoff_learned = 0;
  std::int64_t used = 0;
  int solved_models = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const bool maximize = trial % 2 == 1;
    const Model model = random_program(rng, maximize);
    const std::vector<std::vector<double>> points = feasible_points(model);

    testing::TempFile out("", ".json");
    const Solution with = solve(model, searching(true, out.path()));
    const Solution without = solve(model, searching(false, ""));
    if (points.empty()) {
      EXPECT_EQ(with.status, SolveStatus::kInfeasible) << "trial " << trial;
      EXPECT_EQ(without.status, SolveStatus::kInfeasible) << "trial " << trial;
    } else {
      ++solved_models;
      double best = internal_objective(model, points.front());
      for (const std::vector<double>& x : points) {
        best = std::min(best, internal_objective(model, x));
      }
      const double external = maximize ? -best : best;
      EXPECT_EQ(with.status, SolveStatus::kOptimal)
          << "trial " << trial << ": " << with.message;
      EXPECT_EQ(without.status, SolveStatus::kOptimal) << "trial " << trial;
      EXPECT_NEAR(with.objective, external, 1e-7) << "trial " << trial;
      EXPECT_NEAR(without.objective, external, 1e-7) << "trial " << trial;
    }

    const nlohmann::json report = read_json(out.path());
    used +=
        report["nodes_pruned"].get<std::int64_t>() + report["tightenings"].get<std::int64_t>();
    for (const nlohmann::json& conflict : report["conflicts"]) {
      if (conflict["source"].get<std::string>() == "cutoff") {
        ++cutoff_learned;
        ASSERT_TRUE(conflict.contains("cutoff")) << conflict.dump();
      }
      // A conflict learned once a cutoff conflict is held may have leaned on one, and carries
      // a cutoff too; one without holds at no feasible point at all.
      const bool relative = conflict.contains("cutoff");
      for (const std::vector<double>& x : points) {
        if (!literals_hold(conflict, x)) continue;
        if (relative) {
          ASSERT_GT(internal_objective(model, x), conflict["cutoff"].get<double>())
              << "trial " << trial << ": conflict " << conflict.dump()
              << " holds at a point that beats its cutoff";
        } else {
          FAIL() << "trial " << trial << ": conflict " << conflict.dump()
                 << " holds at a feasible point";
        }
      }
    }
  }
  // The test would pass vacuously on a search that never learned from a cutoff.
  EXPECT_GT(solved_models, 100);
  EXPECT_GT(cutoff_learned, 50) << "conflicts learned from nodes pruned by bound";
  EXPECT_GT(used, 10) << "conflicts that pruned a node or fixed a bound";
}

TEST(ConflictCutoff, ARerunLearnsTheSameCutoffConflicts) {
  std::mt19937 rng(5031);
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = random_program(rng, trial % 2 == 0);
    testing::TempFile first("", ".json");
    testing::TempFile second("", ".json");
    const Solution a = solve(model, searching(true, first.path()));
    const Solution b = solve(model, searching(true, second.path()));
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.nodes, b.nodes) << "trial " << trial;
    nlohmann::json x = read_json(first.path());
    nlohmann::json y = read_json(second.path());
    x.erase("seconds");
    y.erase("seconds");
    EXPECT_EQ(x, y) << "trial " << trial;
    for (const nlohmann::json& conflict : x["conflicts"]) {
      if (conflict["source"].get<std::string>() == "cutoff") {
        ++compared;
        break;
      }
    }
  }
  EXPECT_GT(compared, 5);
}

TEST(ConflictCutoff, AQuadraticObjectiveLearnsNothingFromACutoff) {
  // The proof is a linear one; an MIQP node's bound is not, so no cutoff conflict is
  // attempted there, and the answer is the enumerated one.
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_lower = {0.0, 0.0, 0.0};
  model.col_upper = {3.0, 3.0, 3.0};
  model.col_type.assign(3, VarType::kInteger);
  model.col_cost = {-1.3, -2.1, 0.7};
  model.matrix.reset(1, 3);
  model.matrix.add_entry(0, 0, 2.0);
  model.matrix.add_entry(0, 1, 3.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {7.5};
  model.hessian.reset(3, 3);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.add_entry(2, 2, 1.0);
  model.hessian.finalize();

  double best = std::numeric_limits<double>::infinity();
  for (int a = 0; a <= 3; ++a) {
    for (int b = 0; b <= 3; ++b) {
      for (int c = 0; c <= 3; ++c) {
        if (2 * a + 3 * b + c > 7.5) continue;
        const double value = -1.3 * a - 2.1 * b + 0.7 * c + 0.5 * (a * a + b * b + c * c);
        best = std::min(best, value);
      }
    }
  }
  testing::TempFile out("", ".json");
  const Solution solved = solve(model, searching(true, out.path()));
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, best, 1e-6);
  std::ifstream in(out.path());
  std::stringstream text;
  text << in.rdbuf();
  if (text.str().empty()) return;  // nothing written means nothing learned
  for (const nlohmann::json& conflict : nlohmann::json::parse(text.str())["conflicts"]) {
    EXPECT_NE(conflict["source"].get<std::string>(), "cutoff") << conflict.dump();
  }
}

}  // namespace
}  // namespace sankhya
