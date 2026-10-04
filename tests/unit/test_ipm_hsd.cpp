// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd).
// See src/ipm/hsd.cpp for the method and its citations.
//
// What #475 asks of it, each held by its own tests:
//   1. A primal infeasible LP ends `infeasible` with a Farkas vector, a dual infeasible one
//      with a feasible point ends `unbounded` with a ray and that point; each certificate is
//      re-checked here by the functions every engine's proof is held to.
//   2. A feasible LP ends optimal at the default path's objective, on hand-built models with
//      equality rows, fixed and free columns, ranged rows and a maximisation, and on KKT
//      instances whose optimum is known analytically.
//   3. With the option off nothing changes: the default loop does not certify, and says so.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

Options hsd(bool on, bool scaling = true) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("ipm_hsd", on);
  options.set_bool("scaling", scaling);
  return options;
}

Model empty_model(Index rows, Index cols) {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(cols), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(rows), -kInfinity);
  model.row_upper.assign(static_cast<std::size_t>(rows), kInfinity);
  model.matrix.reset(rows, cols);
  return model;
}

/// The engine alone, with and without the Ruiz scaling, so both branches of solve_scaled()
/// are held to the same answer.
Solution engine(const Model& model, bool scaling = true) {
  Logger quiet(stdout, LogLevel::kOff);
  return ipm::solve_ipm(model, hsd(true, scaling), quiet);
}

// ---- primal infeasible ------------------------------------------------------------------

/// x1 + x2 >= 3 with 0 <= x <= 1: the box gives at most 2.
Model box_infeasible() {
  Model model = empty_model(1, 2);
  model.col_cost = {1.0, 1.0};
  model.col_upper = {1.0, 1.0};
  model.row_lower = {3.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  return model;
}

/// x1, x2 free, x3 >= 0: x1 + x2 >= 2, x1 - x2 >= 2, x1 + x3 <= 1. The proof needs all three
/// rows, y = (1, 1, -2), with A'y vanishing on the two free columns.
Model three_row_infeasible() {
  Model model = empty_model(3, 3);
  model.col_cost = {1.0, -1.0, 2.0};
  model.col_lower = {-kInfinity, -kInfinity, 0.0};
  model.row_lower = {2.0, 2.0, -kInfinity};
  model.row_upper = {kInfinity, kInfinity, 1.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.add_entry(2, 0, 1.0);
  model.matrix.add_entry(2, 2, 1.0);
  model.matrix.finalize();
  return model;
}

/// Two equality rows that cannot both hold on free columns: x1 + x2 = 1, 2 x1 + 2 x2 = 5.
Model equality_infeasible() {
  Model model = empty_model(2, 2);
  model.col_lower = {-kInfinity, -kInfinity};
  model.col_cost = {1.0, 2.0};
  model.row_lower = {1.0, 5.0};
  model.row_upper = {1.0, 5.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 2.0);
  model.matrix.add_entry(1, 1, 2.0);
  model.matrix.finalize();
  return model;
}

/// A fixed column carries the contradiction: x1 = 4 (fixed), x1 + x2 <= 3, x2 >= 2.
Model fixed_column_infeasible() {
  Model model = empty_model(1, 2);
  model.col_cost = {1.0, 1.0};
  model.col_lower = {4.0, 2.0};
  model.col_upper = {4.0, kInfinity};
  model.row_upper = {3.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  return model;
}

/// Primal AND dual infeasible: x1 - x2 >= 1 and x1 - x2 <= 0, min -x1 - x2. Along (1, 1) the
/// objective falls and no row moves, but there is no point to start from: infeasible is the
/// verdict that can be proved.
Model both_infeasible() {
  Model model = empty_model(2, 2);
  model.col_cost = {-1.0, -1.0};
  model.row_lower = {1.0, -kInfinity};
  model.row_upper = {kInfinity, 0.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.finalize();
  return model;
}

void expect_farkas(const Model& model, const Solution& solution, const std::string& name) {
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << name << ": " << solution.message;
  ASSERT_EQ(solution.farkas_dual.size(), static_cast<std::size_t>(model.num_rows())) << name;
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual, &why))
      << name << ": " << why;
}

TEST(InteriorPointHsd, InfeasibleLpsEndWithAFarkasCertificate) {
  for (const auto& [name, model] : {std::pair<std::string, Model>{"box", box_infeasible()},
                                    {"three rows", three_row_infeasible()},
                                    {"equalities", equality_infeasible()},
                                    {"fixed column", fixed_column_infeasible()},
                                    {"both infeasible", both_infeasible()}}) {
    for (const bool scaling : {true, false}) {
      expect_farkas(model, engine(model, scaling), name + (scaling ? "" : " (unscaled)"));
    }
    // Through solve(), presolve off so the engine is what proves it, and on.
    for (const bool presolve : {false, true}) {
      Options options = hsd(true);
      options.set_bool("presolve", presolve);
      const Solution solved = solve(model, options);
      expect_farkas(model, solved,
                    name + (presolve ? " via solve()" : " via solve(), no presolve"));
    }
  }
}

TEST(InteriorPointHsd, OffTheDefaultLoopCertifiesNothing) {
  // The behaviour #475 replaces: the default loop has no way to conclude infeasibility, so
  // an infeasible LP never comes back `infeasible` with a certificate from it.
  for (const Model& model : {box_infeasible(), three_row_infeasible(), equality_infeasible()}) {
    Logger quiet(stdout, LogLevel::kOff);
    const Solution solution = ipm::solve_ipm(model, hsd(false), quiet);
    EXPECT_NE(solution.status, SolveStatus::kInfeasible) << solution.message;
    EXPECT_TRUE(solution.farkas_dual.empty());
  }
}

// ---- dual infeasible (unbounded) --------------------------------------------------------

/// min -x2 s.t. x1 - x2 <= 1, x >= 0. Feasible (x = 0); along (0, 1) the row falls (it has no
/// lower side), no bound blocks and the objective falls.
Model row_unbounded() {
  Model model = empty_model(1, 2);
  model.col_cost = {1.0, -1.0};
  model.row_upper = {1.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.finalize();
  return model;
}

/// Free columns and an equality: min x1 s.t. x1 + x2 = 1, x2 <= 10 + x3, x3 >= 0, maximised
/// as -x1 so the sense is exercised: max -x1 runs to +infinity along (-1, 1, 1).
Model free_unbounded_maximise() {
  Model model = empty_model(2, 3);
  model.sense = ObjSense::kMaximize;
  model.col_cost = {-1.0, 0.0, 0.0};
  model.col_lower = {-kInfinity, -kInfinity, 0.0};
  model.row_lower = {1.0, -kInfinity};
  model.row_upper = {1.0, 10.0};
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.add_entry(1, 2, -1.0);
  model.matrix.finalize();
  return model;
}

void expect_ray(const Model& model, const Solution& solution, const std::string& name) {
  EXPECT_EQ(solution.status, SolveStatus::kUnbounded) << name << ": " << solution.message;
  ASSERT_EQ(solution.primal_ray.size(), static_cast<std::size_t>(model.num_cols())) << name;
  std::string why;
  EXPECT_TRUE(ray_proves_unbounded(model, solution.primal_ray, &why)) << name << ": " << why;
  // The other half of the claim: the point the ray starts from is feasible.
  EXPECT_LE(solution.primal_infeasibility_scaled, tol::kPrimalFeasibility) << name;
}

TEST(InteriorPointHsd, UnboundedLpsEndWithARayAndAFeasiblePoint) {
  for (const auto& [name, model] : {std::pair<std::string, Model>{"row", row_unbounded()},
                                    {"free columns, maximised", free_unbounded_maximise()}}) {
    for (const bool scaling : {true, false}) {
      expect_ray(model, engine(model, scaling), name + (scaling ? "" : " (unscaled)"));
    }
    Options options = hsd(true);
    options.set_bool("presolve", false);
    expect_ray(model, solve(model, options), name + " via solve()");
  }
}

// ---- feasible: the default path's answer -------------------------------------------------

/// Equality rows, a fixed column, a free column, a ranged row and finite upper bounds.
Model mixed_feasible(ObjSense sense) {
  Model model = empty_model(3, 4);
  model.sense = sense;
  const double s = sense == ObjSense::kMaximize ? -1.0 : 1.0;
  model.col_cost = {s * 1.0, s * -2.0, s * 3.0, s * 0.5};
  model.col_lower = {0.0, -kInfinity, 2.0, -1.0};
  model.col_upper = {4.0, kInfinity, 2.0, 3.0};
  model.row_lower = {3.0, -2.0, 1.0};
  model.row_upper = {3.0, 5.0, kInfinity};
  // x1 + x2 + x3 = 3; -2 <= x2 - x4 <= 5; x1 + x4 >= 1.
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.add_entry(1, 3, -1.0);
  model.matrix.add_entry(2, 0, 1.0);
  model.matrix.add_entry(2, 3, 1.0);
  model.matrix.finalize();
  return model;
}

TEST(InteriorPointHsd, FeasibleLpsMatchTheDefaultPath) {
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    const Model model = mixed_feasible(sense);
    const Solution off = solve(model, hsd(false));
    ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
    for (const bool scaling : {true, false}) {
      const Solution on = solve(model, hsd(true, scaling));
      EXPECT_EQ(on.status, SolveStatus::kOptimal) << on.message;
      EXPECT_NEAR(on.objective, off.objective, 1e-7 * std::max(1.0, std::fabs(off.objective)));
      const Solution alone = engine(model, scaling);
      EXPECT_EQ(alone.status, SolveStatus::kOptimal) << alone.message;
      EXPECT_NEAR(alone.objective, off.objective,
                  1e-6 * std::max(1.0, std::fabs(off.objective)));
      EXPECT_TRUE(alone.farkas_dual.empty());
      EXPECT_TRUE(alone.primal_ray.empty());
    }
  }
}

TEST(InteriorPointHsd, AgreesWithTheAnalyticOptimumOnKktInstances) {
  // test_ipm.cpp's KKT check, with the embedding on, the engine alone and crossover off, so
  // the objective is the interior point's own.
  std::mt19937_64 rng(475475);
  oracle::GeneratorConfig config;
  config.min_rows = 3;
  config.max_rows = 12;
  config.min_cols = 3;
  config.max_cols = 12;
  config.magnitude = 6;
  int failed = 0;
  std::string first_failure;
  for (int trial = 0; trial < 150; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution ipm = engine(model);
    const double expected = static_cast<double>(instance.optimal_objective);
    const bool ok =
        ipm.status == SolveStatus::kOptimal &&
        std::fabs(ipm.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected));
    if (!ok && ++failed == 1) {
      first_failure = "trial " + std::to_string(trial) + ": " + to_string(ipm.status) +
                      " objective " + std::to_string(ipm.objective) + " expected " +
                      std::to_string(expected) + " (" + ipm.message + ")\n" +
                      instance.lp.to_text();
    }
  }
  EXPECT_EQ(failed, 0) << first_failure;
}

}  // namespace
}  // namespace sankhya
