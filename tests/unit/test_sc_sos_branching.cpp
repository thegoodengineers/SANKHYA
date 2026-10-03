// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound on semi-continuous columns and special ordered sets, and the
// binary reformulation that is its A/B partner (#754).
//
// The gate is SearchAgreesWithEnumeration. A search that branches on a set can lose the
// optimum the way any branch and bound can - a split that drops a member's support, a
// relaxation that is not one - and still report a feasible, optimal-looking answer. Here
// every instance is small enough to enumerate: each semi-continuous column on or off, each
// SOS1's one allowed member, each SOS2's one allowed pair, every combination solved as an
// ordinary LP (or MILP) and the best kept. The native search and the reformulation must both
// match it.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/sc_sos.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

Options reformulating() {
  Options options = quiet();
  options.set_bool("sos_reformulate", true);
  return options;
}

/// A model with dense rows given as lists; bounds and costs per column.
Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& lower, const std::vector<double>& upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.resize_columns(n);
  model.col_cost = cost;
  model.col_lower = lower;
  model.col_upper = upper;
  model.resize_rows(m);
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
  return model;
}

SosSet set_of(std::uint8_t type, std::vector<Index> columns) {
  SosSet set;
  set.type = type;
  set.columns = std::move(columns);
  for (std::size_t k = 0; k < set.columns.size(); ++k)
    set.weights.push_back(static_cast<double>(k + 1));
  return set;
}

/// Both routes, the same answer, and a point that meets every condition.
void expect_both_routes(const Model& model, SolveStatus status, double objective) {
  ASSERT_EQ(model.validate(), "");
  for (const bool reformulate : {false, true}) {
    const Solution s = solve(model, reformulate ? reformulating() : quiet());
    SCOPED_TRACE(reformulate ? "sos_reformulate" : "native");
    ASSERT_EQ(s.status, status) << s.message;
    if (status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, objective, 1e-6 * std::max(1.0, std::fabs(objective)));
    ASSERT_EQ(s.col_value.size(), static_cast<std::size_t>(model.num_cols()));
    EXPECT_LE(semicontinuous_and_sos_violation(model, s.col_value.data()), 1e-6);
    EXPECT_LE(s.integrality_violation, 1e-6);
    EXPECT_LE(s.primal_infeasibility, 1e-6);
  }
}

TEST(ScSosSearch, ASemiContinuousColumnJumpsToItsRunRange) {
  // min x, x >= 1, x in {0} or [2, 5]: the relaxation stops at 1, inside the gap.
  Model model = build({{1.0}}, {1.0}, {kInfinity}, {1.0}, {2.0}, {5.0});
  model.semicontinuous = {0};
  expect_both_routes(model, SolveStatus::kOptimal, 2.0);
}

TEST(ScSosSearch, OffIsChosenWhenItIsCheaper) {
  // min x + 2y, x + y >= 2, x in {0} or [3, 10]: x = 0 costs 4, x = 3 costs 3; the
  // relaxation's x = 2 is in the gap.
  Model model = build({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 2.0}, {3.0, 0.0}, {10.0, 10.0});
  model.semicontinuous = {0};
  expect_both_routes(model, SolveStatus::kOptimal, 3.0);
  // With y cheap, off wins: x = 0, y = 2 costs 1.
  model.col_cost = {1.0, 0.5};
  expect_both_routes(model, SolveStatus::kOptimal, 1.0);
}

TEST(ScSosSearch, AnUnboundedRunRangeStaysNativeUnderReformulation) {
  // x in {0} or [4, inf), y in {0} or [2, 5], x >= 1, y >= 1: optimum 4 + 2. The
  // reformulation has an M for y and none for x, which it leaves to the native branching.
  Model model = build({{1.0, 0.0}, {0.0, 1.0}}, {1.0, 1.0}, {kInfinity, kInfinity}, {1.0, 1.0},
                      {4.0, 2.0}, {kInfinity, 5.0});
  model.semicontinuous = {0, 1};
  expect_both_routes(model, SolveStatus::kOptimal, 6.0);
  const Solution s = solve(model, reformulating());
  EXPECT_NE(s.message.find("binary reformulation"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("1 left to native"), std::string::npos) << s.message;
}

TEST(ScSosSearch, InfeasibleWhenTheRunRangeMissesTheRows) {
  // 1 <= x <= 3 by rows, x in {0} or [5, 10]: no point.
  Model model = build({{1.0}}, {1.0}, {3.0}, {1.0}, {5.0}, {10.0});
  model.semicontinuous = {0};
  expect_both_routes(model, SolveStatus::kInfeasible, 0.0);
}

TEST(ScSosSearch, Sos1KeepsOneMember) {
  // min -x1 - 2x2 - 3x3, x in [0,1]^3, sum <= 2: the relaxation takes x2 = x3 = 1 (-5);
  // with one member allowed the best is x3 = 1 (-3).
  Model model = build({{1.0, 1.0, 1.0}}, {-kInfinity}, {2.0}, {-1.0, -2.0, -3.0},
                      {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  model.sos = {set_of(1, {0, 1, 2})};
  expect_both_routes(model, SolveStatus::kOptimal, -3.0);
}

TEST(ScSosSearch, Sos2PiecewiseLinearNonconvexCurve) {
  // f on breakpoints x = 0, 1, 2, 3 with values 0, -2, 3, -4, written with weights l_k:
  // x = sum b_k l_k, sum l_k = 1, l SOS2, x <= 2.5. The relaxation mixes l_1 and l_3 to
  // reach -3.5 at x = 2.5; on the curve the best is f(1) = -2 (the last segment reaches only
  // f(2.5) = -0.5).
  Model model =
      build({{0.0, 1.0, 2.0, 3.0}, {1.0, 1.0, 1.0, 1.0}}, {-kInfinity, 1.0}, {2.5, 1.0},
            {0.0, -2.0, 3.0, -4.0}, {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0});
  model.sos = {set_of(2, {0, 1, 2, 3})};
  expect_both_routes(model, SolveStatus::kOptimal, -2.0);
}

TEST(ScSosSearch, ASemiContinuousQuadraticObjective) {
  // min (x - 1.5)^2 = 0.5 * 2 x^2 - 3x + 2.25, x in {0} or [2, 4]: 0.25 at x = 2.
  Model model = build({}, {}, {}, {-3.0}, {2.0}, {4.0});
  model.objective_offset = 2.25;
  model.hessian.reset(1, 1);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.finalize();
  model.semicontinuous = {0};
  for (const bool reformulate : {false, true}) {
    const Solution s = solve(model, reformulate ? reformulating() : quiet());
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
    EXPECT_NEAR(s.objective, 0.25, 1e-5);
    EXPECT_NEAR(s.col_value[0], 2.0, 1e-5);
  }
}

TEST(ScSosSearch, PresolveStandsAsideAndTheClassIsMilp) {
  Model model = build({{1.0}}, {1.0}, {kInfinity}, {1.0}, {2.0}, {5.0});
  model.semicontinuous = {0};
  const Solution s = solve(model, quiet());
  EXPECT_EQ(s.algorithm, "branch-and-bound");
  EXPECT_NE(s.presolve_report.skipped_because.find("#754"), std::string::npos)
      << s.presolve_report.skipped_because;
}

TEST(ScSosSearch, ACertificateIsRefusedNotFaked) {
  Model model = build({{1.0}}, {1.0}, {kInfinity}, {1.0}, {2.0}, {5.0});
  model.semicontinuous = {0};
  Options options = quiet();
  options.set_string("write_certificate", "sc_sos_refused.vipr");
  const Solution s = solve(model, options);
  EXPECT_EQ(s.status, SolveStatus::kNotSolved);
  EXPECT_NE(s.message.find("VIPR"), std::string::npos) << s.message;
  std::remove("sc_sos_refused.vipr");
}

// ---- The gate: enumeration over every on/off and every allowed support -----------------

struct Instance {
  Model model;
  std::vector<Index> sc;  ///< semi-continuous columns
};

/// Six bounded continuous columns (a seventh, integer, on odd seeds), three random rows and
/// one covering row; columns 0 and 1 semi-continuous, an SOS1 over {1, 2, 3} and an SOS2
/// over {3, 4, 5}, so the conditions overlap on columns 1 and 3.
Instance random_instance(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> coef(-3, 3);
  std::uniform_int_distribution<int> cost(-6, 6);
  std::uniform_int_distribution<int> upper(3, 8);
  const bool with_integer = seed % 2 == 1;
  const Index n = with_integer ? 7 : 6;
  std::vector<std::vector<double>> rows(4, std::vector<double>(static_cast<std::size_t>(n)));
  std::vector<double> row_lower(4, -kInfinity);
  std::vector<double> row_upper(4, kInfinity);
  for (std::size_t i = 0; i < 3; ++i) {
    for (Index j = 0; j < n; ++j) rows[i][static_cast<std::size_t>(j)] = coef(rng);
    row_upper[i] = std::uniform_int_distribution<int>(2, 12)(rng);
  }
  for (Index j = 0; j < n; ++j) rows[3][static_cast<std::size_t>(j)] = 1.0;
  row_lower[3] = std::uniform_int_distribution<int>(1, 6)(rng) + 0.5;
  std::vector<double> c(static_cast<std::size_t>(n));
  std::vector<double> lo(static_cast<std::size_t>(n), 0.0);
  std::vector<double> hi(static_cast<std::size_t>(n));
  for (Index j = 0; j < n; ++j) {
    c[static_cast<std::size_t>(j)] = cost(rng);
    hi[static_cast<std::size_t>(j)] = upper(rng);
  }
  for (Index j : {0, 1}) {
    const auto u = static_cast<std::size_t>(j);
    lo[u] = 0.5 + std::uniform_int_distribution<int>(1, static_cast<int>(hi[u]) - 1)(rng);
  }
  Instance inst;
  inst.model = build(rows, row_lower, row_upper, c, lo, hi);
  if (with_integer) inst.model.col_type[6] = VarType::kInteger;
  inst.model.semicontinuous = {0, 1};
  inst.model.sos = {set_of(1, {1, 2, 3}), set_of(2, {3, 4, 5})};
  inst.sc = {0, 1};
  return inst;
}

/// The best objective over every combination, solved without the conditions; +inf when
/// none is feasible.
double enumerate(const Model& model) {
  Model plain = model;
  plain.semicontinuous.clear();
  plain.sos.clear();
  const std::vector<SosSet>& sets = model.sos;
  double best = kInfinity;
  // Semi-continuous on/off: 2^|sc|. SOS1: the allowed member. SOS2: the allowed pair.
  const std::size_t sc_count = model.semicontinuous.size();
  const std::size_t sos1 = sets[0].columns.size();
  const std::size_t sos2 = sets[1].columns.size() - 1;
  for (std::size_t mask = 0; mask < (std::size_t{1} << sc_count); ++mask) {
    for (std::size_t a = 0; a < sos1; ++a) {
      for (std::size_t b = 0; b < sos2; ++b) {
        Model trial = plain;
        bool empty = false;
        const auto fix_zero = [&](Index j) {
          const auto u = static_cast<std::size_t>(j);
          if (trial.col_lower[u] > 0.0) empty = true;
          trial.col_upper[u] = 0.0;
        };
        for (std::size_t k = 0; k < sc_count; ++k) {
          const auto u = static_cast<std::size_t>(model.semicontinuous[k]);
          if (((mask >> k) & 1U) == 0) {
            trial.col_lower[u] = 0.0;
            trial.col_upper[u] = 0.0;
          }
        }
        for (std::size_t k = 0; k < sos1; ++k) {
          if (k != a) fix_zero(sets[0].columns[k]);
        }
        for (std::size_t k = 0; k < sets[1].columns.size(); ++k) {
          if (k != b && k != b + 1) fix_zero(sets[1].columns[k]);
        }
        if (empty) continue;  // an "on" column a set forces to 0: no point here
        const Solution s = solve(trial, quiet());
        if (s.status == SolveStatus::kOptimal) best = std::min(best, s.objective);
        EXPECT_TRUE(s.status == SolveStatus::kOptimal || s.status == SolveStatus::kInfeasible)
            << to_string(s.status);
      }
    }
  }
  return best;
}

TEST(ScSosSearch, SearchAgreesWithEnumeration) {
  int optimal = 0;
  int infeasible = 0;
  int binding = 0;
  for (unsigned seed = 1; seed <= 200; ++seed) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    const Instance inst = random_instance(seed);
    ASSERT_EQ(inst.model.validate(), "");
    const double best = enumerate(inst.model);
    if (std::isinf(best)) {
      expect_both_routes(inst.model, SolveStatus::kInfeasible, 0.0);
      ++infeasible;
    } else {
      expect_both_routes(inst.model, SolveStatus::kOptimal, best);
      ++optimal;
      // Did the conditions matter? The relaxation (semi-continuous columns from 0, sets
      // dropped) is strictly better whenever the search had to branch on one of them.
      Model relaxed = inst.model;
      relaxed.col_lower[0] = 0.0;
      relaxed.col_lower[1] = 0.0;
      relaxed.semicontinuous.clear();
      relaxed.sos.clear();
      const Solution r = solve(relaxed, quiet());
      if (r.status == SolveStatus::kOptimal && r.objective < best - 1e-6) ++binding;
    }
  }
  // The generator is only a gate if it produces both verdicts, and mostly solvable ones.
  RecordProperty("optimal", optimal);
  RecordProperty("infeasible", infeasible);
  RecordProperty("binding", binding);
  std::printf(
      "enumeration gate: %d optimal (%d where the relaxation was strictly better), %d "
      "infeasible\n",
      optimal, binding, infeasible);
  EXPECT_GE(optimal, 100);
  EXPECT_GE(binding, 50) << "too few instances where the conditions change the optimum";
  EXPECT_GE(infeasible, 1);
}

}  // namespace
}  // namespace sankhya
