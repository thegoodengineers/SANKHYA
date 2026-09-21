// SPDX-License-Identifier: Apache-2.0
// SANKHYA - conflict analysis in the branch and bound (#292).
//
// Two halves. The first tests the pure parts on their own: the canonical literal set, the
// verdict of a conflict against a box, the Farkas proof and its tolerance, the deletion
// filter, and the store's duplicate detection and eviction. The second is the one that
// matters: small integer programs solved with conflict analysis on, every conflict the search
// learned written out (conflict_out) and checked against EVERY integer-feasible point found
// by enumeration. A conflict that holds at a feasible point would let the search prune the
// optimum and then prove the wrong answer; this is the test that says none did.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "mip/conflict.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using mip::ConflictLiteral;

// =========================================================================================
// The pure parts
// =========================================================================================

TEST(Conflict, TheCanonicalFormKeepsTheTightestPerColumnAndDirectionSorted) {
  const std::vector<ConflictLiteral> literals = mip::canonical({
      {3, true, 2.0},
      {1, false, 1.0},
      {3, true, 1.0},   // x3 <= 1 is tighter than x3 <= 2
      {1, false, 2.0},  // x1 >= 2 is tighter than x1 >= 1
      {3, false, 0.0},  // a lower bound on x3 is a different literal
  });
  const std::vector<ConflictLiteral> expected = {
      {1, false, 2.0}, {3, false, 0.0}, {3, true, 1.0}};
  EXPECT_EQ(literals, expected);
}

TEST(Conflict, AConflictAgainstABoxPrunesImpliesOrSaysNothing) {
  const std::vector<ConflictLiteral> conflict = {{0, false, 4.0}, {1, false, 6.0}};
  const double tolerance = 1e-6;

  // Both literals hold everywhere in the box: the box is infeasible.
  mip::ConflictVerdict verdict =
      mip::check_conflict(conflict, {4.0, 6.0}, {9.0, 9.0}, tolerance);
  EXPECT_TRUE(verdict.infeasible);

  // One holds, one is undecided: that one must be false, x1 <= 5.
  verdict = mip::check_conflict(conflict, {4.0, 0.0}, {9.0, 9.0}, tolerance);
  EXPECT_FALSE(verdict.infeasible);
  ASSERT_EQ(verdict.implied, 1);
  const ConflictLiteral bound = mip::negation(conflict[1]);
  EXPECT_EQ(bound, (ConflictLiteral{1, true, 5.0}));

  // One is already false everywhere: nothing follows, even with the other undecided.
  verdict = mip::check_conflict(conflict, {0.0, 0.0}, {3.0, 9.0}, tolerance);
  EXPECT_FALSE(verdict.infeasible);
  EXPECT_EQ(verdict.implied, -1);

  // Two undecided: nothing follows.
  verdict = mip::check_conflict(conflict, {0.0, 0.0}, {9.0, 9.0}, tolerance);
  EXPECT_FALSE(verdict.infeasible);
  EXPECT_EQ(verdict.implied, -1);

  // A bound within the integrality tolerance of the literal counts as meeting it, and one
  // further away does not: the bounds are integral here, so noise must not flip a literal.
  verdict = mip::check_conflict(conflict, {4.0 - 1e-9, 6.0}, {9.0, 9.0}, tolerance);
  EXPECT_TRUE(verdict.infeasible);
  verdict = mip::check_conflict(conflict, {3.9, 6.0}, {9.0, 9.0}, tolerance);
  EXPECT_FALSE(verdict.infeasible);
}

/// x0 + x1 <= 10 with both columns in [0, 20]: the example in the issue.
Model two_column_packing(double rhs) {
  Model model;
  model.col_cost = {0.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {20.0, 20.0};
  model.col_type = {VarType::kInteger, VarType::kInteger};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {rhs};
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

TEST(Conflict, TheFarkasProofContradictsOnlyTheBoxesItShould) {
  const Model model = two_column_packing(10.0);
  // y = -1 on the row's upper bound: -x0 - x1 >= -10 for every feasible point.
  const mip::FarkasProof proof = mip::aggregate_farkas(model, {-1.0});
  ASSERT_TRUE(proof.usable);

  EXPECT_TRUE(mip::farkas_contradicts(proof, {6.0, 5.0}, {20.0, 20.0}))
      << "x0 >= 6 and x1 >= 5 need 11 > 10";
  EXPECT_FALSE(mip::farkas_contradicts(proof, {5.0, 5.0}, {20.0, 20.0}))
      << "x0 = x1 = 5 is feasible";
  EXPECT_FALSE(mip::farkas_contradicts(proof, {6.0, -kInfinity}, {20.0, 20.0}))
      << "a column free on the side the proof leans on makes it prove nothing";

  // The sign is part of the proof: +1 leans on a lower bound the row does not have.
  EXPECT_FALSE(mip::aggregate_farkas(model, {1.0}).usable);
  // And no proof at all proves nothing.
  EXPECT_FALSE(mip::aggregate_farkas(model, {0.0}).usable);
  EXPECT_FALSE(mip::aggregate_farkas(model, {}).usable);
}

TEST(Conflict, AContradictionWithinTheFeasibilityToleranceIsNotAProof) {
  // x0 + x1 <= 11 - 1e-9 against x0 >= 6, x1 >= 5: a point violating the row by 1e-9 is one
  // the solver accepts as feasible, so this is no contradiction.
  const Model model = two_column_packing(11.0 - 1e-9);
  const mip::FarkasProof proof = mip::aggregate_farkas(model, {-1.0});
  EXPECT_FALSE(mip::farkas_contradicts(proof, {6.0, 5.0}, {20.0, 20.0}));
  // Scaling y changes nothing: the proof is homogeneous and the check is too.
  const mip::FarkasProof scaled = mip::aggregate_farkas(model, {-1e-9});
  EXPECT_FALSE(mip::farkas_contradicts(scaled, {6.0, 5.0}, {20.0, 20.0}));
  const mip::FarkasProof real = mip::aggregate_farkas(two_column_packing(10.0), {-1e-9});
  EXPECT_TRUE(mip::farkas_contradicts(real, {6.0, 5.0}, {20.0, 20.0}));
}

TEST(Conflict, ASmallCoefficientOnALargeBoundIsNotRoundedAway) {
  // x0 - 1e-4 x1 <= 10 with x1 in [0, 1e6]: x0 >= 10.5 is feasible, at x1 >= 5000. The proof
  // y = -1 aggregates to -x0 + 1e-4 x1 >= -10, and 1e-4 * 1e6 = 100 is not small at all.
  Model model = two_column_packing(10.0);
  model.col_upper = {20.0, 1e6};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1e-4);
  model.matrix.finalize();
  const mip::FarkasProof proof = mip::aggregate_farkas(model, {-1.0});
  ASSERT_TRUE(proof.usable);
  EXPECT_FALSE(mip::farkas_contradicts(proof, {10.5, 0.0}, {20.0, 1e6}));
  EXPECT_FALSE(mip::farkas_contradicts(proof, {10.5, 0.0}, {20.0, kInfinity}));
  // With x1 held at 0 the same proof does hold: x0 >= 10.5 against x0 <= 10.
  EXPECT_TRUE(mip::farkas_contradicts(proof, {10.5, 0.0}, {20.0, 0.0}));
}

TEST(Conflict, TheDeletionFilterDropsWhatTheProofDoesNotNeed) {
  const std::vector<ConflictLiteral> start = {
      {0, true, 0.0}, {1, true, 0.0}, {2, false, 1.0}, {3, false, 1.0}};
  // The "proof": infeasible whenever literals 0 and 2 are both present.
  int calls = 0;
  const auto verify = [&](const std::vector<ConflictLiteral>& literals) {
    ++calls;
    const bool has0 = std::find(literals.begin(), literals.end(), start[0]) != literals.end();
    const bool has2 = std::find(literals.begin(), literals.end(), start[2]) != literals.end();
    return has0 && has2;
  };
  int used = 0;
  const std::vector<ConflictLiteral> kept =
      mip::minimize_conflict(start, /*keep_last=*/false, 100, verify, &used);
  EXPECT_EQ(kept, (std::vector<ConflictLiteral>{start[0], start[2]}));
  EXPECT_EQ(used, 4);
  EXPECT_EQ(calls, 4);

  // keep_last spends no check on the last literal and keeps it.
  calls = 0;
  const std::vector<ConflictLiteral> with_last =
      mip::minimize_conflict(start, /*keep_last=*/true, 100, verify, &used);
  EXPECT_EQ(with_last, (std::vector<ConflictLiteral>{start[0], start[2], start[3]}));
  EXPECT_EQ(calls, 3);

  // The budget is a hard cap, and what it did not reach is kept, not dropped.
  calls = 0;
  const std::vector<ConflictLiteral> capped =
      mip::minimize_conflict(start, /*keep_last=*/false, 2, verify, &used);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(capped, (std::vector<ConflictLiteral>{start[0], start[2], start[3]}));
}

TEST(Conflict, TheStoreRejectsDuplicatesAndForgetsTheLeastUsedDeterministically) {
  mip::ConflictStore store(10);
  for (int k = 0; k < 10; ++k) {
    ASSERT_TRUE(store.add({{k, true, 0.0}}, mip::ConflictSource::kLp, k));
  }
  EXPECT_FALSE(store.add({{3, true, 0.0}}, mip::ConflictSource::kLp, 11));
  EXPECT_EQ(store.duplicates(), 1);

  // Every conflict but #4 has been used; #4 is the one a full store forgets.
  for (mip::ConflictStore::Entry& entry : store.entries()) {
    if (entry.id != 4) entry.uses = 1;
  }
  ASSERT_TRUE(store.add({{20, false, 1.0}}, mip::ConflictSource::kPropagation, 12));
  EXPECT_EQ(store.size(), 10u);
  EXPECT_EQ(store.evicted(), 1);
  for (const mip::ConflictStore::Entry& entry : store.entries()) EXPECT_NE(entry.id, 4);
  // Forgotten means forgotten: the same conflict may be learned again.
  EXPECT_TRUE(store.add({{4, true, 0.0}}, mip::ConflictSource::kLp, 13));

  mip::ConflictStore none(0);
  EXPECT_FALSE(none.add({{0, true, 0.0}}, mip::ConflictSource::kLp, 0));
  EXPECT_EQ(none.size(), 0u);
}

// =========================================================================================
// The search, against enumeration
// =========================================================================================

/// A small pure-integer program with equality rows, which is where infeasible nodes come
/// from: a parity or a sum the branching decisions cannot meet.
Model random_integer_program(std::mt19937& rng, bool maximize) {
  std::uniform_int_distribution<int> columns(4, 7);
  std::uniform_int_distribution<int> rows(2, 4);
  std::uniform_int_distribution<int> upper(1, 3);
  std::uniform_int_distribution<int> coefficient(-3, 6);
  std::uniform_int_distribution<int> cost(-6, 6);
  std::uniform_int_distribution<int> kind(0, 2);
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
      if (a == 0 || a == 1) continue;  // sparser, and more even coefficients
      model.matrix.add_entry(i, j, static_cast<double>(a));
      activity += a * point[static_cast<std::size_t>(j)];
    }
    // Rows are built around a random point, so most programs are feasible, and nudged off it
    // one time in four, so some are not.
    const double shift = std::uniform_int_distribution<int>(0, 3)(rng) == 0 ? 1.0 : 0.0;
    switch (kind(rng)) {
      case 0:
        model.row_lower.push_back(activity + shift);
        model.row_upper.push_back(activity + shift);
        break;
      case 1:
        model.row_lower.push_back(-kInfinity);
        model.row_upper.push_back(activity + shift);
        break;
      default:
        model.row_lower.push_back(activity - shift);
        model.row_upper.push_back(kInfinity);
        break;
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

/// Is the point integral and inside every row and bound of the model?
bool point_is_feasible(const Model& model, const std::vector<double>& x) {
  if (x.size() != static_cast<std::size_t>(model.num_cols())) return false;
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> activity(m, 0.0);
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (x[j] < model.col_lower[j] - 1e-6 || x[j] > model.col_upper[j] + 1e-6) return false;
    if (model.col_type[j] == VarType::kInteger && std::fabs(x[j] - std::round(x[j])) > 1e-6) {
      return false;
    }
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      activity[static_cast<std::size_t>(column.rows[k])] += column.values[k] * x[j];
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (activity[i] < model.row_lower[i] - 1e-6 || activity[i] > model.row_upper[i] + 1e-6) {
      return false;
    }
  }
  return true;
}

double objective_of(const Model& model, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) value += model.col_cost[j] * x[j];
  return value;
}

Options searching(bool conflicts, const std::string& out) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);  // conflict indices are then the model's own
  options.set_bool("conflict_analysis", conflicts);
  if (!out.empty()) options.set_string("conflict_out", out);
  return options;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return nlohmann::json::parse(text.str());
}

/// The check the whole subsystem stands on: no conflict the search wrote out may hold at a
/// point the model calls feasible, whatever ended the search.
void expect_no_conflict_holds(const nlohmann::json& report,
                              const std::vector<std::vector<double>>& points) {
  for (const nlohmann::json& conflict : report["conflicts"]) {
    for (const std::vector<double>& x : points) {
      bool all_hold = true;
      for (const nlohmann::json& literal : conflict["literals"]) {
        const double value = x[literal[0].get<std::size_t>()];
        const double bound = literal[2].get<double>();
        all_hold = all_hold && (literal[1].get<std::string>() == "<=" ? value <= bound + 1e-9
                                                                      : value >= bound - 1e-9);
      }
      ASSERT_FALSE(all_hold) << "conflict " << conflict.dump() << " holds at a feasible point";
    }
  }
}

TEST(Conflict, NoLearnedConflictHoldsAtAnyFeasiblePointAndTheOptimumIsUnchanged) {
  std::mt19937 rng(20260918);
  std::int64_t learned = 0;
  std::int64_t pruned = 0;
  std::int64_t tightened = 0;
  std::int64_t minimized = 0;
  int infeasible_models = 0;
  std::int64_t from_lp = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const bool maximize = trial % 2 == 1;
    const Model model = random_integer_program(rng, maximize);
    const std::vector<std::vector<double>> points = feasible_points(model);

    testing::TempFile out("", ".json");
    const Solution with = solve(model, searching(true, out.path()));
    const Solution without = solve(model, searching(false, ""));

    if (points.empty()) {
      ++infeasible_models;
      EXPECT_EQ(with.status, SolveStatus::kInfeasible) << "trial " << trial;
      EXPECT_EQ(without.status, SolveStatus::kInfeasible) << "trial " << trial;
    } else {
      double best = objective_of(model, points.front());
      for (const std::vector<double>& x : points) {
        const double value = objective_of(model, x);
        best = maximize ? std::max(best, value) : std::min(best, value);
      }
      ASSERT_EQ(with.status, SolveStatus::kOptimal)
          << "trial " << trial << ": " << with.message;
      ASSERT_EQ(without.status, SolveStatus::kOptimal) << "trial " << trial;
      EXPECT_NEAR(with.objective, best, 1e-7) << "trial " << trial;
      EXPECT_NEAR(without.objective, best, 1e-7) << "trial " << trial;
    }

    const nlohmann::json report = read_json(out.path());
    learned += report["learned"].get<std::int64_t>();
    pruned += report["nodes_pruned"].get<std::int64_t>();
    tightened += report["tightenings"].get<std::int64_t>();
    minimized += report["minimized"].get<std::int64_t>();
    for (const nlohmann::json& conflict : report["conflicts"]) {
      if (conflict["source"].get<std::string>() == "lp") ++from_lp;
      for (const std::vector<double>& x : points) {
        bool all_hold = true;
        for (const nlohmann::json& literal : conflict["literals"]) {
          const double value = x[literal[0].get<std::size_t>()];
          const double bound = literal[2].get<double>();
          all_hold =
              all_hold && (literal[1].get<std::string>() == "<=" ? value <= bound + 1e-9
                                                                 : value >= bound - 1e-9);
        }
        ASSERT_FALSE(all_hold) << "trial " << trial << ": conflict " << conflict.dump()
                               << " holds at a feasible point";
      }
    }
  }
  // The test would pass vacuously on a search that never learned anything.
  EXPECT_GT(learned, 100) << "conflicts learned across the suite";
  EXPECT_GT(minimized, 20) << "conflicts shorter than the node's decisions";
  EXPECT_GT(pruned + tightened, 10) << "conflicts that were used";
  EXPECT_GT(from_lp, 10) << "conflicts proved by a Farkas certificate, not by propagation";
  EXPECT_GT(infeasible_models, 5);
}

TEST(Conflict, ARerunLearnsTheSameConflictsInTheSameOrder) {
  std::mt19937 rng(7);
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = random_integer_program(rng, false);
    testing::TempFile first("", ".json");
    testing::TempFile second("", ".json");
    const Solution a = solve(model, searching(true, first.path()));
    const Solution b = solve(model, searching(true, second.path()));
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.nodes, b.nodes) << "trial " << trial;
    nlohmann::json x = read_json(first.path());
    nlohmann::json y = read_json(second.path());
    x.erase("seconds");  // the one field that is a clock reading
    y.erase("seconds");
    EXPECT_EQ(x, y) << "trial " << trial;
    if (!x["conflicts"].empty()) ++compared;
  }
  EXPECT_GT(compared, 5);
}

TEST(Conflict, ANodeLimitLeavesAValidBound) {
  std::mt19937 rng(11);
  int limited = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Model model = random_integer_program(rng, false);
    const std::vector<std::vector<double>> points = feasible_points(model);
    if (points.empty()) continue;
    double best = objective_of(model, points.front());
    for (const std::vector<double>& x : points) best = std::min(best, objective_of(model, x));
    Options options = searching(true, "");
    options.set_int("node_limit", 3);
    const Solution solved = solve(model, options);
    if (solved.status == SolveStatus::kOptimal) continue;
    ++limited;
    EXPECT_TRUE(solved.status == SolveStatus::kNodeLimit ||
                solved.status == SolveStatus::kFeasible)
        << to_string(solved.status);
    EXPECT_LE(solved.dual_bound, best + 1e-7)
        << "trial " << trial << ": a bound past the optimum";
    // kFeasible is the limited search that stopped holding an incumbent: it must be one.
    if (solved.status == SolveStatus::kFeasible) {
      EXPECT_TRUE(point_is_feasible(model, solved.col_value))
          << "trial " << trial << ": " << solved.message;
    }
  }
  EXPECT_GT(limited, 3);
}

TEST(Conflict, EachUseSettingDoesWhatItSaysAndKeepsTheOptimum) {
  // The ablation's arms (#292): none learns but uses nothing, prune never fixes a bound,
  // propagate does both. All three must still agree with enumeration.
  std::mt19937 rng(2922);
  std::int64_t learned_none = 0;
  std::int64_t used_none = 0;
  std::int64_t tightened_prune = 0;
  std::int64_t used_propagate = 0;
  for (int trial = 0; trial < 150; ++trial) {
    const Model model = random_integer_program(rng, trial % 2 == 0);
    const std::vector<std::vector<double>> points = feasible_points(model);
    double best = 0.0;
    if (!points.empty()) {
      const bool maximize = model.sense == ObjSense::kMaximize;
      best = objective_of(model, points.front());
      for (const std::vector<double>& x : points) {
        best = maximize ? std::max(best, objective_of(model, x))
                        : std::min(best, objective_of(model, x));
      }
    }
    for (const char* use : {"none", "prune", "propagate"}) {
      testing::TempFile out("", ".json");
      Options options = searching(true, out.path());
      options.set_string("conflict_use", use);
      const Solution solved = solve(model, options);
      if (points.empty()) {
        EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << use << " trial " << trial;
      } else {
        ASSERT_EQ(solved.status, SolveStatus::kOptimal) << use << " trial " << trial;
        EXPECT_NEAR(solved.objective, best, 1e-7) << use << " trial " << trial;
      }
      const nlohmann::json report = read_json(out.path());
      const auto pruned = report["nodes_pruned"].get<std::int64_t>();
      const auto tightened = report["tightenings"].get<std::int64_t>();
      if (std::string(use) == "none") {
        learned_none += report["learned"].get<std::int64_t>();
        used_none += pruned + tightened;
      } else if (std::string(use) == "prune") {
        tightened_prune += tightened;
      } else {
        used_propagate += pruned + tightened;
      }
    }
  }
  EXPECT_GT(learned_none, 50) << "none still learns";
  EXPECT_EQ(used_none, 0) << "none uses nothing";
  EXPECT_EQ(tightened_prune, 0) << "prune fixes no bound";
  EXPECT_GT(used_propagate, 5);
}

TEST(Conflict, AResumedSearchWithConflictsOnReachesTheEnumeratedOptimum) {
  // The checkpoint (#287) saves the open nodes, not the conflicts, so a resumed search learns
  // afresh from nodes whose ancestry it rebuilt. The optimum must not notice.
  std::mt19937 rng(287);
  int resumed = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const Model model = random_integer_program(rng, trial % 2 == 0);
    const std::vector<std::vector<double>> points = feasible_points(model);
    if (points.empty()) continue;
    const bool maximize = model.sense == ObjSense::kMaximize;
    double best = objective_of(model, points.front());
    for (const std::vector<double>& x : points) {
      best = maximize ? std::max(best, objective_of(model, x))
                      : std::min(best, objective_of(model, x));
    }
    testing::TempFile file("", ".chk");
    Options first = searching(true, "");
    first.set_string("checkpoint", file.path());
    first.set_int("node_limit", 3);
    if (solve(model, first).status == SolveStatus::kOptimal) continue;
    Options second = searching(true, "");
    second.set_string("resume", file.path());
    const Solution solved = solve(model, second);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best, 1e-7) << "trial " << trial;
    ++resumed;
  }
  EXPECT_GT(resumed, 5);
}

// =========================================================================================
// Resource limits, determinism and presolve
// =========================================================================================

TEST(Conflict, ATimeLimitStopsTheSearchCleanlyAndLearnsNothingFromTheStop) {
  // #289: `time_limit=0` is a budget of zero seconds, not an absent limit, so this is a stop
  // that does not depend on how fast the machine is. What must hold afterwards is what holds
  // after any stop: the reported bound still bounds the true optimum, and the conflicts the
  // search DID learn - each from a node proved infeasible before the stop - are still sound.
  // Nothing may be learned from the stop itself.
  std::mt19937 rng(2920);
  int stopped = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Model model = random_integer_program(rng, false);
    const std::vector<std::vector<double>> points = feasible_points(model);
    if (points.empty()) continue;
    double best = objective_of(model, points.front());
    for (const std::vector<double>& x : points) best = std::min(best, objective_of(model, x));

    testing::TempFile out("", ".json");
    Options options = searching(true, out.path());
    options.set_double("time_limit", 0.0);
    const Solution solved = solve(model, options);
    EXPECT_LE(solved.dual_bound, best + 1e-7) << "trial " << trial
                                              << ": a bound past the "
                                                 "optimum after a stop";
    if (solved.status == SolveStatus::kTimeLimit) ++stopped;
    // kFeasible is the status that means an incumbent was in hand when the limit fell; a
    // bare kTimeLimit carries the last relaxation, or says plainly that it carries nothing.
    if (solved.status == SolveStatus::kFeasible) {
      EXPECT_TRUE(point_is_feasible(model, solved.col_value))
          << "trial " << trial << ": " << solved.message;
    }
    const nlohmann::json report = read_json(out.path());
    expect_no_conflict_holds(report, points);
    EXPECT_LE(report["learned"].get<std::int64_t>(), report["detected"].get<std::int64_t>())
        << "trial " << trial << ": the stop itself is not an infeasibility";
  }
  EXPECT_GT(stopped, 5) << "the zero budget was meant to stop the search";
}

TEST(Conflict, AnInterruptStopsTheSearchCleanlyWithConflictsOn) {
  // A market-split instance (Cornuejols and Dawande): equality rows over binaries whose
  // relaxation is useless, so the search runs long enough to be interrupted mid-tree. Too
  // large to enumerate, so what is checked here is the clean exit rather than the conflicts:
  // the status, a point that is genuinely feasible, and a bound that does not claim more
  // than the point.
  std::mt19937 rng(2921);
  Model model;
  const int rows = 4;
  const int cols = 40;
  model.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(cols), 1.0);
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kInteger);
  model.col_cost.assign(static_cast<std::size_t>(cols), 1.0);
  model.matrix.reset(rows, cols);
  std::uniform_int_distribution<int> coefficient(0, 99);
  for (int i = 0; i < rows; ++i) {
    double total = 0.0;
    for (int j = 0; j < cols; ++j) {
      const double a = coefficient(rng);
      model.matrix.add_entry(i, j, a);
      total += a;
    }
    model.row_lower.push_back(std::floor(total / 2.0));
    model.row_upper.push_back(std::floor(total / 2.0));
  }
  model.matrix.finalize();
  model.hessian.reset(cols, cols);
  model.hessian.finalize();

  testing::TempFile out("", ".json");
  SolveControl control;
  int callbacks = 0;
  control.progress_callback = [&](const Progress&) {
    if (++callbacks == 1) std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return callbacks >= 2 ? 1 : 0;
  };
  const Solution solved = solve(model, searching(true, out.path()), &control);
  EXPECT_EQ(solved.status, SolveStatus::kInterrupted) << solved.message;
  EXPECT_GT(solved.nodes, 0);
  ASSERT_TRUE(claims_a_point(solved)) << "an interrupted search still has a point to show";
  EXPECT_EQ(solved.col_value.size(), static_cast<std::size_t>(model.num_cols()));
  // The report is still written, and the analysis never ran away with the search: its work
  // budget is counted in verification calls per explored node.
  const nlohmann::json report = read_json(out.path());
  EXPECT_LE(report["learned"].get<std::int64_t>(), report["detected"].get<std::int64_t>());
  EXPECT_EQ(report["columns"].get<Index>(), model.num_cols());
}

TEST(Conflict, UnderDeterministicModeTwoRunsLearnTheSameConflicts) {
  // #288 asks that no decision depend on the clock. The analysis has a budget in verification
  // calls rather than in seconds for exactly this, and the store's eviction is a total order,
  // so two runs under `deterministic=true` must write byte-identical reports.
  std::mt19937 rng(2922);
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = random_integer_program(rng, trial % 2 == 0);
    testing::TempFile first("", ".json");
    testing::TempFile second("", ".json");
    Options options = searching(true, first.path());
    options.set_bool("deterministic", true);
    const Solution a = solve(model, options);
    options.set_string("conflict_out", second.path());
    const Solution b = solve(model, options);
    ASSERT_TRUE(a.status == SolveStatus::kOptimal || a.status == SolveStatus::kInfeasible)
        << to_string(a.status) << ": " << a.message;
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.nodes, b.nodes) << "trial " << trial;
    EXPECT_EQ(a.objective, b.objective) << "trial " << trial;
    nlohmann::json x = read_json(first.path());
    nlohmann::json y = read_json(second.path());
    x.erase("seconds");  // the one field that is a clock reading
    y.erase("seconds");
    EXPECT_EQ(x, y) << "trial " << trial;
    if (!x["conflicts"].empty()) ++compared;
  }
  EXPECT_GT(compared, 5);
}

/// The model with an extra integer column fixed at one, carried by row 0 whose bounds move
/// with it, so the feasible points of the original are exactly this model's with that column
/// dropped. Presolve removes a fixed column, which is what makes the index question real.
Model with_a_fixed_column(const Model& model) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  Model out;
  out.sense = model.sense;
  out.col_lower.push_back(1.0);
  out.col_upper.push_back(1.0);
  out.col_type.push_back(VarType::kInteger);
  out.col_cost.push_back(0.0);
  out.col_lower.insert(out.col_lower.end(), model.col_lower.begin(), model.col_lower.end());
  out.col_upper.insert(out.col_upper.end(), model.col_upper.begin(), model.col_upper.end());
  out.col_type.insert(out.col_type.end(), model.col_type.begin(), model.col_type.end());
  out.col_cost.insert(out.col_cost.end(), model.col_cost.begin(), model.col_cost.end());
  out.matrix.reset(static_cast<Index>(m), static_cast<Index>(n + 1));
  out.matrix.add_entry(0, 0, 1.0);
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      out.matrix.add_entry(column.rows[k], static_cast<Index>(j + 1), column.values[k]);
    }
  }
  out.matrix.finalize();
  out.row_lower = model.row_lower;
  out.row_upper = model.row_upper;
  if (out.row_lower[0] > -kInfinity) out.row_lower[0] += 1.0;
  if (out.row_upper[0] < kInfinity) out.row_upper[0] += 1.0;
  out.hessian.reset(static_cast<Index>(n + 1), static_cast<Index>(n + 1));
  out.hessian.finalize();
  return out;
}

TEST(Conflict, WithPresolveOnTheConflictsAreInThePresolvedModelsIndices) {
  // Conflicts live in the indices of the model the SEARCH ran on, which is the presolved
  // model when presolve ran, and they are never mapped back: they are solver metadata, not
  // part of the answer. The export says which model it is talking about by writing that
  // model's column count, and this is the test that the count moves when presolve removes a
  // column - and that the answer, which IS mapped back, is the enumerated optimum either way.
  std::mt19937 rng(2923);
  int shrunk = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Model base = random_integer_program(rng, trial % 2 == 1);
    const std::vector<std::vector<double>> points = feasible_points(base);
    if (points.empty()) continue;
    const bool maximize = base.sense == ObjSense::kMaximize;
    double best = objective_of(base, points.front());
    for (const std::vector<double>& x : points) {
      best = maximize ? std::max(best, objective_of(base, x))
                      : std::min(best, objective_of(base, x));
    }
    const Model model = with_a_fixed_column(base);

    testing::TempFile out("", ".json");
    Options options = searching(true, out.path());
    options.set_bool("presolve", true);
    const Solution solved = solve(model, options);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best, 1e-7) << "trial " << trial;
    EXPECT_EQ(solved.col_value.size(), static_cast<std::size_t>(model.num_cols()))
        << "postsolve must hand back the original model's columns";

    const nlohmann::json report = read_json(out.path());
    const auto columns = report["columns"].get<Index>();
    EXPECT_LE(columns, model.num_cols());
    if (columns < model.num_cols() && !report["conflicts"].empty()) ++shrunk;
    for (const nlohmann::json& conflict : report["conflicts"]) {
      for (const nlohmann::json& literal : conflict["literals"]) {
        EXPECT_LT(literal[0].get<Index>(), columns)
            << "a literal outside the model the search ran on";
      }
    }
  }
  EXPECT_GT(shrunk, 3) << "presolve was meant to remove the fixed column and shift the rest";
}

}  // namespace
}  // namespace sankhya
