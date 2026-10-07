// SPDX-License-Identifier: Apache-2.0
// SANKHYA - warm start through the public entry point (#218), and across an edit that added
// or removed rows and columns (#913).
//
// THE NUMBER IS THE PROOF. A planner solves, moves a bound or a price, and solves again;
// the re-solve from the previous basis must take a small fraction of the cold solve's
// pivots, and the answer must be the cold solve's answer. Wall clock proves nothing on a
// machine whose speed changes minute to minute; the pivot count does not.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/tolerances.hpp"
#include "support/model_edit.hpp"

namespace sankhya {
namespace {

Options quiet(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", algorithm);
  return options;
}

Model netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

// SolveControl holds an atomic and cannot be moved, so the basis is copied into one in place.
void start_from(const Solution& previous, SolveControl* control) {
  control->start_col_status = previous.col_status;
  control->start_row_status = previous.row_status;
}

TEST(WarmStart, ATightenedBoundResolvesInAFewDualPivots) {
  // Solve (presolve on, the default), tighten one bound of a basic column, and re-solve
  // from the previous basis: the dual simplex restart. The old basis stays dual feasible,
  // so the pivots are the ones that restore primal feasibility - a handful.
  for (const char* name : {"afiro", "adlittle", "share2b", "israel"}) {
    Model model = netlib(name);
    const Solution first = solve(model, quiet("auto"));
    ASSERT_EQ(first.status, SolveStatus::kOptimal) << name << ": " << first.message;

    // The basic column with the largest value: halving its upper bound (or capping it when
    // it has none) is a change the optimum has to react to.
    Index pick = -1;
    for (Index j = 0; j < model.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (first.col_status[u] != BasisStatus::kBasic) continue;
      if (pick < 0 || first.col_value[u] > first.col_value[static_cast<std::size_t>(pick)]) {
        pick = j;
      }
    }
    ASSERT_GE(pick, 0) << name;
    const auto p = static_cast<std::size_t>(pick);
    if (first.col_value[p] <= 0.0) continue;  // nothing to tighten on this instance
    model.col_upper[p] = 0.5 * first.col_value[p];

    const Solution cold = solve(model, quiet("auto"));
    SolveControl control;
    start_from(first, &control);
    const Solution warm = solve(model, quiet("auto"), &control);
    ASSERT_EQ(warm.status, cold.status) << name << ": " << warm.message;
    if (cold.status != SolveStatus::kOptimal) continue;  // tightening can make it infeasible
    EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)))
        << name;
    EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;
    EXPECT_LE(warm.iterations * 5, cold.iterations + 5)
        << name << ": warm " << warm.iterations << " pivots against " << cold.iterations
        << " cold";
    EXPECT_LE(warm.primal_infeasibility, tol::kPrimalFeasibility);
  }
}

TEST(WarmStart, AChangedCostResolvesInAFewPrimalPivots) {
  // A cost change keeps the old basis primal feasible, so the PRIMAL simplex is the restart.
  for (const char* name : {"afiro", "adlittle", "share2b"}) {
    Model model = netlib(name);
    const Solution first = solve(model, quiet("simplex"));
    ASSERT_EQ(first.status, SolveStatus::kOptimal) << name << ": " << first.message;
    // Make the cheapest basic column expensive: the optimum has to move away from it.
    Index pick = -1;
    for (Index j = 0; j < model.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (first.col_status[u] != BasisStatus::kBasic || first.col_value[u] <= 0.0) continue;
      if (pick < 0 || model.col_cost[u] < model.col_cost[static_cast<std::size_t>(pick)]) {
        pick = j;
      }
    }
    ASSERT_GE(pick, 0) << name;
    const auto p = static_cast<std::size_t>(pick);
    model.col_cost[p] += 10.0 * std::max(1.0, std::fabs(model.col_cost[p]));

    const Solution cold = solve(model, quiet("simplex"));
    SolveControl control;
    start_from(first, &control);
    const Solution warm = solve(model, quiet("simplex"), &control);
    ASSERT_EQ(warm.status, SolveStatus::kOptimal) << name << ": " << warm.message;
    ASSERT_EQ(cold.status, SolveStatus::kOptimal) << name << ": " << cold.message;
    EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)))
        << name;
    EXPECT_LE(warm.iterations * 5, cold.iterations + 5)
        << name << ": warm " << warm.iterations << " pivots against " << cold.iterations
        << " cold";
  }
}

TEST(WarmStart, AWarmResolveThatTurnsInfeasibleStillCarriesACertificate) {
  // #218 meeting #217: the certificate is a property of the verdict, not of the start.
  Model model = netlib("afiro");
  const Solution first = solve(model, quiet("auto"));
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  // Force a contradiction: cap every column at 1e3 and demand a row activity of 1e9, which
  // no point inside that box can reach (raising the row alone would make afiro unbounded
  // instead, through its free directions).
  for (double& upper : model.col_upper) upper = std::min(upper, 1e3);
  model.row_lower[0] = 1e9;
  model.row_upper[0] = kInfinity;
  SolveControl control;
  start_from(first, &control);
  const Solution warm = solve(model, quiet("auto"), &control);
  ASSERT_EQ(warm.status, SolveStatus::kInfeasible) << warm.message;
  EXPECT_EQ(static_cast<Index>(warm.farkas_dual.size()), model.num_rows()) << warm.message;
}

TEST(WarmStart, ManyRandomEditsGiveTheColdAnswer) {
  // #524: a warm re-solve is only a faster route to the cold answer. Over many random edits
  // of costs, column bounds and row sides on four Netlib models, each edit applied to the
  // previous edited model and re-solved from the previous answer's basis, the warm and the
  // cold solve must agree on the status, and on the objective when both are optimal. A
  // fixed seed keeps the sequence reproducible; the edits are kept modest so most models
  // stay feasible, and an edit that makes one infeasible is checked like any other.
  std::mt19937_64 rng(524);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int compared = 0;
  int optimal_pairs = 0;
  for (const char* name : {"afiro", "adlittle", "sc50a", "share2b"}) {
    Model model = netlib(name);
    Solution previous = solve(model, quiet("auto"));
    ASSERT_EQ(previous.status, SolveStatus::kOptimal) << name << ": " << previous.message;
    for (int edit = 0; edit < 60; ++edit) {
      const double kind = unit(rng);
      if (kind < 0.4) {
        const auto j = static_cast<std::size_t>(rng() % model.col_cost.size());
        model.col_cost[j] *= 0.5 + unit(rng);  // a price within a factor of two
      } else if (kind < 0.7) {
        const auto j = static_cast<std::size_t>(rng() % model.col_upper.size());
        const double value = previous.status == SolveStatus::kOptimal
                                 ? previous.col_value[j]
                                 : std::max(model.col_lower[j], 0.0);
        // Cap the column near its current value, loosened again half the time, so the
        // change binds without forcing the model infeasible.
        const double cap = value * (0.8 + 0.4 * unit(rng)) + 1.0;
        model.col_upper[j] =
            std::isfinite(model.col_upper[j])
                ? std::max(model.col_lower[j], std::min(model.col_upper[j], cap))
                : std::max(model.col_lower[j], cap);
      } else {
        const auto i = static_cast<std::size_t>(rng() % model.row_lower.size());
        const double factor = 0.9 + 0.2 * unit(rng);
        if (std::isfinite(model.row_lower[i])) model.row_lower[i] *= factor;
        if (std::isfinite(model.row_upper[i])) model.row_upper[i] *= factor;
        if (model.row_lower[i] > model.row_upper[i])
          std::swap(model.row_lower[i], model.row_upper[i]);
      }
      const Solution cold = solve(model, quiet("auto"));
      SolveControl control;
      start_from(previous, &control);
      const Solution warm = solve(model, quiet("auto"), &control);
      ++compared;
      ASSERT_EQ(warm.status, cold.status) << name << " edit " << edit << ": warm "
                                          << warm.message << " / cold " << cold.message;
      if (cold.status == SolveStatus::kOptimal) {
        ++optimal_pairs;
        EXPECT_NEAR(warm.objective, cold.objective,
                    1e-7 * std::max(1.0, std::fabs(cold.objective)))
            << name << " edit " << edit;
        previous = warm;
      }
    }
  }
  EXPECT_EQ(compared, 240);
  EXPECT_GE(optimal_pairs, 120) << "most edits should leave the model feasible";
}

// ---- #913: a warm re-solve after adding or removing rows and columns ---------------------

using testing::add_column;
using testing::add_row;
using testing::Entry;
using testing::remove_column;
using testing::remove_row;

TEST(WarmStart, ABasisIsMappedByNameOntoAModelWithRowsAndColumnsAddedAndRemoved) {
  Model before = netlib("afiro");
  const Solution first = solve(before, quiet("auto"));
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  // Remove one basic column and one row whose slack is nonbasic (a binding row), add a
  // column and a row.
  Index basic_col = -1;
  for (Index j = 0; j < before.num_cols() && basic_col < 0; ++j) {
    if (first.col_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic) basic_col = j;
  }
  Index binding_row = -1;
  for (Index i = 0; i < before.num_rows() && binding_row < 0; ++i) {
    if (first.row_status[static_cast<std::size_t>(i)] != BasisStatus::kBasic) binding_row = i;
  }
  ASSERT_GE(basic_col, 0);
  ASSERT_GE(binding_row, 0);
  Model after = before;
  const std::string removed_col = after.col_names[static_cast<std::size_t>(basic_col)];
  remove_column(&after, basic_col);
  remove_row(&after, binding_row);
  add_column(&after, "NEWCOL", 1.0, 0.0, 10.0, {{0, 0, 1.0}, {1, 0, -1.0}});
  add_row(&after, "NEWROW", -kInfinity, 1e3, {{0, 0, 1.0}, {0, 1, 1.0}});

  std::vector<BasisStatus> cols;
  std::vector<BasisStatus> rows;
  BasisMapping mapping;
  ASSERT_TRUE(map_basis_by_name(before, first.col_status, first.row_status, after, &cols, &rows,
                                &mapping));
  ASSERT_EQ(static_cast<Index>(cols.size()), after.num_cols());
  ASSERT_EQ(static_cast<Index>(rows.size()), after.num_rows());
  EXPECT_TRUE(mapping.structural());
  EXPECT_EQ(mapping.matched_cols, before.num_cols() - 1);
  EXPECT_EQ(mapping.matched_rows, before.num_rows() - 1);
  EXPECT_EQ(mapping.new_cols, 1);
  EXPECT_EQ(mapping.new_rows, 1);
  EXPECT_EQ(mapping.removed_cols, 1);
  EXPECT_EQ(mapping.removed_rows, 1);
  EXPECT_EQ(mapping.removed_basic, 1);            // the column; the row's slack was nonbasic
  EXPECT_EQ(cols.back(), BasisStatus::kAtLower);  // the new column, at its lower bound
  EXPECT_EQ(rows.back(), BasisStatus::kBasic);    // the new row, on its slack
  // Every surviving entry keeps its status, found by name, not position.
  for (Index j = 0; j + 1 < after.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index old = j < basic_col ? j : j + 1;
    EXPECT_EQ(cols[u], first.col_status[static_cast<std::size_t>(old)]) << after.col_names[u];
    EXPECT_NE(after.col_names[u], removed_col);
  }

  // The basic column and the binding row gone, a new basic slack in: the count of basic
  // entries comes out at num_rows() here, and the solve repairs the rank if it must.
  const Solution cold = solve(after, quiet("auto"));
  SolveControl control;
  control.start_col_status = cols;
  control.start_row_status = rows;
  const Solution warm = solve(after, quiet("auto"), &control);
  ASSERT_EQ(warm.status, cold.status) << warm.message;
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;
  EXPECT_NEAR(warm.objective, cold.objective, 1e-7 * std::max(1.0, std::fabs(cold.objective)));
  EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;

  // Without names there is nothing to match by, and the helper says so.
  Model unnamed = after;
  unnamed.col_names.clear();
  unnamed.row_names.clear();
  EXPECT_FALSE(
      map_basis_by_name(before, first.col_status, first.row_status, unnamed, &cols, &rows));
}

TEST(WarmStart, AShortOrLongBasisIsCompletedBeforeTheSolve) {
  // A basis with one basic entry too few (a basic column removed) and one too many (a
  // binding row removed) is completed, not thrown away: the message says how.
  Model model = netlib("afiro");
  const Solution first = solve(model, quiet("auto"));
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  for (const int surplus : {-2, -1, 1, 2}) {
    SolveControl control;
    start_from(first, &control);
    int changed = 0;
    for (std::size_t j = 0; j < control.start_col_status.size() && changed < std::abs(surplus);
         ++j) {
      BasisStatus& s = control.start_col_status[j];
      if (surplus < 0 && s == BasisStatus::kBasic) {
        s = BasisStatus::kAtLower;
        ++changed;
      } else if (surplus > 0 && s == BasisStatus::kAtLower) {
        s = BasisStatus::kBasic;
        ++changed;
      }
    }
    ASSERT_EQ(changed, std::abs(surplus));
    const Solution warm = solve(model, quiet("auto"), &control);
    ASSERT_EQ(warm.status, SolveStatus::kOptimal) << surplus << ": " << warm.message;
    EXPECT_NEAR(warm.objective, first.objective, 1e-7 * std::fabs(first.objective)) << surplus;
    EXPECT_NE(warm.message.find("was completed with"), std::string::npos)
        << surplus << ": " << warm.message;
    EXPECT_LT(warm.iterations, first.iterations) << surplus;
  }
}

TEST(WarmStart, ManyRandomStructuralEditsGiveTheColdAnswer) {
  // #913: rows and columns added and removed on four Netlib models, each step applied to
  // the previous edited model and re-solved warm from the previous answer's basis, mapped
  // onto the new model by name. The warm and the cold solve must agree on the status, and
  // on the objective to 1e-7 relative when both are optimal; the pivot counts of the two
  // are recorded, and warm must take fewer on most edits. A fixed seed keeps the sequence
  // reproducible. An edit that leaves the model without an optimum is checked like any
  // other and then undone, so the walk continues from a model that has one.
  std::mt19937_64 rng(913);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int compared = 0;
  int optimal_pairs = 0;
  int warm_fewer = 0;
  int warm_routes = 0;
  int completed = 0;  // warm starts whose basis the edit left short or long
  int added_cols = 0;
  int added_rows = 0;
  int removed_cols = 0;
  int removed_rows = 0;
  std::int64_t warm_pivots = 0;
  std::int64_t cold_pivots = 0;
  int fresh = 0;
  for (const char* name : {"afiro", "adlittle", "sc50a", "share2b"}) {
    Model model = netlib(name);
    Solution previous = solve(model, quiet("auto"));
    ASSERT_EQ(previous.status, SolveStatus::kOptimal) << name << ": " << previous.message;
    double cost_scale = 0.0;
    for (const double c : model.col_cost) cost_scale = std::max(cost_scale, std::fabs(c));
    for (int step = 0; step < 40; ++step) {
      Model edited = model;
      const int edits = 1 + static_cast<int>(rng() % 3);
      int step_added_cols = 0;
      int step_added_rows = 0;
      int step_removed_cols = 0;
      int step_removed_rows = 0;
      for (int e = 0; e < edits; ++e) {
        const double kind = unit(rng);
        const Index n = edited.num_cols();
        const Index m = edited.num_rows();
        if (kind < 0.25 && n > 2) {
          remove_column(&edited, static_cast<Index>(rng() % static_cast<std::uint64_t>(n)));
          ++step_removed_cols;
        } else if (kind < 0.5 && m > 2) {
          remove_row(&edited, static_cast<Index>(rng() % static_cast<std::uint64_t>(m)));
          ++step_removed_rows;
        } else if (kind < 0.75) {
          // A new activity: a few rows, a price of either sign, a finite capacity.
          std::vector<Entry> column;
          const int nonzeros = 1 + static_cast<int>(rng() % 4);
          for (int t = 0; t < nonzeros; ++t) {
            const double sign = unit(rng) < 0.5 ? -1.0 : 1.0;
            column.push_back({static_cast<Index>(rng() % static_cast<std::uint64_t>(m)), 0,
                              sign * (0.5 + unit(rng))});
          }
          const double cost = (unit(rng) - 0.3) * std::max(1.0, cost_scale);
          add_column(&edited, "NEWC" + std::to_string(fresh++), cost, 0.0,
                     1.0 + 100.0 * unit(rng), column);
          ++step_added_cols;
        } else {
          // A new limit on a few columns, placed near their activity at the previous
          // optimum so it binds about half the time.
          std::vector<Entry> row;
          const int nonzeros = 2 + static_cast<int>(rng() % 4);
          double activity = 0.0;
          for (int t = 0; t < nonzeros; ++t) {
            const auto j = static_cast<Index>(rng() % static_cast<std::uint64_t>(n));
            const double value = 0.5 + unit(rng);
            row.push_back({0, j, value});
            // Columns this step added have no previous value; they start at zero.
            const std::string& col = edited.col_names[static_cast<std::size_t>(j)];
            for (Index k = 0; k < model.num_cols(); ++k) {
              if (model.col_names[static_cast<std::size_t>(k)] == col) {
                activity += value * previous.col_value[static_cast<std::size_t>(k)];
                break;
              }
            }
          }
          const double bound = activity * (0.9 + 0.2 * unit(rng));
          const double slack = 1e-3 * (1.0 + std::fabs(activity));
          if (unit(rng) < 0.5) {
            add_row(&edited, "NEWR" + std::to_string(fresh++), -kInfinity, bound + slack, row);
          } else {
            add_row(&edited, "NEWR" + std::to_string(fresh++), bound - slack, kInfinity, row);
          }
          ++step_added_rows;
        }
      }
      ASSERT_EQ(edited.validate(), "") << name << " step " << step;

      const Solution cold = solve(edited, quiet("auto"));
      SolveControl control;
      ASSERT_TRUE(map_basis_by_name(model, previous.col_status, previous.row_status, edited,
                                    &control.start_col_status, &control.start_row_status));
      const Solution warm = solve(edited, quiet("auto"), &control);
      ++compared;
      ASSERT_EQ(warm.status, cold.status) << name << " step " << step << ": warm "
                                          << warm.message << " / cold " << cold.message;
      if (cold.status != SolveStatus::kOptimal) continue;  // undone: `model` is unchanged
      ++optimal_pairs;
      EXPECT_NEAR(warm.objective, cold.objective,
                  1e-7 * std::max(1.0, std::fabs(cold.objective)))
          << name << " step " << step;
      const bool warm_route = warm.message.find("presolve bypassed") != std::string::npos &&
                              warm.message.find("#883") == std::string::npos;
      warm_routes += warm_route ? 1 : 0;
      completed += warm.message.find("was completed with") != std::string::npos ? 1 : 0;
      warm_fewer += warm.iterations < cold.iterations ? 1 : 0;
      warm_pivots += warm.iterations;
      cold_pivots += cold.iterations;
      added_cols += step_added_cols;
      added_rows += step_added_rows;
      removed_cols += step_removed_cols;
      removed_rows += step_removed_rows;
      model = std::move(edited);
      previous = warm;
    }
  }
  // The record: what the edits were, and the pivots warm against cold.
  std::printf(
      "[ structural ] %d steps compared, %d optimal pairs (%d columns added, %d removed; %d "
      "rows added, %d removed); warm route on %d, its basis completed on %d; warm fewer "
      "pivots on %d; pivots warm %lld against cold %lld\n",
      compared, optimal_pairs, added_cols, removed_cols, added_rows, removed_rows, warm_routes,
      completed, warm_fewer, static_cast<long long>(warm_pivots),
      static_cast<long long>(cold_pivots));
  EXPECT_EQ(compared, 160);
  EXPECT_GE(optimal_pairs, 80) << "most edits should leave the model with an optimum";
  EXPECT_GE(added_cols, 20);
  EXPECT_GE(added_rows, 20);
  EXPECT_GE(removed_cols, 20);
  EXPECT_GE(removed_rows, 20);
  EXPECT_GE(4 * warm_routes, 3 * optimal_pairs) << "the warm start should seldom fall back";
  EXPECT_GE(completed, 20) << "the edits should leave many bases short or long";
  EXPECT_GT(2 * warm_fewer, optimal_pairs) << "warm should take fewer pivots on most edits";
  EXPECT_LT(warm_pivots, cold_pivots);
}

TEST(WarmStart, ABasisOfTheWrongShapeRunsColdUnlessMappedByName) {
  // Statuses by index that do not fit the model, with no names to place them by: ignored,
  // and the solve runs cold, as before #913.
  {
    Model model = netlib("afiro");
    SolveControl control;
    control.start_col_status.assign(3, BasisStatus::kBasic);
    control.start_row_status.assign(2, BasisStatus::kBasic);
    const Solution s = solve(model, quiet("auto"), &control);
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
    EXPECT_NEAR(s.objective, -464.75314286, 1e-6 * 464.0);
    EXPECT_EQ(s.message.find("warm start"), std::string::npos) << s.message;
  }
  // The basis of afiro before the edit below: passed as it is it does not fit and the solve
  // runs cold; mapped by name it warm-starts and gives the cold answer.
  Model before = netlib("afiro");
  const Solution first = solve(before, quiet("auto"));
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  // A column at its lower bound of zero removed (the optimum does not need it), and a new
  // capacity row and a new column added.
  Model after = before;
  Index at_zero = -1;
  for (Index j = 0; j < before.num_cols() && at_zero < 0; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (first.col_status[u] == BasisStatus::kAtLower && before.col_lower[u] == 0.0) at_zero = j;
  }
  ASSERT_GE(at_zero, 0);
  remove_column(&after, at_zero);
  add_column(&after, "NEWCOL", 1.0, 0.0, 10.0, {{0, 0, 1.0}});
  add_row(&after, "NEWROW", -kInfinity, 1e3, {{0, 0, 1.0}, {0, 1, 1.0}});
  const Solution cold = solve(after, quiet("auto"));
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;
  {
    SolveControl control;
    start_from(first, &control);
    const Solution s = solve(after, quiet("auto"), &control);
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
    EXPECT_NEAR(s.objective, cold.objective, 1e-7 * std::max(1.0, std::fabs(cold.objective)));
    EXPECT_EQ(s.message.find("warm start"), std::string::npos) << s.message;
  }
  SolveControl control;
  ASSERT_TRUE(map_basis_by_name(before, first.col_status, first.row_status, after,
                                &control.start_col_status, &control.start_row_status));
  const Solution warm = solve(after, quiet("auto"), &control);
  ASSERT_EQ(warm.status, SolveStatus::kOptimal) << warm.message;
  EXPECT_NEAR(warm.objective, cold.objective, 1e-7 * std::max(1.0, std::fabs(cold.objective)));
  EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;
}

// #913 part 2, through the public entry point: SolveControl::start_col_value /
// start_row_dual, not solve_pdhg() called directly (test_pdhg.cpp covers that function in
// isolation). Before this test, nothing in solve.cpp ever read those fields - the struct
// pdhg::solve_pdhg takes existed and was tested, but no caller through solve() ever built
// one, so the feature was unreachable end to end despite the unit-level tests passing.
//
// presolve=false: unlike the LP basis warm start (which bypasses presolve, #218), a PDHG
// warm iterate does not - want_pdhg never sets chosen.engine's supports_warm_start, so the
// model solve_pdhg actually sees is the PRESOLVED one, which can hold fewer columns than
// `first.col_value` (in the ORIGINAL model's space). The engine's own size check then finds
// a mismatch and runs cold, silently and safely, but without presolve's reduction this test
// could not tell "ran cold because the sizes did not match" from "ran cold because the
// plumbing was never wired" - which is the bug this test exists to catch. Left as a
// documented limitation rather than fixed here: bypassing presolve for PDHG the way the
// basis warm start does is a larger change this slice does not make.
TEST(WarmStart, APdhgIterateThroughSolveControlResolvesInFewerIterationsThanCold) {
  Model model = netlib("afiro");
  Options options = quiet("pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_bool("presolve", false);
  const Solution first = solve(model, options);
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;

  // A small cost change: the optimum moves, but the previous iterate is still a reasonable
  // starting point for it.
  Model edited = model;
  for (Index j = 0; j < edited.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    edited.col_cost[u] *= 1.01;
  }
  const Solution cold = solve(edited, options);
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;

  SolveControl control;
  control.start_col_value = first.col_value;
  control.start_row_dual = first.row_dual;
  const Solution warm = solve(edited, options, &control);
  ASSERT_EQ(warm.status, SolveStatus::kOptimal) << warm.message;
  EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)));
  EXPECT_LT(warm.iterations, cold.iterations)
      << "warm " << warm.iterations << " vs cold " << cold.iterations;
}

// A warm iterate of the wrong size (model.num_cols() changed) is ignored: solve() must not
// crash or hand the engine a vector it cannot use, and the cold run comes back BIT FOR BIT -
// as it does through a SolveControl that offers no iterate at all. The cold start is the
// default every benchmark CSV was produced with, so wiring the warm start in must not move
// one bit of it.
TEST(WarmStart, APdhgIterateOfTheWrongSizeIsIgnoredAndTheSolveRunsColdBitForBit) {
  Model model = netlib("afiro");
  Options options = quiet("pdhg");
  const Solution cold = solve(model, options);
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;

  SolveControl wrong_size;
  wrong_size.start_col_value.assign(3, 0.0);
  SolveControl none;
  for (SolveControl* control : {&wrong_size, &none}) {
    const Solution s = solve(model, options, control);
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
    EXPECT_EQ(s.iterations, cold.iterations);
    EXPECT_EQ(s.objective, cold.objective);
    EXPECT_EQ(s.col_value, cold.col_value);
    EXPECT_EQ(s.row_dual, cold.row_dual);
  }
}

}  // namespace
}  // namespace sankhya
