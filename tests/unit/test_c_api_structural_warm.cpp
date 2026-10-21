// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a warm re-solve through the C API after rows and columns are added and removed
// (#913).
//
// A planner holding a sankhya_model adds a unit (columns), a limit (a row), or drops a
// period (rows and columns), and calls sankhya_solve_from() with yesterday's solution. The
// basis is placed on the edited model by name - a new row basic on its slack, a new column
// nonbasic at a bound - and the removals repaired with row logicals (Maros, "Computational
// Techniques of the Simplex Method", sec. 9.4; Koberstein, "The dual simplex method,
// techniques for a fast and stable implementation", 2005, ch. 6 on restarting from a
// modified basis). The warm answer must be the cold answer: the objective to 1e-9
// relative, and both points checked here, against a C++ copy of the edited model kept in
// step, for primal feasibility, dual sign, complementarity and a zero duality gap.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/sankhya.h"
#include "sankhya/tolerances.hpp"
#include "support/model_edit.hpp"

namespace sankhya {
namespace {

using testing::Entry;

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

struct Answer {
  sankhya_solve_status status = SANKHYA_NOT_SOLVED;
  double objective = 0.0;
  std::int64_t iterations = 0;
  std::string message;
  std::vector<double> x, activity, y, d;
};

Answer read_answer(const sankhya_solution* s, int cols, int rows) {
  Answer a;
  a.status = sankhya_solution_status(s);
  a.objective = sankhya_solution_objective(s);
  a.iterations = sankhya_solution_iterations(s);
  a.message = sankhya_solution_message(s);
  if (a.status == SANKHYA_OPTIMAL) {
    a.x.resize(static_cast<std::size_t>(cols));
    a.d.resize(static_cast<std::size_t>(cols));
    a.activity.resize(static_cast<std::size_t>(rows));
    a.y.resize(static_cast<std::size_t>(rows));
    EXPECT_EQ(sankhya_solution_col_values(s, a.x.data(), cols), SANKHYA_OK);
    EXPECT_EQ(sankhya_solution_col_duals(s, a.d.data(), cols), SANKHYA_OK);
    EXPECT_EQ(sankhya_solution_row_activities(s, a.activity.data(), rows), SANKHYA_OK);
    EXPECT_EQ(sankhya_solution_row_duals(s, a.y.data(), rows), SANKHYA_OK);
  }
  return a;
}

/// The optimality conditions of a minimisation LP, checked from `model` alone: x within its
/// bounds and Ax within the row bounds, d = c - A'y, each multiplier of the sign its active
/// bound allows and zero where no bound is active, and the dual objective equal to c'x.
/// Empty when they hold; else the first that does not.
std::string verify(const Model& model, const Answer& a) {
  const double ptol = tol::kPrimalFeasibility;
  const double dtol = tol::kDualFeasibility;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> ax(m, 0.0);
  std::vector<double> aty(n, 0.0);
  model.matrix.multiply_add(a.x.data(), ax.data());
  model.matrix.transpose_multiply_add(a.y.data(), aty.data());
  char why[256];
  double primal = model.objective_offset;
  double dual = model.objective_offset;
  // One side of a multiplier's dual objective term: its active bound, or zero when it is
  // within tolerance of zero (so an infinite bound is never multiplied).
  const auto term = [&](double mult, double lower, double upper) {
    if (std::fabs(mult) <= dtol) return 0.0;
    return mult * (mult > 0 ? lower : upper);
  };
  // Sign of a multiplier against its activity: positive only at the lower bound, negative
  // only at the upper (the convention of Solution::row_dual).
  const auto sign_ok = [&](double value, double lower, double upper, double mult) {
    const double scale = 1.0 + std::fabs(value);
    const bool at_lower = std::isfinite(lower) && value <= lower + ptol * scale;
    const bool at_upper = std::isfinite(upper) && value >= upper - ptol * scale;
    return (mult <= dtol || at_lower) && (mult >= -dtol || at_upper);
  };
  for (std::size_t j = 0; j < n; ++j) {
    const double x = a.x[j];
    const double l = model.col_lower[j];
    const double u = model.col_upper[j];
    const double scale = 1.0 + std::fabs(x);
    if (x < l - ptol * scale || x > u + ptol * scale) {
      std::snprintf(why, sizeof why, "column %zu at %.17g outside [%.17g, %.17g]", j, x, l, u);
      return why;
    }
    const double reduced = model.col_cost[j] - aty[j];
    if (std::fabs(reduced - a.d[j]) > dtol * (1.0 + std::fabs(model.col_cost[j]))) {
      std::snprintf(why, sizeof why, "column %zu reduced cost %.17g, c - A'y gives %.17g", j,
                    a.d[j], reduced);
      return why;
    }
    if (!sign_ok(x, l, u, a.d[j])) {
      std::snprintf(why, sizeof why, "column %zu at %.17g in [%.17g, %.17g], reduced cost %.3g",
                    j, x, l, u, a.d[j]);
      return why;
    }
    primal += model.col_cost[j] * x;
    dual += term(a.d[j], l, u);
  }
  for (std::size_t i = 0; i < m; ++i) {
    const double r = ax[i];
    const double l = model.row_lower[i];
    const double u = model.row_upper[i];
    const double scale = 1.0 + std::fabs(r);
    if (r < l - ptol * scale || r > u + ptol * scale) {
      std::snprintf(why, sizeof why, "row %zu activity %.17g outside [%.17g, %.17g]", i, r, l,
                    u);
      return why;
    }
    if (!sign_ok(r, l, u, a.y[i])) {
      std::snprintf(why, sizeof why, "row %zu activity %.17g in [%.17g, %.17g], dual %.3g", i,
                    r, l, u, a.y[i]);
      return why;
    }
    dual += term(a.y[i], l, u);
  }
  const double scale = std::max(1.0, std::fabs(primal));
  if (std::fabs(primal - a.objective) > 1e-9 * scale) {
    std::snprintf(why, sizeof why, "objective %.17g, c'x gives %.17g", a.objective, primal);
    return why;
  }
  // The dual objective sums every active bound times its multiplier, so it carries the
  // multipliers' 1e-7 tolerance over bounds of any size: held to that, relative.
  if (std::fabs(primal - dual) > tol::kDualFeasibility * scale) {
    std::snprintf(why, sizeof why, "duality gap: c'x %.17g, dual objective %.17g", primal,
                  dual);
    return why;
  }
  return "";
}

/// A fresh handle holding `model`, built call by call.
sankhya_model* handle_of(const Model& model) {
  sankhya_model* h = sankhya_model_create();
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto k = static_cast<std::size_t>(j);
    EXPECT_EQ(
        sankhya_model_add_column(h, model.col_cost[k], model.col_lower[k], model.col_upper[k],
                                 0, model.col_names[k].c_str(), nullptr),
        SANKHYA_OK);
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto k = static_cast<std::size_t>(i);
    EXPECT_EQ(sankhya_model_add_row(h, model.row_lower[k], model.row_upper[k],
                                    model.row_names[k].c_str(), nullptr),
              SANKHYA_OK);
  }
  for (const Entry& e : testing::entries_of(model)) {
    EXPECT_EQ(sankhya_model_set_coefficient(h, static_cast<int>(e.row), static_cast<int>(e.col),
                                            e.value),
              SANKHYA_OK);
  }
  EXPECT_EQ(sankhya_model_set_objective_offset(h, model.objective_offset), SANKHYA_OK);
  return h;
}

Answer solve_handle(sankhya_model* h, const sankhya_solution* start, sankhya_solution** keep) {
  sankhya_options* options = sankhya_options_create();
  EXPECT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  // The dual simplex is the engine a structural warm start restarts; cold runs it too, so
  // the pivot counts compare one engine with itself.
  EXPECT_EQ(sankhya_options_set_string(options, "algorithm", "dual-simplex"), SANKHYA_OK);
  sankhya_solution* s = nullptr;
  EXPECT_EQ(sankhya_solve_from(h, options, start, &s), SANKHYA_OK) << sankhya_last_error();
  sankhya_options_free(options);
  Answer a = read_answer(s, sankhya_model_num_cols(h), sankhya_model_num_rows(h));
  if (keep != nullptr) {
    *keep = s;
  } else {
    sankhya_solution_free(s);
  }
  return a;
}

TEST(CApiStructuralWarm, ManyRandomEditsThroughTheCApiGiveTheVerifiedColdAnswer) {
  std::mt19937_64 rng(9131);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int compared = 0, optimal_pairs = 0, warm_fewer = 0, mapped = 0, edits_total = 0;
  int added_cols = 0, added_rows = 0, removed_cols = 0, removed_rows = 0;
  std::int64_t warm_pivots = 0, cold_pivots = 0;
  int fresh = 0;
  for (const char* name : {"afiro", "adlittle", "sc50a", "sc105", "share2b", "blend"}) {
    Model model;  // the accepted model, in step with the handle
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    ASSERT_EQ(model.sense, ObjSense::kMinimize) << name;
    // The handle starts as a READ model, so the first edit exercises the thaw of its
    // frozen matrix as well (c_api_edit.cpp).
    sankhya_model* h = sankhya_model_create();
    ASSERT_EQ(sankhya_model_read(h, netlib_path(name).c_str()), SANKHYA_OK) << name;
    sankhya_solution* previous = nullptr;
    const Answer first = solve_handle(h, nullptr, &previous);
    ASSERT_EQ(first.status, SANKHYA_OPTIMAL) << name;
    std::vector<double> x = first.x;  // the accepted model's last point
    double cost_scale = 1.0;
    for (const double c : model.col_cost) cost_scale = std::max(cost_scale, std::fabs(c));

    for (int step = 0; step < 12; ++step) {
      Model edited = model;
      std::vector<double> x_edited = x;
      const int edits = 1 + static_cast<int>(rng() % 3);
      int step_counts[4] = {0, 0, 0, 0};
      for (int e = 0; e < edits; ++e) {
        const double kind = unit(rng);
        const int n = static_cast<int>(edited.num_cols());
        const int m = static_cast<int>(edited.num_rows());
        if (kind < 0.25 && n > 2) {
          const int j = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
          testing::remove_column(&edited, j);
          x_edited.erase(x_edited.begin() + j);
          ASSERT_EQ(sankhya_model_delete_cols(h, 1, &j), SANKHYA_OK) << sankhya_last_error();
          ++step_counts[0];
        } else if (kind < 0.5 && m > 2) {
          const int i = static_cast<int>(rng() % static_cast<std::uint64_t>(m));
          testing::remove_row(&edited, i);
          ASSERT_EQ(sankhya_model_delete_rows(h, 1, &i), SANKHYA_OK) << sankhya_last_error();
          ++step_counts[1];
        } else if (kind < 0.75) {
          // A new activity on a few distinct rows, a price of either sign, a finite range.
          std::vector<Entry> column;
          const int nonzeros = 1 + static_cast<int>(rng() % 4);
          for (int t = 0; t < nonzeros; ++t) {
            const auto i = static_cast<Index>(rng() % static_cast<std::uint64_t>(m));
            if (std::none_of(column.begin(), column.end(),
                             [&](const Entry& c) { return c.row == i; })) {
              column.push_back({i, 0, (unit(rng) < 0.5 ? -1.0 : 1.0) * (0.5 + unit(rng))});
            }
          }
          const double cost = (unit(rng) - 0.3) * cost_scale;
          const double upper = 1.0 + 100.0 * unit(rng);
          const std::string col_name = "NEWC" + std::to_string(fresh++);
          testing::add_column(&edited, col_name, cost, 0.0, upper, column);
          x_edited.push_back(0.0);
          int j = -1;
          ASSERT_EQ(sankhya_model_add_column(h, cost, 0.0, upper, 0, col_name.c_str(), &j),
                    SANKHYA_OK);
          for (const Entry& c : column) {
            ASSERT_EQ(sankhya_model_set_coefficient(h, static_cast<int>(c.row), j, c.value),
                      SANKHYA_OK);
          }
          ++step_counts[2];
        } else {
          // A new limit on a few distinct columns, near their activity at the last point so
          // it binds about half the time.
          std::vector<Entry> row;
          const int nonzeros = 2 + static_cast<int>(rng() % 4);
          double activity = 0.0;
          for (int t = 0; t < nonzeros; ++t) {
            const auto j = static_cast<Index>(rng() % static_cast<std::uint64_t>(n));
            if (std::any_of(row.begin(), row.end(), [&](const Entry& r) { return r.col == j; }))
              continue;
            const double value = 0.5 + unit(rng);
            row.push_back({0, j, value});
            activity += value * x_edited[static_cast<std::size_t>(j)];
          }
          const double bound = activity * (0.9 + 0.2 * unit(rng));
          const double slack = 1e-3 * (1.0 + std::fabs(activity));
          const bool upper_side = unit(rng) < 0.5;
          const double lo = upper_side ? -kInfinity : bound - slack;
          const double hi = upper_side ? bound + slack : kInfinity;
          const std::string row_name = "NEWR" + std::to_string(fresh++);
          testing::add_row(&edited, row_name, lo, hi, row);
          int i = -1;
          ASSERT_EQ(sankhya_model_add_row(h, lo, hi, row_name.c_str(), &i), SANKHYA_OK);
          for (const Entry& r : row) {
            ASSERT_EQ(sankhya_model_set_coefficient(h, i, static_cast<int>(r.col), r.value),
                      SANKHYA_OK);
          }
          ++step_counts[3];
        }
      }
      ASSERT_EQ(edited.validate(), "") << name << " step " << step;
      ASSERT_EQ(sankhya_model_num_cols(h), static_cast<int>(edited.num_cols()));
      ASSERT_EQ(sankhya_model_num_rows(h), static_cast<int>(edited.num_rows()));

      const Answer cold = solve_handle(h, nullptr, nullptr);
      sankhya_solution* warm_kept = nullptr;
      const Answer warm = solve_handle(h, previous, &warm_kept);
      ++compared;
      ASSERT_EQ(warm.status, cold.status) << name << " step " << step << ": warm "
                                          << warm.message << " / cold " << cold.message;
      if (cold.status != SANKHYA_OPTIMAL) {
        // Undone: the handle is rebuilt from the accepted model, and the walk goes on from
        // a model that has an optimum, still warm from `previous`.
        sankhya_solution_free(warm_kept);
        sankhya_model_free(h);
        h = handle_of(model);
        continue;
      }
      ++optimal_pairs;
      edits_total += edits;
      EXPECT_NEAR(warm.objective, cold.objective,
                  1e-9 * std::max(1.0, std::fabs(cold.objective)))
          << name << " step " << step;
      EXPECT_EQ(verify(edited, cold), "") << name << " step " << step << " cold";
      EXPECT_EQ(verify(edited, warm), "") << name << " step " << step << " warm";
      mapped += warm.message.find("presolve bypassed") != std::string::npos &&
                        warm.message.find("#883") == std::string::npos
                    ? 1
                    : 0;
      if (warm.iterations >= cold.iterations && compared < 12) {
        std::printf("DIAG %s step %d warm %lld cold %lld: %s\n", name, step,
                    static_cast<long long>(warm.iterations),
                    static_cast<long long>(cold.iterations), warm.message.c_str());
      }
      warm_fewer += warm.iterations < cold.iterations ? 1 : 0;
      warm_pivots += warm.iterations;
      cold_pivots += cold.iterations;
      removed_cols += step_counts[0];
      removed_rows += step_counts[1];
      added_cols += step_counts[2];
      added_rows += step_counts[3];
      model = std::move(edited);
      x = warm.x;
      sankhya_solution_free(previous);
      previous = warm_kept;
    }
    sankhya_solution_free(previous);
    sankhya_model_free(h);
  }
  std::printf(
      "[ c api structural ] %d steps compared, %d optimal pairs over %d edits (%d columns "
      "added, %d removed; %d rows added, %d removed); warm route on %d; warm fewer pivots on "
      "%d; pivots warm %lld against cold %lld\n",
      compared, optimal_pairs, edits_total, added_cols, removed_cols, added_rows, removed_rows,
      mapped, warm_fewer, static_cast<long long>(warm_pivots),
      static_cast<long long>(cold_pivots));
  EXPECT_EQ(compared, 72);
  EXPECT_GE(edits_total, 50);
  EXPECT_GE(optimal_pairs, 50);
  EXPECT_GE(std::min({added_cols, added_rows, removed_cols, removed_rows}), 10);
  EXPECT_GE(4 * mapped, 3 * optimal_pairs) << "the warm start should seldom fall back";
  EXPECT_LT(warm_pivots, cold_pivots);
}

TEST(CApiStructuralWarm, DeletingRowsAndColumnsKeepsTheRestOfAReadModel) {
  // afiro read from its file, row 0 and columns 1 and 3 removed (given out of order and
  // with a repeat), against the same edits on the C++ model: the same optimum. A
  // coefficient set on a read model adds to its matrix rather than replacing it.
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &model).ok);
  sankhya_model* h = sankhya_model_create();
  ASSERT_EQ(sankhya_model_read(h, netlib_path("afiro").c_str()), SANKHYA_OK);
  const int nnz = sankhya_model_num_nonzeros(h);
  ASSERT_EQ(sankhya_model_set_coefficient(h, 0, 0, 2.5), SANKHYA_OK);
  EXPECT_EQ(sankhya_model_num_nonzeros(h), nnz + (model.matrix.at(0, 0) == 0.0 ? 1 : 0));
  std::vector<Entry> entries = testing::entries_of(model);
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [](const Entry& e) { return e.row == 0 && e.col == 0; }),
                entries.end());
  entries.push_back({0, 0, 2.5});
  testing::rebuild(&model, entries);

  const int rows[] = {0};
  const int cols[] = {3, 1, 3};
  ASSERT_EQ(sankhya_model_delete_rows(h, 1, rows), SANKHYA_OK) << sankhya_last_error();
  ASSERT_EQ(sankhya_model_delete_cols(h, 3, cols), SANKHYA_OK) << sankhya_last_error();
  testing::remove_row(&model, 0);
  testing::remove_column(&model, 3);
  testing::remove_column(&model, 1);
  EXPECT_EQ(sankhya_model_num_rows(h), static_cast<int>(model.num_rows()));
  EXPECT_EQ(sankhya_model_num_cols(h), static_cast<int>(model.num_cols()));
  EXPECT_EQ(sankhya_model_num_nonzeros(h), static_cast<int>(model.num_nonzeros()));

  const int outside[] = {static_cast<int>(model.num_cols())};
  EXPECT_EQ(sankhya_model_delete_cols(h, 1, outside), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_model_num_cols(h), static_cast<int>(model.num_cols()));

  const Answer a = solve_handle(h, nullptr, nullptr);
  ASSERT_EQ(a.status, SANKHYA_OPTIMAL) << a.message;
  EXPECT_EQ(verify(model, a), "");
  sankhya_model_free(h);
}

}  // namespace
}  // namespace sankhya
