// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Dantzig-Wolfe decomposition of an LP over linking rows (#1048).
//
// Tested against the ordinary solve of the same model, which shares none of its code path:
// random LPs whose blocks share only rows (every column bounded and every row built around one
// point of the box, so each is feasible and bounded), a model whose blocks' own optima violate
// the linking row and so needs the first phase, models that must fall back to the monolithic
// engines with the right verdict, and thread counts that must not change a bit.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options dantzig_wolfe(int blocks) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("decomposition", "dantzig-wolfe");
  o.set_int("decomposition_blocks", blocks);
  return o;
}

Solution monolithic(const Model& model) {
  Options o;
  o.set_bool("log_to_console", false);
  return solve(model, o);
}

/// `blocks` groups of columns, each with rows of its own, and `linking` rows over columns of
/// every block.
Model linked_blocks(std::mt19937& rng, int blocks, int rows, int cols, int linking) {
  const Index n = blocks * cols;
  const Index m = blocks * rows + linking;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> coefficient(-3.0, 3.0);
  std::uniform_real_distribution<double> cost(-5.0, 5.0);
  std::uniform_real_distribution<double> slack(0.0, 2.0);
  std::uniform_int_distribution<int> kind(0, 2);
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  std::vector<double> point(static_cast<std::size_t>(n));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_lower[u] = 0.0;
    model.col_upper[u] = 5.0;
    model.col_cost[u] = cost(rng);
    point[u] = 3.0 * unit(rng);
  }
  model.matrix.reset(m, n);
  Index row = 0;
  const auto add_row = [&](const std::vector<Index>& columns) {
    double activity = 0.0;
    for (const Index c : columns) {
      double a = coefficient(rng);
      if (std::fabs(a) < 0.2) a = 1.0;
      model.matrix.add_entry(row, c, a);
      activity += a * point[static_cast<std::size_t>(c)];
    }
    const auto u = static_cast<std::size_t>(row);
    switch (kind(rng)) {
      case 0: model.row_upper[u] = activity + slack(rng); break;
      case 1: model.row_lower[u] = activity - slack(rng); break;
      default:
        model.row_lower[u] = activity - slack(rng);
        model.row_upper[u] = activity + slack(rng);
    }
    ++row;
  };
  for (int b = 0; b < blocks; ++b) {
    for (int r = 0; r < rows; ++r) {
      // A chain through the block's columns, so that each block is connected.
      std::vector<Index> columns{b * cols + r % cols, b * cols + (r + 1) % cols};
      const Index extra = b * cols + static_cast<Index>(unit(rng) * cols);
      if (std::find(columns.begin(), columns.end(), extra) == columns.end())
        columns.push_back(extra);
      add_row(columns);
    }
  }
  for (int r = 0; r < linking; ++r) {
    std::vector<Index> columns;
    for (int b = 0; b < blocks; ++b) columns.push_back(b * cols + (r + b) % cols);
    add_row(columns);
  }
  model.matrix.finalize();
  return model;
}

void expect_agrees(const Solution& s, const Solution& reference, const std::string& what) {
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << what << ": " << s.message;
  EXPECT_NEAR(s.objective, reference.objective,
              1e-6 * std::max(1.0, std::fabs(reference.objective)))
      << what << " (" << s.algorithm << ")";
  EXPECT_LE(s.primal_infeasibility_scaled, 1e-7) << what;
  EXPECT_LE(s.dual_infeasibility_scaled, 1e-7) << what;
}

TEST(DantzigWolfe, IsReachedOnlyByItsOwnModeAndAutoNeverRunsIt) {
  std::mt19937 rng(1048);
  const Model model = linked_blocks(rng, 3, 6, 5, 2);
  Options automatic = dantzig_wolfe(3);
  automatic.set_string("decomposition", "auto");
  EXPECT_NE(solve(model, automatic).algorithm, "dantzig-wolfe");
  EXPECT_EQ(Options().get_string("decomposition"), "off");
}

TEST(DantzigWolfe, AgreesWithTheMonolithicSolveOnRandomLinkedBlocks) {
  std::mt19937 rng(1048525);
  int decomposed = 0;
  const int trials = 60;
  for (int trial = 0; trial < trials; ++trial) {
    const int blocks = 2 + trial % 3;
    const Model model = linked_blocks(rng, blocks, 4 + trial % 3, 5, 1 + trial % 3);
    const Solution reference = monolithic(model);
    ASSERT_EQ(reference.status, SolveStatus::kOptimal) << "trial " << trial;
    const Solution s = solve(model, dantzig_wolfe(blocks));
    expect_agrees(s, reference, "trial " + std::to_string(trial));
    if (s.algorithm == "dantzig-wolfe") {
      ++decomposed;
      EXPECT_NE(s.message.find("Dantzig-Wolfe"), std::string::npos) << s.message;
      EXPECT_LE(s.relative_gap, 1e-7) << "trial " << trial;
    }
  }
  // The test would pass vacuously if every trial quietly fell back to the monolithic engines.
  EXPECT_GT(decomposed, trials * 7 / 10) << decomposed << " of " << trials << " decomposed";
}

TEST(DantzigWolfe, BlocksWhoseOwnOptimaBreakTheLinkingRowNeedTheFirstPhase) {
  // Two blocks, each x_a + x_b <= 8 over [0, 10] at cost 1 and 2: on their own both stay at
  // zero, which the linking row x_0 + x_2 = 12 forbids. The first phase finds weights that meet
  // it; the second the cheapest, x_0 = 8 and x_2 = 4 (cost 8 + 4 = 12).
  Model model;
  model.resize_columns(4);
  model.resize_rows(3);
  model.matrix.reset(3, 4);
  for (Index j = 0; j < 4; ++j) {
    model.col_lower[static_cast<std::size_t>(j)] = 0.0;
    model.col_upper[static_cast<std::size_t>(j)] = 10.0;
    model.col_cost[static_cast<std::size_t>(j)] = j % 2 == 0 ? 1.0 : 2.0;
  }
  for (Index b = 0; b < 2; ++b) {
    model.matrix.add_entry(b, 2 * b, 1.0);
    model.matrix.add_entry(b, 2 * b + 1, 1.0);
    model.row_upper[static_cast<std::size_t>(b)] = 8.0;
  }
  model.matrix.add_entry(2, 0, 1.0);
  model.matrix.add_entry(2, 2, 1.0);
  model.row_lower[2] = 12.0;
  model.row_upper[2] = 12.0;
  model.matrix.finalize();
  const Solution reference = monolithic(model);
  ASSERT_EQ(reference.status, SolveStatus::kOptimal);
  const Solution s = solve(model, dantzig_wolfe(2));
  expect_agrees(s, reference, "first phase");
  EXPECT_EQ(s.algorithm, "dantzig-wolfe") << s.message;
  EXPECT_NEAR(s.objective, 12.0, 1e-9);
  EXPECT_NEAR(s.col_value[0] + s.col_value[2], 12.0, 1e-7);
}

TEST(DantzigWolfe, AnInfeasibleLpIsReportedInfeasibleByTheOrdinaryEngines) {
  std::mt19937 rng(5);
  Model model = linked_blocks(rng, 2, 5, 4, 1);
  // The linking row asks for more than every column at its upper bound can give.
  const Index last = model.num_rows() - 1;
  double reach = 0.0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.rows[k] == last) reach += std::fabs(column.values[k]) * 5.0;
    }
  }
  model.row_lower[static_cast<std::size_t>(last)] = reach + 1.0;
  model.row_upper[static_cast<std::size_t>(last)] = kInfinity;
  ASSERT_EQ(monolithic(model).status, SolveStatus::kInfeasible);
  const Solution s = solve(model, dantzig_wolfe(2));
  EXPECT_EQ(s.status, SolveStatus::kInfeasible) << s.message;
  EXPECT_NE(s.algorithm, "dantzig-wolfe");
}

TEST(DantzigWolfe, ABlockUnboundedForItsOwnCostsIsStillSolvedRight) {
  // Block 0's column x_0 is free above at cost -1 and its own row does not cap it; only the
  // linking row x_0 + x_2 <= 10 does. Pricing at the block's own costs is unbounded, so the
  // method needs a ray - or declines - and either way the answer must be the monolithic one.
  Model model;
  model.resize_columns(4);
  model.resize_rows(3);
  model.matrix.reset(3, 4);
  for (Index j = 0; j < 4; ++j) {
    model.col_lower[static_cast<std::size_t>(j)] = 0.0;
    model.col_upper[static_cast<std::size_t>(j)] = 4.0;
    model.col_cost[static_cast<std::size_t>(j)] = 0.5;
  }
  model.col_upper[0] = kInfinity;
  model.col_cost[0] = -1.0;
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.row_lower[0] = -2.0;  // x_0 - x_1 >= -2: no cap on x_0
  model.matrix.add_entry(1, 2, 1.0);
  model.matrix.add_entry(1, 3, 1.0);
  model.row_lower[1] = 1.0;
  model.matrix.add_entry(2, 0, 1.0);
  model.matrix.add_entry(2, 2, 1.0);
  model.row_upper[2] = 10.0;
  model.matrix.finalize();
  const Solution reference = monolithic(model);
  ASSERT_EQ(reference.status, SolveStatus::kOptimal) << reference.message;
  expect_agrees(solve(model, dantzig_wolfe(2)), reference, "unbounded block");
}

TEST(DantzigWolfe, TheThreadCountDoesNotChangeABit) {
  std::mt19937 rng(77);
  for (int trial = 0; trial < 6; ++trial) {
    const Model model = linked_blocks(rng, 4, 5, 4, 2);
    Options one = dantzig_wolfe(4);
    one.set_int("decomposition_threads", 1);
    Options many = dantzig_wolfe(4);
    many.set_int("decomposition_threads", 4);
    const Solution a = solve(model, one);
    const Solution b = solve(model, many);
    ASSERT_EQ(a.status, SolveStatus::kOptimal) << a.message;
    ASSERT_EQ(b.status, SolveStatus::kOptimal) << b.message;
    EXPECT_EQ(a.algorithm, b.algorithm);
    EXPECT_EQ(a.objective, b.objective) << "trial " << trial;
    EXPECT_EQ(a.col_value, b.col_value) << "trial " << trial;
    EXPECT_EQ(a.row_dual, b.row_dual) << "trial " << trial;
  }
}

}  // namespace
}  // namespace sankhya
