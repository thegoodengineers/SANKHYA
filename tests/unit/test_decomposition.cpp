// SPDX-License-Identifier: Apache-2.0
// SANKHYA - block-angular detection and Benders decomposition of an LP (#525).
//
// Detection is tested for what it must always do - never hand over a structure whose blocks
// share a column - and for what it should find: a staircase splits at the one stock that links
// its halves, a dense LP is not called structured, blocks that share only rows are found as
// such. Benders is tested against the ordinary solve of the same model, which shares none of
// its code path: random block-angular LPs (every column bounded, so every one is feasible and
// bounded by construction), a model that forces feasibility cuts, models that must fall back
// to the monolithic engines and say so, and thread counts that must not change a bit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "decomp/structure.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

using decomp::BlockStructure;
using decomp::Detection;
using decomp::DetectionOptions;
using decomp::Linking;

Options quiet(const char* mode) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("decomposition", mode);
  return o;
}

/// The invariant detection must never break: a column a block owns is in that block's rows
/// only, a row a block owns holds no other block's column, and the counts are the counts.
void expect_structurally_valid(const Model& model, const BlockStructure& s) {
  ASSERT_EQ(s.row_block.size(), static_cast<std::size_t>(model.num_rows()));
  ASSERT_EQ(s.col_block.size(), static_cast<std::size_t>(model.num_cols()));
  std::vector<Index> rows_owned(static_cast<std::size_t>(s.blocks), 0);
  std::vector<Index> cols_owned(static_cast<std::size_t>(s.blocks), 0);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const Index b = s.row_block[static_cast<std::size_t>(i)];
    ASSERT_LT(b, s.blocks);
    if (b >= 0) ++rows_owned[static_cast<std::size_t>(b)];
  }
  for (Index j = 0; j < model.num_cols(); ++j) {
    const Index owner = s.col_block[static_cast<std::size_t>(j)];
    ASSERT_LT(owner, s.blocks);
    if (owner >= 0) ++cols_owned[static_cast<std::size_t>(owner)];
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] == 0.0) continue;
      const Index b = s.row_block[static_cast<std::size_t>(column.rows[k])];
      if (s.kind == Linking::kColumns && owner >= 0) {
        ASSERT_EQ(b, owner) << "column " << j << " owned by block " << owner << " is in row "
                            << column.rows[k] << " of block " << b;
      }
    }
  }
  if (s.kind == Linking::kRows) {
    // Mirror image: a row a block owns holds columns of that block only.
    for (Index j = 0; j < model.num_cols(); ++j) {
      const Index owner = s.col_block[static_cast<std::size_t>(j)];
      const ColumnView column = model.matrix.column(j);
      for (Index k = 0; k < column.size; ++k) {
        if (column.values[k] == 0.0) continue;
        const Index b = s.row_block[static_cast<std::size_t>(column.rows[k])];
        if (b >= 0) {
          ASSERT_EQ(owner, b);
        }
      }
    }
  }
  EXPECT_EQ(rows_owned, s.block_rows);
  EXPECT_EQ(cols_owned, s.block_columns);
}

/// T periods, each with a stock s_t, a purchase b_t and a run r_t:
///   s_t - s_{t-1} - b_t + r_t = 0,  r_t <= 5,  b_t + r_t >= 1.
/// The stock is the only thing that joins one period to the next.
Model staircase(int periods) {
  const Index n = 3 * periods;
  const Index m = 3 * periods;
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  model.matrix.reset(m, n);
  for (int t = 0; t < periods; ++t) {
    const Index s = 3 * t, b = 3 * t + 1, r = 3 * t + 2;
    const Index balance = 3 * t, cap = 3 * t + 1, need = 3 * t + 2;
    for (const Index j : {s, b, r}) {
      model.col_lower[static_cast<std::size_t>(j)] = 0.0;
      model.col_upper[static_cast<std::size_t>(j)] = 10.0;
    }
    model.col_cost[static_cast<std::size_t>(s)] = 0.1;
    model.col_cost[static_cast<std::size_t>(b)] = 1.0 + 0.05 * t;
    model.col_cost[static_cast<std::size_t>(r)] = -0.5;
    model.matrix.add_entry(balance, s, 1.0);
    if (t > 0) model.matrix.add_entry(balance, 3 * (t - 1), -1.0);
    model.matrix.add_entry(balance, b, -1.0);
    model.matrix.add_entry(balance, r, 1.0);
    model.row_lower[static_cast<std::size_t>(balance)] = 0.0;
    model.row_upper[static_cast<std::size_t>(balance)] = 0.0;
    model.matrix.add_entry(cap, r, 1.0);
    model.row_upper[static_cast<std::size_t>(cap)] = 5.0;
    model.matrix.add_entry(need, b, 1.0);
    model.matrix.add_entry(need, r, 1.0);
    model.row_lower[static_cast<std::size_t>(need)] = 1.0;
  }
  model.matrix.finalize();
  return model;
}

/// `blocks` groups of local rows and columns, `linking` columns that appear in local rows of
/// several blocks, and a few rows on the linking columns alone. Every column is bounded, and
/// every row is built around one point of the box, so the LP is feasible and bounded.
Model block_angular(std::mt19937& rng, int blocks, int rows_per_block, int cols_per_block,
                    int linking, int master_rows) {
  const int local_cols = blocks * cols_per_block;
  const Index n = local_cols + linking;
  const Index m = blocks * rows_per_block + master_rows;
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
  const auto finish_row = [&](double activity) {
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
  const auto entry = [&](Index col, double* activity) {
    double a = coefficient(rng);
    if (std::fabs(a) < 0.2) a = 1.0;
    model.matrix.add_entry(row, col, a);
    *activity += a * point[static_cast<std::size_t>(col)];
  };
  for (int b = 0; b < blocks; ++b) {
    for (int r = 0; r < rows_per_block; ++r) {
      double activity = 0.0;
      std::vector<int> used;
      const int entries = 2 + static_cast<int>(unit(rng) * 2.0);
      for (int e = 0; e < entries; ++e) {
        const int c = b * cols_per_block + static_cast<int>(unit(rng) * cols_per_block);
        if (std::find(used.begin(), used.end(), c) != used.end()) continue;
        used.push_back(c);
        entry(c, &activity);
      }
      if (linking > 0) {
        std::vector<int> picked;
        const int touches = static_cast<int>(unit(rng) * 2.0);
        for (int e = 0; e < touches; ++e) {
          const int c = local_cols + static_cast<int>(unit(rng) * linking);
          if (std::find(picked.begin(), picked.end(), c) != picked.end()) continue;
          picked.push_back(c);
          entry(c, &activity);
        }
      }
      finish_row(activity);
    }
  }
  for (int r = 0; r < master_rows; ++r) {
    double activity = 0.0;
    std::vector<int> used;
    for (int e = 0; e < 2 && linking > 0; ++e) {
      const int c = local_cols + static_cast<int>(unit(rng) * linking);
      if (std::find(used.begin(), used.end(), c) != used.end()) continue;
      used.push_back(c);
      entry(c, &activity);
    }
    finish_row(activity);
  }
  model.matrix.finalize();
  return model;
}

// =========================================================================================
// Detection
// =========================================================================================

TEST(DecompositionDetection, AStaircaseSplitsAtTheStockThatLinksItsHalves) {
  const Model model = staircase(8);
  DetectionOptions options;
  options.blocks = 2;
  const Detection d = decomp::detect_block_structure(model, Linking::kColumns, options);
  ASSERT_TRUE(d.accepted) << d.description;
  EXPECT_EQ(d.structure.blocks, 2);
  // Only stocks can join periods, and a contiguous split crosses one boundary.
  EXPECT_LE(d.structure.linking_columns, 3) << d.description;
  EXPECT_LE(d.structure.linking_fraction, 0.15) << d.description;
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (d.structure.col_block[static_cast<std::size_t>(j)] == decomp::kLinking) {
      EXPECT_EQ(j % 3, 0) << "column " << j << " is linking and is not a stock";
    }
  }
  expect_structurally_valid(model, d.structure);
}

TEST(DecompositionDetection, ADenseLpHasNoStrongStructure) {
  std::mt19937 rng(525);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  const Index n = 24;
  const Index m = 24;
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) model.matrix.add_entry(i, j, value(rng) + 3.0);
    model.row_upper[static_cast<std::size_t>(i)] = 10.0;
  }
  model.matrix.finalize();
  const Detection d = decomp::detect_block_structure(model, Linking::kColumns, {});
  EXPECT_FALSE(d.accepted) << d.description;
}

TEST(DecompositionDetection, BlocksThatShareOnlyRowsAreFoundAsLinkingRows) {
  // Three groups of columns, each with its own rows, and two rows that all three share.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> value(0.5, 2.0);
  const int groups = 3, cols = 6, local_rows = 4, shared = 2;
  const Index n = groups * cols;
  const Index m = groups * local_rows + shared;
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  model.matrix.reset(m, n);
  for (int g = 0; g < groups; ++g) {
    for (int r = 0; r < local_rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if ((r + c) % 2 == 0)
          model.matrix.add_entry(g * local_rows + r, g * cols + c, value(rng));
      }
      model.row_upper[static_cast<std::size_t>(g * local_rows + r)] = 5.0;
    }
  }
  for (int s = 0; s < shared; ++s) {
    for (int g = 0; g < groups; ++g) {
      model.matrix.add_entry(groups * local_rows + s, g * cols + s, value(rng));
    }
    model.row_upper[static_cast<std::size_t>(groups * local_rows + s)] = 9.0;
  }
  model.matrix.finalize();
  DetectionOptions options;
  options.blocks = 3;
  options.max_linking_fraction = 0.25;
  const Detection rows = decomp::detect_block_structure(model, Linking::kRows, options);
  ASSERT_TRUE(rows.accepted) << rows.description;
  EXPECT_EQ(rows.structure.blocks, 3);
  EXPECT_LE(rows.structure.linking_rows, shared + 1) << rows.description;
  expect_structurally_valid(model, rows.structure);
}

TEST(DecompositionDetection, EveryStructureItReturnsIsValidAndTheSameOnARerun) {
  std::mt19937 rng(2525);
  int found = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = block_angular(rng, 3 + trial % 3, 6, 5, 2 + trial % 4, trial % 3);
    DetectionOptions options;
    options.blocks = 2 + trial % 3;
    options.max_linking_fraction = 1.0;
    options.max_largest_share = 1.0;
    for (const Linking kind : {Linking::kColumns, Linking::kRows}) {
      const Detection a = decomp::detect_block_structure(model, kind, options);
      const Detection b = decomp::detect_block_structure(model, kind, options);
      EXPECT_EQ(a.accepted, b.accepted);
      EXPECT_EQ(a.structure.row_block, b.structure.row_block) << "trial " << trial;
      EXPECT_EQ(a.structure.col_block, b.structure.col_block) << "trial " << trial;
      if (!a.accepted) continue;
      ++found;
      expect_structurally_valid(model, a.structure);
    }
  }
  EXPECT_GT(found, 40);
}

// =========================================================================================
// Benders, against the ordinary solve
// =========================================================================================

Options forced(int blocks) {
  Options o = quiet("benders");
  o.set_int("decomposition_blocks", blocks);
  return o;
}

TEST(Decomposition, TheDefaultIsOffAndSolvesWithTheOrdinaryEngines) {
  EXPECT_EQ(Options().get_string("decomposition"), "off");
  const Model model = staircase(8);
  Options o;
  o.set_bool("log_to_console", false);
  const Solution s = solve(model, o);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.algorithm, "benders");
}

TEST(Decomposition, AgreesWithTheMonolithicSolveOnRandomBlockAngularLps) {
  std::mt19937 rng(525525);
  int decomposed = 0;
  int trials = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const int blocks = 2 + trial % 3;
    const Model model = block_angular(rng, blocks, 5 + trial % 3, 4, 2 + trial % 4, trial % 3);
    Options off = quiet("off");
    const Solution reference = solve(model, off);
    ASSERT_EQ(reference.status, SolveStatus::kOptimal) << "trial " << trial;
    const Solution s = solve(model, forced(blocks));
    ++trials;
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << "trial " << trial << ": " << s.message;
    EXPECT_NEAR(s.objective, reference.objective,
                1e-6 * std::max(1.0, std::fabs(reference.objective)))
        << "trial " << trial << " (" << s.algorithm << ")";
    EXPECT_LE(s.primal_infeasibility_scaled, 1e-7) << "trial " << trial;
    EXPECT_LE(s.dual_infeasibility_scaled, 1e-7) << "trial " << trial;
    if (s.algorithm == "benders") {
      ++decomposed;
      EXPECT_NE(s.message.find("Benders"), std::string::npos) << s.message;
      EXPECT_LE(s.relative_gap, 1e-7) << "trial " << trial;
    }
  }
  // The test would pass vacuously if every trial quietly fell back to the monolithic engines.
  EXPECT_GT(decomposed, trials * 7 / 10)
      << decomposed << " of " << trials << " were decomposed";
}

TEST(Decomposition, AStaircaseIsDecomposedAndAgrees) {
  const Model model = staircase(12);
  const Solution reference = solve(model, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kOptimal);
  const Solution s = solve(model, forced(3));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm, "benders");
  EXPECT_NEAR(s.objective, reference.objective,
              1e-6 * std::max(1.0, std::fabs(reference.objective)));
  // A run that converged in one round used no cut but the completing ones, and so says nothing
  // about whether the cuts are right.
  EXPECT_GE(s.iterations, 2) << s.message;
}

TEST(Decomposition, ABlockInfeasibleAtTheFirstMasterPointNeedsFeasibilityCuts) {
  // Two blocks of three rows each, joined by one linking column y with cost 1: the master wants
  // y = 0, but each block needs x_a + x_b + y >= 5 with x_a, x_b <= 1, so y >= 3 is forced, and
  // only a feasibility cut can say so.
  Model model;
  const Index n = 5;  // x0 x1 | x2 x3 | y
  model.resize_columns(n);
  model.resize_rows(4);
  model.matrix.reset(4, n);
  for (Index j = 0; j < 4; ++j) {
    model.col_lower[static_cast<std::size_t>(j)] = 0.0;
    model.col_upper[static_cast<std::size_t>(j)] = 1.0;
    model.col_cost[static_cast<std::size_t>(j)] = 0.5;
  }
  model.col_lower[4] = 0.0;
  model.col_upper[4] = 10.0;
  model.col_cost[4] = 1.0;
  const Index first[2] = {0, 2};
  for (Index b = 0; b < 2; ++b) {
    model.matrix.add_entry(2 * b, first[b], 1.0);
    model.matrix.add_entry(2 * b, first[b] + 1, 1.0);
    model.matrix.add_entry(2 * b, 4, 1.0);
    model.row_lower[static_cast<std::size_t>(2 * b)] = 5.0;
    model.matrix.add_entry(2 * b + 1, first[b], 1.0);
    model.matrix.add_entry(2 * b + 1, first[b] + 1, -1.0);
    model.row_lower[static_cast<std::size_t>(2 * b + 1)] = -1.0;
    model.row_upper[static_cast<std::size_t>(2 * b + 1)] = 1.0;
  }
  model.matrix.finalize();
  const Solution reference = solve(model, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kOptimal);
  const Solution s = solve(model, forced(2));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm, "benders");
  EXPECT_NEAR(s.objective, reference.objective, 1e-7);
  EXPECT_NEAR(s.col_value[4], 3.0, 1e-6);
}

TEST(Decomposition, AnInfeasibleLpIsReportedInfeasibleByTheOrdinaryEngines) {
  Model model = staircase(6);
  // Period 0 must buy and run at least 100 in total, and each of the two is capped at 10.
  model.row_lower[2] = 100.0;
  const Solution reference = solve(model, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kInfeasible) << reference.message;
  const Solution s = solve(model, forced(2));
  EXPECT_EQ(s.status, SolveStatus::kInfeasible) << s.message;
  EXPECT_NE(s.algorithm, "benders");
}

TEST(Decomposition, AnUnboundedLpIsLeftToTheOrdinaryEngines) {
  Model model = staircase(6);
  // Period 0's purchase and run both free above, with the run paying more than the purchase
  // costs: buying and running one more barrel each lowers the cost by 0.4, without end.
  model.col_upper[1] = kInfinity;
  model.col_upper[2] = kInfinity;
  model.col_cost[1] = 0.1;
  model.row_upper[1] = kInfinity;
  const Solution reference = solve(model, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kUnbounded) << reference.message;
  const Solution s = solve(model, forced(2));
  EXPECT_EQ(s.status, SolveStatus::kUnbounded) << s.message;
  EXPECT_NE(s.algorithm, "benders");
}

TEST(Decomposition, AColumnTheBalanceRowCapsIsBoundedAndTheAnswerAgrees) {
  // The run is free above and its cap row is gone, but period 0's balance row ties it to a
  // purchase that is capped, so the LP is bounded: the case an earlier version of the unbounded
  // test built by mistake. Benders must find the same optimum.
  Model model = staircase(6);
  model.col_upper[2] = kInfinity;
  model.row_upper[1] = kInfinity;
  const Solution reference = solve(model, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kOptimal) << reference.message;
  const Solution s = solve(model, forced(2));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, reference.objective,
              1e-6 * std::max(1.0, std::fabs(reference.objective)));
}

TEST(Decomposition, ModelsItDoesNotTakeAreSolvedAsIfTheOptionWereOff) {
  std::mt19937 rng(11);
  const Model base = block_angular(rng, 3, 6, 4, 3, 1);
  const Solution reference = solve(base, quiet("off"));
  ASSERT_EQ(reference.status, SolveStatus::kOptimal);

  Model maximise = base;
  maximise.sense = ObjSense::kMaximize;
  const Solution top = solve(maximise, forced(3));
  EXPECT_NE(top.algorithm, "benders") << "a maximisation is not decomposed";
  EXPECT_EQ(top.status, solve(maximise, quiet("off")).status);

  Model integer = base;
  integer.col_type[0] = VarType::kInteger;
  const Solution mixed = solve(integer, forced(3));
  EXPECT_NE(mixed.algorithm, "benders") << "an integer model is not decomposed";
  EXPECT_EQ(mixed.status, solve(integer, quiet("off")).status);
}

TEST(Decomposition, TheThreadCountDoesNotChangeABit) {
  std::mt19937 rng(99);
  for (int trial = 0; trial < 8; ++trial) {
    const Model model = block_angular(rng, 4, 6, 4, 3, 1);
    Options one = forced(4);
    one.set_int("decomposition_threads", 1);
    Options many = forced(4);
    many.set_int("decomposition_threads", 4);
    const Solution a = solve(model, one);
    const Solution b = solve(model, many);
    ASSERT_EQ(a.status, SolveStatus::kOptimal) << a.message;
    ASSERT_EQ(b.status, SolveStatus::kOptimal) << b.message;
    EXPECT_EQ(a.algorithm, b.algorithm);
    EXPECT_EQ(a.objective, b.objective) << "trial " << trial;
    EXPECT_EQ(a.col_value, b.col_value) << "trial " << trial;
    EXPECT_EQ(a.row_dual, b.row_dual) << "trial " << trial;
    EXPECT_EQ(a.iterations, b.iterations) << "trial " << trial;
  }
}

}  // namespace
}  // namespace sankhya
