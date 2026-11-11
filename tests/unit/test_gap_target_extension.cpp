// SPDX-License-Identifier: Apache-2.0
// SANKHYA - searching on past the gap target (#1158, mip_gap_target_extension).
//
// Meeting the relative gap target proves the incumbent is within it, not that it is the
// optimum; MIPLIB 3 dcmulti stopped there on every seed 3.2e-5 above its published value.
// The option searches on for a share of the nodes already explored. It may only ever make
// the answer better: a closed tree must report the enumerated optimum, and a limit reached
// while it searches must report the target's verdict (optimal within the gap), never a lost
// one. Every model here is small enough to enumerate.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr int kCols = 14;
constexpr int kRows = 3;
constexpr double kLooseGap = 0.3;

/// A 0/1 multi-row knapsack, maximised; weights and profits from the seed, the weights
/// also kept dense in `weights` for the enumeration.
Model knapsack(std::uint32_t seed, std::vector<std::vector<double>>* weights) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> weight(1, 20);
  std::uniform_int_distribution<int> profit(1, 30);
  Model m;
  m.sense = ObjSense::kMaximize;
  m.col_cost.resize(kCols);
  for (auto& c : m.col_cost) c = profit(rng);
  m.col_lower.assign(kCols, 0.0);
  m.col_upper.assign(kCols, 1.0);
  m.col_type.assign(kCols, VarType::kInteger);
  m.row_lower.assign(kRows, -std::numeric_limits<double>::infinity());
  m.row_upper.assign(kRows, 0.0);
  m.matrix.reset(kRows, kCols);
  weights->assign(kRows, std::vector<double>(kCols, 0.0));
  for (Index i = 0; i < kRows; ++i) {
    double total = 0.0;
    for (Index j = 0; j < kCols; ++j) {
      const double w = weight(rng);
      total += w;
      (*weights)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = w;
      m.matrix.add_entry(i, j, w);
    }
    m.row_upper[static_cast<std::size_t>(i)] = std::floor(total / 3.0);
  }
  m.matrix.finalize();
  return m;
}

double enumerate(const Model& m, const std::vector<std::vector<double>>& weights) {
  double best = -std::numeric_limits<double>::infinity();
  for (std::uint32_t mask = 0; mask < (1U << kCols); ++mask) {
    std::vector<double> activity(kRows, 0.0);
    double value = 0.0;
    for (Index j = 0; j < kCols; ++j) {
      if ((mask >> j & 1U) == 0) continue;
      value += m.col_cost[static_cast<std::size_t>(j)];
      for (Index i = 0; i < kRows; ++i) {
        activity[static_cast<std::size_t>(i)] +=
            weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }
    }
    bool fits = true;
    for (Index i = 0; i < kRows; ++i) {
      fits = fits &&
             activity[static_cast<std::size_t>(i)] <= m.row_upper[static_cast<std::size_t>(i)];
    }
    if (fits && value > best) best = value;
  }
  return best;
}

Options loose_options() {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_bool("root_cut_loop", false);
  o.set_bool("enable_root_cuts", false);
  o.set_string("mip_heur_fj", "off");
  o.set_string("mip_heur_parity", "off");
  o.set_double("mip_relative_gap", kLooseGap);
  o.set_int("mip_threads", 1);  // the extension is the one search's; a parallel tree stops
  return o;
}

TEST(GapTargetExtension, AClosedTreeReportsTheEnumeratedOptimum) {
  int stopped_short = 0;  // models the plain gap stop left above the optimum
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    std::vector<std::vector<double>> weights;
    const Model m = knapsack(seed, &weights);
    const double optimum = enumerate(m, weights);

    const Solution plain = solve(m, loose_options());
    ASSERT_EQ(plain.status, SolveStatus::kOptimal) << "seed " << seed << ": " << plain.message;
    if (plain.objective < optimum - 1e-9) ++stopped_short;

    // Off (0) is the search without the option, node for node.
    Options off = loose_options();
    off.set_double("mip_gap_target_extension", 0.0);
    const Solution zero = solve(m, off);
    EXPECT_EQ(zero.nodes, plain.nodes) << "seed " << seed;
    EXPECT_EQ(zero.objective, plain.objective) << "seed " << seed;

    // A budget no tree this size can use: the search closes and proves the optimum.
    Options on = loose_options();
    on.set_double("mip_gap_target_extension", 1e6);
    const Solution closed = solve(m, on);
    ASSERT_EQ(closed.status, SolveStatus::kOptimal)
        << "seed " << seed << ": " << closed.message;
    EXPECT_NEAR(closed.objective, optimum, 1e-9) << "seed " << seed;
    EXPECT_EQ(closed.message.find("not exhausted"), std::string::npos)
        << "seed " << seed << ": " << closed.message;
    EXPECT_GE(closed.objective, plain.objective - 1e-9) << "seed " << seed;
  }
  // Load-bearing: the loose target must leave some model above its optimum, or the closed
  // runs above prove nothing the plain stop did not.
  EXPECT_GT(stopped_short, 0);
}

TEST(GapTargetExtension, ALimitWhileSearchingOnReportsTheTargetsVerdict) {
  int limited = 0;
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    std::vector<std::vector<double>> weights;
    const Model m = knapsack(seed, &weights);
    const Solution plain = solve(m, loose_options());
    if (plain.message.find("not exhausted") == std::string::npos) continue;  // closed anyway

    // The node limit lands one node past where the plain search met the target, so the
    // extension is cut short by it.
    Options on = loose_options();
    on.set_double("mip_gap_target_extension", 1e6);
    on.set_int("node_limit", plain.nodes + 1);
    const Solution cut = solve(m, on);
    if (cut.message.find("not exhausted") == std::string::npos) continue;  // closed in time
    ++limited;
    EXPECT_EQ(cut.status, SolveStatus::kOptimal) << "seed " << seed << ": " << cut.message;
    EXPECT_GE(cut.objective, plain.objective - 1e-9) << "seed " << seed;
    EXPECT_LE(cut.relative_gap, kLooseGap + 1e-9) << "seed " << seed;
    EXPECT_GE(cut.dual_bound, cut.objective - 1e-9) << "seed " << seed;  // maximise
  }
  EXPECT_GT(limited, 0);
}

}  // namespace
}  // namespace sankhya
