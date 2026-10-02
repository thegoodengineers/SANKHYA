// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut pool's row removal (#497), watched through its test seam.
//
// With mip_cut_pooling an aged cut row is deleted from the node LP and kept in the pool, and
// appended again when a node's LP point violates it. The fuzz sweeps (test_branch_and_bound,
// test_debug_solution) check the answers; this file checks the mechanism itself: that a cut
// only comes back when the point violates it, that it comes back as the row it was, that a
// cut is only removed while it is in the LP and only re-added while it is out, and that the
// counters the stats JSON reports are the events that happened.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "mip/cut_pool_audit.hpp"

namespace sankhya::mip {
namespace {

/// Records every pool event of one solve and clears the hook when it goes out of scope.
class Recorder {
 public:
  Recorder() {
    cut_pool_audit_for_testing() = [this](const CutPoolEvent& event) {
      events.push_back(event);
    };
  }
  ~Recorder() { cut_pool_audit_for_testing() = nullptr; }
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  std::vector<CutPoolEvent> events;
};

/// A pure-binary instance of 10 to 14 columns: knapsack rows sum a x <= b, which covers and
/// MIR cuts are separated from, and a covering row, maximising profit.
Model knapsack_instance(std::mt19937_64& rng) {
  std::uniform_int_distribution<Index> width(10, 14);
  std::uniform_int_distribution<Index> height(2, 4);
  std::uniform_int_distribution<int> weight(1, 9);
  std::uniform_int_distribution<int> profit(1, 40);
  std::uniform_int_distribution<int> percent(0, 99);
  const Index n = width(rng);
  const Index knapsacks = height(rng);
  Model model;
  model.resize_columns(n);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_type[u] = VarType::kInteger;
    model.col_upper[u] = 1.0;
    model.col_cost[u] = -static_cast<double>(profit(rng));
  }
  model.resize_rows(knapsacks + 1);
  model.matrix.reset(knapsacks + 1, n);
  for (Index i = 0; i < knapsacks; ++i) {
    int sum = 0;
    for (Index j = 0; j < n; ++j) {
      if (percent(rng) >= 70) continue;
      const int a = weight(rng);
      sum += a;
      model.matrix.add_entry(i, j, a);
    }
    model.row_upper[static_cast<std::size_t>(i)] = std::max(1, sum / 2);
  }
  for (Index j = 0; j < n; ++j) {
    if (percent(rng) < 30) model.matrix.add_entry(knapsacks, j, 1.0);
  }
  model.row_lower[static_cast<std::size_t>(knapsacks)] = 1.0;
  model.matrix.finalize();
  return model;
}

/// Few cuts a round, an age limit of 1 and no heuristics: the root stays open, rows age out
/// at nearly every node, and the tree is deep enough for removed cuts to be violated again.
Options pool_options(bool pooling) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_bool("enable_root_cuts", true);
  options.set_bool("root_cut_loop", false);
  options.set_int("tree_cut_depth", 4);
  options.set_int("cut_max_per_round", 2);
  options.set_int("tree_cut_rows_per_round", 2);
  options.set_int("cut_support_floor", 100);
  options.set_bool("mip_heuristics", false);
  options.set_int("mip_threads", 1);
  options.set_bool("mip_cut_pooling", pooling);
  options.set_int("mip_cut_age_limit", 1);
  return options;
}

TEST(CutPool, ARemovedCutIsAppendedAgainOnlyWhenViolatedAndAsTheSameRow) {
  std::mt19937_64 rng(20261002);
  Options reference = pool_options(false);
  reference.set_bool("enable_root_cuts", false);
  std::int64_t removed = 0;
  std::int64_t readded = 0;
  int instances = 0;
  for (int attempt = 0; attempt < 200 && readded < 20; ++attempt) {
    const Model model = knapsack_instance(rng);
    const Solution plain = solve(model, reference);
    if (plain.status != SolveStatus::kOptimal) continue;
    Recorder recorder;
    const Solution pooled = solve(model, pool_options(true));
    ASSERT_EQ(pooled.status, SolveStatus::kOptimal) << "attempt " << attempt;
    EXPECT_NEAR(pooled.objective, plain.objective, 1e-6) << "attempt " << attempt;
    ++instances;

    // Which pooled cuts are out of the LP, event by event: a cut is removed only while it is
    // in, and appended again only while it is out.
    std::set<std::size_t> out;
    std::int64_t removed_here = 0;
    std::int64_t readded_here = 0;
    for (const CutPoolEvent& event : recorder.events) {
      if (event.kind == CutPoolEvent::Kind::kRemoved) {
        EXPECT_TRUE(out.insert(event.pool_index).second)
            << "cut " << event.pool_index << " removed twice, attempt " << attempt;
        // A row is removed when its logical has been basic: the point does not violate it.
        EXPECT_LE(event.activity - event.rhs, 1e-6 * (1.0 + std::fabs(event.rhs)))
            << "attempt " << attempt;
        EXPECT_EQ(event.row, -1);
        ++removed_here;
      } else {
        EXPECT_EQ(out.erase(event.pool_index), 1u)
            << "cut " << event.pool_index << " appended while in the LP, attempt " << attempt;
        EXPECT_GT(event.activity - event.rhs, tol::kCutViolationTolerance)
            << "cut " << event.pool_index << " appended again without being violated";
        EXPECT_TRUE(event.row_is_the_cut)
            << "cut " << event.pool_index << " came back as a different row, attempt "
            << attempt;
        EXPECT_GE(event.row, 0);
        EXPECT_LT(event.row, event.lp_rows);
        ++readded_here;
      }
    }
    // The stats JSON's counters are these events, no more and no fewer.
    EXPECT_EQ(pooled.cut_rows_removed, removed_here);
    EXPECT_EQ(pooled.cut_rows_readded, readded_here);
    EXPECT_EQ(pooled.cut_rows_aged_out, removed_here);
    EXPECT_EQ(pooled.cuts_reactivated, readded_here);
    EXPECT_GT(pooled.node_lp_rows_max, 0);
    EXPECT_TRUE(std::isfinite(pooled.node_lp_rows_mean));
    removed += removed_here;
    readded += readded_here;
    if (HasFailure()) return;
  }
  EXPECT_GT(instances, 10);
  EXPECT_GT(removed, 0) << "no cut row was ever removed";
  EXPECT_GT(readded, 0) << "no removed cut was ever appended again";
  std::printf("[  INFO    ] cut pool: %d instances, %lld rows removed, %lld appended again\n",
              instances, static_cast<long long>(removed), static_cast<long long>(readded));
}

TEST(CutPool, WithTheOptionOffNothingIsRemoved) {
  std::mt19937_64 rng(20261002);
  std::int64_t aged = 0;
  for (int attempt = 0; attempt < 40; ++attempt) {
    const Model model = knapsack_instance(rng);
    Recorder recorder;
    const Solution s = solve(model, pool_options(false));
    EXPECT_TRUE(recorder.events.empty()) << "attempt " << attempt;
    EXPECT_EQ(s.cut_rows_removed, 0);
    EXPECT_EQ(s.cut_rows_readded, 0);
    aged += s.cut_rows_aged_out;
  }
  EXPECT_GT(aged, 0) << "no row aged out, so the check above saw nothing";
}

}  // namespace
}  // namespace sankhya::mip
