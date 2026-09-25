// SPDX-License-Identifier: Apache-2.0
// SANKHYA - tests for batched GPU PDHG node bounding (#520).
//
// Two obligations:
//   1. With gpu_batch_nodes=false (the default) the function returns an empty
//      result and the search is unchanged.
//   2. With gpu_batch_nodes=true and a CUDA device: on a batch of K identical
//      node LPs whose optimal objective exceeds the incumbent, all K nodes are
//      marked pruned. On a batch of K feasible node LPs, the safe dual bound
//      is finite for each.
//
// References:
//   Applegate et al., NeurIPS 2021; Huang et al., arXiv:2601.21990.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "gpu/pdhg_batch.hpp"  // always available; returns empty without CUDA

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#endif

namespace sankhya {
namespace {

/// Tiny LP: min x  s.t.  x >= 1, x in [0, 2].
/// Optimal value = 1.  K copies of this node LP.
Model tiny_lp_batch_model() {
  Model m;
  m.col_cost = {1.0};
  m.col_lower = {0.0};
  m.col_upper = {2.0};
  m.col_type = {VarType::kContinuous};
  m.matrix.reset(1, 1);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.finalize();
  m.row_lower = {1.0};
  m.row_upper = {kInfinity};
  m.hessian.reset(1, 1);
  m.hessian.finalize();
  return m;
}

Options batch_options(bool on) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_bool("gpu_batch_nodes", on);
  return o;
}

// ---- OFF by default: returns empty ------------------------------------

TEST(GpuBatch, ReturnEmptyWhenOptionOff) {
  const Model model = tiny_lp_batch_model();
  const int K = 4;
  const std::vector<double> lb(static_cast<std::size_t>(K), 0.0);
  const std::vector<double> ub(static_cast<std::size_t>(K), 2.0);
  const auto res = gpu::solve_batch_nodes(model, K, lb, ub, 100.0, batch_options(false));
  EXPECT_TRUE(res.empty());
}

// ---- With CUDA: batch of infeasible-objective nodes (obj > incumbent) pruned --

TEST(GpuBatch, PrunesNodesWhoseBoundExceedsIncumbent) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  // K copies of the tiny LP (optimal = 1). Incumbent = 0.5 (below optimal).
  // Every node has a dual bound >= 1 > 0.5: all should be pruned.
  const Model model = tiny_lp_batch_model();
  const int K = 4;
  std::vector<double> lb(static_cast<std::size_t>(K), 0.0);
  std::vector<double> ub(static_cast<std::size_t>(K), 2.0);
  const double incumbent = 0.5;
  Options o = batch_options(true);
  o.set_int("gpu_batch_max_iter", 1000);
  const auto res = gpu::solve_batch_nodes(model, K, lb, ub, incumbent, o);
  ASSERT_EQ(static_cast<int>(res.size()), K);
  for (int k = 0; k < K; ++k) {
    EXPECT_TRUE(res[static_cast<std::size_t>(k)].pruned)
        << "node " << k << " should be pruned (bound " << res[static_cast<std::size_t>(k)].dual_bound
        << " > incumbent " << incumbent << ")";
  }
#endif
}

// ---- With CUDA: search unchanged when batch_nodes is true -------------------

TEST(GpuBatch, SearchReachesCorrectOptimumWithBatchEnabled) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  // A simple set-partition MILP: unique optimum regardless of node-bounding strategy.
  Model m;
  const int n = 6;
  m.col_cost.resize(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) m.col_cost[static_cast<std::size_t>(j)] = 1.0 + 0.1 * j;
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  m.matrix.reset(1, static_cast<Index>(n));
  for (Index j = 0; j < static_cast<Index>(n); ++j) m.matrix.add_entry(0, j, 1.0);
  m.matrix.finalize();
  m.row_lower = {1.0};
  m.row_upper = {1.0};
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();

  Options without;
  without.set_bool("log_to_console", false);
  const Solution ref = solve(m, without);
  ASSERT_EQ(ref.status, SolveStatus::kOptimal);

  Options with_batch;
  with_batch.set_bool("log_to_console", false);
  with_batch.set_bool("gpu_batch_nodes", true);
  const Solution batched = solve(m, with_batch);
  ASSERT_EQ(batched.status, SolveStatus::kOptimal);
  EXPECT_NEAR(batched.objective, ref.objective, 1e-6);
#endif
}

}  // namespace
}  // namespace sankhya
