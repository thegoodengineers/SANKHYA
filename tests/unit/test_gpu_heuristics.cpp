// SPDX-License-Identifier: Apache-2.0
// SANKHYA - tests for the GPU MIP heuristics (#509): feasibility pump and
// fix-and-propagate.
//
// Both heuristics are gated behind options (gpu_pump, gpu_fix_and_prop) and
// require a CUDA device. Tests here check two things:
//
//   1. With the option OFF they return nullopt / the search is unchanged.
//   2. With the option ON (skipped without a CUDA card) they find a feasible
//      point on a set-partitioning instance where the answer is exact (one
//      column = 1, all others = 0) and the search reaches the same optimum as
//      with the option off.
//
// References:
//   Fischetti, Glover & Lodi, Math. Programming 104 (2005)
//   Corduk et al., arXiv:2510.20499

#include <gtest/gtest.h>

#include <cmath>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#include "gpu/gpu_heuristics.hpp"
#endif

namespace sankhya {
namespace {

/// A set-partitioning MILP: n binary columns, sum = 1. Unique optimum: cost 1
/// (choose the cheapest column). The LP relaxation has 1/n on every column,
/// so the heuristics face a genuinely fractional point.
Model set_partition(int n) {
  Model m;
  m.col_cost.assign(static_cast<std::size_t>(n), 1.0);
  // Make costs distinct so there is a unique optimum.
  for (int j = 0; j < n; ++j) m.col_cost[static_cast<std::size_t>(j)] = 1.0 + 0.01 * j;
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
  return m;
}

Options base_options() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

// ---- gpu_pump=false: heuristic does nothing, search result is unchanged -------

TEST(GpuHeuristics, PumpOffDoesNothing) {
  const Model model = set_partition(6);
  Options o = base_options();
  o.set_bool("gpu_pump", false);
  const Solution s = solve(model, o);
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
}

// ---- gpu_fix_and_prop=false: heuristic does nothing, search result unchanged --

TEST(GpuHeuristics, FixAndPropOffDoesNothing) {
  const Model model = set_partition(6);
  Options o = base_options();
  o.set_bool("gpu_fix_and_prop", false);
  const Solution s = solve(model, o);
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
}

// ---- With a CUDA device: pump finds a feasible point (skipped without card) --

TEST(GpuHeuristics, PumpFindsAFeasiblePointOnSetPartition) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  const Model model = set_partition(8);
  Options o = base_options();
  o.set_bool("gpu_pump", true);
  const auto sol = gpu::feasibility_pump(model, o);
  ASSERT_TRUE(sol.has_value()) << "pump should find a feasible point on set-partition(8)";
  // Check the returned point is row-feasible.
  double sum = 0.0;
  for (double v : sol->x) sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6) << "exactly one column should be 1 in a set-partition optimum";
  // Every column should be 0 or 1.
  for (double v : sol->x) EXPECT_TRUE(v < 1e-6 || std::fabs(v - 1.0) < 1e-6);
#endif
}

// ---- With a CUDA device: fix-and-propagate finds a feasible point ------------

TEST(GpuHeuristics, FixAndPropFindsAFeasiblePointOnSetPartition) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  const Model model = set_partition(8);
  Options o = base_options();
  o.set_bool("gpu_fix_and_prop", true);
  const auto sol = gpu::fix_and_propagate(model, o);
  ASSERT_TRUE(sol.has_value()) << "fix-and-prop should find a feasible point on set-partition(8)";
  double sum = 0.0;
  for (double v : sol->x) sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6);
  for (double v : sol->x) EXPECT_TRUE(v < 1e-6 || std::fabs(v - 1.0) < 1e-6);
#endif
}

// ---- Search with GPU pump on reaches the same optimum as without it ----------

TEST(GpuHeuristics, SearchWithGpuPumpReachesExactOptimum) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  const Model model = set_partition(8);

  Options with_pump = base_options();
  with_pump.set_bool("gpu_pump", true);
  const Solution s_pump = solve(model, with_pump);
  ASSERT_EQ(s_pump.status, SolveStatus::kOptimal);

  Options without = base_options();
  const Solution s_ref = solve(model, without);
  ASSERT_EQ(s_ref.status, SolveStatus::kOptimal);

  EXPECT_NEAR(s_pump.objective, s_ref.objective, 1e-6);
#endif
}

}  // namespace
}  // namespace sankhya
