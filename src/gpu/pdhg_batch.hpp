// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Batched GPU PDHG for branch-and-bound node bounding and strong branching (#520).
//
// References (written from the papers; per ENGINEERING_RULES.md no solver source was
// consulted):
//   [BPDHG] Applegate et al., "Practical Large-Scale LP via PDHG", NeurIPS 2021,
//           extended to batch: n x K and m x K iterates, one SpMM per product.
//   [SBB]   Berthold et al. / Huang et al., "Batched First-Order Strong Branching on the
//           GPU", arXiv:2601.21990. The scoring protocol this implementation targets.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include <vector>

namespace sankhya::gpu {

/// Result for one LP in the batch.
struct BatchNodeResult {
  double dual_bound;  ///< safe dual bound from early-stopped PDHG (#519 guard)
  int iterations;
  bool pruned;  ///< dual_bound > incumbent
};

/// Solve K node LPs in one batched PDHG pass.
///
/// Each LP shares the constraint matrix A; only column bounds differ (stored in
/// `col_lb` and `col_ub`, each of size ncols * K, column-major).  The dual bound
/// per node is valid for pruning when the safe-bound guard (#519) holds.
///
/// Returns an empty vector when `options.get_bool("gpu_batch_nodes")` is false
/// or no CUDA device is present; the caller falls back to sequential node LP solves.
[[nodiscard]] std::vector<BatchNodeResult> solve_batch_nodes(
    const Model& model, int K,
    const std::vector<double>& col_lb,  // ncols * K, column-major
    const std::vector<double>& col_ub,  // ncols * K, column-major
    double incumbent, const Options& options);

}  // namespace sankhya::gpu
