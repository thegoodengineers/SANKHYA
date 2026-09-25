// SPDX-License-Identifier: Apache-2.0
// Non-CUDA stub for pdhg_batch (#520).
// Provides the same symbol as pdhg_batch.cu so non-CUDA builds link correctly.
// The CUDA implementation in pdhg_batch.cu shadows this file when CUDA is enabled.
#ifndef SANKHYA_ENABLE_CUDA

#include "gpu/pdhg_batch.hpp"

namespace sankhya::gpu {

std::vector<BatchNodeResult> solve_batch_nodes(const Model& /*model*/, int /*K*/,
                                               const std::vector<double>& /*col_lb*/,
                                               const std::vector<double>& /*col_ub*/,
                                               double /*incumbent*/,
                                               const Options& /*options*/) {
  return {};
}

}  // namespace sankhya::gpu

#endif  // !SANKHYA_ENABLE_CUDA
