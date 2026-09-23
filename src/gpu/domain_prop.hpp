// SPDX-License-Identifier: Apache-2.0
// GPU activity-based bound propagation for MIP nodes (#510).
// Reference: Sofranac, Gleixner & Pokutta, arXiv:2009.07785, Algorithm 1.
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include <vector>

namespace sankhya::gpu {

struct PropResult {
  std::vector<double> col_lb;  // tightened lower bounds (same size as model.num_cols())
  std::vector<double> col_ub;  // tightened upper bounds
  bool infeasible;             // true if a bound cross was detected
};

/// Propagate variable bounds using row activities on the GPU.
/// Returns the input bounds unchanged (and infeasible=false) when
/// gpu_domain_prop=false or no CUDA device is available.
[[nodiscard]] PropResult propagate_bounds(const Model& model, const std::vector<double>& col_lb,
                                          const std::vector<double>& col_ub,
                                          const Options& options);

}  // namespace sankhya::gpu
