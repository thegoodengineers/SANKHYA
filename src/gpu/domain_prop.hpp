// SPDX-License-Identifier: Apache-2.0
// GPU activity-based bound propagation (#510).
// Reference: Sofranac, Gleixner & Pokutta, "Accelerated domain propagation for mixed-integer
// linear programs on GPUs", arXiv:2009.07785 (2020), Algorithm 1.
//
// The same synchronous rounds as the CPU reference mip::propagate_jacobi
// (src/mip/domain_propagation.hpp), computed one thread per row and one thread per column,
// with the same arithmetic in the same order: the two return identical bounds, which
// tests/unit/test_domain_propagation.cpp holds them to.
#pragma once

#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::gpu {

struct PropResult {
  std::vector<double> col_lb;  ///< tightened lower bounds (model.num_cols())
  std::vector<double> col_ub;  ///< tightened upper bounds
  bool infeasible = false;     ///< a row or a column proved the box empty
  bool ran = false;            ///< false when no device was available or a CUDA call failed
  int rounds = 0;
  long long tightened = 0;
};

/// Propagate [col_lb, col_ub] on the device for at most `rounds_limit` rounds. When no device
/// is available, or a CUDA call fails, returns the input bounds with ran = false; the caller
/// then uses the CPU reference.
[[nodiscard]] PropResult propagate_bounds(const Model& model, const std::vector<double>& col_lb,
                                          const std::vector<double>& col_ub, int rounds_limit,
                                          double integrality);

}  // namespace sankhya::gpu
