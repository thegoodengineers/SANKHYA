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

/// Where the device's wall time goes (#510): each phase timed on the host clock, every phase
/// ending in a synchronous CUDA call, so the phases add up to the call's wall time.
struct PropPhases {
  double context = 0.0;   ///< device probe and the first CUDA call that needs the context
  double host = 0.0;      ///< the row-major copy and the index narrowing, on the host
  double allocate = 0.0;  ///< cudaMalloc of every buffer
  double upload = 0.0;    ///< host to device copies of the matrix, the sides and the bounds
  double rounds = 0.0;    ///< the kernels, with the per-round flag and counter read-backs
  double download = 0.0;  ///< device to host copy of the tightened bounds
  double release = 0.0;   ///< cudaFree of every buffer
};

struct PropResult {
  std::vector<double> col_lb;  ///< tightened lower bounds (model.num_cols())
  std::vector<double> col_ub;  ///< tightened upper bounds
  bool infeasible = false;     ///< a row or a column proved the box empty
  bool ran = false;            ///< false when no device was available or a CUDA call failed
  int rounds = 0;
  long long tightened = 0;
  PropPhases phases;
};

/// Propagate [col_lb, col_ub] on the device for at most `rounds_limit` rounds. When no device
/// is available, or a CUDA call fails, returns the input bounds with ran = false; the caller
/// then uses the CPU reference.
[[nodiscard]] PropResult propagate_bounds(const Model& model, const std::vector<double>& col_lb,
                                          const std::vector<double>& col_ub, int rounds_limit,
                                          double integrality);

}  // namespace sankhya::gpu
