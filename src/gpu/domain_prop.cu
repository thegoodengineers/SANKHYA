// SPDX-License-Identifier: Apache-2.0
// GPU activity-based bound propagation (#510).
//
// Algorithm (Sofranac, Gleixner & Pokutta, arXiv:2009.07785, §3):
//   For each row i, compute activity bounds
//     act_min_i = sum_{a_ij > 0} a_ij * lb_j + sum_{a_ij < 0} a_ij * ub_j
//     act_max_i = sum_{a_ij > 0} a_ij * ub_j + sum_{a_ij < 0} a_ij * lb_j
//   and tighten col bounds by reversing the contribution of each variable:
//     lb_j >= (rhs_lo_i - (act_max_i - a_ij * ub_j)) / a_ij   (a_ij > 0)
//     ub_j <= (rhs_hi_i - (act_min_i - a_ij * lb_j)) / a_ij   (a_ij > 0)
//   Symmetric for a_ij < 0.  One thread per row; atomic min/max on col arrays.
//
// Current state: stub — returns input bounds unchanged.
// Kernel sketch is in the comments above; activate by setting gpu_domain_prop=true
// (option added in src/util/options.cpp) once the kernel is written and tested.

#include "domain_prop.hpp"
#include "device.hpp"

namespace sankhya::gpu {

PropResult
propagate_bounds(const Model& model,
                 const std::vector<double>& col_lb,
                 const std::vector<double>& col_ub,
                 const Options& options)
{
    if (!options.get_bool("gpu_domain_prop") || !device_available(nullptr)) {
        return {col_lb, col_ub, false};
    }

    // TODO(#510): launch propagation kernel. Until then the model is unread; -Werror under
  // nvcc says so unless told otherwise.
  (void)model;
    //   1. Copy CSC matrix A, col_lb, col_ub to device.
    //   2. Launch act_bounds_kernel<<<nrows/256+1, 256>>> to fill d_act_min, d_act_max.
    //   3. Launch tighten_kernel<<<ncols/256+1, 256>>> for each col using atomicMin/Max.
    //   4. Repeat until convergence or max_rounds reached.
    //   5. Copy tightened bounds back; detect infeasibility (lb > ub + eps).
    //   6. Free device memory; return PropResult.

    return {col_lb, col_ub, false};
}

}  // namespace sankhya::gpu
