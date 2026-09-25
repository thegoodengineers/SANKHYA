// SPDX-License-Identifier: Apache-2.0
// C++20 implementation of batch_safe_bound — compiled as a .cpp so it can use std::span.
#include "pdhg_batch_bound.hpp"

#include "../core/safe_bound.hpp"

#include <span>

namespace sankhya::gpu {

double batch_safe_bound(const Model& model, const double* y, int m) {
  const sankhya::SafeBoundProblem sbp{
      &model.matrix,
      std::span<const double>{model.col_cost},
      std::span<const double>{model.row_lower},
      std::span<const double>{model.row_upper},
      std::span<const double>{model.col_lower},
      std::span<const double>{model.col_upper},
  };
  const std::span<const double> y_span(y, static_cast<std::size_t>(m));
  const sankhya::SafeBound sb = sankhya::safe_dual_bound(sbp, y_span);
  return sb.value;
}

}  // namespace sankhya::gpu
