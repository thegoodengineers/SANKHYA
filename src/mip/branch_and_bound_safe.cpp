// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: pruning on a safe bound (#519).
//
// With `safe_bounds` on, every node LP's bound is recomputed from its row duals by the
// Neumaier-Shcherbina formula with outward rounding (src/core/safe_bound.hpp), and THAT is
// what prunes the node and what its children inherit. The objective of the relaxation's
// primal point - the "believed" bound - is only as good as the point's feasibility, and the
// search logs how far above the safe bound it ever sat.
//
// Reference: A. Neumaier and O. Shcherbina, "Safe bounds in linear and mixed-integer linear
// programming", Mathematical Programming 99 (2004) 283-296.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/safe_bound.hpp"

namespace sankhya::mip {

double BranchAndBound::safe_node_bound(const Solution& relaxation, double believed) {
  ++safe_bound_nodes_;
  double safe = -std::numeric_limits<double>::infinity();
  if (relaxation.row_dual.size() == static_cast<std::size_t>(working_.num_rows())) {
    // working_ holds the node's box, its cut rows and any objective row, which is exactly
    // the LP the duals belong to.
    safe = safe_dual_bound(working_, relaxation.row_dual).value;
  }
  if (!std::isfinite(safe)) {
    ++safe_bound_infinite_;
  } else {
    const double gap = believed - safe;
    safe_bound_max_gap_ = std::max(safe_bound_max_gap_, gap);
    safe_bound_max_rel_gap_ =
        std::max(safe_bound_max_rel_gap_, gap / std::max(1.0, std::fabs(believed)));
  }
  if (can_prune(believed) && !can_prune(safe)) ++safe_bound_refusals_;
  return safe;
}

void BranchAndBound::report_safe_bounds() const {
  if (!safe_bounds_) return;
  logger_.info(
      "Safe bounds (#519): {} node bound(s), {} with no finite safe bound; max believed - safe "
      "{:.3e} ({:.3e} relative); {} node(s) the believed bound would have pruned and the safe "
      "one did not",
      safe_bound_nodes_, safe_bound_infinite_, safe_bound_max_gap_, safe_bound_max_rel_gap_,
      safe_bound_refusals_);
}

}  // namespace sankhya::mip
