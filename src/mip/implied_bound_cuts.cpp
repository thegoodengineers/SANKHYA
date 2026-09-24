// SPDX-License-Identifier: Apache-2.0
// SANKHYA - implied-bound cuts (#499). See implied_bound_cuts.hpp for the derivation.

#include "implied_bound_cuts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

struct Entry {
  Index column = -1;
  double value = 0.0;
};

[[nodiscard]] bool is_binary(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  return model.col_type[u] == VarType::kInteger && model.col_lower[u] == 0.0 &&
         model.col_upper[u] == 1.0;
}

/// The cut from one side, written a_x x + a_y y <= b (a lower side is passed negated).
/// Appends it to `out` when the point violates it.
void separate_side(const Model& model, const std::vector<double>& x, Index jx, double a_x,
                   Index jy, double a_y, double b, std::vector<Cut>* out) {
  const auto ux = static_cast<std::size_t>(jx);
  const auto uy = static_cast<std::size_t>(jy);
  const double y = x[uy];
  if (y <= tol::kIntegrality || y >= 1.0 - tol::kIntegrality) return;
  const auto widen = [](double v, double direction) {
    return v + direction * tol::kPrimalFeasibility * std::max(1.0, std::fabs(v));
  };
  Cut cut;
  cut.family = CutFamily::kImpliedBound;
  cut.coeff.assign(model.col_cost.size(), 0.0);
  if (a_x > 0.0) {
    // Upper bounds on x in the two cases, capped by x's own upper bound.
    const double cap = model.col_upper[ux];
    const double u0 = std::min(cap, widen(b / a_x, 1.0));
    const double u1 = std::min(cap, widen((b - a_y) / a_x, 1.0));
    if (!std::isfinite(u0) || !std::isfinite(u1)) return;
    // x - (u1 - u0) y <= u0
    cut.coeff[ux] = 1.0;
    cut.coeff[uy] = -(u1 - u0);
    cut.rhs = u0;
  } else {
    // Lower bounds on x in the two cases, capped by x's own lower bound.
    const double floor_x = model.col_lower[ux];
    const double l0 = std::max(floor_x, widen(b / a_x, -1.0));
    const double l1 = std::max(floor_x, widen((b - a_y) / a_x, -1.0));
    if (!std::isfinite(l0) || !std::isfinite(l1)) return;
    // x >= l0 + (l1 - l0) y, written -x + (l1 - l0) y <= -l0
    cut.coeff[ux] = -1.0;
    cut.coeff[uy] = l1 - l0;
    cut.rhs = -l0;
  }
  const double activity = cut.coeff[ux] * x[ux] + cut.coeff[uy] * y;
  if (activity - cut.rhs <= tol::kCutViolationTolerance * std::max(1.0, std::fabs(cut.rhs))) {
    return;
  }
  out->push_back(std::move(cut));
}

}  // namespace

std::vector<Cut> implied_bound_cuts(const Model& model, const std::vector<double>& x) {
  std::vector<Cut> cuts;
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<Index>(x.size()) != n || m == 0) return cuts;
  // One pass over the columns gives every row's first two entries and its entry count.
  std::vector<std::array<Entry, 2>> first(static_cast<std::size_t>(m));
  std::vector<int> count(static_cast<std::size_t>(m), 0);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto r = static_cast<std::size_t>(column.rows[k]);
      if (count[r] < 2) first[r][static_cast<std::size_t>(count[r])] = {j, column.values[k]};
      ++count[r];
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto r = static_cast<std::size_t>(i);
    if (count[r] != 2) continue;
    Entry e_x = first[r][0];
    Entry e_y = first[r][1];
    if (!is_binary(model, e_y.column)) std::swap(e_x, e_y);
    if (!is_binary(model, e_y.column)) continue;
    if (model.col_type[static_cast<std::size_t>(e_x.column)] != VarType::kContinuous) continue;
    if (e_x.value == 0.0) continue;
    if (std::isfinite(model.row_upper[r])) {
      separate_side(model, x, e_x.column, e_x.value, e_y.column, e_y.value, model.row_upper[r],
                    &cuts);
    }
    if (std::isfinite(model.row_lower[r])) {
      separate_side(model, x, e_x.column, -e_x.value, e_y.column, -e_y.value,
                    -model.row_lower[r], &cuts);
    }
  }
  return cuts;
}

}  // namespace sankhya::mip
