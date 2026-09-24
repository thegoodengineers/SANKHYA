// SPDX-License-Identifier: Apache-2.0
// SANKHYA - coefficient tightening and big-M strengthening (#511). See coef_tightening.hpp
// for the references, the reduction and why the integer points are unchanged.

#include "presolve/coef_tightening.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {
namespace {

/// Products and sums of integers below 2^40 are exact in a double (53-bit mantissa), with
/// room for 2^13 of them to be added up. A row whose every term is such an integer has an
/// exact maximum activity, and needs no margin on g; on MIPLIB that is most rows, and the
/// tightened coefficients then stay integers.
constexpr double kExactIntegerLimit = 1099511627776.0;  // 2^40

[[nodiscard]] bool finite(double v) {
  return std::fabs(v) < kInfinity;
}

[[nodiscard]] bool exact_integer(double v) {
  return std::fabs(v) <= kExactIntegerLimit && std::floor(v) == v;
}

/// The rows of a column-compressed matrix, as positions into its value array: the values
/// are edited in place and the CSC arrays handed back to the matrix at the end, so the
/// column order and the sparsity pattern never change.
struct RowIndex {
  std::vector<Index> start;     ///< num_rows + 1 offsets into `position`
  std::vector<Index> position;  ///< CSC positions, grouped by row
  std::vector<Index> column;    ///< the column of each CSC position
};

RowIndex index_rows(Index m, Index n, const std::vector<Index>& starts,
                    const std::vector<Index>& rows) {
  RowIndex index;
  const auto nnz = rows.size();
  index.start.assign(static_cast<std::size_t>(m) + 1, 0);
  index.position.resize(nnz);
  index.column.resize(nnz);
  for (const Index r : rows) ++index.start[static_cast<std::size_t>(r) + 1];
  for (std::size_t i = 0; i < static_cast<std::size_t>(m); ++i) {
    index.start[i + 1] += index.start[i];
  }
  std::vector<Index> next(index.start.begin(), index.start.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index k = starts[static_cast<std::size_t>(j)];
         k < starts[static_cast<std::size_t>(j) + 1]; ++k) {
      const auto u = static_cast<std::size_t>(k);
      index.position[static_cast<std::size_t>(next[static_cast<std::size_t>(rows[u])]++)] = k;
      index.column[u] = j;
    }
  }
  return index;
}

/// Activity range of one row over the current box, with the count of infinite contributions
/// kept so a single infinite term can still be excluded when propagating onto that column.
struct Activity {
  double min = 0.0;
  double max = 0.0;
  int min_infinite = 0;
  int max_infinite = 0;
  double magnitude = 0.0;  ///< sum of |finite terms|, the scale of the rounding error
  bool exact = true;       ///< every finite term an exact integer, so min and max are exact
};

class Tightener {
 public:
  explicit Tightener(Model* model)
      : model_(model),
        m_(model->num_rows()),
        n_(model->num_cols()),
        starts_(model->matrix.column_starts()),
        rows_(model->matrix.row_indices()),
        values_(model->matrix.values()),
        index_(index_rows(m_, n_, starts_, rows_)) {}

  CoefficientTighteningStats run() {
    round_integer_bounds();
    std::vector<bool> tightened_row(static_cast<std::size_t>(m_), false);
    for (int round = 0; round < tol::kPresolveCoefficientRounds; ++round) {
      ++stats_.rounds;
      bool changed = false;
      for (Index i = 0; i < m_ && !stats_.found_crossing_bounds; ++i) {
        changed = propagate(i) || changed;
      }
      if (stats_.found_crossing_bounds) break;
      for (Index i = 0; i < m_; ++i) {
        if (tighten(i)) {
          changed = true;
          tightened_row[static_cast<std::size_t>(i)] = true;
        }
      }
      if (!changed) break;
    }
    stats_.rows_tightened =
        static_cast<Count>(std::count(tightened_row.begin(), tightened_row.end(), true));
    if (stats_.coefficients_tightened > 0) {
      model_->matrix.assign_columns(m_, n_, std::move(starts_), std::move(rows_),
                                    std::move(values_));
    }
    return stats_;
  }

 private:
  [[nodiscard]] bool is_integer(Index j) const {
    return model_->col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
  }
  double& lower(Index j) { return model_->col_lower[static_cast<std::size_t>(j)]; }
  double& upper(Index j) { return model_->col_upper[static_cast<std::size_t>(j)]; }

  /// The argument steps an integer column by whole units from its bound, which needs the
  /// bound to be an integer. Inward, as presolve rounds: an integer bound of 2.5 means 2.
  void round_integer_bounds() {
    for (Index j = 0; j < n_; ++j) {
      if (!is_integer(j)) continue;
      if (finite(lower(j))) lower(j) = std::ceil(lower(j) - tol::kIntegrality);
      if (finite(upper(j))) upper(j) = std::floor(upper(j) + tol::kIntegrality);
    }
  }

  /// Activity of `sign` * row i over the current box.
  Activity activity(Index i, double sign) {
    Activity a;
    for (Index p = index_.start[static_cast<std::size_t>(i)];
         p < index_.start[static_cast<std::size_t>(i) + 1]; ++p) {
      const Index k = index_.position[static_cast<std::size_t>(p)];
      const Index j = index_.column[static_cast<std::size_t>(k)];
      const double c = sign * values_[static_cast<std::size_t>(k)];
      const double low = c > 0.0 ? lower(j) : upper(j);
      const double high = c > 0.0 ? upper(j) : lower(j);
      if (finite(low)) {
        const double t = c * low;
        a.min += t;
        a.magnitude += std::fabs(t);
        a.exact = a.exact && exact_integer(t);
      } else {
        ++a.min_infinite;
      }
      if (finite(high)) {
        const double t = c * high;
        a.max += t;
        a.magnitude += std::fabs(t);
        a.exact = a.exact && exact_integer(t);
      } else {
        ++a.max_infinite;
      }
    }
    return a;
  }

  /// Bound propagation (Savelsbergh 1994, sec. 2; Brearley, Mitra & Williams 1975): each
  /// column's bound implied by the row's bounds and the other columns' activity range.
  /// The activity is computed once per row, before any of this row's columns move; a bound
  /// that moved earlier in the pass only makes the stale range wider, and a wider range
  /// implies a weaker bound, never a wrong one.
  bool propagate(Index i) {
    const double row_lower = model_->row_lower[static_cast<std::size_t>(i)];
    const double row_upper = model_->row_upper[static_cast<std::size_t>(i)];
    if (!finite(row_lower) && !finite(row_upper)) return false;
    const Activity act = activity(i, 1.0);
    const double row_scale = std::max(finite(row_lower) ? std::fabs(row_lower) : 0.0,
                                      finite(row_upper) ? std::fabs(row_upper) : 0.0) +
                             act.magnitude;
    bool changed = false;
    for (Index p = index_.start[static_cast<std::size_t>(i)];
         p < index_.start[static_cast<std::size_t>(i) + 1] && !stats_.found_crossing_bounds;
         ++p) {
      const Index k = index_.position[static_cast<std::size_t>(p)];
      const Index j = index_.column[static_cast<std::size_t>(k)];
      const double a = values_[static_cast<std::size_t>(k)];
      if (std::fabs(a) <= tol::kZeroDrop) continue;
      const double lo = lower(j);
      const double up = upper(j);
      const double own_min = a > 0.0 ? lo : up;
      const double own_max = a > 0.0 ? up : lo;
      // The implied bound (bound - rest) / a, loosened by a margin unless it is exact: an
      // integer numerator (every term and the bound exact integers) that a divides exactly.
      const auto implied = [&](double bound, double rest, bool upward) {
        const double numerator = bound - rest;
        const double v = numerator / a;
        const bool exact = act.exact && exact_integer(bound) && v * a == numerator;
        const double slack =
            exact ? 0.0
                  : tol::kPresolveCoefficientSafety * std::max(1.0, row_scale) / std::fabs(a);
        return upward ? v + slack : v - slack;
      };
      if (finite(row_upper)) {
        // a x_j <= row_upper - (minimum of the rest)
        const int others = act.min_infinite - (finite(own_min) ? 0 : 1);
        if (others == 0) {
          const double rest = finite(own_min) ? act.min - a * own_min : act.min;
          changed = (a > 0.0 ? set_upper(j, implied(row_upper, rest, true))
                             : set_lower(j, implied(row_upper, rest, false))) ||
                    changed;
        }
      }
      if (finite(row_lower)) {
        // a x_j >= row_lower - (maximum of the rest)
        const int others = act.max_infinite - (finite(own_max) ? 0 : 1);
        if (others == 0) {
          const double rest = finite(own_max) ? act.max - a * own_max : act.max;
          changed = (a > 0.0 ? set_lower(j, implied(row_lower, rest, false))
                             : set_upper(j, implied(row_lower, rest, true))) ||
                    changed;
        }
      }
    }
    return changed;
  }

  bool set_upper(Index j, double v) {
    if (!std::isfinite(v) || std::fabs(v) > tol::kPresolveMaxPropagatedBound) return false;
    if (is_integer(j)) v = std::floor(v + tol::kIntegrality);
    const double old = upper(j);
    const bool improves =
        !finite(old) ||
        (is_integer(j) ? v < old
                       : v < old - tol::kPresolveBoundMinStep * std::max(1.0, std::fabs(old)));
    if (!improves) return false;
    const double lo = lower(j);
    if (finite(lo) && v < lo) {
      if (v < lo - tol::kPrimalFeasibility * std::max(1.0, std::fabs(lo))) {
        stats_.found_crossing_bounds = true;
        return false;
      }
      v = lo;
      if (v >= old) return false;
    }
    upper(j) = v;
    ++stats_.bounds_tightened;
    return true;
  }

  bool set_lower(Index j, double v) {
    if (!std::isfinite(v) || std::fabs(v) > tol::kPresolveMaxPropagatedBound) return false;
    if (is_integer(j)) v = std::ceil(v - tol::kIntegrality);
    const double old = lower(j);
    const bool improves =
        !finite(old) ||
        (is_integer(j) ? v > old
                       : v > old + tol::kPresolveBoundMinStep * std::max(1.0, std::fabs(old)));
    if (!improves) return false;
    const double up = upper(j);
    if (finite(up) && v > up) {
      if (v > up + tol::kPrimalFeasibility * std::max(1.0, std::fabs(up))) {
        stats_.found_crossing_bounds = true;
        return false;
      }
      v = up;
      if (v <= old) return false;
    }
    lower(j) = v;
    ++stats_.bounds_tightened;
    return true;
  }

  /// Coefficient tightening on one side of row i. A one-sided row, or a ranged one whose
  /// other side the box already implies (that side is dropped when the row changes, which
  /// is exact: the box stays in the model). Never an equality.
  bool tighten(Index i) {
    const auto r = static_cast<std::size_t>(i);
    const double row_lower = model_->row_lower[r];
    const double row_upper = model_->row_upper[r];
    if (finite(row_lower) && finite(row_upper) && row_lower == row_upper) return false;
    if (finite(row_upper) && tighten_side(i, +1.0)) return true;
    if (finite(row_lower) && tighten_side(i, -1.0)) return true;
    return false;
  }

  /// `sign` * row <= `sign` * bound: +1 is the row's upper side, -1 its lower side.
  bool tighten_side(Index i, double sign) {
    const auto r = static_cast<std::size_t>(i);
    const double other = sign > 0.0 ? model_->row_lower[r] : model_->row_upper[r];
    double b = sign > 0.0 ? model_->row_upper[r] : -model_->row_lower[r];
    const Activity act = activity(i, sign);
    if (act.max_infinite > 0) return false;
    const bool exact = act.exact && exact_integer(b);
    const double margin =
        exact ? 0.0
              : tol::kPresolveCoefficientSafety * std::max(1.0, act.magnitude + std::fabs(b));
    // The other side, if any, must already hold on the whole box: sign * row >= sign *
    // other is implied when the minimum activity reaches it.
    if (finite(other)) {
      if (act.min_infinite > 0 || act.min < sign * other + margin) return false;
    }
    const double g = act.max - b + margin;  // rounded up, never down
    if (g <= 0.0) return false;             // the side cannot bind on the box
    double largest = 0.0;
    for (Index p = index_.start[r]; p < index_.start[r + 1]; ++p) {
      largest = std::max(
          largest,
          std::fabs(
              values_[static_cast<std::size_t>(index_.position[static_cast<std::size_t>(p)])]));
    }
    if (g < tol::kPresolveCoefficientMinStep * std::max(1.0, largest)) return false;

    Count count = 0;
    for (Index p = index_.start[r]; p < index_.start[r + 1]; ++p) {
      const Index k = index_.position[static_cast<std::size_t>(p)];
      const Index j = index_.column[static_cast<std::size_t>(k)];
      if (!is_integer(j)) continue;
      double& value = values_[static_cast<std::size_t>(k)];
      const double c = sign * value;
      const double d = std::fabs(c) - g;
      if (d <= tol::kPresolveCoefficientMinStep * std::max(1.0, std::fabs(c))) continue;
      // c > 0 is maximised at the upper bound and the row moves by d there; c < 0 at the
      // lower bound. Both are finite: act.max is.
      if (c > 0.0) {
        b -= d * upper(j);
        value = sign * g;
      } else {
        b += d * lower(j);
        value = -sign * g;
      }
      ++count;
    }
    if (count == 0) return false;
    stats_.coefficients_tightened += count;
    if (sign > 0.0) {
      model_->row_upper[r] = b;
      model_->row_lower[r] = -kInfinity;
    } else {
      model_->row_lower[r] = -b;
      model_->row_upper[r] = kInfinity;
    }
    return true;
  }

  Model* model_;
  Index m_;
  Index n_;
  std::vector<Index> starts_;
  std::vector<Index> rows_;
  std::vector<double> values_;
  RowIndex index_;
  CoefficientTighteningStats stats_;
};

}  // namespace

CoefficientTighteningStats tighten_coefficients(Model* model) {
  if (model->num_rows() == 0 || !model->has_integrality()) return {};
  return Tightener(model).run();
}

}  // namespace sankhya::presolve
