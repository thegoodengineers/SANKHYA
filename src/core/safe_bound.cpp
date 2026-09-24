// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the Neumaier-Shcherbina safe LP bound (#519). See safe_bound.hpp for the
// mathematics, the citation and the rounding argument.

#include "core/safe_bound.hpp"

#include "sankhya/tolerances.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMaxShrink = tol::kSafeBoundMaxShrink;
constexpr double kShrinkMargin = tol::kSafeBoundShrinkMargin;

// ---- Outward rounding ------------------------------------------------------------------
//
// Each helper returns a double on the stated side of the EXACT result of the operation on
// its (exact) double operands. Round-to-nearest puts the computed result within half an ulp
// of the exact one; one step of nextafter moves it past it. Two cases are recognised as
// exact and not widened, which keeps integer data and 0/1 matrices exact through the whole
// computation (a reduced cost that is exactly zero then stays exactly zero):
//
//  * a product with a zero operand, or with a power of two (including +-1) whose result is
//    a normal number: scaling by 2^k only changes the exponent;
//  * a sum whose rounding error, computed exactly by Knuth's TwoSum (additions only, so no
//    contraction into a fused multiply-add can disturb it), is zero - and when it is not,
//    its SIGN says which side of the exact sum the computed one is on, so only the wrong
//    side is stepped. Knuth, "The Art of Computer Programming" vol. 2, sec. 4.2.2, thm. B.

[[nodiscard]] double down(double v) noexcept {
  return std::nextafter(v, -kInf);
}
[[nodiscard]] double up(double v) noexcept {
  return std::nextafter(v, kInf);
}
[[nodiscard]] bool power_of_two(double v) noexcept {
  int exponent = 0;
  return std::isfinite(v) && std::fabs(std::frexp(v, &exponent)) == 0.5;
}
/// True when a * b was computed exactly (see above).
[[nodiscard]] bool exact_product(double a, double b, double p) noexcept {
  return (power_of_two(a) || power_of_two(b)) && std::isnormal(p);
}
[[nodiscard]] double mul_down(double a, double b) noexcept {
  if (a == 0.0 || b == 0.0) return 0.0;
  const double p = a * b;
  return exact_product(a, b, p) ? p : down(p);
}
[[nodiscard]] double mul_up(double a, double b) noexcept {
  if (a == 0.0 || b == 0.0) return 0.0;
  const double p = a * b;
  return exact_product(a, b, p) ? p : up(p);
}
/// The exact rounding error of s = a + b: a + b = s + error exactly (Knuth's TwoSum), when
/// s is finite.
[[nodiscard]] double sum_error(double a, double b, double s) noexcept {
  const double bb = s - a;
  return (a - (s - bb)) + (b - bb);
}
[[nodiscard]] double add_down(double a, double b) noexcept {
  if (b == 0.0) return a;
  if (a == 0.0) return b;
  const double s = a + b;
  if (!std::isfinite(s)) return down(s);
  // error >= 0: s is already below the exact sum. Written so a NaN error widens.
  return sum_error(a, b, s) >= 0.0 ? s : down(s);
}
[[nodiscard]] double add_up(double a, double b) noexcept {
  if (b == 0.0) return a;
  if (a == 0.0) return b;
  const double s = a + b;
  if (!std::isfinite(s)) return up(s);
  return sum_error(a, b, s) <= 0.0 ? s : up(s);
}
[[nodiscard]] double sub_down(double a, double b) noexcept {
  return add_down(a, -b);
}
[[nodiscard]] double sub_up(double a, double b) noexcept {
  return add_up(a, -b);
}
[[nodiscard]] double div_down(double a, double b) noexcept {
  return a == 0.0 ? 0.0 : down(a / b);
}
[[nodiscard]] double div_up(double a, double b) noexcept {
  return a == 0.0 ? 0.0 : up(a / b);
}

[[nodiscard]] bool finite(double v) noexcept {
  return is_finite_bound(v) && std::isfinite(v);
}

/// Per-row activity range over the column box, for the implied bounds: the finite part of
/// the minimum (rounded down) and of the maximum (rounded up), and how many columns
/// contribute an infinite amount to each.
struct RowActivity {
  double min_finite = 0.0;
  double max_finite = 0.0;
  Index min_infinite = 0;
  Index max_infinite = 0;
};

class ImpliedBounds {
 public:
  explicit ImpliedBounds(const SafeBoundProblem& p) : p_(p) {}

  /// The tightest upper (or lower) bound on column j that one row and the other columns'
  /// bounds imply, rounded outward; +inf (-inf) when no row gives one.
  double upper(Index j) { return implied(j, true); }
  double lower(Index j) { return implied(j, false); }

 private:
  void build() {
    const SparseMatrix& a = *p_.matrix;
    rows_.assign(static_cast<std::size_t>(a.num_rows()), RowActivity{});
    for (Index j = 0; j < a.num_cols(); ++j) {
      const double l = p_.col_lower[static_cast<std::size_t>(j)];
      const double u = p_.col_upper[static_cast<std::size_t>(j)];
      const ColumnView col = a.column(j);
      for (Index k = 0; k < col.size; ++k) {
        RowActivity& row = rows_[static_cast<std::size_t>(col.rows[k])];
        const double v = col.values[k];
        const double at_min = v > 0.0 ? l : u;
        const double at_max = v > 0.0 ? u : l;
        if (finite(at_min)) {
          row.min_finite = add_down(row.min_finite, mul_down(v, at_min));
        } else {
          ++row.min_infinite;
        }
        if (finite(at_max)) {
          row.max_finite = add_up(row.max_finite, mul_up(v, at_max));
        } else {
          ++row.max_infinite;
        }
      }
    }
    built_ = true;
  }

  /// Lower bound on the row's minimum activity without column j's term `v * x_j`.
  [[nodiscard]] double min_rest(const RowActivity& row, double v, double l, double u) const {
    const double at_min = v > 0.0 ? l : u;
    if (!finite(at_min)) return row.min_infinite == 1 ? row.min_finite : -kInf;
    return row.min_infinite == 0 ? sub_down(row.min_finite, mul_up(v, at_min)) : -kInf;
  }
  /// Upper bound on the row's maximum activity without column j's term.
  [[nodiscard]] double max_rest(const RowActivity& row, double v, double l, double u) const {
    const double at_max = v > 0.0 ? u : l;
    if (!finite(at_max)) return row.max_infinite == 1 ? row.max_finite : kInf;
    return row.max_infinite == 0 ? sub_up(row.max_finite, mul_down(v, at_max)) : kInf;
  }

  double implied(Index j, bool want_upper) {
    if (!built_) build();
    const auto uj = static_cast<std::size_t>(j);
    const double l = p_.col_lower[uj];
    const double u = p_.col_upper[uj];
    double best = want_upper ? kInf : -kInf;
    const ColumnView col = p_.matrix->column(j);
    for (Index k = 0; k < col.size; ++k) {
      const auto i = static_cast<std::size_t>(col.rows[k]);
      const double v = col.values[k];
      const RowActivity& row = rows_[i];
      const double lo_side = p_.row_lower[i];
      const double hi_side = p_.row_upper[i];
      // v x_j <= hi_side - (rest) gives an upper bound when v > 0 and a lower one when
      // v < 0; v x_j >= lo_side - (rest) the other way round.
      const bool use_hi = want_upper == (v > 0.0);
      double candidate;
      if (use_hi) {
        if (!finite(hi_side)) continue;
        const double rest = min_rest(row, v, l, u);
        if (!finite(rest)) continue;
        const double numerator = sub_up(hi_side, rest);  // >= v x_j
        candidate = want_upper ? div_up(numerator, v) : div_down(numerator, v);
      } else {
        if (!finite(lo_side)) continue;
        const double rest = max_rest(row, v, l, u);
        if (!finite(rest)) continue;
        const double numerator = sub_down(lo_side, rest);  // <= v x_j
        candidate = want_upper ? div_up(numerator, v) : div_down(numerator, v);
      }
      if (!std::isfinite(candidate)) continue;
      best = want_upper ? std::min(best, candidate) : std::max(best, candidate);
    }
    return best;
  }

  const SafeBoundProblem& p_;
  std::vector<RowActivity> rows_;
  bool built_ = false;
};

/// Lower bound on min r.x over r in [r_lo, r_hi], x in [l, u], both endpoints finite.
[[nodiscard]] double corner_minimum(double r_lo, double r_hi, double l, double u) noexcept {
  return std::min({mul_down(r_lo, l), mul_down(r_lo, u), mul_down(r_hi, l), mul_down(r_hi, u)});
}

/// The bound from exactly this y. When it is -inf because of columns whose reduced cost has
/// the wrong sign for their missing bound, `*shrink` receives the smallest factor e such
/// that (1 - e) y would, to first order, give those columns the right sign - which exists
/// only when each such column's cost itself has that sign - and +inf when none does.
SafeBound evaluate(const SafeBoundProblem& problem, std::span<const double> y,
                   bool imply_bounds, double* shrink) {
  SafeBound result;
  *shrink = 0.0;
  const SparseMatrix& a = *problem.matrix;
  const auto m = static_cast<std::size_t>(a.num_rows());
  const auto n = static_cast<std::size_t>(a.num_cols());

  // ---- The multipliers actually used, and the rows' part of the bound -----------------
  std::vector<double> used(m, 0.0);
  double total = 0.0;  // a lower bound on the bound, accumulated rounding down
  for (std::size_t i = 0; i < m; ++i) {
    const double yi = y[i];
    if (yi == 0.0) continue;
    const double side = yi > 0.0 ? problem.row_lower[i] : problem.row_upper[i];
    if (!std::isfinite(yi) || !finite(side)) {
      ++result.dropped_multipliers;
      continue;
    }
    used[i] = yi;
    total = add_down(total, mul_down(yi, side));
  }

  // ---- The columns' part: min r_j x_j over the box, r_j = c_j - (A'y)_j -----------------
  ImpliedBounds implied(problem);
  bool unbounded = false;
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView col = a.column(static_cast<Index>(j));
    double s_lo = 0.0;
    double s_hi = 0.0;
    for (Index k = 0; k < col.size; ++k) {
      const double yi = used[static_cast<std::size_t>(col.rows[k])];
      if (yi == 0.0) continue;
      s_lo = add_down(s_lo, mul_down(col.values[k], yi));
      s_hi = add_up(s_hi, mul_up(col.values[k], yi));
    }
    const double c = problem.cost.empty() ? 0.0 : problem.cost[j];
    const double r_lo = sub_down(c, s_hi);
    const double r_hi = sub_up(c, s_lo);
    if (r_lo == 0.0 && r_hi == 0.0) continue;    // exactly zero: no bound is needed
    if (std::isnan(r_lo) || std::isnan(r_hi)) {  // an overflow upstream: nothing is proved
      ++result.unbounded_columns;
      unbounded = true;
      *shrink = kInf;
      continue;
    }

    double l = problem.col_lower[j];
    double u = problem.col_upper[j];
    const bool need_lower = r_hi > 0.0;  // some r in the interval is positive
    const bool need_upper = r_lo < 0.0;  // some r in the interval is negative
    if (need_lower && !finite(l) && imply_bounds) {
      l = implied.lower(static_cast<Index>(j));
      if (finite(l)) ++result.implied_bounds;
    }
    if (need_upper && !finite(u) && imply_bounds) {
      u = implied.upper(static_cast<Index>(j));
      if (finite(u)) ++result.implied_bounds;
    }
    if ((need_lower && !finite(l)) || (need_upper && !finite(u))) {
      ++result.unbounded_columns;
      unbounded = true;
      // (1 - e) y moves r_j to (1 - e) r_j + e c_j: towards c_j, which fixes the sign only
      // when c_j already has the sign the missing bound needs.
      if (need_upper && !finite(u)) {
        *shrink = c > 0.0 ? std::max(*shrink, -r_lo / (c - r_lo)) : kInf;
      }
      if (need_lower && !finite(l)) {
        *shrink = c < 0.0 ? std::max(*shrink, r_hi / (r_hi - c)) : kInf;
      }
      continue;
    }
    double term;
    if (!need_upper) {
      term = std::min(mul_down(r_lo, l), mul_down(r_hi, l));  // r >= 0: the minimum is at l
    } else if (!need_lower) {
      term = std::min(mul_down(r_lo, u), mul_down(r_hi, u));  // r <= 0: at u
    } else {
      term = corner_minimum(r_lo, r_hi, l, u);
    }
    total = add_down(total, term);
  }
  if (unbounded || std::isnan(total)) return result;
  result.value = total;
  return result;
}

}  // namespace

SafeBound safe_dual_bound(const SafeBoundProblem& problem, std::span<const double> y,
                          bool imply_bounds) {
  SafeBound result;
  if (problem.matrix == nullptr) return result;
  const auto m = static_cast<std::size_t>(problem.matrix->num_rows());
  const auto n = static_cast<std::size_t>(problem.matrix->num_cols());
  if (y.size() != m || problem.row_lower.size() != m || problem.row_upper.size() != m ||
      problem.col_lower.size() != n || problem.col_upper.size() != n ||
      (!problem.cost.empty() && problem.cost.size() != n)) {
    return result;
  }
  // The rounding argument needs round-to-nearest; anything else and the bound is refused.
  if (std::fegetround() != FE_TONEAREST) return result;

  double shrink = 0.0;
  result = evaluate(problem, y, imply_bounds, &shrink);
  if (std::isfinite(result.value) || !(shrink < kMaxShrink)) return result;
  // THE ONE RETRY. A basic column with no upper bound has a reduced cost that is zero only
  // to rounding, and an interval straddling zero needs a bound it does not have. Scaling y
  // towards zero pulls every reduced cost towards its cost, which for a column with a
  // positive cost (the common case: a covering row, a nonnegative flow) is the right side.
  // The scaled vector is just another y, computed any way we like - the bound it yields is
  // exact whatever rounding produced it - and it costs about e * |b.y| of the bound.
  const double factor = std::min(kMaxShrink, kShrinkMargin * shrink);
  std::vector<double> scaled(y.begin(), y.end());
  for (double& v : scaled) v *= 1.0 - factor;
  double unused = 0.0;
  SafeBound retry = evaluate(problem, scaled, imply_bounds, &unused);
  if (!std::isfinite(retry.value)) return result;
  retry.shrink = factor;
  return retry;
}

SafeBound safe_dual_bound(const Model& model, const std::vector<double>& row_dual,
                          bool imply_bounds) {
  const double sense = model.sense_multiplier();
  std::vector<double> cost(model.col_cost.size());
  for (std::size_t j = 0; j < cost.size(); ++j) cost[j] = sense * model.col_cost[j];
  std::vector<double> y(row_dual.size());
  for (std::size_t i = 0; i < y.size(); ++i) y[i] = sense * row_dual[i];
  SafeBoundProblem problem;
  problem.matrix = &model.matrix;
  problem.cost = cost;
  problem.row_lower = model.row_lower;
  problem.row_upper = model.row_upper;
  problem.col_lower = model.col_lower;
  problem.col_upper = model.col_upper;
  return safe_dual_bound(problem, y, imply_bounds);
}

}  // namespace sankhya
