// SPDX-License-Identifier: Apache-2.0
// SANKHYA - c-MIR (#498). See mir_cmir.hpp for the method, the citations and the validity
// argument; this file is the bookkeeping.
#include "mir_cmir.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "mir_cuts.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// Largest ratio between a cut's largest and smallest coefficient, as in mir_cuts.cpp.
constexpr double kMaxDynamism = 1e6;
/// Effort caps, so a base row of thousands of integer columns costs a bounded number of
/// trials. Marchand & Wolsey (2001) try the divisors of every integer column strictly inside
/// its bounds; the cap keeps the ones with the largest LP distance from a bound.
constexpr std::size_t kMaxDivisorCandidates = 12;
constexpr std::size_t kMaxComplementTrials = 16;

[[nodiscard]] bool integral(double v) {
  return std::fabs(v - std::round(v)) <= tol::kIntegrality;
}

/// An integer column of the base, with its (merged) coefficient in the model's column.
struct IntegerTerm {
  Index column = -1;
  double a = 0.0;
  double lower = 0.0;
  double upper = 0.0;
  bool complemented = false;  ///< y = upper - x rather than x - lower
};

enum class ContinuousKind { kLower, kUpper, kVariableUpper, kVariableLower };

/// A continuous column of the base, replaced by a slack s >= 0 with coefficient h.
struct ContinuousTerm {
  Index column = -1;
  ContinuousKind kind = ContinuousKind::kLower;
  double h = 0.0;
  double bound = 0.0;  ///< the simple bound, for kLower / kUpper
  VariableBound vb;    ///< for the two variable kinds
};

struct Trial {
  bool ok = false;
  double efficacy = 0.0;
  Cut cut;
};

/// The MIR inequality with `divisor` on the base given by the terms and `b` (the
/// right-hand side after the continuous substitutions), back in the model's columns.
Trial evaluate(const Model& model, const Solution& solution,
               const std::vector<double>& col_lower, const std::vector<double>& col_upper,
               const std::vector<IntegerTerm>& integers,
               const std::vector<ContinuousTerm>& continuous, double b, double divisor,
               double min_violation) {
  Trial trial;
  std::vector<double> coefficient;
  std::vector<bool> is_integer;
  coefficient.reserve(integers.size() + continuous.size());
  for (const IntegerTerm& t : integers) {
    // y = x - lower, or y = upper - x: a x = a lower + a y, or a upper - a y.
    coefficient.push_back(t.complemented ? -t.a : t.a);
    b -= t.a * (t.complemented ? t.upper : t.lower);
    is_integer.push_back(true);
  }
  for (const ContinuousTerm& t : continuous) {
    coefficient.push_back(t.h);
    is_integer.push_back(false);
  }
  std::vector<double> rounded;
  double rhs = 0.0;
  if (!std::isfinite(b) ||
      !mir_inequality(coefficient, is_integer, b, divisor, &rounded, &rhs)) {
    return trial;
  }
  // Back to the model's columns, sparse, then merged.
  std::vector<std::pair<Index, double>> terms;
  for (std::size_t k = 0; k < integers.size(); ++k) {
    const double g = rounded[k];
    if (g == 0.0) continue;
    const IntegerTerm& t = integers[k];
    if (t.complemented) {
      terms.emplace_back(t.column, -g);
      rhs -= g * t.upper;
    } else {
      terms.emplace_back(t.column, g);
      rhs += g * t.lower;
    }
  }
  for (std::size_t k = 0; k < continuous.size(); ++k) {
    const double g = rounded[integers.size() + k];
    if (g == 0.0) continue;
    const ContinuousTerm& t = continuous[k];
    switch (t.kind) {
      case ContinuousKind::kLower:  // s = x - l
        terms.emplace_back(t.column, g);
        rhs += g * t.bound;
        break;
      case ContinuousKind::kUpper:  // s = u - x
        terms.emplace_back(t.column, -g);
        rhs -= g * t.bound;
        break;
      case ContinuousKind::kVariableUpper:  // s = m y + c - x
        terms.emplace_back(t.column, -g);
        terms.emplace_back(t.vb.indicator, g * t.vb.slope);
        rhs -= g * t.vb.constant;
        break;
      case ContinuousKind::kVariableLower:  // s = x - m y - c
        terms.emplace_back(t.column, g);
        terms.emplace_back(t.vb.indicator, -g * t.vb.slope);
        rhs += g * t.vb.constant;
        break;
    }
  }
  std::sort(terms.begin(), terms.end(),
            [](const auto& p, const auto& q) { return p.first < q.first; });
  Cut cut;
  cut.family = CutFamily::kMir;
  cut.coeff.assign(static_cast<std::size_t>(model.num_cols()), 0.0);
  for (const auto& [column, value] : terms)
    cut.coeff[static_cast<std::size_t>(column)] += value;
  // A coefficient at or under the solver's zero drop is dropped when the row is appended;
  // pay for it here so what is appended is still valid: e x >= min(e l, e u) over the box.
  // The box is the one the cut is derived under (the GLOBAL bounds in the tree), never the
  // node's.
  double lhs = 0.0;
  double norm_squared = 0.0;
  double largest = 0.0;
  double smallest = std::numeric_limits<double>::infinity();
  for (const auto& [column, unused] : terms) {
    (void)unused;
    const auto u = static_cast<std::size_t>(column);
    double& e = cut.coeff[u];
    if (e == 0.0) continue;
    if (std::fabs(e) <= tol::kZeroDrop) {
      const double lo = col_lower[u];
      const double hi = col_upper[u];
      if (!is_finite_bound(lo) || !is_finite_bound(hi)) return trial;
      rhs -= std::min(e * lo, e * hi);
      e = 0.0;
      continue;
    }
    lhs += e * solution.col_value[u];
    norm_squared += e * e;
    largest = std::max(largest, std::fabs(e));
    smallest = std::min(smallest, std::fabs(e));
  }
  if (largest == 0.0 || largest / smallest > kMaxDynamism || !std::isfinite(rhs)) return trial;
  const double violation = lhs - rhs;
  if (violation <= min_violation * std::max(1.0, std::fabs(rhs))) return trial;
  cut.rhs = rhs;
  trial.ok = true;
  trial.efficacy = violation / std::sqrt(norm_squared);
  trial.cut = std::move(cut);
  return trial;
}

}  // namespace

VariableBounds find_variable_bounds(const Model& model) {
  VariableBounds found;
  const auto n = static_cast<std::size_t>(model.num_cols());
  found.upper.assign(n, {});
  found.lower.assign(n, {});
  // Row-wise: the two entries of every two-nonzero row.
  std::vector<int> count(static_cast<std::size_t>(model.num_rows()), 0);
  std::vector<std::pair<Index, double>> first(static_cast<std::size_t>(model.num_rows()));
  std::vector<std::pair<Index, double>> second(static_cast<std::size_t>(model.num_rows()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] == 0.0) continue;
      const auto r = static_cast<std::size_t>(column.rows[k]);
      if (count[r] == 0) first[r] = {j, column.values[k]};
      if (count[r] == 1) second[r] = {j, column.values[k]};
      ++count[r];
    }
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto r = static_cast<std::size_t>(i);
    if (count[r] != 2) continue;
    auto [xc, a] = first[r];
    auto [yc, b] = second[r];
    const auto type = [&](Index j) { return model.col_type[static_cast<std::size_t>(j)]; };
    if (type(xc) == VarType::kInteger) {
      std::swap(xc, yc);
      std::swap(a, b);
    }
    if (type(xc) == VarType::kInteger || type(yc) != VarType::kInteger) continue;
    if (std::fabs(a) <= tol::kZeroDrop || std::fabs(b) <= tol::kZeroDrop) continue;
    bool any = false;
    // a x + b y <= U:  x <= U/a - (b/a) y when a > 0, x >= it when a < 0.
    if (is_finite_bound(model.row_upper[r])) {
      const VariableBound vb{yc, -b / a, model.row_upper[r] / a};
      (a > 0.0 ? found.upper : found.lower)[static_cast<std::size_t>(xc)].push_back(vb);
      any = true;
    }
    // a x + b y >= L:  x >= L/a - (b/a) y when a > 0, x <= it when a < 0.
    if (is_finite_bound(model.row_lower[r])) {
      const VariableBound vb{yc, -b / a, model.row_lower[r] / a};
      (a > 0.0 ? found.lower : found.upper)[static_cast<std::size_t>(xc)].push_back(vb);
      any = true;
    }
    if (any) ++found.rows;
  }
  return found;
}

std::optional<Cut> separate_cmir(const Model& model, const Solution& solution,
                                 const std::vector<double>& col_lower,
                                 const std::vector<double>& col_upper,
                                 const std::vector<Index>& columns,
                                 const std::vector<double>& values, double rhs,
                                 const VariableBounds& bounds, double min_violation) {
  const std::vector<double>& x = solution.col_value;
  double b = rhs;
  std::map<Index, double> integer_coefficient;  // ordered: deterministic trials
  std::vector<ContinuousTerm> continuous;
  // An indicator is usable only when it can itself be bound-substituted later.
  const auto usable_indicator = [&](Index y) {
    const auto u = static_cast<std::size_t>(y);
    return (is_finite_bound(col_lower[u]) && integral(col_lower[u])) ||
           (is_finite_bound(col_upper[u]) && integral(col_upper[u]));
  };
  for (std::size_t k = 0; k < columns.size(); ++k) {
    const Index j = columns[k];
    const double coef = values[k];
    if (coef == 0.0) continue;
    const auto u = static_cast<std::size_t>(j);
    if (model.col_type[u] == VarType::kInteger) {
      integer_coefficient[j] += coef;
      continue;
    }
    // The closest bound, simple or variable, at the LP point; a variable one wins a tie,
    // because it is the one that carries the structure.
    ContinuousTerm term;
    term.column = j;
    double best = std::numeric_limits<double>::infinity();
    bool found = false;
    if (is_finite_bound(col_lower[u]) && x[u] - col_lower[u] < best) {
      best = x[u] - col_lower[u];
      term.kind = ContinuousKind::kLower;
      term.bound = col_lower[u];
      found = true;
    }
    if (is_finite_bound(col_upper[u]) && col_upper[u] - x[u] < best) {
      best = col_upper[u] - x[u];
      term.kind = ContinuousKind::kUpper;
      term.bound = col_upper[u];
      found = true;
    }
    for (const VariableBound& vb : bounds.upper[u]) {
      if (!usable_indicator(vb.indicator)) continue;
      const double d =
          vb.slope * x[static_cast<std::size_t>(vb.indicator)] + vb.constant - x[u];
      if (d <= best) {
        best = d;
        term.kind = ContinuousKind::kVariableUpper;
        term.vb = vb;
        found = true;
      }
    }
    for (const VariableBound& vb : bounds.lower[u]) {
      if (!usable_indicator(vb.indicator)) continue;
      const double d =
          x[u] - vb.slope * x[static_cast<std::size_t>(vb.indicator)] - vb.constant;
      if (d <= best) {
        best = d;
        term.kind = ContinuousKind::kVariableLower;
        term.vb = vb;
        found = true;
      }
    }
    if (!found) return std::nullopt;  // a free continuous column has no substitute
    switch (term.kind) {
      case ContinuousKind::kLower:  // x = l + s
        term.h = coef;
        b -= coef * term.bound;
        break;
      case ContinuousKind::kUpper:  // x = u - s
        term.h = -coef;
        b -= coef * term.bound;
        break;
      case ContinuousKind::kVariableUpper:  // x = m y + c - s
        term.h = -coef;
        integer_coefficient[term.vb.indicator] += coef * term.vb.slope;
        b -= coef * term.vb.constant;
        break;
      case ContinuousKind::kVariableLower:  // x = m y + c + s
        term.h = coef;
        integer_coefficient[term.vb.indicator] += coef * term.vb.slope;
        b -= coef * term.vb.constant;
        break;
    }
    continuous.push_back(term);
  }

  std::vector<IntegerTerm> integers;
  bool any_fractional = false;
  for (const auto& [j, a] : integer_coefficient) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = col_lower[u];
    const double hi = col_upper[u];
    const bool has_lo = is_finite_bound(lo) && integral(lo);
    const bool has_hi = is_finite_bound(hi) && integral(hi);
    if (std::fabs(a) <= tol::kZeroDrop) {
      // Cancelled to rounding: dropped, paid for over the box so the base stays valid.
      if (a == 0.0) continue;
      if (!is_finite_bound(lo) || !is_finite_bound(hi)) return std::nullopt;
      b -= std::min(a * lo, a * hi);
      continue;
    }
    if (!has_lo && !has_hi) return std::nullopt;
    IntegerTerm t;
    t.column = j;
    t.a = a;
    t.lower = has_lo ? std::round(lo) : lo;
    t.upper = has_hi ? std::round(hi) : hi;
    // The nearer bound, as the plain MIR does.
    t.complemented = has_hi && (!has_lo || hi - x[u] < x[u] - lo);
    integers.push_back(t);
    if (!integral(x[u])) any_fractional = true;
  }
  if (!any_fractional || integers.empty()) return std::nullopt;

  // Divisors: 1 and |a_j| of each integer column strictly inside its bounds, the ones
  // furthest from a bound first.
  std::vector<std::pair<double, double>> inside;  // (distance to the nearer bound, |a|)
  for (const IntegerTerm& t : integers) {
    const double v = x[static_cast<std::size_t>(t.column)];
    const double to_lower = is_finite_bound(t.lower) ? v - t.lower : kInfinity;
    const double to_upper = is_finite_bound(t.upper) ? t.upper - v : kInfinity;
    const double distance = std::min(to_lower, to_upper);
    if (distance > tol::kIntegrality) inside.emplace_back(distance, std::fabs(t.a));
  }
  std::sort(inside.begin(), inside.end(),
            [](const auto& p, const auto& q) { return p.first > q.first; });
  std::vector<double> divisors{1.0};
  for (const auto& [distance, magnitude] : inside) {
    (void)distance;
    if (divisors.size() > kMaxDivisorCandidates) break;
    if (std::none_of(divisors.begin(), divisors.end(),
                     [&](double d) { return std::fabs(d - magnitude) <= 1e-12 * d; })) {
      divisors.push_back(magnitude);
    }
  }

  Trial best;
  double best_divisor = 0.0;
  const auto consider = [&](const std::vector<IntegerTerm>& ints, double divisor) {
    Trial t = evaluate(model, solution, col_lower, col_upper, ints, continuous, b, divisor,
                       min_violation);
    if (t.ok && (!best.ok || t.efficacy > best.efficacy)) {
      best = std::move(t);
      best_divisor = divisor;
      return true;
    }
    return false;
  };
  for (const double divisor : divisors) consider(integers, divisor);
  if (!best.ok) return std::nullopt;
  const double found_divisor = best_divisor;
  for (const double scale : {2.0, 4.0, 8.0}) consider(integers, found_divisor / scale);

  // Complementation, nearest the middle of the box first: flip, keep what helps.
  std::vector<std::size_t> order;
  for (std::size_t k = 0; k < integers.size(); ++k) {
    if (is_finite_bound(integers[k].lower) && is_finite_bound(integers[k].upper) &&
        integral(integers[k].lower) && integral(integers[k].upper)) {
      order.push_back(k);
    }
  }
  const auto from_middle = [&](std::size_t k) {
    const IntegerTerm& t = integers[k];
    return std::fabs(x[static_cast<std::size_t>(t.column)] - 0.5 * (t.lower + t.upper));
  };
  std::stable_sort(order.begin(), order.end(), [&](std::size_t p, std::size_t q) {
    return from_middle(p) < from_middle(q);
  });
  if (order.size() > kMaxComplementTrials) order.resize(kMaxComplementTrials);
  for (const std::size_t k : order) {
    integers[k].complemented = !integers[k].complemented;
    if (!consider(integers, best_divisor)) integers[k].complemented = !integers[k].complemented;
  }
  return std::move(best.cut);
}

}  // namespace sankhya::mip
