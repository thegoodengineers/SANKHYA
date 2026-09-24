// SPDX-License-Identifier: Apache-2.0
// SANKHYA - feasibility-based bound tightening for the spatial branch and bound (#514).
//
// Belotti, Lee, Liberti, Margot and Wachter, "Branching and bounds tightening techniques
// for non-convex MINLP", Optimization Methods and Software 24 (2009), sec. 3: every row
// rl <= sum of terms <= ru is read as an interval equation. FORWARD, each term's range over
// the box is summed into the row's range; if that misses [rl, ru] the box holds no feasible
// point. BACKWARD, each term must lie in [rl - (the others' largest), ru - (the others'
// smallest)], which bounds its column: a linear term a x_j directly, a product c x_a x_b
// through interval division by the other factor when that factor's range excludes zero, a
// square through the square root. The objective, when an incumbent exists, is one more row:
// objective <= incumbent.
//
// SOUNDNESS. A tightened bound removes only points that violate a row, so no feasible point
// is lost - provided the arithmetic does not round inward. Every derived bound is relaxed
// outward by kFbbtSafety relative before it is applied, and a bound is moved only when it
// improves by more than kFbbtMinImprovement relative, which also ends the passes: a
// sequence of ever-smaller moves (the classic FBBT stall on x = y, y = x/2 style rows)
// stops instead of running forever.

#include <algorithm>
#include <cmath>
#include <vector>

#include "global/global_internal.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::global {
namespace {

struct Interval {
  double lo;
  double hi;
};

/// x * y where a zero factor wins over an infinite one: the corner of an interval product
/// with a zero bound is zero, whatever the other bound is.
double mul(double x, double y) {
  return (x == 0.0 || y == 0.0) ? 0.0 : x * y;
}

Interval times(Interval a, Interval b) {
  const double c[4] = {mul(a.lo, b.lo), mul(a.lo, b.hi), mul(a.hi, b.lo), mul(a.hi, b.hi)};
  return {*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
}

Interval scale(Interval a, double c) {
  return c >= 0.0 ? Interval{mul(c, a.lo), mul(c, a.hi)} : Interval{mul(c, a.hi), mul(c, a.lo)};
}

Interval square(Interval a) {
  const double lo = a.lo * a.lo;
  const double hi = a.hi * a.hi;
  if (a.lo >= 0.0) return {lo, hi};
  if (a.hi <= 0.0) return {hi, lo};
  return {0.0, std::max(lo, hi)};
}

enum class TermKind { kLinear, kProduct };

struct Term {
  TermKind kind;
  Index index;  ///< the column, or the product
  double coefficient;
};

class Tightener {
 public:
  Tightener(const Problem& problem, double feasibility_tolerance, Box* box)
      : problem_(problem), tolerance_(feasibility_tolerance), box_(*box) {}

  bool infeasible() const noexcept { return infeasible_; }
  Count tightened() const noexcept { return tightened_; }
  bool changed_this_pass() const noexcept { return changed_; }
  void start_pass() noexcept { changed_ = false; }

  Interval column(Index j) const {
    const auto u = static_cast<std::size_t>(j);
    return {box_.lower[u], box_.upper[u]};
  }

  Interval term_range(const Term& term) const {
    if (term.kind == TermKind::kLinear) return scale(column(term.index), term.coefficient);
    const Product& product = problem_.products[static_cast<std::size_t>(term.index)];
    const Interval range = product.square() ? square(column(product.a))
                                            : times(column(product.a), column(product.b));
    return scale(range, term.coefficient);
  }

  /// One row: lower <= sum(terms) <= upper.
  void propagate(const std::vector<Term>& terms, double lower, double upper) {
    if (terms.empty() || infeasible_) return;
    ranges_.resize(terms.size());
    double lo_sum = 0.0;
    double hi_sum = 0.0;
    int lo_inf = 0;
    int hi_inf = 0;
    for (std::size_t t = 0; t < terms.size(); ++t) {
      ranges_[t] = term_range(terms[t]);
      if (std::isfinite(ranges_[t].lo))
        lo_sum += ranges_[t].lo;
      else
        ++lo_inf;
      if (std::isfinite(ranges_[t].hi))
        hi_sum += ranges_[t].hi;
      else
        ++hi_inf;
    }
    // Forward: the row's range must meet [lower, upper].
    if (hi_inf == 0 && lower - hi_sum > slack(lower)) return mark_infeasible();
    if (lo_inf == 0 && lo_sum - upper > slack(upper)) return mark_infeasible();

    // Backward: each term against the others.
    for (std::size_t t = 0; t < terms.size() && !infeasible_; ++t) {
      const Interval r = ranges_[t];
      const bool own_lo_inf = !std::isfinite(r.lo);
      const bool own_hi_inf = !std::isfinite(r.hi);
      const double rest_lo =
          (lo_inf - (own_lo_inf ? 1 : 0)) > 0 ? -kInfinity : lo_sum - (own_lo_inf ? 0.0 : r.lo);
      const double rest_hi =
          (hi_inf - (own_hi_inf ? 1 : 0)) > 0 ? kInfinity : hi_sum - (own_hi_inf ? 0.0 : r.hi);
      const double allowed_lo =
          std::isfinite(lower) && std::isfinite(rest_hi) ? lower - rest_hi : -kInfinity;
      const double allowed_hi =
          std::isfinite(upper) && std::isfinite(rest_lo) ? upper - rest_lo : kInfinity;
      if (!std::isfinite(allowed_lo) && !std::isfinite(allowed_hi)) continue;
      backward(terms[t], allowed_lo, allowed_hi);
    }
  }

 private:
  double slack(double reference) const {
    return tolerance_ * std::max(1.0, std::isfinite(reference) ? std::fabs(reference) : 1.0);
  }

  void mark_infeasible() { infeasible_ = true; }

  /// term.coefficient * (column or product) in [lo, hi].
  void backward(const Term& term, double lo, double hi) {
    const double c = term.coefficient;
    if (c == 0.0) return;
    Interval value = c > 0.0 ? Interval{lo / c, hi / c} : Interval{hi / c, lo / c};
    if (term.kind == TermKind::kLinear) {
      tighten(term.index, value.lo, value.hi);
      return;
    }
    const Product& product = problem_.products[static_cast<std::size_t>(term.index)];
    if (product.square()) {
      if (value.hi < 0.0) {
        if (value.hi < -slack(value.hi)) return mark_infeasible();
        value.hi = 0.0;
      }
      const Interval x = column(product.a);
      if (std::isfinite(value.hi)) {
        const double root = std::sqrt(value.hi);
        tighten(product.a, -root, root);
      }
      if (value.lo > 0.0 && std::isfinite(value.lo)) {
        const double root = std::sqrt(value.lo);
        // |x| >= root: x cannot lie in (-root, root), so the side the box leaves decides.
        if (x.lo > -root) tighten(product.a, root, kInfinity);
        if (x.hi < root) tighten(product.a, -kInfinity, -root);
      }
      return;
    }
    divide_into(product.a, value, column(product.b));
    divide_into(product.b, value, column(product.a));
  }

  /// x_target * other in `value`: x_target in value / other when other excludes zero.
  void divide_into(Index target, Interval value, Interval other) {
    if (!(other.lo > 0.0 || other.hi < 0.0)) return;
    if (!std::isfinite(value.lo) && !std::isfinite(value.hi)) return;
    const Interval reciprocal{1.0 / other.hi, 1.0 / other.lo};
    const Interval x = times(value, reciprocal);
    tighten(target, x.lo, x.hi);
  }

  void tighten(Index j, double lo, double hi) {
    const auto u = static_cast<std::size_t>(j);
    double& lower = box_.lower[u];
    double& upper = box_.upper[u];
    if (std::isfinite(lo)) {
      lo -= tol::kFbbtSafety * std::max(1.0, std::fabs(lo));
      // Above the upper bound by no more than the feasibility tolerance: the column is fixed
      // at that bound. By more: no point of the box satisfies the row.
      if (lo > upper) {
        if (lo - upper > slack(upper)) return mark_infeasible();
        lo = upper;
      }
      if (lo > lower + tol::kFbbtMinImprovement * std::max(1.0, std::fabs(lower)) ||
          (!std::isfinite(lower) && lo > lower) || (lo == upper && lower < upper)) {
        lower = lo;
        changed_ = true;
        ++tightened_;
      }
    }
    if (std::isfinite(hi)) {
      hi += tol::kFbbtSafety * std::max(1.0, std::fabs(hi));
      if (hi < lower) {
        if (lower - hi > slack(lower)) return mark_infeasible();
        hi = lower;
      }
      if (hi < upper - tol::kFbbtMinImprovement * std::max(1.0, std::fabs(upper)) ||
          (!std::isfinite(upper) && hi < upper) || (hi == lower && upper > lower)) {
        upper = hi;
        changed_ = true;
        ++tightened_;
      }
    }
  }

  const Problem& problem_;
  double tolerance_;
  Box& box_;
  std::vector<Interval> ranges_;
  bool infeasible_ = false;
  bool changed_ = false;
  Count tightened_ = 0;
};

}  // namespace

FbbtOutcome fbbt(const Problem& problem, double cutoff, double feasibility_tolerance,
                 Box* box) {
  std::vector<std::vector<Term>> rows(static_cast<std::size_t>(problem.m));
  for (Index i = 0; i < problem.m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    for (const auto& [column, value] : problem.row_linear[u]) {
      rows[u].push_back({TermKind::kLinear, column, value});
    }
    for (const auto& [index, value] : problem.row_products[u]) {
      rows[u].push_back({TermKind::kProduct, index, value});
    }
  }
  std::vector<Term> objective;
  if (std::isfinite(cutoff)) {
    for (Index j = 0; j < problem.n; ++j) {
      const double c = problem.cost[static_cast<std::size_t>(j)];
      if (c != 0.0) objective.push_back({TermKind::kLinear, j, c});
    }
    for (const auto& [index, value] : problem.objective_products) {
      objective.push_back({TermKind::kProduct, index, value});
    }
  }

  Tightener tightener(problem, feasibility_tolerance, box);
  for (int pass = 0; pass < tol::kFbbtMaxPasses; ++pass) {
    tightener.start_pass();
    for (Index i = 0; i < problem.m && !tightener.infeasible(); ++i) {
      const auto u = static_cast<std::size_t>(i);
      tightener.propagate(rows[u], problem.row_lower[u], problem.row_upper[u]);
    }
    if (!objective.empty() && !tightener.infeasible()) {
      tightener.propagate(objective, -kInfinity, cutoff - problem.offset);
    }
    if (tightener.infeasible() || !tightener.changed_this_pass()) break;
  }
  return {tightener.infeasible(), tightener.tightened()};
}

}  // namespace sankhya::global
