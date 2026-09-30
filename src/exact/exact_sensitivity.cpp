// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certified shadow prices and sensitivity ranges (#757). See
// exact_sensitivity.hpp for the citations and the definitions.
//
// Sparse: each basis is factorised once by the exact LU (exact_lu.hpp) in the float LU's
// pivot order, and every B^{-1} e_i and e_p^T B^{-1} below is one sparse solve. A basis change
// in the parametric dual simplex is a product-form update. The dense Gauss-Jordan inverse this
// replaced was O(m^3) and capped at 300 rows; the only limit now is the time budget.

#include "exact/exact_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "exact/exact_basis.hpp"
#include "exact/rational.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;
/// Pivots allowed to one one-sided derivative before it declines. Bland's rule makes the
/// dual simplex finite; this only bounds the time on a pathological model.
constexpr int kMaxParametricPivots = 1000;

/// A value that may be infinite: ranges are +inf, one-sided derivatives either sign.
struct Ext {
  int inf = 0;  ///< 0 finite, +1 or -1 infinite
  Rational v;
};
Ext finite(Rational r) {
  return {0, std::move(r)};
}
Ext infinite(int sign) {
  return {sign, Rational(0)};
}
/// The smaller of two values, neither of them -inf.
Ext smaller(const Ext& a, const Ext& b) {
  if (a.inf > 0) return b;
  if (b.inf > 0) return a;
  return a.v <= b.v ? a : b;
}
Ext times(int sign, const Ext& e) {
  return {sign * e.inf, sign < 0 ? -e.v : e.v};
}
std::string text(const Ext& e) {
  if (e.inf != 0) return e.inf > 0 ? "inf" : "-inf";
  return e.v.to_string();
}
double nearest(const Ext& e) {
  return e.inf != 0 ? e.inf * kInfinity : e.v.to_double();
}

/// The float report agrees with the exact value to tol::kSensitivityAgreement, relative to
/// max(1, |exact|). Compared in exact arithmetic, so the comparison itself cannot round.
bool agrees(double reported, const Ext& exact) {
  if (exact.inf != 0) return std::isinf(reported) && (reported > 0) == (exact.inf > 0);
  if (!std::isfinite(reported)) return false;
  const Rational diff = Rational::from_double(reported) - exact.v;
  const Rational size = exact.v.sign() < 0 ? -exact.v : exact.v;
  const Rational scale = size > Rational(1) ? size : Rational(1);
  const Rational allowed = scale * Rational::from_double(tol::kSensitivityAgreement);
  return (diff.sign() < 0 ? -diff : diff) <= allowed;
}

/// d v / d t at t = 0 from the side `direction` (+1 right, -1 left), minimise space: the
/// lexicographic dual simplex on row `row`'s bounds shifted by t, Bland's rule throughout.
/// nullopt when it does not settle within kMaxParametricPivots or meets a singular basis.
std::optional<Ext> one_sided(const Problem& problem, const Basis& start, Index row,
                             int direction, const Deadline& deadline) {
  const auto m = static_cast<Sz>(problem.m);
  const auto logical = static_cast<Sz>(problem.n + row);
  std::optional<Basis> own;  // copied on the first pivot: most rows need none
  for (int pivot = 0; pivot < kMaxParametricPivots; ++pivot) {
    const Basis& basis = own.has_value() ? *own : start;
    // First-order motion of each basic variable relative to its own (shifting) bounds.
    const bool shifted = basis.status[logical] != Nb::kBasic;
    const std::vector<Rational> motion =
        shifted ? basis.column_of_inverse(problem, row) : std::vector<Rational>(m, Rational(0));
    Sz leave_p = m;
    Nb leave_to = Nb::kLower;
    Index leave_k = -1;
    for (Sz p = 0; p < m; ++p) {
      const auto k = static_cast<Sz>(basis.basic[p]);
      Rational slope = motion[p] * Rational(direction);
      if (k == logical) slope -= Rational(direction);
      const Rational& x = basis.value[k];
      Nb to = Nb::kBasic;
      if (slope.sign() < 0 && problem.has_lower[k] && x == problem.lower[k]) to = Nb::kLower;
      if (slope.sign() > 0 && problem.has_upper[k] && x == problem.upper[k]) to = Nb::kUpper;
      if (to != Nb::kBasic && (leave_k < 0 || static_cast<Index>(k) < leave_k)) {
        leave_p = p;
        leave_to = to;
        leave_k = static_cast<Index>(k);
      }
    }
    if (leave_p == m) return finite(basis.y[static_cast<Sz>(row)]);
    const bool must_rise = leave_to == Nb::kLower;
    Index entering = -1;
    Rational best(0);
    std::vector<Rational> row_of_inverse;
    const std::vector<std::pair<Sz, Rational>> tableau =
        basis.tableau_row(problem, leave_p, &row_of_inverse);
    for (const auto& [k, a] : tableau) {
      const Nb st = basis.status[k];
      if (problem.fixed(static_cast<Index>(k))) continue;
      // x_p = ... - a x_k: raising x_k moves x_p by -a, lowering it by +a.
      const bool can_rise = st == Nb::kLower || st == Nb::kFree;
      const bool can_fall = st == Nb::kUpper || st == Nb::kFree;
      const bool helps = (can_rise && ((a.sign() < 0) == must_rise)) ||
                         (can_fall && ((a.sign() > 0) == must_rise));
      if (!helps) continue;
      Rational ratio = basis.d[k] / a;
      if (ratio.sign() < 0) ratio = -ratio;
      if (entering < 0 || ratio < best) {
        best = ratio;
        entering = static_cast<Index>(k);
      }
    }
    if (entering < 0) return infinite(direction);  // no feasible point on that side
    if (!own.has_value()) own = start;
    if (!own->pivot_degenerate(problem, leave_p, entering, leave_to, tableau, row_of_inverse,
                               deadline)) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

SensitivityResult certify_sensitivity(const Model& model, const Solution& solution,
                                      double seconds) {
  SensitivityResult result;
  const Deadline deadline(seconds);
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (solution.status != SolveStatus::kOptimal) {
    result.message = "the status is not optimal";
    return result;
  }
  if (model.hessian.num_nonzeros() > 0 || model.num_integer_columns() > 0) {
    result.message = "certified sensitivity covers plain LPs only";
    return result;
  }
  if (static_cast<Index>(solution.col_status.size()) != n ||
      static_cast<Index>(solution.row_status.size()) != m ||
      static_cast<Index>(solution.col_ranging_lower.size()) != n ||
      static_cast<Index>(solution.row_ranging_lower.size()) != m) {
    result.message = "no basis or no floating-point ranging to certify";
    return result;
  }
  try {
    const Problem problem(model);
    Basis basis;
    const LoadResult loaded = load_basis(model, solution, problem, deadline, &basis);
    if (!loaded.message.empty()) {
      result.verdict = loaded.verdict;
      result.message = loaded.message;
      return result;
    }
    const std::string why = not_optimal(problem, basis);
    if (!why.empty()) {
      result.verdict = ExactVerdict::kFailed;
      result.message = "the reported basis is not exactly optimal: " + why;
      return result;
    }
    const int sense = problem.sense;
    const auto sm = static_cast<Sz>(m);
    std::vector<Sz> position(static_cast<Sz>(n + m), sm);
    for (Sz p = 0; p < sm; ++p) position[static_cast<Sz>(basis.basic[p])] = p;

    // --- Columns: reduced cost and cost range. ---
    for (Index j = 0; j < n; ++j) {
      deadline.check();
      const auto u = static_cast<Sz>(j);
      Ext lo = infinite(1);
      Ext hi = infinite(1);
      const Nb st = basis.status[u];
      if (st != Nb::kBasic) {
        if (problem.fixed(j)) {
          // never enters: both sides unbounded
        } else if (st == Nb::kLower) {
          lo = finite(basis.d[u]);
        } else if (st == Nb::kUpper) {
          hi = finite(-basis.d[u]);
        } else {
          lo = hi = finite(Rational(0));
        }
      } else {
        for (const auto& [k, a] : basis.tableau_row(problem, position[u])) {
          const Nb sk = basis.status[k];
          if (problem.fixed(static_cast<Index>(k))) continue;
          const Rational& dk = basis.d[k];
          if (sk == Nb::kFree) {
            lo = hi = finite(Rational(0));
          } else if (sk == Nb::kLower) {
            if (a.sign() > 0) {
              hi = smaller(hi, finite(dk / a));
            } else {
              lo = smaller(lo, finite(dk / -a));
            }
          } else if (a.sign() > 0) {
            lo = smaller(lo, finite(-dk / a));
          } else {
            hi = smaller(hi, finite(-dk / -a));
          }
        }
        if (lo.inf == 0 && lo.v.sign() < 0) lo = finite(Rational(0));
        if (hi.inf == 0 && hi.v.sign() < 0) hi = finite(Rational(0));
      }
      if (sense < 0) std::swap(lo, hi);
      Solution::ExactSensitivityEntry entry;
      const Ext value = finite(sense < 0 ? -basis.d[u] : basis.d[u]);
      entry.value = text(value);
      entry.range_lower = text(lo);
      entry.range_upper = text(hi);
      entry.certified = agrees(solution.col_dual[u], value) &&
                        agrees(solution.col_ranging_lower[u], lo) &&
                        agrees(solution.col_ranging_upper[u], hi);
      result.columns.push_back(std::move(entry));
    }

    // --- Rows: dual, RHS range and the shadow price interval. ---
    for (Index i = 0; i < m; ++i) {
      deadline.check();
      const auto r = static_cast<Sz>(i);
      Ext down = infinite(1);
      Ext up = infinite(1);
      const std::vector<Rational> column = basis.column_of_inverse(problem, i);
      for (Sz p = 0; p < sm; ++p) {
        const Rational& v = column[p];
        if (v.is_zero()) continue;
        const auto k = static_cast<Sz>(basis.basic[p]);
        const Rational& x = basis.value[k];
        const std::optional<Rational> gap_down =
            problem.has_lower[k] ? std::optional<Rational>(x - problem.lower[k]) : std::nullopt;
        const std::optional<Rational> gap_up =
            problem.has_upper[k] ? std::optional<Rational>(problem.upper[k] - x) : std::nullopt;
        const std::optional<Rational>& limits_up = v.sign() > 0 ? gap_up : gap_down;
        const std::optional<Rational>& limits_down = v.sign() > 0 ? gap_down : gap_up;
        const Rational size = v.sign() > 0 ? v : -v;
        if (limits_up.has_value()) up = smaller(up, finite(*limits_up / size));
        if (limits_down.has_value()) down = smaller(down, finite(*limits_down / size));
      }
      if (down.inf == 0 && down.v.sign() < 0) down = finite(Rational(0));
      if (up.inf == 0 && up.v.sign() < 0) up = finite(Rational(0));
      const Ext dual = finite(basis.y[r]);
      const bool right_here = up.inf != 0 || up.v.sign() > 0;
      const bool left_here = down.inf != 0 || down.v.sign() > 0;
      const std::optional<Ext> right =
          right_here ? dual : one_sided(problem, basis, i, 1, deadline);
      const std::optional<Ext> left =
          left_here ? dual : one_sided(problem, basis, i, -1, deadline);
      if (!right.has_value() || !left.has_value()) {
        result.verdict = ExactVerdict::kDeclined;
        result.message =
            fmt::format("row {}: the parametric dual simplex did not settle within {} pivots",
                        i, kMaxParametricPivots);
        return result;
      }
      Solution::ExactSensitivityEntry entry;
      const Ext model_dual = times(sense, dual);
      const Ext model_left = times(sense, *left);
      const Ext model_right = times(sense, *right);
      entry.value = text(model_dual);
      entry.range_lower = text(down);
      entry.range_upper = text(up);
      entry.shadow_left = text(model_left);
      entry.shadow_right = text(model_right);
      entry.shadow_left_value = nearest(model_left);
      entry.shadow_right_value = nearest(model_right);
      entry.certified = agrees(solution.row_dual[r], model_dual) &&
                        agrees(solution.row_ranging_lower[r], down) &&
                        agrees(solution.row_ranging_upper[r], up);
      result.rows.push_back(std::move(entry));
    }
    result.verdict = ExactVerdict::kVerified;
    return result;
  } catch (const ExactBudgetExceeded&) {
    result.verdict = ExactVerdict::kDeclined;
    result.message = fmt::format(
        "the exact derivation ran past its {} s budget (option exact_seconds)", seconds);
    result.columns.clear();
    result.rows.clear();
    return result;
  } catch (const RationalOverflow&) {
    result.verdict = ExactVerdict::kDeclined;
    result.message = "no exact rational result: a non-finite value in the data";
    result.columns.clear();
    result.rows.clear();
    return result;
  }
}

}  // namespace sankhya::exact

namespace sankhya::exact {

void apply_certified_sensitivity(const Model& model, Solution* solution, Logger& logger,
                                 double seconds) {
  SensitivityResult sensitivity = certify_sensitivity(model, *solution, seconds);
  switch (sensitivity.verdict) {
    case ExactVerdict::kVerified: {
      solution->sensitivity_status = Solution::ExactVerification::kVerified;
      std::size_t certified = 0;
      std::size_t kinked = 0;
      for (const auto& entry : sensitivity.columns) certified += entry.certified ? 1U : 0U;
      for (const auto& entry : sensitivity.rows) {
        certified += entry.certified ? 1U : 0U;
        kinked += entry.shadow_left != entry.shadow_right ? 1U : 0U;
      }
      logger.info(
          "Certified sensitivity (#757): {} of {} reported duals and ranges agree with the "
          "exact values, the rest are corrected in the .sol; {} row(s) with a two-sided "
          "shadow price",
          certified, sensitivity.columns.size() + sensitivity.rows.size(), kinked);
      solution->exact_col_sensitivity = std::move(sensitivity.columns);
      solution->exact_row_sensitivity = std::move(sensitivity.rows);
      break;
    }
    case ExactVerdict::kDeclined:
      solution->sensitivity_status = Solution::ExactVerification::kDeclined;
      solution->sensitivity_message = sensitivity.message;
      logger.info("Certified sensitivity declined: {}", sensitivity.message);
      break;
    case ExactVerdict::kFailed:
      solution->sensitivity_status = Solution::ExactVerification::kFailed;
      solution->sensitivity_message = sensitivity.message;
      logger.warning("Certified sensitivity FAILED: {}", sensitivity.message);
      break;
  }
}

}  // namespace sankhya::exact
