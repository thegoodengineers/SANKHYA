// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certified shadow prices and sensitivity ranges (#757). See
// exact_sensitivity.hpp for the citations and the definitions.
//
// Dense: the basis inverse is formed once per basis by Gauss-Jordan elimination over exact
// rationals, which is O(m^3) and so capped at the same row count as the basis check
// (exact_verify.cpp). Above it the certification declines; the independent verifier
// (tools/verify_solution_sensitivity.py, sparse, arbitrary precision) covers larger models.

#include "exact/exact_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "exact/rational.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;
using Matrix = std::vector<std::vector<Rational>>;

constexpr Index kMaxRowsForExactSensitivity = 300;
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

enum class Nb : unsigned char { kBasic, kLower, kUpper, kFree };

/// The model as [A | -I]: variables 0..n-1 structural, n+i the logical of row i, whose
/// value is the row activity and whose bounds are the row's. Costs in minimise space.
struct Problem {
  Index n = 0;
  Index m = 0;
  int sense = 1;
  std::vector<Rational> cost, lower, upper;
  std::vector<char> has_lower, has_upper;
  std::vector<std::vector<std::pair<Index, Rational>>> columns;

  explicit Problem(const Model& model)
      : n(model.num_cols()), m(model.num_rows()), sense(model.sense_multiplier() < 0 ? -1 : 1) {
    const auto total = static_cast<Sz>(n + m);
    cost.resize(total);
    lower.resize(total);
    upper.resize(total);
    has_lower.resize(total);
    has_upper.resize(total);
    columns.resize(total);
    for (Index k = 0; k < n + m; ++k) {
      const auto u = static_cast<Sz>(k);
      const bool structural = k < n;
      const auto r = static_cast<Sz>(k - n);
      const double lo = structural ? model.col_lower[u] : model.row_lower[r];
      const double hi = structural ? model.col_upper[u] : model.row_upper[r];
      cost[u] = structural ? Rational::from_double(sense * model.col_cost[u]) : Rational(0);
      has_lower[u] = lo > -kInfinity;
      has_upper[u] = hi < kInfinity;
      if (has_lower[u]) lower[u] = Rational::from_double(lo);
      if (has_upper[u]) upper[u] = Rational::from_double(hi);
      if (structural) {
        const ColumnView view = model.matrix.column(k);
        for (Index q = 0; q < view.size; ++q) {
          columns[u].emplace_back(view.rows[q], Rational::from_double(view.values[q]));
        }
      } else {
        columns[u].emplace_back(k - n, Rational(-1));
      }
    }
  }
  [[nodiscard]] bool fixed(Index k) const {
    const auto u = static_cast<Sz>(k);
    return has_lower[u] && has_upper[u] && lower[u] == upper[u];
  }
};

/// Gauss-Jordan inverse; nullopt when exactly singular.
std::optional<Matrix> invert(Matrix a) {
  const Sz m = a.size();
  Matrix inv(m, std::vector<Rational>(m, Rational(0)));
  for (Sz i = 0; i < m; ++i) inv[i][i] = Rational(1);
  for (Sz col = 0; col < m; ++col) {
    Sz pivot = m;
    for (Sz r = col; r < m; ++r) {
      if (!a[r][col].is_zero()) {
        pivot = r;
        break;
      }
    }
    if (pivot == m) return std::nullopt;
    std::swap(a[pivot], a[col]);
    std::swap(inv[pivot], inv[col]);
    const Rational scale = Rational(1) / a[col][col];
    for (Sz c = 0; c < m; ++c) {
      if (!a[col][c].is_zero()) a[col][c] = a[col][c] * scale;
      if (!inv[col][c].is_zero()) inv[col][c] = inv[col][c] * scale;
    }
    for (Sz r = 0; r < m; ++r) {
      if (r == col || a[r][col].is_zero()) continue;
      const Rational factor = a[r][col];
      for (Sz c = 0; c < m; ++c) {
        if (!a[col][c].is_zero()) a[r][c] = a[r][c] - factor * a[col][c];
        if (!inv[col][c].is_zero()) inv[r][c] = inv[r][c] - factor * inv[col][c];
      }
    }
  }
  return inv;
}

/// One basis: its inverse, the point at t = 0, the duals and reduced costs (minimise space).
struct Basis {
  std::vector<Index> basic;  ///< position -> variable
  std::vector<Nb> status;    ///< per variable
  Matrix inverse;            ///< B^{-1}, row p for basis position p
  std::vector<Rational> value, y, d;

  /// False when the basis is exactly singular.
  bool build(const Problem& problem) {
    const auto m = static_cast<Sz>(problem.m);
    const auto total = static_cast<Sz>(problem.n + problem.m);
    Matrix b(m, std::vector<Rational>(m, Rational(0)));
    for (Sz p = 0; p < m; ++p) {
      for (const auto& [i, a] : problem.columns[static_cast<Sz>(basic[p])]) {
        b[static_cast<Sz>(i)][p] = a;
      }
    }
    std::optional<Matrix> inv = invert(std::move(b));
    if (!inv.has_value()) return false;
    inverse = std::move(*inv);
    value.assign(total, Rational(0));
    std::vector<Rational> rhs(m, Rational(0));
    for (Sz k = 0; k < total; ++k) {
      switch (status[k]) {
        case Nb::kLower: value[k] = problem.lower[k]; break;
        case Nb::kUpper: value[k] = problem.upper[k]; break;
        case Nb::kFree:
        case Nb::kBasic: continue;
      }
      if (value[k].is_zero()) continue;
      for (const auto& [i, a] : problem.columns[k]) {
        rhs[static_cast<Sz>(i)] = rhs[static_cast<Sz>(i)] - a * value[k];
      }
    }
    for (Sz p = 0; p < m; ++p) {
      Rational x(0);
      for (Sz i = 0; i < m; ++i) {
        if (!inverse[p][i].is_zero() && !rhs[i].is_zero()) x += inverse[p][i] * rhs[i];
      }
      value[static_cast<Sz>(basic[p])] = x;
    }
    y.assign(m, Rational(0));
    for (Sz p = 0; p < m; ++p) {
      const Rational& c = problem.cost[static_cast<Sz>(basic[p])];
      if (c.is_zero()) continue;
      for (Sz i = 0; i < m; ++i) {
        if (!inverse[p][i].is_zero()) y[i] += c * inverse[p][i];
      }
    }
    d.assign(total, Rational(0));
    for (Sz k = 0; k < total; ++k) {
      if (status[k] == Nb::kBasic) continue;
      Rational dk = problem.cost[k];
      for (const auto& [i, a] : problem.columns[k]) dk -= y[static_cast<Sz>(i)] * a;
      d[k] = dk;
    }
    return true;
  }

  /// Row p of B^{-1} [A | -I] at variable k.
  [[nodiscard]] Rational alpha(const Problem& problem, Sz p, Sz k) const {
    Rational sum(0);
    for (const auto& [i, a] : problem.columns[k]) {
      const Rational& w = inverse[p][static_cast<Sz>(i)];
      if (!w.is_zero()) sum += w * a;
    }
    return sum;
  }
};

/// Why the basis is not exactly optimal, or empty.
std::string not_optimal(const Problem& problem, const Basis& basis) {
  for (Sz k = 0; k < basis.status.size(); ++k) {
    const Rational& x = basis.value[k];
    if (basis.status[k] == Nb::kBasic) {
      if ((problem.has_lower[k] && x < problem.lower[k]) ||
          (problem.has_upper[k] && x > problem.upper[k])) {
        return fmt::format("variable {} is basic outside its bounds, exactly", k);
      }
      continue;
    }
    if (problem.fixed(static_cast<Index>(k))) continue;
    const int s = basis.d[k].sign();
    if ((basis.status[k] == Nb::kLower && s < 0) || (basis.status[k] == Nb::kUpper && s > 0) ||
        (basis.status[k] == Nb::kFree && s != 0)) {
      return fmt::format("variable {} has an exact reduced cost of the wrong sign", k);
    }
  }
  return {};
}

/// d v / d t at t = 0 from the side `direction` (+1 right, -1 left), minimise space: the
/// lexicographic dual simplex on row `row`'s bounds shifted by t, Bland's rule throughout.
/// nullopt when it does not settle within kMaxParametricPivots or meets a singular basis.
std::optional<Ext> one_sided(const Problem& problem, Basis basis, Index row, int direction) {
  const auto m = static_cast<Sz>(problem.m);
  const auto logical = static_cast<Sz>(problem.n + row);
  for (int pivot = 0; pivot < kMaxParametricPivots; ++pivot) {
    // First-order motion of each basic variable relative to its own (shifting) bounds.
    const bool shifted = basis.status[logical] != Nb::kBasic;
    Sz leave_p = m;
    Nb leave_to = Nb::kLower;
    Index leave_k = -1;
    for (Sz p = 0; p < m; ++p) {
      const auto k = static_cast<Sz>(basis.basic[p]);
      Rational slope =
          shifted ? basis.inverse[p][static_cast<Sz>(row)] * Rational(direction) : Rational(0);
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
    for (Sz k = 0; k < basis.status.size(); ++k) {
      const Nb st = basis.status[k];
      if (st == Nb::kBasic || problem.fixed(static_cast<Index>(k))) continue;
      const Rational a = basis.alpha(problem, leave_p, k);
      if (a.is_zero()) continue;
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
    basis.basic[leave_p] = entering;
    basis.status[static_cast<Sz>(entering)] = Nb::kBasic;
    basis.status[static_cast<Sz>(leave_k)] = leave_to;
    if (!basis.build(problem)) return std::nullopt;
  }
  return std::nullopt;
}

Nb nb_of(BasisStatus status) {
  switch (status) {
    case BasisStatus::kAtLower:
    case BasisStatus::kFixed: return Nb::kLower;
    case BasisStatus::kAtUpper: return Nb::kUpper;
    case BasisStatus::kNonbasicFree: return Nb::kFree;
    default: return Nb::kBasic;
  }
}

}  // namespace

SensitivityResult certify_sensitivity(const Model& model, const Solution& solution) {
  SensitivityResult result;
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
  if (m > kMaxRowsForExactSensitivity) {
    result.message = fmt::format("{} rows exceeds the {}-row cap for dense exact inversion", m,
                                 kMaxRowsForExactSensitivity);
    return result;
  }
  try {
    const Problem problem(model);
    Basis basis;
    basis.status.resize(static_cast<Sz>(n + m));
    for (Index k = 0; k < n + m; ++k) {
      const auto u = static_cast<Sz>(k);
      const bool structural = k < n;
      const auto r = static_cast<Sz>(k - n);
      const BasisStatus reported = structural ? solution.col_status[u] : solution.row_status[r];
      if (reported == BasisStatus::kBasic) {
        basis.basic.push_back(k);
        basis.status[u] = Nb::kBasic;
        continue;
      }
      const double lo = structural ? model.col_lower[u] : model.row_lower[r];
      const double hi = structural ? model.col_upper[u] : model.row_upper[r];
      const double at = structural ? solution.col_value[u] : solution.row_activity[r];
      const BasisStatus resolved = resolved_status(reported, lo, hi, at);
      const Nb nb = nb_of(resolved);
      if (resolved == BasisStatus::kUnknown || (nb == Nb::kLower && !(lo > -kInfinity)) ||
          (nb == Nb::kUpper && !(hi < kInfinity))) {
        result.verdict = ExactVerdict::kFailed;
        result.message = fmt::format("variable {} is nonbasic at no finite bound", k);
        return result;
      }
      basis.status[u] = nb;
    }
    if (static_cast<Index>(basis.basic.size()) != m) {
      result.message = fmt::format("the reported basis has {} basic variable(s) for {} row(s)",
                                   basis.basic.size(), m);
      return result;
    }
    if (!basis.build(problem)) {
      result.verdict = ExactVerdict::kFailed;
      result.message = "the reported basis is exactly singular";
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
        for (Sz k = 0; k < basis.status.size(); ++k) {
          const Nb sk = basis.status[k];
          if (sk == Nb::kBasic || problem.fixed(static_cast<Index>(k))) continue;
          const Rational a = basis.alpha(problem, position[u], k);
          if (a.is_zero()) continue;
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
      const auto r = static_cast<Sz>(i);
      Ext down = infinite(1);
      Ext up = infinite(1);
      for (Sz p = 0; p < sm; ++p) {
        const Rational& v = basis.inverse[p][r];
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
      const std::optional<Ext> right = right_here ? dual : one_sided(problem, basis, i, 1);
      const std::optional<Ext> left = left_here ? dual : one_sided(problem, basis, i, -1);
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

void apply_certified_sensitivity(const Model& model, Solution* solution, Logger& logger) {
  SensitivityResult sensitivity = certify_sensitivity(model, *solution);
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
