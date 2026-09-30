// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an LP basis in exact rational arithmetic (#757). See exact_basis.hpp.

#include "exact/exact_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/types.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;

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

Problem::Problem(const Model& model)
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
  scale.resize(total);
  scaled_columns.resize(total);
  scaled_rows.resize(static_cast<Sz>(m));
  for (Sz k = 0; k < total; ++k) {
    std::vector<Rational> values;
    for (const auto& entry : columns[k]) values.push_back(entry.second);
    std::vector<BigInt> integers;
    scale[k] = common_denominator(values, &integers);
    for (Sz q = 0; q < integers.size(); ++q) {
      const Index i = columns[k][q].first;
      scaled_columns[k].emplace_back(i, integers[q]);
      scaled_rows[static_cast<Sz>(i)].emplace_back(static_cast<Index>(k), integers[q]);
    }
  }
}

bool Basis::build(const Problem& problem, const Deadline& deadline) {
  std::vector<const RationalColumn*> columns;
  columns.reserve(basic.size());
  for (const Index k : basic) columns.push_back(&problem.columns[static_cast<Sz>(k)]);
  if (!lu.factorize(columns, problem.m, deadline)) return false;
  refresh(problem);
  return true;
}

std::vector<Rational> Basis::column_of_inverse(const Problem& problem, Index row) const {
  std::vector<Rational> v(static_cast<Sz>(problem.m), Rational(0));
  v[static_cast<Sz>(row)] = Rational(1);
  lu.solve(v);
  return v;
}

std::vector<Rational> Basis::tableau_column(const Problem& problem, Index k) const {
  std::vector<Rational> alpha(static_cast<Sz>(problem.m), Rational(0));
  for (const auto& [i, a] : problem.columns[static_cast<Sz>(k)]) {
    alpha[static_cast<Sz>(i)] = a;
  }
  lu.solve(alpha);
  return alpha;
}

std::vector<std::pair<Sz, Rational>> Basis::tableau_row(
    const Problem& problem, Sz p, std::vector<Rational>* row_of_inverse) const {
  std::vector<Rational> w(static_cast<Sz>(problem.m), Rational(0));
  w[p] = Rational(1);
  lu.solve_transpose(w);
  std::vector<BigInt> numerators;
  const BigInt lcd = common_denominator(w, &numerators);
  std::vector<BigInt> sum(status.size(), BigInt(0));
  std::vector<Sz> touched;
  for (Sz i = 0; i < w.size(); ++i) {
    if (numerators[i].is_zero()) continue;
    for (const auto& [k, a] : problem.scaled_rows[i]) {
      const auto sk = static_cast<Sz>(k);
      if (status[sk] == Nb::kBasic) continue;
      if (sum[sk].is_zero()) touched.push_back(sk);
      sum[sk] = sum[sk] + numerators[i] * a;
    }
  }
  std::sort(touched.begin(), touched.end());
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
  std::vector<std::pair<Sz, Rational>> out;
  for (const Sz k : touched) {
    if (!sum[k].is_zero()) {
      out.emplace_back(k, Rational(std::move(sum[k]), lcd * problem.scale[k]));
    }
  }
  if (row_of_inverse != nullptr) *row_of_inverse = std::move(w);
  return out;
}

bool Basis::pivot(const Problem& problem, Sz position, Index entering, Nb to,
                  const std::vector<Rational>& alpha, const Deadline& deadline) {
  if (alpha[position].is_zero()) return false;
  status[static_cast<Sz>(basic[position])] = to;
  status[static_cast<Sz>(entering)] = Nb::kBasic;
  basic[position] = entering;
  if (lu.num_updates() + 1 >= kRefactorEvery) return build(problem, deadline);
  lu.update(static_cast<Index>(position), alpha);
  refresh(problem);
  return true;
}

bool Basis::pivot_degenerate(const Problem& problem, Sz position, Index entering, Nb to,
                             const std::vector<std::pair<Sz, Rational>>& row,
                             const std::vector<Rational>& row_of_inverse,
                             const Deadline& deadline) {
  const auto q = static_cast<Sz>(entering);
  const auto leaving = static_cast<Sz>(basic[position]);
  Rational alpha_q(0);
  for (const auto& [k, a] : row) {
    if (k == q) alpha_q = a;
  }
  if (alpha_q.is_zero()) return false;
  const Rational theta = d[q] / alpha_q;
  const std::vector<Rational> alpha = tableau_column(problem, entering);
  if (alpha[position].is_zero()) return false;
  status[leaving] = to;
  status[q] = Nb::kBasic;
  basic[position] = entering;
  if (lu.num_updates() + 1 >= kRefactorEvery) return build(problem, deadline);
  lu.update(static_cast<Index>(position), alpha);
  if (!theta.is_zero()) {
    for (const auto& [k, a] : row) d[k] -= theta * a;
    for (Sz i = 0; i < y.size(); ++i) {
      if (!row_of_inverse[i].is_zero()) y[i] += theta * row_of_inverse[i];
    }
  }
  d[q] = Rational(0);
  d[leaving] = -theta;
  return true;
}

void Basis::refresh(const Problem& problem) {
  const auto m = static_cast<Sz>(problem.m);
  const auto total = static_cast<Sz>(problem.n + problem.m);
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
      rhs[static_cast<Sz>(i)] -= a * value[k];
    }
  }
  lu.solve(rhs);
  for (Sz p = 0; p < m; ++p) value[static_cast<Sz>(basic[p])] = std::move(rhs[p]);
  y.assign(m, Rational(0));
  for (Sz p = 0; p < m; ++p) y[p] = problem.cost[static_cast<Sz>(basic[p])];
  lu.solve_transpose(y);
  std::vector<BigInt> numerators;
  const BigInt lcd = common_denominator(y, &numerators);
  d.assign(total, Rational(0));
  for (Sz k = 0; k < total; ++k) {
    if (status[k] == Nb::kBasic) continue;
    BigInt sum(0);
    for (const auto& [i, a] : problem.scaled_columns[k]) {
      const BigInt& yi = numerators[static_cast<Sz>(i)];
      if (!yi.is_zero()) sum = sum + yi * a;
    }
    d[k] = sum.is_zero() ? problem.cost[k]
                         : problem.cost[k] - Rational(std::move(sum), lcd * problem.scale[k]);
  }
}

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

LoadResult load_basis(const Model& model, const Solution& solution, const Problem& problem,
                      const Deadline& deadline, Basis* basis) {
  LoadResult result;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  basis->basic.clear();
  basis->status.assign(static_cast<Sz>(n + m), Nb::kBasic);
  for (Index k = 0; k < n + m; ++k) {
    const auto u = static_cast<Sz>(k);
    const bool structural = k < n;
    const auto r = static_cast<Sz>(k - n);
    const BasisStatus reported = structural ? solution.col_status[u] : solution.row_status[r];
    if (reported == BasisStatus::kBasic) {
      basis->basic.push_back(k);
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
    basis->status[u] = nb;
  }
  if (static_cast<Index>(basis->basic.size()) != m) {
    result.message = fmt::format("the reported basis has {} basic variable(s) for {} row(s)",
                                 basis->basic.size(), m);
    return result;
  }
  if (!basis->build(problem, deadline)) {
    result.verdict = ExactVerdict::kFailed;
    result.message = "the reported basis is exactly singular";
    return result;
  }
  return result;
}

}  // namespace sankhya::exact
