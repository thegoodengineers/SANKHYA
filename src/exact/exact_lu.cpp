// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse exact LU of an LP basis (#757). See exact_lu.hpp for the citations and why
// the pivot order is the float LU's.
//
// THE ALGEBRA, written out because a sign or an order error here does not crash. Step k
// pivots on (r_k, p_k) and subtracts l_ik times row r_k from every other active row i with a
// nonzero in position p_k. With M_k that transformation, E = M_{m-1} ... M_0 and E B = U,
// where U[r_k][p_l] is nonzero only for l >= k.
//
//   FTRAN  B x = b:    U x = E b. Apply M_0 first (b_i -= l_ik b_{r_k}, k ascending), then
//                      back-substitute k descending, pushing x_{p_k} into the rows above.
//   BTRAN  B^T y = c:  U^T w = c, k ascending (w_{r_k} from the column of U at p_k), then
//                      y = M_0^T ... M_{m-1}^T w, k descending: w_{r_k} -= sum_i l_ik w_i.
//
// With etas E_1 .. E_t on top (B_t = B E_1 ... E_t), FTRAN applies them oldest first after
// the base solve and BTRAN newest first before it.

#include "exact/exact_lu.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "la/lu.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;

/// The float LU's pivot sequence for this matrix, as (row, position) per step, or empty when
/// the float factorisation declares the matrix singular to working precision.
std::vector<std::pair<Index, Index>> float_order(
    const std::vector<const RationalColumn*>& columns, Index m) {
  const auto sm = static_cast<Sz>(m);
  std::vector<std::vector<Index>> rows(sm);
  std::vector<std::vector<double>> values(sm);
  std::vector<LuColumn> lu_columns(sm);
  for (Sz p = 0; p < sm; ++p) {
    for (const auto& [i, v] : *columns[p]) {
      rows[p].push_back(i);
      values[p].push_back(v.to_double());
    }
    lu_columns[p] = {rows[p].data(), values[p].data(), static_cast<Index>(rows[p].size())};
  }
  SparseLu lu;
  if (!lu.factorize(lu_columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold)) return {};
  std::vector<std::pair<Index, Index>> order;
  order.reserve(sm);
  for (Sz k = 0; k < sm; ++k) order.emplace_back(lu.pivot_rows()[k], lu.pivot_columns()[k]);
  return order;
}

}  // namespace

BigInt common_denominator(const std::vector<Rational>& v, std::vector<BigInt>* numerators) {
  BigInt lcd(1);
  for (const Rational& e : v) {
    if (e.is_zero() || e.denominator() == lcd || e.denominator().is_one()) continue;
    const BigInt g = BigInt::gcd(lcd, e.denominator());
    lcd = lcd * (g.is_one() ? e.denominator() : e.denominator() / g);
  }
  numerators->assign(v.size(), BigInt(0));
  for (Sz i = 0; i < v.size(); ++i) {
    const Rational& e = v[i];
    if (e.is_zero()) continue;
    (*numerators)[i] =
        e.denominator() == lcd ? e.numerator() : e.numerator() * (lcd / e.denominator());
  }
  return lcd;
}

bool ExactLu::factorize(const std::vector<const RationalColumn*>& columns, Index m,
                        const Deadline& deadline) {
  m_ = m;
  const auto sm = static_cast<Sz>(m);
  pivot_row_.assign(sm, -1);
  pivot_position_.assign(sm, -1);
  pivot_value_.assign(sm, Rational(0));
  lower_.assign(sm, {});
  upper_.assign(sm, {});
  eta_position_.clear();
  eta_.clear();
  hinted_steps_ = 0;
  factor_nonzeros_ = 0;

  // The active submatrix, row-wise, and for each position the rows that may hold a nonzero
  // in it (lazily: a row whose entry cancelled, or that has pivoted, is skipped on use).
  std::vector<RationalColumn> row(sm);
  std::vector<std::vector<Index>> rows_of(sm);
  for (Sz p = 0; p < sm; ++p) {
    for (const auto& [i, v] : *columns[p]) {
      if (v.is_zero()) continue;
      row[static_cast<Sz>(i)].emplace_back(static_cast<Index>(p), v);
      rows_of[p].push_back(i);
    }
  }

  std::vector<std::pair<Index, Index>> order = float_order(columns, m);
  if (order.empty()) {
    // No float order: positions sparsest first. Correct, only slower.
    std::vector<Index> by_count(sm);
    for (Sz p = 0; p < sm; ++p) by_count[p] = static_cast<Index>(p);
    std::stable_sort(by_count.begin(), by_count.end(), [&](Index a, Index b) {
      return rows_of[static_cast<Sz>(a)].size() < rows_of[static_cast<Sz>(b)].size();
    });
    for (const Index p : by_count) order.emplace_back(-1, p);
  }

  std::vector<char> row_done(sm, 0);
  std::vector<Index> step_of_position(sm, -1);
  std::vector<RationalColumn> u_row(sm);  // per step: U's row, by position
  // Scatter workspace for row updates, by position.
  std::vector<Rational> work(sm, Rational(0));
  std::vector<char> in_work(sm, 0);

  auto value_at = [&](Sz i, Index p) -> const Rational* {
    for (const auto& [q, v] : row[i]) {
      if (q == p) return &v;
    }
    return nullptr;
  };

  for (Sz k = 0; k < sm; ++k) {
    deadline.check();
    const Index p = order[k].second;
    const auto sp = static_cast<Sz>(p);
    // The pivot row: the float order's if it is exactly nonzero there, else the active row
    // with a nonzero in this position and the fewest entries.
    Index r = -1;
    const Index hint = order[k].first;
    if (hint >= 0 && !row_done[static_cast<Sz>(hint)] && value_at(static_cast<Sz>(hint), p)) {
      r = hint;
      ++hinted_steps_;
    } else {
      for (const Index i : rows_of[sp]) {
        const auto si = static_cast<Sz>(i);
        if (row_done[si] || value_at(si, p) == nullptr) continue;
        if (r < 0 || row[si].size() < row[static_cast<Sz>(r)].size()) r = i;
      }
    }
    if (r < 0) return false;  // no active nonzero in this column: exactly dependent
    const auto sr = static_cast<Sz>(r);
    const Rational pivot = *value_at(sr, p);
    pivot_row_[k] = r;
    pivot_position_[k] = p;
    pivot_value_[k] = pivot;
    step_of_position[sp] = static_cast<Index>(k);
    row_done[sr] = 1;
    for (const auto& [q, v] : row[sr]) {
      if (q != p) u_row[k].emplace_back(q, v);
    }

    // Eliminate position p from every other active row that holds it.
    std::vector<Index> targets = rows_of[sp];
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    for (const Index i : targets) {
      const auto si = static_cast<Sz>(i);
      if (row_done[si]) continue;
      const Rational* entry = value_at(si, p);
      if (entry == nullptr) continue;
      const Rational factor = *entry / pivot;
      lower_[k].emplace_back(i, factor);
      for (const auto& [q, v] : row[si]) {
        work[static_cast<Sz>(q)] = v;
        in_work[static_cast<Sz>(q)] = 1;
      }
      for (const auto& [q, v] : row[sr]) {
        const auto sq = static_cast<Sz>(q);
        if (q == p) continue;
        if (!in_work[sq]) {
          in_work[sq] = 1;
          work[sq] = Rational(0);
          rows_of[sq].push_back(i);  // fill
        }
        work[sq] -= factor * v;
      }
      RationalColumn updated;
      updated.reserve(row[si].size() + row[sr].size());
      for (const auto& [q, v] : row[si]) {
        const auto sq = static_cast<Sz>(q);
        if (in_work[sq] && q != p && !work[sq].is_zero()) updated.emplace_back(q, work[sq]);
        in_work[sq] = 0;
      }
      for (const auto& [q, v] : row[sr]) {
        const auto sq = static_cast<Sz>(q);
        if (in_work[sq] && q != p && !work[sq].is_zero()) updated.emplace_back(q, work[sq]);
        in_work[sq] = 0;
      }
      row[si] = std::move(updated);
    }
    row[sr].clear();
    rows_of[sp].clear();
    rows_of[sp].shrink_to_fit();
  }

  // U row-wise by position -> column-wise by step.
  for (Sz k = 0; k < sm; ++k) {
    for (auto& [q, v] : u_row[k]) {
      const Index l = step_of_position[static_cast<Sz>(q)];
      upper_[static_cast<Sz>(l)].emplace_back(static_cast<Index>(k), std::move(v));
    }
    factor_nonzeros_ += static_cast<Index>(u_row[k].size() + lower_[k].size()) + 1;
  }
  return true;
}

void ExactLu::solve(std::vector<Rational>& b) const {
  const auto sm = static_cast<Sz>(m_);
  for (Sz k = 0; k < sm; ++k) {
    const Rational& br = b[static_cast<Sz>(pivot_row_[k])];
    if (br.is_zero()) continue;
    const Rational top = br;
    for (const auto& [i, l] : lower_[k]) b[static_cast<Sz>(i)] -= l * top;
  }
  std::vector<Rational> x(sm, Rational(0));
  for (Sz k = sm; k-- > 0;) {
    const Rational& v = b[static_cast<Sz>(pivot_row_[k])];
    if (v.is_zero()) continue;
    const Rational xk = v / pivot_value_[k];
    for (const auto& [j, u] : upper_[k]) {
      b[static_cast<Sz>(pivot_row_[static_cast<Sz>(j)])] -= u * xk;
    }
    x[static_cast<Sz>(pivot_position_[k])] = xk;
  }
  for (Sz e = 0; e < eta_.size(); ++e) {
    const auto p = static_cast<Sz>(eta_position_[e]);
    if (x[p].is_zero()) continue;
    Rational pivot(0);
    for (const auto& [i, a] : eta_[e]) {
      if (static_cast<Sz>(i) == p) pivot = a;
    }
    const Rational xp = x[p] / pivot;
    for (const auto& [i, a] : eta_[e]) {
      if (static_cast<Sz>(i) != p) x[static_cast<Sz>(i)] -= a * xp;
    }
    x[p] = xp;
  }
  b = std::move(x);
}

void ExactLu::solve_transpose(std::vector<Rational>& c) const {
  const auto sm = static_cast<Sz>(m_);
  for (Sz e = eta_.size(); e-- > 0;) {
    const auto p = static_cast<Sz>(eta_position_[e]);
    Rational sum = c[p];
    Rational pivot(0);
    for (const auto& [i, a] : eta_[e]) {
      const auto si = static_cast<Sz>(i);
      if (si == p) {
        pivot = a;
      } else if (!c[si].is_zero()) {
        sum -= a * c[si];
      }
    }
    c[p] = sum / pivot;
  }
  std::vector<Rational> w(sm, Rational(0));
  for (Sz k = 0; k < sm; ++k) {
    Rational sum = c[static_cast<Sz>(pivot_position_[k])];
    for (const auto& [j, u] : upper_[k]) {
      const Rational& wj = w[static_cast<Sz>(pivot_row_[static_cast<Sz>(j)])];
      if (!wj.is_zero()) sum -= u * wj;
    }
    if (!sum.is_zero()) w[static_cast<Sz>(pivot_row_[k])] = sum / pivot_value_[k];
  }
  for (Sz k = sm; k-- > 0;) {
    const auto r = static_cast<Sz>(pivot_row_[k]);
    for (const auto& [i, l] : lower_[k]) {
      const Rational& wi = w[static_cast<Sz>(i)];
      if (!wi.is_zero()) w[r] -= l * wi;
    }
  }
  c = std::move(w);
}

void ExactLu::update(Index position, const std::vector<Rational>& alpha) {
  RationalColumn eta;
  for (Sz i = 0; i < alpha.size(); ++i) {
    if (!alpha[i].is_zero()) eta.emplace_back(static_cast<Index>(i), alpha[i]);
  }
  eta_position_.push_back(position);
  eta_.push_back(std::move(eta));
}

}  // namespace sankhya::exact
