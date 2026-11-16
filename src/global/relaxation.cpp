// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the McCormick relaxation of a box, and the bound its LP multipliers prove (#514).
//
// THE ENVELOPE. For w = x_a * x_b over the box [la, ua] x [lb, ub], McCormick (1976) gives
// the convex and concave envelopes as the four inequalities obtained by multiplying out the
// nonnegative products of the bound distances:
//
//     (x_a - la)(x_b - lb) >= 0   ->   w >= lb x_a + la x_b - la lb
//     (ua - x_a)(ub - x_b) >= 0   ->   w >= ub x_a + ua x_b - ua ub
//     (x_a - la)(ub - x_b) >= 0   ->   w <= ub x_a + la x_b - la ub
//     (ua - x_a)(x_b - lb) >= 0   ->   w <= lb x_a + ua x_b - ua lb
//
// Al-Khayyal and Falk ("Jointly constrained biconvex programming", Math. Oper. Res. 8,
// 1983) proved these are the convex hull of the graph of x_a x_b over a box, so nothing
// tighter is available from the one product alone. A square w = x^2 is convex: its concave
// envelope is the secant w <= (l + u) x - l u, and it is under-estimated by tangents, taken
// here at l, u and the midpoint (Tawarmalani and Sahinidis 2002, ch. 4).
//
// RLT ROWS (`global_rlt`). The envelope sees one product at a time. A linear row a'x <= u
// multiplied by a bound factor y - ly >= 0 of a column y gives (u - a'x)(y - ly) >= 0, which
// is linear in x and the products x_k y; when every such product is already a column w, the
// row couples the products the way the model does, which no single envelope can (Sherali and
// Adams, "A Reformulation-Linearization Technique...", Kluwer 1999; Liberti, Int. Trans.
// Oper. Res. 11, 2004). On the Q-formulation of pooling, sum_i q_il = 1 times f_lj and the pool
// capacity times q_il are exactly the rows the PQ-formulation adds (Quesada and Grossmann,
// Comput. Chem. Eng. 19, 1995; Tawarmalani and Sahinidis 2002, ch. 9). Each row holds at every
// point of the model in the box, so the relaxation stays a relaxation.
//
// THE BOUND. The LP optimum is a lower bound on the box only if the LP was solved to
// optimality exactly; a dual simplex stops on tolerances. dual_bound_from_multipliers()
// instead evaluates the Lagrangian bound at the multipliers the LP returned, which is a
// valid lower bound for ANY multipliers (weak duality), so an inexact LP can only make the
// bound weaker, never wrong (Neumaier and Shcherbina 2004).

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

#include "global/global_internal.hpp"

namespace sankhya::global {
namespace {

struct Range {
  double lower;
  double upper;
};

Range product_range(double la, double ua, double lb, double ub) {
  const double c[4] = {la * lb, la * ub, ua * lb, ua * ub};
  return {*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
}

Range square_range(double l, double u) {
  const double lo = l * l;
  const double hi = u * u;
  if (l >= 0.0) return {lo, hi};
  if (u <= 0.0) return {hi, lo};
  return {0.0, std::max(lo, hi)};
}

}  // namespace

Model build_relaxation(const Problem& problem, const Box& box, double cutoff) {
  const Index n = problem.n;
  const auto num_products = static_cast<Index>(problem.products.size());
  // Four rows per product: the McCormick envelope, or a secant and three tangents.
  const Index envelope_rows = 4 * num_products;
  const auto rlt_count = static_cast<Index>(problem.rlt.size());
  const Index cutoff_rows = std::isfinite(cutoff) ? 1 : 0;

  Model lp;
  lp.name = "mccormick";
  lp.resize_columns(n + num_products);
  lp.resize_rows(problem.m + envelope_rows + rlt_count + cutoff_rows);
  lp.objective_offset = problem.offset;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    lp.col_cost[u] = problem.cost[u];
    lp.col_lower[u] = box.lower[u];
    lp.col_upper[u] = box.upper[u];
  }
  for (const auto& [index, value] : problem.objective_products) {
    lp.col_cost[static_cast<std::size_t>(n + index)] += value;
  }

  lp.matrix.reset(problem.m + envelope_rows + rlt_count + cutoff_rows, n + num_products);
  for (Index i = 0; i < problem.m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    lp.row_lower[u] = problem.row_lower[u];
    lp.row_upper[u] = problem.row_upper[u];
    for (const auto& [column, value] : problem.row_linear[u]) {
      lp.matrix.add_entry(i, column, value);
    }
    for (const auto& [index, value] : problem.row_products[u]) {
      lp.matrix.add_entry(i, n + index, value);
    }
  }

  Index row = problem.m;
  const auto add = [&](Index r, Index column, double value) {
    if (value != 0.0) lp.matrix.add_entry(r, column, value);
  };
  const auto set_row = [&](Index r, double lower, double upper) {
    lp.row_lower[static_cast<std::size_t>(r)] = lower;
    lp.row_upper[static_cast<std::size_t>(r)] = upper;
  };
  for (Index p = 0; p < num_products; ++p) {
    const Product& product = problem.products[static_cast<std::size_t>(p)];
    const Index w = n + p;
    const double la = box.lower[static_cast<std::size_t>(product.a)];
    const double ua = box.upper[static_cast<std::size_t>(product.a)];
    const double lb = box.lower[static_cast<std::size_t>(product.b)];
    const double ub = box.upper[static_cast<std::size_t>(product.b)];
    const Range range = product.square() ? square_range(la, ua) : product_range(la, ua, lb, ub);
    lp.col_lower[static_cast<std::size_t>(w)] = range.lower;
    lp.col_upper[static_cast<std::size_t>(w)] = range.upper;
    if (!product.square()) {
      // w - lb x_a - la x_b >= -la lb
      add(row, w, 1.0);
      add(row, product.a, -lb);
      add(row, product.b, -la);
      set_row(row++, -la * lb, kInfinity);
      // w - ub x_a - ua x_b >= -ua ub
      add(row, w, 1.0);
      add(row, product.a, -ub);
      add(row, product.b, -ua);
      set_row(row++, -ua * ub, kInfinity);
      // w - ub x_a - la x_b <= -la ub
      add(row, w, 1.0);
      add(row, product.a, -ub);
      add(row, product.b, -la);
      set_row(row++, -kInfinity, -la * ub);
      // w - lb x_a - ua x_b <= -ua lb
      add(row, w, 1.0);
      add(row, product.a, -lb);
      add(row, product.b, -ua);
      set_row(row++, -kInfinity, -ua * lb);
    } else {
      // Secant: w - (l + u) x <= -l u.
      add(row, w, 1.0);
      add(row, product.a, -(la + ua));
      set_row(row++, -kInfinity, -la * ua);
      // Tangents at l, u and the midpoint: w - 2 t x >= -t^2.
      for (const double t : {la, ua, 0.5 * (la + ua)}) {
        add(row, w, 1.0);
        add(row, product.a, -2.0 * t);
        set_row(row++, -t * t, kInfinity);
      }
    }
  }
  for (const RltRow& rlt : problem.rlt) {
    const auto& linear = problem.row_linear[static_cast<std::size_t>(rlt.row)];
    const double lower = problem.row_lower[static_cast<std::size_t>(rlt.row)];
    const double upper = problem.row_upper[static_cast<std::size_t>(rlt.row)];
    const double c = rlt.side == 'u' ? upper : lower;  // 'e': lower == upper == b
    // (a'x - c)(y - d) = a'w - d a'x - c y + c d, times +1 or -1 to make both factors >= 0.
    const double d = rlt.side == 'e'     ? 0.0
                     : rlt.factor == 'l' ? box.lower[static_cast<std::size_t>(rlt.column)]
                                         : box.upper[static_cast<std::size_t>(rlt.column)];
    std::map<Index, double> entries;
    for (std::size_t k = 0; k < linear.size(); ++k) {
      entries[n + rlt.products[k]] += linear[k].second;
      entries[linear[k].first] -= d * linear[k].second;
    }
    entries[rlt.column] -= c;
    for (const auto& [column, value] : entries) add(row, column, value);
    const double rhs = -c * d;
    if (rlt.side == 'e') {
      set_row(row++, 0.0, 0.0);
    } else if ((rlt.side == 'l') == (rlt.factor == 'l')) {
      set_row(row++, rhs, kInfinity);  // both factors as written, product >= 0
    } else {
      set_row(row++, -kInfinity, rhs);  // one factor negated
    }
  }
  if (cutoff_rows > 0) {
    for (Index j = 0; j < lp.num_cols(); ++j)
      add(row, j, lp.col_cost[static_cast<std::size_t>(j)]);
    set_row(row++, -kInfinity, cutoff - lp.objective_offset);
  }
  // Every coefficient here is data the box determines, not arithmetic residue (#590).
  lp.matrix.finalize(0.0);
  return lp;
}

std::vector<RltRow> rlt_rows(const Problem& problem) {
  std::map<std::pair<Index, Index>, Index> index_of;
  std::vector<std::vector<Index>> partners(static_cast<std::size_t>(problem.n));
  for (std::size_t p = 0; p < problem.products.size(); ++p) {
    const Product& product = problem.products[p];
    index_of.emplace(std::make_pair(product.a, product.b), static_cast<Index>(p));
    partners[static_cast<std::size_t>(product.a)].push_back(product.b);
    if (!product.square()) partners[static_cast<std::size_t>(product.b)].push_back(product.a);
  }
  for (std::vector<Index>& list : partners) std::sort(list.begin(), list.end());

  // ponytail: every (row, y) pair is kept, which on a dense product graph is O(rows * n)
  // rows of up to n entries; add a budget if a model with one ever matters.
  std::vector<RltRow> out;
  for (Index i = 0; i < problem.m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto& linear = problem.row_linear[u];
    const double lower = problem.row_lower[u];
    const double upper = problem.row_upper[u];
    if (!problem.row_products[u].empty() || linear.empty()) continue;
    if (!std::isfinite(lower) && !std::isfinite(upper)) continue;
    std::vector<Index> candidates = partners[static_cast<std::size_t>(linear.front().first)];
    for (std::size_t k = 1; k < linear.size() && !candidates.empty(); ++k) {
      const auto& next = partners[static_cast<std::size_t>(linear[k].first)];
      std::vector<Index> both;
      std::set_intersection(candidates.begin(), candidates.end(), next.begin(), next.end(),
                            std::back_inserter(both));
      candidates = std::move(both);
    }
    for (const Index y : candidates) {
      RltRow base;
      base.row = i;
      base.column = y;
      for (const auto& term : linear) {
        const Index k = term.first;
        base.products.push_back(index_of.at(std::make_pair(std::min(k, y), std::max(k, y))));
      }
      if (lower == upper) {
        out.push_back(base);
        continue;
      }
      for (const char side : {'l', 'u'}) {
        if (!std::isfinite(side == 'l' ? lower : upper)) continue;
        for (const char factor : {'l', 'u'}) {
          RltRow row = base;
          row.side = side;
          row.factor = factor;
          out.push_back(std::move(row));
        }
      }
    }
  }
  return out;
}

double dual_bound_from_multipliers(const Model& lp, const std::vector<double>& y) {
  if (static_cast<Index>(y.size()) != lp.num_rows()) return -kInfinity;
  std::vector<double> projected = y;
  for (Index i = 0; i < lp.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    if ((projected[u] > 0.0 && !std::isfinite(lp.row_lower[u])) ||
        (projected[u] < 0.0 && !std::isfinite(lp.row_upper[u])) ||
        !std::isfinite(projected[u])) {
      projected[u] = 0.0;
    }
  }
  double bound = lp.objective_offset;
  for (Index i = 0; i < lp.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    // A multiplier that would lean on a bound the row does not have is taken as zero: the
    // bound holds for ANY multipliers, so projecting them costs validity nothing, while a
    // rounding-size multiplier of the wrong sign would otherwise make the bound -infinity.
    const double yi = projected[u];
    if (yi > 0.0) {
      bound += yi * lp.row_lower[u];
    } else if (yi < 0.0) {
      bound += yi * lp.row_upper[u];
    }
  }
  for (Index j = 0; j < lp.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    double d = lp.col_cost[u];
    const ColumnView column = lp.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      d -= column.values[k] * projected[static_cast<std::size_t>(column.rows[k])];
    if (d > 0.0) {
      if (!std::isfinite(lp.col_lower[u])) return -kInfinity;
      bound += d * lp.col_lower[u];
    } else if (d < 0.0) {
      if (!std::isfinite(lp.col_upper[u])) return -kInfinity;
      bound += d * lp.col_upper[u];
    }
  }
  return bound;
}

}  // namespace sankhya::global
