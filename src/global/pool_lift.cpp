// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a pooling model in the P-formulation lifted to the Q-formulation's proportions
// before the global search (#514, `global_pool_lift`).
//
// THE P-FORMULATION (Haverly 1978) of one pool: inflows f_i (i in I), outflows f_j (j in J),
// all >= 0, a balance row sum_I f_i = sum_J f_j, and per quality k a row
//
//     sum_I a_ik f_i + alpha_k p_k sum_J f_j = 0           (the pool's quality p_k =
//     -a_ik/alpha_k
//                                                           averaged over the inflows)
//
// with the products p_k f_j. Its McCormick relaxation boxes each p_k separately, which is why
// adhya3 P stays open at 60 s while the same instance in the PQ-formulation closes in 80 nodes.
//
// THE LIFT. Add the Q-formulation's proportions q_i in [0, 1] (Ben-Tal, Eiger and Gershovitz,
// "Global minimization by reducing the duality gap", Math. Programming 63, 1994) and the rows
//
//     sum_I q_i = 1
//     f_i - sum_J q_i f_j = 0                         for each i in I
//     alpha_k p_k f_j + sum_I a_ik q_i f_j = 0        for each quality k and each j in J
//     the pool's capacity rows (rows over exactly I with one coefficient) restated over J
//
// keeping EVERY row and column of the model as it is. Valid at every point of the model:
// with throughput Y = sum_J f_j > 0 take q_i = f_i / Y (the quality row then says
// alpha_k p_k = -sum_I a_ik q_i, and times f_j that is the third row); with Y = 0 every f is 0
// by the balance row and the bounds, and any q on the simplex does. The objective does not see
// q, so the lifted model has the model's optimum and every point of it, cut to the first n
// columns, is a point of the model (its rows are all there and every incumbent is checked
// against them). Under `global_rlt` (on by default) rlt_rows() then multiplies sum_I q_i = 1 by
// each f_j and the restated capacity by each q_i: the rows the PQ-formulation adds (Quesada and
// Grossmann, "Global optimization of bilinear process networks with multicomponent flows",
// Comput. Chem. Eng. 19, 1995; Tawarmalani and Sahinidis, "Convexification and Global
// Optimization in Continuous and Mixed-Integer Nonlinear Programming", Kluwer 2002, ch. 9, that
// the pq relaxation dominates the p relaxation). The third row makes each relaxed p_k f_j the
// combination of relaxed q_i f_j the Q-formulation uses, so every pool-quality vector is held
// in the convex hull of its sources' quality vectors at once rather than one quality at a time.
// Recognising the structure inside a generic QCQP: Ceccon and Misener, "Solving the pooling
// problem at scale with extensible solver GALINI", Comput. Chem. Eng. 159 (2022).
//
// WHAT IS RECOGNISED, exactly: a quality row is an equality with right-hand side 0 whose every
// product is one column p times another (at least two others), all with one coefficient alpha;
// its balance row is an
// equality with right-hand side 0 and no products holding each of those other columns (J) with
// one coefficient c and every other column (I, at least two, a superset of the quality row's
// linear columns) with -c; every column of I and J has lower bound >= 0. Coefficients are
// compared exactly, and the added rows copy the model's coefficients, so nothing is divided.
// A pool written another way (an explicit throughput column, pool-to-pool arcs whose quality
// is itself a product, inventories) is not matched and the search is the one it was.

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "global/global_internal.hpp"

namespace sankhya::global {
namespace {

struct Quality {
  Index p = -1;
  double alpha = 0.0;
  std::vector<std::pair<Index, double>> linear;  ///< the row's (inflow, a_ik)
};

struct Pool {
  std::vector<Index> inflows;   ///< I
  std::vector<Index> outflows;  ///< J, ascending
  std::vector<Quality> qualities;
  std::vector<Index> quality_rows;
};

double coefficient(const std::vector<std::pair<Index, double>>& row, Index column) {
  for (const auto& [j, a] : row) {
    if (j == column) return a;
  }
  return 0.0;
}

/// The balance row of the pool whose outflows are `outflows`, and its inflows, or -1.
Index find_balance(const Problem& problem, const std::vector<std::vector<Index>>& rows_of,
                   Index skip, const std::vector<Index>& outflows,
                   std::vector<Index>* inflows) {
  for (const Index b : rows_of[static_cast<std::size_t>(outflows.front())]) {
    const auto u = static_cast<std::size_t>(b);
    if (b == skip || problem.row_lower[u] != 0.0 || problem.row_upper[u] != 0.0 ||
        !problem.row_products[u].empty()) {
      continue;
    }
    const auto& row = problem.row_linear[u];
    const double c = coefficient(row, outflows.front());
    std::vector<Index> in;
    std::size_t out = 0;
    bool ok = c != 0.0;
    for (const auto& [j, a] : row) {
      if (!ok) break;
      if (std::binary_search(outflows.begin(), outflows.end(), j)) {
        ok = a == c;
        ++out;
      } else {
        ok = a == -c;
        in.push_back(j);
      }
    }
    if (ok && out == outflows.size() && in.size() >= 2) {
      *inflows = std::move(in);
      return b;
    }
  }
  return -1;
}

}  // namespace

PoolLift lift_pools(const QcqpModel& model, QcqpModel* lifted) {
  const Problem problem = build_problem(model);
  const auto n = static_cast<std::size_t>(problem.n);
  std::vector<std::vector<Index>> rows_of(n);
  for (Index r = 0; r < problem.m; ++r) {
    for (const auto& term : problem.row_linear[static_cast<std::size_t>(r)]) {
      rows_of[static_cast<std::size_t>(term.first)].push_back(r);
    }
  }
  const auto nonnegative = [&](Index j) {
    return problem.col_lower[static_cast<std::size_t>(j)] >= 0.0;
  };

  std::map<Index, Pool> pools;  // by balance row, so the lift is in row order
  for (Index r = 0; r < problem.m; ++r) {
    const auto u = static_cast<std::size_t>(r);
    const auto& products = problem.row_products[u];
    if (products.empty() || problem.row_lower[u] != 0.0 || problem.row_upper[u] != 0.0)
      continue;
    const double alpha = products.front().second;
    const Product& first = problem.products[static_cast<std::size_t>(products.front().first)];
    for (const Index p : {first.a, first.b}) {
      std::vector<Index> outflows;
      bool ok = alpha != 0.0;
      for (const auto& [index, value] : products) {
        if (!ok) break;
        const Product& product = problem.products[static_cast<std::size_t>(index)];
        ok = value == alpha && !product.square() && (product.a == p || product.b == p);
        outflows.push_back(product.a == p ? product.b : product.a);
      }
      // One outflow: q_i f_j is then f_i itself and the proportions couple nothing. That shape
      // is also every path row of a Q- or PQ-formulated file (v_ij - q_i f_j = 0 with the
      // balance sum_i v_ij = f_j), which must be left as it is.
      if (!ok || outflows.size() < 2) continue;
      std::sort(outflows.begin(), outflows.end());
      const auto& linear = problem.row_linear[u];
      std::vector<Index> inflows;
      const Index balance = find_balance(problem, rows_of, r, outflows, &inflows);
      if (balance < 0) continue;
      ok = std::find(inflows.begin(), inflows.end(), p) == inflows.end() &&
           std::all_of(inflows.begin(), inflows.end(), nonnegative) &&
           std::all_of(outflows.begin(), outflows.end(), nonnegative);
      for (const auto& term : linear) {
        ok = ok && std::find(inflows.begin(), inflows.end(), term.first) != inflows.end();
      }
      if (!ok) continue;
      Pool& pool = pools[balance];
      pool.inflows = std::move(inflows);
      pool.outflows = std::move(outflows);
      pool.qualities.push_back({p, alpha, linear});
      pool.quality_rows.push_back(r);
      break;
    }
  }
  if (pools.empty()) return {};

  // The new columns (one q per inflow of each pool) and rows, as row-wise linear and product
  // parts; they are appended after the model's own.
  const Model& lin = model.linear;
  const Index n0 = lin.num_cols();
  const Index m0 = lin.num_rows();
  struct NewRow {
    double lower = 0.0;
    double upper = 0.0;
    std::vector<std::pair<Index, double>> linear;
    std::vector<QuadraticTerm> products;  ///< `row` filled in when appended
    std::string name;
  };
  std::vector<NewRow> rows;
  std::vector<std::string> q_names;
  const auto col_name = [&](Index j) {
    return lin.col_names.empty() ? "x" + std::to_string(j)
                                 : lin.col_names[static_cast<std::size_t>(j)];
  };
  const auto product = [](Index a, Index b, double value) {
    return QuadraticTerm{0, std::min(a, b), std::max(a, b), value};
  };
  Index next = n0;
  PoolLift lift;
  for (const auto& [balance, pool] : pools) {
    lift.records.push_back({balance, pool.quality_rows, {}});
    std::vector<Index> q;
    for (const Index i : pool.inflows) {
      q.push_back(next++);
      q_names.push_back("pool_lift_q_" + col_name(i));
    }
    const std::string tag = lin.row_names.empty()
                                ? std::to_string(balance)
                                : lin.row_names[static_cast<std::size_t>(balance)];
    NewRow simplex{1.0, 1.0, {}, {}, "pool_lift_sumq_" + tag};
    for (const Index qi : q) simplex.linear.emplace_back(qi, 1.0);
    rows.push_back(std::move(simplex));
    for (std::size_t k = 0; k < pool.inflows.size(); ++k) {
      NewRow inflow{
          0.0, 0.0, {{pool.inflows[k], 1.0}}, {}, "pool_lift_in_" + col_name(pool.inflows[k])};
      for (const Index j : pool.outflows) inflow.products.push_back(product(q[k], j, -1.0));
      rows.push_back(std::move(inflow));
    }
    for (const Quality& quality : pool.qualities) {
      for (const Index j : pool.outflows) {
        NewRow path{0.0,
                    0.0,
                    {},
                    {product(quality.p, j, quality.alpha)},
                    "pool_lift_pq_" + col_name(quality.p) + "_" + col_name(j)};
        for (std::size_t k = 0; k < pool.inflows.size(); ++k) {
          const double a = coefficient(quality.linear, pool.inflows[k]);
          if (a != 0.0) path.products.push_back(product(q[k], j, a));
        }
        rows.push_back(std::move(path));
      }
    }
    // A capacity row over exactly the inflows, with one coefficient, holds over the outflows
    // too since their sums are equal; over the outflows rlt_rows() can multiply it by each q_i.
    std::vector<Index> sorted_in = pool.inflows;
    std::sort(sorted_in.begin(), sorted_in.end());
    for (const Index c : rows_of[static_cast<std::size_t>(sorted_in.front())]) {
      const auto u = static_cast<std::size_t>(c);
      const auto& row = problem.row_linear[u];
      if (c == balance || !problem.row_products[u].empty() || row.size() != sorted_in.size())
        continue;
      const double beta = row.front().second;
      bool same = true;
      for (const auto& [j, a] : row) {
        same = same && a == beta && std::binary_search(sorted_in.begin(), sorted_in.end(), j);
      }
      if (!same) continue;
      lift.records.back().capacities.push_back(c);
      NewRow capacity{
          problem.row_lower[u], problem.row_upper[u], {}, {}, "pool_lift_cap_" + tag};
      for (const Index j : pool.outflows) capacity.linear.emplace_back(j, beta);
      rows.push_back(std::move(capacity));
    }
  }

  *lifted = model;
  Model& out = lifted->linear;
  const Index n1 = next;
  const auto m1 = m0 + static_cast<Index>(rows.size());
  out.resize_columns(n1);
  out.resize_rows(m1);
  for (Index j = n0; j < n1; ++j) {
    const auto u = static_cast<std::size_t>(j);
    out.col_upper[u] = 1.0;
    if (!out.col_names.empty()) out.col_names[u] = q_names[u - static_cast<std::size_t>(n0)];
  }
  out.matrix.reset(m1, n1);
  for (Index j = 0; j < n0; ++j) {
    const ColumnView column = lin.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      out.matrix.add_entry(column.rows[k], j, column.values[k]);
  }
  for (std::size_t t = 0; t < rows.size(); ++t) {
    const Index r = m0 + static_cast<Index>(t);
    const auto u = static_cast<std::size_t>(r);
    out.row_lower[u] = rows[t].lower;
    out.row_upper[u] = rows[t].upper;
    if (!out.row_names.empty()) out.row_names[u] = rows[t].name;
    for (const auto& [j, a] : rows[t].linear) out.matrix.add_entry(r, j, a);
    for (QuadraticTerm term : rows[t].products) {
      term.row = r;
      lifted->quadratic.push_back(term);
    }
  }
  out.matrix.finalize(0.0);
  if (lin.has_quadratic_objective()) {
    out.hessian.reset(n1, n1);
    for (Index j = 0; j < n0; ++j) {
      const ColumnView column = lin.hessian.column(j);
      for (Index k = 0; k < column.size; ++k)
        out.hessian.add_entry(column.rows[k], j, column.values[k]);
    }
    out.hessian.finalize(0.0);
  }
  lift.pools = static_cast<Count>(pools.size());
  lift.columns = n1 - n0;
  lift.rows = m1 - m0;
  return lift;
}

}  // namespace sankhya::global
