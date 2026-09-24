// SPDX-License-Identifier: Apache-2.0
// SANKHYA - QcqpModel's checks and evaluation, and the row-wise Problem the global search
// works on (#514).

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "global/global_internal.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya {

std::string QcqpModel::validate() const {
  const std::string base = linear.validate();
  if (!base.empty()) return base;
  const Index n = linear.num_cols();
  const Index m = linear.num_rows();
  std::set<std::tuple<Index, Index, Index>> seen;
  for (const QuadraticTerm& term : quadratic) {
    if (term.row < 0 || term.row >= m) {
      return fmt::format("quadratic term names row {}, outside [0, {})", term.row, m);
    }
    if (term.first < 0 || term.second >= n || term.first > term.second) {
      return fmt::format(
          "quadratic term in row {} names columns ({}, {}); need 0 <= first "
          "<= second < {}",
          term.row, term.first, term.second, n);
    }
    if (!std::isfinite(term.value)) {
      return fmt::format("quadratic term in row {} has coefficient {}", term.row, term.value);
    }
    if (!seen.insert({term.row, term.first, term.second}).second) {
      return fmt::format("quadratic term ({}, {}) repeated in row {}", term.first, term.second,
                         term.row);
    }
  }
  return {};
}

std::vector<double> QcqpModel::row_activity(const std::vector<double>& x) const {
  std::vector<double> activity(static_cast<std::size_t>(linear.num_rows()), 0.0);
  linear.matrix.multiply(x.data(), activity.data());
  for (const QuadraticTerm& term : quadratic) {
    activity[static_cast<std::size_t>(term.row)] += term.value *
                                                    x[static_cast<std::size_t>(term.first)] *
                                                    x[static_cast<std::size_t>(term.second)];
  }
  return activity;
}

double QcqpModel::objective(const std::vector<double>& x) const {
  return linear.evaluate_objective(x.data());
}

namespace global {

Problem build_problem(const QcqpModel& model) {
  const Model& lin = model.linear;
  Problem p;
  p.source = &model;
  p.n = lin.num_cols();
  p.m = lin.num_rows();
  p.sigma = lin.sense_multiplier();
  p.offset = p.sigma * lin.objective_offset;
  p.cost.resize(static_cast<std::size_t>(p.n));
  for (Index j = 0; j < p.n; ++j) {
    p.cost[static_cast<std::size_t>(j)] = p.sigma * lin.col_cost[static_cast<std::size_t>(j)];
  }
  p.col_lower = lin.col_lower;
  p.col_upper = lin.col_upper;
  p.row_lower = lin.row_lower;
  p.row_upper = lin.row_upper;
  p.row_linear.assign(static_cast<std::size_t>(p.m), {});
  p.row_products.assign(static_cast<std::size_t>(p.m), {});
  for (Index j = 0; j < p.n; ++j) {
    const ColumnView column = lin.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      p.row_linear[static_cast<std::size_t>(column.rows[k])].emplace_back(j, column.values[k]);
    }
  }

  std::map<std::pair<Index, Index>, Index> index_of;
  const auto product_index = [&](Index a, Index b) {
    const auto key = std::make_pair(std::min(a, b), std::max(a, b));
    const auto found = index_of.find(key);
    if (found != index_of.end()) return found->second;
    const auto index = static_cast<Index>(p.products.size());
    p.products.push_back({key.first, key.second});
    index_of.emplace(key, index);
    return index;
  };
  for (const QuadraticTerm& term : model.quadratic) {
    p.row_products[static_cast<std::size_t>(term.row)].emplace_back(
        product_index(term.first, term.second), term.value);
  }
  // The objective's 0.5 x'Hx over H's lower triangle: an off-diagonal stored entry stands
  // for two entries of the symmetric H, so it is the product's full coefficient, while a
  // diagonal entry keeps its 0.5 (the convention of Model::evaluate_objective).
  if (lin.has_quadratic_objective()) {
    std::map<Index, double> by_product;
    for (Index j = 0; j < lin.hessian.num_cols(); ++j) {
      const ColumnView column = lin.hessian.column(j);
      for (Index k = 0; k < column.size; ++k) {
        const Index i = column.rows[k];
        const double scale = i == j ? 0.5 : 1.0;
        by_product[product_index(i, j)] += p.sigma * scale * column.values[k];
      }
    }
    for (const auto& [index, value] : by_product) {
      if (value != 0.0) p.objective_products.emplace_back(index, value);
    }
  }
  p.in_product.assign(static_cast<std::size_t>(p.n), 0);
  for (const Product& product : p.products) {
    p.in_product[static_cast<std::size_t>(product.a)] = 1;
    p.in_product[static_cast<std::size_t>(product.b)] = 1;
  }
  return p;
}

double min_objective(const Problem& problem, const std::vector<double>& x) {
  return problem.sigma * problem.source->objective(x);
}

}  // namespace global
}  // namespace sankhya
