// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a nonlinear model as the smooth NLP an optimizer iterates on (NLP stage 1).
// See nlp_problem.hpp for the form and the conventions.

#include "nlp/nlp_problem.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"

namespace sankhya::nlp {
namespace {

using Pair = std::pair<Index, Index>;  // (row, col), row >= col

bool by_column(const Pair& a, const Pair& b) {
  return a.second != b.second ? a.second < b.second : a.first < b.first;
}

}  // namespace

std::string NlpProblem::build(const NonlinearModel& model, NlpProblem* out) {
  const std::string problem = model.validate();
  if (!problem.empty()) return problem;
  const Model& base = model.base;
  NlpProblem p;
  p.n_ = base.num_cols();
  p.linear_rows_ = base.num_rows();
  p.m_ = p.linear_rows_ + static_cast<Index>(model.constraints.size());
  p.sense_ = base.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  p.offset_ = base.objective_offset;
  p.cost_ = base.col_cost;
  p.x_lower_ = base.col_lower;
  p.x_upper_ = base.col_upper;
  for (double& v : p.x_lower_) v = normalize_infinity(v);
  for (double& v : p.x_upper_) v = normalize_infinity(v);
  p.g_lower_ = base.row_lower;
  p.g_upper_ = base.row_upper;
  for (const NonlinearConstraint& c : model.constraints) {
    p.g_lower_.push_back(c.lower);
    p.g_upper_.push_back(c.upper);
  }
  for (double& v : p.g_lower_) v = normalize_infinity(v);
  for (double& v : p.g_upper_) v = normalize_infinity(v);

  std::vector<Pair> pattern;
  // Q, lower triangle including the diagonal (model.hpp). An entry stored above the diagonal
  // is folded below it, which is the same term of the symmetric matrix.
  p.q_starts_.assign(1, 0);
  if (base.has_quadratic_objective()) {
    const SparseMatrix& q = base.hessian;
    for (Index j = 0; j < q.num_cols(); ++j) {
      const ColumnView col = q.column(j);
      for (Index k = 0; k < col.size; ++k) {
        const Index i = col.rows[k];
        p.q_rows_.push_back(i);
        p.q_values_.push_back(col.values[k]);
        pattern.emplace_back(std::max(i, j), std::min(i, j));
      }
      p.q_starts_.push_back(static_cast<Index>(p.q_rows_.size()));
    }
  }
  if (model.objective != kNoExpr) {
    p.has_objective_expression_ = true;
    p.objective_expression_ = CompiledExpression(model.graph, model.objective);
    const auto& own = p.objective_expression_.hessian_pattern();
    pattern.insert(pattern.end(), own.begin(), own.end());
  }
  for (const NonlinearConstraint& c : model.constraints) {
    p.expressions_.emplace_back(model.graph, c.expression);
    const auto& own = p.expressions_.back().hessian_pattern();
    pattern.insert(pattern.end(), own.begin(), own.end());
  }
  std::sort(pattern.begin(), pattern.end(), by_column);
  pattern.erase(std::unique(pattern.begin(), pattern.end()), pattern.end());
  if (!nonzero_count_fits(pattern.size())) {
    return fmt::format("the Hessian of the Lagrangian would hold {} entries", pattern.size());
  }
  p.h_starts_.assign(static_cast<std::size_t>(p.n_) + 1, 0);
  for (const auto& [row, col] : pattern) {
    ++p.h_starts_[static_cast<std::size_t>(col) + 1];
    p.h_rows_.push_back(row);
  }
  for (Index j = 0; j < p.n_; ++j) {
    p.h_starts_[static_cast<std::size_t>(j) + 1] += p.h_starts_[static_cast<std::size_t>(j)];
  }

  // Every contribution's position in the union pattern, found once.
  for (Index j = 0; j + 1 < static_cast<Index>(p.q_starts_.size()); ++j) {
    for (Index k = p.q_starts_[static_cast<std::size_t>(j)];
         k < p.q_starts_[static_cast<std::size_t>(j) + 1]; ++k) {
      const Index i = p.q_rows_[static_cast<std::size_t>(k)];
      p.q_slot_.push_back(p.hessian_slot(std::max(i, j), std::min(i, j)));
    }
  }
  const auto slots_of = [&p](const CompiledExpression& e) {
    std::vector<Index> slots;
    for (const auto& [row, col] : e.hessian_pattern())
      slots.push_back(p.hessian_slot(row, col));
    return slots;
  };
  if (p.has_objective_expression_) p.objective_slot_ = slots_of(p.objective_expression_);
  for (const CompiledExpression& e : p.expressions_) p.expression_slot_.push_back(slots_of(e));

  // The Jacobian: the linear rows' coefficients, then each expression's columns.
  p.j_starts_.assign(1, 0);
  p.a_starts_.assign(1, 0);
  if (p.linear_rows_ > 0) {
    const CsrView rows(base.matrix);
    for (Index i = 0; i < p.linear_rows_; ++i) {
      const ColumnView r = rows.row(i);
      std::vector<std::pair<Index, double>> entries;
      for (Index k = 0; k < r.size; ++k) entries.emplace_back(r.rows[k], r.values[k]);
      std::sort(entries.begin(), entries.end());
      for (const auto& [col, value] : entries) {
        p.a_cols_.push_back(col);
        p.a_values_.push_back(value);
        p.j_cols_.push_back(col);
      }
      p.a_starts_.push_back(static_cast<Index>(p.j_cols_.size()));
      p.j_starts_.push_back(static_cast<Index>(p.j_cols_.size()));
    }
  }
  for (const CompiledExpression& e : p.expressions_) {
    p.j_cols_.insert(p.j_cols_.end(), e.variables().begin(), e.variables().end());
    p.j_starts_.push_back(static_cast<Index>(p.j_cols_.size()));
  }
  if (!nonzero_count_fits(p.j_cols_.size())) {
    return fmt::format("the Jacobian would hold {} entries", p.j_cols_.size());
  }
  *out = std::move(p);
  return {};
}

Index NlpProblem::hessian_slot(Index row, Index col) const {
  const auto first = h_rows_.begin() + h_starts_[static_cast<std::size_t>(col)];
  const auto last = h_rows_.begin() + h_starts_[static_cast<std::size_t>(col) + 1];
  const auto it = std::lower_bound(first, last, row);
  if (it == last || *it != row) return -1;
  return static_cast<Index>(it - h_rows_.begin());
}

bool NlpProblem::objective(const std::vector<double>& x, double* value,
                           Evaluation* error) const {
  double f = offset_;
  for (Index j = 0; j < n_; ++j)
    f += cost_[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  for (Index j = 0; j + 1 < static_cast<Index>(q_starts_.size()); ++j) {
    for (Index k = q_starts_[static_cast<std::size_t>(j)];
         k < q_starts_[static_cast<std::size_t>(j) + 1]; ++k) {
      const Index i = q_rows_[static_cast<std::size_t>(k)];
      const double term = q_values_[static_cast<std::size_t>(k)] *
                          x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)];
      f += i == j ? 0.5 * term : term;  // an off-diagonal entry stands for two of Q
    }
  }
  if (has_objective_expression_) {
    double e = 0.0;
    if (!objective_expression_.value(x, &e, error)) return false;
    f += e;
  }
  *value = sense_ * f;
  return true;
}

bool NlpProblem::objective_gradient(const std::vector<double>& x, double* value,
                                    std::vector<double>* gradient, Evaluation* error) const {
  if (!objective(x, value, error)) return false;
  gradient->assign(cost_.begin(), cost_.end());
  std::vector<double>& g = *gradient;
  for (Index j = 0; j + 1 < static_cast<Index>(q_starts_.size()); ++j) {
    for (Index k = q_starts_[static_cast<std::size_t>(j)];
         k < q_starts_[static_cast<std::size_t>(j) + 1]; ++k) {
      const auto i = static_cast<std::size_t>(q_rows_[static_cast<std::size_t>(k)]);
      const auto jj = static_cast<std::size_t>(j);
      const double q = q_values_[static_cast<std::size_t>(k)];
      g[i] += q * x[jj];
      if (i != jj) g[jj] += q * x[i];
    }
  }
  if (has_objective_expression_) {
    double e = 0.0;
    if (!objective_expression_.gradient(x, &e, &scratch_, error)) return false;
    const auto& cols = objective_expression_.variables();
    for (std::size_t k = 0; k < cols.size(); ++k) {
      g[static_cast<std::size_t>(cols[k])] += scratch_[k];
    }
  }
  for (double& v : g) v *= sense_;
  return true;
}

bool NlpProblem::constraints(const std::vector<double>& x, std::vector<double>* values,
                             Evaluation* error) const {
  values->assign(static_cast<std::size_t>(m_), 0.0);
  for (Index i = 0; i < linear_rows_; ++i) {
    double s = 0.0;
    for (Index k = a_starts_[static_cast<std::size_t>(i)];
         k < a_starts_[static_cast<std::size_t>(i) + 1]; ++k) {
      s += a_values_[static_cast<std::size_t>(k)] *
           x[static_cast<std::size_t>(a_cols_[static_cast<std::size_t>(k)])];
    }
    (*values)[static_cast<std::size_t>(i)] = s;
  }
  for (std::size_t e = 0; e < expressions_.size(); ++e) {
    double v = 0.0;
    if (!expressions_[e].value(x, &v, error)) return false;
    (*values)[static_cast<std::size_t>(linear_rows_) + e] = v;
  }
  return true;
}

bool NlpProblem::jacobian(const std::vector<double>& x, std::vector<double>* values,
                          Evaluation* error) const {
  values->assign(j_cols_.size(), 0.0);
  std::copy(a_values_.begin(), a_values_.end(), values->begin());
  for (std::size_t e = 0; e < expressions_.size(); ++e) {
    double v = 0.0;
    if (!expressions_[e].gradient(x, &v, &scratch_, error)) return false;
    const auto start =
        static_cast<std::size_t>(j_starts_[static_cast<std::size_t>(linear_rows_) + e]);
    std::copy(scratch_.begin(), scratch_.end(),
              values->begin() + static_cast<std::ptrdiff_t>(start));
  }
  return true;
}

bool NlpProblem::hessian(const std::vector<double>& x, double sigma,
                         const std::vector<double>& lambda, std::vector<double>* values,
                         Evaluation* error) const {
  values->assign(h_rows_.size(), 0.0);
  std::vector<double>& h = *values;
  const double objective_weight = sigma * sense_;
  if (objective_weight != 0.0) {
    for (std::size_t k = 0; k < q_values_.size(); ++k) {
      h[static_cast<std::size_t>(q_slot_[k])] += objective_weight * q_values_[k];
    }
    if (has_objective_expression_) {
      hess_scratch_.assign(objective_slot_.size(), 0.0);
      if (!objective_expression_.add_hessian(x, objective_weight, &hess_scratch_, error)) {
        return false;
      }
      for (std::size_t k = 0; k < objective_slot_.size(); ++k) {
        h[static_cast<std::size_t>(objective_slot_[k])] += hess_scratch_[k];
      }
    }
  }
  for (std::size_t e = 0; e < expressions_.size(); ++e) {
    const double weight = lambda[static_cast<std::size_t>(linear_rows_) + e];
    if (weight == 0.0 || expression_slot_[e].empty()) continue;
    hess_scratch_.assign(expression_slot_[e].size(), 0.0);
    if (!expressions_[e].add_hessian(x, weight, &hess_scratch_, error)) return false;
    for (std::size_t k = 0; k < expression_slot_[e].size(); ++k) {
      h[static_cast<std::size_t>(expression_slot_[e][k])] += hess_scratch_[k];
    }
  }
  return true;
}

}  // namespace sankhya::nlp
