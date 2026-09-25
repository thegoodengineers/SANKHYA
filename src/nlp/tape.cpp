// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one expression compiled for repeated evaluation (NLP stage 1). See tape.hpp.

#include "nlp/tape.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "nlp/derivative_rules.hpp"

namespace sankhya::nlp {
namespace {

Evaluation failure(EvalError error, ExprId at, std::string message) {
  Evaluation result;
  result.error = error;
  result.failed_at = at;
  result.message = std::move(message);
  return result;
}

}  // namespace

CompiledExpression::CompiledExpression(const ExpressionGraph& graph, ExprId root) {
  const std::vector<ExprId> order = graph.tape(root);
  std::vector<std::size_t> local(graph.size(), 0);
  child_start_.push_back(0);
  for (std::size_t p = 0; p < order.size(); ++p) {
    const ExprId id = order[p];
    local[static_cast<std::size_t>(id)] = p;
    const Node& n = graph.node(id);
    nodes_.push_back(n);
    ids_.push_back(id);
    for (const ExprId c : n.children) child_pos_.push_back(local[static_cast<std::size_t>(c)]);
    child_start_.push_back(child_pos_.size());
    if (n.op == Op::kVariable) variables_.push_back(n.variable);
  }
  // Each column is one interned node, so each appears once on the tape.
  std::sort(variables_.begin(), variables_.end());
  variable_of_.assign(nodes_.size(), 0);
  position_of_variable_.assign(variables_.size(), 0);
  for (std::size_t p = 0; p < nodes_.size(); ++p) {
    if (nodes_[p].op != Op::kVariable) continue;
    const auto k = static_cast<std::size_t>(
        std::lower_bound(variables_.begin(), variables_.end(), nodes_[p].variable) -
        variables_.begin());
    variable_of_[p] = k;
    position_of_variable_[k] = p;
  }
  pattern_ = graph.hessian_pattern(root);
  for (std::size_t slot = 0; slot < pattern_.size(); ++slot) {
    const auto [row, col] = pattern_[slot];
    if (sweeps_.empty() || sweeps_.back().column != col) sweeps_.push_back(Sweep{col, {}});
    const auto k = static_cast<std::size_t>(
        std::lower_bound(variables_.begin(), variables_.end(), row) - variables_.begin());
    sweeps_.back().entries.emplace_back(slot, position_of_variable_[k]);
  }
  values_.assign(nodes_.size(), 0.0);
  tangent_.assign(nodes_.size(), 0.0);
  adjoint_.assign(nodes_.size(), 0.0);
  adjoint_dot_.assign(nodes_.size(), 0.0);
}

bool CompiledExpression::forward(const std::vector<double>& x, Evaluation* error) const {
  std::string why;
  for (std::size_t p = 0; p < nodes_.size(); ++p) {
    const Node& n = nodes_[p];
    double result = 0.0;
    if (n.op == Op::kVariable) {
      result = x[static_cast<std::size_t>(n.variable)];
    } else {
      child_.resize(child_start_[p + 1] - child_start_[p]);
      for (std::size_t k = 0; k < child_.size(); ++k) {
        child_[k] = values_[child_pos_[child_start_[p] + k]];
      }
      if (!detail::apply(n, child_.data(), &result, &why)) {
        if (error != nullptr) {
          *error =
              failure(EvalError::kDomain, ids_[p],
                      child_.empty() ? why : fmt::format("{} (argument {:g})", why, child_[0]));
        }
        return false;
      }
    }
    if (!std::isfinite(result)) {
      if (error != nullptr) {
        *error = failure(EvalError::kNonFinite, ids_[p],
                         fmt::format("{} overflowed to {}", to_string(n.op), result));
      }
      return false;
    }
    values_[p] = result;
  }
  return true;
}

bool CompiledExpression::local_partial(std::size_t p, std::size_t k, bool with_tangent,
                                       double* d, double* d_dot) const {
  const Node& n = nodes_[p];
  double child[2] = {0.0, 0.0};
  double child_dot[2] = {0.0, 0.0};
  const std::size_t count = child_start_[p + 1] - child_start_[p];
  const std::size_t gathered = n.op == Op::kSum ? 0 : std::min<std::size_t>(count, 2);
  for (std::size_t c = 0; c < gathered; ++c) {
    const std::size_t at = child_pos_[child_start_[p] + c];
    child[c] = values_[at];
    child_dot[c] = tangent_[at];
  }
  return detail::partial(n, k, values_[p], child, with_tangent ? child_dot : nullptr, d, d_dot);
}

bool CompiledExpression::value(const std::vector<double>& x, double* out,
                               Evaluation* error) const {
  if (nodes_.empty()) {
    *out = 0.0;
    return true;
  }
  if (!forward(x, error)) return false;
  *out = values_.back();
  return true;
}

bool CompiledExpression::gradient(const std::vector<double>& x, double* value,
                                  std::vector<double>* out, Evaluation* error) const {
  out->assign(variables_.size(), 0.0);
  if (nodes_.empty()) {
    *value = 0.0;
    return true;
  }
  if (!forward(x, error)) return false;
  *value = values_.back();
  std::fill(adjoint_.begin(), adjoint_.end(), 0.0);
  adjoint_.back() = 1.0;
  for (std::size_t p = nodes_.size(); p-- > 0;) {
    const double bar = adjoint_[p];
    if (bar == 0.0) continue;
    if (nodes_[p].op == Op::kVariable) {
      (*out)[variable_of_[p]] += bar;
      continue;
    }
    for (std::size_t k = 0; k < child_start_[p + 1] - child_start_[p]; ++k) {
      double d = 0.0;
      if (!local_partial(p, k, false, &d, nullptr)) {
        if (error != nullptr) {
          *error = failure(
              EvalError::kDomain, ids_[p],
              fmt::format("{} has a value here but no derivative", to_string(nodes_[p].op)));
        }
        return false;
      }
      adjoint_[child_pos_[child_start_[p] + k]] += bar * d;
    }
  }
  return true;
}

bool CompiledExpression::add_hessian(const std::vector<double>& x, double weight,
                                     std::vector<double>* out, Evaluation* error) const {
  if (out->size() < pattern_.size()) out->resize(pattern_.size(), 0.0);
  if (sweeps_.empty() || weight == 0.0) return true;
  if (!forward(x, error)) return false;
  const auto no_derivative = [&](std::size_t p) {
    if (error != nullptr) {
      *error = failure(
          EvalError::kDomain, ids_[p],
          fmt::format("{} has a value here but no derivative", to_string(nodes_[p].op)));
    }
    return false;
  };
  for (const Sweep& sweep : sweeps_) {
    // Tangent sweep along e_column: every node's derivative with respect to that column.
    for (std::size_t p = 0; p < nodes_.size(); ++p) {
      const Node& n = nodes_[p];
      if (n.op == Op::kVariable) {
        tangent_[p] = n.variable == sweep.column ? 1.0 : 0.0;
        continue;
      }
      double t = 0.0;
      for (std::size_t k = 0; k < child_start_[p + 1] - child_start_[p]; ++k) {
        const double dot = tangent_[child_pos_[child_start_[p] + k]];
        if (dot == 0.0) continue;
        double d = 0.0;
        if (!local_partial(p, k, false, &d, nullptr)) return no_derivative(p);
        t += d * dot;
      }
      tangent_[p] = t;
    }
    // Reverse sweep carrying each adjoint and its derivative along e_column.
    std::fill(adjoint_.begin(), adjoint_.end(), 0.0);
    std::fill(adjoint_dot_.begin(), adjoint_dot_.end(), 0.0);
    adjoint_.back() = 1.0;
    for (std::size_t p = nodes_.size(); p-- > 0;) {
      const double bar = adjoint_[p];
      const double bar_dot = adjoint_dot_[p];
      if (bar == 0.0 && bar_dot == 0.0) continue;
      for (std::size_t k = 0; k < child_start_[p + 1] - child_start_[p]; ++k) {
        double d = 0.0;
        double d_dot = 0.0;
        if (!local_partial(p, k, true, &d, &d_dot)) return no_derivative(p);
        const std::size_t c = child_pos_[child_start_[p] + k];
        adjoint_[c] += bar * d;
        adjoint_dot_[c] += bar_dot * d + bar * d_dot;
      }
    }
    for (const auto& [slot, row_position] : sweep.entries) {
      const double h = adjoint_dot_[row_position];
      if (!std::isfinite(h)) return no_derivative(nodes_.size() - 1);
      (*out)[slot] += weight * h;
    }
  }
  return true;
}

}  // namespace sankhya::nlp
