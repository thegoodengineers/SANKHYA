// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one expression compiled for repeated evaluation (NLP stage 1).
//
// WHY A SECOND EVALUATOR. ExpressionGraph evaluates over node ids, so every call sizes its
// work arrays to the WHOLE graph: a model with a thousand constraints of three columns each
// pays a thousand graph-sized fills per Hessian. An optimizer asks for the same expressions
// at a new point every iteration, so each is compiled once here into its own tape - the
// nodes it reaches, renumbered 0..L-1 in evaluation order, with its children's local
// positions - and evaluated over arrays of length L. The per-node rules are the SAME
// functions ExpressionGraph uses (derivative_rules.hpp), and tests compare the two.
//
// DERIVATIVES. The gradient is one reverse sweep (Griewank and Walther, "Evaluating
// Derivatives", 2nd ed., SIAM 2008, ch. 3). The Hessian is forward-over-reverse (ch. 5): for
// each column j that is the COLUMN of an entry of the structural lower-triangle pattern
// (sparsity.cpp), one tangent sweep along e_j and one second-order reverse sweep give column
// j of the Hessian. Columns with no nonlinear interaction are never swept, so a linear
// expression costs no Hessian work at all and x_1^2 + ... + x_n^2 costs n sweeps of length 1
// each rather than n sweeps of the whole sum.
//
// NOT THREAD SAFE PER OBJECT. The work arrays are members, so one CompiledExpression must not
// be evaluated from two threads at once; copy it per thread. It holds no pointer into the
// graph, so a copy outlives the graph it came from.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "nlp/expression.hpp"

namespace sankhya::nlp {

class CompiledExpression {
 public:
  CompiledExpression() = default;
  CompiledExpression(const ExpressionGraph& graph, ExprId root);

  /// The distinct columns the expression reads, ascending: the pattern of its gradient.
  [[nodiscard]] const std::vector<Index>& variables() const noexcept { return variables_; }

  /// Its structural Hessian pattern, (row, col) with row >= col, sorted by column then row.
  [[nodiscard]] const std::vector<std::pair<Index, Index>>& hessian_pattern() const noexcept {
    return pattern_;
  }

  /// The value at x (x has the graph's num_variables() entries).
  bool value(const std::vector<double>& x, double* out, Evaluation* error) const;

  /// The value, and the gradient as one entry per variables() in that order.
  bool gradient(const std::vector<double>& x, double* value, std::vector<double>* out,
                Evaluation* error) const;

  /// weight times the Hessian, ADDED into `out`, one entry per hessian_pattern() in that
  /// order (`out` is resized to the pattern if it is shorter).
  bool add_hessian(const std::vector<double>& x, double weight, std::vector<double>* out,
                   Evaluation* error) const;

 private:
  bool forward(const std::vector<double>& x, Evaluation* error) const;
  bool local_partial(std::size_t p, std::size_t k, bool with_tangent, double* d,
                     double* d_dot) const;

  std::vector<Node> nodes_;               ///< the tape, in evaluation order
  std::vector<ExprId> ids_;               ///< the graph id of each, for error reports
  std::vector<std::size_t> child_start_;  ///< children of p: child_pos_[start[p], start[p+1])
  std::vector<std::size_t> child_pos_;
  std::vector<Index> variables_;
  std::vector<std::size_t> variable_of_;  ///< for a kVariable position: its index in variables_
  std::vector<std::pair<Index, Index>> pattern_;
  /// The columns swept for the Hessian, and for each the pattern entries of that column
  /// with the local position of the ROW's variable node.
  struct Sweep {
    Index column = 0;
    std::vector<std::pair<std::size_t, std::size_t>> entries;  ///< (pattern slot, local row)
  };
  std::vector<Sweep> sweeps_;
  std::vector<std::size_t> position_of_variable_;  ///< variables_[k] -> its local position

  mutable std::vector<double> values_;
  mutable std::vector<double> tangent_;
  mutable std::vector<double> adjoint_;
  mutable std::vector<double> adjoint_dot_;
  mutable std::vector<double> child_;
};

}  // namespace sankhya::nlp
