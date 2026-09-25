// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a nonlinear model as the smooth NLP an optimizer iterates on (NLP stage 1).
//
// THE FORM. Every NonlinearModel is presented as
//
//     minimize    f(x)
//     subject to  g_lower <= g(x) <= g_upper
//                 x_lower <= x <= x_upper
//
// with f = s * (offset + c'x + 0.5 x'Qx + objective expression), s = +1 to minimise and -1 to
// maximise, and g = [ A x ; the nonlinear constraint expressions ], the model's linear rows
// FIRST and its nonlinear constraints after, in their own order. That is the order of the
// multipliers an engine reports, so Solution::row_dual has num_rows + constraints.size()
// entries in the same order.
//
// THE DERIVATIVES are exact (tape.hpp): the gradient of f, the Jacobian of g on a FIXED
// row-wise pattern, and the Hessian of the Lagrangian
//
//     W(x, sigma, lambda) = sigma * d2 f(x) + sum_i lambda_i d2 g_i(x)
//
// on a FIXED lower-triangle pattern, the union of every expression's structural pattern
// (sparsity.cpp) and of Q. "Fixed" is what a sparse factorization needs: the symbolic
// analysis runs once and every iteration refills the same positions. This is the interface of
// Wachter and Biegler, "On the implementation of an interior-point filter line-search
// algorithm for large-scale nonlinear programming", Math. Programming 106(1), 2006 -
// written from the paper, not from any solver's code.
//
// AN EVALUATION FAILURE IS NOT A VIOLATION. log(x) at x = -1 is a point outside the
// function's domain; every call returns false there with the Evaluation saying where, and an
// optimizer treats that as a step to shorten, never as a number.
//
// LIFETIME AND THREADS. Holds a copy of what it needs, not a pointer into the model. The
// compiled expressions carry work arrays (tape.hpp), so one NlpProblem is used by one thread;
// copy it for another.
#pragma once

#include <string>
#include <vector>

#include "nlp/nonlinear_model.hpp"
#include "nlp/tape.hpp"

namespace sankhya::nlp {

class NlpProblem {
 public:
  /// Build from a model that validates. Returns empty on success, else why not.
  [[nodiscard]] static std::string build(const NonlinearModel& model, NlpProblem* out);

  [[nodiscard]] Index num_variables() const noexcept { return n_; }
  [[nodiscard]] Index num_constraints() const noexcept { return m_; }
  /// How many of the constraints are the model's linear rows (the first ones).
  [[nodiscard]] Index num_linear_rows() const noexcept { return linear_rows_; }
  [[nodiscard]] const std::vector<double>& x_lower() const noexcept { return x_lower_; }
  [[nodiscard]] const std::vector<double>& x_upper() const noexcept { return x_upper_; }
  [[nodiscard]] const std::vector<double>& g_lower() const noexcept { return g_lower_; }
  [[nodiscard]] const std::vector<double>& g_upper() const noexcept { return g_upper_; }
  /// +1 when the model minimises, -1 when it maximises: model objective = sense() * f.
  [[nodiscard]] double sense() const noexcept { return sense_; }

  /// f(x), the MINIMISATION form.
  bool objective(const std::vector<double>& x, double* value, Evaluation* error) const;
  /// f(x) and its gradient, dense, n entries.
  bool objective_gradient(const std::vector<double>& x, double* value,
                          std::vector<double>* gradient, Evaluation* error) const;
  /// g(x), m entries.
  bool constraints(const std::vector<double>& x, std::vector<double>* values,
                   Evaluation* error) const;

  /// The Jacobian of g, row-wise: row i's columns are jacobian_columns()[starts[i],
  /// starts[i+1]), ascending, and jacobian() fills the values in that layout.
  [[nodiscard]] const std::vector<Index>& jacobian_starts() const noexcept { return j_starts_; }
  [[nodiscard]] const std::vector<Index>& jacobian_columns() const noexcept { return j_cols_; }
  bool jacobian(const std::vector<double>& x, std::vector<double>* values,
                Evaluation* error) const;

  /// The Hessian of the Lagrangian, lower triangle by column: column j's rows are
  /// hessian_rows()[starts[j], starts[j+1]), ascending, row >= j.
  [[nodiscard]] const std::vector<Index>& hessian_starts() const noexcept { return h_starts_; }
  [[nodiscard]] const std::vector<Index>& hessian_rows() const noexcept { return h_rows_; }
  /// sigma * d2 f + sum_i lambda[i] d2 g_i at x, in the hessian pattern's layout. `lambda`
  /// has m entries; the linear rows' multipliers are read and contribute nothing.
  bool hessian(const std::vector<double>& x, double sigma, const std::vector<double>& lambda,
               std::vector<double>* values, Evaluation* error) const;

 private:
  /// The slot of (row, col), row >= col, in the Hessian pattern; -1 if absent.
  [[nodiscard]] Index hessian_slot(Index row, Index col) const;

  Index n_ = 0;
  Index m_ = 0;
  Index linear_rows_ = 0;
  double sense_ = 1.0;
  double offset_ = 0.0;
  std::vector<double> x_lower_, x_upper_, g_lower_, g_upper_;
  std::vector<double> cost_;  ///< c, in the model's own sense
  // Q, lower triangle by column, in the model's own sense.
  std::vector<Index> q_starts_, q_rows_;
  std::vector<double> q_values_;
  std::vector<Index> q_slot_;  ///< each Q entry's Hessian slot
  bool has_objective_expression_ = false;
  CompiledExpression objective_expression_;
  std::vector<Index> objective_slot_;  ///< its pattern -> Hessian slot
  // The linear rows, row-wise.
  std::vector<Index> a_starts_, a_cols_;
  std::vector<double> a_values_;
  std::vector<CompiledExpression> expressions_;  ///< the nonlinear constraints
  std::vector<std::vector<Index>> expression_slot_;
  std::vector<Index> j_starts_, j_cols_;
  std::vector<Index> h_starts_, h_rows_;
  mutable std::vector<double> scratch_;
  mutable std::vector<double> hess_scratch_;
};

}  // namespace sankhya::nlp
