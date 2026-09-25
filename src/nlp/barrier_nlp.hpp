// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the equality-constrained form the NLP interior point iterates on (NLP stage 2).
//
// THE FORM. Wachter and Biegler, "On the implementation of an interior-point filter
// line-search algorithm for large-scale nonlinear programming", Math. Programming 106(1)
// (2006), state the method for
//
//     minimize f(w)  subject to  c(w) = 0,  w_L <= w <= w_U                       (eq. 1)
//
// and reach it from g_L <= g(x) <= g_U by a slack per inequality row: c_i = g_i(x) - s_i with
// g_L,i <= s_i <= g_U,i, and c_i = g_i(x) - g_L,i for an equality row (SlackedNlp). The
// feasibility restoration phase (sec. 3.3) is the same method on another problem of this
// form (RestorationNlp), so the interior point is written once, against this interface.
//
// Every Hessian pattern here CONTAINS ITS WHOLE DIAGONAL, present even where the value is
// zero, because the interior point adds the barrier term Sigma and the inertia correction
// delta_w there; a missing diagonal entry would be a missing pivot.
#pragma once

#include <vector>

#include "nlp/expression.hpp"
#include "nlp/nlp_problem.hpp"

namespace sankhya::nlp {

using Vec = std::vector<double>;

class BarrierNlp {
 public:
  virtual ~BarrierNlp() = default;

  [[nodiscard]] Index n() const noexcept { return n_; }
  [[nodiscard]] Index m() const noexcept { return m_; }
  [[nodiscard]] const Vec& lower() const noexcept { return lower_; }
  [[nodiscard]] const Vec& upper() const noexcept { return upper_; }
  /// The Jacobian of c, row-wise, columns ascending within a row.
  [[nodiscard]] const std::vector<Index>& jac_starts() const noexcept { return jac_starts_; }
  [[nodiscard]] const std::vector<Index>& jac_cols() const noexcept { return jac_cols_; }
  /// The Hessian of the Lagrangian, lower triangle by column, rows ascending, diagonal present.
  [[nodiscard]] const std::vector<Index>& hess_starts() const noexcept { return hess_starts_; }
  [[nodiscard]] const std::vector<Index>& hess_rows() const noexcept { return hess_rows_; }

  virtual bool objective(const Vec& w, double* f, Evaluation* error) const = 0;
  virtual bool gradient(const Vec& w, double* f, Vec* g, Evaluation* error) const = 0;
  virtual bool constraints(const Vec& w, Vec* c, Evaluation* error) const = 0;
  virtual bool jacobian(const Vec& w, Vec* values, Evaluation* error) const = 0;
  /// sigma d2f + sum_i lambda_i d2c_i, in the hessian pattern's layout.
  virtual bool hessian(const Vec& w, double sigma, const Vec& lambda, Vec* values,
                       Evaluation* error) const = 0;

 protected:
  Index n_ = 0;
  Index m_ = 0;
  Vec lower_, upper_;
  std::vector<Index> jac_starts_, jac_cols_;
  std::vector<Index> hess_starts_, hess_rows_;
};

/// NlpProblem with a slack per inequality row: w = (x, s).
class SlackedNlp final : public BarrierNlp {
 public:
  /// `problem` must outlive this object.
  explicit SlackedNlp(const NlpProblem* problem);

  /// Row i's slack column, or -1 for an equality row.
  [[nodiscard]] Index slack_of(Index row) const {
    return slack_of_[static_cast<std::size_t>(row)];
  }
  [[nodiscard]] const NlpProblem& problem() const noexcept { return *p_; }

  bool objective(const Vec& w, double* f, Evaluation* error) const override;
  bool gradient(const Vec& w, double* f, Vec* g, Evaluation* error) const override;
  bool constraints(const Vec& w, Vec* c, Evaluation* error) const override;
  bool jacobian(const Vec& w, Vec* values, Evaluation* error) const override;
  bool hessian(const Vec& w, double sigma, const Vec& lambda, Vec* values,
               Evaluation* error) const override;

 private:
  void split(const Vec& w) const;  ///< w's first n_x entries into x_

  const NlpProblem* p_;
  Index n_x_ = 0;
  std::vector<Index> slack_of_;
  Vec rhs_;                          ///< g_L for an equality row, 0 otherwise
  std::vector<Index> problem_slot_;  ///< NlpProblem Hessian slot -> this pattern's slot
  mutable Vec x_, scratch_;
};

/// The feasibility restoration problem of sec. 3.3 over `inner`:
///
///     minimize   rho sum_i (p_i + n_i) + zeta/2 sum_j d_j^2 (w_j - r_j)^2
///     subject to c(w) - p + n = 0,  w_L <= w <= w_U,  p, n >= 0
///
/// with r the point restoration started from and d_j = min(1, 1 / |r_j|) (eq. 29-30).
/// Variables are (w, p, n).
class RestorationNlp final : public BarrierNlp {
 public:
  RestorationNlp(const BarrierNlp* inner, Vec reference, double zeta, double rho);

  [[nodiscard]] const BarrierNlp& inner() const noexcept { return *inner_; }

  bool objective(const Vec& w, double* f, Evaluation* error) const override;
  bool gradient(const Vec& w, double* f, Vec* g, Evaluation* error) const override;
  bool constraints(const Vec& w, Vec* c, Evaluation* error) const override;
  bool jacobian(const Vec& w, Vec* values, Evaluation* error) const override;
  bool hessian(const Vec& w, double sigma, const Vec& lambda, Vec* values,
               Evaluation* error) const override;

 private:
  void split(const Vec& w) const;

  const BarrierNlp* inner_;
  Vec reference_, weight_;  ///< r and d^2
  double zeta_ = 0.0;
  double rho_ = 0.0;
  Index inner_n_ = 0;
  std::vector<Index> diagonal_slot_;  ///< slot of (j, j) for every variable
  mutable Vec w_inner_, scratch_;
};

}  // namespace sankhya::nlp
