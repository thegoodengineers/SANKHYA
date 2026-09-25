// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the equality-constrained form the NLP interior point iterates on (NLP stage 2).
// See barrier_nlp.hpp.

#include "nlp/barrier_nlp.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya::nlp {

// ---- SlackedNlp -----------------------------------------------------------------------------

SlackedNlp::SlackedNlp(const NlpProblem* problem) : p_(problem) {
  n_x_ = p_->num_variables();
  m_ = p_->num_constraints();
  lower_ = p_->x_lower();
  upper_ = p_->x_upper();
  Index next = n_x_;
  for (Index i = 0; i < m_; ++i) {
    const double lo = p_->g_lower()[static_cast<std::size_t>(i)];
    const double hi = p_->g_upper()[static_cast<std::size_t>(i)];
    if (lo == hi && std::isfinite(lo)) {
      slack_of_.push_back(-1);
      rhs_.push_back(lo);
    } else {
      slack_of_.push_back(next++);
      rhs_.push_back(0.0);
      lower_.push_back(lo);
      upper_.push_back(hi);
    }
  }
  n_ = next;

  jac_starts_.assign(1, 0);
  const auto& ps = p_->jacobian_starts();
  const auto& pc = p_->jacobian_columns();
  for (Index i = 0; i < m_; ++i) {
    for (Index k = ps[static_cast<std::size_t>(i)]; k < ps[static_cast<std::size_t>(i) + 1];
         ++k) {
      jac_cols_.push_back(pc[static_cast<std::size_t>(k)]);
    }
    if (slack_of_[static_cast<std::size_t>(i)] >= 0) {
      jac_cols_.push_back(slack_of_[static_cast<std::size_t>(i)]);
    }
    jac_starts_.push_back(static_cast<Index>(jac_cols_.size()));
  }

  // The problem's Hessian pattern, with the diagonal added where it is missing.
  const auto& hs = p_->hessian_starts();
  const auto& hr = p_->hessian_rows();
  hess_starts_.assign(1, 0);
  problem_slot_.assign(hr.size(), 0);
  for (Index j = 0; j < n_; ++j) {
    const bool in_problem = j < n_x_;
    const Index begin = in_problem ? hs[static_cast<std::size_t>(j)] : 0;
    const Index end = in_problem ? hs[static_cast<std::size_t>(j) + 1] : 0;
    if (begin == end || hr[static_cast<std::size_t>(begin)] != j) hess_rows_.push_back(j);
    for (Index k = begin; k < end; ++k) {
      problem_slot_[static_cast<std::size_t>(k)] = static_cast<Index>(hess_rows_.size());
      hess_rows_.push_back(hr[static_cast<std::size_t>(k)]);
    }
    hess_starts_.push_back(static_cast<Index>(hess_rows_.size()));
  }
}

void SlackedNlp::split(const Vec& w) const {
  x_.assign(w.begin(), w.begin() + n_x_);
}

bool SlackedNlp::objective(const Vec& w, double* f, Evaluation* error) const {
  split(w);
  return p_->objective(x_, f, error);
}

bool SlackedNlp::gradient(const Vec& w, double* f, Vec* g, Evaluation* error) const {
  split(w);
  if (!p_->objective_gradient(x_, f, &scratch_, error)) return false;
  g->assign(static_cast<std::size_t>(n_), 0.0);
  std::copy(scratch_.begin(), scratch_.end(), g->begin());
  return true;
}

bool SlackedNlp::constraints(const Vec& w, Vec* c, Evaluation* error) const {
  split(w);
  if (!p_->constraints(x_, c, error)) return false;
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(i);
    (*c)[u] -= rhs_[u];
    if (slack_of_[u] >= 0) (*c)[u] -= w[static_cast<std::size_t>(slack_of_[u])];
  }
  return true;
}

bool SlackedNlp::jacobian(const Vec& w, Vec* values, Evaluation* error) const {
  split(w);
  if (!p_->jacobian(x_, &scratch_, error)) return false;
  values->resize(jac_cols_.size());
  const auto& ps = p_->jacobian_starts();
  std::size_t out = 0;
  for (Index i = 0; i < m_; ++i) {
    for (Index k = ps[static_cast<std::size_t>(i)]; k < ps[static_cast<std::size_t>(i) + 1];
         ++k) {
      (*values)[out++] = scratch_[static_cast<std::size_t>(k)];
    }
    if (slack_of_[static_cast<std::size_t>(i)] >= 0) (*values)[out++] = -1.0;
  }
  return true;
}

bool SlackedNlp::hessian(const Vec& w, double sigma, const Vec& lambda, Vec* values,
                         Evaluation* error) const {
  split(w);
  if (!p_->hessian(x_, sigma, lambda, &scratch_, error)) return false;
  values->assign(hess_rows_.size(), 0.0);
  for (std::size_t k = 0; k < scratch_.size(); ++k) {
    (*values)[static_cast<std::size_t>(problem_slot_[k])] = scratch_[k];
  }
  return true;
}

// ---- RestorationNlp -------------------------------------------------------------------------

RestorationNlp::RestorationNlp(const BarrierNlp* inner, Vec reference, double zeta, double rho)
    : inner_(inner), reference_(std::move(reference)), zeta_(zeta), rho_(rho) {
  inner_n_ = inner_->n();
  m_ = inner_->m();
  n_ = inner_n_ + 2 * m_;
  lower_ = inner_->lower();
  upper_ = inner_->upper();
  lower_.resize(static_cast<std::size_t>(n_), 0.0);
  upper_.resize(static_cast<std::size_t>(n_), kInfinity);
  // d_j = min(1, 1 / |r_j|), eq. (30): the proximity term is relative for large entries.
  for (const double r : reference_) {
    const double d = std::fabs(r) > 1.0 ? 1.0 / std::fabs(r) : 1.0;
    weight_.push_back(d * d);
  }

  jac_starts_.assign(1, 0);
  for (Index i = 0; i < m_; ++i) {
    for (Index k = inner_->jac_starts()[static_cast<std::size_t>(i)];
         k < inner_->jac_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
      jac_cols_.push_back(inner_->jac_cols()[static_cast<std::size_t>(k)]);
    }
    jac_cols_.push_back(inner_n_ + i);       // p_i
    jac_cols_.push_back(inner_n_ + m_ + i);  // n_i
    jac_starts_.push_back(static_cast<Index>(jac_cols_.size()));
  }
  // The inner pattern, which holds its diagonal, then the diagonal of p and n: the inner
  // slots are a prefix, so the inner Hessian copies across unchanged.
  hess_starts_ = inner_->hess_starts();
  hess_rows_ = inner_->hess_rows();
  for (Index j = inner_n_; j < n_; ++j) {
    hess_rows_.push_back(j);
    hess_starts_.push_back(static_cast<Index>(hess_rows_.size()));
  }
  diagonal_slot_.assign(static_cast<std::size_t>(n_), 0);
  for (Index j = 0; j < n_; ++j) {
    diagonal_slot_[static_cast<std::size_t>(j)] = hess_starts_[static_cast<std::size_t>(j)];
  }
}

void RestorationNlp::split(const Vec& w) const {
  w_inner_.assign(w.begin(), w.begin() + inner_n_);
}

bool RestorationNlp::objective(const Vec& w, double* f, Evaluation* /*error*/) const {
  double v = 0.0;
  for (Index j = inner_n_; j < n_; ++j) v += rho_ * w[static_cast<std::size_t>(j)];
  for (Index j = 0; j < inner_n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = w[u] - reference_[u];
    v += 0.5 * zeta_ * weight_[u] * d * d;
  }
  *f = v;
  return true;
}

bool RestorationNlp::gradient(const Vec& w, double* f, Vec* g, Evaluation* error) const {
  (void)objective(w, f, error);
  g->assign(static_cast<std::size_t>(n_), rho_);
  for (Index j = 0; j < inner_n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    (*g)[u] = zeta_ * weight_[u] * (w[u] - reference_[u]);
  }
  return true;
}

bool RestorationNlp::constraints(const Vec& w, Vec* c, Evaluation* error) const {
  split(w);
  if (!inner_->constraints(w_inner_, c, error)) return false;
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(i);
    (*c)[u] += -w[static_cast<std::size_t>(inner_n_ + i)] +
               w[static_cast<std::size_t>(inner_n_ + m_ + i)];
  }
  return true;
}

bool RestorationNlp::jacobian(const Vec& w, Vec* values, Evaluation* error) const {
  split(w);
  if (!inner_->jacobian(w_inner_, &scratch_, error)) return false;
  values->resize(jac_cols_.size());
  std::size_t out = 0;
  for (Index i = 0; i < m_; ++i) {
    for (Index k = inner_->jac_starts()[static_cast<std::size_t>(i)];
         k < inner_->jac_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
      (*values)[out++] = scratch_[static_cast<std::size_t>(k)];
    }
    (*values)[out++] = -1.0;
    (*values)[out++] = 1.0;
  }
  return true;
}

bool RestorationNlp::hessian(const Vec& w, double sigma, const Vec& lambda, Vec* values,
                             Evaluation* error) const {
  split(w);
  // The inner problem's objective is not part of this one: only its constraints' curvature.
  if (!inner_->hessian(w_inner_, 0.0, lambda, &scratch_, error)) return false;
  values->assign(hess_rows_.size(), 0.0);
  std::copy(scratch_.begin(), scratch_.end(), values->begin());
  for (Index j = 0; j < inner_n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    (*values)[static_cast<std::size_t>(diagonal_slot_[u])] += sigma * zeta_ * weight_[u];
  }
  return true;
}

}  // namespace sankhya::nlp
