// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the state of one run of the filter interior point (NLP stage 2), shared by
// filter_ipm.cpp (setup, the main loop, the optimality error), filter_ipm_step.cpp (the
// Newton step, the inertia correction and the filter line search) and
// filter_ipm_restoration.cpp (the restoration phase). Internal; see filter_ipm.hpp.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "nlp/filter_ipm.hpp"
#include "nlp/nlp_kkt.hpp"

namespace sankhya::nlp::detail {

class FilterMethod {
 public:
  FilterMethod(const BarrierNlp& p, const IpmSettings& s, const IpmHooks& h);
  IpmResult run(IpmIterate start);

 private:
  enum class StepOutcome { kAccepted, kRestoration, kStopped };

  // filter_ipm.cpp
  [[nodiscard]] double theta(const Vec& c) const;
  [[nodiscard]] double barrier(const Vec& w, double f) const;
  void jacobian_transpose_times(const Vec& y, Vec* out) const;
  void dual_residual(Vec* r) const;
  [[nodiscard]] double error(double mu) const;
  [[nodiscard]] bool in_filter(double theta, double phi) const;
  bool evaluate_all(Evaluation* error);
  void push_interior();
  void least_squares_multipliers();
  [[nodiscard]] IpmResult finish(IpmExit exit, std::string message) const;

  // filter_ipm_step.cpp
  StepOutcome step();
  bool factorize_with_inertia_correction(const Vec& sigma);
  [[nodiscard]] double fraction_to_boundary(const Vec& dw) const;
  [[nodiscard]] double fraction_to_boundary_dual(const Vec& dzl, const Vec& dzu) const;
  void safeguard_multipliers();

  // filter_ipm_restoration.cpp
  IpmExit restoration(std::string* message);

  const BarrierNlp& p_;
  IpmSettings s_;
  const IpmHooks& h_;
  Index n_ = 0;
  Index m_ = 0;
  Vec lo_, up_;
  std::vector<char> has_lo_, has_up_, fixed_;

  Vec w_, lam_, zl_, zu_;
  double mu_ = 0.0;
  double tau_ = 0.0;
  double f_ = 0.0;
  Vec grad_, c_, jac_, hess_;
  NlpKkt kkt_;
  double theta_max_ = 0.0;
  double theta_min_ = 0.0;
  std::vector<std::pair<double, double>> filter_;
  double delta_w_last_ = 0.0;
  Count iterations_ = 0;
};

}  // namespace sankhya::nlp::detail
