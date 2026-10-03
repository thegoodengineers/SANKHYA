// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd):
// the iteration's state, shared by hsd.cpp (the Newton step), hsd_certificate.cpp (the rays)
// and hsd_run.cpp (the loop and the answer). See hsd.cpp for the method and its citations.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/resource_limits.hpp"
#include "la/ldl.hpp"
#include "la/scaling.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::ipm::hsd {

/// The default path's primal and dual regularization, step rule and iteration cap (ipm.cpp).
constexpr double kPrimalRegularization = 1e-8;
constexpr double kDualRegularization = 1e-10;
constexpr double kRegularizationRaise = 1e4;
constexpr int kMaxRegularizationRaises = 2;
constexpr double kStepToBoundary = 0.995;
constexpr Count kMaxIterations = 300;
constexpr int kRefinementSteps = 2;
/// ipm.cpp's #582 rule: converged in the scaled measures, the model-space ones are given this
/// many iterations to improve by kModelSpaceProgress before the point is reported feasible.
constexpr int kModelSpaceStallIterations = 3;
constexpr double kModelSpaceProgress = 0.9;
/// Within this factor of every tolerance, a stop on a failed direction reports the point.
constexpr double kNearlyConverged = 10.0;
/// The Farkas projection (project_farkas) is tried once tau < kProjectAt kappa, in at most
/// kProjectRounds rounds, each one least-squares solve with the columns it does not target
/// weighted kProjectWeight (and the same shift on the rows).
constexpr double kProjectAt = 1e-4;
constexpr int kProjectRounds = 8;
constexpr double kProjectWeight = 1e-10;
constexpr double kProjectAnchor = 1e-6;
constexpr double kProjectPin = 1e2;

enum class Verdict {
  kOptimal,
  kFeasible,
  kInfeasible,
  kUnbounded,
  kLimit,
  kNumericalError,
  kDeclined
};

struct Outcome {
  Verdict verdict = Verdict::kNumericalError;
  SolveStatus limit = SolveStatus::kIterationLimit;
  std::string message;
  Count iterations = 0;
  bool have_point = false;
  std::vector<double> x;            ///< n structurals of the point x / tau
  std::vector<double> y;            ///< m row multipliers y / tau, minimisation sense
  std::vector<double> d;            ///< n reduced costs, minimisation sense
  std::vector<double> certificate;  ///< Farkas (m) or ray (n), as the checker accepted it
  std::string proof;
};

class Homogeneous {
 public:
  Homogeneous(const Model& model, const Options& options, Logger& logger, SolveControl* control,
              const Timer& clock, const Scaling* scaling, const Model* original,
              bool feasibility_only)
      : model_(model),
        options_(options),
        logger_(logger),
        control_(control),
        clock_(clock),
        scaling_(scaling),
        original_(original),
        feasibility_only_(feasibility_only) {}

  Outcome run();

 private:
  void build();
  void residuals();
  [[nodiscard]] bool factorize();
  void solve_normal(std::vector<double>* rhs);
  void second_system();
  void direction(double eta, double r_mu_t);
  [[nodiscard]] bool direction_is_finite() const;
  [[nodiscard]] double max_step() const;
  [[nodiscard]] bool model_space_holds();
  [[nodiscard]] bool try_certificates(Outcome* out);
  [[nodiscard]] bool proves(bool farkas, const std::vector<double>& v, std::string* why) const;
  [[nodiscard]] std::vector<double> project_farkas(std::vector<double> y);
  Outcome finish(Verdict verdict, std::string message, Count iterations);
  void constraint_times(const std::vector<double>& v, std::vector<double>* out) const;
  void constraint_transpose_times(const std::vector<double>& w, std::vector<double>* out) const;
  [[nodiscard]] bool stop_requested() const {
    return (limits_.has_time_limit() && limits_.time_exhausted(clock_.elapsed_seconds())) ||
           (control_ != nullptr && control_->interruption_requested());
  }

  const Model& model_;
  const Options& options_;
  Logger& logger_;
  SolveControl* control_;
  const Timer& clock_;
  const Scaling* scaling_;
  const Model* original_;  ///< the caller's model when model_ is its scaled copy, else null
  const bool feasibility_only_;
  ResourceLimits limits_;
  SparseLdl::ShouldStop should_stop_;

  Index n_ = 0, m_ = 0, total_ = 0;
  Index bound_count_ = 0;
  std::vector<double> cost_, lower_, upper_, b_;
  std::vector<char> has_lower_, has_upper_, fixed_;
  double c_norm_ = 0.0;

  std::vector<double> x_, y_, sl_, zl_, su_, zu_;
  double tau_ = 1.0, kappa_ = 1.0;
  std::vector<double> r_p_, r_l_, r_u_, r_d_;
  double r_g_ = 0.0, mu_ = 0.0, mu0_ = 0.0;
  double cx_ = 0.0, dual_ray_ = 0.0;  ///< c'x and b'y + l'z_l - u'z_u, unnormalized
  double primal_inf_ = 0.0, dual_inf_ = 0.0, gap_ = 0.0, max_product_ = 0.0;
  double data_norm_ = 0.0;        ///< the largest finite bound and |b_i|
  double primal_data_inf_ = 0.0;  ///< the primal residual of x / tau over 1 + data_norm_
  double model_primal_ = 0.0, model_dual_ = 0.0;

  std::vector<double> theta_;
  SparseMatrix normal_lower_;
  SparseLdl ldl_;
  bool analyzed_ = false;
  double delta_ = kDualRegularization;
  std::int64_t max_factor_nonzeros_ = -1;

  std::vector<double> dx2_, dy2_;
  double g2_ = 0.0;
  std::vector<double> dx_, dy_, dsl_, dzl_, dsu_, dzu_;
  std::vector<double> rmu_l_, rmu_u_;
  double dtau_ = 0.0, dkappa_ = 0.0;
};

/// The vector with every entry at or below kIpmHsdCertificateRounding of its largest zeroed.
[[nodiscard]] std::vector<double> rounded(const std::vector<double>& v);

}  // namespace sankhya::ipm::hsd
