// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd): the
// predictor-corrector loop, its stops, and the Solution it hands back. See hsd.cpp for the
// method and its citations.

#include "ipm/hsd.hpp"
#include "ipm/hsd_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/certificate.hpp"
#include "sankhya/tolerances.hpp"
#include "util/memory.hpp"

namespace sankhya::ipm {
namespace hsd {

Outcome Homogeneous::finish(Verdict verdict, std::string message, Count iterations) {
  Outcome out;
  out.verdict = verdict;
  out.iterations = iterations;
  out.message = std::move(message);
  const bool point = verdict == Verdict::kOptimal || verdict == Verdict::kFeasible ||
                     verdict == Verdict::kLimit;
  if (!point || !(tau_ > 0.0) || !std::isfinite(tau_)) return out;
  out.x.assign(static_cast<std::size_t>(n_), 0.0);
  out.d.assign(static_cast<std::size_t>(n_), 0.0);
  out.y.assign(static_cast<std::size_t>(m_), 0.0);
  for (Index i = 0; i < m_; ++i) {
    out.y[static_cast<std::size_t>(i)] = y_[static_cast<std::size_t>(i)] / tau_;
  }
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    out.x[u] = fixed_[u] != 0 ? lower_[u] : x_[u] / tau_;
    if (fixed_[u] != 0) {
      const ColumnView column = model_.matrix.column(j);
      double dot = 0.0;
      for (Index p = 0; p < column.size; ++p) {
        dot += column.values[p] * out.y[static_cast<std::size_t>(column.rows[p])];
      }
      out.d[u] = cost_[u] - dot;
    } else {
      out.d[u] = (zl_[u] - zu_[u]) / tau_;
    }
  }
  out.have_point =
      std::all_of(out.x.begin(), out.x.end(), [](double v) { return std::isfinite(v); }) &&
      std::all_of(out.y.begin(), out.y.end(), [](double v) { return std::isfinite(v); });
  if (!out.have_point && verdict != Verdict::kLimit) {
    out.verdict = Verdict::kNumericalError;
    out.message += "; the point x / tau is not finite";
  }
  return out;
}

Outcome Homogeneous::run() {
  limits_ = ResourceLimits(options_, logger_);
  should_stop_ = [this] { return stop_requested(); };
  max_factor_nonzeros_ = options_.get_int("ipm_max_factor_nonzeros");
  if (max_factor_nonzeros_ == 0) {
    max_factor_nonzeros_ =
        static_cast<std::int64_t>(auto_factor_budget(physical_memory_bytes()));
  }
  const std::int64_t ordering_entries = options_.get_int("ipm_max_ordering_entries");
  ldl_.set_ordering_budget(ordering_entries < 0 ? static_cast<std::size_t>(-1)
                           : ordering_entries == 0
                               ? auto_ordering_budget(physical_memory_bytes())
                               : static_cast<std::size_t>(ordering_entries));
  ldl_.set_supernodal(options_.get_bool("ipm_supernodal"));
  build();
  Count iterations = 0;
  int model_stalled = 0, tiny_steps = 0;
  double best_model_excess = std::numeric_limits<double>::infinity();
  for (;; ++iterations) {
    residuals();
    if (iterations == 0) mu0_ = mu_;
    if (!feasibility_only_) {
      logger_.iteration(iterations,
                        model_.sense_multiplier() * cx_ / tau_ + model_.objective_offset,
                        primal_inf_, dual_inf_, clock_.elapsed_seconds());
    }
    logger_.verbose(
        "hsd iteration {}{}: tau {:.3e}, kappa {:.3e}, mu {:.2e}, primal {:.2e}, dual {:.2e}, "
        "gap {:.2e}, worst product {:.2e}",
        iterations, feasibility_only_ ? " (feasibility)" : "", tau_, kappa_, mu_, primal_inf_,
        dual_inf_, gap_, max_product_);
    if (!std::isfinite(mu_) || !std::isfinite(tau_) || !std::isfinite(kappa_)) {
      return finish(Verdict::kNumericalError,
                    fmt::format("the iterate stopped being finite at iteration {}", iterations),
                    iterations);
    }
    // tau > kappa: the limit leans to the optimum. And the c = 0 solve, whose point is the
    // other half of an unboundedness claim, holds its residual against the data as well as
    // against the point: x / tau grows without bound as tau falls on an infeasible model,
    // and a violation of 1 against a point of norm 1e9 is 1e-9 relative to it.
    const bool scaled_converged =
        tau_ > kappa_ && primal_inf_ <= tol::kIpmHsdTolerance &&
        (!feasibility_only_ || primal_data_inf_ <= tol::kIpmHsdTolerance) &&
        (feasibility_only_ ||
         (dual_inf_ <= tol::kIpmHsdTolerance && gap_ <= tol::kIpmHsdTolerance &&
          max_product_ <= tol::kIpmHsdTolerance));
    if (scaled_converged) {
      if (model_space_holds()) {
        return finish(Verdict::kOptimal,
                      fmt::format("homogeneous self-dual embedding (#475): converged at "
                                  "iteration {} with tau {:.2e}, kappa {:.2e}",
                                  iterations, tau_, kappa_),
                      iterations);
      }
      const double excess = std::max(model_primal_ / tol::kPrimalFeasibility,
                                     model_dual_ / tol::kDualFeasibility);
      if (excess < kModelSpaceProgress * best_model_excess) {
        best_model_excess = excess;
        model_stalled = 0;
      } else if (++model_stalled >= kModelSpaceStallIterations) {
        return finish(
            Verdict::kFeasible,
            fmt::format("homogeneous self-dual embedding (#475): converged in the scaled space "
                        "but the point measures {:.1e} / {:.1e} in the model's units: a "
                        "feasible point, not a proof (#582)",
                        model_primal_, model_dual_),
            iterations);
      }
    }
    // THE RAY SIDE. kappa > tau: the limit is leaning to a certificate.
    Outcome proved;
    if (kappa_ > tau_ && try_certificates(&proved)) {
      proved.iterations = iterations;
      proved.message = fmt::format(
          "homogeneous self-dual embedding (#475): tau {:.2e}, kappa {:.2e} at iteration {}; "
          "{}",
          tau_, kappa_, iterations,
          proved.verdict == Verdict::kInfeasible ? "the dual ray is a Farkas certificate"
                                                 : "the primal ray proves unboundedness");
      return proved;
    }
    const auto stop_without_verdict = [&](Verdict verdict, const std::string& why) {
      // Last look at the ray side, whatever tau and kappa say.
      Outcome last;
      if (try_certificates(&last)) {
        last.iterations = iterations;
        last.message = fmt::format(
            "homogeneous self-dual embedding (#475): {}; tau {:.2e}, kappa {:.2e}, and the "
            "ray in hand is a certificate",
            why, tau_, kappa_);
        return last;
      }
      return finish(verdict,
                    fmt::format("homogeneous self-dual embedding (#475): {} (tau {:.2e}, kappa "
                                "{:.2e}, relative infeasibility {:.1e} / {:.1e}, gap {:.1e})",
                                why, tau_, kappa_, primal_inf_, dual_inf_, gap_),
                    iterations);
    };
    // A STEP THAT CANNOT BE TAKEN, near the end. As in ipm.cpp (#209), an iterate within a
    // decade of every tolerance that also holds in the model's units is the answer; otherwise
    // a point feasible to 1e-6 is reported as feasible, and anything else as the failure.
    const auto stop_on_failure = [&](const std::string& why) {
      const double slack = kNearlyConverged * tol::kIpmHsdTolerance;
      const bool nearly = tau_ > kappa_ && primal_inf_ <= slack && dual_inf_ <= slack &&
                          gap_ <= slack && max_product_ <= slack && !feasibility_only_;
      if (nearly && model_space_holds()) {
        return finish(Verdict::kOptimal,
                      fmt::format("homogeneous self-dual embedding (#475): {}; the iterate "
                                  "before it is within a decade of every tolerance",
                                  why),
                      iterations);
      }
      const bool usable =
          !feasibility_only_ && primal_inf_ <= 1e-6 && dual_inf_ <= 1e-6 && kappa_ < tau_;
      return stop_without_verdict(usable ? Verdict::kFeasible : Verdict::kNumericalError, why);
    };
    if (iterations >= kMaxIterations || limits_.iterations_exhausted(iterations)) {
      Outcome out = stop_without_verdict(
          Verdict::kLimit,
          fmt::format("iteration limit reached after {} iterations", iterations));
      out.limit = SolveStatus::kIterationLimit;
      return out;
    }
    if (stop_requested()) {
      const bool interrupted = control_ != nullptr && control_->interruption_requested();
      Outcome out = stop_without_verdict(Verdict::kLimit,
                                         interrupted ? "interrupted" : "time limit reached");
      out.limit = interrupted ? SolveStatus::kInterrupted : SolveStatus::kTimeLimit;
      return out;
    }
    if (mu_ <= tol::kIpmHsdMuFloor * mu0_) {
      return stop_on_failure("mu fell to its floor with neither tau nor kappa decided");
    }

    bool finite = false;
    for (int raise = 0; raise <= kMaxRegularizationRaises && !finite; ++raise) {
      if (raise > 0) delta_ *= kRegularizationRaise;
      if (!factorize()) {
        if (ldl_.stopped_early() || stop_requested()) {
          Outcome out = stop_without_verdict(Verdict::kLimit, "time limit reached");
          out.limit = SolveStatus::kTimeLimit;
          return out;
        }
        if (ldl_.factor_too_large() || ldl_.ordering_too_large() || ldl_.pattern_too_large()) {
          return finish(Verdict::kDeclined,
                        "homogeneous self-dual embedding (#475): the normal equations' factor "
                        "is over ipm_max_factor_nonzeros or ipm_max_ordering_entries",
                        iterations);
        }
        return stop_without_verdict(Verdict::kNumericalError,
                                    "the normal equations could not be factorized");
      }
      second_system();
      // PREDICTOR: eta = 1, every product to zero.
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        rmu_l_[u] = has_lower_[u] != 0 ? -sl_[u] * zl_[u] : 0.0;
        rmu_u_[u] = has_upper_[u] != 0 ? -su_[u] * zu_[u] : 0.0;
      }
      direction(1.0, -tau_ * kappa_);
      if (!direction_is_finite()) continue;
      const double alpha_aff = std::min(1.0, max_step());
      double mu_aff = (tau_ + alpha_aff * dtau_) * (kappa_ + alpha_aff * dkappa_);
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        if (has_lower_[u] != 0) {
          mu_aff += (sl_[u] + alpha_aff * dsl_[u]) * (zl_[u] + alpha_aff * dzl_[u]);
        }
        if (has_upper_[u] != 0) {
          mu_aff += (su_[u] + alpha_aff * dsu_[u]) * (zu_[u] + alpha_aff * dzu_[u]);
        }
      }
      mu_aff /= static_cast<double>(bound_count_ + 1);
      const double ratio = mu_ > 0.0 ? mu_aff / mu_ : 0.0;
      const double sigma = std::clamp(ratio * ratio * ratio, 0.0, 1.0);
      // CORRECTOR: eta = 1 - sigma, centring plus the predictor's second-order term.
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        if (has_lower_[u] != 0) {
          rmu_l_[u] = sigma * mu_ - sl_[u] * zl_[u] - dsl_[u] * dzl_[u];
        }
        if (has_upper_[u] != 0) {
          rmu_u_[u] = sigma * mu_ - su_[u] * zu_[u] - dsu_[u] * dzu_[u];
        }
      }
      direction(1.0 - sigma, sigma * mu_ - tau_ * kappa_ - dtau_ * dkappa_);
      finite = direction_is_finite();
    }
    if (!finite) {
      return stop_on_failure(
          fmt::format("the Newton direction was not finite at iteration {} "
                      "after {} regularization raise(s)",
                      iterations, kMaxRegularizationRaises));
    }
    const double alpha = std::min(1.0, kStepToBoundary * max_step());
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      x_[u] += alpha * dx_[u];
      if (has_lower_[u] != 0) {
        sl_[u] += alpha * dsl_[u];
        zl_[u] += alpha * dzl_[u];
      }
      if (has_upper_[u] != 0) {
        su_[u] += alpha * dsu_[u];
        zu_[u] += alpha * dzu_[u];
      }
    }
    for (std::size_t i = 0; i < y_.size(); ++i) y_[i] += alpha * dy_[i];
    tau_ += alpha * dtau_;
    kappa_ += alpha * dkappa_;
    tiny_steps = alpha < 1e-8 ? tiny_steps + 1 : 0;
    if (tiny_steps >= 5) {
      return stop_on_failure(fmt::format("the iteration stalled at iteration {} (step {:.1e})",
                                         iterations, alpha));
    }
  }
}

}  // namespace hsd

using hsd::Homogeneous;
using hsd::Outcome;
using hsd::Verdict;

void announce_homogeneous(const Options& options, Logger& logger) {
  std::string ignored;
  const auto note = [&ignored](const char* name) {
    ignored += ignored.empty() ? name : std::string(", ") + name;
  };
  if (options.get_bool("ipm_proximal_regularization")) note("ipm_proximal_regularization");
  if (options.get_bool("ipm_dense_columns")) note("ipm_dense_columns");
  if (options.get_string("ipm_normal_side") != "rows") note("ipm_normal_side");
  if (options.get_string("ipm_linear_solver") == "cudss") note("ipm_linear_solver=cudss");
  if (options.get_int("ipm_centrality_correctors") > 0) note("ipm_centrality_correctors");
  if (!ignored.empty()) {
    logger.info(
        "interior point: ipm_hsd factors the plain normal equations on the CPU; {} do(es) not "
        "apply and is ignored",
        ignored);
  }
}

void keep_only_a_proved_certificate(const Model& model, Solution* solution, Logger& logger) {
  std::string why;
  if (solution->status == SolveStatus::kInfeasible && !solution->farkas_dual.empty()) {
    if (farkas_proves_infeasible(model, solution->farkas_dual, &why)) return;
    solution->farkas_dual.clear();
  } else if (solution->status == SolveStatus::kUnbounded && !solution->primal_ray.empty()) {
    if (ray_proves_unbounded(model, solution->primal_ray, &why)) return;
    solution->primal_ray.clear();
  } else {
    return;
  }
  logger.warning(
      "interior point (#475): the certificate does not hold in the model's units: {}", why);
  solution->message += fmt::format(
      "; the certificate held on the scaled model but not in the model's units ({}), so the "
      "verdict is withdrawn",
      why);
  solution->status = SolveStatus::kNumericalError;
  std::fill(solution->col_value.begin(), solution->col_value.end(), 0.0);
  solution->recompute_quality(model);
}

Solution solve_homogeneous(const Model& model, const Options& options, Logger& logger,
                           SolveControl* control, const Timer& clock, const Scaling* scaling,
                           const Model* original) {
  logger.info("Interior point (homogeneous self-dual embedding, #475): {} rows, {} columns",
              model.num_rows(), model.num_cols());
  logger.begin_iteration_table();
  Homogeneous main(model, options, logger, control, clock, scaling, original, false);
  Outcome outcome = main.run();
  if (!outcome.proof.empty() && !outcome.certificate.empty()) {
    logger.verbose("interior point (#475): the checker's reading of the certificate: {}",
                   outcome.proof);
  }

  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "ipm";
  solution.iterations = outcome.iterations;
  solution.col_status.clear();
  solution.row_status.clear();
  const double sense = model.sense_multiplier();
  const double worst_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  solution.dual_bound = worst_bound;

  if (outcome.verdict == Verdict::kUnbounded) {
    // THE OTHER HALF OF AN UNBOUNDEDNESS CLAIM: a feasible point. The embedding with c = 0
    // ends at one (tau > 0) or at a Farkas certificate (kappa > 0, primal and dual both
    // infeasible).
    Homogeneous feasibility(model, options, logger, control, clock, scaling, original, true);
    Outcome point = feasibility.run();
    if (point.verdict == Verdict::kOptimal && point.have_point) {
      solution.status = SolveStatus::kUnbounded;
      solution.primal_ray = std::move(outcome.certificate);
      solution.col_value = std::move(point.x);
      solution.message = fmt::format(
          "{}; the feasible point it starts from is the c = 0 embedding's ({} iterations)",
          outcome.message, point.iterations);
    } else if (point.verdict == Verdict::kInfeasible) {
      solution.status = SolveStatus::kInfeasible;
      solution.farkas_dual = std::move(point.certificate);
      solution.message = fmt::format(
          "{}; the c = 0 embedding then proved the model infeasible as well", outcome.message);
    } else {
      solution.status = SolveStatus::kNumericalError;
      solution.message = fmt::format(
          "{}, but no feasible point to start it from was found ({}), so unboundedness is not "
          "claimed",
          outcome.message, point.message);
    }
    solution.iterations += point.iterations;
  } else if (outcome.verdict == Verdict::kInfeasible) {
    solution.status = SolveStatus::kInfeasible;
    solution.farkas_dual = std::move(outcome.certificate);
    solution.message = outcome.message;
  } else {
    solution.message = outcome.message;
    switch (outcome.verdict) {
      case Verdict::kOptimal: solution.status = SolveStatus::kOptimal; break;
      case Verdict::kFeasible: solution.status = SolveStatus::kFeasible; break;
      case Verdict::kLimit: solution.status = outcome.limit; break;
      case Verdict::kDeclined: solution.status = SolveStatus::kNumericalError; break;
      default: solution.status = SolveStatus::kNumericalError; break;
    }
    if (outcome.have_point && solution.status != SolveStatus::kNumericalError) {
      solution.col_value = std::move(outcome.x);
      for (std::size_t j = 0; j < outcome.d.size(); ++j)
        solution.col_dual[j] = sense * outcome.d[j];
      for (std::size_t i = 0; i < outcome.y.size(); ++i)
        solution.row_dual[i] = sense * outcome.y[i];
      if (solution.status == SolveStatus::kOptimal) {
        solution.dual_bound = model.evaluate_objective(solution.col_value.data());
      }
    }
  }
  logger.info("IPM (homogeneous self-dual): {} iterations, {}", solution.iterations,
              to_string(solution.status));
  solution.solve_seconds = clock.elapsed_seconds();
  solution.recompute_quality(model);
  return solution;
}

}  // namespace sankhya::ipm
