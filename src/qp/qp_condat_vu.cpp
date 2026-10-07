// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex QP by a primal-dual proximal method with a forward step.
//
// References
//   Condat, "A primal-dual splitting method for convex optimization involving Lipschitzian,
//     proximable and linear composite terms", J. Optimization Theory and Applications 158
//     (2013).
//   Vu, "A splitting algorithm for dual monotone inclusions involving cocoercive operators",
//     Advances in Computational Mathematics 38 (2013).
//   Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//     applications to imaging", JMIV 40 (2011) - the Q = 0 special case, which is what
//     src/pdhg/ implements for LP.
//
// WHY THIS SHAPE. The LP engine already solves
//
//     min_x max_y   c'x + y'Ax - sigma_C(y),      x in box
//
// by alternating a projected gradient step in x with a proximal step in y. A convex QP adds
// a SMOOTH term, 0.5 x'Qx, whose gradient Qx can simply be added to the primal step - that is
// exactly the Condat-Vu extension of Chambolle-Pock, and it is why the issue recommended this
// route over an active-set or interior-point QP: the projection and Moreau machinery carry
// over unchanged and only the gradient and the step-size rule change.
//
//     x_{k+1} = proj_box( x_k - tau (c + Q x_k + A' y_k) )
//     xbar    = 2 x_{k+1} - x_k
//     y_{k+1} = prox_{sigma sigma_C}( y_k + sigma A xbar )
//
// CONVERGENCE CONDITION. Condat-Vu requires
//
//     1/tau  -  sigma ||A||^2  >=  L / 2,        L = ||Q||_2
//
// Chambolle-Pock's LP condition is the L = 0 case. Both norms are estimated by power
// iteration rather than bounded by a Frobenius or row-sum estimate: a loose bound on ||A||
// forces a small tau and costs iterations directly, and the estimate is computed once.
//
// THIS IS A SEPARATE FILE FROM src/pdhg/ ON PURPOSE. That engine is tuned and verified
// against the whole Netlib set; folding a quadratic term through its adaptive step size and
// restart logic would put the LP path at risk to save duplication in a first QP engine.
// ENGINEERING_RULES.md is explicit that a wrong answer scores zero, and the LP path is the
// thing most of the project's evidence rests on.
//
// ACCELERATION (#493; the PID weight on by default, Halpern off). qp_halpern replaces
// z <- T z by the restarted, reflected Halpern iteration of Lu & Yang (arXiv:2407.16144),
// carried to this operator as their PDQP (arXiv:2311.07710) carries PDHG to QP;
// qp_primal_weight_pid moves the primal weight sigma / tau at each restart by a PID controller
// on the log of the primal-dual movement ratio (Lu, Peng & Yang, arXiv:2507.14051). Restarts
// are decided on the fixed-point residual ||T z - z||. The machinery and its derivation from
// Condat's relaxation bound are in qp_first_order_accel.hpp. Convergence is checked, and the
// answer reported, at T z - a point of the box - never at the anchored combination.
//
// WHERE THE ARITHMETIC RUNS (#493, the GPU half). The step, the fixed-point residual, the
// Halpern blend and the restart bookkeeping are a QpOperator (qp_operator.hpp): the host
// implementation is the arithmetic this file always ran, and qp_gpu=true asks for the device
// one (src/gpu/qp_device.cu), which keeps the iterate on the card and returns scalars. This
// loop is the same either way; the residual evaluation every 50 iterations reads the
// evaluated point back and runs here, so both paths report through one code.

#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "../gpu/qp_device.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "convexity.hpp"
#include "qp_first_order_accel.hpp"
#include "qp_operator.hpp"

namespace sankhya::qp {
namespace {

/// Componentwise projection onto [lower, upper], tolerant of infinite sides.
[[nodiscard]] double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// Power iteration for the largest singular value of A, i.e. ||A||_2.
///
/// Twenty iterations from the configured seed. This is used only to pick a step size, so a
/// few percent of underestimate would break the convergence condition - the result is
/// therefore inflated by a small margin below rather than used raw. The seed comes from
/// random_seed rather than from a literal so that every randomised decision in the solver
/// answers to one option (#288); the default reproduces the previous starting vector.
[[nodiscard]] double spectral_norm(const SparseMatrix& matrix, Index rows, Index cols,
                                   unsigned seed) {
  if (rows == 0 || cols == 0 || matrix.num_nonzeros() == 0) return 0.0;
  std::mt19937 rng(26119u + seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);

  std::vector<double> v(static_cast<std::size_t>(cols));
  for (double& value : v) value = unit(rng);
  std::vector<double> av(static_cast<std::size_t>(rows), 0.0);

  double estimate = 0.0;
  for (int iteration = 0; iteration < 20; ++iteration) {
    double norm = 0.0;
    for (const double value : v) norm += value * value;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return 0.0;
    for (double& value : v) value /= norm;

    std::fill(av.begin(), av.end(), 0.0);
    matrix.multiply_add(v.data(), av.data());
    std::fill(v.begin(), v.end(), 0.0);
    matrix.transpose_multiply_add(av.data(), v.data());

    double next = 0.0;
    for (const double value : v) next += value * value;
    estimate = std::sqrt(std::sqrt(next));  // ||A^T A||^{1/2} converges to ||A||
  }
  return estimate;
}

/// Largest eigenvalue of the symmetric Q held as a lower triangle, by the same route.
[[nodiscard]] double hessian_norm(const Model& model, unsigned seed) {
  const Index n = model.num_cols();
  if (model.hessian.num_nonzeros() == 0 || n == 0) return 0.0;

  std::mt19937 rng(20260826u + seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> v(static_cast<std::size_t>(n));
  for (double& value : v) value = unit(rng);
  std::vector<double> qv(static_cast<std::size_t>(n), 0.0);

  double estimate = 0.0;
  for (int iteration = 0; iteration < 20; ++iteration) {
    double norm = 0.0;
    for (const double value : v) norm += value * value;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return 0.0;
    for (double& value : v) value /= norm;

    hessian_multiply(model, v, &qv);

    double next = 0.0;
    for (std::size_t i = 0; i < qv.size(); ++i) next += qv[i] * v[i];
    estimate = std::fabs(next);
    v = qv;
  }
  return estimate;
}

/// The operator qp_gpu asks for, or the host one with a warning when the device cannot be
/// had. The host operator is never announced: it is what the engine always was.
std::unique_ptr<QpOperator> choose_operator(const Model& model, const Options& options,
                                            Logger& logger) {
  if (options.get_bool("qp_gpu")) {
    std::string reason;
    std::unique_ptr<QpOperator> device = gpu::make_qp_device_operator(model, &reason);
    if (device != nullptr) {
      logger.info("QP: the first-order operator runs on the device (qp_gpu, #493)");
      return device;
    }
    logger.warning(
        "QP: qp_gpu=true but the device operator is unavailable ({}); running on "
        "the host",
        reason);
  }
  return make_host_operator(model);
}

}  // namespace

Solution solve_convex_qp(const Model& model, const Options& options, Logger& logger,
                         SolveControl* control, const QpFirstOrderWarmStart* warm_start) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "qp-condat-vu";

  // One interpretation of every limit, shared with every other engine (#289). Read before
  // the convexity test, which answers to the same clock (#835).
  const ResourceLimits limits(options, logger);

  // ---- convexity, before any arithmetic ---------------------------------------------------
  const ConvexityResult convexity =
      check_convexity(model, convexity_deadline(limits, timer, control));
  if (convexity.stopped) {
    stopped_before_convexity(limits, timer, control, convexity, &solution);
    return solution;
  }
  if (convexity.verdict != Convexity::kConvex) {
    solution.status = SolveStatus::kModelError;
    solution.message =
        convexity.verdict == Convexity::kIndefinite
            ? fmt::format(
                  "the quadratic objective is not convex: {}. A non-convex QP has "
                  "local minima and saddle points, and this engine would return one "
                  "of them labelled optimal, so it is refused instead",
                  convexity.detail)
            : fmt::format("convexity could not be established: {}", convexity.detail);
    logger.warning("{}", solution.message);
    solution.solve_seconds = timer.elapsed_seconds();
    return solution;
  }

  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const auto un = static_cast<std::size_t>(n);
  const auto um = static_cast<std::size_t>(m);
  const double sense = model.sense_multiplier();

  // ---- step sizes -------------------------------------------------------------------------
  const auto seed = static_cast<unsigned>(options.get_int("random_seed"));
  const double norm_a = spectral_norm(model.matrix, m, n, seed);
  const double norm_q = hessian_norm(model, seed);

  // Condat-Vu needs 1/tau - sigma ||A||^2 >= L/2. Take sigma ||A||^2 = 1/(2 tau) and solve,
  // with a 5% margin because both norms are power-iteration ESTIMATES and an underestimate
  // would violate the condition rather than merely slow convergence.
  const double margin = 1.05;
  const double l = margin * norm_q;
  const double a2 = margin * margin * std::max(norm_a * norm_a, 1e-12);
  double tau = 1.0 / (l / 2.0 + std::sqrt(a2) + 1e-12);
  double sigma = (m > 0) ? (1.0 / tau - l / 2.0) / (2.0 * a2) : 0.0;

  logger.verbose("QP step sizes: ||A|| ~ {:.4g}, ||Q|| ~ {:.4g}, tau {:.4g}, sigma {:.4g}",
                 norm_a, norm_q, tau, sigma);

  // ---- the arithmetic, on the host or the device ------------------------------------------
  std::unique_ptr<QpOperator> op = choose_operator(model, options, logger);
  if (op->where() != std::string("host")) solution.algorithm = "qp-condat-vu-cuda";
  std::vector<double> x(un, 0.0), y(um, 0.0);  // the evaluated point, when read back
  std::vector<double> qx(un, 0.0), at_y(un, 0.0), ax(um, 0.0);

  // #981: a point from elsewhere (typically the interior point's stalled iterate,
  // qp_ipm_stall_handoff in src/core/solve.cpp), in place of the projection of zero. row_dual
  // is in Solution::row_dual's convention (solution.row_dual = -sense * y below), so the
  // internal y is its negation times sense; a size that does not match is left at zero,
  // which is what the cold start already does.
  if (warm_start != nullptr && !warm_start->empty() && warm_start->col_value.size() == un) {
    std::vector<double> y_internal(um, 0.0);
    if (warm_start->row_dual.size() == um) {
      for (std::size_t i = 0; i < um; ++i) y_internal[i] = -sense * warm_start->row_dual[i];
    }
    if (op->upload(warm_start->col_value, y_internal)) {
      logger.verbose("QP first-order: warm start from an offered point (#981)");
    }
  }

  const double tolerance = options.get_double("qp_tolerance");
  // A first-order method with no iteration limit still needs a stopping point, so an absent
  // limit becomes this engine's own ceiling rather than an unbounded loop.
  constexpr Count kIterationCeiling = 1000000;

  Count iterations = 0;
  std::string message;
  SolveStatus status = SolveStatus::kIterationLimit;

  StopController stop(control, timer, limits);
  SolveStatus stop_status;

  // ---- #493: Halpern restarts and the PID primal weight, both off by default ---------------
  // With both off none of this is touched and the loop below is the plain iteration, the
  // same arithmetic in the same order. The weight only means something with rows to price.
  const bool use_halpern = options.get_bool("qp_halpern");
  const bool use_pid = options.get_bool("qp_primal_weight_pid") && m > 0;
  const bool use_restarts = use_halpern || use_pid;
  const PidGains gains{options.get_double("qp_pid_kp"), options.get_double("qp_pid_ki"),
                       options.get_double("qp_pid_kd")};
  PidState pid;
  double omega = m > 0 ? condat_vu_weight_of(tau, sigma) : 0.0;
  double reflection =
      use_halpern ? tol::kQpHalpernReflectionShare * condat_vu_reflection_max(l, tau) : 0.0;
  bool device_error = false;
  if (use_halpern && !op->set_anchor()) device_error = true;
  if (use_pid && !op->set_restart()) device_error = true;
  Count period_steps = 0;
  Count restarts = 0;
  double period_residual = 0.0;  // ||T z - z|| at the period's first step
  double fixed_point_residual = 0.0;
  if (use_restarts) {
    logger.verbose(
        "QP acceleration: halpern {}, pid primal weight {} (kp {}, ki {}, kd {}), "
        "initial weight {:.4g}, reflection {:.4g}",
        use_halpern, use_pid, gains.kp, gains.ki, gains.kd, omega, reflection);
  }

  while (!device_error) {
    // Time outranks the counters when both are exhausted at one safe point (#289).
    if (limits.time_exhausted(timer.elapsed_seconds())) {
      status = SolveStatus::kTimeLimit;
      message = limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0);
      break;
    }
    if (limits.iterations_exhausted(iterations) ||
        (limits.iteration_limit() < 0 && iterations >= kIterationCeiling)) {
      status = SolveStatus::kIterationLimit;
      message = limits.iteration_limit() >= 0
                    ? limits.describe(LimitReason::kIterations, timer.elapsed_seconds(),
                                      iterations, 0)
                    : fmt::format(
                          "stopped at this engine's own ceiling of {} iterations, no "
                          "iteration_limit having been set",
                          kIterationCeiling);
      break;
    }
    ++iterations;
    // T z: the primal step with Q x, then the dual prox (qp_operator.hpp).
    if (!op->step(tau, sigma)) {
      device_error = true;
      break;
    }

    if (use_restarts) {
      // ||T z - z|| in the diagonal of Condat's metric, diag(I / tau, I / sigma). The period's
      // first value is the reference the restart test compares against, in the same norm.
      fixed_point_residual = op->fixed_point_residual(tau, sigma);
      if (!std::isfinite(fixed_point_residual)) {
        device_error = true;
        break;
      }
      if (period_steps == 0) period_residual = fixed_point_residual;
      ++period_steps;
    }
    // z <- T z, or under Halpern z <- w ((1 + rho) T z - rho z) + (1 - w) z_anchor, T z
    // staying where convergence is measured and what is reported.
    if (!op->advance(static_cast<double>(period_steps - 1), use_halpern ? reflection : -1.0)) {
      device_error = true;
      break;
    }

    // ---- termination, every 50 iterations -------------------------------------------------
    if (iterations % 50 != 0) continue;

    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iterations;
              p.objective = kInfinity;  // QP does not track it mid-loop
              p.best_bound = -kInfinity;
              return p;
            },
            &stop_status)) {
      status = stop_status;
      message = limits.describe(
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          timer.elapsed_seconds(), iterations, 0);
      break;
    }

    // The point every test below is made at: T z, in the box. Without Halpern it is z itself.
    if (!op->download(use_halpern, &x, &y)) {
      device_error = true;
      break;
    }

    // Primal residual: the worst row-bound violation.
    double primal_residual = 0.0;
    if (m > 0) {
      model.matrix.multiply(x.data(), ax.data());
      for (Index i = 0; i < m; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double projected = project(ax[u], model.row_lower[u], model.row_upper[u]);
        primal_residual = std::max(primal_residual, std::fabs(ax[u] - projected));
      }
    }

    // Dual residual: the projected-gradient stationarity measure. At an optimum, taking a
    // unit gradient step and projecting back changes nothing.
    hessian_multiply(model, x, &qx);
    std::fill(at_y.begin(), at_y.end(), 0.0);
    if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
    double dual_residual = 0.0;
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = sense * model.col_cost[u] + sense * qx[u] + at_y[u];
      const double stepped = project(x[u] - gradient, model.col_lower[u], model.col_upper[u]);
      dual_residual = std::max(dual_residual, std::fabs(x[u] - stepped));
    }

    if (primal_residual <= tolerance && dual_residual <= tolerance) {
      status = SolveStatus::kOptimal;
      break;
    }
    if (iterations % 5000 == 0) {
      logger.iteration(iterations, model.evaluate_objective(x.data()), primal_residual,
                       dual_residual, timer.elapsed_seconds());
    }

    // ---- #493: restart on the fixed-point residual ----------------------------------------
    if (use_restarts) {
      const bool sufficient =
          fixed_point_residual <= tol::kQpRestartSufficientDecay * period_residual;
      const bool artificial =
          period_steps >=
          std::max<Count>(50, static_cast<Count>(tol::kQpRestartArtificialShare *
                                                 static_cast<double>(iterations)));
      if (sufficient || artificial) {
        // Restart at T z, the point the residual was measured at.
        if (use_halpern && !op->take_tz()) {
          device_error = true;
          break;
        }
        if (use_pid) {
          double dx_norm = 0.0, dy_norm = 0.0;
          if (!op->restart_distance(&dx_norm, &dy_norm) || !op->set_restart()) {
            device_error = true;
            break;
          }
          omega = pid_primal_weight(omega, dx_norm, dy_norm, gains, &pid);
          const CondatVuSteps steps = condat_vu_steps_at_weight(l, a2, omega);
          tau = steps.tau;
          sigma = steps.sigma;
          if (use_halpern) reflection = tol::kQpHalpernReflectionShare * steps.reflection_max;
        }
        if (use_halpern && !op->set_anchor()) {
          device_error = true;
          break;
        }
        period_steps = 0;
        ++restarts;
        logger.verbose(
            "QP restart {} at iteration {} ({}): fixed-point residual {:.3e}, weight {:.4g}, "
            "tau {:.4g}, sigma {:.4g}",
            restarts, iterations, sufficient ? "decay" : "artificial", fixed_point_residual,
            omega, tau, sigma);
      }
    }
  }

  if (device_error) {
    // The device failed mid-solve: say so and hand the rest of the budget to the host path,
    // as the LP engines do. The host operator cannot fail, so this never recurses.
    Options on_host = options;
    on_host.set_bool("qp_gpu", false);
    if (limits.has_time_limit())
      on_host.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    logger.warning(
        "QP: the device operator failed after {} iterations; running on the host "
        "with the remaining budget",
        iterations);
    return solve_convex_qp(model, on_host, logger, control);
  }

  if (status == SolveStatus::kIterationLimit && message.empty()) {
    message = limits.describe(LimitReason::kIterations, timer.elapsed_seconds(), iterations, 0);
  }

  // Halpern's report is T z from the last step taken; before any step there is only z.
  if (!op->download(use_halpern && iterations > 0, &x, &y)) {
    solution.status = SolveStatus::kNotSolved;
    solution.message = "the QP operator could not return its iterate";
    solution.solve_seconds = timer.elapsed_seconds();
    return solution;
  }
  if (use_restarts) {
    logger.verbose("QP acceleration: {} restarts, final weight {:.4g}", restarts, omega);
  }

  solution.status = status;
  solution.message = message;
  solution.col_value.assign(x.begin(), x.end());
  for (Index i = 0; i < m; ++i) {
    solution.row_dual[static_cast<std::size_t>(i)] = -sense * y[static_cast<std::size_t>(i)];
  }
  hessian_multiply(model, x, &qx);
  std::fill(at_y.begin(), at_y.end(), 0.0);
  if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_dual[u] = model.col_cost[u] + qx[u] + sense * at_y[u];
  }
  solution.iterations = iterations;
  solution.solve_seconds = timer.elapsed_seconds();
  // An unfinished run has proven nothing about the bound, so it must not present its
  // incumbent as one - the same rule the LP engines follow.
  solution.dual_bound = status == SolveStatus::kOptimal
                            ? model.evaluate_objective(x.data())
                            : (model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity);
  solution.recompute_quality(model);
  return solution;
}

}  // namespace sankhya::qp
