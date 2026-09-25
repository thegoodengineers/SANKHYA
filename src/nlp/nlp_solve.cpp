// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the general NLP engine's entry point (NLP stage 2). See nlp_solve.hpp.

#include "nlp/nlp_solve.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <fmt/format.h>

#include "nlp/barrier_nlp.hpp"
#include "nlp/filter_ipm.hpp"
#include "nlp/nlp_kkt_check.hpp"
#include "nlp/nlp_problem.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp {
namespace {

/// The model's multipliers in minimisation space from the method's iterate: y = -lambda
/// for a row (so a positive y prices its lower bound), d = z_L - z_U for a column, and for a
/// fixed column - which carries no bound multiplier in the method - the d stationarity
/// asks for, grad f - J^T y.
void to_model_terms(const NlpProblem& problem, const IpmIterate& it, Vec* x, Vec* y, Vec* d) {
  const auto n = static_cast<std::size_t>(problem.num_variables());
  const auto m = static_cast<std::size_t>(problem.num_constraints());
  x->assign(it.w.begin(), it.w.begin() + static_cast<std::ptrdiff_t>(n));
  y->assign(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) (*y)[i] = -it.lambda[i];
  d->assign(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) (*d)[j] = it.zl[j] - it.zu[j];
  const Vec& lo = problem.x_lower();
  const Vec& up = problem.x_upper();
  bool any_fixed = false;
  for (std::size_t j = 0; j < n; ++j) any_fixed = any_fixed || (lo[j] == up[j]);
  if (!any_fixed) return;
  double f = 0.0;
  Vec grad, jac;
  Evaluation error;
  if (!problem.objective_gradient(*x, &f, &grad, &error) ||
      !problem.jacobian(*x, &jac, &error)) {
    return;
  }
  Vec jt_y(n, 0.0);
  for (Index i = 0; i < problem.num_constraints(); ++i) {
    for (Index k = problem.jacobian_starts()[static_cast<std::size_t>(i)];
         k < problem.jacobian_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
      jt_y[static_cast<std::size_t>(problem.jacobian_columns()[static_cast<std::size_t>(k)])] +=
          (*y)[static_cast<std::size_t>(i)] * jac[static_cast<std::size_t>(k)];
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (lo[j] == up[j]) (*d)[j] = grad[j] - jt_y[j];
  }
}

}  // namespace

Solution solve_nlp_relaxation(const NonlinearModel& model, const Options& options,
                              const Vec& start, SolveControl* control, Logger& logger) {
  const Timer timer;
  Solution out;
  out.algorithm = "nlp-ipm";
  NlpProblem problem;
  const std::string built = NlpProblem::build(model, &problem);
  if (!built.empty()) {
    out.status = SolveStatus::kModelError;
    out.message = built;
    return out;
  }
  const SlackedNlp slacked(&problem);
  const auto n = static_cast<std::size_t>(problem.num_variables());

  // The start: the caller's, else the model's, else zeros; each slack at its row's value
  // there, when the row can be evaluated (the method pushes both inside their bounds).
  Vec x0 = start.size() == n ? start : (model.start.size() == n ? model.start : Vec(n, 0.0));
  Vec w0 = x0;
  w0.resize(static_cast<std::size_t>(slacked.n()), 0.0);
  {
    Vec clamped = x0, g;
    for (std::size_t j = 0; j < n; ++j) {
      clamped[j] = std::min(std::max(clamped[j], problem.x_lower()[j]), problem.x_upper()[j]);
    }
    Evaluation error;
    if (problem.constraints(clamped, &g, &error)) {
      for (Index i = 0; i < problem.num_constraints(); ++i) {
        const Index s = slacked.slack_of(i);
        if (s >= 0) w0[static_cast<std::size_t>(s)] = g[static_cast<std::size_t>(i)];
      }
    }
  }

  const double primal_tol = tol::kPrimalFeasibility;
  const double dual_tol = tol::kDualFeasibility;
  const double compl_tol = tol::kComplementarity;
  IpmSettings settings;
  settings.mu_min = std::min(primal_tol, dual_tol) / tol::kNlpKappaEpsilon;
  settings.tolerance = dual_tol;
  const std::int64_t limit = options.get_int("iteration_limit");
  settings.max_iterations = limit >= 0 ? static_cast<Count>(limit) : tol::kNlpMaxIterations;
  const double time_limit = options.get_double("time_limit");
  settings.should_stop = [&]() {
    return timer.elapsed_seconds() > time_limit ||
           (control != nullptr && control->interruption_requested());
  };

  IpmHooks hooks;
  Vec x, y, d;
  hooks.converged = [&](const IpmIterate& it, double mu) {
    // Only once the barrier parameter has reached its floor: before that the bound
    // multipliers are still the method's early guesses, and z_L - z_U can cancel into a d
    // that passes the check at a point the method never examined. Measured on HS45: from its
    // start pushed to x = 0.01, where the gradient is ~1e-10, the check passed at iteration 0
    // with objective 2, at the problem's worst point; the published optimum is 1.
    if (mu > settings.mu_min) return false;
    to_model_terms(problem, it, &x, &y, &d);
    return check_nlp_kkt(problem, x, y, d, primal_tol, dual_tol, compl_tol).passed;
  };
  hooks.log = [&](Count it, double f, double theta, double dual, double mu, double alpha,
                  bool restoration) {
    logger.verbose("NLP {:>5}{} f {:+.10e} |c|_1 {:.2e} dual {:.2e} mu {:.1e} alpha {:.2e}", it,
                   restoration ? "r" : " ", problem.sense() * f, theta, dual, mu, alpha);
  };
  const IpmResult result = run_filter_ipm(slacked, IpmIterate{w0, {}, {}, {}}, settings, hooks);

  to_model_terms(problem, result.at, &x, &y, &d);
  const NlpKktReport report = check_nlp_kkt(problem, x, y, d, primal_tol, dual_tol, compl_tol);
  const double sense = problem.sense();
  out.iterations = result.iterations;
  out.col_value = x;
  Evaluation error;
  double f = 0.0;
  if (problem.constraints(x, &out.row_activity, &error) && problem.objective(x, &f, &error)) {
    out.objective = sense * f;
  }
  out.row_dual.resize(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) out.row_dual[i] = sense * y[i];
  out.col_dual.resize(d.size());
  for (std::size_t j = 0; j < d.size(); ++j) out.col_dual[j] = sense * d[j];
  out.primal_infeasibility = report.primal_absolute;
  out.dual_infeasibility = std::max(report.stationarity, report.sign);
  const double no_bound = sense > 0.0 ? -kInfinity : kInfinity;
  out.dual_bound = no_bound;

  switch (result.exit) {
    case IpmExit::kConverged: {
      const ConvexityReport convexity = model.convexity();
      if (convexity.convex) {
        out.status = SolveStatus::kOptimal;
        out.dual_bound = out.objective;
        out.message =
            "a KKT point of a problem proved convex by the composition rules: a global optimum";
      } else {
        out.status = SolveStatus::kLocallyOptimal;
        out.message = fmt::format(
            "a KKT point to the project tolerances: a LOCAL optimum; not proved global because "
            "{}",
            convexity.reasons.empty() ? "convexity was not established"
                                      : convexity.reasons.front());
      }
      break;
    }
    case IpmExit::kLocallyInfeasible:
      out.status = SolveStatus::kLocallyInfeasible;
      out.message = result.message;
      break;
    case IpmExit::kIterationLimit:
      out.status = SolveStatus::kIterationLimit;
      out.stopped_by = LimitReason::kIterations;
      out.message = result.message;
      break;
    case IpmExit::kStopped:
      if (control != nullptr && control->interruption_requested()) {
        out.status = SolveStatus::kInterrupted;
        out.stopped_by = LimitReason::kInterrupt;
      } else {
        out.status = SolveStatus::kTimeLimit;
        out.stopped_by = LimitReason::kTime;
      }
      out.message = result.message;
      break;
    case IpmExit::kAccepted:
    case IpmExit::kRestorationFailed:
    case IpmExit::kEvaluationFailed:
    case IpmExit::kFactorizationFailed:
      // No certificate of optimality. A point inside the model is still worth returning as
      // `feasible`; anything else is a numerical failure, said with the method's reason.
      out.status = report.evaluated && report.primal_ok ? SolveStatus::kFeasible
                                                        : SolveStatus::kNumericalError;
      out.message = fmt::format("{}: {}", to_string(result.exit), result.message);
      break;
  }
  if (!claims_a_point(out.status)) {
    const std::string message = out.message;
    const SolveStatus status = out.status;
    const Count iterations = out.iterations;
    out = Solution{};
    out.status = status;
    out.message = message;
    out.iterations = iterations;
    out.algorithm = "nlp-ipm";
  }
  out.solve_seconds = timer.elapsed_seconds();
  logger.info("NLP interior point: {} after {} iterations{}{}", to_string(out.status),
              out.iterations, out.message.empty() ? "" : " - ", out.message);
  return out;
}

Solution solve_nlp(const NonlinearModel& model, const Options& options, SolveControl* control) {
  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  const std::string problem = model.validate();
  if (!problem.empty()) {
    Solution out;
    out.status = SolveStatus::kModelError;
    out.message = problem;
    return out;
  }
  if (model.base.has_integrality()) {
    Solution out;
    out.status = SolveStatus::kNotSolved;
    out.message =
        "the model has integer columns (a MINLP); the NLP engine solves continuous models";
    return out;
  }
  return solve_nlp_relaxation(model, options, {}, control, logger);
}

}  // namespace sankhya::nlp
