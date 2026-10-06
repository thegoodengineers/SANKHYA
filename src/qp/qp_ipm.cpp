// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex QP by a proximal (primal-dual regularized) interior point (#490).
//
// References
//   Friedlander & Orban, "A primal-dual regularized interior-point method for convex quadratic
//     programs", Mathematical Programming Computation 4 (2012) - the regularized Newton
//     system and why it is quasi-definite.
//   Schwan, Jiang, Kuhn & Jones, "PIQP: a proximal interior-point quadratic programming
//     solver", arXiv 2304.00290 (2023), the paper only - proximal centres moved to the
//     current iterate every iteration, so the regularization changes the matrix and never the
//     residual being driven to zero.
//   Mehrotra, "On the implementation of a primal-dual interior point method", SIAM J. Optim.
//     2(4) (1992) - the predictor-corrector, sigma = (mu_aff / mu)^3.
//   Vanderbei, "Symmetric quasidefinite matrices", SIAM J. Optim. 5(1) (1995) - a
//     quasi-definite matrix has an LDL^T with D of known signs under EVERY symmetric
//     permutation, so the fill-reducing ordering needs no numerical pivoting.
//
// THE PROBLEM AS THE ITERATION SEES IT. Every row becomes an equality: an equality row stays
// as it is, and a ranged or one-sided row a'x gets a slack column w, a'x - w = 0, carrying the
// row's bounds as its own. Fixed columns are substituted out. What is left is
//
//     min  g'v + 0.5 v'Hv   s.t.  M v = b,   l <= v <= u     (some bounds infinite)
//
// with H = sense * Q on the original columns and zero on the slacks: equality constraints and
// bounds, nothing else, which is the first slice #490 allows. The multipliers are y (free, on
// M v = b) and z_l, z_u >= 0 on the finite bounds, and the KKT conditions are
//
//     H v + g - M'y - z_l + z_u = 0,   M v = b,   (v - l) z_l = mu,   (u - v) z_u = mu.
//
// THE NEWTON SYSTEM. Eliminating dz_l and dz_u leaves, with Theta^{-1} = Z_l S_l^{-1} +
// Z_u S_u^{-1} and the proximal parameters rho (on v) and delta (on y):
//
//     [ -(H + Theta^{-1} + rho I)   M' ] [dv]   [ r_d - rc_l ./ s_l + rc_u ./ s_u ]
//     [  M                      delta I ] [dy] = [ b - M v                        ]
//
// Negative definite over positive definite: quasi-definite for any rho, delta > 0, even when
// H is singular and a column is free (Theta^{-1} = 0). It is factored by the project's own
// sparse LDL^T (src/la/ldl.cpp) with the pivot signs enforced (factorize_quasidefinite), the
// AMD ordering unchanged. A pivot that comes out with the wrong sign or too small in
// magnitude means rho and delta are too small for the arithmetic; they are raised and the
// matrix refactored, the "raised on small pivots" rule of the issue.
//
// INFEASIBILITY AND UNBOUNDEDNESS (#893, option qp_ipm_detect_infeasibility, off by default):
// certificates read off the iterates and reported only when the project's own checker accepts
// them; see the #893 comment above farkas_candidate below. Off, an infeasible model runs to
// the iteration ceiling or a non-finite iterate and says so, as before.
//
// STALL DETECTION AND CROSS-ENGINE HANDOFF (#981, qp_ipm_stall_handoff, off by default): a
// warm run has always been abandoned on a stall, read off the three relative measures (#494,
// below); with the option on, a COLD run with no explicit iteration_limit is held to the same
// test in place of the fixed kIterationCeiling, so an iterate that is still falling keeps
// going on whatever time is left, and only a genuine stall stops it. The caller
// (src/core/solve.cpp) reads the stall back through QpIpmWarmResult::stalled and hands the
// iterate in QpIpmWarmResult::final_point to the first-order engine (qp_condat_vu.cpp) for
// the time that remains, and the reverse direction - that engine's own ceiling or time limit
// handed to this one as a QpIpmWarmStart - is the existing warm-start plumbing below, called
// from the same place. Nothing here is new arithmetic: both engines and the stall test
// already existed (#490, #493, #494); #981 is the orchestration between them.
//
// WARM START (#494, #893): when the caller offers a QpIpmWarmStart whose col_value is the
// model's own size, the usual least-squares start (Mehrotra 1992 sec. 7, "after Mehrotra"
// below) is skipped. The offered iterate - point, row and bound multipliers, proximal
// parameters - is shifted into this model's interior instead (qp_ipm_warm.cpp, after Gondzio
// 1998), and the iteration runs from there unchanged. A warm run that fails (a numerical
// error, an iterate that stops being finite) or stalls (its worst relative measure not down
// by kQpIpmWarmStallFactor in kQpIpmWarmStallWindow iterations) is abandoned and the model
// solved again from the cold start, on what is left of the iteration and time limits, so a
// warm start can cost iterations but never an answer. Every solve also hands back its own
// save point through QpIpmWarmResult: the first iterate within kQpIpmWarmSaveLevel, not the
// optimum. Scaling is still not done (#490's "Not done").

#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "la/ldl.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "convexity.hpp"
#include "qp_ipm_system.hpp"
#include "qp_ipm_warm.hpp"

namespace sankhya::qp {
namespace {

using ipm_detail::hessian_times;
using ipm_detail::inf_norm;
using ipm_detail::KktMatrix;
using ipm_detail::Standard;

constexpr double kFractionToBoundary = 0.995;  // Mehrotra (1992), and #490
constexpr Count kIterationCeiling = 200;       // an IPM that has not converged by here won't
constexpr Count kIterationsForProducts = 5;    // extra iterations to close the products

/// Solve K x = rhs with the factors, then two steps of iterative refinement against the
/// regularized K itself: the factors belong to K up to rounding and any lifted pivot, and the
/// refinement recovers what that costs without changing the system being solved.
void solve_refined(const SparseLdl& ldl, const SparseMatrix& k, const std::vector<double>& rhs,
                   std::vector<double>* x) {
  *x = rhs;
  ldl.solve(x->data());
  std::vector<double> kx(rhs.size()), correction(rhs.size());
  for (int step = 0; step < 2; ++step) {
    KktMatrix::multiply(k, *x, &kx);
    for (std::size_t i = 0; i < rhs.size(); ++i) correction[i] = rhs[i] - kx[i];
    ldl.solve(correction.data());
    for (std::size_t i = 0; i < rhs.size(); ++i) (*x)[i] += correction[i];
  }
}

// ---- infeasibility and unboundedness from the proximal iterates (#893) -------------------
//
// References
//   Banjac, Goulart, Stellato & Boyd, "Infeasibility detection in the alternating direction
//     method of multipliers for convex optimization", J. Optim. Theory Appl. 183 (2019), the
//     paper only - for a proximal splitting method on a primal (dual) infeasible convex QP,
//     the differences of successive dual (primal) iterates converge to a Farkas vector (a
//     primal ray), so a certificate can be read off the iteration that failed to converge.
//   Farkas (1902) and Schrijver (1986, sec. 7.3) for the two proofs themselves; see
//     include/sankhya/certificate.hpp.
//
// WHERE THE CANDIDATES COME FROM. This iteration is not the paper's ADMM, so its convergence
// result is not borrowed as a guarantee; it only says where to look. The proximal terms tie
// each step to the residual it cannot remove: the second block row of the Newton system is
// M dv + delta dy = b - M v, so on a primal infeasible model, where b - M v stays away from
// zero, dy grows like (b - M v) / delta and points along the residual. And the residual
// itself is a candidate: the smallest b - M v over the box, r = b - M v*, satisfies
// (M'r)'v <= (M'r)'v* = b'r - |r|^2 < b'r for every v in the box, a Farkas proof. The dual
// side is the mirror image: the smallest stationarity residual r = H w + g - M'y - z_l + z_u
// has H r = 0, M r = 0, r on the right side of every finite bound, and g'r = |r|^2 > 0, so
// -r is a primal ray, and the primal step dv follows it as the iterate runs away. So, per
// iteration, six candidates: the multipliers y, the primal residual b - M v and the step dy
// for infeasibility; the point v, minus the dual residual and the step dv for unboundedness.
//
// WHAT IS REPORTED. None of the above is trusted. A candidate is mapped to the model's own
// rows or columns and handed to farkas_proves_infeasible or ray_proves_unbounded
// (src/core/certificate.cpp), the checker every other engine's proof is held to, against the
// model this engine was given; both signs are tried, as verify_and_keep_certificate does, and
// the candidate is offered rounded first (tol::kQpIpmCertificateRounding) and then as it came.
// A ray is also only half of an unboundedness claim (tools/verify_solution.py checks the point
// first), so `unbounded` is reported only once the current iterate is primal feasible at
// primal_feasibility_tolerance, measured the way Solution::recompute_quality measures every
// answer; a verified ray waiting for such a point does not stop the iteration. Anything that
// does not check out changes nothing, and the iteration's own reporting stands.

struct Detected {
  SolveStatus status = SolveStatus::kNotSolved;  // kInfeasible or kUnbounded once found
  std::vector<double> certificate;               // model rows (Farkas) or model columns (ray)
  std::string source;
};

[[nodiscard]] bool all_finite(const std::vector<double>& x) {
  return std::all_of(x.begin(), x.end(), [](double e) { return std::isfinite(e); });
}

/// A vector over the iteration's rows, scattered to the model's: a row the standard form
/// dropped (no finite side) gets zero.
std::vector<double> on_model_rows(const Standard& s, const std::vector<double>& internal) {
  std::vector<double> out(s.row_of.size(), 0.0);
  for (std::size_t i = 0; i < s.row_of.size(); ++i) {
    if (s.row_of[i] >= 0) out[i] = internal[static_cast<std::size_t>(s.row_of[i])];
  }
  return out;
}

/// A vector over the iteration's columns, on the model's: fixed columns get zero, slack
/// columns are dropped (the checkers recompute row activities from the columns).
std::vector<double> on_model_columns(const Standard& s, const std::vector<double>& internal) {
  std::vector<double> out(static_cast<std::size_t>(s.n), 0.0);
  for (std::size_t j = 0; j < out.size(); ++j) {
    if (s.column_of[j] >= 0) out[j] = internal[static_cast<std::size_t>(s.column_of[j])];
  }
  return out;
}

/// Offer `candidate` to `proves`: rounded (tol::kQpIpmCertificateRounding) when that changes
/// it, then as it came, each with both signs. The first vector accepted lands in `accepted`.
template <typename Proves>
bool offer(std::vector<double> candidate, const Proves& proves, std::vector<double>* accepted) {
  if (!all_finite(candidate)) return false;
  double largest = 0.0;
  for (const double e : candidate) largest = std::max(largest, std::fabs(e));
  if (largest == 0.0) return false;
  std::vector<double> rounded = candidate;
  bool changed = false;
  for (double& e : rounded) {
    if (e != 0.0 && std::fabs(e) <= tol::kQpIpmCertificateRounding * largest) {
      e = 0.0;
      changed = true;
    }
  }
  std::vector<std::vector<double>*> tries;
  if (changed) tries.push_back(&rounded);
  tries.push_back(&candidate);
  for (std::vector<double>* vector : tries) {
    for (int sign = 0; sign < 2; ++sign) {
      if (proves(*vector)) {
        *accepted = std::move(*vector);
        return true;
      }
      for (double& e : *vector) e = -e;
    }
  }
  return false;
}

/// Is `internal`, on the model's rows, a Farkas proof?
bool farkas_candidate(const Model& model, const Standard& s,
                      const std::vector<double>& internal, std::vector<double>* proof) {
  return offer(
      on_model_rows(s, internal),
      [&](const std::vector<double>& y) { return farkas_proves_infeasible(model, y); }, proof);
}

/// Is `internal`, on the model's columns, an unbounded ray?
bool ray_candidate(const Model& model, const Standard& s, const std::vector<double>& internal,
                   std::vector<double>* ray) {
  return offer(
      on_model_columns(s, internal),
      [&](const std::vector<double>& d) { return ray_proves_unbounded(model, d); }, ray);
}

/// One run of the iteration, from the cold start or from `warm` (#494, #893).
struct Run {
  const QpIpmWarmStart* warm = nullptr;  ///< null: the cold start
  Count iterations_before = 0;           ///< spent by an abandoned warm run, against the limits
  QpIpmWarmStart* save = nullptr;        ///< receives the save point, when not null
  bool stalled = false;                  ///< set when the run stopped for lack of progress
  // #981, qp_ipm_stall_handoff: the iterate the run stopped at, whatever its quality -
  // unlike `save`, filled unconditionally (not only once tol::kQpIpmWarmSaveLevel is
  // reached), so a genuine stall can still hand its point to another engine.
  QpIpmWarmStart* final_point = nullptr;
};

Solution iterate(const Model& model, const Standard& s, const Options& options, Logger& logger,
                 SolveControl* control, const Timer& timer, const ResourceLimits& limits,
                 Run* run);

}  // namespace

Solution solve_convex_qp_ipm(const Model& model, const Options& options, Logger& logger,
                             SolveControl* control, const QpIpmWarmStart* warm_start,
                             QpIpmWarmResult* warm_result) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "qp-ipm";
  if (warm_result != nullptr) *warm_result = QpIpmWarmResult{};

  // Read before the convexity test, which answers to the same clock (#835).
  const ResourceLimits limits(options, logger);

  // ---- convexity, before any arithmetic: the refusal is the Condat-Vu engine's, unchanged --
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
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] > model.col_upper[u]) {
      solution.status = SolveStatus::kNotSolved;
      solution.message = fmt::format(
          "column {} has crossed bounds; the interior point needs "
          "an interior",
          j);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
  }

  const Standard s = ipm_detail::standardize(model);
  const bool warm_usable = warm_start != nullptr && !warm_start->empty() &&
                           warm_start->col_value.size() == static_cast<std::size_t>(s.n);
  QpIpmWarmStart* save = warm_result != nullptr ? &warm_result->next : nullptr;
  QpIpmWarmStart* final_point = warm_result != nullptr ? &warm_result->final_point : nullptr;
  Run first{warm_usable ? warm_start : nullptr, 0, save, false, final_point};
  Solution solved = iterate(model, s, options, logger, control, timer, limits, &first);
  if (!warm_usable) {
    if (warm_result != nullptr) {
      warm_result->stalled = first.stalled && solved.status == SolveStatus::kIterationLimit;
    }
    return solved;
  }
  if (warm_result != nullptr) warm_result->warm_used = true;
  // A WARM RUN THAT FAILED OR STALLED is abandoned for the cold start: the answer must not
  // depend on where the iteration began. A verdict (optimal, or a certificate the checker
  // accepted) and a limit stand as they are.
  const bool failed = solved.status == SolveStatus::kNumericalError ||
                      solved.status == SolveStatus::kNotSolved ||
                      (solved.status == SolveStatus::kIterationLimit && first.stalled);
  if (!failed || !warm_start->retry_cold) {
    if (warm_result != nullptr) {
      warm_result->stalled = first.stalled && solved.status == SolveStatus::kIterationLimit;
    }
    return solved;
  }
  logger.verbose(
      "QP interior point: the warm start {} after {} iterations ({}); solving again "
      "from the cold start",
      first.stalled ? "stalled" : "failed", solved.iterations, solved.message);
  if (save != nullptr) *save = QpIpmWarmStart{};
  Run cold{nullptr, solved.iterations, save, false, final_point};
  Solution again = iterate(model, s, options, logger, control, timer, limits, &cold);
  again.iterations += solved.iterations;
  if (warm_result != nullptr) {
    warm_result->fell_back = true;
    warm_result->abandoned_iterations = solved.iterations;
    warm_result->stalled = cold.stalled && again.status == SolveStatus::kIterationLimit;
  }
  return again;
}

namespace {

Solution iterate(const Model& model, const Standard& s, const Options& options, Logger& logger,
                 SolveControl* control, const Timer& timer, const ResourceLimits& limits,
                 Run* run) {
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "qp-ipm";
  const QpIpmWarmStart* warm_start = run->warm;
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);
  const double tolerance = options.get_double("qp_ipm_tolerance");
  const bool detect = options.get_bool("qp_ipm_detect_infeasibility");  // #893
  // #981: generalize the warm run's stall test (#494) to the cold run too, in place of the
  // fixed kIterationCeiling, and report a genuine stall so the caller can hand the point to
  // the first-order engine for whatever time is left. See the file header and the comment at
  // the stall test below.
  const bool stall_handoff = options.get_bool("qp_ipm_stall_handoff");
  const double primal_tolerance = options.get_double("primal_feasibility_tolerance");
  StopController stop(control, timer, limits);

  // ---- starting point: strictly inside every finite bound, unit bound multipliers ---------
  std::vector<double> v(nc, 0.0), y(nr, 0.0), zl(nc, 0.0), zu(nc, 0.0);
  std::vector<bool> has_lower(nc), has_upper(nc);
  Index bound_count = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    has_lower[j] = is_finite_bound(s.lower[j]);
    has_upper[j] = is_finite_bound(s.upper[j]);
    const double width = (has_lower[j] && has_upper[j]) ? s.upper[j] - s.lower[j] : kInfinity;
    const double inset = std::min(1.0, 0.5 * width);
    double value = 0.0;
    if (has_lower[j] && value < s.lower[j] + inset) value = s.lower[j] + inset;
    if (has_upper[j] && value > s.upper[j] - inset) value = s.upper[j] - inset;
    v[j] = value;
    if (has_lower[j]) {
      zl[j] = 1.0;
      ++bound_count;
    }
    if (has_upper[j]) {
      zu[j] = 1.0;
      ++bound_count;
    }
  }

  KktMatrix kkt(s);
  SparseLdl ldl;
  const Index dim = s.cols + s.rows;
  std::vector<signed char> signs(static_cast<std::size_t>(dim), 1);
  for (std::size_t j = 0; j < nc; ++j) signs[j] = -1;

  const auto should_stop = [&]() { return limits.time_exhausted(timer.elapsed_seconds()); };
  {
    const SparseMatrix pattern = kkt.build(std::vector<double>(nc, 1.0), 1.0, 1.0);
    if (!ldl.analyze(pattern, should_stop)) {
      solution.status = ldl.stopped_early() ? SolveStatus::kTimeLimit : SolveStatus::kNotSolved;
      solution.message =
          ldl.stopped_early()
              ? limits.describe(LimitReason::kTime, timer.elapsed_seconds(), 0, 0)
              : "the KKT system's ordering could not be computed";
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }
  }

  std::vector<double> hv(nc), mv(nr), mty(nc), rd(nc), rp(nr), theta_inverse(nc);
  std::vector<double> rhs(static_cast<std::size_t>(dim)), step(static_cast<std::size_t>(dim));
  std::vector<double> dv(nc), dy(nr), dzl(nc), dzu(nc), rcl(nc), rcu(nc), dv_aff(nc);
  std::vector<double> dzl_aff(nc), dzu_aff(nc);
  double rho = options.get_double("qp_ipm_regularization");
  double delta = rho;
  const double regularization_floor = rho;

  SolveStatus status = SolveStatus::kIterationLimit;
  std::string message;
  Count iterations = 0;
  Count iterations_past_relative = 0;
  std::vector<double> kept_v, kept_y;
  double primal_rel = kInfinity, dual_rel = kInfinity, gap_rel = kInfinity;
  double last_mu = 0.0;  // #981: mu at the point the loop stopped, for final_point below
  // #494: the worst relative measure at each iteration of a warm run, for the stall test.
  std::vector<double> worst_measure;

  const auto residuals = [&]() {
    hessian_times(s.h, v, &hv);
    std::fill(mv.begin(), mv.end(), 0.0);
    s.m.multiply_add(v.data(), mv.data());
    std::fill(mty.begin(), mty.end(), 0.0);
    if (s.rows > 0) s.m.transpose_multiply_add(y.data(), mty.data());
    for (std::size_t j = 0; j < nc; ++j) rd[j] = hv[j] + s.g[j] - mty[j] - zl[j] + zu[j];
    for (std::size_t i = 0; i < nr; ++i) rp[i] = s.b[i] - mv[i];
  };

  // The step for a given complementarity target: rcl and rcu hold sigma mu - s z (- the
  // Mehrotra cross term), and the result lands in dv, dy, dzl, dzu.
  const auto newton = [&](const SparseMatrix& k) {
    for (std::size_t j = 0; j < nc; ++j) {
      double r = rd[j];
      if (has_lower[j]) r -= rcl[j] / (v[j] - s.lower[j]);
      if (has_upper[j]) r += rcu[j] / (s.upper[j] - v[j]);
      rhs[j] = r;
    }
    for (std::size_t i = 0; i < nr; ++i) rhs[nc + i] = rp[i];
    solve_refined(ldl, k, rhs, &step);
    for (std::size_t j = 0; j < nc; ++j) {
      dv[j] = step[j];
      dzl[j] = has_lower[j] ? (rcl[j] - zl[j] * dv[j]) / (v[j] - s.lower[j]) : 0.0;
      dzu[j] = has_upper[j] ? (rcu[j] + zu[j] * dv[j]) / (s.upper[j] - v[j]) : 0.0;
    }
    for (std::size_t i = 0; i < nr; ++i) dy[i] = step[nc + i];
  };

  // The largest alpha in (0, 1] keeping every slack and bound multiplier non-negative.
  const auto longest_step = [&]() {
    double alpha = 1.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        if (dv[j] < 0.0) alpha = std::min(alpha, -(v[j] - s.lower[j]) / dv[j]);
        if (dzl[j] < 0.0) alpha = std::min(alpha, -zl[j] / dzl[j]);
      }
      if (has_upper[j]) {
        if (dv[j] > 0.0) alpha = std::min(alpha, (s.upper[j] - v[j]) / dv[j]);
        if (dzu[j] < 0.0) alpha = std::min(alpha, -zu[j] / dzu[j]);
      }
    }
    return alpha;
  };

  // ---- #893: candidates from the iterates, reported only when the checker accepts them ----
  Detected detected;
  bool ray_awaiting_point = false;
  const auto model_point = [&]() {
    std::vector<double> x(static_cast<std::size_t>(s.n));
    for (std::size_t j = 0; j < x.size(); ++j) {
      const Index kcol = s.column_of[j];
      x[j] = kcol < 0 ? s.fixed_value[j] : v[static_cast<std::size_t>(kcol)];
    }
    return x;
  };
  // A Farkas candidate over the iteration's rows and a ray candidate over its columns; an
  // empty vector is not examined.
  const auto examine = [&](const std::vector<double>& on_rows, const char* rows_source,
                           const std::vector<double>& on_columns, const char* columns_source) {
    std::vector<double> proof;
    if (!on_rows.empty() && farkas_candidate(model, s, on_rows, &proof)) {
      detected = {SolveStatus::kInfeasible, std::move(proof), rows_source};
      return true;
    }
    if (!on_columns.empty() && ray_candidate(model, s, on_columns, &proof)) {
      Solution probe;
      probe.col_value = model_point();
      probe.recompute_quality(model);
      if (probe.primal_infeasibility_scaled <= primal_tolerance) {
        detected = {SolveStatus::kUnbounded, std::move(proof), columns_source};
        return true;
      }
      ray_awaiting_point = true;
    }
    return false;
  };
  std::vector<double> minus_rd(detect ? nc : 0);

  // ---- starting point ----------------------------------------------------------------------
  // Warm (#494, #893): the caller's iterate shifted into this model's interior, see
  // qp_ipm_warm.cpp. Cold, after Mehrotra (1992, sec. 7), adapted to bounds: v and y from
  // min g'v + v'(H + I)v/2 s.t. M v = b, one factorization of the same pattern with
  // Theta^{-1} = I, so the start already nearly satisfies the rows (from the unit start the
  // primal residual stayed at 1.0 for thirty iterations on qpcboei2). v is then moved inside
  // its bounds by a margin that grows with how far outside the point was, and the bound
  // multipliers take the sign-split stationarity residual plus a shift that balances them
  // against the slacks.
  if (warm_start != nullptr) {
    const ipm_detail::WarmShift shift = ipm_detail::warm_start_iterate(
        model, s, *warm_start, regularization_floor, &v, &y, &zl, &zu);
    rho = shift.rho;
    delta = shift.delta;
    logger.verbose(
        "QP interior point: warm start, {} slack(s) moved inside, {} multiplier(s) recentred "
        "on mu {:.2e}, rho {:.1e}",
        shift.moved_inside, shift.recentred, shift.mu, rho);
  } else {
    std::fill(theta_inverse.begin(), theta_inverse.end(), 1.0);
    const SparseMatrix k0 = kkt.build(theta_inverse, rho, delta);
    if (ldl.factorize_quasidefinite(k0, signs, tol::kQpIpmPivotShare * std::min(rho, delta),
                                    should_stop) &&
        ldl.regularized_pivots() == 0) {
      for (std::size_t j = 0; j < nc; ++j) rhs[j] = s.g[j];
      for (std::size_t i = 0; i < nr; ++i) rhs[nc + i] = s.b[i];
      solve_refined(ldl, k0, rhs, &step);
      if (std::all_of(step.begin(), step.end(), [](double x) { return std::isfinite(x); })) {
        std::copy(step.begin(), step.begin() + static_cast<std::ptrdiff_t>(nc), v.begin());
        std::copy(step.begin() + static_cast<std::ptrdiff_t>(nc), step.end(), y.begin());
      }
    }
    double worst = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) worst = std::min(worst, v[j] - s.lower[j]);
      if (has_upper[j]) worst = std::min(worst, s.upper[j] - v[j]);
    }
    const double margin = std::max(1.0, -1.5 * worst);
    for (std::size_t j = 0; j < nc; ++j) {
      const double width = (has_lower[j] && has_upper[j]) ? s.upper[j] - s.lower[j] : kInfinity;
      const double inset = std::min(margin, 0.5 * width);
      if (has_lower[j] && v[j] < s.lower[j] + inset) v[j] = s.lower[j] + inset;
      if (has_upper[j] && v[j] > s.upper[j] - inset) v[j] = s.upper[j] - inset;
      zl[j] = 0.0;
      zu[j] = 0.0;
    }
    residuals();  // with z = 0, rd is the stationarity residual H v + g - M'y
    double slack_times_z = 0.0, slack_sum = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        zl[j] = std::max(rd[j], 0.0);
        slack_times_z += (v[j] - s.lower[j]) * zl[j];
        slack_sum += v[j] - s.lower[j];
      }
      if (has_upper[j]) {
        zu[j] = std::max(-rd[j], 0.0);
        slack_times_z += (s.upper[j] - v[j]) * zu[j];
        slack_sum += s.upper[j] - v[j];
      }
    }
    const double shift = std::max(1.0, slack_sum > 0.0 ? 0.5 * slack_times_z / slack_sum : 0.0);
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) zl[j] += shift;
      if (has_upper[j]) zu[j] += shift;
    }
  }

  while (true) {
    residuals();
    double complementarity = 0.0;
    double largest_product = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      if (has_lower[j]) {
        const double product = (v[j] - s.lower[j]) * zl[j];
        complementarity += product;
        largest_product = std::max(largest_product, product);
      }
      if (has_upper[j]) {
        const double product = (s.upper[j] - v[j]) * zu[j];
        complementarity += product;
        largest_product = std::max(largest_product, product);
      }
    }
    // A row's product as the KKT check will measure it: its multiplier against the distance
    // from the row ACTIVITY a'x = w - r_p to the nearer bound, not from the slack w, which
    // differs by the primal residual (primalc1: 3.9e-06 on a row whose w-product was tiny).
    for (Index k = 0; k < s.slacks; ++k) {
      const auto u = static_cast<std::size_t>(s.free_n + k);
      const auto r = static_cast<std::size_t>(s.slack_row[static_cast<std::size_t>(k)]);
      const double activity = v[u] - rp[r];
      double distance = kInfinity;
      if (has_lower[u]) distance = std::min(distance, std::fabs(activity - s.lower[u]));
      if (has_upper[u]) distance = std::min(distance, std::fabs(s.upper[u] - activity));
      if (std::isfinite(distance)) {
        largest_product = std::max(largest_product, std::fabs(y[r]) * distance);
      }
    }
    const double mu =
        bound_count > 0 ? complementarity / static_cast<double>(bound_count) : 0.0;
    last_mu = mu;

    // Relative residuals, and the Dorn duality gap: primal g'v + v'Hv/2 against
    // b'y + l'z_l - u'z_u - v'Hv/2.
    double vhv = 0.0, gv = 0.0, dual = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
      vhv += v[j] * hv[j];
      gv += s.g[j] * v[j];
      if (has_lower[j]) dual += s.lower[j] * zl[j];
      if (has_upper[j]) dual -= s.upper[j] * zu[j];
    }
    for (std::size_t i = 0; i < nr; ++i) dual += s.b[i] * y[i];
    const double primal_objective = gv + 0.5 * vhv;
    const double dual_objective = dual - 0.5 * vhv;
    primal_rel = inf_norm(rp) / (1.0 + std::max(inf_norm(mv), inf_norm(s.b)));
    dual_rel = inf_norm(rd) / (1.0 + std::max({inf_norm(hv), inf_norm(s.g), inf_norm(mty)}));
    gap_rel = std::fabs(primal_objective - dual_objective) /
              (1.0 + std::max(std::fabs(primal_objective), std::fabs(dual_objective)));

    logger.iteration(iterations, primal_objective, primal_rel, dual_rel,
                     timer.elapsed_seconds());
    // The relative gap alone is not enough: an optimality claim is also held to an ABSOLUTE
    // complementarity product per item (tol::kComplementarity, the verifier's test), which a
    // relative gap of 1e-9 on an objective of 5e+05 does not imply (qadlittl: 1.1e-06 on one
    // row). A tenth of it leaves room for the mapping back to the model's rows and columns.
    //
    // When the relative measures are met and the products are not, a few more iterations
    // usually close them; when they cannot (a row whose multiplier is 1e+03 needs its
    // activity on the bound to 1e-10, below what the residual reaches in double precision),
    // the point is handed back as an optimality claim and the in-process KKT gate in solve()
    // decides, withdrawing it to feasible with the product named if it is not backed. The
    // last point that met the relative measures is kept, so an iterate that stops being
    // finite while mu is pushed to zero cannot cost the answer already found.
    const bool relative_met =
        primal_rel <= tolerance && dual_rel <= tolerance && gap_rel <= tolerance;
    // #494: the save point a later solve can start from, the first iterate within
    // kQpIpmWarmSaveLevel: advanced, and still well inside the bounds (Gondzio 1998). A
    // qp_ipm_tolerance looser than that level saves the first iterate that meets it instead.
    if (run->save != nullptr && run->save->empty() &&
        (relative_met ||
         (primal_rel <= tol::kQpIpmWarmSaveLevel && dual_rel <= tol::kQpIpmWarmSaveLevel &&
          gap_rel <= tol::kQpIpmWarmSaveLevel))) {
      *run->save = ipm_detail::save_iterate(model, s, v, y, zl, zu, rho, delta, mu,
                                            run->iterations_before + iterations);
    }
    if (relative_met &&
        (largest_product <= tol::kQpIpmComplementarityShare * tol::kComplementarity ||
         iterations_past_relative >= kIterationsForProducts)) {
      status = SolveStatus::kOptimal;
      break;
    }
    if (relative_met) {
      ++iterations_past_relative;
      kept_v = v;
      kept_y = y;
    }
    if (detect && iterations > 0) {
      for (std::size_t j = 0; j < nc; ++j) minus_rd[j] = -rd[j];
      if (examine(y, "multipliers y", v, "point v") ||
          examine(rp, "primal residual b - M v", minus_rd, "negated dual residual")) {
        status = detected.status;
        break;
      }
    }
    // #494, generalized by #981 (qp_ipm_stall_handoff): a run that has stopped making
    // progress is handed back - to the cold start when it was warm (#494, unconditional), or
    // to the caller to try the first-order engine when it was cold and the option is on
    // (#981). The test is the same either way: the largest of the three relative measures
    // has not fallen by kQpIpmWarmStallFactor over the last kQpIpmWarmStallWindow iterations
    // (Mehrotra 1992's predictor-corrector gains an order of magnitude every two or three
    // iterations while converging, so ten without one is a stall, not slow progress).
    if (warm_start != nullptr || stall_handoff) {
      const double worst = std::max({primal_rel, dual_rel, gap_rel});
      worst_measure.push_back(worst);
      const auto window = static_cast<std::size_t>(tol::kQpIpmWarmStallWindow);
      if (worst_measure.size() > window &&
          !(worst <=
            tol::kQpIpmWarmStallFactor * worst_measure[worst_measure.size() - 1 - window])) {
        status = SolveStatus::kIterationLimit;
        message = fmt::format(
            "{} stalled: the worst relative measure is {:.1e} after {} iterations, {:.1e} "
            "{} iterations before",
            warm_start != nullptr ? "the warm start" : "the interior point", worst, iterations,
            worst_measure[worst_measure.size() - 1 - window], window);
        run->stalled = true;
        break;
      }
    }
    if (limits.time_exhausted(timer.elapsed_seconds())) {
      status = SolveStatus::kTimeLimit;
      message = limits.describe(LimitReason::kTime, timer.elapsed_seconds(),
                                run->iterations_before + iterations, 0);
      break;
    }
    const Count spent = run->iterations_before + iterations;
    // #981: with qp_ipm_stall_handoff, the fixed ceiling is replaced by the stall test above
    // for a cold run with no explicit iteration_limit - it already broke the loop once the
    // worst measure stopped falling, so reaching here means it is STILL falling and keeps
    // going on whatever time is left (checked just above). Off, or with an explicit
    // iteration_limit, nothing here changes.
    if (limits.iterations_exhausted(spent) ||
        (limits.iteration_limit() < 0 && !stall_handoff && spent >= kIterationCeiling)) {
      status = SolveStatus::kIterationLimit;
      message =
          limits.iteration_limit() >= 0
              ? limits.describe(LimitReason::kIterations, timer.elapsed_seconds(), spent, 0)
              : fmt::format(
                    "stopped at this engine's own ceiling of {} iterations with "
                    "relative residuals {:.1e} primal, {:.1e} dual, {:.1e} gap",
                    kIterationCeiling, primal_rel, dual_rel, gap_rel);
      break;
    }
    SolveStatus stop_status;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iterations;
              p.objective = kInfinity;
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
    ++iterations;

    // ---- factor, raising the proximal parameters while a pivot comes out wrong ----------
    for (std::size_t j = 0; j < nc; ++j) {
      double t = 0.0;
      if (has_lower[j]) t += zl[j] / (v[j] - s.lower[j]);
      if (has_upper[j]) t += zu[j] / (s.upper[j] - v[j]);
      theta_inverse[j] = t;
    }
    SparseMatrix k;
    bool factored = false;
    for (int attempt = 0; attempt < tol::kQpIpmRegularizationAttempts && !factored; ++attempt) {
      k = kkt.build(theta_inverse, rho, delta);
      if (!ldl.factorize_quasidefinite(k, signs, tol::kQpIpmPivotShare * std::min(rho, delta),
                                       should_stop)) {
        break;
      }
      if (ldl.regularized_pivots() == 0) {
        factored = true;
      } else {
        rho *= tol::kQpIpmRegularizationRaise;
        delta *= tol::kQpIpmRegularizationRaise;
      }
    }
    if (!factored) {
      status = ldl.stopped_early() ? SolveStatus::kTimeLimit : SolveStatus::kNumericalError;
      message =
          ldl.stopped_early()
              ? limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0)
              : fmt::format(
                    "the regularized KKT system lost its quasi-definite sign "
                    "pattern at rho = {:.1e}",
                    rho);
      break;
    }

    // ---- predictor ------------------------------------------------------------------------
    for (std::size_t j = 0; j < nc; ++j) {
      rcl[j] = has_lower[j] ? -(v[j] - s.lower[j]) * zl[j] : 0.0;
      rcu[j] = has_upper[j] ? -(s.upper[j] - v[j]) * zu[j] : 0.0;
    }
    newton(k);
    double sigma = 0.0;
    if (bound_count > 0) {
      const double alpha_aff = longest_step();
      double affine = 0.0;
      for (std::size_t j = 0; j < nc; ++j) {
        if (has_lower[j]) {
          affine += (v[j] - s.lower[j] + alpha_aff * dv[j]) * (zl[j] + alpha_aff * dzl[j]);
        }
        if (has_upper[j]) {
          affine += (s.upper[j] - v[j] - alpha_aff * dv[j]) * (zu[j] + alpha_aff * dzu[j]);
        }
      }
      const double mu_aff = affine / static_cast<double>(bound_count);
      sigma = mu > 0.0 ? std::pow(mu_aff / mu, 3.0) : 0.0;
      sigma = std::min(1.0, std::max(0.0, sigma));

      // ---- corrector: the centring target and Mehrotra's second-order term ---------------
      dv_aff = dv;
      dzl_aff = dzl;
      dzu_aff = dzu;
      for (std::size_t j = 0; j < nc; ++j) {
        if (has_lower[j]) {
          rcl[j] = sigma * mu - (v[j] - s.lower[j]) * zl[j] - dv_aff[j] * dzl_aff[j];
        }
        if (has_upper[j]) {
          rcu[j] = sigma * mu - (s.upper[j] - v[j]) * zu[j] + dv_aff[j] * dzu_aff[j];
        }
      }
      newton(k);
    }

    // #893: the step itself, before it is taken - on a model that has no optimum this is the
    // direction the iterate runs away along, and the step after it may not be finite.
    if (detect && examine(dy, "step dy", dv, "step dv")) {
      status = detected.status;
      break;
    }

    const double alpha =
        bound_count > 0 ? std::min(1.0, kFractionToBoundary * longest_step()) : 1.0;
    for (std::size_t j = 0; j < nc; ++j) {
      v[j] += alpha * dv[j];
      zl[j] += alpha * dzl[j];
      zu[j] += alpha * dzu[j];
    }
    for (std::size_t i = 0; i < nr; ++i) y[i] += alpha * dy[i];
    const auto finite = [](const std::vector<double>& x) {
      return std::all_of(x.begin(), x.end(), [](double e) { return std::isfinite(e); });
    };
    if (!finite(v) || !finite(y) || !finite(zl) || !finite(zu)) {
      if (!kept_v.empty()) {
        v = kept_v;  // the last point that met the relative measures; the gate judges it
        y = kept_y;
        status = SolveStatus::kOptimal;
      } else {
        status = SolveStatus::kNumericalError;
        message = "the interior point's iterate stopped being finite";
      }
      break;
    }

    // The proximal parameters follow mu down, never below the configured floor (the
    // centres move with the iterate, so a large rho slows the method and never biases it).
    rho = std::max(regularization_floor, std::min(rho, 0.1 * mu));
    delta = rho;
  }

  // ---- back to the caller's model -------------------------------------------------------
  const double sense = model.sense_multiplier();
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index kcol = s.column_of[u];
    solution.col_value[u] = kcol < 0 ? s.fixed_value[u] : v[static_cast<std::size_t>(kcol)];
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const Index r = s.row_of[u];
    solution.row_dual[u] = r < 0 ? 0.0 : sense * y[static_cast<std::size_t>(r)];
  }
  // Reduced costs in the model's own sense, c + Q x - A' row_dual, the quantity the Condat-Vu
  // engine reports and the KKT check derives.
  std::vector<double> qx(static_cast<std::size_t>(s.n), 0.0);
  hessian_times(model.hessian, solution.col_value, &qx);
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double d = model.col_cost[u] + qx[u];
    const ColumnView column = model.matrix.column(j);
    for (Index p = 0; p < column.size; ++p) {
      d -= column.values[p] * solution.row_dual[static_cast<std::size_t>(column.rows[p])];
    }
    solution.col_dual[u] = d;
  }
  solution.status = status;
  solution.message = message;
  solution.iterations = iterations;
  solution.solve_seconds = timer.elapsed_seconds();
  solution.dual_bound = status == SolveStatus::kOptimal
                            ? model.evaluate_objective(solution.col_value.data())
                            : (model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity);
  if (status == SolveStatus::kInfeasible || status == SolveStatus::kUnbounded) {
    const bool infeasible = status == SolveStatus::kInfeasible;
    solution.message = fmt::format(
        "{} at iteration {}: the {} of the proximal iterates is {} that the certificate "
        "checker accepts against this model (#893)",
        infeasible ? "primal infeasible" : "unbounded", iterations, detected.source,
        infeasible ? "a Farkas certificate" : "a ray, from a primal feasible point,");
    if (infeasible) {
      // #191: a verdict with no point does not get a point, and its bound is the worst value
      // the objective can take, on the model's own sense. Cleared before the measurement
      // below, which then has no point to measure (#505: no numbers for a point not claimed).
      solution.farkas_dual = std::move(detected.certificate);
      solution.col_value.clear();
      solution.col_dual.clear();
      solution.row_dual.clear();
      solution.row_activity.clear();
      solution.dual_bound = model.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
    } else {
      solution.primal_ray = std::move(detected.certificate);
    }
    logger.info("QP interior point: {}", solution.message);
  } else if (ray_awaiting_point) {
    solution.message += fmt::format(
        "{}a ray the certificate checker accepts was found, but no primal feasible iterate to "
        "go with it, so unboundedness is not claimed (#893)",
        solution.message.empty() ? "" : "; ");
  }
  solution.recompute_quality(model);
  logger.verbose(
      "QP interior point: {} iterations, relative residuals {:.2e} primal, {:.2e} "
      "dual, {:.2e} gap",
      iterations, primal_rel, dual_rel, gap_rel);
  return solution;
}

}  // namespace
}  // namespace sankhya::qp
