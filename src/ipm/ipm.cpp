// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Mehrotra's predictor-corrector interior-point method for LP (#56).
//
// References:
//   Mehrotra, S., "On the implementation of a primal-dual interior point method", SIAM J.
//     Optimization 2 (1992), 575-601 - the predictor-corrector and the centering rule.
//   Wright, S.J., "Primal-Dual Interior-Point Methods", SIAM (1997), ch. 10-11 - the
//     bounded-variable form, the normal equations, the starting point and the step rule.
//   Altman & Gondzio (1999), see la/ldl.hpp - primal and dual regularization.
//
// THE FORM SOLVED. The same one the simplex uses:  [A | -I] [x; s] = 0  with bounds on every
// component, l <= x <= u for the structurals and row_lower <= s <= row_upper for the
// logicals. A finite lower bound contributes a slack s_l = x - l >= 0 with multiplier
// z_l >= 0, a finite upper bound s_u = u - x >= 0 with z_u >= 0, and the dual constraint
// reads c - Abar^T y - z_l + z_u = 0. Free variables have neither and are held by the
// primal regularization alone.
//
// Every Newton system reduces to the normal equations  (Abar Theta Abar^T + delta I) dy = r
// with Theta diagonal, which is SPD and, because Abar carries -I, at least as well
// conditioned as Theta_s on the logical block. The sparse LDL^T of la/ldl.hpp is analyzed
// once and refactorized every iteration.
//
// WHAT THIS METHOD DOES NOT DO. It produces no basis, so it cannot warm-start branch and
// bound and the simplex remains the node engine. It does not certify infeasibility or
// unboundedness: when it fails to converge it says so and hands back a numerical error,
// never a claim it cannot prove. Both are stated in the option's description.

#include "sankhya/ipm.hpp"
#include "util/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/resource_limits.hpp"
#include "ipm/ipm_testing.hpp"
#include "ipm/model_space.hpp"
#include "ipm/proximal_system.hpp"
#include "ipm/dense_columns.hpp"
#include "la/ldl.hpp"
#include "la/normal_pattern.hpp"
#include "la/scaling.hpp"
#include "util/profiler.hpp"

#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

/// Convergence: relative primal and dual infeasibility and relative complementarity all
/// below this. Tighter than the reported tolerances (1e-7) so the point survives being
/// re-measured against the original model in recompute_quality().
constexpr double kIpmTolerance = 1e-8;
/// The largest single complementarity product s_k z_k at convergence, relative to the
/// variable's magnitude. The average (mu) is what the method drives down; the status guard
/// judges every product on its own against the primal scale, so an iterate whose average
/// is 1e-8 and whose worst product is 1e-7 would be downgraded to `feasible` after
/// converging. One or two more Mehrotra steps close that gap.
///
/// 1e-8, a decade inside the guard's 1e-7, and not tighter: measured on a KKT instance
/// (test_ipm.cpp, trial 109 of the oracle sweep) the worst relative product reaches 2.3e-9
/// at iteration 12 and then plateaus at 1.05e-9 while the normal equations, by then
/// dominated by Theta entries of 1e+8, hand the factorization pivots at its floor. A
/// threshold of 1e-9 turned a point optimal to ten digits into eleven stalled iterations
/// and a `feasible` verdict; 1e-8 accepts it at iteration 12 and still clears the guard.
constexpr double kIpmComplementarity = 1e-8;
/// The relative duality gap at convergence. 1e-8, the same as the feasibility measures,
/// and MEASURED: 1e-9 and 1e-10 were tried against the verifier's strong-duality threshold
/// and stalled one or two of 150 KKT instances a hair short with the objective already
/// exact, while buying nothing on the instances the verifier rejects (pilot4, scagr25):
/// those rejections arise in original units after postsolve, where an absolute
/// complementarity product of 2e-6 is 1e-9 in the scaled units this loop converges in.
/// A tolerance in one space cannot be met by tightening a tolerance in the other; the
/// verifier's verdict on those instances is reported as it stands.
constexpr double kIpmGap = 1e-8;
/// Primal regularization rho added to every Theta^-1 (Altman & Gondzio): holds free
/// variables and keeps Theta finite as a slack goes to zero.
/// THE BARRIER'S LAST WORD (#209). Near the solution theta = 1 / (z/s) spans the gap between
/// the variables at bounds (z/s -> infinity) and the basic ones (z/s -> 0), and past some
/// point the normal equations A theta A^T are singular to working precision whatever the
/// regularization: on the 20,000-row staircase model the factorization went from 9
/// regularized pivots to 5,090 in one step, at a relative gap of 1.0e-07, and the direction
/// it produced was NaN (#205 kept the answer; this keeps the method from asking). A jump of
/// this size in one factorization, while the iterate is already within one decade of every
/// convergence tolerance, means the method has reached the resolution the barrier has left
/// - Mehrotra-type codes treat it as termination, and so does this one: the iterate in hand
/// is reported as converged, and the status guard in solve() measures it against the
/// project's tolerances like every other optimal claim. Wright, *Primal-Dual Interior-Point
/// Methods* (1997), chapter 11.
constexpr double kBarrierExhaustedSlack = 10.0;  ///< within this factor of each tolerance
/// How much the dual regularization grows when a Newton direction is not finite (#209), and
/// how many times it may grow: 1e-10 -> 1e-6 -> 1e-2 is as far as a diagonal shift can go
/// before the direction stops being a Newton direction at all.
constexpr double kRegularizationRaise = 1e4;
constexpr Count kMaxRegularizationRaises = 2;
constexpr Count kBarrierExhaustedPivots = 64;  ///< at least this many pivots regularized...
constexpr double kBarrierExhaustedFraction = 0.01;  ///< ...or this fraction of the rows

constexpr double kPrimalRegularization = 1e-8;
/// Dual regularization delta on the diagonal of the normal equations, and the pivot floor
/// the factorization enforces.
constexpr double kDualRegularization = 1e-10;
/// Fraction of the way to the boundary a step may go (Mehrotra's eta).
constexpr double kStepToBoundary = 0.995;
constexpr int kMaxIterations = 300;
/// Iterative refinement steps on each normal-equations solve.
constexpr int kRefinementSteps = 2;
constexpr int kRuizIterations = 10;
/// Iterations the loop may spend converged in its scaled measures while the model-space
/// measures fail to improve by kModelSpaceProgress before it stops and reports the point as
/// feasible (#582). Measured on finnis and d2q06c: once the scaled residuals are at their
/// floor the model-space ones do not move at all, so a short window loses nothing.
constexpr int kModelSpaceStallIterations = 3;
constexpr double kModelSpaceProgress = 0.9;
/// At most this many columns go to the dense-column correction (#467): each costs one solve
/// with the sparse factor per factorization to build the Schur complement, and the k x k
/// complement is factored densely. The densest are taken first.
constexpr Index kMaxDenseColumns = 100;

class InteriorPoint {
 public:
  InteriorPoint(const Model& model, const Options& options, Logger& logger,
                SolveControl* control = nullptr, const WarmStart* warm = nullptr,
                const Timer* clock = nullptr)
      : model_(model),
        options_(options),
        logger_(logger),
        control_(control),
        warm_(warm),
        clock_(clock) {}

  /// The Ruiz factors solve_scaled() built, so residuals() can measure the iterate in
  /// MODEL space beside the scaled space it iterates in (#582). Null means no scaling.
  void set_scaling(const Scaling* scaling) { scaling_ = scaling; }

  Solution run();

 private:
  void build();
  void apply_warm_start();
  void residuals();
  /// The model-space half of residuals() (#582).
  void model_space_residuals();
  /// The ten rows carrying the worst model-space violation, with their scale factors, at
  /// verbose, for a stop whose model-space residual is above the tolerance (#582).
  void log_worst_model_rows() const;
  /// The same table for the dual side: the ten variables with the worst model-space dual
  /// violation, their scaled dual residual and their Ruiz factor (#582).
  void log_worst_model_duals() const;
  [[nodiscard]] bool direction_is_finite() const;
  [[nodiscard]] bool factorize();

  /// The deadline handed down to the linear algebra, so a time limit is not defeated by one
  /// very long ordering or factorization (#193). Empty when there is no limit.
  SparseLdl::ShouldStop should_stop_;
  /// The limits this solve runs under, interpreted where every other engine interprets
  /// them (#289). Held by the object because should_stop_ captures `this`.
  ResourceLimits limits_;
  /// Set when the deadline fired inside normal_equations_lower(), which the factorization
  /// never got to see - run() reads it beside ldl_.stopped_early() to report a time limit
  /// rather than a numerical failure (#232).
  bool assembly_stopped_ = false;
  /// The set-up's own deadline (#357): ipm_setup_share of the time limit, measured on the
  /// solve's clock, covering the ordering AND the first factorization. A set-up that has
  /// not finished by then is abandoned and the solve DECLINES (kNotSolved) rather than
  /// running the whole budget out with nothing to show, so the dispatcher can hand the
  /// model to another engine on the time that is left. Infinite when there is no time
  /// limit or the share is 1.
  double ordering_deadline_ = std::numeric_limits<double>::infinity();
  const Timer* run_clock_ = nullptr;
  bool ordering_declined_ = false;
  void newton_direction();
  [[nodiscard]] double step_length(const std::vector<double>& s, const std::vector<double>& ds,
                                   const std::vector<double>& t,
                                   const std::vector<double>& dt) const;
  Solution finish(SolveStatus status, const std::string& message, Count iterations,
                  double seconds);

  /// c_j - a_j^T y for a fixed structural column, whose multipliers do not exist.
  [[nodiscard]] double fixed_reduced_cost(Index j) const {
    const ColumnView column = model_.matrix.column(j);
    double dot = 0.0;
    for (Index p = 0; p < column.size; ++p) {
      dot += column.values[p] * y_[static_cast<std::size_t>(column.rows[p])];
    }
    return cost_[static_cast<std::size_t>(j)] - dot;
  }

  /// Abar v for v over all n + m variables: A v_x - v_s.
  void constraint_times(const std::vector<double>& v, std::vector<double>* out) const;
  /// Abar^T w, over all n + m variables.
  void constraint_transpose_times(const std::vector<double>& w, std::vector<double>* out) const;

  const Model& model_;
  const Options& options_;
  Logger& logger_;
  SolveControl* control_ = nullptr;
  const WarmStart* warm_ = nullptr;
  /// The clock the time limit is measured on. solve_scaled() starts it BEFORE scaling and
  /// copying the model, which on a 500,000-row polish is tens of seconds that run()'s own
  /// timer never saw (#232); null means run() keeps its own.
  const Timer* clock_ = nullptr;
  /// The factor the ordering predicts is compared with polish_max_factor_nonzeros (with a
  /// warm start) or ipm_max_factor_nonzeros (#246), and factorize() sets the flag instead of
  /// building it.
  std::int64_t max_factor_nonzeros_ = -1;
  bool factor_too_large_ = false;
  /// Set when the refusal came BEFORE assembly (#467): the normal equations alone would
  /// hold this many lower-triangle nonzeros (at least), and the factor at least as many.
  std::int64_t predicted_normal_nonzeros_ = -1;
  /// The dense-column path (#467, option ipm_dense_columns): the columns split off the
  /// normal equations and corrected for by Sherman-Morrison-Woodbury inside conjugate
  /// gradients. Inactive (no columns) by default, and then nothing below changes.
  DenseColumnCorrection dense_;
  bool dense_schur_failed_ = false;
  Count pcg_iterations_ = 0;
  Count pcg_solves_ = 0;
  double worst_pcg_residual_ = 0.0;
  /// Solve the normal equations for the factors in ldl_: the plain LDL^T solve with
  /// kRefinementSteps of iterative refinement against normal_lower_, or, on the dense-column
  /// path, preconditioned conjugate gradients against the whole of A Theta A^T.
  void solve_normal(std::vector<double>* rhs);

  Index n_ = 0;
  Index m_ = 0;
  Index total_ = 0;
  std::vector<double> cost_;   ///< minimisation sense, over the n structurals (0 for logicals)
  std::vector<double> lower_;  ///< over all total_ variables
  std::vector<double> upper_;
  std::vector<bool> has_lower_;
  std::vector<bool> has_upper_;
  /// l == u: a constant, not a variable. No slack, no multiplier, no dual condition, Theta 0.
  std::vector<bool> fixed_;
  Index bound_count_ = 0;

  // Iterates.
  std::vector<double> x_, y_, sl_, zl_, su_, zu_;
  // Residuals.
  std::vector<double> r_b_, r_c_, r_l_, r_u_;
  double primal_infeasibility_ = 0.0;
  double dual_infeasibility_ = 0.0;
  /// THE SAME MEASUREMENT THE STATUS GUARD MAKES (#582): the largest primal violation in
  /// model space, each row's over max(1, the largest term of its activity sum) and each
  /// bound's over max(1, |x|), exactly Solution::recompute_quality's
  /// primal_infeasibility_scaled. On irish-electricity the scaled residual above read
  /// 1.2e-8 while this read 1.0e-4: the Ruiz factors on its worst rows are four decades,
  /// and a residual that is small in the units the loop iterates in is not small in the
  /// units the model is written in. The loop converges only when both hold.
  double model_primal_infeasibility_ = 0.0;
  /// Per row, the model-space relative violation behind model_primal_infeasibility_, for
  /// the worst-rows table at a stop that does not meet it.
  std::vector<double> model_row_violation_;
  /// THE DUAL HALF OF THE SAME MEASUREMENT (#582): the residual part of the guard's
  /// dual_infeasibility_scaled - the consistency of zl - zu with c - A^T y over each
  /// column's terms, and the sign conditions - on the point this iterate would be reported
  /// as. The scaled dual residual is Dc times the model's on a structural, and the loop
  /// divides it by 1 + ||c||, not by each column's own terms: on finnis a scaled 7.8e-9 is
  /// 7.0e-6 in the model, on a column whose Ruiz factor is 1.
  double model_dual_infeasibility_ = 0.0;
  /// The guard's complementarity_violation, the largest absolute |multiplier| * distance.
  /// Logged, not required: a product is the same number in both spaces, and the guard
  /// already downgrades an optimal claim that fails it (the claim is then not a proof, but
  /// it is also not a disagreement between two measurements, which is what #582 is).
  double model_complementarity_ = 0.0;
  std::vector<double> model_dual_violation_;
  /// The guard's primal and dual residual measures hold on the point as it would be
  /// reported: the loop may claim optimal only then (#582).
  [[nodiscard]] bool model_space_holds() const {
    return model_primal_infeasibility_ <= tol::kPrimalFeasibility &&
           model_dual_infeasibility_ <= tol::kDualFeasibility;
  }
  /// How far the worse model-space residual is from its tolerance, as a ratio (<= 1 holds).
  [[nodiscard]] double model_space_excess() const {
    return std::max(model_primal_infeasibility_ / tol::kPrimalFeasibility,
                    model_dual_infeasibility_ / tol::kDualFeasibility);
  }
  /// model_space_holds() on the point as finish() would REPORT it, which for an optimal
  /// status is after dual purification. When only the dual side fails, the purification is
  /// tried here: kept if every model-space measure then holds, rolled back otherwise, so a
  /// failed attempt leaves the iterate exactly as it was. On finnis the dual residual sat at
  /// 7.0e-6 in model units - 7.8e-9 in the loop's, where every per-term scale is replaced by
  /// 1 + ||c|| - for eight iterations; the purification is what makes that point a proof.
  /// `throttled`: inside the loop, try the purification (a refactorization) only when the
  /// model-space dual residual has improved by kModelSpaceProgress since the last try; the
  /// final stops always try it.
  [[nodiscard]] bool model_space_holds_as_reported(bool throttled = false);
  bool purified_ = false;
  double best_model_excess_ = std::numeric_limits<double>::infinity();
  int model_stalled_ = 0;
  double last_purified_dual_ = std::numeric_limits<double>::infinity();
  const Scaling* scaling_ = nullptr;
  double mu_ = 0.0;
  double max_product_ = 0.0;
  double objective_ = 0.0;

  // Normal equations.
  std::vector<double> theta_;
  SparseMatrix normal_lower_;
  SparseLdl ldl_;
  bool analyzed_ = false;

  // THE PROXIMAL PATH (#473, ipm_proximal_regularization, off by default): the regularized
  // augmented system of ipm/proximal_system.hpp in place of the normal equations. ldl_ then
  // holds ITS factors, so the reporting below reads the same object on either path; the
  // dual purification, which needs the normal equations, gets its own factorization.
  std::unique_ptr<ProximalSystem> proximal_;
  double proximal_reg_ = tol::kIpmProximalStart;
  SparseLdl purify_ldl_;
  bool purify_analyzed_ = false;
  Count refinement_steps_ = 0;
  /// The worst RELATIVE unregularized residual a refined solve handed back (review of #473).
  double worst_refinement_residual_ = 0.0;
  /// Refined solves, and those left above tol::kIpmProximalRefinementTarget. A miss is not
  /// silent: it is counted, reported in the IPM line and in the message of every stop that
  /// is not optimal, and it shrinks rho for the next factorization (refinement_missed_).
  Count refinement_solves_ = 0;
  Count refinement_misses_ = 0;
  bool refinement_missed_ = false;
  /// "augmented system" on the proximal path, "normal equations" otherwise: what a message
  /// about the factorization is about.
  [[nodiscard]] const char* system_name() const noexcept {
    return proximal_ != nullptr ? "augmented system" : "normal equations";
  }
  /// The proximal half of newton_direction(): the Newton system solved through the
  /// augmented factors with refinement on the unregularized matrix, for the given g.
  void proximal_newton_direction(const std::vector<double>& g);

  // Directions.
  std::vector<double> dx_, dy_, dsl_, dzl_, dsu_, dzu_;
  std::vector<double> r_mu_l_, r_mu_u_;
  Count factorizations_ = 0;
  Count regularized_pivots_ = 0;
  /// The dual regularization the factorization runs with (#209). It starts at
  /// kDualRegularization and is raised when a Newton direction comes back non-finite near
  /// the end, where the barrier has left the normal equations rank deficient at working
  /// precision: a stronger diagonal makes the factor well defined again at the cost of a
  /// slightly inexact direction, which the next iteration's residuals absorb.
  double dual_regularization_ = kDualRegularization;
  Count regularization_raises_ = 0;
  /// Whether the barrier-exhausted stop has already spent its one extra iteration (#392).
  bool barrier_retry_used_ = false;

  // THE BEST ITERATE IS KEPT. Near the optimum the normal equations lose conditioning and
  // an iteration can drift; when the loop then stalls or hits a limit, the point returned
  // is the best one seen by the convergence measures, not the last one computed.
  // The slacks are part of the iterate. They were not saved at first, so restoring the
  // best point left it paired with whatever slacks the broken iteration had produced, and
  // the barrier parameter recomputed from those read NaN beside a point that was fine.
  struct Snapshot {
    double merit = std::numeric_limits<double>::infinity();
    std::vector<double> x, y, zl, zu, sl, su;
  } best_;
  void remember_if_best(double merit);
  void restore_best();
  bool purify_duals();
};

void InteriorPoint::remember_if_best(double merit) {
  // A NaN ITERATE LOOKS PERFECT. Every violation measure is a running maximum built from
  // comparisons, and every comparison with NaN is false, so an iterate full of NaN reports
  // zero infeasibility and zero complementarity - a merit of exactly 0.0, better than any
  // real point. That is how a converged 20,000-row solve lost its 1e-7 iterate: the NaN
  // that replaced it scored 0.0 and was remembered as the best. The merit is not enough to
  // check, since it is the very thing that lies; the iterate itself has to be finite.
  if (!std::isfinite(merit) || !std::isfinite(mu_) || !std::isfinite(objective_)) return;
  if (!(merit < best_.merit)) return;
  best_.merit = merit;
  best_.x = x_;
  best_.y = y_;
  best_.zl = zl_;
  best_.zu = zu_;
  best_.sl = sl_;
  best_.su = su_;
}

void InteriorPoint::restore_best() {
  if (best_.x.empty()) return;
  x_ = best_.x;
  y_ = best_.y;
  zl_ = best_.zl;
  zu_ = best_.zu;
  sl_ = best_.sl;
  su_ = best_.su;
}

void InteriorPoint::build() {
  n_ = model_.num_cols();
  m_ = model_.num_rows();
  total_ = n_ + m_;
  const double sense = model_.sense_multiplier();
  cost_.assign(static_cast<std::size_t>(total_), 0.0);
  lower_.resize(static_cast<std::size_t>(total_));
  upper_.resize(static_cast<std::size_t>(total_));
  has_lower_.assign(static_cast<std::size_t>(total_), false);
  has_upper_.assign(static_cast<std::size_t>(total_), false);
  fixed_.assign(static_cast<std::size_t>(total_), false);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    cost_[u] = sense * model_.col_cost[u];
    lower_[u] = model_.col_lower[u];
    upper_[u] = model_.col_upper[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(n_ + i);
    lower_[u] = model_.row_lower[static_cast<std::size_t>(i)];
    upper_[u] = model_.row_upper[static_cast<std::size_t>(i)];
  }
  bound_count_ = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    // A FIXED VARIABLE IS A CONSTANT. Every equality row's logical is one, and the
    // bounded form would give it two slacks that both have to vanish: no interior, Theta
    // driven to zero, complementarity unreachable. It is pinned, carries no bound pair,
    // and drops out of the normal equations; its value still enters Abar x.
    if (lower_[u] == upper_[u] && is_finite_bound(lower_[u])) {
      fixed_[u] = true;
      continue;
    }
    has_lower_[u] = is_finite_bound(lower_[u]);
    has_upper_[u] = is_finite_bound(upper_[u]);
    bound_count_ += (has_lower_[u] ? 1 : 0) + (has_upper_[u] ? 1 : 0);
  }

  // STARTING POINT (Wright, sec. 11.3, simplified): every variable inside its box with a
  // margin, every slack at least 1 so the first iterate is comfortably interior, every
  // multiplier 1. The residuals r_l, r_u absorb any gap between x and its slacks.
  x_.assign(static_cast<std::size_t>(total_), 0.0);
  sl_.assign(static_cast<std::size_t>(total_), 0.0);
  su_.assign(static_cast<std::size_t>(total_), 0.0);
  zl_.assign(static_cast<std::size_t>(total_), 0.0);
  zu_.assign(static_cast<std::size_t>(total_), 0.0);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) {
      x_[u] = lower_[u];
      continue;
    }
    if (has_lower_[u] && has_upper_[u]) {
      x_[u] = 0.5 * (lower_[u] + upper_[u]);
    } else if (has_lower_[u]) {
      x_[u] = lower_[u] + 1.0;
    } else if (has_upper_[u]) {
      x_[u] = upper_[u] - 1.0;
    } else {
      x_[u] = 0.0;
    }
  }
  // THE LOGICALS START CONSISTENT WITH THE STRUCTURALS: s = A x exactly, so the constraint
  // residual r_b is zero at the first iterate and the bound residuals r_l, r_u carry the
  // whole infeasibility. Starting every logical at "bound + 1" instead left r_b of the
  // size of A x, and the affine direction's attempt to close it in one step sent mu from
  // 1 to 1e+9 on israel and stocfor1 before the method could recover - or not.
  if (m_ > 0) {
    std::vector<double> activity(static_cast<std::size_t>(m_), 0.0);
    model_.matrix.multiply_add(x_.data(), activity.data());
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(n_ + i);
      if (!fixed_[u]) x_[u] = activity[static_cast<std::size_t>(i)];
    }
  }
  // THE DUALS START CONSISTENT WITH THE COSTS, for the same reason: with y = 0 the dual
  // residual is c - z_l + z_u, and z_l = max(c, 1), z_u = max(-c, 1) makes it vanish
  // wherever |c| >= 1 and small elsewhere. Starting every multiplier at 1 instead left a
  // dual residual the size of the scaled cost vector, and the first dual step took z to
  // that size at once: on stocfor1 (scaled) mu went 1 -> 3e6 in four iterations and the
  // run never recovered. The primal slacks are floored at 1 so the first iterate is
  // comfortably interior; r_l and r_u absorb whatever that costs in consistency.
  //
  // THE SLACKS ARE SHIFTED TOGETHER, NOT FLOORED ONE BY ONE (#375). Flooring each slack at 1
  // left the residual of a badly placed variable - a logical whose row activity at the
  // midpoint start sits 1e4 outside its row bounds - as r_u = -1e4 against a slack of 1, and
  // the Newton direction that closes it drives other slacks negative at once: on stocfor2
  // the primal step was 1e-3 for thirty iterations while mu climbed from 2 to 5e8, and the
  // solve reached the iteration limit. Mehrotra's starting point (Mehrotra, SIAM J. Optim.
  // 2 (1992), sec. 7; Wright, Primal-Dual Interior-Point Methods, sec. 11.3) shifts EVERY
  // slack by the same amount, 1.5 times the worst violation, so the step that repairs the
  // worst one is affordable everywhere, and then balances slacks against multipliers so
  // the complementarity products start comparable. The residuals r_l, r_u carry the
  // uniform shift instead of one variable's whole violation.
  double worst_slack = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    if (has_lower_[u]) worst_slack = std::min(worst_slack, x_[u] - lower_[u]);
    if (has_upper_[u]) worst_slack = std::min(worst_slack, upper_[u] - x_[u]);
  }
  const double slack_shift = std::max(1.0, -1.5 * worst_slack);
  double product_sum = 0.0;
  double slack_sum = 0.0;
  double dual_sum = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    if (has_lower_[u]) {
      sl_[u] = x_[u] - lower_[u] + slack_shift;
      zl_[u] = std::max(cost_[u], 1.0);
      product_sum += sl_[u] * zl_[u];
      slack_sum += sl_[u];
      dual_sum += zl_[u];
    }
    if (has_upper_[u]) {
      su_[u] = upper_[u] - x_[u] + slack_shift;
      zu_[u] = std::max(-cost_[u], 1.0);
      product_sum += su_[u] * zu_[u];
      slack_sum += su_[u];
      dual_sum += zu_[u];
    }
  }
  if (product_sum > 0.0) {
    const double slack_balance = 0.5 * product_sum / dual_sum;
    const double dual_balance = 0.5 * product_sum / slack_sum;
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (fixed_[u]) continue;
      if (has_lower_[u]) {
        sl_[u] += slack_balance;
        zl_[u] += dual_balance;
      }
      if (has_upper_[u]) {
        su_[u] += slack_balance;
        zu_[u] += dual_balance;
      }
    }
  }
  y_.assign(static_cast<std::size_t>(m_), 0.0);
  if (warm_ != nullptr) apply_warm_start();
  const auto alloc = [&](std::vector<double>& v, Index size) {
    v.assign(static_cast<std::size_t>(size), 0.0);
  };
  alloc(r_b_, m_);
  alloc(r_c_, total_);
  alloc(r_l_, total_);
  alloc(r_u_, total_);
  alloc(theta_, total_);
  alloc(dx_, total_);
  alloc(dy_, m_);
  alloc(dsl_, total_);
  alloc(dzl_, total_);
  alloc(dsu_, total_);
  alloc(dzu_, total_);
  alloc(r_mu_l_, total_);
  alloc(r_mu_u_, total_);
}

void InteriorPoint::constraint_times(const std::vector<double>& v,
                                     std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.multiply_add(v.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(i)] -= v[static_cast<std::size_t>(n_ + i)];
  }
}

void InteriorPoint::constraint_transpose_times(const std::vector<double>& w,
                                               std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.transpose_multiply_add(w.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(n_ + i)] = -w[static_cast<std::size_t>(i)];
  }
}

// A WARM START REPLACES THE POINT, NOT THE FLOORS. The caller's x is projected into its
// box and kept; the logicals are set to A x so r_b starts at zero exactly as the cold start
// arranges; y is the caller's row duals in minimisation sense; and each multiplier pair is
// read off the caller's reduced cost, z_l - z_u = d. What a first-order point cannot
// supply is an interior: its slacks and multipliers are within 1e-6 of zero on every
// active bound, and a Newton step from there is cut to nothing by the fraction-to-boundary
// rule. So the slacks and multipliers are floored at polish_start_margin and r_l, r_u, r_c
// absorb the inconsistency, as they do for the cold start's floor of 1 - the floor is what
// the method has left to remove. The margin was measured, not chosen: on the staircase
// family 0.1 polishes the 1,000- and 5,000-row instances in 7 and 12 iterations against a
// cold start's 18 and 24, while 0.01, 0.001 and 0.0001 start so close to the boundary that
// the barrier vanishes before the residuals do and the factorization breaks down at the
// end (the failure #205 made survivable). Closer is not better here.
//
// A logical's column of Abar = [A -I] is -e_i, so its reduced cost is y_i itself.

void InteriorPoint::apply_warm_start() {
  const WarmStart& w = *warm_;
  const double kWarmFloor = options_.get_double("polish_start_margin");
  if (static_cast<Index>(w.col_value.size()) != n_ ||
      static_cast<Index>(w.row_dual.size()) != m_ ||
      static_cast<Index>(w.col_dual.size()) != n_) {
    warm_ = nullptr;
    return;
  }
  for (const std::vector<double>* v : {&w.col_value, &w.row_dual, &w.col_dual}) {
    for (const double value : *v) {
      if (!std::isfinite(value)) {
        warm_ = nullptr;
        return;
      }
    }
  }
  const double sense = model_.sense_multiplier();
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (fixed_[u]) continue;
    double x = w.col_value[u];
    if (has_lower_[u]) x = std::max(x, lower_[u]);
    if (has_upper_[u]) x = std::min(x, upper_[u]);
    x_[u] = x;
  }
  if (m_ > 0) {
    std::vector<double> activity(static_cast<std::size_t>(m_), 0.0);
    model_.matrix.multiply_add(x_.data(), activity.data());
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(n_ + i);
      if (!fixed_[u]) x_[u] = activity[static_cast<std::size_t>(i)];
    }
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(i);
    y_[u] = sense * w.row_dual[u];
  }
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    const double d = k < n_ ? sense * w.col_dual[u] : y_[static_cast<std::size_t>(k - n_)];
    if (has_lower_[u]) {
      sl_[u] = std::max(x_[u] - lower_[u], kWarmFloor);
      zl_[u] = std::max(d, kWarmFloor);
    }
    if (has_upper_[u]) {
      su_[u] = std::max(upper_[u] - x_[u], kWarmFloor);
      zu_[u] = std::max(-d, kWarmFloor);
    }
  }
}

bool InteriorPoint::direction_is_finite() const {
  const auto finite = [](const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
  };
  return finite(dx_) && finite(dy_) && finite(dsl_) && finite(dzl_) && finite(dsu_) &&
         finite(dzu_);
}

void InteriorPoint::residuals() {
  // r_b = -(Abar x); r_c = c - Abar^T y - z_l + z_u; r_l = x - l - s_l; r_u = u - x - s_u.
  constraint_times(x_, &r_b_);
  double x_norm = 0.0;
  for (Index i = 0; i < m_; ++i)
    r_b_[static_cast<std::size_t>(i)] = -r_b_[static_cast<std::size_t>(i)];
  constraint_transpose_times(y_, &r_c_);
  double c_norm = 0.0;
  double complementarity = 0.0;
  max_product_ = 0.0;
  objective_ = 0.0;
  primal_infeasibility_ = 0.0;
  dual_infeasibility_ = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    r_c_[u] = fixed_[u] ? 0.0 : cost_[u] - r_c_[u] - zl_[u] + zu_[u];
    objective_ += cost_[u] * x_[u];
    c_norm = std::max(c_norm, std::fabs(cost_[u]));
    x_norm = std::max(x_norm, std::fabs(x_[u]));
    dual_infeasibility_ = std::max(dual_infeasibility_, std::fabs(r_c_[u]));
    if (has_lower_[u]) {
      r_l_[u] = x_[u] - lower_[u] - sl_[u];
      complementarity += sl_[u] * zl_[u];
      max_product_ = std::max(max_product_, sl_[u] * zl_[u] / (1.0 + std::fabs(x_[u])));
      primal_infeasibility_ = std::max(primal_infeasibility_, std::fabs(r_l_[u]));
    }
    if (has_upper_[u]) {
      r_u_[u] = upper_[u] - x_[u] - su_[u];
      complementarity += su_[u] * zu_[u];
      max_product_ = std::max(max_product_, su_[u] * zu_[u] / (1.0 + std::fabs(x_[u])));
      primal_infeasibility_ = std::max(primal_infeasibility_, std::fabs(r_u_[u]));
    }
  }
  for (Index i = 0; i < m_; ++i) {
    primal_infeasibility_ =
        std::max(primal_infeasibility_, std::fabs(r_b_[static_cast<std::size_t>(i)]));
  }
  primal_infeasibility_ /= 1.0 + x_norm;
  dual_infeasibility_ /= 1.0 + c_norm;
  mu_ = bound_count_ > 0 ? complementarity / static_cast<double>(bound_count_) : 0.0;
  model_space_residuals();
}

void InteriorPoint::model_space_residuals() {
  // The guard's own measurement, made on the scaled iterate by ipm/model_space.cpp: rows
  // and bounds over their terms for the primal side, the consistency of zl - zu with
  // c - A^T y, the sign conditions and the relative products for the dual side, all in the
  // model's units. One pass over the matrix per side, the cost of the residual itself.
  ScaledIterate it;
  it.matrix = &model_.matrix;
  it.cost = &cost_;
  it.lower = &lower_;
  it.upper = &upper_;
  it.x = &x_;
  it.y = &y_;
  it.zl = &zl_;
  it.zu = &zu_;
  it.scaling = scaling_;
  ModelSpaceMeasure measure = measure_in_model_space(it);
  model_primal_infeasibility_ = measure.primal;
  model_dual_infeasibility_ = measure.dual_residual;
  model_complementarity_ = measure.complementarity;
  model_row_violation_ = std::move(measure.row_violation);
  model_dual_violation_ = std::move(measure.dual_violation);
}

void InteriorPoint::log_worst_model_rows() const {
  if (model_row_violation_.empty()) return;
  std::vector<Index> order(static_cast<std::size_t>(m_));
  for (Index i = 0; i < m_; ++i) order[static_cast<std::size_t>(i)] = i;
  const auto worse = [&](Index a, Index b) {
    return model_row_violation_[static_cast<std::size_t>(a)] >
           model_row_violation_[static_cast<std::size_t>(b)];
  };
  const std::size_t shown = std::min<std::size_t>(10, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                    order.end(), worse);
  logger_.verbose(
      "interior point: model-space primal infeasibility {:.3e} against {:.0e}; the worst "
      "rows (scaled residual | Ruiz row factor | model-space relative violation):",
      model_primal_infeasibility_, tol::kPrimalFeasibility);
  for (std::size_t q = 0; q < shown; ++q) {
    const Index i = order[q];
    const auto u = static_cast<std::size_t>(i);
    const double factor = scaling_ == nullptr ? 1.0 : scaling_->row[u];
    const std::string name = u < model_.row_names.size() && !model_.row_names[u].empty()
                                 ? model_.row_names[u]
                                 : fmt::format("row {}", i);
    logger_.verbose("  {:<24} {:.3e} | {:.3e} | {:.3e}", name, std::fabs(r_b_[u]), factor,
                    model_row_violation_[u]);
  }
}

void InteriorPoint::log_worst_model_duals() const {
  if (model_dual_violation_.empty()) return;
  std::vector<Index> order(static_cast<std::size_t>(total_));
  for (Index k = 0; k < total_; ++k) order[static_cast<std::size_t>(k)] = k;
  const auto worse = [&](Index a, Index b) {
    return model_dual_violation_[static_cast<std::size_t>(a)] >
           model_dual_violation_[static_cast<std::size_t>(b)];
  };
  const std::size_t shown = std::min<std::size_t>(10, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                    order.end(), worse);
  logger_.verbose(
      "interior point: model-space dual infeasibility {:.3e} against {:.0e}; the worst "
      "variables (scaled dual residual | Ruiz factor | model-space relative violation):",
      model_dual_infeasibility_, tol::kDualFeasibility);
  for (std::size_t q = 0; q < shown; ++q) {
    const Index k = order[q];
    const auto u = static_cast<std::size_t>(k);
    std::string name;
    double factor = 1.0;
    if (k < n_) {
      name = u < model_.col_names.size() && !model_.col_names[u].empty()
                 ? model_.col_names[u]
                 : fmt::format("column {}", k);
      if (scaling_ != nullptr) factor = scaling_->column[u];
    } else {
      const auto r = static_cast<std::size_t>(k - n_);
      name = r < model_.row_names.size() && !model_.row_names[r].empty()
                 ? fmt::format("row {}", model_.row_names[r])
                 : fmt::format("row {}", k - n_);
      if (scaling_ != nullptr) factor = scaling_->row[r];
    }
    logger_.verbose("  {:<24} {:.3e} | {:.3e} | {:.3e}", name, std::fabs(r_c_[u]), factor,
                    model_dual_violation_[u]);
  }
}

bool InteriorPoint::factorize() {
  // Theta = (z_l/s_l + z_u/s_u + rho)^-1 over every variable; the logical block of
  // Abar Theta Abar^T is the diagonal Theta_s, added as a row shift.
  std::vector<double> theta_x(static_cast<std::size_t>(n_));
  std::vector<double> row_shift(static_cast<std::size_t>(m_));
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    double inverse = kPrimalRegularization;
    if (has_lower_[u]) inverse += zl_[u] / sl_[u];
    if (has_upper_[u]) inverse += zu_[u] / su_[u];
    theta_[u] = fixed_[u] ? 0.0 : 1.0 / inverse;
    if (k < n_) {
      theta_x[u] = theta_[u];
    } else {
      row_shift[static_cast<std::size_t>(k - n_)] = theta_[u];
    }
  }
  Profiler* profiler = logger_.profiler();
  // THE SIZE IS KNOWN BEFORE THE MATRIX IS BUILT (#467). The factor's pattern contains the
  // lower triangle of the normal equations, so a normal-equations matrix over the factor
  // budget is a factor over it too, and the ordering would refuse it - after an assembly
  // that on Linf_520c took 112 s and 474 million nonzeros, and on bdry2 would need 7.9e9.
  // Counting from the pattern of A costs at most the assembly's arithmetic and none of its
  // memory, and usually one pass over the column counts.
  if (!analyzed_ && max_factor_nonzeros_ >= 0) {
    std::vector<char> skip(static_cast<std::size_t>(n_), 0);
    for (Index j = 0; j < n_; ++j) {
      const auto u = static_cast<std::size_t>(j);
      skip[u] = fixed_[u] || (dense_.active() && dense_.mask()[u] != 0) ? 1 : 0;
    }
    const NormalPrediction prediction =
        predict_normal_nonzeros(model_.matrix, skip, max_factor_nonzeros_, should_stop_);
    if (prediction.stopped) {
      assembly_stopped_ = true;
      return false;
    }
    if (prediction.over_cap) {
      factor_too_large_ = true;
      predicted_normal_nonzeros_ = prediction.nonzeros;
      return false;
    }
    logger_.verbose("interior point: normal equations predicted at {} {} nonzeros",
                    prediction.exact ? "exactly" : "at most", prediction.nonzeros);
  }
  std::vector<double> theta_sparse;
  std::vector<double> assembly_shift;
  if (dense_.active()) {
    dense_.sparse_theta(theta_x, &theta_sparse);
    (void)dense_.preconditioner_shift(model_.matrix, theta_x, row_shift, dual_regularization_,
                                      &assembly_shift);
  }
  bool assembled = false;
  // The proximal path (#473) factors the augmented system instead; its Theta^-1 carries no
  // regularization, which the system adds itself as rho and delta.
  std::vector<double> theta_inverse;
  if (proximal_ != nullptr) {
    theta_inverse.assign(static_cast<std::size_t>(total_), 0.0);
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (has_lower_[u]) theta_inverse[u] += zl_[u] / sl_[u];
      if (has_upper_[u]) theta_inverse[u] += zu_[u] / su_[u];
    }
    // The floor rises with each recovery from a non-finite direction (#209), by the
    // proximal path's own factor and to its own cap (1e-8 -> 1e-6 -> 1e-4), not the default
    // path's x1e4, which took rho to 1. A refinement that missed its target since the last
    // factorization shrinks rho toward that floor first.
    const double floor = ProximalSystem::recovery_floor(regularization_raises_);
    proximal_reg_ =
        ProximalSystem::next_regularization(proximal_reg_, mu_, floor, refinement_missed_);
    refinement_missed_ = false;
    proximal_->assemble(theta_inverse, proximal_reg_, proximal_reg_);
    assembled = true;
  } else {
    ProfileScope timed(profiler, "normal equations", ProfileMode::kDetailed);
    assembled = normal_equations_lower(model_.matrix, dense_.active() ? theta_sparse : theta_x,
                                       dense_.active() ? assembly_shift : row_shift,
                                       dual_regularization_, &normal_lower_, should_stop_);
  }
  const SparseMatrix& system = proximal_ != nullptr ? proximal_->matrix() : normal_lower_;
  if (!assembled) {
    assembly_stopped_ = true;
    return false;
  }
  // The set-up - the ordering and the FIRST factorization - answers to two clocks: the
  // solve's deadline, and its own share of it (#357). On the random 20,000-row scale shape
  // the ordering finishes in 2 s but predicts a 57-million-nonzero factor, and that first
  // factorization ran the whole 120 s out with 0 iterations; abandoning it at the share
  // turns that into a decline another engine can act on. Later factorizations run under
  // the solve's deadline alone: by then the cost per iteration is known and paid for.
  bool setup_past_share = false;
  const SparseLdl::ShouldStop setup_stop = [this, &setup_past_share] {
    if (should_stop_ && should_stop_()) return true;
    if (run_clock_ != nullptr && run_clock_->elapsed_seconds() > ordering_deadline_) {
      setup_past_share = true;
      return true;
    }
    return false;
  };
  if (!analyzed_) {
    logger_.verbose("interior point: {} assembled ({} nonzeros) at {:.2f}s", system_name(),
                    system.num_nonzeros(),
                    clock_ != nullptr ? clock_->elapsed_seconds() : -1.0);
    Timer ordering_clock;
    ldl_.set_factor_budget(max_factor_nonzeros_);
    bool analysed = false;
    {
      ProfileScope timed(profiler, "ordering", ProfileMode::kDetailed);
      analysed = ldl_.analyze(system, setup_stop);
    }
    if (!analysed) {
      // The pattern count passed the cap before the pattern was stored (#246); the exact
      // size is unknown, and the message below says "more than".
      if (ldl_.factor_too_large()) factor_too_large_ = true;
      if (setup_past_share && !(should_stop_ && should_stop_())) ordering_declined_ = true;
      return false;
    }
    logger_.verbose(
        "interior point: {} {} nonzeros, ordered and analysed in "
        "{:.2f}s, factor {} nonzeros",
        system_name(), system.num_nonzeros(), ordering_clock.elapsed_seconds(),
        ldl_.factor_nonzeros() + ldl_.dimension());
    analyzed_ = true;
    // The ordering knows the factor's size before a single entry of it exists. A polish
    // that would need a 9-million-nonzero factor for a 7,000-row random-family model (#193)
    // is not a polish, and the caller has a perfectly good first-order answer to keep.
    if (max_factor_nonzeros_ >= 0 &&
        static_cast<std::int64_t>(ldl_.factor_nonzeros() + ldl_.dimension()) >
            max_factor_nonzeros_) {
      factor_too_large_ = true;
      return false;
    }
  }
  bool factored = false;
  {
    ProfileScope timed(profiler, "factorization", ProfileMode::kDetailed);
    const SparseLdl::ShouldStop& stop = factorizations_ == 0 ? setup_stop : should_stop_;
    if (proximal_ != nullptr) {
      int attempts = 0;
      factored = proximal_->factorize(ldl_, theta_inverse, &proximal_reg_, stop, &attempts);
      factorizations_ += attempts > 1 ? attempts - 1 : 0;
    } else {
      factored = ldl_.factorize(normal_lower_, dual_regularization_, stop);
    }
  }
  if (!factored) {
    if (setup_past_share && !(should_stop_ && should_stop_())) ordering_declined_ = true;
    return false;
  }
  ++factorizations_;
  regularized_pivots_ += ldl_.regularized_pivots();
  if (dense_.active() &&
      !dense_.prepare(ldl_, model_.matrix, theta_x, row_shift, dual_regularization_)) {
    dense_schur_failed_ = true;
    return false;
  }
  return true;
}

void InteriorPoint::solve_normal(std::vector<double>* rhs) {
  if (dense_.active()) {
    const PcgReport report = dense_.solve(rhs->data());
    ++pcg_solves_;
    pcg_iterations_ += report.iterations;
    worst_pcg_residual_ = std::max(worst_pcg_residual_, report.relative_residual);
    return;
  }
  // Solve with iterative refinement against the matrix actually built (the factors carry
  // the regularization; the residual is measured against the unregularized-by-pivot M).
  const std::vector<double> b = *rhs;
  ldl_.solve(rhs->data());
  const auto multiply_normal = [&](const std::vector<double>& v, std::vector<double>* out) {
    // M is stored as its lower triangle: M v = L v + L^T v - diag v.
    std::fill(out->begin(), out->end(), 0.0);
    for (Index j = 0; j < m_; ++j) {
      const ColumnView column = normal_lower_.column(j);
      const double vj = v[static_cast<std::size_t>(j)];
      double dot = 0.0;
      for (Index p = 0; p < column.size; ++p) {
        const Index i = column.rows[p];
        const double a = column.values[p];
        (*out)[static_cast<std::size_t>(i)] += a * vj;
        if (i != j) dot += a * v[static_cast<std::size_t>(i)];
      }
      (*out)[static_cast<std::size_t>(j)] += dot;
    }
  };
  std::vector<double> residual(static_cast<std::size_t>(m_));
  for (int step = 0; step < kRefinementSteps; ++step) {
    multiply_normal(*rhs, &residual);
    for (Index i = 0; i < m_; ++i) {
      residual[static_cast<std::size_t>(i)] =
          b[static_cast<std::size_t>(i)] - residual[static_cast<std::size_t>(i)];
    }
    ldl_.solve(residual.data());
    for (Index i = 0; i < m_; ++i)
      (*rhs)[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
  }
}

/// One Newton direction for the current r_mu terms: solves the normal equations for dy,
/// then recovers dx, ds, dz. The factorization in ldl_ is the current one.
void InteriorPoint::newton_direction() {
  // g_k = r_c - r_mu_l/s_l + z_l r_l/s_l + r_mu_u/s_u - z_u r_u/s_u; rhs = r_b + Abar Theta g.
  std::vector<double> g(static_cast<std::size_t>(total_));
  std::vector<double> theta_g(static_cast<std::size_t>(total_));
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    double value = r_c_[u];
    if (has_lower_[u]) value += (-r_mu_l_[u] + zl_[u] * r_l_[u]) / sl_[u];
    if (has_upper_[u]) value += (r_mu_u_[u] - zu_[u] * r_u_[u]) / su_[u];
    g[u] = value;
    theta_g[u] = theta_[u] * value;
  }
  if (proximal_ != nullptr) {
    proximal_newton_direction(g);
    return;
  }
  std::vector<double> rhs(static_cast<std::size_t>(m_));
  constraint_times(theta_g, &rhs);
  for (Index i = 0; i < m_; ++i)
    rhs[static_cast<std::size_t>(i)] += r_b_[static_cast<std::size_t>(i)];

  dy_ = rhs;
  solve_normal(&dy_);

  // dx = Theta (Abar^T dy - g); ds_l = dx + r_l; ds_u = -dx + r_u;
  // dz_l = (r_mu_l - z_l ds_l)/s_l; dz_u = (r_mu_u - z_u ds_u)/s_u.
  constraint_transpose_times(dy_, &dx_);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    dx_[u] = theta_[u] * (dx_[u] - g[u]);
    if (has_lower_[u]) {
      dsl_[u] = dx_[u] + r_l_[u];
      dzl_[u] = (r_mu_l_[u] - zl_[u] * dsl_[u]) / sl_[u];
    }
    if (has_upper_[u]) {
      dsu_[u] = -dx_[u] + r_u_[u];
      dzu_[u] = (r_mu_u_[u] - zu_[u] * dsu_[u]) / su_[u];
    }
  }
}

void InteriorPoint::proximal_newton_direction(const std::vector<double>& g) {
  const ProximalSystem::Refinement refined =
      proximal_->solve(ldl_, g, r_b_, tol::kIpmProximalRefinementSteps, &dx_, &dy_);
  refinement_steps_ += refined.steps;
  ++refinement_solves_;
  // THE REFINEMENT CAN FAIL, AND NOT SILENTLY (review of #473). The residual is measured
  // relative to the right-hand side, so the target means the same on every model. A finite
  // miss is counted and shrinks rho for the next factorization; a non-finite one is a
  // non-finite direction, which the #209 recovery in run() handles.
  const double relative = refined.relative_residual();
  if (std::isfinite(relative)) {
    worst_refinement_residual_ = std::max(worst_refinement_residual_, relative);
    if (relative > tol::kIpmProximalRefinementTarget) {
      ++refinement_misses_;
      refinement_missed_ = true;
    }
  }
  // ds and dz from dx exactly as the normal-equations path recovers them.
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u]) {
      dsl_[u] = dx_[u] + r_l_[u];
      dzl_[u] = (r_mu_l_[u] - zl_[u] * dsl_[u]) / sl_[u];
    }
    if (has_upper_[u]) {
      dsu_[u] = -dx_[u] + r_u_[u];
      dzu_[u] = (r_mu_u_[u] - zu_[u] * dsu_[u]) / su_[u];
    }
  }
}

double InteriorPoint::step_length(const std::vector<double>& s, const std::vector<double>& ds,
                                  const std::vector<double>& t,
                                  const std::vector<double>& dt) const {
  double alpha = 1.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] && ds[u] < 0.0) alpha = std::min(alpha, -s[u] / ds[u]);
    if (has_upper_[u] && dt[u] < 0.0) alpha = std::min(alpha, -t[u] / dt[u]);
  }
  return alpha;
}

Solution InteriorPoint::finish(SolveStatus status, const std::string& message, Count iterations,
                               double seconds) {
  Solution solution;
  solution.allocate_for(model_);
  solution.status = status;
  solution.algorithm = "ipm";
  solution.message = message;
  solution.iterations = iterations;
  solution.solve_seconds = seconds;
  if (proximal_ == nullptr) {
    logger_.info("IPM: {} iterations, {} factorizations, {} regularized pivot(s) in total",
                 iterations, factorizations_, regularized_pivots_);
  } else {
    const std::string refinement =
        fmt::format("{} of {} refined solve(s) above the {:.0e} refinement target",
                    refinement_misses_, refinement_solves_, tol::kIpmProximalRefinementTarget);
    logger_.info("IPM: {} iterations, {} factorizations, {} regularized pivot(s) in total, {}",
                 iterations, factorizations_, regularized_pivots_, refinement);
    logger_.verbose(
        "interior point: proximal regularization {:.1e} at the end, {} refinement "
        "corrections on the unregularized system, worst relative residual kept {:.1e}",
        proximal_reg_, refinement_steps_, worst_refinement_residual_);
    // Every stop that is not a proof says how the refinement went: a feasible point or a
    // limit reached on directions that missed their target is a different story from one
    // reached on exact ones.
    if (status != SolveStatus::kOptimal && status != SolveStatus::kNotSolved) {
      solution.message += (solution.message.empty() ? "" : "; ") + refinement;
    }
  }
  if (dense_.active()) {
    logger_.info(
        "IPM: dense-column correction over {} column(s): {} conjugate-gradient step(s) in {} "
        "solve(s), worst relative residual {:.1e}",
        dense_.columns().size(), pcg_iterations_, pcg_solves_, worst_pcg_residual_);
  }
  // THE WORST ROWS AND VARIABLES, AT EVERY STOP THAT HANDS BACK A POINT (#582). A limit stop
  // restores the best iterate without re-measuring it, so it is measured here first: the
  // table must describe the point that is returned. On irish-electricity the stop that shows
  // the disagreement is the time limit, which the table did not cover before.
  const bool returns_point =
      status == SolveStatus::kOptimal || status == SolveStatus::kFeasible ||
      status == SolveStatus::kIterationLimit || status == SolveStatus::kTimeLimit ||
      status == SolveStatus::kInterrupted;
  if (returns_point && logger_.enabled(LogLevel::kVerbose)) {
    residuals();
    if (model_primal_infeasibility_ > tol::kPrimalFeasibility) log_worst_model_rows();
    if (model_dual_infeasibility_ > tol::kDualFeasibility) log_worst_model_duals();
  }
  bool have_point = status == SolveStatus::kOptimal || status == SolveStatus::kFeasible ||
                    status == SolveStatus::kIterationLimit ||
                    status == SolveStatus::kTimeLimit || status == SolveStatus::kInterrupted;

  // A LIMIT IS NOT A LICENCE TO REPORT NONSENSE (#194). Reaching the time limit means the
  // iterate in hand is the answer, and normally it is a real point. It is not one if the
  // iteration has broken down numerically: on a generated 5000x5000 instance this method ran
  // out of time with x_ full of NaN, reported it as its point, and the objective computed
  // from it went into a results CSV as `nan` - into the file that is this project's only
  // evidence, in the column that says whether the answer was right.
  //
  // A NaN is not a number a reader can compare, and it does not announce itself: it
  // propagates through every arithmetic operation downstream looking like data. So the
  // iterate is checked before it is called a point, and a broken one is reported as the
  // numerical failure it is.
  if (have_point) {
    const bool finite_point =
        std::all_of(x_.begin(), x_.begin() + n_, [](double v) { return std::isfinite(v); });
    if (!finite_point) {
      have_point = false;
      status = SolveStatus::kNumericalError;
      solution.status = status;
      solution.message =
          message +
          "; the iterate is not finite, so it is reported as a numerical failure rather than "
          "as a point (#194)";
    }
  }

  // THE BEST ITERATE, ATTACHED TO A NUMERICAL ERROR FOR THE CROSSOVER (#474). A numerical
  // error claims no point (#200), and without crossover_from_nonoptimal its vectors stay zero.
  // With it, the best finite iterate the loop kept is written into them so the crossover can
  // start from it; the status stays numerical_error, and crossover_when_wanted() zeroes the
  // vectors again unless the simplex turns them into a vertex it proves. Never for a polish
  // (a warm start), whose caller keeps its own first-order answer instead.
  bool attach_best = false;
  if (!have_point && status == SolveStatus::kNumericalError && warm_ == nullptr &&
      options_.get_bool("crossover") && options_.get_bool("crossover_from_nonoptimal") &&
      !best_.x.empty()) {
    restore_best();
    const auto finite = [](const std::vector<double>& v) {
      return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
    };
    attach_best = finite(x_) && finite(y_) && finite(zl_) && finite(zu_);
    if (attach_best) {
      solution.message += fmt::format(
          "; the best iterate (merit {:.1e}) is attached for crossover_from_nonoptimal, and "
          "is not claimed as a point",
          best_.merit);
    }
  }
  if (!have_point && !attach_best) {
    solution.recompute_quality(model_);
    solution.dual_bound = model_.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
    return solution;
  }
  if (status == SolveStatus::kOptimal && !purified_) purify_duals();
  const double sense = model_.sense_multiplier();
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_value[u] = x_[u];
    // The reduced cost is what the dual constraint says it is at convergence: z_l - z_u;
    // for a fixed column, c - a^T y, which no sign condition constrains.
    solution.col_dual[u] = sense * (fixed_[u] ? fixed_reduced_cost(j) : zl_[u] - zu_[u]);
  }
  for (Index i = 0; i < m_; ++i) {
    solution.row_dual[static_cast<std::size_t>(i)] = sense * y_[static_cast<std::size_t>(i)];
  }
  // No basis: the statuses stay as allocate_for() left them, and the header says why.
  solution.col_status.clear();
  solution.row_status.clear();
  if (status == SolveStatus::kOptimal) {
    solution.dual_bound = model_.evaluate_objective(solution.col_value.data());
  } else {
    solution.dual_bound = model_.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model_);
  return solution;
}

// DUAL PURIFICATION (#209). The barrier method converges in its own measures - on the
// 20,000-row staircase model to a relative gap of 4e-8 with a dual residual of 5e-12 - and
// the status guard in solve() still finds one column, interior by three units, carrying a
// recomputed reduced cost of 1.7e-5: the residual of y on that column, tiny in the norm the
// loop watches and 12% over the tolerance in the per-term measure the guard uses. At an
// optimum the reduced cost of every strictly interior column is exactly zero, and that is a
// linear condition on y alone: A_I^T y = c_I over the interior set I. So the last thing the
// method does is the dual half of a crossover (Bixby, "Solving real-world linear programs",
// Operations Research 50 (2002), sec. 4 on crossover; Andersen & Ye, "Combining interior
// point and pivoting algorithms", Management Science 42 (1996)): the least-squares
// correction dy = (A_I A_I^T + eps I)^-1 A_I (c_I - A_I^T y) through the same normal
// equations the iterations use, accepted only if the guard's own measure of the duals,
// evaluated here on the working problem, gets smaller. The primal point is not touched, so
// nothing the primal side proved is put at risk; a correction that does not help is
// discarded and the log says so.
namespace {
constexpr double kPurifyInteriorFraction =
    1e-5;                              ///< slack per unit of |x| that counts as interior
constexpr double kPurifyShift = 1e-8;  ///< diagonal shift on rows with no interior logical
}  // namespace

bool InteriorPoint::model_space_holds_as_reported(bool throttled) {
  if (model_space_holds()) return true;
  // Purification moves y and z only; a primal failure is not its to repair.
  if (model_primal_infeasibility_ > tol::kPrimalFeasibility) return false;
  // A purification is a full refactorization of the normal equations, and a failed one is
  // rolled back but still paid for. Inside the loop it is retried only once the dual
  // residual has moved by kModelSpaceProgress since the last try, the same progress the
  // stall test asks for (review of #611); the final stops always try it.
  if (throttled && model_dual_infeasibility_ >= kModelSpaceProgress * last_purified_dual_) {
    return false;
  }
  last_purified_dual_ = model_dual_infeasibility_;
  const std::vector<double> y = y_;
  const std::vector<double> zl = zl_;
  const std::vector<double> zu = zu_;
  if (purify_duals()) {
    model_space_residuals();
    if (model_space_holds()) {
      purified_ = true;
      return true;
    }
  }
  y_ = y;
  zl_ = zl;
  zu_ = zu;
  model_space_residuals();
  return false;
}

bool InteriorPoint::purify_duals() {
  // On the proximal path ldl_ holds the augmented system's factors; the normal equations
  // the purification needs are ordered once, on first use, into their own object (#473).
  if (m_ == 0 || total_ == 0 || (!analyzed_ && proximal_ == nullptr)) return false;
  const auto T = static_cast<std::size_t>(total_);
  const auto M = static_cast<std::size_t>(m_);

  const auto reduced_costs = [&](const std::vector<double>& y, std::vector<double>* d) {
    constraint_transpose_times(y, d);
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      (*d)[u] = fixed_[u] ? 0.0 : cost_[u] - (*d)[u];
    }
  };
  // Two measures, both in the working problem's units and each relative to its own terms as
  // Solution::recompute_quality measures: what the step targets - the reduced cost of every
  // interior variable, which should be exactly zero - and what it must not break - a
  // multiplier pushing against a bound that does not exist. The complementarity products of
  // the near-active variables are left out on purpose: x sitting 1e-9 inside its bound with
  // a reduced cost of 1 is a product no change of y can move, and it is the largest term
  // on the model that motivated this step, so judging by it would reject every correction.
  const auto column_scale = [&](const std::vector<double>& y, Index k) {
    const auto u = static_cast<std::size_t>(k);
    double scale = std::max(1.0, std::fabs(cost_[u]));
    if (k < n_) {
      const ColumnView column = model_.matrix.column(k);
      for (Index q = 0; q < column.size; ++q) {
        scale = std::max(
            scale, std::fabs(column.values[q] * y[static_cast<std::size_t>(column.rows[q])]));
      }
    } else {
      scale = std::max(scale, std::fabs(y[static_cast<std::size_t>(k - n_)]));
    }
    return scale;
  };
  std::vector<char> interior(T, 0);
  const auto measures = [&](const std::vector<double>& y, const std::vector<double>& d,
                            double* worst_interior, double* worst_sign) {
    *worst_interior = 0.0;
    *worst_sign = 0.0;
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (fixed_[u]) continue;
      const double scale = column_scale(y, k);
      const double dk = d[u];
      if (interior[u]) *worst_interior = std::max(*worst_interior, std::fabs(dk) / scale);
      if (dk > 0.0 && !has_lower_[u]) *worst_sign = std::max(*worst_sign, dk / scale);
      if (dk < 0.0 && !has_upper_[u]) *worst_sign = std::max(*worst_sign, -dk / scale);
    }
  };

  // The interior set: every unfixed variable whose slack to each of its bounds is more than
  // a fraction of its own size. A free variable is interior by definition.
  Index count = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    const double tau = kPurifyInteriorFraction * std::max(1.0, std::fabs(x_[u]));
    bool inside = true;
    if (has_lower_[u] && sl_[u] <= tau) inside = false;
    if (has_upper_[u] && su_[u] <= tau) inside = false;
    interior[u] = inside ? 1 : 0;
    if (inside) ++count;
  }
  if (count == 0) return false;

  std::vector<double> d(T);
  reduced_costs(y_, &d);
  double interior_before = 0.0, sign_before = 0.0;
  measures(y_, d, &interior_before, &sign_before);

  // Right-hand side A_I d_I, and the matrix A_I A_I^T with a small shift on the rows that
  // have no interior logical (their logical, if interior, contributes exactly 1).
  std::vector<double> restricted(T, 0.0);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (interior[u]) restricted[u] = d[u];
  }
  std::vector<double> rhs(M);
  constraint_times(restricted, &rhs);
  std::vector<double> theta_x(static_cast<std::size_t>(n_), kPurifyShift);
  std::vector<double> row_shift(M, kPurifyShift);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (!interior[u]) continue;
    if (k < n_) {
      theta_x[u] = 1.0;
    } else {
      row_shift[static_cast<std::size_t>(k - n_)] = 1.0;
    }
  }
  // On the dense-column path (#467) the dense columns stay out of the assembled matrix,
  // whose pattern the analysis fixed, and come back through the correction.
  std::vector<double> theta_sparse;
  std::vector<double> assembly_shift;
  if (dense_.active()) {
    dense_.sparse_theta(theta_x, &theta_sparse);
    (void)dense_.preconditioner_shift(model_.matrix, theta_x, row_shift, dual_regularization_,
                                      &assembly_shift);
  }
  if (!normal_equations_lower(model_.matrix, dense_.active() ? theta_sparse : theta_x,
                              dense_.active() ? assembly_shift : row_shift,
                              dual_regularization_, &normal_lower_, should_stop_)) {
    return false;
  }
  SparseLdl& ldl = proximal_ != nullptr ? purify_ldl_ : ldl_;
  if (proximal_ != nullptr && !purify_analyzed_) {
    ldl.set_factor_budget(max_factor_nonzeros_);
    if (!ldl.analyze(normal_lower_, should_stop_)) return false;
    purify_analyzed_ = true;
  }
  if (!ldl.factorize(normal_lower_, dual_regularization_, should_stop_)) return false;
  ++factorizations_;
  if (dense_.active() &&
      !dense_.prepare(ldl_, model_.matrix, theta_x, row_shift, dual_regularization_)) {
    return false;
  }

  std::vector<double> dy = rhs;
  if (dense_.active()) {
    (void)dense_.solve(dy.data());
  } else {
    ldl.solve(dy.data());
  }
  if (!std::all_of(dy.begin(), dy.end(), [](double v) { return std::isfinite(v); }))
    return false;

  std::vector<double> candidate(y_);
  for (Index i = 0; i < m_; ++i)
    candidate[static_cast<std::size_t>(i)] += dy[static_cast<std::size_t>(i)];
  std::vector<double> d_after(T);
  reduced_costs(candidate, &d_after);
  double interior_after = 0.0, sign_after = 0.0;
  measures(candidate, d_after, &interior_after, &sign_after);
  const bool helps = interior_after < interior_before;
  const bool safe = sign_after <= std::max(sign_before, kIpmTolerance);
  if (!helps || !safe) {
    logger_.verbose(
        "interior point: dual purification over {} interior columns would take their worst "
        "relative reduced cost from {:.3e} to {:.3e} and the worst sign violation from "
        "{:.3e} to {:.3e}; not applied",
        count, interior_before, interior_after, sign_before, sign_after);
    return false;
  }

  y_ = candidate;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    const double dk = d_after[u];
    if (has_lower_[u] && has_upper_[u]) {
      zl_[u] = dk >= 0.0 ? dk : 0.0;
      zu_[u] = dk >= 0.0 ? 0.0 : -dk;
    } else if (has_lower_[u]) {
      zl_[u] = dk;
      zu_[u] = 0.0;
    } else if (has_upper_[u]) {
      zl_[u] = 0.0;
      zu_[u] = -dk;
    } else {
      zl_[u] = 0.0;
      zu_[u] = 0.0;
    }
  }
  logger_.verbose(
      "interior point: dual purification over {} interior columns took their worst relative "
      "reduced cost from {:.3e} to {:.3e}; worst sign violation {:.3e} -> {:.3e}",
      count, interior_before, interior_after, sign_before, sign_after);
  return true;
}

Solution InteriorPoint::run() {
  Timer own_clock;
  const Timer& timer = clock_ != nullptr ? *clock_ : own_clock;
  // One interpretation of every limit, shared with every other engine (#289). This used to
  // read `time_limit > 0.0`, which made a budget of zero seconds mean NO limit here while it
  // meant "stop at once" in the simplex and in PDHG: measured on bandm.mps, time_limit=0
  // returned optimal from this engine and time_limit from the other three.
  limits_ = ResourceLimits(options_, logger_);
  const double time_limit = limits_.time_limit();
  // Handed to the linear algebra so the clock is not only consulted between iterations.
  // Captured by reference to the local timer, which outlives every call that uses it.
  if (limits_.has_time_limit()) {
    if (control_ != nullptr) {
      // Observe both the wall-clock deadline and any external SolveControl interruption.
      should_stop_ = [&timer, this, control = control_] {
        return limits_.time_exhausted(timer.elapsed_seconds()) ||
               control->interruption_requested();
      };
    } else {
      should_stop_ = [&timer, this] { return limits_.time_exhausted(timer.elapsed_seconds()); };
    }
  } else if (control_ != nullptr) {
    // No time limit, but interruption is still supported.
    should_stop_ = [control = control_] { return control->interruption_requested(); };
  }
  max_factor_nonzeros_ = options_.get_int(warm_ != nullptr ? "polish_max_factor_nonzeros"
                                                           : "ipm_max_factor_nonzeros");
  // 0 means "sized from this machine" (#576): the constants these options shipped with
  // were the 7.7 GB laptop's, and on a 96 GB node they abandoned orderings with 86 GB
  // free. The polish keeps its own explicit cap.
  if (warm_ == nullptr && max_factor_nonzeros_ == 0) {
    max_factor_nonzeros_ =
        static_cast<std::int64_t>(auto_factor_budget(physical_memory_bytes()));
  }
  // The ordering's own budget (#246): the one phase that can run the machine out of memory
  // before it can say how large the factor would be.
  const std::int64_t ordering_entries = options_.get_int("ipm_max_ordering_entries");
  ldl_.set_ordering_budget(ordering_entries < 0 ? static_cast<std::size_t>(-1)
                           : ordering_entries == 0
                               ? auto_ordering_budget(physical_memory_bytes())
                               : static_cast<std::size_t>(ordering_entries));
  ldl_.set_supernodal(options_.get_bool("ipm_supernodal"));
  run_clock_ = &timer;
  ordering_declined_ = false;
  ordering_deadline_ = std::numeric_limits<double>::infinity();
  {
    const double share = options_.get_double("ipm_setup_share");
    if (limits_.has_time_limit() && share < 1.0) {
      ordering_deadline_ = timer.elapsed_seconds() + share * limits_.time_limit();
    }
  }
  build();
  if (options_.get_bool("ipm_proximal_regularization")) {
    proximal_ = std::make_unique<ProximalSystem>(model_.matrix, fixed_);
    purify_ldl_.set_ordering_budget(ldl_.ordering_budget());
    // The supernodal kernel (#470) factors the positive-definite normal equations only; the
    // signed quasi-definite factorization this path runs is the scalar one whatever the
    // option says (la/ldl.cpp), and a user who set both is told so, once.
    if (ldl_.supernodal()) {
      logger_.info(
          "interior point: ipm_supernodal does not apply to the augmented system of "
          "ipm_proximal_regularization, whose signed factorization stays scalar");
    }
  }
  logger_.verbose("interior point: built in {:.2f}s from the start of the solve",
                  timer.elapsed_seconds());
  if (options_.get_bool("ipm_dense_columns")) {
    // A fixed column has Theta 0 and contributes nothing to the normal equations, so it is
    // never dense in the sense that matters here.
    std::vector<char> eligible(static_cast<std::size_t>(n_), 0);
    for (Index j = 0; j < n_; ++j) {
      eligible[static_cast<std::size_t>(j)] = fixed_[static_cast<std::size_t>(j)] ? 0 : 1;
    }
    dense_.set_columns(
        model_.matrix,
        find_dense_columns(model_.matrix, eligible,
                           options_.get_double("ipm_dense_column_factor"), kMaxDenseColumns));
    if (dense_.active()) {
      Index largest = 0;
      for (const Index j : dense_.columns()) {
        largest = std::max(largest, model_.matrix.column(j).size);
      }
      logger_.info(
          "Interior point: {} dense column(s) (the largest with {} entries) split off the "
          "normal equations and corrected for by Sherman-Morrison-Woodbury (#467)",
          dense_.columns().size(), largest);
    }
  }
  // A polish arrives with most of its budget spent by the first-order phase; a build that
  // already used the rest must not go on to assemble and order for nothing (#232).
  if (should_stop_ && should_stop_()) {
    const bool interrupted = control_ != nullptr && control_->interruption_requested();
    return finish(
        interrupted ? SolveStatus::kInterrupted : SolveStatus::kTimeLimit,
        interrupted
            ? "interrupted before the first iteration"
            : fmt::format("time limit {:g}s reached before the first iteration", time_limit),
        0, timer.elapsed_seconds());
  }
  logger_.info("Interior point: {} rows, {} columns, {} nonzeros", m_, n_,
               model_.num_nonzeros());
  logger_.begin_iteration_table();

  Count iterations = 0;
  double previous_mu = std::numeric_limits<double>::infinity();
  int stalled = 0;

  for (;; ++iterations) {
    residuals();
    logger_.iteration(iterations,
                      model_.sense_multiplier() * objective_ + model_.objective_offset,
                      primal_infeasibility_, dual_infeasibility_, timer.elapsed_seconds());
    const double relative_gap =
        mu_ * static_cast<double>(bound_count_) / (1.0 + std::fabs(objective_));

    // AN ITERATE THAT IS NOT A NUMBER ENDS THE SOLVE, NOW - and before it can be remembered,
    // compared or logged as anything else. On a 20,000-row staircase model this method
    // converged normally to a relative gap of 1.0e-07 at iteration 26, the next
    // factorization broke down as the barrier vanished (5,090 regularized pivots in one
    // step), and the iterate came back NaN. The loop then ran 270 MORE iterations on NaN:
    // every guard below is a comparison and every comparison with NaN is false, so the
    // convergence test could not pass and the stall detector could not trip. Worse, the
    // NaN iterate's violation measures all read 0.0 for the same reason, so it was
    // remembered as the best point with merit 0.0, and the real answer was overwritten.
    //
    // The best FINITE iterate is the answer. It is restored, re-measured, and reported as
    // feasible when it meets the feasibility tolerances - never as optimal, which only the
    // test below may say - or as the numerical failure it is when it does not.
    if (!std::isfinite(mu_) || !std::isfinite(primal_infeasibility_) ||
        !std::isfinite(dual_infeasibility_) || !std::isfinite(objective_)) {
      restore_best();
      residuals();
      const double best_gap =
          mu_ * static_cast<double>(bound_count_) / (1.0 + std::fabs(objective_));
      const bool usable = best_.merit < std::numeric_limits<double>::infinity() &&
                          std::isfinite(objective_) && primal_infeasibility_ <= 1e-6 &&
                          dual_infeasibility_ <= 1e-6;
      return finish(
          usable ? SolveStatus::kFeasible : SolveStatus::kNumericalError,
          fmt::format("the iterate stopped being finite at iteration {}; {}", iterations,
                      usable ? fmt::format("the best earlier iterate is reported as "
                                           "a feasible point (relative gap {:.1e}, "
                                           "infeasibility {:.1e} / {:.1e})",
                                           best_gap, primal_infeasibility_, dual_infeasibility_)
                             : fmt::format("the best earlier iterate does not meet "
                                           "the feasibility tolerances (relative "
                                           "gap {:.1e}, infeasibility {:.1e} / "
                                           "{:.1e}, merit {:.1e})",
                                           best_gap, primal_infeasibility_, dual_infeasibility_,
                                           best_.merit)),
          iterations, timer.elapsed_seconds());
    }

    remember_if_best(
        std::max({primal_infeasibility_, dual_infeasibility_, relative_gap, max_product_}));
    logger_.verbose(
        "ipm iteration {}: mu {:.2e}, relative gap {:.2e}, worst relative product {:.2e}, "
        "primal {:.2e} scaled / {:.2e} model, dual {:.2e} scaled / {:.2e} model, "
        "regularized pivots so far {}",
        iterations, mu_, relative_gap, max_product_, primal_infeasibility_,
        model_primal_infeasibility_, dual_infeasibility_, model_dual_infeasibility_,
        regularized_pivots_);
    // BOTH SPACES (#582): the scaled residual is what the loop drives down; the model-space
    // one is what the answer is judged by, and on a badly scaled model they differ by the
    // Ruiz factors of the worst rows. Optimal is claimed only when the point would pass the
    // guard's own measurement, primal and dual.
    const bool scaled_converged =
        primal_infeasibility_ <= kIpmTolerance && dual_infeasibility_ <= kIpmTolerance &&
        relative_gap <= kIpmGap && max_product_ <= kIpmComplementarity;
    if (scaled_converged && model_space_holds_as_reported(/*throttled=*/true)) {
      return finish(SolveStatus::kOptimal,
                    barrier_retry_used_
                        ? fmt::format("converged at a relative gap of {:.1e} one step after "
                                      "the barrier vanished (#392)",
                                      relative_gap)
                        : std::string{},
                    iterations, timer.elapsed_seconds());
    }
    // CONVERGED IN THE LOOP'S UNITS, NOT IN THE MODEL'S (#582). Every scaled measure holds
    // and the model-space ones do not, even after purification. The loop goes on while they
    // improve; when kModelSpaceStallIterations pass without kModelSpaceProgress, the point is
    // at the floor the regularization leaves, and it is reported as the feasible point it is
    // with both measurements in the message, for the status guard to judge. Before this the
    // loop ran on to the iteration limit (d2q06c: 300 iterations at a relative gap of 1.7e-9).
    if (scaled_converged) {
      const double excess = model_space_excess();
      if (excess < kModelSpaceProgress * best_model_excess_) {
        best_model_excess_ = excess;
        model_stalled_ = 0;
      } else if (++model_stalled_ >= kModelSpaceStallIterations) {
        return finish(
            SolveStatus::kFeasible,
            fmt::format("converged in the scaled space (relative gap {:.1e}, infeasibility "
                        "{:.1e} / {:.1e}) but the point measures {:.1e} / {:.1e} in the "
                        "model's units, complementarity {:.1e}, and {} more iterations did "
                        "not reduce that: a feasible point, not a proof (#582)",
                        relative_gap, primal_infeasibility_, dual_infeasibility_,
                        model_primal_infeasibility_, model_dual_infeasibility_,
                        model_complementarity_, kModelSpaceStallIterations),
            iterations, timer.elapsed_seconds());
      }
    }
    // The count is cumulative over the iterations where the scaled measures hold, not
    // consecutive: on pilot87 those measures flicker in and out, and a consecutive count
    // never reached three, so the loop ran to the time limit (review of #611, measured).
    if (iterations >= kMaxIterations || limits_.iterations_exhausted(iterations)) {
      restore_best();
      return finish(
          SolveStatus::kIterationLimit,
          fmt::format("iteration limit reached after {} iterations; relative "
                      "infeasibility {:.1e} / {:.1e}, gap {:.1e}",
                      iterations, primal_infeasibility_, dual_infeasibility_, relative_gap),
          iterations, timer.elapsed_seconds());
    }
    if (should_stop_ && should_stop_()) {
      restore_best();
      const bool interrupted = control_ != nullptr && control_->interruption_requested();
      return finish(
          interrupted ? SolveStatus::kInterrupted : SolveStatus::kTimeLimit,
          interrupted ? "interrupted" : fmt::format("time limit {:g}s reached", time_limit),
          iterations, timer.elapsed_seconds());
    }
    if (!factorize()) {
      if (factor_too_large_ && predicted_normal_nonzeros_ >= 0) {
        return finish(
            warm_ != nullptr ? SolveStatus::kNotSolved : SolveStatus::kNumericalError,
            fmt::format(
                "declined before assembly: the normal equations alone would hold at least {} "
                "lower-triangle nonzeros, above {} = {}, and their factor at least as many "
                "(#467); {}raise the option or use another engine",
                predicted_normal_nonzeros_,
                warm_ != nullptr ? "polish_max_factor_nonzeros" : "ipm_max_factor_nonzeros",
                max_factor_nonzeros_,
                dense_.active() || options_.get_bool("ipm_dense_columns")
                    ? ""
                    : "ipm_dense_columns=true splits dense columns off, or "),
            iterations, timer.elapsed_seconds());
      }
      if (dense_schur_failed_) {
        return finish(SolveStatus::kNumericalError,
                      "the Schur complement of the dense-column correction was not "
                      "numerically positive definite (#467)",
                      iterations, timer.elapsed_seconds());
      }
      if (factor_too_large_) {
        // A declined polish leaves the first-order answer standing, so it is not a failure;
        // a declined plain solve has nothing to fall back on and says so as one.
        return finish(
            warm_ != nullptr ? SolveStatus::kNotSolved : SolveStatus::kNumericalError,
            fmt::format(
                "declined: the factor of the {} would hold {} nonzeros, "
                "above {} = {}; raise the option or use another engine",
                system_name(),
                ldl_.factor_too_large()
                    ? fmt::format("more than {}", max_factor_nonzeros_)
                    : fmt::format("{}", ldl_.factor_nonzeros() + ldl_.dimension()),
                warm_ != nullptr ? "polish_max_factor_nonzeros" : "ipm_max_factor_nonzeros",
                max_factor_nonzeros_),
            iterations, timer.elapsed_seconds());
      }
      if (ordering_declined_) {
        // Declined, not failed and not out of time: the ordering did not finish within
        // its share, and the rest of the budget is handed back for another engine (#357).
        return finish(
            SolveStatus::kNotSolved,
            fmt::format("the interior point declined: the ordering and first factorization "
                        "of the {} ({} factor nonzeros) did not finish within "
                        "ipm_setup_share = {:g} of the {:g}s time limit ({:.1f}s), so the "
                        "factorization is not affordable here",
                        system_name(), ldl_.factor_nonzeros() + ldl_.dimension(),
                        options_.get_double("ipm_setup_share"), limits_.time_limit(),
                        timer.elapsed_seconds()),
            iterations, timer.elapsed_seconds());
      }
      if (ldl_.ordering_too_large()) {
        return finish(
            SolveStatus::kNumericalError,
            fmt::format("the ordering of the {} was abandoned: its quotient "
                        "graph passed ipm_max_ordering_entries = {} list entries, so the "
                        "fill-in is beyond what this machine can hold (#246); raise the "
                        "option or use another engine",
                        system_name(), ldl_.ordering_budget()),
            iterations, timer.elapsed_seconds());
      }
      // Told to stop rather than unable to: the difference matters to a reader, and to the
      // status guard. Neither is a point, but only one of them is a failure.
      if (ldl_.stopped_early() || assembly_stopped_) {
        restore_best();
        return finish(
            SolveStatus::kTimeLimit,
            fmt::format("time limit {:g}s reached inside the {}, which was abandoned",
                        time_limit,
                        assembly_stopped_ ? fmt::format("assembly of the {}", system_name())
                                          : std::string("factorization")),
            iterations, timer.elapsed_seconds());
      }
      if (ldl_.pattern_too_large()) {
        return finish(SolveStatus::kNumericalError,
                      fmt::format("the factor of the {} would hold more than "
                                  "{} nonzeros, which this build cannot address (#305)",
                                  system_name(), kMaxNonzeros),
                      iterations, timer.elapsed_seconds());
      }
      return finish(SolveStatus::kNumericalError,
                    fmt::format("the {} could not be factorized", system_name()), iterations,
                    timer.elapsed_seconds());
    }

    // THE BARRIER'S LAST WORD (#209, and the constants above). On the 20,000-row staircase
    // model the factorization at a relative gap of 4e-8 had to regularize 2,199 of 18,227
    // pivots where the one before regularized 6, and the direction it produced was NaN. A
    // spike like that at an iterate within a decade of every tolerance is the matrix saying
    // the barrier is gone: the columns pinned to their bounds have left the normal equations
    // and what remains is rank deficient at working precision. Capping z/s was tried and
    // changes nothing - the capped factorization regularizes 2,194 and its step is NaN too -
    // because the deficiency is in the rank of the active columns, not in their weights. So
    // the iterate measured at the top of this loop is the answer, reported as converged and
    // measured by the status guard in solve() against the project's tolerances like every
    // other optimal claim; on that model the guard finds its dual side 12% over the 1e-7
    // tolerance and reports it feasible at a relative error of 1.4e-8, which is what it is.
    {
      const Count regularized_now = ldl_.regularized_pivots();
      const auto spike_threshold = std::max<Count>(
          kBarrierExhaustedPivots,
          static_cast<Count>(kBarrierExhaustedFraction * static_cast<double>(m_)));
      const bool nearly_converged =
          primal_infeasibility_ <= kBarrierExhaustedSlack * kIpmTolerance &&
          dual_infeasibility_ <= kBarrierExhaustedSlack * kIpmTolerance &&
          relative_gap <= kBarrierExhaustedSlack * kIpmGap &&
          max_product_ <= kBarrierExhaustedSlack * kIpmComplementarity;
      if (regularized_now >= spike_threshold && nearly_converged) {
        // ONE MORE STEP ON A STRONGER DIAGONAL BEFORE THE STOP (#392). The iterate here is
        // within a decade of every tolerance, but on the 20,000-row staircase its worst
        // complementarity product was 1.9e-6 against the 1e-6 the independent verifier
        // accepts, and the stop above handed it back as it stood: the status guard then
        // reported a feasible point rather than a proof. The factorization that spiked is
        // rank deficient at working precision, but #389's recovery showed that a diagonal
        // raised by kRegularizationRaise makes the factor well defined at the price of a
        // slightly inexact direction - and one such step is exactly what a nearly converged
        // iterate needs to bring its worst products down to the mean. It is taken once: the
        // regularization is raised, the normal equations refactorized, and the loop falls
        // through to the predictor-corrector below. The next visit to this block, whatever
        // the step did, restores the best iterate seen and stops; the merit that chooses it
        // includes the worst relative product, so a step that made things worse costs one
        // iteration and nothing else.
        bool one_more_step = false;
        if (!barrier_retry_used_ && regularization_raises_ < kMaxRegularizationRaises) {
          barrier_retry_used_ = true;
          dual_regularization_ *= kRegularizationRaise;
          ++regularization_raises_;
          logger_.verbose(
              "interior point: the barrier vanished at iteration {} ({} of {} pivots "
              "regularized); regularization raised to {:.1e} for one more step",
              iterations, regularized_now, m_, dual_regularization_);
          one_more_step = factorize();
        }
        if (!one_more_step) {
          restore_best();
          residuals();
          const double best_gap =
              mu_ * static_cast<double>(bound_count_) / (1.0 + std::fabs(objective_));
          // THE STATUS FOLLOWS THE MEASUREMENT OF THE RESTORED POINT (#576). The test that
          // brought the loop here allows kBarrierExhaustedSlack on every tolerance, so the
          // iterate handed back can sit a decade above the feasibility tolerance while its
          // gap is at rounding. Claiming optimal on the gap alone sent irish-electricity to
          // the status guard as "converged at a relative gap of 8.6e-15" and came back
          // numerical_error, because the guard measured what the claim did not. Optimal
          // only when every tolerance holds on the restored point; otherwise the point is
          // reported as what it is, feasible to the slack, with the numbers in the message.
          const bool optimal_here =
              primal_infeasibility_ <= kIpmTolerance && dual_infeasibility_ <= kIpmTolerance &&
              best_gap <= kIpmGap && max_product_ <= kIpmComplementarity &&
              model_space_holds_as_reported();
          return finish(
              optimal_here ? SolveStatus::kOptimal : SolveStatus::kFeasible,
              optimal_here
                  ? fmt::format("converged at a relative gap of {:.1e} when the barrier "
                                "vanished: {} of {} pivots regularized in one factorization",
                                best_gap, regularized_now, m_)
                  : fmt::format("the barrier vanished ({} of {} pivots regularized in one "
                                "factorization) with the best iterate at relative gap {:.1e} "
                                "but infeasibility {:.1e} / {:.1e} scaled, {:.1e} / {:.1e} in "
                                "model space, and worst product {:.1e}, above the {:.0e} "
                                "tolerance: a feasible point to that slack, not a proof (#576, "
                                "#582)",
                                regularized_now, m_, best_gap, primal_infeasibility_,
                                dual_infeasibility_, model_primal_infeasibility_,
                                model_dual_infeasibility_, max_product_, kIpmTolerance),
              iterations, timer.elapsed_seconds());
        }
      }
    }

    // PREDICTOR then CORRECTOR, as one step: the corrector's right-hand side carries the
    // predictor's products ds dz, so a predictor that is not finite poisons the corrector
    // whatever the factorization behind it, and a recovery has to redo both (#209).
    const auto predictor_corrector = [&]() {
      // PREDICTOR: the affine-scaling direction (mu-terms = -s z).
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        r_mu_l_[u] = has_lower_[u] ? -sl_[u] * zl_[u] : 0.0;
        r_mu_u_[u] = has_upper_[u] ? -su_[u] * zu_[u] : 0.0;
      }
      newton_direction();
      // Tests only (ipm_testing.hpp): poison this direction to reach the #209 recovery.
      if (testing::take_poisoned_direction() && !dx_.empty()) {
        dx_[0] = std::numeric_limits<double>::quiet_NaN();
      }
      if (!direction_is_finite()) return false;
      const double alpha_p_aff = step_length(sl_, dsl_, su_, dsu_);
      const double alpha_d_aff = step_length(zl_, dzl_, zu_, dzu_);
      double mu_aff = 0.0;
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        if (has_lower_[u])
          mu_aff += (sl_[u] + alpha_p_aff * dsl_[u]) * (zl_[u] + alpha_d_aff * dzl_[u]);
        if (has_upper_[u])
          mu_aff += (su_[u] + alpha_p_aff * dsu_[u]) * (zu_[u] + alpha_d_aff * dzu_[u]);
      }
      mu_aff = bound_count_ > 0 ? mu_aff / static_cast<double>(bound_count_) : 0.0;
      const double ratio = mu_ > 0.0 ? mu_aff / mu_ : 0.0;
      const double sigma = std::min(1.0, ratio * ratio * ratio);

      // CORRECTOR: centering plus the second-order term from the predictor.
      for (Index k = 0; k < total_; ++k) {
        const auto u = static_cast<std::size_t>(k);
        if (has_lower_[u]) r_mu_l_[u] = sigma * mu_ - sl_[u] * zl_[u] - dsl_[u] * dzl_[u];
        if (has_upper_[u]) r_mu_u_[u] = sigma * mu_ - su_[u] * zu_[u] - dsu_[u] * dzu_[u];
      }
      newton_direction();
      return direction_is_finite();
    };
    bool step_is_finite = predictor_corrector();
    // A NON-FINITE DIRECTION IS CAUGHT BEFORE IT IS TAKEN (#209). On the 5,000-row
    // staircase model the iterate at a relative gap of 2e-8 is finite and measured, and the
    // factorization behind the NEXT step has pivots just above the regularization floor
    // whose reciprocals overflow the solve: the direction is NaN without a single pivot
    // having been regularized, so the spike test above cannot see it. Rather than take the
    // step (and then throw the iterate away at the top of the next loop), the
    // regularization is raised by 1e4, the normal equations are refactorized and the
    // predictor and corrector recomputed. A stronger diagonal makes the factor well defined
    // at the price of a slightly inexact Newton step, and the residuals the next iteration
    // measures absorb that; up to kMaxRegularizationRaises raises, after which the current
    // iterate is the answer, judged as the barrier-exhausted stop judges one.
    if (!step_is_finite) {
      while (!step_is_finite && regularization_raises_ < kMaxRegularizationRaises) {
        dual_regularization_ *= kRegularizationRaise;
        ++regularization_raises_;
        logger_.verbose(
            "interior point: non-finite direction at iteration {}; "
            "regularization raised to {:.1e} and the step recomputed",
            iterations,
            proximal_ != nullptr ? ProximalSystem::recovery_floor(regularization_raises_)
                                 : dual_regularization_);
        if (!factorize()) break;
        step_is_finite = predictor_corrector();
      }
      if (!step_is_finite) {
        const bool nearly_converged =
            primal_infeasibility_ <= kBarrierExhaustedSlack * kIpmTolerance &&
            dual_infeasibility_ <= kBarrierExhaustedSlack * kIpmTolerance &&
            relative_gap <= kBarrierExhaustedSlack * kIpmGap &&
            max_product_ <= kBarrierExhaustedSlack * kIpmComplementarity;
        restore_best();
        residuals();
        const bool usable = std::isfinite(objective_) && primal_infeasibility_ <= 1e-6 &&
                            dual_infeasibility_ <= 1e-6;
        // Converged only if the point handed back also holds in the model's units (#582):
        // "within a decade" of the scaled tolerances is not a claim about the model.
        const bool converged_here = nearly_converged && model_space_holds_as_reported();
        return finish(
            converged_here ? SolveStatus::kOptimal
                           : (usable ? SolveStatus::kFeasible : SolveStatus::kNumericalError),
            fmt::format(
                "the Newton direction was not finite at iteration {} after {} "
                "regularization raise(s); {}",
                iterations, regularization_raises_,
                converged_here
                    ? std::string("the iterate before it is within a decade of every "
                                  "tolerance and is reported as converged")
                : nearly_converged
                    ? fmt::format("the iterate before it is within a decade of every "
                                  "scaled tolerance but measures {:.1e} / {:.1e} in the "
                                  "model's units, so it is reported as it stands (#582)",
                                  model_primal_infeasibility_, model_dual_infeasibility_)
                    : std::string("the best iterate is reported as it stands")),
            iterations, timer.elapsed_seconds());
      }
    }
    const double alpha_p = std::min(1.0, kStepToBoundary * step_length(sl_, dsl_, su_, dsu_));
    const double alpha_d = std::min(1.0, kStepToBoundary * step_length(zl_, dzl_, zu_, dzu_));

    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      x_[u] += alpha_p * dx_[u];
      if (has_lower_[u]) {
        sl_[u] += alpha_p * dsl_[u];
        zl_[u] += alpha_d * dzl_[u];
      }
      if (has_upper_[u]) {
        su_[u] += alpha_p * dsu_[u];
        zu_[u] += alpha_d * dzu_[u];
      }
    }
    for (Index i = 0; i < m_; ++i)
      y_[static_cast<std::size_t>(i)] += alpha_d * dy_[static_cast<std::size_t>(i)];

    // A method that stops moving is not converging; say so rather than spin to the limit.
    if (mu_ >= 0.999 * previous_mu && alpha_p < 1e-6 && alpha_d < 1e-6) {
      if (++stalled >= 5) {
        // The best iterate is returned as a FEASIBLE point when it met the feasibility
        // tolerances, for the status guard to judge; otherwise as the numerical failure it
        // is. Never as optimal: the convergence test above is the only thing that says so.
        restore_best();
        residuals();
        const bool usable = best_.merit < std::numeric_limits<double>::infinity() &&
                            primal_infeasibility_ <= 1e-6 && dual_infeasibility_ <= 1e-6;
        return finish(usable ? SolveStatus::kFeasible : SolveStatus::kNumericalError,
                      fmt::format("the interior-point iteration stalled after {} iterations "
                                  "(steps {:.1e} / {:.1e}); {}",
                                  iterations, alpha_p, alpha_d,
                                  usable ? "the best iterate is reported as a feasible point"
                                         : "the model may be infeasible or unbounded, which "
                                           "this method does not certify"),
                      iterations, timer.elapsed_seconds());
      }
    } else {
      stalled = 0;
    }
    previous_mu = mu_;
  }
}

/// Scaled solve: Ruiz + Pock-Chambolle equilibration, exactly as the simplex entry point
/// applies it, then the point, the row duals and the reduced costs are mapped back:
/// x = Dc xhat, y = Dr yhat, d = Dc^-1 dhat (la/scaling.hpp).
Solution solve_scaled(const Model& model, const Options& options, Logger& logger,
                      SolveControl* control, const WarmStart* warm) {
  // The clock the time limit runs on starts HERE. Scaling a 500,000-row model and copying
  // it are tens of seconds, and a polish's budget of 30 used to begin only after them.
  Timer clock;
  if (!options.get_bool("scaling")) {
    InteriorPoint engine(model, options, logger, control, warm, &clock);
    return engine.run();
  }
  const Scaling scaling = build_scaling(model, model.col_cost, kRuizIterations);
  logger.verbose("interior point: scaled in {:.2f}s", clock.elapsed_seconds());
  Model scaled = model;
  scaled.matrix = scaling.matrix;
  scaled.col_cost = scaling.cost;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    scaled.col_lower[u] =
        is_finite_bound(model.col_lower[u]) ? model.col_lower[u] / dc : model.col_lower[u];
    scaled.col_upper[u] =
        is_finite_bound(model.col_upper[u]) ? model.col_upper[u] / dc : model.col_upper[u];
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double dr = scaling.row[u];
    scaled.row_lower[u] =
        is_finite_bound(model.row_lower[u]) ? model.row_lower[u] * dr : model.row_lower[u];
    scaled.row_upper[u] =
        is_finite_bound(model.row_upper[u]) ? model.row_upper[u] * dr : model.row_upper[u];
  }
  // The warm start lives in the original units and is mapped the way the answer is mapped
  // back below, inverted: xhat = x / Dc, yhat = y / Dr, dhat = d * Dc.
  WarmStart scaled_warm;
  if (warm != nullptr && static_cast<Index>(warm->col_value.size()) == n &&
      static_cast<Index>(warm->row_dual.size()) == m &&
      static_cast<Index>(warm->col_dual.size()) == n) {
    scaled_warm = *warm;
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      scaled_warm.col_value[u] /= scaling.column[u];
      scaled_warm.col_dual[u] *= scaling.column[u];
    }
    for (Index i = 0; i < m; ++i) {
      const auto u = static_cast<std::size_t>(i);
      scaled_warm.row_dual[u] /= scaling.row[u];
    }
    warm = &scaled_warm;
  }
  logger.verbose("interior point: scaled model and warm start ready at {:.2f}s",
                 clock.elapsed_seconds());
  InteriorPoint engine(scaled, options, logger, control, warm, &clock);
  engine.set_scaling(&scaling);
  Solution solution = engine.run();
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    if (u < solution.col_value.size()) solution.col_value[u] *= dc;
    if (u < solution.col_dual.size()) solution.col_dual[u] /= dc;
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (u < solution.row_dual.size()) solution.row_dual[u] *= scaling.row[u];
  }
  if (solution.status == SolveStatus::kOptimal) {
    solution.dual_bound = model.evaluate_objective(solution.col_value.data());
  }
  solution.recompute_quality(model);
  return solution;
}

}  // namespace

Solution solve_ipm(const Model& model, const Options& options, Logger& logger,
                   SolveControl* control) {
  return solve_scaled(model, options, logger, control, nullptr);
}

Solution solve_ipm(const Model& model, const Options& options, Logger& logger,
                   const WarmStart* warm) {
  return solve_scaled(model, options, logger, nullptr, warm);
}

Solution solve_ipm(const Model& model, const Options& options, Logger& logger,
                   SolveControl* control, const WarmStart* warm) {
  return solve_scaled(model, options, logger, control, warm);
}

}  // namespace sankhya::ipm
