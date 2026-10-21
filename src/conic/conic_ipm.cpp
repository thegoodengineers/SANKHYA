// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the conic interior point (#755): the homogeneous self-dual embedding of
//
//     min c'x  s.t.  A x = b,  x in K        max b'y  s.t.  A'y + s = c,  s in K
//
// with K a product of a free block, a non-negative block and second-order cones (K is
// self-dual), solved by a Mehrotra predictor-corrector in the Nesterov-Todd scaling.
//
// References
//   Nesterov & Todd, "Self-scaled barriers and interior-point methods for convex programming",
//     Math. Oper. Res. 22 (1997), and "Primal-dual interior-point methods for self-scaled
//     cones", SIAM J. Optim. 8 (1998) - the scaling W with W^{-1} x = W s = lambda.
//   Alizadeh & Goldfarb, "Second-order cone programming", Math. Programming 95 (2003),
//     sections 4-6 - the Jordan algebra of the cone (jordan.hpp), the NT direction and the
//     step to the boundary.
//   Andersen, Roos & Terlaky, "On implementing a primal-dual interior-point method for conic
//     quadratic optimization", Math. Programming 95 (2003) - the homogeneous model for cones,
//     the stopping rules and the certificates read off tau and kappa.
//   Xu, Hung & Ye, "A simplified homogeneous and self-dual linear programming algorithm and
//     its implementation", Annals of OR 62 (1996); Andersen & Andersen, "The MOSEK interior
//     point optimizer for linear programming: an implementation of the homogeneous
//     algorithm" (2000) - the embedding this project already runs for LP (src/ipm/hsd.cpp).
//   Vandenberghe, "The CVXOPT linear and quadratic cone program solvers" (2010), the
//     document only - the reduced Newton system with the scaled complementarity eliminated.
//   Mehrotra, "On the implementation of a primal-dual interior point method", SIAM J. Optim.
//     2 (1992) - predictor-corrector, sigma = (mu_aff / mu)^3, the fraction to the boundary.
//   Altman & Gondzio, "Regularized symmetric indefinite systems in interior point methods for
//     linear and quadratic optimization", Optim. Methods Softw. 11 (1999); Vanderbei,
//     "Symmetric quasidefinite matrices", SIAM J. Optim. 5 (1995) - the regularized
//     augmented system and why the project's signed LDL^T factors it without pivoting.
//
// THE EMBEDDING. Unknowns (x, tau) in K x R_+ and (y, s, kappa) with (s, kappa) in K x R_+:
//
//     A x - b tau = 0,   -A'y + c tau - s = 0,   b'y - c'x - kappa = 0,
//
// which always has a solution: tau > 0 at the limit gives the optimum x/tau, y/tau, s/tau;
// kappa > 0 gives a ray, either (y, s) proving primal infeasibility (A'y + s = 0, b'y > 0)
// or x proving unboundedness (A x = 0, c'x < 0). Residuals r_p = b tau - A x,
// r_d = A'y + s - c tau, r_g = kappa + c'x - b'y, and mu = (x's + tau kappa) / (nu + 1) with
// nu the number of non-negative columns plus the number of cones.
//
// THE NEWTON SYSTEM. In the NT scaling the complementarity is linearized as
// lambda o (W^{-1} dx + W ds) = d_s, so ds = W^{-1} L(lambda)^{-1} d_s - W^{-2} dx and the
// dual equation becomes W^{-2} dx - A'dy + c dtau = r_d + W^{-1} L(lambda)^{-1} d_s =: rd~.
// With tau kappa linearized as kappa dtau + tau dkappa = d_kappa, the step solves
//
//     [ -W^{-2}  A' ] [dx]   [ -rd~ ]         [c]
//     [  A       0  ] [dy] = [  r_p ] + dtau  [b]
//
// twice (once per right-hand side, one factorization), and the gap row fixes dtau:
// dtau = (r_g + d_kappa/tau + c'u1 - b'u2) / (kappa/tau - c'p1 + b'p2). The (1,1) block is
// -W^{-2} - rho I (rho on the free block, where W is absent), the (2,2) block delta I,
// quasi-definite for any rho, delta > 0 and factored by factorize_quasidefinite; the
// regularization is taken back out by refinement against the unregularized matrix, exactly
// as the convex QP interior point does (qp_ipm_system.cpp). A cone's W^{-2} = P(w^{-1}) is a
// dense block, written out in full.
// ponytail: dense cone blocks in the KKT matrix, O(k^2) per cone of size k; the product-form
// factorization of Andersen, Roos & Terlaky (2003, section 6) is the upgrade for wide cones.

#include "conic/conic_ipm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "conic/jordan.hpp"
#include "la/ldl.hpp"
#include "qp/qp_ipm_system.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::conic {
namespace {

double inf_norm(const std::vector<double>& v) {
  double n = 0.0;
  for (const double e : v) n = std::max(n, std::fabs(e));
  return n;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

/// The KKT matrix's fixed lower-triangular pattern, filled with values per iteration.
class Kkt {
 public:
  explicit Kkt(const ConicForm& f) : f_(f) {
    const Index n = f.n;
    const Index dim = n + f.m;
    block_.assign(static_cast<std::size_t>(n), -1);
    for (std::size_t c = 0; c < f.cones.size(); ++c) {
      for (Index i = 0; i < f.cones[c].size; ++i) {
        block_[static_cast<std::size_t>(f.cones[c].start + i)] = static_cast<Index>(c);
      }
    }
    starts_.assign(static_cast<std::size_t>(dim) + 1, 0);
    for (Index j = 0; j < n; ++j) {
      rows_.push_back(j);
      const Index block = block_of(j);
      if (block >= 0) {
        const ConeBlock& c = f.cones[static_cast<std::size_t>(block)];
        for (Index i = j + 1; i < c.start + c.size; ++i) rows_.push_back(i);
      }
      const ColumnView col = f.a.column(j);
      for (Index p = 0; p < col.size; ++p) rows_.push_back(n + col.rows[p]);
      starts_[static_cast<std::size_t>(j) + 1] = static_cast<Index>(rows_.size());
    }
    for (Index i = 0; i < f.m; ++i) {
      rows_.push_back(n + i);
      starts_[static_cast<std::size_t>(n + i) + 1] = static_cast<Index>(rows_.size());
    }
    signs_.assign(static_cast<std::size_t>(dim), 1);
    for (Index j = 0; j < n; ++j) signs_[static_cast<std::size_t>(j)] = -1;
  }

  [[nodiscard]] Index block_of(Index j) const { return block_[static_cast<std::size_t>(j)]; }

  /// theta[j] = s_j / x_j on the non-negative block; v_k = w_k^{-1} per cone.
  SparseMatrix build(const std::vector<double>& theta,
                     const std::vector<std::vector<double>>& v, double rho,
                     double delta) const {
    const Index n = f_.n;
    std::vector<double> values(rows_.size(), 0.0);
    for (Index j = 0; j < n; ++j) {
      std::size_t p = static_cast<std::size_t>(starts_[static_cast<std::size_t>(j)]);
      const Index block = block_of(j);
      if (block < 0) {
        const bool nonneg = j >= f_.free_n;
        values[p++] = -(nonneg ? theta[static_cast<std::size_t>(j)] : 0.0) - rho;
      } else {
        const ConeBlock& c = f_.cones[static_cast<std::size_t>(block)];
        const std::vector<double>& vk = v[static_cast<std::size_t>(block)];
        const double d = jordan::det(vk.data(), c.size);
        const Index i0 = j - c.start;
        // -(P(v))_{ij} = -2 v_i v_j + det(v) J_ij
        for (Index i = i0; i < c.size; ++i) {
          double e = -2.0 * vk[static_cast<std::size_t>(i0)] * vk[static_cast<std::size_t>(i)];
          if (i == i0) e += (i0 == 0 ? d : -d) - rho;
          values[p++] = e;
        }
      }
      const ColumnView col = f_.a.column(j);
      for (Index q = 0; q < col.size; ++q) values[p++] = col.values[q];
    }
    for (Index i = 0; i < f_.m; ++i) {
      values[static_cast<std::size_t>(starts_[static_cast<std::size_t>(n + i)])] = delta;
    }
    SparseMatrix k;
    k.assign_columns(n + f_.m, n + f_.m, starts_, rows_, std::move(values));
    return k;
  }

  [[nodiscard]] const std::vector<signed char>& signs() const noexcept { return signs_; }

 private:
  const ConicForm& f_;
  std::vector<Index> block_;  ///< the cone each position belongs to, -1 outside every cone
  std::vector<Index> starts_;
  std::vector<Index> rows_;
  std::vector<signed char> signs_;
};

}  // namespace

HsdResult solve_hsd(const ConicForm& f, const Options& options, Logger& logger,
                    SolveControl* control, const Timer& timer) {
  using jordan::arrow_solve;
  using jordan::product;
  using jordan::quadratic;

  const ResourceLimits limits(options, logger);
  StopController stop(control, timer, limits);
  const auto n = static_cast<std::size_t>(f.n);
  const auto m = static_cast<std::size_t>(f.m);
  const auto nf = static_cast<std::size_t>(f.free_n);
  const auto nl = static_cast<std::size_t>(f.nonneg_n);
  const std::size_t ncones = f.cones.size();
  const double nu = static_cast<double>(nl + ncones);
  const double b_norm = inf_norm(f.b);
  const double c_norm = inf_norm(f.c);
  const double a_max = f.a.max_abs_value();
  const double tol = tol::kConicTolerance;

  HsdResult out;
  // ---- the starting point: e on every cone, tau = kappa = 1 (Andersen & Andersen 2000) ----
  std::vector<double> x(n, 0.0), s(n, 0.0), y(m, 0.0);
  for (std::size_t j = nf; j < nf + nl; ++j) x[j] = s[j] = 1.0;
  for (const ConeBlock& c : f.cones)
    x[static_cast<std::size_t>(c.start)] = s[static_cast<std::size_t>(c.start)] = 1.0;
  double tau = 1.0;
  double kappa = 1.0;

  Kkt kkt(f);
  SparseLdl ldl;
  bool analyzed = false;
  double rho = tol::kConicRegularization;
  double delta = rho;

  std::vector<double> rp(m), rd(n), rdt(n), dx(n), dy(m), ds(n), dxa(n), dsa(n);
  std::vector<double> theta(n, 0.0);
  std::vector<std::vector<double>> w(ncones), wh(ncones), wmh(ncones), v(ncones), lam(ncones);
  std::vector<double> scratch, tmp, tmp2;
  std::vector<double> rhs(n + m), sol_u(n + m), sol_p(n + m), p1(n), p2(m), u1(n), u2(m);
  const auto should_stop_ldl = [&]() { return limits.time_exhausted(timer.elapsed_seconds()); };

  double mu0 = 0.0;
  Count iterations = 0;
  SolveStatus status = SolveStatus::kIterationLimit;
  std::string message;
  double pres = 0.0, dres = 0.0, gap = 0.0;

  const auto residuals = [&]() {
    // r_p = b tau - A x ; r_d = A'y + s - c tau ; r_g = kappa + c'x - b'y
    for (std::size_t i = 0; i < m; ++i) rp[i] = f.b[i] * tau;
    f.a.multiply_add(x.data(), rp.data(), -1.0);
    for (std::size_t j = 0; j < n; ++j) rd[j] = s[j] - f.c[j] * tau;
    f.a.transpose_multiply_add(y.data(), rd.data(), 1.0);
    const double cx = dot(f.c, x);
    const double by = dot(f.b, y);
    pres = inf_norm(rp) / (tau * (1.0 + b_norm));
    dres = inf_norm(rd) / (tau * (1.0 + c_norm));
    gap = std::fabs(cx - by) / (tau + std::fabs(by));
    return kappa + cx - by;
  };

  for (;;) {
    const double rg = residuals();
    double mu = tau * kappa;
    for (std::size_t j = nf; j < n; ++j) mu += x[j] * s[j];
    mu /= nu + 1.0;
    if (iterations == 0) mu0 = mu;
    logger.verbose(
        "conic ipm {:4d}  pres {:.2e}  dres {:.2e}  gap {:.2e}  mu {:.2e}  tau {:.2e}  kappa "
        "{:.2e}",
        iterations, pres, dres, gap, mu, tau, kappa);

    if (pres <= tol && dres <= tol && gap <= tol) {
      status = SolveStatus::kOptimal;
      break;
    }
    // ---- certificates read off a ray (Andersen, Roos & Terlaky 2003, section 5) ----------
    //
    // A RAY IS A PROOF ONLY AT THE DATA'S SCALE (#1061 review). A dual ray (y, s), b'y = 1,
    // with residual r = A'y + s proves infeasibility only if no feasible x has
    // b'y = x'(A'y) = x'(r - s) <= x'r, i.e. only if |r| is small against 1 / |x| for every
    // feasible x - and a feasible model with a large optimum has large x. Min t, t >= |x|,
    // x = 1e9 produced a dual ray with |r| = 6e-9 at iteration 4, small against the ray's own
    // terms and still no proof: x = 1e9 gives x'r of order 6. So the residual is weighed by
    // max(1, |b|), the size the data lets a feasible point have, and the unboundedness ray's
    // by max(1, |c|) the same way. kappa > tau stays: it is necessary, not sufficient.
    {
      const double by = dot(f.b, y);
      if (by > 0.0) {
        // A'y + s = 0 on the normalized ray, relative to its own terms and the data's scale.
        std::vector<double> r(n);
        for (std::size_t j = 0; j < n; ++j) r[j] = s[j] / by;
        f.a.transpose_multiply_add(y.data(), r.data(), 1.0 / by);
        const double scale =
            (1.0 + a_max * inf_norm(y) / by + inf_norm(s) / by) / std::max(1.0, b_norm);
        if (inf_norm(r) <= tol::kConicCertificateTolerance * scale && kappa > tau) {
          status = SolveStatus::kInfeasible;
          out.y.assign(m, 0.0);
          out.s.assign(n, 0.0);
          for (std::size_t i = 0; i < m; ++i) out.y[i] = y[i] / by;
          for (std::size_t j = 0; j < n; ++j) out.s[j] = s[j] / by;
          message = fmt::format(
              "primal infeasible: a dual ray with A'y + s = 0 (residual {:.2e}) "
              "and b'y = 1 at iteration {}",
              inf_norm(r) / scale, iterations);
          break;
        }
      }
      const double cx = dot(f.c, x);
      if (cx < 0.0) {
        std::vector<double> r(m, 0.0);
        f.a.multiply_add(x.data(), r.data(), -1.0 / cx);
        const double scale = (1.0 + a_max * inf_norm(x) / -cx) / std::max(1.0, c_norm);
        if (inf_norm(r) <= tol::kConicCertificateTolerance * scale && kappa > tau) {
          status = SolveStatus::kUnbounded;
          out.x.assign(n, 0.0);
          for (std::size_t j = 0; j < n; ++j) out.x[j] = x[j] / -cx;
          message = fmt::format(
              "unbounded: a primal ray with A x = 0 (residual {:.2e}) and "
              "c'x = -1 at iteration {}",
              inf_norm(r) / scale, iterations);
          break;
        }
      }
    }
    if (mu <= tol::kConicMuFloor * mu0) {
      status = SolveStatus::kNumericalError;
      message = fmt::format(
          "the embedding's mu fell to {:.1e} of its start without tau or kappa deciding "
          "(tau {:.2e}, kappa {:.2e}); the model is near the boundary between feasible and "
          "infeasible",
          mu / mu0, tau, kappa);
      break;
    }
    // ---- limits -------------------------------------------------------------------------
    if (limits.time_exhausted(timer.elapsed_seconds())) {
      status = SolveStatus::kTimeLimit;
      out.stopped_by = LimitReason::kTime;
      message = limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0);
      break;
    }
    if (limits.iterations_exhausted(iterations) ||
        (limits.iteration_limit() < 0 && iterations >= tol::kConicIterationCeiling)) {
      status = SolveStatus::kIterationLimit;
      out.stopped_by = LimitReason::kIterations;
      message =
          limits.iteration_limit() >= 0
              ? limits.describe(LimitReason::kIterations, timer.elapsed_seconds(), iterations,
                                0)
              : fmt::format(
                    "no convergence in {} iterations (pres {:.1e}, dres {:.1e}, gap {:.1e})",
                    iterations, pres, dres, gap);
      break;
    }
    SolveStatus stop_status;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.iterations = iterations;
              p.objective = kInfinity;
              p.best_bound = -kInfinity;
              return p;
            },
            &stop_status)) {
      status = stop_status;
      out.stopped_by =
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      message = limits.describe(out.stopped_by, timer.elapsed_seconds(), iterations, 0);
      break;
    }

    // ---- the Nesterov-Todd scaling ------------------------------------------------------
    for (std::size_t j = nf; j < nf + nl; ++j) theta[j] = s[j] / x[j];
    for (std::size_t k = 0; k < ncones; ++k) {
      const ConeBlock& c = f.cones[k];
      const auto sz = static_cast<std::size_t>(c.size);
      const auto st = static_cast<std::size_t>(c.start);
      w[k].resize(sz);
      wh[k].resize(sz);
      wmh[k].resize(sz);
      v[k].resize(sz);
      lam[k].resize(sz);
      scratch.resize(3 * sz);
      jordan::nt_point(&x[st], &s[st], c.size, scratch.data(), w[k].data());
      jordan::sqrt(w[k].data(), c.size, wh[k].data());
      jordan::inverse(w[k].data(), c.size, v[k].data());
      jordan::sqrt(v[k].data(), c.size, wmh[k].data());
      quadratic(wmh[k].data(), &x[st], c.size, lam[k].data());  // lambda = W^{-1} x
    }

    // ---- factorization ------------------------------------------------------------------
    SparseMatrix k0 = kkt.build(theta, v, 0.0, 0.0);
    SparseMatrix k = kkt.build(theta, v, rho, delta);
    if (!analyzed) {
      if (!ldl.analyze(k, should_stop_ldl)) {
        status = ldl.stopped_early() ? SolveStatus::kTimeLimit : SolveStatus::kNumericalError;
        message = ldl.stopped_early() ? "the time limit ended the symbolic analysis"
                                      : "the KKT matrix could not be analysed";
        break;
      }
      analyzed = true;
    }
    bool factored = false;
    for (int attempt = 0; attempt < tol::kQpIpmRegularizationAttempts; ++attempt) {
      if (!ldl.factorize_quasidefinite(k, kkt.signs(), tol::kQpIpmPivotShare * rho,
                                       should_stop_ldl)) {
        break;
      }
      if (ldl.regularized_pivots() == 0) {
        factored = true;
        break;
      }
      rho *= tol::kQpIpmRegularizationRaise;
      delta = rho;
      k = kkt.build(theta, v, rho, delta);
    }
    if (!factored) {
      if (ldl.stopped_early()) {
        status = SolveStatus::kTimeLimit;
        out.stopped_by = LimitReason::kTime;
        message = limits.describe(LimitReason::kTime, timer.elapsed_seconds(), iterations, 0);
      } else {
        status = SolveStatus::kNumericalError;
        message = fmt::format(
            "the KKT system lost its quasi-definite sign pattern at rho = {:.1e}", rho);
      }
      break;
    }

    // K [p1; p2] = [c; b], once per factorization.
    for (std::size_t j = 0; j < n; ++j) rhs[j] = f.c[j];
    for (std::size_t i = 0; i < m; ++i) rhs[n + i] = f.b[i];
    qp::ipm_detail::solve_refined(ldl, k, rhs, &sol_p);
    qp::ipm_detail::refine_unregularized(ldl, k0, rhs, &sol_p);
    for (std::size_t j = 0; j < n; ++j) p1[j] = sol_p[j];
    for (std::size_t i = 0; i < m; ++i) p2[i] = sol_p[n + i];
    const double cp1 = dot(f.c, p1);
    const double bp2 = dot(f.b, p2);

    // The direction for a complementarity target d_s (per block, in the scaled space) and
    // d_kappa. Writes dx, dy, ds and returns (dtau, dkappa).
    std::vector<double> dsv(n, 0.0);  // d_s laid out over the columns
    double dtau = 0.0;
    double dkappa = 0.0;
    // eta scales the three residuals: 1 in the predictor, 1 - sigma in the corrector, so the
    // residuals fall with mu at one rate (Xu, Hung & Ye 1996; Andersen & Andersen 2000).
    const auto newton = [&](double dk, double eta) {
      // rd~ = eta r_d + W^{-1} L(lambda)^{-1} d_s
      for (std::size_t j = 0; j < n; ++j) rdt[j] = eta * rd[j];
      for (std::size_t j = nf; j < nf + nl; ++j)
        rdt[j] += dsv[j] / x[j];  // W^{-1} lambda^{-1} d
      for (std::size_t k_ = 0; k_ < ncones; ++k_) {
        const ConeBlock& c = f.cones[k_];
        const auto st = static_cast<std::size_t>(c.start);
        tmp.resize(static_cast<std::size_t>(c.size));
        tmp2.resize(static_cast<std::size_t>(c.size));
        arrow_solve(lam[k_].data(), &dsv[st], c.size, tmp.data());
        quadratic(wmh[k_].data(), tmp.data(), c.size, tmp2.data());
        for (Index i = 0; i < c.size; ++i)
          rdt[st + static_cast<std::size_t>(i)] += tmp2[static_cast<std::size_t>(i)];
      }
      for (std::size_t j = 0; j < n; ++j) rhs[j] = -rdt[j];
      for (std::size_t i = 0; i < m; ++i) rhs[n + i] = eta * rp[i];
      qp::ipm_detail::solve_refined(ldl, k, rhs, &sol_u);
      qp::ipm_detail::refine_unregularized(ldl, k0, rhs, &sol_u);
      for (std::size_t j = 0; j < n; ++j) u1[j] = sol_u[j];
      for (std::size_t i = 0; i < m; ++i) u2[i] = sol_u[n + i];
      dtau = (eta * rg + dk / tau + dot(f.c, u1) - dot(f.b, u2)) / (kappa / tau - cp1 + bp2);
      for (std::size_t j = 0; j < n; ++j) dx[j] = u1[j] + dtau * p1[j];
      for (std::size_t i = 0; i < m; ++i) dy[i] = u2[i] + dtau * p2[i];
      dkappa = (dk - kappa * dtau) / tau;
      // ds = W^{-1} L(lambda)^{-1} d_s - W^{-2} dx
      for (std::size_t j = 0; j < nf; ++j) ds[j] = 0.0;
      for (std::size_t j = nf; j < nf + nl; ++j) ds[j] = (dsv[j] - s[j] * dx[j]) / x[j];
      for (std::size_t k_ = 0; k_ < ncones; ++k_) {
        const ConeBlock& c = f.cones[k_];
        const auto st = static_cast<std::size_t>(c.start);
        tmp.resize(static_cast<std::size_t>(c.size));
        tmp2.resize(static_cast<std::size_t>(c.size));
        arrow_solve(lam[k_].data(), &dsv[st], c.size, tmp.data());
        quadratic(wmh[k_].data(), tmp.data(), c.size, &ds[st]);
        quadratic(v[k_].data(), &dx[st], c.size, tmp2.data());
        for (Index i = 0; i < c.size; ++i)
          ds[st + static_cast<std::size_t>(i)] -= tmp2[static_cast<std::size_t>(i)];
      }
    };
    // The largest step keeping (x, tau) and (s, kappa) inside their cones.
    const auto max_step = [&]() {
      double alpha = std::numeric_limits<double>::infinity();
      for (std::size_t j = nf; j < nf + nl; ++j) {
        if (dx[j] < 0.0) alpha = std::min(alpha, -x[j] / dx[j]);
        if (ds[j] < 0.0) alpha = std::min(alpha, -s[j] / ds[j]);
      }
      for (const ConeBlock& c : f.cones) {
        const auto st = static_cast<std::size_t>(c.start);
        alpha = std::min(alpha, jordan::max_step(&x[st], &dx[st], c.size));
        alpha = std::min(alpha, jordan::max_step(&s[st], &ds[st], c.size));
      }
      if (dtau < 0.0) alpha = std::min(alpha, -tau / dtau);
      if (dkappa < 0.0) alpha = std::min(alpha, -kappa / dkappa);
      return alpha;
    };

    // ---- predictor: d_s = -lambda o lambda, d_kappa = -tau kappa ---------------------------
    for (std::size_t j = nf; j < nf + nl; ++j) dsv[j] = -x[j] * s[j];
    for (std::size_t k_ = 0; k_ < ncones; ++k_) {
      const ConeBlock& c = f.cones[k_];
      product(lam[k_].data(), lam[k_].data(), c.size, &dsv[static_cast<std::size_t>(c.start)]);
      for (Index i = 0; i < c.size; ++i) dsv[static_cast<std::size_t>(c.start + i)] *= -1.0;
    }
    newton(-tau * kappa, 1.0);
    const double alpha_aff = std::min(1.0, max_step());
    double mu_aff = (tau + alpha_aff * dtau) * (kappa + alpha_aff * dkappa);
    for (std::size_t j = nf; j < n; ++j)
      mu_aff += (x[j] + alpha_aff * dx[j]) * (s[j] + alpha_aff * ds[j]);
    mu_aff /= nu + 1.0;
    const double sigma = std::clamp(std::pow(std::max(0.0, mu_aff) / mu, 3.0), 0.0, 1.0);
    dxa = dx;
    dsa = ds;
    const double dtau_aff = dtau;
    const double dkappa_aff = dkappa;

    // ---- corrector: d_s = sigma mu e - lambda o lambda - (W^{-1} dx_aff) o (W ds_aff)
    // --------
    for (std::size_t j = nf; j < nf + nl; ++j)
      dsv[j] = sigma * mu - x[j] * s[j] - dxa[j] * dsa[j];
    for (std::size_t k_ = 0; k_ < ncones; ++k_) {
      const ConeBlock& c = f.cones[k_];
      const auto st = static_cast<std::size_t>(c.start);
      const auto sz = static_cast<std::size_t>(c.size);
      tmp.resize(sz);
      tmp2.resize(sz);
      std::vector<double> cross(sz);
      quadratic(wmh[k_].data(), &dxa[st], c.size, tmp.data());  // W^{-1} dx_aff
      quadratic(wh[k_].data(), &dsa[st], c.size, tmp2.data());  // W ds_aff
      product(tmp.data(), tmp2.data(), c.size, cross.data());
      product(lam[k_].data(), lam[k_].data(), c.size, &dsv[st]);
      for (std::size_t i = 0; i < sz; ++i)
        dsv[st + i] = (i == 0 ? sigma * mu : 0.0) - dsv[st + i] - cross[i];
    }
    newton(sigma * mu - tau * kappa - dtau_aff * dkappa_aff, 1.0 - sigma);
    const double alpha = std::min(1.0, tol::kConicFractionToBoundary * max_step());

    for (std::size_t j = 0; j < n; ++j) {
      x[j] += alpha * dx[j];
      s[j] += alpha * ds[j];
    }
    for (std::size_t i = 0; i < m; ++i) y[i] += alpha * dy[i];
    tau += alpha * dtau;
    kappa += alpha * dkappa;
    ++iterations;
    const auto finite = [](const std::vector<double>& z) {
      return std::all_of(z.begin(), z.end(), [](double e) { return std::isfinite(e); });
    };
    if (!finite(x) || !finite(s) || !finite(y) || !std::isfinite(tau) ||
        !std::isfinite(kappa)) {
      status = SolveStatus::kNumericalError;
      message = "the iterate stopped being finite";
      break;
    }
  }

  out.status = status;
  out.message = message;
  out.iterations = iterations;
  out.primal_residual = pres;
  out.dual_residual = dres;
  out.gap = gap;
  if (status == SolveStatus::kOptimal) {
    out.x.assign(n, 0.0);
    out.y.assign(m, 0.0);
    out.s.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      out.x[j] = x[j] / tau;
      out.s[j] = s[j] / tau;
    }
    for (std::size_t i = 0; i < m; ++i) out.y[i] = y[i] / tau;
  }
  return out;
}

}  // namespace sankhya::conic
