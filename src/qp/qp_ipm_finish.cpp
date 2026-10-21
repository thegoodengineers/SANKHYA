// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish after the QP interior point (#980, qp_ipm_finish, on by
// default since #1016). Called from src/qp/qp_ipm.cpp once its own loop reports optimal with a
// complementarity product still above the in-process KKT gate's share.

#include "qp_ipm_finish.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "la/dense_qr.hpp"
#include "la/ldl.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::qp::ipm_detail {

double largest_complementarity_product(
    const Standard& s, const std::vector<bool>& has_lower, const std::vector<bool>& has_upper,
    const std::vector<double>& v, const std::vector<double>& y, const std::vector<double>& zl,
    const std::vector<double>& zu, const std::vector<double>& rp) {
  const auto nc = static_cast<std::size_t>(s.cols);
  double largest = 0.0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j]) largest = std::max(largest, std::fabs((v[j] - s.lower[j]) * zl[j]));
    if (has_upper[j]) largest = std::max(largest, std::fabs((s.upper[j] - v[j]) * zu[j]));
  }
  // ...and as the gate prices a column: the DERIVED reduced cost H v + g - M^T y (which
  // carries the stationarity residual, not only z_l - z_u) times the distance to the nearer
  // finite bound. On qgrow15, qgfrdxpn and qshell every z-product was under the share and this
  // one was not, so the finish was never tried and the gate withdrew the claim.
  std::vector<double> d(nc, 0.0), mty(nc, 0.0);
  hessian_times(s.h, v, &d);
  if (s.rows > 0) s.m.transpose_multiply_add(y.data(), mty.data());
  for (std::size_t j = 0; j < nc; ++j) {
    const double below = has_lower[j] ? v[j] - s.lower[j] : kInfinity;
    const double above = has_upper[j] ? s.upper[j] - v[j] : kInfinity;
    const double slack = std::min(below, above);
    if (std::isfinite(slack)) {
      largest = std::max(largest, std::fabs(d[j] + s.g[j] - mty[j]) * std::fabs(slack));
    }
  }
  // A slack's row: the row's multiplier against the activity's distance to its nearer side.
  for (Index slack = 0; slack < s.slacks; ++slack) {
    const auto uu = static_cast<std::size_t>(s.free_n + slack);
    const auto rr = static_cast<std::size_t>(s.slack_row[static_cast<std::size_t>(slack)]);
    const double activity = v[uu] - rp[rr];
    double distance = kInfinity;
    if (has_lower[uu]) distance = std::min(distance, std::fabs(activity - s.lower[uu]));
    if (has_upper[uu]) distance = std::min(distance, std::fabs(s.upper[uu] - activity));
    if (std::isfinite(distance)) largest = std::max(largest, std::fabs(y[rr]) * distance);
  }
  return largest;
}

// ---- #980: an active-set finish after the interior point ---------------------------------
//
// References
//   Stellato, Banjac, Goulart, Bemporad & Boyd, "OSQP: an operator splitting solver for
//     quadratic programs", Math. Prog. Comp. 12 (2020), sec. 5.3 - solution polishing: fix the
//     active set an iterate has found and solve the reduced equality-constrained KKT system.
//   Businger & Golub, Numer. Math. 7 (1965) - QR with column pivoting (src/la/dense_qr.cpp),
//     here choosing an independent subset of the active rows.
//   El-Bakry, Tapia & Zhang, "A study of indicators for identifying zero variables in
//     interior-point methods", SIAM Review 36 (1994) - a bound whose slack is below its
//     multiplier is active.
//   Nocedal & Wright, Numerical Optimization, 2nd ed. (2006), sec. 16.5 - the primal active-set
//     rule of one change to the working set per step.
//
// WHY RANK-REVEALING. On a degenerate optimum (qseba: 68 of 972 active rows dependent) the
// reduced KKT system is singular, its right-hand side consistent only to the 1e-9 the data is
// printed to, and any minimum-norm or regularized solve spreads multipliers of 1e+5 over
// rows it leaves 1e-10 off their bounds: a product of 1e-5 the gate refuses. A BASIC solution
// does not: the independent rows QR with column pivoting picks are met exactly, the dependent
// ones are met to the data's own consistency. The primal point is solved as a correction to
// the interior point's (so a direction the reduced system leaves free keeps the interior
// point's value), and the multipliers are the interior point's own, moved the least on the
// chosen rows to satisfy stationarity exactly: on a degenerate dual the basic multipliers can
// differ from the interior point's in sign (qship12s, 396 columns), its own do not.
//
// Every candidate answers to the acceptance test at the end, so a wrong guess of the active
// set costs a rejected finish, never a wrong answer.
namespace {

enum class Pin : signed char { kFree, kLower, kUpper };

/// The dense product of the reduced system is at most this many entries (one QR of the
/// active rows restricted to the free columns); above it the finish declines.
/// ponytail: dense QR caps the finish at 2e7 entries (160 MB); a sparse rank-revealing QR
/// would lift it, and no Maros-Meszaros holdout needs it.
constexpr double kDenseLimit = 2e7;
/// ...and the reduced KKT system itself is solved densely up to this many entries (dimension
/// 2000, about 5e9 flops for its QR); above it the regularized sparse LDL^T and refinement.
/// ponytail: dense COD caps the least-norm solve at dimension 2000; above it the sparse path
/// can drift along a singular direction and the finish is refused, never wrong.
constexpr double kDenseKktLimit = 4e6;

struct Candidate {
  std::vector<double> v, y, zl, zu, residual;  // residual = H v + g - M^T y
};

/// The power of two nearest 1 / sqrt(size): a scaling that adds no rounding of its own.
double unit_scale(double size) {
  return size > 0.0 ? std::exp2(std::round(-0.5 * std::log2(size))) : 1.0;
}

/// Symmetric Ruiz equilibration of the dense symmetric n x n matrix `a`, in place: returns
/// D with every row of D A D near unit largest magnitude (Ruiz, "A scaling algorithm to
/// equilibrate both rows and columns norms in matrices", RAL-TR-2001-034). The QR's rank
/// test is relative to its first pivot, and unscaled, a row of 1e-12 coefficients beside a
/// Hessian of order 1 reads as dependent and drops out of the solve (#1079).
std::vector<double> equilibrate(std::vector<double>* a, std::size_t n) {
  std::vector<double> d(n, 1.0), r(n);
  for (int pass = 0; pass < tol::kQpIpmFinishEquilibrationPasses; ++pass) {
    std::fill(r.begin(), r.end(), 0.0);
    for (std::size_t c = 0; c < n; ++c) {
      for (std::size_t i = 0; i < n; ++i) r[c] = std::max(r[c], std::fabs((*a)[c * n + i]));
    }
    for (std::size_t i = 0; i < n; ++i) r[i] = unit_scale(r[i]);
    for (std::size_t c = 0; c < n; ++c) {
      for (std::size_t i = 0; i < n; ++i) (*a)[c * n + i] *= r[i] * r[c];
    }
    for (std::size_t i = 0; i < n; ++i) d[i] *= r[i];
  }
  return d;
}

/// x scaled entrywise by d.
std::vector<double> scaled(const std::vector<double>& d, std::vector<double> x) {
  for (std::size_t i = 0; i < x.size(); ++i) x[i] *= d[i];
  return x;
}

/// One reduced solve for a fixed working set. Returns false when the system is too large or
/// will not factor.
bool solve_reduced(const Standard& s, const std::vector<Pin>& pin,
                   const std::vector<double>& v0, const std::vector<double>& y0,
                   double regularization_floor, const std::function<bool()>& should_stop,
                   Candidate* out) {
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);
  std::vector<Index> new_col(nc, -1);
  std::vector<double> fixed(nc, 0.0);
  Index nf = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (pin[j] == Pin::kFree) {
      new_col[j] = nf++;
    } else {
      fixed[j] = pin[j] == Pin::kLower ? s.lower[j] : s.upper[j];
    }
  }
  if (nf == 0 || static_cast<double>(nf) * static_cast<double>(nr) > kDenseLimit) return false;

  // B = M restricted to the free columns, transposed: nf x rows, column r is row r.
  const auto unf = static_cast<std::size_t>(nf);
  std::vector<double> b(unf * nr, 0.0);
  for (std::size_t j = 0; j < nc; ++j) {
    if (new_col[j] < 0) continue;
    const ColumnView a = s.m.column(static_cast<Index>(j));
    for (Index p = 0; p < a.size; ++p) {
      b[static_cast<std::size_t>(a.rows[p]) * unf + static_cast<std::size_t>(new_col[j])] =
          a.values[p];
    }
  }
  // Each row to unit norm before the rank decision, which is relative to the largest row: a
  // row of 1e-12 coefficients is not dependent on one of order 1 (#1079). w undoes it below.
  std::vector<double> w(nr, 1.0);
  for (std::size_t r = 0; r < nr; ++r) {
    double sum = 0.0;
    for (std::size_t j = 0; j < unf; ++j) sum += b[r * unf + j] * b[r * unf + j];
    w[r] = unit_scale(sum);
    for (std::size_t j = 0; j < unf; ++j) b[r * unf + j] *= w[r];
  }
  DenseQr qr;
  qr.factor(b, nf, s.rows, tol::kQpIpmFinishRankTolerance);
  std::vector<Index> row_sel(nr, -1);
  const Index rank = qr.rank();
  for (Index k = 0; k < rank; ++k)
    row_sel[static_cast<std::size_t>(qr.pivots()[static_cast<std::size_t>(k)])] = k;

  // The reduced KKT system over the free columns and the chosen rows, in KktMatrix's lower
  // layout; `k` regularized for the factorization, `exact` without, refined against.
  const Index dim = nf + rank;
  SparseMatrix k(dim, dim), exact(dim, dim);
  const auto both = [&](Index r, Index c, double value) {
    k.add_entry(r, c, value);
    exact.add_entry(r, c, value);
  };
  std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0), base(rhs.size(), 0.0);
  for (Index oj = 0; oj < s.free_n; ++oj) {
    const ColumnView q = s.h.column(oj);
    for (Index p = 0; p < q.size; ++p) {
      const auto urow = static_cast<std::size_t>(q.rows[p]);
      const auto ucol = static_cast<std::size_t>(oj);
      const bool row_free = new_col[urow] >= 0, col_free = new_col[ucol] >= 0;
      if (row_free && col_free) {
        both(new_col[urow], new_col[ucol], -q.values[p]);
      } else if (row_free) {
        rhs[static_cast<std::size_t>(new_col[urow])] += q.values[p] * fixed[ucol];
      } else if (col_free && urow != ucol) {
        rhs[static_cast<std::size_t>(new_col[ucol])] += q.values[p] * fixed[urow];
      }
    }
  }
  std::vector<double> b_adj = s.b;
  for (std::size_t j = 0; j < nc; ++j) {
    const ColumnView a = s.m.column(static_cast<Index>(j));
    if (new_col[j] < 0) {
      for (Index p = 0; p < a.size; ++p) {
        b_adj[static_cast<std::size_t>(a.rows[p])] -= a.values[p] * fixed[j];
      }
      continue;
    }
    const auto nj = static_cast<std::size_t>(new_col[j]);
    k.add_entry(new_col[j], new_col[j], -regularization_floor);
    rhs[nj] += s.g[j];
    base[nj] = v0[j];
    for (Index p = 0; p < a.size; ++p) {
      const Index r = row_sel[static_cast<std::size_t>(a.rows[p])];
      if (r >= 0) both(nf + r, new_col[j], a.values[p]);
    }
  }
  for (std::size_t r = 0; r < nr; ++r) {
    const Index sel = row_sel[r];
    if (sel < 0) continue;
    k.add_entry(nf + sel, nf + sel, regularization_floor);
    rhs[static_cast<std::size_t>(nf + sel)] = b_adj[r];
    base[static_cast<std::size_t>(nf + sel)] = y0[r];
  }
  // Every entry kept, as standardize() keeps them for s.h and s.m: the default finalize()
  // drops entries below tol::kZeroDrop (1e-11), and a coefficient that small is still the
  // model's. Maros-Meszaros huestis has row coefficients from 2e-21 up, multiplied by row
  // duals near 1e9: dropping column 177's 9.9e-12 made K * base read a stationarity residual
  // of 9.5e-3 at an interior point whose true one is 2e-10, the finish "corrected" an exact
  // value away from the optimum, and then cycled (#1079).
  k.finalize(0.0);
  exact.finalize(0.0);

  // Solve for the correction to the interior point's own point (base).
  std::vector<double> kb(rhs.size());
  KktMatrix::multiply(exact, base, &kb);
  for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] -= kb[i];
  std::vector<double> d;
  const auto udim = static_cast<std::size_t>(dim);
  if (static_cast<double>(dim) * static_cast<double>(dim) <= kDenseKktLimit) {
    // Small enough to be dense: the LEAST-NORM correction by the complete orthogonal
    // decomposition. The system stays singular after the row selection when free columns
    // carry no curvature (qgrow15: rank 600 of 814), and the least-norm correction leaves
    // those directions at the interior point's values; a regularized solve refined against
    // the singular system drifts along them instead.
    std::vector<double> dense(udim * udim, 0.0);
    for (Index col = 0; col < dim; ++col) {
      const ColumnView e = exact.column(col);
      for (Index p = 0; p < e.size; ++p) {
        const auto r = static_cast<std::size_t>(e.rows[p]);
        const auto c = static_cast<std::size_t>(col);
        dense[c * udim + r] = e.values[p];
        dense[r * udim + c] = e.values[p];
      }
    }
    // Solved as (D K D) z = D rhs, d = D z: the rank decision on the equilibrated matrix.
    const std::vector<double> e = equilibrate(&dense, udim);
    DenseQr kqr;
    kqr.factor(dense, dim, dim, tol::kQpIpmFinishRankTolerance);
    d = scaled(e, kqr.min_norm_least_squares(scaled(e, rhs)));
    std::vector<double> kd(udim), r(udim);
    for (int pass = 0; pass < 2; ++pass) {  // refinement in working precision
      KktMatrix::multiply(exact, d, &kd);
      for (std::size_t i = 0; i < udim; ++i) r[i] = rhs[i] - kd[i];
      const std::vector<double> dd = scaled(e, kqr.min_norm_least_squares(scaled(e, r)));
      for (std::size_t i = 0; i < udim; ++i) d[i] += dd[i];
    }
  } else {
    std::vector<signed char> signs(udim, 1);
    for (Index j = 0; j < nf; ++j) signs[static_cast<std::size_t>(j)] = -1;
    SparseLdl ldl;
    if (!ldl.analyze(k, should_stop) ||
        !ldl.factorize_quasidefinite(k, signs, tol::kQpIpmPivotShare * regularization_floor,
                                     should_stop)) {
      return false;
    }
    solve_refined(ldl, k, rhs, &d);
    refine_unregularized(ldl, exact, rhs, &d);
  }
  if (!std::all_of(d.begin(), d.end(), [](double x) { return std::isfinite(x); })) return false;

  out->v.assign(nc, 0.0);
  for (std::size_t j = 0; j < nc; ++j) {
    out->v[j] = new_col[j] >= 0 ? v0[j] + d[static_cast<std::size_t>(new_col[j])] : fixed[j];
  }
  // Multipliers: the interior point's, moved the least (in norm, over all rows) to make the
  // free columns' stationarity exact (B dy = residual_F, least-norm least squares, twice).
  out->y = y0;
  std::vector<double> hv(nc, 0.0), mty(nc, 0.0);
  hessian_times(s.h, out->v, &hv);
  out->residual.assign(nc, 0.0);
  for (int pass = 0; pass < 2; ++pass) {
    std::fill(mty.begin(), mty.end(), 0.0);
    if (s.rows > 0) s.m.transpose_multiply_add(out->y.data(), mty.data());
    std::vector<double> rf(unf, 0.0);
    for (std::size_t j = 0; j < nc; ++j) {
      out->residual[j] = hv[j] + s.g[j] - mty[j];
      if (new_col[j] >= 0) rf[static_cast<std::size_t>(new_col[j])] = out->residual[j];
    }
    if (pass == 1 || rank == 0) break;
    // The least-norm correction over every row, dependent ones included: on a degenerate dual
    // the correction on the chosen rows alone is larger and flips signs (qgrow15).
    const std::vector<double> dy = scaled(w, qr.min_norm_least_squares(rf));
    for (std::size_t r = 0; r < nr; ++r) out->y[r] += dy[r];
  }
  out->zl.assign(nc, 0.0);
  out->zu.assign(nc, 0.0);
  for (std::size_t j = 0; j < nc; ++j) {
    if (pin[j] == Pin::kLower) out->zl[j] = out->residual[j];
    if (pin[j] == Pin::kUpper) out->zu[j] = -out->residual[j];
  }
  return true;
}

}  // namespace

bool active_set_finish(const Standard& s, const std::vector<bool>& has_lower,
                       const std::vector<bool>& has_upper, double tolerance,
                       double primal_tolerance, double regularization_floor,
                       const std::function<bool()>& should_stop, std::vector<double>* v,
                       std::vector<double>* y, std::vector<double>* zl,
                       std::vector<double>* zu) {
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);

  // The working set: a bound is active when the point sits within the scaled distance of it,
  // or when its slack is below its own (right-signed) multiplier.
  std::vector<Pin> pin(nc, Pin::kFree);
  for (std::size_t j = 0; j < nc; ++j) {
    const double dl = has_lower[j] ? (*v)[j] - s.lower[j] : kInfinity;
    const double du = has_upper[j] ? s.upper[j] - (*v)[j] : kInfinity;
    // Each bound against its own size: a column on [0, 1000] is not at 0 when it is 0.05.
    const double near_lower = tol::kQpIpmFinishActiveDistance *
                              (has_lower[j] ? std::max(1.0, std::fabs(s.lower[j])) : 1.0);
    const double near_upper = tol::kQpIpmFinishActiveDistance *
                              (has_upper[j] ? std::max(1.0, std::fabs(s.upper[j])) : 1.0);
    if (has_lower[j] && dl <= du && (dl <= near_lower)) {
      pin[j] = Pin::kLower;
    } else if (has_upper[j] && (du <= near_upper)) {
      pin[j] = Pin::kUpper;
    }
  }

  Candidate c;
  bool settled = false;
  std::vector<int> changes(nc, 0);
  for (int round = 0; round < tol::kQpIpmFinishRounds && !settled; ++round) {
    // On the solve's own clock, between rounds and inside the sparse factorization (the
    // dense path is bounded by kDenseKktLimit): boyd1 (93,261 columns) spent 55 s in these
    // rounds past a 60 s limit the interior point had already used 30 s of, and with the
    // check between rounds alone still 14 s past it inside one factorization. A finish that
    // runs out of time declines, and the interior point's point stands.
    if (should_stop && should_stop()) return false;
    if (!solve_reduced(s, pin, *v, *y, regularization_floor, should_stop, &c)) return false;
    // One change per round, the worst offender first: a free column outside its bounds goes
    // onto the bound it violates; else a pinned bound with a wrong-signed multiplier is
    // released; else a free column whose reduced cost the active rows cannot zero goes onto
    // the bound that sign calls for.
    std::size_t worst = nc;
    double worst_size = 0.0;
    Pin to = Pin::kFree;
    const auto consider = [&](std::size_t j, double size, Pin target) {
      if (size > worst_size) {
        worst_size = size;
        worst = j;
        to = target;
      }
    };
    for (std::size_t j = 0; j < nc; ++j) {
      if (pin[j] != Pin::kFree) continue;
      if (has_lower[j] && c.v[j] < s.lower[j] - primal_tolerance)
        consider(j, s.lower[j] - c.v[j], Pin::kLower);
      if (has_upper[j] && c.v[j] > s.upper[j] + primal_tolerance)
        consider(j, c.v[j] - s.upper[j], Pin::kUpper);
    }
    if (worst == nc) {
      for (std::size_t j = 0; j < nc; ++j) {
        if (pin[j] == Pin::kLower && c.zl[j] < -primal_tolerance)
          consider(j, -c.zl[j], Pin::kFree);
        if (pin[j] == Pin::kUpper && c.zu[j] < -primal_tolerance)
          consider(j, -c.zu[j], Pin::kFree);
      }
    }
    if (worst == nc) {
      double dual_scale = 1.0;
      for (std::size_t j = 0; j < nc; ++j) dual_scale = std::max(dual_scale, std::fabs(s.g[j]));
      for (std::size_t j = 0; j < nc; ++j) {
        if (pin[j] != Pin::kFree) continue;
        const double r = c.residual[j];
        if (std::fabs(r) <= tolerance * dual_scale) continue;
        if (r > 0.0 && has_lower[j]) consider(j, r, Pin::kLower);
        if (r < 0.0 && has_upper[j]) consider(j, -r, Pin::kUpper);
      }
    }
    if (worst == nc) {
      settled = true;
    } else {
      // A column asked to change a third time is cycling (huestis: pinned, released, pinned
      // again on the same column): the working set is not converging, so stop.
      if (++changes[worst] > 2) return false;
      pin[worst] = to;
    }
  }
  if (!settled) return false;

  // ---- acceptance: never worse than the point the interior point already found -------------
  std::vector<double> hv(nc, 0.0), mty(nc, 0.0), mv(nr, 0.0), rp(nr, 0.0), rd(nc, 0.0);
  hessian_times(s.h, c.v, &hv);
  if (s.rows > 0) s.m.transpose_multiply_add(c.y.data(), mty.data());
  s.m.multiply_add(c.v.data(), mv.data());
  for (std::size_t r = 0; r < nr; ++r) rp[r] = s.b[r] - mv[r];
  for (std::size_t j = 0; j < nc; ++j) rd[j] = hv[j] + s.g[j] - mty[j] - c.zl[j] + c.zu[j];
  const double primal_rel = inf_norm(rp) / (1.0 + std::max(inf_norm(mv), inf_norm(s.b)));
  const double dual_rel =
      inf_norm(rd) / (1.0 + std::max({inf_norm(hv), inf_norm(s.g), inf_norm(mty)}));
  if (!(primal_rel <= tolerance) || !(dual_rel <= tolerance)) return false;
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j] && c.v[j] < s.lower[j] - primal_tolerance) return false;
    if (has_upper[j] && c.v[j] > s.upper[j] + primal_tolerance) return false;
    if (c.zl[j] < -primal_tolerance || c.zu[j] < -primal_tolerance) return false;
  }
  const double largest_product =
      largest_complementarity_product(s, has_lower, has_upper, c.v, c.y, c.zl, c.zu, rp);
  if (largest_product > tol::kQpIpmComplementarityShare * tol::kComplementarity) return false;

  *v = std::move(c.v);
  *y = std::move(c.y);
  *zl = std::move(c.zl);
  *zu = std::move(c.zu);
  return true;
}

}  // namespace sankhya::qp::ipm_detail
