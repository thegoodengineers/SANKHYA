// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish after the QP interior point (#980, qp_ipm_finish, off by
// default). Called from src/qp/qp_ipm.cpp once its own loop reports optimal with a
// complementarity product still above the in-process KKT gate's share.

#include "qp_ipm_finish.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "la/ldl.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::qp::ipm_detail {

double largest_complementarity_product(const Standard& s, const std::vector<bool>& has_lower,
                                       const std::vector<bool>& has_upper,
                                       const std::vector<double>& v, const std::vector<double>& y,
                                       const std::vector<double>& zl,
                                       const std::vector<double>& zu,
                                       const std::vector<double>& rp) {
  const auto nc = static_cast<std::size_t>(s.cols);
  double largest = 0.0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j]) largest = std::max(largest, std::fabs((v[j] - s.lower[j]) * zl[j]));
    if (has_upper[j]) largest = std::max(largest, std::fabs((s.upper[j] - v[j]) * zu[j]));
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
// Reference: Stellato, Banjac, Goulart, Bemporad & Boyd, "OSQP: an operator splitting solver
// for quadratic programs", Math. Prog. Comp. 12 (2020), sec. 5.3 - solution polishing: fix
// the active set an ADMM iterate has found, and solve the reduced equality-constrained KKT
// system on what is left exactly. This is the same idea after this engine's interior point
// instead of after ADMM, built entirely from pieces this engine already has (the same
// Standard, the project's own sparse LDL^T and the solve_refined helper above), so it is
// orchestration of the existing IPM output rather than a new primal algorithm; no separate
// numerical-method justification is added beyond this citation.
//
// ACTIVE-SET TEST. A bound is called active when the point already sits on it to a loose,
// scale-aware distance (tol::kQpIpmFinishActiveDistance), independent of its multiplier: at
// convergence of the central path essentially every bound's complementarity product is
// small (that is what "the relative measures are met" means in aggregate), so the product
// alone cannot tell an active bound from an interior one whose multiplier merely happens to
// be small; the distance to the bound can. Calling a bound active when it should have stayed
// interior only costs a rejected finish below, never a wrong answer - see the acceptance
// test, which is the property this function exists to protect.
bool active_set_finish(const Standard& s, const std::vector<bool>& has_lower,
                       const std::vector<bool>& has_upper, double tolerance,
                       double primal_tolerance, double regularization_floor,
                       std::vector<double>* v, std::vector<double>* y, std::vector<double>* zl,
                       std::vector<double>* zu) {
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);

  std::vector<bool> active_lower(nc, false), active_upper(nc, false);
  std::vector<double> fixed(nc, 0.0);
  Index active_count = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (!has_lower[j] && !has_upper[j]) continue;
    const double distance_lower = has_lower[j] ? (*v)[j] - s.lower[j] : kInfinity;
    const double distance_upper = has_upper[j] ? s.upper[j] - (*v)[j] : kInfinity;
    const double scale = std::max({1.0, has_lower[j] ? std::fabs(s.lower[j]) : 0.0,
                                   has_upper[j] ? std::fabs(s.upper[j]) : 0.0});
    if (distance_lower <= distance_upper && has_lower[j] &&
        distance_lower <= tol::kQpIpmFinishActiveDistance * scale) {
      active_lower[j] = true;
      fixed[j] = s.lower[j];
      ++active_count;
    } else if (has_upper[j] && distance_upper <= tol::kQpIpmFinishActiveDistance * scale) {
      active_upper[j] = true;
      fixed[j] = s.upper[j];
      ++active_count;
    }
  }
  if (active_count == 0) return false;  // nothing identified to pin: leave the IPM's point

  std::vector<Index> new_col(nc, -1);
  Index nf = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (!active_lower[j] && !active_upper[j]) new_col[static_cast<std::size_t>(j)] = nf++;
  }
  if (nf == 0) return false;  // every column pinned: no reduced system to solve

  // ---- build the reduced KKT system: free columns and all rows, active columns folded ------
  // into the right-hand side - the same lower-triangle layout as KktMatrix::build (the
  // variable block negative definite, M attached once and symmetrized by KktMatrix::multiply
  // inside solve_refined, the row block delta I), assembled with add_entry/finalize instead
  // of a fixed pattern because the reduction changes the pattern every time it runs.
  const Index dim = nf + s.rows;
  SparseMatrix k(dim, dim);
  std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
  for (Index oj = 0; oj < s.free_n; ++oj) {
    const auto uoj = static_cast<std::size_t>(oj);
    const ColumnView q = s.h.column(oj);
    for (Index p = 0; p < q.size; ++p) {
      const Index row = q.rows[p];
      const double value = q.values[p];
      if (row == oj) {
        if (new_col[uoj] >= 0) k.add_entry(new_col[uoj], new_col[uoj], -value);
        continue;
      }
      const auto urow = static_cast<std::size_t>(row);
      const bool row_free = new_col[urow] >= 0, col_free = new_col[uoj] >= 0;
      if (row_free && col_free) {
        k.add_entry(new_col[urow], new_col[uoj], -value);
      } else if (row_free && !col_free) {
        rhs[static_cast<std::size_t>(new_col[urow])] += value * fixed[uoj];
      } else if (!row_free && col_free) {
        rhs[static_cast<std::size_t>(new_col[uoj])] += value * fixed[urow];
      }
    }
  }
  std::vector<double> b_adj = s.b;
  for (std::size_t j = 0; j < nc; ++j) {
    const ColumnView a = s.m.column(static_cast<Index>(j));
    if (new_col[j] >= 0) {
      const auto nj = static_cast<std::size_t>(new_col[j]);
      k.add_entry(new_col[j], new_col[j], -regularization_floor);
      rhs[nj] += s.g[j];
      for (Index p = 0; p < a.size; ++p) k.add_entry(nf + a.rows[p], new_col[j], a.values[p]);
    } else {
      for (Index p = 0; p < a.size; ++p) {
        b_adj[static_cast<std::size_t>(a.rows[p])] -= a.values[p] * fixed[j];
      }
    }
  }
  for (Index r = 0; r < s.rows; ++r) {
    k.add_entry(nf + r, nf + r, regularization_floor);
    rhs[static_cast<std::size_t>(nf + r)] += b_adj[static_cast<std::size_t>(r)];
  }
  k.finalize();

  std::vector<signed char> signs(static_cast<std::size_t>(dim), 1);
  for (Index j = 0; j < nf; ++j) signs[static_cast<std::size_t>(j)] = -1;
  SparseLdl ldl;
  if (!ldl.analyze(k) ||
      !ldl.factorize_quasidefinite(k, signs, tol::kQpIpmPivotShare * regularization_floor)) {
    return false;  // the reduced system refuses to factor: keep the IPM's own point
  }
  std::vector<double> solved;
  solve_refined(ldl, k, rhs, &solved);
  if (!std::all_of(solved.begin(), solved.end(), [](double x) { return std::isfinite(x); })) {
    return false;
  }

  std::vector<double> new_v(nc), new_y(nr, 0.0);
  for (std::size_t j = 0; j < nc; ++j) {
    new_v[j] = new_col[j] >= 0 ? solved[static_cast<std::size_t>(new_col[j])] : fixed[j];
  }
  for (std::size_t r = 0; r < nr; ++r) new_y[r] = solved[static_cast<std::size_t>(nf) + r];

  // The multipliers on the pinned columns, from the same stationarity residual the cold
  // start above derives z from: z = 0 everywhere, then split by sign on the columns that are
  // pinned (both signs, for a column fixed at equal lower and upper bounds).
  std::vector<double> hv(nc, 0.0), mty(nc, 0.0);
  hessian_times(s.h, new_v, &hv);
  if (s.rows > 0) s.m.transpose_multiply_add(new_y.data(), mty.data());
  std::vector<double> new_zl(nc, 0.0), new_zu(nc, 0.0);
  for (std::size_t j = 0; j < nc; ++j) {
    if (!active_lower[j] && !active_upper[j]) continue;
    const double residual = hv[j] + s.g[j] - mty[j];
    if (active_lower[j]) new_zl[j] = active_upper[j] ? std::max(residual, 0.0) : residual;
    if (active_upper[j]) new_zu[j] = active_lower[j] ? std::max(-residual, 0.0) : -residual;
  }

  // ---- acceptance: never worse than the point the interior point already found -------------
  std::vector<double> mv(nr, 0.0), rp(nr, 0.0), rd(nc, 0.0);
  s.m.multiply_add(new_v.data(), mv.data());
  for (std::size_t r = 0; r < nr; ++r) rp[r] = s.b[r] - mv[r];
  for (std::size_t j = 0; j < nc; ++j) rd[j] = hv[j] + s.g[j] - mty[j] - new_zl[j] + new_zu[j];
  const double primal_rel = inf_norm(rp) / (1.0 + std::max(inf_norm(mv), inf_norm(s.b)));
  const double dual_rel =
      inf_norm(rd) / (1.0 + std::max({inf_norm(hv), inf_norm(s.g), inf_norm(mty)}));
  if (!(primal_rel <= tolerance) || !(dual_rel <= tolerance)) return false;
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j] && new_v[j] < s.lower[j] - primal_tolerance) return false;
    if (has_upper[j] && new_v[j] > s.upper[j] + primal_tolerance) return false;
    if (new_zl[j] < -primal_tolerance || new_zu[j] < -primal_tolerance) return false;
  }
  const double largest_product = largest_complementarity_product(
      s, has_lower, has_upper, new_v, new_y, new_zl, new_zu, rp);
  if (largest_product > tol::kQpIpmComplementarityShare * tol::kComplementarity) return false;

  *v = std::move(new_v);
  *y = std::move(new_y);
  *zl = std::move(new_zl);
  *zu = std::move(new_zu);
  return true;
}

}  // namespace sankhya::qp::ipm_detail
