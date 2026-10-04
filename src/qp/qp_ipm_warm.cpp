// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the warm start of the proximal QP interior point (#494, #893).
//
// References
//   Gondzio, "Warm start of the primal-dual method applied in the cutting-plane scheme",
//     Mathematical Programming 83 (1998) 125-143 - the idea this follows: hand on an
//     advanced but still well-centred iterate of the previous solve rather than its optimum
//     (an optimum sits on the boundary, where the next solve's first steps are blocked), and
//     restore positivity and centrality where the modification broke them before the usual
//     iteration resumes.
//   Gondzio, "Multiple centrality corrections in a primal-dual method for linear
//     programming", Computational Optimization and Applications 6 (1996) 137-156 - the
//     centrality box [beta_min mu, beta_max mu] (0.1 and 10) the complementarity products are
//     moved into.
//   Yildirim & Wright, "Warm-start strategies in interior-point methods for linear
//     programming", SIAM J. Optim. 12(3) (2002) - why a saved iterate with a larger mu
//     tolerates a larger change to the problem, which is what a branching step is.
//
// WHAT A BRANCH CHANGES. A child node is its parent with one column's bound tightened past
// the parent's value of it (or a binary fixed, which the standard form then substitutes out).
// The parent's saved iterate is therefore primal infeasible for the child in that column
// alone, and its multipliers are still those of a well-centred point of a nearby problem.
//
// THE SHIFT.
//
//   1. the point is mapped onto the child's columns (fixed ones drop out), each slack column
//      recovered as its row's activity a'x, so the rows the point satisfied stay satisfied;
//   2. every column is moved to at least d = kQpIpmWarmShiftDistance (0.1) from each of its
//      finite bounds (half the width when the column is narrower than 2d): the branched
//      column, now outside its bound, and any that sat closer than d;
//   3. the bound multipliers are the saved ones (recomputed from the stationarity residual,
//      sign-split, when the caller gave none), then each moved into the centrality box:
//      z in [0.1 mu / s, 10 mu / s] for its slack s, mu the average product at the shifted
//      point, so every product is between a tenth and ten times the average and none is zero;
//   4. the row multipliers are the saved ones, and the proximal parameters rho and delta
//      start where the saved iterate had them (raised there by a pivot that came out wrong,
//      so the child need not raise them again), never below the configured floor.
//
// The iteration then runs from that point exactly as from the cold start; it is an infeasible
// interior point method, so the primal residual step 2 introduced is driven to zero like any
// other. Nothing here can change what counts as optimal: qp_ipm.cpp's stopping test is the
// same, and a warm run that fails or stalls is abandoned for the cold start (qp_ipm.cpp).

#include "qp_ipm_warm.hpp"

#include <algorithm>
#include <cmath>

#include "sankhya/tolerances.hpp"

namespace sankhya::qp::ipm_detail {
namespace {

/// The model row of each slack column, -1 where the standard form has none.
std::vector<Index> model_row_of_slack(const Standard& s) {
  std::vector<Index> model_row_of(static_cast<std::size_t>(s.rows), -1);
  for (std::size_t i = 0; i < s.row_of.size(); ++i) {
    if (s.row_of[i] >= 0)
      model_row_of[static_cast<std::size_t>(s.row_of[i])] = static_cast<Index>(i);
  }
  std::vector<Index> out(static_cast<std::size_t>(s.slacks), -1);
  for (Index k = 0; k < s.slacks; ++k) {
    out[static_cast<std::size_t>(k)] =
        model_row_of[static_cast<std::size_t>(s.slack_row[static_cast<std::size_t>(k)])];
  }
  return out;
}

}  // namespace

WarmShift warm_start_iterate(const Model& model, const Standard& s, const QpIpmWarmStart& warm,
                             double regularization_floor, std::vector<double>* v_out,
                             std::vector<double>* y_out, std::vector<double>* zl_out,
                             std::vector<double>* zu_out) {
  const auto nc = static_cast<std::size_t>(s.cols);
  const auto nr = static_cast<std::size_t>(s.rows);
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double>& v = *v_out;
  std::vector<double>& y = *y_out;
  std::vector<double>& zl = *zl_out;
  std::vector<double>& zu = *zu_out;
  v.assign(nc, 0.0);
  y.assign(nr, 0.0);
  zl.assign(nc, 0.0);
  zu.assign(nc, 0.0);
  const std::vector<Index> slack_model_row = model_row_of_slack(s);
  const double sense = model.sense_multiplier();

  // ---- 1. the point, and each slack as its row's activity ---------------------------------
  for (std::size_t j = 0; j < n; ++j) {
    if (s.column_of[j] >= 0) v[static_cast<std::size_t>(s.column_of[j])] = warm.col_value[j];
  }
  // With the slacks at zero, (M v)_r is the activity of the kept columns; a slack row's b_r is
  // minus what the fixed columns contribute, so a'x = (M v)_r - b_r. (#924 took (M v)_r alone,
  // which is off by the fixed columns' share once a branch has fixed one.)
  std::vector<double> mv(nr, 0.0);
  if (s.rows > 0) s.m.multiply_add(v.data(), mv.data());
  for (Index k = 0; k < s.slacks; ++k) {
    const auto r = static_cast<std::size_t>(s.slack_row[static_cast<std::size_t>(k)]);
    v[static_cast<std::size_t>(s.free_n + k)] = mv[r] - s.b[r];
  }

  // ---- 4 (first half). the row multipliers --------------------------------------------------
  if (warm.row_dual.size() == m) {
    for (std::size_t i = 0; i < m; ++i) {
      if (s.row_of[i] >= 0) y[static_cast<std::size_t>(s.row_of[i])] = sense * warm.row_dual[i];
    }
  }

  // ---- 2. into the interior -----------------------------------------------------------------
  const double distance = tol::kQpIpmWarmShiftDistance;
  WarmShift shift;
  std::vector<bool> has_lower(nc), has_upper(nc);
  for (std::size_t j = 0; j < nc; ++j) {
    has_lower[j] = is_finite_bound(s.lower[j]);
    has_upper[j] = is_finite_bound(s.upper[j]);
    if (!std::isfinite(v[j])) v[j] = 0.0;
    const double width = (has_lower[j] && has_upper[j]) ? s.upper[j] - s.lower[j] : kInfinity;
    const double inset = std::min(distance, 0.5 * width);
    const double before = v[j];
    if (has_lower[j] && v[j] < s.lower[j] + inset) v[j] = s.lower[j] + inset;
    if (has_upper[j] && v[j] > s.upper[j] - inset) v[j] = s.upper[j] - inset;
    if (v[j] != before) ++shift.moved_inside;
  }

  // ---- 3. the bound multipliers
  // ---------------------------------------------------------------
  const bool column_z = warm.col_z_lower.size() == n && warm.col_z_upper.size() == n;
  const bool row_z = warm.row_z_lower.size() == m && warm.row_z_upper.size() == m;
  std::vector<bool> given(nc, false);
  if (column_z) {
    for (std::size_t j = 0; j < n; ++j) {
      const Index k = s.column_of[j];
      if (k < 0) continue;
      zl[static_cast<std::size_t>(k)] = warm.col_z_lower[j];
      zu[static_cast<std::size_t>(k)] = warm.col_z_upper[j];
      given[static_cast<std::size_t>(k)] = true;
    }
  }
  if (row_z) {
    for (Index k = 0; k < s.slacks; ++k) {
      const Index i = slack_model_row[static_cast<std::size_t>(k)];
      if (i < 0) continue;
      const auto u = static_cast<std::size_t>(s.free_n + k);
      zl[u] = warm.row_z_lower[static_cast<std::size_t>(i)];
      zu[u] = warm.row_z_upper[static_cast<std::size_t>(i)];
      given[u] = true;
    }
  }
  if (!std::all_of(given.begin(), given.end(), [](bool g) { return g; })) {
    // The stationarity residual H v + g - M'y with z = 0, sign-split: the multipliers that
    // would make the point dual feasible, as the cold start takes them.
    std::vector<double> hv(nc, 0.0), mty(nc, 0.0);
    hessian_times(s.h, v, &hv);
    if (s.rows > 0) s.m.transpose_multiply_add(y.data(), mty.data());
    for (std::size_t j = 0; j < nc; ++j) {
      if (given[j]) continue;
      const double rd = hv[j] + s.g[j] - mty[j];
      zl[j] = std::max(rd, 0.0);
      zu[j] = std::max(-rd, 0.0);
    }
  }
  // The centring target: the average product at the SHIFTED point, not the saved iterate's
  // mu. The shift has moved slacks, so the products the saved mu described no longer exist;
  // on the 80 random MIQPs of tests/unit/test_miqp_node_ipm.cpp, with the save level and
  // shift below, recentring on the saved mu took 2,903 node IPM iterations, recentring on
  // this average 2,088, and the cold start 2,617. Never below kQpIpmWarmCentringFloor, so the
  // box is not empty.
  double sum = 0.0;
  Index count = 0;
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j]) {
      sum += (v[j] - s.lower[j]) * (std::isfinite(zl[j]) ? std::max(zl[j], 0.0) : 0.0);
      ++count;
    }
    if (has_upper[j]) {
      sum += (s.upper[j] - v[j]) * (std::isfinite(zu[j]) ? std::max(zu[j], 0.0) : 0.0);
      ++count;
    }
  }
  const double mu = std::max(count > 0 ? sum / static_cast<double>(count) : 0.0,
                             tol::kQpIpmWarmCentringFloor);
  const auto centre = [&](double slack, double* z) {
    const double low = tol::kQpIpmWarmCentringLow * mu / slack;
    const double high = tol::kQpIpmWarmCentringHigh * mu / slack;
    const double before = *z;
    *z = std::isfinite(*z) ? std::min(std::max(*z, low), high) : low;
    if (*z != before) ++shift.recentred;
  };
  for (std::size_t j = 0; j < nc; ++j) {
    if (has_lower[j]) {
      centre(v[j] - s.lower[j], &zl[j]);
    } else {
      zl[j] = 0.0;
    }
    if (has_upper[j]) {
      centre(s.upper[j] - v[j], &zu[j]);
    } else {
      zu[j] = 0.0;
    }
  }
  for (double& e : y) {
    if (!std::isfinite(e)) e = 0.0;
  }

  // ---- 4 (second half). the proximal parameters
  // ----------------------------------------------
  const auto usable = [&](double p) {
    return std::isfinite(p) && p > regularization_floor ? p : regularization_floor;
  };
  shift.rho = usable(warm.rho);
  shift.delta = usable(warm.delta);
  shift.mu = mu;
  return shift;
}

QpIpmWarmStart save_iterate(const Model& model, const Standard& s, const std::vector<double>& v,
                            const std::vector<double>& y, const std::vector<double>& zl,
                            const std::vector<double>& zu, double rho, double delta, double mu,
                            Count iterations) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const double sense = model.sense_multiplier();
  QpIpmWarmStart warm;
  warm.col_value.assign(n, 0.0);
  warm.col_z_lower.assign(n, 0.0);
  warm.col_z_upper.assign(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    const Index k = s.column_of[j];
    if (k < 0) {
      warm.col_value[j] = s.fixed_value[j];
      continue;
    }
    const auto u = static_cast<std::size_t>(k);
    warm.col_value[j] = v[u];
    warm.col_z_lower[j] = zl[u];
    warm.col_z_upper[j] = zu[u];
  }
  warm.row_dual.assign(m, 0.0);
  warm.row_z_lower.assign(m, 0.0);
  warm.row_z_upper.assign(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    if (s.row_of[i] >= 0) warm.row_dual[i] = sense * y[static_cast<std::size_t>(s.row_of[i])];
  }
  const std::vector<Index> slack_model_row = model_row_of_slack(s);
  for (Index k = 0; k < s.slacks; ++k) {
    const Index i = slack_model_row[static_cast<std::size_t>(k)];
    if (i < 0) continue;
    const auto u = static_cast<std::size_t>(s.free_n + k);
    warm.row_z_lower[static_cast<std::size_t>(i)] = zl[u];
    warm.row_z_upper[static_cast<std::size_t>(i)] = zu[u];
  }
  warm.rho = rho;
  warm.delta = delta;
  warm.mu = mu;
  warm.iterations = iterations;
  return warm;
}

}  // namespace sankhya::qp::ipm_detail
