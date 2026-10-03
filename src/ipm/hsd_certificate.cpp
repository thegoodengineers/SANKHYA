// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the certificates of the homogeneous self-dual embedding (#475, ipm_hsd): the
// rays read off the iterate, cleaned, and offered to the project's own checker. See hsd.cpp
// for the method and its citations.

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

namespace sankhya::ipm::hsd {

/// The vector with every entry at or below kIpmHsdCertificateRounding of its largest zeroed.
std::vector<double> rounded(const std::vector<double>& v) {
  double largest = 0.0;
  for (const double x : v) largest = std::max(largest, std::fabs(x));
  std::vector<double> out(v.size(), 0.0);
  for (std::size_t k = 0; k < v.size(); ++k) {
    if (std::fabs(v[k]) > tol::kIpmHsdCertificateRounding * largest) out[k] = v[k];
  }
  return out;
}

bool Homogeneous::proves(bool farkas, const std::vector<double>& v, std::string* why) const {
  // On the model this method iterates on, and, when that is a scaled copy, on the caller's
  // model too, the vector mapped back as solve_scaled() maps it (y = Dr yhat, d = Dc dhat):
  // a proof that holds in only one of the two units is not reported.
  if (!(farkas ? farkas_proves_infeasible(model_, v, why)
               : ray_proves_unbounded(model_, v, why))) {
    return false;
  }
  if (original_ == nullptr || scaling_ == nullptr) return true;
  std::vector<double> mapped(v);
  const std::vector<double>& factor = farkas ? scaling_->row : scaling_->column;
  for (std::size_t k = 0; k < mapped.size() && k < factor.size(); ++k) mapped[k] *= factor[k];
  std::string there;
  const bool held = farkas ? farkas_proves_infeasible(*original_, mapped, &there)
                           : ray_proves_unbounded(*original_, mapped, &there);
  if (why != nullptr) *why = held ? there : "in the model's units, " + there;
  return held;
}

std::vector<double> Homogeneous::project_farkas(std::vector<double> y) {
  // THE FARKAS VECTOR PROJECTED ONTO WHAT THE CHECKER ASKS. Aggregating the rows with y gives
  // d = A'y, and the proof fails on any column whose box cannot absorb its d_j: d_j > 0 with
  // no upper bound, d_j < 0 with no lower bound (a free column either way). Where both of a
  // column's z and s go to zero the Theta-weighted correction above has no weight to work
  // with, so this one chooses the weights itself: 1 on the offending set V, kProjectWeight
  // everywhere else, and solves min sum_V (a_j'(y + dy))^2 + w (sum_rest (a_j'dy)^2 +
  // ||dy||^2) through the same assembly and the same SparseLdl (a refactorization, whose
  // pattern is the analysed one), then drops the multipliers of the wrong sign for their row.
  // A few rounds, because fixing V moves the rest. Least squares, not an LP: the checker
  // decides whether the result is a proof.
  if (m_ == 0 || !analyzed_) return {};
  const auto zero_wrong_signs = [&](std::vector<double>* v) {
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(i);
      if (((*v)[u] > 0.0 && !is_finite_bound(model_.row_lower[u])) ||
          ((*v)[u] < 0.0 && !is_finite_bound(model_.row_upper[u]))) {
        (*v)[u] = 0.0;
      }
    }
  };
  const auto N = static_cast<std::size_t>(n_);
  for (int round = 0; round < kProjectRounds; ++round) {
    zero_wrong_signs(&y);
    std::vector<double> d(N, 0.0);
    model_.matrix.transpose_multiply_add(y.data(), d.data());
    std::vector<double> theta_x(N, kProjectWeight);
    std::vector<double> row_shift(static_cast<std::size_t>(m_), kProjectAnchor);
    // A one-sided row whose multiplier is at zero may not move to the wrong sign, which the
    // next round would undo by zeroing it: such a row is pinned by a large shift instead.
    double largest = 0.0;
    for (const double v : y) largest = std::max(largest, std::fabs(v));
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const bool one_sided =
          !is_finite_bound(model_.row_lower[u]) || !is_finite_bound(model_.row_upper[u]);
      if (one_sided && std::fabs(y[u]) <= tol::kIpmHsdCertificateRounding * largest) {
        row_shift[u] = kProjectPin;
      }
    }
    std::vector<double> target(static_cast<std::size_t>(total_), 0.0);
    // V: the columns leaning the wrong way, and the open-sided ones whose d_j is at rounding
    // size either way (a correction elsewhere would tip them), each held to d_j = 0.
    double d_largest = 0.0;
    for (const double v : d) d_largest = std::max(d_largest, std::fabs(v));
    bool any = false;
    for (std::size_t j = 0; j < N; ++j) {
      if (fixed_[j] != 0) continue;
      const bool open = has_upper_[j] == 0 || has_lower_[j] == 0;
      const bool wrong =
          (d[j] > 0.0 && has_upper_[j] == 0) || (d[j] < 0.0 && has_lower_[j] == 0);
      if (wrong || (open && std::fabs(d[j]) <= tol::kIpmHsdCertificateRounding * d_largest)) {
        theta_x[j] = 1.0;
        target[j] = -d[j];
        any = any || wrong;
      }
    }
    if (!any) break;
    if (!normal_equations_lower(model_.matrix, theta_x, row_shift, delta_, &normal_lower_,
                                should_stop_) ||
        !ldl_.factorize(normal_lower_, delta_, should_stop_)) {
      return {};
    }
    std::vector<double> dy(static_cast<std::size_t>(m_));
    constraint_times(target, &dy);
    solve_normal(&dy);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += dy[i];
    if (!std::all_of(y.begin(), y.end(), [](double v) { return std::isfinite(v); })) return {};
  }
  zero_wrong_signs(&y);
  return y;
}

bool Homogeneous::try_certificates(Outcome* out) {
  // Both rays are read off the iterate as it stands (scale is irrelevant to a ray), offered
  // rounded and then as they came, either sign for the Farkas vector, and kept only when the
  // checker accepts one. b'y + l'z_l - u'z_u > 0 and c'x < 0 are the embedding's own signs
  // that the iterate leans to that side; the checker is the guarantee.
  //
  // THE RAYS CLEANED. The checker allows a free column no rounding in A'y beyond 1e-11 of its
  // own entries, and the iterate's A'y carries the dual residual, which shrinks only with mu.
  // The correction dy minimizing ||Theta^(1/2) (Abar'dy - r)|| for r = -(Abar'y + z_l - z_u),
  // solved by M = Abar Theta Abar' with the current factors, removes it where Theta is large
  // (free and interior columns, Theta up to 1 / rho) and leaves it where a bound's multiplier
  // can absorb it: ipm.cpp's dual purification, applied to the ray. Its primal mirror,
  // dx = Theta Abar' M^-1 (-Abar x), takes the row residual out of the primal ray. Each is
  // one more candidate; the checker still decides.
  const auto T = static_cast<std::size_t>(total_);
  std::string why;
  if (dual_ray_ > 0.0 && m_ > 0) {
    std::vector<double> cleaned;
    if (analyzed_) {
      std::vector<double> aty(T), weighted(T, 0.0);
      constraint_transpose_times(y_, &aty);
      for (std::size_t u = 0; u < T; ++u) {
        if (fixed_[u] == 0) weighted[u] = -theta_[u] * (aty[u] + zl_[u] - zu_[u]);
      }
      std::vector<double> dy(static_cast<std::size_t>(m_));
      constraint_times(weighted, &dy);
      solve_normal(&dy);
      cleaned = y_;
      for (std::size_t i = 0; i < cleaned.size(); ++i) cleaned[i] += dy[i];
      if (!std::all_of(cleaned.begin(), cleaned.end(),
                       [](double v) { return std::isfinite(v); })) {
        cleaned.clear();
      }
    }
    // A multiplier of the wrong sign for its row (one leaning on a side the row does not
    // have) is the dual residual of z_l - z_u on that row's logical, not part of the ray: the
    // signed copy sets it to zero. The checker sees both.
    const auto signed_copy = [&](const std::vector<double>& y) {
      std::vector<double> kept(y);
      bool changed = false;
      for (Index i = 0; i < m_; ++i) {
        const auto u = static_cast<std::size_t>(i);
        if ((kept[u] > 0.0 && !is_finite_bound(model_.row_lower[u])) ||
            (kept[u] < 0.0 && !is_finite_bound(model_.row_upper[u]))) {
          kept[u] = 0.0;
          changed = true;
        }
      }
      if (!changed) kept.clear();
      return kept;
    };
    const auto offer = [&](std::vector<double> y) {
      for (int flip = 0; flip < 2; ++flip) {
        if (flip == 1) {
          for (double& v : y) v = -v;
        }
        std::vector<double> kept = signed_copy(y);
        for (std::vector<double>* candidate : {&y, &kept}) {
          if (candidate->empty()) continue;
          if (!proves(true, *candidate, &why)) continue;
          out->proof = why;
          out->certificate = std::move(*candidate);
          out->verdict = Verdict::kInfeasible;
          return true;
        }
      }
      return false;
    };
    if (offer(rounded(y_)) || offer(y_)) return true;
    if (!cleaned.empty() && (offer(rounded(cleaned)) || offer(cleaned))) return true;
    // Deep in the ray side, the projection (project_farkas) too: it refactors, so only here.
    if (tau_ < kProjectAt * kappa_) {
      for (const double sign : {1.0, -1.0}) {
        std::vector<double> start = cleaned.empty() ? y_ : cleaned;
        for (double& v : start) v *= sign;
        std::vector<double> projected = project_farkas(std::move(start));
        if (!projected.empty() && (offer(rounded(projected)) || offer(std::move(projected)))) {
          return true;
        }
      }
    }
    logger_.verbose("hsd: no Farkas candidate checks out: {}", why);
  }
  if (!feasibility_only_ && cx_ < 0.0 && n_ > 0) {
    std::vector<double> cleaned;
    if (analyzed_ && m_ > 0) {
      std::vector<double> residual(static_cast<std::size_t>(m_)), dx(T);
      constraint_times(x_, &residual);
      for (double& v : residual) v = -v;
      solve_normal(&residual);
      constraint_transpose_times(residual, &dx);
      cleaned = x_;
      for (std::size_t u = 0; u < T; ++u) cleaned[u] += theta_[u] * dx[u];
      cleaned.resize(static_cast<std::size_t>(n_));
      if (!std::all_of(cleaned.begin(), cleaned.end(),
                       [](double v) { return std::isfinite(v); })) {
        cleaned.clear();
      }
    }
    const auto offer = [&](std::vector<double> ray) {
      for (Index j = 0; j < n_; ++j) {
        if (fixed_[static_cast<std::size_t>(j)] != 0) ray[static_cast<std::size_t>(j)] = 0.0;
      }
      if (!proves(false, ray, &why)) return false;
      out->proof = why;
      out->certificate = std::move(ray);
      out->verdict = Verdict::kUnbounded;
      return true;
    };
    std::vector<double> ray(x_.begin(), x_.begin() + n_);
    if (offer(rounded(ray)) || offer(std::move(ray))) return true;
    if (!cleaned.empty() && (offer(rounded(cleaned)) || offer(std::move(cleaned)))) return true;
    logger_.verbose("hsd: no ray candidate checks out: {}", why);
  }
  return false;
}

}  // namespace sankhya::ipm::hsd
