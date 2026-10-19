// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the Jordan algebra of the second-order cone (#755), the arithmetic the conic
// interior point's Nesterov-Todd scaling is written in.
//
// References
//   Alizadeh & Goldfarb, "Second-order cone programming", Mathematical Programming 95 (2003),
//     sections 4 and 6 - the Jordan product, the quadratic representation P(u), the spectral
//     decomposition that gives square roots and inverses, and the NT scaling point.
//   Nesterov & Todd, "Self-scaled barriers and interior-point methods for convex programming",
//     Mathematics of Operations Research 22 (1997) - the scaling itself.
//   Faraut & Koranyi, "Analysis on Symmetric Cones" (Oxford, 1994), ch. II - the identities
//     P(u)^{-1} = P(u^{-1}) and P(u^{1/2})^2 = P(u) used below.
//
// A vector x of the cone Q^k = { (x_0, x_1) : x_0 >= ||x_1|| } has
//     x o y     = (x'y, x_0 y_1 + y_0 x_1)            the Jordan product, identity e = (1, 0)
//     det(x)    = x_0^2 - ||x_1||^2                   the eigenvalues are x_0 +- ||x_1||
//     x^{-1}    = J x / det(x),  J = diag(1, -I)
//     x^{1/2}   = (r, x_1 / (2 r)),  r = sqrt((x_0 + sqrt(det x)) / 2)
//     P(u) v    = 2 u (u'v) - det(u) J v              so P(u) = 2 u u' - det(u) J
// Every function below takes raw pointers and a length so the cone blocks of one long
// vector can be addressed in place.
#pragma once

#include <cmath>
#include <limits>

#include "sankhya/types.hpp"

namespace sankhya::conic::jordan {

inline double dot(const double* a, const double* b, Index k) {
  double s = 0.0;
  for (Index i = 0; i < k; ++i) s += a[i] * b[i];
  return s;
}

inline double det(const double* x, Index k) {
  double tail = 0.0;
  for (Index i = 1; i < k; ++i) tail += x[i] * x[i];
  return x[0] * x[0] - tail;
}

/// out = x o y.
inline void product(const double* x, const double* y, Index k, double* out) {
  const double head = dot(x, y, k);
  for (Index i = 1; i < k; ++i) out[i] = x[0] * y[i] + y[0] * x[i];
  out[0] = head;
}

/// out = x^{-1}; the caller guarantees det(x) > 0.
inline void inverse(const double* x, Index k, double* out) {
  const double d = det(x, k);
  out[0] = x[0] / d;
  for (Index i = 1; i < k; ++i) out[i] = -x[i] / d;
}

/// out = x^{1/2}; the caller guarantees x in the interior of the cone.
inline void sqrt(const double* x, Index k, double* out) {
  const double r = std::sqrt(0.5 * (x[0] + std::sqrt(det(x, k))));
  out[0] = r;
  for (Index i = 1; i < k; ++i) out[i] = x[i] / (2.0 * r);
}

/// out = P(u) v = 2 u (u'v) - det(u) J v.
inline void quadratic(const double* u, const double* v, Index k, double* out) {
  const double uv = dot(u, v, k);
  const double d = det(u, k);
  out[0] = 2.0 * u[0] * uv - d * v[0];
  for (Index i = 1; i < k; ++i) out[i] = 2.0 * u[i] * uv + d * v[i];
}

/// Solve lambda o u = r for u: the arrow matrix L(lambda) = [lambda_0, lambda_1'; lambda_1,
/// lambda_0 I] inverted in closed form (Alizadeh & Goldfarb 2003, eq. 12). The caller
/// guarantees lambda in the interior of the cone, so det(lambda) > 0.
inline void arrow_solve(const double* lambda, const double* r, Index k, double* out) {
  const double d = det(lambda, k);
  const double l0 = lambda[0];
  double tail = 0.0;  // lambda_1' r_1
  for (Index i = 1; i < k; ++i) tail += lambda[i] * r[i];
  out[0] = (l0 * r[0] - tail) / d;
  // u_1 = (r_1 - lambda_1 u_0) / lambda_0
  for (Index i = 1; i < k; ++i) out[i] = (r[i] - lambda[i] * out[0]) / l0;
}

/// The largest alpha >= 0 with x + alpha dx in the cone, +inf when every alpha is. x is in the
/// cone's interior. The boundary is reached where det(x + alpha dx) = 0 with x_0 + alpha
/// dx_0 >= 0: a quadratic in alpha, det(x) + 2 alpha x'J dx + alpha^2 det(dx) = 0.
inline double max_step(const double* x, const double* dx, Index k) {
  const double a = det(dx, k);
  double xjdx = x[0] * dx[0];
  for (Index i = 1; i < k; ++i) xjdx -= x[i] * dx[i];
  const double b = 2.0 * xjdx;
  const double c = det(x, k);  // > 0
  const double inf = std::numeric_limits<double>::infinity();
  double alpha = inf;
  if (a < 0.0) {
    // One positive root: the step leaves the cone exactly once.
    alpha = (-b - std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
  } else if (a > 0.0) {
    const double disc = b * b - 4.0 * a * c;
    if (disc >= 0.0 && b < 0.0) alpha = (-b - std::sqrt(disc)) / (2.0 * a);
  } else if (b < 0.0) {
    alpha = -c / b;
  }
  // The root found is where det vanishes; it is the cone's boundary only if the head stays
  // non-negative there (det also vanishes on the mirror cone -x_0 >= ||x_1||).
  if (dx[0] < 0.0) alpha = std::fmin(alpha, -x[0] / dx[0]);
  return alpha < 0.0 ? 0.0 : alpha;
}

/// The Nesterov-Todd scaling point w of the pair (x, s), both interior: w = P(x^{1/2})
/// (P(x^{1/2}) s)^{-1/2}, the unique interior point with P(w) s = x (Alizadeh & Goldfarb
/// 2003, section 6; Nesterov & Todd 1997). `scratch` holds 3k doubles.
inline void nt_point(const double* x, const double* s, Index k, double* scratch, double* w) {
  double* half = scratch;    // x^{1/2}
  double* ps = scratch + k;  // P(x^{1/2}) s
  double* root = scratch + 2 * k;
  sqrt(x, k, half);
  quadratic(half, s, k, ps);
  inverse(ps, k, root);  // (P(x^{1/2}) s)^{-1}
  sqrt(root, k, ps);     // (P(x^{1/2}) s)^{-1/2}
  quadratic(half, ps, k, w);
}

}  // namespace sankhya::conic::jordan
