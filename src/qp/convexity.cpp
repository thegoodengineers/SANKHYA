// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity test. See convexity.hpp for why a refusal is the right default, and for
// why there are two implementations of the same decision.

#include "convexity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "la/ldl.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::qp {
namespace {

/// The dense reference is O(n^2) memory and O(n^3) time, so it is only ever run on a matrix
/// small enough for both to be free. Production goes through the sparse test at every size;
/// this bound exists so the reference stays a reference.
constexpr Index kDenseReferenceLimit = 2000;

/// The tolerance of the decision, relative to max(1, largest |Q_ii|): Q is accepted when
/// Q + shift * I factorizes with every pivot positive, shift = kCurvatureTolerance * that
/// scale, so an accepted Q has no eigenvalue below -shift (#835). It is measured against the
/// largest diagonal entry rather than an absolute number because an indefinite direction in a
/// badly scaled Q would otherwise hide under a fixed tolerance.
///
/// 1e-12 is the threshold QPLIB itself counts a negative eigenvalue by (its NOBJQUADNEGEV
/// statistic, doc.html), and #835 needs both of its sides. Below it: QPLIB_10056's smallest
/// eigenvalue is -1.2e-14 against a largest diagonal of 6.3, rounding in a matrix of rank 31,
/// and the factorization's own rounding on it stays far below the shift (its smallest shifted
/// pivot equals the shift to three digits in every one of 12 orders tried). Above it: of the
/// 130 QPLIB instances listed as non-convex that are small enough to measure densely, the one
/// whose smallest eigenvalue is nearest zero is QPLIB_10046, -2.6e-11 against a largest
/// diagonal of 7.9 (-3.3e-12 relative), and it must stay refused. The old slack, 1e-10, used as
/// the shift, accepted it and QPLIB_10048 and QPLIB_10074 from the same family (measured).
constexpr double kCurvatureTolerance = 1e-12;

/// How many coordinates of a witness are tried as a short direction of their own (below).
constexpr std::size_t kSharpenCoordinates = 4;

/// "column 1 (BN)" when the model names its columns, "column 1" otherwise. A refusal is read
/// by the person who wrote the file, and they know the column by its name, not its index.
[[nodiscard]] std::string column_label(const Model& model, Index j) {
  const auto uj = static_cast<std::size_t>(j);
  if (uj < model.col_names.size() && !model.col_names[uj].empty()) {
    return fmt::format("column {} ({})", j, model.col_names[uj]);
  }
  return fmt::format("column {}", j);
}

/// Q in MINIMIZATION sense, as the lower triangle of a symmetric matrix.
///
/// The engines minimise sense * (c'x + 0.5 x'Qx), so the Hessian they actually see is
/// sense * Q. What has to be positive semidefinite is that, not Q. For a MAXIMIZATION model
/// the requirement is therefore that Q be NEGATIVE semidefinite - a concave objective - and
/// testing Q itself would reject every well posed concave maximisation while accepting the
/// convex ones, which are exactly the unbounded-above cases that must be refused.
///
/// Model documents the Hessian as lower-triangular and both the readers and the C API
/// normalise to (max, min); an entry that arrives above the diagonal anyway is folded onto
/// its mirror rather than ignored, because ignoring it would test a different matrix.
[[nodiscard]] SparseMatrix minimization_lower_triangle(const Model& model) {
  const Index n = model.num_cols();
  const double sense = model.sense_multiplier();
  SparseMatrix lower(n, n);
  lower.reserve(static_cast<std::size_t>(model.hessian.num_nonzeros()));
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      lower.add_entry(std::max(i, j), std::min(i, j), sense * column.values[k]);
    }
  }
  lower.finalize();
  return lower;
}

/// Q_ij of the symmetric matrix whose lower triangle is `lower`, for either order of i, j.
[[nodiscard]] double entry(const SparseMatrix& lower, Index i, Index j) {
  const ColumnView column = lower.column(std::min(i, j));
  const Index row = std::max(i, j);
  for (Index k = 0; k < column.size; ++k) {
    if (column.rows[k] == row) return column.values[k];
  }
  return 0.0;
}

/// x^T Q x, evaluated so that its sign can be trusted when it is small.
///
/// A witness is only worth reporting if x^T Q x < 0 is a fact about Q and not about the
/// arithmetic, and near the tolerance the two are close: QPLIB_10046's directions curve by
/// about -1e-11 against an |x|^T |Q| |x| near 8, where the standard bound for plain summation
/// of its 22,500 terms, m eps |x|^T |Q| |x|, is about 4e-11. So the form is computed with
/// error-free transformations (Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J.
/// Sci. Comput. 26 (2005), Algorithm Dot2): each product is split exactly into its rounded
/// value and its error with an fma (as primal_simplex.cpp's refinement does), each addition
/// with Knuth's TwoSum, and the errors are summed beside the value. The result is as accurate
/// as if computed in twice the working precision: |error| <= eps |value| + (2 m eps)^2 |x|^T
/// |Q| |x| for m terms, which bound() returns.
struct Form {
  double value = 0.0;
  double magnitude = 0.0;  ///< |x|^T |Q| |x|
  double terms = 0.0;

  [[nodiscard]] double bound() const {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double gamma = 2.0 * (terms + 1.0) * eps;
    return eps * std::fabs(value) + gamma * gamma * magnitude;
  }
};

[[nodiscard]] Form quadratic_form(const SparseMatrix& lower, const std::vector<double>& x) {
  Form form;
  double sum = 0.0;
  double errors = 0.0;
  for (Index j = 0; j < lower.num_cols(); ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    if (xj == 0.0) continue;
    const ColumnView column = lower.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const double xi = x[static_cast<std::size_t>(column.rows[k])];
      if (xi == 0.0) continue;
      // The stored entry stands for both halves of an off-diagonal pair; doubling is exact.
      const double weight = (column.rows[k] == j) ? 1.0 : 2.0;
      const double p1 = column.values[k] * xi;
      const double e1 = std::fma(column.values[k], xi, -p1);
      const double p2 = p1 * xj;
      const double e2 = std::fma(p1, xj, -p2);
      const double term = weight * p2;
      const double s = sum + term;  // TwoSum: sum + term == s + its error, exactly
      const double back = s - sum;
      errors += (sum - (s - back)) + (term - back);
      sum = s;
      errors += weight * (e2 + e1 * xj);  // e1 * xj is rounded, but it is eps^2 small
      form.magnitude += std::fabs(term);
      form.terms += 1.0;
    }
  }
  form.value = sum + errors;
  return form;
}

/// A direction of negative curvature, checked against Q itself.
struct Witness {
  std::vector<double> x;
  double curvature = 0.0;  ///< x^T Q x / x^T x
  std::string along;       ///< which columns, when the direction is short enough to name
  bool verified = false;   ///< x^T Q x < 0 by more than its own rounding error
};

/// From the direction the factorization produced, the most negative of: that direction, and
/// every one- and two-column direction among the column it failed at and the largest
/// coordinates of it.
///
/// The factorization's own direction is a valid certificate but can be a poor one: beside a
/// zero diagonal the shift is the pivot, the multipliers are 1/shift, and z^T Q z / z^T z comes
/// out near -shift although the eigenvalue is -1 ([[0, 1], [1, 0]] is exactly that). The
/// curvature that made the pivot fail lives on the few coordinates the multipliers blew up,
/// and on two of them a 2 x 2 principal submatrix gives the exact smallest eigenvalue in
/// closed form - a direction a reader can check by hand.
[[nodiscard]] Witness sharpen(const Model& model, const SparseMatrix& lower,
                              const std::vector<double>& z, Index column) {
  const auto n = static_cast<std::size_t>(lower.num_cols());
  // A multiplier can overflow when a pivot is positive but tiny; such a coordinate is still
  // one of the largest, so it ranks as infinite rather than as a NaN no ordering can hold.
  const auto size_of = [&](Index i) {
    const double v = std::fabs(z[static_cast<std::size_t>(i)]);
    return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
  };
  std::vector<Index> picks = {column};
  {
    std::vector<Index> order;
    for (std::size_t i = 0; i < n; ++i) {
      if (z[i] != 0.0 && static_cast<Index>(i) != column)
        order.push_back(static_cast<Index>(i));
    }
    const std::size_t keep = std::min(order.size(), kSharpenCoordinates - 1);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep),
                      order.end(), [&](Index a, Index b) { return size_of(a) > size_of(b); });
    picks.insert(picks.end(), order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep));
  }

  Witness best;
  best.x = z;
  double norm_squared = 0.0;
  for (const double v : z) norm_squared += v * v;
  best.curvature = norm_squared > 0.0 ? quadratic_form(lower, z).value / norm_squared : 0.0;
  if (!std::isfinite(best.curvature)) best.curvature = std::numeric_limits<double>::infinity();

  const auto consider = [&](double curvature, std::vector<std::pair<Index, double>> coords,
                            std::string along) {
    if (!(curvature < best.curvature)) return;
    best.curvature = curvature;
    best.x.assign(n, 0.0);
    for (const auto& [i, v] : coords) best.x[static_cast<std::size_t>(i)] = v;
    best.along = std::move(along);
  };
  for (std::size_t a = 0; a < picks.size(); ++a) {
    const Index i = picks[a];
    const double qii = entry(lower, i, i);
    consider(qii, {{i, 1.0}}, fmt::format(", along {} alone", column_label(model, i)));
    for (std::size_t b = a + 1; b < picks.size(); ++b) {
      const Index j = picks[b];
      const double qij = entry(lower, i, j);
      if (qij == 0.0) continue;  // decoupled: the one-column directions already cover it
      const double qjj = entry(lower, j, j);
      // The smaller eigenvalue of [[qii, qij], [qij, qjj]], eigenvector (qij, lambda - qii).
      const double lambda = 0.5 * (qii + qjj) - std::hypot(0.5 * (qii - qjj), qij);
      const double v1 = qij;
      const double v2 = lambda - qii;
      const double norm = std::hypot(v1, v2);
      consider(lambda, {{i, v1 / norm}, {j, v2 / norm}},
               fmt::format(", along {} and {} alone", column_label(model, i),
                           column_label(model, j)));
    }
  }

  // Reported and verified from the same evaluation, accurate to its stated bound.
  const Form form = quadratic_form(lower, best.x);
  double chosen_norm = 0.0;
  for (const double v : best.x) chosen_norm += v * v;
  best.verified = std::isfinite(form.value) && form.value + form.bound() < 0.0;
  if (best.verified) best.curvature = form.value / chosen_norm;
  return best;
}

/// The verdict and detail for a pivot of Q + shift * I that was not positive.
[[nodiscard]] ConvexityResult indefinite(const Model& model, const SparseMatrix& lower,
                                         const std::vector<double>& z, Index column,
                                         double pivot, double shift, const char* method) {
  ConvexityResult result;
  const Witness witness = sharpen(model, lower, z, column);
  if (!witness.verified) {
    result.detail = fmt::format(
        "{} of Q + {:.3g} I reached a pivot of {:.6g} at {} in minimization sense, but no "
        "direction it exhibits has x^T Q x below its own rounding error, so convexity is "
        "neither proved nor refuted at this tolerance",
        method, shift, pivot, column_label(model, column));
    return result;  // kUnverified
  }
  result.verdict = Convexity::kIndefinite;
  result.witness = witness.x;
  result.witness_curvature = witness.curvature;
  result.detail = fmt::format(
      "{} of Q + {:.3g} I reached a pivot of {:.6g} at {} in minimization sense; it exhibits a "
      "direction x with x^T Q x / x^T x = {:.6g}{}, in which the objective curves downward, so "
      "the model is non-convex",
      method, shift, pivot, column_label(model, column), witness.curvature, witness.along);
  return result;
}

}  // namespace

ConvexityResult check_convexity_dense(const Model& model) {
  ConvexityResult result;
  const Index n = model.num_cols();

  if (model.hessian.num_nonzeros() == 0) {
    result.verdict = Convexity::kConvex;
    result.detail = "the objective has no quadratic term";
    return result;
  }
  if (n > kDenseReferenceLimit) {
    result.detail = fmt::format(
        "{} columns exceeds the {} the dense reference test holds in "
        "memory; check_convexity() decides this size sparsely",
        n, kDenseReferenceLimit);
    return result;  // kUnverified
  }

  const SparseMatrix lower = minimization_lower_triangle(model);

  // The stored entries are Q itself (the 0.5 lives in the objective, not in the storage), so
  // an off-diagonal entry appears twice.
  const auto un = static_cast<std::size_t>(n);
  std::vector<double> q(un * un, 0.0);
  double largest_diagonal = 0.0;
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = lower.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      const double value = column.values[k];
      q[static_cast<std::size_t>(i) * un + static_cast<std::size_t>(j)] = value;
      q[static_cast<std::size_t>(j) * un + static_cast<std::size_t>(i)] = value;
      if (i == j) largest_diagonal = std::max(largest_diagonal, std::fabs(value));
    }
  }
  const double shift = kCurvatureTolerance * std::max(1.0, largest_diagonal);

  // LDL^T of Q + shift * I without row interchanges, every pivot required positive: the rule
  // check_semidefinite() applies sparsely, and why it is the right one is written there.
  std::vector<double> l(un * un, 0.0);
  std::vector<double> d(un, 0.0);
  for (Index j = 0; j < n; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    double pivot = q[uj * un + uj] + shift;
    for (Index k = 0; k < j; ++k) {
      const auto uk = static_cast<std::size_t>(k);
      pivot -= l[uj * un + uk] * l[uj * un + uk] * d[uk];
    }
    if (!(pivot > 0.0)) {
      // z = L^{-T} e_j over the first j + 1 columns: z^T (Q + shift I) z = pivot.
      std::vector<double> z(un, 0.0);
      z[uj] = 1.0;
      for (Index k = j - 1; k >= 0; --k) {
        const auto uk = static_cast<std::size_t>(k);
        double sum = 0.0;
        for (Index i = k + 1; i <= j; ++i) {
          const auto ui = static_cast<std::size_t>(i);
          sum += l[ui * un + uk] * z[ui];
        }
        z[uk] = -sum;
      }
      return indefinite(model, lower, z, j, pivot, shift, "LDL^T");
    }
    d[uj] = pivot;
    for (Index i = j + 1; i < n; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      double sum = q[ui * un + uj];
      for (Index k = 0; k < j; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        sum -= l[ui * un + uk] * l[uj * un + uk] * d[uk];
      }
      l[ui * un + uj] = sum / pivot;
    }
  }

  result.verdict = Convexity::kConvex;
  result.detail =
      fmt::format("LDL^T of Q + {:.3g} I completed with every pivot positive", shift);
  return result;
}

ConvexityResult check_convexity(const Model& model) {
  ConvexityResult result;

  if (model.hessian.num_nonzeros() == 0) {
    result.verdict = Convexity::kConvex;
    result.detail = "the objective has no quadratic term";
    return result;
  }

  const SparseMatrix lower = minimization_lower_triangle(model);
  SparseLdl ldl;
  const SemidefiniteReport report = ldl.check_semidefinite(lower, kCurvatureTolerance);
  switch (report.verdict) {
    case SemidefiniteReport::Verdict::kPositiveSemidefinite:
      result.verdict = Convexity::kConvex;
      result.detail = fmt::format(
          "sparse LDL^T of Q + {:.3g} I over {} nonzeros completed with every pivot positive",
          report.shift, lower.num_nonzeros());
      return result;
    case SemidefiniteReport::Verdict::kIndefinite:
      return indefinite(model, lower, report.witness, report.column, report.pivot, report.shift,
                        "sparse LDL^T");
    case SemidefiniteReport::Verdict::kUndecided: break;
  }

  result.detail = "the Hessian could not be ordered and factorized, so convexity is unproven";
  return result;  // kUnverified
}

}  // namespace sankhya::qp
