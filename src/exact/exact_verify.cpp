// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational verification of a reported optimal LP basis (#521). See
// exact_verify.hpp for the citations and what this deliberately does NOT do (repair a basis,
// only check the one in hand).
//
// EVERYTHING IN THIS FILE EITHER PROVES THE BASIS EXACTLY OR SAYS WHY IT DID NOT TRY. There
// is no tolerance anywhere below - exact arithmetic has no "close enough", so a bound
// violation of any size, or a nonzero reduced cost of the wrong sign of any size, is
// kFailed. An __int128 overflow is kDeclined, not kFailed: it says nothing about whether the
// double answer was right, only that this module could not check it. Reporting kFailed for
// an overflow would manufacture a false negative exactly as reporting kVerified past an
// overflow would manufacture a false positive; both are worse than declining.
//
// SCOPE, STATED ONCE. Plain LP only (a quadratic objective or an integer column declines
// outright - see verify_basis_exact). A dense m x m elimination, so a row-count cap keeps
// this from being attempted on a model where it would be correct but impractically slow.
// Every model coefficient is converted from the double the reader already produced via
// Rational::from_double - bit-exact to that double, not to whatever decimal a human wrote in
// the MPS file, which is the precision this feature can actually promise given the model
// object it receives has already been through one rounding step before this code ever runs.

#include "exact/exact_verify.hpp"

#include <optional>
#include <vector>

#include <fmt/format.h>

#include "exact/rational.hpp"
#include "sankhya/types.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;

/// A row cap for the dense O(m^3) elimination below. Chosen so a verification attempt stays
/// on the order of seconds rather than minutes: 300^3 is 27 million Rational multiply-adds,
/// each a handful of __int128 operations. Declining above this is an honest capacity limit,
/// not a correctness one - nothing above this size is silently wrong, it is simply not
/// attempted.
constexpr Index kMaxRowsForExactVerification = 300;

std::string to_fraction_string(const Rational& value) {
  return fmt::format("{}/{}", static_cast<long long>(value.numerator()),
                     static_cast<long long>(value.denominator()));
}

/// Solve `matrix * x = rhs` exactly, `matrix` given as m row-vectors of m entries (a dense
/// square matrix, mutated in place as the elimination workspace). Partial pivoting picks any
/// nonzero entry in the working column - correctness in exact arithmetic does not depend on
/// magnitude, only need-not-be-zero - preferring the first one found, which is the cheapest
/// to locate. Returns std::nullopt when every candidate pivot in some column is exactly
/// zero: the matrix is exactly singular, which for a basis matrix means the reported basis
/// is not actually a basis in exact arithmetic.
std::optional<std::vector<Rational>> solve_dense_exact(
    std::vector<std::vector<Rational>> matrix, std::vector<Rational> rhs) {
  const Sz m = rhs.size();
  std::vector<Sz> row_of(m);
  for (Sz i = 0; i < m; ++i) row_of[i] = i;

  for (Sz col = 0; col < m; ++col) {
    Sz pivot_row = m;
    for (Sz r = col; r < m; ++r) {
      if (!matrix[row_of[r]][col].is_zero()) {
        pivot_row = r;
        break;
      }
    }
    if (pivot_row == m) return std::nullopt;  // exactly singular
    std::swap(row_of[col], row_of[pivot_row]);

    const Sz pivot = row_of[col];
    const Rational pivot_value = matrix[pivot][col];
    for (Sz r = col + 1; r < m; ++r) {
      const Sz row = row_of[r];
      if (matrix[row][col].is_zero()) continue;
      const Rational factor = matrix[row][col] / pivot_value;
      for (Sz c = col; c < m; ++c) matrix[row][c] = matrix[row][c] - factor * matrix[pivot][c];
      rhs[row] = rhs[row] - factor * rhs[pivot];
    }
  }

  std::vector<Rational> x(m);
  for (Sz i = m; i-- > 0;) {
    const Sz row = row_of[i];
    Rational sum = rhs[row];
    for (Sz c = i + 1; c < m; ++c) sum = sum - matrix[row][c] * x[c];
    x[i] = sum / matrix[row][i];
  }
  return x;
}

/// The exact value a nonbasic column or row-slack holds at its reported bound. kFixed uses
/// the lower bound (equal to the upper by definition); kNonbasicFree is exactly zero, the
/// only value at which a free variable can be nonbasic without an unbounded reduced cost.
Rational nonbasic_value(BasisStatus status, double lower, double upper) {
  switch (status) {
    case BasisStatus::kAtLower:
    case BasisStatus::kFixed: return Rational::from_double(lower);
    case BasisStatus::kAtUpper: return Rational::from_double(upper);
    case BasisStatus::kNonbasicFree: return Rational(0);
    case BasisStatus::kBasic:
    case BasisStatus::kUnknown:
      break;  // unreachable: callers only ask this of nonbasic positions
  }
  return Rational(0);
}

}  // namespace

ExactResult verify_basis_exact(const Model& model, const Solution& solution) {
  ExactResult result;

  if (solution.status != SolveStatus::kOptimal) {
    result.message = fmt::format("status is {}, exact verification is defined at optimal only",
                                 to_string(solution.status));
    return result;
  }
  if (model.hessian.num_nonzeros() > 0) {
    result.message =
        "the objective is quadratic; exact verification covers the LP engines only";
    return result;
  }
  if (model.num_integer_columns() > 0) {
    result.message =
        "the model has integer columns; a basis proves an LP relaxation optimal, not a MILP";
    return result;
  }
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (static_cast<Index>(solution.col_status.size()) != n ||
      static_cast<Index>(solution.row_status.size()) != m) {
    result.message = "the engine produced no basis (the interior point and PDHG do not)";
    return result;
  }
  if (m > kMaxRowsForExactVerification) {
    result.message = fmt::format(
        "{} rows exceeds the {}-row cap for a dense exact elimination in reasonable time", m,
        kMaxRowsForExactVerification);
    return result;
  }

  const Sz sn = static_cast<Sz>(n);
  const Sz sm = static_cast<Sz>(m);

  // --- Reconstruct the basis layout - identical to src/simplex/ranging.cpp's, since both
  // read the same Solution contract (BasisStatus, one entry per column and per row). ---
  std::vector<Index> basis_col(sm, -1);    // basis_col[p]: 0..n-1 structural, n..n+m-1 = n+row
  std::vector<Index> pos_of(sn + sm, -1);  // pos_of[col]: basis position, -1 if nonbasic
  {
    Index p = 0;
    for (Index j = 0; j < n; ++j) {
      if (solution.col_status[static_cast<Sz>(j)] == BasisStatus::kBasic) {
        if (p < m) {
          basis_col[static_cast<Sz>(p)] = j;
          pos_of[static_cast<Sz>(j)] = p;
        }
        ++p;
      }
    }
    for (Index i = 0; i < m; ++i) {
      if (solution.row_status[static_cast<Sz>(i)] == BasisStatus::kBasic) {
        if (p < m) {
          basis_col[static_cast<Sz>(p)] = n + i;
          pos_of[static_cast<Sz>(n + i)] = p;
        }
        ++p;
      }
    }
    if (p != m) {
      result.message =
          fmt::format("the reported basis has {} basic variable(s) for {} row(s)", p, m);
      return result;
    }
  }

  try {
    const double sense = model.sense_multiplier();

    // --- Exact model data, converted once. ---
    std::vector<Rational> col_cost(sn), col_cost_min_sense(sn), col_lower_exact(sn),
        col_upper_exact(sn);
    for (Index j = 0; j < n; ++j) {
      const Sz sj = static_cast<Sz>(j);
      col_cost[sj] = Rational::from_double(model.col_cost[sj]);
      col_cost_min_sense[sj] = Rational::from_double(sense * model.col_cost[sj]);
    }

    // --- Basis matrix B (dense, m x m: row i, column p) and its transpose (built directly,
    // not derived, so a bug in one cannot hide in the other's transpose). ---
    std::vector<std::vector<Rational>> basis(sm, std::vector<Rational>(sm, Rational(0)));
    std::vector<std::vector<Rational>> basis_transpose(sm,
                                                       std::vector<Rational>(sm, Rational(0)));
    std::vector<Rational> basic_cost_min_sense(sm);
    for (Index p = 0; p < m; ++p) {
      const Sz sp = static_cast<Sz>(p);
      const Index col = basis_col[sp];
      if (col < n) {
        basic_cost_min_sense[sp] = col_cost_min_sense[static_cast<Sz>(col)];
        const ColumnView view = model.matrix.column(col);
        for (Index k = 0; k < view.size; ++k) {
          const Sz row = static_cast<Sz>(view.rows[k]);
          const Rational value = Rational::from_double(view.values[k]);
          basis[row][sp] = value;
          basis_transpose[sp][row] = value;
        }
      } else {
        const Sz row = static_cast<Sz>(col - n);
        basic_cost_min_sense[sp] = Rational(0);  // a row's own slack is never costed
        basis[row][sp] = Rational(-1);
        basis_transpose[sp][row] = Rational(-1);
      }
    }

    // --- Primal RHS: B x_B = -(sum over nonbasic columns of their exact column * their
    // exact bound value), from the homogeneous extended system [A | -I][x; s] = 0. ---
    std::vector<Rational> rhs(sm, Rational(0));
    for (Index j = 0; j < n; ++j) {
      if (pos_of[static_cast<Sz>(j)] != -1) continue;  // basic, contributes through B itself
      const Sz sj = static_cast<Sz>(j);
      const Rational value =
          nonbasic_value(solution.col_status[sj], model.col_lower[sj], model.col_upper[sj]);
      if (value.is_zero()) continue;
      const ColumnView view = model.matrix.column(j);
      for (Index k = 0; k < view.size; ++k) {
        const Sz row = static_cast<Sz>(view.rows[k]);
        rhs[row] = rhs[row] - Rational::from_double(view.values[k]) * value;
      }
    }
    for (Index i = 0; i < m; ++i) {
      if (pos_of[static_cast<Sz>(n + i)] != -1) continue;  // this row's slack is basic
      const Sz si = static_cast<Sz>(i);
      const Rational value =
          nonbasic_value(solution.row_status[si], model.row_lower[si], model.row_upper[si]);
      // logical column i is -e_i, so its contribution to row i is -(-1)*value = +value.
      rhs[si] = rhs[si] + value;
    }

    const auto x_basic = solve_dense_exact(basis, rhs);
    if (!x_basic.has_value()) {
      result.verdict = ExactVerdict::kFailed;
      result.message = "the reported basis is exactly singular - it is not a basis at all";
      return result;
    }

    // --- Exact primal feasibility: every basic value inside its bounds, no tolerance. ---
    for (Index p = 0; p < m; ++p) {
      const Sz sp = static_cast<Sz>(p);
      const Index col = basis_col[sp];
      const double lower = col < n ? model.col_lower[static_cast<Sz>(col)]
                                   : model.row_lower[static_cast<Sz>(col - n)];
      const double upper = col < n ? model.col_upper[static_cast<Sz>(col)]
                                   : model.row_upper[static_cast<Sz>(col - n)];
      const Rational& value = (*x_basic)[sp];
      if (lower > -kInfinity && value < Rational::from_double(lower)) {
        result.verdict = ExactVerdict::kFailed;
        result.message = fmt::format(
            "basic position {} (of {} at basis_col {}) is below its exact lower bound", p,
            col < n ? "column" : "row", col < n ? col : col - n);
        return result;
      }
      if (upper < kInfinity && value > Rational::from_double(upper)) {
        result.verdict = ExactVerdict::kFailed;
        result.message = fmt::format(
            "basic position {} (of {} at basis_col {}) is above its exact upper bound", p,
            col < n ? "column" : "row", col < n ? col : col - n);
        return result;
      }
    }

    // --- Exact dual: B^T y = c_B (minimize sense). ---
    const auto y = solve_dense_exact(basis_transpose, basic_cost_min_sense);
    if (!y.has_value()) {
      result.verdict = ExactVerdict::kFailed;
      result.message = "the reported basis's transpose is exactly singular";
      return result;
    }

    // --- Exact dual feasibility: every nonbasic reduced cost has the sign optimality
    // requires (minimize sense), exactly - kNonbasicFree needs exactly zero. ---
    for (Index j = 0; j < n; ++j) {
      const Sz sj = static_cast<Sz>(j);
      if (pos_of[sj] != -1) continue;  // basic: zero by construction of y
      Rational reduced_cost = col_cost_min_sense[sj];
      const ColumnView view = model.matrix.column(j);
      for (Index k = 0; k < view.size; ++k) {
        reduced_cost = reduced_cost - (*y)[static_cast<Sz>(view.rows[k])] *
                                          Rational::from_double(view.values[k]);
      }
      const BasisStatus status = solution.col_status[sj];
      const bool ok = status == BasisStatus::kFixed ||
                      (status == BasisStatus::kAtLower && reduced_cost.sign() >= 0) ||
                      (status == BasisStatus::kAtUpper && reduced_cost.sign() <= 0) ||
                      (status == BasisStatus::kNonbasicFree && reduced_cost.is_zero());
      if (!ok) {
        result.verdict = ExactVerdict::kFailed;
        result.message = fmt::format("column {} has the wrong-signed exact reduced cost for {}",
                                     j, to_string(status));
        return result;
      }
    }
    for (Index i = 0; i < m; ++i) {
      const Sz si = static_cast<Sz>(i);
      if (pos_of[static_cast<Sz>(n + i)] != -1) continue;  // basic
      const Rational reduced_cost = (*y)[si];  // logical column is -e_i, cost 0: 0-y*(-1)=y
      const BasisStatus status = solution.row_status[si];
      const bool ok = status == BasisStatus::kFixed ||
                      (status == BasisStatus::kAtLower && reduced_cost.sign() >= 0) ||
                      (status == BasisStatus::kAtUpper && reduced_cost.sign() <= 0) ||
                      (status == BasisStatus::kNonbasicFree && reduced_cost.is_zero());
      if (!ok) {
        result.verdict = ExactVerdict::kFailed;
        result.message = fmt::format(
            "row {}'s slack has the wrong-signed exact reduced cost "
            "for {}",
            i, to_string(status));
        return result;
      }
    }

    // --- Exact objective, in the model's OWN sense (not the internal minimize sense). ---
    Rational objective = Rational::from_double(model.objective_offset);
    std::vector<Rational> col_value_exact(sn);
    for (Index j = 0; j < n; ++j) {
      const Sz sj = static_cast<Sz>(j);
      col_value_exact[sj] = pos_of[sj] != -1
                                ? (*x_basic)[static_cast<Sz>(pos_of[sj])]
                                : nonbasic_value(solution.col_status[sj], model.col_lower[sj],
                                                 model.col_upper[sj]);
      objective = objective + col_cost[sj] * col_value_exact[sj];
    }

    result.verdict = ExactVerdict::kVerified;
    result.exact_objective = to_fraction_string(objective);
    result.exact_col_value.resize(sn);
    for (Index j = 0; j < n; ++j) {
      result.exact_col_value[static_cast<Sz>(j)] =
          to_fraction_string(col_value_exact[static_cast<Sz>(j)]);
    }
    return result;
  } catch (const RationalOverflow&) {
    result.verdict = ExactVerdict::kDeclined;
    result.message =
        "an intermediate value overflowed exact __int128 arithmetic; the model or basis is "
        "too large or too ill-scaled to verify this way";
    return result;
  }
}

}  // namespace sankhya::exact
