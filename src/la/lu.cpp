// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse LU with Markowitz pivoting. See lu.hpp for the references and for why
// DenseLu is kept alive as the oracle for this file.
//
// ============================================================================================
// THE TWO SOLVES, DERIVED. Read this before changing either of them.
// ============================================================================================
//
// The elimination applies, at step k, the transformation
//
//     M_k = I - sum_i mult_{i,k} e_i e_{r_k}^T
//
// to the active submatrix, where mult_{i,k} = a[i][c_k] / a[r_k][c_k]. After all m steps
//
//     M_{m-1} ... M_1 M_0 A = U                                                        (1)
//
// where U is upper triangular once its rows are read in the order r_0, r_1, ... and its
// columns in the order c_0, c_1, ...: pivot row r_k has entries only in columns c_k .. c_m-1.
// Inverting (1),
//
//     A = M_0^-1 M_1^-1 ... M_{m-1}^-1 U,      M_k^-1 = I + sum_i mult_{i,k} e_i e_{r_k}^T
//
// FTRAN, solve A x = b.
//   Left-multiply by M_{m-1} ... M_0 and use (1):   U x = (M_{m-1} ... M_0) b.
//   Applying M_k to a vector v is  v[i] -= mult_{i,k} * v[r_k]  for each recorded i, and the
//   factors must be applied in INCREASING k, the order they were produced.
//   Then back-substitute for x in DECREASING k:
//       x[c_k] = ( b[r_k] - sum_{l>k} U[r_k][c_l] * x[c_l] ) / a[r_k][c_k]
//
// BTRAN, solve A^T y = b.
//   A^T = U^T M_{m-1}^-T ... M_0^-T, so with z = (M_{m-1}^-T ... M_0^-T) y we need first
//       U^T z = b
//   U^T is lower triangular in the same orderings, so this is a forward substitution in
//   INCREASING k. Written in push form, which needs U by row exactly as it is stored:
//       z[r_k] = b[c_k] / a[r_k][c_k]
//       then for each (l > k, u) in row r_k:   b[c_l] -= u * z[r_k]
//   Then recover y from z = (M_{m-1}^-T ... M_0^-T) y, i.e. y = M_0^T M_1^T ... M_{m-1}^T z.
//   M_k^T v subtracts sum_i mult_{i,k} v[i] from component r_k alone, and the outermost
//   factor is M_0^T, so these are applied in DECREASING k - the reverse of FTRAN.
//
// The asymmetry is the whole point: FTRAN is L-then-U ascending-then-descending, BTRAN is
// U-then-L ascending-then-descending. Swapping either order produces a plausible vector.
// ============================================================================================

#include "lu.hpp"
#include "lu_workspace.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// Entries whose magnitude falls below this after an update are treated as exact
/// cancellation and removed. Keeping them would inflate the fill counts that drive the
/// refactorization trigger, and they carry no information.
constexpr double kDropTolerance = tol::kZeroDrop;

/// A basis update is rejected when its pivot element is small relative to the largest entry
/// of alpha. This is the update's counterpart to the Markowitz threshold in the
/// factorization: a tiny pivot divides through every subsequent solve and is how a product
/// form quietly loses accuracy over a few hundred iterations.
constexpr double kUpdatePivotThreshold = 1e-7;

/// A Forrest-Tomlin update is refused when its new diagonal is this small against the
/// largest term of the sum that produced it (#395; Tomlin 1972, the "loss of significance"
/// test): the digits the cancellation destroyed cannot be recovered by any later step.
constexpr double kFtCancellationThreshold = 1e-4;
/// ... and when the row eta it would file has an entry above this (#395): the eta is
/// applied on every later solve, and its size is the factor by which each solve's rounding
/// is amplified. Measured with kFtCancellationThreshold on a 200-row harness (six seeds,
/// 128 sparse column replacements each, drift against a fresh factorization): unguarded
/// 2.5e-3; the cancellation test alone 2.1e-5; both, 4.3e-8 against 4.2e-7 for the product
/// form on the same sequences, at 31 refusals in 768 updates.
constexpr double kFtRowEtaBound = 1e6;

/// Refactorize once the eta file reaches this many updates, whatever its size. Bounds the
/// worst-case drift by bounding how long any single factorization is trusted.
///
/// Measured (#68): 128 passes every correctness gate - 320 unit tests and the rational
/// oracle at 0 mismatches - while a sweep to 512 made d2q06c fail outright, so the ceiling
/// is real and this sits well inside it. The time-based break-even in the simplex normally
/// fires first; this is the backstop for a model whose solves are so cheap that it does not.
constexpr Index kMaxEtaCount = 128;

}  // namespace

// =========================================================================================
// Factorization
// =========================================================================================

bool SparseLu::factorize(const std::vector<LuColumn>& columns, Index m, double pivot_tolerance,
                         double markowitz_threshold, const ShouldStop& should_stop) {
  m_ = m;
  stopped_early_ = false;
  pivot_row_.clear();
  pivot_col_.clear();
  pivot_value_.clear();
  l_start_.assign(1, 0);
  l_rows_.clear();
  l_values_.clear();
  u_start_.assign(1, 0);
  u_steps_.clear();
  u_values_.clear();
  eta_start_.assign(1, 0);
  eta_rows_.clear();
  eta_values_.clear();
  eta_pivot_position_.clear();
  eta_pivot_value_.clear();
  base_nonzeros_ = 0;
  ft_active_ = false;
  ft_base_row_nonzeros_ = 0;
  ft_row_fill_ = 0;
  ft_step_of_position_.clear();
  ft_position_.clear();
  ft_step_at_.clear();
  ft_diag_.clear();
  ft_row_.clear();
  ft_col_.clear();
  ft_reta_pivot_step_.clear();
  ft_reta_start_.assign(1, 0);
  ft_reta_steps_.clear();
  ft_reta_values_.clear();
  work_.assign(static_cast<std::size_t>(m), 0.0);
  smallest_pivot_ = 0.0;
  largest_pivot_ = 0.0;
  dependent_positions_.clear();
  uncovered_rows_.clear();

  if (m == 0) return true;
  if (static_cast<Index>(columns.size()) != m) return false;

  Workspace w;
  w.init(m);
  w.fast = !reference_elimination_;

  // The row-wise value copy the singleton fast path reads (#463, lu_workspace.hpp). A row
  // named twice in one column is merged by the general update (the last value wins), which
  // the row copy cannot reproduce, so such an input takes the reference path throughout.
  std::vector<Index> last_column_of_row(static_cast<std::size_t>(m), -1);
  for (Index j = 0; j < m; ++j) {
    const LuColumn& column = columns[static_cast<std::size_t>(j)];
    const auto uj = static_cast<std::size_t>(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index row = column.rows[k];
      const double value = column.values[k];
      if (row < 0 || row >= m) return false;
      if (value == 0.0) continue;
      const auto ur = static_cast<std::size_t>(row);
      if (last_column_of_row[ur] == j) w.fast = false;
      last_column_of_row[ur] = j;
      w.col_rows[uj].push_back(row);
      w.col_values[uj].push_back(value);
      w.row_cols[ur].push_back(j);
      w.row_values[ur].push_back(value);
      ++w.col_count[uj];
      ++w.row_count[ur];
    }
  }

  if (!eliminate(w, pivot_tolerance, markowitz_threshold, should_stop)) return false;

  // U was recorded against column indices, because at the moment a pivot row is retired the
  // step at which each of its columns will itself be eliminated is not yet known. Translate
  // the whole array once, now that the permutation is complete.
  std::vector<Index> step_of_column(static_cast<std::size_t>(m), -1);
  for (Index k = 0; k < m; ++k) {
    step_of_column[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] = k;
  }
  for (Index& entry : u_steps_) {
    const Index step = step_of_column[static_cast<std::size_t>(entry)];
    if (step < 0) return false;  // a column that was never pivotal: structurally singular
    entry = step;
  }
  build_column_u();
  build_row_l();
  base_nonzeros_ = factor_nonzeros();
  return true;
}

void SparseLu::build_row_l() {
  // Transpose L from step-major (step k holds the rows its multipliers touch) into
  // row-major by STEP: for the row retired at step k, the earlier steps j < k that hold a
  // multiplier on it. A multiplier of step j lives on a row still active after step j, so
  // that row's own step is later than j and the map is well defined.
  const Index m = m_;
  std::vector<Index> step_of_row(static_cast<std::size_t>(m), -1);
  for (Index k = 0; k < m; ++k) {
    step_of_row[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] = k;
  }
  lr_start_.assign(static_cast<std::size_t>(m) + 1, 0);
  for (const Index row : l_rows_) {
    ++lr_start_[static_cast<std::size_t>(step_of_row[static_cast<std::size_t>(row)]) + 1];
  }
  for (Index k = 0; k < m; ++k) {
    lr_start_[static_cast<std::size_t>(k) + 1] += lr_start_[static_cast<std::size_t>(k)];
  }
  lr_steps_.assign(static_cast<std::size_t>(lr_start_[static_cast<std::size_t>(m)]), 0);
  lr_values_.assign(lr_steps_.size(), 0.0);
  std::vector<Index> fill(lr_start_.begin(), lr_start_.end() - 1);
  for (Index j = 0; j < m; ++j) {
    for (Index p = l_start_[static_cast<std::size_t>(j)];
         p < l_start_[static_cast<std::size_t>(j) + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index k = step_of_row[static_cast<std::size_t>(l_rows_[up])];
      const auto slot = static_cast<std::size_t>(fill[static_cast<std::size_t>(k)]++);
      lr_steps_[slot] = j;
      lr_values_[slot] = l_values_[up];
    }
  }
}

void SparseLu::build_column_u() {
  // Transpose the row-wise U (row-step k holds the steps j > k of its off-pivot entries)
  // into column form: column-step j holds the row-steps i < j that push into it.
  const Index m = m_;
  uc_start_.assign(static_cast<std::size_t>(m) + 1, 0);
  for (Index k = 0; k < m; ++k) {
    for (Index p = u_start_[static_cast<std::size_t>(k)];
         p < u_start_[static_cast<std::size_t>(k) + 1]; ++p) {
      const Index j = u_steps_[static_cast<std::size_t>(p)];
      if (j <= k) continue;
      ++uc_start_[static_cast<std::size_t>(j) + 1];
    }
  }
  for (Index j = 0; j < m; ++j) {
    uc_start_[static_cast<std::size_t>(j) + 1] += uc_start_[static_cast<std::size_t>(j)];
  }
  uc_steps_.assign(static_cast<std::size_t>(uc_start_[static_cast<std::size_t>(m)]), 0);
  uc_values_.assign(uc_steps_.size(), 0.0);
  std::vector<Index> fill(uc_start_.begin(), uc_start_.end() - 1);
  for (Index k = 0; k < m; ++k) {
    for (Index p = u_start_[static_cast<std::size_t>(k)];
         p < u_start_[static_cast<std::size_t>(k) + 1]; ++p) {
      const Index j = u_steps_[static_cast<std::size_t>(p)];
      if (j <= k) continue;
      const auto slot = static_cast<std::size_t>(fill[static_cast<std::size_t>(j)]++);
      uc_steps_[slot] = k;
      uc_values_[slot] = u_values_[static_cast<std::size_t>(p)];
    }
  }
}

// =========================================================================================
// Basis update
// =========================================================================================

bool SparseLu::update(Index leaving_position, const double* alpha) {
  if (m_ == 0) return false;
  if (leaving_position < 0 || leaving_position >= m_) return false;
  // The scheme the caller chose (use_forrest_tomlin) is honoured here, so the simplex has
  // one call and one set of counters whichever form is in play.
  if (forrest_tomlin_) return update_forrest_tomlin(leaving_position, alpha);
  // The symmetric guard to update_forrest_tomlin()'s own check: once Forrest-Tomlin mode
  // has folded any update into U, appending a product-form eta here would never be read by
  // either solve() path - U itself, not an outer eta file, is what represents those updates.
  if (ft_active_) return false;

  const auto pivot_index = static_cast<std::size_t>(leaving_position);
  const double pivot = alpha[pivot_index];

  // Reject rather than divide by something too small. The factorization is left untouched,
  // so the caller can refactorize and retry the same pivot on fresh factors.
  double largest = 0.0;
  for (Index i = 0; i < m_; ++i) {
    largest = std::max(largest, std::fabs(alpha[static_cast<std::size_t>(i)]));
  }
  if (!std::isfinite(pivot)) return false;
  if (std::fabs(pivot) < kUpdatePivotThreshold * std::max(1.0, largest)) return false;

  for (Index i = 0; i < m_; ++i) {
    const double value = alpha[static_cast<std::size_t>(i)];
    if (i == leaving_position || std::fabs(value) < kDropTolerance) continue;
    if (!std::isfinite(value)) return false;
    eta_rows_.push_back(i);
    eta_values_.push_back(value);
  }
  eta_start_.push_back(static_cast<Index>(eta_rows_.size()));
  eta_pivot_position_.push_back(leaving_position);
  eta_pivot_value_.push_back(pivot);
  return true;
}

bool SparseLu::should_refactorize() const noexcept {
  // Only the drift bound remains here. The fill-ratio rule that used to sit beside it was
  // replaced by the simplex's measured break-even (see primal_simplex.cpp): a fill ratio
  // assumes a fixed relationship between eta size and eta cost that no single constant
  // captured across instances.
  if (ft_active_) return static_cast<Index>(ft_reta_pivot_step_.size()) >= kMaxEtaCount;
  return eta_count() >= kMaxEtaCount;
}

// =========================================================================================
// Solves
// =========================================================================================

void SparseLu::forward_l(double* b) const {
  // Apply the elimination factors in increasing k. See the derivation at the top.
  for (Index k = 0; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const double pivot_component = b[static_cast<std::size_t>(pivot_row_[uk])];
    if (pivot_component == 0.0) continue;  // hyper-sparsity, in its cheapest form
    const Index begin = l_start_[uk];
    const Index end = l_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      b[static_cast<std::size_t>(l_rows_[up])] -= l_values_[up] * pivot_component;
    }
  }
}

void SparseLu::apply_etas(double* b) const {
  // Then the recorded updates, OLDEST FIRST: x = E_k^-1 ... E_1^-1 (B_0^-1 b). Applying
  // E^-1 is z_p = y_p / alpha_p followed by z_i = y_i - alpha_i z_p.
  const Index etas = eta_count();
  for (Index k = 0; k < etas; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const auto pivot_index = static_cast<std::size_t>(eta_pivot_position_[uk]);
    const double scaled = b[pivot_index] / eta_pivot_value_[uk];
    b[pivot_index] = scaled;
    if (scaled == 0.0) continue;
    const Index begin = eta_start_[uk];
    const Index end = eta_start_[uk + 1];
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      b[static_cast<std::size_t>(eta_rows_[ut])] -= eta_values_[ut] * scaled;
    }
  }
}

void SparseLu::solve(double* b) const {
  if (m_ == 0) return;
  forward_l(b);

  if (ft_active_) {
    // Same gather as below (row-permuted into step order - pivot_row_ never changes, so
    // this is already indexed by step). Then the row-eta file, oldest first, matching
    // apply_etas()'s order for the same reason: each was computed against the state the
    // ones before it left behind. Then the Forrest-Tomlin back-substitution instead of the
    // uc_ push, then the same scatter.
    for (Index k = 0; k < m_; ++k) {
      work_[static_cast<std::size_t>(k)] =
          b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])];
    }
    ft_apply_retas(work_.data());
    // No zero-fill: ft_back_substitute() visits every position exactly once and writes
    // ft_scratch_ at every one, so a prior call's contents are fully overwritten regardless.
    ft_back_substitute(work_.data(), ft_scratch_.data());
    for (Index s = 0; s < m_; ++s) {
      b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(s)])] =
          ft_scratch_[static_cast<std::size_t>(s)];
    }
    return;
  }

  // HYPER-SPARSE BACK-SUBSTITUTION (#68; Gilbert & Peierls 1988). U is walked by COLUMN in
  // decreasing step order, and a step whose result is exactly zero pushes nothing: on a
  // right-hand side with a handful of nonzeros - the entering column of a large sparse
  // basis - this touches a small fraction of U. The gather in solve_reference() reads all
  // of U regardless; the two must agree to rounding, and a test says so.
  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])];
  }
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = work_[uk] / pivot_value_[uk];
    work_[uk] = value;
    if (value == 0.0) continue;
    const Index begin = uc_start_[uk];
    const Index end = uc_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      work_[static_cast<std::size_t>(uc_steps_[up])] -= uc_values_[up] * value;
    }
  }
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] =
        work_[static_cast<std::size_t>(k)];
  }

  apply_etas(b);
}

void SparseLu::solve_reference(double* b) const {
  // Tests only, and it reads u_start_/uc_start_/pivot_value_ directly - none of which a
  // Forrest-Tomlin update ever touches, so this would silently answer for the ORIGINAL
  // basis rather than the updated one. solve() itself dispatches correctly; this does not.
  assert(!ft_active_ && "solve_reference() ignores update_forrest_tomlin(); use solve()");
  if (m_ == 0) return;
  forward_l(b);
  // Back-substitute in decreasing k. work_ holds x indexed by elimination step, so that the
  // inner loop can index U directly; it is scattered back to column order at the end.
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    double sum = b[static_cast<std::size_t>(pivot_row_[uk])];
    const Index begin = u_start_[uk];
    const Index end = u_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      if (step <= k) continue;  // the pivot entry itself
      sum -= u_values_[up] * work_[static_cast<std::size_t>(step)];
    }
    work_[uk] = sum / pivot_value_[uk];
  }

  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] =
        work_[static_cast<std::size_t>(k)];
  }

  apply_etas(b);
}

void SparseLu::apply_etas_transposed(double* b) const {
  // The updates come FIRST here and in the REVERSE order to FTRAN:
  // x = B_0^-T (E_1^-T ... E_k^-T b). Applying E^-T touches one component,
  // v_p <- (v_p - sum_{i != p} alpha_i v_i) / alpha_p, leaving the rest alone.
  for (Index k = eta_count() - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index begin = eta_start_[uk];
    const Index end = eta_start_[uk + 1];
    double accumulated = 0.0;
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      accumulated += eta_values_[ut] * b[static_cast<std::size_t>(eta_rows_[ut])];
    }
    const auto pivot_index = static_cast<std::size_t>(eta_pivot_position_[uk]);
    b[pivot_index] = (b[pivot_index] - accumulated) / eta_pivot_value_[uk];
  }
}

void SparseLu::forward_u_transposed() const {
  // Forward-substitute through U^T in increasing k, in push form, on work_ indexed by step.
  // A step whose result is exactly zero pushes nothing.
  std::vector<double>& z = work_;
  for (Index k = 0; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk] / pivot_value_[uk];
    z[uk] = value;
    if (value == 0.0) continue;
    const Index begin = u_start_[uk];
    const Index end = u_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      if (step <= k) continue;
      z[static_cast<std::size_t>(step)] -= u_values_[up] * value;
    }
  }
}

void SparseLu::apply_transposed_l() const {
  // HYPER-SPARSE TRANSPOSED ELIMINATION (#243; Gilbert & Peierls 1988), shared by both
  // update schemes since neither ever touches L. Apply M_k^T in DECREASING k to work_,
  // already indexed by step by the caller (forward_u_transposed(), or
  // ft_forward_substitute() + ft_apply_retas_transposed()). M_k^T subtracts, from the
  // component on pivot row r_k, the multipliers of step k times the components on the rows
  // they touch; every row a step-k multiplier touches is retired at a LATER step, so in
  // decreasing k the component on r_k is final the moment step k is reached and can be
  // pushed into every earlier step that holds a multiplier on r_k - which is exactly what
  // the row-wise L lists. A zero component pushes nothing, so a sparse rho costs the rows it
  // reaches rather than the whole of L, which the gather in solve_transpose_reference()
  // reads regardless.
  std::vector<double>& z = work_;
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk];
    if (value == 0.0) continue;
    const Index begin = lr_start_[uk];
    const Index end = lr_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      z[static_cast<std::size_t>(lr_steps_[up])] -= lr_values_[up] * value;
    }
  }
}

void SparseLu::solve_transpose(double* b) const {
  if (m_ == 0) return;

  if (ft_active_) {
    // Gather via pivot_col_ exactly as below, run the Forrest-Tomlin forward substitution in
    // place of forward_u_transposed(), then the row-eta file NEWEST first (the reverse of
    // FTRAN's order above, for the same reason apply_etas_transposed() reverses update()'s).
    for (Index k = 0; k < m_; ++k) {
      work_[static_cast<std::size_t>(k)] =
          b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
    }
    ft_forward_substitute(work_.data());
    ft_apply_retas_transposed(work_.data());
  } else {
    apply_etas_transposed(b);

    // work_ is indexed by step from here to the end: the right-hand side enters through the
    // pivot columns and the answer leaves through the pivot rows, and both triangular
    // passes run in step space in between, so nothing is scattered to row order and
    // gathered back.
    for (Index k = 0; k < m_; ++k) {
      work_[static_cast<std::size_t>(k)] =
          b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
    }
    forward_u_transposed();
  }

  // The L^T pass (lr_, untouched by either update scheme) and the final scatter are shared.
  apply_transposed_l();
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] =
        work_[static_cast<std::size_t>(k)];
  }
}

void SparseLu::solve_transpose_reference(double* b) const {
  // Tests only - see the note on solve_reference() above; the same staleness applies here.
  assert(!ft_active_ &&
         "solve_transpose_reference() ignores update_forrest_tomlin(); use solve_transpose()");
  if (m_ == 0) return;
  apply_etas_transposed(b);

  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
  }
  forward_u_transposed();

  // z is indexed by step and belongs on the pivot ROWS; place it there before applying the
  // transposed elimination factors, which are indexed by row.
  std::vector<double>& z = work_;
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] =
        z[static_cast<std::size_t>(k)];
  }

  // Apply M_k^T in DECREASING k - the reverse of FTRAN, and the half of this file most
  // likely to be "corrected" into agreement with solve() by someone who has not read the
  // derivation above. This is the gather form: every entry of L is read whatever the
  // density of the result.
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index begin = l_start_[uk];
    const Index end = l_start_[uk + 1];
    double accumulated = 0.0;
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      accumulated += l_values_[up] * b[static_cast<std::size_t>(l_rows_[up])];
    }
    b[static_cast<std::size_t>(pivot_row_[uk])] -= accumulated;
  }
}

// =========================================================================================
// Basis update (Forrest-Tomlin) - see lu.hpp for the derivation.
// =========================================================================================

Index SparseLu::ft_extra_nonzeros() const noexcept {
  return ft_row_fill_ - ft_base_row_nonzeros_ + static_cast<Index>(ft_reta_steps_.size());
}

void SparseLu::ft_init() {
  const Index m = m_;
  ft_step_of_position_.assign(static_cast<std::size_t>(m), -1);
  for (Index k = 0; k < m; ++k) {
    ft_step_of_position_[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] = k;
  }
  ft_position_.assign(static_cast<std::size_t>(m), 0);
  ft_step_at_.assign(static_cast<std::size_t>(m), 0);
  for (Index k = 0; k < m; ++k) {
    ft_position_[static_cast<std::size_t>(k)] = k;
    ft_step_at_[static_cast<std::size_t>(k)] = k;
  }
  ft_diag_.assign(pivot_value_.begin(), pivot_value_.end());

  // Copy U's off-diagonal entries out of the frozen row/column-wise arrays factorize() built
  // (u_start_/u_steps_/u_values_, and their column-wise mirror uc_ - both already keyed by
  // step) into the mutable STEP-keyed lists an update folds fill into. U itself is left
  // exactly as factorize() produced it; only these copies ever change.
  ft_row_.assign(static_cast<std::size_t>(m), {});
  ft_col_.assign(static_cast<std::size_t>(m), {});
  for (Index k = 0; k < m; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    for (Index p = u_start_[uk]; p < u_start_[uk + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      const double value = u_values_[up];
      ft_row_[uk].emplace_back(step, value);
      ft_col_[static_cast<std::size_t>(step)].emplace_back(k, value);
    }
  }
  ft_base_row_nonzeros_ = static_cast<Index>(u_steps_.size());
  ft_row_fill_ = ft_base_row_nonzeros_;
  ft_scratch_.assign(static_cast<std::size_t>(m), 0.0);
  ft_btran_scratch_.assign(static_cast<std::size_t>(m), 0.0);
  ft_btran_touched_.clear();
  ft_spike_.assign(static_cast<std::size_t>(m), 0.0);
  ft_spike_marked_.assign(static_cast<std::size_t>(m), false);
  ft_spike_touched_.clear();
  ft_work_nz_.clear();
  ft_reta_pivot_step_.clear();
  ft_reta_start_.assign(1, 0);
  ft_reta_steps_.clear();
  ft_reta_values_.clear();
  ft_active_ = true;
}

void SparseLu::ft_set(Index owner_step, Index referenced_step, double value) {
  auto& r = ft_row_[static_cast<std::size_t>(owner_step)];
  bool found = false;
  for (auto& entry : r) {
    if (entry.first == referenced_step) {
      entry.second = value;
      found = true;
      break;
    }
  }
  if (!found) {
    r.emplace_back(referenced_step, value);
    ++ft_row_fill_;
  }

  auto& c = ft_col_[static_cast<std::size_t>(referenced_step)];
  found = false;
  for (auto& entry : c) {
    if (entry.first == owner_step) {
      entry.second = value;
      found = true;
      break;
    }
  }
  if (!found) c.emplace_back(owner_step, value);
}

void SparseLu::ft_erase(Index owner_step, Index referenced_step) {
  auto& r = ft_row_[static_cast<std::size_t>(owner_step)];
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (r[i].first == referenced_step) {
      r[i] = r.back();
      r.pop_back();
      --ft_row_fill_;
      break;
    }
  }
  auto& c = ft_col_[static_cast<std::size_t>(referenced_step)];
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i].first == owner_step) {
      c[i] = c.back();
      c.pop_back();
      break;
    }
  }
}

void SparseLu::ft_btran_unit(Index step) const {
  // Clear exactly the entries the PREVIOUS call left nonzero, in O(prior nnz) rather than
  // O(m): a push out of this loop only ever reaches a position >= its source's own position
  // (lu.hpp's own invariant - a step's row only references positions at or after it), and
  // this loop starts at step's own position, so nothing before it is ever written below -
  // whatever a fill(0, m) used to zero there was never read either way. ft_btran_scratch_ is
  // touched ONLY here and in update_forrest_tomlin() reading this call's own output, unlike
  // ft_scratch_ (see lu.hpp) - the incremental clear below is only valid because nothing else
  // writes into this buffer between two calls.
  for (Index touched : ft_btran_touched_) {
    ft_btran_scratch_[static_cast<std::size_t>(touched)] = 0.0;
  }
  ft_btran_touched_.clear();

  // e~^T = e_step^T U_current^-1: forward substitution seeded with a single 1, in increasing
  // POSITION order (ft_step_at_ says which step that position currently holds), pushing each
  // resolved step's value into the later steps its own row references. Positions before
  // step's own start at 0 and push nothing - not merely usually zero, but zero BY THE
  // INVARIANT above - so the loop starts there directly rather than discovering it m times
  // over (issue #279's follow-up: "the partial BTRAN started at the leaving position").
  const auto us0 = static_cast<std::size_t>(step);
  ft_btran_scratch_[us0] = 1.0;
  for (Index k = ft_position_[us0]; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index occupant = ft_step_at_[uk];
    const auto us = static_cast<std::size_t>(occupant);
    const double value = ft_btran_scratch_[us] / ft_diag_[us];
    ft_btran_scratch_[us] = value;
    if (value == 0.0) continue;
    ft_btran_touched_.push_back(occupant);
    for (const auto& [other_step, coefficient] : ft_row_[us]) {
      ft_btran_scratch_[static_cast<std::size_t>(other_step)] -= coefficient * value;
    }
  }
}

void SparseLu::ft_apply_retas(double* residual_by_step) const {
  // FTRAN: R_1^-1 ... R_k^-1, oldest first - each R^-1 = I - e_p r^T touches only component
  // p, subtracting r's dot product with the whole vector (Forrest & Tomlin 1972, eq. 8-9).
  const Index count = static_cast<Index>(ft_reta_pivot_step_.size());
  for (Index k = 0; k < count; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index begin = ft_reta_start_[uk];
    const Index end = ft_reta_start_[uk + 1];
    double dot = 0.0;
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      dot +=
          ft_reta_values_[ut] * residual_by_step[static_cast<std::size_t>(ft_reta_steps_[ut])];
    }
    residual_by_step[static_cast<std::size_t>(ft_reta_pivot_step_[uk])] -= dot;
  }
}

void SparseLu::ft_apply_retas_transposed(double* z_by_step) const {
  // BTRAN: R_k^-T ... R_1^-T, newest first - R^-T = I - r e_p^T scatters component p's value
  // into every entry r touches (the transpose of the dot-product-into-one-component above).
  const Index count = static_cast<Index>(ft_reta_pivot_step_.size());
  for (Index k = count - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const double pivot_value = z_by_step[static_cast<std::size_t>(ft_reta_pivot_step_[uk])];
    if (pivot_value == 0.0) continue;
    const Index begin = ft_reta_start_[uk];
    const Index end = ft_reta_start_[uk + 1];
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      z_by_step[static_cast<std::size_t>(ft_reta_steps_[ut])] -=
          ft_reta_values_[ut] * pivot_value;
    }
  }
}

void SparseLu::ft_back_substitute(double* residual_by_step, double* solution_by_step) const {
  // FTRAN's back-substitution, hyper-sparse: decreasing position, pushing each resolved
  // step's value into the earlier-position steps that reference it (ft_col_), exactly as the
  // frozen uc_ push does when no update has touched U yet - both arrays are step-indexed
  // throughout; ft_step_at_ only ever decides WHICH step position k resolves.
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index step = ft_step_at_[uk];
    const auto us = static_cast<std::size_t>(step);
    const double value = residual_by_step[us] / ft_diag_[us];
    solution_by_step[us] = value;
    if (value == 0.0) continue;
    for (const auto& [other_step, coefficient] : ft_col_[us]) {
      residual_by_step[static_cast<std::size_t>(other_step)] -= coefficient * value;
    }
  }
}

void SparseLu::ft_forward_substitute(double* z_by_step) const {
  // BTRAN's forward substitution, hyper-sparse: increasing position, pushing each resolved
  // step's value into the later-position steps its own row references (ft_row_), the mirror
  // image of the back-substitution above and of forward_u_transposed() before any update.
  for (Index k = 0; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index step = ft_step_at_[uk];
    const auto us = static_cast<std::size_t>(step);
    const double value = z_by_step[us] / ft_diag_[us];
    z_by_step[us] = value;
    if (value == 0.0) continue;
    for (const auto& [other_step, coefficient] : ft_row_[us]) {
      z_by_step[static_cast<std::size_t>(other_step)] -= coefficient * value;
    }
  }
}

bool SparseLu::update_forrest_tomlin(Index leaving_position, const double* alpha) {
  if (m_ == 0) return false;
  if (leaving_position < 0 || leaving_position >= m_) return false;
  if (!ft_active_) {
    // Mixing the two update schemes on one factorization is a caller bug, not a case to
    // support: an eta already on file was never folded into U, so U alone no longer
    // represents the basis and folding into it now would be silently wrong.
    if (eta_count() > 0) return false;
    ft_init();
  }

  const Index m = m_;
  const auto up = static_cast<std::size_t>(leaving_position);
  const Index s0 = ft_step_of_position_[up];
  const Index q = ft_position_[static_cast<std::size_t>(s0)];

  // THE PIVOT IS ALPHA'S ENTRY AT THE LEAVING POSITION, HERE AS IN THE PRODUCT FORM (#395).
  // det(B_new)/det(B) = alpha_p, and the Forrest-Tomlin fold realises that same ratio as
  // new_diagonal / old_diagonal(s0): the test on the new diagonal alone lets a pivot through
  // whenever the old diagonal is large. The product form's own rule, on the same alpha, is
  // applied first so both schemes refuse the same pivots.
  double largest_alpha = 0.0;
  for (Index i = 0; i < m; ++i) {
    largest_alpha = std::max(largest_alpha, std::fabs(alpha[static_cast<std::size_t>(i)]));
  }
  const double alpha_p = alpha[up];
  if (!std::isfinite(alpha_p) || !std::isfinite(largest_alpha)) return false;
  if (std::fabs(alpha_p) < kUpdatePivotThreshold * std::max(1.0, largest_alpha)) return false;

  // Recover the column FTRAN would see just after L, i.e. a~ = L^-1 P a (Huangfu & Hall,
  // eq. 11 via #243's alpha), from the fully solved `alpha` the caller already has:
  // work_by_step[s] = alpha[pivot_col_[s]] is what back-substitution through the CURRENT U
  // produced from a~, so multiplying that same U forward through it recovers a~ again - the
  // identity U * work_by_step == a~, read the other way. This loop is the one place this
  // update genuinely must touch every position: `alpha` is a plain dense array with no
  // sparsity pattern of its own, so finding its nonzeros costs one linear scan regardless.
  // It costs nothing extra to collect that pattern into ft_work_nz_ while scanning, though -
  // issue #279's follow-up ("a sparse spike... walk ft_row_/ft_col_ from the nonzeros of
  // alpha") is what everything from here on reads instead of a second 0..m sweep.
  ft_work_nz_.clear();
  for (Index s = 0; s < m; ++s) {
    const auto us = static_cast<std::size_t>(s);
    const double value = alpha[static_cast<std::size_t>(pivot_col_[us])];
    work_[us] = value;
    if (value != 0.0) ft_work_nz_.push_back(s);
  }
  // ft_spike_ is member scratch, sized once by ft_init(), for the same reason work_ is: this
  // runs on the pivot path several hundred times a second. Cleared here from whatever the
  // PREVIOUS call left touched (O(prior nnz)), then rebuilt as a genuine sparse
  // matrix-vector product: spike = U * work, computed by walking work's nonzero columns
  // (ft_work_nz_) and pushing each one's contribution into every row that references it
  // (ft_col_) plus its own diagonal term - the column-wise dual of the row-wise gather this
  // replaces, and mathematically identical to it: a row untouched by this push has no
  // nonzero term in the original sum either, since every one of its entries multiplies some
  // work_[other_step] that this loop has already established is exactly 0.
  std::vector<double>& spike = ft_spike_;
  for (Index touched : ft_spike_touched_) {
    const auto ut = static_cast<std::size_t>(touched);
    spike[ut] = 0.0;
    ft_spike_marked_[ut] = false;
  }
  ft_spike_touched_.clear();
  const auto spike_add = [&](Index index, double delta) {
    const auto ui = static_cast<std::size_t>(index);
    if (!ft_spike_marked_[ui]) {
      ft_spike_marked_[ui] = true;
      spike[ui] = delta;
      ft_spike_touched_.push_back(index);
    } else {
      spike[ui] += delta;
    }
  };
  for (Index s : ft_work_nz_) {
    const auto us = static_cast<std::size_t>(s);
    const double w = work_[us];
    spike_add(s, ft_diag_[us] * w);
    for (const auto& [row, coefficient] : ft_col_[us]) spike_add(row, coefficient * w);
  }

  // The row eta: r^T = u-bar_p^T U^-1, where u-bar_p is s0's OWN off-diagonal row (about to
  // be eliminated, since s0 is moving to the last position and nothing can validly reference
  // a row that isn't there any more). Forrest & Tomlin's shortcut (eq. 12): computing the
  // partial BTRAN of a unit vector at s0 gives the same r after scaling by -diag(s0), since
  // u-bar_p^T = e_p^T U - diag(s0) e_p^T, and e_p^T U U^-1 cancels the first term. r is never
  // materialised as its own m-sized array below (issue #279's follow-up: "a sparse r") -
  // ft_btran_touched_ IS r's support (s0 itself excluded, always exactly 0 by
  // construction), and ft_btran_scratch_[step] scaled by -old_diagonal_s0 is its value, so
  // both read sites just walk that list.
  const double old_diagonal_s0 = ft_diag_[static_cast<std::size_t>(s0)];
  ft_btran_unit(s0);

  // Applying R^-1 to the spike modifies only its s0 entry (R^-1 = I - e_s0 r^T): the new
  // diagonal, once s0 reaches the last position, is spike[s0] minus r's dot product with the
  // whole spike. Every OTHER entry of the spike is installed unchanged below - R^-1 does not
  // touch them, which is the entire reason this costs one small row eta and not a rewrite of
  // every step the spike touches. Both sums below are over the TOUCHED lists rather than
  // 0..m: an entry missing from ft_spike_touched_ is exactly 0 by the clear above, and one
  // missing from ft_btran_touched_ is exactly 0 by ft_btran_unit's own contract.
  double largest = 0.0;
  for (Index touched : ft_spike_touched_) {
    largest = std::max(largest, std::fabs(spike[static_cast<std::size_t>(touched)]));
  }
  double dot = 0.0;
  double largest_term = std::fabs(spike[static_cast<std::size_t>(s0)]);
  double largest_r = 0.0;
  for (Index step : ft_btran_touched_) {
    if (step == s0) continue;
    const double r_value = -old_diagonal_s0 * ft_btran_scratch_[static_cast<std::size_t>(step)];
    const double term = r_value * spike[static_cast<std::size_t>(step)];
    largest_term = std::max(largest_term, std::fabs(term));
    largest_r = std::max(largest_r, std::fabs(r_value));
    dot += term;
  }
  const double new_diagonal = spike[static_cast<std::size_t>(s0)] - dot;
  // CANCELLATION IS THE INSTABILITY, NOT THE PIVOT'S SIZE (#395). The new diagonal is a
  // sum whose terms r_j * spike_j can be six orders larger than the result: on a 200-row
  // random basis with sparse entering columns the row eta reached 2e5 by the 17th update and
  // 6e6 by the 91st, the terms 1.5e5 and 1.5e6 against new diagonals of 1e-2 and 0.35, and
  // the folded factors drifted from a fresh factorization by 1e-5 and then 3e-2 while the
  // product form on the same sequence stayed at 6e-8. Tomlin (1972) makes this the
  // stability test of the method: when the pivot is small against the terms that produced
  // it, the digits are gone, and the only safe answer is a fresh factorization. The update
  // is refused - the simplex refactorizes and retries the same pivot on fresh factors.
  if (std::fabs(new_diagonal) < kFtCancellationThreshold * largest_term) return false;
  // THE ROW ETA'S SIZE IS THE OTHER HALF OF THE SAME TEST. Every later FTRAN subtracts
  // r^T y from one entry and every BTRAN adds r times one entry to the rest: entries of 1e7
  // in r turn the rounding of a solve into an error of 1e-9 on an answer of 1, and the next
  // fold copies that error into U for good. On the six-seed harness the cancellation test
  // alone left the drift at 2e-5 and this bound brought it to 4e-8, the product form's own
  // level, for 31 refusals in 768 updates.
  if (largest_r > kFtRowEtaBound) return false;
  // Both rejections happen here, before the first write below: a caller that reads false
  // may keep using this instance exactly as it would after update() returning false.
  if (!std::isfinite(new_diagonal)) return false;
  if (std::fabs(new_diagonal) < kUpdatePivotThreshold * std::max(1.0, largest)) return false;

  // Eliminate s0's own old row (u-bar_p): it is invalid the moment s0 stops being the last
  // position's occupant, and R above is exactly what accounts for it - nothing else needs to
  // change to compensate, which is the point of computing r from it rather than rewriting
  // every step downstream by hand.
  const auto old_row = ft_row_[static_cast<std::size_t>(s0)];
  for (const auto& [other_step, value] : old_row) ft_erase(s0, other_step);

  // Replace s0's old column (other steps' existing references to it): stale regardless of
  // value, since column s0 is being replaced outright, not perturbed.
  const auto old_column = ft_col_[static_cast<std::size_t>(s0)];
  for (const auto& [other_step, value] : old_column) ft_erase(other_step, s0);

  // Install the new column: every OTHER step whose row now references s0, unchanged from
  // the spike (R^-1 never touched these). Valid regardless of position, once s0 is last. A
  // step outside ft_spike_touched_ has spike 0, so it would never pass the threshold below
  // anyway - walking the touched list instead of 0..m changes nothing this loop installs.
  for (Index step : ft_spike_touched_) {
    if (step == s0) continue;
    const double value = spike[static_cast<std::size_t>(step)];
    if (std::fabs(value) >= kDropTolerance) ft_set(step, s0, value);
  }
  ft_diag_[static_cast<std::size_t>(s0)] = new_diagonal;

  // Move s0 to the last position, shifting every step between its old and new position down
  // by one to keep the permutation a bijection. Every OTHER step's own row/column entries -
  // untouched above - stay exactly as valid as they were: a uniform shift preserves every
  // existing "referenced step's position >= referencing step's position" relationship (see
  // lu.hpp), so this is pure relabelling, no arithmetic.
  if (q < m - 1) {
    for (Index c = q; c <= m - 2; ++c) {
      const Index moving = ft_step_at_[static_cast<std::size_t>(c) + 1];
      ft_step_at_[static_cast<std::size_t>(c)] = moving;
      ft_position_[static_cast<std::size_t>(moving)] = c;
    }
    ft_step_at_[static_cast<std::size_t>(m - 1)] = s0;
    ft_position_[static_cast<std::size_t>(s0)] = m - 1;
  }

  // File the row eta: R is applied between L and U on every later solve (ft_apply_retas() /
  // ft_apply_retas_transposed()), not folded into U - see lu.hpp for why U alone cannot
  // represent it. Walking ft_btran_touched_ (r's own support, s0 already excluded above)
  // rather than 0..m is the "sparse r" this file's other reads of it already rely on.
  ft_reta_pivot_step_.push_back(s0);
  for (Index step : ft_btran_touched_) {
    if (step == s0) continue;
    const double value = -old_diagonal_s0 * ft_btran_scratch_[static_cast<std::size_t>(step)];
    if (std::fabs(value) >= kDropTolerance) {
      ft_reta_steps_.push_back(step);
      ft_reta_values_.push_back(value);
    }
  }
  ft_reta_start_.push_back(static_cast<Index>(ft_reta_steps_.size()));

  return true;
}

}  // namespace sankhya
