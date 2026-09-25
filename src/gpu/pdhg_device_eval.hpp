// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the PDHG convergence evaluation on the device (#478 item 3): host-side half.
//
// pdhg::evaluate (src/pdhg/pdhg_evaluate.cpp) is the reference: the residuals of the
// UNSCALED problem at a candidate (x, y). The CUDA engines used to copy x, y and both running
// sums off the card every 40 iterations and run it on the CPU, which at 4M rows took half
// the solve. Here the same quantities are formed on the card from the scaled iterate and the
// products the engine already knows how to take, and only the handful of sums below cross to
// the host.
//
// The scaled products give the unscaled ones by the diagonal scaling of src/la/scaling.hpp,
// Ahat = Dr A Dc, x = Dc xhat, y = Dr yhat:
//     A x   = Dr^-1 (Ahat xhat)          (the row activity)
//     A^T y = Dc^-1 (Ahat^T yhat)        (so d = c + Dc^-1 Ahat^T yhat)
// which is the same number as the host's A x in exact arithmetic and differs from it by
// rounding only, and is why a device evaluation agrees with the host one to rounding rather
// than to the bit (tests/unit/test_pdhg_device_evaluation.cpp holds it to a bound derived
// from the dot products' error, Higham 2002 section 3.1).
//
// Every sum is fixed-order: a CUB BlockReduce per block into its own slot, then one block per
// sum reading the slots in index order, as the step-rule scalars are summed
// (pdhg_reduce.cuh). With deterministic=true the products are CSR_ALG2 non-transpose products
// on A and on an explicit A^T, so an evaluation is the same bits every run.
//
// This header is host-only (no CUDA types) so the test can include it; the device half,
// the evaluator the engines drive, is pdhg_device_eval.cuh.
#pragma once

#include <vector>

#include "../la/scaling.hpp"
#include "../pdhg/pdhg_evaluate.hpp"

namespace sankhya::gpu::eval {

/// The row-side sums of one evaluated point over one card's rows.
struct RowSums {
  double primal_sq = 0.0;        ///< sum of squared row-bound violations of A x
  double support = 0.0;          ///< sum of y_i * (the row bound y_i prices against)
  double dual_sq = 0.0;          ///< sum of y_i^2 where no row bound can absorb y_i
  double complementarity = 0.0;  ///< max |y_i| * slack_i (a max, not a sum)
  double restart_sq = 0.0;       ///< ||y - y_restart||^2 in scaled space (current point only)
};

/// The column-side sums of one evaluated point.
struct ColumnSums {
  double dual_sq = 0.0;          ///< sum of d_j^2 where no column bound can absorb d_j
  double bound = 0.0;            ///< sum of d_j * (the column bound d_j prices against)
  double complementarity = 0.0;  ///< max |d_j| * slack_j
  double objective = 0.0;        ///< c^T x
  double restart_sq = 0.0;       ///< ||x - x_restart||^2 in scaled space (current point only)
};

inline constexpr int kRowSums = 5;
inline constexpr int kColumnSums = 5;
inline constexpr int kSumsPerPoint = kRowSums + kColumnSums;  ///< RowSums, then ColumnSums
inline constexpr int kMaxPoints = 2;                          ///< current and average

/// The sums of point `point` from a finished evaluation's kMaxPoints * kSumsPerPoint doubles.
[[nodiscard]] RowSums row_sums(const double* sums, int point);
[[nodiscard]] ColumnSums column_sums(const double* sums, int point);

/// Adds another card's row sums (the multi-GPU engine calls this in slot order, so the
/// cross-card total is fixed-order too); complementarity takes the max.
void accumulate(RowSums& into, const RowSums& part);

/// The residuals from the sums, by the formulas of pdhg::evaluate.
[[nodiscard]] pdhg::Residuals assemble(const pdhg::Problem& problem, const RowSums& rows,
                                       const ColumnSums& columns);

/// What the test seam returns.
struct TestEvaluation {
  pdhg::Residuals residuals;
  double restart_dx = 0.0;  ///< ||x_s - x_restart||, scaled
  double restart_dy = 0.0;  ///< ||y_s - y_restart||, scaled
};

/// Test seam (tests/unit/test_pdhg_device_evaluation.cpp): evaluate the scaled point
/// (x_s, y_s) on the current device, with the products taken as the deterministic single-GPU
/// engine takes them. With count > 0 the point is a running SUM and is averaged on the device
/// first (the average path, which has no restart distances). False when there is no device
/// or any CUDA call fails.
[[nodiscard]] bool evaluate_for_testing(const pdhg::Problem& problem, const Scaling& scaling,
                                        const std::vector<double>& x_s,
                                        const std::vector<double>& y_s,
                                        const std::vector<double>& x_restart,
                                        const std::vector<double>& y_restart, double count,
                                        TestEvaluation* out);

}  // namespace sankhya::gpu::eval
