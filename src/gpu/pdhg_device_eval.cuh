// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the PDHG convergence evaluation on the device (#478 item 3): device half.
//
// Included only by CUDA translation units. The design, the scaling identities and the
// references are in pdhg_device_eval.hpp.
//
// One DeviceEvaluator per card. It holds the UNSCALED problem data the residuals are measured
// against (for its rows, and for every column when it owns the column side), the scale
// factors, the restart reference and the best point so far, all on the card. The engine
// takes the two products for each point it evaluates, into ax() and into aty() or its own
// A^T y buffer, and the evaluator reduces; finish() then brings kMaxPoints * kSumsPerPoint
// doubles to pinned host memory, which is the only traffic an evaluation causes.
//
// Points: 0 is the current iterate, 1 the running average. Every call enqueues on the stream
// given, which must be the stream the engine's products run on.
#pragma once

#include <cuda_runtime.h>

#include "pdhg_device_eval.hpp"

namespace sankhya::gpu::eval {

class DeviceEvaluator {
 public:
  DeviceEvaluator() = default;
  DeviceEvaluator(const DeviceEvaluator&) = delete;
  DeviceEvaluator& operator=(const DeviceEvaluator&) = delete;
  ~DeviceEvaluator();

  /// Upload rows [row_begin, row_end) of the unscaled problem, and every column when
  /// `columns`; x_start is the scaled starting point, the first restart reference. The
  /// device must be current. False on any CUDA failure.
  [[nodiscard]] bool init(const pdhg::Problem& problem, const Scaling& scaling, int row_begin,
                          int row_end, bool columns, const double* x_start_device);

  /// Scratch for the engine's products: A x of the point (this card's rows) and A^T y of it
  /// (every column; allocated only with the column side).
  [[nodiscard]] double* ax() const { return ax_; }
  [[nodiscard]] double* aty() const { return aty_; }
  /// The average, x_sum / count and y_sum / count (scaled), written by average().
  [[nodiscard]] double* x_average() const { return x_avg_; }
  [[nodiscard]] double* y_average() const { return y_avg_; }

  /// x_average = x_sum / count on every column, y_average = y_sum / count on this card's
  /// rows: the same division the host path makes.
  [[nodiscard]] bool average(cudaStream_t s, const double* x_sum, const double* y_sum,
                             double count);
  /// Row-side sums of `point` from y (this card's rows) and A x in `ax`. The restart
  /// distance is formed only when with_restart (point 0).
  [[nodiscard]] bool rows(cudaStream_t s, int point, const double* y, const double* ax,
                          bool with_restart);
  /// Column-side sums of `point` from x and A^T y in `aty`. Column side only.
  [[nodiscard]] bool columns(cudaStream_t s, int point, const double* x, const double* aty,
                             bool with_restart);
  /// Fixed-order totals of `points` points, then their copy to host_sums(); the caller
  /// synchronises `s` before reading it. Points not evaluated read as zero.
  [[nodiscard]] bool finish(cudaStream_t s, int points);
  [[nodiscard]] const double* host_sums() const { return host_sums_; }

  /// Copy a point into the restart reference, or into the best point. x is copied only on
  /// the column side, y always (this card's rows).
  [[nodiscard]] bool set_restart(cudaStream_t s, const double* x, const double* y);
  [[nodiscard]] bool set_best(cudaStream_t s, const double* x, const double* y);
  /// Synchronous download of the best point (scaled): n values when the column side is
  /// here (else x is left alone), and this card's rows into y.
  [[nodiscard]] bool download_best(cudaStream_t s, double* x, double* y) const;

  [[nodiscard]] bool has_columns() const { return columns_; }

 private:
  bool copy_point(cudaStream_t s, double* x_dst, double* y_dst, const double* x,
                  const double* y) const;

  int n_ = 0, m_ = 0;  // m_: this card's rows
  int bn_ = 0, bm_ = 0;
  bool columns_ = false;
  // Rows (this card's).
  double *row_scale_{}, *row_lo_{}, *row_hi_{};
  double *ax_{}, *y_avg_{}, *y_restart_{}, *y_best_{};
  // Columns: problem data only with the column side; x_avg on every card (each card takes
  // A_k x of the average for its own rows).
  double *col_scale_{}, *cost_{}, *col_lo_{}, *col_hi_{};
  double *aty_{}, *x_avg_{}, *x_restart_{}, *x_best_{};
  // Block partials, kMaxPoints * (kRowSums * bm + kColumnSums * bn); totals; pinned copy.
  double* partials_{};
  double* sums_{};
  double* host_sums_{};
  int device_ = -1;
};

}  // namespace sankhya::gpu::eval
