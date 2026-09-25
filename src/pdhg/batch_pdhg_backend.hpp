// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the arithmetic of the batched PDHG (#520), behind one interface with two
// implementations: the CPU reference (batch_pdhg_cpu.cpp) and the CUDA kernels
// (src/gpu/pdhg_batch.cu). Internal to src/pdhg and src/gpu.
//
// THE SPLIT. Everything that decides - step sizes, primal weights, when to restart and to
// what - is host code in batch_pdhg.cpp, run once for either backend. A backend only does
// the vector arithmetic: the iterations, the partial sums the decisions are made from, the
// restarts it is told to make. Both backends perform every product, sum and quotient in the
// same order with the same rounding (the device uses __dmul_rn / __dadd_rn / __dsub_rn /
// __ddiv_rn so nothing is contracted into a fused multiply-add; the CPU file is compiled
// with -ffp-contract=off), so the two produce the same iterates bit for bit, take the same
// restart decisions, and return the same duals. tests/unit/test_batch_pdhg.cpp holds them
// to that.
//
// LAYOUT. K LPs share the matrix; LP k's value of column j sits at j * K + k and of row i at
// i * K + k, so the K threads that work on one row or column read the same matrix entries
// and neighbouring vector entries (the n x K block of #661's design, stored row-major).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::pdhg {

/// The scaled problem the backends iterate on. Every array is the Ruiz / Pock-Chambolle
/// scaled one (la/scaling.hpp): Ahat = Dr A Dc, chat = Dc c, row bounds Dr r, boxes l / Dc.
struct BatchData {
  Index rows = 0;
  Index cols = 0;
  Index count = 0;  ///< K, the LPs in the batch
  /// Ahat row-wise (CSR): entries of row i at row_start[i] .. row_start[i+1], ascending column.
  std::vector<Index> row_start;
  std::vector<Index> row_index;
  std::vector<double> row_value;
  /// Ahat column-wise (CSC, i.e. Ahat^T row-wise): ascending row within a column.
  std::vector<Index> col_start;
  std::vector<Index> col_index;
  std::vector<double> col_value;
  std::vector<double> cost;       ///< cols
  std::vector<double> row_lower;  ///< rows
  std::vector<double> row_upper;  ///< rows
  std::vector<double> col_lower;  ///< cols * K, interleaved
  std::vector<double> col_upper;  ///< cols * K, interleaved
  std::vector<double> x0;         ///< cols * K, interleaved, inside the box
  std::vector<double> y0;         ///< rows * K, interleaved
};

/// Quantities the restart test needs, per iterate and LP, as partial sums over chunks of
/// kBatchPdhgChunk rows or columns. Row-side quantities, in this order:
///   0  sum of squared primal residuals (distance of (Ax)_i from [rl_i, ru_i])
///   1  the rows' part of the dual objective: y_i rl_i for y_i > 0, y_i ru_i for y_i < 0
///   2  squared distance of y from the last restart point
/// Column-side quantities, with g = c - A^T y:
///   0  sum of squared dual residuals (the part of g_j no finite bound of the box absorbs)
///   1  the primal objective c'x
///   2  the columns' part of the dual objective: g_j l_j for g_j > 0, g_j u_j for g_j < 0,
///      finite bounds only
///   3  squared distance of x from the last restart point
inline constexpr int kBatchRowQuantities = 3;
inline constexpr int kBatchColQuantities = 4;
/// Two iterates are measured: 0 the current one, 1 the average since the last restart.
inline constexpr int kBatchIterates = 2;

/// Partial sums, laid out [iterate][quantity][chunk][lp].
struct BatchPartials {
  std::vector<double> rows;
  std::vector<double> cols;
};

/// What restart() does with one LP.
enum class BatchRestart : signed char { kNone = 0, kToCurrent = 1, kToAverage = 2 };

class BatchBackend {
 public:
  virtual ~BatchBackend() = default;
  BatchBackend() = default;
  BatchBackend(const BatchBackend&) = delete;
  BatchBackend& operator=(const BatchBackend&) = delete;

  /// `iterations` PDHG iterations of every LP, LP k with primal step tau[k] and dual step
  /// sigma[k]. Each iteration adds the new point into the running sums.
  virtual void iterate(Count iterations, const std::vector<double>& tau,
                       const std::vector<double>& sigma) = 0;
  /// The restart statistics at the current point and at the average (the current point for
  /// an LP whose sums are empty).
  virtual void measure(BatchPartials* partials) = 0;
  /// Restart the LPs as told: the chosen point becomes the current one and the restart
  /// point, and the running sums are emptied. kNone leaves an LP alone.
  virtual void restart(const std::vector<BatchRestart>& action) = 0;
  /// The current and average duals, rows * K interleaved, in the scaled space.
  virtual void read_duals(std::vector<double>* current, std::vector<double>* average) = 0;
  /// False once a device call has failed; the caller then discards the whole run.
  [[nodiscard]] virtual bool healthy() const = 0;
};

/// Number of chunks covering `size` rows or columns.
[[nodiscard]] Index batch_chunks(Index size);

/// The CPU reference backend.
[[nodiscard]] std::unique_ptr<BatchBackend> make_cpu_batch_backend(const BatchData& data);

}  // namespace sankhya::pdhg
