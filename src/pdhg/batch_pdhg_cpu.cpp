// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the batched PDHG's CPU reference backend (#520). See batch_pdhg_backend.hpp.
//
// Every floating-point step below is mirrored, in the same order, by the CUDA kernels in
// src/gpu/pdhg_batch.cu; change one and the other must change with it, or the test that
// holds the two to identical duals fails. This file is compiled with -ffp-contract=off (see
// CMakeLists.txt) so that a*b + c is two rounded operations here as it is on the device.
//
// The iteration is Chambolle & Pock (2011), Algorithm 1, on the saddle point
//
//     min_{l <= x <= u} max_y   c'x - y'Ax + sum_i min_{rl_i <= t <= ru_i} y_i t
//
// whose y is the multiplier in the sign convention of core/safe_bound.hpp (y_i > 0 prices the
// row's lower side). With v = y - sigma A xbar, the dual step's prox is, by Moreau's identity,
//
//     y+ = v - proj_[-sigma ru, -sigma rl](v)
//
// which is max(0, v + sigma rl) on a >= row, min(0, v + sigma ru) on a <= row and
// v + sigma b on an equality. The primal step is the projected gradient step
// x+ = proj_[l,u](x - tau (c - A'y)) and xbar = 2 x+ - x the extrapolation.

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "batch_pdhg_backend.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {

Index batch_chunks(Index size) {
  return (size + tol::kBatchPdhgChunk - 1) / tol::kBatchPdhgChunk;
}

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

inline double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

class CpuBatchBackend final : public BatchBackend {
 public:
  explicit CpuBatchBackend(const BatchData& data)
      : d_(data),
        k_(static_cast<std::size_t>(data.count)),
        x_(data.x0),
        y_(data.y0),
        x_sum_(data.x0.size(), 0.0),
        y_sum_(data.y0.size(), 0.0),
        x_restart_(data.x0),
        y_restart_(data.y0),
        xbar_(data.x0.size(), 0.0),
        count_(k_, 0.0) {}

  void iterate(Count iterations, const std::vector<double>& tau,
               const std::vector<double>& sigma) override {
    const auto n = static_cast<std::size_t>(d_.cols);
    const auto m = static_cast<std::size_t>(d_.rows);
    for (Count t = 0; t < iterations; ++t) {
      for (std::size_t j = 0; j < n; ++j) {
        const auto begin = static_cast<std::size_t>(d_.col_start[j]);
        const auto end = static_cast<std::size_t>(d_.col_start[j + 1]);
        for (std::size_t k = 0; k < k_; ++k) {
          double aty = 0.0;
          for (std::size_t p = begin; p < end; ++p) {
            aty =
                aty + d_.col_value[p] * y_[static_cast<std::size_t>(d_.col_index[p]) * k_ + k];
          }
          const std::size_t at = j * k_ + k;
          const double g = d_.cost[j] - aty;
          const double v = x_[at] - tau[k] * g;
          const double next = clamp(v, d_.col_lower[at], d_.col_upper[at]);
          xbar_[at] = 2.0 * next - x_[at];
          x_[at] = next;
          x_sum_[at] = x_sum_[at] + next;
        }
      }
      for (std::size_t i = 0; i < m; ++i) {
        const auto begin = static_cast<std::size_t>(d_.row_start[i]);
        const auto end = static_cast<std::size_t>(d_.row_start[i + 1]);
        const double rl = d_.row_lower[i];
        const double ru = d_.row_upper[i];
        for (std::size_t k = 0; k < k_; ++k) {
          double ax = 0.0;
          for (std::size_t p = begin; p < end; ++p) {
            ax = ax +
                 d_.row_value[p] * xbar_[static_cast<std::size_t>(d_.row_index[p]) * k_ + k];
          }
          const std::size_t at = i * k_ + k;
          const double v = y_[at] - sigma[k] * ax;
          const double lo = std::isfinite(ru) ? -(sigma[k] * ru) : -kInf;
          const double hi = std::isfinite(rl) ? -(sigma[k] * rl) : kInf;
          const double next = v - clamp(v, lo, hi);
          y_[at] = next;
          y_sum_[at] = y_sum_[at] + next;
        }
      }
      for (double& c : count_) c = c + 1.0;
    }
  }

  void measure(BatchPartials* partials) override {
    const Index row_chunks = batch_chunks(d_.rows);
    const Index col_chunks = batch_chunks(d_.cols);
    partials->rows.assign(static_cast<std::size_t>(kBatchIterates * kBatchRowQuantities) *
                              static_cast<std::size_t>(row_chunks) * k_,
                          0.0);
    partials->cols.assign(static_cast<std::size_t>(kBatchIterates * kBatchColQuantities) *
                              static_cast<std::size_t>(col_chunks) * k_,
                          0.0);
    for (int s = 0; s < kBatchIterates; ++s) {
      for (Index c = 0; c < row_chunks; ++c) {
        for (std::size_t k = 0; k < k_; ++k) row_chunk(s, c, row_chunks, k, &partials->rows);
      }
      for (Index c = 0; c < col_chunks; ++c) {
        for (std::size_t k = 0; k < k_; ++k) col_chunk(s, c, col_chunks, k, &partials->cols);
      }
    }
  }

  void restart(const std::vector<BatchRestart>& action) override {
    for (std::size_t k = 0; k < k_; ++k) {
      if (action[k] == BatchRestart::kNone) continue;
      const bool to_average = action[k] == BatchRestart::kToAverage && count_[k] > 0.0;
      reset(&x_, &x_sum_, &x_restart_, d_.cols, k, to_average);
      reset(&y_, &y_sum_, &y_restart_, d_.rows, k, to_average);
      count_[k] = 0.0;
    }
  }

  void read_duals(std::vector<double>* current, std::vector<double>* average) override {
    *current = y_;
    average->resize(y_.size());
    for (std::size_t i = 0; i < static_cast<std::size_t>(d_.rows); ++i) {
      for (std::size_t k = 0; k < k_; ++k) (*average)[i * k_ + k] = mean(y_, y_sum_, i, k);
    }
  }

  [[nodiscard]] bool healthy() const override { return true; }

 private:
  /// The average of an iterate since the last restart, or the current value before any
  /// iteration has been summed.
  [[nodiscard]] double mean(const std::vector<double>& value, const std::vector<double>& sum,
                            std::size_t index, std::size_t k) const {
    const std::size_t at = index * k_ + k;
    return count_[k] > 0.0 ? sum[at] / count_[k] : value[at];
  }
  [[nodiscard]] double point(int s, const std::vector<double>& value,
                             const std::vector<double>& sum, std::size_t index,
                             std::size_t k) const {
    return s == 0 ? value[index * k_ + k] : mean(value, sum, index, k);
  }

  void reset(std::vector<double>* value, std::vector<double>* sum, std::vector<double>* anchor,
             Index size, std::size_t k, bool to_average) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(size); ++i) {
      const std::size_t at = i * k_ + k;
      if (to_average) (*value)[at] = (*sum)[at] / count_[k];
      (*anchor)[at] = (*value)[at];
      (*sum)[at] = 0.0;
    }
  }

  void row_chunk(int s, Index c, Index chunks, std::size_t k, std::vector<double>* out) const {
    double q[kBatchRowQuantities] = {0.0, 0.0, 0.0};
    const Index first = c * tol::kBatchPdhgChunk;
    const Index last =
        first + tol::kBatchPdhgChunk < d_.rows ? first + tol::kBatchPdhgChunk : d_.rows;
    for (Index row = first; row < last; ++row) {
      const auto i = static_cast<std::size_t>(row);
      double ax = 0.0;
      for (auto p = static_cast<std::size_t>(d_.row_start[i]);
           p < static_cast<std::size_t>(d_.row_start[i + 1]); ++p) {
        ax = ax + d_.row_value[p] *
                      point(s, x_, x_sum_, static_cast<std::size_t>(d_.row_index[p]), k);
      }
      const double rl = d_.row_lower[i];
      const double ru = d_.row_upper[i];
      const double r = ax < rl ? rl - ax : (ax > ru ? ax - ru : 0.0);
      q[0] = q[0] + r * r;
      const double y = point(s, y_, y_sum_, i, k);
      const double part = y > 0.0 ? (std::isfinite(rl) ? y * rl : 0.0)
                                  : (y < 0.0 ? (std::isfinite(ru) ? y * ru : 0.0) : 0.0);
      q[1] = q[1] + part;
      const double move = y - y_restart_[i * k_ + k];
      q[2] = q[2] + move * move;
    }
    for (int quantity = 0; quantity < kBatchRowQuantities; ++quantity) {
      const std::size_t at = ((static_cast<std::size_t>(s * kBatchRowQuantities + quantity) *
                                   static_cast<std::size_t>(chunks) +
                               static_cast<std::size_t>(c)) *
                              k_) +
                             k;
      (*out)[at] = q[quantity];
    }
  }

  void col_chunk(int s, Index c, Index chunks, std::size_t k, std::vector<double>* out) const {
    double q[kBatchColQuantities] = {0.0, 0.0, 0.0, 0.0};
    const Index first = c * tol::kBatchPdhgChunk;
    const Index last =
        first + tol::kBatchPdhgChunk < d_.cols ? first + tol::kBatchPdhgChunk : d_.cols;
    for (Index col = first; col < last; ++col) {
      const auto j = static_cast<std::size_t>(col);
      double aty = 0.0;
      for (auto p = static_cast<std::size_t>(d_.col_start[j]);
           p < static_cast<std::size_t>(d_.col_start[j + 1]); ++p) {
        aty = aty + d_.col_value[p] *
                        point(s, y_, y_sum_, static_cast<std::size_t>(d_.col_index[p]), k);
      }
      const std::size_t at = j * k_ + k;
      const double g = d_.cost[j] - aty;
      const double lb = d_.col_lower[at];
      const double ub = d_.col_upper[at];
      const bool lf = std::isfinite(lb);
      const bool uf = std::isfinite(ub);
      const double residual =
          lf && uf ? 0.0 : (lf ? (g < 0.0 ? g : 0.0) : (uf ? (g > 0.0 ? g : 0.0) : g));
      q[0] = q[0] + residual * residual;
      const double x = point(s, x_, x_sum_, j, k);
      q[1] = q[1] + d_.cost[j] * x;
      const double part = g > 0.0 ? (lf ? g * lb : 0.0) : (g < 0.0 ? (uf ? g * ub : 0.0) : 0.0);
      q[2] = q[2] + part;
      const double move = x - x_restart_[at];
      q[3] = q[3] + move * move;
    }
    for (int quantity = 0; quantity < kBatchColQuantities; ++quantity) {
      const std::size_t at = ((static_cast<std::size_t>(s * kBatchColQuantities + quantity) *
                                   static_cast<std::size_t>(chunks) +
                               static_cast<std::size_t>(c)) *
                              k_) +
                             k;
      (*out)[at] = q[quantity];
    }
  }

  const BatchData& d_;
  std::size_t k_;
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> x_sum_;
  std::vector<double> y_sum_;
  std::vector<double> x_restart_;
  std::vector<double> y_restart_;
  std::vector<double> xbar_;
  std::vector<double> count_;  ///< iterations summed since LP k's last restart
};

}  // namespace

std::unique_ptr<BatchBackend> make_cpu_batch_backend(const BatchData& data) {
  return std::make_unique<CpuBatchBackend>(data);
}

}  // namespace sankhya::pdhg
