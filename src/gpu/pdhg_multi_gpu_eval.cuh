// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the row-partitioned multi-GPU PDHG's convergence evaluation on the cards (#478
// item 3, #295).
//
// Included only by CUDA translation units.
//
// One eval::DeviceEvaluator (pdhg_device_eval.cuh) per card, over the card's own rows; card 0
// also owns the column side, since x is replicated bit-identical on every card. A point is
// evaluated by
//   every card:  A_k x -> the row-side sums of its rows;  A_k^T y_k -> its partial,
//   exchange:    A^T y = sum of the partials in slot order, on every card (the iteration's
//                own fixed-order exchange, multi_gpu_exchange.hpp), into d_aty,
//   card 0:      the column-side sums from x and A^T y,
// and each card's eval::kSumsPerPoint totals per point come to the host, where the row sums
// are added in slot order - fixed-order end to end, so a run stays bitwise repeatable and
// {0, 1} gives the bits {0, 0} gives. d_aty is rewritten at the start of every iteration, so
// the evaluation borrows it. The best point and the restart reference stay on the cards.
#pragma once

#include <memory>
#include <vector>

#include "multi_gpu_exchange.hpp"
#include "pdhg_device_eval.cuh"
#include "pdhg_multi_gpu_device.hpp"

namespace sankhya::gpu::multi {

class CardEvaluation {
 public:
  /// One evaluator per card, cards in slot order; false (and nothing held) on any failure.
  [[nodiscard]] bool init(const std::vector<DeviceState*>& cards, PartialSumExchange* exchange,
                          const pdhg::Problem& problem, const Scaling& scaling);

  /// Residuals of the current iterate and, with `average_count` > 0, of the running average
  /// (the sums over that many iterates), and the current iterate's scaled distances to the
  /// restart reference. Synchronises every card.
  [[nodiscard]] bool evaluate(double average_count, pdhg::Residuals* current,
                              pdhg::Residuals* average, double* restart_dx, double* restart_dy);
  /// The current iterate (or the average of the last evaluate()) becomes the best point.
  [[nodiscard]] bool keep_best(bool average);
  /// The current iterate becomes the restart reference.
  [[nodiscard]] bool set_restart();
  /// The best point, scaled: x (n) from card 0, y (m) assembled from every card's rows.
  [[nodiscard]] bool download_best(std::vector<double>* x, std::vector<double>* y);

 private:
  eval::DeviceEvaluator& of(const DeviceState& c) {
    return *evaluators_[static_cast<std::size_t>(c.slot)];
  }
  template <typename Fn>
  bool on_every_card(Fn&& fn);
  bool synchronize();
  bool enqueue_point(int point, bool average);
  bool copy(bool best, bool average);

  std::vector<DeviceState*> cards_;
  PartialSumExchange* exchange_ = nullptr;
  const pdhg::Problem* problem_ = nullptr;
  std::vector<std::unique_ptr<eval::DeviceEvaluator>> evaluators_;
};

}  // namespace sankhya::gpu::multi
