// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the multi-GPU PDHG's convergence evaluation on the cards (#478 item 3, #295).
// Design: pdhg_multi_gpu_eval.cuh. References: Applegate et al., PDLP, NeurIPS 2021,
// sections 3.3 and 4.3 (the KKT error evaluated); Lu & Yang, cuPDLP.jl, arXiv:2311.12180
// (evaluation on the device); Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.,
// SIAM 2003, ch. 11 (the row-block distributed products).

#include "pdhg_multi_gpu_eval.cuh"

#include <cmath>

namespace sankhya::gpu::multi {

template <typename Fn>
bool CardEvaluation::on_every_card(Fn&& fn) {
  for (DeviceState* c : cards_) {
    if (cudaSetDevice(c->device_id) != cudaSuccess) return false;
    if (!fn(*c)) return false;
  }
  return true;
}

bool CardEvaluation::synchronize() {
  return on_every_card(
      [](DeviceState& c) { return cudaStreamSynchronize(c.stream) == cudaSuccess; });
}

bool CardEvaluation::init(const std::vector<DeviceState*>& cards, PartialSumExchange* exchange,
                          const pdhg::Problem& problem, const Scaling& scaling) {
  cards_ = cards;
  exchange_ = exchange;
  problem_ = &problem;
  const bool ready = on_every_card([&](DeviceState& c) {
    evaluators_.push_back(std::make_unique<eval::DeviceEvaluator>());
    return evaluators_.back()->init(problem, scaling, c.row_start, c.row_start + c.local_m,
                                    c.slot == 0, c.d_x);
  });
  if (!ready) evaluators_.clear();
  return ready;
}

bool CardEvaluation::enqueue_point(int point, bool average) {
  const bool rows = on_every_card([&](DeviceState& c) {
    eval::DeviceEvaluator& ev = of(c);
    double* x = average ? ev.x_average() : c.d_x;
    const double* y = average ? ev.y_average() : c.d_y;
    return spmv_ax(c, x, ev.ax()) && ev.rows(c.stream, point, y, ev.ax(), !average) &&
           spmv_aty(c, y);
  });
  DeviceState& c0 = *cards_[0];
  if (!rows || !exchange_->enqueue() || cudaSetDevice(c0.device_id) != cudaSuccess)
    return false;
  eval::DeviceEvaluator& ev0 = of(c0);
  return ev0.columns(c0.stream, point, average ? ev0.x_average() : c0.d_x, c0.d_aty, !average);
}

bool CardEvaluation::evaluate(double average_count, pdhg::Residuals* current,
                              pdhg::Residuals* average, double* restart_dx,
                              double* restart_dy) {
  const bool with_average = average_count > 0.0;
  if (!enqueue_point(0, false)) return false;
  if (with_average) {
    // No card may overwrite its A_k^T y partial for the average while another card may still
    // be copying the current point's.
    if (!synchronize()) return false;
    if (!on_every_card([&](DeviceState& c) {
          return of(c).average(c.stream, c.d_xsum, c.d_ysum, average_count);
        }) ||
        !enqueue_point(1, true))
      return false;
  }
  const int points = with_average ? 2 : 1;
  if (!on_every_card([&](DeviceState& c) { return of(c).finish(c.stream, points); }) ||
      !synchronize())
    return false;
  eval::RowSums rows[eval::kMaxPoints];
  for (const DeviceState* c : cards_) {  // slot order: a fixed-order cross-card total
    for (int p = 0; p < points; ++p)
      eval::accumulate(rows[p], eval::row_sums(of(*c).host_sums(), p));
  }
  const double* sums0 = of(*cards_[0]).host_sums();
  const eval::ColumnSums columns0 = eval::column_sums(sums0, 0);
  *current = eval::assemble(*problem_, rows[0], columns0);
  *restart_dx = std::sqrt(columns0.restart_sq);
  *restart_dy = std::sqrt(rows[0].restart_sq);
  if (with_average) *average = eval::assemble(*problem_, rows[1], eval::column_sums(sums0, 1));
  return true;
}

bool CardEvaluation::copy(bool best, bool average) {
  return on_every_card([&](DeviceState& c) {
    eval::DeviceEvaluator& ev = of(c);
    const double* x = average ? ev.x_average() : c.d_x;
    const double* y = average ? ev.y_average() : c.d_y;
    return best ? ev.set_best(c.stream, x, y) : ev.set_restart(c.stream, x, y);
  });
}

bool CardEvaluation::keep_best(bool average) { return copy(true, average); }

bool CardEvaluation::set_restart() { return copy(false, false); }

bool CardEvaluation::download_best(std::vector<double>* x, std::vector<double>* y) {
  return on_every_card([&](DeviceState& c) {
    return of(c).download_best(c.stream, x->data(), y->data() + c.row_start);
  });
}

}  // namespace sankhya::gpu::multi
