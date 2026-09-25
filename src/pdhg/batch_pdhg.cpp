// SPDX-License-Identifier: Apache-2.0
// SANKHYA - batched PDHG (#520): the set-up, the decisions and the bounds. See batch_pdhg.hpp
// for the method and its references, batch_pdhg_backend.hpp for the arithmetic's split.

#include "batch_pdhg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "../core/safe_bound.hpp"
#include "../la/scaling.hpp"
#include "batch_pdhg_backend.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "../gpu/pdhg_batch.hpp"
#endif

namespace sankhya::pdhg {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kRuizIterations = 10;
constexpr int kPowerIterations = 30;
constexpr unsigned kPowerSeed = 520;

/// The scaled problem, laid out for the backends.
BatchData build_data(const BatchProblem& problem, const Scaling& scaling) {
  const Model& model = *problem.model;
  BatchData d;
  d.rows = model.num_rows();
  d.cols = model.num_cols();
  d.count = problem.count;
  const auto n = static_cast<std::size_t>(d.cols);
  const auto m = static_cast<std::size_t>(d.rows);
  const auto k_count = static_cast<std::size_t>(d.count);

  const CsrView by_row(scaling.matrix);
  d.row_start = by_row.row_starts();
  d.row_index = by_row.column_indices();
  d.row_value = by_row.values();
  d.col_start = scaling.matrix.column_starts();
  d.col_index = scaling.matrix.row_indices();
  d.col_value = scaling.matrix.values();
  d.cost = scaling.cost;
  d.row_lower = scaling.row_lower;
  d.row_upper = scaling.row_upper;

  d.col_lower.resize(n * k_count);
  d.col_upper.resize(n * k_count);
  d.x0.resize(n * k_count);
  for (std::size_t k = 0; k < k_count; ++k) {
    for (std::size_t j = 0; j < n; ++j) {
      const double dc = scaling.column[j];
      const double lo = problem.col_lower[k * n + j];
      const double hi = problem.col_upper[k * n + j];
      const double lo_hat = std::isfinite(lo) ? lo / dc : lo;
      const double hi_hat = std::isfinite(hi) ? hi / dc : hi;
      const double start = problem.primal_start.empty() ? 0.0 : problem.primal_start[j] / dc;
      d.col_lower[j * k_count + k] = lo_hat;
      d.col_upper[j * k_count + k] = hi_hat;
      d.x0[j * k_count + k] = std::min(std::max(start, lo_hat), hi_hat);
    }
  }
  // y = Dr yhat. A start of the wrong sign for a one-sided row is moved to zero: the dual
  // step would do the same on its first iteration.
  d.y0.assign(m * k_count, 0.0);
  for (std::size_t i = 0; i < m && i < problem.dual_start.size(); ++i) {
    double y = problem.dual_start[i] / scaling.row[i];
    if (!std::isfinite(d.row_lower[i])) y = std::min(y, 0.0);
    if (!std::isfinite(d.row_upper[i])) y = std::max(y, 0.0);
    if (!std::isfinite(y)) y = 0.0;
    for (std::size_t k = 0; k < k_count; ++k) d.y0[i * k_count + k] = y;
  }
  return d;
}

/// ||Ahat||_2 by power iteration on Ahat^T Ahat, rounded up by 1% as la/scaling.hpp's
/// estimate_spectral_norm() rounds: an underestimate would oversize the step. Written here,
/// single-threaded over the batch's own arrays, because that routine's A^T x is an OpenMP
/// region, and a caller that runs many small batches pays a thread-team start per product:
/// on a box with a 7-CPU quota and 64 visible cores, test_batch_pdhg.cpp's CPU sweep took
/// 689 s with that routine at the default thread count and 0.1 s with OMP_NUM_THREADS=1.
double spectral_norm(const BatchData& d) {
  const auto n = static_cast<std::size_t>(d.cols);
  const auto m = static_cast<std::size_t>(d.rows);
  if (n == 0 || m == 0 || d.col_value.empty()) return 1.0;
  std::mt19937 rng(kPowerSeed);
  std::uniform_real_distribution<double> spread(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& value : v) value = spread(rng);
  std::vector<double> av(m);
  double norm = 0.0;
  for (int step = 0; step < kPowerIterations; ++step) {
    double length = 0.0;
    for (const double value : v) length += value * value;
    length = std::sqrt(length);
    if (!(length > 0.0) || !std::isfinite(length)) return 1.0;
    for (double& value : v) value /= length;
    for (std::size_t i = 0; i < m; ++i) {
      double sum = 0.0;
      for (auto p = static_cast<std::size_t>(d.row_start[i]);
           p < static_cast<std::size_t>(d.row_start[i + 1]); ++p) {
        sum += d.row_value[p] * v[static_cast<std::size_t>(d.row_index[p])];
      }
      av[i] = sum;
    }
    double next = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (auto p = static_cast<std::size_t>(d.col_start[j]);
           p < static_cast<std::size_t>(d.col_start[j + 1]); ++p) {
        sum += d.col_value[p] * av[static_cast<std::size_t>(d.col_index[p])];
      }
      v[j] = sum;
      next += sum * sum;
    }
    norm = std::sqrt(std::sqrt(next));  // ||A^T A v||^(1/2) approaches ||A||_2
  }
  return norm > 0.0 ? norm * 1.01 : 1.0;
}

/// Initial primal weight, [PDLP] sec. 3.3: ||c|| / ||b|| over the finite row bounds, or 1
/// when either is zero.
double initial_primal_weight(const BatchData& d) {
  double cost = 0.0;
  for (const double c : d.cost) cost += c * c;
  double rhs = 0.0;
  for (std::size_t i = 0; i < d.row_lower.size(); ++i) {
    const double lo = std::isfinite(d.row_lower[i]) ? std::fabs(d.row_lower[i]) : 0.0;
    const double hi = std::isfinite(d.row_upper[i]) ? std::fabs(d.row_upper[i]) : 0.0;
    const double b = std::max(lo, hi);
    rhs += b * b;
  }
  cost = std::sqrt(cost);
  rhs = std::sqrt(rhs);
  if (cost > tol::kBatchPdhgMinMove && rhs > tol::kBatchPdhgMinMove) return cost / rhs;
  return 1.0;
}

/// One iterate's restart statistics for one LP, summed from the partials in chunk order.
struct Stats {
  double primal_residual = 0.0;  ///< squared
  double dual_residual = 0.0;    ///< squared
  double primal_objective = 0.0;
  double dual_objective = 0.0;
  double primal_move = 0.0;  ///< squared
  double dual_move = 0.0;    ///< squared
};

Stats sum_stats(const BatchPartials& p, int s, std::size_t k, std::size_t k_count,
                Index row_chunks, Index col_chunks) {
  const auto total = [&](const std::vector<double>& v, int quantities, int q, Index chunks) {
    double sum = 0.0;
    for (Index c = 0; c < chunks; ++c) {
      sum +=
          v[((static_cast<std::size_t>(s * quantities + q) * static_cast<std::size_t>(chunks) +
              static_cast<std::size_t>(c)) *
             k_count) +
            k];
    }
    return sum;
  };
  Stats st;
  st.primal_residual = total(p.rows, kBatchRowQuantities, 0, row_chunks);
  st.dual_objective = total(p.rows, kBatchRowQuantities, 1, row_chunks) +
                      total(p.cols, kBatchColQuantities, 2, col_chunks);
  st.dual_move = total(p.rows, kBatchRowQuantities, 2, row_chunks);
  st.dual_residual = total(p.cols, kBatchColQuantities, 0, col_chunks);
  st.primal_objective = total(p.cols, kBatchColQuantities, 1, col_chunks);
  st.primal_move = total(p.cols, kBatchColQuantities, 3, col_chunks);
  return st;
}

/// The weighted KKT error of [cuPDLP] sec. 2: sqrt(w^2 |r_p|^2 + |r_d|^2 / w^2 + gap^2).
double kkt_error(const Stats& st, double weight) {
  const double gap = st.primal_objective - st.dual_objective;
  const double value =
      weight * weight * st.primal_residual + st.dual_residual / (weight * weight) + gap * gap;
  return std::isfinite(value) ? std::sqrt(value) : kInf;
}

std::unique_ptr<BatchBackend> make_backend(const BatchData& data, BatchBackendKind kind,
                                           BatchResult* result) {
  if (kind == BatchBackendKind::kDevice) {
#ifdef SANKHYA_ENABLE_CUDA
    std::string why;
    std::unique_ptr<BatchBackend> device = gpu::make_device_batch_backend(data, &why);
    if (device) {
      result->on_device = true;
      return device;
    }
    result->note = "device unavailable (" + why + "); ran on the CPU";
#else
    result->note = "this build has no CUDA backend; ran on the CPU";
#endif
  }
  return make_cpu_batch_backend(data);
}

}  // namespace

BatchResult solve_batch(const BatchProblem& problem, const BatchSettings& settings) {
  const Timer clock;
  BatchResult result;
  const Model& model = *problem.model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const auto k_count = static_cast<std::size_t>(std::max<Index>(problem.count, 0));
  result.bound.assign(k_count, -kInf);
  result.dual.assign(m * k_count, 0.0);
  result.restarts.assign(k_count, 0);
  if (k_count == 0 || problem.cost.size() != n || problem.col_lower.size() != n * k_count ||
      problem.col_upper.size() != n * k_count ||
      (!problem.primal_start.empty() && problem.primal_start.size() != n)) {
    result.note = "malformed batch";
    return result;
  }

  const Scaling scaling = build_scaling(model, problem.cost, kRuizIterations);
  const BatchData data = build_data(problem, scaling);
  const double eta = tol::kBatchPdhgStepFraction / spectral_norm(data);
  std::vector<double> weight(k_count, initial_primal_weight(data));
  std::vector<double> tau(k_count);
  std::vector<double> sigma(k_count);
  const auto set_steps = [&](std::size_t k) {
    tau[k] = eta / weight[k];
    sigma[k] = eta * weight[k];
  };
  for (std::size_t k = 0; k < k_count; ++k) set_steps(k);

  std::unique_ptr<BatchBackend> backend = make_backend(data, settings.backend, &result);
  const Index row_chunks = batch_chunks(data.rows);
  const Index col_chunks = batch_chunks(data.cols);

  // The restart state of [PDLP] sec. 3.2 per LP: the KKT error at the last restart, the
  // candidate's at the last check (for the no-local-progress test), and where the epoch began.
  std::vector<double> restart_kkt(k_count, kInf);
  std::vector<double> last_candidate(k_count, kInf);
  std::vector<Count> epoch_start(k_count, 0);
  BatchPartials partials;
  if (settings.iterations > 0) {
    backend->measure(&partials);
    for (std::size_t k = 0; k < k_count; ++k) {
      restart_kkt[k] =
          kkt_error(sum_stats(partials, 0, k, k_count, row_chunks, col_chunks), weight[k]);
    }
  }

  Count done = 0;
  std::vector<BatchRestart> action(k_count, BatchRestart::kNone);
  while (done < settings.iterations && backend->healthy()) {
    const Count step = std::min(tol::kBatchPdhgCheckInterval, settings.iterations - done);
    backend->iterate(step, tau, sigma);
    done += step;
    if (done >= settings.iterations || clock.elapsed_seconds() > settings.time_limit) break;

    backend->measure(&partials);
    for (std::size_t k = 0; k < k_count; ++k) {
      const Stats current = sum_stats(partials, 0, k, k_count, row_chunks, col_chunks);
      const Stats average = sum_stats(partials, 1, k, k_count, row_chunks, col_chunks);
      const double kkt_current = kkt_error(current, weight[k]);
      const double kkt_average = kkt_error(average, weight[k]);
      const bool to_average = kkt_average < kkt_current;
      const double candidate = to_average ? kkt_average : kkt_current;
      const bool sufficient = candidate <= tol::kBatchPdhgRestartSufficient * restart_kkt[k];
      const bool necessary = candidate <= tol::kBatchPdhgRestartNecessary * restart_kkt[k] &&
                             candidate > last_candidate[k];
      const bool artificial = static_cast<double>(done - epoch_start[k]) >=
                              tol::kBatchPdhgRestartArtificial * static_cast<double>(done);
      last_candidate[k] = candidate;
      action[k] = BatchRestart::kNone;
      if (!(sufficient || necessary || artificial)) continue;

      action[k] = to_average ? BatchRestart::kToAverage : BatchRestart::kToCurrent;
      // The primal weight update of [PDLP] sec. 3.3, from how far each side moved over the
      // epoch: log w <- theta log(dy / dx) + (1 - theta) log w.
      const Stats& chosen = to_average ? average : current;
      const double dx = std::sqrt(chosen.primal_move);
      const double dy = std::sqrt(chosen.dual_move);
      if (std::isfinite(dx) && std::isfinite(dy) && dx > tol::kBatchPdhgMinMove &&
          dy > tol::kBatchPdhgMinMove) {
        const double theta = tol::kBatchPdhgWeightSmoothing;
        weight[k] = std::exp(theta * std::log(dy / dx) + (1.0 - theta) * std::log(weight[k]));
        set_steps(k);
      }
      restart_kkt[k] = candidate;
      last_candidate[k] = kInf;
      epoch_start[k] = done;
      ++result.restarts[k];
    }
    backend->restart(action);
  }
  result.iterations = done;
  if (!backend->healthy()) {
    result.note = "a device call failed; no bound from this batch";
    result.seconds = clock.elapsed_seconds();
    return result;
  }

  // THE BOUNDS. Both the current and the average multipliers are tried and the larger bound
  // kept: each is valid on its own, so the maximum is too.
  std::vector<double> current;
  std::vector<double> average;
  backend->read_duals(&current, &average);
  std::vector<double> y(m);
  for (std::size_t k = 0; k < k_count; ++k) {
    SafeBoundProblem lp;
    lp.matrix = &model.matrix;
    lp.cost = problem.cost;
    lp.row_lower = model.row_lower;
    lp.row_upper = model.row_upper;
    lp.col_lower = std::span<const double>(problem.col_lower).subspan(k * n, n);
    lp.col_upper = std::span<const double>(problem.col_upper).subspan(k * n, n);
    for (const std::vector<double>* source : {&current, &average}) {
      // Any y is allowed, so a diverged entry is replaced by zero rather than trusted.
      for (std::size_t i = 0; i < m; ++i) {
        const double value = scaling.row[i] * (*source)[i * k_count + k];
        y[i] = std::isfinite(value) ? value : 0.0;
      }
      double bound = safe_dual_bound(lp, y).value;
      if (std::isnan(bound)) bound = -kInf;
      if (source == &current || bound > result.bound[k]) {
        result.bound[k] = bound;
        std::copy(y.begin(), y.end(), result.dual.begin() + static_cast<std::ptrdiff_t>(k * m));
      }
    }
  }
  result.seconds = clock.elapsed_seconds();
  return result;
}

}  // namespace sankhya::pdhg
