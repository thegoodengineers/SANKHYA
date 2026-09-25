// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU restarted PDHG for LP (#295).
//
// References (derived from the papers; no solver source consulted):
//   [CP11]   Chambolle & Pock, JMIV 40(1) 2011.
//   [PDLP]   Applegate et al., NeurIPS 2021.
//   [cuPDLP] Lu & Yang, arXiv:2311.12180.
//
// Partitioning scheme:
//   - Rows of A are split into K contiguous blocks balanced by nonzeros plus rows
//     (partition_rows_by_nonzeros): card k owns rows [r_k, r_{k+1}).
//   - The primal x (n) and the problem constants are REPLICATED, bit-identical, on every card.
//   - The dual y (m) is PARTITIONED: card k holds y[r_k..r_{k+1}).
//
// Per-iteration communication:
//   1. A^T y = sum_k A_k^T y_k: an all-gather of the n-vector partials, card to card over
//      P2P where every pair allows it, else staged through pinned host memory; each card sums
//      the partials in slot order (multi_gpu_exchange.hpp).
//   2. Three scalars per card (movement and interaction partials), downloaded and summed on
//      the host in slot order: the step-size rule needs them on the host anyway.
// Every reduction is order-fixed (pdhg_multi_gpu_device.hpp), so a run is reproducible bit for
// bit, and the same device list on one card or on K cards of one architecture gives the same
// bits.
//
// Fallback: one device (unless gpu_partitioned), or any device setup failure, delegates to
// solve_pdhg_gpu(); a CUDA error mid-solve hands the remaining budget to the CPU PDHG.

#include "pdhg_multi_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// fmt/format.h uses Unicode string literals that nvcc cannot parse; use snprintf instead.

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "../la/scaling.hpp"
#include "../pdhg/pdhg_evaluate.hpp"
#include "multi_device.hpp"
#include "multi_gpu_exchange.hpp"
#include "pdhg_gpu.hpp"
#include "pdhg_multi_gpu_device.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu {
namespace {

constexpr Count kEvaluationInterval = 40;

using multi::DeviceState;
using pdhg::euclidean_norm;
using pdhg::evaluate;
using pdhg::Problem;
using pdhg::Residuals;

// Pinned host memory for the per-card scalars, so their downloads are truly asynchronous.
struct PinnedDoubles {
  double* p = nullptr;
  explicit PinnedDoubles(std::size_t count) {
    if (cudaMallocHost(reinterpret_cast<void**>(&p), count * sizeof(double)) != cudaSuccess)
      p = nullptr;
  }
  PinnedDoubles(const PinnedDoubles&) = delete;
  PinnedDoubles& operator=(const PinnedDoubles&) = delete;
  ~PinnedDoubles() {
    if (p) cudaFreeHost(p);
  }
};

}  // anonymous namespace

// ---- Public entry point --------------------------------------------------

Solution solve_pdhg_multi_gpu(const Model& model, const Options& options,
                               const std::vector<int>& device_ids, Logger& logger,
                               SolveControl* control) {
  // One device: the single-card engine, unless the partitioned engine is asked for by name
  // (gpu_partitioned), which is how its one-card baseline is measured (#295).
  if (device_ids.empty() || (device_ids.size() == 1 && !options.get_bool("gpu_partitioned"))) {
    return solve_pdhg_gpu(model, options, logger, control);
  }
  if (device_ids.size() > static_cast<std::size_t>(multi::kMaxDevices)) {
    logger.warning("Multi-GPU PDHG: {} devices requested, at most {} supported; single-GPU",
                   device_ids.size(), multi::kMaxDevices);
    return solve_pdhg_gpu(model, options, logger, control);
  }

  // Every requested device must exist.
  const int total_devices = device_count();
  for (int id : device_ids) {
    if (id < 0 || id >= total_devices) {
      logger.warning(
          "Multi-GPU PDHG: device {} requested but only {} devices are present; "
          "falling back to single-GPU",
          id, total_devices);
      return solve_pdhg_gpu(model, options, logger, control);
    }
  }

  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cuda-multi";

  const std::string validation = model.validate();
  if (!validation.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = validation;
    return solution;
  }

  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);
  const int ni = static_cast<int>(n);
  const int K = static_cast<int>(device_ids.size());

  // ---- Unscaled problem --------------------------------------------------
  Problem prob;
  prob.model = &model;
  prob.cost.resize(n);
  for (Index j = 0; j < cols; ++j)
    prob.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  prob.cost_norm = euclidean_norm(prob.cost);
  {
    double bsq = 0.0;
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double b = is_finite_bound(model.row_lower[u])
                           ? model.row_lower[u]
                           : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
      bsq += b * b;
    }
    prob.bound_norm = std::sqrt(bsq);
  }

  // ---- Precondition (CPU, one-off) ---------------------------------------
  constexpr int kRuizIter = 10;
  constexpr int kPowerIter = 30;
  const Scaling scaling = build_scaling(model, prob.cost, kRuizIter);
  const double spectral_norm = estimate_spectral_norm(
      scaling.matrix, kPowerIter, static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  // ---- Build CSR on CPU --------------------------------------------------
  const CsrView full_csr(scaling.matrix);

  // ---- Solver parameters -------------------------------------------------
  const double tolerance = options.get_double("pdhg_tolerance");
  const ResourceLimits limits(options, logger);
  const Count iteration_limit =
      limits.iteration_limit() < 0 ? 1000000 : static_cast<Count>(limits.iteration_limit());
  const bool use_restarts = options.get_bool("pdhg_restart");
  const bool stop_at_request = options.get_bool("pdhg_stop_at_request");

  // ---- Initial x (scaled) ------------------------------------------------
  std::vector<double> x0_scaled(n);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double v = 0.0;
    if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
      v = scaling.col_lower[u];
    if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
      v = scaling.col_upper[u];
    x0_scaled[u] = v;
  }

  // ---- Row partitioning, balanced by nonzeros (#295) ---------------------
  const std::vector<RowPartition> parts =
      partition_rows_by_nonzeros(full_csr.row_starts(), device_ids);
  const bool allow_peer = options.get_bool("gpu_peer_access");

  logger.info(
      "Solving LP with CUDA multi-GPU ({} devices) restarted PDHG: {} rows, {} columns, {} "
      "nonzeros",
      K, rows, cols, model.num_nonzeros());
  for (const RowPartition& part : parts)
    logger.info("  device {}: rows [{}, {}), {} nonzeros", part.device_id, part.row_start,
                part.row_end, partition_weight(full_csr.row_starts(), part) - part.local_m());
  logger.info("Estimated ||A||_2 = {:.4e}, target tolerance {:.1e}, restarts {}", spectral_norm,
              tolerance, use_restarts ? "on" : "off");

  // ---- Per-card state and the exchange -----------------------------------
  // The cards are declared before the exchange so that the exchange, which synchronizes the
  // cards' streams when it goes, is destroyed first.
  std::vector<std::unique_ptr<DeviceState>> cards;
  std::vector<DeviceState*> card_ptrs;
  bool setup_ok = true;
  for (int k = 0; k < K && setup_ok; ++k) {
    const RowPartition& part = parts[static_cast<std::size_t>(k)];
    cards.push_back(std::make_unique<DeviceState>());
    DeviceState& c = *cards.back();
    c.device_id = part.device_id;
    c.slot = k;
    c.num_slots = K;
    card_ptrs.push_back(&c);
    setup_ok =
        multi::setup_device(c, full_csr, scaling, x0_scaled, part.row_start, part.row_end, ni);
  }
  multi::PartialSumExchange exchange;
  std::string transport_reason;
  if (setup_ok) setup_ok = exchange.init(card_ptrs, allow_peer, &transport_reason);
  PinnedDoubles host_scalars(3 * static_cast<std::size_t>(K));
  if (!setup_ok || host_scalars.p == nullptr) {
    logger.warning(
        "Multi-GPU PDHG: device setup failed; falling back to single-GPU on device {}",
        device_ids[0]);
    return solve_pdhg_gpu(model, options, logger, control);
  }
  const char* transport = multi::to_string(exchange.transport());
  logger.info("A^T y exchange: {} ({})", transport, transport_reason);

  // ---- CPU-side iterate buffers ------------------------------------------
  std::vector<double> h_x(n), h_xsum(n);
  std::vector<double> h_y(m), h_ysum(m);
  std::vector<double> x_unscaled(n), y_unscaled(m);
  std::vector<double> activity(m), reduced_costs(n);
  std::vector<double> x_restart(x0_scaled), y_restart(m, 0.0);

  auto unscale = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    for (Index j = 0; j < cols; ++j)
      x_unscaled[static_cast<std::size_t>(j)] =
          xs[static_cast<std::size_t>(j)] * scaling.column[static_cast<std::size_t>(j)];
    for (Index i = 0; i < rows; ++i)
      y_unscaled[static_cast<std::size_t>(i)] =
          ys[static_cast<std::size_t>(i)] * scaling.row[static_cast<std::size_t>(i)];
  };

  // Assemble full y from partitioned device memory into h_y.
  auto gather_y = [&](bool use_ysum, std::vector<double>& dest) -> bool {
    for (const auto& c : cards) {
      if (c->local_m == 0) continue;
      if (cudaSetDevice(c->device_id) != cudaSuccess) return false;
      if (!multi::download(*c, use_ysum ? c->d_ysum : c->d_y, dest.data() + c->row_start,
                           static_cast<std::size_t>(c->local_m)))
        return false;
    }
    return true;
  };

  // ---- PDHG main loop ----------------------------------------------------
  double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  double omega = 1.0;
  Count iteration = 0, restarts = 0, last_restart = 0, averaged = 0, attempts = 0;
  double restart_kkt = std::numeric_limits<double>::infinity();
  double evaluation_seconds = 0.0;

  Residuals best;
  best.primal = best.dual = best.gap = std::numeric_limits<double>::infinity();
  std::vector<double> best_x(n, 0.0), best_y(m, 0.0);

  bool converged = false, gpu_error = false, logged_table = false;
  StopController stop(control, timer, limits);
  SolveStatus stop_status = SolveStatus::kIterationLimit;

  // Run fn on every card in slot order with that card's device current; false on any error.
  auto on_every_card = [&](auto&& fn) -> bool {
    for (const auto& c : cards) {
      if (cudaSetDevice(c->device_id) != cudaSuccess) return false;
      if (!fn(*c)) return false;
    }
    return true;
  };
  DeviceState& c0 = *cards[0];
  const double loop_start = timer.elapsed_seconds();

  while (!gpu_error) {
    if (iteration >= iteration_limit) break;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iteration;
              p.objective = best.primal;
              p.best_bound = sense * best.dual_objective + model.objective_offset;
              return p;
            },
            &stop_status))
      break;

    const double tau = eta / omega;
    const double sigma = eta * omega;

    // One iteration, enqueued on every card's stream [cuPDLP sec. 3]:
    //   A_k^T y_k -> exchange -> primal step -> A_k x_ext -> dual step -> A_k dx ->
    //   interaction -> scalars to the host.
    const bool enqueued =
        on_every_card([](DeviceState& c) { return multi::spmv_aty(c); }) &&
        exchange.enqueue() &&
        on_every_card([&](DeviceState& c) { return multi::launch_primal(c, tau, omega); }) &&
        on_every_card([](DeviceState& c) { return multi::spmv_ax(c, c.d_ext, c.d_ax); }) &&
        on_every_card([&](DeviceState& c) { return multi::launch_dual(c, sigma, omega); }) &&
        on_every_card([](DeviceState& c) { return multi::spmv_ax(c, c.d_dx, c.d_adx); }) &&
        on_every_card([&](DeviceState& c) {
          return multi::launch_interaction_and_scalars(
              c, host_scalars.p + 3 * static_cast<std::size_t>(c.slot));
        }) &&
        // The one host synchronization per iteration. It also guarantees that no card starts
        // the next iteration's A_k^T y_k, or its staging copy, while another card may still be
        // reading this iteration's partial.
        on_every_card(
            [](DeviceState& c) { return cudaStreamSynchronize(c.stream) == cudaSuccess; });
    if (!enqueued) {
      gpu_error = true;
      break;
    }
    ++attempts;

    // mv_x is computed identically on every card (replicated x): card 0's is used. The dual
    // movement and the interaction are summed over the cards in slot order.
    const double h_mv_x = host_scalars.p[0];
    double h_mv_y = 0.0, h_interaction = 0.0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k) {
      h_mv_y += host_scalars.p[3 * k + 1];
      h_interaction += host_scalars.p[3 * k + 2];
    }
    const double movement = h_mv_x + h_mv_y;
    const double interaction = std::fabs(h_interaction);

    // Adaptive step size [PDLP sec. 3.1].
    const bool no_info = (interaction <= 0.0);
    const double limit =
        no_info ? std::numeric_limits<double>::infinity() : movement / interaction;
    const double exp_val = static_cast<double>(std::max<Count>(2, iteration + 1));
    const double shrink = 1.0 - std::pow(exp_val, -0.3);
    const double grow = 1.0 + std::pow(exp_val, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (eta <= limit) {
      // Accept: swap the iterates on every card and accumulate the running sums.
      if (!on_every_card([](DeviceState& c) {
            std::swap(c.d_x, c.d_xn);
            std::swap(c.d_y, c.d_yn);
            return multi::launch_accumulate(c);
          })) {
        gpu_error = true;
        break;
      }
      ++averaged;
      ++iteration;
    }
    const double eta_ceil = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_info) eta = std::clamp(proposed, 1e-12, eta_ceil);

    // Convergence check on the host every kEvaluationInterval iterations.
    if (iteration == 0) continue;
    if (iteration % kEvaluationInterval != 0 && !no_info) continue;
    const double evaluation_start = timer.elapsed_seconds();

    // x from card 0 (bit-identical on all), y gathered from every card.
    if (cudaSetDevice(c0.device_id) != cudaSuccess || !multi::download(c0, c0.d_x, h_x.data(), n) ||
        !gather_y(false, h_y)) {
      gpu_error = true;
      break;
    }

    unscale(h_x, h_y);
    std::vector<double> cur_x = x_unscaled, cur_y = y_unscaled;
    const Residuals cur = evaluate(prob, cur_x, cur_y, activity, reduced_costs);

    const Residuals* chosen = &cur;
    const std::vector<double>* chosen_x = &cur_x;
    const std::vector<double>* chosen_y = &cur_y;

    Residuals avg;
    std::vector<double> avg_x, avg_y;
    if (averaged > 0) {
      if (cudaSetDevice(c0.device_id) != cudaSuccess ||
          !multi::download(c0, c0.d_xsum, h_xsum.data(), n) || !gather_y(true, h_ysum)) {
        gpu_error = true;
        break;
      }
      const double cnt = static_cast<double>(averaged);
      std::vector<double> xav(n), yav(m);
      for (std::size_t j = 0; j < n; ++j) xav[j] = h_xsum[j] / cnt;
      for (std::size_t i = 0; i < m; ++i) yav[i] = h_ysum[i] / cnt;
      unscale(xav, yav);
      avg_x = x_unscaled;
      avg_y = y_unscaled;
      avg = evaluate(prob, avg_x, avg_y, activity, reduced_costs);
      if (avg.worst() < cur.worst()) {
        chosen = &avg;
        chosen_x = &avg_x;
        chosen_y = &avg_y;
      }
    }

    const Residuals& better = *chosen;
    if (better.worst() < best.worst()) {
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
    }

    if (!logged_table) {
      logger.begin_iteration_table();
      logged_table = true;
    }
    logger.iteration(iteration, sense * better.primal_objective + model.objective_offset,
                     better.primal, better.dual, timer.elapsed_seconds());

    const bool stop_here =
        better.meets_request(tolerance) && (stop_at_request || better.meets_project_standard());
    if (stop_here) {
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
      converged = true;
      evaluation_seconds += timer.elapsed_seconds() - evaluation_start;
      break;
    }

    // Restart logic [PDLP sec. 4.3].
    if (use_restarts) {
      const double kkt = better.worst();
      const Count since = iteration - last_restart;
      const bool sufficient = kkt <= 0.2 * restart_kkt;
      const bool artificial =
          since >= std::max<Count>(kEvaluationInterval,
                                   static_cast<Count>(0.36 * static_cast<double>(iteration)));
      if (sufficient || artificial) {
        std::vector<double> dx_rs(n), dy_rs(m);
        for (std::size_t j = 0; j < n; ++j) dx_rs[j] = h_x[j] - x_restart[j];
        for (std::size_t i = 0; i < m; ++i) dy_rs[i] = h_y[i] - y_restart[i];
        const double dxn = euclidean_norm(dx_rs), dyn = euclidean_norm(dy_rs);
        if (dxn > 1e-12 && dyn > 1e-12) {
          constexpr double theta = 0.5;
          omega = std::exp(theta * std::log(dyn / dxn) + (1.0 - theta) * std::log(omega));
          omega = std::clamp(omega, 1e-6, 1e6);
        }
        // Zero the running sums on every card.
        if (!on_every_card([&](DeviceState& c) {
              if (n > 0 &&
                  cudaMemsetAsync(c.d_xsum, 0, n * sizeof(double), c.stream) != cudaSuccess)
                return false;
              return c.local_m == 0 ||
                     cudaMemsetAsync(c.d_ysum, 0,
                                     static_cast<std::size_t>(c.local_m) * sizeof(double),
                                     c.stream) == cudaSuccess;
            })) {
          gpu_error = true;
          break;
        }
        averaged = 0;
        x_restart = h_x;
        y_restart = h_y;
        restart_kkt = kkt;
        last_restart = iteration;
        ++restarts;
        logger.verbose("restart {} at iteration {}: KKT {:.3e}, primal weight {:.3e}", restarts,
                       iteration, kkt, omega);
      }
    }
    evaluation_seconds += timer.elapsed_seconds() - evaluation_start;
  }  // end while
  const double loop_seconds = timer.elapsed_seconds() - loop_start;

  if (gpu_error) {
    Options remaining = options;
    if (limits.has_time_limit())
      remaining.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    logger.warning(
        "Multi-GPU PDHG: CUDA error after {:.2f}s; falling back to CPU PDHG on remaining "
        "budget",
        timer.elapsed_seconds());
    return pdhg::solve_pdhg(model, remaining, logger, control);
  }

  // ---- Extract solution --------------------------------------------------
  for (Index j = 0; j < cols; ++j)
    solution.col_value[static_cast<std::size_t>(j)] =
        best_x.empty() ? 0.0 : best_x[static_cast<std::size_t>(j)];

  const Residuals final_r = evaluate(prob, best_x, best_y, activity, reduced_costs);
  for (Index j = 0; j < cols; ++j)
    solution.col_dual[static_cast<std::size_t>(j)] =
        sense * reduced_costs[static_cast<std::size_t>(j)];
  for (Index i = 0; i < rows; ++i)
    solution.row_dual[static_cast<std::size_t>(i)] =
        sense * (-best_y[static_cast<std::size_t>(i)]);

  solution.iterations = iteration;
  solution.solve_seconds = timer.elapsed_seconds();

  const bool verifiable = converged && final_r.meets_project_standard();
  if (verifiable) {
    solution.status = SolveStatus::kOptimal;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices, %s exchange) converged after %lld iterations and %lld restarts; "
        "absolute primal %.3e, dual %.3e, relative gap %.3e",
        K, transport, (long long)iteration, (long long)restarts, final_r.absolute_primal,
        final_r.absolute_dual,
        final_r.gap_as_verified);
    solution.message = buf;
  } else if (converged) {
    solution.status = SolveStatus::kFeasible;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices, %s exchange) met requested tolerance %.1e after %lld iterations "
        "but NOT project standard",
        K, transport, tolerance, (long long)iteration);
    solution.message = buf;
  } else {
    solution.status =
        (stop_status == SolveStatus::kTimeLimit || stop_status == SolveStatus::kInterrupted)
            ? stop_status
            : SolveStatus::kIterationLimit;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices, %s exchange) stopped at relative primal %.3e, dual %.3e, "
        "gap %.3e after %lld iterations and %lld restarts (target %.1e)",
        K, transport, final_r.primal, final_r.dual, final_r.gap, (long long)iteration,
        (long long)restarts, tolerance);
    solution.message = buf;
  }

  if (verifiable) {
    solution.dual_bound = sense * final_r.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info(
      "Status: {}  objective {:.10e}  iterations {}  restarts {}  devices {}  time {:.3f}s",
      to_string(solution.status), solution.objective, solution.iterations, restarts, K,
      solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}", final_r.primal,
              final_r.dual, final_r.gap);
  // The split the scaling measurements (#295) need: the device loop per step attempt
  // (accepted or rejected), with the host-side evaluation every kEvaluationInterval
  // iterations taken out, and that evaluation on its own.
  const double device_seconds = loop_seconds - evaluation_seconds;
  logger.info(
      "Timing: loop {:.3f}s, {} step attempts, {:.1f} us per attempt on the cards; host "
      "evaluation {:.3f}s",
      loop_seconds, attempts,
      attempts > 0 ? 1e6 * device_seconds / static_cast<double>(attempts) : 0.0,
      evaluation_seconds);
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::gpu
