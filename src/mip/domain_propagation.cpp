// SPDX-License-Identifier: Apache-2.0
// SANKHYA - synchronous activity-based bound propagation (#510). See domain_propagation.hpp.
//
// Every floating-point step below is mirrored, in the same order, by the CUDA kernels in
// src/gpu/domain_prop.cu; change one and the other must change with it, or the test that
// holds them to identical bounds fails.

#include "domain_propagation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "../gpu/domain_prop.hpp"
#endif

namespace sankhya::mip {

RowMajor row_major(const Model& model) {
  RowMajor r;
  r.rows = model.num_rows();
  r.cols = model.num_cols();
  std::vector<Index> count(static_cast<std::size_t>(r.rows) + 1, 0);
  for (Index j = 0; j < r.cols; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      ++count[static_cast<std::size_t>(column.rows[k]) + 1];
  }
  r.start.assign(static_cast<std::size_t>(r.rows) + 1, 0);
  for (Index i = 0; i < r.rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    r.start[u + 1] = r.start[u] + count[u + 1];
  }
  const auto nnz = static_cast<std::size_t>(r.start.back());
  r.column.assign(nnz, 0);
  r.value.assign(nnz, 0.0);
  std::vector<Index> next(r.start.begin(), r.start.end() - 1);
  // Columns in increasing order, so each row lists its entries by column: one fixed order
  // both propagators sum in.
  for (Index j = 0; j < r.cols; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto slot =
          static_cast<std::size_t>(next[static_cast<std::size_t>(column.rows[k])]++);
      r.column[slot] = j;
      r.value[slot] = column.values[k];
    }
  }
  return r;
}

JacobiPropagation propagate_jacobi(const Model& model, const RowMajor& rows,
                                   std::vector<double>* lower, std::vector<double>* upper,
                                   int rounds_limit, double integrality) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  JacobiPropagation result;
  const auto n = static_cast<std::size_t>(rows.cols);
  std::vector<double>& lo = *lower;
  std::vector<double>& hi = *upper;
  std::vector<double> cand_lo(n);
  std::vector<double> cand_hi(n);
  for (int round = 0; round < rounds_limit; ++round) {
    ++result.rounds;
    std::fill(cand_lo.begin(), cand_lo.end(), -kInf);
    std::fill(cand_hi.begin(), cand_hi.end(), kInf);
    for (Index i = 0; i < rows.rows; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      const auto begin = static_cast<std::size_t>(rows.start[ui]);
      const auto end = static_cast<std::size_t>(rows.start[ui + 1]);
      double min_act = 0.0;
      double max_act = 0.0;
      int min_inf = 0;
      int max_inf = 0;
      for (std::size_t k = begin; k < end; ++k) {
        const double a = rows.value[k];
        const auto j = static_cast<std::size_t>(rows.column[k]);
        const double low = a > 0.0 ? a * lo[j] : a * hi[j];
        const double high = a > 0.0 ? a * hi[j] : a * lo[j];
        if (std::isinf(low)) {
          ++min_inf;
        } else {
          min_act = min_act + low;
        }
        if (std::isinf(high)) {
          ++max_inf;
        } else {
          max_act = max_act + high;
        }
      }
      const double ru = model.row_upper[ui];
      const double rl = model.row_lower[ui];
      if ((min_inf == 0 && std::isfinite(ru) && min_act > ru + tol::kPrimalFeasibility) ||
          (max_inf == 0 && std::isfinite(rl) && max_act < rl - tol::kPrimalFeasibility)) {
        result.infeasible = true;
        return result;
      }
      for (std::size_t k = begin; k < end; ++k) {
        const double a = rows.value[k];
        if (a == 0.0) continue;
        const auto j = static_cast<std::size_t>(rows.column[k]);
        const double own_low = a > 0.0 ? a * lo[j] : a * hi[j];
        const double own_high = a > 0.0 ? a * hi[j] : a * lo[j];
        if (min_inf == 0 && std::isfinite(ru) && std::isfinite(own_low)) {
          const double implied = (ru - (min_act - own_low)) / a;
          if (a > 0.0) {
            cand_hi[j] = std::min(cand_hi[j], implied);
          } else {
            cand_lo[j] = std::max(cand_lo[j], implied);
          }
        }
        if (max_inf == 0 && std::isfinite(rl) && std::isfinite(own_high)) {
          const double implied = (rl - (max_act - own_high)) / a;
          if (a > 0.0) {
            cand_lo[j] = std::max(cand_lo[j], implied);
          } else {
            cand_hi[j] = std::min(cand_hi[j], implied);
          }
        }
      }
    }
    Count changed = 0;
    for (std::size_t j = 0; j < n; ++j) {
      double new_lo = lo[j];
      double new_hi = hi[j];
      if (cand_hi[j] < new_hi - tol::kPropagationMinChange) new_hi = cand_hi[j];
      if (cand_lo[j] > new_lo + tol::kPropagationMinChange) new_lo = cand_lo[j];
      if (model.col_type[j] == VarType::kInteger) {
        if (std::isfinite(new_hi)) new_hi = std::floor(new_hi + integrality);
        if (std::isfinite(new_lo)) new_lo = std::ceil(new_lo - integrality);
      }
      if (new_lo > new_hi + tol::kPrimalFeasibility) {
        result.infeasible = true;
        return result;
      }
      changed += static_cast<Count>(new_lo != lo[j]) + static_cast<Count>(new_hi != hi[j]);
      lo[j] = new_lo;
      hi[j] = new_hi;
    }
    result.tightened += changed;
    if (changed == 0) break;
  }
  return result;
}

Count propagate_root_bounds(Model* model, const Options& options, Logger& logger) {
  const int rounds = tol::kDomainPropagationRounds;
  const double integrality = tol::kIntegrality;
  // Wall time of the whole propagation as the search pays for it: on the device that
  // includes building the row-major copy and the transfers to and from the card.
  const Timer clock;
#ifndef SANKHYA_ENABLE_CUDA
  (void)options;  // only the CUDA build has a backend to choose
#endif
#ifdef SANKHYA_ENABLE_CUDA
  const bool device_allowed = options.get_string("domain_prop_backend") != "cpu";
  const gpu::PropResult device =
      device_allowed ? gpu::propagate_bounds(*model, model->col_lower, model->col_upper, rounds,
                                             integrality)
                     : gpu::PropResult{};
  if (device.ran) {
    logger.info(
        "Domain propagation (#510, GPU): {} round(s), {} bound(s) tightened in {:.6f}s{}",
        device.rounds, device.tightened, clock.elapsed_seconds(),
        device.infeasible ? "; the box is empty, left for the search to prove" : "");
    // Where the device time went: the context is made once per process, so on a solve
    // that has already touched the card it is the probe alone.
    const gpu::PropPhases& t = device.phases;
    logger.verbose(
        "Domain propagation (#510, GPU) phases: context {:.6f}s, host copy {:.6f}s, allocate "
        "{:.6f}s, upload {:.6f}s, rounds {:.6f}s, download {:.6f}s, release {:.6f}s",
        t.context, t.host, t.allocate, t.upload, t.rounds, t.download, t.release);
    if (device.infeasible) return 0;
    model->col_lower = device.col_lb;
    model->col_upper = device.col_ub;
    return static_cast<Count>(device.tightened);
  }
#endif
  std::vector<double> lower = model->col_lower;
  std::vector<double> upper = model->col_upper;
  const JacobiPropagation cpu =
      propagate_jacobi(*model, row_major(*model), &lower, &upper, rounds, integrality);
  logger.info("Domain propagation (#510, CPU): {} round(s), {} bound(s) tightened in {:.6f}s{}",
              cpu.rounds, cpu.tightened, clock.elapsed_seconds(),
              cpu.infeasible ? "; the box is empty, left for the search to prove" : "");
  if (cpu.infeasible) return 0;
  model->col_lower = std::move(lower);
  model->col_upper = std::move(upper);
  return cpu.tightened;
}

}  // namespace sankhya::mip
