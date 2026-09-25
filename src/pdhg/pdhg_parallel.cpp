// SPDX-License-Identifier: Apache-2.0
// SANKHYA - see pdhg_parallel.hpp (#487). Each per-entry expression below is the serial
// loop's in pdhg.cpp, term for term, so the only difference between the two paths is the
// order the three sums are taken in.
#include "pdhg_parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::pdhg {
namespace {

double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// Runs chunk(begin, end) over the fixed chunks of [0, count) on the OpenMP workers and
/// returns the chunk results summed in chunk order on the calling thread. The chunk bounds
/// depend on `count` and tol::kPdhgParallelChunk only, never on the thread count, so the
/// total is the same bits at any number of threads.
template <class Chunk>
double fixed_order_sum(std::size_t count, const Chunk& chunk) {
  const auto size = static_cast<std::size_t>(tol::kPdhgParallelChunk);
  const std::size_t chunks = (count + size - 1) / size;
  std::vector<double> partial(chunks, 0.0);
  const auto last = static_cast<std::int64_t>(chunks);
#ifdef SANKHYA_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t c = 0; c < last; ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * size;
    partial[static_cast<std::size_t>(c)] = chunk(begin, std::min(count, begin + size));
  }
  double total = 0.0;
  for (const double p : partial) total += p;
  return total;
}

/// Runs body(k) for every k in [0, count) on the OpenMP workers, statically partitioned.
template <class Body>
void for_each_index(std::size_t count, const Body& body) {
  const auto last = static_cast<std::int64_t>(count);
#ifdef SANKHYA_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t k = 0; k < last; ++k) body(static_cast<std::size_t>(k));
}

}  // namespace

double parallel_primal_step(const Scaling& scaling, const std::vector<double>& x,
                            const std::vector<double>& at_y, double tau, double omega,
                            std::vector<double>& x_next, std::vector<double>* extrapolated,
                            std::vector<double>* dx) {
  return fixed_order_sum(x.size(), [&](std::size_t begin, std::size_t end) {
    double movement = 0.0;
    for (std::size_t u = begin; u < end; ++u) {
      const double gradient = scaling.cost[u] + at_y[u];
      x_next[u] = project(x[u] - tau * gradient, scaling.col_lower[u], scaling.col_upper[u]);
      if (extrapolated != nullptr) (*extrapolated)[u] = 2.0 * x_next[u] - x[u];
      const double d = x_next[u] - x[u];
      if (dx != nullptr) (*dx)[u] = d;
      movement += 0.5 * omega * d * d;
    }
    return movement;
  });
}

double parallel_dual_step(const Scaling& scaling, const std::vector<double>& y,
                          const std::vector<double>& a_x, double sigma, double omega,
                          std::vector<double>& y_next) {
  return fixed_order_sum(y.size(), [&](std::size_t begin, std::size_t end) {
    double movement = 0.0;
    for (std::size_t u = begin; u < end; ++u) {
      const double v = y[u] + sigma * a_x[u];
      y_next[u] = v - sigma * project(v / sigma, scaling.row_lower[u], scaling.row_upper[u]);
      const double d = y_next[u] - y[u];
      movement += 0.5 * d * d / omega;
    }
    return movement;
  });
}

double parallel_interaction(const std::vector<double>& y_next, const std::vector<double>& y,
                            const std::vector<double>& a, const std::vector<double>* b) {
  return fixed_order_sum(y.size(), [&](std::size_t begin, std::size_t end) {
    double interaction = 0.0;
    if (b == nullptr) {
      for (std::size_t i = begin; i < end; ++i) interaction += (y_next[i] - y[i]) * a[i];
    } else {
      for (std::size_t i = begin; i < end; ++i) {
        interaction += (y_next[i] - y[i]) * (a[i] - (*b)[i]);
      }
    }
    return interaction;
  });
}

void parallel_extrapolate_product(const std::vector<double>& next,
                                  const std::vector<double>& cached, std::vector<double>& out) {
  for_each_index(out.size(), [&](std::size_t i) { out[i] = 2.0 * next[i] - cached[i]; });
}

void parallel_accumulate(std::vector<double>& sum, const std::vector<double>& v) {
  for_each_index(sum.size(), [&](std::size_t k) { sum[k] += v[k]; });
}

}  // namespace sankhya::pdhg
