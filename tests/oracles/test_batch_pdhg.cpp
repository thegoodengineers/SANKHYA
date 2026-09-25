// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the batched PDHG (#520) against the exact oracle, against itself, and the device
// against the CPU.
//
// What is claimed, and tested here:
//  1. Every bound the batch returns is at most the EXACT optimum of its LP (rational
//     arithmetic), at every iteration budget including zero - the bound is safe for any
//     duals, so the budget may only change how tight it is. An unbounded LP gets -inf.
//  2. A batch of K LPs returns, bit for bit, what K batches of one return: nothing leaks
//     between the LPs of a batch.
//  3. The CUDA backend returns, bit for bit, what the CPU reference returns (skipped
//     without a device, or in a build without CUDA).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "pdhg/batch_pdhg.hpp"

namespace sankhya::oracle {
namespace {

/// A rational at least `v` (see test_safe_bound.cpp): "bound <= optimum" stays conservative.
Rational rational_at_least(double v) {
  constexpr int kMinExponent = -60;
  if (v == 0.0) return Rational(0);
  if (std::fabs(v) < std::ldexp(1.0, kMinExponent)) {
    return v < 0.0 ? Rational(0) : Rational(1, static_cast<Rational::Int>(1) << 60);
  }
  int exponent = 0;
  const double mantissa = std::frexp(v, &exponent);
  const auto scaled = static_cast<std::int64_t>(std::ldexp(mantissa, 53));
  const int shift = exponent - 53;
  if (shift >= 0) {
    return Rational(static_cast<Rational::Int>(scaled) *
                    (static_cast<Rational::Int>(1) << shift));
  }
  return Rational(static_cast<Rational::Int>(scaled), static_cast<Rational::Int>(1)
                                                          << (-shift));
}

/// K integer boxes inside [0, u] (u = 8 where the LP has none), each a random sub-box.
struct Boxes {
  std::vector<std::vector<std::int64_t>> lower;
  std::vector<std::vector<std::int64_t>> upper;
};

Boxes random_boxes(std::mt19937_64& rng, const GeneratedLp& lp, int count, bool keep_infinite) {
  Boxes boxes;
  std::uniform_int_distribution<int> coin(0, 3);
  for (int k = 0; k < count; ++k) {
    std::vector<std::int64_t> lo(static_cast<std::size_t>(lp.num_cols), 0);
    std::vector<std::int64_t> hi = lp.upper;
    for (std::size_t j = 0; j < lo.size(); ++j) {
      const bool infinite = hi[j] == kNoUpperBound;
      const std::int64_t top = infinite ? 8 : hi[j];
      if (infinite && !keep_infinite) hi[j] = 8;
      if (coin(rng) == 0 && top > 0) {
        std::uniform_int_distribution<std::int64_t> pick(0, top);
        const std::int64_t a = pick(rng);
        if (coin(rng) < 2) {
          lo[j] = a;
        } else {
          hi[j] = a;
        }
      }
    }
    boxes.lower.push_back(lo);
    boxes.upper.push_back(hi);
  }
  return boxes;
}

pdhg::BatchProblem batch_of(const Model& model, const Boxes& boxes) {
  pdhg::BatchProblem problem;
  problem.model = &model;
  problem.cost = model.col_cost;
  problem.count = static_cast<Index>(boxes.lower.size());
  for (std::size_t k = 0; k < boxes.lower.size(); ++k) {
    for (std::size_t j = 0; j < boxes.lower[k].size(); ++j) {
      problem.col_lower.push_back(static_cast<double>(boxes.lower[k][j]));
      problem.col_upper.push_back(boxes.upper[k][j] == kNoUpperBound
                                      ? std::numeric_limits<double>::infinity()
                                      : static_cast<double>(boxes.upper[k][j]));
    }
  }
  return problem;
}

struct Tally {
  int optimal = 0;
  int infeasible = 0;
  int unbounded = 0;
  int abstained = 0;
  int finite = 0;
  int tight = 0;  ///< within 1e-6 relative of the optimum
};

/// Claim 1 over a sweep of instances, for one iteration budget and backend.
Tally sweep(Count iterations, pdhg::BatchBackendKind backend, int trials, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  GeneratorConfig config;
  config.max_rows = 6;
  config.max_cols = 7;
  Tally tally;
  for (int trial = 0; trial < trials; ++trial) {
    GeneratedLp lp = trial % 2 == 0 ? kkt_lp(rng, config).lp : random_lp(rng, config);
    const Boxes boxes = random_boxes(rng, lp, 6, trial % 3 != 0);
    const Model model = to_model(lp);
    pdhg::BatchSettings settings;
    settings.iterations = iterations;
    settings.backend = backend;
    const pdhg::BatchResult result = pdhg::solve_batch(batch_of(model, boxes), settings);
    for (std::size_t k = 0; k < boxes.lower.size(); ++k) {
      GeneratedLp node = lp;
      node.lower = boxes.lower[k];
      node.upper = boxes.upper[k];
      const OracleResult exact = solve_exact(node);
      const double bound = result.bound[k];
      if (exact.status == OracleStatus::kOptimal) {
        ++tally.optimal;
        if (std::isfinite(bound)) ++tally.finite;
        EXPECT_FALSE(std::isnan(bound));
        EXPECT_LE(rational_at_least(bound), exact.objective)
            << "iterations " << iterations << " LP " << k << ": bound " << bound
            << " above the exact optimum " << exact.objective.to_double() << "\n"
            << node.to_text();
        const double optimum = exact.objective.to_double();
        if (std::isfinite(bound) &&
            optimum - bound <= 1e-6 * std::max(1.0, std::fabs(optimum))) {
          ++tally.tight;
        }
      } else if (exact.status == OracleStatus::kUnbounded) {
        ++tally.unbounded;
        EXPECT_EQ(bound, -std::numeric_limits<double>::infinity())
            << "a finite lower bound on an unbounded LP\n"
            << node.to_text();
      } else if (exact.status == OracleStatus::kInfeasible) {
        ++tally.infeasible;  // any bound is a valid lower bound on +infinity
      } else {
        ++tally.abstained;
      }
    }
  }
  return tally;
}

void report(const char* label, Count iterations, const Tally& t) {
  std::printf(
      "[  INFO    ] %s, %lld iterations: %d optimal LPs (%d with a finite bound, %d within "
      "1e-6 of the optimum), %d infeasible, %d unbounded, %d abstained\n",
      label, static_cast<long long>(iterations), t.optimal, t.finite, t.tight, t.infeasible,
      t.unbounded, t.abstained);
}

bool device_ran(pdhg::BatchResult* out) {
  GeneratedLp lp;
  lp.num_rows = 1;
  lp.num_cols = 1;
  lp.a = {{1}};
  lp.b = {1};
  lp.c = {1};
  lp.upper = {kNoUpperBound};
  const Model model = to_model(lp);
  Boxes boxes;
  boxes.lower = {{0}};
  boxes.upper = {{4}};
  pdhg::BatchSettings settings;
  settings.iterations = 1;
  settings.backend = pdhg::BatchBackendKind::kDevice;
  *out = pdhg::solve_batch(batch_of(model, boxes), settings);
  return out->on_device;
}

TEST(BatchPdhg, EveryBoundIsAtMostTheExactOptimumAtAnyBudget) {
  for (const Count iterations : {Count{0}, Count{7}, Count{64}, Count{300}, Count{3000}}) {
    const Tally t = sweep(iterations, pdhg::BatchBackendKind::kCpu, 120,
                          static_cast<std::uint64_t>(5200 + iterations));
    report("CPU", iterations, t);
    EXPECT_GT(t.optimal, 200);
    EXPECT_GT(t.unbounded + t.infeasible, 20);
    if (iterations == 3000) {
      // Useful, not merely valid: at a full budget most of these tiny LPs are bounded
      // to within 1e-6 of their optimum.
      EXPECT_GT(t.tight, t.optimal / 2);
    }
  }
}

TEST(BatchPdhg, DeviceBoundsAreAtMostTheExactOptimum) {
  pdhg::BatchResult probe;
  if (!device_ran(&probe)) GTEST_SKIP() << "no device: " << probe.note;
  for (const Count iterations : {Count{0}, Count{64}, Count{3000}}) {
    const Tally t = sweep(iterations, pdhg::BatchBackendKind::kDevice, 60,
                          static_cast<std::uint64_t>(7700 + iterations));
    report("device", iterations, t);
    EXPECT_GT(t.optimal, 100);
  }
}

/// A larger sparse LP with a known feasible point, for the bit-for-bit comparisons: enough
/// rows and columns for several chunks of the restart statistics.
Model sparse_model(std::mt19937_64& rng, Index rows, Index cols) {
  Model model;
  model.resize_columns(cols);
  model.resize_rows(rows);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  std::uniform_int_distribution<Index> column(0, cols - 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> point(static_cast<std::size_t>(cols));
  for (double& p : point) p = 4.0 * unit(rng);
  model.matrix.reset(rows, cols);
  std::vector<double> activity(static_cast<std::size_t>(rows), 0.0);
  for (Index i = 0; i < rows; ++i) {
    for (int e = 0; e < 6; ++e) {
      const Index j = column(rng);
      const double a = value(rng);
      model.matrix.add_entry(i, j, a);
      activity[static_cast<std::size_t>(i)] += a * point[static_cast<std::size_t>(j)];
    }
  }
  model.matrix.finalize();
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double slack = unit(rng);
    const int kind = static_cast<int>(unit(rng) * 3.0);
    // Built from the point with a margin; the exact row activity after duplicate summing
    // may differ slightly, which only matters for feasibility, not for the comparison.
    model.row_lower[u] =
        kind == 1 ? -std::numeric_limits<double>::infinity() : activity[u] - slack;
    model.row_upper[u] =
        kind == 0 ? std::numeric_limits<double>::infinity() : activity[u] + slack;
  }
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_cost[u] = value(rng);
    model.col_lower[u] = 0.0;
    model.col_upper[u] = unit(rng) < 0.3 ? std::numeric_limits<double>::infinity() : 6.0;
  }
  return model;
}

pdhg::BatchProblem sparse_batch(std::mt19937_64& rng, const Model& model, Index count) {
  pdhg::BatchProblem problem;
  problem.model = &model;
  problem.cost = model.col_cost;
  problem.count = count;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (Index k = 0; k < count; ++k) {
    for (Index j = 0; j < model.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      double lo = model.col_lower[u];
      double hi = model.col_upper[u];
      if (unit(rng) < 0.1) {
        const double cut = std::floor(4.0 * unit(rng));
        if (unit(rng) < 0.5) {
          hi = cut;
        } else {
          lo = cut;
        }
      }
      problem.col_lower.push_back(lo);
      problem.col_upper.push_back(hi);
    }
  }
  problem.primal_start.assign(static_cast<std::size_t>(model.num_cols()), 1.0);
  problem.dual_start.assign(static_cast<std::size_t>(model.num_rows()), 0.25);
  return problem;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

TEST(BatchPdhg, ABatchOfKIsKBatchesOfOneBitForBit) {
  std::mt19937_64 rng(520001);
  const Model model = sparse_model(rng, 300, 420);
  const Index count = 9;
  const pdhg::BatchProblem batch = sparse_batch(rng, model, count);
  pdhg::BatchSettings settings;
  settings.iterations = 700;
  const pdhg::BatchResult together = pdhg::solve_batch(batch, settings);
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  int finite = 0;
  for (Index k = 0; k < count; ++k) {
    pdhg::BatchProblem one = batch;
    one.count = 1;
    const auto uk = static_cast<std::size_t>(k);
    one.col_lower.assign(batch.col_lower.begin() + static_cast<std::ptrdiff_t>(uk * n),
                         batch.col_lower.begin() + static_cast<std::ptrdiff_t>((uk + 1) * n));
    one.col_upper.assign(batch.col_upper.begin() + static_cast<std::ptrdiff_t>(uk * n),
                         batch.col_upper.begin() + static_cast<std::ptrdiff_t>((uk + 1) * n));
    const pdhg::BatchResult alone = pdhg::solve_batch(one, settings);
    EXPECT_TRUE(same_bits({together.bound[uk]}, {alone.bound[0]}))
        << "LP " << k << ": " << together.bound[uk] << " in the batch, " << alone.bound[0]
        << " alone";
    const std::vector<double> dual(
        together.dual.begin() + static_cast<std::ptrdiff_t>(uk * m),
        together.dual.begin() + static_cast<std::ptrdiff_t>((uk + 1) * m));
    EXPECT_TRUE(same_bits(dual, alone.dual)) << "LP " << k << "'s duals differ";
    EXPECT_EQ(together.restarts[uk], alone.restarts[0]);
    if (std::isfinite(alone.bound[0])) ++finite;
  }
  EXPECT_GT(finite, 0) << "no LP got a finite bound, so the comparison proves little";
}

TEST(BatchPdhg, DeviceEqualsTheCpuReferenceBitForBit) {
  pdhg::BatchResult probe;
  if (!device_ran(&probe)) GTEST_SKIP() << "no device: " << probe.note;
  std::mt19937_64 rng(520002);
  // 1,300 rows and 1,700 columns: six and seven chunks of the restart statistics.
  const Model model = sparse_model(rng, 1300, 1700);
  const pdhg::BatchProblem batch = sparse_batch(rng, model, 24);
  pdhg::BatchSettings settings;
  settings.iterations = 1000;
  settings.backend = pdhg::BatchBackendKind::kCpu;
  const pdhg::BatchResult cpu = pdhg::solve_batch(batch, settings);
  settings.backend = pdhg::BatchBackendKind::kDevice;
  const pdhg::BatchResult device = pdhg::solve_batch(batch, settings);
  ASSERT_TRUE(device.on_device) << device.note;
  EXPECT_TRUE(same_bits(cpu.bound, device.bound));
  EXPECT_TRUE(same_bits(cpu.dual, device.dual));
  EXPECT_EQ(cpu.restarts, device.restarts);
  int finite = 0;
  double largest = 0.0;
  for (std::size_t k = 0; k < cpu.bound.size(); ++k) {
    if (std::isfinite(cpu.bound[k])) ++finite;
    if (std::isfinite(cpu.bound[k]) && std::isfinite(device.bound[k])) {
      largest = std::max(largest, std::fabs(cpu.bound[k] - device.bound[k]));
    }
  }
  std::printf(
      "[  INFO    ] device vs CPU: %zu LPs, %d finite bounds, largest difference %.3e; "
      "%.3fs CPU, %.3fs device\n",
      cpu.bound.size(), finite, largest, cpu.seconds, device.seconds);
  EXPECT_GT(finite, 0);
}

}  // namespace
}  // namespace sankhya::oracle
