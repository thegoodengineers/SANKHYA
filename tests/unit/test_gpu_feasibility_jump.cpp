// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU Feasibility Jump (#508).
//
// The device search is held to the CPU reference (src/mip/feasibility_jump.cpp) on what
// matters, not on its path: where the CPU search finds a feasible point the device search
// finds one too; every point it returns is re-measured in this file against every row, bound
// and integrality; on a model with no integer point it returns nothing; a run is reproducible
// for a seed. Then the branch and bound with gpu_feasibility_jump=true is judged by the exact
// rational oracle, and with it off (the default) the solve is bit for bit the one without it.
// The device tests skip, and say so, in a build without CUDA or on a machine without a card.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "mip/feasibility_jump.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#include "gpu/gpu_fj.hpp"
#endif

namespace sankhya::mip {
namespace {

Model make_model(const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                 const std::vector<double>& cost, const std::vector<double>& lower,
                 const std::vector<double>& upper, const std::vector<bool>& integer) {
  Model m;
  const auto n = cost.size();
  m.col_cost = cost;
  m.col_lower = lower;
  m.col_upper = upper;
  for (std::size_t j = 0; j < n; ++j) {
    m.col_type.push_back(integer[j] ? VarType::kInteger : VarType::kContinuous);
  }
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

/// Independent of the code under test: integral, inside the box, every row satisfied to the
/// absolute tolerance offer_incumbent() applies, summed densely here.
bool feasible(const Model& m, const std::vector<double>& x, std::string* why) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) {
    *why = "wrong length";
    return false;
  }
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger && x[j] != std::round(x[j])) {
      *why = "column " + std::to_string(j) + " fractional";
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) {
      *why = "column " + std::to_string(j) + " outside its box";
      return false;
    }
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-7 || activity > m.row_upper[u] + 1e-7) {
      *why = "row " + std::to_string(i) + " violated, activity " + std::to_string(activity);
      return false;
    }
  }
  return true;
}

[[maybe_unused]] double objective(const Model& m, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) value += m.col_cost[j] * x[j];
  return value;
}

[[maybe_unused]] FeasibilityJumpSettings settings_for(Count work, std::uint64_t seed) {
  FeasibilityJumpSettings settings;
  settings.work_limit = work;
  settings.seed = seed;
  return settings;
}

[[maybe_unused]] FeasibilityJumpResult run_cpu(const Model& m, Count work = 200000,
                                               std::uint64_t seed = 1) {
  return feasibility_jump(m, feasibility_jump_zero_start(m), settings_for(work, seed));
}

oracle::GeneratedLp random_milp(std::mt19937_64& rng, int trial) {
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  oracle::GeneratedLp lp = oracle::random_lp(rng, config);
  lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
  for (Index j = 0; j < lp.num_cols; ++j) {
    lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 3 == 0 ? 0 : 1;
  }
  return lp;
}

Model as_milp(const oracle::GeneratedLp& lp) {
  Model model = oracle::to_model(lp);
  for (Index j = 0; j < lp.num_cols; ++j) {
    if (lp.integral[static_cast<std::size_t>(j)] != 0) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
  return model;
}

/// The hand models of the CPU tests (test_feasibility_jump.cpp), each with a feasible point.
[[maybe_unused]] std::vector<std::pair<std::string, Model>> hand_models() {
  std::vector<std::pair<std::string, Model>> out;
  out.emplace_back("cover",
                   make_model({{1, 1, 0, 0, 0, 0, 0, 0},
                               {0, 1, 1, 0, 0, 0, 0, 1},
                               {0, 0, 1, 1, 0, 0, 0, 0},
                               {1, 0, 0, 0, 1, 1, 0, 0},
                               {0, 0, 0, 0, 0, 1, 1, 0},
                               {0, 0, 0, 1, 0, 0, 1, 1}},
                              std::vector<double>(6, 1.0), std::vector<double>(6, kInfinity),
                              std::vector<double>(8, 1.0), std::vector<double>(8, 0.0),
                              std::vector<double>(8, 1.0), std::vector<bool>(8, true)));
  out.emplace_back(
      "general integer equalities",
      make_model({{1, 1, 1}, {1, -1, 0}, {0, 1, 1}}, {7.0, 1.0, 3.0}, {7.0, 1.0, kInfinity},
                 {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, {20.0, 20.0, 20.0}, {true, true, true}));
  out.emplace_back("integer and continuous",
                   make_model({{1, 1}, {2, -5}}, {2.5, 5.5}, {kInfinity, 5.5}, {1.0, 1.0},
                              {0.0, 0.0}, {10.0, 0.3}, {true, false}));
  constexpr int k = 4;
  std::vector<std::vector<double>> rows;
  for (int r = 0; r < k; ++r) {
    std::vector<double> row(k * k, 0.0);
    for (int c = 0; c < k; ++c) row[static_cast<std::size_t>(r * k + c)] = 1.0;
    rows.push_back(row);
  }
  for (int c = 0; c < k; ++c) {
    std::vector<double> row(k * k, 0.0);
    for (int r = 0; r < k; ++r) row[static_cast<std::size_t>(r * k + c)] = 1.0;
    rows.push_back(row);
  }
  std::vector<double> cost;
  for (int e = 0; e < k * k; ++e) cost.push_back(static_cast<double>((e * 7) % 11));
  out.emplace_back(
      "assignment",
      make_model(rows, std::vector<double>(2 * k, 1.0), std::vector<double>(2 * k, 1.0), cost,
                 std::vector<double>(k * k, 0.0), std::vector<double>(k * k, 1.0),
                 std::vector<bool>(k * k, true)));
  Model knapsack =
      make_model({{5, 7, 4, 4}}, {-kInfinity}, {12.0}, {10.0, 13.0, 7.0, 8.0},
                 {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0}, {true, true, true, true});
  knapsack.sense = ObjSense::kMaximize;
  out.emplace_back("knapsack, maximised", knapsack);
  return out;
}

#ifdef SANKHYA_ENABLE_CUDA
gpu::FjDeviceResult run_gpu(const Model& m, Count work = 200000, std::uint64_t seed = 1,
                            int restarts = 0) {
  return gpu::feasibility_jump(m, feasibility_jump_zero_start(m), settings_for(work, seed),
                               restarts);
}

void expect_all_feasible(const Model& m, const gpu::FjDeviceResult& r,
                         const std::string& name) {
  EXPECT_EQ(r.rejected, 0) << name << ": the host check refused a device point";
  for (std::size_t k = 0; k < r.search.points.size(); ++k) {
    std::string why;
    EXPECT_TRUE(feasible(m, r.search.points[k], &why)) << name << " point " << k << ": " << why;
    if (k > 0) {
      const double sense = m.sense == ObjSense::kMaximize ? -1.0 : 1.0;
      EXPECT_LT(sense * objective(m, r.search.points[k]),
                sense * objective(m, r.search.points[k - 1]))
          << name << ": each point handed over improves on the last";
    }
  }
}
#endif

#ifdef SANKHYA_ENABLE_CUDA
#define SKIP_WITHOUT_DEVICE() \
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed."
#else
#define SKIP_WITHOUT_DEVICE() \
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed."
#endif

TEST(GpuFeasibilityJump, FindsAPointOnEveryHandModelTheCpuSearchSolves) {
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  for (const auto& [name, m] : hand_models()) {
    ASSERT_FALSE(run_cpu(m).points.empty()) << name << ": the CPU reference finds a point";
    const gpu::FjDeviceResult r = run_gpu(m);
    ASSERT_TRUE(r.ran) << name << ": " << r.reason;
    EXPECT_FALSE(r.search.points.empty())
        << name << ": " << r.search.moves << " moves, " << r.search.weight_updates
        << " updates over " << r.restarts << " restarts";
    expect_all_feasible(m, r, name);
  }
#endif
}

TEST(GpuFeasibilityJump, SolvesTheModelsTheCpuSearchSolvesOnRandomMilps) {
  // The fuzz of the CPU test, with the device search alongside: on every random MILP where
  // the CPU search finds a point the device search must find one too, whatever it returns is
  // feasible, and on a MILP the exact oracle proves infeasible it returns nothing.
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  std::mt19937_64 rng(5080);
  int cpu_found = 0;
  int gpu_found = 0;
  int infeasible_checked = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const Model model = as_milp(lp);
    const auto seed = static_cast<std::uint64_t>(trial);
    const FeasibilityJumpResult cpu = run_cpu(model, 100000, seed);
    const gpu::FjDeviceResult device = run_gpu(model, 100000, seed, 16);
    ASSERT_TRUE(device.ran) << device.reason;
    expect_all_feasible(model, device, "trial " + std::to_string(trial));
    cpu_found += cpu.points.empty() ? 0 : 1;
    gpu_found += device.search.points.empty() ? 0 : 1;
    if (!cpu.points.empty()) {
      EXPECT_FALSE(device.search.points.empty())
          << "trial " << trial << ": the CPU search found a point, the device search none\n"
          << lp.to_text();
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      ++infeasible_checked;
      EXPECT_TRUE(device.search.points.empty())
          << "trial " << trial << ": a point on an infeasible MILP";
    }
  }
  std::printf("[ gpu fj   ] random MILPs with a point: CPU %d, GPU %d of 300; %d infeasible\n",
              cpu_found, gpu_found, infeasible_checked);
  EXPECT_GT(cpu_found, 50);
  EXPECT_GE(gpu_found, cpu_found);
  EXPECT_GT(infeasible_checked, 20);
#endif
}

TEST(GpuFeasibilityJump, ReturnsNothingWhenNoIntegerPointExists) {
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  const Model parity =
      make_model({{2, 2}}, {3.0}, {3.0}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  const gpu::FjDeviceResult a = run_gpu(parity, 50000);
  EXPECT_TRUE(a.ran);
  EXPECT_TRUE(a.search.points.empty());
  const Model cover = make_model({{1, 1}}, {3.0}, {kInfinity}, {1.0, 1.0}, {0.0, 0.0},
                                 {1.0, 1.0}, {true, true});
  EXPECT_TRUE(run_gpu(cover, 50000).search.points.empty());
  const Model empty_box = make_model({{1}}, {0.0}, {kInfinity}, {1.0}, {0.2}, {0.8}, {true});
  const gpu::FjDeviceResult r = run_gpu(empty_box);
  EXPECT_TRUE(r.ran);
  EXPECT_TRUE(r.search.points.empty());
  EXPECT_EQ(r.search.work, 0);
#endif
}

TEST(GpuFeasibilityJump, HandsEachPointOverAsItIsFound) {
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  Model m = make_model({{5, 7, 4, 4}}, {-kInfinity}, {12.0}, {10.0, 13.0, 7.0, 8.0},
                       {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0}, {true, true, true, true});
  m.sense = ObjSense::kMaximize;
  std::vector<std::vector<double>> seen;
  FeasibilityJumpSettings settings = settings_for(200000, 1);
  settings.on_point = [&seen](const std::vector<double>& x) { seen.push_back(x); };
  const gpu::FjDeviceResult r =
      gpu::feasibility_jump(m, feasibility_jump_zero_start(m), settings);
  ASSERT_FALSE(r.search.points.empty());
  EXPECT_EQ(seen, r.search.points);
  // 13 + 8 (b and d, weight 11) is the optimum; the objective phase must get above zero.
  EXPECT_GT(objective(m, r.search.points.back()), 0.0);
  expect_all_feasible(m, r, "knapsack");
#endif
}

TEST(GpuFeasibilityJump, IsReproducibleForASeed) {
  // No atomics and counter-based draws: the same seed, restarts and budget give the same
  // points, moves and work, bit for bit.
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  std::mt19937_64 rng(5081);
  for (int trial = 0; trial < 20; ++trial) {
    const Model model = as_milp(random_milp(rng, trial));
    const gpu::FjDeviceResult a = run_gpu(model, 100000, 7, 12);
    const gpu::FjDeviceResult b = run_gpu(model, 100000, 7, 12);
    EXPECT_EQ(a.search.points, b.search.points) << "trial " << trial;
    EXPECT_EQ(a.search.moves, b.search.moves) << "trial " << trial;
    EXPECT_EQ(a.search.work, b.search.work) << "trial " << trial;
    EXPECT_EQ(a.search.weight_updates, b.search.weight_updates) << "trial " << trial;
  }
#endif
}

TEST(GpuFeasibilityJump, StopsWhenAskedBetweenLaunches) {
  SKIP_WITHOUT_DEVICE();
#ifdef SANKHYA_ENABLE_CUDA
  // No integer point, an unbounded budget: only the stop can end it.
  const Model parity =
      make_model({{2, 2}}, {3.0}, {3.0}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  FeasibilityJumpSettings settings = settings_for(std::numeric_limits<Count>::max(), 1);
  int polls = 0;
  settings.should_stop = [&polls]() { return ++polls >= 3; };
  const gpu::FjDeviceResult r =
      gpu::feasibility_jump(parity, feasibility_jump_zero_start(parity), settings, 4);
  EXPECT_TRUE(r.ran);
  EXPECT_EQ(polls, 3);
  EXPECT_EQ(r.launches, 3);
  EXPECT_TRUE(r.search.points.empty());
#endif
}

// ---- Through the branch and bound: every build, with or without a card. -----------------

Options fj_options() {
  Options options;
  options.set_string("mip_heur_fj", "on");
  options.set_bool("presolve", false);
  options.set_double("mip_relative_gap", 0.0);
  options.set_double("mip_absolute_gap", 0.0);
  return options;
}

TEST(GpuFeasibilityJump, OffByDefaultLeavesTheSolveBitForBitUnchanged) {
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("gpu_feasibility_jump"));
  std::mt19937_64 rng(5082);
  const Options unset = fj_options();
  Options off = fj_options();
  off.set_bool("gpu_feasibility_jump", false);
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = as_milp(random_milp(rng, trial));
    const Solution a = solve(model, unset);
    const Solution b = solve(model, off);
    EXPECT_EQ(a.status, b.status) << "trial " << trial;
    EXPECT_EQ(a.col_value, b.col_value) << "trial " << trial;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(a.objective),
              std::bit_cast<std::uint64_t>(b.objective))
        << "trial " << trial;
    EXPECT_EQ(a.nodes, b.nodes) << "trial " << trial;
  }
}

TEST(GpuFeasibilityJump, TheSearchWithItOnAgreesWithTheExactOracle) {
  // End to end with gpu_feasibility_jump=true: the device search where there is one, the
  // CPU search where there is not, its points through offer_incumbent(), and the answer
  // judged by exact arithmetic. The log shows which search ran.
  std::mt19937_64 rng(50800);
  Options options = fj_options();
  options.set_bool("gpu_feasibility_jump", true);
  options.set_bool("log_to_console", true);
  options.set_string("log_level", "verbose");
  int compared = 0;
  int on_device = 0;
  int on_cpu = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    const Model model = as_milp(lp);
    testing::internal::CaptureStdout();
    const Solution got = solve(model, options);
    std::fflush(stdout);
    const std::string log = testing::internal::GetCapturedStdout();
    if (log.find("Feasibility jump on the GPU") != std::string::npos) ++on_device;
    if (log.find("the CPU search runs") != std::string::npos) ++on_cpu;
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      EXPECT_EQ(got.status, SolveStatus::kInfeasible) << "trial " << trial << "\n"
                                                      << lp.to_text();
    } else {
      ASSERT_EQ(got.status, SolveStatus::kOptimal)
          << "trial " << trial << ": " << got.message << "\n"
          << lp.to_text();
      const double want = exact.objective.to_double();
      EXPECT_NEAR(got.objective, want, 1e-6 * std::max(1.0, std::fabs(want)))
          << "trial " << trial << "\n"
          << lp.to_text();
      std::string why;
      EXPECT_TRUE(feasible(model, got.col_value, &why)) << "trial " << trial << ": " << why;
    }
    ++compared;
  }
  std::printf("[ gpu fj   ] %d compared; FJ on the device in %d searches, on the CPU in %d\n",
              compared, on_device, on_cpu);
  EXPECT_GT(compared, 70);
  EXPECT_GT(on_device + on_cpu, 0) << "feasibility jump never ran";
#ifdef SANKHYA_ENABLE_CUDA
  if (gpu::device_available(nullptr)) {
    EXPECT_GT(on_device, 0);
    EXPECT_EQ(on_cpu, 0) << "a device is present, yet the CPU search ran";
  } else {
    EXPECT_EQ(on_device, 0);
  }
#else
  EXPECT_EQ(on_device, 0);
  EXPECT_GT(on_cpu, 0) << "without CUDA the CPU search must run in its place";
#endif
}

}  // namespace
}  // namespace sankhya::mip
