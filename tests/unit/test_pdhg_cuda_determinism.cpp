// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU PDHG under deterministic=true is bit-for-bit repeatable (#478, #383).
//
// The CUDA engine used to sum the step-rule scalars with atomicAdd, whose order the scheduler
// picks, and to run A^T y as cuSPARSE's transpose product, which the cuSPARSE documentation
// does not promise to be repeatable; two runs on one card took different trajectories
// (#448). Under deterministic=true it now sums in a fixed order and runs both products as
// non-transpose CSR_ALG2 products on an explicit A^T (src/gpu/pdhg_reduce.cuh,
// src/gpu/pdhg_gpu.cu). What that buys is checked here the only way it can be: solve the
// same model twice and compare the answers to the bit - iteration count, objective, every
// primal value and every dual - on the per-iteration path and on the CUDA-graph device loop.
//
// Skipped, not passed, when the build has no CUDA or the machine has no device.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

std::string repository_path(const std::string& relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

Options deterministic_gpu(bool device_loop, long long iteration_limit) {
  Options o;
  o.set_bool("log_to_console", true);  // the log is read below to prove which path ran
  o.set_string("algorithm", "pdhg");
  o.set_bool("gpu", true);
  o.set_bool("deterministic", true);
  o.set_bool("gpu_on_device_loop", device_loop);
  // The polish hands the point to the interior point; the claim here is about PDHG.
  o.set_bool("pdhg_polish", false);
  o.set_double("pdhg_tolerance", 1e-8);
  o.set_int("iteration_limit", iteration_limit);
  return o;
}

bool cuda_ran(const Solution& s) {
  return s.algorithm.find("cuda") != std::string::npos;
}

struct LoggedRun {
  Solution solution;
  std::string log;
};

LoggedRun solve_logged(const Model& model, const Options& options) {
  ::testing::internal::CaptureStdout();
  LoggedRun run;
  run.solution = solve(model, options);
  std::fflush(stdout);
  run.log = ::testing::internal::GetCapturedStdout();
  return run;
}

/// The first index at which two vectors differ in their bits, or -1.
long long first_difference(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(double)) != 0) return static_cast<long long>(k);
  }
  return -1;
}

bool skip_without_a_device() {
  Model probe;
  if (!io::read_model(repository_path("data/netlib/afiro.mps"), &probe).ok) return true;
  Options options = deterministic_gpu(false, 1);
  options.set_bool("log_to_console", false);
  return !cuda_ran(solve(probe, options));
}

struct Instance {
  const char* label;
  const char* path;
  long long iteration_limit;
};

// A Netlib instance solved to 1e-8, a larger Netlib one and a 3,000-row staircase from the
// scale generator (data/scale/README.md) held to a fixed iteration budget: the budget is
// enough for several restarts, which is where a one-ULP difference used to turn into a
// different trajectory, and bounds the test's time on a card.
const std::vector<Instance> kInstances = {
    {"stocfor1", "data/netlib/stocfor1.mps", 1000000},
    {"25fv47", "data/netlib/25fv47.mps", 20000},
    {"staircase-3000", "data/scale/staircase-3000-seed7.mps.gz", 20000},
};

TEST(PdhgCudaDeterminism, TwoSolvesGiveTheSameBitsOnBothDevicePaths) {
  if (skip_without_a_device()) {
    GTEST_SKIP() << "CUDA backend or device not present: skipped, not passed.";
  }
  for (const Instance& instance : kInstances) {
    Model model;
    const io::ReadResult read = io::read_model(repository_path(instance.path), &model);
    ASSERT_TRUE(read.ok) << instance.path << ": " << read.error;
    for (const bool device_loop : {false, true}) {
      const Options options = deterministic_gpu(device_loop, instance.iteration_limit);
      const LoggedRun first = solve_logged(model, options);
      const LoggedRun second = solve_logged(model, options);
      const std::string what =
          std::string(instance.label) + (device_loop ? " (device loop)" : " (per-iteration)");
      ASSERT_TRUE(cuda_ran(first.solution)) << what << ": " << first.solution.message;
      EXPECT_NE(first.log.find("GPU PDHG: deterministic"), std::string::npos)
          << what << ": the deterministic device path did not run\n"
          << first.log.substr(0, 2000);
      if (device_loop) {
        EXPECT_NE(first.log.find("device loop on"), std::string::npos)
            << what << ": the device loop did not run";
      }
      const Solution& a = first.solution;
      const Solution& b = second.solution;
      std::cout << what << ": " << to_string(a.status) << " / " << to_string(b.status)
                << ", iterations " << a.iterations << " / " << b.iterations << ", objective "
                << a.objective << "\n";
      EXPECT_EQ(a.status, b.status) << what;
      EXPECT_EQ(a.iterations, b.iterations) << what;
      EXPECT_EQ(std::memcmp(&a.objective, &b.objective, sizeof(double)), 0)
          << what << ": objectives " << a.objective << " and " << b.objective;
      EXPECT_EQ(first_difference(a.col_value, b.col_value), -1)
          << what << ": the primal points differ";
      EXPECT_EQ(first_difference(a.row_dual, b.row_dual), -1) << what << ": the duals differ";
      EXPECT_EQ(first_difference(a.col_dual, b.col_dual), -1)
          << what << ": the reduced costs differ";
    }
  }
}

TEST(PdhgCudaDeterminism, ThePerIterationPathAndTheDeviceLoopReachTheSameAnswer) {
  // #478 item 4. Each path is repeatable on its own (above); against each other they are held
  // to the stopping tolerance, not to the bit, and the iteration counts are printed rather
  // than asserted equal. The two paths apply the same formulas to the same products, but
  // (a) the per-iteration path evaluates at every 40th accepted step and the device loop at
  // the first 32-iteration block boundary past it, so restarts are decided at different
  // iterates, and (b) the step rule's pow() runs in the host libm on one path and in CUDA's
  // on the other, which the CUDA Programming Guide allows to differ by up to 2 ulp. Either is
  // a different trajectory from that point on, so equal counts are not expected. share2b is
  // left out: neither engine meets 1e-8 on it inside the budget (test_pdhg_cuda_regression).
  if (skip_without_a_device()) {
    GTEST_SKIP() << "CUDA backend or device not present: skipped, not passed.";
  }
  const std::vector<const char*> names = {"afiro", "sc50a", "sc50b",    "adlittle",
                                          "blend", "sc105", "stocfor1", "israel"};
  constexpr double kAgreementTol = 1e-8;  // the stopping tolerance both runs are asked for
  int agreed = 0;
  int equal_counts = 0;
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(
        io::read_model(repository_path(std::string("data/netlib/") + name + ".mps"), &model).ok)
        << name;
    const Solution host = solve_logged(model, deterministic_gpu(false, 1000000)).solution;
    const Solution loop = solve_logged(model, deterministic_gpu(true, 1000000)).solution;
    ASSERT_TRUE(cuda_ran(host) && cuda_ran(loop)) << name;
    const bool both =
        (host.status == SolveStatus::kOptimal || host.status == SolveStatus::kFeasible) &&
        (loop.status == SolveStatus::kOptimal || loop.status == SolveStatus::kFeasible);
    EXPECT_TRUE(both) << name << ": per-iteration " << host.message << "; device loop "
                      << loop.message;
    const double scale = std::max({1.0, std::fabs(host.objective), std::fabs(loop.objective)});
    const double diff = std::fabs(host.objective - loop.objective);
    EXPECT_LE(diff, kAgreementTol * scale)
        << name << ": per-iteration " << host.objective << ", device loop " << loop.objective;
    if (both && diff <= kAgreementTol * scale) ++agreed;
    if (host.iterations == loop.iterations) ++equal_counts;
    std::cout << name << ": per-iteration " << host.iterations << " iterations, device loop "
              << loop.iterations << " iterations, relative objective difference "
              << diff / scale << "\n";
  }
  std::cout << "per-iteration against device loop, deterministic: " << agreed << "/"
            << names.size() << " agree at " << kAgreementTol << ", " << equal_counts << "/"
            << names.size() << " with equal iteration counts\n";
  EXPECT_EQ(agreed, static_cast<int>(names.size()));
}

}  // namespace
}  // namespace sankhya
