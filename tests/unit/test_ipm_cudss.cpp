// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations factored by NVIDIA cuDSS (#489).
//
// The device tests skip without a cuDSS build or a CUDA device; the fallback tests run in
// every other build, so the CPU build proves the option changes nothing there.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "gpu/cudss_factor.hpp"
#include "la/ldl.hpp"
#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

/// A cuDSS factor ready on a device, or the reason there is none.
bool device_ready(std::string* reason) {
  if (!gpu::CudssFactor::compiled()) {
    *reason = "built without SANKHYA_ENABLE_CUDSS";
    return false;
  }
  gpu::CudssFactor probe;
  return probe.initialize(reason);
}

Model committed_netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

Options ipm_options(const char* linear_solver) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("crossover", false);  // the interior point's own answer is judged
  options.set_string("ipm_linear_solver", linear_solver);
  return options;
}

/// The engine itself, with its log captured.
std::string solve_ipm_logged(const Model& model, const Options& options, Solution* out) {
  std::FILE* stream = std::tmpfile();
  EXPECT_NE(stream, nullptr);
  if (stream == nullptr) return {};
  {
    Logger logger(stream, LogLevel::kInfo);
    *out = ipm::solve_ipm(model, options, logger);
  }
  std::fflush(stream);
  std::rewind(stream);
  std::string text;
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) text += buffer;
  std::fclose(stream);
  return text;
}

/// The lower triangle of A A^T + I for a random sparse A: symmetric positive definite, with
/// the pattern normal_equations_lower() gives the interior point.
SparseMatrix random_normal_equations(Index rows, Index cols, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_int_distribution<Index> row(0, rows - 1);
  SparseMatrix a(rows, cols);
  for (Index j = 0; j < cols; ++j) {
    std::vector<Index> picked;
    for (int k = 0; k < 3; ++k) picked.push_back(row(rng));
    std::sort(picked.begin(), picked.end());
    picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
    for (const Index i : picked) a.add_entry(i, j, value(rng));
  }
  a.finalize();
  SparseMatrix lower;
  const std::vector<double> theta(static_cast<std::size_t>(cols), 1.0);
  EXPECT_TRUE(normal_equations_lower(a, theta, {}, 1.0, &lower));
  return lower;
}

/// The lower triangle of a 3 x 3 symmetric matrix [[1, 1, 0], [1, d, 0], [0, 0, 1]].
SparseMatrix three_by_three(double d) {
  SparseMatrix lower(3, 3);
  lower.add_entry(0, 0, 1.0);
  lower.add_entry(1, 0, 1.0);
  lower.add_entry(1, 1, d);
  lower.add_entry(2, 2, 1.0);
  lower.finalize();
  return lower;
}

TEST(IpmCudss, TheOptionDefaultsToTheCpuFactor) {
  const Options options;
  EXPECT_EQ(options.get_string("ipm_linear_solver"), "cpu");
}

TEST(IpmCudss, WithoutTheBackendEveryCallDeclinesWithAReason) {
  if (gpu::CudssFactor::compiled()) GTEST_SKIP() << "this build has the cuDSS backend";
  gpu::CudssFactor factor;
  std::string reason;
  EXPECT_FALSE(factor.initialize(&reason));
  EXPECT_NE(reason.find("SANKHYA_ENABLE_CUDSS"), std::string::npos) << reason;
  EXPECT_FALSE(factor.analyze(three_by_three(2.0), &reason));
  EXPECT_EQ(factor.factorize(three_by_three(2.0), 1e-10, &reason), gpu::CudssOutcome::kFailed);
  double b[3] = {1.0, 2.0, 3.0};
  EXPECT_FALSE(factor.solve(b, &reason));
  EXPECT_EQ(b[0], 1.0);
}

TEST(IpmCudss, SolvesAsTheCpuFactorDoesOnPositiveDefiniteSystems) {
  std::string reason;
  if (!device_ready(&reason)) GTEST_SKIP() << reason;
  for (const std::uint64_t seed : {1u, 2u, 3u}) {
    const SparseMatrix lower = random_normal_equations(400, 900, seed);
    SparseLdl cpu;
    ASSERT_TRUE(cpu.analyze(lower));
    ASSERT_TRUE(cpu.factorize(lower, 1e-10));
    gpu::CudssFactor device;
    ASSERT_TRUE(device.analyze(lower, &reason)) << reason;
    ASSERT_EQ(device.factorize(lower, 1e-10, &reason), gpu::CudssOutcome::kFactored) << reason;
    EXPECT_EQ(device.regularized_pivots(), 0);
    std::vector<double> x_cpu(400);
    for (std::size_t i = 0; i < x_cpu.size(); ++i) x_cpu[i] = std::sin(static_cast<double>(i));
    std::vector<double> x_device = x_cpu;
    cpu.solve(x_cpu.data());
    ASSERT_TRUE(device.solve(x_device.data(), &reason)) << reason;
    double worst = 0.0;
    double largest = 0.0;
    for (std::size_t i = 0; i < x_cpu.size(); ++i) {
      worst = std::max(worst, std::fabs(x_cpu[i] - x_device[i]));
      largest = std::max(largest, std::fabs(x_cpu[i]));
    }
    // Two backward-stable factorizations of a matrix with condition number in the tens,
    // in different orders: they agree to a few hundred ulps of the solution.
    EXPECT_LE(worst, 1e-12 * std::max(1.0, largest)) << "seed " << seed;
    // New values on the same pattern: a second factorization reuses the analysis.
    std::vector<double> doubled = lower.values();
    for (double& v : doubled) v *= 2.0;
    SparseMatrix scaled;
    scaled.assign_columns(lower.num_rows(), lower.num_cols(), lower.column_starts(),
                          lower.row_indices(), std::move(doubled));
    ASSERT_EQ(device.factorize(scaled, 1e-10, &reason), gpu::CudssOutcome::kFactored);
    std::vector<double> rhs(400);
    for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] = std::sin(static_cast<double>(i));
    ASSERT_TRUE(device.solve(rhs.data(), &reason));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      EXPECT_NEAR(rhs[i], 0.5 * x_cpu[i], 1e-12 * std::max(1.0, largest));
    }
    EXPECT_EQ(device.timing().factorizations, 2);
  }
}

TEST(IpmCudss, TheCpuPivotRuleSmallPivotsLiftedNegativeOnesDeclined) {
  std::string reason;
  if (!device_ready(&reason)) GTEST_SKIP() << reason;
  gpu::CudssFactor device;
  // An exactly singular 2 x 2 block: one zero pivot, lifted to the regularization, counted.
  ASSERT_TRUE(device.analyze(three_by_three(1.0), &reason)) << reason;
  ASSERT_EQ(device.factorize(three_by_three(1.0), 1e-8, &reason), gpu::CudssOutcome::kFactored)
      << reason;
  EXPECT_EQ(device.regularized_pivots(), 1);
  // A pivot of -1e-6 (in either elimination order): the CPU rule would lift it to +1e-8, and
  // the device, which keeps it, must say so rather than hand back an indefinite factor.
  EXPECT_EQ(device.factorize(three_by_three(1.0 - 1e-6), 1e-8, &reason),
            gpu::CudssOutcome::kIndefinite);
  EXPECT_NE(reason.find("negative pivot"), std::string::npos) << reason;
  double b[3] = {1.0, 2.0, 3.0};
  EXPECT_FALSE(device.solve(b, &reason));  // no usable factors after a decline
  // A different pattern is refused, not factored with the wrong structure.
  EXPECT_EQ(device.factorize(random_normal_equations(3, 4, 7), 1e-8, &reason),
            gpu::CudssOutcome::kFailed);
}

TEST(IpmCudss, WithoutADeviceTheCpuFactorIsKeptBitForBit) {
  std::string reason;
  if (device_ready(&reason)) GTEST_SKIP() << "a cuDSS device is present";
  for (const char* name : {"afiro", "adlittle", "sc50a", "blend", "israel"}) {
    const Model model = committed_netlib(name);
    Solution cpu;
    Solution asked;
    (void)solve_ipm_logged(model, ipm_options("cpu"), &cpu);
    const std::string log = solve_ipm_logged(model, ipm_options("cudss"), &asked);
    EXPECT_NE(log.find("ipm_linear_solver = cudss is unavailable"), std::string::npos) << log;
    EXPECT_EQ(asked.status, cpu.status) << name;
    EXPECT_EQ(asked.iterations, cpu.iterations) << name;
    EXPECT_EQ(asked.objective, cpu.objective) << name;
    EXPECT_EQ(asked.col_value, cpu.col_value) << name;
  }
}

TEST(IpmCudss, TheExcludedPathsKeepTheCpuFactorAndSaySo) {
  const Model model = committed_netlib("afiro");
  Options options = ipm_options("cudss");
  options.set_bool("ipm_proximal_regularization", true);
  Solution solution;
  const std::string log = solve_ipm_logged(model, options, &solution);
  EXPECT_NE(log.find("does not apply with ipm_proximal_regularization"), std::string::npos)
      << log;
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
}

TEST(IpmCudss, NetlibOptimaMatchTheCpuFactor) {
  std::string reason;
  if (!device_ready(&reason)) GTEST_SKIP() << reason;
  const std::regex device_line(
      R"(IPM: cuDSS (\d+) factorization\(s\) in [0-9.]+s, (\d+) solve\(s\) .*; (\d+) declined )"
      R"(and redone on the CPU; on the device to the end)");
  const char* names[] = {"afiro",    "adlittle", "sc50a",  "sc105",  "blend",   "share2b",
                         "stocfor1", "israel",   "bandm",  "scfxm1", "25fv47",  "ship04s",
                         "scsd8",    "ganges",   "pilot4", "d2q06c", "80bau3b", "greenbea"};
  int compared = 0;
  for (const char* name : names) {
    const Model model = committed_netlib(name);
    Solution cpu;
    Solution device;
    (void)solve_ipm_logged(model, ipm_options("cpu"), &cpu);
    const std::string log = solve_ipm_logged(model, ipm_options("cudss"), &device);
    std::smatch match;
    const bool used = std::regex_search(log, match, device_line);
    EXPECT_TRUE(used) << name << ": the device was not used to the end\n" << log;
    std::cout << name << ": cpu " << to_string(cpu.status) << " " << cpu.iterations
              << " it obj " << fmt::format("{:.12e}", cpu.objective) << " | cudss "
              << to_string(device.status) << " " << device.iterations << " it obj "
              << fmt::format("{:.12e}", device.objective);
    if (used) {
      std::cout << " (" << match[1].str() << " device factorizations, " << match[3].str()
                << " declined)";
      EXPECT_GT(std::stol(match[1].str()), 0) << name;
    }
    std::cout << "\n";
    // The CPU factor's verdict is the reference: where it proves optimality the device must
    // too, at the same optimum within the interior point's own accuracy.
    if (cpu.status != SolveStatus::kOptimal) continue;
    EXPECT_EQ(device.status, SolveStatus::kOptimal) << name << ": " << device.message;
    if (device.status == SolveStatus::kOptimal) {
      EXPECT_NEAR(device.objective, cpu.objective,
                  1e-6 * std::max(1.0, std::fabs(cpu.objective)))
          << name;
      ++compared;
    }
  }
  std::cout << "cuDSS against CPU factor: " << compared << " Netlib optima compared\n";
}

}  // namespace
}  // namespace sankhya
