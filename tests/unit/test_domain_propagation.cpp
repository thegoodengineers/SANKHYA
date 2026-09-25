// SPDX-License-Identifier: Apache-2.0
// SANKHYA - synchronous activity-based bound propagation, CPU reference and GPU (#510).
//
// Four obligations: the rule itself on a case worked by hand; validity, checked against every
// integer point of small random boxes (no point that satisfies the rows may be cut off); the
// CUDA propagator returns the SAME bounds as the CPU reference, bit for bit, on random models,
// on the Netlib models in the repository and on every MIPLIB instance fetched into
// data/miplib and data/miplib-tier2 (bench/runners/fetch_miplib.py; skipped, and said so,
// without a card or without the files); and a search with root propagation on reaches the
// exact MILP optimum.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "mip/domain_propagation.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#include "gpu/domain_prop.hpp"
#endif

namespace sankhya::mip {
namespace {

/// Integer columns in [lo, hi], rows given densely with [row_lo, row_hi].
Model integer_model(const std::vector<std::vector<double>>& a,
                    const std::vector<double>& row_lo, const std::vector<double>& row_hi,
                    double lo, double hi) {
  const auto m = static_cast<Index>(a.size());
  const auto n = static_cast<Index>(a.front().size());
  Model model;
  model.resize_columns(n);
  model.resize_rows(m);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_lower[u] = lo;
    model.col_upper[u] = hi;
    model.col_type[u] = VarType::kInteger;
  }
  model.row_lower = row_lo;
  model.row_upper = row_hi;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  return model;
}

TEST(DomainPropagation, AWorkedCase) {
  // x + y <= 3, x - y >= 1, both integer in [0, 10]. Round 1: x, y <= 3 from the first row,
  // x >= 1 from the second. Round 2: y <= 3 - 1 = 2 from the first, and from the second
  // y <= x - 1 <= 2. Nothing moves in round 3.
  Model model =
      integer_model({{1.0, 1.0}, {1.0, -1.0}}, {-kInfinity, 1.0}, {3.0, kInfinity}, 0.0, 10.0);
  std::vector<double> lo = model.col_lower;
  std::vector<double> hi = model.col_upper;
  const JacobiPropagation r =
      propagate_jacobi(model, row_major(model), &lo, &hi, 50, tol::kIntegrality);
  ASSERT_FALSE(r.infeasible);
  EXPECT_EQ(lo[0], 1.0);
  EXPECT_EQ(hi[0], 3.0);
  EXPECT_EQ(lo[1], 0.0);
  EXPECT_EQ(hi[1], 2.0);
  EXPECT_GE(r.rounds, 2);
}

TEST(DomainPropagation, AnEmptyBoxIsReported) {
  // x + y >= 5 with x, y in [0, 2]: the maximum activity 4 is below 5.
  Model model = integer_model({{1.0, 1.0}}, {5.0}, {kInfinity}, 0.0, 2.0);
  std::vector<double> lo = model.col_lower;
  std::vector<double> hi = model.col_upper;
  EXPECT_TRUE(
      propagate_jacobi(model, row_major(model), &lo, &hi, 50, tol::kIntegrality).infeasible);
}

TEST(DomainPropagation, NoIntegerPointOfTheRowsIsCutOff) {
  std::mt19937_64 rng(5100);
  std::uniform_int_distribution<int> coef(-4, 4);
  std::uniform_int_distribution<int> rhs(-6, 10);
  int boxes = 0;
  int points = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const std::size_t n = 3;
    const std::size_t m = 2 + static_cast<std::size_t>(trial % 3);
    std::vector<std::vector<double>> a(m, std::vector<double>(n));
    std::vector<double> row_lo(m);
    std::vector<double> row_hi(m);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) a[i][j] = coef(rng);
      const int b = rhs(rng);
      row_lo[i] = trial % 2 == 0 ? -kInfinity : b - 6;
      row_hi[i] = b;
    }
    const Model model = integer_model(a, row_lo, row_hi, -3.0, 4.0);
    std::vector<double> lo = model.col_lower;
    std::vector<double> hi = model.col_upper;
    const JacobiPropagation r =
        propagate_jacobi(model, row_major(model), &lo, &hi, 50, tol::kIntegrality);
    ++boxes;
    // Every integer point of the ORIGINAL box that satisfies every row must survive.
    for (int x0 = -3; x0 <= 4; ++x0) {
      for (int x1 = -3; x1 <= 4; ++x1) {
        for (int x2 = -3; x2 <= 4; ++x2) {
          const int x[3] = {x0, x1, x2};
          bool feasible = true;
          for (std::size_t i = 0; i < m && feasible; ++i) {
            double act = 0.0;
            for (std::size_t j = 0; j < n; ++j) act += a[i][j] * x[j];
            feasible = act >= row_lo[i] && act <= row_hi[i];
          }
          if (!feasible) continue;
          ++points;
          ASSERT_FALSE(r.infeasible) << "trial " << trial << ": a feasible point exists";
          for (std::size_t j = 0; j < n; ++j) {
            EXPECT_GE(x[j], lo[j]) << "trial " << trial;
            EXPECT_LE(x[j], hi[j]) << "trial " << trial;
          }
        }
      }
    }
  }
  EXPECT_GT(points, 1000) << boxes << " boxes";
}

#ifdef SANKHYA_ENABLE_CUDA
const std::filesystem::path kRepo =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();

/// The device and the CPU reference from the same box, `rounds_limit` rounds at most: the same
/// verdict, rounds and count, and every bound the same 64 bits (so -0.0 against 0.0, which
/// operator== would accept, is a difference). Returns what the CPU reference did.
JacobiPropagation same_bounds(const Model& model, const std::string& name, int rounds_limit) {
  std::vector<double> lo = model.col_lower;
  std::vector<double> hi = model.col_upper;
  const JacobiPropagation cpu =
      propagate_jacobi(model, row_major(model), &lo, &hi, rounds_limit, tol::kIntegrality);
  const gpu::PropResult device = gpu::propagate_bounds(model, model.col_lower, model.col_upper,
                                                       rounds_limit, tol::kIntegrality);
  EXPECT_TRUE(device.ran) << name;
  if (!device.ran) return cpu;
  EXPECT_EQ(device.infeasible, cpu.infeasible) << name;
  if (cpu.infeasible || device.infeasible) return cpu;
  EXPECT_EQ(device.rounds, cpu.rounds) << name;
  EXPECT_EQ(device.tightened, cpu.tightened) << name;
  int differing = 0;
  for (std::size_t j = 0; j < lo.size(); ++j) {
    const bool lower_same =
        std::bit_cast<std::uint64_t>(device.col_lb[j]) == std::bit_cast<std::uint64_t>(lo[j]);
    const bool upper_same =
        std::bit_cast<std::uint64_t>(device.col_ub[j]) == std::bit_cast<std::uint64_t>(hi[j]);
    if (lower_same && upper_same) continue;
    if (++differing <= 5) {
      ADD_FAILURE() << name << " column " << j << ": device [" << device.col_lb[j] << ", "
                    << device.col_ub[j] << "], CPU [" << lo[j] << ", " << hi[j] << "]";
    }
  }
  EXPECT_EQ(differing, 0) << name << ": columns whose bounds differ in any bit";
  return cpu;
}
#endif

TEST(DomainPropagation, TheGpuReturnsTheCpuBounds) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  std::mt19937_64 rng(5101);
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::uniform_int_distribution<int> pick(0, 3);
  for (int trial = 0; trial < 60; ++trial) {
    const auto n = static_cast<std::size_t>(20 + trial);
    const auto m = static_cast<std::size_t>(10 + trial / 2);
    std::vector<std::vector<double>> a(m, std::vector<double>(n, 0.0));
    std::vector<double> row_lo(m);
    std::vector<double> row_hi(m);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (pick(rng) == 0) a[i][j] = value(rng);
      }
      row_hi[i] = 3.0 * static_cast<double>(n) / 4.0 + value(rng);
      row_lo[i] = pick(rng) == 0 ? row_hi[i] - 10.0 : -kInfinity;
    }
    Model model = integer_model(a, row_lo, row_hi, 0.0, 9.0);
    for (std::size_t j = 0; j < n; j += 3) model.col_type[j] = VarType::kContinuous;
    same_bounds(model, "random " + std::to_string(trial), 50);
  }
  const std::filesystem::path netlib = kRepo / "data/netlib";
  int compared = 0;
  for (const char* name : {"afiro", "sc50a", "sc105", "blend", "stocfor1", "adlittle",
                           "share2b", "scagr7", "boeing2", "brandy"}) {
    Model model;
    if (!io::read_model((netlib / (std::string(name) + ".mps")).string(), &model).ok) continue;
    same_bounds(model, name, 50);
    ++compared;
  }
  EXPECT_GE(compared, 5);
#endif
}

TEST(DomainPropagation, TheGpuReturnsTheCpuBoundsOnMiplib) {
  // The issue's acceptance item: "identical fixpoint bounds to the CPU propagator on the
  // MIPLIB set". MIPLIB is fetched, not tracked (data/miplib* is gitignored), so this runs
  // over whatever `bench/runners/fetch_miplib.py` (and `--tier 2`) put there, every file of
  // it, with a round cap far above the solver's own so each model is taken to its fixpoint.
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  std::vector<std::filesystem::path> files;
  for (const char* dir : {"data/miplib", "data/miplib-tier2"}) {
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(kRepo / dir, error)) {
      const std::string file = entry.path().filename().string();
      if (file.ends_with(".mps.gz") || file.ends_with(".mps")) files.push_back(entry.path());
    }
  }
  if (files.empty()) {
    GTEST_SKIP() << "no MIPLIB instance under data/miplib or data/miplib-tier2 (run "
                    "bench/runners/fetch_miplib.py): skipped, not passed.";
  }
  std::sort(files.begin(), files.end());
  constexpr int kRoundCap = 1000;
  int compared = 0;
  int at_fixpoint = 0;
  for (const auto& path : files) {
    Model model;
    const std::string name = path.filename().string();
    const auto read = io::read_model(path.string(), &model);
    ASSERT_TRUE(read.ok) << name << ": " << read.error;
    const JacobiPropagation cpu = same_bounds(model, name, kRoundCap);
    ++compared;
    if (cpu.rounds < kRoundCap) ++at_fixpoint;
    std::printf("[ miplib    ] %-28s %6d rows %6d cols %4d round(s) %6lld tightened%s\n",
                name.c_str(), static_cast<int>(model.num_rows()),
                static_cast<int>(model.num_cols()), cpu.rounds,
                static_cast<long long>(cpu.tightened), cpu.infeasible ? " (box empty)" : "");
  }
  std::printf(
      "[ miplib    ] %d instance(s) compared bit for bit, %d at a fixpoint within %d "
      "rounds\n",
      compared, at_fixpoint, kRoundCap);
  EXPECT_EQ(compared, static_cast<int>(files.size()));
#endif
}

TEST(DomainPropagation, ASearchWithRootPropagationReachesTheExactOptimum) {
  std::mt19937_64 rng(5102);
  oracle::GeneratorConfig config;
  config.bounded_column_probability = 1.0;
  int compared = 0;
  for (int k = 0; k < 40; ++k) {
    oracle::GeneratedLp lp = oracle::kkt_lp(rng, config).lp;
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    Model model = oracle::to_model(lp);
    for (auto& t : model.col_type) t = VarType::kInteger;
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("presolve", false);
    options.set_bool("gpu_domain_prop", true);
    const Solution solved = solve(model, options);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(solved.objective, exact.objective.to_double(), 1e-6) << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 10);
}

}  // namespace
}  // namespace sankhya::mip
