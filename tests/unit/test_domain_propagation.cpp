// SPDX-License-Identifier: Apache-2.0
// SANKHYA - synchronous activity-based bound propagation, CPU reference and GPU (#510).
//
// Four obligations: the rule itself on a case worked by hand; validity, checked against every
// integer point of small random boxes (no point that satisfies the rows may be cut off); the
// CUDA propagator returns the SAME bounds as the CPU reference on random models and on the
// Netlib models in the repository (skipped, and said so, without a card); and a search with
// root propagation on reaches the exact MILP optimum.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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
    const int n = 3;
    const int m = 2 + trial % 3;
    std::vector<std::vector<double>> a(static_cast<std::size_t>(m),
                                       std::vector<double>(static_cast<std::size_t>(n)));
    std::vector<double> row_lo(static_cast<std::size_t>(m));
    std::vector<double> row_hi(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) a[i][j] = coef(rng);
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
          for (int i = 0; i < m && feasible; ++i) {
            double act = 0.0;
            for (int j = 0; j < n; ++j) act += a[i][j] * x[j];
            feasible = act >= row_lo[i] && act <= row_hi[i];
          }
          if (!feasible) continue;
          ++points;
          ASSERT_FALSE(r.infeasible) << "trial " << trial << ": a feasible point exists";
          for (int j = 0; j < n; ++j) {
            EXPECT_GE(x[j], lo[static_cast<std::size_t>(j)]) << "trial " << trial;
            EXPECT_LE(x[j], hi[static_cast<std::size_t>(j)]) << "trial " << trial;
          }
        }
      }
    }
  }
  EXPECT_GT(points, 1000) << boxes << " boxes";
}

TEST(DomainPropagation, TheGpuReturnsTheCpuBounds) {
#ifndef SANKHYA_ENABLE_CUDA
  GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
#else
  if (!gpu::device_available(nullptr)) GTEST_SKIP() << "no CUDA device: skipped, not passed.";
  const auto same = [](const Model& model, const std::string& name) {
    std::vector<double> lo = model.col_lower;
    std::vector<double> hi = model.col_upper;
    const JacobiPropagation cpu =
        propagate_jacobi(model, row_major(model), &lo, &hi, 50, tol::kIntegrality);
    const gpu::PropResult device =
        gpu::propagate_bounds(model, model.col_lower, model.col_upper, 50, tol::kIntegrality);
    ASSERT_TRUE(device.ran) << name;
    ASSERT_EQ(device.infeasible, cpu.infeasible) << name;
    if (cpu.infeasible) return;
    EXPECT_EQ(device.rounds, cpu.rounds) << name;
    EXPECT_EQ(device.tightened, cpu.tightened) << name;
    for (std::size_t j = 0; j < lo.size(); ++j) {
      EXPECT_EQ(device.col_lb[j], lo[j]) << name << " column " << j;
      EXPECT_EQ(device.col_ub[j], hi[j]) << name << " column " << j;
    }
  };
  std::mt19937_64 rng(5101);
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::uniform_int_distribution<int> pick(0, 3);
  for (int trial = 0; trial < 60; ++trial) {
    const int n = 20 + trial;
    const int m = 10 + trial / 2;
    std::vector<std::vector<double>> a(static_cast<std::size_t>(m),
                                       std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<double> row_lo(static_cast<std::size_t>(m));
    std::vector<double> row_hi(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        if (pick(rng) == 0) a[i][j] = value(rng);
      }
      row_hi[i] = 3.0 * n / 4.0 + value(rng);
      row_lo[i] = pick(rng) == 0 ? row_hi[i] - 10.0 : -kInfinity;
    }
    Model model = integer_model(a, row_lo, row_hi, 0.0, 9.0);
    for (int j = 0; j < n; j += 3)
      model.col_type[static_cast<std::size_t>(j)] = VarType::kContinuous;
    same(model, "random " + std::to_string(trial));
  }
  const std::filesystem::path netlib =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data/netlib";
  int compared = 0;
  for (const char* name : {"afiro", "sc50a", "sc105", "blend", "stocfor1", "adlittle",
                           "share2b", "scagr7", "boeing2", "brandy"}) {
    Model model;
    if (!io::read_model((netlib / (std::string(name) + ".mps")).string(), &model).ok) continue;
    same(model, name);
    ++compared;
  }
  EXPECT_GE(compared, 5);
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
