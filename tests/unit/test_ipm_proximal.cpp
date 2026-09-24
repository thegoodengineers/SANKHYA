// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the proximal primal-dual regularization of the LP interior point (#473).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/proximal_system.hpp"
#include "la/ldl.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/lp_generator.hpp"

namespace sankhya {
namespace {

Options proximal_ipm(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("ipm_proximal_regularization", on);
  return options;
}

/// Dense Gaussian elimination with partial pivoting: the reference the refined solve is held
/// to, sharing no code with the sparse factorization.
std::vector<double> dense_solve(std::vector<std::vector<double>> a, std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::fabs(a[i][k]) > std::fabs(a[pivot][k])) pivot = i;
    }
    std::swap(a[k], a[pivot]);
    std::swap(b[k], b[pivot]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i][k] / a[k][k];
      for (std::size_t j = k; j < n; ++j) a[i][j] -= f * a[k][j];
      b[i] -= f * b[k];
    }
  }
  std::vector<double> x(n);
  for (std::size_t k = n; k-- > 0;) {
    double s = b[k];
    for (std::size_t j = k + 1; j < n; ++j) s -= a[k][j] * x[j];
    x[k] = s / a[k][k];
  }
  return x;
}

TEST(ProximalSystem, RefinementSolvesTheUnregularizedNewtonSystem) {
  // A random 6 x 9 A, Theta^-1 spread over twelve decades as near an optimum, one free
  // structural (Theta^-1 = 0), one fixed structural and one fixed logical. The refined
  // direction must satisfy the Newton system WITHOUT rho and delta, checked against a dense
  // solve of that system in the full (x, s, y) space.
  std::mt19937_64 rng(473);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  const Index m = 6;
  const Index n = 9;
  SparseMatrix a(m, n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < m; ++i) {
      if ((i + j) % 3 != 1 || i == j % m) a.add_entry(i, j, value(rng));
    }
  }
  a.finalize();
  const Index total = n + m;
  std::vector<bool> fixed(static_cast<std::size_t>(total), false);
  fixed[4] = true;
  fixed[static_cast<std::size_t>(n + 2)] = true;
  std::vector<double> theta_inverse(static_cast<std::size_t>(total));
  for (Index k = 0; k < total; ++k) {
    theta_inverse[static_cast<std::size_t>(k)] = std::pow(10.0, -6.0 + (k * 7) % 13);
  }
  theta_inverse[7] = 0.0;  // a free structural
  std::vector<double> g(static_cast<std::size_t>(total)), r_b(static_cast<std::size_t>(m));
  for (double& v : g) v = value(rng);
  for (double& v : r_b) v = value(rng);

  ipm::ProximalSystem system(a, fixed);
  SparseLdl ldl;
  const double reg = 1e-6;
  system.assemble(theta_inverse, reg, reg);
  ASSERT_TRUE(ldl.analyze(system.matrix()));
  double used = reg;
  int attempts = 0;
  ASSERT_TRUE(system.factorize(ldl, theta_inverse, &used, {}, &attempts));
  EXPECT_EQ(ldl.regularized_pivots(), 0);
  std::vector<double> dx(static_cast<std::size_t>(total)), dy(static_cast<std::size_t>(m));
  const ipm::ProximalSystem::Refinement refined =
      system.solve(ldl, g, r_b, tol::kIpmProximalRefinementSteps, &dx, &dy);

  // The dense reference: unknowns are the unfixed variables and y.
  std::vector<Index> free_vars;
  for (Index k = 0; k < total; ++k) {
    if (!fixed[static_cast<std::size_t>(k)]) free_vars.push_back(k);
  }
  const std::size_t f = free_vars.size();
  const std::size_t dim = f + static_cast<std::size_t>(m);
  std::vector<std::vector<double>> k0(dim, std::vector<double>(dim, 0.0));
  std::vector<double> rhs(dim, 0.0);
  const auto abar = [&](Index i, Index k) {
    if (k >= n) return k - n == i ? -1.0 : 0.0;
    const ColumnView column = a.column(k);
    for (Index p = 0; p < column.size; ++p) {
      if (column.rows[p] == i) return column.values[p];
    }
    return 0.0;
  };
  for (std::size_t q = 0; q < f; ++q) {
    const Index k = free_vars[q];
    k0[q][q] = -theta_inverse[static_cast<std::size_t>(k)];
    rhs[q] = g[static_cast<std::size_t>(k)];
    for (Index i = 0; i < m; ++i) {
      const double v = abar(i, k);
      k0[q][f + static_cast<std::size_t>(i)] = v;
      k0[f + static_cast<std::size_t>(i)][q] = v;
    }
  }
  for (Index i = 0; i < m; ++i)
    rhs[f + static_cast<std::size_t>(i)] = r_b[static_cast<std::size_t>(i)];
  const std::vector<double> exact = dense_solve(k0, rhs);

  double scale = 1.0;
  for (const double v : exact) scale = std::max(scale, std::fabs(v));
  for (std::size_t q = 0; q < f; ++q) {
    EXPECT_NEAR(dx[static_cast<std::size_t>(free_vars[q])], exact[q], 1e-8 * scale)
        << "variable " << free_vars[q];
  }
  for (Index i = 0; i < m; ++i) {
    EXPECT_NEAR(dy[static_cast<std::size_t>(i)], exact[f + static_cast<std::size_t>(i)],
                1e-8 * scale)
        << "row " << i;
  }
  EXPECT_EQ(dx[4], 0.0);
  EXPECT_EQ(dx[static_cast<std::size_t>(n + 2)], 0.0);
  // The refinement is what closed the gap the regularization opened.
  EXPECT_GT(refined.steps, 0);
  EXPECT_LT(refined.final_residual, refined.first_residual);
}

TEST(ProximalSystem, TheRegularizationFollowsMuDownAndStopsAtTheFloor) {
  using ipm::ProximalSystem;
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-6, 1.0, 1e-8), 1e-6);
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-6, 1e-5, 1e-8),
                   tol::kIpmProximalShare * 1e-5);
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-6, 1e-12, 1e-8), 1e-8);
  // A raised floor (the dual regularization's recovery) wins over the schedule.
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-8, 1e-12, 1e-4), 1e-4);
}

TEST(InteriorPointProximal, AgreesWithTheExactOracleOnKktInstances) {
  // The same 150 KKT instances, the same seed and the same 1e-6 bar as the default path's
  // test in test_ipm.cpp: the proximal path must not lose one.
  std::mt19937_64 rng(56001);
  oracle::GeneratorConfig config;
  config.min_rows = 3;
  config.max_rows = 12;
  config.min_cols = 3;
  config.max_cols = 12;
  config.magnitude = 6;
  int failed = 0;
  std::string first_failure;
  for (int trial = 0; trial < 150; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    // Without crossover, so the interior point's own answer is what is judged.
    Options options = proximal_ipm(true);
    options.set_bool("crossover", false);
    const Solution ipm = solve(model, options);
    const double expected = static_cast<double>(instance.optimal_objective);
    const bool ok =
        ipm.status == SolveStatus::kOptimal &&
        std::fabs(ipm.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected));
    if (!ok) {
      ++failed;
      if (first_failure.empty()) {
        first_failure = "trial " + std::to_string(trial) + ": " + to_string(ipm.status) +
                        " objective " + std::to_string(ipm.objective) + " expected " +
                        std::to_string(expected) + " (" + ipm.message + ")";
      }
    }
  }
  EXPECT_EQ(failed, 0) << first_failure;
}

TEST(InteriorPointProximal, SolvesTheCommittedNetlibInstancesToTheSimplexAnswer) {
  const char* names[] = {"afiro", "adlittle", "sc50a",    "sc50b",
                         "blend", "share2b",  "stocfor1", "israel"};
  for (const char* name : names) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(name) + ".mps"))
            .string();
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;
    Options simplex_options;
    simplex_options.set_bool("log_to_console", false);
    simplex_options.set_string("algorithm", "dual-simplex");
    const Solution simplex = solve(model, simplex_options);
    ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << name << ": " << simplex.message;
    // Without crossover, so the interior point's own answer is what is judged.
    Options options = proximal_ipm(true);
    options.set_bool("crossover", false);
    const Solution ipm = solve(model, options);
    EXPECT_EQ(ipm.status, SolveStatus::kOptimal) << name << ": " << ipm.message;
    if (ipm.status == SolveStatus::kOptimal) {
      EXPECT_NEAR(ipm.objective, simplex.objective,
                  1e-6 * std::max(1.0, std::fabs(simplex.objective)))
          << name;
    }
  }
}

TEST(InteriorPointProximal, OffByDefaultAndTheDefaultPathIsUnchanged) {
  Options options;
  EXPECT_FALSE(options.get_bool("ipm_proximal_regularization"));
  // The same model through both paths: the default path's iteration count is its own, and
  // the proximal one reaches the same optimum.
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/afiro.mps")
          .string();
  ASSERT_TRUE(io::read_model(path, &model).ok);
  const Solution plain = solve(model, proximal_ipm(false));
  const Solution proximal = solve(model, proximal_ipm(true));
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  ASSERT_EQ(proximal.status, SolveStatus::kOptimal) << proximal.message;
  EXPECT_NEAR(plain.objective, proximal.objective, 1e-6 * std::fabs(plain.objective));
}

}  // namespace
}  // namespace sankhya
