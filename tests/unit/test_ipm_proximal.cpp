// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the proximal primal-dual regularization of the LP interior point (#473).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/ipm_testing.hpp"
#include "ipm/proximal_system.hpp"
#include "la/ldl.hpp"
#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
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

Model netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

/// The engine itself (no presolve, no crossover, no status guard), with its log captured.
std::string solve_ipm_logged(const Model& model, const Options& options, LogLevel level,
                             Solution* out) {
  std::FILE* stream = std::tmpfile();
  EXPECT_NE(stream, nullptr);
  if (stream == nullptr) return {};
  {
    Logger logger(stream, level);
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

/// The first number the pattern's one capture group matches in `text`, or NaN.
double captured(const std::string& text, const char* pattern) {
  std::smatch match;
  if (!std::regex_search(text, match, std::regex(pattern))) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::stod(match[1].str());
}

TEST(ProximalSystem, TheResidualIsReportedRelativeToTheRightHandSide) {
  // Review of #473: an absolute residual of 1e-9 is exact on a right-hand side of 1e6 and
  // useless on one of 1e-9. The report divides by max(1, |g|, |r_b|), and the target the
  // interior point holds it to is on that relative measure.
  SparseMatrix a(2, 3);
  a.add_entry(0, 0, 1.0);
  a.add_entry(0, 1, 2.0);
  a.add_entry(1, 1, -1.0);
  a.add_entry(1, 2, 3.0);
  a.finalize();
  const std::vector<bool> fixed(5, false);
  const std::vector<double> theta_inverse = {1e-3, 1e2, 1.0, 1e4, 1e-2};
  const std::vector<double> g = {3e6, -1.0, 2.0, 0.5, -4.0};
  const std::vector<double> r_b = {1.0, -2e5};
  ipm::ProximalSystem system(a, fixed);
  SparseLdl ldl;
  system.assemble(theta_inverse, 1e-6, 1e-6);
  ASSERT_TRUE(ldl.analyze(system.matrix()));
  double reg = 1e-6;
  int attempts = 0;
  ASSERT_TRUE(system.factorize(ldl, theta_inverse, &reg, {}, &attempts));
  std::vector<double> dx(5), dy(2);
  // No corrections: the plain regularized solve.
  const ipm::ProximalSystem::Refinement plain = system.solve(ldl, g, r_b, 0, &dx, &dy);
  EXPECT_EQ(plain.steps, 0);
  EXPECT_DOUBLE_EQ(plain.scale, 3e6);
  EXPECT_DOUBLE_EQ(plain.relative_residual(), plain.final_residual / 3e6);
  // With corrections the relative residual reaches the target on this well-posed system.
  const ipm::ProximalSystem::Refinement refined =
      system.solve(ldl, g, r_b, tol::kIpmProximalRefinementSteps, &dx, &dy);
  EXPECT_DOUBLE_EQ(refined.scale, 3e6);
  EXPECT_LE(refined.final_residual, plain.final_residual);
  EXPECT_LE(refined.relative_residual(), tol::kIpmProximalRefinementTarget)
      << "absolute " << refined.final_residual;
}

TEST(ProximalSystem, AMissedRefinementShrinksRhoAndARecoveryRaisesItsFloorToTheCap) {
  using ipm::ProximalSystem;
  // A miss shrinks the previous rho by kIpmProximalRefinementShrink before the mu rule...
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-6, 1.0, 1e-8, true),
                   1e-6 * tol::kIpmProximalRefinementShrink);
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(1e-6, 1.0, 1e-8, false), 1e-6);
  // ...and never below the floor.
  EXPECT_DOUBLE_EQ(ProximalSystem::next_regularization(2e-8, 1.0, 1e-8, true), 1e-8);
  // The recovery floor: 1e-8, then x kIpmProximalRecoveryRaise per raise, capped. The
  // default path's x1e4 would have made these 1e-4 and 1.
  EXPECT_DOUBLE_EQ(ProximalSystem::recovery_floor(0), tol::kIpmProximalFloor);
  EXPECT_DOUBLE_EQ(ProximalSystem::recovery_floor(1),
                   tol::kIpmProximalFloor * tol::kIpmProximalRecoveryRaise);
  EXPECT_LE(ProximalSystem::recovery_floor(2), tol::kIpmProximalRecoveryCap);
  EXPECT_DOUBLE_EQ(ProximalSystem::recovery_floor(2), tol::kIpmProximalRecoveryCap);
  EXPECT_DOUBLE_EQ(ProximalSystem::recovery_floor(40), tol::kIpmProximalRecoveryCap);
}

TEST(ProximalSystem, ALiftedPivotRaisesRhoAndFactorsAgain) {
  // Column 2 of A is empty, so its pivot is -(Theta^-1 + rho) under every ordering. A
  // Theta^-1 of -2 rho stands in for the rounding that gives a pivot the wrong sign in a real
  // factorization: the first attempt lifts it, the raise by kIpmProximalRaise makes it
  // -(98 rho), well past the threshold, and the second attempt is clean.
  SparseMatrix a(2, 3);
  a.add_entry(0, 0, 1.0);
  a.add_entry(1, 0, 1.0);
  a.add_entry(1, 1, 2.0);
  a.finalize();
  const std::vector<bool> fixed(5, false);
  const double start = 1e-6;
  const auto factor = [&](double t, double* reg, int* attempts, SparseLdl* ldl) {
    const std::vector<double> theta_inverse = {1.0, 1.0, t, 1.0, 1.0};
    ipm::ProximalSystem system(a, fixed);
    system.assemble(theta_inverse, start, start);
    EXPECT_TRUE(ldl->analyze(system.matrix()));
    *reg = start;
    EXPECT_TRUE(system.factorize(*ldl, theta_inverse, reg, {}, attempts));
  };
  {
    SparseLdl ldl;
    double reg = 0.0;
    int attempts = 0;
    factor(-2.0 * start, &reg, &attempts, &ldl);
    EXPECT_GE(attempts, 2);
    EXPECT_EQ(attempts, 2);
    EXPECT_DOUBLE_EQ(reg, start * tol::kIpmProximalRaise);
    EXPECT_EQ(ldl.regularized_pivots(), 0);
  }
  {
    // Two raises needed: -5e3 rho is still of the wrong sign at 100 rho, not at 1e4 rho.
    SparseLdl ldl;
    double reg = 0.0;
    int attempts = 0;
    factor(-5e3 * start, &reg, &attempts, &ldl);
    EXPECT_EQ(attempts, 3);
    EXPECT_DOUBLE_EQ(reg, start * tol::kIpmProximalRaise * tol::kIpmProximalRaise);
    EXPECT_EQ(ldl.regularized_pivots(), 0);
  }
  {
    // Past what kIpmProximalAttempts factorizations can raise: the lifted pivot is left to
    // the caller, which sees it counted.
    SparseLdl ldl;
    double reg = 0.0;
    int attempts = 0;
    factor(-1.0, &reg, &attempts, &ldl);
    EXPECT_EQ(attempts, tol::kIpmProximalAttempts);
    EXPECT_EQ(ldl.regularized_pivots(), 1);
  }
}

TEST(InteriorPointProximal, ARefinementMissIsCountedInTheLogAndInALimitStopsMessage) {
  const Model model = netlib("afiro");
  Options options = proximal_ipm(true);
  options.set_int("iteration_limit", 3);
  Solution limited;
  const std::string log = solve_ipm_logged(model, options, LogLevel::kInfo, &limited);
  ASSERT_EQ(limited.status, SolveStatus::kIterationLimit) << limited.message;
  // Three iterations, each a predictor and a corrector: six refined solves, every one counted.
  const std::string counted = "of 6 refined solve(s) above the 1e-08 refinement target";
  EXPECT_NE(limited.message.find(counted), std::string::npos) << limited.message;
  EXPECT_NE(log.find(counted), std::string::npos) << log;
  const double misses = captured(log, R"(regularized pivot\(s\) in total, (\d+) of 6 refined)");
  ASSERT_FALSE(std::isnan(misses)) << log;
  EXPECT_LE(misses, 6.0);
  // The default path's line and messages are untouched.
  Solution plain;
  options.set_bool("ipm_proximal_regularization", false);
  const std::string plain_log = solve_ipm_logged(model, options, LogLevel::kInfo, &plain);
  EXPECT_EQ(plain.message.find("refined solve"), std::string::npos) << plain.message;
  EXPECT_EQ(plain_log.find("refined solve"), std::string::npos) << plain_log;
}

TEST(InteriorPointProximal, ANonFiniteDirectionRaisesRhoToItsCapAndTheSolveFinishes) {
  // Modelled on InteriorPoint.ANonFiniteNewtonDirectionIsRecoveredByRaisingTheRegularization
  // (#209), with the non-finite direction forced (ipm_testing.hpp) rather than waited for:
  // the first two predictor directions come back NaN, so the recovery raises twice. The
  // proximal floor must go 1e-8 -> 1e-6 -> 1e-4 and stop there; before the review it
  // followed the dual regularization's x1e4 to 1.
  const Model model = netlib("afiro");
  Options simplex_options;
  simplex_options.set_bool("log_to_console", false);
  simplex_options.set_string("algorithm", "dual-simplex");
  const Solution simplex = solve(model, simplex_options);
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << simplex.message;

  ipm::testing::poison_next_directions.store(2);
  Solution recovered;
  const std::string log =
      solve_ipm_logged(model, proximal_ipm(true), LogLevel::kVerbose, &recovered);
  const int left = ipm::testing::poison_next_directions.exchange(0);
  EXPECT_EQ(left, 0) << "the poisoned directions were not all taken";
  std::size_t raises = 0;
  for (std::size_t at = log.find("non-finite direction at iteration 0");
       at != std::string::npos; at = log.find("non-finite direction at iteration 0", at + 1)) {
    ++raises;
  }
  EXPECT_EQ(raises, 2U) << log;
  // The first raise is logged with the floor it set: 1e-6, not the default path's 1e-4.
  const double floor_after = captured(log, R"(regularization raised to ([0-9.e+-]+) and)");
  const double expected_floor = tol::kIpmProximalFloor * tol::kIpmProximalRecoveryRaise;
  EXPECT_NEAR(floor_after, expected_floor, 1e-9 * expected_floor) << log;
  const double rho_at_end = captured(log, R"(proximal regularization ([0-9.e+-]+) at the end)");
  ASSERT_FALSE(std::isnan(rho_at_end)) << log;
  EXPECT_LE(rho_at_end, tol::kIpmProximalRecoveryCap * (1.0 + 1e-9)) << log;
  ASSERT_EQ(recovered.status, SolveStatus::kOptimal) << recovered.message;
  EXPECT_NEAR(recovered.objective, simplex.objective, 1e-6 * std::fabs(simplex.objective));
}

TEST(InteriorPointProximal, ARankDeficientLpWithAFreeColumnNeedsNoLiftedPivot) {
  // min x0 + 2 x1 + 3 x2 + 1.5 x3, x3 free, x0..x2 >= 0:
  //   x0 + x1 + x2 + x3 = 4      (row 0)
  //   x0 + x1 + x2 + x3 = 4      (row 1, a duplicate)
  //   2x0 + 2x1 + 2x2 + 2x3 = 8  (row 2, doubled)
  //   x0 - x3 <= 2               (row 3)
  // Rows 0-2 have rank 1 and their logicals are fixed, so A Theta A^T is singular whatever
  // Theta, and the free column's Theta of 1e8 makes it worse: the normal equations lift a
  // pivot. The augmented system with rho, delta > 0 is quasi-definite and is factored with
  // none lifted. The optimum is x0 = 3, x3 = 1, objective 4.5.
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {1.0, 2.0, 3.0, 1.5};
  model.col_lower = {0.0, 0.0, 0.0, -kInfinity};
  model.col_upper = {kInfinity, kInfinity, kInfinity, kInfinity};
  model.col_type.assign(4, VarType::kContinuous);
  model.row_lower = {4.0, 4.0, 8.0, -kInfinity};
  model.row_upper = {4.0, 4.0, 8.0, 2.0};
  model.matrix.reset(4, 4);
  for (Index j = 0; j < 4; ++j) {
    model.matrix.add_entry(0, j, 1.0);
    model.matrix.add_entry(1, j, 1.0);
    model.matrix.add_entry(2, j, 2.0);
  }
  model.matrix.add_entry(3, 0, 1.0);
  model.matrix.add_entry(3, 3, -1.0);
  model.matrix.finalize();
  model.hessian.reset(4, 4);
  model.hessian.finalize();

  Options simplex_options;
  simplex_options.set_bool("log_to_console", false);
  simplex_options.set_bool("presolve", false);
  simplex_options.set_string("algorithm", "dual-simplex");
  const Solution simplex = solve(model, simplex_options);
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << simplex.message;
  EXPECT_NEAR(simplex.objective, 4.5, 1e-9);

  const auto run = [&](bool proximal, Solution* out) {
    Options options = proximal_ipm(proximal);
    options.set_bool("presolve", false);
    options.set_bool("crossover", false);
    return solve_ipm_logged(model, options, LogLevel::kInfo, out);
  };
  const char* pivots = R"(IPM: \d+ iterations, \d+ factorizations, (\d+) regularized pivot)";
  Solution on;
  const std::string on_log = run(true, &on);
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, simplex.objective, 1e-6 * std::fabs(simplex.objective));
  EXPECT_EQ(captured(on_log, pivots), 0.0) << on_log;
  EXPECT_NE(on_log.find(" 0 regularized pivot(s) in total"), std::string::npos) << on_log;

  Solution off;
  const std::string off_log = run(false, &off);
  EXPECT_GE(captured(off_log, pivots), 1.0) << off_log;
  if (off.status == SolveStatus::kOptimal) {
    EXPECT_NEAR(off.objective, simplex.objective, 1e-6 * std::fabs(simplex.objective));
  }
}

TEST(InteriorPointProximal, OffByDefaultAndTheDefaultPathIsTheExplicitlyOffPath) {
  Options defaults;
  EXPECT_FALSE(defaults.get_bool("ipm_proximal_regularization"));
  // The default path, pinned against the same solve with the option set false explicitly:
  // the same iterations, the same objective and the same point, bit for bit. Afiro's
  // default-path count is its own; the proximal one is only held to the same optimum.
  const Model model = netlib("afiro");
  Options unset;
  unset.set_bool("log_to_console", false);
  unset.set_string("algorithm", "ipm");
  const Solution by_default = solve(model, unset);
  const Solution plain = solve(model, proximal_ipm(false));
  const Solution proximal = solve(model, proximal_ipm(true));
  ASSERT_EQ(by_default.status, SolveStatus::kOptimal) << by_default.message;
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  ASSERT_EQ(proximal.status, SolveStatus::kOptimal) << proximal.message;
  EXPECT_EQ(by_default.iterations, plain.iterations);
  EXPECT_EQ(by_default.objective, plain.objective);
  EXPECT_EQ(by_default.col_value, plain.col_value);
  EXPECT_EQ(by_default.message, plain.message);
  EXPECT_NEAR(plain.objective, proximal.objective, 1e-6 * std::fabs(plain.objective));
}

}  // namespace
}  // namespace sankhya
