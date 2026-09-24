// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Gondzio's multiple centrality correctors in the LP interior point (#472).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/centrality.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/lp_generator.hpp"

namespace sankhya {
namespace {

Options ipm_with_correctors(std::int64_t cap) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("crossover", false);  // the interior point's own answer is judged
  options.set_int("ipm_centrality_correctors", cap);
  return options;
}

TEST(CentralityCorrectors, TheAspirationStepIsColomboAndGondzios) {
  EXPECT_DOUBLE_EQ(ipm::aspiration_step(0.0), 0.3);
  EXPECT_DOUBLE_EQ(ipm::aspiration_step(0.2), 0.6);
  EXPECT_DOUBLE_EQ(ipm::aspiration_step(0.6), 1.0);
  EXPECT_DOUBLE_EQ(ipm::aspiration_step(1.0), 1.0);
}

TEST(CentralityCorrectors, ProductsOutsideTheBandArePulledBackAndTheRestLeftAlone) {
  // mu_target 1: the band is [0.1, 10]. At the trial point (steps 1, 1) the products are
  // s * z of (s + ds, z + dz): 0.01 (below), 1 (inside), 20 (above, pulled down by 10), 1000
  // (above, pulled down by no more than 10), and one entry with no bound, untouched.
  const std::vector<double> s = {0.1, 1.0, 4.0, 100.0, 5.0};
  const std::vector<double> ds = {0.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> z = {0.1, 1.0, 5.0, 10.0, 5.0};
  const std::vector<double> dz = {0.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<bool> present = {true, true, true, true, false};
  std::vector<double> r_mu(5, 0.0);
  const Index corrected = ipm::add_centrality_term(s, ds, z, dz, present, 1.0, 1.0, 1.0, &r_mu);
  EXPECT_EQ(corrected, 3);
  EXPECT_NEAR(r_mu[0], 0.1 - 0.01, 1e-15);
  EXPECT_EQ(r_mu[1], 0.0);
  EXPECT_NEAR(r_mu[2], 10.0 - 20.0, 1e-12);
  EXPECT_NEAR(r_mu[3], -10.0, 1e-12);  // capped at -beta_max * mu_target
  EXPECT_EQ(r_mu[4], 0.0);
  // The trial point is where the products are measured, not the current one.
  std::vector<double> moved(1, 0.0);
  EXPECT_EQ(
      ipm::add_centrality_term({1.0}, {-0.95}, {1.0}, {0.0}, {true}, 1.0, 1.0, 1.0, &moved), 1);
  EXPECT_NEAR(moved[0], 0.1 - 0.05, 1e-12);
}

TEST(CentralityCorrectors, TheBudgetFollowsTheFactorShapeAndTheCap) {
  EXPECT_EQ(ipm::corrector_budget(1e6, 1e3, 0), 0);  // off
  EXPECT_EQ(ipm::corrector_budget(2e3, 1e3, 5), 1);  // a sparse factor: ratio 0.75
  EXPECT_EQ(ipm::corrector_budget(1e6, 1e3, 3), 3);  // ratio 250, capped
  EXPECT_EQ(ipm::corrector_budget(1e6, 1e3, 10), 8);
  EXPECT_EQ(ipm::corrector_budget(1e6, 0.0, 5), 0);
}

TEST(CentralityCorrectors, AgreeWithTheExactOracleOnKktInstances) {
  // The 150 KKT instances of test_ipm.cpp with their analytic optima, correctors on.
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
    const Solution ipm = solve(model, ipm_with_correctors(5));
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

TEST(CentralityCorrectors, SolveTheCommittedNetlibInstancesToTheSimplexAnswer) {
  const char* names[] = {"afiro", "adlittle", "sc50a",    "sc50b",
                         "blend", "share2b",  "stocfor1", "israel"};
  Count with = 0;
  Count without = 0;
  for (const char* name : names) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(name) + ".mps"))
            .string();
    ASSERT_TRUE(io::read_model(path, &model).ok) << path;
    Options simplex_options;
    simplex_options.set_bool("log_to_console", false);
    simplex_options.set_string("algorithm", "dual-simplex");
    const Solution simplex = solve(model, simplex_options);
    ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << name;
    const Solution ipm = solve(model, ipm_with_correctors(5));
    const Solution plain = solve(model, ipm_with_correctors(0));
    EXPECT_EQ(ipm.status, SolveStatus::kOptimal) << name << ": " << ipm.message;
    if (ipm.status == SolveStatus::kOptimal) {
      EXPECT_NEAR(ipm.objective, simplex.objective,
                  1e-6 * std::max(1.0, std::fabs(simplex.objective)))
          << name;
    }
    with += ipm.iterations;
    without += plain.iterations;
  }
  // Recorded, not asserted: the count is what the Netlib run in the PR measures.
  std::cout << "committed Netlib: " << without << " iterations without correctors, " << with
            << " with\n";
}

TEST(CentralityCorrectors, OffByDefault) {
  Options options;
  EXPECT_EQ(options.get_int("ipm_centrality_correctors"), 0);
}

}  // namespace
}  // namespace sankhya
