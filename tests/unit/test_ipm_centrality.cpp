// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Gondzio's multiple centrality correctors in the LP interior point (#472).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/centrality.hpp"
#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
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

/// The engine itself (no presolve, no crossover, no status guard), with its log captured.
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

/// The counts of the "centrality correctors:" log line: kept, tried, the most tried in one
/// iteration and the largest per-iteration budget. NaN when the line is absent.
struct CorrectorLine {
  double kept = std::numeric_limits<double>::quiet_NaN();
  double tried = std::numeric_limits<double>::quiet_NaN();
  double most_tried = std::numeric_limits<double>::quiet_NaN();
  double most_budget = std::numeric_limits<double>::quiet_NaN();
};

CorrectorLine corrector_line(const std::string& log) {
  CorrectorLine line;
  std::smatch match;
  const std::regex pattern(
      R"(centrality correctors: (\d+) kept of (\d+) tried, at most (\d+) tried in one )"
      R"(iteration under a budget of at most (\d+))");
  if (std::regex_search(log, match, pattern)) {
    line.kept = std::stod(match[1].str());
    line.tried = std::stod(match[2].str());
    line.most_tried = std::stod(match[3].str());
    line.most_budget = std::stod(match[4].str());
  }
  return line;
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
    const Model model = committed_netlib(name);
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

TEST(CentralityCorrectors, TheDefaultIsBitForBitTheOptionSetOff) {
  // Review of #668: with the option at its default the interior point must be exactly the
  // one without correctors - same iterations, same objective, same point, to the last bit.
  for (const char* name : {"afiro", "adlittle", "sc50a", "blend", "share2b", "israel"}) {
    const Model model = committed_netlib(name);
    Options by_default;
    by_default.set_bool("log_to_console", false);
    by_default.set_string("algorithm", "ipm");
    by_default.set_bool("crossover", false);
    const Solution untouched = solve(model, by_default);
    const Solution off = solve(model, ipm_with_correctors(0));
    EXPECT_EQ(untouched.status, off.status) << name;
    EXPECT_EQ(untouched.iterations, off.iterations) << name;
    EXPECT_EQ(untouched.objective, off.objective) << name;  // exact, not near
    EXPECT_EQ(untouched.col_value, off.col_value) << name;
    EXPECT_EQ(untouched.row_dual, off.row_dual) << name;
  }
}

TEST(CentralityCorrectors, SomeAreKeptOnTheCommittedNetlibInstances) {
  // The log line is the evidence a corrector did anything: over the committed Netlib models
  // with the option on, at least one corrector is kept, and never more than were tried.
  double kept = 0.0;
  double tried = 0.0;
  for (const char* name :
       {"afiro", "adlittle", "sc50a", "sc50b", "blend", "share2b", "stocfor1", "israel"}) {
    const Model model = committed_netlib(name);
    Solution solution;
    const std::string log = solve_ipm_logged(model, ipm_with_correctors(5), &solution);
    const CorrectorLine line = corrector_line(log);
    ASSERT_FALSE(std::isnan(line.kept)) << name << ": no corrector line in\n" << log;
    EXPECT_LE(line.kept, line.tried) << name;
    EXPECT_LE(line.most_tried, line.most_budget) << name;
    std::cout << name << ": " << line.kept << " kept of " << line.tried << " tried, "
              << solution.iterations << " iterations\n";
    kept += line.kept;
    tried += line.tried;
  }
  EXPECT_GT(kept, 0.0) << "no corrector kept in " << tried << " tried";
}

/// A covering LP whose normal equations fill in: m rows, 2m columns meeting 8 random rows
/// each, plus one column meeting every row. Its row-side factor is nearly dense, so the
/// budget from the factor's shape alone allows several correctors per iteration.
Model filled_cover(Index m) {
  std::mt19937_64 rng(472668);
  std::uniform_real_distribution<double> value(1.0, 2.0);
  const Index n = 2 * m + 1;
  Model model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(m, n);
  std::vector<Index> rows(static_cast<std::size_t>(m));
  std::iota(rows.begin(), rows.end(), Index{0});
  for (Index j = 0; j + 1 < n; ++j) {
    std::shuffle(rows.begin(), rows.end(), rng);
    std::vector<Index> picked(rows.begin(), rows.begin() + 8);
    std::sort(picked.begin(), picked.end());
    for (const Index i : picked) model.matrix.add_entry(i, j, value(rng));
  }
  for (Index i = 0; i < m; ++i) model.matrix.add_entry(i, n - 1, 1.0);
  model.matrix.finalize();
  model.col_cost.resize(static_cast<std::size_t>(n));
  for (double& c : model.col_cost) c = value(rng);
  model.col_cost[static_cast<std::size_t>(n - 1)] = 3.0;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), 1.0);
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  return model;
}

void expect_same_optimum(const Solution& got, const Solution& reference, const char* path) {
  ASSERT_EQ(got.status, SolveStatus::kOptimal) << path << ": " << got.message;
  EXPECT_NEAR(got.objective, reference.objective,
              1e-6 * std::max(1.0, std::fabs(reference.objective)))
      << path;
}

TEST(CentralityCorrectors, AtMostOnePerIterationWhereASolveIsAConjugateGradient) {
  // Review of #668: on the dense-column path (#467) and the n x n side (#469) every solve is
  // a preconditioned conjugate gradient, so the factor-shape budget does not describe its
  // cost: at most kIpmCentralityConjugateGradientCorrectors per iteration there, and at most
  // kIpmCentralityProximalCorrectors on the refined solves of the proximal path (#473). The
  // row side on the same model shows the cap is what binds, not the model.
  const Model model = filled_cover(80);
  Solution plain;
  const CorrectorLine row_side =
      corrector_line(solve_ipm_logged(model, ipm_with_correctors(5), &plain));
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;
  EXPECT_GE(row_side.most_budget, 2.0);

  Options dense = ipm_with_correctors(5);
  dense.set_bool("ipm_dense_columns", true);
  dense.set_double("ipm_dense_column_factor", 2.0);
  Solution split;
  const std::string dense_log = solve_ipm_logged(model, dense, &split);
  ASSERT_NE(dense_log.find("dense-column correction over 1 column"), std::string::npos)
      << dense_log;
  const CorrectorLine on_dense = corrector_line(dense_log);
  ASSERT_FALSE(std::isnan(on_dense.most_tried)) << dense_log;
  EXPECT_LE(on_dense.most_budget, tol::kIpmCentralityConjugateGradientCorrectors);
  EXPECT_LE(on_dense.most_tried, tol::kIpmCentralityConjugateGradientCorrectors);
  expect_same_optimum(split, plain, "dense columns");

  Options columns = ipm_with_correctors(5);
  columns.set_string("ipm_normal_side", "columns");
  Solution column_side;
  const std::string column_log = solve_ipm_logged(model, columns, &column_side);
  ASSERT_NE(column_log.find("n x n side (#469)"), std::string::npos) << column_log;
  const CorrectorLine on_columns = corrector_line(column_log);
  ASSERT_FALSE(std::isnan(on_columns.most_tried)) << column_log;
  EXPECT_LE(on_columns.most_budget, tol::kIpmCentralityConjugateGradientCorrectors);
  EXPECT_LE(on_columns.most_tried, tol::kIpmCentralityConjugateGradientCorrectors);
  expect_same_optimum(column_side, plain, "n x n side");

  Options proximal = ipm_with_correctors(5);
  proximal.set_bool("ipm_proximal_regularization", true);
  Solution refined;
  const std::string proximal_log = solve_ipm_logged(model, proximal, &refined);
  const CorrectorLine on_proximal = corrector_line(proximal_log);
  ASSERT_FALSE(std::isnan(on_proximal.most_tried)) << proximal_log;
  EXPECT_LE(on_proximal.most_budget, tol::kIpmCentralityProximalCorrectors);
  EXPECT_LE(on_proximal.most_tried, tol::kIpmCentralityProximalCorrectors);
  expect_same_optimum(refined, plain, "proximal");
}

}  // namespace
}  // namespace sankhya
