// SPDX-License-Identifier: Apache-2.0
// SANKHYA - IIS deletion filter tests (#217).
//
// The tests follow the same discipline as test_cuts.cpp and test_certificate.cpp: every
// positive test has a paired negative control that verifies the algorithm can distinguish
// what is necessary from what is not. Passing only a positive test is not evidence.
//
// Algorithm: Chinneck & Dravnieks, "Locating minimal infeasible constraint sets in linear
// programs", ORSA J. Computing 3(2) (1991).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

/// Build a simple LP model from dense row data.
Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& col_lower, const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(col_lower.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  model.col_names.clear();
  model.row_names.clear();
  return model;
}

Options silent_options() {
  Options opts;
  opts.set_bool("log_to_console", false);
  opts.set_bool("compute_iis", true);
  // Disable presolve so the dual simplex runs and produces a Farkas dual.
  // When presolve detects infeasibility it does not produce a Farkas certificate,
  // and compute_iis returns early. That is a known limitation, not tested here.
  opts.set_bool("presolve", false);
  return opts;
}

// =========================================================================================
// IIS::SimpleConflict
//
// Two rows that directly conflict: x >= 3 and x <= 2. Both are necessary.
// Two irrelevant rows (y >= 1, z <= 10) must not appear in the IIS.
// =========================================================================================

TEST(IIS, SimpleConflict) {
  // Variables: x, y, z  (3 columns)
  // Rows:
  //   R0: x >= 3       (lb=3, ub=inf)
  //   R1: x <= 2       (lb=-inf, ub=2)
  //   R2: y >= 1       (lb=1, ub=inf)   <- irrelevant
  //   R3: z <= 10      (lb=-inf, ub=10) <- irrelevant
  Model model = make_lp({{1, 0, 0},                          // R0
                         {1, 0, 0},                          // R1
                         {0, 1, 0},                          // R2
                         {0, 0, 1}},                         // R3
                        {3.0, -kInfinity, 1.0, -kInfinity},  // row_lower
                        {kInfinity, 2.0, kInfinity, 10.0},   // row_upper
                        {0.0, 0.0, 0.0},                     // col_lower
                        {kInfinity, kInfinity, kInfinity});  // col_upper

  const Options opts = silent_options();
  const Solution sol = solve(model, opts);

  ASSERT_EQ(sol.status, SolveStatus::kInfeasible);
  // Must have computed an IIS.
  EXPECT_FALSE(sol.iis_rows.empty());

  // The IIS must contain exactly R0 and R1.
  EXPECT_EQ(sol.iis_rows.size(), 2u);
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{0}), sol.iis_rows.end());
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{1}), sol.iis_rows.end());

  // Irrelevant rows must not appear.
  EXPECT_EQ(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{2}), sol.iis_rows.end());
  EXPECT_EQ(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{3}), sol.iis_rows.end());

  // No column bounds should be in the IIS for this example.
  EXPECT_TRUE(sol.iis_col_lo.empty());
  EXPECT_TRUE(sol.iis_col_hi.empty());
}

// =========================================================================================
// IIS::TheCertificateAndTheWitnessesProveBothProperties
//
// What the .sol file will carry, checked here in the solver's own terms (#217, third
// acceptance box): the Farkas vector's support lies inside the IIS, and every element has a
// witness point that satisfies all the other elements and violates its own.
// =========================================================================================

TEST(IIS, TheCertificateAndTheWitnessesProveBothProperties) {
  // The conflict is x >= 3 against x <= 2, buried under a column bound the certificate may
  // touch (x <= 10 is slack) and two irrelevant rows.
  Model model =
      make_lp({{1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {3.0, -kInfinity, 1.0, -kInfinity},
              {kInfinity, 2.0, kInfinity, 10.0}, {0.0, 0.0, 0.0}, {10.0, kInfinity, kInfinity});
  const Solution sol = solve(model, silent_options());
  ASSERT_EQ(sol.status, SolveStatus::kInfeasible) << sol.message;
  ASSERT_EQ(sol.iis_rows.size(), 2u);
  ASSERT_FALSE(sol.iis_inconclusive);

  // 1. The certificate is the subsystem's: nonzero multipliers only on IIS rows.
  ASSERT_EQ(sol.farkas_dual.size(), 4u);
  for (Index i = 0; i < 4; ++i) {
    const bool in_iis =
        std::find(sol.iis_rows.begin(), sol.iis_rows.end(), i) != sol.iis_rows.end();
    if (!in_iis) {
      EXPECT_EQ(sol.farkas_dual[static_cast<std::size_t>(i)], 0.0) << "row " << i;
    }
  }
  EXPECT_TRUE(sol.farkas_dual[0] != 0.0 && sol.farkas_dual[1] != 0.0);

  // 2. One witness per element; each satisfies the other IIS rows and violates its own.
  const std::size_t elements =
      sol.iis_rows.size() + sol.iis_col_lo.size() + sol.iis_col_hi.size();
  ASSERT_EQ(sol.iis_witnesses.size(), elements);
  for (std::size_t w = 0; w < sol.iis_rows.size(); ++w) {
    const std::vector<double>& x = sol.iis_witnesses[w];
    ASSERT_EQ(x.size(), 3u);
    for (std::size_t v = 0; v < sol.iis_rows.size(); ++v) {
      const auto row = static_cast<std::size_t>(sol.iis_rows[v]);
      const double activity = x[0];  // both IIS rows are x alone
      const bool inside =
          activity >= model.row_lower[row] - 1e-7 && activity <= model.row_upper[row] + 1e-7;
      if (v == w) {
        EXPECT_FALSE(inside) << "witness " << w << " must violate its own row";
      } else {
        EXPECT_TRUE(inside) << "witness " << w << " must satisfy row " << row;
      }
    }
  }
}

// =========================================================================================
// IIS::RedundantRowExcluded
//
// Conflict: x >= 3 and x <= 2. Plus a redundant row x <= 1 (tighter than R1).
// The deletion filter must exclude the redundant row.
// =========================================================================================

TEST(IIS, RedundantRowExcluded) {
  // R0: x >= 3  (necessary)
  // R1: x <= 2  (necessary: without this, x >= 3 and x <= 1 alone are infeasible)
  // R2: x <= 1  (redundant: R0 and R1 already prove infeasibility without R2)
  //
  // Note: the certificate names all three, but only two are irreducible.
  // The filter must drop whichever one it finds is redundant (either R1 or R2, since
  // both together with R0 are redundant; exactly one of them must survive alongside R0).
  // We check that exactly two rows appear in the IIS (one is R0).
  Model model = make_lp({{1, 0},                        // R0: x >= 3
                         {1, 0},                        // R1: x <= 2
                         {1, 0}},                       // R2: x <= 1
                        {3.0, -kInfinity, -kInfinity},  // row_lower
                        {kInfinity, 2.0, 1.0},          // row_upper
                        {0.0, 0.0},                     // col_lower
                        {kInfinity, kInfinity});        // col_upper

  const Options opts = silent_options();
  const Solution sol = solve(model, opts);

  ASSERT_EQ(sol.status, SolveStatus::kInfeasible);
  ASSERT_FALSE(sol.iis_rows.empty());

  // IIS must have exactly 2 rows (R0 is necessary; exactly one of R1/R2 is).
  EXPECT_EQ(sol.iis_rows.size(), 2u);
  // R0 must appear.
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{0}), sol.iis_rows.end());
  // Not all three should appear (the filter reduced it).
  EXPECT_LT(sol.iis_rows.size(), 3u);
}

// =========================================================================================
// IIS::ColumnBoundInConflict
//
// Conflict between a row lower bound and a column upper bound.
// Row: x >= 5. Column: x in [0, 3] (upper bound 3 < 5).
// Both the row AND the column upper bound are in the IIS.
// =========================================================================================

TEST(IIS, ColumnBoundInConflict) {
  // Single variable x: row says x >= 5 but col_upper = 3.
  Model model = make_lp({{1.0}},      // R0: x >= 5
                        {5.0},        // row_lower
                        {kInfinity},  // row_upper
                        {0.0},        // col_lower
                        {3.0});       // col_upper

  const Options opts = silent_options();
  const Solution sol = solve(model, opts);

  ASSERT_EQ(sol.status, SolveStatus::kInfeasible);

  // The IIS must mention R0 (row) AND the upper bound on x (col_hi index 0).
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{0}), sol.iis_rows.end());
  EXPECT_NE(std::find(sol.iis_col_hi.begin(), sol.iis_col_hi.end(), Index{0}),
            sol.iis_col_hi.end());
}

// =========================================================================================
// IIS::ComputeIisDisabled
//
// When compute_iis=false the iis_rows vector must remain empty.
// =========================================================================================

TEST(IIS, ComputeIisDisabled) {
  Model model = make_lp({{1, 0}, {1, 0}}, {3.0, -kInfinity}, {kInfinity, 2.0}, {0.0, 0.0},
                        {kInfinity, kInfinity});

  Options opts = silent_options();
  opts.set_bool("compute_iis", false);
  const Solution sol = solve(model, opts);

  ASSERT_EQ(sol.status, SolveStatus::kInfeasible);
  EXPECT_TRUE(sol.iis_rows.empty());
  EXPECT_TRUE(sol.iis_col_lo.empty());
  EXPECT_TRUE(sol.iis_col_hi.empty());
}

// =========================================================================================
// IIS::FeasibleModelHasNoIis
//
// A feasible model must not produce an IIS (sanity check).
// =========================================================================================

TEST(IIS, FeasibleModelHasNoIis) {
  Model model = make_lp({{1.0}},  // x >= 0 and x <= 10 -> feasible
                        {0.0}, {10.0}, {0.0}, {kInfinity});

  const Options opts = silent_options();
  const Solution sol = solve(model, opts);

  EXPECT_NE(sol.status, SolveStatus::kInfeasible);
  EXPECT_TRUE(sol.iis_rows.empty());
  EXPECT_TRUE(sol.iis_col_lo.empty());
  EXPECT_TRUE(sol.iis_col_hi.empty());
}

// =========================================================================================
// IIS::ConflictBuriedAmongManyRows
//
// A conflict between two rows is buried among ten irrelevant rows. The IIS must name
// exactly those two rows and nothing else.
// =========================================================================================

TEST(IIS, ConflictBuriedAmongManyRows) {
  // Variables: x (single variable). 10 irrelevant rows plus the conflicting pair.
  // Irrelevant rows: -100 <= x <= 100 (repeated), which are satisfied by any x in [-100,100].
  // Conflict: R10: x >= 5, R11: x <= 2. These together are infeasible.
  const int n_irrelevant = 10;
  std::vector<std::vector<double>> rows;
  std::vector<double> row_lo, row_hi;

  for (int k = 0; k < n_irrelevant; ++k) {
    rows.push_back({1.0});
    row_lo.push_back(-100.0);
    row_hi.push_back(100.0);
  }
  // Conflict:
  rows.push_back({1.0});  // R10: x >= 5
  row_lo.push_back(5.0);
  row_hi.push_back(kInfinity);

  rows.push_back({1.0});  // R11: x <= 2
  row_lo.push_back(-kInfinity);
  row_hi.push_back(2.0);

  Model model = make_lp(rows, row_lo, row_hi, {0.0}, {kInfinity});
  const Options opts = silent_options();
  const Solution sol = solve(model, opts);

  ASSERT_EQ(sol.status, SolveStatus::kInfeasible);
  ASSERT_FALSE(sol.iis_rows.empty());

  // Must contain exactly the two conflicting rows (indices 10 and 11).
  EXPECT_EQ(sol.iis_rows.size(), 2u);
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{n_irrelevant}),
            sol.iis_rows.end());
  EXPECT_NE(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), Index{n_irrelevant + 1}),
            sol.iis_rows.end());

  // Irrelevant rows must not appear.
  for (Index i = 0; i < Index{n_irrelevant}; ++i) {
    EXPECT_EQ(std::find(sol.iis_rows.begin(), sol.iis_rows.end(), i), sol.iis_rows.end())
        << "Irrelevant row " << i << " should not be in the IIS";
  }
}

// =========================================================================================
// The witnesses, held to tools/verify_solution.py's own measure (#475)
//
// The verifier judges a witness row by row, its violation over max(1, |activity|, |bounds|),
// against the primal tolerance; the status guard that accepts a trial point measures a row
// over the largest term of its sum. Where terms cancel the second is the looser, and on
// Netlib klein2, from the interior point's certificate under ipm_hsd, four witnesses missed
// the verifier's measure by up to 2.2e-7 and the file was rejected. A claimed IIS now carries
// only witnesses that pass the verifier's rule, or claims nothing.
// =========================================================================================

double verifier_violation(double value, double lower, double upper) {
  const double below = std::isfinite(lower) ? lower - value : 0.0;
  const double above = std::isfinite(upper) ? value - upper : 0.0;
  double scale = std::max(1.0, std::fabs(value));
  if (std::isfinite(lower)) scale = std::max(scale, std::fabs(lower));
  if (std::isfinite(upper)) scale = std::max(scale, std::fabs(upper));
  return std::max({below, above, 0.0}) / scale;
}

/// The failures of the claimed witnesses under the verifier's rule; empty when every one
/// satisfies the other elements and violates its own.
std::vector<std::string> witness_failures(const Model& model, const Solution& sol) {
  std::vector<std::string> failures;
  if (sol.iis_inconclusive) return failures;
  struct Element {
    int kind;
    Index index;
  };
  std::vector<Element> iis;
  for (const Index i : sol.iis_rows) iis.push_back({0, i});
  for (const Index j : sol.iis_col_lo) iis.push_back({1, j});
  for (const Index j : sol.iis_col_hi) iis.push_back({2, j});
  if (sol.iis_witnesses.size() != iis.size()) {
    failures.push_back("witness count");
    return failures;
  }
  for (std::size_t w = 0; w < iis.size(); ++w) {
    const std::vector<double>& x = sol.iis_witnesses[w];
    std::vector<double> activity(static_cast<std::size_t>(model.num_rows()), 0.0);
    model.matrix.multiply_add(x.data(), activity.data());
    for (std::size_t e = 0; e < iis.size(); ++e) {
      const auto u = static_cast<std::size_t>(iis[e].index);
      const double v =
          iis[e].kind == 0
              ? verifier_violation(activity[u], model.row_lower[u], model.row_upper[u])
          : iis[e].kind == 1 ? verifier_violation(x[u], model.col_lower[u], kInfinity)
                             : verifier_violation(x[u], -kInfinity, model.col_upper[u]);
      if (e == w ? v <= 1e-7 : v > 1e-7) {
        failures.push_back("witness " + std::to_string(w) + " on element " + std::to_string(e) +
                           ": " + std::to_string(v));
      }
    }
  }
  return failures;
}

TEST(IIS, Klein2WitnessesUnderTheEmbeddingHoldInTheVerifiersMeasure) {
  const std::filesystem::path path =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
      "data/netlib-infeasible/klein2.mps";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "data/netlib-infeasible/klein2.mps is not fetched "
                    "(python bench/runners/fetch_netlib_infeasible.py klein2)";
  }
  Model model;
  const io::ReadResult read = io::read_model(path.string(), &model);
  ASSERT_TRUE(read.ok) << read.error;
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("ipm_hsd", true);
  const Solution sol = solve(model, options);
  ASSERT_EQ(sol.status, SolveStatus::kInfeasible) << sol.message;
  ASSERT_FALSE(sol.iis_rows.empty());
  const std::vector<std::string> failures = witness_failures(model, sol);
  EXPECT_TRUE(failures.empty()) << failures.size() << " failure(s), first: " << failures[0];
}

TEST(IIS, EveryClaimedWitnessHoldsInTheVerifiersMeasureOnRandomInfeasibleLps) {
  std::mt19937_64 rng(475217);
  oracle::GeneratorConfig config;
  int claimed = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    const Model model = oracle::to_model(lp);
    const Solution sol = solve(model, silent_options());
    if (sol.status != SolveStatus::kInfeasible || sol.iis_rows.empty()) continue;
    if (!sol.iis_inconclusive) ++claimed;
    const std::vector<std::string> failures = witness_failures(model, sol);
    EXPECT_TRUE(failures.empty()) << "trial " << trial << ": " << failures[0] << "\n"
                                  << lp.to_text();
  }
  EXPECT_GT(claimed, 20);
}

}  // namespace
}  // namespace sankhya
