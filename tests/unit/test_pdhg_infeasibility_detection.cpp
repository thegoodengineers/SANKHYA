// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG infeasibility/unboundedness detection from restart iterate differences
// (#484). See src/pdhg/pdhg.cpp's detect_certificate_from_restart() for the mechanism and
// its citation.
//
// Two properties, each its own test, matching #484's acceptance criteria exactly:
//   1. No optimal Netlib instance is ever reported infeasible - the false-positive risk this
//      feature exists to avoid, held over the nine committed reference instances
//      (test_pdhg_cuda_regression.cpp's own list, reused here rather than re-derived).
//   2. A genuinely infeasible model IS detected and carries a certificate that
//      tools/verify_solution.py-equivalent checks (farkas_proves_infeasible, the same
//      function the .sol writer and the verifier both hold every engine's certificate to)
//      accept - the positive case, so this file is not all negative controls.

#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

std::string repository_path(const char* relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

Options detection_options() {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("algorithm", "pdhg");
  o.set_bool("pdhg_detect_infeasibility", true);
  o.set_bool("pdhg_polish", false);  // isolate the first-order engine's own verdict
  o.set_int("iteration_limit", 1000000);
  return o;
}

// The nine instances committed in data/netlib/reference.json (test_pdhg_cuda_regression.cpp
// uses the same list); every one is genuinely optimal.
const std::vector<const char*> kNetlibInstances = {
    "afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1", "israel",
};

TEST(PdhgInfeasibilityDetection, NoOptimalNetlibInstanceIsEverReportedInfeasible) {
  for (const char* name : kNetlibInstances) {
    const std::string path =
        repository_path((std::string("data/netlib/") + name + ".mps").c_str());
    Model model;
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;

    const Solution solution = solve(model, detection_options());
    EXPECT_NE(solution.status, SolveStatus::kInfeasible)
        << name << ": a genuinely optimal instance must never be reported infeasible, "
        << "message: " << solution.message;
    EXPECT_NE(solution.status, SolveStatus::kUnbounded)
        << name << ": a genuinely optimal instance must never be reported unbounded, "
        << "message: " << solution.message;
  }
}

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
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
  return model;
}

TEST(PdhgInfeasibilityDetection, AGenuinelyInfeasibleModelIsDetectedWithAValidCertificate) {
  // x + y <= 1  and  x + y >= 3, x,y >= 0: 1 < 3, infeasible by construction, small enough
  // for PDHG to find a ray quickly rather than needing the full iteration budget.
  const Model model = make_lp({{1.0, 1.0}, {1.0, 1.0}}, {-kInfinity, 3.0}, {1.0, kInfinity},
                              {0.0, 0.0}, {0.0, 0.0}, {kInfinity, kInfinity});

  const Solution solution = solve(model, detection_options());
  ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  ASSERT_FALSE(solution.farkas_dual.empty())
      << "reported infeasible but carries no certificate: " << solution.message;

  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual, &why)) << why;
}

/// Columns x0.., rows given densely with [row_lo, row_hi], x >= 0 with no upper bound.
Model dense_lp(const std::vector<double>& cost, const std::vector<std::vector<double>>& rows,
               const std::vector<double>& row_lo, const std::vector<double>& row_hi) {
  Model m;
  const auto n = static_cast<Index>(cost.size());
  const auto r = static_cast<Index>(rows.size());
  m.resize_columns(n);
  m.col_cost = cost;
  m.resize_rows(r);
  m.row_lower = row_lo;
  m.row_upper = row_hi;
  m.matrix.reset(r, n);
  for (Index i = 0; i < r; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) m.matrix.add_entry(i, j, v);
    }
  }
  m.matrix.finalize();
  return m;
}

TEST(PdhgInfeasibilityDetection, AnInfeasibleModelWithAnImprovingRayIsNotUnbounded) {
  // min -x1; x2 + x3 <= 1; 2 x2 + 3 x3 >= 4; x >= 0. Infeasible (x2 + x3 <= 1 caps
  // 2 x2 + 3 x3 at 3), and x1 is an improving direction no row touches. A ray alone proves
  // the dual infeasible, not the primal feasible, so the engine must not say unbounded
  // (review of #652: it did, at iteration 1).
  const Model model = dense_lp({-1.0, 0.0, 0.0}, {{0.0, 1.0, 1.0}, {0.0, 2.0, 3.0}},
                               {-kInfinity, 4.0}, {1.0, kInfinity});
  for (const bool halpern : {false, true}) {
    Options o = detection_options();
    o.set_bool("presolve", false);
    o.set_bool("pdhg_halpern", halpern);
    if (halpern) o.set_bool("pdhg_restart", false);
    const Solution s = solve(model, o);
    EXPECT_NE(s.status, SolveStatus::kUnbounded) << (halpern ? "halpern: " : "") << s.message;
    EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  }
}

TEST(PdhgInfeasibilityDetection, FeasibleBoundedBadlyScaledLpsAreNeverCertifiedOtherwise) {
  // A x <= b with A > 0, b > 0 and x >= 0: x = 0 is feasible and every column is bounded by
  // the rows, so the LP is feasible and bounded. Rows scaled by 10^k for k in [-3, 3]. In
  // review of #652, 8 such LPs came back infeasible and 41 unbounded, from near-converged
  // restart differences small enough to slip under the checkers' absolute floors.
  std::mt19937_64 rng(4840);
  std::uniform_real_distribution<double> entry(0.1, 50.0);
  std::uniform_int_distribution<int> power(-3, 3);
  std::uniform_real_distribution<double> price(-5.0, 5.0);
  int solved = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const std::size_t n = 6 + static_cast<std::size_t>(trial % 7);
    const std::size_t m = 4 + static_cast<std::size_t>(trial % 5);
    std::vector<std::vector<double>> a(m, std::vector<double>(n));
    std::vector<double> hi(m);
    for (std::size_t i = 0; i < m; ++i) {
      const double scale = std::pow(10.0, power(rng));
      for (std::size_t j = 0; j < n; ++j) a[i][j] = scale * entry(rng);
      hi[i] = scale * 100.0;
    }
    std::vector<double> cost(n);
    for (double& c : cost) c = price(rng);
    const Model model = dense_lp(cost, a, std::vector<double>(m, -kInfinity), hi);
    for (const bool halpern : {false, true}) {
      Options o = detection_options();
      o.set_bool("presolve", false);
      o.set_bool("pdhg_halpern", halpern);
      if (halpern) o.set_bool("pdhg_restart", false);
      o.set_int("iteration_limit", 200000);
      const Solution s = solve(model, o);
      EXPECT_NE(s.status, SolveStatus::kInfeasible) << "trial " << trial << " " << s.message;
      EXPECT_NE(s.status, SolveStatus::kUnbounded) << "trial " << trial << " " << s.message;
      EXPECT_NE(s.status, SolveStatus::kInfeasibleOrUnbounded)
          << "trial " << trial << " " << s.message;
      ++solved;
    }
  }
  EXPECT_EQ(solved, 120);
}

}  // namespace
}  // namespace sankhya
