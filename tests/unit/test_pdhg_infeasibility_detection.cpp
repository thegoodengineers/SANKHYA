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

#include <filesystem>
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

}  // namespace
}  // namespace sankhya
