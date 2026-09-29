// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a Farkas certificate from the elastic LP (#559).
//
// The elastic LP's row duals are offered as a certificate only after the in-process check
// accepts them, so the tests below hold both directions: an infeasible model gets a vector
// that farkas_proves_infeasible accepts, and a feasible model gets nothing at all.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "core/elastic_certificate.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

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
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

// x1 + x2 >= 3 with both columns in [0, 1]: the row needs 3, the box gives at most 2. A second,
// satisfiable row (x1 - x2 <= 5) must not spoil the certificate.
TEST(ElasticCertificate, InfeasibleModelGetsACertificateTheCheckAccepts) {
  const double inf = kInfinity;
  const Model model =
      make_lp({{1.0, 1.0}, {1.0, -1.0}}, {3.0, -inf}, {inf, 5.0}, {0.0, 0.0}, {1.0, 1.0});
  const std::vector<double> y = detail::farkas_from_elastic(model, quiet(), 10.0);
  ASSERT_EQ(y.size(), 2U);
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, y, &why)) << why;
}

// A multiplier may only lean on a side the row has: the returned vector never puts a
// positive weight on a row without a lower bound, or a negative one on a row without an
// upper bound.
TEST(ElasticCertificate, NoMultiplierLeansOnAMissingRowSide) {
  const double inf = kInfinity;
  const Model model =
      make_lp({{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}}, {4.0, -inf, -inf},
              {inf, 0.5, 0.5}, {0.0, 0.0, 0.0}, {2.0, 2.0, 2.0});
  const std::vector<double> y = detail::farkas_from_elastic(model, quiet(), 10.0);
  ASSERT_EQ(y.size(), 3U);
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (y[i] > 0.0) {
      EXPECT_TRUE(std::isfinite(model.row_lower[i])) << "row " << i;
    }
    if (y[i] < 0.0) {
      EXPECT_TRUE(std::isfinite(model.row_upper[i])) << "row " << i;
    }
  }
  std::string why;
  EXPECT_TRUE(farkas_proves_infeasible(model, y, &why)) << why;
}

// A feasible model has an elastic optimum of zero, and zero proves nothing.
TEST(ElasticCertificate, FeasibleModelGetsNothing) {
  const double inf = kInfinity;
  const Model model =
      make_lp({{1.0, 1.0}, {1.0, -1.0}}, {1.5, -inf}, {inf, 5.0}, {0.0, 0.0}, {1.0, 1.0});
  EXPECT_TRUE(detail::farkas_from_elastic(model, quiet(), 10.0).empty());
}

// Crossed column bounds are a contradiction no row multiplier can express.
TEST(ElasticCertificate, CrossedColumnBoundsAreLeftAlone) {
  const double inf = kInfinity;
  const Model model = make_lp({{1.0, 1.0}}, {0.0}, {inf}, {2.0, 0.0}, {1.0, 1.0});
  EXPECT_TRUE(detail::farkas_from_elastic(model, quiet(), 10.0).empty());
}

// No time, no attempt.
TEST(ElasticCertificate, NoBudgetNoAttempt) {
  const double inf = kInfinity;
  const Model model = make_lp({{1.0, 1.0}}, {3.0}, {inf}, {0.0, 0.0}, {1.0, 1.0});
  EXPECT_TRUE(detail::farkas_from_elastic(model, quiet(), 0.0).empty());
}

TEST(ElasticCertificate, OptionIsOnByDefault) {
  const Options options;
  EXPECT_TRUE(options.get_bool("certificate_elastic"));
}

}  // namespace
}  // namespace sankhya
