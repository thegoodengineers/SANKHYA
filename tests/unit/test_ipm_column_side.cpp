// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations on the n x n side (#469).
//
// The column side solves the same system as the row side, M dy = r with
// M = A Theta A^T + D, by conjugate gradients preconditioned with the Woodbury identity on
// the factors of N = Theta^-1 + A^T D^-1 A. Held here: on the constraint matrices of Netlib
// instances both sides give the same direction to 1e-10 relative, with some columns fixed
// (Theta 0) as the interior point has them; and the interior point on the column side
// reaches the optimum it reaches on the row side, on Netlib and on a tall model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/column_side.hpp"
#include "la/ldl.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

std::string netlib(const std::string& name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (name + ".mps"))
      .string();
}

double relative_difference(const std::vector<double>& x, const std::vector<double>& y) {
  double diff = 0.0;
  double size = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    diff = std::max(diff, std::fabs(x[i] - y[i]));
    size = std::max(size, std::fabs(y[i]));
  }
  return diff / std::max(size, 1e-300);
}

/// Theta log-uniform over two decades with every seventh column fixed (Theta 0), a logical
/// on every row (so M is well conditioned and the reference is accurate to what is
/// compared), the row side as the interior point runs it - LDL^T of M and two refinement
/// steps - against the column side.
void expect_same_direction(const std::string& name, double agreement) {
  Model model;
  const io::ReadResult read = io::read_model(netlib(name), &model);
  ASSERT_TRUE(read.ok) << name << ": " << read.error;
  const SparseMatrix& a = model.matrix;
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  std::mt19937_64 rng(static_cast<std::uint64_t>(m) * 104729u + static_cast<std::uint64_t>(n));
  std::uniform_real_distribution<double> decades(-1.0, 1.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> theta(static_cast<std::size_t>(n));
  std::vector<bool> fixed(static_cast<std::size_t>(n), false);
  for (std::size_t j = 0; j < theta.size(); ++j) {
    theta[j] = std::pow(10.0, decades(rng));
    if (j % 7 == 3) {
      fixed[j] = true;
      theta[j] = 0.0;
    }
  }
  std::vector<double> shift(static_cast<std::size_t>(m));
  for (double& v : shift) v = unit(rng) + 0.1;
  const double delta = 1e-10;
  std::vector<double> rhs(static_cast<std::size_t>(m));
  for (double& r : rhs) r = decades(rng);

  SparseMatrix lower_m;
  ASSERT_TRUE(normal_equations_lower(a, theta, shift, delta, &lower_m));
  SparseLdl rows;
  ASSERT_TRUE(rows.analyze(lower_m));
  ASSERT_TRUE(rows.factorize(lower_m, delta));
  std::vector<double> reference = rhs;
  rows.solve(reference.data());
  for (int step = 0; step < 2; ++step) {
    std::vector<double> residual(static_cast<std::size_t>(m), 0.0);
    std::vector<double> atv(static_cast<std::size_t>(n), 0.0);
    a.transpose_multiply_add(reference.data(), atv.data());
    for (std::size_t j = 0; j < atv.size(); ++j) atv[j] *= theta[j];
    a.multiply_add(atv.data(), residual.data());
    for (std::size_t i = 0; i < residual.size(); ++i) {
      residual[i] = rhs[i] - residual[i] - (shift[i] + delta) * reference[i];
    }
    rows.solve(residual.data());
    for (std::size_t i = 0; i < residual.size(); ++i) reference[i] += residual[i];
  }

  ipm::ColumnSide side;
  side.set_matrix(a, fixed);
  SparseMatrix lower_n;
  ASSERT_TRUE(side.assemble(theta, shift, delta, &lower_n, {}));
  ASSERT_EQ(lower_n.num_rows(), n) << name;
  SparseLdl columns;
  ASSERT_TRUE(columns.analyze(lower_n));
  ASSERT_TRUE(columns.factorize(lower_n, delta));
  std::vector<double> x = rhs;
  const ipm::ColumnSideReport report = side.solve(columns, x.data());
  EXPECT_TRUE(report.converged) << name << ": backward error " << report.backward_error;
  EXPECT_LE(report.backward_error, 1e-14) << name;
  EXPECT_LE(relative_difference(x, reference), agreement)
      << name << ", " << report.iterations << " CG steps";
}

TEST(InteriorPointColumnSide, TheDirectionIsTheRowSidesOnNetlib) {
  for (const char* name : {"afiro", "adlittle", "sc50a", "sc50b", "blend", "share2b", "scagr7",
                           "stocfor1", "bandm", "brandy", "scsd1", "ship04s"}) {
    expect_same_direction(name, 1e-10);
  }
}

TEST(InteriorPointColumnSide, BothSidesAreCountedExactlyBeforeTheyAreBuilt) {
  for (const char* name : {"afiro", "sc50a", "scagr7", "stocfor1", "israel"}) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib(name), &model).ok) << name;
    const SparseMatrix& a = model.matrix;
    const auto n = static_cast<std::size_t>(a.num_cols());
    const auto m = static_cast<std::size_t>(a.num_rows());
    std::vector<bool> fixed(n, false);
    std::vector<double> theta(n, 1.0);
    for (std::size_t j = 2; j < n; j += 5) {
      fixed[j] = true;
      theta[j] = 0.0;
    }
    const std::vector<double> ones(m, 1.0);
    SparseMatrix lower_m;
    ASSERT_TRUE(normal_equations_lower(a, theta, ones, 0.0, &lower_m));
    ipm::ColumnSide side;
    side.set_matrix(a, fixed);
    SparseMatrix lower_n;
    ASSERT_TRUE(side.assemble(theta, ones, 1e-10, &lower_n, {}));
    const std::int64_t count_m = lower_m.num_nonzeros();
    const std::int64_t count_n = lower_n.num_nonzeros();
    EXPECT_FALSE(side.row_side_exceeds(count_m, {})) << name;
    EXPECT_TRUE(side.row_side_exceeds(count_m - 1, {})) << name;
    EXPECT_FALSE(side.column_side_exceeds(count_n, {})) << name;
    EXPECT_TRUE(side.column_side_exceeds(count_n - 1, {})) << name;
  }
}

Options ipm_options(const char* side) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_string("ipm_normal_side", side);
  return options;
}

TEST(InteriorPointColumnSide, TheInteriorPointReachesTheSameOptimumOnNetlib) {
  for (const char* name : {"afiro", "adlittle", "sc50a", "sc50b", "blend", "share2b", "scagr7",
                           "stocfor1", "israel"}) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib(name), &model).ok) << name;
    const Solution rows = solve(model, ipm_options("rows"));
    const Solution columns = solve(model, ipm_options("columns"));
    ASSERT_EQ(rows.status, SolveStatus::kOptimal) << name << ": " << rows.message;
    ASSERT_EQ(columns.status, SolveStatus::kOptimal) << name << ": " << columns.message;
    EXPECT_NEAR(columns.objective, rows.objective,
                1e-8 * std::max(1.0, std::fabs(rows.objective)))
        << name;
  }
}

/// A tall model: 600 inequality rows over 20 columns, a random nonnegative cover
/// problem - min c x s.t. A x >= 1, x >= 0 - with every row meeting four columns.
Model tall_cover(std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<Index> pick(0, 19);
  std::uniform_real_distribution<double> value(0.5, 2.0);
  const Index m = 600;
  const Index n = 20;
  Model model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    std::vector<Index> used;
    while (used.size() < 4) {
      const Index j = pick(rng);
      if (std::find(used.begin(), used.end(), j) == used.end()) used.push_back(j);
    }
    for (const Index j : used) model.matrix.add_entry(i, j, value(rng));
  }
  model.matrix.finalize();
  model.col_cost.resize(static_cast<std::size_t>(n));
  for (double& c : model.col_cost) c = value(rng);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), 1.0);
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  return model;
}

TEST(InteriorPointColumnSide, ATallModelIsSolvedTheSameOnEitherSideAndByAuto) {
  const Model model = tall_cover(469);
  const Solution simplex = [&] {
    Options o;
    o.set_bool("log_to_console", false);
    o.set_string("algorithm", "dual-simplex");
    return solve(model, o);
  }();
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << simplex.message;
  for (const char* side : {"rows", "columns", "auto"}) {
    const Solution s = solve(model, ipm_options(side));
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << side << ": " << s.message;
    EXPECT_NEAR(s.objective, simplex.objective,
                1e-7 * std::max(1.0, std::fabs(simplex.objective)))
        << side;
  }
}

}  // namespace
}  // namespace sankhya
