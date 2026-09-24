// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense columns in the interior point's normal equations (#467).
//
// Three things are held here. The prediction of the size of A Theta A^T is exact when it
// says so, and a bound in the right direction when it does not. The dense-column solve -
// the sparse factor, the Woodbury correction, conjugate gradients on the whole system -
// gives the direction the plain factorization of the whole system gives, to 1e-10
// relative, on the constraint matrices of Netlib instances. And the interior point with
// the option on reaches the optimum it reaches without it, while a normal-equations matrix
// over the factor budget is declined before it is built.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/dense_columns.hpp"
#include "la/ldl.hpp"
#include "la/normal_pattern.hpp"
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

SparseMatrix random_matrix(std::mt19937_64& rng, Index m, Index n, double density,
                           Index dense_columns) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  SparseMatrix a(m, n);
  for (Index j = 0; j < n; ++j) {
    const double d = j < dense_columns ? 0.8 : density;
    bool any = false;
    for (Index i = 0; i < m; ++i) {
      if (unit(rng) < d) {
        a.add_entry(i, j, value(rng) + (value(rng) > 0 ? 0.5 : -0.5));
        any = true;
      }
    }
    if (!any) a.add_entry(static_cast<Index>(unit(rng) * static_cast<double>(m)), j, 1.0);
  }
  a.finalize(0.0);
  return a;
}

std::int64_t assembled_nonzeros(const SparseMatrix& a, const std::vector<char>& skip) {
  std::vector<double> theta(static_cast<std::size_t>(a.num_cols()), 1.0);
  for (std::size_t j = 0; j < skip.size(); ++j) {
    if (skip[j] != 0) theta[j] = 0.0;
  }
  SparseMatrix lower;
  EXPECT_TRUE(normal_equations_lower(a, theta, {}, 1e-10, &lower));
  return lower.num_nonzeros();
}

TEST(NormalPattern, TheExactCountIsTheAssembledCount) {
  std::mt19937_64 rng(467);
  for (int trial = 0; trial < 20; ++trial) {
    const SparseMatrix a = random_matrix(rng, 60, 90, 0.05, trial % 3);
    std::vector<char> skip(90, 0);
    if (trial % 2 == 1) {
      for (std::size_t j = 0; j < skip.size(); j += 7) skip[j] = 1;
    }
    const std::int64_t truth = assembled_nonzeros(a, skip);
    // No cap: the upper bound, which is an upper bound.
    const NormalPrediction loose = predict_normal_nonzeros(a, skip, -1);
    EXPECT_GE(loose.nonzeros, truth) << "trial " << trial;
    // A cap just above the truth sits between the bounds (or at the upper one): the count
    // is exact or a bound that is not over the cap.
    const NormalPrediction exact = predict_normal_nonzeros(a, skip, truth);
    EXPECT_FALSE(exact.over_cap) << "trial " << trial;
    if (exact.exact) {
      EXPECT_EQ(exact.nonzeros, truth) << "trial " << trial;
    }
    EXPECT_GE(exact.nonzeros, truth) << "trial " << trial;
    // One below the truth: over the cap, and the reported count is a true lower bound.
    const NormalPrediction over = predict_normal_nonzeros(a, skip, truth - 1);
    EXPECT_TRUE(over.over_cap) << "trial " << trial;
    EXPECT_LE(over.nonzeros, truth) << "trial " << trial;
    EXPECT_GT(over.nonzeros, truth - 1) << "trial " << trial;
  }
}

TEST(NormalPattern, OneDenseColumnIsRefusedFromItsCountAlone) {
  // 3,000 rows, one column meeting all of them: its block alone is 4.5 million entries.
  const Index m = 3000;
  SparseMatrix a(m, m + 1);
  for (Index i = 0; i < m; ++i) {
    a.add_entry(i, i, 1.0);
    a.add_entry(i, m, 1.0);
  }
  a.finalize(0.0);
  const NormalPrediction p = predict_normal_nonzeros(a, {}, 1000000);
  EXPECT_TRUE(p.over_cap);
  EXPECT_EQ(p.nonzeros, static_cast<std::int64_t>(m) + std::int64_t{m} * (m - 1) / 2);
  // Without that column the matrix is its diagonal.
  std::vector<char> skip(static_cast<std::size_t>(m + 1), 0);
  skip[static_cast<std::size_t>(m)] = 1;
  const NormalPrediction q = predict_normal_nonzeros(a, skip, 1000000);
  EXPECT_FALSE(q.over_cap);
  EXPECT_EQ(q.nonzeros, m);
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

/// The normal equations of a Netlib instance's A, theta log-uniform over two decades and a
/// logical theta on every row, solved two ways: the plain LDL^T of the whole M with two
/// refinement steps (the default path), and the dense-column correction with the `k`
/// densest columns split off. Every row carries a logical so that M is well conditioned and
/// the reference itself is accurate to the 1e-10 compared: with equality rows on a
/// rank-deficient A (brandy) M is singular up to delta = 1e-10, and neither path has a
/// direction to 1e-10 to compare. Equality rows are exercised by the solves below.
void expect_same_direction(const std::string& name, Index k, double agreement) {
  Model model;
  const io::ReadResult read = io::read_model(netlib(name), &model);
  ASSERT_TRUE(read.ok) << name << ": " << read.error;
  const SparseMatrix& a = model.matrix;
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  std::mt19937_64 rng(static_cast<std::uint64_t>(m) * 7919u + static_cast<std::uint64_t>(n));
  std::uniform_real_distribution<double> decades(-1.0, 1.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> theta(static_cast<std::size_t>(n));
  for (double& t : theta) t = std::pow(10.0, decades(rng));
  std::vector<double> shift(static_cast<std::size_t>(m), 0.0);
  for (double& v : shift) v = unit(rng) + 0.1;
  const double delta = 1e-10;
  std::vector<double> rhs(static_cast<std::size_t>(m));
  for (double& r : rhs) r = decades(rng);

  // The whole system, factored.
  SparseMatrix full;
  ASSERT_TRUE(normal_equations_lower(a, theta, shift, delta, &full));
  SparseLdl whole;
  ASSERT_TRUE(whole.analyze(full));
  ASSERT_TRUE(whole.factorize(full, delta));
  std::vector<double> reference = rhs;
  whole.solve(reference.data());
  for (int step = 0; step < 2; ++step) {
    std::vector<double> residual(static_cast<std::size_t>(m), 0.0);
    std::vector<double> atv(static_cast<std::size_t>(n), 0.0);
    a.transpose_multiply_add(reference.data(), atv.data());
    for (Index j = 0; j < n; ++j)
      atv[static_cast<std::size_t>(j)] *= theta[static_cast<std::size_t>(j)];
    a.multiply_add(atv.data(), residual.data());
    for (Index i = 0; i < m; ++i) {
      const auto u = static_cast<std::size_t>(i);
      residual[u] = rhs[u] - residual[u] - (shift[u] + delta) * reference[u];
    }
    whole.solve(residual.data());
    for (Index i = 0; i < m; ++i)
      reference[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
  }

  // The dense-column path: the k densest columns split off.
  ipm::DenseColumnCorrection correction;
  correction.set_columns(a, ipm::find_dense_columns(a, {}, 1e-9, k));
  ASSERT_EQ(static_cast<Index>(correction.columns().size()), std::min(k, n)) << name;
  std::vector<double> sparse_theta;
  correction.sparse_theta(theta, &sparse_theta);
  std::vector<double> assembly_shift;
  (void)correction.preconditioner_shift(a, theta, shift, delta, &assembly_shift);
  SparseMatrix sparse;
  ASSERT_TRUE(normal_equations_lower(a, sparse_theta, assembly_shift, delta, &sparse));
  SparseLdl part;
  ASSERT_TRUE(part.analyze(sparse));
  ASSERT_TRUE(part.factorize(sparse, delta));
  ASSERT_TRUE(correction.prepare(part, a, theta, shift, delta)) << name;
  std::vector<double> x = rhs;
  const ipm::PcgReport report = correction.solve(x.data());
  EXPECT_TRUE(report.converged) << name << ": residual " << report.relative_residual;
  // The backward error is at the rounding floor whatever the conditioning.
  EXPECT_LE(report.relative_residual, 1e-14) << name << " with " << k << " dense columns";
  EXPECT_LE(relative_difference(x, reference), agreement)
      << name << " with " << k << " dense columns, " << report.iterations << " CG steps";
}

TEST(DenseColumnCorrection, TheDirectionIsTheWholeSystemsOnNetlib) {
  for (const char* name : {"afiro", "adlittle", "sc50a", "blend", "share2b", "scagr7",
                           "stocfor1", "bandm", "brandy"}) {
    for (const Index k : {1, 3, 10}) expect_same_direction(name, k, 1e-10);
  }
}

// ISRAEL IS HELD TO WHAT ITS CONDITIONING ALLOWS. Its normal equations here have terms of
// 2e7 against a right-hand side of order one; both solves reach a backward error of 1e-16
// and still differ by 6e-10 relative, which puts the condition number near 6e6: neither
// direction is known to 1e-10, and a test demanding it would be testing the reference.
TEST(DenseColumnCorrection, TheDirectionOnIsraelAgreesToItsConditioning) {
  for (const Index k : {1, 3, 10}) expect_same_direction("israel", k, 1e-8);
}

TEST(DenseColumnCorrection, FindsOnlyColumnsOverTheThreshold) {
  SparseMatrix a(100, 4);
  for (Index i = 0; i < 100; ++i) a.add_entry(i, 1, 1.0);  // 100 entries
  for (Index i = 0; i < 60; ++i) a.add_entry(i, 3, 1.0);   // 60 entries
  for (Index i = 0; i < 5; ++i) a.add_entry(i, 0, 1.0);    // 5 entries
  a.add_entry(7, 2, 1.0);
  a.finalize(0.0);
  // 10 sqrt(100) = 100: nothing is over it; 5 sqrt(100) = 50: columns 1 and 3.
  EXPECT_TRUE(ipm::find_dense_columns(a, {}, 10.0, 100).empty());
  EXPECT_EQ(ipm::find_dense_columns(a, {}, 5.0, 100), (std::vector<Index>{1, 3}));
  // The cap keeps the densest; ineligible columns are never chosen.
  EXPECT_EQ(ipm::find_dense_columns(a, {}, 5.0, 1), (std::vector<Index>{1}));
  EXPECT_EQ(ipm::find_dense_columns(a, {1, 0, 1, 1}, 5.0, 100), (std::vector<Index>{3}));
}

Options ipm_options(bool dense) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("ipm_dense_columns", dense);
  options.set_double("ipm_dense_column_factor", 2.0);
  return options;
}

TEST(DenseColumnCorrection, TheInteriorPointReachesTheSameOptimumOnNetlib) {
  for (const char* name :
       {"afiro", "adlittle", "sc50a", "blend", "share2b", "scagr7", "israel", "stocfor1"}) {
    Model model;
    const io::ReadResult read = io::read_model(netlib(name), &model);
    ASSERT_TRUE(read.ok) << name;
    const Solution plain = solve(model, ipm_options(false));
    const Solution split = solve(model, ipm_options(true));
    ASSERT_EQ(plain.status, SolveStatus::kOptimal) << name << ": " << plain.message;
    ASSERT_EQ(split.status, SolveStatus::kOptimal) << name << ": " << split.message;
    EXPECT_NEAR(split.objective, plain.objective,
                1e-8 * std::max(1.0, std::fabs(plain.objective)))
        << name;
  }
}

/// min sum x_i + 2 y  s.t.  x_i + y >= 1 for m rows, x, y >= 0: the column y meets every
/// row. y = 1 covers every row at cost 2, which beats sum x_i = m, so the optimum is 2.
Model one_dense_column(Index m) {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.matrix.reset(m, m + 1);
  for (Index i = 0; i < m; ++i) {
    model.matrix.add_entry(i, i, 1.0);
    model.matrix.add_entry(i, m, 1.0);
  }
  model.matrix.finalize();
  model.col_cost.assign(static_cast<std::size_t>(m + 1), 1.0);
  model.col_cost[static_cast<std::size_t>(m)] = 2.0;
  model.col_lower.assign(static_cast<std::size_t>(m + 1), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(m + 1), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(m + 1), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), 1.0);
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  return model;
}

TEST(DenseColumnCorrection, OverBudgetIsDeclinedBeforeAssemblyAndSolvedWithTheOption) {
  const Model model = one_dense_column(3000);
  Options off = ipm_options(false);
  off.set_bool("presolve", false);
  off.set_int("ipm_max_factor_nonzeros", 1000000);
  const Solution declined = solve(model, off);
  EXPECT_EQ(declined.status, SolveStatus::kNumericalError) << declined.message;
  EXPECT_NE(declined.message.find("declined before assembly"), std::string::npos)
      << declined.message;
  EXPECT_EQ(declined.iterations, 0);

  Options on = off;
  on.set_bool("ipm_dense_columns", true);
  on.set_double("ipm_dense_column_factor", 10.0);
  const Solution solved = solve(model, on);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, 2.0, 1e-7);
}

}  // namespace
}  // namespace sankhya
