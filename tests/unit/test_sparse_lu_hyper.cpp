// SPDX-License-Identifier: Apache-2.0
// SANKHYA - hyper-sparse FTRAN and BTRAN (#464).
//
// The claim in lu_hyper.cpp is not "an accurate solve" but THE SAME solve: the reached steps
// are visited in the order the full loops visit them, so every result must equal the full
// loops' to the last bit (a -0.0 against a +0.0 compares equal, which is the one difference
// the file owns up to). That is what lets the simplex keep its pivots, and so its iteration
// counts, with the option on. Held here three ways: random sparse bases, on fresh factors
// and after product-form updates, with unit, sparse and dense right-hand sides (the last
// exercises the fall-back); a residual through the original matrix so that "equal to the
// full loops" cannot hide a shared mistake; and whole simplex solves on Netlib, primal and
// dual, whose iteration counts and objectives must not move.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "la/lu.hpp"
#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

/// A random sparse nonsingular basis: a permuted diagonal plus a few entries per column,
/// the shape of a simplex basis that is mostly slacks with some structural columns.
struct Basis {
  Index m = 0;
  std::vector<std::vector<Index>> rows;
  std::vector<std::vector<double>> values;

  [[nodiscard]] std::vector<LuColumn> columns() const {
    std::vector<LuColumn> out(static_cast<std::size_t>(m));
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j].size = static_cast<Index>(rows[j].size());
      out[j].rows = rows[j].data();
      out[j].values = values[j].data();
    }
    return out;
  }

  /// B x, for the residual check.
  [[nodiscard]] std::vector<double> times(const std::vector<double>& x) const {
    std::vector<double> out(static_cast<std::size_t>(m), 0.0);
    for (std::size_t j = 0; j < rows.size(); ++j) {
      for (std::size_t p = 0; p < rows[j].size(); ++p) {
        out[static_cast<std::size_t>(rows[j][p])] += values[j][p] * x[j];
      }
    }
    return out;
  }
  /// B^T y.
  [[nodiscard]] std::vector<double> transpose_times(const std::vector<double>& y) const {
    std::vector<double> out(static_cast<std::size_t>(m), 0.0);
    for (std::size_t j = 0; j < rows.size(); ++j) {
      for (std::size_t p = 0; p < rows[j].size(); ++p) {
        out[j] += values[j][p] * y[static_cast<std::size_t>(rows[j][p])];
      }
    }
    return out;
  }
};

std::vector<Index> sample_rows(std::mt19937* rng, Index m, Index count) {
  std::uniform_int_distribution<Index> row(0, m - 1);
  std::vector<Index> out;
  while (static_cast<Index>(out.size()) < count) {
    const Index r = row(*rng);
    if (std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
  }
  return out;
}

Basis random_basis(std::mt19937* rng, Index m, int extra_per_column) {
  Basis basis;
  basis.m = m;
  basis.rows.resize(static_cast<std::size_t>(m));
  basis.values.resize(static_cast<std::size_t>(m));
  std::vector<Index> permutation(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) permutation[static_cast<std::size_t>(i)] = i;
  std::shuffle(permutation.begin(), permutation.end(), *rng);
  std::uniform_real_distribution<double> diagonal(1.0, 4.0);
  std::uniform_real_distribution<double> off(-1.0, 1.0);
  std::uniform_int_distribution<int> extras(0, extra_per_column);
  for (Index j = 0; j < m; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    const Index d = permutation[uj];
    basis.rows[uj].push_back(d);
    basis.values[uj].push_back(diagonal(*rng) * (off(*rng) < 0.0 ? -1.0 : 1.0));
    for (const Index r : sample_rows(rng, m, extras(*rng))) {
      if (r == d) continue;
      basis.rows[uj].push_back(r);
      basis.values[uj].push_back(0.3 * off(*rng));
    }
  }
  return basis;
}

std::vector<double> sparse_vector(std::mt19937* rng, Index m, Index nonzeros) {
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::vector<double> out(static_cast<std::size_t>(m), 0.0);
  for (const Index r : sample_rows(rng, m, nonzeros))
    out[static_cast<std::size_t>(r)] = value(*rng);
  return out;
}

/// FTRAN and BTRAN of `rhs` through both factorizations: equal to the bit, and small
/// residuals when `basis` is the matrix the factors represent (no updates applied).
void expect_same_solves(const SparseLu& full, const SparseLu& hyper,
                        const std::vector<double>& rhs, const Basis* basis,
                        const std::string& where) {
  std::vector<double> x_full = rhs;
  std::vector<double> x_hyper = rhs;
  full.solve(x_full.data());
  hyper.solve(x_hyper.data());
  std::vector<double> y_full = rhs;
  std::vector<double> y_hyper = rhs;
  full.solve_transpose(y_full.data());
  hyper.solve_transpose(y_hyper.data());
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    ASSERT_EQ(x_full[i], x_hyper[i]) << where << ": FTRAN entry " << i;
    ASSERT_EQ(y_full[i], y_hyper[i]) << where << ": BTRAN entry " << i;
  }
  if (basis == nullptr) return;
  // The residual is a guard against a mistake the two paths would share, which shows as an
  // O(1) residual, not a measure of accuracy: threshold pivoting at tau = 0.01 allows element
  // growth, and one of these matrices (trial 31, m = 137) leaves 3.2e-9 on a dense
  // right-hand side through the full loops as well. Hence 1e-8 relative to the solution.
  double x_norm = 0.0;
  double y_norm = 0.0;
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    x_norm = std::max(x_norm, std::fabs(x_hyper[i]));
    y_norm = std::max(y_norm, std::fabs(y_hyper[i]));
  }
  const std::vector<double> bx = basis->times(x_hyper);
  const std::vector<double> bty = basis->transpose_times(y_hyper);
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    EXPECT_NEAR(bx[i], rhs[i], 1e-8 * (1.0 + x_norm)) << where << ": B x residual at " << i;
    EXPECT_NEAR(bty[i], rhs[i], 1e-8 * (1.0 + y_norm)) << where << ": B^T y residual at " << i;
  }
}

TEST(SparseLuHyper, SolvesAreTheFullLoopsToTheBitOnFreshAndUpdatedFactors) {
  std::mt19937 rng(464);
  std::int64_t hyper_ftrans = 0;
  std::int64_t hyper_btrans = 0;
  for (int trial = 0; trial < 40; ++trial) {
    std::uniform_int_distribution<Index> size(80, 600);
    const Index m = size(rng);
    const Basis basis = random_basis(&rng, m, trial % 3 == 0 ? 1 : 3);
    SparseLu full;
    SparseLu hyper;
    hyper.use_hyper_sparse(true);
    const std::vector<LuColumn> columns = basis.columns();
    ASSERT_TRUE(full.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold));
    ASSERT_TRUE(hyper.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold));

    const std::string tag = "trial " + std::to_string(trial) + ", m " + std::to_string(m);
    for (int k = 0; k < 20; ++k) {
      std::vector<double> unit(static_cast<std::size_t>(m), 0.0);
      unit[static_cast<std::size_t>(sample_rows(&rng, m, 1)[0])] = 1.0;
      expect_same_solves(full, hyper, unit, &basis, tag + " unit");
      expect_same_solves(full, hyper, sparse_vector(&rng, m, 3), &basis, tag + " sparse");
    }
    // A dense right-hand side takes the full loops in both, and must still agree.
    expect_same_solves(full, hyper, sparse_vector(&rng, m, m / 2), &basis, tag + " dense");

    // Product-form updates: the base factors are unchanged, the etas are applied around
    // them, and both instances record the same etas.
    for (int update = 0; update < 12; ++update) {
      const std::vector<double> entering = sparse_vector(&rng, m, 4);
      std::vector<double> alpha = entering;
      full.solve(alpha.data());
      const Index leaving = sample_rows(&rng, m, 1)[0];
      const bool a = full.update(leaving, alpha.data());
      const bool b = hyper.update(leaving, alpha.data());
      ASSERT_EQ(a, b) << tag;
      if (!a) break;
      std::vector<double> unit(static_cast<std::size_t>(m), 0.0);
      unit[static_cast<std::size_t>(sample_rows(&rng, m, 1)[0])] = 1.0;
      expect_same_solves(full, hyper, unit, nullptr, tag + " after updates");
      expect_same_solves(full, hyper, sparse_vector(&rng, m, 2), nullptr,
                         tag + " after updates");
    }
    hyper_ftrans += hyper.solve_stats().ftran_hyper;
    hyper_btrans += hyper.solve_stats().btran_hyper;
    EXPECT_EQ(full.solve_stats().ftran_hyper, 0);
  }
  // The path under test was actually taken, not fallen back from every time. Of about
  // 2,600 solves each way, the ones whose reach stays under 10% of m: the transposed reach
  // through the row-wise L is the wider of the two on these matrices.
  std::cout << "[ hyper ] symbolic-reach solves: " << hyper_ftrans << " FTRAN, " << hyper_btrans
            << " BTRAN\n";
  EXPECT_GT(hyper_ftrans, 500);
  EXPECT_GT(hyper_btrans, 500);
}

TEST(SparseLuHyper, NetlibIterationCountsAndObjectivesDoNotMove) {
  // The acceptance item of #464: the arithmetic is the same, so the simplex takes the same
  // pivots. Primal and dual, presolve off so the engines do the work.
  const char* instances[] = {"afiro",  "sc50b",   "share2b", "stocfor1",
                             "degen2", "ship04s", "sctap1"};
  for (const char* algorithm : {"simplex", "dual-simplex"}) {
    for (const char* name : instances) {
      Model model;
      const std::string path =
          (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
           "data/netlib" / (std::string(name) + ".mps"))
              .string();
      const io::ReadResult read = io::read_model(path, &model);
      ASSERT_TRUE(read.ok) << path << ": " << read.error;
      Solution runs[2];
      for (int hyper = 0; hyper < 2; ++hyper) {
        Options options;
        options.set_bool("log_to_console", false);
        options.set_bool("presolve", false);
        options.set_string("algorithm", algorithm);
        options.set_bool("lu_hyper_sparse", hyper == 1);
        runs[hyper] = solve(model, options);
      }
      EXPECT_EQ(runs[0].status, SolveStatus::kOptimal) << name << " " << algorithm;
      EXPECT_EQ(runs[0].status, runs[1].status) << name << " " << algorithm;
      EXPECT_EQ(runs[0].iterations, runs[1].iterations) << name << " " << algorithm;
      EXPECT_EQ(runs[0].objective, runs[1].objective) << name << " " << algorithm;
    }
  }
}

}  // namespace
}  // namespace sankhya
