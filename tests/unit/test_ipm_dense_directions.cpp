// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the dense-column directions against the default path's, on the systems the
// interior point really meets (#467).
//
// test_ipm_dense_columns.cpp compares the two solves on Netlib constraint matrices with a
// synthetic Theta over two decades and a logical on every row: systems conditioned well
// enough that both answers are known to 1e-10. The interior point does not stay there. Near
// the optimum Theta spans twenty decades, M = A Theta A^T + D + delta I approaches
// singularity, and no solver, ours or another, knows the direction to 1e-10. So this test
// records every Newton-direction solve of a real run with the dense columns forced (factor
// 1: each of the instances below then has between 5 and 100), solves each of those systems
// again the way the default path does - the sparse LDL^T of the whole M, pivots lifted at
// delta, two steps of iterative refinement (ipm.cpp, solve_normal) - and holds the two to
//
//     ||x_dense - x_default||_inf / ||x_default||_inf
//         <= max(1e-10, 10 * kappa_inf(M) * (eta_dense + eta_default)),
//
// the first-order perturbation bound for two backward-stable solves of one system (Higham,
// "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM 2002, thm. 7.2), with eta
// the normwise backward error of each answer (Rigal and Gaches; Higham sec. 7.1) and kappa
// estimated by Hager's method (Higham sec. 15.3; Hager, SIAM J. Sci. Stat. Comput. 5, 1984).
// Where M is well conditioned the bound is below 1e-10 and the test demands 1e-10; where it
// is not, the test demands what the conditioning allows of EITHER path, and says how many
// solves were held to 1e-10 outright.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ipm/dense_columns.hpp"
#include "ipm/ipm_testing.hpp"
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

/// One recorded dense-column solve: the system M = A Theta A^T + diag(shift) + delta I, the
/// right-hand side, the answer and whether the conjugate gradients called it converged.
struct Recorded {
  SparseMatrix a;
  std::vector<double> theta;
  std::vector<double> shift;
  double delta = 0.0;
  std::vector<double> rhs;
  std::vector<double> solution;
  bool converged = false;
};

double inf_norm(const std::vector<double>& v) {
  double largest = 0.0;
  for (const double x : v) largest = std::max(largest, std::fabs(x));
  return largest;
}

/// The whole system of one recorded solve, factored the way the default path factors it.
class WholeSystem {
 public:
  explicit WholeSystem(const Recorded& r) : r_(r) {
    ok_ = normal_equations_lower(r.a, r.theta, r.shift, r.delta, &lower_) &&
          ldl_.analyze(lower_) && ldl_.factorize(lower_, r.delta);
    // ||M||_inf from the stored lower triangle: every off-diagonal entry counts in two rows.
    std::vector<double> rows(static_cast<std::size_t>(lower_.num_rows()), 0.0);
    for (Index j = 0; j < lower_.num_cols(); ++j) {
      const ColumnView column = lower_.column(j);
      for (Index p = 0; p < column.size; ++p) {
        const Index i = column.rows[p];
        rows[static_cast<std::size_t>(i)] += std::fabs(column.values[p]);
        if (i != j) rows[static_cast<std::size_t>(j)] += std::fabs(column.values[p]);
      }
    }
    norm_ = inf_norm(rows);
  }
  [[nodiscard]] bool ok() const { return ok_; }
  [[nodiscard]] double norm() const { return norm_; }

  /// M v, from A: the product the dense-column path's conjugate gradients use.
  [[nodiscard]] std::vector<double> multiply(const std::vector<double>& v) const {
    std::vector<double> atv(static_cast<std::size_t>(r_.a.num_cols()), 0.0);
    r_.a.transpose_multiply_add(v.data(), atv.data());
    for (std::size_t j = 0; j < atv.size(); ++j) atv[j] *= r_.theta[j];
    std::vector<double> out(v.size(), 0.0);
    r_.a.multiply_add(atv.data(), out.data());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] += (r_.shift[i] + r_.delta) * v[i];
    return out;
  }

  /// The default path's solve: the factor, then two steps of iterative refinement.
  [[nodiscard]] std::vector<double> solve(const std::vector<double>& b) const {
    std::vector<double> x = b;
    ldl_.solve(x.data());
    for (int step = 0; step < 2; ++step) {
      const std::vector<double> mx = multiply(x);
      std::vector<double> residual(b.size());
      for (std::size_t i = 0; i < b.size(); ++i) residual[i] = b[i] - mx[i];
      ldl_.solve(residual.data());
      for (std::size_t i = 0; i < b.size(); ++i) x[i] += residual[i];
    }
    return x;
  }

  /// The normwise backward error of x for M x = b (Rigal and Gaches).
  [[nodiscard]] double backward_error(const std::vector<double>& x,
                                      const std::vector<double>& b) const {
    const std::vector<double> mx = multiply(x);
    std::vector<double> residual(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) residual[i] = b[i] - mx[i];
    return inf_norm(residual) / (norm_ * inf_norm(x) + inf_norm(b));
  }

  /// kappa_inf(M) = ||M||_inf ||M^-1||_inf, the second factor by Hager's estimator of the
  /// 1-norm of M^-1, which for a symmetric M is its infinity norm. A lower bound, in
  /// practice within a small factor; the test's factor of 10 covers it.
  [[nodiscard]] double condition() const {
    const std::size_t m = static_cast<std::size_t>(lower_.num_rows());
    std::vector<double> x(m, 1.0 / static_cast<double>(m));
    double estimate = 0.0;
    for (int step = 0; step < 5; ++step) {
      const std::vector<double> y = solve(x);
      double one_norm = 0.0;
      for (const double v : y) one_norm += std::fabs(v);
      if (step > 0 && one_norm <= estimate) break;
      estimate = one_norm;
      std::vector<double> sign(m);
      for (std::size_t i = 0; i < m; ++i) sign[i] = y[i] < 0.0 ? -1.0 : 1.0;
      const std::vector<double> z = solve(sign);
      std::size_t best = 0;
      double zx = 0.0;
      for (std::size_t i = 0; i < m; ++i) {
        zx += z[i] * x[i];
        if (std::fabs(z[i]) > std::fabs(z[best])) best = i;
      }
      if (std::fabs(z[best]) <= zx) break;
      std::fill(x.begin(), x.end(), 0.0);
      x[best] = 1.0;
    }
    return norm_ * estimate;
  }

 private:
  const Recorded& r_;
  SparseMatrix lower_;
  SparseLdl ldl_;
  bool ok_ = false;
  double norm_ = 0.0;
};

std::vector<Recorded> record_run(const std::string& name, Solution* solution) {
  Model model;
  const io::ReadResult read = io::read_model(netlib(name), &model);
  EXPECT_TRUE(read.ok) << name << ": " << read.error;
  std::vector<Recorded> recorded;
  ipm::testing::dense_solve_observer =
      [&](const ipm::DenseColumnCorrection& correction, const std::vector<double>& rhs,
          const std::vector<double>& answer, const ipm::PcgReport& report) {
        Recorded r;
        r.a = *correction.matrix();
        r.theta = correction.theta();
        r.shift = correction.row_shift();
        if (r.shift.empty()) r.shift.assign(rhs.size(), 0.0);
        r.delta = correction.delta();
        r.rhs = rhs;
        r.solution = answer;
        r.converged = report.converged;
        recorded.push_back(std::move(r));
      };
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("ipm_dense_columns", true);
  options.set_double("ipm_dense_column_factor", 1.0);
  *solution = solve(model, options);
  ipm::testing::dense_solve_observer = nullptr;
  return recorded;
}

TEST(DenseColumnDirections, AgreeWithTheDefaultPathOnTheIteratesOfARealSolve) {
  int solves = 0;
  int held_to_1e10 = 0;
  int unconverged = 0;
  double worst_ratio = 0.0;  // the difference over the bound it is held to
  for (const char* name : {"adlittle", "blend", "share2b", "scagr7", "israel", "bandm",
                           "brandy", "agg", "capri", "e226", "fit1d"}) {
    Solution solution;
    const std::vector<Recorded> recorded = record_run(name, &solution);
    EXPECT_EQ(solution.status, SolveStatus::kOptimal) << name << ": " << solution.message;
    ASSERT_FALSE(recorded.empty()) << name << " has no dense column at factor 1";
    int model_compared = 0;
    int model_held = 0;
    double model_worst = 0.0;
    for (std::size_t s = 0; s < recorded.size(); ++s) {
      const Recorded& r = recorded[s];
      // An unconverged solve is never used as a direction (the interior point raises its
      // regularization and solves again), so it has no direction to compare.
      if (!r.converged) {
        ++unconverged;
        continue;
      }
      const WholeSystem whole(r);
      ASSERT_TRUE(whole.ok()) << name << " solve " << s;
      const std::vector<double> reference = whole.solve(r.rhs);
      double diff = 0.0;
      for (std::size_t i = 0; i < reference.size(); ++i) {
        diff = std::max(diff, std::fabs(r.solution[i] - reference[i]));
      }
      const double size = inf_norm(reference);
      if (size == 0.0) continue;
      const double relative = diff / size;
      const double eta =
          whole.backward_error(r.solution, r.rhs) + whole.backward_error(reference, r.rhs);
      const double bound = std::max(1e-10, 10.0 * whole.condition() * eta);
      EXPECT_LE(relative, bound) << name << " solve " << s << ": eta " << eta << " |x| "
                                 << inf_norm(r.solution) << " |ref| " << size << " |b| "
                                 << inf_norm(r.rhs) << " kappa " << whole.condition() << " |M| "
                                 << whole.norm() << " delta " << r.delta;
      ++solves;
      ++model_compared;
      if (relative <= 1e-10) {
        ++held_to_1e10;
        ++model_held;
      }
      model_worst = std::max(model_worst, relative);
      worst_ratio = std::max(worst_ratio, relative / bound);
    }
    std::cout << "[ dense directions ] " << name << ": " << model_compared << " of "
              << recorded.size() << " solves converged and compared, " << model_held
              << " within 1e-10, worst relative difference " << model_worst << "\n";
  }
  std::cout << "[ dense directions ] " << solves << " converged solves compared, "
            << held_to_1e10 << " within 1e-10 relative, " << unconverged
            << " unconverged (not used as directions), worst difference / bound " << worst_ratio
            << "\n";
  EXPECT_GT(solves, 0);
  // Most systems of a run are conditioned well enough for 1e-10 (491 of 508 when this was
  // written); a fall below nine in ten says the dense-column solve lost accuracy that the
  // conditioning does not explain, even if each solve stays inside its own bound.
  EXPECT_GE(10 * held_to_1e10, 9 * solves);
}

}  // namespace
}  // namespace sankhya
