// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CPU/CUDA PDHG numerical agreement tests (issue #283).
//
// Solves the nine committed Netlib instances (data/netlib/reference.json) with both the
// CPU PDHG engine (algorithm=pdhg, gpu=false) and the CUDA PDHG engine (algorithm=pdhg,
// gpu=true) and holds the two objectives to the error the stopping rule GUARANTEES for
// the two points actually returned - derived below, per instance and per run - rather than
// to a fixed relative number.
//
// WHY NOT A FIXED 1e-8. This file used to demand |obj_cpu - obj_cuda| <= 1e-8 max(1, |obj|),
// on the premise that two answers which each stopped at 1e-8 agree to 1e-8. The stopping
// rule does not say that. Its gap terms are tight enough (kOptimal needs |p - d| <=
// kDualityGap max(1, |p|) = 1e-9 relative), but it also admits an ABSOLUTE primal residual
// up to kPrimalFeasibility = 1e-7 and an absolute dual residual up to kDualFeasibility =
// 1e-7, and a residual moves the objective by the residual times the size of the optimal
// multipliers or of the optimal point. On blend (|obj| = 30.8) that is worth more than 1e-8
// relative: two correct runs differed by 1.25e-8 on a card (#711), both kOptimal, while
// the gap term alone allows 1e-9. The bound below is what the definitions imply.
//
// THE BOUND. Minimise-space LP min c'x, rl <= A x <= ru, l <= x <= u, optimum p* at (x*, y*)
// with reduced costs d* = c + A'y* (tests take x*, y*, d* from the simplex). For a returned
// point (x, y) with objective p = c'x:
//   below:  c'x - p* = sum_j d*_j (x_j - x*_j) - sum_i y*_i ((A x)_i - (A x*)_i)
//                    >= -||y*||_2 P - ||d*||_1 V,
//           P = ||row-bound violation of A x||_2 (pdhg::evaluate's absolute primal residual),
//           V = max column-bound violation of x (zero before postsolve; measured here);
//   above:  p* >= L(y) >= d(y) - D M, where d(y) is pdhg::evaluate's dual objective (the
//           Lagrangian over the absorbable terms), D its absolute dual residual (the terms it
//           cannot absorb, 2-norm) and M = sqrt(||x*||^2 + ||A x*||^2) - so
//           c'x - p* <= (p - d(y)) + D M.
// Hence |p - p*| <= e(x, y) = max(||y*|| P + ||d*||_1 V, max(0, p - d(y)) + D M) for EACH
// run, and |p_cpu - p_cuda| <= e_cpu + e_cuda, with every residual measured by
// pdhg::evaluate on the ORIGINAL model at the point solve() returned (after postsolve; the
// solver's own guarantees are in the presolved space, so they are re-measured here).
// References: Bertsimas & Tsitsiklis, "Introduction to Linear Optimization", 1997, section
// 4.3 (weak duality and the Lagrangian bound); Applegate et al., PDLP, NeurIPS 2021,
// section 3.3 (the residuals). The simplex optimum enters only through the norms ||y*||,
// ||d*||_1 and M; p* itself cancels.
//
// Nothing is loosened: a run whose objective is further from the other than both runs'
// own residuals permit is a genuine disagreement and fails. The derived bound and the
// relative difference are both printed per instance.
//
// When the CUDA backend is not compiled in (gpu=true falls back to pdhg-cpu), the test
// calls GTEST_SKIP so that "skipped" is distinguishable from "passed" in CI output.
// Depends on #16 and #17.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pdhg/pdhg_evaluate.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

std::string repository_path(const char* relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

// The norms of a reference optimum (x*, y*, d*) that the bound needs (file header).
struct ReferenceNorms {
  double y_norm = 0.0;    ///< ||y*||_2
  double d_l1 = 0.0;      ///< ||d*||_1
  double x_extent = 0.0;  ///< M = sqrt(||x*||^2 + ||A x*||^2)
  bool ok = false;
};

ReferenceNorms reference_norms(const Model& model) {
  Options simplex;
  simplex.set_bool("log_to_console", false);
  const Solution s = solve(model, simplex);
  ReferenceNorms r;
  r.ok = s.status == SolveStatus::kOptimal;
  if (!r.ok) return r;
  std::vector<double> ax(static_cast<std::size_t>(model.num_rows()), 0.0);
  model.matrix.multiply(s.col_value.data(), ax.data());
  double y2 = 0.0, x2 = 0.0;
  for (const double v : s.row_dual) y2 += v * v;
  for (const double v : s.col_dual) r.d_l1 += std::fabs(v);
  for (const double v : s.col_value) x2 += v * v;
  for (const double v : ax) x2 += v * v;
  r.y_norm = std::sqrt(y2);
  r.x_extent = std::sqrt(x2);
  return r;
}

// e(x, y) of the file header for one returned solution: how far its objective can be from
// the optimum given its own residuals, measured on the original model.
double objective_error_bound(const Model& model, const Solution& run,
                             const ReferenceNorms& ref) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const double sense = model.sense_multiplier();
  pdhg::Problem prob;
  prob.model = &model;
  prob.cost.resize(n);
  for (std::size_t j = 0; j < n; ++j) prob.cost[j] = sense * model.col_cost[j];
  // bound_norm and cost_norm only scale the RELATIVE residuals, which the bound does not use.
  std::vector<double> y(m), activity, reduced;
  for (std::size_t i = 0; i < m; ++i) y[i] = -sense * run.row_dual[i];  // row_dual = -sense y
  const pdhg::Residuals r = pdhg::evaluate(prob, run.col_value, y, activity, reduced);
  double column_violation = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const double x = run.col_value[j];
    if (is_finite_bound(model.col_lower[j]))
      column_violation = std::max(column_violation, model.col_lower[j] - x);
    if (is_finite_bound(model.col_upper[j]))
      column_violation = std::max(column_violation, x - model.col_upper[j]);
  }
  const double below = ref.y_norm * r.absolute_primal + ref.d_l1 * column_violation;
  const double above =
      std::max(0.0, r.primal_objective - r.dual_objective) + r.absolute_dual * ref.x_extent;
  return std::max(below, above);
}

// The objectives agree when they are within the sum of the two runs' bounds; `allowed`
// returns that sum so the caller can print it.
bool objectives_agree(const Model& model, const Solution& cpu, const Solution& gpu,
                      const ReferenceNorms& ref, double* allowed) {
  *allowed = objective_error_bound(model, cpu, ref) + objective_error_bound(model, gpu, ref);
  return std::fabs(cpu.objective - gpu.objective) <= *allowed;
}

Options pdhg_regression_options(bool gpu) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("algorithm", "pdhg");
  o.set_bool("gpu", gpu);
  // The polish (#229) hands the PDHG point to the interior-point engine and tests a
  // different code path.  Disabled here so the comparison is purely between the two
  // first-order engines.
  o.set_bool("pdhg_polish", false);
  // 1e-8: the tighter of the two project-standard tolerances (ENGINEERING_RULES.md). The
  // two runs do not take the same iterates (#451, #456), so their objectives are held to
  // the bound both runs' residuals imply (file header), not to each other's bits.
  o.set_double("pdhg_tolerance", 1e-8);
  // Netlib small instances need up to ~200k iterations at 1e-4 (adlittle, test_pdhg.cpp).
  // At 1e-8 the count is higher; 1e6 is the budget here.
  o.set_int("iteration_limit", 1000000);
  return o;
}

// Returns true when the CUDA backend actually ran rather than falling back to pdhg-cpu.
bool cuda_was_used(const Solution& s) {
  return s.algorithm.find("cuda") != std::string::npos ||
         s.algorithm.find("gpu") != std::string::npos;
}

// The nine instances committed in data/netlib/reference.json, ordered small to large.
const std::vector<const char*> kNetlibInstances = {
    "afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1", "israel",
};

// =========================================================================================

TEST(PdhgCudaRegression, NineNetlibInstancesAgreeAtTheStoppingTolerance) {
  // --- CUDA availability probe ---
  // Solve the smallest committed instance (afiro) with gpu=true.  If the engine falls back
  // to the CPU path the algorithm field is "pdhg-cpu" and the test skips rather than fails,
  // so "skipped" and "passed" are distinguishable in CI.
  const std::string probe_path = repository_path("data/netlib/afiro.mps");
  Model probe_model;
  const io::ReadResult probe_read = io::read_model(probe_path, &probe_model);
  ASSERT_TRUE(probe_read.ok) << probe_path << ": " << probe_read.error
                             << " (instance is committed; a failure here means the file moved)";
  // One iteration is enough to learn which engine ran; a full solve at 1e-8 here would
  // cost every CPU-only CI run a solve it then throws away.
  Options probe_options = pdhg_regression_options(/*gpu=*/true);
  probe_options.set_int("iteration_limit", 1);
  const Solution probe = solve(probe_model, probe_options);
  if (!cuda_was_used(probe)) {
    GTEST_SKIP() << "CUDA backend not in this build (gpu=true ran as \"" << probe.algorithm
                 << "\"): skipped, not passed.  Depends on #16 and #17.";
  }

  // --- Nine Netlib instances ---
  int agreed = 0;
  int failed = 0;

  for (const char* name : kNetlibInstances) {
    const std::string path =
        repository_path((std::string("data/netlib/") + name + ".mps").c_str());
    Model model;
    const io::ReadResult r = io::read_model(path, &model);
    ASSERT_TRUE(r.ok) << path << ": " << r.error;

    const Solution cpu = solve(model, pdhg_regression_options(/*gpu=*/false));
    const Solution gpu = solve(model, pdhg_regression_options(/*gpu=*/true));

    const bool cpu_converged = cpu.status == SolveStatus::kOptimal;
    // On a real card the GPU may come back `feasible` where the CPU says `optimal`: the
    // status guard measured its worst complementarity product a hair above 1e-6 (adlittle
    // on an L4: 1.347e-6, #456), which the nondeterministic reductions explain (#451).  It is
    // accepted here ONLY through the same objective agreement an optimal answer must pass
    // below; it is not a free pass, and the log names it.
    const bool gpu_converged =
        gpu.status == SolveStatus::kOptimal || gpu.status == SolveStatus::kFeasible;
    if (gpu.status == SolveStatus::kFeasible) {
      std::cout << name << ": CUDA reports feasible, not optimal (" << gpu.message
                << "); held to the objective agreement below\n";
    }

    // Both engines stopped short (share2b hits the 1e6-iteration budget on both at 1e-8):
    // that is agreement in failure, asserted as such - the GPU must reproduce the CPU's
    // status, not vanish from the count.
    if (!cpu_converged && !gpu_converged) {
      EXPECT_EQ(cpu.status, gpu.status)
          << name << ": CPU " << to_string(cpu.status) << " (" << cpu.message << ") but CUDA "
          << to_string(gpu.status) << " (" << gpu.message << ")";
      std::cout << name << ": neither engine converged at 1e-8 inside the budget (CPU "
                << to_string(cpu.status) << ", CUDA " << to_string(gpu.status)
                << "); agreement in failure\n";
      if (cpu.status == gpu.status)
        ++agreed;
      else
        ++failed;
      continue;
    }

    EXPECT_TRUE(cpu_converged) << name << " CPU: " << cpu.message;
    EXPECT_TRUE(gpu_converged) << name << " CUDA: " << gpu.message;
    if (!cpu_converged || !gpu_converged) {
      ++failed;
      continue;
    }

    const ReferenceNorms ref = reference_norms(model);
    ASSERT_TRUE(ref.ok) << name << ": the simplex reference did not solve";
    const double scale = std::max({1.0, std::fabs(cpu.objective), std::fabs(gpu.objective)});
    const double diff = std::fabs(cpu.objective - gpu.objective);
    double allowed = 0.0;
    const bool ok = objectives_agree(model, cpu, gpu, ref, &allowed);

    std::cout << name << ": cpu=" << cpu.objective << "  cuda=" << gpu.objective
              << "  rel=" << diff / scale << "  derived bound rel=" << allowed / scale
              << (ok ? "  PASS" : "  FAIL") << "\n";

    EXPECT_TRUE(ok) << name << ": objective mismatch\n"
                    << "  CPU  objective : " << cpu.objective << "\n"
                    << "  CUDA objective : " << gpu.objective << "\n"
                    << "  relative diff  : " << diff / scale << "\n"
                    << "  allowed (derived from both runs' residuals): " << allowed / scale;
    if (ok) {
      ++agreed;
    } else {
      ++failed;
    }
  }

  std::cout << "CPU/CUDA agreement: " << agreed << "/"
            << static_cast<int>(kNetlibInstances.size())
            << " instances passed (per-instance derived bound)\n";
  EXPECT_EQ(failed, 0);
}

TEST(PdhgCudaRegression, TheTwoMatvecPathOnTheDeviceAgreesWithTheCpuAtTheStoppingTolerance) {
  // #479 on the device: with pdhg_two_matvec the CUDA loop takes one A-product per iteration
  // and derives A xbar and A dx from the cached A x_k. The derived vectors differ from the
  // computed ones by rounding, so it is held to the same bound as the three-product path
  // above: the CPU answer at the stopping tolerance, on the instances both converge on.
  const std::string probe_path = repository_path("data/netlib/afiro.mps");
  Model probe_model;
  ASSERT_TRUE(io::read_model(probe_path, &probe_model).ok) << probe_path;
  Options probe_options = pdhg_regression_options(/*gpu=*/true);
  probe_options.set_int("iteration_limit", 1);
  if (!cuda_was_used(solve(probe_model, probe_options))) {
    GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
  }
  int agreed = 0;
  int compared = 0;
  for (const char* name : kNetlibInstances) {
    Model model;
    ASSERT_TRUE(
        io::read_model(repository_path((std::string("data/netlib/") + name + ".mps").c_str()),
                       &model)
            .ok)
        << name;
    const Solution cpu = solve(model, pdhg_regression_options(/*gpu=*/false));
    Options two = pdhg_regression_options(/*gpu=*/true);
    two.set_string("pdhg_two_matvec", "true");
    const Solution gpu = solve(model, two);
    ASSERT_TRUE(cuda_was_used(gpu)) << name << ": " << gpu.message;
    const bool cpu_converged = cpu.status == SolveStatus::kOptimal;
    const bool gpu_converged =
        gpu.status == SolveStatus::kOptimal || gpu.status == SolveStatus::kFeasible;
    if (!cpu_converged && !gpu_converged) continue;  // share2b: neither, as above
    ++compared;
    EXPECT_TRUE(gpu_converged) << name << " two-mat-vec on CUDA: " << gpu.message;
    const ReferenceNorms ref = reference_norms(model);
    ASSERT_TRUE(ref.ok) << name << ": the simplex reference did not solve";
    double allowed = 0.0;
    const bool ok = objectives_agree(model, cpu, gpu, ref, &allowed);
    EXPECT_TRUE(ok) << name << ": CPU " << cpu.objective << " (" << cpu.iterations
                    << " iterations), CUDA two-mat-vec " << gpu.objective << " ("
                    << gpu.iterations << "), allowed " << allowed;
    if (gpu_converged && ok) ++agreed;
  }
  EXPECT_GE(compared, 8);
  EXPECT_EQ(agreed, compared);
}

TEST(PdhgCudaRegression, TheDeviceLoopAgreesWithTheCpuAtTheStoppingTolerance) {
  // #478: with gpu_on_device_loop one iteration is captured as a CUDA graph and replayed
  // kPdhgDeviceLoopBlock times per host synchronisation, the adaptive step rule running on
  // the device. Same bound as the per-iteration path, alone and with #479's two-mat-vec, and
  // the log must say the loop ran: a silent fallback to the per-iteration path would pass the
  // agreement check without testing anything.
  const std::string probe_path = repository_path("data/netlib/afiro.mps");
  Model probe_model;
  ASSERT_TRUE(io::read_model(probe_path, &probe_model).ok) << probe_path;
  Options probe_options = pdhg_regression_options(/*gpu=*/true);
  probe_options.set_int("iteration_limit", 1);
  if (!cuda_was_used(solve(probe_model, probe_options))) {
    GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
  }
  for (const bool two_matvec : {false, true}) {
    int agreed = 0;
    int compared = 0;
    for (const char* name : kNetlibInstances) {
      Model model;
      ASSERT_TRUE(
          io::read_model(repository_path((std::string("data/netlib/") + name + ".mps").c_str()),
                         &model)
              .ok)
          << name;
      const Solution cpu = solve(model, pdhg_regression_options(/*gpu=*/false));
      Options loop = pdhg_regression_options(/*gpu=*/true);
      loop.set_bool("gpu_on_device_loop", true);
      loop.set_string("pdhg_two_matvec", two_matvec ? "true" : "false");
      loop.set_bool("log_to_console", true);
      ::testing::internal::CaptureStdout();
      const Solution gpu = solve(model, loop);
      std::fflush(stdout);
      const std::string log = ::testing::internal::GetCapturedStdout();
      ASSERT_TRUE(cuda_was_used(gpu)) << name << ": " << gpu.message;
      EXPECT_NE(log.find("device loop on"), std::string::npos)
          << name << ": the device loop did not run\n"
          << log.substr(0, 2000);
      const bool cpu_converged = cpu.status == SolveStatus::kOptimal;
      const bool gpu_converged =
          gpu.status == SolveStatus::kOptimal || gpu.status == SolveStatus::kFeasible;
      if (!cpu_converged && !gpu_converged) continue;
      ++compared;
      EXPECT_TRUE(gpu_converged) << name << " device loop: " << gpu.message;
      const ReferenceNorms ref = reference_norms(model);
      ASSERT_TRUE(ref.ok) << name << ": the simplex reference did not solve";
      double allowed = 0.0;
      const bool ok = objectives_agree(model, cpu, gpu, ref, &allowed);
      EXPECT_TRUE(ok) << name << (two_matvec ? " (two-mat-vec)" : "") << ": CPU "
                      << cpu.objective << " (" << cpu.iterations
                      << " iterations), CUDA device loop " << gpu.objective << " ("
                      << gpu.iterations << "), allowed " << allowed;
      if (gpu_converged && ok) ++agreed;
    }
    EXPECT_GE(compared, 8) << (two_matvec ? "two-mat-vec" : "three products");
    EXPECT_EQ(agreed, compared) << (two_matvec ? "two-mat-vec" : "three products");
  }
}

TEST(PdhgCudaRegression, AConvergedCudaRunCarriesItsKktCrossings) {
  // #486 on the device: the CUDA loop records the first crossings of the relative KKT error
  // as the CPU engine does. A run that converged to 1e-8 relative crossed all three levels,
  // in order, and its CSV row must not read "never reached".
  Model model;
  ASSERT_TRUE(io::read_model(repository_path("data/netlib/afiro.mps"), &model).ok);
  const Solution gpu = solve(model, pdhg_regression_options(/*gpu=*/true));
  if (!cuda_was_used(gpu))
    GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
  ASSERT_EQ(gpu.status, SolveStatus::kOptimal) << gpu.message;
  EXPECT_GT(gpu.kkt_1e4_iterations, 0);
  EXPECT_GE(gpu.kkt_1e6_iterations, gpu.kkt_1e4_iterations);
  EXPECT_GE(gpu.kkt_1e8_iterations, gpu.kkt_1e6_iterations);
  EXPECT_LE(gpu.kkt_1e8_iterations, gpu.iterations);
  EXPECT_TRUE(std::isfinite(gpu.kkt_1e4_seconds));
  EXPECT_LE(gpu.kkt_1e4_seconds, gpu.kkt_1e8_seconds);
}

}  // namespace
}  // namespace sankhya
