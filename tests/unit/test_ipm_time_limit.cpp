// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point honours its time limit inside the linear algebra (#468).
//
// Measured before this file existed: Linf_520c (93,326 rows) with a 90 s limit returned
// after 153 s on one machine and 320 s on another, having assembled normal equations of 474
// million lower-triangle entries. The assembly asked the deadline every 256 rows, and one
// column that meets every row makes 256 rows 24 million multiply-adds; the assembled
// triplets were then sorted by SparseMatrix::finalize(), and the ordering built its graph
// from every entry, and neither asked at all. Both tests here use the same shape in
// miniature - a column in every row, as the L-infinity objective's `t` is - so the normal
// equations are dense.

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/cudss_factor.hpp"
#include "ipm/ipm_testing.hpp"
#include "la/ldl.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

/// min t  s.t.  x_i + t >= 1 + i/m,  0 <= x_i <= 1/2,  t free: m rows, one of them per x_i,
/// and the column of t in every one of them; the optimum is t = 0.5 + (m-1)/m. The value is
/// not what is tested, the shape is: A Theta A^T is dense, m(m+1)/2 lower-triangle entries.
Model l_infinity_shape(Index m) {
  Model model;
  const Index n = m + 1;
  const Index t = m;
  model.sense = ObjSense::kMinimize;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_cost[static_cast<std::size_t>(t)] = 1.0;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 0.5);
  model.col_lower[static_cast<std::size_t>(t)] = -kInfinity;
  model.col_upper[static_cast<std::size_t>(t)] = kInfinity;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.resize(static_cast<std::size_t>(m));
  model.row_upper.assign(static_cast<std::size_t>(m), kInfinity);
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    model.row_lower[static_cast<std::size_t>(i)] =
        1.0 + static_cast<double>(i) / static_cast<double>(m);
    model.matrix.add_entry(i, i, 1.0);
    model.matrix.add_entry(i, t, 1.0);
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(InteriorPointTimeLimit, TheAssemblyAsksTheDeadlineInProportionToItsWork) {
  // Deterministic, no clock: the predicate never fires and only counts how often it was
  // asked. The dense column makes every row of the product cost the whole column, m*m
  // multiply-adds in all; a check every 256 rows asked m/256 = 8 times for 4.2 million of
  // them, one look per half a million. The deadline must now be asked at least once per
  // 100,000 units of work (it is asked every 65,536).
  const Index m = 2048;
  const Model model = l_infinity_shape(m);
  std::vector<double> theta(static_cast<std::size_t>(m) + 1, 1.0);
  std::size_t asked = 0;
  SparseMatrix normal;
  ASSERT_TRUE(normal_equations_lower(model.matrix, theta, {}, 1e-8, &normal, [&asked] {
    ++asked;
    return false;
  }));
  const auto work = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  EXPECT_GE(asked, work / 100000) << "asked " << asked << " times for " << work << " work";

  // And asking changed nothing: the product is complete, dense, and equal entry for entry to
  // the one built with no deadline at all.
  SparseMatrix plain;
  ASSERT_TRUE(normal_equations_lower(model.matrix, theta, {}, 1e-8, &plain));
  EXPECT_EQ(normal.num_nonzeros(), static_cast<Index>(m) * (m + 1) / 2);
  ASSERT_EQ(plain.num_nonzeros(), normal.num_nonzeros());
  EXPECT_EQ(plain.row_indices(), normal.row_indices());
  EXPECT_EQ(plain.values(), normal.values());
  EXPECT_EQ(plain.column_starts(), normal.column_starts());
}

// #907: cuDSS does not consult the deadline once a call has started, so the device path can
// only ever decline to START a late factorize() or solve() - and real hardware (an A100 on
// refinery_year) showed the solve running 19 iterations and 9,700s past its limit, which is
// far more than one overrun factorization costs. This fakes a slow device entirely on the
// CPU (the real backend never builds or runs here) to prove, without cuDSS or a GPU, that the
// deadline stops the loop after the first factorization or solve that is already running
// late, not 19 of them.
// Call counts live on the test's stack, not on the device: InteriorPoint owns and destroys
// the device inside solve_ipm(), so anything the test wants to read afterwards must outlive it.
class FakeSlowDevice : public gpu::LinearSolverDevice {
 public:
  FakeSlowDevice(std::chrono::duration<double> per_call_delay, int* factorize_calls,
                 int* solve_calls)
      : per_call_delay_(per_call_delay),
        factorize_calls_(factorize_calls),
        solve_calls_(solve_calls) {}

  bool initialize(std::string* /*reason*/) override { return true; }

  bool analyze(const SparseMatrix& lower, std::string* /*reason*/) override {
    return ldl_.analyze(lower);
  }

  gpu::CudssOutcome factorize(const SparseMatrix& lower, double regularization,
                              std::string* /*reason*/) override {
    std::this_thread::sleep_for(per_call_delay_);
    ++timing_.factorizations;
    ++*factorize_calls_;
    if (!ldl_.factorize(lower, regularization)) return gpu::CudssOutcome::kFailed;
    return gpu::CudssOutcome::kFactored;
  }

  bool solve(double* b, std::string* /*reason*/) override {
    std::this_thread::sleep_for(per_call_delay_);
    ++timing_.solves;
    ++*solve_calls_;
    ldl_.solve(b);
    return true;
  }

  Count regularized_pivots() const noexcept override { return ldl_.regularized_pivots(); }
  std::int64_t factor_nonzeros() const noexcept override {
    return static_cast<std::int64_t>(ldl_.factor_nonzeros()) + ldl_.dimension();
  }
  const gpu::CudssTiming& timing() const noexcept override { return timing_; }

 private:
  std::chrono::duration<double> per_call_delay_;
  int* factorize_calls_;
  int* solve_calls_;
  SparseLdl ldl_;
  gpu::CudssTiming timing_;
};

TEST(InteriorPointTimeLimit, TheDeviceDeadlineStopsAfterOneLateCallNotNineteen) {
  // Each factorize() or solve() the fake device is asked for takes 60ms - a handful of them
  // already exceed the limit below, exactly as one cuDSS call on a huge system would. Before
  // the fix this fake reproduced the real failure on an A100 (refinery_year: 19 iterations,
  // 9,700s past the limit, #907) and WORSE: normal_solve()'s own mid-solve CPU fallback (the
  // device was factored but a solve of it was declined because the deadline had already
  // passed) could itself be too late to even analyse, and the NEXT solve of that same
  // factorization - the refinement loop calls normal_solve() more than once per
  // factorization - fell through to SparseLdl::solve() with no factors of the right
  // dimension, which segfaulted (stl_vector.h, out of bounds) rather than returning a status
  // at all. The fix is that once a factorization proves unusable every later solve of it
  // says so too (ipm.cpp, normal_factor_unusable_) instead of calling solve() blind.
  constexpr double kLimit = 0.03;
  constexpr double kPerCallDelay = 0.06;
  constexpr double kMargin = 2.0;  // generous: this asserts "not 19 iterations", not a bound
  int factorize_calls = 0;
  int solve_calls = 0;
  ipm::testing::fake_device_factory = [&] {
    return std::unique_ptr<gpu::LinearSolverDevice>(new FakeSlowDevice(
        std::chrono::duration<double>(kPerCallDelay), &factorize_calls, &solve_calls));
  };
  const Model model = l_infinity_shape(60);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("ipm_linear_solver", "cudss");
  options.set_double("time_limit", kLimit);
  options.set_double("ipm_setup_share", 1.0);
  Logger quiet(nullptr);
  const auto started = std::chrono::steady_clock::now();
  const Solution solved = ipm::solve_ipm(model, options, quiet);
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ipm::testing::fake_device_factory = nullptr;
  // Not a crash, not a hang, and not nineteen iterations run to a convergence the budget
  // never allowed: the solve stops - with time_limit, or with whatever status the loop's own
  // non-finite-direction recovery (#209) reports for the NaN a declined fallback hands
  // back - soon after the first factorization or solve that started late, never at optimal.
  EXPECT_NE(solved.status, SolveStatus::kOptimal) << solved.message;
  // A handful of calls - the one that started late, plus whatever the predictor and corrector
  // of the single iteration before it needed - never nineteen iterations' worth.
  EXPECT_LE(factorize_calls, 3) << solved.message;
  EXPECT_LE(wall, kLimit + kMargin) << solved.message;
}

TEST(InteriorPointTimeLimit, ADenseColumnDoesNotCarryTheSolvePastItsTimeLimit) {
  // The clock this time, on the whole engine: a limit of half a second on a model whose
  // normal equations hold 4.5 million entries and whose factor is dense. The engine must
  // answer time_limit - not a numerical error, not optimal - and within the limit plus a
  // margin that covers returning, freeing what was built and a loaded CI machine.
  constexpr double kLimit = 0.5;
  constexpr double kMargin = 0.75;
  const Model model = l_infinity_shape(3000);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("time_limit", kLimit);
  // The set-up share would decline at a fifth of the limit with not_solved (#357); this test
  // is about the limit itself, so the set-up is given all of it.
  options.set_double("ipm_setup_share", 1.0);
  Logger quiet(nullptr);
  const auto started = std::chrono::steady_clock::now();
  const Solution solved = ipm::solve_ipm(model, options, quiet);
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  EXPECT_EQ(solved.status, SolveStatus::kTimeLimit) << solved.message;
  EXPECT_LE(wall, kLimit + kMargin) << solved.message;
}

}  // namespace
}  // namespace sankhya
