// SPDX-License-Identifier: Apache-2.0
// SANKHYA - two-card multi-GPU PDHG (#295): the same answer as one card, the same bits run to
// run, and the same bits through the host-staged fallback as through peer-to-peer.
//
// Every test here needs two CUDA devices and calls GTEST_SKIP with fewer, so "skipped" and
// "passed" stay distinguishable in CI, whose runners have none.
//
// What "the same answer" means. The two-card run sums A^T y over the cards, so its
// floating-point additions are ordered differently from a one-card run and the trajectories
// part after the first rounding difference; two runs that each stopped at the tolerance tol
// can be held to agree to tol in the objective and no tighter (the argument of
// test_pdhg_cuda_regression.cpp, #456). That is the bound, against both one-card engines:
// the partitioned engine on one card (the like-for-like baseline, gpu_partitioned) and the
// production single-GPU engine. Where one run converges and the other does not inside the
// budget that is a failure; where neither does, the statuses must agree.
//
// What "the same bits" means. Every reduction on the multi-GPU path has a fixed order
// (src/gpu/pdhg_multi_gpu_device.hpp), so two runs, the peer and the host-staged transport,
// and the same two-block partition on one card or on two, must agree exactly: iteration
// count, objective, and every primal and dual value, compared with ==, not a tolerance.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/multi_device.hpp"
#include "gpu/pdhg_gpu.hpp"
#include "gpu/pdhg_multi_gpu.hpp"
#endif

namespace sankhya {
namespace {

#ifdef SANKHYA_ENABLE_CUDA

std::string repository_path(const std::string& relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

// The stopping tolerance of every solve here, and the objective agreement demanded of two
// solves that each met it (file header).
constexpr double kTolerance = 1e-6;

Options solve_options() {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_double("pdhg_tolerance", kTolerance);
  o.set_bool("pdhg_polish", false);
  o.set_int("iteration_limit", 400000);
  return o;
}

bool converged(const Solution& s) {
  return s.status == SolveStatus::kOptimal || s.status == SolveStatus::kFeasible;
}

// The skip every test starts with.
#define REQUIRE_TWO_CARDS()                                                    \
  do {                                                                         \
    if (gpu::device_count() < 2)                                               \
      GTEST_SKIP() << "needs two CUDA devices, found " << gpu::device_count(); \
  } while (0)

Model read_netlib(const std::string& name) {
  Model model;
  const std::string path = repository_path("data/netlib/" + name + ".mps");
  const io::ReadResult r = io::read_model(path, &model);
  EXPECT_TRUE(r.ok) << path << ": " << r.error;
  return model;
}

// A sparse LP with a known optimum, built backwards from a chosen KKT pair: the construction
// of bench/runners/generate_large_lp.py ("random" structure) and tests/oracles kkt_lp, plus
// a few long linking rows so that splitting by rows and splitting by nonzeros differ.
//   min c^T x  s.t.  A x >= b,  0 <= x <= u;  x*, y* >= 0 chosen, d = c - A^T y* >= 0 with
//   d_j = 0 wherever x*_j > 0, b tight on rows with y*_i > 0 and slack elsewhere.
struct SyntheticLp {
  Model model;
  double optimum = 0.0;
};

SyntheticLp synthetic_kkt_lp(int rows, int cols, int nnz_per_col, int linking_rows,
                             int linking_length, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto draw = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  const auto ur = static_cast<std::size_t>(rows);
  const auto uc = static_cast<std::size_t>(cols);
  std::vector<double> x(uc), y(ur);
  for (auto& v : x) v = draw(0, 2) == 0 ? 0.0 : draw(1, 9);
  for (auto& v : y) v = draw(0, 1) == 0 ? 0.0 : draw(1, 9);
  // Linking rows are the first rows: the dense block a row-count split puts on card 0.
  std::vector<std::vector<std::pair<int, double>>> by_col(uc);
  for (int j = 0; j < cols; ++j) {
    std::vector<int> picked;
    while (static_cast<int>(picked.size()) < nnz_per_col) {
      const int i = draw(linking_rows, rows - 1);
      if (std::find(picked.begin(), picked.end(), i) == picked.end()) picked.push_back(i);
    }
    for (int i : picked) by_col[static_cast<std::size_t>(j)].push_back({i, draw(1, 9)});
  }
  for (int i = 0; i < linking_rows; ++i)
    for (int t = 0; t < linking_length; ++t)
      by_col[static_cast<std::size_t>(draw(0, cols - 1))].push_back({i, draw(1, 9)});

  SyntheticLp lp;
  Model& m = lp.model;
  m.matrix.reset(rows, cols);
  std::vector<double> activity(ur, 0.0);
  m.col_cost.assign(uc, 0.0);
  for (std::size_t j = 0; j < uc; ++j) {
    double aty = 0.0;
    for (const auto& [i, a] : by_col[j]) {
      m.matrix.add_entry(i, static_cast<Index>(j), a);
      aty += a * y[static_cast<std::size_t>(i)];
      activity[static_cast<std::size_t>(i)] += a * x[j];
    }
    const double d = x[j] > 0.0 ? 0.0 : draw(1, 9);
    m.col_cost[j] = d + aty;
  }
  m.matrix.finalize();
  m.col_lower.assign(uc, 0.0);
  m.col_upper.resize(uc);
  for (std::size_t j = 0; j < uc; ++j) m.col_upper[j] = x[j] + draw(0, 3);
  m.col_type.assign(uc, VarType::kContinuous);
  m.row_lower.resize(ur);
  m.row_upper.assign(ur, kInfinity);
  for (std::size_t i = 0; i < ur; ++i)
    m.row_lower[i] = y[i] > 0.0 ? activity[i] : activity[i] - draw(1, 9);
  for (std::size_t j = 0; j < uc; ++j) lp.optimum += m.col_cost[j] * x[j];
  return lp;
}

void expect_bitwise_equal(const Solution& a, const Solution& b, const std::string& what) {
  EXPECT_EQ(a.status, b.status) << what;
  EXPECT_EQ(a.iterations, b.iterations) << what;
  EXPECT_EQ(a.objective, b.objective) << what;  // exact, by design
  ASSERT_EQ(a.col_value.size(), b.col_value.size()) << what;
  ASSERT_EQ(a.row_dual.size(), b.row_dual.size()) << what;
  std::size_t differing = 0;
  for (std::size_t j = 0; j < a.col_value.size(); ++j) {
    if (a.col_value[j] != b.col_value[j]) ++differing;
    if (a.col_dual[j] != b.col_dual[j]) ++differing;
  }
  for (std::size_t i = 0; i < a.row_dual.size(); ++i)
    if (a.row_dual[i] != b.row_dual[i]) ++differing;
  EXPECT_EQ(differing, 0u) << what << ": primal or dual values differ";
}

// Holds `two` to `one` at the stopping tolerance; returns true when it agreed.
bool agrees(const Solution& one, const Solution& two, const std::string& what) {
  if (!converged(one) && !converged(two)) {
    EXPECT_EQ(one.status, two.status) << what << ": neither converged, statuses differ";
    std::cout << what << ": neither converged (" << to_string(one.status) << ", "
              << to_string(two.status) << ")\n";
    return one.status == two.status;
  }
  EXPECT_TRUE(converged(one)) << what << " one card: " << one.message;
  EXPECT_TRUE(converged(two)) << what << " two cards: " << two.message;
  if (!converged(one) || !converged(two)) return false;
  const double scale = std::max({1.0, std::fabs(one.objective), std::fabs(two.objective)});
  const double rel = std::fabs(one.objective - two.objective) / scale;
  std::cout << what << ": one card " << one.objective << " (" << one.iterations
            << " it), two cards " << two.objective << " (" << two.iterations
            << " it), relative difference " << rel << "\n";
  EXPECT_LE(rel, kTolerance) << what;
  return rel <= kTolerance;
}

const std::vector<std::string> kNetlibNine = {
    "afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1", "israel",
};

TEST(MultiGpuTwoCards, NineNetlibInstancesMatchOneCardAtTheStoppingTolerance) {
  REQUIRE_TWO_CARDS();
  Logger silent(nullptr);
  Options partitioned = solve_options();
  partitioned.set_bool("gpu_partitioned", true);
  int agreed = 0;
  for (const std::string& name : kNetlibNine) {
    const Model model = read_netlib(name);
    const Solution two = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 1}, silent);
    ASSERT_EQ(two.algorithm, "pdhg-cuda-multi") << name << ": the two-card path did not run";
    const Solution one_partitioned = gpu::solve_pdhg_multi_gpu(model, partitioned, {0}, silent);
    ASSERT_EQ(one_partitioned.algorithm, "pdhg-cuda-multi") << name;
    const Solution one_engine = gpu::solve_pdhg_gpu(model, solve_options(), silent);
    const bool a = agrees(one_partitioned, two, name + " (partitioned engine, one card)");
    const bool b = agrees(one_engine, two, name + " (single-GPU engine)");
    agreed += (a && b) ? 1 : 0;
  }
  std::cout << "two cards against one: " << agreed << "/" << kNetlibNine.size()
            << " instances agree at " << kTolerance << "\n";
  EXPECT_EQ(agreed, static_cast<int>(kNetlibNine.size()));
}

TEST(MultiGpuTwoCards, ALargerSyntheticLpMatchesOneCardAndItsKnownOptimum) {
  REQUIRE_TWO_CARDS();
  Logger silent(nullptr);
  // 20,000 x 20,000, five nonzeros a column plus eight linking rows of 2,000.
  const SyntheticLp lp = synthetic_kkt_lp(20000, 20000, 5, 8, 2000, 295);
  // The iteration BUDGET here is 1,000,000, not the file's 400,000; the tolerance is
  // unchanged. What binds on this model is the project standard's complementarity bound
  // (1e-6), which the run meets somewhere between roughly 270,000 and 420,000 iterations
  // depending on where its restarts fall. On two A100s (#478 item 3): single engine 266,920
  // (evaluation on the device) and 310,920 (on the host); two cards 415,240 (on the cards)
  // and 360,240 (on the host); the device and host evaluations agreed to six digits at
  // every checkpoint of the 415,240 run, complementarity included. A rounding-level
  // difference in a restart decision moves the count by 100,000, so 400,000 was a coin toss.
  Options budget = solve_options();
  budget.set_int("iteration_limit", 1000000);
  const Solution two = gpu::solve_pdhg_multi_gpu(lp.model, budget, {0, 1}, silent);
  ASSERT_EQ(two.algorithm, "pdhg-cuda-multi");
  Options partitioned = budget;
  partitioned.set_bool("gpu_partitioned", true);
  const Solution one_partitioned =
      gpu::solve_pdhg_multi_gpu(lp.model, partitioned, {0}, silent);
  const Solution one_engine = gpu::solve_pdhg_gpu(lp.model, budget, silent);
  agrees(one_partitioned, two, "synthetic 20000x20000 (partitioned engine, one card)");
  agrees(one_engine, two, "synthetic 20000x20000 (single-GPU engine)");
  ASSERT_TRUE(converged(two)) << two.message;
  const double scale = std::max(1.0, std::fabs(lp.optimum));
  EXPECT_LE(std::fabs(two.objective - lp.optimum) / scale, kTolerance)
      << "two cards " << two.objective << " against the constructed optimum " << lp.optimum;
}

TEST(MultiGpuTwoCards, TwoRunsAndOneOrTwoCardsGiveTheSameBits) {
  REQUIRE_TWO_CARDS();
  Logger silent(nullptr);
  const SyntheticLp lp = synthetic_kkt_lp(5000, 5000, 5, 4, 800, 7);
  for (const auto* which : {"sc105", "synthetic"}) {
    const Model model = std::string(which) == "sc105" ? read_netlib("sc105") : lp.model;
    const Solution first = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 1}, silent);
    const Solution second = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 1}, silent);
    ASSERT_EQ(first.algorithm, "pdhg-cuda-multi");
    expect_bitwise_equal(first, second, std::string(which) + ": run 1 against run 2");
    // The same two-block partition with both blocks on card 0.
    const Solution one_card = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 0}, silent);
    expect_bitwise_equal(first, one_card, std::string(which) + ": {0,1} against {0,0}");
  }
}

TEST(MultiGpuTwoCards, HostStagedFallbackGivesTheSameBitsAsPeerToPeer) {
  REQUIRE_TWO_CARDS();
  Logger silent(nullptr);
  const SyntheticLp lp = synthetic_kkt_lp(5000, 5000, 5, 4, 800, 11);
  Options staged = solve_options();
  staged.set_bool("gpu_peer_access", false);
  for (const auto* which : {"israel", "synthetic"}) {
    const Model model = std::string(which) == "israel" ? read_netlib("israel") : lp.model;
    const Solution host = gpu::solve_pdhg_multi_gpu(model, staged, {0, 1}, silent);
    ASSERT_EQ(host.algorithm, "pdhg-cuda-multi");
    EXPECT_NE(host.message.find("host-staged exchange"), std::string::npos) << host.message;
    const Solution peer = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 1}, silent);
    if (gpu::can_peer_access(0, 1) && gpu::can_peer_access(1, 0)) {
      EXPECT_NE(peer.message.find("peer-to-peer exchange"), std::string::npos) << peer.message;
    } else {
      std::cout << "devices 0 and 1 have no P2P: both runs were host-staged\n";
    }
    expect_bitwise_equal(host, peer, std::string(which) + ": host-staged against peer");
  }
}

TEST(MultiGpuTwoCards, EvaluationOnTheCardsMatchesTheHostEvaluation) {
  // #478 item 3: the KKT residuals, gap and restart distances reduced on the cards (the
  // default) against pdhg::evaluate on the host (gpu_device_evaluation=false, the
  // reference). The two agree to rounding (test_pdhg_device_evaluation.cpp), so a restart can
  // fall one evaluation apart and the runs are held to the stopping tolerance, not the bit;
  // each is bitwise repeatable on its own (TwoRunsAndOneOrTwoCardsGiveTheSameBits runs the
  // default, the cards).
  REQUIRE_TWO_CARDS();
  Logger silent(nullptr);
  Options host = solve_options();
  host.set_bool("gpu_device_evaluation", false);
  const SyntheticLp lp = synthetic_kkt_lp(5000, 5000, 5, 4, 800, 478);
  int agreed = 0, compared = 0;
  // Instances that meet the project standard inside the file's budget on both paths
  // (stocfor1 and israel reach the iteration limit on either, which would compare nothing).
  for (const std::string& name :
       {std::string("afiro"), std::string("sc50a"), std::string("adlittle"),
        std::string("blend"), std::string("sc105"), std::string("synthetic")}) {
    const Model model = name == "synthetic" ? lp.model : read_netlib(name);
    const Solution cards = gpu::solve_pdhg_multi_gpu(model, solve_options(), {0, 1}, silent);
    const Solution on_host = gpu::solve_pdhg_multi_gpu(model, host, {0, 1}, silent);
    ASSERT_EQ(cards.algorithm, "pdhg-cuda-multi") << name;
    ASSERT_EQ(on_host.algorithm, "pdhg-cuda-multi") << name;
    ++compared;
    agreed += agrees(on_host, cards, name + " (host evaluation against the cards)") ? 1 : 0;
  }
  EXPECT_EQ(agreed, compared);
}

#else  // no CUDA backend in this build: the same tests, visibly skipped

TEST(MultiGpuTwoCards, NineNetlibInstancesMatchOneCardAtTheStoppingTolerance) {
  GTEST_SKIP() << "CUDA backend not compiled in (SANKHYA_ENABLE_CUDA=OFF)";
}
TEST(MultiGpuTwoCards, ALargerSyntheticLpMatchesOneCardAndItsKnownOptimum) {
  GTEST_SKIP() << "CUDA backend not compiled in (SANKHYA_ENABLE_CUDA=OFF)";
}
TEST(MultiGpuTwoCards, TwoRunsAndOneOrTwoCardsGiveTheSameBits) {
  GTEST_SKIP() << "CUDA backend not compiled in (SANKHYA_ENABLE_CUDA=OFF)";
}
TEST(MultiGpuTwoCards, HostStagedFallbackGivesTheSameBitsAsPeerToPeer) {
  GTEST_SKIP() << "CUDA backend not compiled in (SANKHYA_ENABLE_CUDA=OFF)";
}
TEST(MultiGpuTwoCards, EvaluationOnTheCardsMatchesTheHostEvaluation) {
  GTEST_SKIP() << "CUDA backend not compiled in (SANKHYA_ENABLE_CUDA=OFF)";
}

#endif  // SANKHYA_ENABLE_CUDA

}  // namespace
}  // namespace sankhya
