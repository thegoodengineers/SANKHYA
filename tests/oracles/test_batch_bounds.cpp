// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound with batched PDHG bounds (#520), against the exact MILP oracle.
//
//  * With gpu_batch_nodes and gpu_batch_strong_branching on, the search reaches the exact
//    optimum on the MILP fuzz instances, and the batch demonstrably ran and pruned.
//  * EVERY prune a batched bound made is re-checked in exact arithmetic: the LP optimum of
//    the pruned box is at least the bound, and the MILP optimum of the box is at least the
//    cutoff less the pruning margin - so nothing better than the incumbent was discarded.
//  * With both options off (the default) the search is the old one, bit for bit.
//  * The same with the CUDA backend, skipped without a device.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "mip/batch_audit.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "pdhg/batch_pdhg.hpp"
#include "support/temp_file.hpp"

namespace sankhya::oracle {
namespace {

Rational rational_at_least(double v) {
  constexpr int kMinExponent = -60;
  if (v == 0.0) return Rational(0);
  if (std::fabs(v) < std::ldexp(1.0, kMinExponent)) {
    return v < 0.0 ? Rational(0) : Rational(1, static_cast<Rational::Int>(1) << 60);
  }
  int exponent = 0;
  const double mantissa = std::frexp(v, &exponent);
  const auto scaled = static_cast<std::int64_t>(std::ldexp(mantissa, 53));
  const int shift = exponent - 53;
  if (shift >= 0) {
    return Rational(static_cast<Rational::Int>(scaled) *
                    (static_cast<Rational::Int>(1) << shift));
  }
  return Rational(static_cast<Rational::Int>(scaled), static_cast<Rational::Int>(1)
                                                          << (-shift));
}

Options quiet() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

Options batched(const char* backend) {
  Options o = quiet();
  o.set_bool("gpu_batch_nodes", true);
  o.set_bool("gpu_batch_strong_branching", true);
  o.set_string("gpu_batch_backend", backend);
  o.set_int("gpu_batch_size", 4);  // small, so the K-full trigger fires on small trees too
  o.set_int("gpu_batch_iterations", 400);
  return o;
}

/// A fuzz instance as test_branch_and_bound.cpp's MILP gate draws them: every column
/// integer and bounded.
struct Instance {
  GeneratedLp lp;
  Model model;
};

Instance draw(std::mt19937_64& rng, bool wide) {
  GeneratorConfig config;
  config.min_rows = 2;
  config.max_rows = wide ? 6 : 5;
  config.min_cols = wide ? 8 : 2;
  config.max_cols = wide ? 12 : 6;
  config.magnitude = 4;
  if (wide) config.density = 0.4;
  Instance in;
  in.lp = random_lp(rng, config);
  in.lp.integral.assign(static_cast<std::size_t>(in.lp.num_cols), 1);
  for (std::int64_t& u : in.lp.upper) {
    if (u == kNoUpperBound) u = 6;
  }
  in.model = to_model(in.lp);
  for (Index j = 0; j < in.model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    in.model.col_type[u] = VarType::kInteger;
    in.model.col_upper[u] = static_cast<double>(in.lp.upper[u]);
  }
  return in;
}

/// The batch counters of one solve (profile_out), summed into `into`.
Solution solve_counted(const Model& model, Options options,
                       std::map<std::string, std::int64_t>* into) {
  testing::TempFile file("", ".json");
  options.set_string("profile", "basic");
  options.set_string("profile_out", file.path());
  const Solution s = solve(model, options);
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  if (!text.str().empty()) {
    const nlohmann::json counters = nlohmann::json::parse(text.str())["counters"];
    for (const char* name :
         {"batch calls", "batch device calls", "batch LPs", "batch bounds raised",
          "batch nodes pruned", "batch children closed"}) {
      if (counters.contains(name)) (*into)[name] += counters[name].get<std::int64_t>();
    }
  }
  return s;
}

/// Sweep `trials` instances through the search with `options`; every answer must match
/// the oracle's. Returns the summed batch counters.
std::map<std::string, std::int64_t> reaches_exact_optimum(const Options& options, int trials,
                                                          std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::map<std::string, std::int64_t> counters;
  int optimal = 0;
  int infeasible = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const Instance in = draw(rng, trial % 2 == 1);
    const OracleResult exact = solve_exact_milp(in.lp, 20000);
    if (exact.status != OracleStatus::kOptimal && exact.status != OracleStatus::kInfeasible) {
      continue;
    }
    const Solution s = solve_counted(in.model, options, &counters);
    if (exact.status == OracleStatus::kInfeasible) {
      EXPECT_EQ(s.status, SolveStatus::kInfeasible) << in.lp.to_text();
      ++infeasible;
      continue;
    }
    const double expected = exact.objective.to_double();
    const bool stopped_on_gap = s.message.find("gap target") != std::string::npos;
    if (s.status != SolveStatus::kOptimal) {
      ADD_FAILURE() << to_string(s.status) << "\n" << in.lp.to_text();
      continue;
    }
    const double scale = std::max(1.0, std::fabs(expected));
    if (stopped_on_gap) {
      // Optimal within the gap target: never better than the optimum, never worse than the
      // target allows (the default relative gap 1e-4).
      EXPECT_GE(s.objective, expected - 1e-6 * scale) << in.lp.to_text();
      EXPECT_LE(s.objective, expected + 1e-4 * scale + 1e-6) << in.lp.to_text();
    } else {
      EXPECT_NEAR(s.objective, expected, 1e-6 * scale) << in.lp.to_text();
    }
    ++optimal;
  }
  std::printf(
      "[  INFO    ] %d at the exact optimum, %d agreed infeasible; batch: %lld call(s) (%lld "
      "on "
      "the device), %lld LP(s), %lld bound(s) raised, %lld node(s) pruned, %lld "
      "strong-branching child(ren) closed\n",
      optimal, infeasible, static_cast<long long>(counters["batch calls"]),
      static_cast<long long>(counters["batch device calls"]),
      static_cast<long long>(counters["batch LPs"]),
      static_cast<long long>(counters["batch bounds raised"]),
      static_cast<long long>(counters["batch nodes pruned"]),
      static_cast<long long>(counters["batch children closed"]));
  EXPECT_GT(optimal, trials / 3);
  return counters;
}

TEST(BatchBound, SearchReachesTheExactOptimumOnTheCpu) {
  const auto counters = reaches_exact_optimum(batched("cpu"), 240, 20260925);
  EXPECT_GT(counters.at("batch calls"), 0);
  EXPECT_GT(counters.at("batch nodes pruned") + counters.at("batch children closed"), 0)
      << "the batch never pruned, so this sweep proves nothing about its prunes";
}

bool device_available() {
  Model model;
  model.resize_columns(1);
  model.resize_rows(1);
  model.col_cost = {1.0};
  model.col_upper = {1.0};
  model.row_lower = {0.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();
  pdhg::BatchProblem problem;
  problem.model = &model;
  problem.cost = {1.0};
  problem.count = 1;
  problem.col_lower = {0.0};
  problem.col_upper = {1.0};
  pdhg::BatchSettings settings;
  settings.backend = pdhg::BatchBackendKind::kDevice;
  return pdhg::solve_batch(problem, settings).on_device;
}

TEST(BatchBound, SearchReachesTheExactOptimumOnTheDevice) {
  if (!device_available()) GTEST_SKIP() << "no CUDA device (or no CUDA in this build)";
  const auto counters = reaches_exact_optimum(batched("auto"), 120, 20260926);
  EXPECT_GT(counters.at("batch device calls"), 0);
  EXPECT_EQ(counters.at("batch device calls"), counters.at("batch calls"));
}

TEST(BatchBound, EveryBatchedPruneIsRecheckedByTheOracle) {
  // Presolve, cuts and symmetry rows off so the search's columns and rows are the model's:
  // a pruned box is then a box of THIS model, which the oracle can solve as it stands.
  Options options = batched("cpu");
  options.set_bool("presolve", false);
  options.set_bool("enable_root_cuts", false);
  options.set_bool("mip_symmetry", false);
  std::vector<mip::BatchPruneRecord> records;
  mip::batch_prune_audit_for_testing() = [&](const mip::BatchPruneRecord& r) {
    records.push_back(r);
  };
  std::mt19937_64 rng(5200520);
  int checked_nodes = 0;
  int checked_children = 0;
  int lp_infeasible = 0;
  int skipped = 0;
  for (int trial = 0; trial < 240; ++trial) {
    const Instance in = draw(rng, trial % 2 == 1);
    records.clear();
    (void)solve(in.model, options);
    for (const mip::BatchPruneRecord& r : records) {
      GeneratedLp box = in.lp;
      box.lower.assign(static_cast<std::size_t>(box.num_cols), 0);
      bool integral = true;
      for (std::size_t j = 0; j < r.col_lower.size(); ++j) {
        const double lo = r.col_lower[j];
        const double hi = r.col_upper[j];
        integral = integral && lo >= 0.0 && std::floor(lo) == lo && std::floor(hi) == hi;
        box.lower[j] = static_cast<std::int64_t>(lo);
        box.upper[j] = static_cast<std::int64_t>(hi);
      }
      if (!integral) {
        ++skipped;
        continue;
      }
      // The bound is a bound: the exact LP optimum over the box is at least it.
      const OracleResult lp = solve_exact(box);
      if (lp.status == OracleStatus::kOptimal) {
        EXPECT_LE(rational_at_least(r.bound), lp.objective)
            << "batched bound " << r.bound << " above the box's exact LP optimum "
            << lp.objective.to_double() << "\n"
            << box.to_text();
      } else {
        EXPECT_NE(lp.status, OracleStatus::kUnbounded) << box.to_text();
        ++lp_infeasible;
      }
      // The prune lost nothing: no integer point of the box beats the cutoff by more than
      // the margin the search allows itself.
      const OracleResult milp = solve_exact_milp(box, 20000);
      if (milp.status == OracleStatus::kOptimal) {
        const double slack = 1e-9 * std::max(1.0, std::fabs(r.cutoff));
        EXPECT_GE(milp.objective.to_double(), r.cutoff - r.margin - slack)
            << "the pruned box holds an integer point of objective "
            << milp.objective.to_double() << " against the cutoff " << r.cutoff << "\n"
            << box.to_text();
      } else {
        EXPECT_TRUE(milp.status == OracleStatus::kInfeasible) << to_string(milp.status);
      }
      ++(r.strong_branching ? checked_children : checked_nodes);
    }
  }
  mip::batch_prune_audit_for_testing() = nullptr;
  std::printf(
      "[  INFO    ] re-checked %d pruned node(s) and %d closed strong-branching child(ren) "
      "exactly (%d of the boxes LP-infeasible); %d skipped as non-integral boxes\n",
      checked_nodes, checked_children, lp_infeasible, skipped);
  EXPECT_GT(checked_nodes, 0);
  EXPECT_GT(checked_children, 0);
  EXPECT_EQ(skipped, 0);
}

bool same_solution(const Solution& a, const Solution& b) {
  const auto bits = [](const std::vector<double>& x, const std::vector<double>& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0);
  };
  return a.status == b.status && bits({a.objective}, {b.objective}) &&
         bits({a.dual_bound}, {b.dual_bound}) && bits(a.col_value, b.col_value) &&
         a.nodes == b.nodes && a.iterations == b.iterations;
}

TEST(BatchBound, OffByDefaultIsTheOldPathBitForBit) {
  const Options defaults = quiet();
  EXPECT_FALSE(defaults.get_bool("gpu_batch_nodes"));
  EXPECT_FALSE(defaults.get_bool("gpu_batch_strong_branching"));
  // Off, with every other batch knob moved: none of them may be read.
  Options off = quiet();
  off.set_bool("gpu_batch_nodes", false);
  off.set_bool("gpu_batch_strong_branching", false);
  off.set_int("gpu_batch_size", 3);
  off.set_int("gpu_batch_iterations", 17);
  off.set_string("gpu_batch_backend", "cpu");
  std::mt19937_64 rng(520520);
  int compared = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Instance in = draw(rng, trial % 2 == 1);
    const Solution a = solve(in.model, defaults);
    const Solution b = solve(in.model, off);
    EXPECT_TRUE(same_solution(a, b)) << in.lp.to_text();
    if (a.nodes > 1) ++compared;
  }
  EXPECT_GT(compared, 5) << "too few instances branched for the comparison to mean much";
}

}  // namespace
}  // namespace sankhya::oracle
