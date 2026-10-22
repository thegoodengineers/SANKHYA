// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual_stall_reperturb (src/simplex/dual_simplex.cpp) against the exact oracle: the
// re-perturbation moves the costs the dual loop works on, so the oracle is what says every
// move was undone before the answer left. The degenerate generator is where stalls live.

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

TEST(FuzzDualReperturb, DegenerateAndRandomInstancesAgainstTheOracle) {
  std::mt19937_64 rng(1006500605);
  GeneratorConfig config;
  int optimal = 0, other = 0, skipped = 0, mismatched = 0;
  std::vector<std::string> failures;
  for (int trial = 0; trial < 2000; ++trial) {
    const GeneratedLp lp = trial < 1000 ? degenerate_lp(rng, config) : random_lp(rng, config);
    const OracleResult exact = solve_exact(lp);
    if (exact.status == OracleStatus::kOverflow ||
        exact.status == OracleStatus::kIterationLimit) {
      ++skipped;
      continue;
    }
    Options options;
    options.set_bool("log_to_console", false);
    options.set_string("algorithm", "dual-simplex");
    options.set_bool("dual_stall_reperturb", true);
    const Solution s = solve(to_model(lp), options);
    bool agree = false;
    switch (exact.status) {
      case OracleStatus::kOptimal: {
        const double expected = exact.objective.to_double();
        agree = s.status == SolveStatus::kOptimal &&
                std::fabs(s.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected));
        optimal += agree ? 1 : 0;
        break;
      }
      case OracleStatus::kInfeasible:
        agree = s.status == SolveStatus::kInfeasible;
        other += agree ? 1 : 0;
        break;
      case OracleStatus::kUnbounded:
        agree = s.status == SolveStatus::kUnbounded;
        other += agree ? 1 : 0;
        break;
      case OracleStatus::kOverflow:
      case OracleStatus::kIterationLimit: break;
    }
    if (agree) continue;
    ++mismatched;
    if (failures.size() < 5) {
      failures.push_back(std::string("oracle ") + to_string(exact.status) + " solver " +
                         to_string(s.status) + " objective " + std::to_string(s.objective) +
                         "\n" + lp.to_text());
    }
  }
  std::cout << "\n=== 2000 instances (degenerate + random) under dual_stall_reperturb ===\n"
            << "  agreed optimal " << optimal << "\n  agreed infeasible/unbounded " << other
            << "\n  oracle abstained " << skipped << "\n  MISMATCHED " << mismatched << "\n";
  for (const std::string& f : failures)
    std::cout << "\n--- failing instance ---\n" << f << "\n";
  EXPECT_EQ(mismatched, 0);
  EXPECT_GT(optimal + other, 1600);
}

}  // namespace
}  // namespace sankhya::oracle
