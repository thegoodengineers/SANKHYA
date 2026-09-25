// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the NLP interior point on the Hock-Schittkowski problems (NLP stage 2).
//
// THE PROBLEMS are data/nlp/hs/*.nl, written by Pyomo's .nl writer from Vanderbei's published
// AMPL transcriptions of W. Hock and K. Schittkowski, "Test Examples for Nonlinear Programming
// Codes", LNEMS 187, Springer (1981) (bench/runners/hs_mod_to_nl.py; the source URL and the
// sha256 of every downloaded model are in data/nlp/hs/REFERENCE.csv, with the published
// optimal objective each model file states). They are read by the solver's own .nl reader,
// so this is also a test of that reader against a third party's writer.
//
// WHAT COUNTS AS RIGHT. A local method may legitimately stop at a different local optimum
// than the published one. So each answer is judged in three ways:
//   * it is never `optimal` (a GLOBAL claim) with an objective worse than the published one
//     - that would be a wrong answer, and fails the test outright;
//   * every `optimal` / `locally_optimal` answer passed the KKT check at the project
//     tolerances (it cannot be labelled so otherwise, nlp_solve.cpp);
//   * it MATCHES when its objective is within 1e-6 relative of the published value, the
//     precision the values are published to. Problems that do not match are listed in
//     kKnownMisses with the reason, and a problem that starts matching or stops matching
//     fails the test, so the list cannot drift from the truth in either direction.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/nl_reader.hpp"
#include "nlp/nlp_solve.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {
namespace {

const std::filesystem::path kData =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / "nlp" /
    "hs";

/// Problems whose answer does not match the published optimum, with why. Measured, not
/// guessed: each entry was added from a run of this test, and the reason is what the run
/// showed.
const std::map<std::string, std::string> kKnownMisses = {
    // Another local minimum, which a local method may legitimately reach from the published
    // start. Each is a KKT point that tools/verify_solution.py verified with its own reader and
    // derivatives (bench/runners/nlp_bench.py), and each is labelled locally_optimal.
    {"hs002",
     "x = (-1.22103, 1.5): the local minimum on the bound x2 >= 1.5 left of the "
     "origin, objective 4.94123; the published one is at x1 = 1.2244"},
    {"hs016",
     "x = (-0.5, 0.707107): a vertex where x1 >= -0.5 and x1 + x2^2 >= 0 are both "
     "active, objective 23.1447"},
    {"hs020",
     "x = (-0.5, 0.866025): a vertex where x1 >= -0.5 and x1^2 + x2^2 >= 1 are both "
     "active, objective 40.1987"},
    {"hs044", "x = (3, 0, 4, 0): a vertex of the linear constraints, objective -13"},
    {"hs108", "another local minimum, objective -0.674981 against -0.866025"},
    // Degenerate: at the optimum (1, 0) the gradient of (1 - x1)^3 - x2 >= 0 does not satisfy
    // any constraint qualification, so no KKT multipliers exist there. The method stops at
    // x1 = 1.00117, where the violation (1 - x1)^3 = -1.6e-9 is inside the tolerance and the
    // KKT test passes; the objective is 2.3e-3 below the optimum because the point is that
    // slightly infeasible.
    {"hs013",
     "degenerate, no multipliers at the optimum: stops 1.2e-3 from it, inside the "
     "feasibility tolerance"},
};

struct Reference {
  std::string problem;
  double objective = 0.0;
};

std::vector<Reference> references() {
  std::vector<Reference> out;
  std::ifstream in(kData / "REFERENCE.csv");
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    std::stringstream fields(line);
    Reference r;
    std::string value;
    std::getline(fields, r.problem, ',');
    std::getline(fields, value, ',');
    r.objective = std::stod(value);
    out.push_back(r);
  }
  return out;
}

TEST(NlpHockSchittkowski, PublishedOptimaAreReachedAndNothingFalseIsClaimed) {
  const std::vector<Reference> refs = references();
  ASSERT_EQ(refs.size(), 70u) << "data/nlp/hs/REFERENCE.csv";
  Options options;
  options.set_bool("log_to_console", false);
  int matched = 0, local = 0, global = 0;
  std::set<std::string> misses;
  std::printf("%-8s %-20s %18s %18s %10s %6s\n", "problem", "status", "objective", "published",
              "rel.gap", "iters");
  for (const Reference& ref : refs) {
    std::unique_ptr<NonlinearModel> model;
    const io::ReadResult read = read_nl((kData / (ref.problem + ".nl")).string(), &model);
    ASSERT_TRUE(read.ok) << ref.problem << ": " << read.error;
    const Solution s = solve_nlp(*model, options);
    const bool has_point =
        s.status == SolveStatus::kOptimal || s.status == SolveStatus::kLocallyOptimal;
    const double gap =
        std::fabs(s.objective - ref.objective) / std::max(1.0, std::fabs(ref.objective));
    const bool match = has_point && gap <= 1e-6;
    std::printf("%-8s %-20s %18.10g %18.10g %10.2e %6lld\n", ref.problem.c_str(),
                to_string(s.status), s.objective, ref.objective, gap,
                static_cast<long long>(s.iterations));
    if (s.status == SolveStatus::kOptimal) {
      ++global;
      // A global claim may not be worse than the published optimum (minimisation).
      EXPECT_LE(s.objective, ref.objective + 1e-6 * std::max(1.0, std::fabs(ref.objective)))
          << ref.problem << " claims a global optimum worse than the published one";
    }
    if (s.status == SolveStatus::kLocallyOptimal) ++local;
    if (match) {
      ++matched;
    } else {
      misses.insert(ref.problem);
    }
  }
  std::printf("matched %d of %zu (optimal %d, locally_optimal %d)\n", matched, refs.size(),
              global, local);
  std::set<std::string> known;
  for (const auto& [name, why] : kKnownMisses) known.insert(name);
  for (const std::string& name : misses) {
    EXPECT_TRUE(known.count(name) == 1) << name << " does not match and is not listed";
  }
  for (const std::string& name : known) {
    EXPECT_TRUE(misses.count(name) == 1) << name << " is listed as a miss but now matches";
  }
}

}  // namespace
}  // namespace sankhya::nlp
