// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP on published instances (NLP stage 3).
//
// data/nlp/minlplib: 17 MINLPLib instances (https://www.minlplib.org) that MINLPLib marks
// convex, that have integer columns, closed gaps (primal bound = dual bound) and only the
// operators the .nl reader takes, and whose relaxation THIS solver's composition rules prove
// convex - the condition under which the branch and bound runs at all. The published primal
// bound and the sha256 of each downloaded file are in data/nlp/minlplib/REFERENCE.csv.
//
// Judged as a MIP answer is: `optimal` must sit within the gap target (1e-4 relative) of the
// published optimum, and NO answer's objective may be better than the published optimum
// (that would be a wrong answer - a point outside the model, or a bound that is not one).
// Instances that do not reach a proved optimum are listed with what the run showed, and the
// list is checked in both directions.

#include <algorithm>
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
    "minlplib";

/// Measured with bench/runners/nlp_bench.py at a 60 s limit: the answer is honest and
/// verified, and not a proved optimum, because the NLP engine could not solve one node's
/// relaxation (an iteration limit) - the branch and bound then keeps its incumbent as
/// `feasible` rather than call it optimal, or reports that it found none.
const std::map<std::string, std::string> kKnownMisses = {
    {"clay0203m",
     "at this test's 20 s limit no incumbent yet (time_limit, no point); at 60 s "
     "feasible at the published value 41573.26, a node relaxation having hit the "
     "NLP iteration limit, so no proof"},
    {"m3",
     "feasible at the published value 37.8; node 25's relaxation hit the NLP "
     "iteration limit, so no proof"},
    {"jit1",
     "numerical_error: the root relaxation cycles through restoration for 3000 "
     "iterations (objective of order 1e5)"},
};

struct Reference {
  std::string problem;
  double objective = 0.0;
  bool maximize = false;
};

std::vector<Reference> references() {
  std::vector<Reference> out;
  std::ifstream in(kData / "REFERENCE.csv");
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream fields(line);
    Reference r;
    std::string value, dual, sense;
    std::getline(fields, r.problem, ',');
    std::getline(fields, value, ',');
    std::getline(fields, dual, ',');
    std::getline(fields, sense, ',');
    r.objective = std::stod(value);
    r.maximize = sense == "max";
    out.push_back(r);
  }
  return out;
}

TEST(MinlpLib, ConvexInstancesReachTheirPublishedOptimaAndNothingBetter) {
  const std::vector<Reference> refs = references();
  ASSERT_EQ(refs.size(), 17u);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("time_limit", 20.0);
  std::set<std::string> misses;
  int proved = 0;
  for (const Reference& ref : refs) {
    std::unique_ptr<NonlinearModel> model;
    const io::ReadResult read = read_nl((kData / (ref.problem + ".nl")).string(), &model);
    ASSERT_TRUE(read.ok) << ref.problem << ": " << read.error;
    const Solution s = solve_nlp(*model, options);
    const double scale = std::max(1.0, std::fabs(ref.objective));
    const double better =
        ref.maximize ? s.objective - ref.objective : ref.objective - s.objective;
    std::printf("%-18s %-16s %20.10g %20.10g nodes %lld\n", ref.problem.c_str(),
                to_string(s.status), s.objective, ref.objective,
                static_cast<long long>(s.nodes));
    if (claims_a_point(s)) {
      EXPECT_LE(better, 1e-4 * scale)
          << ref.problem << " claims a point better than the optimum";
    }
    if (s.status == SolveStatus::kOptimal &&
        std::fabs(s.objective - ref.objective) <= 1e-4 * scale) {
      ++proved;
    } else {
      misses.insert(ref.problem);
    }
  }
  std::printf("proved optimal at the published value: %d of %zu\n", proved, refs.size());
  for (const std::string& name : misses) {
    EXPECT_TRUE(kKnownMisses.count(name) == 1) << name << " is not proved and not listed";
  }
  for (const auto& [name, why] : kKnownMisses) {
    EXPECT_TRUE(misses.count(name) == 1) << name << " is listed but now proved: " << why;
  }
}

}  // namespace
}  // namespace sankhya::nlp
