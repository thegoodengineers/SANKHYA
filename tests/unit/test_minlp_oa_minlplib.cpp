// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by outer approximation on published instances (#528).
//
// The same 17 MINLPLib instances as test_minlp_minlplib.cpp (data/nlp/minlplib, published
// primal bounds in REFERENCE.csv), solved with `minlp_method=oa`. Three properties, none of
// which depends on how fast the machine is:
//
//   * NO ANSWER IS BETTER THAN THE PUBLISHED OPTIMUM. A point or a bound past it is a wrong
//     answer - a cut that removed a feasible point, or a point outside the model.
//   * EVERY `optimal` IS AT THE PUBLISHED OPTIMUM, within the MIP gap target.
//   * THE INSTANCES OUTER APPROXIMATION SOLVES IN A FRACTION OF A SECOND ARE PROVED.
//
// Which of the others it also proves inside the limit is a measurement, not a property: it is
// in bench/results (docs/BENCHMARKS.md section 2e), not asserted here. A slower or sanitized
// build would turn an assertion on it into noise.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

/// Proved at the published value in well under a second each, measured by
/// bench/runners/nlp_bench.py with `--solver-option minlp_method=oa`. The rest of the set -
/// clay0203m, jit1, ball_mk2_10, cvxnonsep_pcon20 - are measured, not asserted, for the reason
/// in the header.
const std::set<std::string> kMustProve = {
    "batchdes", "ex1223", "ex1223a", "ex1223b", "fac1",     "flay02m",  "gbd",
    "m3",       "nvs03",  "st_e14",  "syn05m",  "synthes2", "synthes3",
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

TEST(MinlpOaLib, ConvexInstancesReachTheirPublishedOptimaAndNothingBetter) {
  const std::vector<Reference> refs = references();
  ASSERT_EQ(refs.size(), 17u);
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("minlp_method", "oa");
  options.set_double("time_limit", 10.0);
  std::set<std::string> proved;
  for (const Reference& ref : refs) {
    std::unique_ptr<NonlinearModel> model;
    const io::ReadResult read = read_nl((kData / (ref.problem + ".nl")).string(), &model);
    ASSERT_TRUE(read.ok) << ref.problem << ": " << read.error;
    const Solution s = solve_nlp(*model, options);
    const double scale = std::max(1.0, std::fabs(ref.objective));
    const double better =
        ref.maximize ? s.objective - ref.objective : ref.objective - s.objective;
    std::printf("%-18s %-16s %20.10g %20.10g rounds-nodes %lld\n", ref.problem.c_str(),
                to_string(s.status), s.objective, ref.objective,
                static_cast<long long>(s.nodes));
    EXPECT_EQ(s.algorithm, "minlp-oa") << ref.problem;
    if (claims_a_point(s)) {
      EXPECT_LE(better, 1e-4 * scale)
          << ref.problem << " claims a point better than the published optimum";
    }
    if (s.status == SolveStatus::kOptimal) {
      EXPECT_NEAR(s.objective, ref.objective, 1e-4 * scale)
          << ref.problem << " is called optimal away from the published optimum";
      // The bound it proves cannot be past the published optimum either.
      if (ref.maximize) {
        EXPECT_GE(s.dual_bound, ref.objective - 1e-5 * scale) << ref.problem;
      } else {
        EXPECT_LE(s.dual_bound, ref.objective + 1e-5 * scale) << ref.problem;
      }
      proved.insert(ref.problem);
    }
  }
  std::printf("outer approximation proved optimal at the published value: %zu of %zu\n",
              proved.size(), refs.size());
  for (const std::string& name : kMustProve) {
    EXPECT_EQ(proved.count(name), 1u) << name << " is not proved by outer approximation";
  }
}

}  // namespace
}  // namespace sankhya::nlp
