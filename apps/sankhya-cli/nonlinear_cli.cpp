// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the CLI's handling of nonlinear (.nl) models (NLP stage 1).

#include "nonlinear_cli.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "nlp/nl_reader.hpp"
#include "nlp/nlp_problem.hpp"
#include "nlp/nonlinear_model.hpp"

namespace sankhya::cli {
namespace {

std::unique_ptr<nlp::NonlinearModel> read_or_report(const std::string& path) {
  std::unique_ptr<nlp::NonlinearModel> model;
  std::vector<std::string> notes;
  const io::ReadResult result = nlp::read_nl(path, &model, &notes);
  if (!result.ok) {
    fmt::print(stderr, "error: {}\n", result.error);
    return nullptr;
  }
  for (const std::string& note : notes) fmt::print("note: {}\n", note);
  return model;
}

}  // namespace

int nonlinear_info(const std::string& path) {
  const std::unique_ptr<nlp::NonlinearModel> model = read_or_report(path);
  if (!model) return 3;
  const Model& base = model->base;
  Index nonlinear_rows = 0;
  for (const nlp::NonlinearConstraint& c : model->constraints) {
    const int degree = model->graph.degree(c.expression);
    if (degree < 0 || degree > 1) ++nonlinear_rows;
  }
  nlp::NlpProblem problem;
  const std::string built = nlp::NlpProblem::build(*model, &problem);
  fmt::print("{:<26}{}\n", "file", path);
  fmt::print("{:<26}{}\n", "class", nlp::to_string(model->classify()));
  fmt::print("{:<26}{}\n", "sense",
             base.sense == ObjSense::kMaximize ? "maximize" : "minimize");
  fmt::print("{:<26}{} ({} integer)\n", "columns", base.num_cols(), base.num_integer_columns());
  fmt::print("{:<26}{} ({} nonlinear)\n", "constraints", model->constraints.size(),
             nonlinear_rows);
  fmt::print("{:<26}{}\n", "nonlinear objective",
             model->objective == nlp::kNoExpr ? "no" : "yes");
  fmt::print("{:<26}{}\n", "expression nodes", model->graph.size());
  fmt::print("{:<26}{}\n", "starting point", model->start.empty() ? "none" : "from the file");
  if (built.empty()) {
    fmt::print("{:<26}{}\n", "Jacobian nonzeros", problem.jacobian_columns().size());
    fmt::print("{:<26}{} (lower triangle)\n", "Lagrangian Hessian nonz.",
               problem.hessian_rows().size());
  } else {
    fmt::print("{:<26}{}\n", "not an NLP", built);
  }
  const nlp::ConvexityReport convexity = model->convexity();
  fmt::print("{:<26}{}\n", "relaxation convex",
             convexity.convex ? "proved, by the composition rules" : "not proved");
  const std::size_t shown = std::min<std::size_t>(convexity.reasons.size(), 5);
  for (std::size_t k = 0; k < shown; ++k) fmt::print("  - {}\n", convexity.reasons[k]);
  if (convexity.reasons.size() > shown) {
    fmt::print("  ... and {} more\n", convexity.reasons.size() - shown);
  }
  const std::vector<std::string> risks = model->domain_risks();
  fmt::print("{:<26}{}\n", "domain risks", risks.size());
  for (std::size_t k = 0; k < std::min<std::size_t>(risks.size(), 5); ++k) {
    fmt::print("  - {}\n", risks[k]);
  }
  return 0;
}

int nonlinear_solve(const std::string& path, const Options& options) {
  (void)options;
  const std::unique_ptr<nlp::NonlinearModel> model = read_or_report(path);
  if (!model) return 3;
  const std::string problem = model->validate();
  if (!problem.empty()) {
    fmt::print(stderr, "error: {}\n", problem);
    return 5;
  }
  // Stage 1 reads and differentiates nonlinear models; the engine that solves a general one
  // is not in this build, and solving the linear part alone would be a different problem.
  fmt::print("\n{:<22}{}\n", "status", to_string(SolveStatus::kNotSolved));
  fmt::print(
      "{:<22}model class {}: this build represents and differentiates it but has no "
      "engine for it\n",
      "message", nlp::to_string(model->classify()));
  return 5;
}

}  // namespace sankhya::cli
