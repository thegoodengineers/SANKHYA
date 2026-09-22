// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_engine.hpp"

#include <fmt/format.h>

namespace sankhya::engine {

ProblemClass classify(const Model& model) {
  const bool integral = model.has_integrality();
  const bool quadratic = model.has_quadratic_objective();
  if (integral && quadratic) return ProblemClass::kMiqp;
  if (integral) return ProblemClass::kMilp;
  if (quadratic) return ProblemClass::kQp;
  return ProblemClass::kLp;
}

const char* to_string(ProblemClass problem_class) {
  switch (problem_class) {
    case ProblemClass::kLp: return "LP";
    case ProblemClass::kMilp: return "MILP";
    case ProblemClass::kQp: return "QP";
    case ProblemClass::kMiqp: return "MIQP";
  }
  return "unknown";
}

bool EngineCapabilities::accepts(ProblemClass problem_class) const {
  switch (problem_class) {
    case ProblemClass::kLp: return lp;
    case ProblemClass::kMilp: return milp;
    case ProblemClass::kQp: return qp;
    case ProblemClass::kMiqp: return miqp;
  }
  return false;
}

bool SolverEngine::supports(const Model& model) const {
  return capabilities().accepts(classify(model));
}

Solution SolverEngine::solve(const Model& model, const Options& options, Logger& logger,
                             SolveControl* control) const {
  if (!supports(model)) return unsupported_class_result(name(), model);
  return solve_verified(model, options, logger, control);
}

Solution unsupported_class_result(const std::string& engine_name, const Model& model) {
  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message =
      fmt::format("{} does not support {} models", engine_name, to_string(classify(model)));
  return solution;
}

}  // namespace sankhya::engine
