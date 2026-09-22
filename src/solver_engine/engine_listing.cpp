// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/engine_listing.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace sankhya::engine {
namespace {

/// The engines the source tree has that a build can leave out. Named here rather than
/// discovered, because an engine that is not compiled in is not in the registry to be asked.
std::vector<std::string> not_in_this_build() {
  std::vector<std::string> missing;
#ifndef SANKHYA_ENABLE_CUDA
  missing.emplace_back("pdhg-gpu");
#endif
  return missing;
}

std::string classes_of(const EngineCapabilities& caps) {
  std::string out;
  for (const ProblemClass problem_class :
       {ProblemClass::kLp, ProblemClass::kMilp, ProblemClass::kQp, ProblemClass::kMiqp}) {
    if (!caps.accepts(problem_class)) continue;
    if (!out.empty()) out += ",";
    out += to_string(problem_class);
  }
  return out;
}

/// How a caller reaches the engine: by name through `algorithm`, through `algorithm=pdhg`
/// with the GPU flag, or by the problem class alone.
std::string reached_by(const SolverRegistry& registry, const SolverEngine& engine) {
  const std::vector<std::string> names = registry.algorithm_names();
  if (std::find(names.begin(), names.end(), engine.name()) != names.end()) {
    return "algorithm=" + engine.name();
  }
  if (engine.capabilities().supports_gpu) return "algorithm=pdhg with gpu=true, or auto";
  return "the problem class";
}

const char* yes(bool value) {
  return value ? "yes" : "-";
}

}  // namespace

std::string format_engines_text(const SolverRegistry& registry) {
  constexpr const char* kRow = "{:<17} {:<12} {:<38} {:<5} {:<6} {:<5} {:<5} {:<7} {:<8}\n";
  std::string out = fmt::format(kRow, "engine", "classes", "reached by", "warm", "basis",
                                "duals", "certs", "interrupt", "determ.");
  for (const std::string& name : registry.names()) {
    const SolverEngine* engine = registry.find(name);
    if (engine == nullptr) continue;
    const EngineCapabilities caps = engine->capabilities();
    out += fmt::format(kRow, name, classes_of(caps), reached_by(registry, *engine),
                       yes(caps.supports_warm_start), yes(caps.supports_basis),
                       yes(caps.supports_duals), yes(caps.supports_certificates),
                       yes(caps.supports_interrupt), yes(caps.supports_deterministic_mode));
  }
  out += "\n";
  for (const std::string& name : registry.names()) {
    const SolverEngine* engine = registry.find(name);
    if (engine == nullptr || engine->summary().empty()) continue;
    out += fmt::format("{:<17} {}\n", name, engine->summary());
    if (!engine->source().empty()) out += fmt::format("{:<17} {}\n", "", engine->source());
  }
  for (const std::string& name : not_in_this_build()) {
    out += fmt::format(
        "\n{} is in the source tree and not in this build (SANKHYA_ENABLE_CUDA is off).\n",
        name);
  }
  return out;
}

std::string format_engines_json(const SolverRegistry& registry) {
  nlohmann::json doc;
  doc["engines"] = nlohmann::json::array();
  for (const std::string& name : registry.names()) {
    const SolverEngine* engine = registry.find(name);
    if (engine == nullptr) continue;
    const EngineCapabilities caps = engine->capabilities();
    const std::vector<std::string> names = registry.algorithm_names();
    nlohmann::json row;
    row["name"] = name;
    row["reached_by"] = reached_by(registry, *engine);
    row["selectable_by_algorithm"] = std::find(names.begin(), names.end(), name) != names.end();
    row["classes"] = {
        {"LP", caps.lp}, {"MILP", caps.milp}, {"QP", caps.qp}, {"MIQP", caps.miqp}};
    row["capabilities"] = {{"gpu", caps.supports_gpu},
                           {"warm_start", caps.supports_warm_start},
                           {"duals", caps.supports_duals},
                           {"basis", caps.supports_basis},
                           {"certificates", caps.supports_certificates},
                           {"interrupt", caps.supports_interrupt},
                           {"deterministic_mode", caps.supports_deterministic_mode}};
    row["summary"] = engine->summary();
    row["source"] = engine->source();
    doc["engines"].push_back(row);
  }
  doc["not_in_this_build"] = not_in_this_build();
  return doc.dump(2) + "\n";
}

}  // namespace sankhya::engine
