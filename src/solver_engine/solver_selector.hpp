// SPDX-License-Identifier: Apache-2.0
// SANKHYA - SolverSelector (#297): chooses a SolverEngine from a registry for a model, for a
// caller that wants to go through the registry rather than solve()'s own dispatch.
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "solver_engine/solver_registry.hpp"

namespace sankhya::engine {

/// What the selector decided, and why.
struct EngineChoice {
  const SolverEngine* engine = nullptr;  ///< nullptr when nothing in the registry qualifies
  std::string rule;
  std::string reason;
};

/// Chooses a SolverEngine from `registry` for `model` under `options`.
///
/// For an LP this defers entirely to select_engine() (src/core/engine_selection.hpp), the
/// measured rule table solve()'s own dispatcher already uses for `algorithm=auto` (#284) -
/// so a caller going through the registry gets the same answer solve() would give, not a
/// second opinion re-derived here. GPU routing (`use_gpu` on that rule table's answer) stays
/// solve()'s concern: the algorithm name it returns for a GPU-eligible model is still
/// "pdhg", exactly as it is for the CPU path, and this selector resolves it to the
/// registry's "pdhg" engine either way.
///
/// For MILP, QP and MIQP there is one built-in engine each today, and the rule says so
/// ("only-candidate"). An explicit `algorithm` naming a registered engine is honoured as
/// given (rule "requested"); a name that is not registered, or that names an engine whose
/// capabilities() reject the model's class, comes back with `engine == nullptr` and `reason`
/// says which.
[[nodiscard]] EngineChoice select(const SolverRegistry& registry, const Model& model,
                                  const Options& options, bool warm_start = false);

}  // namespace sankhya::engine
