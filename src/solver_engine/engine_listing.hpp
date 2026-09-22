// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the engine listing (#297): every registered engine, what it solves and what its
// answer carries, for people (`sankhya engines`) and for scripts (`--format json`).
//
// EVERY FLAG HERE IS READ FROM THE ENGINE, not typed into a table beside it: the rows are
// SolverEngine::capabilities() and the two description strings each engine carries, so the
// listing cannot say something the registry does not. Which names the `algorithm` option
// accepts comes from the same predicate solve() uses (SolverRegistry::algorithm_names), and
// tests/unit/test_solver_engine.cpp holds the option's own choices list to it.
#pragma once

#include <string>

#include "solver_engine/solver_registry.hpp"

namespace sankhya::engine {

/// One row per registered engine, in registration order, followed by a note naming any
/// engine the source tree has that this binary was built without.
[[nodiscard]] std::string format_engines_text(const SolverRegistry& registry);

/// The same as JSON: `{"engines": [...], "not_in_this_build": [...]}`, one object per engine
/// with its name, the classes it accepts, whether `algorithm` selects it, its capabilities,
/// and where it lives.
[[nodiscard]] std::string format_engines_json(const SolverRegistry& registry);

}  // namespace sankhya::engine
