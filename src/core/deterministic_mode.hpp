// SPDX-License-Identifier: Apache-2.0
// SANKHYA - deterministic mode (#288), shared by src/core/solve.cpp and every SolverEngine
// wrapper (src/solver_engine/builtin_engines.cpp) so a caller reaching an engine through
// either path gets the identical reproducibility guarantee (#297 review, deterministic
// execution context).
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// The options a deterministic solve actually runs with.
///
/// WHAT DETERMINISTIC MODE IS. Every decision the solver makes is a function of the model and
/// the options, except the ones that ask the clock: how much time is left decides when a
/// solve stops, how long the polish may run, and how the time limit is split between a
/// first-order pass and its finish. Those answers differ between two runs on one machine and
/// between two machines, so a run that ends on any of them is not reproducible. This turns
/// each of them into its deterministic counterpart, once, before any engine sees the options.
///
/// WHAT IT IS NOT. It does not make floating-point arithmetic associative, and it makes no
/// claim about a different compiler, a different CPU or a different build of this project.
/// The guarantee is: same build, same machine, same model, same options, same numbers.
///
/// A time limit is REFUSED rather than quietly ignored. A caller who asked for both
/// determinism and a deadline has asked for two things that cannot both hold, and the one
/// thing worse than picking for them is picking silently.
///
/// Returns `requested` unchanged when `options.deterministic` is false, so a caller can call
/// this unconditionally - which is what solve() and every SolverEngine wrapper now do.
[[nodiscard]] Options apply_deterministic_mode(const Options& requested, Logger& logger);

}  // namespace sankhya
