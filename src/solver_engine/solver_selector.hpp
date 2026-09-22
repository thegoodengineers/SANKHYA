// SPDX-License-Identifier: Apache-2.0
// SANKHYA - SolverSelector (#297): chooses a SolverEngine from a registry for a model. This
// IS how src/core/solve.cpp decides which engine to run (full integration) - it is not a
// parallel opinion solve() ignores. A caller going through the registry directly
// (src/solver_engine/solver_engine_dispatch.hpp) gets the identical decision for the same
// reason: there is one selection algorithm, called from both places.
#pragma once

#include <string>

#include "sankhya/logging.hpp"
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
/// FOR AN LP, `algorithm=auto` defers entirely to select_engine() (#284,
/// src/core/engine_selection.hpp) - the same measured rule table - so both callers of this
/// function (solve.cpp's dispatcher and a caller going through the registry directly) get
/// the identical answer. When that rule table's answer is GPU-eligible (`use_gpu`) or the
/// caller set the `gpu` option, and this build has SANKHYA_ENABLE_CUDA, `logger` is used to
/// probe the device (only on `algorithm=auto`, to avoid the CUDA runtime init cost
/// otherwise) and, if the registry has a "pdhg-gpu" engine AND gpu::gpu_pdhg_is_safe()
/// agrees (the SAME shared guard #297's A1 fix introduced - this selector cannot bypass it
/// any more than solve.cpp can), that engine is chosen instead of the CPU "pdhg" one (#297,
/// hardware context / selector diagram: "Model + Engine + Hardware capabilities + User
/// configuration -> Compatible engine"). An EXPLICIT `algorithm=pdhg` with the `gpu` option
/// set is upgraded the same way, so the choice does not depend on how "pdhg" was reached.
/// Anything that declines falls back to CPU "pdhg".
///
/// FOR MILP, QP AND MIQP there is one built-in engine each today, and `algorithm=auto`
/// resolves to it with rule "only-candidate" - the SAME candidates() lookup
/// src/core/solve.cpp's own MILP/QP/MIQP branches now call directly for their (today always
/// true) "does a registered engine exist for this class" discovery check, so there is one
/// list of candidates, not two. An EXPLICIT `algorithm` IS checked against the registry and
/// that engine's capabilities() here, for all four classes uniformly - a deliberate choice
/// (#297 review, B2). solve.cpp's own MILP/QP/MIQP branches deliberately do NOT call this
/// function for that check (only candidates() - see above): they still never read
/// `algorithm` for those three classes, exactly as before #297, so solve()'s behavior -
/// `algorithm` is accepted but ignored for MILP/QP/MIQP - is genuinely unchanged. A caller
/// going through the registry directly with select() gets the stricter, more honest contract
/// instead: an `algorithm` that does not name a class-appropriate registered engine is
/// refused (`engine == nullptr`) rather than silently solved by something else. Both
/// behaviors are intentional and independently regression-tested
/// (tests/unit/test_solver_engine.cpp, tests/unit/test_engine_selection.cpp).
[[nodiscard]] EngineChoice select(const SolverRegistry& registry, const Model& model,
                                  const Options& options, Logger& logger,
                                  bool warm_start = false);

}  // namespace sankhya::engine
