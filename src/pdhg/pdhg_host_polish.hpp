// SPDX-License-Identifier: Apache-2.0
// SANKHYA - feasibility polishing of an answer another engine returned (#1177).
//
// The CUDA PDHG engine has no feasibility polish of its own. This runs the CPU engine's
// FeasibilityPolisher (#483, [AHLL25] Applegate, Hinder, Lu & Lubin, arXiv:2501.07018, the
// feasibility polishing section) on the host, from the point and duals the card handed back:
// the card does the iterations, the host polishes. The rule for adopting the polished pair is
// the CPU engine's final polish (pdhg.cpp): a polished x is kept only when it is primal
// feasible to kPdhgTight relative and the project's absolute tolerance, and it replaces the
// card's answer when that answer had not met the request or the polished pair meets it too.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::pdhg {

/// Polish `first`, a first-order answer to `model` (values, row duals in the Solution's sign
/// convention), when pdhg_feasibility_polish is on and `first` is neither optimal nor without
/// a point. Each phase gets kPdhgFinalPolishBudgetMultiplier steps per iteration `first` took;
/// the time limit, read on `timer`, bounds it. `first` is left as it was when nothing is
/// adopted, apart from a note in its message.
void polish_engine_answer(Solution* first, const Model& model, const Options& options,
                          Logger& logger, SolveControl* control, const Timer& timer);

}  // namespace sankhya::pdhg
