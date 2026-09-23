// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an in-process KKT check of an LP answer against the model it answers (#476).
//
// WHY A SECOND COPY OF THE VERIFIER'S CHECKS. tools/verify_solution.py is the project's
// independent verifier: it shares no code with the solver and reads only the written .sol
// file, and it must stay that way. But the engine race (#476) has to judge an answer INSIDE
// the process, before it lets that answer stop the other engines, and it cannot shell out to
// Python in the middle of a solve. So the checks the verifier makes of an optimal LP answer
// are reimplemented here, in C++, from the model and the Solution alone - the same measures,
// the same scalings and the same thresholds, so an answer that passes here is one the
// verifier would pass - without the verifier linking or calling any of it. The Python tool is
// unchanged and remains the independent judge of everything written to disk.
//
// The checks, each named in the verdict when it fails, are those of verify_solution.py for
// status `optimal` on an LP: the point's size and finiteness, column bounds and recomputed
// row activities (each relative to the scale of the terms it came from), agreement of the
// reported activities, the recomputed objective, the reduced costs against c - A^T y, the
// sign conditions of the column and row multipliers, complementary slackness, the basis when
// every status is known, and strong duality with the per-item accounting the verifier does.
//
// References: the KKT conditions of LP, e.g. Nocedal & Wright, "Numerical Optimization",
// 2nd ed. (2006), thm. 13.1; the scalings are verify_solution.py's and are argued there.
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {

/// The thresholds the check applies. The defaults are the project's (tolerances.hpp) and the
/// verifier's command-line defaults, which are the same numbers.
struct KktTolerances {
  double primal = tol::kPrimalFeasibility;  ///< scaled bound and row violation
  double dual = tol::kDualFeasibility;      ///< scaled multiplier sign violation
  double duality_gap = tol::kDualityGap;    ///< relative gap beyond the accounted shares
};

/// The outcome: passed, or the first check that failed and the number that failed it.
struct KktVerdict {
  bool passed = false;
  std::string check;   ///< the verifier's name for the check that failed; empty on a pass
  std::string detail;  ///< the measurement, in the verifier's words
};

/// Does `solution`, which claims `optimal`, satisfy the KKT conditions of the LP `model` at
/// `tolerances`? `model` must be the ORIGINAL model (the one the caller asked about), and the
/// solution's vectors must be in its rows and columns. A model with integer columns or a
/// Hessian is not an LP and is refused with check "class".
[[nodiscard]] KktVerdict check_lp_optimality(const Model& model, const Solution& solution,
                                             const KktTolerances& tolerances = {});

}  // namespace sankhya
