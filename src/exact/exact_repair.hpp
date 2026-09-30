// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact repair of an LP basis that is optimal only to tolerance (#757).
//
// The double-precision simplex stops when every primal and dual infeasibility is below its
// tolerance (1e-7). Rebuilt in exact rational arithmetic, such a basis is often not optimal
// at all: a basic value 1e-12 below its bound, a reduced cost of -1e-15 on a nonbasic
// column. On Netlib that was 34 of the 96 bases #757's CSV covered, and exact duals and
// ranges of a non-optimal basis are not defined, so the certified sensitivity had nothing
// to certify.
//
// The repair takes exact simplex pivots from that basis until it is exactly optimal. This is
// the approach of Koch ("The final NETLIB-LP results", Operations Research Letters 32, 2004),
// who checked the bases a floating-point simplex returned in exact rational arithmetic and,
// where one was not optimal, continued from it until an exactly optimal basis was found; and
// of Applegate, Cook, Dash and Espinoza ("Exact solutions to linear programming problems",
// Operations Research Letters 35, 2007), whose exact solver warm-starts from the floating-
// point optimal basis and proves or corrects it in rational arithmetic. Here:
//
//   1. Primal infeasible and dual feasible: the dual simplex, Bland's rule (the smallest
//      infeasible basic variable leaves; the entering ratio test breaks ties by the smallest
//      index), which is finite.
//   2. Primal and dual infeasible: the costs of the dual infeasible nonbasic variables are
//      shifted by exactly their reduced costs, which makes the basis dual feasible; step 1
//      runs; the true costs come back. This is the cost shifting a floating-point dual
//      simplex uses (Koberstein, PhD thesis, Paderborn, 2005), here in exact
//      arithmetic, where removing the shift leaves no residue to clean up but real dual
//      infeasibilities, which step 3 removes.
//   3. Primal feasible and dual infeasible: the primal simplex with Bland's rule and bound
//      flips, which keeps primal feasibility and is finite.
//
// Every quantity is exact: the pivots are chosen by exact signs and exact ratios, and the
// basis the repair returns is checked exactly optimal (not_optimal) with the true costs
// before it is reported. A basis that cannot be repaired within the pivot cap or the time
// budget is left as the solver reported it, with the reason.
#pragma once

#include <string>
#include <vector>

#include "exact/exact_verify.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya::exact {

struct RepairResult {
  /// kVerified: the basis below is exactly optimal (unchanged when no step was needed);
  /// kDeclined: not attempted or not finished; kFailed: the model is exactly infeasible or
  /// unbounded from this basis, so the double-precision `optimal` was wrong.
  ExactVerdict verdict = ExactVerdict::kDeclined;
  std::string message;
  int dual_pivots = 0;
  int primal_pivots = 0;
  int bound_flips = 0;
  int shifted_costs = 0;  ///< dual infeasible costs shifted, then restored (step 2)
  [[nodiscard]] bool changed() const { return dual_pivots + primal_pivots + bound_flips > 0; }
  /// The repaired basis and its exact point and duals, rounded once to the nearest double,
  /// in the model's sense. Filled only on kVerified.
  std::vector<BasisStatus> col_status, row_status;
  std::vector<double> col_value, col_dual, row_dual;
};

/// `solution` must be an optimal plain LP with a basis. Declines otherwise.
[[nodiscard]] RepairResult repair_basis_exact(const Model& model, const Solution& solution,
                                              double seconds = kInfinity);

/// repair_basis_exact; when the basis changed, it and its exact point and duals (nearest
/// doubles) replace the reported ones and every quality measure is recomputed. solve()
/// calls it under option "exact" before the exact check and the certified sensitivity.
/// Returns true when the reported basis was replaced.
bool apply_exact_repair(const Model& model, Solution* solution, Logger& logger, double seconds);

}  // namespace sankhya::exact
