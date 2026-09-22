// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts for the fixed-charge structure (#419).
//
// THE STRUCTURE. A fixed-charge row is a flow balance over arcs whose flow is switched on by
// a binary: sum_{j in N+} x_j - sum_{j in N-} x_j <= b with 0 <= x_j <= u_j y_j, y_j in
// {0, 1}. The variable upper bound x_j <= u_j y_j is a row of the model in its own right
// (x_j - u_j y_j <= 0, two nonzeros), which is how the structure is recognised here: every
// such row names an indicator and a capacity for its continuous column, and every other row
// whose inflows carry an indicator is a single-node flow set. A binary that sits directly in
// the flow row is its own indicator (x_j = u_j y_j with u_j its coefficient), and a column
// with neither indicator nor switch is a flow whose switch is always on.
//
// THE INEQUALITY. Padberg, Van Roy and Wolsey, "Valid linear inequalities for fixed charge
// problems", Operations Research 33 (1985): for a flow cover C+ subset of N+ with
// lambda = sum_{C+} u_j - b > 0,
//
//     sum_{C+} x_j + sum_{C+} (u_j - lambda)^+ (1 - y_j)
//         <= b + sum_{j in L-} lambda y_j + sum_{j in N- \ L-} x_j
//
// for any L- subset of N-. Why it holds: with T the arcs of C+ switched on, the left side is
// at most min(b', sum_T u_j) + sum_{C+ \ T} (u_j - lambda)^+, where b' is b plus whatever the
// outflows carry; when the capacity switched off, S = sum_{C+ \ T} u_j, is at least lambda
// that is at most b, and when it is less than lambda every (u_j - lambda)^+ in it is zero
// and the bound sum_T u_j = b + lambda - S is all that is left, which is below b + lambda,
// the least the right side can be once any L- arc is on; with every L- arc off those arcs
// carry nothing and b' is b plus the N- \ L- flows, the right side exactly. Separation
// picks C+ from the arcs whose indicator is at least a half at the LP point, extends it by
// the next-best arcs until it is a cover, and puts each outflow on the right in the form
// that is smaller at the LP point. The lifted coefficients of Gu, Nemhauser and Savelsbergh,
// "Lifted flow cover inequalities for mixed 0-1 integer programs", Math. Programming 85
// (1999), for the arcs left out of the cover are not built here: the arcs outside C+ get a
// zero coefficient, which is valid and weaker.
//
// VALIDITY IS BY CONSTRUCTION AND CHECKED IN EXACT ARITHMETIC. tests/unit/test_flow_cover_cuts
// fixes every indicator assignment of small random fixed-charge models and maximises each
// cut's violation over the continuous part with the rational simplex oracle: the maximum
// must be non-positive, exactly. The shared filter in cuts.hpp decides density and dynamism.

#pragma once

#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

struct FlowCoverStats {
  int vub_rows = 0;   ///< variable-upper-bound rows x <= u y recognised
  int flow_rows = 0;  ///< row sides with at least one switched inflow, tried
  int covers = 0;     ///< of those, where a cover with lambda > 0 was found
  int cuts = 0;       ///< violated cuts returned
};

/// Flow cover cuts violated at `solution.col_value`, from the rows whose inflows carry an
/// indicator. `col_lower` / `col_upper` are the bounds the structure is read under: the
/// GLOBAL ones, so a cut is valid at every node.
[[nodiscard]] std::vector<Cut> generate_flow_cover_cuts(const Model& model,
                                                        const Solution& solution,
                                                        const std::vector<double>& col_lower,
                                                        const std::vector<double>& col_upper,
                                                        FlowCoverStats* stats = nullptr);

}  // namespace sankhya::mip
