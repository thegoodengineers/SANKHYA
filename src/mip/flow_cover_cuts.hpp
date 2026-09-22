// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover inequalities for the fixed-charge / single-node flow family (#419).
//
// THE STRUCTURE. Three MIPLIB instances that never reach the published optimum at the 60 s
// limit (ran12x21, ran13x13, k16x240b; #419) share a shape none of the existing families is
// built for: a continuous flow variable x_j switched on by a binary y_j through a variable
// upper bound x_j <= u_j y_j, with several such (x_j, y_j, u_j) triples sharing one capacity
// row sum_j x_j <= b. That is the "single-node 0-1 flow set"
//
//     X = { (x, y) in R^n_+ x {0,1}^n :  sum_j x_j <= b,  x_j <= u_j y_j  for every j }
//
// which is exactly the structure MIR is weak on: MIR bound-substitutes each continuous x_j
// against ITS OWN bound, but the bound that actually matters here is u_j y_j, a bound the row
// itself does not carry - it lives in a separate row. Gomory and the knapsack-cover family
// don't apply either (the row is not a pure 0/1 knapsack; it has continuous columns).
//
// THE INEQUALITY (Padberg, Van Roy & Wolsey, "Valid linear inequalities for fixed charge
// problems", Operations Research 33 (1985)). Let C be a subset of the row's FULL flow-column
// set that is a FLOW COVER: sum_{j in C} u_j = b + lambda, lambda > 0 (the excess). Split C
// into C+ = { j in C : u_j > lambda } and C- = C \ C+. The flow cover inequality
//
//     sum_{j in C} x_j + sum_{j in C+} (u_j - lambda) (1 - y_j)  <=  b
//
// is valid for every point of X (PVW 1985, Theorem 1; also Wolsey, "Integer Programming",
// section 9.4, and Nemhauser & Wolsey, "Integer and Combinatorial Optimization", section
// II.3.6). The textbook instance the unit test checks by hand (x1+x2+x3-y1<=9, from capacities
// u=(6,5,4,3) and budget b=10 on sum_j x_j<=b, cover C={1,2,3}, excess lambda=5, C+={1}) is
// this inequality. Because the theorem is proved by a case split on which cover members are
// switched on that is easy to get subtly wrong by hand, the obligation cuts.hpp sets for every
// family - an exact-arithmetic validity gate plus enumeration over random models - is what
// this file is actually checked against, not a from-scratch re-derivation in a comment.
//
// GENERAL COEFFICIENTS. A capacity row sum_j a_j x_j <= b with a_j > 0 is handled by the
// substitution z_j = a_j x_j, U_j = a_j u_j (so z_j <= U_j y_j and sum_j z_j <= b is exactly
// the set above); the returned cut is stated back in x_j with coefficient a_j.
//
// LIFTING (partial, #419's "where practical"). The full sequential/superadditive lifting of
// Gu, Nemhauser & Savelsbergh ("Lifted flow cover inequalities for mixed 0-1 integer
// programs", Math. Programming 85 (1999)) solves a lifting subproblem per left-out variable
// and is future work. What is implemented here is the valid special case that needs no
// subproblem: for a flow column j on the same row but left out of the cover, the variable
// upper bound x_j <= u_j y_j is ITSELF a valid inequality (a_j x_j - a_j u_j y_j <= 0), and
// adding any non-negative multiple of a valid inequality to another preserves validity. The
// multiple a_j is added whenever it costs no violation at the current point - i.e. whenever
// the variable upper bound is already tight there (a_j x*_j - a_j u_j y*_j >= -tolerance, so
// the added term is ~0 and the cut's measured violation does not drop - which lets the cut
// carry the leftover column's coefficients without weakening the round's separation.
//
// AGGREGATION AND SINGLE-NODE RELAXATION. #419 scopes multi-row aggregation to "only as far
// as the architecture justifies": this generator is one-row separation, exactly the interface
// mir_cuts.cpp already exposes (Model, Solution, global bounds, Stats out-param), with the one
// piece of row-crossing structure the theorem itself needs - locating each flow column's own
// variable-upper-bound row - handled the same way mir_cuts.cpp scans rows_of(model). A single-
// node relaxation that aggregates several capacity rows around one implied network node is the
// natural next step the issue names, and is left for a follow-up once this family has a
// measurement to build on.
#pragma once

#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// What one call found, for the tests, the log and the stats block.
struct FlowCoverStats {
  int cuts = 0;            ///< cuts returned
  int lifted_cuts = 0;     ///< of those, at least one leftover column was folded in
  int lifted_terms = 0;    ///< leftover (x_j, y_j) pairs folded into some cut
  int vub_rows_found = 0;  ///< columns for which a variable-upper-bound row was located
};

/// Flow cover cuts from the model's capacity rows at `solution`, valid under `col_lower` /
/// `col_upper` (pass the GLOBAL bounds from a tree node so the cut holds everywhere, as
/// mir_cuts.cpp's node overload does). Every returned cut is valid for the integer-feasible
/// set of `model` by construction; the caller's filter decides which are worth adding.
[[nodiscard]] std::vector<Cut> generate_flow_cover_cuts(const Model& model,
                                                        const Solution& solution,
                                                        const std::vector<double>& col_lower,
                                                        const std::vector<double>& col_upper,
                                                        FlowCoverStats* stats = nullptr);

}  // namespace sankhya::mip
