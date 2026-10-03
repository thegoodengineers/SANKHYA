// SPDX-License-Identifier: Apache-2.0
// SANKHYA - semi-continuous columns and special ordered sets (#754): the checks every part of
// the solver measures a point against, in one place.
//
// Not a public header. Model::validate() and Solution::recompute_quality() call into it, and
// so does the branch and bound, so the search, the status guard and the answer's reported
// violation all mean the same thing by "the point breaks a set".
#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

/// How far x is from the semi-continuous domain {0} or [l, +inf) on its lower side: the
/// smaller of the two moves that would repair it, min(x, l - x), when 0 < x < l, and 0
/// otherwise. The upper bound is an ordinary bound and is measured as one.
[[nodiscard]] double semicontinuous_violation(double x, double lower);

/// How far a point is from satisfying one special ordered set: for type 1 the second
/// largest |x| among the members (all but one must vanish), for type 2 the smallest, over
/// every pair of adjacent members, of the largest |x| outside that pair. Zero when the set
/// is satisfied exactly. `x` is indexed by column.
[[nodiscard]] double sos_violation(const SosSet& set, const double* x);

/// The largest of the two measures above over the whole model; 0 when it has neither.
[[nodiscard]] double semicontinuous_and_sos_violation(const Model& model, const double* x);

/// The structural checks validate() makes on the two fields; empty when they are well formed.
[[nodiscard]] std::string validate_semicontinuous_and_sos(const Model& model);

/// The set with its members in increasing weight order (stable, so equal weights keep the
/// file's order and validate() then refuses them). Both readers end with this.
[[nodiscard]] SosSet sorted_by_weight(const SosSet& set);

/// One flag per column, set for the semi-continuous ones; empty when there are none.
[[nodiscard]] std::vector<char> semicontinuous_mask(const Model& model);

}  // namespace sankhya
