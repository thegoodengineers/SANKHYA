// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sankhya/types.hpp"

namespace sankhya::la {

/// The dual step of a primal-dual splitting on the row box C = [lo, hi]: the prox of
/// sigma times the support function of C at v. Moreau's identity (Moreau 1965; Parikh and
/// Boyd, "Proximal Algorithms", 2014, section 2.5) gives it as v - sigma proj_C(v / sigma),
/// which equals v - proj_[sigma lo, sigma hi](v) because sigma > 0 scales the box.
///
/// The second form is the one computed. Where v lies inside the scaled box (a free row, or a
/// row away from both its bounds) the answer is 0 exactly, and that is what makes a free
/// row's multiplier 0. The first form returns v - sigma * fl(v / sigma) there, the rounding
/// error of the division: nonzero for about one point in eleven unfused, and for every point
/// once the compiler fuses the multiply and subtract into an FMA (#1100). On a side the point
/// is past, both forms compute v - sigma * bound, the same bits.
[[nodiscard]] inline double moreau_dual_prox(double v, double sigma, double lo, double hi) {
  if (is_finite_bound(lo) && v < sigma * lo) return v - sigma * lo;
  if (is_finite_bound(hi) && v > sigma * hi) return v - sigma * hi;
  return 0.0;
}

}  // namespace sankhya::la
