// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the CLI's handling of nonlinear (.nl) models (NLP stage 1).
#pragma once

#include <string>

#include "sankhya/options.hpp"

namespace sankhya::cli {

/// `sankhya info model.nl`: the structure a judge would otherwise count by hand - columns,
/// integer columns, constraints and how many are nonlinear, the problem class, whether the
/// continuous relaxation is PROVED convex (and if not, which part stops it), the domain risks
/// the bounds allow, and the sizes of the Jacobian and of the Hessian of the Lagrangian.
/// Returns the CLI exit code.
int nonlinear_info(const std::string& path);

/// `sankhya solve model.nl`: the NLP interior point (src/nlp/nlp_solve.hpp), the summary the
/// linear path prints, and the .sol / stats files over the model's solution frame. Returns the
/// CLI exit code.
int nonlinear_solve(const std::string& path, const Options& options,
                    const std::string& solution_path, const std::string& stats_path);

}  // namespace sankhya::cli
