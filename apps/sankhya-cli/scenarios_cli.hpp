// SPDX-License-Identifier: Apache-2.0
// SANKHYA - `sankhya scenarios MODEL SCENARIOS.csv` (#752).
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::cli {

/// Solve `model` (read from `model_path`) under every row of `csv_path` and print one line
/// per scenario; with `out_path`, also write them as CSV. With `compare`, each scenario is
/// also solved on its own - the file read again, presolved, started cold, as N separate
/// `sankhya solve` calls would - and the two objectives and times are reported side by side.
/// Returns 0 when every scenario's answer was verified (and, with `compare`, agrees with its
/// single solve to tol::kScenarioAgreement), 1 when any was not, 3 on an input error.
int run_scenarios(const Model& model, const std::string& model_path,
                  const std::string& csv_path, const Options& options,
                  const std::string& out_path, bool compare);

}  // namespace sankhya::cli
