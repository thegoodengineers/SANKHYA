// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the model writers' semi-continuous and SOS output (#754), the inverse of
// mps_sos.cpp, mps_bounds.cpp's SC bound and lp_sos.cpp. Internal to src/io/.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::io {

/// The BOUNDS lines of one semi-continuous column: its lower end when it is not 0 (LO, or LI
/// for a semi-integer column), then " SC BND <name> <u>", which the reader reads as the
/// upper end and the flag together.
void write_mps_semicontinuous_bounds(std::FILE* out, const std::string& name, double lower,
                                     double upper, bool integer);

/// The SOS section, one " S1 SOS <name> <priority>" header per set and one
/// "    <set> <column> <weight>" line per member; nothing when the model has no set.
void write_mps_sos(std::FILE* out, const Model& model, const std::vector<std::string>& columns);

/// The LP format's "Semi-Continuous" and "SOS" sections; nothing for a model with neither.
void write_lp_sc_sos(std::FILE* out, const Model& model,
                     const std::vector<std::string>& columns);

}  // namespace sankhya::io
