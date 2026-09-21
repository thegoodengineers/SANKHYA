// SPDX-License-Identifier: Apache-2.0
#include "mip/components.hpp"

namespace sankhya::mip {

MilpComponents describe_components(const Options& options) {
  MilpComponents components;
  components.node_selection = options.get_string("mip_node_selection");
  components.branching = options.get_string("mip_branching");
  components.relaxation_engine = options.get_string("mip_node_engine");
  components.cuts_enabled = options.get_bool("enable_root_cuts");
  components.heuristics_enabled = options.get_bool("mip_heuristics");
  components.conflict_analysis_enabled = options.get_bool("conflict_analysis");
  return components;
}

}  // namespace sankhya::mip
