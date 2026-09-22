// SPDX-License-Identifier: Apache-2.0
#include "mip/components.hpp"

#include "mip/heuristics.hpp"

namespace sankhya::mip {

MilpComponents describe_components(const Options& options) {
  MilpComponents components;
  components.node_selection = options.get_string("mip_node_selection");
  components.branching = options.get_string("mip_branching");
  components.relaxation_engine = options.get_string("mip_node_engine");
  components.cuts_enabled = options.get_bool("enable_root_cuts");
  // The same resolution the search makes (#414): the master switch and every heuristic's
  // own auto/on/off, so a single `mip_heur_rens=on` reads as heuristics on.
  const HeuristicSchedule schedule = HeuristicSchedule::from(options);
  components.heuristics_enabled = schedule.any_optional();
  components.heuristics = schedule.names();
  components.conflict_analysis_enabled = options.get_bool("conflict_analysis");
  return components;
}

}  // namespace sankhya::mip
