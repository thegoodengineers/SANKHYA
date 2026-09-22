// SPDX-License-Identifier: Apache-2.0
#include "core/deterministic_mode.hpp"

#include <cstdint>
#include <limits>

namespace sankhya {
namespace {

/// The value the termination options carry when nothing is to stop the solve on the clock.
constexpr double kNoWallClockLimit = std::numeric_limits<double>::max();

}  // namespace

Options apply_deterministic_mode(const Options& requested, Logger& logger) {
  if (!requested.get_bool("deterministic")) return requested;

  const Options defaults;
  Options effective = requested;

  const double time_limit = requested.get_double("time_limit");
  if (time_limit != defaults.get_double("time_limit")) {
    logger.warning(
        "deterministic: time_limit={:g}s is refused - a solve that stops on the clock "
        "returns a different answer on a slower machine. Use iteration_limit or node_limit, "
        "which count the same on every machine",
        time_limit);
    effective.set_double("time_limit", defaults.get_double("time_limit"));
  }

  const double polish_seconds = requested.get_double("polish_max_seconds");
  if (polish_seconds != defaults.get_double("polish_max_seconds")) {
    logger.warning(
        "deterministic: polish_max_seconds={:g} is refused; the polish is bounded by "
        "polish_max_factor_nonzeros, which is a property of the model rather than of the "
        "machine",
        polish_seconds);
  }
  // Even at its default the seconds budget would decide whether the polish finishes, so it
  // goes entirely, set to the same no-limit sentinel the termination options use. What
  // bounds the polish afterwards is polish_max_factor_nonzeros, a property of the model.
  effective.set_double("polish_max_seconds", kNoWallClockLimit);

  const std::int64_t threads = requested.get_int("threads");
  if (threads == defaults.get_int("threads")) {
    effective.set_int("threads", 1);
  } else {
    // The measurement in #57 says the column loops are bit-identical at 1 and 8 threads, and
    // a caller who set threads has read that. Saying so here is the difference between
    // relying on the claim and relying on it knowingly.
    logger.warning(
        "deterministic: running on {} threads; reproducibility then rests on the parallel "
        "reductions being order-independent, which #57 measured but this mode does not "
        "re-check",
        threads);
  }
  return effective;
}

}  // namespace sankhya
