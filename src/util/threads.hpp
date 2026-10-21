// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the one place that talks to the OpenMP runtime (#57).
#pragma once

#include <cstdint>

#include "sankhya/logging.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#ifdef SANKHYA_HAVE_OPENMP
#include <omp.h>
#endif

namespace sankhya {

namespace detail {
/// The `omp_min_work` of the solve running on this thread. Thread-local, like OpenMP's own
/// team size: a thread the solver starts begins at the default, not at its parent's value.
inline thread_local std::int64_t omp_min_work_value = tol::kOmpMinParallelWork;
}  // namespace detail

/// Whether a column loop of `work` entries (nonzeros plus columns it reads) is worth an
/// OpenMP team. Below the threshold the fork and join cost more than the loop: on a 26-column
/// node LP of the spatial branch and bound, a team per simplex iteration cost about 100x of
/// the node throughput on a loaded 4-core machine (#514). Used as each loop's `if` clause, so
/// the loop body, and with it the answer, is the same either way.
[[nodiscard]] inline bool omp_worth_it(std::int64_t work) {
  return work >= detail::omp_min_work_value;
}

/// Apply the `threads` option to the OpenMP runtime, once per solve. 0 means one thread per
/// hardware core (the runtime's default). Without OpenMP the option is acknowledged in the
/// log and nothing else happens: there is no thread pool to size.
inline void apply_thread_option(const Options& options, Logger& logger) {
  const std::int64_t requested = options.get_int("threads");
  detail::omp_min_work_value = options.get_int("omp_min_work");
#ifdef SANKHYA_HAVE_OPENMP
  if (requested > 0) omp_set_num_threads(static_cast<int>(requested));
  logger.debug("Threads: {} worker(s) for the column loops (OpenMP)",
               requested > 0 ? static_cast<int>(requested) : omp_get_max_threads());
#else
  if (requested != 1) {
    logger.info("Threads: {} requested, but this build has no OpenMP; running single-threaded",
                requested);
  }
#endif
}

}  // namespace sankhya
