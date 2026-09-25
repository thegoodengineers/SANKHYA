// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one Feasibility Jump run on one instance, CPU (#506) or GPU (#508), for the A/B
// in bench/runners/gpu_fj_ab.py. Not part of the default build (EXCLUDE_FROM_ALL).
//
//     sankhya-fj-ab <model.mps[.gz]> <cpu|gpu> <seconds> [seed] [restarts]
//
// Runs the search from the box point closest to zero with an unbounded work budget, stopped
// by the clock, and prints one line of key=value pairs: when the first feasible point was
// handed over (seconds from the call, the device's context and transfers included), how many
// points, the best objective in the model's own sense, the effort, and whether every point
// passed mip::feasibility_jump_point_is_feasible() against the model read from the file. A
// run whose points do not all pass says verified=0, and the runner counts it as not found.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mip/feasibility_jump.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/gpu_fj.hpp"
#endif

int main(int argc, char** argv) {
  using namespace sankhya;
  using Clock = std::chrono::steady_clock;
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model> <cpu|gpu> <seconds> [seed] [restarts]\n", argv[0]);
    return 2;
  }
  const std::string engine = argv[2];
  const double limit = std::atof(argv[3]);
  const std::uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
  const int restarts = argc > 5 ? std::atoi(argv[5]) : 0;
  Model model;
  const io::ReadResult loaded = io::read_model(argv[1], &model);
  if (!loaded.ok) {
    std::printf("status=read_error message=\"%s\"\n", loaded.error.c_str());
    return 1;
  }
  const double sense = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  const auto objective = [&model](const std::vector<double>& x) {
    double v = model.objective_offset;
    for (std::size_t j = 0; j < x.size(); ++j) v += model.col_cost[j] * x[j];
    return v;
  };

  mip::FeasibilityJumpSettings settings;
  settings.work_limit = std::numeric_limits<Count>::max();
  settings.seed = seed;
  const Clock::time_point start = Clock::now();
  const auto elapsed = [start]() {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  double first = -1.0;
  settings.on_point = [&first, &elapsed](const std::vector<double>&) {
    if (first < 0.0) first = elapsed();
  };
  settings.should_stop = [&elapsed, limit]() { return elapsed() >= limit; };
  const std::vector<double> from = mip::feasibility_jump_zero_start(model);

  mip::FeasibilityJumpResult found;
  int used_restarts = 1;
  long long launches = 0;
  long long rejected = 0;
  std::string reason;
  if (engine == "cpu") {
    found = mip::feasibility_jump(model, from, settings);
  } else if (engine == "gpu") {
#ifdef SANKHYA_ENABLE_CUDA
    gpu::FjDeviceResult device = gpu::feasibility_jump(model, from, settings, restarts);
    if (!device.ran) {
      std::printf("status=no_device message=\"%s\"\n", device.reason.c_str());
      return 1;
    }
    found = std::move(device.search);
    used_restarts = device.restarts;
    launches = static_cast<long long>(device.launches);
    rejected = static_cast<long long>(device.rejected);
    reason = device.reason;
#else
    (void)restarts;
    std::printf("status=no_cuda_build\n");
    return 1;
#endif
  } else {
    std::fprintf(stderr, "engine must be cpu or gpu\n");
    return 2;
  }
  const double seconds = elapsed();
  bool verified = true;
  for (const std::vector<double>& x : found.points) {
    verified = verified && mip::feasibility_jump_point_is_feasible(model, x, tol::kIntegrality);
  }
  const bool any = !found.points.empty();
  std::printf(
      "status=%s engine=%s first_seconds=%.6f points=%zu best_objective=%.17g verified=%d "
      "seconds=%.3f work=%lld moves=%lld weight_updates=%lld restarts=%d launches=%lld "
      "rejected=%lld sense=%s message=\"%s\"\n",
      any ? "found" : "none", engine.c_str(), first, found.points.size(),
      any ? objective(found.points.back()) : 0.0, verified ? 1 : 0, seconds,
      static_cast<long long>(found.work), static_cast<long long>(found.moves),
      static_cast<long long>(found.weight_updates), used_restarts, launches, rejected,
      sense < 0.0 ? "max" : "min", reason.c_str());
  return 0;
}
