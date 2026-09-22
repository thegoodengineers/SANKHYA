// SPDX-License-Identifier: Apache-2.0
// SANKHYA - see pdhg_gpu_guard.hpp. Named .cu (no kernel code inside) so it is picked up by
// CMakeLists.txt's `file(GLOB "src/gpu/*.cu")` without a build-system edit; it calls only
// the plain-C++ device-query and arithmetic functions declared in device.hpp/gpu_memory.hpp.
#include "gpu/pdhg_gpu_guard.hpp"

#include "gpu/device.hpp"
#include "gpu/gpu_memory.hpp"

namespace sankhya::gpu {

bool gpu_pdhg_is_safe(const Model& model, const Options& options, Logger& logger) {
  if (options.get_bool("deterministic")) {
    logger.warning(
        "GPU PDHG: deterministic=true is incompatible with atomicAdd reductions (#383); "
        "falling back to CPU PDHG");
    return false;
  }

  int cap_major = 0, cap_minor = 0;
  if (device_compute_capability(&cap_major, &cap_minor)) {
    if (!is_supported_compute_capability(cap_major, cap_minor)) {
      logger.warning(
          "GPU PDHG: device compute {}.{} is below the minimum compiled architecture "
          "({}.{}); falling back to CPU PDHG",
          cap_major, cap_minor, kMinComputeArch / 10, kMinComputeArch % 10);
      return false;
    }
  }

  std::size_t free_bytes = 0, total_bytes = 0;
  if (device_free_memory(&free_bytes, &total_bytes)) {
    // Uses the caller's `model` dimensions as given (pre-presolve when called from
    // solve.cpp): conservative (overestimates), safe.
    const std::size_t required =
        estimate_pdhg_gpu_memory(model.num_rows(), model.num_cols(), model.num_nonzeros());
    const std::size_t reserve = vram_reserve(total_bytes);
    logger.info(
        "GPU PDHG memory: required {:.0f} MiB, available {:.0f} MiB, reserve {:.0f} MiB",
        static_cast<double>(required) / 1048576.0, static_cast<double>(free_bytes) / 1048576.0,
        static_cast<double>(reserve) / 1048576.0);
    if (required + reserve > free_bytes) {
      logger.warning(
          "GPU PDHG: estimated {:.0f} MiB + {:.0f} MiB reserve exceeds {:.0f} MiB free "
          "VRAM; falling back to CPU PDHG",
          static_cast<double>(required) / 1048576.0, static_cast<double>(reserve) / 1048576.0,
          static_cast<double>(free_bytes) / 1048576.0);
      return false;
    }
  }

  return true;
}

}  // namespace sankhya::gpu
