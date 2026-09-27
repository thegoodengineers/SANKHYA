// SPDX-License-Identifier: Apache-2.0
// SANKHYA - see pdhg_gpu_guard.hpp. Named .cu (no kernel code inside) so it is picked up by
// CMakeLists.txt's `file(GLOB "src/gpu/*.cu")` without a build-system edit; it calls only
// the plain-C++ device-query and arithmetic functions declared in device.hpp/gpu_memory.hpp.
#include "gpu/pdhg_gpu_guard.hpp"

#include "gpu/device.hpp"
#include "gpu/gpu_memory.hpp"
#include "gpu/multi_device.hpp"

namespace sankhya::gpu {

bool gpu_pdhg_is_safe(const Model& model, const Options& options, Logger& logger) {
  // Both GPU engines are bit-for-bit repeatable under deterministic=true: the single-device
  // engine since #478 (fixed-order reductions, both products non-transpose CSR_ALG2 on an
  // explicit A^T), the multi-device engine since its reductions were written that way
  // (pdhg_multi_gpu_device.hpp) and its cross-card sum runs in slot order
  // (multi_gpu_exchange.hpp; tests/unit/test_multi_gpu_solve.cpp,
  // TwoRunsAndOneOrTwoCardsGiveTheSameBits). #383's refusal of the multi-device path under
  // the flag is gone (#295).
  const bool deterministic = options.get_bool("deterministic");

  // Every card named in gpu_devices must meet the compiled minimum, not only device 0
  // (#295): a set with one older card would launch on it and fail at run time. A card that
  // does not exist is left to the engine, which says so and runs on one card.
  const std::vector<int> device_ids = parse_device_ids(options.get_string("gpu_devices"));
  for (int id : device_ids) {
    int cap_major = 0, cap_minor = 0;
    const bool known = device_ids.size() > 1
                           ? device_compute_capability_of(id, &cap_major, &cap_minor)
                           : device_compute_capability(&cap_major, &cap_minor);
    if (known && !is_supported_compute_capability(cap_major, cap_minor)) {
      logger.warning(
          "GPU PDHG: device {} compute {}.{} is below the minimum compiled architecture "
          "({}.{}); falling back to CPU PDHG",
          id, cap_major, cap_minor, kMinComputeArch / 10, kMinComputeArch % 10);
      return false;
    }
  }

  std::size_t free_bytes = 0, total_bytes = 0;
  if (device_free_memory(&free_bytes, &total_bytes)) {
    // Uses the caller's `model` dimensions as given (pre-presolve when called from
    // solve.cpp): conservative (overestimates), safe.
    // Deterministic mode holds A^T as a second CSR matrix on the device (#478).
    const std::size_t required =
        estimate_pdhg_gpu_memory(model.num_rows(), model.num_cols(), model.num_nonzeros()) +
        (deterministic
             ? estimate_pdhg_gpu_transpose_memory(model.num_cols(), model.num_nonzeros())
             : 0);
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
