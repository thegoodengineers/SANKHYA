// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU CUDA device utilities (#295).
//
// CUDA-requiring functions only. Compiled when SANKHYA_ENABLE_CUDA is ON via the
// CMakeLists glob of src/gpu/*.cu.

#include "multi_device.hpp"

#include <cuda_runtime.h>

namespace sankhya::gpu {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

bool can_peer_access(int from, int to) {
  if (from == to) return true;
  int can = 0;
  if (cudaDeviceCanAccessPeer(&can, from, to) != cudaSuccess) return false;
  return can != 0;
}

bool device_compute_capability_of(int id, int* major, int* minor) {
  if (id < 0 || id >= device_count()) return false;
  int cc_major = 0, cc_minor = 0;
  if (cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, id) != cudaSuccess ||
      cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, id) != cudaSuccess)
    return false;
  if (major) *major = cc_major;
  if (minor) *minor = cc_minor;
  return true;
}

bool device_memory_of(int id, std::size_t* free_bytes, std::size_t* total_bytes) {
  if (id < 0 || id >= device_count()) return false;
  int previous = 0;
  if (cudaGetDevice(&previous) != cudaSuccess) return false;
  if (cudaSetDevice(id) != cudaSuccess) return false;
  std::size_t free_val = 0, total_val = 0;
  const bool ok = cudaMemGetInfo(&free_val, &total_val) == cudaSuccess;
  (void)cudaSetDevice(previous);
  if (!ok) return false;
  if (free_bytes) *free_bytes = free_val;
  if (total_bytes) *total_bytes = total_val;
  return true;
}

}  // namespace sankhya::gpu
