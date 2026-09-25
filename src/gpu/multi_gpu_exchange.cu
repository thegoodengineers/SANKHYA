// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cross-card sum of A^T y in the row-partitioned multi-GPU PDHG (#295).
// Design and the NCCL decision: multi_gpu_exchange.hpp.

#include "multi_gpu_exchange.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "multi_device.hpp"

namespace sankhya::gpu::multi {
namespace {

struct SlotPointers {
  const double* p[kMaxDevices];
};

// out[j] = ((p_0[j] + p_1[j]) + p_2[j]) + ... in slot order: the same additions in the same
// order on every card, whatever the transport, so the result is identical everywhere.
__global__ void k_fixed_order_sum(SlotPointers s, int slots, double* __restrict__ out, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  double acc = s.p[0][j];
  for (int k = 1; k < slots; ++k) acc += s.p[k][j];
  out[j] = acc;
}

}  // namespace

const char* to_string(Transport t) {
  return t == Transport::kPeer ? "peer-to-peer" : "host-staged";
}

PartialSumExchange::~PartialSumExchange() {
  for (std::size_t k = 0; k < cards_.size(); ++k) {
    cudaSetDevice(cards_[k]->device_id);
    cudaStreamSynchronize(cards_[k]->stream);
    if (k < ready_.size() && ready_[k]) cudaEventDestroy(ready_[k]);
    if (k < staged_.size() && staged_[k]) cudaEventDestroy(staged_[k]);
  }
  if (host_stage_) cudaFreeHost(host_stage_);
  // Leave the process's peer mappings as they were found.
  for (const auto& [from, to] : enabled_) {
    cudaSetDevice(from);
    cudaDeviceDisablePeerAccess(to);
  }
}

bool PartialSumExchange::init(const std::vector<DeviceState*>& cards, bool allow_peer,
                              std::string* why) {
  cards_ = cards;
  const std::size_t k_cards = cards_.size();
  if (k_cards == 0 || k_cards > static_cast<std::size_t>(kMaxDevices)) return false;
  ready_.assign(k_cards, nullptr);
  staged_.assign(k_cards, nullptr);
  for (std::size_t k = 0; k < k_cards; ++k) {
    if (cudaSetDevice(cards_[k]->device_id) != cudaSuccess) return false;
    if (cudaEventCreateWithFlags(&ready_[k], cudaEventDisableTiming) != cudaSuccess) return false;
    if (cudaEventCreateWithFlags(&staged_[k], cudaEventDisableTiming) != cudaSuccess)
      return false;
  }

  // Distinct physical devices; two slots on one card copy within that card.
  std::vector<int> physical;
  for (const DeviceState* c : cards_) {
    bool seen = false;
    for (int p : physical) seen = seen || p == c->device_id;
    if (!seen) physical.push_back(c->device_id);
  }

  bool peer = allow_peer;
  std::string reason = allow_peer ? "" : "gpu_peer_access=false";
  for (std::size_t a = 0; peer && a < physical.size(); ++a)
    for (std::size_t b = 0; peer && b < physical.size(); ++b) {
      if (a == b) continue;
      if (!can_peer_access(physical[a], physical[b])) {
        peer = false;
        reason = "device " + std::to_string(physical[a]) + " cannot access device " +
                 std::to_string(physical[b]) + " by P2P";
      }
    }
  for (std::size_t a = 0; peer && a < physical.size(); ++a)
    for (std::size_t b = 0; peer && b < physical.size(); ++b) {
      if (a == b) continue;
      if (cudaSetDevice(physical[a]) != cudaSuccess) return false;
      const cudaError_t e = cudaDeviceEnablePeerAccess(physical[b], 0);
      if (e == cudaSuccess) {
        enabled_.emplace_back(physical[a], physical[b]);
      } else if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();  // clear the (non-sticky) error: access is already on
      } else {
        cudaGetLastError();
        peer = false;
        reason = "cudaDeviceEnablePeerAccess(" + std::to_string(physical[a]) + " -> " +
                 std::to_string(physical[b]) + ") failed: " + cudaGetErrorString(e);
      }
    }

  if (peer) {
    transport_ = Transport::kPeer;
    if (why)
      *why = physical.size() > 1 ? "peer access enabled between every pair of cards"
                                 : "all slots on one card";
    return true;
  }
  transport_ = Transport::kHostStaged;
  if (why) *why = reason;
  const std::size_t n = static_cast<std::size_t>(cards_[0]->n);
  if (n > 0 &&
      cudaMallocHost(reinterpret_cast<void**>(&host_stage_), k_cards * n * sizeof(double)) !=
          cudaSuccess)
    return false;
  return true;
}

bool PartialSumExchange::enqueue() {
  const int k_cards = static_cast<int>(cards_.size());
  const int n = cards_[0]->n;
  if (n == 0) return true;
  const auto bytes = static_cast<std::size_t>(n) * sizeof(double);
  const auto un = static_cast<std::size_t>(n);

  for (int k = 0; k < k_cards; ++k) {
    DeviceState& c = *cards_[static_cast<std::size_t>(k)];
    if (cudaSetDevice(c.device_id) != cudaSuccess) return false;
    if (transport_ == Transport::kHostStaged) {
      if (cudaMemcpyAsync(host_stage_ + static_cast<std::size_t>(k) * un, c.d_partial, bytes,
                          cudaMemcpyDeviceToHost, c.stream) != cudaSuccess)
        return false;
      if (cudaEventRecord(staged_[static_cast<std::size_t>(k)], c.stream) != cudaSuccess)
        return false;
    } else {
      if (cudaEventRecord(ready_[static_cast<std::size_t>(k)], c.stream) != cudaSuccess)
        return false;
    }
  }

  for (int j = 0; j < k_cards; ++j) {
    DeviceState& dst = *cards_[static_cast<std::size_t>(j)];
    if (cudaSetDevice(dst.device_id) != cudaSuccess) return false;
    SlotPointers slots{};
    for (int k = 0; k < k_cards; ++k) {
      if (k == j) {
        slots.p[k] = dst.d_partial;
        continue;
      }
      const DeviceState& src = *cards_[static_cast<std::size_t>(k)];
      double* into = dst.d_recv + static_cast<std::size_t>(k) * un;
      if (transport_ == Transport::kPeer) {
        if (cudaStreamWaitEvent(dst.stream, ready_[static_cast<std::size_t>(k)], 0) !=
                cudaSuccess ||
            cudaMemcpyPeerAsync(into, dst.device_id, src.d_partial, src.device_id, bytes,
                                dst.stream) != cudaSuccess)
          return false;
      } else {
        if (cudaStreamWaitEvent(dst.stream, staged_[static_cast<std::size_t>(k)], 0) !=
                cudaSuccess ||
            cudaMemcpyAsync(into, host_stage_ + static_cast<std::size_t>(k) * un, bytes,
                            cudaMemcpyHostToDevice, dst.stream) != cudaSuccess)
          return false;
      }
      slots.p[k] = into;
    }
    k_fixed_order_sum<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize, 0, dst.stream>>>(
        slots, k_cards, dst.d_aty, n);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
  }
  return true;
}

}  // namespace sankhya::gpu::multi
