// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cross-card sum of A^T y in the row-partitioned multi-GPU PDHG (#295).
//
// Included only by CUDA translation units.
//
// Each card k holds p_k = A_k^T y_k (n doubles). Every card needs A^T y = p_0 + ... + p_{K-1}.
// The exchange is an all-gather followed by a local sum: card j receives every other card's
// p_k into slot k of its receive buffer, then forms ((p_0 + p_1) + p_2) + ... on the device
// in slot order. The sum is the same sequence of double additions on every card, so the
// replicated primal iterate stays bit-identical across cards, and it is the same sequence
// whichever transport moved the partials, so the two transports give the same bits.
//
// Transports:
//   kPeer        cudaMemcpyPeerAsync, device to device, after cudaDeviceEnablePeerAccess on
//                every ordered pair of distinct cards (NVLink or PCIe P2P). No host copy,
//                no host synchronization: card j's stream waits on card k's "partial ready"
//                event and then pulls p_k.
//   kHostStaged  card k copies p_k into pinned host memory on its own stream, card j's
//                stream waits for that copy and uploads it. Used when a pair of cards cannot
//                reach each other, or when the option gpu_peer_access is false.
// Both are asynchronous; the solver's one host synchronization per iteration (the step-size
// scalars) is what keeps the next iteration from overwriting a partial or a staging slot
// before every reader has taken it.
//
// Why not NCCL: for K = 2 on one node an all-gather of the partials moves exactly the bytes
// a ring all-reduce moves, over the same NVLink, and NCCL's reduction order is chosen by its
// algorithm selection (ring or tree, by size and topology), which would tie the answer to
// NCCL's tuning. A second, optional dependency buys nothing measurable here; the reasoning
// is recorded in docs/PROVENANCE.md (judgement call 28).
#pragma once

#include <cuda_runtime.h>

#include <string>
#include <utility>
#include <vector>

#include "pdhg_multi_gpu_device.hpp"

namespace sankhya::gpu::multi {

enum class Transport { kPeer, kHostStaged };

[[nodiscard]] const char* to_string(Transport t);

class PartialSumExchange {
 public:
  PartialSumExchange() = default;
  PartialSumExchange(const PartialSumExchange&) = delete;
  PartialSumExchange& operator=(const PartialSumExchange&) = delete;
  ~PartialSumExchange();

  /// cards are in slot order and already set up. Tries peer access on every pair of distinct
  /// physical devices when allow_peer; falls back to host staging when any pair lacks it.
  /// why receives a one-line reason for the transport chosen. Returns false on a CUDA error.
  [[nodiscard]] bool init(const std::vector<DeviceState*>& cards, bool allow_peer,
                          std::string* why);

  [[nodiscard]] Transport transport() const { return transport_; }

  /// Enqueue on every card: d_aty = sum over slots 0..K-1 of the cards' d_partial, in slot
  /// order. Every card's spmv_aty must already be enqueued.
  [[nodiscard]] bool enqueue();

 private:
  std::vector<DeviceState*> cards_;
  Transport transport_ = Transport::kHostStaged;
  std::vector<cudaEvent_t> ready_;   // per card: partial computed
  std::vector<cudaEvent_t> staged_;  // per card: partial copied to the host (staged only)
  double* host_stage_ = nullptr;     // K * n pinned doubles (staged only)
  std::vector<std::pair<int, int>> enabled_;  // peer access this object turned on
};

}  // namespace sankhya::gpu::multi
