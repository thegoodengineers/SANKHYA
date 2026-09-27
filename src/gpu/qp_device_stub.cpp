// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the QP device operator's factory in a build without CUDA (#493): declines with
// a reason, so src/qp/qp_condat_vu.cpp keeps the host operator and nothing in the CPU build
// includes a CUDA header. The CUDA definition is in qp_device.cu.

#include "qp_device.hpp"

#ifndef SANKHYA_ENABLE_CUDA

namespace sankhya::gpu {

std::unique_ptr<qp::QpOperator> make_qp_device_operator(const Model& /*model*/,
                                                        std::string* reason) {
  if (reason != nullptr) {
    *reason = "this build has no CUDA backend (configure with -DSANKHYA_ENABLE_CUDA=ON)";
  }
  return nullptr;
}

}  // namespace sankhya::gpu

#endif  // !SANKHYA_ENABLE_CUDA
