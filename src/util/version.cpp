// SPDX-License-Identifier: Apache-2.0
// SANKHYA - build identification. The macros are supplied by CMake.

#include "sankhya/version.hpp"

#include <string>

#include <fmt/format.h>

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#endif

#ifndef SANKHYA_VERSION
#define SANKHYA_VERSION "0.0.0-unconfigured"
#endif
#ifndef SANKHYA_GIT_COMMIT
#define SANKHYA_GIT_COMMIT "unknown"
#endif
#ifndef SANKHYA_BUILD_TYPE
#define SANKHYA_BUILD_TYPE "unknown"
#endif
#ifndef SANKHYA_COMPILER
#define SANKHYA_COMPILER "unknown"
#endif

namespace sankhya {

const char* version_string() noexcept {
  return SANKHYA_VERSION;
}
const char* git_commit() noexcept {
  return SANKHYA_GIT_COMMIT;
}
const char* build_type() noexcept {
  return SANKHYA_BUILD_TYPE;
}
const char* compiler_string() noexcept {
  return SANKHYA_COMPILER;
}

bool cuda_enabled() noexcept {
#ifdef SANKHYA_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

const char* cuda_device_description() noexcept {
#ifdef SANKHYA_ENABLE_CUDA
  static const std::string desc = []() -> std::string {
    std::string d;
    return gpu::device_available(&d) ? d : std::string("no device: ") + d;
  }();
  return desc.c_str();
#else
  return "none (CUDA not compiled in)";
#endif
}

const char* repository() noexcept {
  return "thegoodengineers/SANKHYA";
}
const char* repository_url() noexcept {
  return "https://github.com/thegoodengineers/SANKHYA";
}

const char* banner() noexcept {
  static const std::string text = fmt::format(
      "SANKHYA {} ({}, {}, {}, CUDA {}, GPU {})", version_string(), git_commit(), build_type(),
      compiler_string(), cuda_enabled() ? "on" : "off", cuda_device_description());
  return text.c_str();
}

}  // namespace sankhya
