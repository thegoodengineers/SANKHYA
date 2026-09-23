// SPDX-License-Identifier: Apache-2.0
// SANKHYA - banner format tests (issue #354).
//
// Pins the GPU field of the one-line banner so that parsers (bench runners, CI scripts)
// are not silently broken by a format change.  The CPU build always produces the fixed
// string "GPU none (CUDA not compiled in)"; the CUDA build produces "GPU <device-string>".

#include <string_view>

#include <gtest/gtest.h>

#include "sankhya/version.hpp"

namespace sankhya {
namespace {

TEST(Banner, GpuFieldPresent) {
  const std::string_view b = banner();
  EXPECT_NE(b.find("GPU "), std::string_view::npos) << "banner: " << b;
}

#ifndef SANKHYA_ENABLE_CUDA
TEST(Banner, CpuBuildGpuField) {
  const std::string_view b = banner();
  EXPECT_NE(b.find("GPU none (CUDA not compiled in)"), std::string_view::npos)
      << "banner: " << b;
}
#endif

// The canonical identity (#538), pinned so a refactor cannot quietly drop or change it.
TEST(Banner, CanonicalRepository) {
  EXPECT_STREQ(repository(), "thegoodengineers/SANKHYA");
  EXPECT_STREQ(repository_url(), "https://github.com/thegoodengineers/SANKHYA");
}

}  // namespace
}  // namespace sankhya
