// SPDX-License-Identifier: Apache-2.0
// SANKHYA - libFuzzer target for the free-format MPS dialect (#534).
//
// Seeded from demo/*.mps and the Netlib set's free-format members via
// fuzz/corpus/mps_free/. See fuzz_mps_fixed.cpp for why a temp file, and the corpus policy.

#include <cstdint>

#include "fuzz_common.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  sankhya::fuzz::ScratchFile file(data, size);
  if (!file.ok()) return 0;

  sankhya::Model model;
  (void)sankhya::io::read_mps(file.path(), &model, sankhya::io::MpsFormat::kFree);
  return 0;
}
