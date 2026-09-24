// SPDX-License-Identifier: Apache-2.0
// SANKHYA - libFuzzer target for the CPLEX LP reader (#534).
//
// Seeded from demo/*.lp via fuzz/corpus/lp/. See fuzz_mps_fixed.cpp for why a temp file, and
// the corpus policy.

#include <cstdint>

#include "fuzz_common.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  sankhya::fuzz::ScratchFile file(data, size);
  if (!file.ok()) return 0;

  sankhya::Model model;
  (void)sankhya::io::read_lp(file.path(), &model);
  return 0;
}
