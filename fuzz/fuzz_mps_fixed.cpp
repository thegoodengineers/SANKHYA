// SPDX-License-Identifier: Apache-2.0
// SANKHYA - libFuzzer target for the fixed-column MPS dialect (#534).
//
// Seeded from Netlib and MIPLIB instances (bench/runners/fetch_data.py's corpus, all
// fixed-format historically) via fuzz/corpus/mps_fixed/. A crash here becomes a regression
// test: the minimized reproducer is checked in under fuzz/corpus/mps_fixed/ and this target
// then covers it on every run, the same way libFuzzer's own corpus accumulation works.

#include <cstdint>

#include "fuzz_common.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  sankhya::fuzz::ScratchFile file(data, size);
  if (!file.ok()) return 0;

  sankhya::Model model;
  // ok() == false is an ordinary parse failure (ReadResult::error), not a bug: the harness
  // exists to catch a crash or a sanitizer report, not to require every random buffer to be
  // a valid model.
  (void)sankhya::io::read_mps(file.path(), &model, sankhya::io::MpsFormat::kFixed);
  return 0;
}
