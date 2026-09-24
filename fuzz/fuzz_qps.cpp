// SPDX-License-Identifier: Apache-2.0
// SANKHYA - libFuzzer target for QPS (MPS's QUADOBJ/QMATRIX/QSECTION extension, #534).
//
// This is the same read_mps() the plain LP targets call - QPS is not a separate reader
// (src/io/mps_reader.cpp parses QUADOBJ inline) - but a mutation corpus seeded from plain LP
// files would need to get lucky to ever synthesize a QUADOBJ section, and Hessian parsing
// (src/io/mps_reader.cpp's column-pair-and-value records, which never appeared in COLUMNS)
// is exactly the kind of new field format most likely to have its own bugs. A target seeded
// from demo/*_qp.mps and demo/miqp_blend.mps via fuzz/corpus/qps/ spends its budget there
// instead of diluting it across every plain-LP mutation the mps_free target already covers.

#include <cstdint>

#include "fuzz_common.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  sankhya::fuzz::ScratchFile file(data, size);
  if (!file.ok()) return 0;

  sankhya::Model model;
  (void)sankhya::io::read_mps(file.path(), &model, sankhya::io::MpsFormat::kAuto);
  return 0;
}
