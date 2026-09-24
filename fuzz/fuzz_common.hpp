// SPDX-License-Identifier: Apache-2.0
// SANKHYA - shared plumbing for the reader fuzz targets (#534).
//
// read_mps() and read_lp() take a path, not a buffer (include/sankhya/io.hpp: readers report
// "path:line: message", which wants a real file for the CLI and Python callers this API
// otherwise serves). A fuzz target gets a byte buffer from libFuzzer, so every target here
// writes that buffer to a fresh temporary file, calls the real reader on it exactly the way
// the CLI would, and removes it - the same reader every model in this project goes through,
// with no separate buffer-parsing path that could hide a bug the real one has.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace sankhya::fuzz {

// A path a concurrent run of the same target cannot collide on: libFuzzer forks worker
// processes, and mkstemp's O_EXCL guarantees a distinct file per call even when several
// workers race this in the same directory.
class ScratchFile {
 public:
  explicit ScratchFile(const uint8_t* data, size_t size) {
    char pattern[] = "/tmp/sankhya_fuzz_XXXXXX";
    const int fd = mkstemp(pattern);
    if (fd < 0) return;
    path_ = pattern;  // only once mkstemp actually created this file
    size_t written = 0;
    bool write_failed = false;
    while (written < size) {
      const ssize_t n = write(fd, data + written, size - written);
      if (n <= 0) {
        write_failed = true;
        break;
      }
      written += static_cast<size_t>(n);
    }
    close(fd);
    ok_ = !write_failed;
  }

  ~ScratchFile() {
    if (!path_.empty()) std::remove(path_.c_str());
  }

  ScratchFile(const ScratchFile&) = delete;
  ScratchFile& operator=(const ScratchFile&) = delete;

  [[nodiscard]] const std::string& path() const { return path_; }
  [[nodiscard]] bool ok() const { return ok_; }

 private:
  std::string path_;
  bool ok_ = false;
};

}  // namespace sankhya::fuzz
