#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage one release archive from a finished build (#748).
#
#     scripts/package_release.sh BUILD_DIR FLAVOR OUT_DIR [EXTRA_LIB ...]
#
# FLAVOR names the archive (linux-x86_64-cpu, windows-x64-cpu, linux-x86_64-cuda). EXTRA_LIB
# are runtime libraries copied beside the binary (libgomp on Linux, the MinGW DLLs the shared
# library imports on Windows), so the archive runs on a machine with no compiler.
#
# The archive keeps the repository's layout - the binary and library in build/, the Python
# package in bindings/python, the verifier in tools/ - because every script and the bindings
# already look there: demo/run_demo.sh, the notebook and `import sankhya` work from the
# unpacked directory exactly as from a checkout, with nothing to configure.
#
# MANIFEST.txt ties the binary to its source: what `sankhya version` prints (version, commit,
# build type, compiler, CUDA), the link dependencies of the binary and library AS SHIPPED,
# and the sha256 of every file. The provenance check (docs/PROVENANCE.md: no solver library
# linked) runs here on the shipped files, not only on the CI build, and fails the packaging.
set -euo pipefail

BUILD="$1"
FLAVOR="$2"
OUT="$3"
shift 3

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE=sankhya
[ -f "$BUILD/sankhya.exe" ] && EXE=sankhya.exe
VERSION="$("$BUILD/$EXE" version | head -1)"
SEMVER="$(printf '%s' "$VERSION" | sed -n 's/^SANKHYA \([^ ]*\) .*/\1/p')"
NAME="sankhya-${SEMVER}-${FLAVOR}"
STAGE="$OUT/$NAME"

rm -rf "$STAGE"
mkdir -p "$STAGE/build" "$STAGE/bench/runners" "$STAGE/docs" "$STAGE/scripts"
cp "$BUILD/$EXE" "$STAGE/build/"
for lib in "$BUILD"/libsankhya.so* "$BUILD"/libsankhya.dll "$BUILD"/libsankhya.dylib; do
  [ -e "$lib" ] && cp -P "$lib" "$STAGE/build/"
done
for extra in "$@"; do cp -L "$extra" "$STAGE/build/"; done

cp -r "$REPO/include" "$REPO/tools" "$REPO/demo" "$REPO/notebooks" "$STAGE/"
mkdir -p "$STAGE/bindings" "$STAGE/bench/case_studies"
cp -r "$REPO/bindings/python" "$STAGE/bindings/"
cp -r "$REPO/bench/case_studies/refinery" "$STAGE/bench/case_studies/"
cp "$REPO/bench/runners/stamp.py" "$STAGE/bench/runners/"
cp "$REPO/scripts/binary_provenance.sh" "$STAGE/scripts/"
cp "$REPO/README.md" "$STAGE/"
cp "$REPO/docs/PROVENANCE.md" "$REPO/docs/sbom.spdx.json" "$STAGE/docs/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Link dependencies of what ships. ldd exists on Linux and in MSYS2; objdump is the fallback.
deps() {
  if command -v ldd >/dev/null 2>&1; then ldd "$1" || true
  else objdump -p "$1" | grep 'DLL Name' || true; fi
}

{
  echo "$VERSION"
  echo "flavor      $FLAVOR"
  echo "packaged    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  command -v nvcc >/dev/null 2>&1 && echo "nvcc        $(nvcc --version | tail -1)"
  echo
  for f in "$STAGE/build/$EXE" "$STAGE"/build/libsankhya.so "$STAGE"/build/libsankhya.dll; do
    [ -e "$f" ] || continue
    echo "=== link dependencies of build/$(basename "$f") ==="
    deps "$f"
    echo
  done
} > "$STAGE/MANIFEST.txt"

# The red line, checked on the shipped files: library names only (first field), because a
# load address can spell 'cbc' in hex.
if grep -E '^\s' "$STAGE/MANIFEST.txt" | awk '{print $1, $3}' |
   grep -Ei 'cbc|clp|highs|scip|soplex|glpk|lpsolve|osqp|ortools|gurobi|cplex|xpress'; then
  echo "FAIL: a solver library is linked into the release binaries" >&2
  exit 1
fi
echo "provenance: no solver library among the shipped binaries' dependencies"

( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum ) \
  > "$STAGE/SHA256SUMS"

cd "$OUT"
case "$FLAVOR" in
  windows*) (command -v zip >/dev/null && zip -qr "$NAME.zip" "$NAME") ||
            powershell -NoProfile -Command "Compress-Archive -Path '$NAME' -DestinationPath '$NAME.zip'"
            sha256sum "$NAME.zip" > "$NAME.zip.sha256" ;;
  *)        tar -czf "$NAME.tar.gz" "$NAME"
            sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" ;;
esac
echo "packaged $OUT/$NAME"
