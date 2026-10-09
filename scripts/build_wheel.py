# SPDX-License-Identifier: Apache-2.0
"""Build the Python wheel from a finished build (#748).

    python scripts/build_wheel.py BUILD_DIR PLATFORM_TAG OUT_DIR [EXTRA_LIB ...]

The bindings are ctypes over the C API (bindings/python/README.md), so the wheel holds no
compiled extension: it is bindings/python/sankhya with the shared library, the CLI and their
runtime libraries (EXTRA_LIB: libgomp on Linux, the MinGW DLLs on Windows) copied into the
package directory, where _library.py and _executable.py look first. Nothing in it depends on
the Python version, so ONE wheel per platform, tagged py3-none-<platform>, serves every
CPython from 3.9 on; CI installs it on each of 3.9 to 3.13 and solves and verifies there.

The wheel is written directly as the zip PEP 427 describes, with the standard library, so
building it needs no build backend and the tag is exactly the one given.
"""

from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(build: Path, platform_tag: str, out: Path, extras: list[Path]) -> Path:
    exe = build / ("sankhya.exe" if (build / "sankhya.exe").exists() else "sankhya")
    banner = subprocess.run([str(exe), "version"], capture_output=True, text=True,
                            check=True).stdout.splitlines()[0]
    match = re.match(r"SANKHYA (\S+) ", banner)
    if not match:
        raise SystemExit(f"cannot read a version from: {banner}")
    version = match.group(1)

    # (name inside the wheel, source file). Libraries are copied with symlinks followed, and
    # the Linux library under its bare name, which is the one _library.py asks for.
    files: list[tuple[str, Path]] = []
    package = REPO / "bindings" / "python" / "sankhya"
    for source in sorted(package.rglob("*.py")):
        files.append(("sankhya/" + source.relative_to(package).as_posix(), source))
    for name in ("libsankhya.dll", "libsankhya.so", "libsankhya.dylib"):
        if (build / name).exists():
            files.append((f"sankhya/{name}", (build / name).resolve()))
    files.append((f"sankhya/{exe.name}", exe))
    files += [(f"sankhya/{extra.name}", extra.resolve()) for extra in extras]

    dist_info = f"sankhya-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.1\nName: sankhya\nVersion: " + version + "\n"
        "Summary: LP, MILP and convex QP solver written from scratch (SIH 2026, PS26119)\n"
        "Home-page: https://github.com/thegoodengineers/SANKHYA\n"
        "License: Apache-2.0\nRequires-Python: >=3.9\n"
    )
    wheel = ("Wheel-Version: 1.0\nGenerator: scripts/build_wheel.py\n"
             "Root-Is-Purelib: false\nTag: py3-none-" + platform_tag + "\n")

    out.mkdir(parents=True, exist_ok=True)
    target = out / f"sankhya-{version}-py3-none-{platform_tag}.whl"
    record: list[str] = []

    def add(archive: zipfile.ZipFile, name: str, data: bytes, executable: bool) -> None:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o755 if executable else 0o644) << 16
        archive.writestr(info, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")

    with zipfile.ZipFile(target, "w") as archive:
        for name, source in files:
            add(archive, name, source.read_bytes(), not name.endswith(".py"))
        add(archive, f"{dist_info}/METADATA", metadata.encode(), False)
        add(archive, f"{dist_info}/WHEEL", wheel.encode(), False)
        record.append(f"{dist_info}/RECORD,,")
        archive.writestr(zipfile.ZipInfo(f"{dist_info}/RECORD", date_time=(1980, 1, 1, 0, 0, 0)),
                         "\n".join(record) + "\n")
    print(target)
    return target


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), [Path(a) for a in sys.argv[4:]])
