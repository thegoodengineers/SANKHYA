#!/usr/bin/env python3
"""Generate TRUSS and STOCFOR3, the two Netlib LPs netlib.org ships as generator bundles.

    python bench/runners/generate_netlib_bundles.py            # writes data/netlib/{truss,stocfor3}.mps
    python bench/runners/generate_netlib_bundles.py --check    # regenerate and compare with the tracked files

`lp/data/truss` and `lp/data/stocfor3` are not EMPS files but shell bundles of Fortran
generators and their data, which is why `emps` rejects them and why #742 first listed them as
not shipped (#745):

- truss: Michael Ferris's `truss.f` reads `loads.dat`, `nodepos.dat` and `numfrnds.dat` and
  writes the MPS file `mps`.
- stocfor3: Gus Gassmann's `std2mps.f` + `input.f` read the SMPS time file on unit 1, the
  EMPS-expanded core file on unit 2 and `stoch3.frs` on unit 3, and write the deterministic
  equivalent in five pieces on units 11 to 15, which are concatenated in that order. The 1987
  source passes an INTEGER*2 to IABS, which gfortran now rejects; the generic ABS computes
  the same value for every integer kind, so IABS( is rewritten to ABS( and nothing else.

Each generated model is checked against the readme's summary table (rows including the
objective, columns, nonzero coefficients) before it is written. STOCFOR3's generator also
writes 3,752 coefficients that are exactly zero; the table counts nonzeros, so the check does
too. The generated files are committed, so CI and fetch_data.py need no Fortran compiler;
fetch_data.py pins them by sha256 (GENERATED). Nothing here is linked into SANKHYA: these are
benchmark instances produced by Netlib's own generators.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_data  # noqa: E402

GFORTRAN_CANDIDATES = ["gfortran", r"C:\msys64\ucrt64\bin\gfortran.exe"]
# A freshly linked executable can be refused by Windows Smart App Control, one link at a time,
# so a few equivalent builds are tried in turn; each is the same program.
BUILD_VARIANTS = [["-O1", "-static"], ["-O0", "-static"], ["-O2", "-static"], ["-O1"], ["-Os", "-static"]]


def unbundle(text: str, into: Path) -> None:
    """Netlib's `sh` bundle: `sed >NAME <<'//GO.SYSIN DD NAME' 's/^-//'`, then '-'-prefixed lines."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("sed >") and "<<" in line:
            name = line.split(">", 1)[1].split()[0]
            end = f"//GO.SYSIN DD {name}"
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != end:
                body.append(lines[i][1:] if lines[i].startswith("-") else lines[i])
                i += 1
            (into / name).write_text("\n".join(body) + "\n", encoding="latin-1")
        i += 1


def build(sources: list[Path], work: Path) -> Path:
    compiler = next((c for c in GFORTRAN_CANDIDATES if shutil.which(c) or Path(c).exists()), None)
    if compiler is None:
        raise SystemExit("gfortran not found (MSYS2: pacman -S mingw-w64-ucrt-x86_64-gcc-fortran)")
    env = dict(os.environ)
    env["PATH"] = str(Path(shutil.which(compiler) or compiler).parent) + os.pathsep + env["PATH"]
    for k, flags in enumerate(BUILD_VARIANTS):
        exe = work / f"gen{k}.exe"
        subprocess.run([compiler, "-std=legacy", "-w", *flags, *map(str, sources), "-o", str(exe)],
                       cwd=work, env=env, check=True)
        try:
            subprocess.run([str(exe)], cwd=work, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=60)
            return exe
        except OSError as error:  # blocked by the OS; try the next link
            print(f"  build {k} refused ({error}); relinking", file=sys.stderr)
    raise SystemExit("every build of the generator was refused by the OS")


def dimensions(mps: Path) -> tuple[int, int, int]:
    """Rows including the objective, columns, nonzero coefficients - the readme's three counts."""
    section, rows, cols, nonzeros = None, 0, set(), 0
    for line in mps.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            section = line.split()[0]
            continue
        fields = line.split()
        if section == "ROWS":
            rows += 1
        elif section == "COLUMNS":
            cols.add(fields[0])
            nonzeros += sum(float(fields[i + 1]) != 0.0 for i in range(1, len(fields) - 1, 2))
    return rows, len(cols), nonzeros


def generate_truss(bundle: str, work: Path) -> Path:
    unbundle(bundle, work)
    exe = build([work / "truss.f"], work)
    (work / "mps").unlink(missing_ok=True)
    subprocess.run([str(exe)], cwd=work, check=True, stdin=subprocess.DEVNULL, capture_output=True)
    return work / "mps"


def generate_stocfor3(bundle: str, work: Path, emps: Path) -> Path:
    unbundle(bundle, work)
    for name in ("std2mps.f", "input.f"):
        path = work / name
        path.write_text(path.read_text(encoding="latin-1").replace("IABS(", "ABS("), encoding="latin-1")
    exe = build([work / "std2mps.f", work / "input.f"], work)
    for stale in work.glob("fort.*"):
        stale.unlink()
    fetch_data.decompress(emps, work / "core.mpc", work / "fort.2")
    shutil.copy(work / "time7.frs", work / "fort.1")
    shutil.copy(work / "stoch3.frs", work / "fort.3")
    subprocess.run([str(exe)], cwd=work, check=True, stdin=subprocess.DEVNULL, capture_output=True)
    out = work / "stocfor3.mps"
    out.write_bytes(b"".join((work / f"fort.{u}").read_bytes() for u in range(11, 16)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare with the tracked files, writing nothing")
    args = parser.parse_args()
    published = fetch_data.parse_summary_table(
        (fetch_data.DATA_DIR / "readme").read_text(encoding="latin-1"))
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        emps, _ = fetch_data.build_emps(root)
        for name in ("truss", "stocfor3"):
            work = root / name
            work.mkdir()
            bundle = fetch_data.download(f"{fetch_data.NETLIB_BASE}/{name}").decode("latin-1")
            made = generate_truss(bundle, work) if name == "truss" else generate_stocfor3(bundle, work, emps)
            entry = published[name]
            want = (entry["published_rows"], entry["published_cols"], entry["published_nonzeros"])
            got = dimensions(made)
            digest = fetch_data.sha256(made.read_bytes())
            ok = got == want
            print(f"  {name:<9} rows/cols/nonzeros {got} readme {want} {'ok' if ok else 'MISMATCH'}  sha256 {digest}")
            if not ok:
                failures += 1
                continue
            target = fetch_data.DATA_DIR / f"{name}.mps"
            if args.check:
                same = target.exists() and fetch_data.sha256(target.read_bytes()) == digest
                print(f"  {name:<9} tracked copy {'identical' if same else 'DIFFERS'}")
                failures += not same
            else:
                shutil.copy(made, target)
                print(f"  {name:<9} wrote {target.relative_to(fetch_data.REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
