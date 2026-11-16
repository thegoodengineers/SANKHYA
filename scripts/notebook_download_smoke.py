# SPDX-License-Identifier: Apache-2.0
"""Run the walkthrough notebook's download cell on its own, then use what it unpacked (#748).

    python scripts/notebook_download_smoke.py NOTEBOOK

Run from an EMPTY directory, so the cell finds no solver and takes option A: it must pick the
newest published release with an archive for this platform, match the archive against that
release's SHA256SUMS and unpack it. The unpacked CLI then solves the crude blend demo and the
unpacked tools/verify_solution.py must accept the .sol. This is what a reader with no compiler
gets from the notebook's first cell, against the releases as they are published today.
"""

import json
import subprocess
import sys
from pathlib import Path

cells = [c for c in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["cells"]
         if c["cell_type"] == "code"]
assert not any(Path.cwd().iterdir()), "run from an empty directory"
scope = {}
exec("".join(cells[0]["source"]), scope)  # the download cell, exactly as the notebook has it

root = scope["RELEASE_ROOT"]
assert root is not None, "the download cell unpacked nothing"
exe = root / "build" / scope["EXE"]
mps = root / "demo" / "crude_blend.mps"
sol = Path.cwd() / "blend.sol"
subprocess.run([str(exe), "version"], check=True)
subprocess.run([str(exe), "solve", str(mps), "--write-sol", str(sol)], check=True)
subprocess.run([sys.executable, str(root / "tools" / "verify_solution.py"), str(mps), str(sol)],
               check=True)
print("notebook download smoke: verified from", root.name)
