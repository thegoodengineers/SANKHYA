# SPDX-License-Identifier: Apache-2.0
"""Check an INSTALLED wheel solves and verifies the crude blend demo (#748).

    <venv python> scripts/wheel_smoke.py REPO_DIR

Run with the interpreter of a clean venv the wheel was pip-installed into, from a directory
outside the checkout, so `import sankhya` can only find the installed package. The CLI the
wheel carries writes the .sol, tools/verify_solution.py (pure Python, links nothing of ours)
checks it, and the bindings' own objective must equal the verified one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import sankhya

repo = Path(sys.argv[1]).resolve()
mps = repo / "demo" / "crude_blend.mps"
package = Path(sankhya.__file__).resolve().parent
assert repo not in package.parents, f"imported the checkout, not the wheel: {package}"
print("sankhya", sankhya.version(), "from", package)

result = sankhya.Model.read(str(mps)).solve()
assert result.optimal, result.status
print("bindings objective", result.objective)

exe = sankhya.locate_executable()
assert exe.parent == package, f"CLI not from the wheel: {exe}"
with tempfile.TemporaryDirectory() as tmp:
    sol = Path(tmp) / "blend.sol"
    subprocess.run([str(exe), "solve", str(mps), "--write-sol", str(sol)], check=True)
    subprocess.run([sys.executable, str(repo / "tools" / "verify_solution.py"), str(mps), str(sol)],
                   check=True)
    objective = next(float(line.split()[-1]) for line in sol.read_text().splitlines()
                     if line.split()[:1] == ["objective"])
assert abs(objective - result.objective) <= 1e-9 * max(1.0, abs(objective)), (objective,
                                                                             result.objective)
print("wheel smoke: verified, objective", objective)
