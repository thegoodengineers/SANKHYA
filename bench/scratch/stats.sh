#!/bin/bash
# Scratch: per attempt, how often the primal loop's update is declined and phase 1 is
# completed, over Netlib, Kennington and the stress set, on main.
set +e
c=aa5ff25815709176d46117d2a80c01967d337114
bin=.b/$c/build/sankhya
python3 bench/runners/fetch_kennington.py > /dev/null 2>&1
one() { # set file limit
  name=$(basename "$2"); name=${name%.gz}; name=${name%.mps}
  timeout $(( $3 * 3 + 30 )) "$bin" solve "$2" --time-limit $3 --option log_level=debug 2>&1 | python3 -c '
import re, sys
p1 = 0; attempts = []; status = "?"; secs = "?"
for line in sys.stdin:
    if line.startswith("Phase 1 complete"): p1 += 1
    m = re.match(r"Basis \((.*?)\): (\d+) refactorizations over (\d+) iterations, (\d+) declined as unsafe, (\d+) forced", line)
    if m:
        attempts.append((int(m.group(3)), int(m.group(2)), int(m.group(4)), int(m.group(5)), p1)); p1 = 0
    m = re.match(r"Result: (\S+) .* (\S+)s$", line.strip())
    if m: status, secs = m.group(1), m.group(2)
print("STAT", sys.argv[1], sys.argv[2], status, secs, "|", " ; ".join("it %d rf %d declined %d forced %d p1 %d" % a for a in attempts))
' "$1" "$name"
}
for f in data/netlib/*.mps; do one netlib "$f" 60; done
for f in data/kennington/*.mps data/kennington/*.mps.gz; do [ -f "$f" ] && one kennington "$f" 60; done
for f in data/stress/scaled_*.mps; do one stress "$f" 60; done
