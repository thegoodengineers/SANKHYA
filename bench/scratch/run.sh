#!/bin/bash
# Scratch: where main's solve of scaled_maros-r7 sits past its time limit.
set +e
c=aa5ff25815709176d46117d2a80c01967d337114
bin=.b/$c/build/sankhya
log="$OUT/log-maros-r7.txt"
stdbuf -o0 -e0 "$bin" solve data/stress/scaled_maros-r7.mps --time-limit 60 --option log_level=debug > "$log" 2>&1 &
pid=$!
for t in 70 100 130 160; do
  while [ "$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')" != "" ] && [ "$(ps -o etimes= -p $pid | tr -d ' ')" -lt $t ]; do sleep 2; done
  kill -0 $pid 2>/dev/null || break
  echo "TRACE at ${t}s"
  sudo gdb -p $pid -batch -ex "bt 25" 2>/dev/null | grep -E "^#" | cut -c1-200 | sed "s/^/TRACE /"
done
kill -9 $pid 2>/dev/null
echo "TRACE log lines $(wc -l < "$log")"
grep -v "^ipm iteration\|^ *[0-9]\+ \+[-0-9.e+]\+ \|^$\|^-----\|^ Iter\|^  R[0-9]" "$log" | tail -40 | cut -c1-260 | sed "s/^/TAIL /"
