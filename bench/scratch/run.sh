#!/bin/bash
set +e
bin=.b/5c77d3e849b343be9186eb34a2cf9b38198881dc/build/sankhya
for m in scaled_degen3; do
  log="$OUT/log-D-$m.txt"; start=$(date +%s.%N)
  timeout 200 "$bin" solve data/stress/$m.mps --time-limit 60 --write-sol "$OUT/D-$m.sol" --option log_level=debug > "$log" 2>&1; code=$?
  echo "RUN D $m exit $code wall $(echo "$(date +%s.%N) - $start" | bc)"
  grep -E "^Result|^message" "$log" | cut -c1-300 | sed "s/^/RUN D $m | /"
  echo "RUN D $m | phase-1 completions $(grep -c 'Phase 1 complete' "$log"), refactorized $(grep -c '^refactorized at' "$log"), re-priced $(grep -c 'through its own column' "$log")"
  [ -f "$OUT/D-$m.sol" ] && python3 tools/verify_solution.py data/stress/$m.mps "$OUT/D-$m.sol" 2>&1 | grep -E "VERIFIED|REJECTED" | sed "s/^/RUN D $m | verifier /"
done
python3 bench/runners/stress.py --binary "$bin" --machine scratch --solvers sankhya --out "$OUT/results/stress-D.csv" 2>&1 | grep -E "^(scaled_|sankhya )" | grep -v " correct " | sed "s/^/STRESS-D /"
bash bench/scratch/stats.sh
