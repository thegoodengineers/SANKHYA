#!/bin/bash
set +e
one() { # label commit model
  bin=.b/$2/build/sankhya; log="$OUT/log-$1-$3.txt"; start=$(date +%s.%N)
  timeout 200 "$bin" solve data/stress/$3.mps --time-limit 60 --write-sol "$OUT/$1-$3.sol" --option log_level=debug > "$log" 2>&1; code=$?
  echo "RUN $1 $3 exit $code wall $(echo "$(date +%s.%N) - $start" | bc)"
  grep -E "^Result|^message" "$log" | cut -c1-300 | sed "s/^/RUN $1 $3 | /"
  echo "RUN $1 $3 | phase-1 completions $(grep -c 'Phase 1 complete' "$log"), refactorized $(grep -c '^refactorized at' "$log")"
  [ -f "$OUT/$1-$3.sol" ] && python3 tools/verify_solution.py data/stress/$3.mps "$OUT/$1-$3.sol" 2>&1 | grep -E "VERIFIED|REJECTED" | sed "s/^/RUN $1 $3 | verifier /"
}
one V0 5c0df2dac2f02adefa7263289065c94c4ed82a4b scaled_degen3
one V1 ea9a56a8bc86095c6552d4b83546a13d28622d5d scaled_degen3
one V0 5c0df2dac2f02adefa7263289065c94c4ed82a4b scaled_pilot.ja
one V1 ea9a56a8bc86095c6552d4b83546a13d28622d5d scaled_pilot.ja
python3 bench/runners/stress.py --binary .b/331f646d9dc8f2d4152121e7e969981fbd3baa68/build/sankhya --machine scratch --solvers sankhya --out "$OUT/results/stress-R.csv" 2>&1 | grep -E "^(scaled_|sankhya )" | grep -v " correct " | sed "s/^/STRESS-R /"
python3 bench/runners/stress.py --binary .b/ea9a56a8bc86095c6552d4b83546a13d28622d5d/build/sankhya --machine scratch --solvers sankhya --out "$OUT/results/stress-V1.csv" 2>&1 | grep -E "^(scaled_|sankhya )" | grep -v " correct " | sed "s/^/STRESS-V1 /"
