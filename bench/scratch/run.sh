#!/bin/bash
# Scratch: the two candidate fixes on the rows they are for.
set +e
run() { # commit model cap
  bin=.b/$1/build/sankhya; log="$OUT/log-$1-$2.txt"; start=$(date +%s.%N)
  timeout $3 "$bin" solve data/stress/$2.mps --time-limit 60 --write-sol "$OUT/$1-$2.sol" --option log_level=debug > "$log" 2>&1; code=$?
  end=$(date +%s.%N)
  echo "RUN $1 $2 exit $code wall $(echo "$end - $start" | bc)"
  grep -E "^Result|^Safe bound|^message|^Prescaled|^Retry|exact repair|Phase 1 complete" "$log" | awk '{k=$0; gsub(/[0-9]+/,"#",k); if(c[substr(k,1,40)]++<2) print}' | cut -c1-300 | sed "s/^/RUN $1 $2 | /"
  echo "RUN $1 $2 | phase-1 completions $(grep -c 'Phase 1 complete' "$log"), refactorized $(grep -c '^refactorized at' "$log")"
  [ -f "$OUT/$1-$2.sol" ] && python3 tools/verify_solution.py data/stress/$2.mps "$OUT/$1-$2.sol" 2>&1 | grep -E "VERIFIED|REJECTED|FAIL" | cut -c1-200 | sed "s/^/RUN $1 $2 | verifier /"
}
run 9a98ddd1406e0c9d7185e124d9e8af1c5cd795d0 scaled_degen3 200
run ddfb09a6772c3b8f35213dacdf09a0e74a9fbb49 scaled_maros-r7 250
python3 bench/runners/stress.py --binary .b/9a98ddd1406e0c9d7185e124d9e8af1c5cd795d0/build/sankhya --machine scratch --solvers sankhya --out "$OUT/results/stress-A.csv" 2>&1 | grep -E "^(scaled_|sankhya )" | grep -v " correct " | sed "s/^/STRESS-A /"
python3 bench/runners/stress.py --binary .b/9a98ddd1406e0c9d7185e124d9e8af1c5cd795d0/build/sankhya --machine scratch --solvers sankhya --out "$OUT/results/stress-A2.csv" --instances none 2>&1 | tail -3 | sed "s/^/STRESS-A /"
