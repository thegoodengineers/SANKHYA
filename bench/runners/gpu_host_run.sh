#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - the #888 / #759 GPU evidence on a rented card, from a clone, legs resumable.
#
#   bench/runners/gpu_host_run.sh HOST [REF] [ARCH]      HOST = root@ip, REF = branch/tag/sha
#                                                        (default main), ARCH = sm (default 80)
#   bench/runners/gpu_host_run.sh HOST REF ARCH fetch    copy every CSV the host has back here
#
# What it does on the host (all under $WORK, nothing else touched): a Python venv with cmake,
# ninja, highspy and cuopt-cu12 (reused when present), a clone of REF built with
# -DSANKHYA_ENABLE_CUDA=ON, the instances, then the legs in priority order, each started
# with setsid nohup, logging to $WORK/logs/, each runner appending and flushing its CSV per
# row so a dropped connection loses only the solve in flight:
#   1. gpu_refinery_year.py   refinery year: GPU PDHG -> crossover -> verify; HiGHS 1 & N threads
#   2. mittelmann_cuopt.py    cuOpt vs our GPU PDHG, LPfeas-14, 300 s, 1e-4 then 1e-8
#   3. gpu_refinery_year.py   the million-row refinery, only after the two above
# The CPU thread count is the cgroup quota, not nproc (a container says 256 and means 14).
# Every runner leg runs under `|| true`: a runner exits non-zero when it has failures to
# report, and that must not end the script (the 23 Sep 2026 trap).
set -u
HOST=${1:?usage: gpu_host_run.sh HOST [REF] [ARCH] [fetch]}
REF=${2:-main}
ARCH=${3:-80}
MODE=${4:-run}
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_e2e}
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 $HOST"
WORK=${WORK:-/home/jovyan/sankhya-888}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "$MODE" = fetch ]; then
  mkdir -p "$HERE/bench/results"
  for f in $($SSH "ls $WORK/src/bench/results/*.csv 2>/dev/null | xargs -n1 basename"); do
    case "$f" in gpu-refinery-year-*|lpfeas14-cuopt-*)
      $SSH "cat $WORK/src/bench/results/$f" > "$HERE/bench/results/$f" && echo "fetched $f" ;;
    esac
  done
  exit 0
fi

# The remote script, written on the host and run detached; its stdout is the leg log.
$SSH "mkdir -p $WORK/logs && cat > $WORK/run.sh" <<EOF
#!/usr/bin/env bash
set -u
WORK=$WORK; REF=$REF; ARCH=$ARCH
cd \$WORK
log() { echo "[\$(date -u +%H:%M:%S)] \$*"; }
# cgroup CPU quota (v1 or v2), the honest thread count
quota=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || cut -d' ' -f1 /sys/fs/cgroup/cpu.max 2>/dev/null)
period=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || cut -d' ' -f2 /sys/fs/cgroup/cpu.max 2>/dev/null)
if [ -n "\$quota" ] && [ "\$quota" != "max" ] && [ "\$quota" -gt 0 ] 2>/dev/null; then
  THREADS=\$(( quota / period )); else THREADS=\$(nproc); fi
[ "\$THREADS" -gt 16 ] && THREADS=16
log "threads \$THREADS"
# venv: cmake, ninja, highspy; cuOpt reused from an earlier venv when it is there
if [ -x /home/jovyan/cuopt-venv/bin/python ] && /home/jovyan/cuopt-venv/bin/python -c 'import cuopt' 2>/dev/null; then
  VENV=/home/jovyan/cuopt-venv; log "cuOpt venv found at \$VENV"
else
  VENV=\$WORK/venv; [ -d \$VENV ] || python3 -m venv \$VENV
fi
PY=\$VENV/bin/python
# a venv made without pip (the cuOpt one on TIR, 22 Oct 2026) gets one from ensurepip
\$PY -m pip --version > /dev/null 2>&1 || \$PY -m ensurepip --upgrade 2>&1 | tail -1
\$PY -m pip install -q cmake ninja highspy 2>&1 | tail -2
\$PY -c 'import cuopt' 2>/dev/null || \$PY -m pip install -q --extra-index-url=https://pypi.nvidia.com cuopt-cu12 2>&1 | tail -3
\$PY -c 'import importlib.metadata as m; print("cuopt-cu12", m.version("cuopt-cu12"))' || log "cuOpt NOT installed; leg 2 will record it"
export PATH=\$VENV/bin:/usr/local/cuda/bin:\$PATH
# clone and build (fresh clone at REF, never a dirty tree: the stamp must be clean)
if [ ! -d src ]; then git clone -q https://github.com/thegoodengineers/SANKHYA.git src; fi
( cd src && git fetch -q origin && { git checkout -q "\$REF" 2>/dev/null || git checkout -q "origin/\$REF"; } && { git merge -q --ff-only "origin/\$REF" 2>/dev/null || true; } ) || { log "checkout of \$REF failed"; exit 1; }
( cd src && git log --oneline -1 && git status --porcelain --untracked-files=no | head -3 )
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSANKHYA_ENABLE_CUDA=ON \\
  -DSANKHYA_CUDA_ARCHITECTURES=\$ARCH -DSANKHYA_BUILD_TESTS=OFF \\
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > logs/cmake.log 2>&1 || { tail -20 logs/cmake.log; exit 1; }
cmake --build build -j\$THREADS --target sankhya > logs/build.log 2>&1 || { grep -E "error|FAILED" logs/build.log | head; exit 1; }
BIN=\$WORK/build/sankhya; \$BIN version
mkdir -p inst
# leg 1: the refinery year
log "leg 1: refinery year"
\$PY src/bench/runners/gpu_refinery_year.py --binary \$BIN --card a100 --keep inst \\
  --instances refinery_year --cpu-threads 4 --time-limit \${LIMIT1:-600} --repeats \${REPEATS:-1} \\
  --host-label "E2E TIR A100" > logs/leg1.log 2>&1 || true
log "leg 1 done: \$(ls src/bench/results/gpu-refinery-year-*.csv 2>/dev/null)"
# leg 2: cuOpt vs ours on LPfeas-14, 1e-4 first, then 1e-8 into the same CSV
log "leg 2: LPfeas-14"
\$PY src/bench/runners/mittelmann_cuopt.py --fetch > logs/fetch.log 2>&1 || true
for tol in 1e-4 1e-8; do
  \$PY src/bench/runners/mittelmann_cuopt.py --binary \$BIN --card a100 --time-limit 300 \\
    --tolerances \$tol --reference-time-limit 600 >> logs/leg2.log 2>&1 || true
done
log "leg 2 done"
# leg 3: the million-row refinery, only now
log "leg 3: million-refinery"
\$PY src/bench/runners/gpu_refinery_year.py --binary \$BIN --card a100 --keep inst \\
  --instances million-refinery --cpu-threads 4 --time-limit \${LIMIT3:-900} --repeats 1 \\
  --host-label "E2E TIR A100" > logs/leg3.log 2>&1 || true
log "all legs done"
EOF
$SSH "chmod +x $WORK/run.sh && cd $WORK && setsid nohup bash run.sh > logs/run.log 2>&1 < /dev/null & disown; echo started"
echo "started on $HOST; follow with: $SSH tail -f $WORK/logs/run.log"
