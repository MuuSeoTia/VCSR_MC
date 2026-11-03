#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/opt/rocm/bin:$PATH
mkdir -p logs
CORPUS="${CORPUS:-bench200/corpus200.csv}"
D="${D:-$PWD/matrices200}"
TAG="${TAG:-MI300A_S32T128}"
GPU="${GPU:-0}"
OS="${OS:-128}"
WARMUP="${WARMUP:-3}"
TILEK="${TILEK:-128}"
SEGW="${SEGW:-32}"
SEGW_ARG=${SEGW:+--segw $SEGW}
export HIP_VISIBLE_DEVICES="$GPU"

export ALGOS="${ALGOS:-rocsparse rocsparse_fp16 vcsr_mc vcsr_mc_fp32}"
export REPS="${REPS:-3}"
NODE_CPUS=$(cat /sys/devices/system/node/node"$GPU"/cpulist 2>/dev/null)
LOG="logs/run_$TAG.log"
[ -x build/spmm_vcsr ] || { echo "build first: ./build.sh"; exit 1; }
[ -d "$D/mtx" ]        || { echo "fetch first: ./get_matrices.sh"; exit 1; }
{
  echo "### host: $(hostname)   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rocm-smi --showproductname 2>/dev/null | grep -E "Card Series|GFX Version" | head -2
  echo "### corpus: $CORPUS ($(($(wc -l < "$CORPUS")-1)) matrices)"
  echo "### protocol: repeat=10 warmup=$WARMUP, mean and median recorded; O in {$OS}; tileK=min(O,$TILEK) ${SEGW_ARG:-segw default}"
  echo "### kernels (one process each): $ALGOS"
  echo "### device: HIP_VISIBLE_DEVICES=$GPU, CPUs bound to NUMA node $GPU ($NODE_CPUS)"
  echo "### repetitions: $REPS per (matrix, kernel); per-matrix median reported"
  if [ -n "$NODE_CPUS" ] && taskset -c "$NODE_CPUS" true 2>/dev/null; then
    BIND=(taskset -c "$NODE_CPUS")
  else
    echo "### WARNING: cannot bind to node $GPU CPUs (run with --exclusive); unbound"
    BIND=()
  fi
  CSVS=()
  for O in $OS; do
    TK=$(( O < TILEK ? O : TILEK ))
    export EXTRA="--O $O --tileK $TK $SEGW_ARG --repeat 10 --warmup $WARMUP --stat median"
    CSV="results200_${TAG}_O$O.csv"
    echo; echo "### O=$O tileK=$TK -> $CSV   $(date -u +%H:%M:%SZ)"
    time "${BIND[@]}" python3 bench200/run200.py ./build/spmm_vcsr "$CORPUS" "$D/mtx" "$CSV" > /dev/null
    echo "### rows: $(($(wc -l < "$CSV")-1))"
    CSVS+=("$CSV")
  done
  echo; echo "### rerunning noisy matrices   $(date -u +%H:%M:%SZ)"
  "${BIND[@]}" python3 rerun_noisy.py ./build/spmm_vcsr "$CORPUS" "$D/mtx" "${CSVS[@]}"
  echo
  python3 orig_table.py "${CSVS[@]}"
} 2>&1 | tee "$LOG"
echo "-> $LOG"
