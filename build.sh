#!/bin/bash

set -uo pipefail
cd "$(dirname "$0")"
export PATH=/opt/rocm/bin:$PATH
mkdir -p logs
LOG=logs/build.log
{
  echo "### host: $(hostname)   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  hipcc --version | head -1
  ARCH="${ARCH:-$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1)}"
  echo "### arch: $ARCH"
  [ -z "$ARCH" ] && { echo "no AMD GPU detected -- run this on a ROCm node"; exit 1; }
  rm -rf build
  cmake -S . -B build -DGPU_ARCH="$ARCH" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
        -DCMAKE_PREFIX_PATH=/opt/rocm
  cmake --build build -j"${SLURM_CPUS_PER_TASK:-8}"
  echo "### binary: $(ls -l build/spmm_vcsr | awk '{print $5}') bytes"
  echo "### code objects: $(strings build/spmm_vcsr | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z]+' | sort -u | paste -sd' ')"
} 2>&1 | tee "$LOG"
echo "-> $LOG"
