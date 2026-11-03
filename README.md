# VCSR-MC vs rocSPARSE on AMD MI300A

VCSR-MC SpMM (sparse A * dense B = dense C) on AMD MI300A GPU

## 1. Get the matrices

```bash
./get_matrices.sh
```

This downloads the matrices listed in `bench200/corpus200.csv` from SuiteSparse into `matrices200/mtx/` 

## 2. Build

```bash
./build.sh
```

This builds `build/spmm_vcsr`. The GPU target is detected with `rocminfo`; set `ARCH=gfx942` to choose it yourself.

## 3. Run

```bash
./run.sh
```

Defaults:
- kernels `rocsparse`, `rocsparse_fp16`, `vcsr_mc` and `vcsr_mc_fp32`;
- 3 warm-up launches, then the median of 10 timed launches;
- 3 fresh processes per (matrix, kernel), with noisy matrices rerun;

Every result is validated against hipSPARSE.

Results go to `results200_MI300A_S32T128_O128.csv`, and the log to `logs/`. At the end the script prints a speedup table (orig_table.py). It has one row for each of our kernels, vcsr_mc (FP16 inputs) and vcsr_mc_fp32 (FP32 inputs), both against rocSPARSE FP32 (rocsparse). The FP32 row is the matched-precision comparison. The CSV also holds the rocsparse_fp16 times, for comparing vcsr_mc against rocSPARSE at FP16.

To time a single matrix:

```bash
build/spmm_vcsr --mtx matrices200/mtx/enron.mtx --algo vcsr_mc,rocsparse_fp16 --O 128 --segw 32 --tileK 128 --warmup 3 --stat median
```

## Files

```
build.sh, CMakeLists.txt     build
get_matrices.sh              download the corpus
run.sh                       the benchmark
bench200/run200.py           one process per (matrix, kernel) over the corpus
rerun_noisy.py               reruns matrices whose repetitions disagree
orig_table.py, analyze.py    speedup table
src/vcsr_mc.cu               the VCSR-MC kernels (FP16, FP32) and format construction
src/kernels.cu               the hipSPARSE baseline (FP32 and FP16)
src/main.cpp                 driver: runs, times and validates the kernels
include/                     Matrix Market reader, CSR, format struct
reference/vcsr_tc.cu         VCSR-TC (Tensor Core CUDA)
```
