#include "hip/hip_runtime.h"
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <vector>
#include "csr.hpp"


// Factor A is divided by before FP16 conversion (and C multiplied by after); vcsr_mc.cu.
float half_scale(const CSR &csr);


struct VCSRMC {
    int M{0}, N{0}, segw{16}, bundle{16}, num_segments{0};
    std::vector<int> group_ptr, group_depth, group_seg_base, group_rows, lcol;
    std::vector<float> val;
    std::vector<int> item_gbeg, item_gend;   // groups [gbeg, gend) of one segment per item
    float scale{1.0f};
    bool fp32{false};   // `vcsr_mc_fp32`: FP32 operands and MFMA, no FP16 conversion, scale = 1
};
VCSRMC csr_to_vcsr_mc(const CSR &csr, bool fp32 = false, int segw = 16);   // segw: 16, 32, 48 or 64
void upload_vcsr_mc_device(const VCSRMC &V);
void run_vcsr_mc_spmm_gpu(const VCSRMC &V, const float* dB, int O, int tileK,
                              float* dC, hipStream_t st, float &ms);
void destroy_vcsr_mc_device(hipStream_t st);
