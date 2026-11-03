// vcsr_tc.cu - VCSR with Tensor Core acceleration
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "csr.hpp"

using namespace nvcuda;

struct VCSRTC {
    int M{0}, N{0};
    int segw{16};
    int bundle{16};
    std::vector<int> group_ptr;
    std::vector<int> group_depth;
    std::vector<int> group_seg_base;
    std::vector<int> group_rows;
    std::vector<int> lcol;
    std::vector<float> val;
    int num_segments{0};
};

VCSRTC csr_to_vcsr_tc(const CSR &csr, int bundle) {
    const int segw = 16;
    bundle = 16;
    
    VCSRTC V;
    V.M = csr.M; V.N = csr.N; V.segw = segw; V.bundle = bundle;
    int S = (csr.N + segw - 1) / segw;
    V.num_segments = S;
    
    struct Entry { int lcol; float v; };
    std::vector<std::unordered_map<int, std::vector<Entry>>> seg_entries(S);
    
    for (int r = 0; r < csr.M; ++r) {
        for (int p = csr.row_ptr[r]; p < csr.row_ptr[r+1]; ++p) {
            int c = csr.col[p];
            float v = csr.val[p];
            int sid = c / segw;
            int lc = c % segw;
            seg_entries[sid][r].push_back({lc, v});
        }
    }
    
    for (int sid = 0; sid < S; ++sid) {
        for (auto &kv : seg_entries[sid]) {
            std::sort(kv.second.begin(), kv.second.end(),
                      [](const Entry &a, const Entry &b) { return a.lcol < b.lcol; });
        }
    }
    
    for (int sid = 0; sid < S; ++sid) {
        std::vector<int> active_rows;
        for (auto &kv : seg_entries[sid]) {
            if (!kv.second.empty()) active_rows.push_back(kv.first);
        }
        
        std::sort(active_rows.begin(), active_rows.end(), [&](int a, int b) {
            return seg_entries[sid][a].size() > seg_entries[sid][b].size();
        });
        
        for (size_t start = 0; start < active_rows.size(); start += bundle) {
            size_t end = std::min(start + (size_t)bundle, active_rows.size());
            
            int depth = 0;
            for (size_t i = start; i < end; ++i) {
                depth = std::max(depth, (int)seg_entries[sid][active_rows[i]].size());
            }
            if (depth == 0) continue;
            
            V.group_ptr.push_back((int)V.val.size());
            V.group_depth.push_back(depth);
            V.group_seg_base.push_back(sid * segw);
            
            for (int lane = 0; lane < bundle; ++lane) {
                int ridx = (int)start + lane;
                V.group_rows.push_back(ridx < (int)active_rows.size() ? active_rows[ridx] : -1);
            }
            
            for (int d = 0; d < depth; ++d) {
                for (int lane = 0; lane < bundle; ++lane) {
                    int ridx = (int)start + lane;
                    if (ridx < (int)active_rows.size()) {
                        int row = active_rows[ridx];
                        auto &vec = seg_entries[sid][row];
                        if (d < (int)vec.size()) {
                            V.lcol.push_back(vec[d].lcol);
                            V.val.push_back(vec[d].v);
                        } else {
                            V.lcol.push_back(-1);
                            V.val.push_back(0.f);
                        }
                    } else {
                        V.lcol.push_back(-1);
                        V.val.push_back(0.f);
                    }
                }
            }
        }
    }
    V.group_ptr.push_back((int)V.val.size());
    return V;
}

namespace vcsr_tc_dev {
    static int* d_group_ptr = nullptr;
    static int* d_group_depth = nullptr;
    static int* d_group_seg_base = nullptr;
    static int* d_group_rows = nullptr;
    static int* d_lcol = nullptr;
    static float* d_val = nullptr;
    static bool uploaded = false;
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void vcsr_tc_kernel(
    int O, int bundle, int segw, int N, int num_groups,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_seg_base,
    const int* __restrict__ group_rows,
    const int* __restrict__ lcol,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    __shared__ half A_tile[WMMA_M * WMMA_K];
    __shared__ half B_tile[WMMA_K * WMMA_N];
    __shared__ float C_tile[WMMA_M * WMMA_N];
    
    int gid = blockIdx.x;
    int k_tile = blockIdx.y;
    if (gid >= num_groups) return;
    
    int lane = threadIdx.x;
    int seg_base = group_seg_base[gid];
    int base = group_ptr[gid];
    int depth = group_depth[gid];
    int k0 = k_tile * WMMA_N;
    
    if (k0 >= O) return;
    int actual_N = min(WMMA_N, O - k0);
    
    for (int i = lane; i < WMMA_M * WMMA_N; i += blockDim.x) C_tile[i] = 0.0f;
    __syncthreads();
    
    for (int d = 0; d < depth; ++d) {
        for (int i = lane; i < WMMA_M * WMMA_K; i += blockDim.x) A_tile[i] = __float2half(0.0f);
        __syncthreads();
        
        if (lane < bundle) {
            int idx = base + d * bundle + lane;
            int lc = lcol[idx];
            if (lc >= 0 && lc < segw) {
                A_tile[lane * WMMA_K + lc] = __float2half(aval[idx]);
            }
        }
        __syncthreads();
        
        for (int i = lane; i < WMMA_K * WMMA_N; i += blockDim.x) {
            int brow = i / WMMA_N, bcol = i % WMMA_N;
            int global_row = seg_base + brow;
            B_tile[i] = (global_row < N && bcol < actual_N) 
                ? __float2half(B[global_row * ldb + k0 + bcol]) : __float2half(0.0f);
        }
        __syncthreads();
        
        if (lane < 32) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            
            wmma::load_matrix_sync(a_frag, A_tile, WMMA_K);
            wmma::load_matrix_sync(b_frag, B_tile, WMMA_N);
            wmma::load_matrix_sync(c_frag, C_tile, WMMA_N, wmma::mem_row_major);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(C_tile, c_frag, WMMA_N, wmma::mem_row_major);
        }
        __syncthreads();
    }
    
    for (int i = lane; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int m = i / WMMA_N, n = i % WMMA_N;
        if (n < actual_N && m < bundle) {
            int row = group_rows[gid * bundle + m];
            if (row >= 0) atomicAdd(&C[row * ldc + k0 + n], C_tile[i]);
        }
    }
}

void run_vcsr_tc_spmm_gpu(const VCSRTC &V, const float* dB, int O,
                          float* dC, cudaStream_t st, float &ms) {
    using namespace vcsr_tc_dev;
    int num_groups = (int)V.group_depth.size();
    if (num_groups == 0) { ms = 0; return; }
    
    if (!uploaded) {
        cudaMalloc(&d_group_ptr, sizeof(int) * V.group_ptr.size());
        cudaMalloc(&d_group_depth, sizeof(int) * V.group_depth.size());
        cudaMalloc(&d_group_seg_base, sizeof(int) * V.group_seg_base.size());
        cudaMalloc(&d_group_rows, sizeof(int) * V.group_rows.size());
        cudaMalloc(&d_lcol, sizeof(int) * V.lcol.size());
        cudaMalloc(&d_val, sizeof(float) * V.val.size());
        cudaMemcpy(d_group_ptr, V.group_ptr.data(), sizeof(int) * V.group_ptr.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_depth, V.group_depth.data(), sizeof(int) * V.group_depth.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_seg_base, V.group_seg_base.data(), sizeof(int) * V.group_seg_base.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_rows, V.group_rows.data(), sizeof(int) * V.group_rows.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lcol, V.lcol.data(), sizeof(int) * V.lcol.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, V.val.data(), sizeof(float) * V.val.size(), cudaMemcpyHostToDevice);
        uploaded = true;
    }
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    
    dim3 grid(num_groups, (O + WMMA_N - 1) / WMMA_N);
    dim3 block(32);
    
    cudaEventRecord(t0, st);
    vcsr_tc_kernel<<<grid, block, 0, st>>>(O, V.bundle, V.segw, V.N, num_groups,
        d_group_ptr, d_group_depth, d_group_seg_base, d_group_rows, d_lcol, d_val, dB, O, dC, O);
    cudaEventRecord(t1, st);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void destroy_vcsr_tc_device(cudaStream_t) {
    using namespace vcsr_tc_dev;
    if (d_group_ptr) cudaFree(d_group_ptr);
    if (d_group_depth) cudaFree(d_group_depth);
    if (d_group_seg_base) cudaFree(d_group_seg_base);
    if (d_group_rows) cudaFree(d_group_rows);
    if (d_lcol) cudaFree(d_lcol);
    if (d_val) cudaFree(d_val);
    d_group_ptr = d_group_depth = d_group_seg_base = d_group_rows = d_lcol = nullptr;
    d_val = nullptr; uploaded = false;
}


