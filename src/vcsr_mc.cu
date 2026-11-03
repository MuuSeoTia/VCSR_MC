#include "hip/hip_runtime.h"
#include "cuda_check.hpp"
// vcsr_mc.cu - VCSR-MC: the VCSR-TC SpMM kernel (github.com/MuuSeoTia/VCSR-SPMM) on AMD CDNA matrix cores (MFMA)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cmath>
#include "vcsr_formats.hpp"

// FP16 range scaling: A is divided by this before conversion, C multiplied back
float half_scale(const CSR &csr) {
    constexpr float kHalfHeadroom = 60000.0f;
    constexpr float kHalfMinNormal = 6.103515625e-05f;
    float amax = 0.0f;
    for (float v : csr.val) amax = std::max(amax, std::fabs(v));
    if (amax > kHalfHeadroom) return amax / kHalfHeadroom;
    if (amax > 0.0f && amax < kHalfMinNormal) return amax / kHalfHeadroom;
    return 1.0f;
}

// Format construction (host): segment-major Segmented VCSR, 16-row groups, work items
VCSRMC csr_to_vcsr_mc(const CSR &csr, bool fp32, int segw) {
    if (segw != 16 && segw != 32 && segw != 48 && segw != 64)
        throw std::runtime_error("vcsr_mc supports segw in {16, 32, 48, 64}");
    const int bundle = 16;

    VCSRMC V;
    V.M = csr.M; V.N = csr.N; V.segw = segw; V.bundle = bundle;
    V.fp32 = fp32;
    V.scale = fp32 ? 1.0f : half_scale(csr);
    const float inv_scale = 1.0f / V.scale;
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
                            V.val.push_back(vec[d].v * inv_scale);
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

    // Work items: runs of up to `chunk` groups of one segment, heaviest first
    const int G = (int)V.group_depth.size();
    const int chunk = std::max(1, std::min(32, G / 4096));
    struct Item { int b, e; long long work; };
    std::vector<Item> items;
    for (int g = 0; g < G;) {
        int e = g;
        while (e < G && e - g < chunk && V.group_seg_base[e] == V.group_seg_base[g]) ++e;
        items.push_back({g, e, (long long)V.group_ptr[e] - V.group_ptr[g]});
        g = e;
    }
    std::stable_sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.work > b.work; });
    for (const Item &it : items) { V.item_gbeg.push_back(it.b); V.item_gend.push_back(it.e); }
    return V;
}

// Device copies of the format
namespace vcsr_mc_dev {
static int *d_group_ptr = nullptr, *d_group_depth = nullptr, *d_group_seg_base = nullptr;
static int *d_group_rows = nullptr, *d_lcol = nullptr, *d_item_gbeg = nullptr, *d_item_gend = nullptr;
static float* d_val = nullptr;
static bool uploaded = false;
}

typedef _Float16 amd_half4 __attribute__((ext_vector_type(4)));
typedef float    amd_float4 __attribute__((ext_vector_type(4)));


constexpr int AMD_WAVE = 64;
constexpr int a_ld16(int SW) { return SW == 16 ? 20 : SW + 4; }
constexpr int B_PAD = 4;

constexpr int A_BUFS = 3;

// FP16 kernel: v_mfma_f32_16x16x16f16, FP16 A and B in LDS, FP32 accumulate
template <int NSLAB, int SW>
__global__ __launch_bounds__(NSLAB * AMD_WAVE)
void vcsr_mc(
    int O, int N, int num_items, float scale,
    const int* __restrict__ item_gbeg,
    const int* __restrict__ item_gend,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_seg_base,
    const int* __restrict__ group_rows,
    const int* __restrict__ lcol,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    constexpr int TK = NSLAB * 16;
    constexpr int NTHREAD = NSLAB * AMD_WAVE;
    constexpr int B_LD = TK + B_PAD;
    constexpr int A_LD = a_ld16(SW);
    constexpr int KC = SW / 16;
    __shared__ _Float16 As[A_BUFS][16 * A_LD];
    __shared__ _Float16 Bs[SW * B_LD];

    const int item = blockIdx.x;
    if (item >= num_items) return;
    const int tid = threadIdx.x;
    const int wave = tid / AMD_WAVE, lane = tid % AMD_WAVE;
    const int n = lane % 16, q = lane / 16;
    const int k0 = blockIdx.y * TK;
    const int gb = item_gbeg[item], ge = item_gend[item];
    const int seg_base = group_seg_base[gb];
    const int bc = wave * 16 + n;
    const int col = k0 + bc;
    amd_half4 b[KC];

    for (int i = tid; i < A_BUFS * 16 * A_LD; i += NTHREAD) (&As[0][0])[i] = (_Float16)0;

    for (int e = tid; e < SW * TK / 4; e += NTHREAD) {
        const int r = e / (TK / 4), c = (e % (TK / 4)) * 4;
        const int gr = seg_base + r, gc = k0 + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < N) {
            const float* src = B + (size_t)gr * ldb + gc;
            if ((ldb & 3) == 0 && gc + 3 < O) v = *reinterpret_cast<const float4*>(src);
            else {
                if (gc     < O) v.x = src[0];
                if (gc + 1 < O) v.y = src[1];
                if (gc + 2 < O) v.z = src[2];
                if (gc + 3 < O) v.w = src[3];
            }
        }
        *reinterpret_cast<amd_half4*>(&Bs[r * B_LD + c]) =
            amd_half4{(_Float16)v.x, (_Float16)v.y, (_Float16)v.z, (_Float16)v.w};
    }
    __syncthreads();

    #pragma unroll
    for (int c = 0; c < KC; ++c) {
        const int r0 = 16 * c + 4 * q;
        b[c] = amd_half4{Bs[r0 * B_LD + bc], Bs[(r0 + 1) * B_LD + bc],
                         Bs[(r0 + 2) * B_LD + bc], Bs[(r0 + 3) * B_LD + bc]};
    }

    for (int g = gb; g < ge; ++g) {
        const int buf = (g - gb) % A_BUFS;
        const int base = group_ptr[g], depth = group_depth[g];

        for (int i = tid; i < depth * 16; i += NTHREAD) {
            const int lc = lcol[base + i];
            if (lc >= 0 && lc < SW) As[buf][(i % 16) * A_LD + lc] = (_Float16)aval[base + i];
        }
        __syncthreads();
        if (g > gb) {
            const int pbuf = (g - 1 - gb) % A_BUFS;
            const int pbase = group_ptr[g - 1], pdepth = group_depth[g - 1];
            for (int i = tid; i < pdepth * 16; i += NTHREAD) {
                const int lc = lcol[pbase + i];
                if (lc >= 0 && lc < SW) As[pbuf][(i % 16) * A_LD + lc] = (_Float16)0;
            }
        }
        amd_float4 d = amd_float4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < KC; ++c) {
            const amd_half4 a = *reinterpret_cast<const amd_half4*>(&As[buf][n * A_LD + 16 * c + 4 * q]);
            d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b[c], d, 0, 0, 0);
        }
        if (col < O) {
            const int* rows = group_rows + (size_t)g * 16 + 4 * q;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                if (rows[i] >= 0) atomicAdd(&C[(size_t)rows[i] * ldc + col], d[i] * scale);
        }
    }
}

constexpr int B_PAD32 = 16;

// FP32 kernel: v_mfma_f32_16x16x4f32, FP32 throughout
template <int NSLAB, int SW>
__global__ __launch_bounds__(NSLAB * AMD_WAVE)
void vcsr_mc_fp32(
    int O, int N, int num_items,
    const int* __restrict__ item_gbeg,
    const int* __restrict__ item_gend,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_seg_base,
    const int* __restrict__ group_rows,
    const int* __restrict__ lcol,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    constexpr int TK = NSLAB * 16;
    constexpr int NTHREAD = NSLAB * AMD_WAVE;
    constexpr int B_LD = TK + B_PAD32;
    constexpr int A_LD32 = SW + 2;
    constexpr int KC = SW / 4;
    __shared__ float As[A_BUFS][16 * A_LD32];
    __shared__ float Bs[SW * B_LD];

    const int item = blockIdx.x;
    if (item >= num_items) return;
    const int tid = threadIdx.x;
    const int wave = tid / AMD_WAVE, lane = tid % AMD_WAVE;
    const int n = lane % 16, q = lane / 16;
    const int k0 = blockIdx.y * TK;
    const int gb = item_gbeg[item], ge = item_gend[item];
    const int seg_base = group_seg_base[gb];
    const int bc = wave * 16 + n;
    const int col = k0 + bc;
    float b[KC];

    for (int i = tid; i < A_BUFS * 16 * A_LD32; i += NTHREAD) (&As[0][0])[i] = 0.f;

    for (int e = tid; e < SW * TK / 4; e += NTHREAD) {
        const int r = e / (TK / 4), c = (e % (TK / 4)) * 4;
        const int gr = seg_base + r, gc = k0 + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < N) {
            const float* src = B + (size_t)gr * ldb + gc;
            if ((ldb & 3) == 0 && gc + 3 < O) v = *reinterpret_cast<const float4*>(src);
            else {
                if (gc     < O) v.x = src[0];
                if (gc + 1 < O) v.y = src[1];
                if (gc + 2 < O) v.z = src[2];
                if (gc + 3 < O) v.w = src[3];
            }
        }
        *reinterpret_cast<float4*>(&Bs[r * B_LD + c]) = v;
    }
    __syncthreads();
    #pragma unroll
    for (int c = 0; c < KC; ++c) b[c] = Bs[(4 * c + q) * B_LD + bc];

    for (int g = gb; g < ge; ++g) {
        const int buf = (g - gb) % A_BUFS;
        const int base = group_ptr[g], depth = group_depth[g];

        for (int i = tid; i < depth * 16; i += NTHREAD) {
            const int lc = lcol[base + i];
            if (lc >= 0 && lc < SW) As[buf][(i % 16) * A_LD32 + lc] = aval[base + i];
        }
        __syncthreads();

        if (g > gb) {
            const int pbuf = (g - 1 - gb) % A_BUFS;
            const int pbase = group_ptr[g - 1], pdepth = group_depth[g - 1];
            for (int i = tid; i < pdepth * 16; i += NTHREAD) {
                const int lc = lcol[pbase + i];
                if (lc >= 0 && lc < SW) As[pbuf][(i % 16) * A_LD32 + lc] = 0.f;
            }
        }
        amd_float4 d = amd_float4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < KC; ++c)
            d = __builtin_amdgcn_mfma_f32_16x16x4f32(As[buf][n * A_LD32 + 4 * c + q], b[c], d, 0, 0, 0);
        // Merge into C with atomicAdd straight from the accumulator registers
        if (col < O) {
            const int* rows = group_rows + (size_t)g * 16 + 4 * q;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                if (rows[i] >= 0) atomicAdd(&C[(size_t)rows[i] * ldc + col], d[i]);
        }
    }
}

// Upload the format
void upload_vcsr_mc_device(const VCSRMC &V) {
    using namespace vcsr_mc_dev;
    if (uploaded) return;
    auto up = [](auto** dst, const auto& s) {
        using T = typename std::decay_t<decltype(s)>::value_type;
        CUDA_CHECK(hipMalloc(dst, sizeof(T) * std::max<size_t>(s.size(), 1)));
        if (!s.empty()) CUDA_CHECK(hipMemcpy(*dst, s.data(), sizeof(T) * s.size(), hipMemcpyHostToDevice));
    };
    up(&d_group_ptr, V.group_ptr);
    up(&d_group_depth, V.group_depth);
    up(&d_group_seg_base, V.group_seg_base);
    up(&d_group_rows, V.group_rows);
    up(&d_lcol, V.lcol);
    up(&d_val, V.val);
    up(&d_item_gbeg, V.item_gbeg);
    up(&d_item_gend, V.item_gend);
    uploaded = true;
}

// Launch: one workgroup per (work item, T_K-column block); T_K/16 wavefronts each
void run_vcsr_mc_spmm_gpu(const VCSRMC &V, const float* dB, int O, int tileK,
                              float* dC, hipStream_t st, float &ms) {
    using namespace vcsr_mc_dev;
    const int num_items = (int)V.item_gbeg.size();
    if (num_items == 0) { ms = 0; return; }
    if (!uploaded) upload_vcsr_mc_device(V);

    hipEvent_t t0, t1;
    CUDA_CHECK(hipEventCreate(&t0)); CUDA_CHECK(hipEventCreate(&t1));
    auto launch_sw = [&](auto slabs, auto segw) {
        constexpr int NSLAB = decltype(slabs)::value, SW = decltype(segw)::value;
        const dim3 grid(num_items, (O + NSLAB * 16 - 1) / (NSLAB * 16));
        const dim3 block(NSLAB * AMD_WAVE);
        if (V.fp32)
            vcsr_mc_fp32<NSLAB, SW><<<grid, block, 0, st>>>(O, V.N, num_items,
                d_item_gbeg, d_item_gend, d_group_ptr, d_group_depth, d_group_seg_base,
                d_group_rows, d_lcol, d_val, dB, O, dC, O);
        else
            vcsr_mc<NSLAB, SW><<<grid, block, 0, st>>>(O, V.N, num_items, V.scale,
                d_item_gbeg, d_item_gend, d_group_ptr, d_group_depth, d_group_seg_base,
                d_group_rows, d_lcol, d_val, dB, O, dC, O);
    };
    auto launch = [&](auto slabs) {
        if      (V.segw == 32) launch_sw(slabs, std::integral_constant<int, 32>{});
        else if (V.segw == 48) launch_sw(slabs, std::integral_constant<int, 48>{});
        else if (V.segw == 64) launch_sw(slabs, std::integral_constant<int, 64>{});
        else                   launch_sw(slabs, std::integral_constant<int, 16>{});
    };
    CUDA_CHECK(hipEventRecord(t0, st));
    if      (tileK == 16)  launch(std::integral_constant<int, 1>{});
    else if (tileK == 32)  launch(std::integral_constant<int, 2>{});
    else if (tileK == 64)  launch(std::integral_constant<int, 4>{});
    else if (tileK == 128) launch(std::integral_constant<int, 8>{});
    else {
        CUDA_CHECK(hipEventDestroy(t0)); CUDA_CHECK(hipEventDestroy(t1));
        throw std::runtime_error("vcsr_mc supports tileK in {16, 32, 64, 128}");
    }
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipEventRecord(t1, st));
    CUDA_CHECK(hipEventSynchronize(t1));
    CUDA_CHECK(hipEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(hipEventDestroy(t0)); CUDA_CHECK(hipEventDestroy(t1));
}

// Free the device copies
void destroy_vcsr_mc_device(hipStream_t) {
    using namespace vcsr_mc_dev;
    for (int** p : {&d_group_ptr, &d_group_depth, &d_group_seg_base, &d_group_rows, &d_lcol,
                    &d_item_gbeg, &d_item_gend})
        if (*p) { CUDA_CHECK(hipFree(*p)); *p = nullptr; }
    if (d_val) { CUDA_CHECK(hipFree(d_val)); d_val = nullptr; }
    uploaded = false;
}