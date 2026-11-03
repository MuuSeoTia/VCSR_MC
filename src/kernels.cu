#include "hip/hip_runtime.h"
#include "cuda_check.hpp"
#include <hip/hip_runtime.h>
#ifdef HAVE_ROCSPARSE
#if __has_include(<hipsparse/hipsparse.h>)
#include <hipsparse/hipsparse.h>
#else
#include <hipsparse.h>
#endif
#define ROCSPARSE_CHECK(call) do { \
    const hipsparseStatus_t status_ = (call); \
    if (status_ != HIPSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "rocSPARSE error %s:%d: %s: status %d\n", __FILE__, __LINE__, #call, int(status_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)
#define ROCSPARSE_THROW(call) do { \
    const hipsparseStatus_t status_ = (call); \
    if (status_ != HIPSPARSE_STATUS_SUCCESS) \
        throw std::runtime_error(std::string(#call) + " returned status " + std::to_string(int(status_))); \
} while (0)
#endif
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include "csr.hpp"

#ifdef HAVE_ROCSPARSE
namespace {
struct RocsparseContext {
  hipsparseHandle_t handle{nullptr};
  hipStream_t stream{nullptr};
  void* buffer{nullptr};
  size_t buffer_bytes{0};
  hipsparseSpMatDescr_t matA{nullptr};
  hipsparseDnMatDescr_t matB{nullptr}, matC{nullptr};
  const void *key_rowptr{nullptr}, *key_col{nullptr}, *key_val{nullptr};
  const void *key_B{nullptr}, *key_C{nullptr};
  int key_M{-1}, key_N{-1}, key_nnz{-1}, key_O{-1};
  hipsparseSpMMAlg_t key_alg{HIPSPARSE_SPMM_ALG_DEFAULT};
  float setup_ms{0.f};
};

RocsparseContext& rocsparse_context(int which = 0) {
  static RocsparseContext ctxs[2];
  RocsparseContext& ctx = ctxs[which];
  if (!ctx.handle) {
    ROCSPARSE_CHECK(hipsparseCreate(&ctx.handle));
    CUDA_CHECK(hipStreamCreate(&ctx.stream));
    ROCSPARSE_CHECK(hipsparseSetStream(ctx.handle, ctx.stream));
  }
  return ctx;
}
}

float run_rocsparse_spmm(const int M, const int N, const int nnz,
                        const int* d_rowptr, const int* d_col, const float* d_val,
                        const float* dB, int O, float* dC, float* setup_ms, int alg_id)
{
  RocsparseContext& ctx = rocsparse_context();
  const hipsparseSpMMAlg_t alg = alg_id == 1 ? HIPSPARSE_SPMM_CSR_ALG1
                               : alg_id == 2 ? HIPSPARSE_SPMM_CSR_ALG2
                               : alg_id == 3 ? HIPSPARSE_SPMM_CSR_ALG3
                               :               HIPSPARSE_SPMM_ALG_DEFAULT;
  const float alpha = 1.f, beta = 0.f;

  const bool same = ctx.matA && ctx.key_rowptr == d_rowptr && ctx.key_col == d_col &&
                    ctx.key_val == d_val && ctx.key_B == dB && ctx.key_C == dC &&
                    ctx.key_M == M && ctx.key_N == N && ctx.key_nnz == nnz && ctx.key_O == O &&
                    ctx.key_alg == alg;
  if (!same) {
    if (ctx.matA) {
      ROCSPARSE_CHECK(hipsparseDestroySpMat(ctx.matA));
      ROCSPARSE_CHECK(hipsparseDestroyDnMat(ctx.matB));
      ROCSPARSE_CHECK(hipsparseDestroyDnMat(ctx.matC));
      ctx.matA = nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ROCSPARSE_CHECK(hipsparseCreateCsr(&ctx.matA, M, N, nnz,
                      (void*)d_rowptr, (void*)d_col, (void*)d_val,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    ROCSPARSE_CHECK(hipsparseCreateDnMat(&ctx.matB, N, O, O, (void*)dB,
                        HIP_R_32F, HIPSPARSE_ORDER_ROW));
    ROCSPARSE_CHECK(hipsparseCreateDnMat(&ctx.matC, M, O, O, (void*)dC,
                        HIP_R_32F, HIPSPARSE_ORDER_ROW));
    size_t bufferSize = 0;
    ROCSPARSE_THROW(hipsparseSpMM_bufferSize(ctx.handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
                            HIP_R_32F, alg, &bufferSize));
    if (bufferSize > ctx.buffer_bytes) {
      if (ctx.buffer) CUDA_CHECK(hipFree(ctx.buffer));
      CUDA_CHECK(hipMalloc(&ctx.buffer, bufferSize));
      ctx.buffer_bytes = bufferSize;
    }
    ROCSPARSE_THROW(hipsparseSpMM_preprocess(ctx.handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
                            HIP_R_32F, alg, ctx.buffer));
    CUDA_CHECK(hipStreamSynchronize(ctx.stream));
    ctx.setup_ms = std::chrono::duration<float, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    ctx.key_rowptr = d_rowptr; ctx.key_col = d_col; ctx.key_val = d_val;
    ctx.key_B = dB; ctx.key_C = dC;
    ctx.key_M = M; ctx.key_N = N; ctx.key_nnz = nnz; ctx.key_O = O; ctx.key_alg = alg;
  }
  if (setup_ms) *setup_ms = ctx.setup_ms;

  hipEvent_t a, b;
  CUDA_CHECK(hipEventCreate(&a));
  CUDA_CHECK(hipEventCreate(&b));

  CUDA_CHECK(hipEventRecord(a, ctx.stream));
  ROCSPARSE_THROW(hipsparseSpMM(ctx.handle,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
               HIP_R_32F,
               alg,
               ctx.buffer));
  CUDA_CHECK(hipEventRecord(b, ctx.stream));
  CUDA_CHECK(hipEventSynchronize(b));

  float ms = 0.f;
  CUDA_CHECK(hipEventElapsedTime(&ms, a, b));

  CUDA_CHECK(hipEventDestroy(a));
  CUDA_CHECK(hipEventDestroy(b));
  return ms;
}

float run_rocsparse_spmm_fp16(const int M, const int N, const int nnz,
                             const int* d_rowptr, const int* d_col, const void* d_val16,
                             const void* dB16, int O, float* dC, float scale, float* setup_ms)
{
  RocsparseContext& ctx = rocsparse_context(1);
  const hipsparseSpMMAlg_t alg = HIPSPARSE_SPMM_ALG_DEFAULT;
  const float alpha = scale, beta = 0.f;
  const bool same = ctx.matA && ctx.key_rowptr == d_rowptr && ctx.key_col == d_col &&
                    ctx.key_val == d_val16 && ctx.key_B == dB16 && ctx.key_C == dC &&
                    ctx.key_M == M && ctx.key_N == N && ctx.key_nnz == nnz && ctx.key_O == O;
  if (!same) {
    if (ctx.matA) {
      ROCSPARSE_CHECK(hipsparseDestroySpMat(ctx.matA));
      ROCSPARSE_CHECK(hipsparseDestroyDnMat(ctx.matB));
      ROCSPARSE_CHECK(hipsparseDestroyDnMat(ctx.matC));
      ctx.matA = nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ROCSPARSE_CHECK(hipsparseCreateCsr(&ctx.matA, M, N, nnz,
                      (void*)d_rowptr, (void*)d_col, (void*)d_val16,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_16F));
    ROCSPARSE_CHECK(hipsparseCreateDnMat(&ctx.matB, N, O, O, (void*)dB16,
                        HIP_R_16F, HIPSPARSE_ORDER_ROW));
    ROCSPARSE_CHECK(hipsparseCreateDnMat(&ctx.matC, M, O, O, (void*)dC,
                        HIP_R_32F, HIPSPARSE_ORDER_ROW));
    size_t bufferSize = 0;
    ROCSPARSE_THROW(hipsparseSpMM_bufferSize(ctx.handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
                            HIP_R_32F, alg, &bufferSize));
    if (bufferSize > ctx.buffer_bytes) {
      if (ctx.buffer) CUDA_CHECK(hipFree(ctx.buffer));
      CUDA_CHECK(hipMalloc(&ctx.buffer, bufferSize));
      ctx.buffer_bytes = bufferSize;
    }
    ROCSPARSE_THROW(hipsparseSpMM_preprocess(ctx.handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
                            HIP_R_32F, alg, ctx.buffer));
    CUDA_CHECK(hipStreamSynchronize(ctx.stream));
    ctx.setup_ms = std::chrono::duration<float, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    ctx.key_rowptr = d_rowptr; ctx.key_col = d_col; ctx.key_val = d_val16;
    ctx.key_B = dB16; ctx.key_C = dC;
    ctx.key_M = M; ctx.key_N = N; ctx.key_nnz = nnz; ctx.key_O = O; ctx.key_alg = alg;
  }
  if (setup_ms) *setup_ms = ctx.setup_ms;

  hipEvent_t a, b;
  CUDA_CHECK(hipEventCreate(&a));
  CUDA_CHECK(hipEventCreate(&b));
  CUDA_CHECK(hipEventRecord(a, ctx.stream));
  ROCSPARSE_THROW(hipsparseSpMM(ctx.handle,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, ctx.matA, ctx.matB, &beta, ctx.matC,
               HIP_R_32F, alg, ctx.buffer));
  CUDA_CHECK(hipEventRecord(b, ctx.stream));
  CUDA_CHECK(hipEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(hipEventElapsedTime(&ms, a, b));
  CUDA_CHECK(hipEventDestroy(a));
  CUDA_CHECK(hipEventDestroy(b));
  return ms;
}
#endif
