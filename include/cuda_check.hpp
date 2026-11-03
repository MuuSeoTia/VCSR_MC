#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    const hipError_t status_ = (call); \
    if (status_ != hipSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s: %s\n", \
                     __FILE__, __LINE__, #call, hipGetErrorString(status_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Request larger dynamic shared memory only when the existing launch needs it.
template <typename Kernel>
inline void check_shared_memory(Kernel kernel, size_t bytes) {
    int device = 0, normal = 0, optin = 0;
    CUDA_CHECK(hipGetDevice(&device));
    CUDA_CHECK(hipDeviceGetAttribute(&normal, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
    if (bytes <= static_cast<size_t>(normal)) return;
    CUDA_CHECK(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, device));
    if (bytes > static_cast<size_t>(optin))
        throw std::runtime_error("segw * tileK exceeds this GPU's shared-memory limit");
    CUDA_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)));
}
