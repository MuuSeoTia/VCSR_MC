#include "hip/hip_runtime.h"
#include "cuda_check.hpp"
#include <cerrno>
#include <climits>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <getopt.h>
#include "mmio.hpp"
#include "csr.hpp"
#include "vcsr_formats.hpp"
#include "reorder.hpp"

// Forward decls from kernels.cu
float run_rocsparse_spmm(const int M, const int N, const int nnz,
                        const int* d_rowptr, const int* d_col, const float* d_val,
                        const float* dB, int O, float* dC, float* setup_ms = nullptr,
                        int alg_id = 0);
float run_rocsparse_spmm_fp16(const int M, const int N, const int nnz,
                             const int* d_rowptr, const int* d_col, const void* d_val16,
                             const void* dB16, int O, float* dC, float scale, float* setup_ms);
float half_scale(const CSR &csr);

static void usage() {
    printf("Usage: spmm_vcsr --mtx file.mtx --O 128 [options]\n"
           "Algorithms: rocsparse, vcsr_mc (original VCSR-TC, hipified + adapted to CDNA),\n"
           "            vcsr_mc_fp32 (the same, FP32 operands and FP32 MFMA),\n"
           "            rocsparse_fp16 = hipSPARSE default, FP16 A and B, FP32 compute and C\n"
           "Options: --segw 16 --tileK 64 --repeat 10 --warmup 3 --stat mean|median\n"
           "         --reorder none|rcm --check cpu|rocsparse|none --csv --help\n"
           "         --bdist pattern|normal|uniform|wide   (B fill; random ones use a fixed seed)\n"
           "         --lenient   record a failed correctness check as FAIL instead of aborting\n"
           "tileK must be one of 16, 32, 64, 128.\n");
}

static int integer_at_least(const char* text, long minimum) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum || value > INT_MAX)
        throw std::runtime_error(std::string("Expected integer >= ") +
                                 std::to_string(minimum) + ", got: " + text);
    return int(value);
}

static int positive_int(const char* text) { return integer_at_least(text, 1); }

static std::string csv_quote(const std::string& value) {
    std::string out = "\"";
    for (char c : value) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}

static int benchmark_main(int argc, char** argv) {
    std::string mtx, algo = "all";
#ifdef HAVE_ROCSPARSE
    std::string check = "rocsparse";
#else
    std::string check = "cpu";
#endif
    int O = 128, segw = 16, tileK = 64, repeat = 10, warmup = 3;
    constexpr int bundle = 16;   // rows per group; fixed by vcsr_mc (csr_to_vcsr_mc)
    std::string reorder = "none";
    // How the dense input B is filled. "pattern" is the historical deterministic fill
    // (13 values in [-6/7, 6/7], used for every timing result so far); the others are
    // random with a fixed seed, for accuracy checks: "normal" N(0,1), "uniform" U(-1,1),
    // "wide" random sign x log-uniform magnitude in [1e-3, 1e3].
    std::string bdist = "pattern";
    // How the --repeat timed launches are reduced to the reported time: "mean" (every
    // result so far) or "median" (the VCSR-TC paper's protocol: median single launch).
    std::string stat = "mean";
    bool csv = false, lenient = false;
    static option opts[] = {
        {"mtx", required_argument, nullptr, 'm'}, {"O", required_argument, nullptr, 'o'},
        {"segw", required_argument, nullptr, 's'},
        {"tileK", required_argument, nullptr, 't'}, {"repeat", required_argument, nullptr, 'r'},
        {"warmup", required_argument, nullptr, 'w'},
        {"reorder", required_argument, nullptr, 'R'},
        {"algo", required_argument, nullptr, 'a'}, {"check", required_argument, nullptr, 'c'},
        {"csv", no_argument, nullptr, 'j'}, {"help", no_argument, nullptr, 'h'},
        {"bdist", required_argument, nullptr, 'D'}, {"lenient", no_argument, nullptr, 'L'},
        {"stat", required_argument, nullptr, 'S'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': mtx = optarg; break;
            case 'o': O = positive_int(optarg); break;
            case 's': segw = positive_int(optarg); break;
            case 't': tileK = positive_int(optarg); break;
            case 'r': repeat = positive_int(optarg); break;
            case 'w': warmup = integer_at_least(optarg, 0); break;
            case 'R': reorder = optarg; break;
            case 'a': algo = optarg; break;
            case 'c': check = optarg; break;
            case 'D': bdist = optarg; break;
            case 'L': lenient = true; break;
            case 'S': stat = optarg; break;
            case 'j': csv = true; break;
            case 'h': usage(); return 0;
            default: usage(); return 2;
        }
    }
    const std::vector<std::string> algorithms = {
        "all", "rocsparse", "rocsparse_fp16",
        "vcsr_mc", "vcsr_mc_fp32"
    };
    // --algo takes a comma-separated list so one process can time several kernels off a
    // single parse. Reading and converting a large SuiteSparse matrix dominates wall
    // time -- one invocation per algorithm re-pays that cost every time.
    std::vector<std::string> requested;
    for (size_t pos = 0;;) {
        const size_t comma = algo.find(',', pos);
        const size_t end = (comma == std::string::npos) ? algo.size() : comma;
        std::string one = algo.substr(pos, end - pos);
        if (one.empty() || std::find(algorithms.begin(), algorithms.end(), one) == algorithms.end())
            throw std::runtime_error("Unknown algorithm: " + one);
        requested.push_back(one);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    const bool run_all = std::find(requested.begin(), requested.end(), "all") != requested.end();
    auto chosen = [&](const char* name) {
        return std::find(requested.begin(), requested.end(), name) != requested.end();
    };
    if (check != "cpu" && check != "rocsparse" && check != "none")
        throw std::runtime_error("Unknown reference: " + check);
#ifndef HAVE_ROCSPARSE
    if (chosen("rocsparse") || check == "rocsparse")
        throw std::runtime_error("rocSPARSE was disabled at build time");
#endif
    if (mtx.empty() || optind != argc) { usage(); return 2; }
    if (reorder != "none" && reorder != "rcm")
        throw std::runtime_error("Unknown reordering: " + reorder);
    if (stat != "mean" && stat != "median")
        throw std::runtime_error("Unknown --stat: " + stat + " (mean|median)");
    if (tileK != 16 && tileK != 32 && tileK != 64 && tileK != 128)
        throw std::runtime_error("tileK must be one of 16, 32, 64, 128");

    CSR csr = coo_to_csr(read_matrix_market_coo(mtx));
    // Existing kernels use signed 32-bit indexing for flattened B and C.
    if (size_t(csr.N) * O > INT_MAX || size_t(csr.M) * O > INT_MAX ||
        segw > INT_MAX - csr.N + 1 || tileK > INT_MAX - O + 1)
        throw std::runtime_error("Dimensions exceed the kernels' 32-bit indexing limit");
    hipDeviceProp_t device{};
    int device_id;
    CUDA_CHECK(hipGetDevice(&device_id));
    CUDA_CHECK(hipGetDeviceProperties(&device, device_id));
    const size_t elementsB = size_t(csr.N) * O, elementsC = size_t(csr.M) * O;
    const size_t bytesB = elementsB * sizeof(float), bytesC = elementsC * sizeof(float);
    std::vector<float> hB(elementsB);
    if (bdist == "pattern") {
        for (size_t i = 0; i < hB.size(); ++i) hB[i] = float(int(i % 13) - 6) / 7.f;
    } else {
        std::mt19937_64 rng(20260923);
        if (bdist == "normal") {
            std::normal_distribution<float> d(0.f, 1.f);
            for (float& x : hB) x = d(rng);
        } else if (bdist == "uniform") {
            std::uniform_real_distribution<float> d(-1.f, 1.f);
            for (float& x : hB) x = d(rng);
        } else if (bdist == "wide") {
            std::uniform_real_distribution<float> e(-3.f, 3.f);
            std::bernoulli_distribution sign(0.5);
            for (float& x : hB) x = (sign(rng) ? -1.f : 1.f) * std::pow(10.f, e(rng));
        } else {
            throw std::runtime_error("Unknown --bdist: " + bdist + " (pattern|normal|uniform|wide)");
        }
    }
    // Relabelling A's columns and permuting B's rows to match leaves C unchanged, so
    // this is applied once, up front, to every algorithm including the baselines.
    if (reorder == "rcm" && csr.nnz)
        apply_column_permutation(csr, hB, O, rcm_column_order(csr));
    float *dB = nullptr, *dC = nullptr, *d_val = nullptr;
    int *d_rowptr = nullptr, *d_col = nullptr;
    CUDA_CHECK(hipMalloc(&dB, bytesB)); CUDA_CHECK(hipMalloc(&dC, bytesC));
    CUDA_CHECK(hipMalloc(&d_rowptr, sizeof(int) * (size_t(csr.M) + 1)));
    CUDA_CHECK(hipMalloc(&d_col, sizeof(int) * std::max(csr.nnz, 1)));
    CUDA_CHECK(hipMalloc(&d_val, sizeof(float) * std::max(csr.nnz, 1)));
    CUDA_CHECK(hipMemcpy(d_rowptr, csr.row_ptr.data(), sizeof(int) * (size_t(csr.M) + 1), hipMemcpyHostToDevice));
    if (csr.nnz) {
        CUDA_CHECK(hipMemcpy(d_col, csr.col.data(), sizeof(int) * csr.nnz, hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_val, csr.val.data(), sizeof(float) * csr.nnz, hipMemcpyHostToDevice));
    }
    CUDA_CHECK(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    hipStream_t st;
    CUDA_CHECK(hipStreamCreate(&st));

    // Validation is outside kernel event timing. CPU mode is an independent oracle.
    std::vector<double> reference;
    if (check != "none") reference.assign(elementsC, 0.0);
    if (check == "cpu") {
        for (int r = 0; r < csr.M; ++r)
            for (int p = csr.row_ptr[r]; p < csr.row_ptr[r+1]; ++p)
                for (int k = 0; k < O; ++k)
                    reference[size_t(r)*O+k] += double(csr.val[p]) * hB[size_t(csr.col[p])*O+k];
    }
#ifdef HAVE_ROCSPARSE
    if (check == "rocsparse" && csr.nnz) {
        CUDA_CHECK(hipMemset(dC, 0, bytesC));
        run_rocsparse_spmm(csr.M, csr.N, csr.nnz, d_rowptr, d_col, d_val, dB, O, dC);
        std::vector<float> result(elementsC);
        CUDA_CHECK(hipMemcpy(result.data(), dC, bytesC, hipMemcpyDeviceToHost));
        reference.assign(result.begin(), result.end());
    }
#endif
    for (double v : reference)
        if (!std::isfinite(v)) throw std::runtime_error("Reference contains non-finite values");
    if (csv)
        printf("gpu,sm,mtx,O,segw,bundle,tileK,repeat,algo,ms,gflops,validation,rel_l2,setup_ms,ms_mean,ms_median,convert_ms,upload_ms\n");
    else
        printf("Matrix: %d x %d, nnz=%d; O=%d segw=%d bundle=%d tileK=%d repeat=%d warmup=%d reorder=%s; reference=%s\n"
               "Kernel                Time (ms)      GFLOP/s   Validation\n",
               csr.M, csr.N, csr.nnz, O, segw, bundle, tileK, repeat, warmup,
               reorder.c_str(), check.c_str());
    auto selected = [&](const char* name) { return run_all || chosen(name); };
    auto wall_ms = [](auto&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    const float no_setup = 0.f;
    // setup_ms is the kernel's one-time preparation, excluded from `ms` and reported
    // beside it: host format construction plus device upload for our formats, and
    // descriptor creation + workspace sizing + hipsparseSpMM_preprocess for hipSPARSE.
    // It is taken by reference because hipSPARSE only reports it once its first call has run.
    // convert_ms / upload_ms split setup_ms into host format construction and device
    // upload where a kernel reports them (vcsr_mc*); 0 otherwise.
    auto measure = [&](const char* name, const std::function<void(float&)>& run,
                       const float& setup_ms, float convert_ms = 0.f, float upload_ms = 0.f) {
        auto iterate = [&]() {
            CUDA_CHECK(hipMemsetAsync(dC, 0, bytesC, st));
            CUDA_CHECK(hipStreamSynchronize(st));
            float ms = 0;
            if (csr.nnz) run(ms);
            CUDA_CHECK(hipGetLastError());
            CUDA_CHECK(hipStreamSynchronize(st));
            if (!std::isfinite(ms) || ms < 0) throw std::runtime_error("Invalid kernel timing");
            return ms;
        };
        // A kernel's first launch pays lazy module loading, library initialisation and
        // clock ramp-up. Averaging those in makes the reported time a function of
        // --repeat rather than of the kernel, so warm-up iterations are discarded.
        for (int it = 0; it < warmup; ++it) iterate();
        std::vector<double> times(repeat);
        for (double& t : times) t = iterate();
        double rel_l2 = 0;
        bool failed = false;
        if (check != "none") {
            std::vector<float> result(elementsC);
            CUDA_CHECK(hipMemcpy(result.data(), dC, bytesC, hipMemcpyDeviceToHost));
            double error2 = 0, reference2 = 0;
            bool nonfinite = false;
            for (size_t i = 0; i < result.size(); ++i) {
                if (!std::isfinite(result[i])) {
                    if (!lenient) throw std::runtime_error(std::string(name) + ": non-finite output");
                    nonfinite = true;
                    break;
                }
                const double error = double(result[i]) - reference[i];
                error2 += error * error;
                reference2 += reference[i] * reference[i];
            }
            const double error = std::sqrt(error2), norm = std::sqrt(reference2);
            rel_l2 = norm > 0 ? error / norm : error;
            if (nonfinite) { rel_l2 = INFINITY; failed = true; }
            // FP16-input paths (vcsr_mc, rocsparse_fp16) need the
            // looser tolerance; vcsr_mc_fp32 is held to the FP32 bound.
            const std::string kernel_name(name);
            const bool half = kernel_name == "vcsr_mc" || kernel_name == "rocsparse_fp16";
            const double tolerance = (half ? 3e-3 : 1e-4) * norm + 1e-5 * std::sqrt(double(elementsC));
            // Two-sided: the absolute term exists so a near-zero reference does not
            // demand impossible precision, but on its own it let a kernel that
            // returned all zeros pass at rel_l2 = 1.0 (crystm03, whose reference
            // norm is ~1e-13). Require the relative error to be sane as well.
            if (!nonfinite && (error > tolerance || rel_l2 > (half ? 1e-1 : 1e-2))) {
                if (!lenient)
                    throw std::runtime_error(std::string(name) + ": correctness check failed, rel_l2=" + std::to_string(rel_l2));
                failed = true;
            }
        }
        // Both reductions of the same launches go to the CSV; --stat picks which one is
        // `ms` (and GFLOP/s).
        const double ms_mean = std::accumulate(times.begin(), times.end(), 0.0) / repeat;
        std::sort(times.begin(), times.end());
        const double ms_median = repeat % 2 ? times[repeat / 2] : 0.5 * (times[repeat / 2 - 1] + times[repeat / 2]);
        const double ms = stat == "median" ? ms_median : ms_mean;
        const double gflops = ms > 0 ? (2.0 * csr.nnz * O) / (ms * 1e6) : 0;
        const char* validation = check == "none" ? "unchecked" : failed ? "FAIL" : "pass";
        if (csv)
            printf("%s,%d,%s,%d,%d,%d,%d,%d,%s,%.9g,%.9g,%s,%.9g,%.6g,%.9g,%.9g,%.6g,%.6g\n",
                   csv_quote(device.name).c_str(), device.major*10+device.minor, csv_quote(mtx).c_str(),
                   O, segw, bundle, tileK, repeat, name, ms, gflops, validation, rel_l2, setup_ms, ms_mean, ms_median,
                   convert_ms, upload_ms);
        else
            printf("%-20s %10.6f %12.2f   %s (rel_l2=%.3g, setup %.3f ms)\n",
                   name, ms, gflops, validation, rel_l2, setup_ms);
    };
#ifdef HAVE_ROCSPARSE
    float rocsparse_setup = 0.f;
    if (selected("rocsparse")) measure("rocsparse", [&](float& ms) {
        ms = run_rocsparse_spmm(csr.M, csr.N, csr.nnz, d_rowptr, d_col, d_val, dB, O, dC, &rocsparse_setup);
    }, rocsparse_setup);
    // Mixed precision, matching vcsr_mc: FP16 A values (range-scaled by half_scale,
    // undone through alpha) and FP16 B, FP32 compute and output. The FP16 copies are
    // made once, outside the timed region: convert_ms is the host conversion of A and B,
    // upload_ms their copy to the device, and setup_ms adds the library's own setup.
    if (selected("rocsparse_fp16")) {
        const float scale = half_scale(csr), inv = 1.f / scale;
        std::vector<_Float16> val16, B16;
        const float convert = wall_ms([&] {
            val16.resize(csr.nnz); B16.resize(hB.size());
            for (int i = 0; i < csr.nnz; ++i) val16[i] = (_Float16)(csr.val[i] * inv);
            for (size_t i = 0; i < hB.size(); ++i) B16[i] = (_Float16)hB[i];
        });
        void *d_val16 = nullptr, *dB16 = nullptr;
        const float upload = wall_ms([&] {
            CUDA_CHECK(hipMalloc(&d_val16, sizeof(_Float16) * std::max(csr.nnz, 1)));
            CUDA_CHECK(hipMalloc(&dB16, sizeof(_Float16) * std::max<size_t>(B16.size(), 1)));
            if (csr.nnz) CUDA_CHECK(hipMemcpy(d_val16, val16.data(), sizeof(_Float16) * csr.nnz, hipMemcpyHostToDevice));
            CUDA_CHECK(hipMemcpy(dB16, B16.data(), sizeof(_Float16) * B16.size(), hipMemcpyHostToDevice));
        });
        float lib_setup = 0.f, setup = 0.f;
        measure("rocsparse_fp16", [&](float& ms) {
            ms = run_rocsparse_spmm_fp16(csr.M, csr.N, csr.nnz, d_rowptr, d_col, d_val16, dB16, O, dC,
                                        scale, &lib_setup);
            setup = convert + upload + lib_setup;
        }, setup, convert, upload);
        CUDA_CHECK(hipFree(d_val16)); CUDA_CHECK(hipFree(dB16));
    }
#endif
    // The original VCSR-TC kernel after hipify-perl plus the CDNA adaptations listed in
    // src/vcsr_mc.cu (FP16 operands), and its FP32 twin. bundle = 16; segw 16 (the
    // original's) or 32.
    for (const bool fp32 : {false, true}) {
        const char* name = fp32 ? "vcsr_mc_fp32" : "vcsr_mc";
        if (!selected(name)) continue;
        if (tileK != 16 && tileK != 32 && tileK != 64 && tileK != 128) {
            fprintf(stderr, "Skipping %s: needs tileK in {16,32,64,128}\n", name);
            continue;
        }
        if (segw != 16 && segw != 32 && segw != 48 && segw != 64) {
            fprintf(stderr, "Skipping %s: needs segw in {16,32,48,64}\n", name);
            continue;
        }
        VCSRMC V;
        const float convert = wall_ms([&] { V = csr_to_vcsr_mc(csr, fp32, segw); });
        const float upload = wall_ms([&] { upload_vcsr_mc_device(V); });
        measure(name, [&](float& ms) { run_vcsr_mc_spmm_gpu(V, dB, O, tileK, dC, st, ms); },
                convert + upload, convert, upload);
        destroy_vcsr_mc_device(st);
    }
    CUDA_CHECK(hipStreamDestroy(st));
    CUDA_CHECK(hipFree(dB)); CUDA_CHECK(hipFree(dC));
    CUDA_CHECK(hipFree(d_rowptr)); CUDA_CHECK(hipFree(d_col)); CUDA_CHECK(hipFree(d_val));
    return 0;
}

int main(int argc, char** argv) {
    try { return benchmark_main(argc, argv); }
    catch (const std::exception& e) { fprintf(stderr, "Error: %s\n", e.what()); return 2; }
}
