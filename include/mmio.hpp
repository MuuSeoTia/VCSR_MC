#include "hip/hip_runtime.h"
#pragma once
#include <cmath>
#include <climits>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct COO {
    int M{0}, N{0};
    std::vector<int> rows, cols;
    std::vector<float> vals;
};

// Matrix Market coordinates are one-based; the banner defines value/symmetry rules.
inline COO read_matrix_market_coo(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open MTX: " + path);
    std::string line, magic, object, layout, field, symmetry;
    if (!std::getline(in, line)) throw std::runtime_error("Empty MTX file");
    std::istringstream banner(line);
    banner >> magic >> object >> layout >> field >> symmetry;
    if (magic != "%%MatrixMarket" || object != "matrix" || layout != "coordinate")
        throw std::runtime_error("Expected a Matrix Market coordinate matrix");
    if (field != "real" && field != "integer" && field != "pattern")
        throw std::runtime_error("Unsupported MTX field: " + field);
    if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric")
        throw std::runtime_error("Unsupported MTX symmetry: " + symmetry);
    auto next_line = [&]() {
        while (std::getline(in, line)) {
            const auto first = line.find_first_not_of(" \t\r");
            if (first != std::string::npos && line[first] != '%') return true;
        }
        return false;
    };
    long long m, n, count;
    if (!next_line()) throw std::runtime_error("Missing MTX dimensions");
    std::istringstream dims(line);
    if (!(dims >> m >> n >> count) || m <= 0 || n <= 0 || m >= INT_MAX ||
        n >= INT_MAX || count < 0 || count > INT_MAX)
        throw std::runtime_error("Invalid or unsupported MTX dimensions/nnz");
    if (symmetry != "general" && m != n)
        throw std::runtime_error("Symmetric MTX must be square");
    COO coo;
    coo.M = int(m); coo.N = int(n);
    auto append = [&](int r, int c, float v) {
        if (coo.vals.size() >= size_t(INT_MAX))
            throw std::runtime_error("Expanded MTX exceeds 32-bit nnz limit");
        coo.rows.push_back(r); coo.cols.push_back(c); coo.vals.push_back(v);
    };
    for (long long i = 0; i < count; ++i) {
        if (!next_line()) throw std::runtime_error("Truncated MTX data");
        std::istringstream entry(line);
        long long r, c;
        double value = 1.0;
        if (!(entry >> r >> c) || (field != "pattern" && !(entry >> value)))
            throw std::runtime_error("Bad MTX entry at index " + std::to_string(i));
        std::string extra;
        if (entry >> extra) throw std::runtime_error("Extra fields in MTX entry");
        if (r < 1 || r > m || c < 1 || c > n)
            throw std::runtime_error("MTX row/column index out of range (expected one-based indices)");
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            throw std::runtime_error("MTX value is not finite FP32");
        if (symmetry == "skew-symmetric" && r == c && value != 0)
            throw std::runtime_error("Skew-symmetric MTX diagonal must be zero");
        append(int(r-1), int(c-1), float(value));
        if (symmetry != "general" && r != c)
            append(int(c-1), int(r-1), float(symmetry == "symmetric" ? value : -value));
    }
    if (next_line()) throw std::runtime_error("Extra entries after declared MTX nnz");
    return coo;
}
