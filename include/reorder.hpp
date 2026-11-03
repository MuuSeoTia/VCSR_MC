#include "hip/hip_runtime.h"
#pragma once
#include <algorithm>
#include <deque>
#include <vector>
#include "csr.hpp"

// Reverse Cuthill-McKee ordering of the symmetrised sparsity pattern.
//
// Applied to columns only, this clusters the columns a given row touches into a
// narrow band, which is what makes column segmentation pay off: a row whose nonzeros
// are spread over many segments forces one staged B tile per segment. Measured on
// cop20k_A at segw=32 this cuts segments per row from 10.5 to 6.7; on an already
// banded matrix such as bcsstk17 it is a no-op, as expected.
//
// Returns new_position[old_column].
inline std::vector<int> rcm_column_order(const CSR &csr) {
    const int n = csr.N;
    std::vector<int> degree(n, 0);
    // Symmetrise the pattern: an edge exists between columns j1, j2 of the graph
    // whenever A has a nonzero at (j1, j2) or (j2, j1).
    std::vector<int> head(n + 1, 0);
    for (int r = 0; r < csr.M; ++r)
        for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
            const int c = csr.col[p];
            if (r < n) ++head[r + 1];
            ++head[c + 1];
        }
    for (int i = 0; i < n; ++i) head[i + 1] += head[i];
    std::vector<int> adj(head[n]);
    std::vector<int> fill(head.begin(), head.begin() + n);
    for (int r = 0; r < csr.M; ++r)
        for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
            const int c = csr.col[p];
            if (r < n) adj[fill[r]++] = c;
            adj[fill[c]++] = (r < n) ? r : c;
        }
    for (int i = 0; i < n; ++i) {
        auto b = adj.begin() + head[i], e = adj.begin() + head[i + 1];
        std::sort(b, e);
        degree[i] = int(std::unique(b, e) - b);
    }

    std::vector<char> seen(n, 0);
    std::vector<int> order;
    order.reserve(n);
    std::vector<int> roots(n);
    for (int i = 0; i < n; ++i) roots[i] = i;
    std::stable_sort(roots.begin(), roots.end(),
                     [&](int a, int b) { return degree[a] < degree[b]; });

    std::deque<int> queue;
    std::vector<int> neighbours;
    for (int root : roots) {
        if (seen[root]) continue;
        seen[root] = 1;
        queue.push_back(root);
        while (!queue.empty()) {
            const int v = queue.front();
            queue.pop_front();
            order.push_back(v);
            neighbours.clear();
            for (int k = head[v]; k < head[v] + degree[v]; ++k) {
                const int u = adj[k];
                if (!seen[u]) { seen[u] = 1; neighbours.push_back(u); }
            }
            std::sort(neighbours.begin(), neighbours.end(),
                      [&](int a, int b) { return degree[a] < degree[b]; });
            for (int u : neighbours) queue.push_back(u);
        }
    }
    std::reverse(order.begin(), order.end());

    std::vector<int> position(n);
    for (int i = 0; i < int(order.size()); ++i) position[order[i]] = i;
    return position;
}

// Relabel A's columns by `position` and permute B's rows to match, so that
// C = A*B is unchanged: (A P)(P^T B) = A B. Row order of A (and therefore of C) is
// untouched, so validation and output layout are unaffected.
inline void apply_column_permutation(CSR &csr, std::vector<float> &hB, int O,
                                     const std::vector<int> &position) {
    for (int &c : csr.col) c = position[c];
    // Column indices must stay sorted within a row: rocSPARSE requires it, and the
    // VCSR builders assume it when packing a segment's entries by depth.
    std::vector<int> idx;
    std::vector<int> newc;
    std::vector<float> newv;
    for (int r = 0; r < csr.M; ++r) {
        const int s = csr.row_ptr[r], e = csr.row_ptr[r + 1], len = e - s;
        if (len < 2) continue;
        idx.resize(len); newc.resize(len); newv.resize(len);
        for (int i = 0; i < len; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return csr.col[s + a] < csr.col[s + b]; });
        for (int i = 0; i < len; ++i) { newc[i] = csr.col[s + idx[i]]; newv[i] = csr.val[s + idx[i]]; }
        for (int i = 0; i < len; ++i) { csr.col[s + i] = newc[i]; csr.val[s + i] = newv[i]; }
    }
    if (hB.empty()) return;
    std::vector<float> permuted(hB.size());
    for (int old_row = 0; old_row < csr.N; ++old_row) {
        const float* src = hB.data() + size_t(old_row) * O;
        float* dst = permuted.data() + size_t(position[old_row]) * O;
        std::copy(src, src + O, dst);
    }
    hB.swap(permuted);
}
