#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "mmio.hpp"

struct CSR {
int M{0}, N{0}, nnz{0};
std::vector<int> row_ptr; // size M+1
std::vector<int> col; // size nnz
std::vector<float> val; // size nnz
};


inline CSR coo_to_csr(const COO &coo){
CSR csr; csr.M=coo.M; csr.N=coo.N; csr.nnz=(int)coo.vals.size();
csr.row_ptr.assign(csr.M+1,0); csr.col.resize(csr.nnz); csr.val.resize(csr.nnz);
for(int r: coo.rows) { if(r<0||r>=csr.M) throw std::runtime_error("row idx out of range"); csr.row_ptr[r+1]++; }
for(int i=1;i<=csr.M;++i) csr.row_ptr[i]+=csr.row_ptr[i-1];
std::vector<int> ctr = csr.row_ptr;
for(size_t i=0;i<coo.vals.size();++i){
int r=coo.rows[i]; int dst=ctr[r]++;
if(coo.cols[i]<0 || coo.cols[i]>=csr.N) throw std::runtime_error("column idx out of range");
csr.col[dst]=coo.cols[i]; csr.val[dst]=coo.vals[i];
}

// ensure columns are sorted within each row (helps VCSR packing)
for(int r=0;r<csr.M;++r){
int s=csr.row_ptr[r], e=csr.row_ptr[r+1];
std::vector<int> idx(e-s); std::iota(idx.begin(), idx.end(), 0);
std::sort(idx.begin(), idx.end(), [&](int a,int b){ return csr.col[s+a] < csr.col[s+b]; });
std::vector<int> newc(e-s); std::vector<float> newv(e-s);
for(int i=0;i<e-s;++i){ newc[i]=csr.col[s+idx[i]]; newv[i]=csr.val[s+idx[i]]; }
for(int i=0;i<e-s;++i){ csr.col[s+i]=newc[i]; csr.val[s+i]=newv[i]; }
for(int i=s+1;i<e;++i) {
    if(csr.col[i]==csr.col[i-1])
        throw std::runtime_error("Duplicate MTX coordinates are unsupported; coalesce them before benchmarking");
}
}
return csr;
}