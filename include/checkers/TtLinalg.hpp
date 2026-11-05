#pragma once
// TtLinalg.hpp -- GPU linear algebra for the exp_avg "TT + EQ residual" codec.
//
// Pipeline shape follows TZ (decompose -> quantize residual -> lossless) but the
// pieces are ours: a randomized range finder (rocBLAS GEMM + host Jacobi on the
// tiny l x l Gram matrices, no rocSOLVER) drives a 3-core Tensor-Train SVD with
// the bond dimensions HARD-CAPPED (rank_cap) so the reshape is low-rank by
// construction; the truncation residual is handed to the error-bounded EQ
// quantizer (see CheckpointCompressor).
//
// Split 1 -- the only unfolding whose matrix is the size of the shard -- uses
// an APPROXIMATE range finder: one sketch of a few thousand sampled columns,
// no power iterations, no second projection pass.  G1 stays exactly orthonormal
// so the codec's error bound is unaffected; only which rank-r subspace is kept
// changes.  See range_finder_fast in TtLinalg.cpp for why that is the right
// trade on optimizer momentum.
//
//   forward   A(P) --reshape n1 x (n2 n3)--> rSVD(cap) -> G1(n1 x r1), C(r1 x n2n3)
//             C --reshape (r1 n2) x n3--> rSVD(cap) -> G2(r1n2 x r2), G3(r2 x n3)
//   inverse   T1 = G1 . reshape(G2, r1, n2 r2)          (n1 x n2 r2)
//             Ahat = reshape(T1, n1 n2, r2) . G3        (n1 n2 x n3) = P
//
// All matrices are ROW-MAJOR fp32 on the device.  One shared rocblas_handle,
// used under the compressor's codec mutex; set its stream per call.

#include <cstddef>
#include <cstdint>

#include "checkers/GpuDefs.hpp"

namespace checkers {
namespace tt {

// Opaque handle to the process-wide rocblas/cublas handle (lazily created).
void*  blas_handle();
void   blas_set_stream(void* handle, gpu::Stream_t stream);

// ------------------------------------------------------------------ //
//  dims: factor L into a near-cube n1*n2*n3 >= L  (ndim==2 => n3==1)
// ------------------------------------------------------------------ //
struct TtDims {
    uint32_t n1 = 1, n2 = 1, n3 = 1;
    uint64_t padded() const { return uint64_t(n1) * n2 * n3; }
};
TtDims choose_dims(size_t n_elems, uint32_t ndim);

// ------------------------------------------------------------------ //
//  TT cores (device-owned; free with tt_free)
// ------------------------------------------------------------------ //
struct TtCores {
    float*   g1 = nullptr;   // n1 x r1
    float*   g2 = nullptr;   // (r1*n2) x r2
    float*   g3 = nullptr;   // r2 x n3
    uint32_t n1 = 0, n2 = 0, n3 = 0;
    uint32_t r1 = 0, r2 = 0;
    bool     arena_owned = false;   // g1..g3 are TtArena slices: tt_free must not hipFree them
    uint64_t core_elems() const {
        return uint64_t(n1) * r1 + uint64_t(r1) * n2 * r2 + uint64_t(r2) * n3;
    }
};
void tt_free(TtCores& c);

// ------------------------------------------------------------------ //
//  One-time device workspace (2026-11-05)
// ------------------------------------------------------------------ //
// While an arena is set on the calling thread, every device allocation inside
// tt_decompose / tt_reconstruct is a bump-allocated slice of it and every free
// is a no-op, so a decomposition does no hipMalloc / hipFree at all.  The
// caller resets `used` before each tensor.  An allocation that does not fit
// fails (the codec then falls back to raw) -- it never falls back to hipMalloc.
// No arena set (the default) = the old hipMalloc / hipFree behaviour.
struct TtArena {
    uint8_t* base = nullptr;
    size_t   cap  = 0;
    size_t   used = 0;
    size_t   peak = 0;
    uint64_t overflows = 0;
};
void tt_set_arena(TtArena* a);     // this thread; nullptr = off
// Upper bound on the arena bytes one tt_decompose + tt_reconstruct of these
// dims at this rank cap can take.
size_t tt_arena_bytes(const TtDims& dims, uint32_t rank_cap);
// rocBLAS device workspace for every GEMM shape tt_decompose / tt_reconstruct
// runs for these dims (rocBLAS size query, no compute).  0 if unavailable.
size_t tt_blas_workspace_bytes(const TtDims& dims, uint32_t rank_cap);
// Give the shared rocBLAS handle a FIXED workspace of this size, allocated
// now; rocBLAS then never allocates or frees during a GEMM (a GEMM that would
// need more fails instead).  Returns false on error.
bool tt_blas_fix_workspace(size_t bytes);

// Per-stage timings for tt_decompose, in ms.  tt_decompose is 93% of the
// exp_avg stream's GPU time, so it gets attributed rather than guessed at.
struct TtProf {
    double sketch1_ms = 0;   // split-1 sampled range finder (gather + GEMM + Jacobi)
    double gemm_c_ms  = 0;   // C = G1^T A   -- the one full pass over the shard
    double scale_ms   = 0;   // absmax(C) + rescale
    double sketch2_ms = 0;   // split-2 range finder on C (power iterations)
    double gemm_g3_ms = 0;   // G3 = G2^T C
    double alloc_ms   = 0;   // hipMalloc/hipFree inside the decomposition
};

// Decompose a zero-padded fp32 device buffer A (exactly dims.padded() elements)
// into 3 TT cores with every bond <= rank_cap and, within that cap, the smallest
// rank whose kept Frobenius energy >= energy_keep.  Returns false on OOM / bad
// dims (caller then falls back to a lossless codec).  A is READ ONLY.
bool tt_decompose(const float* A, const TtDims& dims,
                  uint32_t rank_cap, double energy_keep,
                  TtCores& out, void* blas, gpu::Stream_t stream,
                  TtProf* prof = nullptr);

// Reconstruct into A_hat (must hold dims.padded() elements). A_hat may not alias
// any core. Uses 2 GEMMs.
bool tt_reconstruct(const TtCores& c, float* A_hat,
                    void* blas, gpu::Stream_t stream);

// Same, with the n1*n2*r2-float intermediate in caller-owned device memory
// (t1_ws), and NO allocation, free or synchronization: the caller syncs the
// stream.  For the restore path, which must not hipMalloc/hipFree per tensor
// (2026-11-05: a hipFree hung there with the GPU idle).
size_t tt_reconstruct_ws_floats(const TtCores& c);   // n1 * n2 * r2
bool tt_reconstruct_ws(const TtCores& c, float* A_hat, float* t1_ws,
                       void* blas, gpu::Stream_t stream);

} // namespace tt
} // namespace checkers
