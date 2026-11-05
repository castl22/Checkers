// TtLinalg.cpp -- randomized TT-SVD (rocBLAS GEMM + host Jacobi) for the
// exp_avg "TT + EQ residual" codec.  See TtLinalg.hpp.

#include "checkers/TtLinalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <mutex>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  include <hip/hip_runtime.h>
#  include <rocblas/rocblas.h>
#  define TT_HAVE_ROCBLAS 1
#else
#  define TT_HAVE_ROCBLAS 0
#endif

namespace checkers {
namespace tt {

// ================================================================= //
//  shared BLAS handle
// ================================================================= //
#if TT_HAVE_ROCBLAS
namespace {
std::once_flag        g_blas_once;
rocblas_handle        g_blas = nullptr;
void make_blas() { rocblas_create_handle(&g_blas); }
}
void* blas_handle() {
    std::call_once(g_blas_once, make_blas);
    return reinterpret_cast<void*>(g_blas);
}
void blas_set_stream(void* h, gpu::Stream_t s) {
    rocblas_set_stream(reinterpret_cast<rocblas_handle>(h),
                       reinterpret_cast<hipStream_t>(s));
}
#else
void* blas_handle() { return nullptr; }
void  blas_set_stream(void*, gpu::Stream_t) {}
#endif

// ================================================================= //
//  small device kernels
// ================================================================= //
#if TT_HAVE_ROCBLAS
namespace {

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__global__ void k_hashfill(float* __restrict__ x, size_t n, uint64_t seed) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t h = splitmix64(seed ^ (uint64_t)i);
    // top 24 bits -> [0,1) -> [-1,1)
    const float u = (float)((h >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
    x[i] = 2.0f * u - 1.0f;
}

// dst(rows x rcols) = first rcols columns of src(rows x lcols), row-major
__global__ void k_copy_cols(const float* __restrict__ src, uint32_t rows,
                            uint32_t lcols, uint32_t rcols,
                            float* __restrict__ dst) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tot = (size_t)rows * rcols;
    if (t >= tot) return;
    const uint32_t i = (uint32_t)(t / rcols);
    const uint32_t j = (uint32_t)(t - (size_t)i * rcols);
    dst[t] = src[(size_t)i * lcols + j];
}

__global__ void k_scale_inplace(float* __restrict__ x, size_t n, float c) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= c;
}

// Ms(m x S) = M(m x n) restricted to `nw` evenly spaced windows of `wlen`
// CONTIGUOUS columns each (S = nw*wlen).  Contiguous windows rather than a
// uniform stride so the read stays coalesced: a stride-61 gather of 16 M
// elements touches 16 M cache lines, this one touches wlen/32 per row-window.
__global__ void k_gather_windows(const float* __restrict__ M, uint32_t m,
                                 uint64_t n, uint32_t S, uint32_t wlen,
                                 uint64_t wstep, float* __restrict__ Ms) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * S) return;
    const uint32_t i  = (uint32_t)(t / S);
    const uint32_t j  = (uint32_t)(t - (size_t)i * S);
    const uint32_t wi = j / wlen;
    uint64_t col = (uint64_t)wi * wstep + (j - wi * wlen);
    if (col >= n) col = n - 1;
    Ms[t] = M[(size_t)i * n + col];
}

__global__ void k_absmax(const float* __restrict__ x, size_t n,
                         unsigned int* __restrict__ out) {
    __shared__ unsigned int sh[256];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float m = 0.0f;
    for (size_t k = i; k < n; k += (size_t)blockDim.x * gridDim.x)
        m = fmaxf(m, fabsf(x[k]));
    sh[threadIdx.x] = __float_as_uint(m);            // m >= 0 => bit order == value order
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out, sh[0]);
}

inline uint32_t grid_for(size_t n, uint32_t b) { return (uint32_t)((n + b - 1) / b); }

// ---- one-time workspace (see tt_set_arena) -------------------------------
thread_local TtArena* t_arena = nullptr;
// device allocation: an arena slice when an arena is set, else hipMalloc
bool dmalloc(void** p, size_t bytes) {
    if (t_arena) {
        const size_t at = (t_arena->used + 255) & ~size_t(255);
        if (at + bytes > t_arena->cap) { t_arena->overflows++; *p = nullptr; return false; }
        *p = t_arena->base + at;
        t_arena->used = at + bytes;
        t_arena->peak = std::max(t_arena->peak, t_arena->used);
        return true;
    }
    return hipMalloc(p, bytes) == hipSuccess;
}
void dfree(void* p) { if (!t_arena && p) (void)hipFree(p); }

#define TT_HIP_OK(call) do { hipError_t _e = (call); if (_e != hipSuccess) return false; } while (0)

// ---- row-major GEMM: C(m x n) = alpha * opA(A) * opB(B) + beta * C -----------
// A is (m x k) if !tA else (k x m); B is (k x n) if !tB else (n x k). Row-major.
inline bool gemm_rm(rocblas_handle h, bool tA, bool tB, int m, int n, int k,
                    float alpha, const float* A, const float* B, float beta, float* C) {
    const rocblas_operation op1 = tB ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation op2 = tA ? rocblas_operation_transpose : rocblas_operation_none;
    const int lda = tB ? k : n;      // first operand passed is B
    const int ldb = tA ? m : k;      // second operand passed is A
    const rocblas_status st = rocblas_sgemm(h, op1, op2, n, m, k,
                                            &alpha, B, lda, A, ldb, &beta, C, n);
    return st == rocblas_status_success;
}

// ---- cyclic Jacobi eigensolver for a symmetric d x d matrix (row-major) ------
// In: A (d*d, symmetric, destroyed).  Out: eval[d] ascending-then-we-sort-desc,
// evec[d*d] row-major with column j = eigenvector j.
void jacobi_sym(std::vector<double>& A, int d,
                std::vector<double>& eval, std::vector<double>& evec) {
    evec.assign((size_t)d * d, 0.0);
    for (int i = 0; i < d; ++i) evec[(size_t)i * d + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < d; ++p)
            for (int q = p + 1; q < d; ++q)
                off += A[(size_t)p * d + q] * A[(size_t)p * d + q];
        if (off < 1e-30) break;
        for (int p = 0; p < d; ++p) {
            for (int q = p + 1; q < d; ++q) {
                const double apq = A[(size_t)p * d + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = A[(size_t)p * d + p];
                const double aqq = A[(size_t)q * d + q];
                const double phi = 0.5 * (aqq - app) / apq;
                const double t = (phi >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(phi) + std::sqrt(phi * phi + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int i = 0; i < d; ++i) {
                    const double aip = A[(size_t)i * d + p];
                    const double aiq = A[(size_t)i * d + q];
                    A[(size_t)i * d + p] = c * aip - s * aiq;
                    A[(size_t)i * d + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < d; ++i) {
                    const double api = A[(size_t)p * d + i];
                    const double aqi = A[(size_t)q * d + i];
                    A[(size_t)p * d + i] = c * api - s * aqi;
                    A[(size_t)q * d + i] = s * api + c * aqi;
                }
                for (int i = 0; i < d; ++i) {
                    const double vip = evec[(size_t)i * d + p];
                    const double viq = evec[(size_t)i * d + q];
                    evec[(size_t)i * d + p] = c * vip - s * viq;
                    evec[(size_t)i * d + q] = s * vip + c * viq;
                }
            }
        }
    }
    eval.resize(d);
    for (int i = 0; i < d; ++i) eval[i] = A[(size_t)i * d + i];
}

struct Dev {
    void* p = nullptr;
    ~Dev() { dfree(p); }
    bool alloc(size_t bytes) { return dmalloc(&p, bytes); }
    float* f() { return reinterpret_cast<float*>(p); }
};

// Degenerate fallback: an all-zero sketch has no range, so hand back the single
// orthonormal column e0.  G1 e0^T is still a valid (zero-energy) projector, the
// residual is then the whole shard and the codec round-trips exactly -- whereas
// returning false makes tt_decompose fail and the caller throw.
bool emit_e0(int m, hipStream_t s, float** Qr_out,
             std::vector<double>& sv_out, int* l_out) {
    float* Qr = nullptr;
    if (!dmalloc((void**)&Qr, sizeof(float) * (size_t)m)) return false;
    if (hipMemsetAsync(Qr, 0, sizeof(float) * (size_t)m, s) != hipSuccess) { dfree(Qr); return false; }
    const float one = 1.0f;
    if (hipMemcpyAsync(Qr, &one, sizeof(float), hipMemcpyHostToDevice, s) != hipSuccess) { dfree(Qr); return false; }
    if (hipStreamSynchronize(s) != hipSuccess) { dfree(Qr); return false; }
    sv_out.assign(1, 0.0);
    *Qr_out = Qr;
    *l_out  = 1;
    return true;
}

// ---- APPROXIMATE range finder: one sketch of a COLUMN SAMPLE --------------
//
// range_finder() below is the accurate version and walks the whole m x n matrix
// six times -- Y = (M M^T)^2 M Omega is five passes and B = Q^T M a sixth.  For
// split 1 of a 512 MB exp_avg shard that matrix is 512 x 262144, and rocBLAS
// runs such huge-K / tiny-N shapes far off peak: it measured 11.1 s of the
// 19.4 s exp_avg compress, more than every other stage combined.
//
// A range finder only has to return A subspace, and column sampling preserves
// one: Range(M) ~ Range(M[:, S]) as soon as |S| >> rank.  So sketch ~4k sampled
// columns instead of 262k, drop the power iterations, and skip the B pass.
// Q = orth(M_S Omega) is still EXACTLY orthonormal, so G1 G1^T is still an
// orthogonal projector and the codec's error bound is untouched -- only WHICH
// rank-r subspace is kept changes, and momentum shards are not low-rank in the
// first place (TZ's own README measures 1.72x), so the truncation residual the
// EQ quantizer then sees barely moves.
bool range_finder_fast(rocblas_handle h, hipStream_t s,
                       const float* M, int m, uint64_t n, uint32_t rank_cap,
                       float** Qr_out, std::vector<double>& sv_out, int* l_out) {
    const int p_os = 8;
    const int l = std::min<int>({(int)rank_cap + p_os, m,
                                 (int)std::min<uint64_t>(n, 1u << 20)});
    if (l < 1) return false;
    const uint32_t B = 256;

    // |S| >= 64 * l so the sample is far from rank-deficient, capped at n.
    const uint64_t want  = std::max<uint64_t>(4096u, (uint64_t)l * 64u);
    const uint64_t S64   = std::min<uint64_t>(n, want);
    const uint32_t nw    = (uint32_t)std::min<uint64_t>(32u, S64);
    const uint32_t wlen  = (uint32_t)std::max<uint64_t>(1u, S64 / nw);
    const uint32_t S     = nw * wlen;
    const uint64_t wstep = (nw > 1 && n > wlen) ? (n - wlen) / (nw - 1) : 0;

    Dev Ms, Omega, Y, G, W, Q;
    if (!Ms.alloc(sizeof(float) * (size_t)m * S)) return false;
    k_gather_windows<<<grid_for((size_t)m * S, B), B, 0, s>>>(
        M, (uint32_t)m, n, S, wlen, wstep, Ms.f());

    // Condition the SAMPLE (a few MB), not the matrix: the Gram is formed in
    // fp32 and an orthonormal basis is scale-invariant, so this is the whole
    // reason the old code copied and normalized all P elements of A.
    unsigned int* d_am = nullptr;
    if (!dmalloc((void**)&d_am, sizeof(unsigned int))) return false;
    hipMemsetAsync(d_am, 0, sizeof(unsigned int), s);
    k_absmax<<<std::min<uint32_t>(256u, grid_for((size_t)m * S, B)), B, 0, s>>>(
        Ms.f(), (size_t)m * S, d_am);
    unsigned int h_am = 0;
    if (hipMemcpyAsync(&h_am, d_am, sizeof(unsigned int), hipMemcpyDeviceToHost, s) != hipSuccess) {
        dfree(d_am); return false;
    }
    hipStreamSynchronize(s);
    dfree(d_am);
    const float smax = __builtin_bit_cast(float, h_am);
    if (!(smax > 0.0f)) return emit_e0(m, s, Qr_out, sv_out, l_out);
    k_scale_inplace<<<grid_for((size_t)m * S, B), B, 0, s>>>(
        Ms.f(), (size_t)m * S, 1.0f / smax);

    if (!Omega.alloc(sizeof(float) * (size_t)S * l)) return false;
    if (!Y.alloc(sizeof(float) * (size_t)m * l))     return false;
    k_hashfill<<<grid_for((size_t)S * l, B), B, 0, s>>>(
        Omega.f(), (size_t)S * l, 0x9E37ULL ^ ((uint64_t)m << 20) ^ S);
    if (!gemm_rm(h, false, false, m, l, (int)S, 1.f, Ms.f(), Omega.f(), 0.f, Y.f()))
        return false;

    // G = Y^T Y (l x l) -> Jacobi -> Q = Y V diag(1/sqrt(lam)), truncated to the
    // numerically effective rank (see range_finder: the sketch is deliberately
    // rank-deficient, and forming Q over near-zero lam produces garbage columns).
    if (!G.alloc(sizeof(float) * (size_t)l * l)) return false;
    if (!gemm_rm(h, true, false, l, l, m, 1.f, Y.f(), Y.f(), 0.f, G.f())) return false;
    TT_HIP_OK(hipStreamSynchronize(s));
    std::vector<float> hGf((size_t)l * l);
    TT_HIP_OK(hipMemcpy(hGf.data(), G.p, sizeof(float) * (size_t)l * l, hipMemcpyDeviceToHost));
    std::vector<double> hG((size_t)l * l), ev, evec;
    for (size_t i = 0; i < hG.size(); ++i) hG[i] = (double)hGf[i];
    jacobi_sym(hG, l, ev, evec);

    std::vector<int> gord(l);
    for (int i = 0; i < l; ++i) gord[i] = i;
    std::sort(gord.begin(), gord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    const double lmax = ev[gord[0]];
    if (!(lmax > 0.0)) return emit_e0(m, s, Qr_out, sv_out, l_out);
    int le = 0;
    while (le < l && ev[gord[le]] > lmax * 1e-9) ++le;
    if (le < 1) le = 1;

    std::vector<float> hW((size_t)l * le);
    sv_out.resize(le);
    for (int j = 0; j < le; ++j) sv_out[j] = std::sqrt(ev[gord[j]]);
    for (int i = 0; i < l; ++i)
        for (int j = 0; j < le; ++j)
            hW[(size_t)i * le + j] = (float)(evec[(size_t)i * l + gord[j]] /
                                             std::sqrt(ev[gord[j]]));
    if (!W.alloc(sizeof(float) * (size_t)l * le)) return false;
    TT_HIP_OK(hipMemcpy(W.p, hW.data(), sizeof(float) * (size_t)l * le, hipMemcpyHostToDevice));

    float* Qr = nullptr;
    if (!dmalloc((void**)&Qr, sizeof(float) * (size_t)m * le)) return false;
    if (!gemm_rm(h, false, false, m, le, l, 1.f, Y.f(), W.f(), 0.f, Qr)) { dfree(Qr); return false; }
    TT_HIP_OK(hipStreamSynchronize(s));
    *Qr_out = Qr;
    *l_out  = le;
    return true;
}

// Randomized range finder.  M is (m x n) row-major fp32 on device.
// Returns Qr (m x le) with columns = leading left singular vectors (descending),
// and sv[le] = singular value estimates (descending), where le <= rank_cap+8 is
// the numerically-effective rank of the sketch.  Caller frees Qr.
//
// Orthonormalization is Gram + Jacobi with an EFFECTIVE-RANK TRUNCATION: the
// sketch Y is deliberately rank-deficient (we oversample past the true rank), so
// forming Q = Y V diag(1/sqrt(lam)) over ALL l directions amplifies the near-zero
// lam into garbage columns.  We keep only lam_j > lam_max * tol, which both
// stabilizes Q and does the rank truncation for free.
bool range_finder(rocblas_handle h, hipStream_t s,
                  const float* M, int m, int n, uint32_t rank_cap,
                  float** Qr_out, std::vector<double>& sv_out, int* l_out) {
    const int p_os = 8;
    const int l = std::min<int>({(int)rank_cap + p_os, m, n});
    if (l < 1) return false;
    const uint32_t B = 256;

    Dev Omega, Y, TmpNL, G, W, Q, Bm, BBt, Ub;
    if (!Omega.alloc(sizeof(float) * (size_t)n * l)) return false;
    if (!Y.alloc(sizeof(float) * (size_t)m * l))     return false;
    if (!TmpNL.alloc(sizeof(float) * (size_t)n * l)) return false;

    k_hashfill<<<grid_for((size_t)n * l, B), B, 0, s>>>(Omega.f(), (size_t)n * l, 0x9E37ULL ^ ((uint64_t)m << 20) ^ n);

    // Y = (M M^T)^2 M Omega   (2 power iterations for a sharper subspace)
    if (!gemm_rm(h, false, false, m, l, n, 1.f, M, Omega.f(), 0.f, Y.f())) return false;
    for (int it = 0; it < 2; ++it) {
        if (!gemm_rm(h, true,  false, n, l, m, 1.f, M, Y.f(), 0.f, TmpNL.f())) return false;
        if (!gemm_rm(h, false, false, m, l, n, 1.f, M, TmpNL.f(), 0.f, Y.f())) return false;
    }

    // G = Y^T Y (l x l)  ->  Jacobi
    if (!G.alloc(sizeof(float) * (size_t)l * l)) return false;
    if (!gemm_rm(h, true, false, l, l, m, 1.f, Y.f(), Y.f(), 0.f, G.f())) return false;
    TT_HIP_OK(hipStreamSynchronize(s));
    std::vector<float> hGf((size_t)l * l);
    TT_HIP_OK(hipMemcpy(hGf.data(), G.p, sizeof(float) * (size_t)l * l, hipMemcpyDeviceToHost));
    std::vector<double> hG((size_t)l * l), ev, evec;
    for (size_t i = 0; i < hG.size(); ++i) hG[i] = (double)hGf[i];
    jacobi_sym(hG, l, ev, evec);

    // effective rank: keep lam_j > lam_max * tol, descending
    std::vector<int> gord(l);
    for (int i = 0; i < l; ++i) gord[i] = i;
    std::sort(gord.begin(), gord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    double lmax = ev[gord[0]];
    if (!(lmax > 0.0)) return emit_e0(m, s, Qr_out, sv_out, l_out);  // Y is all zeros
    int le = 0;
    while (le < l && ev[gord[le]] > lmax * 1e-9) ++le;
    if (le < 1) le = 1;

    // W (l x le) = V[:,top-le] * diag(1/sqrt(lam))
    std::vector<float> hW((size_t)l * le);
    for (int i = 0; i < l; ++i)
        for (int j = 0; j < le; ++j)
            hW[(size_t)i * le + j] = (float)(evec[(size_t)i * l + gord[j]] /
                                             std::sqrt(ev[gord[j]]));
    if (!W.alloc(sizeof(float) * (size_t)l * le)) return false;
    TT_HIP_OK(hipMemcpy(W.p, hW.data(), sizeof(float) * (size_t)l * le, hipMemcpyHostToDevice));
    if (!Q.alloc(sizeof(float) * (size_t)m * le)) return false;
    // Q = Y * W  (m x le, orthonormal columns)
    if (!gemm_rm(h, false, false, m, le, l, 1.f, Y.f(), W.f(), 0.f, Q.f())) return false;

    // B = Q^T M (le x n) ; BBt = B B^T (le x le) ; Jacobi -> singular directions
    if (!Bm.alloc(sizeof(float) * (size_t)le * n)) return false;
    if (!gemm_rm(h, true,  false, le, n, m, 1.f, Q.f(), M, 0.f, Bm.f())) return false;
    if (!BBt.alloc(sizeof(float) * (size_t)le * le)) return false;
    if (!gemm_rm(h, false, true, le, le, n, 1.f, Bm.f(), Bm.f(), 0.f, BBt.f())) return false;
    TT_HIP_OK(hipStreamSynchronize(s));
    std::vector<float> hBBtf((size_t)le * le);
    TT_HIP_OK(hipMemcpy(hBBtf.data(), BBt.p, sizeof(float) * (size_t)le * le, hipMemcpyDeviceToHost));
    std::vector<double> hBBt((size_t)le * le), mu, Ubv;
    for (size_t i = 0; i < hBBt.size(); ++i) hBBt[i] = (double)hBBtf[i];
    jacobi_sym(hBBt, le, mu, Ubv);

    std::vector<int> ord(le);
    for (int i = 0; i < le; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return mu[a] > mu[b]; });
    sv_out.resize(le);
    std::vector<float> hUb((size_t)le * le);
    for (int j = 0; j < le; ++j) {
        sv_out[j] = std::sqrt(std::max(mu[ord[j]], 0.0));
        for (int i = 0; i < le; ++i)
            hUb[(size_t)i * le + j] = (float)Ubv[(size_t)i * le + ord[j]];
    }
    if (!Ub.alloc(sizeof(float) * (size_t)le * le)) return false;
    TT_HIP_OK(hipMemcpy(Ub.p, hUb.data(), sizeof(float) * (size_t)le * le, hipMemcpyHostToDevice));

    float* Qr = nullptr;
    if (!dmalloc((void**)&Qr, sizeof(float) * (size_t)m * le)) return false;
    if (!gemm_rm(h, false, false, m, le, le, 1.f, Q.f(), Ub.f(), 0.f, Qr)) { dfree(Qr); return false; }
    TT_HIP_OK(hipStreamSynchronize(s));
    *Qr_out = Qr;
    *l_out  = le;
    return true;
}

uint32_t adaptive_rank(const std::vector<double>& sv, double energy_keep, uint32_t cap) {
    double tot = 0.0;
    for (double v : sv) tot += v * v;
    if (tot <= 0.0) return 1u;
    double acc = 0.0;
    for (size_t k = 0; k < sv.size(); ++k) {
        acc += sv[k] * sv[k];
        if (acc >= energy_keep * tot)
            return std::min<uint32_t>((uint32_t)k + 1u, cap);
    }
    return std::min<uint32_t>((uint32_t)sv.size(), cap);
}

} // namespace
#endif // TT_HAVE_ROCBLAS

// ================================================================= //
//  public API
// ================================================================= //
TtDims choose_dims(size_t L, uint32_t ndim) {
    TtDims d;
    if (L == 0) return d;
    if (ndim <= 2) {
        // near-square n1*n2 >= L
        uint32_t n2 = (uint32_t)std::llround(std::sqrt((double)L));
        n2 = std::max<uint32_t>(n2, 1u);
        uint32_t n1 = (uint32_t)((L + n2 - 1) / n2);
        d.n1 = std::max<uint32_t>(n1, 1u); d.n2 = n2; d.n3 = 1u;
        return d;
    }
    uint32_t c = (uint32_t)std::llround(std::cbrt((double)L));
    c = std::max<uint32_t>(c, 1u);
    // n1*n2 = c*c, n3 grows to cover L
    d.n1 = c; d.n2 = c;
    d.n3 = (uint32_t)(((uint64_t)L + (uint64_t)c * c - 1) / ((uint64_t)c * c));
    d.n3 = std::max<uint32_t>(d.n3, 1u);
    return d;
}

void tt_free(TtCores& c) {
#if TT_HAVE_ROCBLAS
    if (!c.arena_owned) {                 // arena slices are not ours to free
        if (c.g1) hipFree(c.g1);
        if (c.g2) hipFree(c.g2);
        if (c.g3) hipFree(c.g3);
    }
#endif
    c = TtCores{};
}

#if TT_HAVE_ROCBLAS
bool tt_decompose(const float* A, const TtDims& dims,
                  uint32_t rank_cap, double energy_keep,
                  TtCores& out, void* blas, gpu::Stream_t stream,
                  TtProf* prof) {
    auto h = reinterpret_cast<rocblas_handle>(blas);
    auto s = reinterpret_cast<hipStream_t>(stream);
    rocblas_set_stream(h, s);
    using tt_clk = std::chrono::steady_clock;
    auto tmark = [&](tt_clk::time_point t0) {
        hipStreamSynchronize(s);
        return std::chrono::duration<double, std::milli>(tt_clk::now() - t0).count();
    };
    auto t_stage = tt_clk::now();

    const uint32_t n1 = dims.n1, n2 = dims.n2, n3 = dims.n3;
    const uint64_t P  = dims.padded();
    const uint32_t B  = 256;
    out = TtCores{};
    out.n1 = n1; out.n2 = n2; out.n3 = n3;
    out.arena_owned = (t_arena != nullptr);
    if (P == 0 || rank_cap == 0) return false;

    // ---- split 1:  M1 = A  as  n1 x (n2*n3), sketched from a column sample ----
    //
    // There is no O(1)-normalized copy of A any more.  It cost a P-element
    // device buffer plus a full read+write pass (4 GB of traffic on a 512 MB
    // shard) solely to condition an l x l Gram, and range_finder_fast now
    // normalizes its own few-MB sample instead.  G1 comes out orthonormal
    // either way -- an orthonormal basis is scale-invariant -- so A is read
    // directly and the cores are unchanged in meaning and in format.
    const uint64_t cols1 = (uint64_t)n2 * n3;
    float* Qr1 = nullptr; std::vector<double> sv1; int l1 = 0;
    if (!range_finder_fast(h, s, A, (int)n1, cols1, rank_cap, &Qr1, sv1, &l1)) return false;
    const uint32_t r1 = adaptive_rank(sv1, energy_keep, rank_cap);
    out.r1 = r1;
    if (!dmalloc((void**)&out.g1, sizeof(float) * (size_t)n1 * r1)) { dfree(Qr1); return false; }
    k_copy_cols<<<grid_for((size_t)n1 * r1, B), B, 0, s>>>(Qr1, n1, (uint32_t)l1, r1, out.g1);
    dfree(Qr1);
    if (prof) { prof->sketch1_ms = tmark(t_stage); t_stage = tt_clk::now(); }

    Dev C;
    if (!C.alloc(sizeof(float) * (size_t)r1 * cols1)) return false;
    if (prof) { prof->alloc_ms += tmark(t_stage); t_stage = tt_clk::now(); }
    // C = G1^T * M1   (r1 x cols1) -- the one remaining full pass over A
    if (!gemm_rm(h, true, false, (int)r1, (int)cols1, (int)n1, 1.f, out.g1, A, 0.f, C.f())) return false;
    if (prof) { prof->gemm_c_ms = tmark(t_stage); t_stage = tt_clk::now(); }

    // Scale C to O(1) for split 2.  This is the same conditioning the old
    // whole-tensor normalization gave, but C is r1/n1 of the size (16 MB vs
    // 512 MB) so the two passes are free, and g3 is scaled back at the end.
    float scale = 1.0f;
    {
        const size_t cn = (size_t)r1 * cols1;
        unsigned int* d_am = nullptr;
        if (!dmalloc((void**)&d_am, sizeof(unsigned int))) return false;
        hipMemsetAsync(d_am, 0, sizeof(unsigned int), s);
        k_absmax<<<std::min<uint32_t>(256u, grid_for(cn, B)), B, 0, s>>>(C.f(), cn, d_am);
        unsigned int h_am = 0;
        if (hipMemcpyAsync(&h_am, d_am, sizeof(unsigned int), hipMemcpyDeviceToHost, s) != hipSuccess) {
            dfree(d_am); return false;
        }
        hipStreamSynchronize(s);
        dfree(d_am);
        const float cmax = __builtin_bit_cast(float, h_am);
        if (cmax > 0.0f) {
            scale = cmax;
            k_scale_inplace<<<grid_for(cn, B), B, 0, s>>>(C.f(), cn, 1.0f / cmax);
        }
    }

    if (prof) { prof->scale_ms = tmark(t_stage); t_stage = tt_clk::now(); }

    // ---- split 2:  M2 = C  as  (r1*n2) x n3 ----
    const int rows2 = (int)((uint64_t)r1 * n2);
    float* Qr2 = nullptr; std::vector<double> sv2; int l2 = 0;
    if (!range_finder(h, s, C.f(), rows2, (int)n3, rank_cap, &Qr2, sv2, &l2)) return false;
    const uint32_t r2 = adaptive_rank(sv2, energy_keep, rank_cap);
    out.r2 = r2;
    if (!dmalloc((void**)&out.g2, sizeof(float) * (size_t)rows2 * r2)) { dfree(Qr2); return false; }
    k_copy_cols<<<grid_for((size_t)rows2 * r2, B), B, 0, s>>>(Qr2, (uint32_t)rows2, (uint32_t)l2, r2, out.g2);
    dfree(Qr2);

    if (prof) { prof->sketch2_ms = tmark(t_stage); t_stage = tt_clk::now(); }
    if (!dmalloc((void**)&out.g3, sizeof(float) * (size_t)r2 * n3)) return false;
    // G3 = G2^T * M2   (r2 x n3)
    if (!gemm_rm(h, true, false, (int)r2, (int)n3, rows2, 1.f, out.g2, C.f(), 0.f, out.g3)) return false;
    // undo the O(1) normalization
    k_scale_inplace<<<grid_for((size_t)r2 * n3, B), B, 0, s>>>(out.g3, (size_t)r2 * n3, scale);
    if (hipStreamSynchronize(s) != hipSuccess) return false;
    if (prof) prof->gemm_g3_ms = tmark(t_stage);
    return true;
}

size_t tt_reconstruct_ws_floats(const TtCores& c) {
    return static_cast<size_t>(c.n1) * c.n2 * c.r2;
}

bool tt_reconstruct_ws(const TtCores& c, float* A_hat, float* t1_ws, void* blas, gpu::Stream_t stream) {
    auto h = reinterpret_cast<rocblas_handle>(blas);
    auto s = reinterpret_cast<hipStream_t>(stream);
    rocblas_set_stream(h, s);
    const uint32_t n1 = c.n1, n2 = c.n2, n3 = c.n3, r1 = c.r1, r2 = c.r2;
    if (!n1 || !n2 || !n3 || !r1 || !r2 || !t1_ws) return false;
    // T1 = G1(n1 x r1) * reshape(G2, r1 x (n2*r2))   -> (n1 x n2*r2)
    if (!gemm_rm(h, false, false, (int)n1, (int)((uint64_t)n2 * r2), (int)r1,
                 1.f, c.g1, c.g2, 0.f, t1_ws)) return false;
    // A_hat = reshape(T1, (n1*n2) x r2) * G3(r2 x n3)  -> (n1*n2 x n3) = P
    return gemm_rm(h, false, false, (int)((uint64_t)n1 * n2), (int)n3, (int)r2,
                   1.f, t1_ws, c.g3, 0.f, A_hat);
}

bool tt_reconstruct(const TtCores& c, float* A_hat, void* blas, gpu::Stream_t stream) {
    if (!c.n1 || !c.n2 || !c.n3 || !c.r1 || !c.r2) return false;
    Dev T1;
    if (!T1.alloc(sizeof(float) * tt_reconstruct_ws_floats(c))) return false;
    if (!tt_reconstruct_ws(c, A_hat, T1.f(), blas, stream)) return false;
    return hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)) == hipSuccess;
}
void tt_set_arena(TtArena* a) { t_arena = a; }

// Mirrors the allocations of tt_decompose (range_finder_fast on split 1,
// range_finder on split 2, cores, C, absmax cells, the e0 fallbacks) and
// tt_reconstruct's T1, with every rank at its cap and every effective rank
// at its maximum l -- an upper bound, since the arena never frees.
size_t tt_arena_bytes(const TtDims& d, uint32_t cap) {
    size_t tot = 0;
    auto add = [&](uint64_t floats) { tot += ((floats * 4 + 255) & ~uint64_t(255)) + 256; };
    const uint64_t n1 = d.n1, n2 = d.n2, n3 = d.n3, cols1 = n2 * n3, c = cap;
    // split 1: range_finder_fast(m = n1, n = cols1)
    const uint64_t l1 = std::min<uint64_t>({c + 8, n1, std::min<uint64_t>(cols1, 1u << 20)});
    const uint64_t S64 = std::min<uint64_t>(cols1, std::max<uint64_t>(4096u, l1 * 64u));
    const uint64_t nw = std::min<uint64_t>(32u, S64), wlen = std::max<uint64_t>(1u, S64 / std::max<uint64_t>(nw, 1));
    const uint64_t S = nw * wlen;
    add(n1 * S); add(1); add(S * l1); add(n1 * l1); add(l1 * l1); add(l1 * l1); add(n1 * l1); add(n1);
    add(n1 * c);            // g1
    add(c * cols1);         // C
    add(1);                 // absmax cell
    // split 2: range_finder(m = rows2, n = n3)
    const uint64_t rows2 = c * n2;
    const uint64_t l2 = std::min<uint64_t>({c + 8, rows2, n3});
    add(n3 * l2); add(rows2 * l2); add(n3 * l2); add(l2 * l2); add(l2 * l2); add(rows2 * l2);
    add(l2 * n3); add(l2 * l2); add(l2 * l2); add(rows2 * l2); add(rows2);
    add(rows2 * c);         // g2
    add(c * n3);            // g3
    add(n1 * n2 * c);       // tt_reconstruct's T1
    return tot;
}

size_t tt_blas_workspace_bytes(const TtDims& d, uint32_t cap) {
    auto h = reinterpret_cast<rocblas_handle>(blas_handle());
    if (!h) return 0;
    void* dummy = nullptr;                       // query mode never touches the operands
    if (hipMalloc(&dummy, 256) != hipSuccess) return 0;
    const float* D = static_cast<const float*>(dummy);
    float* Dw = static_cast<float*>(dummy);
    if (rocblas_start_device_memory_size_query(h) != rocblas_status_success) { hipFree(dummy); return 0; }
    const int n1 = (int)d.n1, n2 = (int)d.n2, n3 = (int)d.n3, c = (int)cap;
    const int cols1 = (int)((uint64_t)d.n2 * d.n3);
    const int l1 = (int)std::min<uint64_t>({(uint64_t)c + 8, (uint64_t)n1, std::min<uint64_t>(cols1, 1u << 20)});
    const int S = (int)std::min<uint64_t>(cols1, std::max<uint64_t>(4096u, (uint64_t)l1 * 64u));
    const int rows2 = c * n2;
    const int l2 = std::min({c + 8, rows2, n3});
    // the GEMM shapes of tt_decompose + tt_reconstruct, effective ranks at max
    (void)gemm_rm(h, false, false, n1, l1, S, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, l1, l1, n1, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, false, n1, l1, l1, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, c, cols1, n1, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, false, rows2, l2, n3, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, n3, l2, rows2, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, l2, l2, rows2, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, false, rows2, l2, l2, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, l2, n3, rows2, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, true,  l2, l2, n3, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, true,  false, c, n3, rows2, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, false, n1, n2 * c, c, 1.f, D, D, 0.f, Dw);
    (void)gemm_rm(h, false, false, n1 * n2, n3, c, 1.f, D, D, 0.f, Dw);
    size_t bytes = 0;
    if (rocblas_stop_device_memory_size_query(h, &bytes) != rocblas_status_success) bytes = 0;
    hipFree(dummy);
    return bytes;
}

bool tt_blas_fix_workspace(size_t bytes) {
    auto h = reinterpret_cast<rocblas_handle>(blas_handle());
    if (!h) return false;
    // size 0 would hand memory management back to rocBLAS (grow on demand)
    return rocblas_set_device_memory_size(h, std::max<size_t>(bytes, 4096)) == rocblas_status_success;
}
#else  // no rocBLAS -> codec disabled, caller falls back
bool tt_decompose(const float*, const TtDims&, uint32_t, double,
                  TtCores&, void*, gpu::Stream_t) { return false; }
bool tt_reconstruct(const TtCores&, float*, void*, gpu::Stream_t) { return false; }
size_t tt_reconstruct_ws_floats(const TtCores&) { return 0; }
void   tt_set_arena(TtArena*) {}
size_t tt_arena_bytes(const TtDims&, uint32_t) { return 0; }
size_t tt_blas_workspace_bytes(const TtDims&, uint32_t) { return 0; }
bool   tt_blas_fix_workspace(size_t) { return false; }
bool tt_reconstruct_ws(const TtCores&, float*, float*, void*, gpu::Stream_t) { return false; }
#endif

} // namespace tt
} // namespace checkers
