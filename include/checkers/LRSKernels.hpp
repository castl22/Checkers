#pragma once
// include/checkers/LRSKernels.hpp
//
// LRS-C  (Layer-wise Rank-Structured Compression)  —  GPU kernel declarations.
//
// Three-stage pipeline per 2-D float32 optimizer tensor:
//
//   Stage 1  Decompose  (shard-local, no cross-rank communication)
//     master_weight, exp_avg:   W  =  a_i·1ᵀ  +  1·bⱼᵀ  +  R
//     exp_avg_sq:               log(V+δ)  =  a_i  +  bⱼ  +  R   (log-normal)
//
//   Stage 2  Quantize  (distribution-specific, error-driven variable bit depth)
//     master_weight   Gaussian    8-bit nominal  clip ±4σ
//     exp_avg         Laplacian   6-bit nominal  clip ±8b  (b = σ/√2)
//     exp_avg_sq      Gaussian    4-bit nominal  clip ±4σ  (on log-residual)
//     Bit depth per row is computed from rel_err so the user sets error budget.
//
//   Stage 3  Lossless
//     Column-major packed codes → ANS with hardcoded analytical frequency table.
//     Tables are never stored on disk; decoded from the stream_type tag.
//
// Hardcoded ANS tables are computed once at init from the analytical PDFs:
//     Gaussian   P_k  = Φ((k+1−n/2)/(n/8)) − Φ((k−n/2)/(n/8))
//     Laplacian  P_k  ∝ exp(−√2 · |k−n/2| / (n/16))
// normalized to ANS_M = 4096.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  include <hip/hip_runtime.h>
   using lrs_stream_t = hipStream_t;
#else
#  include <cuda_runtime.h>
   using lrs_stream_t = cudaStream_t;
#endif

namespace checkers {
namespace lrs {

// ─────────────────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────────────────
enum class StreamType  : uint8_t { MasterWeight = 0, ExpAvg = 1, ExpAvgSq = 2 };
enum class Distribution: uint8_t { Gaussian = 0, Laplacian = 1 };

inline Distribution stream_dist(StreamType t) noexcept {
    return (t == StreamType::ExpAvg) ? Distribution::Laplacian : Distribution::Gaussian;
}

// ─────────────────────────────────────────────────────────────────────────────
// ANS scale constant (must agree with ANSCompressor.hpp)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr uint32_t LRS_ANS_M = 4096u;  // 2^12

// ─────────────────────────────────────────────────────────────────────────────
// Hardcoded analytical ANS frequency tables
// ─────────────────────────────────────────────────────────────────────────────
// Returns a pointer to a static uint32_t table of 2^nbits entries summing to
// LRS_ANS_M. Valid (dist, nbits) pairs: (any, 4), (any, 6), (any, 8).
// Thread-safe after the first call; tables are built at static init.
const uint32_t* analytical_freq_table(Distribution dist, int nbits);

// ─────────────────────────────────────────────────────────────────────────────
// LRSMeta — host-side metadata required to reconstruct one tensor
// ─────────────────────────────────────────────────────────────────────────────
struct LRSMeta {
    // Tensor geometry
    int M = 0, N = 0;
    StreamType   stream_type{};
    Distribution dist{};
    float        rel_err     = 0.0f;

    // Mean-field vectors (host)
    std::vector<float>   row_mean;    // a_i  [M] fp32
    std::vector<float>   col_mean;    // b_j  [N] fp32

    // Per-row quantization metadata (host)
    std::vector<float>   row_scale;   // σ_i  [M] std of residual
    std::vector<float>   row_rms;     // rms of original row [M]
    std::vector<uint8_t> row_nbits;   // b_i  [M]  ∈ {4, 6, 8}

    // Sparse outlier list (|R_ij|/σ_i > clip_K)
    // Stored as flat index + fp32 value; typically < 0.05% of elements.
    std::vector<uint32_t> outlier_idx;   // flat index i*N+j
    std::vector<float>    outlier_val;   // original float32 value R_ij

    // Exact-zero bitmap for exp_avg_sq (packed uint32, length ceil(M*N/32))
    std::vector<uint32_t> zero_bitmap;
    uint32_t              n_zeros = 0;

    // Log-transform shift (exp_avg_sq only)
    float log_delta = 1e-30f;
};

// ─────────────────────────────────────────────────────────────────────────────
// LRSCompressResult
// ─────────────────────────────────────────────────────────────────────────────
struct LRSCompressResult {
    LRSMeta              meta;
    std::vector<uint8_t> compressed_bytes; // zstd-compressed column-major codes
};

// ─────────────────────────────────────────────────────────────────────────────
// Public host-side entry points
// ─────────────────────────────────────────────────────────────────────────────

// Compress a 2-D float32 tensor that lives on `device_idx`.
// d_W   : device pointer to [M × N] float32 row-major.
//         WARNING — d_W is MODIFIED IN PLACE during decomposition.
//         The caller must save a copy if the original is needed.
// Returns a fully host-side LRSCompressResult ready to serialize to disk.
LRSCompressResult lrs_compress(
    float*     d_W,
    int        M,
    int        N,
    StreamType stream_type,
    float      rel_err,
    int        device_idx
);

// Reconstruct W on GPU and return as a host float32 vector [M*N].
std::vector<float> lrs_decompress(
    const LRSCompressResult& result,
    int device_idx
);

} // namespace lrs
} // namespace checkers
