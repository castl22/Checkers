#include "checkers/CheckpointCompressor.hpp"

#include "checkers/CheckpointFormat.hpp"
#include "checkers/CodecKernels.hpp"
#include "checkers/GpuDefs.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/PgQuant.hpp"
#include "checkers/TtLinalg.hpp"
#include "checkers/ZstdUtil.hpp"
#include "checkers/logging.hpp"

#include <cerrno>
#include <cfloat>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <system_error>

namespace py = pybind11;

namespace checkers {

// ------------------------------------------------------------------ //
//  helpers
// ------------------------------------------------------------------ //
namespace {

using clock_t_ = std::chrono::steady_clock;

double ms_since(clock_t_::time_point t) {
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t).count();
}

std::string fmt_double(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.9g", v);
    return std::string(b);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (c == '\n')        { o += "\\n"; }
        else                        { o.push_back(c); }
    }
    return o;
}

// cat_index: 0 = ModelState, 1 = MasterWeights, 2 = OptimizerExpAvg,
//            3 = OptimizerExpAvgSq   (tensor_category_index order)
inline bool phase_b_category(int cat_index) {          // per-group RMS-relative quant
    return cat_index == 0 || cat_index == 1;
}
inline bool phase_c_category(int cat_index) {          // single global RMS-relative quant
    return cat_index == 3;
}
inline bool codec_dtype_supported(DataType dt) {
    return dt == DataType::Float32 || dt == DataType::BFloat16 || dt == DataType::Float16;
}

// PgSteps / quantize_steps live in checkers/PgQuant.hpp (shared with the self-test).

} // namespace

// ------------------------------------------------------------------ //
//  WorkItem  (private; declared in the header as an incomplete type)
// ------------------------------------------------------------------ //
struct CheckpointCompressor::WorkItem {
    ShardTensor          st;
    int                  cat_index = -1;
    int                  slot      = -1;

    std::vector<uint8_t> payload;                 // main payload (zstd)
    std::vector<uint8_t> aux;                     // codec side-info (zstd); empty for RawZstd
    uint8_t              flags     = 0;
    uint8_t              codec_id  = static_cast<uint8_t>(CodecId::RawZstd);

    // per-group quant codec params (RmsRelPgQuant)
    double               pg_lo     = 0.0;
    double               pg_step   = 0.0;
    double               d_global  = 0.0;
    uint64_t             n_groups  = 0;
    uint64_t             n_over    = 0;
    uint64_t             n_zero    = 0;
    uint8_t              log_domain = 0;

    // SNR contribution (0/0 => treat this tensor as lossless)
    double               sq_err    = 0.0;
    double               signal    = 0.0;
    bool                 lossy     = false;

    double               d2h_ms    = 0.0;
    double               compute_ms = 0.0;
    bool                 done      = false;
    std::string          error;                   // why done == false (worker error / skipped)
};

namespace {
struct JobCtx {
    std::deque<size_t>      queue;
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    closed = false;
};
} // namespace

// ------------------------------------------------------------------ //
//  singleton / lifecycle
// ------------------------------------------------------------------ //
CheckpointCompressor& CheckpointCompressor::instance() {
    static CheckpointCompressor inst;
    return inst;
}

CheckpointCompressor::~CheckpointCompressor() {
    reset();
}

void CheckpointCompressor::reset() {
    if (finalizer_.joinable()) finalizer_.join();
    pending_.store(false);
    free_codec_scratch();
    pool_.release();
    baseline_free_vram_ = 0;
    staging_vram_delta_ = 0;
    staging_mode_.clear();
    last_metrics_ = CompressionMetrics{};
}

void CheckpointCompressor::configure(const CompressionConfig&    cfg,
                                     std::shared_ptr<RankLogger> logger,
                                     int                         global_rank) {
    reset();
    cfg_         = cfg;
    logger_      = std::move(logger);
    global_rank_ = global_rank;
    CK_GPU_CHECK(gpu::GetDevice(&device_id_));
}

// ------------------------------------------------------------------ //
//  codec scratch (shared, mutex-guarded)
// ------------------------------------------------------------------ //
void CheckpointCompressor::free_codec_scratch() {
    if (scratch_.fp32)       (void)gpu::Free(scratch_.fp32);
    if (scratch_.q)          (void)gpu::Free(scratch_.q);
    if (scratch_.rms)        (void)gpu::Free(scratch_.rms);
    if (scratch_.step)       (void)gpu::Free(scratch_.step);
    if (scratch_.over_idx)   (void)gpu::Free(scratch_.over_idx);
    if (scratch_.over_val)   (void)gpu::Free(scratch_.over_val);
    if (scratch_.over_count) (void)gpu::Free(scratch_.over_count);
    scratch_ = CodecScratch{};
}

void CheckpointCompressor::alloc_codec_scratch(size_t max_elems, size_t expand_elems,
                                               uint32_t group) {
    free_codec_scratch();
    if (max_elems == 0) return;

    const uint32_t g  = std::max<uint32_t>(group, 1u);
    const size_t max_groups = (max_elems + g - 1) / g;
    const size_t over_cap   = std::max<size_t>(1024, max_elems / 32);   // ~3% headroom

    // scratch_.fp32 doubles as (a) the bf16/fp16 -> fp32 expand target for the
    // per-group codec and (b) the log(v) working buffer for the exp_avg_sq
    // LogPwRel codec (fp32 input), so size it for the largest staged tensor.
    const size_t fp32_elems = std::max(expand_elems, max_elems);

    void* p = nullptr;
    if (fp32_elems > 0) {
        CK_GPU_CHECK(gpu::Malloc(&p, fp32_elems * sizeof(float)));   scratch_.fp32 = static_cast<float*>(p);
    }
    CK_GPU_CHECK(gpu::Malloc(&p, max_elems  * sizeof(int16_t)));      scratch_.q          = static_cast<int16_t*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, max_groups * sizeof(float)));        scratch_.rms        = static_cast<float*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, max_groups * sizeof(float)));        scratch_.step       = static_cast<float*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, over_cap   * sizeof(uint32_t)));     scratch_.over_idx   = static_cast<uint32_t*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, over_cap   * sizeof(float)));        scratch_.over_val   = static_cast<float*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, sizeof(uint32_t)));                  scratch_.over_count = static_cast<uint32_t*>(p);

    // GPU bit-pack stage.  max_blocks covers the largest staged tensor; packed
    // is the 16-bit worst case so a dense tensor cannot overrun it.
    const size_t max_blocks = (max_elems + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK;
    CK_GPU_CHECK(gpu::Malloc(&p, max_blocks));                        scratch_.w      = static_cast<uint8_t*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, max_blocks * sizeof(uint64_t)));     scratch_.off    = static_cast<uint64_t*>(p);
    CK_GPU_CHECK(gpu::Malloc(&p, max_elems  * sizeof(uint16_t)));     scratch_.packed = static_cast<uint8_t*>(p);

    scratch_.max_elems  = max_elems;
    scratch_.max_groups = max_groups;
    scratch_.over_cap   = over_cap;
    scratch_.max_blocks = max_blocks;
    scratch_.bytes = (fp32_elems * sizeof(float))
                   + (max_elems  * sizeof(int16_t))
                   + (2 * max_groups * sizeof(float))
                   + (over_cap * (sizeof(uint32_t) + sizeof(float)))
                   + sizeof(uint32_t)
                   + max_blocks * (1 + sizeof(uint64_t))
                   + max_elems * sizeof(uint16_t);
}

// ------------------------------------------------------------------ //
//  uniform scalar quant codec
//    PerGroupRms -> RmsRelPgQuant     (Phase B, weights; linear, per-group step)
//    GlobalRms   -> RmsRelGlobalQuant (Phase C aggressive; linear, one step)
//    LogPwRel    -> LogPwRelQuant     (Phase C default, exp_avg_sq; quantize
//                                      log(v) with one step D = 2*log(1+eps),
//                                      decoder v_hat = exp(q*D) -> hard
//                                      per-element relative bound, no zeroing)
// ------------------------------------------------------------------ //
bool CheckpointCompressor::codec_pergroup(WorkItem& wi, StagingPool::Slot& sl,
                                          int worker_id, double error_bound,
                                          PgStepMode mode) {
    const size_t   n     = wi.st.num_elements;
    const uint32_t group = std::max<uint32_t>(cfg_.group_size, 1u);
    const DataType dt    = wi.st.dtype;
    if (n == 0) return false;

    const bool  log_mode = (mode == PgStepMode::LogPwRel);
    // Guard against log(0); this is NOT a divergence floor -- values below it
    // still round-trip relatively (exp(log(1e-30)) ~= 0, which is exactly what
    // Adam wants for v -> 0).  The relative bound is what keeps 1/(sqrt(v)+eps)
    // intact for the bulk.
    constexpr float LOG_TINY = 1e-30f;

    const uint32_t ng = static_cast<uint32_t>((n + group - 1) / group);
    gpu::Stream_t  cs  = pool_.compute_stream(worker_id);

    if (n > scratch_.max_elems || ng > scratch_.max_groups) {
        if (logger_) logger_->log_message("[compressor] codec_pergroup: '" + wi.st.name +
            "' exceeds codec scratch -- raw fallback");
        return false;
    }

    std::vector<float>    h_rms(ng);
    PgSteps               ps;
    uint32_t              h_over = 0;
    std::vector<uint32_t> h_ovi;
    std::vector<float>    h_ovv;
    std::vector<uint8_t>  aux_raw;
    std::vector<uint8_t>  h_w;            // per-block bit widths, D2H'd once
    size_t                packed_bytes = 0;

    {
        std::lock_guard<std::mutex> lk(codec_mtx_);

        // d_ref = the fp32 buffer we quantize.  LogPwRel quantizes log(v)
        // (fp32 input guaranteed by the dispatch); the others quantize the
        // (expanded) linear tensor.
        const float* d_ref = nullptr;
        if (log_mode) {
            codec::log_clamp_fp32(reinterpret_cast<const float*>(sl.d_buf), n,
                                  LOG_TINY, scratch_.fp32, cs);
            d_ref = scratch_.fp32;
        } else if (dt == DataType::Float32) {
            d_ref = reinterpret_cast<const float*>(sl.d_buf);
        } else {
            codec::expand_to_fp32(sl.d_buf, dt, n, scratch_.fp32, cs);
            d_ref = scratch_.fp32;
        }

        if (!log_mode) {
            codec::group_rms_fp32(d_ref, n, group, scratch_.rms, ng, cs);
            CK_GPU_CHECK(gpu::MemcpyAsync(h_rms.data(), scratch_.rms, ng * sizeof(float),
                                         gpu::Copy::D2H, cs));
            CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        }

        if (log_mode) {
            // exp_avg_sq default: hard per-element relative bound in the log
            // domain.  D/2 = log(1+eps) => v_hat/v in [1/(1+eps), 1+eps].
            // No RMS, no per-group step, no zeroing -- every second moment is
            // preserved to <= eps so Adam's 1/(sqrt(v)+eps) is intact and the
            // restart converges (rms_rel_global drives it to a cliff instead).
            const double D = 2.0 * std::log1p(error_bound) * (1.0 - 1e-6);
            ps.pg_lo   = std::log(D);            // decoder: D_use = exp(pg_lo) = D
            ps.pg_step = 0.0;
            ps.dcodes.assign(ng, 0u);
            ps.shrink.assign(ng, 1.0f);
            ps.d_use.assign(ng, static_cast<float>(D));
            wi.d_global = D;
        } else if (mode == PgStepMode::GlobalRms) {
            // exp_avg_sq: one step from the whole-tensor RMS (equal groups =>
            // mean of the per-group mean-squares is the tensor mean-square).
            // The heavy tail makes D coarse -> the bulk quantizes to 0; the
            // decoder floors v_hat at EXP_AVG_SQ_FLOOR so 1/(sqrt(v_hat)+eps)
            // never collapses (the divergence-cliff guard from the spec).
            constexpr float EXP_AVG_SQ_FLOOR = 1e-16f;   // ~ (Adam eps)^2
            double ms = 0.0;
            for (uint32_t g = 0; g < ng; ++g)
                ms += static_cast<double>(h_rms[g]) * static_cast<double>(h_rms[g]);
            const double grms = std::sqrt(ms / static_cast<double>(ng));
            const double D = 2.0 * error_bound * grms * (1.0 - 1e-6);
            ps.pg_lo   = std::log(std::max(D, 1e-300));
            ps.pg_step = static_cast<double>(EXP_AVG_SQ_FLOOR);  // carried to the decoder
            ps.dcodes.assign(ng, 0u);                            // dcode=0 => D_hat=exp(pg_lo)=D
            ps.shrink.assign(ng, 1.0f);
            ps.d_use.assign(ng, static_cast<float>(D));
            wi.d_global = D;
        } else {
            ps = quantize_steps(h_rms, error_bound);   // per-group step
        }

        CK_GPU_CHECK(gpu::MemcpyAsync(scratch_.step, ps.d_use.data(), ng * sizeof(float),
                                     gpu::Copy::H2D, cs));
        CK_GPU_CHECK(gpu::MemsetAsync(scratch_.over_count, 0, sizeof(uint32_t), cs));
        // Fused [log ->] quantize -> zigzag -> bit-pack widths, one pass.  This
        // replaces quantize_pergroup + a separate zigzag_inplace + the CPU zstd
        // that followed: the payload now never leaves the GPU uncompressed.
        // log_clamp_fp32 above is now redundant for the log path (the kernel
        // does the transform), but d_ref is already built, so leave it -- the
        // codes are identical either way and the dispatch stays readable.
        codec::quantize_fused(d_ref, n, group, scratch_.step,
                              reinterpret_cast<uint16_t*>(scratch_.q),
                              scratch_.over_idx, scratch_.over_val,
                              static_cast<uint32_t>(scratch_.over_cap),
                              scratch_.over_count, 0, LOG_TINY, cs, scratch_.w);
        CK_GPU_CHECK(gpu::MemcpyAsync(&h_over, scratch_.over_count, sizeof(uint32_t),
                                     gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));

        if (h_over > scratch_.over_cap) {
            if (logger_) logger_->log_message("[compressor] codec_pergroup: '" + wi.st.name +
                "' overflow " + std::to_string(h_over) + " > cap " +
                std::to_string(scratch_.over_cap) + " -- raw fallback");
            return false;
        }

        // scratch_.q holds ZIGZAG codes (quantize_fused emits them directly).
        // The SNR check wants signed codes, so it un-zigzags in place and must
        // re-zigzag before the packer below reads the same buffer (as
        // codec_tt_eq does).  Without that, measure_snr=true wrote signed codes
        // that the decoder's unzigzag turned into garbage.
        if (cfg_.measure_snr) {
            double sq = 0.0, sig = 0.0;
            codec::unzigzag_inplace(reinterpret_cast<uint16_t*>(scratch_.q), n, cs);
            if (log_mode) {
                // linear-domain SNR: v_hat = exp(dequant(q)).  Reuse scratch_.fp32
                // (log v is no longer needed); sl.d_buf still holds linear v.
                codec::dequant_pergroup(scratch_.q, n, group, scratch_.step,
                                        scratch_.fp32, cs);
                codec::exp_fp32(scratch_.fp32, n, scratch_.fp32, cs);
                CK_GPU_CHECK(gpu::StreamSynchronize(cs));
                codec::reduce_sq_error_fp32(reinterpret_cast<const float*>(sl.d_buf),
                                            scratch_.fp32, n, &sq, &sig, cs);
            } else {
                codec::reduce_quant_error(scratch_.q, scratch_.step, d_ref, n, group,
                                          &sq, &sig, cs);
            }
            wi.sq_err = sq;
            wi.signal = sig;
            codec::zigzag_inplace(scratch_.q, n, cs);   // restore for the packer
        }

        if (h_over) {
            h_ovi.resize(h_over);
            h_ovv.resize(h_over);
            CK_GPU_CHECK(gpu::MemcpyAsync(h_ovi.data(), scratch_.over_idx,
                                         h_over * sizeof(uint32_t), gpu::Copy::D2H, cs));
            CK_GPU_CHECK(gpu::MemcpyAsync(h_ovv.data(), scratch_.over_val,
                                         h_over * sizeof(float), gpu::Copy::D2H, cs));
        }
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));

        // aux (uncompressed): dcodes[ng] ++ shrink[ng]*4 ++ (idx,val)[h_over]*8
        aux_raw.reserve(ng + ng * 4u + static_cast<size_t>(h_over) * 8u);
        aux_raw.insert(aux_raw.end(), ps.dcodes.begin(), ps.dcodes.end());
        const uint8_t* sp = reinterpret_cast<const uint8_t*>(ps.shrink.data());
        aux_raw.insert(aux_raw.end(), sp, sp + ng * 4u);
        for (uint32_t k = 0; k < h_over; ++k) {
            const uint8_t* ip = reinterpret_cast<const uint8_t*>(&h_ovi[k]);
            const uint8_t* vp = reinterpret_cast<const uint8_t*>(&h_ovv[k]);
            aux_raw.insert(aux_raw.end(), ip, ip + 4);
            aux_raw.insert(aux_raw.end(), vp, vp + 4);
        }
        // ---- GPU bit-pack: the payload never crosses the bus uncompressed --
        // Lossless over the zigzag codes, so the reconstruction and the error
        // bound are bit-identical to the zstd payload this replaces; only the
        // compressed SIZE and the cost differ.  Measured on this shard, the old
        // path put 12.04 GB per rank through a SINGLE-THREADED zstd_compress at
        // ~300 MB/s, which is what made the checkpoint stall 15.4 s while the
        // same codec benchmarks at 0.13 s/rank.
        const uint32_t nblk = static_cast<uint32_t>(
            (n + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
        h_w.resize(nblk);
        CK_GPU_CHECK(gpu::MemcpyAsync(h_w.data(), scratch_.w, nblk, gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        std::vector<uint64_t> h_off(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { h_off[b] = run; run += 32ull * h_w[b]; }
        packed_bytes = static_cast<size_t>(run);
        CK_GPU_CHECK(gpu::MemcpyAsync(scratch_.off, h_off.data(),
                                     (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D, cs));
        codec::bitpack_encode(reinterpret_cast<const uint16_t*>(scratch_.q), n,
                              scratch_.w, scratch_.off, scratch_.packed, nblk, cs);
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        if (packed_bytes)
            CK_GPU_CHECK(gpu::MemcpyAsync(sl.h_buf, scratch_.packed, packed_bytes,
                                         gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
    } // release codec_mtx_ (GPU done; sl.h_buf holds the packed bytes)

    // entropy stage -- parallel across workers.  Only the WIDTH array and the
    // aux blob still see zstd: n/256 bytes and ~5*ng bytes respectively, against
    // the n*2 bytes that used to go through it.  The widths are highly
    // redundant (whole-zero blocks give width 0) and leaving them raw capped
    // the achievable ratio, so this one small zstd call earns its place.
    {
        const uint32_t nblk = static_cast<uint32_t>(
            (n + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
        std::vector<uint8_t> wc = zstd_compress(h_w.data(), nblk, cfg_.zstd_level);
        const uint32_t wcs = static_cast<uint32_t>(wc.size());
        wi.payload.resize(4 + wcs + packed_bytes);
        std::memcpy(wi.payload.data(), &wcs, 4);
        std::memcpy(wi.payload.data() + 4, wc.data(), wcs);
        if (packed_bytes)
            std::memcpy(wi.payload.data() + 4 + wcs, sl.h_buf, packed_bytes);
    }
    wi.aux        = zstd_compress(aux_raw.data(), aux_raw.size(), cfg_.zstd_level);
    wi.flags      = CCH_FLAG_BITPACK;
    wi.codec_id   = static_cast<uint8_t>(
                        log_mode                     ? CodecId::LogPwRelQuant     :
                        mode == PgStepMode::GlobalRms ? CodecId::RmsRelGlobalQuant
                                                      : CodecId::RmsRelPgQuant);
    wi.log_domain = log_mode ? 1u : 0u;
    wi.pg_lo      = ps.pg_lo;
    wi.pg_step    = ps.pg_step;
    wi.n_groups   = ng;
    wi.n_over     = h_over;
    wi.lossy      = true;
    return true;
}

// ------------------------------------------------------------------ //
//  exp_avg codec: capped TT-SVD  ->  EQ error-bounded residual quant
//  (TZ's pipeline shape; EQ's quantizer for the piece).  Returns false
//  (caller falls back to raw+zstd) for small tensors or on any GPU/BLAS
//  failure.
// ------------------------------------------------------------------ //
bool CheckpointCompressor::codec_tt_eq(WorkItem& wi, StagingPool::Slot& sl,
                                       int worker_id, double error_bound) {
    const size_t L = wi.st.num_elements;
    if (wi.st.dtype != DataType::Float32) return false;
    if (L < (1u << 15) || L > 0xFFFFFFFFull) return false;   // small / oversized -> raw
    if (L > scratch_.max_elems) return false;

    gpu::Stream_t cs = pool_.compute_stream(worker_id);
    const uint32_t ndim = std::min<uint32_t>(std::max<uint32_t>(cfg_.tt_ndim, 2u), 3u);
    const uint32_t cap  = std::max<uint32_t>(cfg_.tt_max_rank, 1u);
    const tt::TtDims dims = tt::choose_dims(L, ndim);
    const size_t P = static_cast<size_t>(dims.padded());

    void*     blas       = tt::blas_handle();
    float*    d_A         = nullptr;   // padded shard  [P]
    float*    d_Ahat      = nullptr;   // TT low-rank recon, then full recon  [P]
    float*    d_R         = nullptr;   // residual, then R_hat  [L]
    uint16_t* d_core_u16  = nullptr;   // fp16 packed cores
    float*    d_step1     = nullptr;   // [1]  EQ step
    tt::TtCores cores;

    auto cleanup = [&]() {
        if (d_A)        (void)gpu::Free(d_A);
        if (d_Ahat)     (void)gpu::Free(d_Ahat);
        if (d_R)        (void)gpu::Free(d_R);
        if (d_core_u16) (void)gpu::Free(d_core_u16);
        if (d_step1)    (void)gpu::Free(d_step1);
        tt::tt_free(cores);
    };

    uint32_t             h_over = 0;
    std::vector<uint32_t> h_ovi;
    std::vector<float>    h_ovv;
    std::vector<uint16_t> h_core;
    std::vector<uint8_t>  h_w;            // per-block bit widths, D2H'd once
    uint32_t              tt_nblk = 0;
    size_t                packed_bytes = 0;
    TtHeader             th{};
    double               D = 0.0;
    double               rmin = 0.0, rmax = 0.0;

    {
        std::lock_guard<std::mutex> lk(codec_mtx_);

        if (gpu::Malloc(reinterpret_cast<void**>(&d_A),    P * sizeof(float)) != gpu::Success ||
            gpu::Malloc(reinterpret_cast<void**>(&d_Ahat), P * sizeof(float)) != gpu::Success ||
            gpu::Malloc(reinterpret_cast<void**>(&d_R),    L * sizeof(float)) != gpu::Success ||
            gpu::Malloc(reinterpret_cast<void**>(&d_step1), sizeof(float))    != gpu::Success) {
            cleanup();
            return false;
        }

        // pad-copy the shard into d_A
        CK_GPU_CHECK(gpu::MemsetAsync(d_A, 0, P * sizeof(float), cs));
        CK_GPU_CHECK(gpu::MemcpyAsync(d_A, sl.d_buf, L * sizeof(float), gpu::Copy::D2D, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));

        tt::blas_set_stream(blas, cs);
        if (!tt::tt_decompose(d_A, dims, cap, cfg_.tt_energy_keep, cores, blas, cs)) {
            if (logger_) logger_->log_message("[compressor] codec_tt_eq: '" + wi.st.name +
                "' tt_decompose failed -- raw fallback");
            cleanup();
            return false;
        }
        if (!tt::tt_reconstruct(cores, d_Ahat, blas, cs)) {
            cleanup();
            return false;
        }

        // residual R = A[:L] - Ahat_lr[:L]
        codec::sub_fp32(d_A, d_Ahat, d_R, L, cs);
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));

        // EQ step from the error bound (eq_ebound.py: D = 2*abs_eps*margin)
        codec::reduce_minmax_fp32(d_R, L, &rmin, &rmax, cs);
        const double amax  = std::max(std::fabs(rmin), std::fabs(rmax));
        const double range = rmax - rmin;
        double abs_eps;
        switch (cfg_.tt_eq_mode) {
            case 0:  abs_eps = error_bound;                     break;   // ABS
            default: abs_eps = error_bound * std::max(range, 0.0); break; // REL
        }
        if (!(abs_eps > 0.0)) { cleanup(); return false; }        // degenerate residual
        const double dyn    = amax / std::max(abs_eps, (double)FLT_MIN);
        const double margin = 1.0 - std::min(1e-3, std::max(64.0 * DBL_EPSILON * dyn, 1e-9));
        D = 2.0 * abs_eps * margin;
        if (!(D > 0.0)) { cleanup(); return false; }

        const float Df = static_cast<float>(D);
        CK_GPU_CHECK(gpu::MemcpyAsync(d_step1, &Df, sizeof(float), gpu::Copy::H2D, cs));
        CK_GPU_CHECK(gpu::MemsetAsync(scratch_.over_count, 0, sizeof(uint32_t), cs));

        const uint32_t grp = static_cast<uint32_t>(L);   // single global step
        // Fused quantize -> zigzag -> bit-pack widths, one pass (was
        // quantize_pergroup + a separate zigzag_inplace + CPU zstd).
        codec::quantize_fused(d_R, L, grp, d_step1,
                              reinterpret_cast<uint16_t*>(scratch_.q),
                              scratch_.over_idx, scratch_.over_val,
                              static_cast<uint32_t>(scratch_.over_cap),
                              scratch_.over_count, 0, 0.0f, cs, scratch_.w);
        CK_GPU_CHECK(gpu::MemcpyAsync(&h_over, scratch_.over_count, sizeof(uint32_t),
                                     gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        if (h_over > scratch_.over_cap) { cleanup(); return false; }

        if (cfg_.measure_snr) {
            // scratch_.q holds ZIGZAG codes now; undo for the signed dequant.
            // Re-zigzagged below before packing.
            codec::unzigzag_inplace(reinterpret_cast<uint16_t*>(scratch_.q), L, cs);
            codec::dequant_pergroup(scratch_.q, L, grp, d_step1, d_R, cs);     // d_R := R_hat
            if (h_over)
                codec::scatter_overflow(d_R, scratch_.over_idx, scratch_.over_val, h_over, cs);
            codec::add_fp32(d_Ahat, d_R, d_Ahat, L, cs);                       // d_Ahat := full recon
            double sq = 0.0, sig = 0.0;
            codec::reduce_sq_error_fp32(d_A, d_Ahat, L, &sq, &sig, cs);
            wi.sq_err = sq;
            wi.signal = sig;
            codec::zigzag_inplace(scratch_.q, L, cs);   // restore for the packer
        }

        // ---- GPU bit-pack (see codec_pergroup for why) ----
        tt_nblk = static_cast<uint32_t>((L + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
        h_w.resize(tt_nblk);
        CK_GPU_CHECK(gpu::MemcpyAsync(h_w.data(), scratch_.w, tt_nblk, gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        {
            std::vector<uint64_t> h_off(tt_nblk);
            uint64_t run = 0;
            for (uint32_t b = 0; b < tt_nblk; ++b) { h_off[b] = run; run += 32ull * h_w[b]; }
            packed_bytes = static_cast<size_t>(run);
            CK_GPU_CHECK(gpu::MemcpyAsync(scratch_.off, h_off.data(),
                                         (size_t)tt_nblk * sizeof(uint64_t),
                                         gpu::Copy::H2D, cs));
        }
        codec::bitpack_encode(reinterpret_cast<const uint16_t*>(scratch_.q), L,
                              scratch_.w, scratch_.off, scratch_.packed, tt_nblk, cs);
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
        if (packed_bytes)
            CK_GPU_CHECK(gpu::MemcpyAsync(sl.h_buf, scratch_.packed, packed_bytes,
                                         gpu::Copy::D2H, cs));
        if (h_over) {
            h_ovi.resize(h_over);
            h_ovv.resize(h_over);
            CK_GPU_CHECK(gpu::MemcpyAsync(h_ovi.data(), scratch_.over_idx,
                                         h_over * sizeof(uint32_t), gpu::Copy::D2H, cs));
            CK_GPU_CHECK(gpu::MemcpyAsync(h_ovv.data(), scratch_.over_val,
                                         h_over * sizeof(float), gpu::Copy::D2H, cs));
        }

        // pack cores -> fp16 (g1 ++ g2 ++ g3) and pull to host
        const uint64_t ce = cores.core_elems();
        const size_t   e1 = static_cast<size_t>(cores.n1) * cores.r1;
        const size_t   e2 = static_cast<size_t>(cores.r1) * cores.n2 * cores.r2;
        const size_t   e3 = static_cast<size_t>(cores.r2) * cores.n3;
        if (gpu::Malloc(reinterpret_cast<void**>(&d_core_u16), ce * sizeof(uint16_t)) != gpu::Success) {
            cleanup();
            return false;
        }
        codec::contract_from_fp32(cores.g1, DataType::Float16, e1, d_core_u16, cs);
        codec::contract_from_fp32(cores.g2, DataType::Float16, e2, d_core_u16 + e1, cs);
        codec::contract_from_fp32(cores.g3, DataType::Float16, e3, d_core_u16 + e1 + e2, cs);
        h_core.resize(ce);
        CK_GPU_CHECK(gpu::MemcpyAsync(h_core.data(), d_core_u16, ce * sizeof(uint16_t),
                                     gpu::Copy::D2H, cs));
        CK_GPU_CHECK(gpu::StreamSynchronize(cs));

        th.magic      = TTEQ_MAGIC;
        th.ndim       = static_cast<uint16_t>(ndim);
        th.eq_mode    = static_cast<uint16_t>(cfg_.tt_eq_mode);
        th.n1         = cores.n1;  th.n2 = cores.n2;  th.n3 = cores.n3;
        th.r1         = cores.r1;  th.r2 = cores.r2;
        th.n_elems    = L;
        th.core_elems = ce;
        th.n_over     = h_over;
        th.d_step     = D;
        th.resid_min  = static_cast<float>(rmin);
        th.resid_max  = static_cast<float>(rmax);
        th.core_f16   = 1u;

        cleanup();   // GPU buffers done; h_core / sl.h_buf hold everything
    }

    // aux (uncompressed) = TtHeader ++ core_f16 bytes ++ (idx u32, val f32)[h_over]
    std::vector<uint8_t> aux_raw;
    aux_raw.reserve(sizeof(TtHeader) + h_core.size() * sizeof(uint16_t) +
                    static_cast<size_t>(h_over) * 8u);
    const uint8_t* thp = reinterpret_cast<const uint8_t*>(&th);
    aux_raw.insert(aux_raw.end(), thp, thp + sizeof(TtHeader));
    const uint8_t* cbp = reinterpret_cast<const uint8_t*>(h_core.data());
    aux_raw.insert(aux_raw.end(), cbp, cbp + h_core.size() * sizeof(uint16_t));
    for (uint32_t k = 0; k < h_over; ++k) {
        const uint8_t* ip = reinterpret_cast<const uint8_t*>(&h_ovi[k]);
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(&h_ovv[k]);
        aux_raw.insert(aux_raw.end(), ip, ip + 4);
        aux_raw.insert(aux_raw.end(), vp, vp + 4);
    }

    {
        std::vector<uint8_t> wc = zstd_compress(h_w.data(), tt_nblk, cfg_.zstd_level);
        const uint32_t wcs = static_cast<uint32_t>(wc.size());
        wi.payload.resize(4 + wcs + packed_bytes);
        std::memcpy(wi.payload.data(), &wcs, 4);
        std::memcpy(wi.payload.data() + 4, wc.data(), wcs);
        if (packed_bytes)
            std::memcpy(wi.payload.data() + 4 + wcs, sl.h_buf, packed_bytes);
    }
    wi.aux        = zstd_compress(aux_raw.data(), aux_raw.size(), cfg_.zstd_level);
    wi.flags      = CCH_FLAG_BITPACK;
    wi.codec_id   = static_cast<uint8_t>(CodecId::TtEqResidual);
    wi.log_domain = 0u;
    wi.d_global   = D;
    wi.pg_lo      = static_cast<double>(aux_raw.size());   // exact uncompressed aux size (restart needs it)
    wi.pg_step    = static_cast<double>(cfg_.tt_eq_mode);
    wi.n_groups   = th.core_elems;
    wi.n_over     = h_over;
    wi.lossy      = true;
    return true;
}

// ------------------------------------------------------------------ //
//  allocate the persistent staging pool
// ------------------------------------------------------------------ //
void CheckpointCompressor::allocate(MemoryManager& mgr) {
    const auto staged = mgr.collect_tensors(cfg_.stage_targets);
    const size_t n = staged.size();
    size_t max_bytes = 0, max_elems = 0, expand_elems = 0;
    for (const auto& s : staged) {
        max_bytes = std::max(max_bytes, s.byte_size);
        max_elems = std::max(max_elems, s.num_elements);
        if (s.dtype == DataType::BFloat16 || s.dtype == DataType::Float16)
            expand_elems = std::max(expand_elems, s.num_elements);
    }

    if (max_bytes == 0) {
        if (logger_) logger_->log_message("[compressor] allocate: no staged tensors; pool not created");
        return;
    }

    const uint32_t workers = std::max<uint32_t>(cfg_.compute_streams, 1u);
    uint32_t ring = std::max(cfg_.staging_ring_depth, workers);
    ring = std::min<uint32_t>(ring, static_cast<uint32_t>(std::max<size_t>(1, n)));

    const size_t slot_bytes = (max_bytes + 255u) & ~size_t(255u);

    size_t free_b = 0, total_b = 0;
    (void)gpu::MemGetInfo(&free_b, &total_b);
    baseline_free_vram_ = free_b;

    if (cfg_.staging_budget_mb > 0) {
        const size_t budget = static_cast<size_t>(cfg_.staging_budget_mb) << 20;
        const uint32_t cap = static_cast<uint32_t>(std::max<size_t>(1, budget / slot_bytes));
        if (cap < ring) {
            if (logger_) logger_->log_message("[compressor] staging budget caps ring depth " +
                std::to_string(ring) + " -> " + std::to_string(cap));
            ring = cap;
        }
    }

    pool_.allocate(slot_bytes, ring, workers);
    alloc_codec_scratch(max_elems, expand_elems, cfg_.group_size);

    size_t free_after = 0;
    (void)gpu::MemGetInfo(&free_after, &total_b);
    staging_vram_delta_ = static_cast<int64_t>(baseline_free_vram_) - static_cast<int64_t>(free_after);
    staging_mode_       = (ring >= n) ? "snapshot" : "ring";

    if (logger_) logger_->log_message(
        "[compressor] staging pool ready mode=" + staging_mode_ +
        " ring_depth=" + std::to_string(pool_.ring_depth()) +
        " slot_bytes=" + std::to_string(slot_bytes) +
        " workers=" + std::to_string(workers) +
        " device_bytes=" + std::to_string(pool_.device_bytes()) +
        " pinned_bytes=" + std::to_string(pool_.pinned_bytes()) +
        " codec_scratch_bytes=" + std::to_string(scratch_.bytes) +
        " measured_vram_delta=" + std::to_string(staging_vram_delta_));
}

// ------------------------------------------------------------------ //
//  run  (critical-path snapshot + async compress/write)
// ------------------------------------------------------------------ //
CompressionMetrics CheckpointCompressor::run(MemoryManager&     mgr,
                                             const std::string& output_dir,
                                             const std::array<double, tracked_tensor_category_count>& error_bounds,
                                             uint32_t           targets,
                                             bool               blocking) {
    // finish any prior non-blocking checkpoint before starting a new one;
    // if it failed and nobody called join(), fail here rather than lose it
    if (finalizer_.joinable()) finalizer_.join();
    pending_.store(false);
    if (finalize_error_) {
        std::exception_ptr e = finalize_error_;
        finalize_error_ = nullptr;
        std::rethrow_exception(e);
    }

    const auto t0 = clock_t_::now();

    CompressionMetrics m;
    m.output_dir                 = output_dir;
    m.error_bounds               = error_bounds;
    // representative value = first targeted stream's bound (for legacy display)
    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c)
        if ((targets & (1u << c)) && error_bounds[c] > 0.0) { m.error_bound = error_bounds[c]; break; }
    m.ring_depth                 = pool_.ring_depth();
    m.workers                    = pool_.n_compute_streams();
    m.staging_mode               = staging_mode_;
    m.staging_device_bytes       = pool_.device_bytes();
    m.pinned_host_bytes          = pool_.pinned_bytes();
    m.codec_scratch_device_bytes = scratch_.bytes;
    m.peak_device_overhead_bytes = pool_.device_bytes() + scratch_.bytes;
    m.measured_free_vram_delta_b = staging_vram_delta_;

    uint32_t mask = targets & cfg_.stage_targets;
    if (targets & ~cfg_.stage_targets) {
        if (logger_) logger_->log_message(
            "[compressor] WARNING requested targets include categories that were not staged; "
            "restricting to stage_targets");
    }
    m.targets_mask = mask;

    if (!pool_.allocated() || mask == 0) {
        if (logger_) logger_->log_message("[compressor] run: nothing to do (allocated=" +
            std::to_string(pool_.allocated()) + " mask=" + std::to_string(mask) + ")");
        m.wall_ms     = ms_since(t0);
        last_metrics_ = m;
        return m;
    }

    auto tensors = mgr.collect_tensors(mask);
    if (tensors.empty()) {
        m.wall_ms     = ms_since(t0);
        last_metrics_ = m;
        return m;
    }

    // The snapshot must capture completed post-optimizer-step state.
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    auto work = std::make_shared<std::vector<WorkItem>>();
    work->reserve(tensors.size());
    for (auto& st : tensors) {
        WorkItem wi;
        wi.cat_index = static_cast<int>(tensor_category_index(st.category));
        wi.st        = std::move(st);
        work->push_back(std::move(wi));
    }

    auto ctx = std::make_shared<JobCtx>();

    // ---- background worker pool ----
    const uint32_t nW = pool_.n_compute_streams();
    const std::array<double, tracked_tensor_category_count> ebs = error_bounds;
    auto workers = std::make_shared<std::vector<std::thread>>();
    workers->reserve(nW);
    for (uint32_t w = 0; w < nW; ++w) {
        workers->emplace_back([this, work, ctx, w, ebs]() {
            (void)gpu::SetDevice(device_id_);
            for (;;) {
                size_t idx;
                {
                    std::unique_lock<std::mutex> lk(ctx->mtx);
                    ctx->cv.wait(lk, [&] { return !ctx->queue.empty() || ctx->closed; });
                    if (ctx->queue.empty()) return;
                    idx = ctx->queue.front();
                    ctx->queue.pop_front();
                }

                WorkItem& wi = (*work)[idx];
                StagingPool::Slot& sl = pool_.slot(wi.slot);

                try {
                    CK_GPU_CHECK(gpu::EventSynchronize(sl.copied));
                    const auto tc0 = clock_t_::now();

                    // TEST ONLY (tests/test_save_failures.py): fail every tensor whose
                    // name contains $CHECKERS_TEST_FAIL_TENSOR.  Unset = no effect.
                    if (const char* inj = std::getenv("CHECKERS_TEST_FAIL_TENSOR"))
                        if (*inj && wi.st.name.find(inj) != std::string::npos)
                            throw std::runtime_error("injected failure (CHECKERS_TEST_FAIL_TENSOR)");

                    const double eb = (wi.cat_index >= 0 &&
                                       wi.cat_index < static_cast<int>(tracked_tensor_category_count))
                                          ? ebs[wi.cat_index] : 0.0;

                    bool did_lossy = false;
                    if (!(eb > 0.0)) {
                        // no bound for this stream -> lossless
                    } else if (phase_b_category(wi.cat_index) && codec_dtype_supported(wi.st.dtype)) {
                        did_lossy = codec_pergroup(wi, sl, static_cast<int>(w), eb,
                                                   PgStepMode::PerGroupRms);
                    } else if (phase_c_category(wi.cat_index) && wi.st.dtype == DataType::Float32) {
                        // exp_avg_sq: log-domain pointwise-relative by default
                        // (restart converges); the aggressive global-RMS codec
                        // (~25x, but hits a divergence cliff) is opt-in.
                        did_lossy = codec_pergroup(wi, sl, static_cast<int>(w), eb,
                                                   cfg_.exp_avg_sq_aggressive
                                                       ? PgStepMode::GlobalRms
                                                       : PgStepMode::LogPwRel);
                    } else if (wi.cat_index == 2 && wi.st.dtype == DataType::Float32) {
                        // exp_avg: capped TT-SVD -> EQ error-bounded residual quant.
                        did_lossy = codec_tt_eq(wi, sl, static_cast<int>(w), eb);
                    }

                    if (!did_lossy) {
                        // raw + zstd fallback / non-lossy streams
                        gpu::Stream_t cs = pool_.compute_stream(static_cast<int>(w));
                        const auto td0 = clock_t_::now();
                        CK_GPU_CHECK(gpu::MemcpyAsync(sl.h_buf, sl.d_buf, wi.st.byte_size,
                                                     gpu::Copy::D2H, cs));
                        CK_GPU_CHECK(gpu::StreamSynchronize(cs));
                        wi.d2h_ms = ms_since(td0);

                        auto comp = zstd_compress(sl.h_buf, wi.st.byte_size, cfg_.zstd_level);
                        if (comp.size() >= wi.st.byte_size) {
                            wi.payload.assign(static_cast<const uint8_t*>(sl.h_buf),
                                              static_cast<const uint8_t*>(sl.h_buf) + wi.st.byte_size);
                            wi.flags = CCH_FLAG_RAW_PASSTHROUGH;
                        } else {
                            wi.payload = std::move(comp);
                            wi.flags   = CCH_FLAG_ZSTD;
                        }
                        wi.codec_id = static_cast<uint8_t>(CodecId::RawZstd);
                        wi.aux.clear();
                        wi.lossy = false;
                    }

                    wi.compute_ms = ms_since(tc0);
                    wi.done       = true;
                } catch (const std::exception& e) {
                    if (logger_) logger_->log_message("[compressor] worker error on '" +
                        wi.st.name + "': " + e.what());
                    wi.done  = false;
                    wi.error = std::string("worker error: ") + e.what();
                }
                pool_.give_back(wi.slot);
            }
        });
    }

    // ---- critical path: acquire slot -> D2D -> hand to a worker ----
    for (size_t i = 0; i < work->size(); ++i) {
        WorkItem& wi = (*work)[i];

        if (wi.st.byte_size > pool_.slot_bytes()) {
            if (logger_) logger_->log_message("[compressor] tensor '" + wi.st.name +
                "' (" + std::to_string(wi.st.byte_size) + " B) exceeds slot_bytes " +
                std::to_string(pool_.slot_bytes()) + " -- skipping");
            wi.done  = false;
            wi.error = "exceeds slot_bytes " + std::to_string(pool_.slot_bytes());
            continue;
        }

        const int slot = pool_.acquire();              // blocks when the ring is full
        wi.slot = slot;
        StagingPool::Slot& sl = pool_.slot(slot);

        if (gpu::PointerValid(wi.st.device_ptr) != gpu::Success) {
            (void)gpu::LastError();
            if (logger_) logger_->log_message("[compressor] stale device pointer for '" +
                wi.st.name + "' -- skipping");
            wi.done  = false;
            wi.error = "stale device pointer";
            pool_.give_back(slot);
            continue;
        }

        CK_GPU_CHECK(gpu::MemcpyAsync(sl.d_buf, wi.st.device_ptr, wi.st.byte_size,
                                     gpu::Copy::D2D, pool_.copy_stream()));
        CK_GPU_CHECK(gpu::EventRecord(sl.copied, pool_.copy_stream()));

        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            ctx->queue.push_back(i);
        }
        ctx->cv.notify_one();
    }

    // Wait for every D2D to complete: only now may the optimizer overwrite the
    // live tensors.  This wait is the entire training stall.
    CK_GPU_CHECK(gpu::StreamSynchronize(pool_.copy_stream()));
    m.critical_path_ms = ms_since(t0);

    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->closed = true;
    }
    ctx->cv.notify_all();

    if (blocking) {
        for (auto& th : *workers) th.join();
        auto sm = std::make_shared<CompressionMetrics>(m);
        finalize_and_store(work, sm, output_dir);
        sm->wall_ms = ms_since(t0);
        sm->pending = false;
        last_metrics_ = *sm;
        return *sm;
    }

    // non-blocking: background thread joins workers, writes files, stores metrics
    pending_.store(true);
    auto sm = std::make_shared<CompressionMetrics>(m);
    finalizer_ = std::thread([this, work, workers, sm, output_dir, t0]() {
        for (auto& th : *workers) th.join();
        try {
            finalize_and_store(work, sm, output_dir);
        } catch (const std::exception& e) {
            if (logger_) logger_->log_message(std::string("[compressor] finalize error: ") + e.what());
            finalize_error_ = std::current_exception();   // raised by join() / the next run()
        } catch (...) {
            finalize_error_ = std::current_exception();
        }
        sm->wall_ms   = ms_since(t0);
        sm->pending   = false;
        last_metrics_ = *sm;
        pending_.store(false);
    });

    m.pending     = true;
    last_metrics_ = m;
    return m;
}

CompressionMetrics CheckpointCompressor::join() {
    if (finalizer_.joinable()) finalizer_.join();
    pending_.store(false);
    if (finalize_error_) {
        std::exception_ptr e = finalize_error_;
        finalize_error_ = nullptr;
        std::rethrow_exception(e);
    }
    return last_metrics_;
}

// ------------------------------------------------------------------ //
//  finalize: write one .cch per stream + a manifest
// ------------------------------------------------------------------ //
void CheckpointCompressor::finalize_and_store(
        std::shared_ptr<std::vector<WorkItem>> work,
        std::shared_ptr<CompressionMetrics>    sm,
        std::string                            output_dir) {
    namespace fs = std::filesystem;
    CompressionMetrics& m = *sm;

    std::error_code ec;
    fs::create_directories(output_dir, ec);

    // Every collected tensor must have been compressed.  Until 2026-11-05 a
    // tensor whose worker failed (e.g. std::bad_alloc) or that was skipped was
    // silently left out: the stream file was written without it, self-consistent,
    // and the save reported success (exp1 control_s3 step_100: 23-36 exp_avg_sq
    // tensors missing on 3 ranks, found only at restore).  Now: no file is
    // written for this rank and the save throws.
    std::array<uint32_t, tracked_tensor_category_count> n_expected{};
    {
        size_t n_failed = 0;
        std::string first;
        for (const WorkItem& wi : *work) {
            const int c = wi.cat_index;
            if (c >= 0 && c < static_cast<int>(tracked_tensor_category_count)) n_expected[c]++;
            if (wi.done) continue;
            if (n_failed++ < 3)
                first += (first.empty() ? "" : "; ") + wi.st.name + ": " +
                         (wi.error.empty() ? std::string("not compressed") : wi.error);
        }
        if (n_failed) {
            const std::string msg = "[compressor] save FAILED: " + std::to_string(n_failed) + " of " +
                std::to_string(work->size()) + " tensors were not compressed (" + first +
                (n_failed > 3 ? "; ..." : "") + "); nothing written to " + output_dir +
                " for rank " + std::to_string(global_rank_);
            if (logger_) logger_->log_message(msg);
            throw std::runtime_error(msg);
        }
    }

    std::array<std::vector<size_t>, tracked_tensor_category_count> by_cat;
    for (size_t i = 0; i < work->size(); ++i) {
        const int c = (*work)[i].cat_index;
        if (c >= 0 && c < static_cast<int>(tracked_tensor_category_count))
            by_cat[c].push_back(i);
    }

    // Every write and the close are checked (2026-11-05): a short write
    // (full /dev/shm, Lustre error) used to pass silently.
    auto write_or_throw = [](std::ofstream& f, const void* p, size_t n,
                             const std::string& path, const char* what) {
        if (n == 0) return;
        errno = 0;
        f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("[compressor] write failed (" + std::string(what) + ", " +
                                         std::to_string(n) + " B) to " + path + ": " +
                                         (errno ? std::strerror(errno) : "stream error"));
    };
    auto close_or_throw = [](std::ofstream& f, const std::string& path) {
        errno = 0;
        f.close();
        if (f.fail()) throw std::runtime_error("[compressor] close failed for " + path + ": " +
                                               (errno ? std::strerror(errno) : "stream error"));
    };

    const uint32_t rank_u = static_cast<uint32_t>(global_rank_ < 0 ? 0 : global_rank_);
    char rankbuf[16];
    std::snprintf(rankbuf, sizeof(rankbuf), "%03u", rank_u);

    std::string manifest;
    manifest += "{\n";
    manifest += "  \"rank\": " + std::to_string(rank_u) + ",\n";
    manifest += "  \"error_bound\": " + fmt_double(m.error_bound) + ",\n";
    manifest += "  \"group_size\": " + std::to_string(cfg_.group_size) + ",\n";
    manifest += "  \"kernels_abi\": " + std::to_string(codec::kernels_abi_version()) + ",\n";
    manifest += "  \"streams\": {\n";
    bool first_stream = true;

    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
        if (by_cat[c].empty()) continue;
        const auto cat   = static_cast<TensorCategory>(c);
        const char* sname = tensor_category_name(cat);
        const std::string path = output_dir + "/rank_" + rankbuf + "_" + sname + ".cch";

        const auto tio0 = clock_t_::now();
        const uint32_t nT = static_cast<uint32_t>(by_cat[c].size());

        std::vector<CchIndexEntry> index(nT);
        uint64_t running = 0, tot_orig = 0, tot_pay = 0;
        bool     stream_lossy = false;
        uint32_t stream_codec = static_cast<uint32_t>(CodecId::RawZstd);
        for (uint32_t k = 0; k < nT; ++k) {
            WorkItem& wi = (*work)[by_cat[c][k]];
            CchIndexEntry& e = index[k];
            std::memset(&e, 0, sizeof(e));
            std::strncpy(e.name, wi.st.name.c_str(), sizeof(e.name) - 1);
            e.dtype        = static_cast<uint8_t>(wi.st.dtype);
            e.codec_id     = wi.codec_id;
            e.flags        = wi.flags;
            e.ndim         = static_cast<uint8_t>(std::min<size_t>(wi.st.shape.size(), CCH_MAX_DIMS));
            e.num_elements = wi.st.num_elements;
            e.orig_bytes   = wi.st.byte_size;
            for (uint8_t d = 0; d < e.ndim; ++d) e.shape[d] = wi.st.shape[d];

            e.data_offset  = running;
            e.data_size    = wi.payload.size();
            running       += wi.payload.size();
            e.aux_offset   = running;
            e.aux_size     = wi.aux.size();
            running       += wi.aux.size();

            e.d_global   = wi.d_global;
            e.n_groups   = wi.n_groups;
            e.n_over     = wi.n_over;
            e.n_zero     = wi.n_zero;
            e.log_domain = wi.log_domain;
            e.pg_lo      = wi.pg_lo;
            e.pg_step    = wi.pg_step;

            tot_orig += wi.st.byte_size;
            tot_pay  += wi.payload.size() + wi.aux.size();

            StreamMetrics& s = m.per_stream[c];
            s.n_tensors++;
            s.orig_bytes       += wi.st.byte_size;
            s.compressed_bytes += wi.payload.size() + wi.aux.size();
            s.compute_ms       += wi.compute_ms;
            s.d2h_ms           += wi.d2h_ms;
            if (wi.lossy) {
                stream_lossy = true;
                stream_codec = wi.codec_id;
                s.sq_err += wi.sq_err;
                s.signal += wi.signal;
            }
        }

        CchHeader h;
        std::memset(&h, 0, sizeof(h));
        h.magic               = CCH_MAGIC;
        h.version             = CCH_VERSION;
        h.rank                = rank_u;
        h.category            = static_cast<uint32_t>(c);
        h.codec_id            = stream_codec;
        h.group_size          = cfg_.group_size;
        h.n_tensors           = nT;
        h.error_bound         = m.error_bounds[c];
        h.index_offset        = sizeof(CchHeader);
        h.data_offset         = sizeof(CchHeader) + static_cast<uint64_t>(nT) * sizeof(CchIndexEntry);
        h.total_orig_bytes    = tot_orig;
        h.total_payload_bytes = tot_pay;

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("[compressor] cannot open " + path + ": " + std::strerror(errno));
        write_or_throw(f, &h, sizeof(h), path, "header");
        write_or_throw(f, index.data(), index.size() * sizeof(CchIndexEntry), path, "index");
        for (uint32_t k = 0; k < nT; ++k) {
            WorkItem& wi = (*work)[by_cat[c][k]];
            write_or_throw(f, wi.payload.data(), wi.payload.size(), path, "payload");
            write_or_throw(f, wi.aux.data(), wi.aux.size(), path, "aux");
        }
        close_or_throw(f, path);
        {
            // and the file on disk is exactly as long as its header + index say
            const uint64_t want = h.data_offset + running;
            std::error_code sec;
            const uint64_t got = fs::file_size(path, sec);
            if (sec || got != want)
                throw std::runtime_error("[compressor] " + path + " is " +
                    (sec ? "unreadable (" + sec.message() + ")" : std::to_string(got) + " B") +
                    " after close, header implies " + std::to_string(want) + " B");
        }

        const double io_ms = ms_since(tio0);
        m.per_stream[c].io_ms   += io_ms;
        m.per_stream[c].lossless = !stream_lossy;
        m.io_ms += io_ms;

        const double ratio = tot_pay ? static_cast<double>(tot_orig) / static_cast<double>(tot_pay) : 1.0;
        const char* codec_name =
            (stream_codec == static_cast<uint32_t>(CodecId::RmsRelPgQuant))     ? "rms_rel_pg" :
            (stream_codec == static_cast<uint32_t>(CodecId::LogPwRelQuant))     ? "log_pw_rel" :
            (stream_codec == static_cast<uint32_t>(CodecId::TtEqResidual))      ? "tt_eq" :
            (stream_codec == static_cast<uint32_t>(CodecId::RmsRelGlobalQuant)) ? "rms_rel_global" :
            "raw_zstd";
        double snr_db = 0.0;
        const StreamMetrics& sm_c = m.per_stream[c];
        if (stream_lossy && sm_c.sq_err > 0.0)
            snr_db = 10.0 * std::log10(sm_c.signal / sm_c.sq_err);

        if (!first_stream) manifest += ",\n";
        first_stream = false;
        manifest += "    \"" + std::string(sname) + "\": {";
        manifest += "\"file\": \"rank_" + std::string(rankbuf) + "_" + sname + ".cch\", ";
        manifest += "\"codec\": \"" + std::string(codec_name) + "\", ";
        manifest += "\"error_bound\": " + fmt_double(m.error_bounds[c]) + ", ";
        manifest += "\"n_tensors\": " + std::to_string(nT) + ", ";
        manifest += "\"n_expected\": " + std::to_string(n_expected[c]) + ", ";
        manifest += "\"orig_bytes\": " + std::to_string(tot_orig) + ", ";
        manifest += "\"compressed_bytes\": " + std::to_string(tot_pay) + ", ";
        manifest += "\"ratio\": " + fmt_double(ratio) + ", ";
        manifest += "\"lossless\": " + std::string(stream_lossy ? "false" : "true");
        if (stream_lossy) manifest += ", \"snr_db\": " + fmt_double(snr_db);
        manifest += "}";
        (void)json_escape;
    }

    manifest += "\n  }\n}\n";
    {
        const std::string mpath = output_dir + "/rank_" + rankbuf + "_manifest.json";
        std::ofstream mf(mpath, std::ios::trunc);
        if (!mf) throw std::runtime_error("[compressor] cannot open " + mpath + ": " + std::strerror(errno));
        write_or_throw(mf, manifest.data(), manifest.size(), mpath, "manifest");
        close_or_throw(mf, mpath);
    }

    // aggregate over streams
    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
        const StreamMetrics& s = m.per_stream[c];
        m.orig_bytes       += s.orig_bytes;
        m.compressed_bytes += s.compressed_bytes;
        m.n_tensors        += s.n_tensors;
        m.compute_ms       += s.compute_ms;
        m.d2h_ms           += s.d2h_ms;
        m.sq_err           += s.sq_err;
        m.signal           += s.signal;
        if (!s.lossless) m.lossless = false;
    }

    if (logger_) {
        const double ratio = m.compressed_bytes
            ? static_cast<double>(m.orig_bytes) / static_cast<double>(m.compressed_bytes) : 1.0;
        logger_->log_message(
            "[compressor] done tensors=" + std::to_string(m.n_tensors) +
            " orig_bytes=" + std::to_string(m.orig_bytes) +
            " compressed_bytes=" + std::to_string(m.compressed_bytes) +
            " ratio=" + fmt_double(ratio) +
            " critical_path_ms=" + fmt_double(m.critical_path_ms) +
            " compute_ms=" + fmt_double(m.compute_ms) +
            " d2h_ms=" + fmt_double(m.d2h_ms) +
            " io_ms=" + fmt_double(m.io_ms));
    }
}

// ------------------------------------------------------------------ //
//  metrics -> py::dict   (call with the GIL held)
// ------------------------------------------------------------------ //
py::dict metrics_to_pydict(const CompressionMetrics& m) {
    auto ratio = [](uint64_t o, uint64_t c) {
        return c ? static_cast<double>(o) / static_cast<double>(c) : 1.0;
    };
    auto snr_obj = [](bool lossless, double sq, double sig) -> py::object {
        if (lossless) return py::object(py::str("lossless"));
        if (sq <= 0.0) return py::object(py::float_(std::numeric_limits<double>::infinity()));
        return py::object(py::float_(10.0 * std::log10(sig / sq)));
    };

    py::dict overall;
    overall["orig_bytes"]       = m.orig_bytes;
    overall["compressed_bytes"] = m.compressed_bytes;
    overall["ratio"]            = ratio(m.orig_bytes, m.compressed_bytes);
    overall["snr_db"]           = snr_obj(m.lossless, m.sq_err, m.signal);
    overall["lossless"]         = m.lossless;
    overall["critical_path_ms"] = m.critical_path_ms;
    overall["compute_ms"]       = m.compute_ms;
    overall["d2h_ms"]           = m.d2h_ms;
    overall["io_ms"]            = m.io_ms;
    overall["wall_ms"]          = m.wall_ms;
    overall["n_tensors"]        = m.n_tensors;
    overall["pending"]          = m.pending;

    py::dict per;
    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
        const StreamMetrics& s = m.per_stream[c];
        if (s.n_tensors == 0) continue;
        py::dict d;
        d["n_tensors"]        = s.n_tensors;
        d["error_bound"]      = m.error_bounds[c];
        d["orig_bytes"]       = s.orig_bytes;
        d["compressed_bytes"] = s.compressed_bytes;
        d["ratio"]            = ratio(s.orig_bytes, s.compressed_bytes);
        d["snr_db"]           = snr_obj(s.lossless, s.sq_err, s.signal);
        d["lossless"]         = s.lossless;
        d["compute_ms"]       = s.compute_ms;
        d["d2h_ms"]           = s.d2h_ms;
        d["io_ms"]            = s.io_ms;
        per[tensor_category_name(static_cast<TensorCategory>(c))] = d;
    }

    py::dict mem;
    mem["staging_device_bytes"]         = m.staging_device_bytes;
    mem["pinned_host_bytes"]            = m.pinned_host_bytes;
    mem["codec_scratch_device_bytes"]   = m.codec_scratch_device_bytes;
    mem["peak_device_overhead_bytes"]   = m.peak_device_overhead_bytes;
    mem["measured_free_vram_delta_bytes"] = m.measured_free_vram_delta_b;
    mem["staging_mode"]                 = m.staging_mode;
    mem["ring_depth"]                   = m.ring_depth;
    mem["workers"]                      = m.workers;

    py::dict out;
    out["overall"]     = overall;
    out["per_stream"]  = per;
    out["memory"]      = mem;
    out["output_dir"]  = m.output_dir;
    out["error_bound"] = m.error_bound;
    {
        py::dict ebd;
        for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c)
            ebd[tensor_category_name(static_cast<TensorCategory>(c))] = m.error_bounds[c];
        out["error_bounds"] = ebd;
    }
    return out;
}

} // namespace checkers
