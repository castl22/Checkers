// Standalone exerciser for the per-group RMS quant codec (Phase B), so its
// ratio / SNR can be checked against the Python reference (SOTA/EQ/
// compress_master.py, RMS_REL_PG) on identical tensors before the codec is
// trusted in the training path.
//
//   checkers_py.codec_selftest(np.float32 array, dtype="float32"|"bfloat16"|
//                              "float16", eps=0.05, group=256, zstd_level=3)
//     -> dict(orig_bytes, orig_fp32_bytes, comp_bytes, payload_bytes,
//             aux_bytes, ratio, ratio_vs_fp32, snr_db, quant_snr_db,
//             rmse, max_abs_err, n_over, n_groups, eps, group)

#include "checkers/CheckpointFormat.hpp"
#include "checkers/CodecKernels.hpp"
#include "checkers/GpuDefs.hpp"
#include "checkers/PgQuant.hpp"
#include "checkers/TensorResource.hpp"
#include "checkers/TtLinalg.hpp"
#include "checkers/ZstdUtil.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace checkers {

namespace {

struct DevFree {
    void* p = nullptr;
    DevFree() = default;
    explicit DevFree(void* q) : p(q) {}
    DevFree(const DevFree&)            = delete;
    DevFree& operator=(const DevFree&) = delete;
    DevFree(DevFree&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevFree& operator=(DevFree&& o) noexcept {
        if (this != &o) { if (p) (void)gpu::Free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    ~DevFree() { if (p) (void)gpu::Free(p); }
};

// Pinned, reused host staging buffer.
//
// The selftest used std::vector<uint8_t>(n*2) for the payload, which is (a)
// PAGEABLE -- the D2H ran at 2.7 GB/s instead of ~25 -- and (b) value-
// initialising, so every call memset and first-touched 256 MB.  Together those
// were ~250 ms of the ~276 ms "GPU segment" on a 512 MB tensor: harness cost,
// not codec cost.  Production never paid it -- StagingPool::allocate already
// uses gpu::HostAlloc -- so this only makes the selftest measure the codec.
struct PinnedBuf {
    void*  p = nullptr;
    size_t n = 0;
    void* get(size_t bytes) {
        if (bytes > n) {
            if (p) (void)gpu::HostFree(p);
            void* q = nullptr;
            CK_GPU_CHECK(gpu::HostAlloc(&q, bytes));
            p = q; n = bytes;
        }
        return p;
    }
    ~PinnedBuf() { if (p) (void)gpu::HostFree(p); }
};

// Allocate device scratch AND fault it in.
//
// The allocation itself has always sat outside the timers, but the FIRST TOUCH
// of a fresh mapping did not: on the MI300A APU that is a page fault taken
// inside the kernel, and the selftests hipMalloc/hipFree a dozen n-sized
// buffers on every call -- once per 512 MB chunk.  It showed up as
// master_weights reading bitpack(GPU) 3214 ms against exp_avg_sq's 788 ms for
// the identical kernel on the identical element count.  cuSZp's harness
// preallocates once (_ensure) and lpmm rides torch's caching allocator, so
// paying the fault here is what makes the three measure the same thing.  The
// memset changes no codec output and no peak footprint.
// Deterministic tally of what THIS call asked the driver for.  The harness's
// peak-memory column is sampled from the HIP runtime on a polling thread, which
// can miss short-lived allocations entirely; this is an exact, auditable number
// to check it against.  Reset per call, so it is the simultaneous footprint.
size_t g_dev_alloc_bytes = 0;

void* dmalloc(std::vector<DevFree>& keep, size_t bytes) {
    if (bytes == 0) return nullptr;
    void* p = nullptr;
    CK_GPU_CHECK(gpu::Malloc(&p, bytes));
    CK_GPU_CHECK(gpu::MemsetAsync(p, 0, bytes, nullptr));
    keep.emplace_back(p);
    g_dev_alloc_bytes += bytes;
    return p;
}

DataType parse_storage_dtype(const std::string& s) {
    if (s == "float32" || s == "fp32")  return DataType::Float32;
    if (s == "bfloat16" || s == "bf16") return DataType::BFloat16;
    if (s == "float16" || s == "fp16")  return DataType::Float16;
    throw std::runtime_error("codec_selftest: dtype must be float32 / bfloat16 / float16");
}

} // namespace

py::dict codec_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                        const std::string& dtype_str,
                        double             eps,
                        uint32_t           group,
                        int                zstd_level,
                        bool               global_step,
                        bool               log_pw_rel,
                        int                zstd_workers,
                        bool               legacy_fwd,
                        int                entropy) {
    const size_t n = static_cast<size_t>(arr.size());
    if (n == 0) throw std::runtime_error("codec_selftest: empty array");
    g_dev_alloc_bytes = 0;
    if (group == 0) group = 256;
    if (log_pw_rel) global_step = false;   // mutually exclusive; log path wins
    constexpr float LOG_TINY = 1e-30f;

    // Force the GPU runtime to initialize (ROCm idiom, matches initialize_context)
    // then bind a device, so the self-test works even standalone.
    (void)gpu::Free(nullptr);
    int dev = 0;
    if (gpu::GetDevice(&dev) != gpu::Success) { (void)gpu::SetDevice(0); dev = 0; }
    CK_GPU_CHECK(gpu::SetDevice(dev));
    const DataType dt = parse_storage_dtype(dtype_str);
    const size_t   elem = dtype_element_size(dt);
    const uint32_t ng = static_cast<uint32_t>((n + group - 1) / group);
    const size_t   cap = std::max<size_t>(1024, n / 32);

    std::vector<DevFree> keep;
    keep.reserve(16);
    float*    d_src   = static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    void*     d_store = (dt == DataType::Float32) ? nullptr : dmalloc(keep, n * elem);
    float*    d_ref   = (dt == DataType::Float32) ? d_src
                        : static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    int16_t*  d_q     = static_cast<int16_t*>(dmalloc(keep, n * sizeof(int16_t)));
    float*    d_rms   = static_cast<float*>(dmalloc(keep, ng * sizeof(float)));
    float*    d_step  = static_cast<float*>(dmalloc(keep, ng * sizeof(float)));
    uint32_t* d_ovi   = static_cast<uint32_t*>(dmalloc(keep, cap * sizeof(uint32_t)));
    float*    d_ovv   = static_cast<float*>(dmalloc(keep, cap * sizeof(float)));
    uint32_t* d_ovc   = static_cast<uint32_t*>(dmalloc(keep, sizeof(uint32_t)));
    float*    d_dec    = static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    // The log(v) transform is folded into the fused quantize kernel, so the
    // separate n-element log buffer is only allocated for the legacy path.
    float*    d_qin    = (legacy_fwd && log_pw_rel)
                       ? static_cast<float*>(dmalloc(keep, n * sizeof(float)))
                       : nullptr;
    // Bit-pack scratch, allocated (and faulted in) before ANY clock starts.
    // d_w has to exist before the quantizer runs, because the quantizer is what
    // now fills it.  d_packed is sized for the worst case -- every code 16 bits
    // wide, i.e. the same n*2 the unpacked codes take -- rather than this
    // chunk's packed_bytes, so no hipMalloc ever lands inside a timed stage.
    const uint32_t nblk = static_cast<uint32_t>(
        (n + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
    uint8_t*  d_w      = (entropy == 1)
                       ? static_cast<uint8_t*>(dmalloc(keep, nblk)) : nullptr;
    uint64_t* d_off    = (entropy == 1)
                       ? static_cast<uint64_t*>(dmalloc(keep, (size_t)nblk * sizeof(uint64_t)))
                       : nullptr;
    uint8_t*  d_packed = (entropy == 1)
                       ? static_cast<uint8_t*>(dmalloc(keep, n * sizeof(uint16_t)))
                       : nullptr;

    // ---- timing: the benchmark needs compress and decompress SEPARATELY, so
    // bracket the forward path (H2D + GPU quant + zstd) and the inverse path
    // independently.  Device syncs make the GPU segments real rather than
    // just measuring kernel launches.  Error metrics below are NOT counted.
    using ck_clk = std::chrono::steady_clock;
    auto ck_ms = [](ck_clk::time_point t0) {
        return std::chrono::duration<double, std::milli>(ck_clk::now() - t0).count();
    };
    (void)gpu::DeviceSynchronize();
    const auto t_comp0 = ck_clk::now();

    // H2D is timed separately so the caller can report either boundary:
    // host->host (compress_ms) or device-resident->host (compress_ms - h2d_ms).
    // gpu::Memcpy is synchronous, so bracketing it directly is exact.
    CK_GPU_CHECK(gpu::Memcpy(d_src, arr.data(), n * sizeof(float), gpu::Copy::H2D));
    const double h2d_ms = ck_ms(t_comp0);

    if (dt != DataType::Float32) {   // storage round-trip: ref = what the codec sees
        codec::contract_from_fp32(d_src, dt, n, d_store, nullptr);
        codec::expand_to_fp32(d_store, dt, n, d_ref, nullptr);
    }

    // d_ref is the linear reference we score against.  The quantizer reads it
    // directly: for log_pw_rel the log transform happens inside the fused
    // kernel rather than in a separate pass over an n-element buffer.
    const float* d_qin_ref = d_ref;
    const int    log_dom   = log_pw_rel ? 1 : 0;

    // ---- forward ----
    // Sub-stage timers: the GPU segment was ~300 ms for a 512 MB tensor and the
    // fusion work barely moved it, so attribute it rather than guess again.
    double rms_ms = 0.0, steps_ms = 0.0, quant_ms = 0.0, payload_d2h_ms = 0.0;
    auto stamp = [&](ck_clk::time_point s) {
        (void)gpu::DeviceSynchronize(); return ck_ms(s);
    };
    PgSteps ps;
    ps.shrink.assign(ng, 1.0f);
    ps.dcodes.assign(ng, 0u);
    uint32_t n_over = 0;

    if (legacy_fwd) {
        // The ORIGINAL sequence, kept so the fused path can be A/B'd against it
        // in the same process on the same GPU.  Timings taken across builds or
        // across shared login nodes vary by 5-25% run to run, which is the same
        // order as the effect being measured, so they cannot attribute a change.
        if (log_pw_rel) {
            codec::log_clamp_fp32(d_ref, n, LOG_TINY, d_qin, nullptr);
            d_qin_ref = d_qin;
        }
        codec::group_rms_fp32(d_qin_ref, n, group, d_rms, ng, nullptr);
        std::vector<float> h_rms(ng);
        CK_GPU_CHECK(gpu::Memcpy(h_rms.data(), d_rms, ng * sizeof(float), gpu::Copy::D2H));
        if (log_pw_rel) {
            const double D = 2.0 * std::log1p(eps) * (1.0 - 1e-6);
            ps.pg_lo = std::log(D); ps.pg_step = 0.0;
            ps.d_use.assign(ng, static_cast<float>(D));
        } else if (global_step) {
            double ms = 0.0;
            for (uint32_t g = 0; g < ng; ++g) ms += double(h_rms[g]) * double(h_rms[g]);
            const double D = 2.0 * eps * std::sqrt(ms / double(ng)) * (1.0 - 1e-6);
            ps.pg_lo = std::log(std::max(D, 1e-300)); ps.pg_step = 0.0;
            ps.d_use.assign(ng, static_cast<float>(D));
        } else {
            ps = quantize_steps(h_rms, eps);
        }
        CK_GPU_CHECK(gpu::Memcpy(d_step, ps.d_use.data(), ng * sizeof(float), gpu::Copy::H2D));
        CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));
        codec::quantize_pergroup(d_qin_ref, n, group, d_step, d_q,
                                 d_ovi, d_ovv, static_cast<uint32_t>(cap), d_ovc, nullptr);
        CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
        codec::zigzag_inplace(d_q, n, nullptr);
    } else {
        if (log_pw_rel) {
            // D = 2*log1p(eps) is data-INDEPENDENT.  The legacy path still ran a
            // full group_rms pass over the whole array and D2H'd ng floats before
            // throwing both away; CheckpointCompressor guards this, the selftest
            // did not, so exp_avg_sq paid for a reduction it never used.
            const double D = 2.0 * std::log1p(eps) * (1.0 - 1e-6);
            codec::fill_f32(d_step, ng, static_cast<float>(D), nullptr);
            ps.pg_lo = std::log(D);
            ps.pg_step = 0.0;
        } else if (global_step) {
            codec::group_rms_fp32(d_ref, n, group, d_rms, ng, nullptr);
            std::vector<float> h_rms(ng);
            CK_GPU_CHECK(gpu::Memcpy(h_rms.data(), d_rms, ng * sizeof(float), gpu::Copy::D2H));
            double ms = 0.0;
            for (uint32_t g = 0; g < ng; ++g) ms += double(h_rms[g]) * double(h_rms[g]);
            const double D = 2.0 * eps * std::sqrt(ms / double(ng)) * (1.0 - 1e-6);
            codec::fill_f32(d_step, ng, static_cast<float>(D), nullptr);
            ps.pg_lo = std::log(std::max(D, 1e-300));
            ps.pg_step = 0.0;
        } else {
            { const auto s = ck_clk::now();
              codec::group_rms_fp32(d_ref, n, group, d_rms, ng, nullptr);
              rms_ms = stamp(s); }
            const auto s_steps = ck_clk::now();
            // Steps stay on the HOST, via the shared quantize_steps().  A device
            // version was tried and reverted: it changed a handful of dcodes by
            // one ULP (device vs host log/exp), which moved comp_bytes by 613 B
            // in 122 MB and the ratio in the 4th decimal, for a 1.01-1.09x gain.
            // Not a trade worth making -- the codec output must stay identical.
            std::vector<float> h_rms(ng);
            CK_GPU_CHECK(gpu::Memcpy(h_rms.data(), d_rms, ng * sizeof(float),
                                     gpu::Copy::D2H));
            ps = quantize_steps(h_rms, eps);
            CK_GPU_CHECK(gpu::Memcpy(d_step, ps.d_use.data(), ng * sizeof(float),
                                     gpu::Copy::H2D));
            steps_ms = stamp(s_steps);
        }

        CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));
        const auto s_q = ck_clk::now();
        // Fused [log ->] quantize -> zigzag -> bit-pack widths, one pass.
        // Passing d_w is what removes the separate bitpack_widths kernel.
        codec::quantize_fused(d_qin_ref, n, group, d_step,
                              reinterpret_cast<uint16_t*>(d_q),
                              d_ovi, d_ovv, static_cast<uint32_t>(cap), d_ovc,
                              log_dom, LOG_TINY, nullptr, d_w);
        CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
        quant_ms = stamp(s_q);
    }

    const uint32_t n_over_kept = std::min<uint32_t>(n_over, static_cast<uint32_t>(cap));

    // NOTE: reduce_quant_error used to run HERE, inside the compress bracket.
    // It is a full fp64 reduction over every element and is a measurement, not
    // part of the codec, so it inflated compress_ms (and only for this codec --
    // tt_eq_selftest never had it, which made the two paths incomparable).  It
    // now runs after decompress_ms, where d_q has been un-zigzagged back to
    // quantization codes and is valid again.
    // (zigzag_inplace is gone -- the fused kernel already emitted zigzag u16)
    // entropy == 0 : zstd on the host (original path, still the default)
    // entropy == 1 : GPU fixed-width bit-packing -- the uncompressed payload
    //                never crosses the bus and there is no CPU stage at all.
    //                Lossless over the zigzag codes, so the reconstruction and
    //                SNR are unchanged; only the compressed SIZE differs.
    static PinnedBuf g_zz;                       // reused across calls
    const size_t   zz_bytes = n * sizeof(uint16_t);
    std::vector<uint8_t>  h_w;
    std::vector<uint64_t> h_off;
    static PinnedBuf g_pk;        // pinned, reused target for the packed bytes
    std::vector<uint8_t> w_comp;  // zstd'd width array
    uint8_t*  pk_host  = nullptr; // bit-pack payload (widths ++ packed bits)
    size_t    pk_bytes = 0;
    size_t    packed_bytes = 0;
    double    pack_gpu_ms = 0.0, pack_io_ms = 0.0;
    double    unpack_gpu_ms = 0.0, unpack_io_ms = 0.0;

    std::vector<uint32_t> h_ovi(n_over_kept);
    std::vector<float>    h_ovv(n_over_kept);
    if (n_over_kept) {
        CK_GPU_CHECK(gpu::Memcpy(h_ovi.data(), d_ovi, n_over_kept * sizeof(uint32_t), gpu::Copy::D2H));
        CK_GPU_CHECK(gpu::Memcpy(h_ovv.data(), d_ovv, n_over_kept * sizeof(float),    gpu::Copy::D2H));
    }

    std::vector<uint8_t> aux_raw;
    aux_raw.reserve(ng + ng * 4u + static_cast<size_t>(n_over_kept) * 8u);
    aux_raw.insert(aux_raw.end(), ps.dcodes.begin(), ps.dcodes.end());
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(ps.shrink.data());
    aux_raw.insert(aux_raw.end(), sp, sp + ng * 4u);
    for (uint32_t k = 0; k < n_over_kept; ++k) {
        const uint8_t* ip = reinterpret_cast<const uint8_t*>(&h_ovi[k]);
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(&h_ovv[k]);
        aux_raw.insert(aux_raw.end(), ip, ip + 4);
        aux_raw.insert(aux_raw.end(), vp, vp + 4);
    }

    // Entropy stage timed separately: it is the CPU tail of the pipeline, and a
    // like-for-like comparison against a GPU-only bit-packer needs it isolated
    // rather than silently folded into (or dropped from) the total.
    const auto t_zstd0 = ck_clk::now();
    std::vector<uint8_t> payload, auxc;
    if (entropy == 1) {
        // The widths came out of quantize_fused above.  The legacy forward path
        // does not emit them, so it still needs the standalone pass.
        if (legacy_fwd) {
            const auto s_g0 = ck_clk::now();
            codec::bitpack_widths(reinterpret_cast<const uint16_t*>(d_q), n, d_w,
                                  nblk, nullptr);
            CK_GPU_CHECK(gpu::DeviceSynchronize());
            pack_gpu_ms = ck_ms(s_g0);
        }
        const auto s_io0 = ck_clk::now();
        h_w.resize(nblk);
        CK_GPU_CHECK(gpu::Memcpy(h_w.data(), d_w, nblk, gpu::Copy::D2H));

        // exclusive prefix sum of 32*w[b] -- a block is exactly 32*w bytes,
        // which is why BITPACK_BLOCK is 256 (no cross-block bit alignment).
        h_off.resize(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { h_off[b] = run; run += 32ull * h_w[b]; }
        packed_bytes = static_cast<size_t>(run);
        CK_GPU_CHECK(gpu::Memcpy(d_off, h_off.data(),
                                 (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D));
        pack_io_ms = ck_ms(s_io0);

        const auto s_g1 = ck_clk::now();
        codec::bitpack_encode(reinterpret_cast<const uint16_t*>(d_q), n, d_w,
                              d_off, d_packed, nblk, nullptr);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        pack_gpu_ms += ck_ms(s_g1);
        const auto s_io1 = ck_clk::now();

        // Pinned + reused: a fresh std::vector here was pageable AND
        // value-initialised, which made this stage read 1177 ms against the
        // 177 ms the packing actually takes.  Same defect as the old zstd
        // staging buffer.
        // The WIDTH array is tiny (n/256 bytes) but highly redundant -- after
        // TT most blocks are all-zero, so their width is 0.  Leaving it raw
        // capped the TT ratio at 1024x (13097 -> 919).  zstd on just the widths
        // is ~500x less CPU work than zstd on the payload and restores it.
        w_comp = zstd_compress_mt(h_w.data(), nblk, zstd_level, zstd_workers);
        pk_bytes = 4 + w_comp.size() + packed_bytes;
        pk_host  = static_cast<uint8_t*>(g_pk.get(pk_bytes));
        const uint32_t wcs = static_cast<uint32_t>(w_comp.size());
        std::memcpy(pk_host, &wcs, 4);
        std::memcpy(pk_host + 4, w_comp.data(), wcs);
        if (packed_bytes)
            CK_GPU_CHECK(gpu::Memcpy(pk_host + 4 + wcs, d_packed, packed_bytes,
                                     gpu::Copy::D2H));
        pack_io_ms += ck_ms(s_io1);
    } else {
        uint8_t* const h_zz = static_cast<uint8_t*>(g_zz.get(zz_bytes));
        { const auto s = ck_clk::now();
          CK_GPU_CHECK(gpu::Memcpy(h_zz, d_q, zz_bytes, gpu::Copy::D2H));
          payload_d2h_ms = ck_ms(s); }
        payload = zstd_compress_mt(h_zz, zz_bytes, zstd_level, zstd_workers);
    }
    const auto s_aux = ck_clk::now();
    auxc = zstd_compress_mt(aux_raw.data(), aux_raw.size(), zstd_level, zstd_workers);
    // The aux blob (step codes + overflow) is the last CPU work in the tail;
    // book it with the rest of it so pack_io_ms is ALL of the non-GPU time and
    // the caller can strip exactly that to get a device-only number.
    if (entropy == 1) pack_io_ms += ck_ms(s_aux);
    const double zstd_ms = ck_ms(t_zstd0) - payload_d2h_ms;

    (void)gpu::DeviceSynchronize();
    const double compress_ms = ck_ms(t_comp0);

    // ---- inverse: a TRUE round trip, starting from the COMPRESSED BYTES ----
    // This used to begin at d_q, still device-resident from the forward pass, so
    // it measured neither the entropy decode nor the H2D of the payload -- the
    // decompress-side mirror of the compress-side boundary problem.  Starting
    // from `payload` makes decompress_ms comparable with a compressor that must
    // also unpack its own bitstream.  d_step is reused rather than rebuilt from
    // aux: reconstructing it is an ng-sized operation that lives in the
    // production Restart path, not here, and is negligible beside the payload.
    const auto t_dec0 = ck_clk::now();
    double unzstd_ms = 0.0, dec_h2d_ms = 0.0;
    std::vector<uint8_t> aux_dec;
    if (entropy == 1) {
        // A real decoder rebuilds the offsets from the widths it received, so
        // do that rather than reusing the encoder's array.
        uint32_t wcs = 0; std::memcpy(&wcs, pk_host, 4);
        std::vector<uint8_t> rw = zstd_decompress(pk_host + 4, wcs, nblk);
        std::vector<uint64_t> roff(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { roff[b] = run; run += 32ull * rw[b]; }
        aux_dec   = zstd_decompress(auxc.data(), auxc.size(), aux_raw.size());
        unzstd_ms = ck_ms(t_dec0);

        const auto s2 = ck_clk::now();
        CK_GPU_CHECK(gpu::Memcpy(d_w, rw.data(), nblk, gpu::Copy::H2D));
        CK_GPU_CHECK(gpu::Memcpy(d_off, roff.data(),
                                 (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D));
        if (packed_bytes)
            CK_GPU_CHECK(gpu::Memcpy(d_packed, pk_host + 4 + wcs, packed_bytes,
                                     gpu::Copy::H2D));
        dec_h2d_ms   = ck_ms(s2);
        unpack_io_ms = dec_h2d_ms;

        // Fused unpack -> unzigzag -> dequant -> [exp], one kernel.  exp can
        // only be folded in when there are no overflow values: those are
        // scattered AFTER the dequant and must go through exp too, so with
        // n_over > 0 the separate exp_fp32 pass below still runs.
        const int dec_log = (log_pw_rel && n_over_kept == 0) ? 1 : 0;
        const auto s3 = ck_clk::now();
        codec::bitpack_dequant(d_packed, d_w, d_off, n, group, d_step, d_dec,
                               nblk, dec_log, nullptr);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        unpack_gpu_ms = ck_ms(s3);
        if (n_over_kept)
            codec::scatter_overflow(d_dec, d_ovi, d_ovv, n_over_kept, nullptr);
        if (log_pw_rel && !dec_log)
            codec::exp_fp32(d_dec, n, d_dec, nullptr);
        else if (global_step)
            codec::clamp_floor_fp32(d_dec, n, 1e-16f, nullptr);  // exp_avg_sq floor
    } else {
        static PinnedBuf g_zz_dec;               // pinned target for the decode
        uint8_t* const h_zz_dec = static_cast<uint8_t*>(g_zz_dec.get(zz_bytes));
        (void)zstd_decompress_into(h_zz_dec, zz_bytes, payload.data(), payload.size());
        aux_dec   = zstd_decompress(auxc.data(), auxc.size(), aux_raw.size());
        unzstd_ms = ck_ms(t_dec0);
        const auto s2 = ck_clk::now();
        CK_GPU_CHECK(gpu::Memcpy(d_q, h_zz_dec, zz_bytes, gpu::Copy::H2D));
        dec_h2d_ms = ck_ms(s2);

        codec::unzigzag_inplace(reinterpret_cast<uint16_t*>(d_q), n, nullptr);
        codec::dequant_pergroup(d_q, n, group, d_step, d_dec, nullptr);
        if (n_over_kept)
            codec::scatter_overflow(d_dec, d_ovi, d_ovv, n_over_kept, nullptr);
        if (log_pw_rel)
            codec::exp_fp32(d_dec, n, d_dec, nullptr);            // v_hat = exp(q*D)
        else if (global_step)
            codec::clamp_floor_fp32(d_dec, n, 1e-16f, nullptr);   // exp_avg_sq floor
    }
    (void)aux_dec.size();
    if (dt != DataType::Float32) {
        codec::contract_from_fp32(d_dec, dt, n, d_store, nullptr);
        codec::expand_to_fp32(d_store, dt, n, d_dec, nullptr);
    }

    (void)gpu::DeviceSynchronize();
    const double decompress_ms = ck_ms(t_dec0);

    // ---- measurements: outside BOTH brackets ----
    // The inverse un-zigzagged d_q in place, so it holds quantization codes
    // again and reduce_quant_error is valid here.  d_qin_ref is untouched by
    // the inverse in every dtype/codec combination.
    // quant_snr compares the codes against what the quantizer consumed.  With
    // the log transform fused into the kernel that buffer is no longer
    // materialised, so for log_pw_rel this diagnostic is skipped (reported as
    // +inf).  The headline SNR is unaffected: it is measured in the linear
    // domain, d_ref vs d_dec, below.
    // The bit-pack inverse decodes straight to floats and never un-zigzags d_q,
    // which still holds the FORWARD pass's zigzag words; undo them here, outside
    // the clock, purely so this diagnostic stays valid.
    double quant_sq = 0.0, quant_sig = 0.0;
    if (entropy == 1)
        codec::unzigzag_inplace(reinterpret_cast<uint16_t*>(d_q), n, nullptr);
    if (!log_pw_rel || legacy_fwd) {
        codec::reduce_quant_error(d_q, d_step, d_qin_ref, n, group,
                                  &quant_sq, &quant_sig, nullptr);
    }

    double sq = 0.0, sig = 0.0, maxae = 0.0, maxre = 0.0;
    codec::reduce_sq_error_fp32(d_ref, d_dec, n, &sq, &sig, nullptr);
    codec::reduce_max_abs_err(d_ref, d_dec, n, &maxae, nullptr);
    codec::reduce_max_rel_err(d_ref, d_dec, n, &maxre, nullptr);
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    const size_t comp_bytes  = (entropy == 1 ? pk_bytes : payload.size()) + auxc.size();
    const size_t orig_bytes  = n * elem;
    const size_t orig_fp32   = n * sizeof(float);
    const double snr_db      = (sq > 0.0)      ? 10.0 * std::log10(sig / sq)             : 1e9;
    const double quant_snr   = (quant_sq > 0.0)? 10.0 * std::log10(quant_sig / quant_sq): 1e9;
    const double rmse        = std::sqrt(sq / static_cast<double>(n));

    py::dict d;
    d["compress_ms"]     = compress_ms;    // H2D + GPU quant + zstd  (host->host)
    d["decompress_ms"]   = decompress_ms;  // GPU inverse (excludes error metrics)
    d["h2d_ms"]          = h2d_ms;         // subtract for a device-resident start
    d["zstd_ms"]         = zstd_ms;        // CPU entropy tail, isolated
    d["legacy_fwd"]      = legacy_fwd;
    d["entropy"]         = entropy;
    d["pack_gpu_ms"]     = pack_gpu_ms;
    d["pack_io_ms"]      = pack_io_ms;
    d["unpack_gpu_ms"]   = unpack_gpu_ms;
    d["unpack_io_ms"]    = unpack_io_ms;
    d["packed_bytes"]    = (uint64_t)packed_bytes;
    d["width_bytes"]     = (uint64_t)nblk;
    d["rms_ms"]          = rms_ms;
    d["steps_ms"]        = steps_ms;
    d["quant_ms"]        = quant_ms;
    d["payload_d2h_ms"]  = payload_d2h_ms;
    d["unzstd_ms"]       = unzstd_ms;      // entropy decode (payload + aux)
    d["dec_h2d_ms"]      = dec_h2d_ms;     // H2D of the decoded payload
    d["dev_alloc_bytes"] = (uint64_t)g_dev_alloc_bytes;
    d["orig_bytes"]      = orig_bytes;
    d["orig_fp32_bytes"] = orig_fp32;
    d["comp_bytes"]      = comp_bytes;
    d["payload_bytes"]   = (entropy == 1 ? pk_bytes : payload.size());
    d["aux_bytes"]       = auxc.size();
    d["ratio"]           = static_cast<double>(orig_bytes) / static_cast<double>(comp_bytes);
    d["ratio_vs_fp32"]   = static_cast<double>(orig_fp32)  / static_cast<double>(comp_bytes);
    d["snr_db"]          = snr_db;
    d["quant_snr_db"]    = quant_snr;
    d["rmse"]            = rmse;
    d["max_abs_err"]     = maxae;
    d["max_rel_err"]     = maxre;
    d["n_over"]          = n_over;
    d["n_groups"]        = ng;
    d["eps"]             = eps;
    d["group"]           = group;
    d["dtype"]           = dtype_str;
    d["global_step"]     = global_step;
    d["log_pw_rel"]      = log_pw_rel;
    return d;
}

// ------------------------------------------------------------------ //
//  tt_selftest: TT-SVD decompose + reconstruct round trip (no quant),
//  to validate TtLinalg in isolation.
// ------------------------------------------------------------------ //
py::dict tt_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                     uint32_t rank_cap, uint32_t ndim, double energy_keep) {
    const size_t n = static_cast<size_t>(arr.size());
    if (n == 0) throw std::runtime_error("tt_selftest: empty array");
    if (rank_cap == 0) rank_cap = 16;
    if (energy_keep <= 0.0 || energy_keep > 1.0) energy_keep = 0.999;

    (void)gpu::Free(nullptr);
    int dev = 0;
    if (gpu::GetDevice(&dev) != gpu::Success) { (void)gpu::SetDevice(0); dev = 0; }
    CK_GPU_CHECK(gpu::SetDevice(dev));

    const tt::TtDims dims = tt::choose_dims(n, ndim);
    const size_t P = static_cast<size_t>(dims.padded());

    std::vector<DevFree> keep;
    float* d_A    = static_cast<float*>(dmalloc(keep, P * sizeof(float)));
    float* d_Ahat = static_cast<float*>(dmalloc(keep, P * sizeof(float)));
    CK_GPU_CHECK(gpu::MemsetAsync(d_A, 0, P * sizeof(float), nullptr));
    CK_GPU_CHECK(gpu::Memcpy(d_A, arr.data(), n * sizeof(float), gpu::Copy::H2D));

    void* blas = tt::blas_handle();
    tt::blas_set_stream(blas, nullptr);

    tt::TtCores cores;
    const bool ok_dec = tt::tt_decompose(d_A, dims, rank_cap, energy_keep, cores, blas, nullptr);
    if (!ok_dec) { tt::tt_free(cores); throw std::runtime_error("tt_selftest: tt_decompose failed"); }
    const bool ok_rec = tt::tt_reconstruct(cores, d_Ahat, blas, nullptr);
    if (!ok_rec) { tt::tt_free(cores); throw std::runtime_error("tt_selftest: tt_reconstruct failed"); }

    double sq = 0.0, sig = 0.0, maxae = 0.0;
    codec::reduce_sq_error_fp32(d_A, d_Ahat, n, &sq, &sig, nullptr);
    codec::reduce_max_abs_err(d_A, d_Ahat, n, &maxae, nullptr);
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    const uint64_t core_elems = cores.core_elems();
    const uint32_t r1 = cores.r1, r2 = cores.r2;
    tt::tt_free(cores);

    py::dict d;
    d["n"]           = n;
    d["n1"]          = dims.n1;
    d["n2"]          = dims.n2;
    d["n3"]          = dims.n3;
    d["r1"]          = r1;
    d["r2"]          = r2;
    d["rank_cap"]    = rank_cap;
    d["ndim"]        = ndim;
    d["core_elems"]  = core_elems;
    d["lowrank_frac"]= static_cast<double>(core_elems) / static_cast<double>(n);
    d["rel_l2"]      = (sig > 0.0) ? std::sqrt(sq / sig) : 0.0;
    d["snr_db"]      = (sq > 0.0) ? 10.0 * std::log10(sig / sq) : 1e9;
    d["max_abs_err"] = maxae;
    return d;
}

// ------------------------------------------------------------------ //
//  tt_eq_selftest: the full exp_avg codec (capped TT-SVD -> EQ residual
//  quant -> zstd) forward+inverse round trip on one host array.
// ------------------------------------------------------------------ //
py::dict tt_eq_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                        double eps, uint32_t rank_cap, uint32_t ndim,
                        double energy_keep, int eq_mode, int zstd_level,
                        int zstd_workers, int entropy) {
    const size_t L = static_cast<size_t>(arr.size());
    if (L == 0) throw std::runtime_error("tt_eq_selftest: empty array");
    g_dev_alloc_bytes = 0;
    if (rank_cap == 0) rank_cap = 16;
    if (energy_keep <= 0.0 || energy_keep > 1.0) energy_keep = 0.999;

    (void)gpu::Free(nullptr);
    int dev = 0;
    if (gpu::GetDevice(&dev) != gpu::Success) { (void)gpu::SetDevice(0); dev = 0; }
    CK_GPU_CHECK(gpu::SetDevice(dev));

    const tt::TtDims dims = tt::choose_dims(L, ndim);
    const size_t P = static_cast<size_t>(dims.padded());
    const size_t cap = std::max<size_t>(1024, L / 16);

    // d_A is the input and the SNR reference, so it outlives both phases.  Every
    // buffer the COMPRESS side produces goes in keep_comp, which is cleared --
    // i.e. actually gpu::Free'd -- before the inverse starts, so the inverse has
    // to recompute rather than read a buffer left warm by the forward pass.
    std::vector<DevFree> keep, keep_comp, keep_dec;
    float*    d_A    = static_cast<float*>(dmalloc(keep,      P * sizeof(float)));
    float*    d_Ahat = static_cast<float*>(dmalloc(keep_comp, P * sizeof(float)));
    float*    d_R    = static_cast<float*>(dmalloc(keep_comp, L * sizeof(float)));
    int16_t*  d_q    = static_cast<int16_t*>(dmalloc(keep_comp, L * sizeof(int16_t)));
    float*    d_step = static_cast<float*>(dmalloc(keep_comp, sizeof(float)));
    uint32_t* d_ovi  = static_cast<uint32_t*>(dmalloc(keep_comp, cap * sizeof(uint32_t)));
    float*    d_ovv  = static_cast<float*>(dmalloc(keep_comp, cap * sizeof(float)));
    uint32_t* d_ovc  = static_cast<uint32_t*>(dmalloc(keep_comp, sizeof(uint32_t)));
    // Bit-pack scratch, allocated (and faulted in) before any clock starts --
    // d_w has to exist before the quantizer, which is what fills it now, and
    // d_packed is sized for the 16-bits-per-code worst case so no hipMalloc
    // lands inside a timed stage.  See codec_selftest for the reasoning.
    const size_t   zz_bytes = L * sizeof(uint16_t);
    const uint32_t nblk     = static_cast<uint32_t>(
        (L + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
    uint8_t*  d_w      = (entropy == 1)
                       ? static_cast<uint8_t*>(dmalloc(keep_comp, nblk)) : nullptr;
    uint64_t* d_off    = (entropy == 1)
                       ? static_cast<uint64_t*>(dmalloc(keep_comp, (size_t)nblk * sizeof(uint64_t)))
                       : nullptr;
    uint8_t*  d_packed = (entropy == 1)
                       ? static_cast<uint8_t*>(dmalloc(keep_comp, zz_bytes)) : nullptr;

    // Timing.  Unlike codec_selftest, this routine runs the inverse BEFORE it
    // serializes, so the compress cost is two segments (decompose+quantize,
    // then zstd) that are accumulated around the inverse.
    using ck_clk = std::chrono::steady_clock;
    auto ck_ms = [](ck_clk::time_point t0) {
        return std::chrono::duration<double, std::milli>(ck_clk::now() - t0).count();
    };
    (void)gpu::DeviceSynchronize();
    const auto t_comp0 = ck_clk::now();

    // The memset is the TT zero-padding (part of the codec); only the H2D copy
    // is transfer, so bracket that alone.  Subtract h2d_ms for a device-resident
    // start, matching codec_selftest.
    CK_GPU_CHECK(gpu::MemsetAsync(d_A, 0, P * sizeof(float), nullptr));
    const auto t_h2d0 = ck_clk::now();
    CK_GPU_CHECK(gpu::Memcpy(d_A, arr.data(), L * sizeof(float), gpu::Copy::H2D));
    const double h2d_ms = ck_ms(t_h2d0);

    // exp_avg is Checkers' slowest stream and 71% of its compress time was
    // unattributed; time the TT stages so it can be aimed at.
    double decompose_ms = 0.0, reconstruct_ms = 0.0, resid_ms = 0.0, quant_ms = 0.0;
    auto tstamp = [&](ck_clk::time_point s) {
        (void)gpu::DeviceSynchronize(); return ck_ms(s);
    };
    void* blas = tt::blas_handle();
    tt::blas_set_stream(blas, nullptr);
    tt::TtCores cores;
    tt::TtProf prof;
    { const auto s = ck_clk::now();
      if (!tt::tt_decompose(d_A, dims, rank_cap, energy_keep, cores, blas, nullptr, &prof))
          throw std::runtime_error("tt_eq_selftest: tt_decompose failed");
      decompose_ms = tstamp(s); }
    { const auto s = ck_clk::now();
      if (!tt::tt_reconstruct(cores, d_Ahat, blas, nullptr)) {
          tt::tt_free(cores); throw std::runtime_error("tt_eq_selftest: tt_reconstruct failed");
      }
      reconstruct_ms = tstamp(s); }
    const uint32_t r1 = cores.r1, r2 = cores.r2;
    const uint32_t cn1 = cores.n1, cn2 = cores.n2, cn3 = cores.n3;
    const uint64_t core_elems = cores.core_elems();
    const size_t   e1 = (size_t)cn1 * r1;
    const size_t   e2 = (size_t)r1 * cn2 * r2;
    const size_t   e3 = (size_t)r2 * cn3;

    const auto s_resid = ck_clk::now();
    codec::sub_fp32(d_A, d_Ahat, d_R, L, nullptr);
    double rmin = 0.0, rmax = 0.0;
    codec::reduce_minmax_fp32(d_R, L, &rmin, &rmax, nullptr);
    resid_ms = tstamp(s_resid);
    const double amax  = std::max(std::fabs(rmin), std::fabs(rmax));
    const double range = rmax - rmin;
    double abs_eps = (eq_mode == 0) ? eps : eps * std::max(range, 0.0);
    if (!(abs_eps > 0.0)) abs_eps = std::max(eps, 1e-30);
    const double dyn    = amax / std::max(abs_eps, (double)std::numeric_limits<float>::min());
    const double margin = 1.0 - std::min(1e-3, std::max(64.0 * 2.220446049250313e-16 * dyn, 1e-9));
    const double D  = 2.0 * abs_eps * margin;
    const float  Df = static_cast<float>(D);
    CK_GPU_CHECK(gpu::Memcpy(d_step, &Df, sizeof(float), gpu::Copy::H2D));
    CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));

    const auto s_quant = ck_clk::now();
    const uint32_t grp = static_cast<uint32_t>(L);
    // Fused quantize -> zigzag -> bit-pack widths, one pass.  This used to be
    // quantize_pergroup here plus a separate zigzag_inplace and bitpack_widths
    // in the serialization segment: three passes over the L-element residual.
    codec::quantize_fused(d_R, L, grp, d_step, reinterpret_cast<uint16_t*>(d_q),
                          d_ovi, d_ovv, static_cast<uint32_t>(cap), d_ovc,
                          0, 0.0f, nullptr, d_w);
    uint32_t n_over = 0;
    CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
    n_over = std::min<uint32_t>(n_over, static_cast<uint32_t>(cap));
    quant_ms = tstamp(s_quant);

    (void)gpu::DeviceSynchronize();
    double compress_ms = ck_ms(t_comp0);        // segment 1: decompose + quantize

    // The inverse deliberately does NOT run here.  It used to: it started from
    // device-resident d_q and reused d_Ahat carried over from the forward pass,
    // so decompress_ms covered neither the entropy decode, nor the H2D of the
    // payload, nor the TT reconstruction from the fp16 cores that are what is
    // actually stored -- it timed little more than a dequantize, on the one
    // stream the headline ratio comes from.  It now runs below, from the
    // serialized bytes, after every compress-side buffer has been freed.

    // serialized sizes: bit-packed (or zstd'd) u16 payload + zstd(TtHeader +
    // f16 cores + overflow)
    const auto t_comp1 = ck_clk::now();          // segment 2: serialization
    static PinnedBuf tt_zz;                      // pinned + reused, see PinnedBuf
    uint8_t*  h_zz     = nullptr;
    size_t    packed_bytes = 0, pk_bytes = 0;
    static PinnedBuf tt_pk;
    uint8_t*  pk_host = nullptr;
    std::vector<uint8_t> h_w;
    double    pack_gpu_ms = 0.0, pack_io_ms = 0.0, payload_d2h_ms = 0.0;
    if (entropy == 0) {
        h_zz = static_cast<uint8_t*>(tt_zz.get(zz_bytes));
        const auto s = ck_clk::now();
        CK_GPU_CHECK(gpu::Memcpy(h_zz, d_q, zz_bytes, gpu::Copy::D2H));
        payload_d2h_ms = ck_ms(s);
    }

    float* d_cf16 = static_cast<float*>(dmalloc(keep_comp, core_elems * sizeof(uint16_t)));
    {
        uint16_t* c16 = reinterpret_cast<uint16_t*>(d_cf16);
        codec::contract_from_fp32(cores.g1, DataType::Float16, e1, c16, nullptr);
        codec::contract_from_fp32(cores.g2, DataType::Float16, e2, c16 + e1, nullptr);
        codec::contract_from_fp32(cores.g3, DataType::Float16, e3, c16 + e1 + e2, nullptr);
    }
    static PinnedBuf tt_core;
    const size_t  core_bytes = core_elems * sizeof(uint16_t);
    uint8_t* const h_core    = static_cast<uint8_t*>(tt_core.get(core_bytes));
    CK_GPU_CHECK(gpu::Memcpy(h_core, d_cf16, core_bytes, gpu::Copy::D2H));

    // Real overflow bytes (idx u32, val f32).  This used to be zero padding
    // sized to n_over, which was harmless only because nothing read it back;
    // the decoder below has to scatter them, so they must be serialized.
    std::vector<uint32_t> h_ovi(n_over);
    std::vector<float>    h_ovv(n_over);
    if (n_over) {
        CK_GPU_CHECK(gpu::Memcpy(h_ovi.data(), d_ovi, n_over * sizeof(uint32_t), gpu::Copy::D2H));
        CK_GPU_CHECK(gpu::Memcpy(h_ovv.data(), d_ovv, n_over * sizeof(float),    gpu::Copy::D2H));
    }
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    // Populate the header for real, so the decode is driven by the serialized
    // metadata rather than by values still lying around in host variables.
    TtHeader th{};
    std::memset(&th, 0, sizeof(th));
    th.magic      = TTEQ_MAGIC;
    th.ndim       = static_cast<uint16_t>(ndim);
    th.eq_mode    = static_cast<uint16_t>(eq_mode);
    th.n1 = cn1;  th.n2 = cn2;  th.n3 = cn3;
    th.r1 = r1;   th.r2 = r2;
    th.n_elems    = L;
    th.core_elems = core_elems;
    th.n_over     = n_over;
    th.d_step     = D;
    th.resid_min  = static_cast<float>(rmin);
    th.resid_max  = static_cast<float>(rmax);
    th.core_f16   = 1u;

    std::vector<uint8_t> aux_raw;
    aux_raw.reserve(sizeof(TtHeader) + core_bytes + (size_t)n_over * 8u);
    const uint8_t* thp = reinterpret_cast<const uint8_t*>(&th);
    aux_raw.insert(aux_raw.end(), thp, thp + sizeof(TtHeader));
    aux_raw.insert(aux_raw.end(), h_core, h_core + core_bytes);
    for (uint32_t k = 0; k < n_over; ++k) {
        const uint8_t* ip = reinterpret_cast<const uint8_t*>(&h_ovi[k]);
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(&h_ovv[k]);
        aux_raw.insert(aux_raw.end(), ip, ip + 4);
        aux_raw.insert(aux_raw.end(), vp, vp + 4);
    }

    const auto t_zstd0 = ck_clk::now();
    std::vector<uint8_t> payload, auxc;
    if (entropy == 1) {
        // widths already came out of quantize_fused above
        const auto s_io0 = ck_clk::now();
        h_w.resize(nblk);
        CK_GPU_CHECK(gpu::Memcpy(h_w.data(), d_w, nblk, gpu::Copy::D2H));
        // exclusive prefix sum of 32*w[b] -- a block is exactly 32*w bytes.
        std::vector<uint64_t> h_off(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { h_off[b] = run; run += 32ull * h_w[b]; }
        packed_bytes = (size_t)run;
        CK_GPU_CHECK(gpu::Memcpy(d_off, h_off.data(),
                                 (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D));
        pack_io_ms = ck_ms(s_io0);

        const auto s_g1 = ck_clk::now();
        codec::bitpack_encode(reinterpret_cast<const uint16_t*>(d_q), L, d_w,
                              d_off, d_packed, nblk, nullptr);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        pack_gpu_ms += ck_ms(s_g1);

        const auto s_io1 = ck_clk::now();
        // zstd only the width array -- see the note in codec_selftest.
        auto w_comp = zstd_compress_mt(h_w.data(), nblk, zstd_level, zstd_workers);
        pk_bytes = 4 + w_comp.size() + packed_bytes;
        pk_host  = static_cast<uint8_t*>(tt_pk.get(pk_bytes));
        const uint32_t wcs = static_cast<uint32_t>(w_comp.size());
        std::memcpy(pk_host, &wcs, 4);
        std::memcpy(pk_host + 4, w_comp.data(), wcs);
        if (packed_bytes)
            CK_GPU_CHECK(gpu::Memcpy(pk_host + 4 + wcs, d_packed, packed_bytes, gpu::Copy::D2H));
        pack_io_ms += ck_ms(s_io1);
    } else {
        payload = zstd_compress_mt(h_zz, zz_bytes, zstd_level, zstd_workers);
    }
    const auto s_aux = ck_clk::now();
    auxc = zstd_compress_mt(aux_raw.data(), aux_raw.size(), zstd_level, zstd_workers);
    // Book the aux zstd with the rest of the host tail so pack_io_ms is ALL of
    // the non-GPU time and the caller can strip exactly that.
    if (entropy == 1) pack_io_ms += ck_ms(s_aux);
    const double zstd_ms = ck_ms(t_zstd0);
    (void)gpu::DeviceSynchronize();
    compress_ms += ck_ms(t_comp1);

    // ---- drop EVERYTHING the compress side produced -----------------------
    // Actually gpu::Free'd, not merely abandoned, so the reconstruct below
    // cannot read a warm buffer.  d_A (input / SNR reference) survives.  The
    // bytes the inverse starts from live in pinned HOST memory (pk_host /
    // payload / auxc), so it gets no shortcut.
    tt::tt_free(cores);
    keep_comp.clear();
    d_Ahat = nullptr; d_R   = nullptr; d_q   = nullptr; d_step = nullptr;
    d_ovi  = nullptr; d_ovv = nullptr; d_ovc = nullptr; d_cf16 = nullptr;
    d_w    = nullptr; d_off = nullptr; d_packed = nullptr;

    // ---- decode-side buffers: allocated untimed, then poisoned ------------
    // Allocation stays out of the clock on both sides.  Zeroing d_Ahat2 means a
    // reconstruct that failed to write shows up as a garbage SNR rather than
    // silently inheriting the forward pass's result.
    int16_t*  d_q2    = static_cast<int16_t*>(dmalloc(keep_dec, L * sizeof(int16_t)));
    uint16_t* d_c16_2 = static_cast<uint16_t*>(dmalloc(keep_dec, core_elems * sizeof(uint16_t)));
    float*    d_Ahat2 = static_cast<float*>(dmalloc(keep_dec, P * sizeof(float)));
    float*    d_dec2  = static_cast<float*>(dmalloc(keep_dec, L * sizeof(float)));
    float*    d_step2 = static_cast<float*>(dmalloc(keep_dec, sizeof(float)));
    uint32_t* d_ovi2  = static_cast<uint32_t*>(dmalloc(keep_dec, std::max<size_t>(n_over, 1) * sizeof(uint32_t)));
    float*    d_ovv2  = static_cast<float*>(dmalloc(keep_dec, std::max<size_t>(n_over, 1) * sizeof(float)));
    uint8_t*  d_w2      = static_cast<uint8_t*>(dmalloc(keep_dec, entropy == 1 ? nblk : 0));
    uint64_t* d_off2    = static_cast<uint64_t*>(
        dmalloc(keep_dec, entropy == 1 ? (size_t)nblk * sizeof(uint64_t) : 0));
    uint8_t*  d_packed2 = static_cast<uint8_t*>(
        dmalloc(keep_dec, entropy == 1 ? std::max<size_t>(packed_bytes, 1) : 0));
    tt::TtCores cores2;
    cores2.g1 = static_cast<float*>(dmalloc(keep_dec, e1 * sizeof(float)));
    cores2.g2 = static_cast<float*>(dmalloc(keep_dec, e2 * sizeof(float)));
    cores2.g3 = static_cast<float*>(dmalloc(keep_dec, e3 * sizeof(float)));
    CK_GPU_CHECK(gpu::MemsetAsync(d_Ahat2, 0, P * sizeof(float), nullptr));
    CK_GPU_CHECK(gpu::MemsetAsync(d_dec2,  0, L * sizeof(float), nullptr));
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    // ---- inverse: from the serialized bytes, as they come off storage -----
    const auto t_dec0 = ck_clk::now();
    static PinnedBuf tt_zz_dec;
    uint8_t* h_zz_dec = nullptr;
    std::vector<uint64_t> roff;
    std::vector<uint8_t> rw_tt;
    uint32_t wcs_tt = 0;
    if (entropy == 1) {
        std::memcpy(&wcs_tt, pk_host, 4);
        rw_tt = zstd_decompress(pk_host + 4, wcs_tt, nblk);
        roff.resize(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { roff[b] = run; run += 32ull * rw_tt[b]; }
    } else {
        h_zz_dec = static_cast<uint8_t*>(tt_zz_dec.get(zz_bytes));
        (void)zstd_decompress_into(h_zz_dec, zz_bytes, payload.data(), payload.size());
    }
    auto h_aux_dec = zstd_decompress(auxc.data(), auxc.size(), aux_raw.size());
    const double unzstd_ms = ck_ms(t_dec0);

    TtHeader rh{};
    std::memcpy(&rh, h_aux_dec.data(), sizeof(TtHeader));
    if (rh.magic != TTEQ_MAGIC)
        throw std::runtime_error("tt_eq_selftest: bad TtHeader magic on decode");
    cores2.n1 = rh.n1; cores2.n2 = rh.n2; cores2.n3 = rh.n3;
    cores2.r1 = rh.r1; cores2.r2 = rh.r2;
    const uint32_t rn_over = rh.n_over;
    const float    rDf     = static_cast<float>(rh.d_step);

    const auto t_dech2d0 = ck_clk::now();
    if (entropy == 1) {
        CK_GPU_CHECK(gpu::Memcpy(d_w2, rw_tt.data(), nblk, gpu::Copy::H2D));
        CK_GPU_CHECK(gpu::Memcpy(d_off2, roff.data(),
                                 (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D));
        if (packed_bytes)
            CK_GPU_CHECK(gpu::Memcpy(d_packed2, pk_host + 4 + wcs_tt, packed_bytes, gpu::Copy::H2D));
    } else {
        CK_GPU_CHECK(gpu::Memcpy(d_q2, h_zz_dec, zz_bytes, gpu::Copy::H2D));
    }
    CK_GPU_CHECK(gpu::Memcpy(d_c16_2, h_aux_dec.data() + sizeof(TtHeader),
                             core_elems * sizeof(uint16_t), gpu::Copy::H2D));
    if (rn_over) {
        const uint8_t* ovp = h_aux_dec.data() + sizeof(TtHeader) + core_elems * sizeof(uint16_t);
        std::vector<uint32_t> ri(rn_over);
        std::vector<float>    rv(rn_over);
        for (uint32_t k = 0; k < rn_over; ++k) {
            std::memcpy(&ri[k], ovp + (size_t)k * 8u,     4);
            std::memcpy(&rv[k], ovp + (size_t)k * 8u + 4, 4);
        }
        CK_GPU_CHECK(gpu::Memcpy(d_ovi2, ri.data(), rn_over * sizeof(uint32_t), gpu::Copy::H2D));
        CK_GPU_CHECK(gpu::Memcpy(d_ovv2, rv.data(), rn_over * sizeof(float),    gpu::Copy::H2D));
    }
    CK_GPU_CHECK(gpu::Memcpy(d_step2, &rDf, sizeof(float), gpu::Copy::H2D));
    const double dec_h2d_ms = ck_ms(t_dech2d0);

    // The unpack kernel used to sit INSIDE the H2D bracket above, so a caller
    // stripping dec_h2d_ms to get a device-only number silently dropped the
    // entropy decode with it.  It is now one fused unpack -> unzigzag ->
    // dequant kernel straight into d_dec2, instead of three passes through a
    // materialized L-element code array.
    double unpack_gpu_ms = 0.0;
    if (entropy == 1) {
        const auto s_u = ck_clk::now();
        codec::bitpack_dequant(d_packed2, d_w2, d_off2, L, grp, d_step2, d_dec2,
                               nblk, 0, nullptr);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        unpack_gpu_ms = ck_ms(s_u);
    } else {
        codec::unzigzag_inplace(reinterpret_cast<uint16_t*>(d_q2), L, nullptr);
        codec::dequant_pergroup(d_q2, L, grp, d_step2, d_dec2, nullptr);
    }

    // fp16 cores -> fp32, then rebuild the low-rank tensor from THEM.
    codec::expand_to_fp32(d_c16_2,           DataType::Float16, e1, cores2.g1, nullptr);
    codec::expand_to_fp32(d_c16_2 + e1,      DataType::Float16, e2, cores2.g2, nullptr);
    codec::expand_to_fp32(d_c16_2 + e1 + e2, DataType::Float16, e3, cores2.g3, nullptr);
    tt::blas_set_stream(blas, nullptr);
    if (!tt::tt_reconstruct(cores2, d_Ahat2, blas, nullptr))
        throw std::runtime_error("tt_eq_selftest: tt_reconstruct (decode) failed");

    if (rn_over) codec::scatter_overflow(d_dec2, d_ovi2, d_ovv2, rn_over, nullptr);
    codec::add_fp32(d_Ahat2, d_dec2, d_Ahat2, L, nullptr);      // full reconstruction
    (void)gpu::DeviceSynchronize();
    const double decompress_ms = ck_ms(t_dec0);

    // ---- SNR measured against the tensor that came out of THAT path -------
    double sq = 0.0, sig = 0.0, maxae = 0.0;
    codec::reduce_sq_error_fp32(d_A, d_Ahat2, L, &sq, &sig, nullptr);
    codec::reduce_max_abs_err(d_A, d_Ahat2, L, &maxae, nullptr);

    const size_t comp = (entropy == 1 ? pk_bytes : payload.size()) + auxc.size();
    const size_t orig = L * sizeof(float);

    py::dict d;
    d["n"]            = L;
    d["dims"]         = py::make_tuple(dims.n1, dims.n2, dims.n3);
    d["r1"]           = r1;
    d["r2"]           = r2;
    d["core_elems"]   = core_elems;
    d["core_bytes"]   = auxc.size();
    d["payload_bytes"]= (entropy == 1 ? pk_bytes : payload.size());
    d["entropy"]      = entropy;
    d["dev_alloc_bytes"] = (uint64_t)g_dev_alloc_bytes;
    d["orig_bytes"]   = orig;
    d["comp_bytes"]   = comp;
    d["compress_ms"]  = compress_ms;     // decompose + quantize + serialize
    // entropy decode + H2D + fp16 core expand + TT reconstruct + residual add,
    // starting from the serialized bytes with every compress buffer freed
    d["decompress_ms"] = decompress_ms;
    d["decompose_ms"]  = decompose_ms;   // capped TT-SVD (rSVD + GEMMs)
    d["tt_sketch1_ms"] = prof.sketch1_ms;
    d["tt_gemm_c_ms"]  = prof.gemm_c_ms;
    d["tt_scale_ms"]   = prof.scale_ms;
    d["tt_sketch2_ms"] = prof.sketch2_ms;
    d["tt_gemm_g3_ms"] = prof.gemm_g3_ms;
    d["tt_alloc_ms"]   = prof.alloc_ms;
    d["reconstruct_ms"]= reconstruct_ms;
    d["resid_ms"]      = resid_ms;
    d["quant_ms"]      = quant_ms;
    d["h2d_ms"]        = h2d_ms;         // subtract for a device-resident start
    d["zstd_ms"]       = zstd_ms;        // whole entropy tail, isolated
    d["pack_gpu_ms"]   = pack_gpu_ms;    // bit-pack kernels (entropy==1)
    d["pack_io_ms"]    = pack_io_ms;     // ALL non-GPU time in the tail
    d["payload_d2h_ms"]= payload_d2h_ms; // entropy==0 only
    d["unpack_gpu_ms"] = unpack_gpu_ms;  // bit-unpack kernel (entropy==1)
    d["unzstd_ms"]     = unzstd_ms;      // entropy decode (payload + aux)
    d["dec_h2d_ms"]    = dec_h2d_ms;     // H2D of the decoded payload + cores
    d["packed_bytes"]  = (uint64_t)packed_bytes;
    d["ratio"]        = static_cast<double>(orig) / static_cast<double>(comp);
    d["snr_db"]       = (sq > 0.0) ? 10.0 * std::log10(sig / sq) : 1e9;
    d["rel_l2"]       = (sig > 0.0) ? std::sqrt(sq / sig) : 0.0;
    d["max_abs_err"]  = maxae;
    d["resid_range"]  = range;
    d["eq_step_D"]    = D;
    d["n_over"]       = n_over;
    d["eps"]          = eps;
    d["eq_mode"]      = eq_mode;
    return d;
}

#if 0  // superseded: the restart experiment quantizes in numpy (bit-identical)
// ------------------------------------------------------------------ //
//  codec_roundtrip: run the real GPU codec forward+inverse and return the
//  reconstructed fp32 array.
// ------------------------------------------------------------------ //
py::array_t<float> codec_roundtrip(
        py::array_t<float, py::array::c_style | py::array::forcecast> arr,
        double eps, uint32_t group, bool global_step) {
    const size_t n = static_cast<size_t>(arr.size());
    if (n == 0) throw std::runtime_error("codec_roundtrip: empty array");
    if (group == 0) group = 256;
    constexpr float EXP_AVG_SQ_FLOOR = 1e-16f;

    (void)gpu::Free(nullptr);
    int dev = 0;
    if (gpu::GetDevice(&dev) != gpu::Success) { (void)gpu::SetDevice(0); dev = 0; }
    CK_GPU_CHECK(gpu::SetDevice(dev));

    const uint32_t ng  = static_cast<uint32_t>((n + group - 1) / group);
    const size_t   cap = std::max<size_t>(1024, n / 32);

    std::vector<DevFree> keep;
    keep.reserve(12);
    float*    d_src  = static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    int16_t*  d_q    = static_cast<int16_t*>(dmalloc(keep, n * sizeof(int16_t)));
    float*    d_rms  = static_cast<float*>(dmalloc(keep, ng * sizeof(float)));
    float*    d_step = static_cast<float*>(dmalloc(keep, ng * sizeof(float)));
    uint32_t* d_ovi  = static_cast<uint32_t*>(dmalloc(keep, cap * sizeof(uint32_t)));
    float*    d_ovv  = static_cast<float*>(dmalloc(keep, cap * sizeof(float)));
    uint32_t* d_ovc  = static_cast<uint32_t*>(dmalloc(keep, sizeof(uint32_t)));
    float*    d_dec  = static_cast<float*>(dmalloc(keep, n * sizeof(float)));

    CK_GPU_CHECK(gpu::Memcpy(d_src, arr.data(), n * sizeof(float), gpu::Copy::H2D));

    codec::group_rms_fp32(d_src, n, group, d_rms, ng, nullptr);
    std::vector<float> h_rms(ng);
    CK_GPU_CHECK(gpu::Memcpy(h_rms.data(), d_rms, ng * sizeof(float), gpu::Copy::D2H));

    PgSteps ps;
    if (global_step) {
        double ms = 0.0;
        for (uint32_t g = 0; g < ng; ++g) ms += double(h_rms[g]) * double(h_rms[g]);
        const double D = 2.0 * eps * std::sqrt(ms / double(ng)) * (1.0 - 1e-6);
        ps.d_use.assign(ng, static_cast<float>(D));
    } else {
        ps = quantize_steps(h_rms, eps);
    }
    CK_GPU_CHECK(gpu::Memcpy(d_step, ps.d_use.data(), ng * sizeof(float), gpu::Copy::H2D));
    CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));

    codec::quantize_pergroup(d_src, n, group, d_step, d_q,
                             d_ovi, d_ovv, static_cast<uint32_t>(cap), d_ovc, nullptr);
    uint32_t n_over = 0;
    CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
    n_over = std::min<uint32_t>(n_over, static_cast<uint32_t>(cap));

    codec::dequant_pergroup(d_q, n, group, d_step, d_dec, nullptr);
    if (n_over) codec::scatter_overflow(d_dec, d_ovi, d_ovv, n_over, nullptr);
    if (global_step) codec::clamp_floor_fp32(d_dec, n, EXP_AVG_SQ_FLOOR, nullptr);
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    py::array_t<float> out(static_cast<py::ssize_t>(n));
    CK_GPU_CHECK(gpu::Memcpy(out.mutable_data(), d_dec, n * sizeof(float), gpu::Copy::D2H));
    return out;
}
#endif  // codec_roundtrip

// ══════════════════════════════════════════════════════════════════════════
//  expavg_probe -- measurement-only variants of the exp_avg codec.
//
//  Nothing here is on the production path.  CheckpointCompressor::codec_tt_eq
//  is untouched; this file is a separate translation unit that the compressor
//  never calls, so every option below is additive by construction.
//
//  The shipped codec is reproduced exactly by the defaults
//  (variant="tt_resid", step_mode="global_range", rank_cap=16), which is what
//  makes the other variants comparable to it.
//
//  variant:
//    "nott"         quantize exp_avg directly, no TT stage at all.
//    "tt_resid"     SHIPPED SHAPE: TT cores (fp16) + an ELEMENT-WISE residual
//                   R = A - reconstruct(cores), quantized.
//    "tt_rankquant" Case A: no element-wise residual.  Decompose at the larger
//                   rank_cap, keep the leading rank_f16 bonds in fp16, and
//                   quantize the REMAINING RANK COMPONENTS.  Storage is then
//                   O(core elements), not O(L) -- which is the whole point of
//                   asking the question.
//
//  step_mode (how the uniform quantizer picks its step):
//    "global_range" one step for the whole tensor, D = 2*eps*(max-min).  What
//                   the shipped TT residual uses.
//    "bucket_rms"   per bucket, D_g = 2*eps*RMS_g (the weight codec's rule).
//    "bucket_range" per bucket, D_g = 2*eps*(max_g-min_g).
// ══════════════════════════════════════════════════════════════════════════

namespace {

// Per-bucket min/max.  group_rms_fp32 already covers the RMS rule; there is no
// per-bucket min/max kernel in CodecKernels because nothing on the production
// path needs one, so it lives here with the rest of the probe-only code.
__global__ void k_group_minmax(const float* __restrict__ x, size_t n,
                               uint32_t group, float* __restrict__ gmin,
                               float* __restrict__ gmax, uint32_t ng) {
    const uint32_t g = blockIdx.x;
    if (g >= ng) return;
    const size_t lo = (size_t)g * group;
    const size_t hi = min(lo + group, n);
    float mn =  INFINITY, mx = -INFINITY;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = x[i];
        mn = fminf(mn, v); mx = fmaxf(mx, v);
    }
    __shared__ float smn[256], smx[256];
    smn[threadIdx.x] = mn; smx[threadIdx.x] = mx;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
            smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { gmin[g] = smn[0]; gmax[g] = smx[0]; }
}

// All eight probe accumulators in one pass: squared error, both energies, the
// dot product, the zero counts and the sign agreement.  fp64 throughout --
// these are reductions over ~1e9 elements and fp32 would lose the answer.
__global__ void k_probe_metrics(const float* __restrict__ a,
                                const float* __restrict__ b,
                                size_t n, double* __restrict__ out) {
    __shared__ double s[8][128];
    double acc[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x) {
        const double av = a[i], bv = b[i];
        const double d  = av - bv;
        acc[0] += d * d;
        acc[1] += av * av;
        acc[2] += av * bv;
        acc[3] += bv * bv;
        const bool anz = (av != 0.0), bnz = (bv != 0.0);
        if (anz) acc[4] += 1.0;
        if (bnz) acc[5] += 1.0;
        if (anz && bnz) {
            acc[6] += 1.0;
            if ((av > 0.0) == (bv > 0.0)) acc[7] += 1.0;
        }
    }
    for (int k = 0; k < 8; ++k) s[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st)
            for (int k = 0; k < 8; ++k) s[k][threadIdx.x] += s[k][threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < 8; ++k) atomicAdd(&out[k], s[k][0]);
}

// Case A core split.  `keep` gets the leading F x F rank block, `extra` gets
// the complement; the two have disjoint support and sum back to the original
// cores, so the reconstruction is simply tt_reconstruct(keep_f16 + extra_hat).
__global__ void k_split_g1(const float* src, float* keep, float* extra,
                           uint32_t n1, uint32_t r1, uint32_t F) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n1 * r1) return;
    const bool k = (uint32_t)(i % r1) < F;
    keep[i]  = k ? src[i] : 0.0f;
    extra[i] = k ? 0.0f   : src[i];
}
__global__ void k_split_g2(const float* src, float* keep, float* extra,
                           uint32_t r1, uint32_t n2, uint32_t r2, uint32_t F) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)r1 * n2 * r2) return;
    const uint32_t b = (uint32_t)(i % r2);
    const uint32_t a = (uint32_t)(i / ((size_t)n2 * r2));
    const bool k = (a < F) && (b < F);
    keep[i]  = k ? src[i] : 0.0f;
    extra[i] = k ? 0.0f   : src[i];
}
__global__ void k_split_g3(const float* src, float* keep, float* extra,
                           uint32_t r2, uint32_t n3, uint32_t F) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)r2 * n3) return;
    const bool k = (uint32_t)(i / n3) < F;
    keep[i]  = k ? src[i] : 0.0f;
    extra[i] = k ? 0.0f   : src[i];
}

}  // namespace

// Quantize one host array with Checkers' uniform quantizer and hand back BOTH
// the dequantized values and the real serialized size.  codec_selftest already
// reports a ratio but keeps the reconstruction to itself, and Case A needs the
// values: Python does the SVD and the fp16 split, this does the quantization,
// and Python then reconstructs and scores.  Measurement only, like the rest of
// this file.
py::tuple quant_roundtrip(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                          double eps, const std::string& step_mode,
                          uint32_t bucket, int zstd_level) {
    const size_t n = static_cast<size_t>(arr.size());
    if (n == 0)
        return py::make_tuple(py::bytes("", 0), (uint64_t)0, (uint32_t)0, 999.0);

    const bool bucketed = (step_mode == "bucket_rms" || step_mode == "bucket_range");
    if (!bucketed && step_mode != "global_range")
        throw std::runtime_error("quant_roundtrip: bad step_mode");
    const uint32_t grp = bucketed ? std::max<uint32_t>(bucket, 1u)
                                  : static_cast<uint32_t>(n);
    const uint32_t ng  = bucketed
        ? static_cast<uint32_t>((n + (size_t)grp - 1) / grp) : 1u;

    std::vector<DevFree> keep;
    float* d_x = static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    CK_GPU_CHECK(gpu::Memcpy(d_x, arr.data(), n * sizeof(float), gpu::Copy::H2D));

    std::vector<float> h_step(ng);
    if (!bucketed) {
        double rmin = 0.0, rmax = 0.0;
        codec::reduce_minmax_fp32(d_x, n, &rmin, &rmax, nullptr);
        const double aeps = eps * std::max(rmax - rmin, 0.0);
        if (!(aeps > 0.0)) {                       // constant array -> exact
            std::vector<float> h_raw(n);
            CK_GPU_CHECK(gpu::Memcpy(h_raw.data(), d_x, n * sizeof(float), gpu::Copy::D2H));
            return py::make_tuple(
                py::bytes(reinterpret_cast<const char*>(h_raw.data()), n * sizeof(float)),
                (uint64_t)(n * sizeof(float)), (uint32_t)0, 999.0);
        }
        h_step[0] = static_cast<float>(2.0 * aeps * (1.0 - 1e-6));
    } else if (step_mode == "bucket_rms") {
        float* d_r = static_cast<float*>(dmalloc(keep, (size_t)ng * sizeof(float)));
        codec::group_rms_fp32(d_x, n, grp, d_r, ng, nullptr);
        std::vector<float> h_r(ng);
        CK_GPU_CHECK(gpu::Memcpy(h_r.data(), d_r, (size_t)ng * sizeof(float), gpu::Copy::D2H));
        h_step = quantize_steps(h_r, eps).d_use;
    } else {
        float* d_mn = static_cast<float*>(dmalloc(keep, (size_t)ng * sizeof(float)));
        float* d_mx = static_cast<float*>(dmalloc(keep, (size_t)ng * sizeof(float)));
        k_group_minmax<<<ng, 256>>>(d_x, n, grp, d_mn, d_mx, ng);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        std::vector<float> h_mn(ng), h_mx(ng), h_rg(ng);
        CK_GPU_CHECK(gpu::Memcpy(h_mn.data(), d_mn, (size_t)ng*sizeof(float), gpu::Copy::D2H));
        CK_GPU_CHECK(gpu::Memcpy(h_mx.data(), d_mx, (size_t)ng*sizeof(float), gpu::Copy::D2H));
        for (uint32_t g = 0; g < ng; ++g) h_rg[g] = std::max(h_mx[g] - h_mn[g], 0.0f);
        h_step = quantize_steps(h_rg, eps).d_use;
    }
    float* d_step = static_cast<float*>(dmalloc(keep, (size_t)ng * sizeof(float)));
    CK_GPU_CHECK(gpu::Memcpy(d_step, h_step.data(), (size_t)ng * sizeof(float), gpu::Copy::H2D));

    const size_t cap = std::max<size_t>(n / 64, 1024);
    uint16_t* d_q   = static_cast<uint16_t*>(dmalloc(keep, n * sizeof(uint16_t)));
    uint32_t* d_ovi = static_cast<uint32_t*>(dmalloc(keep, cap * sizeof(uint32_t)));
    float*    d_ovv = static_cast<float*>(dmalloc(keep, cap * sizeof(float)));
    uint32_t* d_ovc = static_cast<uint32_t*>(dmalloc(keep, sizeof(uint32_t)));
    const uint32_t nblk = static_cast<uint32_t>((n + codec::BITPACK_BLOCK - 1)
                                                / codec::BITPACK_BLOCK);
    uint8_t* d_w = static_cast<uint8_t*>(dmalloc(keep, nblk));
    CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));
    codec::quantize_fused(d_x, n, grp, d_step, d_q, d_ovi, d_ovv,
                          static_cast<uint32_t>(cap), d_ovc, 0, 0.0f, nullptr, d_w);
    uint32_t n_over = 0;
    CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
    n_over = std::min<uint32_t>(n_over, static_cast<uint32_t>(cap));

    std::vector<uint8_t> h_w(nblk);
    CK_GPU_CHECK(gpu::Memcpy(h_w.data(), d_w, nblk, gpu::Copy::D2H));
    uint64_t run = 0;
    for (uint32_t b = 0; b < nblk; ++b) run += 32ull * h_w[b];
    std::vector<uint8_t> wc = zstd_compress(h_w.data(), nblk, zstd_level);
    std::vector<uint8_t> sc = zstd_compress(reinterpret_cast<const uint8_t*>(h_step.data()),
                                            h_step.size() * sizeof(float), zstd_level);
    const uint64_t bytes = (uint64_t)run + wc.size() + sc.size() + (uint64_t)n_over * 8u;

    float* d_deq = static_cast<float*>(dmalloc(keep, n * sizeof(float)));
    codec::unzigzag_inplace(d_q, n, nullptr);
    codec::dequant_pergroup(reinterpret_cast<const int16_t*>(d_q), n, grp,
                            d_step, d_deq, nullptr);
    if (n_over) codec::scatter_overflow(d_deq, d_ovi, d_ovv, n_over, nullptr);
    CK_GPU_CHECK(gpu::DeviceSynchronize());
    double dbg_sq = 0.0, dbg_sig = 0.0;
    codec::reduce_sq_error_fp32(d_x, d_deq, n, &dbg_sq, &dbg_sig, nullptr);
    CK_GPU_CHECK(gpu::DeviceSynchronize());
    // D2H into a host vector first, then into the numpy buffer.  Copying
    // straight into out.mutable_data() returned a buffer filled with a single
    // repeated value even though the device-side SNR on d_deq was correct.
    std::vector<float> h_out(n);
    CK_GPU_CHECK(gpu::Memcpy(h_out.data(), d_deq, n * sizeof(float), gpu::Copy::D2H));
    const double dbg_snr = (dbg_sq > 0.0 && dbg_sig > 0.0)
                         ? 10.0 * std::log10(dbg_sig / dbg_sq) : -999.0;
    // Returned as raw bytes for np.frombuffer.  Handing the data back as a
    // py::array_t -- whether by constructing from the pointer or by memcpy into
    // mutable_data() -- produced an array whose buffer did not match the host
    // vector it was built from, while the vector itself was verifiably correct.
    return py::make_tuple(
        py::bytes(reinterpret_cast<const char*>(h_out.data()), n * sizeof(float)),
        bytes, n_over, dbg_snr);
}

py::dict expavg_probe(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                      const std::string& variant,
                      double eps,
                      const std::string& step_mode,
                      uint32_t bucket,
                      uint32_t rank_cap,
                      uint32_t rank_f16,
                      uint32_t ndim,
                      double energy_keep,
                      int zstd_level,
                      uint32_t n1, uint32_t n2, uint32_t n3) {
    const size_t L = static_cast<size_t>(arr.size());
    if (L == 0) throw std::runtime_error("expavg_probe: empty array");
    const bool want_tt     = (variant == "tt_resid" || variant == "tt_rankquant");
    const bool rank_quant  = (variant == "tt_rankquant");
    if (!want_tt && variant != "nott")
        throw std::runtime_error("expavg_probe: variant must be nott|tt_resid|tt_rankquant");
    const bool bucketed = (step_mode == "bucket_rms" || step_mode == "bucket_range");
    if (!bucketed && step_mode != "global_range")
        throw std::runtime_error("expavg_probe: step_mode must be "
                                 "global_range|bucket_rms|bucket_range");
    const uint32_t grp = bucketed ? std::max<uint32_t>(bucket, 1u)
                                  : static_cast<uint32_t>(L);
    const uint32_t ng  = static_cast<uint32_t>((L + (size_t)grp - 1) / grp);

    std::vector<DevFree> keep;
    auto B = [](size_t n, uint32_t t) { return (uint32_t)((n + t - 1) / t); };

    // ---- source, and the buffer the quantizer actually sees -------------
    // Shape the TT sees.  Default: choose_dims, i.e. the artificial near-cube
    // the shipped codec builds from the fragment's element count alone.  With
    // n1/n2 given, use that shape instead -- e.g. a weight fragment's TRUE
    // (rows/world) x cols, to ask whether the cube reshape is what destroys the
    // low-rank structure.
    tt::TtDims dims;
    if (n1 && n2) {
        dims.n1 = n1; dims.n2 = n2; dims.n3 = n3 ? n3 : 1u;
        if ((size_t)dims.n1 * dims.n2 * dims.n3 < L)
            throw std::runtime_error("expavg_probe: explicit dims smaller than the array");
    } else {
        dims = tt::choose_dims(L, std::min(std::max(ndim, 2u), 3u));
    }
    const size_t P  = want_tt ? (size_t)dims.padded() : L;
    float* d_A = static_cast<float*>(dmalloc(keep, P * sizeof(float)));
    CK_GPU_CHECK(gpu::MemsetAsync(d_A, 0, P * sizeof(float), nullptr));
    CK_GPU_CHECK(gpu::Memcpy(d_A, arr.data(), L * sizeof(float), gpu::Copy::H2D));

    float*   d_Ahat = nullptr;          // TT low-rank reconstruction
    float*   d_tgt  = nullptr;          // what gets quantized
    size_t   n_tgt  = 0;
    size_t   core_keep_bytes = 0;
    tt::TtCores cores{};
    void* blas = want_tt ? tt::blas_handle() : nullptr;
    float *d_k1 = nullptr, *d_k2 = nullptr, *d_k3 = nullptr;   // Case A: fp16-rounded keep
    size_t e1 = 0, e2 = 0, e3 = 0, core_elems = 0;

    if (want_tt) {
        if (!tt::tt_decompose(d_A, dims, rank_cap, energy_keep, cores, blas, nullptr))
            throw std::runtime_error("expavg_probe: tt_decompose failed");
        e1 = (size_t)cores.n1 * cores.r1;
        e2 = (size_t)cores.r1 * cores.n2 * cores.r2;
        e3 = (size_t)cores.r2 * cores.n3;
        core_elems = e1 + e2 + e3;
    }

    if (variant == "nott") {
        d_tgt = d_A; n_tgt = L;
    } else if (variant == "tt_resid") {
        // cores stored fp16; residual is element-wise over all L values.
        core_keep_bytes = core_elems * sizeof(uint16_t);
        uint16_t* c16 = static_cast<uint16_t*>(dmalloc(keep, core_elems * sizeof(uint16_t)));
        codec::contract_from_fp32(cores.g1, DataType::Float16, e1, c16, nullptr);
        codec::contract_from_fp32(cores.g2, DataType::Float16, e2, c16 + e1, nullptr);
        codec::contract_from_fp32(cores.g3, DataType::Float16, e3, c16 + e1 + e2, nullptr);
        codec::expand_to_fp32(c16,            DataType::Float16, e1, cores.g1, nullptr);
        codec::expand_to_fp32(c16 + e1,       DataType::Float16, e2, cores.g2, nullptr);
        codec::expand_to_fp32(c16 + e1 + e2,  DataType::Float16, e3, cores.g3, nullptr);
        d_Ahat = static_cast<float*>(dmalloc(keep, P * sizeof(float)));
        if (!tt::tt_reconstruct(cores, d_Ahat, blas, nullptr))
            throw std::runtime_error("expavg_probe: tt_reconstruct failed");
        d_tgt = static_cast<float*>(dmalloc(keep, L * sizeof(float)));
        codec::sub_fp32(d_A, d_Ahat, d_tgt, L, nullptr);
        n_tgt = L;
    } else {
        // Case A: split the cores by rank, fp16 for the leading block and the
        // uniform quantizer for the rest.  No element-wise residual exists.
        const uint32_t F = std::min(rank_f16, std::min(cores.r1, cores.r2));
        float* kc = static_cast<float*>(dmalloc(keep, core_elems * sizeof(float)));
        float* ex = static_cast<float*>(dmalloc(keep, core_elems * sizeof(float)));
        k_split_g1<<<B(e1,256),256>>>(cores.g1, kc,           ex,           cores.n1, cores.r1, F);
        k_split_g2<<<B(e2,256),256>>>(cores.g2, kc+e1,        ex+e1,        cores.r1, cores.n2, cores.r2, F);
        k_split_g3<<<B(e3,256),256>>>(cores.g3, kc+e1+e2,     ex+e1+e2,     cores.r2, cores.n3, F);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        // fp16 round-trip of the kept block, in place
        uint16_t* c16 = static_cast<uint16_t*>(dmalloc(keep, core_elems * sizeof(uint16_t)));
        codec::contract_from_fp32(kc, DataType::Float16, core_elems, c16, nullptr);
        codec::expand_to_fp32(c16, DataType::Float16, core_elems, kc, nullptr);
        // only the kept rank block is actually stored as fp16
        const size_t kept_elems = (size_t)cores.n1 * F + (size_t)F * cores.n2 * F
                                + (size_t)F * cores.n3;
        core_keep_bytes = kept_elems * sizeof(uint16_t);
        d_k1 = kc; d_k2 = kc + e1; d_k3 = kc + e1 + e2;
        d_tgt = ex; n_tgt = core_elems;
    }

    // ---- the uniform quantizer, with whichever step rule ----------------
    const uint32_t q_ng  = bucketed
        ? static_cast<uint32_t>((n_tgt + (size_t)grp - 1) / grp) : 1u;
    const uint32_t q_grp = bucketed ? grp : static_cast<uint32_t>(n_tgt);
    float* d_step = static_cast<float*>(dmalloc(keep, (size_t)q_ng * sizeof(float)));
    std::vector<float> h_step(q_ng);
    double D_global = 0.0, rmin = 0.0, rmax = 0.0;

    if (!bucketed) {
        codec::reduce_minmax_fp32(d_tgt, n_tgt, &rmin, &rmax, nullptr);
        const double range  = rmax - rmin;
        const double aeps   = eps * std::max(range, 0.0);
        const double amax   = std::max(std::fabs(rmin), std::fabs(rmax));
        const double dyn    = amax / std::max(aeps, (double)FLT_MIN);
        const double margin = 1.0 - std::min(1e-3, std::max(64.0 * DBL_EPSILON * dyn, 1e-9));
        D_global = 2.0 * aeps * margin;
        if (!(D_global > 0.0)) throw std::runtime_error("expavg_probe: degenerate range");
        h_step[0] = static_cast<float>(D_global);
    } else if (step_mode == "bucket_rms") {
        float* d_rms = static_cast<float*>(dmalloc(keep, (size_t)q_ng * sizeof(float)));
        codec::group_rms_fp32(d_tgt, n_tgt, q_grp, d_rms, q_ng, nullptr);
        std::vector<float> h_rms(q_ng);
        CK_GPU_CHECK(gpu::Memcpy(h_rms.data(), d_rms, (size_t)q_ng * sizeof(float), gpu::Copy::D2H));
        PgSteps ps = quantize_steps(h_rms, eps);     // D_g = 2*eps*RMS_g
        h_step = ps.d_use;
    } else {
        float* d_mn = static_cast<float*>(dmalloc(keep, (size_t)q_ng * sizeof(float)));
        float* d_mx = static_cast<float*>(dmalloc(keep, (size_t)q_ng * sizeof(float)));
        k_group_minmax<<<q_ng, 256>>>(d_tgt, n_tgt, q_grp, d_mn, d_mx, q_ng);
        CK_GPU_CHECK(gpu::DeviceSynchronize());
        std::vector<float> h_mn(q_ng), h_mx(q_ng), h_rg(q_ng);
        CK_GPU_CHECK(gpu::Memcpy(h_mn.data(), d_mn, (size_t)q_ng*sizeof(float), gpu::Copy::D2H));
        CK_GPU_CHECK(gpu::Memcpy(h_mx.data(), d_mx, (size_t)q_ng*sizeof(float), gpu::Copy::D2H));
        // quantize_steps takes "r" and forms D = 2*eps*r, so feeding it the
        // bucket RANGE gives D_g = 2*eps*(max_g-min_g) with the same
        // decoder-exact step table the RMS rule produces.
        for (uint32_t g = 0; g < q_ng; ++g) h_rg[g] = std::max(h_mx[g] - h_mn[g], 0.0f);
        PgSteps ps = quantize_steps(h_rg, eps);
        h_step = ps.d_use;
    }
    CK_GPU_CHECK(gpu::Memcpy(d_step, h_step.data(), (size_t)q_ng * sizeof(float), gpu::Copy::H2D));

    const size_t cap = std::max<size_t>(n_tgt / 64, 1024);
    uint16_t* d_q   = static_cast<uint16_t*>(dmalloc(keep, n_tgt * sizeof(uint16_t)));
    uint32_t* d_ovi = static_cast<uint32_t*>(dmalloc(keep, cap * sizeof(uint32_t)));
    float*    d_ovv = static_cast<float*>(dmalloc(keep, cap * sizeof(float)));
    uint32_t* d_ovc = static_cast<uint32_t*>(dmalloc(keep, sizeof(uint32_t)));
    const uint32_t nblk = static_cast<uint32_t>((n_tgt + codec::BITPACK_BLOCK - 1)
                                                / codec::BITPACK_BLOCK);
    uint8_t*  d_w   = static_cast<uint8_t*>(dmalloc(keep, nblk));
    CK_GPU_CHECK(gpu::MemsetAsync(d_ovc, 0, sizeof(uint32_t), nullptr));
    codec::quantize_fused(d_tgt, n_tgt, q_grp, d_step, d_q, d_ovi, d_ovv,
                          static_cast<uint32_t>(cap), d_ovc, 0, 0.0f, nullptr, d_w);
    uint32_t n_over = 0;
    CK_GPU_CHECK(gpu::Memcpy(&n_over, d_ovc, sizeof(uint32_t), gpu::Copy::D2H));
    n_over = std::min<uint32_t>(n_over, static_cast<uint32_t>(cap));

    // ---- serialized size ------------------------------------------------
    std::vector<uint8_t> h_w(nblk);
    CK_GPU_CHECK(gpu::Memcpy(h_w.data(), d_w, nblk, gpu::Copy::D2H));
    std::vector<uint64_t> h_off(nblk);
    uint64_t run = 0;
    for (uint32_t b = 0; b < nblk; ++b) { h_off[b] = run; run += 32ull * h_w[b]; }
    const size_t packed_bytes = static_cast<size_t>(run);
    std::vector<uint8_t> wc = zstd_compress(h_w.data(), nblk, zstd_level);
    std::vector<uint8_t> sc = zstd_compress(reinterpret_cast<const uint8_t*>(h_step.data()),
                                            h_step.size() * sizeof(float), zstd_level);
    const size_t total_bytes = sizeof(TtHeader) + core_keep_bytes + packed_bytes
                             + wc.size() + sc.size() + (size_t)n_over * 8u;

    // ---- inverse --------------------------------------------------------
    float* d_hat = static_cast<float*>(dmalloc(keep, P * sizeof(float)));
    float* d_deq = static_cast<float*>(dmalloc(keep, n_tgt * sizeof(float)));
    codec::unzigzag_inplace(d_q, n_tgt, nullptr);
    codec::dequant_pergroup(reinterpret_cast<const int16_t*>(d_q), n_tgt, q_grp,
                            d_step, d_deq, nullptr);
    if (n_over) codec::scatter_overflow(d_deq, d_ovi, d_ovv, n_over, nullptr);

    if (variant == "nott") {
        CK_GPU_CHECK(gpu::Memcpy(d_hat, d_deq, L * sizeof(float), gpu::Copy::D2D));
    } else if (variant == "tt_resid") {
        codec::add_fp32(d_Ahat, d_deq, d_hat, L, nullptr);
    } else {
        // cores = fp16(kept block) + dequantized(remaining rank components)
        codec::add_fp32(d_k1, d_deq,            cores.g1, e1, nullptr);
        codec::add_fp32(d_k2, d_deq + e1,       cores.g2, e2, nullptr);
        codec::add_fp32(d_k3, d_deq + e1 + e2,  cores.g3, e3, nullptr);
        if (!tt::tt_reconstruct(cores, d_hat, blas, nullptr))
            throw std::runtime_error("expavg_probe: tt_reconstruct (decode) failed");
    }
    CK_GPU_CHECK(gpu::DeviceSynchronize());

    // ---- metrics --------------------------------------------------------
    double* d_acc = static_cast<double*>(dmalloc(keep, 8 * sizeof(double)));
    CK_GPU_CHECK(gpu::MemsetAsync(d_acc, 0, 8 * sizeof(double), nullptr));
    k_probe_metrics<<<std::min<uint32_t>(4096u, B(L,128)), 128>>>(d_A, d_hat, L, d_acc);
    CK_GPU_CHECK(gpu::DeviceSynchronize());
    double h[8];
    CK_GPU_CHECK(gpu::Memcpy(h, d_acc, 8 * sizeof(double), gpu::Copy::D2H));

    const uint32_t out_r1 = want_tt ? cores.r1 : 0u;   // read BEFORE tt_free zeroes them
    const uint32_t out_r2 = want_tt ? cores.r2 : 0u;
    if (want_tt) tt::tt_free(cores);

    const double sq = h[0], sig = h[1], dot = h[2], rec = h[3];
    const double anz = h[4], bnz = h[5], both = h[6], agree = h[7];
    py::dict d;
    d["variant"]        = variant;
    d["step_mode"]      = step_mode;
    d["bucket"]         = bucketed ? grp : 0u;
    d["eps"]            = eps;
    d["rank_cap"]       = rank_cap;
    d["rank_f16"]       = rank_quant ? rank_f16 : 0u;
    d["n_elems"]        = static_cast<uint64_t>(L);
    d["dims"]           = py::make_tuple(dims.n1, dims.n2, dims.n3);
    d["snr_db"]         = (sq > 0.0 && sig > 0.0) ? 10.0 * std::log10(sig / sq)
                                                  : std::numeric_limits<double>::infinity();
    d["cosine"]         = (rec > 0.0 && sig > 0.0) ? dot / std::sqrt(rec * sig)
                                                   : std::nan("");
    d["sign_agree"]     = both > 0 ? agree / both : std::nan("");
    d["zero_frac_recon"]= 1.0 - bnz / (double)L;
    d["zero_frac_ref"]  = 1.0 - anz / (double)L;
    d["flattened_frac"] = anz > 0 ? (anz - both) / anz : std::nan("");
    d["energy_ratio"]   = sig > 0.0 ? std::sqrt(rec / sig) : std::nan("");
    d["ratio"]          = (double)(L * sizeof(float)) / (double)total_bytes;
    d["bytes"]          = static_cast<uint64_t>(total_bytes);
    d["core_bytes"]     = static_cast<uint64_t>(core_keep_bytes);
    d["payload_bytes"]  = static_cast<uint64_t>(packed_bytes);
    d["n_over"]         = n_over;
    d["n_buckets"]      = q_ng;
    // raw accumulators, so per-fragment runs can be aggregated EXACTLY into a
    // shard-level figure instead of averaging derived quantities
    d["acc_sq_err"]     = sq;
    d["acc_signal"]     = sig;
    d["acc_dot"]        = dot;
    d["acc_rec_energy"] = rec;
    d["acc_n_ref_nz"]   = anz;
    d["acc_n_rec_nz"]   = bnz;
    d["acc_n_both"]     = both;
    d["acc_n_agree"]    = agree;
    d["r1"]             = out_r1;
    d["r2"]             = out_r2;
    return d;
}

} // namespace checkers
