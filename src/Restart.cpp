#include "checkers/CheckpointCompressor.hpp"

#include "checkers/CheckpointFormat.hpp"
#include "checkers/CodecKernels.hpp"
#include "checkers/GpuDefs.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/TensorResource.hpp"
#include "checkers/TtLinalg.hpp"
#include "checkers/ZstdUtil.hpp"
#include "checkers/logging.hpp"

#include <algorithm>
#include <type_traits>
#include <memory>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace checkers {

namespace {

// Streaming restore (2026-11-05): only the header and index are read up
// front; each tensor's payload + aux are read just before it is decoded, into
// one reused buffer.  (Before, the whole data section -- 14 GB per rank for a
// lossless stream -- was read into memory first.)
struct LoadedStream {
    CchHeader                  header{};
    std::vector<CchIndexEntry> index;
    std::ifstream              f;
    uint64_t                   data_beg = 0;   // file offset of the data section
    uint64_t                   data_len = 0;   // its length
};

// read [off, off + n) of the data section into buf
bool read_data(LoadedStream& ls, uint64_t off, uint64_t n, std::vector<uint8_t>& buf) {
    if (off + n > ls.data_len) return false;
    buf.resize(static_cast<size_t>(n));
    if (n == 0) return true;
    ls.f.clear();
    ls.f.seekg(static_cast<std::streamoff>(ls.data_beg + off), std::ios::beg);
    ls.f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(ls.f);
}

bool read_stream_file(const std::string& path, LoadedStream& out, std::string& err) {
    out.f.open(path, std::ios::binary);
    std::ifstream& f = out.f;
    if (!f) { err = "cannot open"; return false; }

    f.read(reinterpret_cast<char*>(&out.header), sizeof(CchHeader));
    if (!f || out.header.magic != CCH_MAGIC) { err = "bad magic"; return false; }
    if (out.header.version < CCH_VERSION_MIN_READ || out.header.version > CCH_VERSION) {
        err = "unsupported version " + std::to_string(out.header.version); return false;
    }

    // v2: index right after the header, data to EOF.  v3: data, then the
    // index at the end.  Both: the index is at index_offset.
    f.seekg(0, std::ios::end);
    const std::streamoff file_end  = f.tellg();
    const std::streamoff idx_beg   = static_cast<std::streamoff>(out.header.index_offset);
    const std::streamoff idx_bytes = static_cast<std::streamoff>(out.header.n_tensors) *
                                     static_cast<std::streamoff>(sizeof(CchIndexEntry));
    if (idx_beg + idx_bytes > file_end) { err = "truncated index"; return false; }
    out.index.resize(out.header.n_tensors);
    f.seekg(idx_beg, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.index.data()), idx_bytes);
    if (!f) { err = "truncated index"; return false; }

    const std::streamoff data_beg = static_cast<std::streamoff>(out.header.data_offset);
    const std::streamoff data_end = (idx_beg > data_beg) ? idx_beg : file_end;
    if (data_end < data_beg) { err = "truncated data"; return false; }
    out.data_beg = static_cast<uint64_t>(data_beg);
    out.data_len = static_cast<uint64_t>(data_end - data_beg);
    return true;
}

// --------------------------------------------------------------------------
// DecodeScratch (2026-11-05): every GPU buffer the decoders need, allocated
// ONCE per restore -- sized by a pre-pass over all index entries for the
// largest tensor -- and reused for every tensor, plus one non-blocking stream.
// Per tensor there is no hipMalloc / hipFree and no hipDeviceSynchronize, only
// async copies + kernels on that stream and one hipStreamSynchronize.
// (Before: 8-13 hipMalloc + hipFree per tensor; a restore hung inside one of
// those hipFree's with the GPU idle -- exp1, and the 11-05 benchmark.)
// --------------------------------------------------------------------------
struct DecodeNeed {
    size_t fp32 = 0, R = 0, zz = 0, step = 1, over = 0, nblk = 0, packed = 1;
    size_t core = 0, g1 = 0, g2 = 0, g3 = 0, t1 = 0;      // TT
};

struct DecodeScratch {
    gpu::Stream_t s = nullptr;
    float    *fp32 = nullptr, *R = nullptr, *step = nullptr, *ovv = nullptr;
    float    *g1 = nullptr, *g2 = nullptr, *g3 = nullptr, *t1 = nullptr;
    uint16_t *zz = nullptr;
    uint32_t *ovi = nullptr;
    uint8_t  *w = nullptr, *pk = nullptr, *core = nullptr;
    uint64_t *off = nullptr;
    DecodeNeed cap;
    size_t   bytes = 0;

    std::string alloc(const DecodeNeed& n) {
        cap = n;
        std::string err;
        auto one = [&](auto*& p, size_t count, size_t elem) {
            if (!err.empty()) return;
            void* q = nullptr;
            const size_t nb = std::max<size_t>(count * elem, 1);
            if (gpu::Malloc(&q, nb) != gpu::Success) {
                err = "restore scratch: device alloc of " + std::to_string(nb) + " B failed";
                return;
            }
            p = static_cast<std::remove_reference_t<decltype(p)>>(q);
            bytes += nb;
        };
        one(fp32, n.fp32, sizeof(float));   one(R, n.R, sizeof(float));
        one(step, n.step, sizeof(float));   one(ovv, n.over, sizeof(float));
        one(ovi, n.over, sizeof(uint32_t)); one(zz, n.zz, sizeof(uint16_t));
        one(w, n.nblk, 1);                  one(off, n.nblk, sizeof(uint64_t));
        one(pk, n.packed, 1);               one(core, n.core, 1);
        one(g1, n.g1, sizeof(float));       one(g2, n.g2, sizeof(float));
        one(g3, n.g3, sizeof(float));       one(t1, n.t1, sizeof(float));
        if (err.empty() && gpu::StreamCreate(&s) != gpu::Success) {
            s = nullptr; err = "restore scratch: stream create failed";
        }
        return err;
    }
    // once, at the end of the restore
    void release() {
        if (s) { (void)gpu::StreamSynchronize(s); (void)gpu::StreamDestroy(s); s = nullptr; }
        for (void* p : {(void*)fp32, (void*)R, (void*)step, (void*)ovv, (void*)ovi, (void*)zz,
                        (void*)w, (void*)off, (void*)pk, (void*)core, (void*)g1, (void*)g2,
                        (void*)g3, (void*)t1})
            if (p) (void)gpu::Free(p);
        fp32 = R = step = ovv = g1 = g2 = g3 = t1 = nullptr;
        zz = nullptr; ovi = nullptr; w = pk = core = nullptr; off = nullptr;
    }
    ~DecodeScratch() { release(); }
};

// --------------------------------------------------------------------------
// RawZstd (or raw passthrough): payload bytes -> live tensor.
// --------------------------------------------------------------------------
std::string refill_raw(const CchIndexEntry& e, const uint8_t* payload,
                       const struct ShardTensor& st, DecodeScratch& S) {
    std::vector<uint8_t> raw;
    if (e.flags & CCH_FLAG_ZSTD) raw = zstd_decompress(payload, e.data_size, e.orig_bytes);
    else                          raw.assign(payload, payload + e.data_size);

    if (raw.size() != st.byte_size)
        return "size mismatch (file " + std::to_string(raw.size()) + " vs live " +
               std::to_string(st.byte_size) + ")";

    gpu::Error_t rc = gpu::MemcpyAsync(st.device_ptr, raw.data(), st.byte_size, gpu::Copy::H2D, S.s);
    if (rc == gpu::Success) rc = gpu::StreamSynchronize(S.s);   // before `raw` goes away
    if (rc != gpu::Success) return std::string("H2D failed: ") + gpu::ErrorString(rc);
    return {};
}

// --------------------------------------------------------------------------
// RmsRelPgQuant: unzigzag -> per-group dequant -> restore overflow ->
//                contract to storage dtype -> live tensor.
// --------------------------------------------------------------------------
std::string refill_pergroup(const CchIndexEntry& e, uint32_t group,
                            const uint8_t* payload, const uint8_t* aux,
                            const struct ShardTensor& st, DecodeScratch& S) {
    const size_t   n  = e.num_elements;
    const uint32_t ng = static_cast<uint32_t>(e.n_groups);
    const uint32_t no = static_cast<uint32_t>(e.n_over);
    const DataType dt = static_cast<DataType>(e.dtype);
    if (group == 0) return "group_size == 0 in header";
    if (ng == 0)    return "n_groups == 0";

    // --- decode payload and aux blob ---
    // CCH_FLAG_BITPACK: [u32 wcs][zstd(widths)][packed bits].  The widths are
    // what the offsets are rebuilt from, exactly as the encoder built them, so
    // nothing but the widths and the packed stream is needed to invert this.
    const bool bitpacked = (e.flags & CCH_FLAG_BITPACK) != 0;
    const uint32_t nblk = static_cast<uint32_t>(
        (n + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
    std::vector<uint8_t>  zz;
    std::vector<uint8_t>  bw;
    std::vector<uint64_t> boff;
    size_t packed_n = 0;
    const uint8_t* packed_p = nullptr;
    if (bitpacked) {
        if (e.data_size < 4) return "bitpack payload truncated";
        uint32_t wcs = 0; std::memcpy(&wcs, payload, 4);
        if (4u + wcs > e.data_size) return "bitpack width block truncated";
        if (e.flags & CCH_FLAG_RAW_SIDE) {
            if (wcs != nblk) return "raw width block is " + std::to_string(wcs) + " B, expected " +
                                   std::to_string(nblk);
            bw.assign(payload + 4, payload + 4 + nblk);
        } else {
            bw = zstd_decompress(payload + 4, wcs, nblk);
        }
        if (bw.size() < nblk) return "bitpack widths short";
        boff.resize(nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < nblk; ++b) { boff[b] = run; run += 32ull * bw[b]; }
        packed_n = static_cast<size_t>(run);
        if (4u + wcs + packed_n > e.data_size) return "bitpack stream truncated";
        packed_p = payload + 4 + wcs;
    } else {
        zz = zstd_decompress(payload, e.data_size, n * sizeof(uint16_t));
    }
    const size_t aux_raw_n = static_cast<size_t>(ng) + static_cast<size_t>(ng) * 4u
                           + static_cast<size_t>(no) * 8u;
    std::vector<uint8_t> ax;
    if (e.flags & CCH_FLAG_RAW_SIDE) {
        if (e.aux_size != aux_raw_n) return "raw aux is " + std::to_string(e.aux_size) +
                                            " B, expected " + std::to_string(aux_raw_n);
        ax.assign(aux, aux + aux_raw_n);
    } else {
        ax = zstd_decompress(aux, e.aux_size, aux_raw_n);
    }

    const uint8_t* dcodes = ax.data();
    const float*   shrink = reinterpret_cast<const float*>(ax.data() + ng);
    const uint8_t* over   = ax.data() + ng + ng * 4u;

    // --- rebuild decoder-exact per-group steps on host ---
    std::vector<float> d_use(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        const double D_hat = std::exp(e.pg_lo + static_cast<double>(dcodes[g]) * e.pg_step);
        d_use[g] = static_cast<float>(D_hat * static_cast<double>(shrink[g]));
    }

    // --- split overflow (idx u32, val f32) interleaved ---
    std::vector<uint32_t> ovi(no);
    std::vector<float>    ovv(no);
    for (uint32_t k = 0; k < no; ++k) {
        std::memcpy(&ovi[k], over + k * 8u,     4);
        std::memcpy(&ovv[k], over + k * 8u + 4, 4);
    }

    // --- GPU decode, in the restore's scratch (no per-tensor alloc/free) ---
    if (n > S.cap.fp32 || ng > S.cap.step || no > S.cap.over ||
        (bitpacked && (nblk > S.cap.nblk || packed_n > S.cap.packed)) ||
        (!bitpacked && n > S.cap.zz))
        return "decode scratch too small (pre-pass missed this tensor)";
    void *d_zz = S.zz, *d_step = S.step, *d_fp32 = S.fp32, *d_ovi = S.ovi, *d_ovv = S.ovv;
    void *d_w = S.w, *d_off = S.off, *d_pk = S.pk;
    gpu::Stream_t s = S.s;
    (void)gpu::MemcpyAsync(d_step, d_use.data(), ng * sizeof(float), gpu::Copy::H2D, s);
    if (no) {
        (void)gpu::MemcpyAsync(d_ovi, ovi.data(), no * sizeof(uint32_t), gpu::Copy::H2D, s);
        (void)gpu::MemcpyAsync(d_ovv, ovv.data(), no * sizeof(float),    gpu::Copy::H2D, s);
    }

    if (bitpacked) {
        (void)gpu::MemcpyAsync(d_w, bw.data(), nblk, gpu::Copy::H2D, s);
        (void)gpu::MemcpyAsync(d_off, boff.data(), (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D, s);
        if (packed_n)
            (void)gpu::MemcpyAsync(d_pk, packed_p, packed_n, gpu::Copy::H2D, s);
        // log_domain=0 on purpose: the exp is applied AFTER scatter_overflow
        // below, so the overflow values go through it too -- matching the
        // encoder, which captured them in the log domain.
        codec::bitpack_dequant(static_cast<const uint8_t*>(d_pk),
                               static_cast<const uint8_t*>(d_w),
                               static_cast<const uint64_t*>(d_off), n, group,
                               static_cast<const float*>(d_step),
                               static_cast<float*>(d_fp32), nblk, 0, s);
    } else {
        (void)gpu::MemcpyAsync(d_zz, zz.data(), n * sizeof(uint16_t), gpu::Copy::H2D, s);
        codec::unzigzag_inplace(static_cast<uint16_t*>(d_zz), n, s);
        codec::dequant_pergroup(static_cast<const int16_t*>(d_zz), n, group,
                                static_cast<const float*>(d_step),
                                static_cast<float*>(d_fp32), s);
    }
    if (no)
        codec::scatter_overflow(static_cast<float*>(d_fp32),
                                static_cast<const uint32_t*>(d_ovi),
                                static_cast<const float*>(d_ovv), no, s);

    // exp_avg_sq LogPwRelQuant (default): values were quantized in the log
    // domain, so undo it here -- v_hat = exp(q*D).  Hard per-element relative
    // bound, no floor needed.
    if (e.log_domain) {
        codec::exp_fp32(static_cast<float*>(d_fp32), n,
                        static_cast<float*>(d_fp32), s);
    }
    // exp_avg_sq RmsRelGlobalQuant (aggressive, opt-in): floor v_hat so the
    // coarse global step can't drive Adam's 1/(sqrt(v)+eps) to a cliff.  Floor
    // is carried in pg_step (dcode is always 0, so pg_step is otherwise unused).
    else if (e.codec_id == static_cast<uint8_t>(CodecId::RmsRelGlobalQuant)) {
        const float floor = static_cast<float>(e.pg_step);
        if (floor > 0.0f)
            codec::clamp_floor_fp32(static_cast<float*>(d_fp32), n, floor, s);
    }

    if (st.byte_size != n * dtype_element_size(dt)) {
        (void)gpu::StreamSynchronize(s);
        return "live tensor byte size mismatch";
    }
    codec::contract_from_fp32(static_cast<const float*>(d_fp32), dt, n, st.device_ptr, s);

    // this stream only; also keeps the host buffers above alive long enough
    gpu::Error_t rc = gpu::StreamSynchronize(s);
    if (rc != gpu::Success) return std::string("decode sync failed: ") + gpu::ErrorString(rc);
    return {};
}

// --------------------------------------------------------------------------
// TtEqResidual: fp16 TT cores -> reconstruct low-rank part; EQ residual
// (single global step) -> dequant + overflow; sum -> live tensor.
//   aux (decompressed) = TtHeader ++ core_f16[core_elems] ++ (idx u32,val f32)[n_over]
//   payload (decompressed) = uint16 zigzag(EQ quant index)[n_elems]
// --------------------------------------------------------------------------
std::string refill_tt_eq(const CchIndexEntry& e, const uint8_t* payload,
                         const uint8_t* aux, const struct ShardTensor& st, DecodeScratch& S) {
    const DataType dt = static_cast<DataType>(e.dtype);
    if (dt != DataType::Float32) return "tt_eq: non-fp32 live tensor";

    // zstd_decompress needs the EXACT uncompressed size.  codec_tt_eq stashes
    // aux_raw.size() in e.pg_lo (unused by this codec); fall back to computing
    // it from the index (cores are fp16: e.n_groups == core_elems).
    size_t aux_exact = static_cast<size_t>(e.pg_lo);
    if (aux_exact == 0)
        aux_exact = sizeof(TtHeader)
                  + static_cast<size_t>(e.n_groups) * sizeof(uint16_t)
                  + static_cast<size_t>(e.n_over) * 8u;
    std::vector<uint8_t> ax;
    if (e.flags & CCH_FLAG_RAW_SIDE) {
        if (e.aux_size != aux_exact) return "tt_eq: raw aux is " + std::to_string(e.aux_size) +
                                            " B, expected " + std::to_string(aux_exact);
        ax.assign(aux, aux + aux_exact);
    } else {
        ax = zstd_decompress(aux, e.aux_size, aux_exact);
    }
    if (ax.size() < sizeof(TtHeader)) return "tt_eq: aux too short";

    TtHeader th{};
    std::memcpy(&th, ax.data(), sizeof(TtHeader));
    if (th.magic != TTEQ_MAGIC) return "tt_eq: bad TtHeader magic";

    const size_t   L  = static_cast<size_t>(th.n_elems);
    const uint32_t n1 = th.n1, n2 = th.n2, n3 = th.n3, r1 = th.r1, r2 = th.r2;
    const size_t   e1 = static_cast<size_t>(n1) * r1;
    const size_t   e2 = static_cast<size_t>(r1) * n2 * r2;
    const size_t   e3 = static_cast<size_t>(r2) * n3;
    const size_t   ce = e1 + e2 + e3;
    const size_t   P  = static_cast<size_t>(n1) * n2 * n3;
    const uint32_t no = static_cast<uint32_t>(th.n_over);
    if (ce != static_cast<size_t>(th.core_elems)) return "tt_eq: core_elems mismatch";
    if (L != st.num_elements || st.byte_size != L * sizeof(float))
        return "tt_eq: live tensor size mismatch";

    const size_t core_bytes = ce * (th.core_f16 ? sizeof(uint16_t) : sizeof(float));
    if (ax.size() < sizeof(TtHeader) + core_bytes + static_cast<size_t>(no) * 8u)
        return "tt_eq: aux truncated";
    const uint8_t* core_p = ax.data() + sizeof(TtHeader);
    const uint8_t* over_p = core_p + core_bytes;

    std::vector<uint32_t> ovi(no);
    std::vector<float>    ovv(no);
    for (uint32_t k = 0; k < no; ++k) {
        std::memcpy(&ovi[k], over_p + k * 8u,     4);
        std::memcpy(&ovv[k], over_p + k * 8u + 4, 4);
    }

    // CCH_FLAG_BITPACK: [u32 wcs][zstd(widths)][packed bits] -- see refill_pergroup.
    const bool tt_bitpacked = (e.flags & CCH_FLAG_BITPACK) != 0;
    const uint32_t tt_nblk = static_cast<uint32_t>(
        (L + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK);
    std::vector<uint8_t>  zz, tbw;
    std::vector<uint64_t> tboff;
    size_t tt_packed_n = 0;
    const uint8_t* tt_packed_p = nullptr;
    if (tt_bitpacked) {
        if (e.data_size < 4) return "tt_eq: bitpack payload truncated";
        uint32_t wcs = 0; std::memcpy(&wcs, payload, 4);
        if (4u + wcs > e.data_size) return "tt_eq: bitpack width block truncated";
        if (e.flags & CCH_FLAG_RAW_SIDE) {
            if (wcs != tt_nblk) return "tt_eq: raw width block is " + std::to_string(wcs) +
                                       " B, expected " + std::to_string(tt_nblk);
            tbw.assign(payload + 4, payload + 4 + tt_nblk);
        } else {
            tbw = zstd_decompress(payload + 4, wcs, tt_nblk);
        }
        if (tbw.size() < tt_nblk) return "tt_eq: bitpack widths short";
        tboff.resize(tt_nblk);
        uint64_t run = 0;
        for (uint32_t b = 0; b < tt_nblk; ++b) { tboff[b] = run; run += 32ull * tbw[b]; }
        tt_packed_n = static_cast<size_t>(run);
        if (4u + wcs + tt_packed_n > e.data_size) return "tt_eq: bitpack stream truncated";
        tt_packed_p = payload + 4 + wcs;
    } else {
        zz = zstd_decompress(payload, e.data_size, L * sizeof(uint16_t));
        if (zz.size() < L * sizeof(uint16_t)) return "tt_eq: payload truncated";
    }

    // ---- GPU decode, in the restore's scratch (no per-tensor alloc/free) ----
    const size_t t1n = static_cast<size_t>(n1) * n2 * r2;
    if (core_bytes > S.cap.core || e1 > S.cap.g1 || e2 > S.cap.g2 || e3 > S.cap.g3 ||
        P > S.cap.fp32 || L > S.cap.R || no > S.cap.over || t1n > S.cap.t1 ||
        (tt_bitpacked && (tt_nblk > S.cap.nblk || tt_packed_n > S.cap.packed)) ||
        (!tt_bitpacked && L > S.cap.zz))
        return "tt_eq: decode scratch too small (pre-pass missed this tensor)";
    void *d_core = S.core, *d_g1 = S.g1, *d_g2 = S.g2, *d_g3 = S.g3;
    void *d_Ahat = S.fp32, *d_zz = S.zz, *d_R = S.R, *d_step = S.step;
    void *d_ovi = S.ovi, *d_ovv = S.ovv;
    void *d_tw = S.w, *d_toff = S.off, *d_tpk = S.pk;

    gpu::Stream_t s = S.s;
    (void)gpu::MemcpyAsync(d_core, core_p, core_bytes, gpu::Copy::H2D, s);
    // NOT here: on the bit-pack path `zz` is empty, so an unconditional copy of
    // L*2 bytes from it reads off the end of the vector.  The payload upload
    // belongs inside the branch below, where its form is known.
    const float Df = static_cast<float>(th.d_step);
    (void)gpu::MemcpyAsync(d_step, &Df, sizeof(float), gpu::Copy::H2D, s);
    if (no) {
        (void)gpu::MemcpyAsync(d_ovi, ovi.data(), no * sizeof(uint32_t), gpu::Copy::H2D, s);
        (void)gpu::MemcpyAsync(d_ovv, ovv.data(), no * sizeof(float),    gpu::Copy::H2D, s);
    }

    // cores fp16 -> fp32
    if (th.core_f16) {
        const uint16_t* cp = static_cast<const uint16_t*>(d_core);
        codec::expand_to_fp32(cp,        DataType::Float16, e1, static_cast<float*>(d_g1), s);
        codec::expand_to_fp32(cp + e1,   DataType::Float16, e2, static_cast<float*>(d_g2), s);
        codec::expand_to_fp32(cp + e1+e2,DataType::Float16, e3, static_cast<float*>(d_g3), s);
    } else {
        const float* cp = static_cast<const float*>(d_core);
        (void)gpu::MemcpyAsync(d_g1, cp,        e1 * sizeof(float), gpu::Copy::D2D, s);
        (void)gpu::MemcpyAsync(d_g2, cp + e1,   e2 * sizeof(float), gpu::Copy::D2D, s);
        (void)gpu::MemcpyAsync(d_g3, cp + e1+e2,e3 * sizeof(float), gpu::Copy::D2D, s);
    }

    tt::TtCores cores;
    cores.g1 = static_cast<float*>(d_g1); cores.g2 = static_cast<float*>(d_g2);
    cores.g3 = static_cast<float*>(d_g3);
    cores.n1 = n1; cores.n2 = n2; cores.n3 = n3; cores.r1 = r1; cores.r2 = r2;

    void* blas = tt::blas_handle();
    tt::blas_set_stream(blas, s);
    if (!tt::tt_reconstruct_ws(cores, static_cast<float*>(d_Ahat), S.t1, blas, s)) {
        (void)gpu::StreamSynchronize(s);
        return "tt_eq: tt_reconstruct failed";
    }
    cores = tt::TtCores{};             // scratch-owned; never tt_free'd

    // residual: unpack/unzigzag -> dequant (single global step) -> scatter overflow
    if (tt_bitpacked) {
        (void)gpu::MemcpyAsync(d_tw, tbw.data(), tt_nblk, gpu::Copy::H2D, s);
        (void)gpu::MemcpyAsync(d_toff, tboff.data(), (size_t)tt_nblk * sizeof(uint64_t),
                               gpu::Copy::H2D, s);
        if (tt_packed_n)
            (void)gpu::MemcpyAsync(d_tpk, tt_packed_p, tt_packed_n, gpu::Copy::H2D, s);
        codec::bitpack_dequant(static_cast<const uint8_t*>(d_tpk),
                               static_cast<const uint8_t*>(d_tw),
                               static_cast<const uint64_t*>(d_toff), L,
                               static_cast<uint32_t>(L),
                               static_cast<const float*>(d_step),
                               static_cast<float*>(d_R), tt_nblk, 0, s);
    } else {
        (void)gpu::MemcpyAsync(d_zz, zz.data(), L * sizeof(uint16_t), gpu::Copy::H2D, s);
        codec::unzigzag_inplace(static_cast<uint16_t*>(d_zz), L, s);
        codec::dequant_pergroup(static_cast<const int16_t*>(d_zz), L, static_cast<uint32_t>(L),
                                static_cast<const float*>(d_step), static_cast<float*>(d_R), s);
    }
    if (no)
        codec::scatter_overflow(static_cast<float*>(d_R),
                                static_cast<const uint32_t*>(d_ovi),
                                static_cast<const float*>(d_ovv), no, s);

    // full reconstruction = low-rank[:L] + residual  -> live tensor
    codec::add_fp32(static_cast<const float*>(d_Ahat), static_cast<const float*>(d_R),
                    static_cast<float*>(d_Ahat), L, s);
    codec::contract_from_fp32(static_cast<const float*>(d_Ahat), dt, L, st.device_ptr, s);

    gpu::Error_t rc = gpu::StreamSynchronize(s);   // this stream only
    if (rc != gpu::Success) return std::string("tt_eq: decode sync failed: ") + gpu::ErrorString(rc);
    return {};
}

} // namespace

py::dict restart_apply(MemoryManager&              mgr,
                       const std::string&          input_dir,
                       uint32_t                    targets,
                       std::shared_ptr<RankLogger> logger,
                       int                         global_rank) {
    namespace fs = std::filesystem;

    const uint32_t rank_u = static_cast<uint32_t>(global_rank < 0 ? 0 : global_rank);
    char rankbuf[16];
    std::snprintf(rankbuf, sizeof(rankbuf), "%03u", rank_u);

    py::dict restored;
    py::list missing;
    py::list warnings;
    uint64_t total_restored = 0;

    // ---- pass 1: open every target stream, read header + index, and size the
    // decode scratch for the largest tensor of each kind ----
    std::array<std::unique_ptr<LoadedStream>, tracked_tensor_category_count> streams;
    DecodeNeed need;
    {
        std::vector<uint8_t> tmp;
        for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
            if ((targets & (1u << c)) == 0u) continue;
            const char* sname = tensor_category_name(static_cast<TensorCategory>(c));
            const std::string path = input_dir + "/rank_" + rankbuf + "_" + sname + ".cch";
            if (!fs::exists(path)) {
                missing.append(py::str(sname));
                if (logger) logger->log_message(std::string("[restart] no file for stream '") +
                    sname + "' (" + path + ") -- expected from a framework-native checkpoint");
                continue;
            }
            auto ls = std::make_unique<LoadedStream>();
            std::string err;
            if (!read_stream_file(path, *ls, err)) {
                warnings.append(py::str(std::string(sname) + ": " + err));
                if (logger) logger->log_message("[restart] stream '" + std::string(sname) +
                    "' unreadable: " + err);
                continue;
            }
            for (const auto& e : ls->index) {
                const size_t n = static_cast<size_t>(e.num_elements);
                const size_t nblk = (n + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK;
                const bool bp = (e.flags & CCH_FLAG_BITPACK) != 0;
                auto mx = [](size_t& a, size_t b) { a = std::max(a, b); };
                if (e.codec_id == static_cast<uint8_t>(CodecId::RmsRelPgQuant) ||
                    e.codec_id == static_cast<uint8_t>(CodecId::LogPwRelQuant) ||
                    e.codec_id == static_cast<uint8_t>(CodecId::RmsRelGlobalQuant)) {
                    mx(need.fp32, n); mx(need.step, e.n_groups); mx(need.over, e.n_over);
                    if (bp) { mx(need.nblk, nblk); mx(need.packed, e.data_size); }
                    else      mx(need.zz, n);
                } else if (e.codec_id == static_cast<uint8_t>(CodecId::TtEqResidual)) {
                    // the TtHeader (dims, ranks, overflow count) is at the front of aux
                    if (!read_data(*ls, e.aux_offset, e.aux_size, tmp)) continue;
                    std::vector<uint8_t> ax;
                    if (e.flags & CCH_FLAG_RAW_SIDE) ax = tmp;
                    else {
                        size_t exact = static_cast<size_t>(e.pg_lo);
                        if (exact == 0) exact = sizeof(TtHeader) + static_cast<size_t>(e.n_groups) * 2u
                                              + static_cast<size_t>(e.n_over) * 8u;
                        try { ax = zstd_decompress(tmp.data(), tmp.size(), exact); } catch (...) { continue; }
                    }
                    if (ax.size() < sizeof(TtHeader)) continue;
                    TtHeader th{}; std::memcpy(&th, ax.data(), sizeof(TtHeader));
                    if (th.magic != TTEQ_MAGIC) continue;
                    const size_t L = th.n_elems, P = static_cast<size_t>(th.n1) * th.n2 * th.n3;
                    const size_t e1 = static_cast<size_t>(th.n1) * th.r1;
                    const size_t e2 = static_cast<size_t>(th.r1) * th.n2 * th.r2;
                    const size_t e3 = static_cast<size_t>(th.r2) * th.n3;
                    const size_t lb = (L + codec::BITPACK_BLOCK - 1) / codec::BITPACK_BLOCK;
                    mx(need.fp32, P); mx(need.R, L); mx(need.over, th.n_over);
                    mx(need.core, (e1 + e2 + e3) * (th.core_f16 ? 2u : 4u));
                    mx(need.g1, e1); mx(need.g2, e2); mx(need.g3, e3);
                    mx(need.t1, static_cast<size_t>(th.n1) * th.n2 * th.r2);
                    if (bp) { mx(need.nblk, lb); mx(need.packed, e.data_size); }
                    else      mx(need.zz, L);
                }
            }
            streams[c] = std::move(ls);
        }
    }
    DecodeScratch S;
    {
        const std::string aerr = S.alloc(need);
        if (!aerr.empty()) throw std::runtime_error("[restart] " + aerr);
        if (logger) logger->log_message("[restart] decode scratch allocated once: " +
            std::to_string(S.bytes) + " B (largest tensor: " + std::to_string(need.fp32) + " fp32 elems)");
    }

    // ---- pass 2: decode every tensor into the scratch, then the live tensor ----
    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
        const uint32_t bit = 1u << c;
        if (!streams[c]) continue;
        const char* sname = tensor_category_name(static_cast<TensorCategory>(c));
        LoadedStream& ls = *streams[c];

        std::unordered_map<std::string, const CchIndexEntry*> by_name;
        by_name.reserve(ls.index.size());
        for (const auto& e : ls.index) by_name.emplace(std::string(e.name), &e);

        uint64_t n_ok = 0;
        std::vector<uint8_t> buf;          // this tensor's payload + aux, reused
        for (const auto& stv : mgr.collect_tensors(bit)) {
            auto it = by_name.find(stv.name);
            if (it == by_name.end()) {
                warnings.append(py::str(std::string(sname) + ": no entry for '" + stv.name + "'"));
                continue;
            }
            const CchIndexEntry& e = *it->second;
            // payload and aux are adjacent in practice; read the span covering both
            const uint64_t lo = e.aux_size ? std::min(e.data_offset, e.aux_offset) : e.data_offset;
            const uint64_t hi = e.aux_size ? std::max(e.data_offset + e.data_size, e.aux_offset + e.aux_size)
                                           : e.data_offset + e.data_size;
            if (hi > ls.data_len) {
                warnings.append(py::str(std::string(sname) + ": '" + stv.name +
                                        "' payload/aux past the end of the data section"));
                continue;
            }
            if (!read_data(ls, lo, hi - lo, buf)) {
                warnings.append(py::str(std::string(sname) + ": '" + stv.name + "' short read"));
                continue;
            }
            const uint8_t* payload = buf.data() + (e.data_offset - lo);
            const uint8_t* aux     = buf.data() + (e.aux_size ? e.aux_offset - lo : 0);

            std::string werr;
            try {
                if (e.codec_id == static_cast<uint8_t>(CodecId::RmsRelPgQuant)     ||
                    e.codec_id == static_cast<uint8_t>(CodecId::LogPwRelQuant)     ||
                    e.codec_id == static_cast<uint8_t>(CodecId::RmsRelGlobalQuant))
                    werr = refill_pergroup(e, ls.header.group_size, payload, aux, stv, S);
                else if (e.codec_id == static_cast<uint8_t>(CodecId::TtEqResidual))
                    werr = refill_tt_eq(e, payload, aux, stv, S);
                else
                    werr = refill_raw(e, payload, stv, S);
            } catch (const std::exception& ex) {
                werr = ex.what();
            }

            if (!werr.empty()) {
                warnings.append(py::str(std::string(sname) + " '" + stv.name + "': " + werr));
                continue;
            }
            ++n_ok;
        }

        (void)gpu::StreamSynchronize(S.s);
        restored[py::str(sname)] = n_ok;
        total_restored += n_ok;
        if (logger) logger->log_message("[restart] stream '" + std::string(sname) +
            "' refilled " + std::to_string(n_ok) + "/" + std::to_string(ls.index.size()) + " tensors");
    }

    S.release();             // the restore's only frees
    py::dict out;
    out["rank"]           = rank_u;
    out["input_dir"]      = input_dir;
    out["restored"]       = restored;
    out["missing"]        = missing;
    out["warnings"]       = warnings;
    out["total_restored"] = total_restored;
    return out;
}

} // namespace checkers
