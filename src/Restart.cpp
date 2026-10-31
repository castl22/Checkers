#include "checkers/CheckpointCompressor.hpp"

#include "checkers/CheckpointFormat.hpp"
#include "checkers/CodecKernels.hpp"
#include "checkers/GpuDefs.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/TensorResource.hpp"
#include "checkers/TtLinalg.hpp"
#include "checkers/ZstdUtil.hpp"
#include "checkers/logging.hpp"

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

struct LoadedStream {
    CchHeader                  header{};
    std::vector<CchIndexEntry> index;
    std::vector<uint8_t>       data;   // raw data section (payloads + aux concatenated)
};

bool read_stream_file(const std::string& path, LoadedStream& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open"; return false; }

    f.read(reinterpret_cast<char*>(&out.header), sizeof(CchHeader));
    if (!f || out.header.magic != CCH_MAGIC) { err = "bad magic"; return false; }
    if (out.header.version != CCH_VERSION)   { err = "version mismatch"; return false; }

    out.index.resize(out.header.n_tensors);
    f.read(reinterpret_cast<char*>(out.index.data()),
           static_cast<std::streamsize>(out.index.size() * sizeof(CchIndexEntry)));
    if (!f) { err = "truncated index"; return false; }

    f.seekg(0, std::ios::end);
    const std::streamoff file_end = f.tellg();
    const std::streamoff data_beg = static_cast<std::streamoff>(out.header.data_offset);
    if (file_end < data_beg) { err = "truncated data"; return false; }
    out.data.resize(static_cast<size_t>(file_end - data_beg));
    f.seekg(data_beg, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(out.data.size()));
    if (!f) { err = "short read on data section"; return false; }
    return true;
}

// --------------------------------------------------------------------------
// RawZstd (or raw passthrough): payload bytes -> live tensor.
// --------------------------------------------------------------------------
std::string refill_raw(const CchIndexEntry& e, const uint8_t* payload,
                       const struct ShardTensor& st) {
    std::vector<uint8_t> raw;
    if (e.flags & CCH_FLAG_ZSTD) raw = zstd_decompress(payload, e.data_size, e.orig_bytes);
    else                          raw.assign(payload, payload + e.data_size);

    if (raw.size() != st.byte_size)
        return "size mismatch (file " + std::to_string(raw.size()) + " vs live " +
               std::to_string(st.byte_size) + ")";

    gpu::Error_t rc = gpu::Memcpy(st.device_ptr, raw.data(), st.byte_size, gpu::Copy::H2D);
    if (rc != gpu::Success) return std::string("H2D failed: ") + gpu::ErrorString(rc);
    return {};
}

// --------------------------------------------------------------------------
// RmsRelPgQuant: unzigzag -> per-group dequant -> restore overflow ->
//                contract to storage dtype -> live tensor.
// --------------------------------------------------------------------------
std::string refill_pergroup(const CchIndexEntry& e, uint32_t group,
                            const uint8_t* payload, const uint8_t* aux,
                            const struct ShardTensor& st) {
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
        bw = zstd_decompress(payload + 4, wcs, nblk);
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
    std::vector<uint8_t> ax = zstd_decompress(aux, e.aux_size, aux_raw_n);

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

    // --- GPU decode ---
    void *d_zz = nullptr, *d_step = nullptr, *d_fp32 = nullptr, *d_ovi = nullptr, *d_ovv = nullptr;
    void *d_w = nullptr, *d_off = nullptr, *d_pk = nullptr;
    auto cleanup = [&]() {
        if (d_w)    (void)gpu::Free(d_w);
        if (d_off)  (void)gpu::Free(d_off);
        if (d_pk)   (void)gpu::Free(d_pk);
        if (d_zz)   (void)gpu::Free(d_zz);
        if (d_step) (void)gpu::Free(d_step);
        if (d_fp32) (void)gpu::Free(d_fp32);
        if (d_ovi)  (void)gpu::Free(d_ovi);
        if (d_ovv)  (void)gpu::Free(d_ovv);
    };

    if (gpu::Malloc(&d_zz,   n * sizeof(uint16_t)) != gpu::Success ||
        gpu::Malloc(&d_step, ng * sizeof(float))   != gpu::Success ||
        gpu::Malloc(&d_fp32, n * sizeof(float))    != gpu::Success ||
        (no && gpu::Malloc(&d_ovi, no * sizeof(uint32_t)) != gpu::Success) ||
        (no && gpu::Malloc(&d_ovv, no * sizeof(float))    != gpu::Success)) {
        cleanup();
        return "device alloc failed";
    }

    gpu::Stream_t s = nullptr;   // default stream is fine for the one-shot restart path
    (void)gpu::Memcpy(d_step, d_use.data(), ng * sizeof(float),   gpu::Copy::H2D);
    if (no) {
        (void)gpu::Memcpy(d_ovi, ovi.data(), no * sizeof(uint32_t), gpu::Copy::H2D);
        (void)gpu::Memcpy(d_ovv, ovv.data(), no * sizeof(float),    gpu::Copy::H2D);
    }

    if (bitpacked) {
        if (gpu::Malloc(&d_w, nblk) != gpu::Success ||
            gpu::Malloc(&d_off, (size_t)nblk * sizeof(uint64_t)) != gpu::Success ||
            gpu::Malloc(&d_pk, std::max<size_t>(packed_n, 1)) != gpu::Success) {
            cleanup(); return "bitpack device alloc failed";
        }
        (void)gpu::Memcpy(d_w, bw.data(), nblk, gpu::Copy::H2D);
        (void)gpu::Memcpy(d_off, boff.data(), (size_t)nblk * sizeof(uint64_t), gpu::Copy::H2D);
        if (packed_n)
            (void)gpu::Memcpy(d_pk, packed_p, packed_n, gpu::Copy::H2D);
        // log_domain=0 on purpose: the exp is applied AFTER scatter_overflow
        // below, so the overflow values go through it too -- matching the
        // encoder, which captured them in the log domain.
        codec::bitpack_dequant(static_cast<const uint8_t*>(d_pk),
                               static_cast<const uint8_t*>(d_w),
                               static_cast<const uint64_t*>(d_off), n, group,
                               static_cast<const float*>(d_step),
                               static_cast<float*>(d_fp32), nblk, 0, s);
    } else {
        (void)gpu::Memcpy(d_zz, zz.data(), n * sizeof(uint16_t), gpu::Copy::H2D);
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
        cleanup();
        return "live tensor byte size mismatch";
    }
    codec::contract_from_fp32(static_cast<const float*>(d_fp32), dt, n, st.device_ptr, s);

    gpu::Error_t rc = gpu::DeviceSynchronize();
    cleanup();
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
                         const uint8_t* aux, const struct ShardTensor& st) {
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
    std::vector<uint8_t> ax = zstd_decompress(aux, e.aux_size, aux_exact);
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
        tbw = zstd_decompress(payload + 4, wcs, tt_nblk);
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

    // ---- GPU decode ----
    void *d_core = nullptr, *d_g1 = nullptr, *d_g2 = nullptr, *d_g3 = nullptr;
    void *d_Ahat = nullptr, *d_zz = nullptr, *d_R = nullptr, *d_step = nullptr;
    void *d_ovi = nullptr, *d_ovv = nullptr;
    void *d_tw = nullptr, *d_toff = nullptr, *d_tpk = nullptr;
    auto cleanup = [&]() {
        for (void* p : {d_core, d_g1, d_g2, d_g3, d_Ahat, d_zz, d_R, d_step, d_ovi,
                        d_ovv, d_tw, d_toff, d_tpk})
            if (p) (void)gpu::Free(p);
    };
    if (gpu::Malloc(&d_core, core_bytes)          != gpu::Success ||
        gpu::Malloc(&d_g1,  e1 * sizeof(float))   != gpu::Success ||
        gpu::Malloc(&d_g2,  e2 * sizeof(float))   != gpu::Success ||
        gpu::Malloc(&d_g3,  e3 * sizeof(float))   != gpu::Success ||
        gpu::Malloc(&d_Ahat, P * sizeof(float))   != gpu::Success ||
        gpu::Malloc(&d_zz,  L * sizeof(uint16_t)) != gpu::Success ||
        gpu::Malloc(&d_R,   L * sizeof(float))    != gpu::Success ||
        gpu::Malloc(&d_step, sizeof(float))       != gpu::Success ||
        (no && gpu::Malloc(&d_ovi, no * sizeof(uint32_t)) != gpu::Success) ||
        (no && gpu::Malloc(&d_ovv, no * sizeof(float))    != gpu::Success)) {
        cleanup();
        return "tt_eq: device alloc failed";
    }

    gpu::Stream_t s = nullptr;
    (void)gpu::Memcpy(d_core, core_p, core_bytes, gpu::Copy::H2D);
    // NOT here: on the bit-pack path `zz` is empty, so an unconditional copy of
    // L*2 bytes from it reads off the end of the vector.  The payload upload
    // belongs inside the branch below, where its form is known.
    const float Df = static_cast<float>(th.d_step);
    (void)gpu::Memcpy(d_step, &Df, sizeof(float), gpu::Copy::H2D);
    if (no) {
        (void)gpu::Memcpy(d_ovi, ovi.data(), no * sizeof(uint32_t), gpu::Copy::H2D);
        (void)gpu::Memcpy(d_ovv, ovv.data(), no * sizeof(float),    gpu::Copy::H2D);
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
    if (!tt::tt_reconstruct(cores, static_cast<float*>(d_Ahat), blas, s)) {
        cores = tt::TtCores{};         // g1..g3 freed by cleanup, not tt_free
        cleanup();
        return "tt_eq: tt_reconstruct failed";
    }
    cores = tt::TtCores{};             // detach; cleanup() owns d_g1..d_g3

    // residual: unpack/unzigzag -> dequant (single global step) -> scatter overflow
    if (tt_bitpacked) {
        if (gpu::Malloc(&d_tw, tt_nblk) != gpu::Success ||
            gpu::Malloc(&d_toff, (size_t)tt_nblk * sizeof(uint64_t)) != gpu::Success ||
            gpu::Malloc(&d_tpk, std::max<size_t>(tt_packed_n, 1)) != gpu::Success) {
            cleanup(); return "tt_eq: bitpack device alloc failed";
        }
        (void)gpu::Memcpy(d_tw, tbw.data(), tt_nblk, gpu::Copy::H2D);
        (void)gpu::Memcpy(d_toff, tboff.data(), (size_t)tt_nblk * sizeof(uint64_t),
                          gpu::Copy::H2D);
        if (tt_packed_n)
            (void)gpu::Memcpy(d_tpk, tt_packed_p, tt_packed_n, gpu::Copy::H2D);
        codec::bitpack_dequant(static_cast<const uint8_t*>(d_tpk),
                               static_cast<const uint8_t*>(d_tw),
                               static_cast<const uint64_t*>(d_toff), L,
                               static_cast<uint32_t>(L),
                               static_cast<const float*>(d_step),
                               static_cast<float*>(d_R), tt_nblk, 0, s);
    } else {
        (void)gpu::Memcpy(d_zz, zz.data(), L * sizeof(uint16_t), gpu::Copy::H2D);
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

    gpu::Error_t rc = gpu::DeviceSynchronize();
    cleanup();
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

    for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
        const uint32_t bit = 1u << c;
        if ((targets & bit) == 0u) continue;

        const auto  cat   = static_cast<TensorCategory>(c);
        const char* sname = tensor_category_name(cat);
        const std::string path = input_dir + "/rank_" + rankbuf + "_" + sname + ".cch";

        if (!fs::exists(path)) {
            missing.append(py::str(sname));
            if (logger) logger->log_message(std::string("[restart] no file for stream '") +
                sname + "' (" + path + ") -- expected from a framework-native checkpoint");
            continue;
        }

        LoadedStream ls;
        std::string err;
        if (!read_stream_file(path, ls, err)) {
            warnings.append(py::str(std::string(sname) + ": " + err));
            if (logger) logger->log_message("[restart] stream '" + std::string(sname) +
                "' unreadable: " + err);
            continue;
        }

        std::unordered_map<std::string, const CchIndexEntry*> by_name;
        by_name.reserve(ls.index.size());
        for (const auto& e : ls.index) by_name.emplace(std::string(e.name), &e);

        uint64_t n_ok = 0;
        for (const auto& stv : mgr.collect_tensors(bit)) {
            auto it = by_name.find(stv.name);
            if (it == by_name.end()) {
                warnings.append(py::str(std::string(sname) + ": no entry for '" + stv.name + "'"));
                continue;
            }
            const CchIndexEntry& e = *it->second;
            const uint8_t* payload = ls.data.data() + e.data_offset;
            const uint8_t* aux     = ls.data.data() + e.aux_offset;

            std::string werr;
            try {
                if (e.codec_id == static_cast<uint8_t>(CodecId::RmsRelPgQuant)     ||
                    e.codec_id == static_cast<uint8_t>(CodecId::LogPwRelQuant)     ||
                    e.codec_id == static_cast<uint8_t>(CodecId::RmsRelGlobalQuant))
                    werr = refill_pergroup(e, ls.header.group_size, payload, aux, stv);
                else if (e.codec_id == static_cast<uint8_t>(CodecId::TtEqResidual))
                    werr = refill_tt_eq(e, payload, aux, stv);
                else
                    werr = refill_raw(e, payload, stv);
            } catch (const std::exception& ex) {
                werr = ex.what();
            }

            if (!werr.empty()) {
                warnings.append(py::str(std::string(sname) + " '" + stv.name + "': " + werr));
                continue;
            }
            ++n_ok;
        }

        (void)gpu::DeviceSynchronize();
        restored[py::str(sname)] = n_ok;
        total_restored += n_ok;
        if (logger) logger->log_message("[restart] stream '" + std::string(sname) +
            "' refilled " + std::to_string(n_ok) + "/" + std::to_string(ls.index.size()) + " tensors");
    }

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
