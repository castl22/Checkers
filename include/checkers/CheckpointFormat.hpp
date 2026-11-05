#pragma once
// On-disk format for one compressed checkpoint stream.
//
// One file per (rank, stream):  <dir>/rank_<NNN>_<stream>.cch
//   stream in { model_state, master_weights, exp_avg, exp_avg_sq }
// Plus a sidecar:               <dir>/rank_<NNN>_manifest.json
//
// A stream file is fully self-describing and independently decodable, so a
// perturbation study can read / rewrite exactly one stream without touching
// the others.
//
// Layout:
//   [ CchHeader ]                       64 B
//   [ CchIndexEntry * n_tensors ]       fixed stride, tensor order = registry order
//   [ data section ]                    per tensor: main payload, then aux payload
//
// All integers little-endian (x86-64 / same-endian restore only, for now).

#include <cstdint>

namespace checkers {

inline constexpr uint32_t CCH_MAGIC   = 0x31484343u; // "CCH1"
// v2: per-group quant codec params; index right after the header, then data.
// v3 (2026-11-05): header, then data, then the index at the END of the file
//   (header.index_offset points at it), so a writer can append each tensor as
//   soon as it is compressed and write the index last.  Written by every save;
//   no zstd anywhere on the save path (see CCH_FLAG_RAW_SIDE).
// Readers take both: the index is always at header.index_offset, and the data
// section ends at index_offset when the index follows it (v3), else at EOF (v2).
inline constexpr uint32_t CCH_VERSION          = 3u;  // what the writer produces
inline constexpr uint32_t CCH_VERSION_MIN_READ = 2u;  // oldest the reader accepts
inline constexpr uint32_t CCH_MAX_DIMS = 8u;

// Per-tensor flags
inline constexpr uint8_t CCH_FLAG_RAW_PASSTHROUGH = 1u << 0; // payload is raw bytes, no zstd
inline constexpr uint8_t CCH_FLAG_ZSTD            = 1u << 1; // payload is zstd frame
// Payload is GPU fixed-width bit-packed:  [u32 wcs][zstd(widths)][packed bits].
// Per-TENSOR, so a v2 file written with the zstd payload still decodes through
// the old path -- Restart dispatches on this flag, not on the file version.
inline constexpr uint8_t CCH_FLAG_BITPACK         = 1u << 2;
// Side data stored RAW (2026-11-05, every v3 lossy tensor): the bit-pack width
// block is the nblk width bytes themselves (wcs == nblk) and the aux blob is
// the uncompressed blob (per-group dcodes/shrink/overflow, or TtHeader + cores
// + overflow).  Without it both are zstd frames (v2 files).
inline constexpr uint8_t CCH_FLAG_RAW_SIDE        = 1u << 3;

// Aux blob layout for the per-group quant codecs (uncompressed, before zstd):
//   [ n_groups * uint8   dcodes  ]  log-domain step codes
//   [ n_groups * float32 shrink  ]  per-group ULP shrink factor
//   [ n_over   * (uint32 idx + float32 val) ]  overflow-captured elements
// Main payload = zstd( num_elements * uint16  zigzag(quant index) ).

#pragma pack(push, 1)

struct CchHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t rank;
    uint32_t category;      // TensorCategory value
    uint32_t codec_id;      // dominant CodecId for this stream
    uint32_t group_size;
    uint32_t n_tensors;
    uint32_t flags;
    double   error_bound;
    uint64_t index_offset;  // == sizeof(CchHeader)
    uint64_t data_offset;   // == index_offset + n_tensors * sizeof(CchIndexEntry)
    uint64_t total_orig_bytes;
    uint64_t total_payload_bytes;
    uint64_t reserved[2];
};

struct CchIndexEntry {
    char     name[256];
    uint8_t  dtype;         // DataType value
    uint8_t  codec_id;      // CodecId actually used for this tensor
    uint8_t  flags;         // CCH_FLAG_*
    uint8_t  ndim;
    uint32_t pad0;
    uint64_t num_elements;
    uint64_t orig_bytes;
    uint64_t shape[CCH_MAX_DIMS];

    uint64_t data_offset;   // main payload, relative to CchHeader.data_offset
    uint64_t data_size;
    uint64_t aux_offset;    // codec side-info, relative to CchHeader.data_offset
    uint64_t aux_size;

    // codec parameters (used by the lossy codecs; zero for RawZstd)
    double   d_global;      // single global quant step (0 => per-group steps in aux)
    uint64_t n_groups;
    uint64_t n_over;        // overflow-captured elements
    uint64_t n_zero;        // exact-zero elements (log-domain codecs)
    uint8_t  log_domain;
    uint8_t  cparam_pad[7];
    double   pg_lo;         // per-group: log-domain base   (D_hat = exp(pg_lo + dcode*pg_step))
    double   pg_step;       // per-group: log-domain quantum
};

// TT + EQ residual codec (exp_avg).  Written at the FRONT of the aux blob
// (before zstd), followed by the fp16 TT cores (core_elems * 2 bytes,
// g1 ++ g2 ++ g3) and then the residual overflow list
// (n_over * (uint32 idx + float32 val)).  Main payload = zstd( n_elems * uint16
// zigzag(EQ quant index) ).
struct TtHeader {
    uint32_t magic;        // TTEQ_MAGIC
    uint16_t ndim;         // 2 or 3
    uint16_t eq_mode;      // 0 ABS, 1 REL, 2 PW_REL
    uint32_t n1, n2, n3;   // reshape dims (n1*n2*n3 >= n_elems, tail zero-padded)
    uint32_t r1, r2;       // TT bond dimensions actually used
    uint64_t n_elems;      // unpadded shard length
    uint64_t core_elems;   // n1*r1 + r1*n2*r2 + r2*n3
    uint64_t n_over;       // residual overflow-captured elements
    double   d_step;       // EQ uniform quant step (raw residual units)
    float    resid_min;
    float    resid_max;
    uint32_t core_f16;     // 1 => cores stored fp16, 0 => fp32
    uint32_t pad;
};

inline constexpr uint32_t TTEQ_MAGIC = 0x51455454u; // "TTEQ"

#pragma pack(pop)

static_assert(sizeof(TtHeader) == 76, "TtHeader size drift");
static_assert(sizeof(CchHeader) == 88, "CchHeader size drift -- bump CCH_VERSION if intentional");
static_assert(sizeof(CchHeader) % 8 == 0, "CchHeader must stay 8-byte aligned");
static_assert(sizeof(CchIndexEntry) == 432, "CchIndexEntry size drift -- bump CCH_VERSION if intentional");
static_assert(sizeof(CchIndexEntry) % 8 == 0, "CchIndexEntry must stay 8-byte aligned");

} // namespace checkers
