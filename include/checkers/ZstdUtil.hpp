#pragma once
// Tiny inline wrappers around the vendored zstd (third_party/zstd).
// Shared by the compressor and the restart path.

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <zstd.h>

namespace checkers {

inline std::vector<uint8_t> zstd_compress(const void* src, size_t n, int level) {
    const size_t bound = ZSTD_compressBound(n);
    std::vector<uint8_t> out(bound);
    const size_t w = ZSTD_compress(out.data(), bound, src, n, level);
    if (ZSTD_isError(w)) {
        throw std::runtime_error(std::string("[checkers][zstd] compress: ") + ZSTD_getErrorName(w));
    }
    out.resize(w);
    return out;
}

// Multi-threaded variant.  workers<=1 is byte-identical to zstd_compress.
// With workers>1 zstd splits the input internally and still emits ONE standard
// frame, so the stream format and the decompress/restart path are unchanged.
//
// Why this exists: the production compressor gets its parallelism from the
// worker pool (one tensor per thread), but any path with a single buffer --
// codec_selftest, and therefore the SOTA benchmark -- was left with a serial
// entropy stage measured at 0.27 GB/s against a 1.63 GB/s GPU codec, i.e. 85%
// of compress time doing zstd on one core.
inline std::vector<uint8_t> zstd_compress_mt(const void* src, size_t n,
                                             int level, int workers) {
    if (workers <= 1) return zstd_compress(src, n, level);

    const size_t bound = ZSTD_compressBound(n);
    std::vector<uint8_t> out(bound);

    ZSTD_CCtx* ctx = ZSTD_createCCtx();
    if (ctx == nullptr) throw std::runtime_error("[checkers][zstd] createCCtx failed");

    size_t rc = ZSTD_CCtx_setParameter(ctx, ZSTD_c_compressionLevel, level);
    if (ZSTD_isError(rc)) { ZSTD_freeCCtx(ctx);
        throw std::runtime_error(std::string("[checkers][zstd] set level: ") + ZSTD_getErrorName(rc)); }

    // Not fatal if the build lacks ZSTD_MULTITHREAD: zstd reports an error and
    // we fall through to a correct single-threaded compression.
    rc = ZSTD_CCtx_setParameter(ctx, ZSTD_c_nbWorkers, workers);
    if (ZSTD_isError(rc)) {
        ZSTD_freeCCtx(ctx);
        return zstd_compress(src, n, level);
    }

    const size_t w = ZSTD_compress2(ctx, out.data(), bound, src, n);
    ZSTD_freeCCtx(ctx);
    if (ZSTD_isError(w)) {
        throw std::runtime_error(std::string("[checkers][zstd] compress2: ") + ZSTD_getErrorName(w));
    }
    out.resize(w);
    return out;
}

// Decompress into a caller-owned buffer (so the destination can be PINNED and
// reused, instead of a fresh pageable std::vector on every call).
inline size_t zstd_decompress_into(void* dst, size_t dst_cap,
                                   const void* src, size_t n) {
    const size_t w = ZSTD_decompress(dst, dst_cap, src, n);
    if (ZSTD_isError(w)) {
        throw std::runtime_error(std::string("[checkers][zstd] decompress_into: ") +
                                 ZSTD_getErrorName(w));
    }
    return w;
}

inline std::vector<uint8_t> zstd_decompress(const void* src, size_t n, size_t original_bytes) {
    std::vector<uint8_t> out(original_bytes);
    const size_t w = ZSTD_decompress(out.data(), original_bytes, src, n);
    if (ZSTD_isError(w)) {
        throw std::runtime_error(std::string("[checkers][zstd] decompress: ") + ZSTD_getErrorName(w));
    }
    if (w != original_bytes) {
        throw std::runtime_error("[checkers][zstd] decompress: size mismatch (expected " +
                                 std::to_string(original_bytes) + ", got " + std::to_string(w) + ")");
    }
    return out;
}

} // namespace checkers
