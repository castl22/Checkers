#include "checkers/CodecKernels.hpp"
#include "checkers/TensorResource.hpp"   // full DataType definition

#include <RAJA/RAJA.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  include <hip/hip_fp16.h>
#else
#  include <cuda_fp16.h>
#endif

namespace checkers {
namespace codec {

// Thread geometry for the bit-pack kernels.
//
// These kernels are limited by MEMORY-LEVEL PARALLELISM, not by bandwidth or by
// arithmetic.  Measured on one 0.5 GiB chunk, every variant reading the same
// bytes (see also the encode/dequant notes below):
//
//   kernel                               values/lane  loads/lane   read rate
//   k_group_rms (strided loop)                16          16       122 GB/s
//   quantize, scalar                           1           1        33 GB/s
//   quantize, 1 float4 per lane                4           1        19 GB/s
//   quantize, 16 contiguous per lane          16           4        25 GB/s
//
// Two things were tried and REJECTED by measurement, both recorded here so they
// are not tried again:
//   * Re-deriving the codes in a second pass instead of storing them (to avoid
//     writing the 0.25 GiB u16 array) -- 51 ms against 39 ms.  Writes are not
//     expensive on this part; quantizing twice is.
//   * Giving each lane one float4 from each of 4 different blocks, so every
//     wavefront load is perfectly coalesced -- 66 ms against 39 ms.  It needs
//     four full-wavefront reductions per lane, and those cost more than the
//     coalescing saves.
//
// So the compress path uses ONE WAVEFRONT PER BLOCK with VP_C contiguous values
// per lane, and the decompress path uses LPB_D lanes per block with VP_D
// contiguous values per lane -- which is what each direction measured fastest
// at.  In both, the per-block reduction stays inside a wavefront, so there is
// no shared memory and no barrier in the width pass.
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
inline constexpr uint32_t WAVE = 64;      // CDNA (gfx942); build pins this arch
#else
inline constexpr uint32_t WAVE = 32;
#endif
inline constexpr uint32_t BP_WG = 256;                        // threads/workgroup

// compress: one wavefront owns one bit-pack block
inline constexpr uint32_t VP_C   = BITPACK_BLOCK / WAVE;      // values per lane
inline constexpr uint32_t C_BLKS = BP_WG / WAVE;              // blocks/workgroup
inline constexpr uint32_t C_VALS = C_BLKS * BITPACK_BLOCK;    // values/workgroup

// decompress: LPB_D lanes own one bit-pack block
inline constexpr uint32_t VP_D   = 16;                        // values per lane
inline constexpr uint32_t LPB_D  = BITPACK_BLOCK / VP_D;      // lanes per block
inline constexpr uint32_t D_BLKS = BP_WG / LPB_D;             // blocks/workgroup
inline constexpr uint32_t D_VALS = D_BLKS * BITPACK_BLOCK;    // values/workgroup

static_assert(VP_C % 4 == 0 && VP_D % 4 == 0, "float4 loads need a multiple of 4");
static_assert(LPB_D <= WAVE && (WAVE % LPB_D) == 0,
              "a block's lanes must sit inside one wavefront");

#if defined(RAJA_ENABLE_HIP)
using CodecExec   = RAJA::hip_exec<256>;
using CodecReduce = RAJA::hip_reduce;
#elif defined(RAJA_ENABLE_CUDA)
using CodecExec   = RAJA::cuda_exec<256>;
using CodecReduce = RAJA::cuda_reduce;
#else
using CodecExec   = RAJA::seq_exec;
using CodecReduce = RAJA::seq_reduce;
#endif

// v3: exp_avg_sq default codec is LogPwRelQuant (quantize log(v), decode
//     v_hat = exp(q*D)); RmsRelGlobalQuant kept as an opt-in.  Decode still
//     dispatches on codec_id, so v2 stream files round-trip unchanged.
int kernels_abi_version() { return 3; }

// ================================================================= //
//  device helpers
// ================================================================= //
__device__ __forceinline__ float bits_to_f32(uint32_t b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __uint_as_float(b);
#else
    float f; __builtin_memcpy(&f, &b, 4); return f;
#endif
}
__device__ __forceinline__ uint32_t f32_to_bits(float f) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t b; __builtin_memcpy(&b, &f, 4); return b;
#endif
}

// ================================================================= //
//  kernels
// ================================================================= //
__global__ void k_expand_bf16(const uint16_t* __restrict__ in, size_t n,
                              float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bits_to_f32((uint32_t)in[i] << 16);
}

__global__ void k_expand_fp16(const uint16_t* __restrict__ in, size_t n,
                              float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h; __builtin_memcpy(&h, &in[i], sizeof(__half));
    out[i] = __half2float(h);
}

__global__ void k_contract_bf16(const float* __restrict__ in, size_t n,
                                uint16_t* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t b = f32_to_bits(in[i]);
    // round-to-nearest-even to bf16
    const uint32_t lsb  = (b >> 16) & 1u;
    const uint32_t bias = 0x7fffu + lsb;
    b += bias;
    out[i] = (uint16_t)(b >> 16);
}

__global__ void k_contract_fp16(const float* __restrict__ in, size_t n,
                                uint16_t* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half(in[i]);
    __builtin_memcpy(&out[i], &h, sizeof(__half));
}

// one block per group, 256 threads, shared-mem tree reduction in fp64
__global__ void k_group_rms(const float* __restrict__ x, size_t n, uint32_t group,
                            float* __restrict__ out_rms, uint32_t n_groups) {
    const uint32_t g = blockIdx.x;
    if (g >= n_groups) return;

    const size_t base = (size_t)g * group;
    const size_t end  = (base + group < n) ? (base + group) : n;

    double acc = 0.0;
    for (size_t i = base + threadIdx.x; i < end; i += blockDim.x) {
        const double v = (double)x[i];
        acc += v * v;
    }

    __shared__ double sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t cnt = end - base;
        const double ms  = (cnt > 0) ? sh[0] / (double)cnt : 0.0;
        out_rms[g] = (float)sqrt(ms);
    }
}

__global__ void k_quantize_pergroup(const float* __restrict__ x, size_t n, uint32_t group,
                                    const float* __restrict__ step,
                                    int16_t* __restrict__ q,
                                    uint32_t* __restrict__ over_idx,
                                    float* __restrict__ over_val,
                                    uint32_t over_cap,
                                    uint32_t* __restrict__ over_count,
                                    int32_t max_index) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float s  = step[(size_t)(i / group)];
    const float qf = rintf(x[i] / s);
    const int32_t qi = (int32_t)qf;

    if (qi > max_index || qi < -max_index) {
        const uint32_t slot = atomicAdd(over_count, 1u);
        if (slot < over_cap) { over_idx[slot] = (uint32_t)i; over_val[slot] = x[i]; }
        q[i] = 0;
    } else {
        q[i] = (int16_t)qi;
    }
}

// Group index for element i.  `i / group` is a 64-bit integer division, which
// AMD has no instruction for -- the compiler emits a ~70-instruction software
// sequence, once per ELEMENT.  Every group size the codec actually uses (256,
// 4096, or the whole tensor) is a power of two, so the host passes the shift
// and this becomes one instruction; gshift > 31 is the sentinel for the general
// case, which still divides.
__device__ __forceinline__ size_t qf_gidx(size_t i, uint32_t group, uint32_t gshift) {
    return (gshift <= 31u) ? (i >> gshift) : (i / group);
}

// The quantize math, shared by the three kernels that need it.  `capture`
// is false on a re-quantize pass, so the overflow list is filled exactly once.
__device__ __forceinline__ uint16_t qf_code(float v, float s, int32_t max_index,
                                            int log_domain, float log_floor,
                                            size_t i, uint32_t* __restrict__ over_count,
                                            uint32_t* __restrict__ over_idx,
                                            float* __restrict__ over_val,
                                            uint32_t over_cap, bool capture) {
    if (log_domain) { const float vv = v > log_floor ? v : log_floor; v = logf(vv); }
    const int32_t qi = (int32_t)rintf(v / s);
    int16_t q;
    if (qi > max_index || qi < -max_index) {
        if (capture) {
            const uint32_t slot = atomicAdd(over_count, 1u);
            // store the TRANSFORMED value, matching the unfused path
            if (slot < over_cap) { over_idx[slot] = (uint32_t)i; over_val[slot] = v; }
        }
        q = 0;
    } else {
        q = (int16_t)qi;
    }
    const int32_t vq = (int32_t)q;
    return (uint16_t)((vq << 1) ^ (vq >> 15));
}

// OR-reduce across the wavefront, which holds exactly one bit-pack block's
// values (highest set bit of the OR == that of the max, so OR is a valid and
// cheaper reduction for a bit width), then turn it into a width.
__device__ __forceinline__ uint32_t qf_wave_bits(uint32_t acc) {
    for (uint32_t s = WAVE >> 1; s > 0; s >>= 1)
        acc |= (uint32_t)__shfl_xor((int)acc, (int)s, (int)WAVE);
    return acc ? (32u - (uint32_t)__clz((int)acc)) : 0u;
}

// float4 load with a bounds-checked tail.
__device__ __forceinline__ float4 qf_ld4b(const float* __restrict__ x, size_t i, size_t n) {
    if (i + 4 <= n) return *reinterpret_cast<const float4*>(x + i);
    float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i + 0 < n) r.x = x[i + 0];
    if (i + 1 < n) r.y = x[i + 1];
    if (i + 2 < n) r.z = x[i + 2];
    if (i + 3 < n) r.w = x[i + 3];
    return r;
}

// Fused: [log ->] quantize -> zigzag, one pass over the array, codes written
// to memory.  This is the ENTROPY=0 (CPU zstd) path: zstd needs the code array
// on the host, so it has to exist.  The GPU bit-pack path does NOT use this --
// k_quantize_widths / k_quantize_pack below re-derive the codes in registers so
// the array is never written at all.
__global__ void k_quantize_fused(const float* __restrict__ x, size_t n, uint32_t group,
                                 const float* __restrict__ step,
                                 uint16_t* __restrict__ zz,
                                 uint32_t* __restrict__ over_idx,
                                 float* __restrict__ over_val,
                                 uint32_t over_cap,
                                 uint32_t* __restrict__ over_count,
                                 int32_t max_index,
                                 int log_domain, float log_floor,
                                 uint8_t* __restrict__ w, uint32_t n_blocks,
                                 uint32_t gshift, int uniform_group) {
    const uint32_t wv = threadIdx.x / WAVE;          // wavefront == bit-pack block
    const uint32_t l  = threadIdx.x & (WAVE - 1);
    const size_t   i0 = (size_t)blockIdx.x * C_VALS + (size_t)threadIdx.x * VP_C;

    const float4 f = qf_ld4b(x, i0, n);
    const float  vv[4] = {f.x, f.y, f.z, f.w};
    const float  s_u = uniform_group ? step[qf_gidx(i0, group, gshift)] : 0.0f;

    uint32_t acc = 0;
    uint16_t o[VP_C];
    #pragma unroll
    for (uint32_t m = 0; m < VP_C; ++m) {
        const size_t i = i0 + m;
        const float  s = uniform_group ? s_u : step[qf_gidx(i, group, gshift)];
        o[m] = (i < n) ? qf_code(vv[m], s, max_index, log_domain, log_floor, i,
                                 over_count, over_idx, over_val, over_cap, true)
                       : (uint16_t)0;
        acc |= o[m];
    }
    // 2*VP_C bytes, 8-byte aligned (i0 % 4 == 0): one wide store per lane.
    if (i0 + VP_C <= n) {
        *reinterpret_cast<uint64_t*>(zz + i0) =
            (uint64_t)o[0] | ((uint64_t)o[1] << 16)
          | ((uint64_t)o[2] << 32) | ((uint64_t)o[3] << 48);
    } else {
        #pragma unroll
        for (uint32_t m = 0; m < VP_C; ++m) if (i0 + m < n) zz[i0 + m] = o[m];
    }

    if (!w) return;
    const uint32_t bits = qf_wave_bits(acc);
    const uint32_t b    = blockIdx.x * C_BLKS + wv;
    if (l == 0 && b < n_blocks) w[b] = (uint8_t)bits;
}

__global__ void k_bitpack_widths(const uint16_t* __restrict__ zz, size_t n,
                                 uint8_t* __restrict__ w, uint32_t n_blocks) {
    const uint32_t b = blockIdx.x;
    if (b >= n_blocks) return;
    const size_t base = (size_t)b * BITPACK_BLOCK;
    const size_t end  = (base + BITPACK_BLOCK < n) ? base + BITPACK_BLOCK : n;

    uint32_t mx = 0;
    for (size_t i = base + threadIdx.x; i < end; i += blockDim.x)
        mx = max(mx, (uint32_t)zz[i]);

    __shared__ uint32_t sh[BITPACK_BLOCK];
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const uint32_t m = sh[0];
        w[b] = (uint8_t)(m ? (32u - (uint32_t)__clz((int)m)) : 0u);
    }
}

// Pack block b's values at byte offset off[b], w[b] bits each.
//
// One WAVEFRONT per block.  The original ran one 256-thread workgroup per 256
// VALUES -- 4 M workgroups per 512 MB shard, each reading 512 bytes -- and
// inside it every thread did a shared-memory atomicOr to deposit its field and
// then wrote the result back one BYTE at a time; that measured 10 GB/s against
// the 122 GB/s group_rms sustains on the same data.  Now each lane stages VP_C
// codes with one 8-byte load, then the 64 lanes stride over the block's <=128
// output words, each owning whole words it assembles from shared and stores as
// one aligned dword.  off[b] is always a multiple of 32 bytes (a block is
// exactly 32*bits), so nothing is a read-modify-write.
__global__ void k_bitpack_encode(const uint16_t* __restrict__ zz, size_t n,
                                 const uint8_t* __restrict__ w,
                                 const uint64_t* __restrict__ off,
                                 uint8_t* __restrict__ out, uint32_t n_blocks) {
    __shared__ uint16_t sv[C_VALS];

    const uint32_t wv = threadIdx.x / WAVE;
    const uint32_t l  = threadIdx.x & (WAVE - 1);
    const uint32_t b  = blockIdx.x * C_BLKS + wv;
    const size_t   i0 = (size_t)blockIdx.x * C_VALS + (size_t)threadIdx.x * VP_C;
    uint16_t* const dst = sv + threadIdx.x * VP_C;

    if (i0 + VP_C <= n) {
        *reinterpret_cast<uint64_t*>(dst) =
            *reinterpret_cast<const uint64_t*>(zz + i0);
    } else {
        #pragma unroll
        for (uint32_t m = 0; m < VP_C; ++m)
            dst[m] = (i0 + m < n) ? zz[i0 + m] : (uint16_t)0;
    }
    __syncthreads();        // one barrier per C_VALS values

    if (b >= n_blocks) return;
    const uint32_t bits = w[b];
    if (bits == 0) return;                    // block is all zeros: nothing stored

    const uint32_t  words = (BITPACK_BLOCK * bits) >> 5;    // exact: 8*bits, <=128
    const uint32_t  mask  = (1u << bits) - 1u;              // bits <= 16 here
    const uint16_t* src   = sv + wv * BITPACK_BLOCK;
    uint32_t* const o     = reinterpret_cast<uint32_t*>(out + off[b]);
    for (uint32_t u = l; u < words; u += WAVE) {
        const uint32_t bit0 = u << 5;                       // word's first bit
        uint32_t acc = 0;
        for (uint32_t vi = bit0 / bits;
             vi < BITPACK_BLOCK && vi * bits < bit0 + 32u; ++vi) {
            const uint32_t val = (uint32_t)src[vi] & mask;
            const int32_t  sh  = (int32_t)(vi * bits) - (int32_t)bit0;
            acc |= (sh >= 0) ? (val << sh) : (val >> (uint32_t)(-sh));
        }
        o[u] = acc;
    }
}

__global__ void k_bitpack_decode(const uint8_t* __restrict__ in,
                                 const uint8_t* __restrict__ w,
                                 const uint64_t* __restrict__ off,
                                 size_t n, uint16_t* __restrict__ zz,
                                 uint32_t n_blocks) {
    const uint32_t b = blockIdx.x;
    if (b >= n_blocks) return;
    const uint32_t bits = w[b];
    const size_t   base = (size_t)b * BITPACK_BLOCK;
    const size_t   idx  = base + threadIdx.x;

    if (bits == 0) { if (idx < n) zz[idx] = 0; return; }

    __shared__ uint32_t sw[BITPACK_BLOCK / 2];
    const uint32_t words  = (BITPACK_BLOCK * bits + 31u) / 32u;
    const uint8_t* src    = in + off[b];
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
        const uint32_t j = i * 4u;
        sw[i] = (uint32_t)src[j] | ((uint32_t)src[j + 1] << 8)
              | ((uint32_t)src[j + 2] << 16) | ((uint32_t)src[j + 3] << 24);
    }
    __syncthreads();

    if (idx < n) {
        const uint32_t p    = threadIdx.x * bits;
        const uint32_t widx = p >> 5, o = p & 31u;
        uint32_t v = sw[widx] >> o;
        if (o + bits > 32u) v |= sw[widx + 1] << (32u - o);
        zz[idx] = (uint16_t)(v & ((1u << bits) - 1u));
    }
}

// Fused inverse: unpack -> unzigzag -> dequantize -> [exp], one pass.
// Replaces bitpack_decode + unzigzag_inplace + dequant_pergroup (+ exp_fp32),
// which materialized the u16 code array and then re-read it twice.
// Fused inverse: unpack -> unzigzag -> dequantize -> [exp] in ONE pass.
//
// The unfused sequence was bitpack_decode -> unzigzag_inplace ->
// dequant_pergroup -> exp_fp32: four kernels that materialized the u16 code
// array and then re-read it twice, ~12 GB of traffic against this kernel's
// ~4.8 GB.  LPB_D lanes own one block and each emits VP_D floats as coalesced
// 16-byte stores; that geometry measured 15.3 ms per 0.5 GiB chunk against
// 19-20 ms for the wavefront-per-block variants.
__global__ void k_bitpack_dequant(const uint8_t* __restrict__ in,
                                  const uint8_t* __restrict__ w,
                                  const uint64_t* __restrict__ off,
                                  size_t n, uint32_t group,
                                  const float* __restrict__ step,
                                  float* __restrict__ out, uint32_t n_blocks,
                                  int log_domain, uint32_t gshift, int uniform_group) {
    __shared__ uint32_t sw[D_BLKS][BITPACK_BLOCK / 2];   // 8*bits <= 128 words

    const uint32_t t   = threadIdx.x;
    const uint32_t sub = t / LPB_D;              // which block in this workgroup
    const uint32_t sl  = t % LPB_D;              // lane within that block
    const uint32_t b   = blockIdx.x * D_BLKS + sub;
    const uint32_t bits = (b < n_blocks) ? w[b] : 0u;

    // Staged with respect to the barrier: a lane with no work must still reach
    // it, so the guard is on the loads, not a return.
    if (b < n_blocks && bits) {
        const uint32_t words = (BITPACK_BLOCK * bits) >> 5;
        // off[b] is a multiple of 32 bytes (a block is exactly 32*bits), so the
        // packed stream is 4-byte aligned and can be read as words.
        const uint32_t* src = reinterpret_cast<const uint32_t*>(in + off[b]);
        for (uint32_t u = sl; u < words; u += LPB_D) sw[sub][u] = src[u];
    }
    __syncthreads();
    if (b >= n_blocks) return;

    const size_t   i0   = (size_t)blockIdx.x * D_VALS + (size_t)t * VP_D;
    const uint32_t v0   = sl * VP_D;             // value index within the block
    const uint32_t mask = bits ? ((1u << bits) - 1u) : 0u;
    const float    s_u  = uniform_group ? step[qf_gidx(i0, group, gshift)] : 0.0f;

    float o[VP_D];
    #pragma unroll
    for (uint32_t k = 0; k < VP_D; ++k) {
        uint32_t u = 0;
        if (bits) {
            const uint32_t p = (v0 + k) * bits, widx = p >> 5, sh = p & 31u;
            u = sw[sub][widx] >> sh;
            if (sh + bits > 32u) u |= sw[sub][widx + 1] << (32u - sh);
            u &= mask;
        }
        const int16_t q = (int16_t)((u >> 1) ^ (~(u & 1u) + 1u));  // (u>>1) ^ -(u&1)
        const size_t  i = i0 + k;
        const float   s = uniform_group ? s_u
                        : step[qf_gidx(i < n ? i : n - 1, group, gshift)];
        const float   y = (float)q * s;
        o[k] = log_domain ? expf(y) : y;   // expf, not __expf: must match Restart
    }
    if (i0 + VP_D <= n) {
        #pragma unroll
        for (uint32_t k = 0; k < VP_D; k += 4)
            *reinterpret_cast<float4*>(out + i0 + k) =
                make_float4(o[k], o[k + 1], o[k + 2], o[k + 3]);
    } else {
        #pragma unroll
        for (uint32_t k = 0; k < VP_D; ++k)
            if (i0 + k < n) out[i0 + k] = o[k];
    }
}

__global__ void k_fill_f32(float* __restrict__ p, size_t n, float v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// Per-group step codes, mirroring PgQuant.hpp::quantize_steps exactly:
//   D_g   = 2*eps*max(rms, floor_r)
//   code  = clamp(floor((log(D_g) - lo) / step_q), 0, 255)
//   D_use = exp(lo + code*step_q) * (1 - 1e-6)
__global__ void k_pg_codes(const float* __restrict__ rms, uint32_t ng,
                           double eps, double floor_r, double lo, double step_q,
                           uint8_t* __restrict__ dcodes, float* __restrict__ d_use) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng) return;
    double r = (double)rms[g];
    if (r < floor_r) r = floor_r;
    const double lD = log(2.0 * eps * r);
    double code = floor((lD - lo) / step_q);
    if (code < 0.0)   code = 0.0;
    if (code > 255.0) code = 255.0;
    dcodes[g] = (uint8_t)code;
    d_use[g]  = (float)(exp(lo + code * step_q) * (1.0 - 1e-6));
}

__global__ void k_zigzag(int16_t* __restrict__ q, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t v = (int32_t)q[i];
    const uint16_t zz = (uint16_t)((v << 1) ^ (v >> 15));
    __builtin_memcpy(&q[i], &zz, sizeof(uint16_t));
}

__global__ void k_unzigzag(uint16_t* __restrict__ zz, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = (uint32_t)zz[i];
    const int16_t v = (int16_t)((u >> 1) ^ (~(u & 1u) + 1u)); // (u>>1) ^ -(u&1)
    __builtin_memcpy(&zz[i], &v, sizeof(int16_t));
}

__global__ void k_dequant_pergroup(const int16_t* __restrict__ q, size_t n, uint32_t group,
                                   const float* __restrict__ step, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (float)q[i] * step[(size_t)(i / group)];
}

__global__ void k_scatter_overflow(float* __restrict__ out,
                                   const uint32_t* __restrict__ idx,
                                   const float* __restrict__ val,
                                   uint32_t n_over) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_over) return;
    out[idx[k]] = val[k];
}

__global__ void k_log_clamp(const float* __restrict__ v, size_t n, float floor,
                            float* __restrict__ y) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float vv = v[i] > floor ? v[i] : floor;
    y[i] = logf(vv);
}

__global__ void k_exp(const float* __restrict__ y, size_t n, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = expf(y[i]);
}

__global__ void k_clamp_floor(float* __restrict__ x, size_t n, float floor) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (x[i] < floor) x[i] = floor;
}

// ================================================================= //
//  host wrappers
// ================================================================= //
namespace {
inline uint32_t grid_for(size_t n, uint32_t block) {
    return (uint32_t)((n + block - 1) / block);
}
} // namespace

void expand_to_fp32(const void* d_in, DataType dt, size_t n,
                    float* d_out, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    switch (dt) {
        case DataType::Float32:
            (void)gpu::MemcpyAsync(d_out, d_in, n * sizeof(float), gpu::Copy::D2D, stream);
            break;
        case DataType::BFloat16:
            k_expand_bf16<<<grid, block, 0, stream>>>(
                static_cast<const uint16_t*>(d_in), n, d_out);
            break;
        case DataType::Float16:
            k_expand_fp16<<<grid, block, 0, stream>>>(
                static_cast<const uint16_t*>(d_in), n, d_out);
            break;
        default:
            throw std::runtime_error("[codec] expand_to_fp32: unsupported dtype");
    }
}

void contract_from_fp32(const float* d_in, DataType dt, size_t n,
                        void* d_out, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    switch (dt) {
        case DataType::Float32:
            (void)gpu::MemcpyAsync(d_out, d_in, n * sizeof(float), gpu::Copy::D2D, stream);
            break;
        case DataType::BFloat16:
            k_contract_bf16<<<grid, block, 0, stream>>>(
                d_in, n, static_cast<uint16_t*>(d_out));
            break;
        case DataType::Float16:
            k_contract_fp16<<<grid, block, 0, stream>>>(
                d_in, n, static_cast<uint16_t*>(d_out));
            break;
        default:
            throw std::runtime_error("[codec] contract_from_fp32: unsupported dtype");
    }
}

void group_rms_fp32(const float* d_in, size_t n, uint32_t group,
                    float* d_rms, uint32_t n_groups, gpu::Stream_t stream) {
    if (n_groups == 0) return;
    k_group_rms<<<n_groups, 256, 0, stream>>>(d_in, n, group, d_rms, n_groups);
}

void quantize_pergroup(const float* d_in, size_t n, uint32_t group,
                       const float* d_step, int16_t* d_q,
                       uint32_t* d_over_idx, float* d_over_val,
                       uint32_t over_cap, uint32_t* d_over_count,
                       gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_quantize_pergroup<<<grid, block, 0, stream>>>(
        d_in, n, group, d_step, d_q, d_over_idx, d_over_val, over_cap,
        d_over_count, QUANT_MAX_INDEX);
}

// Shift for i/group when group is a power of two, else a >31 sentinel meaning
// "divide".  uniform_group says a lane's consecutive values always land in the
// same group, so the step is looked up once per lane instead of per value.
static inline uint32_t group_shift(uint32_t group) {
    return (group && (group & (group - 1)) == 0) ? (uint32_t)__builtin_ctz(group) : 0xFFu;
}

void quantize_fused(const float* d_in, size_t n, uint32_t group,
                    const float* d_step, uint16_t* d_zz,
                    uint32_t* d_over_idx, float* d_over_val,
                    uint32_t over_cap, uint32_t* d_over_count,
                    int log_domain, float log_floor, gpu::Stream_t stream,
                    uint8_t* d_w) {
    if (n == 0) return;
    const uint32_t n_blocks = (uint32_t)((n + BITPACK_BLOCK - 1) / BITPACK_BLOCK);
    const uint32_t grid = (uint32_t)((n + C_VALS - 1) / C_VALS);
    k_quantize_fused<<<grid, BP_WG, 0, stream>>>(
        d_in, n, group, d_step, d_zz, d_over_idx, d_over_val, over_cap,
        d_over_count, QUANT_MAX_INDEX, log_domain, log_floor, d_w, n_blocks,
        group_shift(group), (group % VP_C == 0) ? 1 : 0);
}

void bitpack_widths(const uint16_t* d_zz, size_t n, uint8_t* d_w,
                    uint32_t n_blocks, gpu::Stream_t stream) {
    if (n_blocks == 0) return;
    k_bitpack_widths<<<n_blocks, BITPACK_BLOCK, 0, stream>>>(d_zz, n, d_w, n_blocks);
}

void bitpack_encode(const uint16_t* d_zz, size_t n, const uint8_t* d_w,
                    const uint64_t* d_off, uint8_t* d_out,
                    uint32_t n_blocks, gpu::Stream_t stream) {
    if (n_blocks == 0) return;
    const uint32_t wgs = (n_blocks + C_BLKS - 1) / C_BLKS;
    k_bitpack_encode<<<wgs, BP_WG, 0, stream>>>(d_zz, n, d_w, d_off, d_out, n_blocks);
}

void bitpack_dequant(const uint8_t* d_in, const uint8_t* d_w,
                     const uint64_t* d_off, size_t n, uint32_t group,
                     const float* d_step, float* d_out, uint32_t n_blocks,
                     int log_domain, gpu::Stream_t stream) {
    if (n_blocks == 0) return;
    const uint32_t wgs = (n_blocks + D_BLKS - 1) / D_BLKS;
    k_bitpack_dequant<<<wgs, BP_WG, 0, stream>>>(
        d_in, d_w, d_off, n, group, d_step, d_out, n_blocks, log_domain,
        group_shift(group), (group % VP_D == 0) ? 1 : 0);
}

void bitpack_decode(const uint8_t* d_in, const uint8_t* d_w,
                    const uint64_t* d_off, size_t n, uint16_t* d_zz,
                    uint32_t n_blocks, gpu::Stream_t stream) {
    if (n_blocks == 0) return;
    k_bitpack_decode<<<n_blocks, BITPACK_BLOCK, 0, stream>>>(
        d_in, d_w, d_off, n, d_zz, n_blocks);
}

void fill_f32(float* d_p, size_t n, float v, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_fill_f32<<<grid, block, 0, stream>>>(d_p, n, v);
}

void pg_steps_device(const float* d_rms, uint32_t ng, double eps,
                     uint8_t* d_dcodes, float* d_use,
                     double* out_pg_lo, double* out_pg_step,
                     gpu::Stream_t stream) {
    if (ng == 0) { *out_pg_lo = 0.0; *out_pg_step = 0.0; return; }

    // pass 1: amax over the per-group RMS (ng is small -- n/group)
    RAJA::ReduceMax<CodecReduce, double> amx(0.0);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, ng),
        [=] RAJA_HOST_DEVICE (size_t i) { amx.max((double)d_rms[i]); });
    const double amax    = amx.get();
    const double floor_r = (amax * 1e-12 > (double)FLT_MIN) ? amax * 1e-12
                                                            : (double)FLT_MIN;

    // pass 2: min/max of log(D_g).  Recomputing the log here costs less than
    // materialising an ng-sized array and reading it back.
    RAJA::ReduceMin<CodecReduce, double> mn(1e300);
    RAJA::ReduceMax<CodecReduce, double> mx(-1e300);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, ng),
        [=] RAJA_HOST_DEVICE (size_t i) {
            double r = (double)d_rms[i];
            if (r < floor_r) r = floor_r;
            const double l = log(2.0 * eps * r);
            mn.min(l);
            mx.max(l);
        });
    const double lo     = mn.get();
    const double hi     = mx.get();
    const double span   = (hi - lo) > 1e-30 ? (hi - lo) : 1e-30;
    const double step_q = span / 255.0;             // (1 << 8) - 1 levels

    // pass 3: codes + decoder-exact steps, written straight to device memory
    const uint32_t block = 256, grid = grid_for(ng, block);
    k_pg_codes<<<grid, block, 0, stream>>>(d_rms, ng, eps, floor_r, lo, step_q,
                                           d_dcodes, d_use);
    *out_pg_lo   = lo;
    *out_pg_step = step_q;
}

void zigzag_inplace(int16_t* d_q, size_t n, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_zigzag<<<grid, block, 0, stream>>>(d_q, n);
}

void unzigzag_inplace(uint16_t* d_zz, size_t n, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_unzigzag<<<grid, block, 0, stream>>>(d_zz, n);
}

void dequant_pergroup(const int16_t* d_q, size_t n, uint32_t group,
                      const float* d_step, float* d_out, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_dequant_pergroup<<<grid, block, 0, stream>>>(d_q, n, group, d_step, d_out);
}

void scatter_overflow(float* d_out, const uint32_t* d_idx, const float* d_val,
                      uint32_t n_over, gpu::Stream_t stream) {
    if (n_over == 0) return;
    const uint32_t block = 256, grid = grid_for(n_over, block);
    k_scatter_overflow<<<grid, block, 0, stream>>>(d_out, d_idx, d_val, n_over);
}

void log_clamp_fp32(const float* d_v, size_t n, float floor,
                    float* d_y, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_log_clamp<<<grid, block, 0, stream>>>(d_v, n, floor, d_y);
}

void exp_fp32(const float* d_y, size_t n, float* d_out, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_exp<<<grid, block, 0, stream>>>(d_y, n, d_out);
}

void clamp_floor_fp32(float* d_x, size_t n, float floor, gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_clamp_floor<<<grid, block, 0, stream>>>(d_x, n, floor);
}

void reduce_quant_error(const int16_t* d_q, const float* d_step, const float* d_ref,
                        size_t n, uint32_t group,
                        double* out_sq_err, double* out_signal,
                        gpu::Stream_t /*stream*/) {
    RAJA::ReduceSum<CodecReduce, double> se(0.0), sg(0.0);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, n),
        [=] RAJA_HOST_DEVICE (size_t i) {
            const double r = (double)d_ref[i];
            const double t = (double)d_q[i] * (double)d_step[(size_t)(i / group)];
            const double e = t - r;
            se += e * e;
            sg += r * r;
        });
    *out_sq_err = se.get();
    *out_signal = sg.get();
}

void reduce_sq_error_fp32(const float* d_ref, const float* d_test, size_t n,
                          double* out_sq_err, double* out_signal,
                          gpu::Stream_t /*stream*/) {
    RAJA::ReduceSum<CodecReduce, double> se(0.0), sg(0.0);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, n),
        [=] RAJA_HOST_DEVICE (size_t i) {
            const double r = (double)d_ref[i];
            const double t = (double)d_test[i];
            const double e = t - r;
            se += e * e;
            sg += r * r;
        });
    *out_sq_err = se.get();
    *out_signal = sg.get();
}

void reduce_max_abs_err(const float* d_ref, const float* d_test, size_t n,
                        double* out_max_abs_err, gpu::Stream_t /*stream*/) {
    RAJA::ReduceMax<CodecReduce, double> mx(0.0);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, n),
        [=] RAJA_HOST_DEVICE (size_t i) {
            const double e = (double)d_test[i] - (double)d_ref[i];
            mx.max(e < 0.0 ? -e : e);
        });
    *out_max_abs_err = mx.get();
}

void reduce_max_rel_err(const float* d_ref, const float* d_test, size_t n,
                        double* out_max_rel_err, gpu::Stream_t /*stream*/) {
    RAJA::ReduceMax<CodecReduce, double> mx(0.0);
    RAJA::forall<CodecExec>(RAJA::TypedRangeSegment<size_t>(0, n),
        [=] RAJA_HOST_DEVICE (size_t i) {
            const double r = (double)d_ref[i];
            double e = (double)d_test[i] - r;
            if (e < 0.0) e = -e;
            const double denom = (r < 0.0 ? -r : r);
            mx.max(e / (denom > 1e-30 ? denom : 1e-30));
        });
    *out_max_rel_err = mx.get();
}

// RAJA's ReduceMin/ReduceMax on HIP took 990 ms to scan 128M floats (~0.5 GB/s)
// and were 50% of the exp_avg compress time.  This is a plain two-stage block
// reduction: grid-stride pass into shared memory, one float per block out, then
// a trivial host fold over MINMAX_BLOCKS values.  min/max is exact, so the
// result is bit-identical to the RAJA version -- this is purely a speed change.
static constexpr uint32_t MINMAX_BLOCKS  = 1024;
static constexpr uint32_t MINMAX_THREADS = 256;

__global__ void k_minmax_partial(const float* __restrict__ x, size_t n,
                                 float* __restrict__ omin, float* __restrict__ omax) {
    __shared__ float smin[MINMAX_THREADS];
    __shared__ float smax[MINMAX_THREADS];
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float mn =  3.402823466e+38f;
    float mx = -3.402823466e+38f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = x[i];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    smin[threadIdx.x] = mn;
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + s]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { omin[blockIdx.x] = smin[0]; omax[blockIdx.x] = smax[0]; }
}

void reduce_minmax_fp32(const float* d_x, size_t n,
                        double* out_min, double* out_max, gpu::Stream_t stream) {
    if (n == 0) { *out_min = 0.0; *out_max = 0.0; return; }

    // Scratch is tiny (2 * 1024 floats) and reused across calls.
    static float* d_pmin = nullptr;
    static float* d_pmax = nullptr;
    if (d_pmin == nullptr) {
        void* a = nullptr; void* b = nullptr;
        CK_GPU_CHECK(gpu::Malloc(&a, MINMAX_BLOCKS * sizeof(float)));
        CK_GPU_CHECK(gpu::Malloc(&b, MINMAX_BLOCKS * sizeof(float)));
        d_pmin = static_cast<float*>(a);
        d_pmax = static_cast<float*>(b);
    }

    const uint32_t blocks = (uint32_t)std::min<size_t>(
        MINMAX_BLOCKS, (n + MINMAX_THREADS - 1) / MINMAX_THREADS);
    k_minmax_partial<<<blocks, MINMAX_THREADS, 0, stream>>>(d_x, n, d_pmin, d_pmax);

    float h_min[MINMAX_BLOCKS], h_max[MINMAX_BLOCKS];
    CK_GPU_CHECK(gpu::Memcpy(h_min, d_pmin, blocks * sizeof(float), gpu::Copy::D2H));
    CK_GPU_CHECK(gpu::Memcpy(h_max, d_pmax, blocks * sizeof(float), gpu::Copy::D2H));

    float mn = h_min[0], mx = h_max[0];
    for (uint32_t i = 1; i < blocks; ++i) {
        if (h_min[i] < mn) mn = h_min[i];
        if (h_max[i] > mx) mx = h_max[i];
    }
    *out_min = (double)mn;
    *out_max = (double)mx;
}

__global__ void k_sub_fp32(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] - b[i];
}
__global__ void k_add_fp32(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

void sub_fp32(const float* d_a, const float* d_b, float* d_out, size_t n,
              gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_sub_fp32<<<grid, block, 0, stream>>>(d_a, d_b, d_out, n);
}
void add_fp32(const float* d_a, const float* d_b, float* d_out, size_t n,
              gpu::Stream_t stream) {
    if (n == 0) return;
    const uint32_t block = 256, grid = grid_for(n, block);
    k_add_fp32<<<grid, block, 0, stream>>>(d_a, d_b, d_out, n);
}

} // namespace codec
} // namespace checkers
