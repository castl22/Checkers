#pragma once
// GPU codec kernels (raw HIP/CUDA __global__ launches behind thin host
// wrappers) shared by the checkpoint compressor and the restart path.
//
// Phase B  master_weights / model_state : per-group RMS -> uniform quant
//   forward : expand_to_fp32 -> group_rms_fp32 -> (host: quantize_steps)
//             -> quantize_pergroup -> zigzag_inplace -> [D2H + zstd]
//   inverse : [zstd + H2D] -> unzigzag_inplace -> dequant_pergroup
//             -> scatter_overflow -> contract_from_fp32 -> [H2D into live tensor]
//
// Every wrapper runs on the caller-supplied stream and touches only
// caller-owned buffers -- no hidden global state.

#include <cstddef>
#include <cstdint>

#include "checkers/GpuDefs.hpp"

namespace checkers {

// enum class DataType : uint8_t  is defined in checkers/TensorResource.hpp
enum class DataType : uint8_t;

namespace codec {

// Bumped whenever the on-GPU codec math changes in a way that would alter
// reconstructed values (i.e. break round-trip against an older stream file).
int kernels_abi_version();

inline constexpr int32_t QUANT_MAX_INDEX = 32000;   // int16 zigzag headroom

// ---- fp32 <-> storage dtype -------------------------------------------------
// dt == Float32 : plain copy.  Float16 / BFloat16 : widen / narrow.
void expand_to_fp32(const void* d_in, DataType dt, size_t n,
                    float* d_out, gpu::Stream_t stream);
void contract_from_fp32(const float* d_in, DataType dt, size_t n,
                        void* d_out, gpu::Stream_t stream);

// ---- log-domain transform (exp_avg_sq) -----------------------------------
// y = log(max(v, floor)).  The clamp is the divergence-cliff guard: below it
// Adam's own epsilon dominates 1/(sqrt(v)+eps), so the exact value is moot.
void log_clamp_fp32(const float* d_v, size_t n, float floor,
                    float* d_y, gpu::Stream_t stream);
// v = exp(y)   (in place allowed: d_out may alias d_y)
void exp_fp32(const float* d_y, size_t n, float* d_out, gpu::Stream_t stream);

// x[i] = max(x[i], floor)   -- exp_avg_sq divergence-cliff guard (Phase C).
void clamp_floor_fp32(float* d_x, size_t n, float floor, gpu::Stream_t stream);

// ---- forward --------------------------------------------------------------
// Per-group RMS = sqrt(mean(x^2)) in fp64, one block per group.
void group_rms_fp32(const float* d_in, size_t n, uint32_t group,
                    float* d_rms, uint32_t n_groups, gpu::Stream_t stream);

// ---- fused forward --------------------------------------------------------
// One kernel replacing  [log_clamp_fp32 ->] quantize_pergroup -> zigzag_inplace.
// Each of those was a separate full pass over the array; fusing them removes
// two round trips through HBM (n*4 B for the log buffer, 2*n*2 B for zigzag)
// and two kernel launches.  Output is the zigzag u16 payload, ready for D2H.
//
//   log_domain != 0 : v = logf(max(x[i], log_floor))   (exp_avg_sq path)
//   log_domain == 0 : v = x[i]
//   q  = rintf(v / step[i/group]), clamped; |q| >= QUANT_MAX_INDEX goes to the
//        overflow lists (storing v, i.e. the transformed value, matching the
//        unfused path) and q=0 is written.
//   zz = zigzag(q)
// *d_over_count must be zeroed by the caller before launch.
//
// d_w (optional): if non-null, ALSO emit the bit-pack width array -- w[b] =
// bits needed for the largest zigzag code in block b -- reduced across the
// wavefront while the codes are still in registers.  That removes the separate
// bitpack_widths pass, which re-read the whole u16 code array to produce one
// byte per 512 bytes read.
void quantize_fused(const float* d_in, size_t n, uint32_t group,
                    const float* d_step, uint16_t* d_zz,
                    uint32_t* d_over_idx, float* d_over_val,
                    uint32_t over_cap, uint32_t* d_over_count,
                    int log_domain, float log_floor, gpu::Stream_t stream,
                    uint8_t* d_w = nullptr);

// Fill n floats with a constant (log_pw_rel uses one data-independent step).
void fill_f32(float* d_p, size_t n, float v, gpu::Stream_t stream);

// Device-side per-group step quantization -- the GPU twin of the host
// quantize_steps() in PgQuant.hpp.  The host version required a D2H of the
// n_groups RMS values, a serial host loop over every group, and an H2D of the
// resulting steps; at group=256 over a 512 MB tensor that is 524288 groups
// round-tripping across PCIe with two blocking syncs.  This computes dcodes and
// d_use in place and returns only the two scalars the stream header needs.
void pg_steps_device(const float* d_rms, uint32_t n_groups, double eps,
                     uint8_t* d_dcodes, float* d_use,
                     double* out_pg_lo, double* out_pg_step,
                     gpu::Stream_t stream);

// ---- GPU fixed-width bit-packing (alternative to the CPU zstd stage) -------
// Replaces the CPU entropy coder, which measured 64%/95% of Checkers' compress
// /decompress time while cuSZp and lpmm have no CPU stage at all.  Same idea as
// cuSZp's packer: per block of BITPACK_BLOCK values, take the max, derive a bit
// width, write exactly that many bits per value.
//
// BITPACK_BLOCK is 256 so a block is always a whole number of bytes: 256*w/8 =
// 32*w, w in [0,16].  That removes all cross-block bit alignment.
//
// This is LOSSLESS over the zigzag codes -- the unpacked values are bit-identical
// to the packed ones -- so the reconstruction, error bound and SNR are unchanged.
// Only the compressed SIZE differs from zstd (fixed width cannot exploit the
// value distribution the way an entropy coder does).
inline constexpr uint32_t BITPACK_BLOCK = 256;

// w[b] from an already-materialized code array (legacy forward path only).
void bitpack_widths(const uint16_t* d_zz, size_t n, uint8_t* d_w,
                    uint32_t n_blocks, gpu::Stream_t stream);

// Pack block b's values at byte offset d_off[b], w[b] bits each.
void bitpack_encode(const uint16_t* d_zz, size_t n, const uint8_t* d_w,
                    const uint64_t* d_off, uint8_t* d_out,
                    uint32_t n_blocks, gpu::Stream_t stream);

// Inverse.  d_zz must hold n elements.
void bitpack_decode(const uint8_t* d_in, const uint8_t* d_w,
                    const uint64_t* d_off, size_t n, uint16_t* d_zz,
                    uint32_t n_blocks, gpu::Stream_t stream);

// Fused inverse: unpack -> unzigzag -> dequantize -> [exp] in ONE pass.
//
//   out[i] = q[i] * step[i/group]           (log_domain == 0)
//   out[i] = expf(q[i] * step[i/group])     (log_domain != 0, exp_avg_sq)
//
// The unfused sequence was bitpack_decode -> unzigzag_inplace ->
// dequant_pergroup -> exp_fp32: four kernels materializing the u16 code array
// and then re-reading it twice.  On a 512 MB fp32 shard that is ~12 GB of
// traffic against this kernel's ~4.8 GB, and the codes never need to exist in
// memory at all.
void bitpack_dequant(const uint8_t* d_in, const uint8_t* d_w,
                     const uint64_t* d_off, size_t n, uint32_t group,
                     const float* d_step, float* d_out, uint32_t n_blocks,
                     int log_domain, gpu::Stream_t stream);

// q[i] = rintf(x[i] / step[i/group]); |q| >= QUANT_MAX_INDEX is captured into
// (d_over_idx, d_over_val) via an atomic counter and stored as q=0.
// *d_over_count must be zeroed by the caller before launch.
void quantize_pergroup(const float* d_in, size_t n, uint32_t group,
                       const float* d_step,
                       int16_t* d_q,
                       uint32_t* d_over_idx, float* d_over_val,
                       uint32_t over_cap, uint32_t* d_over_count,
                       gpu::Stream_t stream);

// In-place zigzag: reads int16 q[i], writes uint16 over the same address.
void zigzag_inplace(int16_t* d_q, size_t n, gpu::Stream_t stream);

// Quantizer error vs a reference fp32 buffer, fp64 accumulation on the GPU.
// (Ignores overflow-restore, so reported SNR is a lower bound.)
void reduce_quant_error(const int16_t* d_q, const float* d_step, const float* d_ref,
                        size_t n, uint32_t group,
                        double* out_sq_err, double* out_signal,
                        gpu::Stream_t stream);

// Plain fp32/fp32 error (used where there is no quant step).
void reduce_sq_error_fp32(const float* d_ref, const float* d_test, size_t n,
                          double* out_sq_err, double* out_signal,
                          gpu::Stream_t stream);

// max |ref - test| over the tensor (fp64 accumulation).
void reduce_max_abs_err(const float* d_ref, const float* d_test, size_t n,
                        double* out_max_abs_err, gpu::Stream_t stream);

// max |ref - test| / max(|ref|, tiny)  -- pointwise-relative error (PW_REL).
void reduce_max_rel_err(const float* d_ref, const float* d_test, size_t n,
                        double* out_max_rel_err, gpu::Stream_t stream);

// min / max over the tensor (fp64 accumulation) -- EQ REL-mode value range.
void reduce_minmax_fp32(const float* d_x, size_t n,
                        double* out_min, double* out_max, gpu::Stream_t stream);

// ---- elementwise (TT + EQ residual codec) --------------------------------
void sub_fp32(const float* d_a, const float* d_b, float* d_out, size_t n,
              gpu::Stream_t stream);   // out = a - b
void add_fp32(const float* d_a, const float* d_b, float* d_out, size_t n,
              gpu::Stream_t stream);   // out = a + b  (out may alias a or b)

// ---- inverse ------------------------------------------------------------
void unzigzag_inplace(uint16_t* d_zz, size_t n, gpu::Stream_t stream);

void dequant_pergroup(const int16_t* d_q, size_t n, uint32_t group,
                      const float* d_step, float* d_out, gpu::Stream_t stream);

void scatter_overflow(float* d_out, const uint32_t* d_idx, const float* d_val,
                      uint32_t n_over, gpu::Stream_t stream);

} // namespace codec
} // namespace checkers
