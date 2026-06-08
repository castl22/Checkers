// src/ANSCompressor.cpp
//
// GPU-accelerated rANS checkpoint compressor – kernels and host implementation.
//
// Kernel summary
// --------------
//  byte_histogram_kernel   – count byte frequencies in a device buffer (256-thread
//                             multi-block; histogram lives entirely in shared mem).
//  rans_encode_kernel      – rANS encode with ANS_LANES=256 independent interleaved
//                             streams (one per thread).  Each thread encodes its
//                             strided slice of the input bytes in REVERSE order and
//                             writes compressed bytes backward into its private region
//                             of the output buffer (no intra-warp communication needed).
//
// Thread/block layout per tensor
// --------------------------------
//  histogram:  ceil(n_bytes / 65536) blocks × 256 threads
//  encoding:   1 block × 256 threads

#include "checkers/ANSCompressor.hpp"
#include "checkers/MemoryManager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  define GPU_CHECK(call)                                                      \
      do {                                                                      \
          hipError_t _e = (call);                                               \
          if (_e != hipSuccess)                                                 \
              throw std::runtime_error(std::string("[ANS] HIP error: ")        \
                                       + hipGetErrorString(_e));               \
      } while (0)
#  define GPU_STREAM_CREATE(s)  GPU_CHECK(hipStreamCreate(&(s)))
#  define GPU_STREAM_SYNC(s)    GPU_CHECK(hipStreamSynchronize(s))
#  define GPU_STREAM_DESTROY(s) GPU_CHECK(hipStreamDestroy(s))
#  define GPU_MALLOC(p, n)      GPU_CHECK(hipMalloc((void**)&(p), (n)))
#  define GPU_FREE(p)           if (p) { hipFree(p); (p) = nullptr; }
#  define GPU_MEMSET(p,v,n,s)   GPU_CHECK(hipMemsetAsync((p),(v),(n),(s)))
#  define GPU_MEMCPY_H2D(d,h,n,s) GPU_CHECK(hipMemcpyAsync((d),(h),(n),hipMemcpyHostToDevice,(s)))
#  define GPU_MEMCPY_D2H(h,d,n,s) GPU_CHECK(hipMemcpyAsync((h),(d),(n),hipMemcpyDeviceToHost,(s)))
#else
#  define GPU_CHECK(call)                                                      \
      do {                                                                      \
          cudaError_t _e = (call);                                              \
          if (_e != cudaSuccess)                                                \
              throw std::runtime_error(std::string("[ANS] CUDA error: ")       \
                                       + cudaGetErrorString(_e));              \
      } while (0)
#  define GPU_STREAM_CREATE(s)  GPU_CHECK(cudaStreamCreate(&(s)))
#  define GPU_STREAM_SYNC(s)    GPU_CHECK(cudaStreamSynchronize(s))
#  define GPU_STREAM_DESTROY(s) GPU_CHECK(cudaStreamDestroy(s))
#  define GPU_MALLOC(p, n)      GPU_CHECK(cudaMalloc((void**)&(p), (n)))
#  define GPU_FREE(p)           if (p) { cudaFree(p); (p) = nullptr; }
#  define GPU_MEMSET(p,v,n,s)   GPU_CHECK(cudaMemsetAsync((p),(v),(n),(s)))
#  define GPU_MEMCPY_H2D(d,h,n,s) GPU_CHECK(cudaMemcpyAsync((d),(h),(n),cudaMemcpyHostToDevice,(s)))
#  define GPU_MEMCPY_D2H(h,d,n,s) GPU_CHECK(cudaMemcpyAsync((h),(d),(n),cudaMemcpyDeviceToHost,(s)))
#endif

namespace checkers {

namespace {

// ====================================================================
// File-format magic & version
// ====================================================================
static constexpr uint32_t ANSC_MAGIC   = 0x414E5343u; // "ANSC"
static constexpr uint32_t ANSC_VERSION = 1u;

// Maximum elements per histogram-kernel block pass
static constexpr size_t HIST_BLOCK_ELEMS = 65536; // 256 threads × 256 iters

inline auto now_ms() {
    return std::chrono::steady_clock::now();
}
inline double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

} // anon namespace

// ====================================================================
// GPU Kernels
// ====================================================================

// --------------------------------------------------------------------
// byte_histogram_kernel
// --------------------------------------------------------------------
// Counts byte (uint8_t) frequencies in d_data[0..n_bytes).
// Uses 256 shared-memory bins (one per possible byte value) for fast
// intra-block accumulation, then atomically merges into g_hist[256].
// Launch with 256 threads per block; grid size = min(4096, ceil(n/256)).
// g_hist must be zero-initialized before the first block launch.
// --------------------------------------------------------------------
__global__ void byte_histogram_kernel(const uint8_t* __restrict__ d_data,
                                       size_t                       n_bytes,
                                       uint32_t* __restrict__       g_hist)
{
    __shared__ uint32_t s_hist[256];
    s_hist[threadIdx.x] = 0;
    __syncthreads();

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n_bytes) {
        atomicAdd(&s_hist[(uint32_t)d_data[idx]], 1u);
        idx += stride;
    }
    __syncthreads();

    atomicAdd(&g_hist[threadIdx.x], s_hist[threadIdx.x]);
}

// --------------------------------------------------------------------
// rans_encode_kernel
// --------------------------------------------------------------------
// Encodes d_data[0..n_bytes) using ANS_LANES=256 interleaved rANS
// streams, one per thread.
//
// Thread t handles bytes at positions: t, t+ANS_LANES, t+2·ANS_LANES, …
// (strided / interleaved access).  Within its lane, thread t processes
// elements in REVERSE ORDER (required for correct rANS streaming).
//
// Output layout in d_out:
//   thread t's raw output region: d_out[ t*max_chunk .. (t+1)*max_chunk )
//   bytes are written BACKWARD from the END of the region.
//   After the kernel, valid bytes start at:
//     d_out + t*max_chunk + d_streams[t].buf_offset
//   and span d_streams[t].byte_count bytes.
//
// Renormalization:
//   x_max[s] = ANS_NORM_BASE * norm_freq[s]
//   While state >= x_max[s]: emit (state & 0xFF), state >>= 8.
//   Then: state = (state/freq << ANS_SCALE_BITS) + cum + (state%freq).
//   This guarantees state stays in [ANS_L, ANS_L*256) after each step.
//
// Worst-case output: 2 bytes per input byte (for freq=1 symbols).
// d_out is therefore allocated as 2*n_bytes to guarantee no overflow.
// --------------------------------------------------------------------
__global__ void rans_encode_kernel(const uint8_t* __restrict__       d_data,
                                    size_t                             n_bytes,
                                    const uint32_t* __restrict__      d_norm_freq,
                                    const uint32_t* __restrict__      d_cum_freq,
                                    const uint32_t* __restrict__      d_x_max,
                                    uint8_t* __restrict__             d_out,
                                    ANSStreamHeader* __restrict__     d_streams,
                                    size_t                            max_chunk)
{
    const uint32_t tid   = threadIdx.x;   // 0 .. ANS_LANES-1
    const size_t   N     = n_bytes;
    const size_t   LANES = ANS_LANES;

    // Elements for thread tid: t, t+LANES, t+2·LANES, …
    const size_t my_count = (N > (size_t)tid) ? ((N - tid + LANES - 1) / LANES) : 0u;

    // Private output region; write backward from end
    uint8_t* const region_start = d_out + (size_t)tid * max_chunk;
    uint8_t* const region_end   = region_start + max_chunk;
    uint8_t*       wptr         = region_end;

    uint32_t state = ANS_L;

    // Encode in reverse order (rANS requirement)
    for (size_t k = my_count; k-- > 0; ) {
        const size_t   data_idx = (size_t)tid + k * LANES;
        const uint8_t  sym      = d_data[data_idx];
        const uint32_t freq     = d_norm_freq[sym];
        const uint32_t cum      = d_cum_freq[sym];
        const uint32_t xm       = d_x_max[sym];

        if (freq == 0) {
            // Symbol absent from table (shared-table mismatch).
            // Treat as escape: emit raw byte and skip rANS update so
            // the stream remains decodable via the fallback raw byte path.
            // In practice this should not occur when ensure_full_alphabet=true.
            if (wptr > region_start) *--wptr = sym;
            continue;
        }

        // Renormalize: emit bytes until state < x_max
        while (state >= xm) {
            if (wptr <= region_start) { wptr = region_start; break; } // overflow guard
            *--wptr = (uint8_t)(state & 0xFFu);
            state >>= 8;
        }

        // rANS state update
        state = ((state / freq) << ANS_SCALE_BITS) + cum + (state % freq);
    }

    // Write metadata
    const uint32_t cnt = (uint32_t)(region_end - wptr);
    d_streams[tid].final_state = state;
    d_streams[tid].byte_count  = cnt;
    d_streams[tid].buf_offset  = (uint32_t)(max_chunk - cnt); // offset from region start
    d_streams[tid].pad         = 0;
}

// --------------------------------------------------------------------
// compact_streams_kernel
// --------------------------------------------------------------------
// Packs each lane's valid compressed bytes into a contiguous output
// buffer, so only the actual compressed bytes need to be transferred
// from GPU to CPU.  Launch with ANS_LANES blocks × 256 threads.
//
//  blockIdx.x  = lane id  (0 .. ANS_LANES-1)
//  threadIdx.x = byte worker within the lane's region
//  d_offsets[ANS_LANES+1]: inclusive prefix sums of byte_count per lane
//    (d_offsets[0]=0, d_offsets[l+1] = d_offsets[l] + d_hdrs[l].byte_count)
// --------------------------------------------------------------------
__global__ void compact_streams_kernel(const uint8_t* __restrict__         d_enc_buf,
                                        const ANSStreamHeader* __restrict__ d_hdrs,
                                        uint8_t* __restrict__               d_compact,
                                        const uint32_t* __restrict__        d_offsets,
                                        size_t                              max_chunk)
{
    const int lane = (int)blockIdx.x;
    const uint32_t cnt     = d_hdrs[lane].byte_count;
    const uint32_t src_off = d_hdrs[lane].buf_offset;
    const uint8_t* src = d_enc_buf + (size_t)lane * max_chunk + src_off;
    uint8_t*       dst = d_compact + d_offsets[lane];
    for (int i = (int)threadIdx.x; i < (int)cnt; i += (int)blockDim.x)
        dst[i] = src[i];
}

// --------------------------------------------------------------------
// extract_mantissa_kernel
// --------------------------------------------------------------------
// For each element in d_src, copies the whole element to d_masked but
// zeroes the low sidecar_per_elem bytes.  Those low bytes are written
// separately to d_sidecar.
//
// dtype layout (IEEE 754 little-endian, low byte first):
//   BF16 (2B):  [7-mantissa-bits | 0-pad]  [sign + 8-exp-bits]
//               → zero byte[0]; sidecar_per_elem = 1
//   FP16 (2B):  [lower-8-mantissa-bits]    [sign + 5-exp + upper2-mantissa]
//               → zero byte[0]; sidecar_per_elem = 1
//   FP32 (4B):  [low-8-mant] [mid-8-mant] [high-7-mant + low-1-exp] [high-exp + sign]
//               → zero bytes[0,1]; sidecar_per_elem = 2
//
// Reconstruction (lossless): original_elem = masked_elem OR sidecar_elem
// (since the zeroed bytes in masked_elem are exactly what sidecar_elem holds).
// --------------------------------------------------------------------
__global__ void extract_mantissa_kernel(const uint8_t* __restrict__ d_src,
                                         uint8_t* __restrict__       d_masked,
                                         uint8_t* __restrict__       d_sidecar,
                                         size_t                      n_elements,
                                         uint32_t                    elem_stride,
                                         uint32_t                    sidecar_per_elem)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    const uint8_t* src = d_src     + i * elem_stride;
    uint8_t*       msk = d_masked  + i * elem_stride;
    uint8_t*       sc  = d_sidecar + i * sidecar_per_elem;

    // Copy entire element to masked, zeroing the low bytes
    for (uint32_t j = 0; j < elem_stride; j++)
        msk[j] = (j < sidecar_per_elem) ? 0u : src[j];

    // Save the low bytes to sidecar
    for (uint32_t j = 0; j < sidecar_per_elem; j++)
        sc[j] = src[j];
}

// ====================================================================
// ANSCompressor – class implementation
// ====================================================================

ANSCompressor::ANSCompressor(std::shared_ptr<RankLogger> logger,
                             size_t n_worker_threads)
    : logger_(std::move(logger))
    , n_workers_(n_worker_threads == 0 ? 1 : n_worker_threads)
    , rank_(logger_ ? logger_->global_rank() : -1)
{
    workers_.resize(n_workers_);
    for (auto& w : workers_) {
        init_worker_gpu(w);
    }
}

ANSCompressor::~ANSCompressor()
{
    for (auto& w : workers_) {
        free_worker_gpu(w);
    }
    free_shared_bufs();
}

void ANSCompressor::init_worker_gpu(WorkerGPU& w)
{
    // Record the current device so worker threads can re-bind to it.
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    GPU_CHECK(hipGetDevice(&w.device_id));
#else
    GPU_CHECK(cudaGetDevice(&w.device_id));
#endif
    GPU_STREAM_CREATE(w.stream);
    GPU_MALLOC(w.d_hist,      256 * sizeof(uint32_t));
    GPU_MALLOC(w.d_streams,   ANS_LANES * sizeof(ANSStreamHeader));
    // Prefix-sum offsets for stream compaction (ANS_LANES+1 entries)
    GPU_MALLOC(w.d_offsets,   (ANS_LANES + 1) * sizeof(uint32_t));
    GPU_MALLOC(w.d_norm_freq, 256 * sizeof(uint32_t));
    GPU_MALLOC(w.d_cum_freq,  257 * sizeof(uint32_t));
    GPU_MALLOC(w.d_x_max,     256 * sizeof(uint32_t));
    // Large dynamic buffers (enc_buf, compact_buf, masked_buf, sidecar_buf)
    // are shared via shared_bufs_ to avoid N-worker GPU memory amplification.
}

void ANSCompressor::free_worker_gpu(WorkerGPU& w)
{
    if (w.stream) { GPU_STREAM_SYNC(w.stream); GPU_STREAM_DESTROY(w.stream); w.stream = nullptr; }
    GPU_FREE(w.d_hist);
    GPU_FREE(w.d_streams);
    GPU_FREE(w.d_offsets);
    GPU_FREE(w.d_norm_freq);
    GPU_FREE(w.d_cum_freq);
    GPU_FREE(w.d_x_max);
}

void ANSCompressor::free_shared_bufs()
{
    GPU_FREE(shared_bufs_.d_enc_buf);
    shared_bufs_.d_enc_cap = 0;
    GPU_FREE(shared_bufs_.d_compact_buf);
    shared_bufs_.d_compact_cap = 0;
    GPU_FREE(shared_bufs_.d_masked_buf);
    shared_bufs_.d_masked_cap = 0;
    GPU_FREE(shared_bufs_.d_sidecar_buf);
    shared_bufs_.d_sidecar_cap = 0;
}

void ANSCompressor::ensure_enc_buf(SharedGPUBufs& b, size_t required)
{
    if (b.d_enc_cap >= required) return;
    if (logger_) logger_->log_message(lp("dbg") +
        "ensure_enc_buf GPU realloc old=" +
        std::to_string(b.d_enc_cap) + " new=" + std::to_string(required));
    GPU_FREE(b.d_enc_buf);
    GPU_MALLOC(b.d_enc_buf, required);
    b.d_enc_cap = required;
}

void ANSCompressor::ensure_compact_buf(SharedGPUBufs& b, size_t required)
{
    if (b.d_compact_cap >= required) return;
    if (logger_) logger_->log_message(lp("dbg") +
        "ensure_compact_buf GPU realloc old=" +
        std::to_string(b.d_compact_cap) + " new=" + std::to_string(required));
    GPU_FREE(b.d_compact_buf);
    GPU_MALLOC(b.d_compact_buf, required);
    b.d_compact_cap = required;
}

void ANSCompressor::ensure_masked_buf(SharedGPUBufs& b, size_t masked_cap, size_t sidecar_cap)
{
    if (b.d_masked_cap < masked_cap) {
        if (logger_) logger_->log_message(lp("dbg") +
            "ensure_masked_buf GPU realloc masked old=" +
            std::to_string(b.d_masked_cap) + " new=" + std::to_string(masked_cap));
        GPU_FREE(b.d_masked_buf);
        GPU_MALLOC(b.d_masked_buf, masked_cap);
        b.d_masked_cap = masked_cap;
    }
    if (b.d_sidecar_cap < sidecar_cap) {
        if (logger_) logger_->log_message(lp("dbg") +
            "ensure_masked_buf GPU realloc sidecar old=" +
            std::to_string(b.d_sidecar_cap) + " new=" + std::to_string(sidecar_cap));
        GPU_FREE(b.d_sidecar_buf);
        GPU_MALLOC(b.d_sidecar_buf, sidecar_cap);
        b.d_sidecar_cap = sidecar_cap;
    }
}

// --------------------------------------------------------------------
// build_byte_histogram
// --------------------------------------------------------------------
void ANSCompressor::build_byte_histogram(const void* d_data,
                                         size_t      n_bytes,
                                         uint32_t    h_hist_out[256],
                                         gpuStream_t stream)
{
    // Use worker 0's persistent histogram buffer
    auto& w = workers_[0];
    gpuStream_t s = stream ? stream : w.stream;

    GPU_MEMSET(w.d_hist, 0, 256 * sizeof(uint32_t), s);

    const uint32_t nblocks = (uint32_t)std::min<size_t>(
        (n_bytes + 255) / 256, 4096);
    byte_histogram_kernel<<<dim3(nblocks), dim3(256), 0, s>>>(
        static_cast<const uint8_t*>(d_data), n_bytes, w.d_hist);

    GPU_MEMCPY_D2H(h_hist_out, w.d_hist, 256 * sizeof(uint32_t), s);
    GPU_STREAM_SYNC(s);
}

// --------------------------------------------------------------------
// normalize_histogram
// --------------------------------------------------------------------
ANSFreqTable ANSCompressor::normalize_histogram(const uint32_t h_raw_hist[256],
                                                bool ensure_full_alphabet) const
{
    ANSFreqTable table{};

    // Count total bytes and distinct symbols
    uint64_t total = 0;
    int n_distinct = 0;
    for (int i = 0; i < 256; i++) {
        total    += h_raw_hist[i];
        n_distinct += (h_raw_hist[i] > 0) ? 1 : 0;
    }

    // Determine effective distinct count (for full-alphabet mode every
    // symbol needs at least 1 slot, so treat all 256 as "present").
    const int effective_distinct = ensure_full_alphabet ? 256 : n_distinct;
    if (effective_distinct == 0) {
        // Degenerate empty tensor: give all slots to symbol 0
        table.norm_freq[0] = ANS_M;
        table.cum_freq[1]  = ANS_M;
        for (int i = 2; i <= 256; i++) table.cum_freq[i] = ANS_M;
        table.x_max[0] = ANS_NORM_BASE * ANS_M;
        for (uint32_t j = 0; j < ANS_M; j++) {
            table.decode_sym[j] = 0; table.decode_freq[j] = ANS_M; table.decode_cum[j] = 0;
        }
        return table;
    }

    // --- Step 1: proportional allocation ---
    uint32_t norm[256] = {};
    uint32_t assigned  = 0;

    for (int i = 0; i < 256; i++) {
        if (!ensure_full_alphabet && h_raw_hist[i] == 0) continue;
        uint32_t slots = (total > 0)
            ? (uint32_t)((uint64_t)std::max(h_raw_hist[i], 1u) * ANS_M / total)
            : 0u;
        if (slots == 0) slots = 1; // guarantee minimum 1
        norm[i]   = slots;
        assigned += slots;
    }

    // --- Step 2: fix rounding error (must reach ANS_M exactly) ---
    struct SF { int sym; double frac; };
    std::vector<SF> fracs;
    fracs.reserve(256);
    for (int i = 0; i < 256; i++) {
        if (norm[i] == 0) continue;
        double exact = (total > 0)
            ? ((double)std::max(h_raw_hist[i], 1u) * ANS_M / (double)total)
            : 1.0;
        fracs.push_back({i, exact - (double)(int)exact});
    }
    std::sort(fracs.begin(), fracs.end(), [](const SF& a, const SF& b){
        return a.frac > b.frac; // descending fractional part
    });

    if (assigned < ANS_M) {
        uint32_t rem = ANS_M - assigned;
        for (size_t k = 0; k < fracs.size() && rem; k++, rem--)
            norm[fracs[k].sym]++;
    } else if (assigned > ANS_M) {
        uint32_t rem = assigned - ANS_M;
        // Take from symbols with smallest fractional part (keep min=1)
        for (int k = (int)fracs.size() - 1; k >= 0 && rem; k--) {
            if (norm[fracs[k].sym] > 1) { norm[fracs[k].sym]--; rem--; }
        }
    }

    // --- Step 3: populate table fields ---
    for (int i = 0; i < 256; i++) table.norm_freq[i] = norm[i];

    table.cum_freq[0] = 0;
    for (int i = 0; i < 256; i++)
        table.cum_freq[i+1] = table.cum_freq[i] + table.norm_freq[i];

    for (int sym = 0; sym < 256; sym++) {
        if (table.norm_freq[sym] == 0) { table.x_max[sym] = 0; continue; }
        table.x_max[sym] = ANS_NORM_BASE * table.norm_freq[sym];
        for (uint32_t j = table.cum_freq[sym]; j < table.cum_freq[sym+1]; j++) {
            table.decode_sym [j] = (uint8_t)sym;
            table.decode_freq[j] = table.norm_freq[sym];
            table.decode_cum [j] = table.cum_freq[sym];
        }
    }

    return table;
}

// --------------------------------------------------------------------
// compress_tensor  (single tensor, uses worker[wid] + shared bufs b)
// Caller must hold shared_bufs_mtx_ for the duration of this call.
// --------------------------------------------------------------------
CompressedTensor ANSCompressor::compress_tensor(const std::string&  name,
                                                const void*         d_data,
                                                size_t              byte_size,
                                                DataType            dtype,
                                                size_t              num_elements,
                                                const ANSFreqTable& table,
                                                bool                owns_table,
                                                const std::string&  anchor_name,
                                                WorkerGPU&          w,
                                                SharedGPUBufs&      b)
{
    CompressedTensor ct;
    ct.name             = name;
    ct.dtype            = dtype;
    ct.original_bytes   = byte_size;
    ct.num_elements     = num_elements;
    ct.owns_table       = owns_table;
    ct.anchor_name      = anchor_name;

    if (byte_size == 0) return ct;

    if (logger_) logger_->log_message(lp("dbg") + "compress_tensor enter name=" + name +
        " byte_size=" + std::to_string(byte_size) +
        " n_elems=" + std::to_string(num_elements));

    // ---- Drain any deferred GPU errors from previous operations ----
    // ROCm/HIP can defer GPU page-fault signals; they are delivered to the process
    // not when the bad kernel runs but when the *next* GPU API call is made on that
    // device, which would manifest as SIGSEGV with no C++ exception.  Calling
    // hipGetLastError() here clears and surfaces any pending device error as a
    // catchable HIP error, converting a would-be SIGSEGV into a logged exception.
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    {
        hipError_t pending = hipGetLastError();
        if (pending != hipSuccess) {
            const std::string msg = lp("gpu_fault") +
                "deferred GPU error detected before compress_tensor for '" + name +
                "': " + hipGetErrorString(pending);
            if (logger_) logger_->log_message(msg);
            throw std::runtime_error(msg);
        }
    }
#else
    {
        cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess) {
            const std::string msg = lp("gpu_fault") +
                "deferred GPU error detected before compress_tensor for '" + name +
                "': " + cudaGetErrorString(pending);
            if (logger_) logger_->log_message(msg);
            throw std::runtime_error(msg);
        }
    }
#endif

    // ---- Validate source pointer before any kernel launch ----
    // If the source buffer was freed/moved by PyTorch between discovery and
    // checkpoint, this detects it early as a logged error rather than SIGSEGV.
    if (d_data) {
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
        hipPointerAttribute_t attr{};
        hipError_t ptr_stat = hipPointerGetAttributes(&attr, d_data);
        if (ptr_stat != hipSuccess) {
            const std::string msg = lp("invalid_ptr") +
                "compress_tensor d_data pointer invalid for '" + name +
                "' ptr=" + std::to_string(reinterpret_cast<uintptr_t>(d_data)) +
                " err=" + hipGetErrorString(ptr_stat);
            if (logger_) logger_->log_message(msg);
            // Return empty passthrough rather than crashing
            std::memset(ct.streams, 0, sizeof(ct.streams));
            ct.streams[0].byte_count = (uint32_t)byte_size;
            return ct;
        }
        if (logger_) logger_->log_message(lp("dbg") + "compress_tensor ptr_ok name=" + name +
            " memoryType=" + std::to_string((int)attr.type));
#endif
    }

    gpuStream_t s = w.stream;

    // ---- Allocate encode output buffer (2× input = worst-case 2 bytes/symbol) ----
    const size_t max_chunk   = ((byte_size + ANS_LANES - 1) / ANS_LANES) * 2; // per-lane bytes
    const size_t enc_buf_sz  = ANS_LANES * max_chunk;
    ensure_enc_buf(b, enc_buf_sz);
    // Compact buffer: at most enc_buf_sz but in practice ≤ byte_size
    ensure_compact_buf(b, enc_buf_sz);

    // ---- Upload encode tables to GPU ----
    auto t_encode_start = now_ms();
    GPU_MEMCPY_H2D(w.d_norm_freq, table.norm_freq, 256 * sizeof(uint32_t), s);
    GPU_MEMCPY_H2D(w.d_cum_freq,  table.cum_freq,  257 * sizeof(uint32_t), s);
    GPU_MEMCPY_H2D(w.d_x_max,     table.x_max,     256 * sizeof(uint32_t), s);

    // ---- Launch encode kernel (1 block, 256 threads) ----
    rans_encode_kernel<<<dim3(1), dim3(ANS_LANES), 0, s>>>(
        static_cast<const uint8_t*>(d_data),
        byte_size,
        w.d_norm_freq,
        w.d_cum_freq,
        w.d_x_max,
        b.d_enc_buf,
        w.d_streams,
        max_chunk);
    GPU_STREAM_SYNC(s);
    ct.encode_ms = elapsed_ms(t_encode_start);

    // ---- Check for GPU errors surfaced by the sync ----
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    {
        hipError_t post_enc = hipGetLastError();
        if (post_enc != hipSuccess) {
            const std::string msg = lp("gpu_fault") +
                "GPU error after encode sync for '" + name +
                "': " + hipGetErrorString(post_enc);
            if (logger_) logger_->log_message(msg);
            throw std::runtime_error(msg);
        }
    }
#else
    {
        cudaError_t post_enc = cudaGetLastError();
        if (post_enc != cudaSuccess) {
            const std::string msg = lp("gpu_fault") +
                "GPU error after encode sync for '" + name +
                "': " + cudaGetErrorString(post_enc);
            if (logger_) logger_->log_message(msg);
            throw std::runtime_error(msg);
        }
    }
#endif

    // ---- Copy stream headers back to CPU (tiny: ANS_LANES * 16 = 4 KB) ----
    auto t_copy_start = now_ms();
    ANSStreamHeader h_streams[ANS_LANES];
    GPU_MEMCPY_D2H(h_streams, w.d_streams, ANS_LANES * sizeof(ANSStreamHeader), s);
    GPU_STREAM_SYNC(s);

    // ---- Compute compaction prefix-sum offsets on CPU (trivial) ----
    uint32_t h_offsets[ANS_LANES + 1];
    h_offsets[0] = 0;
    for (uint32_t t = 0; t < ANS_LANES; t++)
        h_offsets[t + 1] = h_offsets[t] + h_streams[t].byte_count;
    const size_t total_compressed = h_offsets[ANS_LANES];

    // Safety: corrupt stream headers from a GPU fault could produce a wildly large
    // total_compressed, making resize() allocate several GB and the subsequent DMA
    // trigger SIGSEGV.  Cap at 2× original to match the kernel's worst-case.
    if (total_compressed > enc_buf_sz) {
        const std::string msg = lp("corrupt_streams") +
            "total_compressed=" + std::to_string(total_compressed) +
            " > enc_buf_sz=" + std::to_string(enc_buf_sz) +
            " for '" + name + "' — likely deferred GPU fault; falling back to passthrough";
        if (logger_) logger_->log_message(msg);
        std::memset(ct.streams, 0, sizeof(ct.streams));
        ct.streams[0].byte_count = (uint32_t)byte_size;
        return ct;
    }

    // ---- Upload offsets + compact on GPU → copy only actual compressed bytes ----
    GPU_MEMCPY_H2D(w.d_offsets, h_offsets, (ANS_LANES + 1) * sizeof(uint32_t), s);
    compact_streams_kernel<<<dim3(ANS_LANES), dim3(256), 0, s>>>(
        b.d_enc_buf, w.d_streams, b.d_compact_buf, w.d_offsets, max_chunk);

    if (logger_) logger_->log_message(lp("dbg") + "compress_tensor: host resize total_compressed=" +
        std::to_string(total_compressed) + " for name=" + name);
    try {
        ct.data.resize(total_compressed);
    } catch (const std::bad_alloc& ba) {
        if (logger_) logger_->log_message(
            "[ANS][bad_alloc] compress_tensor ct.data.resize(" +
            std::to_string(total_compressed) + ") failed for '" + name + "': " + ba.what());
        throw;
    }
    GPU_MEMCPY_D2H(ct.data.data(), b.d_compact_buf, total_compressed, s);
    GPU_STREAM_SYNC(s);

    // ---- Populate per-lane stream metadata (compacted offsets) ----
    size_t compact_offset = 0;
    for (uint32_t t = 0; t < ANS_LANES; t++) {
        ct.streams[t]            = h_streams[t];
        ct.streams[t].buf_offset = (uint32_t)compact_offset; // offset in compacted buffer
        compact_offset += h_streams[t].byte_count;
    }
    ct.copy_ms = elapsed_ms(t_copy_start);

    // ---- Fallback: store uncompressed if compression expanded the data ----
    if (total_compressed >= byte_size && byte_size > 0) {
        if (logger_) logger_->log_message(lp("dbg") + "compress_tensor: fallback passthrough for '" +
            name + "' total_compressed=" + std::to_string(total_compressed) +
            " byte_size=" + std::to_string(byte_size));
        try {
            std::vector<uint8_t> h_raw(byte_size);
            GPU_MEMCPY_D2H(h_raw.data(), d_data, byte_size, s);
            GPU_STREAM_SYNC(s);
            ct.data = std::move(h_raw);
        } catch (const std::bad_alloc& ba) {
            if (logger_) logger_->log_message(
                lp("bad_alloc") + "compress_tensor fallback alloc(" +
                std::to_string(byte_size) + ") failed for '" + name + "': " + ba.what());
            throw;
        }
        // Zero out stream headers to signal "raw passthrough"
        std::memset(ct.streams, 0, sizeof(ct.streams));
        ct.streams[0].byte_count = (uint32_t)byte_size; // sentinel: whole payload is raw
    }

    if (logger_) logger_->log_message(lp("dbg") + "compress_tensor done name=" + name +
        " compressed=" + std::to_string(ct.data.size()));
    return ct;
}

// ====================================================================
// write_rank_states_file  (binary per-rank .ansc v2)
// ====================================================================
// Layout (little-endian):
//   Header (16 bytes):
//     [4]  magic    0x414E5343 "ANSC"
//     [4]  version  2
//     [4]  rank
//     [4]  n_tensors
//   Index table  (n_tensors × INDEX_ENTRY_BYTES bytes):
//     Per tensor:
//       [256]         name (null-padded)
//       [1]           dtype
//       [1]           is_raw_passthrough
//       [1]           owns_table
//       [5]           padding
//       [8]           original_bytes
//       [8]           num_elements
//       [8]           data_offset  (byte offset into data section below)
//       [8]           data_size
//       [256]         anchor_name  (null-padded; empty when owns_table)
//       [ANS_LANES×16] ANSStreamHeader[ANS_LANES]
//   Data section:
//     All tensors' compressed/raw bytes concatenated in index order.
// ====================================================================
static constexpr uint32_t ANSC_VERSION_V2  = 2u;
static constexpr uint32_t DICT_MAGIC       = 0x44494354u;  // "DICT"
static constexpr uint32_t DICT_VERSION     = 1u;
static constexpr uint32_t MANT_MAGIC       = 0x4D414E54u;  // "MANT"
static constexpr uint32_t MANT_VERSION_V2  = 2u;

void ANSCompressor::write_rank_states_file(
        const std::string&                   path,
        int                                  rank,
        const std::vector<CompressedTensor>& cts) const
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("[ANS] cannot open for write: " + path);

    auto write = [&](const void* p, size_t n) {
        f.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
    };

    // ---- Header ----
    const uint32_t magic    = ANSC_MAGIC;
    const uint32_t version  = ANSC_VERSION_V2;
    const uint32_t rank_u32 = static_cast<uint32_t>(rank < 0 ? 0 : rank);
    const uint32_t n        = static_cast<uint32_t>(cts.size());
    write(&magic,    4);
    write(&version,  4);
    write(&rank_u32, 4);
    write(&n,        4);

    // ---- Build data-section offsets ----
    std::vector<uint64_t> offsets(cts.size() + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < cts.size(); i++)
        offsets[i + 1] = offsets[i] + cts[i].data.size();

    // ---- Index table ----
    for (size_t i = 0; i < cts.size(); i++) {
        const auto& ct = cts[i];

        char name_buf[256] = {};
        std::strncpy(name_buf, ct.name.c_str(), 255);
        write(name_buf, 256);

        bool is_raw = (ct.streams[0].byte_count == (uint32_t)ct.original_bytes);
        for (int t = 1; t < (int)ANS_LANES && is_raw; t++)
            is_raw &= (ct.streams[t].byte_count == 0);

        const uint8_t dtype_u8   = static_cast<uint8_t>(ct.dtype);
        const uint8_t is_raw_u8  = is_raw ? 1u : 0u;
        const uint8_t owns_u8    = ct.owns_table ? 1u : 0u;
        write(&dtype_u8,  1);
        write(&is_raw_u8, 1);
        write(&owns_u8,   1);
        const uint8_t pad5[5] = {};
        write(pad5, 5);

        write(&ct.original_bytes, 8);
        write(&ct.num_elements,   8);
        write(&offsets[i],        8);
        const uint64_t dsz = cts[i].data.size();
        write(&dsz, 8);

        char anchor_buf[256] = {};
        std::strncpy(anchor_buf, ct.anchor_name.c_str(), 255);
        write(anchor_buf, 256);

        write(ct.streams, ANS_LANES * sizeof(ANSStreamHeader));
    }

    // ---- Data section ----
    for (const auto& ct : cts)
        write(ct.data.data(), ct.data.size());
}

// ====================================================================
// write_rank_mantissa_file  (binary per-rank .mant v2)
// ====================================================================
// Layout (little-endian):
//   Header (16 bytes):
//     [4]  magic    0x4D414E54 "MANT"
//     [4]  version  2
//     [4]  rank
//     [4]  n_tensors
//   Index table (n_tensors entries):
//     Per tensor:
//       [256]  name (null-padded)
//       [1]    dtype
//       [1]    bytes_per_elem
//       [6]    padding
//       [8]    n_elements
//       [8]    data_offset (into data section)
//       [8]    data_size
//   Data section:
//     All sidecar bytes concatenated in index order.
// ====================================================================
void ANSCompressor::write_rank_mantissa_file(
        const std::string&                 path,
        int                                rank,
        const std::vector<SidecarEntry>&   sidecars) const
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("[ANS] cannot open mantissa for write: " + path);

    auto write = [&](const void* p, size_t n) {
        f.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
    };

    const uint32_t magic    = MANT_MAGIC;
    const uint32_t version  = MANT_VERSION_V2;
    const uint32_t rank_u32 = static_cast<uint32_t>(rank < 0 ? 0 : rank);
    const uint32_t cnt      = static_cast<uint32_t>(sidecars.size());
    write(&magic,    4);
    write(&version,  4);
    write(&rank_u32, 4);
    write(&cnt,      4);

    // build offsets
    std::vector<uint64_t> offsets(sidecars.size() + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < sidecars.size(); i++)
        offsets[i + 1] = offsets[i] + sidecars[i].si.data.size();

    for (size_t i = 0; i < sidecars.size(); i++) {
        const auto& se = sidecars[i];
        char name_buf[256] = {};
        std::strncpy(name_buf, se.name.c_str(), 255);
        write(name_buf, 256);

        const uint8_t dtype_u8 = static_cast<uint8_t>(se.dtype);
        const uint8_t bpe      = static_cast<uint8_t>(se.si.bytes_per_elem);
        write(&dtype_u8, 1);
        write(&bpe,      1);
        const uint8_t pad6[6] = {};
        write(pad6, 6);

        write(&se.n_elements, 8);
        write(&offsets[i],    8);
        const uint64_t dsz = se.si.data.size();
        write(&dsz, 8);
    }

    for (const auto& se : sidecars)
        write(se.si.data.data(), se.si.data.size());
}

// ====================================================================
// write_rank_tables_file  (binary per-rank .dict)
// ====================================================================
// Layout (little-endian):
//   Header (16 bytes):
//     [4]  magic    0x44494354 "DICT"
//     [4]  version  1
//     [4]  rank
//     [4]  n_tables
//   Per table (fixed 2308 bytes each):
//     [256]      anchor_name (null-padded; the tensor that owns this table)
//     [256×4]    norm_freq[256]
//     [257×4]    cum_freq[257]
// ====================================================================
void ANSCompressor::write_rank_tables_file(
        const std::string&               path,
        int                              rank,
        const std::vector<TableEntry>&   tables) const
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("[ANS] cannot open tables for write: " + path);

    auto write = [&](const void* p, size_t n) {
        f.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
    };

    const uint32_t magic    = DICT_MAGIC;
    const uint32_t version  = DICT_VERSION;
    const uint32_t rank_u32 = static_cast<uint32_t>(rank < 0 ? 0 : rank);
    const uint32_t cnt      = static_cast<uint32_t>(tables.size());
    write(&magic,    4);
    write(&version,  4);
    write(&rank_u32, 4);
    write(&cnt,      4);

    for (const auto& te : tables) {
        char name_buf[256] = {};
        std::strncpy(name_buf, te.anchor_name.c_str(), 255);
        write(name_buf, 256);
        write(te.table.norm_freq, 256 * sizeof(uint32_t));
        write(te.table.cum_freq,  257 * sizeof(uint32_t));
    }
}

// ====================================================================
// extract_mantissa
// ====================================================================
// Launches extract_mantissa_kernel to produce:
//   w.d_masked_buf  – original tensor with low mantissa byte(s) zeroed
//   w.d_sidecar_buf – the extracted low byte(s) per element
// Copies sidecar to host and returns it.  The caller should use
// w.d_masked_buf as the input to subsequent histogram + ANS encode.
//
// Returns SidecarInfo with bytes_per_elem == 0 when the dtype is not
// supported (caller should fall back to the original d_data pointer).
// ====================================================================
ANSCompressor::SidecarInfo ANSCompressor::extract_mantissa(WorkerGPU&  w,
                                                            SharedGPUBufs& b,
                                                            const void* d_data,
                                                            size_t      n_elements,
                                                            DataType    dtype)
{
    SidecarInfo si{};

    uint32_t elem_stride       = 0;
    uint32_t sidecar_per_elem  = 0;
    switch (dtype) {
        case DataType::Float16:  // fp16: 2B/elem, 10 mantissa bits → zero byte[0]
        case DataType::BFloat16: // bf16: 2B/elem,  7 mantissa bits → zero byte[0]
            elem_stride      = 2;
            sidecar_per_elem = 1;
            break;
        case DataType::Float32:  // fp32: 4B/elem, 23 mantissa bits → zero bytes[0,1]
            elem_stride      = 4;
            sidecar_per_elem = 2;
            break;
        default:
            return si; // unsupported – no sidecar
    }

    if (n_elements == 0) return si;

    const size_t masked_bytes  = n_elements * elem_stride;
    const size_t sidecar_bytes = n_elements * sidecar_per_elem;
    if (logger_) logger_->log_message(lp("dbg") + "extract_mantissa enter"
        " n_elements=" + std::to_string(n_elements) +
        " elem_stride=" + std::to_string(elem_stride) +
        " sidecar_per_elem=" + std::to_string(sidecar_per_elem) +
        " masked_bytes=" + std::to_string(masked_bytes) +
        " sidecar_bytes=" + std::to_string(sidecar_bytes));
    ensure_masked_buf(b, masked_bytes, sidecar_bytes);

    const uint32_t block_sz = 256;
    const uint32_t nblocks  = (uint32_t)((n_elements + block_sz - 1) / block_sz);
    extract_mantissa_kernel<<<dim3(nblocks), dim3(block_sz), 0, w.stream>>>(
        static_cast<const uint8_t*>(d_data),
        b.d_masked_buf,
        b.d_sidecar_buf,
        n_elements,
        elem_stride,
        sidecar_per_elem);
    // Kernel launched asynchronously on w.stream; caller can immediately
    // launch histogram on the same stream (ordering guaranteed).

    // Copy sidecar to host (sidecar_bytes ≤ half the original tensor size).
    if (logger_) logger_->log_message(lp("dbg") + "extract_mantissa: host resize sidecar_bytes=" +
        std::to_string(sidecar_bytes));
    try {
        si.data.resize(sidecar_bytes);
    } catch (const std::bad_alloc& ba) {
        if (logger_) logger_->log_message(
            "[ANS][bad_alloc] extract_mantissa si.data.resize(" +
            std::to_string(sidecar_bytes) + ") failed: " + ba.what());
        throw;
    }
    si.bytes_per_elem = sidecar_per_elem;
    GPU_MEMCPY_D2H(si.data.data(), b.d_sidecar_buf, sidecar_bytes, w.stream);
    GPU_STREAM_SYNC(w.stream);
    if (logger_) logger_->log_message(lp("dbg") + "extract_mantissa done");
    return si;
}

// ====================================================================
// compress_and_save_all  – high-level driver
// ====================================================================
void ANSCompressor::compress_and_save_all(const ClusterInfo& clusters,
                                          MemoryManager&     mgr,
                                          const std::string& output_dir)
{
    namespace fs = std::filesystem;
    const auto total_t0 = now_ms();

    fs::create_directories(output_dir);
    fs::create_directories(output_dir + "/mantissa");
    fs::create_directories(output_dir + "/tables");

    const size_t N = mgr.record_count();
    if (N == 0) {
        if (logger_) logger_->log_message(lp("info") + "compress_and_save_all: no tensors to compress");
        return;
    }

    const auto& host_recs = mgr.host_records();
    const size_t active_n = std::min(N, host_recs.size());

    if (logger_)
        logger_->log_message(lp("info") + "compress_and_save_all begin"
            " tensors=" + std::to_string(active_n) +
            " workers=" + std::to_string(n_workers_) +
            " output_dir=" + output_dir);

    // ---- Build work units ----
    struct TensorWork {
        size_t      record_idx;
        std::string name;
        void*       d_ptr;
        size_t      byte_size;
        DataType    dtype;
        size_t      num_elements;
        bool        is_anchor;
        bool        is_singleton;
        int         cluster_id;   // -1 if singleton
    };

    std::vector<TensorWork> work_items;
    work_items.reserve(active_n);

    std::unordered_map<int, bool> cluster_seen;
    for (size_t i = 0; i < active_n; i++) {
        const auto& rec   = host_recs[i];
        const std::string& tname = mgr.get_name_from_index(i);
        if (!rec.d_ptr || rec.num_elements == 0) continue;

        const DataType dtype = static_cast<DataType>(rec.data_type);
        int cid = -1;
        auto it = clusters.tensor_to_cluster.find(tname);
        if (it != clusters.tensor_to_cluster.end()) cid = it->second;

        bool is_singleton = (cid == -1);
        bool is_anchor    = false;
        if (!is_singleton && !cluster_seen[cid]) {
            cluster_seen[cid] = true;
            is_anchor = true;
        }

        work_items.push_back({i, tname, rec.d_ptr,
                              rec.num_elements * dtype_element_size(dtype),
                              dtype, rec.num_elements,
                              is_anchor, is_singleton, cid});
    }

    const size_t n_items = work_items.size();

    // ---- Result collection (one slot per work_item, written by workers) ----
    // No mutex needed: each slot is written by exactly one worker, and all
    // reads happen after the worker threads are joined.
    struct TensorResult {
        CompressedTensor  ct;
        SidecarInfo       si;
        ANSFreqTable      table;       // valid when ct.owns_table == true
        bool              valid{false};
    };
    std::vector<TensorResult> results(n_items);

    // ---- Shared cluster tables (anchor deposits, members consume) ----
    std::mutex                            tables_mtx;
    std::unordered_map<int, ANSFreqTable> cluster_tables;

    // ---- Thread pool ----
    std::mutex              queue_mtx;
    std::condition_variable queue_cv;
    std::deque<size_t>      queue;
    bool                    all_enqueued = false;
    for (size_t i = 0; i < n_items; i++) queue.push_back(i);

    std::mutex   stats_mtx;
    double total_compute_ms = 0.0;
    size_t done_count = 0;

    auto worker_fn = [&](int wid) {
        auto& w = workers_[wid];

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
        (void)hipSetDevice(w.device_id);
#else
        (void)cudaSetDevice(w.device_id);
#endif
        { volatile auto _t0 = now_ms(); volatile auto _t1 = now_ms(); (void)_t0; (void)_t1; }
        if (logger_) logger_->log_message(lp("dbg") +
            "worker" + std::to_string(wid) + " thread started device=" +
            std::to_string(w.device_id));

        while (true) {
            size_t work_idx;
            {
                std::unique_lock<std::mutex> lk(queue_mtx);
                queue_cv.wait(lk, [&]{ return !queue.empty() || all_enqueued; });
                if (queue.empty()) return;
                work_idx = queue.front();
                queue.pop_front();
            }
            queue_cv.notify_all();

            auto& item = work_items[work_idx];
            if (!item.d_ptr || item.byte_size == 0) continue;

            if (logger_) logger_->log_message(lp("dbg") + "worker" + std::to_string(wid) +
                " start tensor='" + item.name +
                "' bytes=" + std::to_string(item.byte_size) +
                " dtype=" + std::to_string((int)item.dtype) +
                " n_elems=" + std::to_string(item.num_elements) +
                " is_anchor=" + std::to_string(item.is_anchor) +
                " is_singleton=" + std::to_string(item.is_singleton) +
                " d_ptr=" + (item.d_ptr ? "ok" : "NULL"));

            try {

            // ---- Drain deferred GPU errors before any kernel launch ----
            // ROCm page-faults are surfaced as SIGSEGV if not drained first.
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
            {
                hipError_t pending = hipGetLastError();
                if (pending != hipSuccess) {
                    if (logger_) logger_->log_message(lp("gpu_fault") +
                        "worker" + std::to_string(wid) +
                        " deferred GPU error before '" + item.name +
                        "': " + hipGetErrorString(pending) + " — skipping");
                    continue;
                }
            }
#else
            {
                cudaError_t pending = cudaGetLastError();
                if (pending != cudaSuccess) {
                    if (logger_) logger_->log_message(lp("gpu_fault") +
                        "worker" + std::to_string(wid) +
                        " deferred GPU error before '" + item.name +
                        "': " + cudaGetErrorString(pending) + " — skipping");
                    continue;
                }
            }
#endif

            // ---- Validate source pointer before launching extract_mantissa ----
            // extract_mantissa_kernel accesses item.d_ptr directly; if the pointer
            // became stale (PyTorch freed / moved the tensor), launching without
            // validation triggers a GPU page-fault → SIGSEGV on ROCm.
            // hipPointerGetAttributes detects unmapped addresses as a catchable error.
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
            {
                hipPointerAttribute_t attr{};
                hipError_t ptr_stat = hipPointerGetAttributes(&attr, item.d_ptr);
                if (ptr_stat != hipSuccess) {
                    if (logger_) logger_->log_message(lp("invalid_ptr") +
                        "worker" + std::to_string(wid) +
                        " d_ptr invalid for '" + item.name +
                        "' err=" + hipGetErrorString(ptr_stat) + " — skipping");
                    continue;
                }
            }
#else
            {
                cudaPointerAttributes attr{};
                cudaError_t ptr_stat = cudaPointerGetAttributes(&attr, item.d_ptr);
                if (ptr_stat != cudaSuccess) {
                    cudaGetLastError(); // clear the error
                    if (logger_) logger_->log_message(lp("invalid_ptr") +
                        "worker" + std::to_string(wid) +
                        " d_ptr invalid for '" + item.name +
                        "' err=" + cudaGetErrorString(ptr_stat) + " — skipping");
                    continue;
                }
            }
#endif

            // ---- Acquire shared GPU buffers (only one worker uses GPU at a time) ----
            // All GPU kernels in this block are fully synchronous (we sync the stream
            // before releasing the lock), so serializing here prevents the N-worker
            // GPU memory amplification that previously caused OOM / SIGSEGV.
            SidecarInfo  si;
            ANSFreqTable table{};
            bool         owns_table  = item.is_anchor || item.is_singleton;
            std::string  anchor_name;
            double hist_ms = 0.0, table_ms = 0.0;
            CompressedTensor ct;

            // Phase 1: extract mantissa + build histogram (under GPU lock)
            {
                std::lock_guard<std::mutex> gpu_lock(shared_bufs_mtx_);
                SharedGPUBufs& b = shared_bufs_;

                si = extract_mantissa(w, b, item.d_ptr, item.num_elements, item.dtype);
                const void* compress_src = (si.bytes_per_elem > 0)
                    ? static_cast<const void*>(b.d_masked_buf)
                    : item.d_ptr;

                if (owns_table) {
                    const auto t_hist0 = now_ms();
                    GPU_MEMSET(w.d_hist, 0, 256 * sizeof(uint32_t), w.stream);
                    const uint32_t nblocks = (uint32_t)std::min<size_t>(
                        (item.byte_size + 255) / 256, 4096);
                    byte_histogram_kernel<<<dim3(nblocks), dim3(256), 0, w.stream>>>(
                        static_cast<const uint8_t*>(compress_src),
                        item.byte_size, w.d_hist);
                    uint32_t h_hist[256] = {};
                    GPU_MEMCPY_D2H(h_hist, w.d_hist, 256 * sizeof(uint32_t), w.stream);
                    GPU_STREAM_SYNC(w.stream);
                    hist_ms = elapsed_ms(t_hist0);

                    const auto t_table0 = now_ms();
                    table = normalize_histogram(h_hist, /*ensure_full_alphabet=*/!item.is_singleton);
                    table_ms = elapsed_ms(t_table0);

                    if (!item.is_singleton) {
                        std::lock_guard<std::mutex> lk(tables_mtx);
                        cluster_tables[item.cluster_id] = table;
                    }
                }
            } // release GPU lock

            // Phase 2: wait for cluster anchor table (no GPU resources held)
            if (!owns_table) {
                while (true) {
                    {
                        std::lock_guard<std::mutex> lk(tables_mtx);
                        auto it = cluster_tables.find(item.cluster_id);
                        if (it != cluster_tables.end()) { table = it->second; break; }
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                auto it2 = clusters.cluster_anchor.find(item.cluster_id);
                if (it2 != clusters.cluster_anchor.end())
                    anchor_name = it2->second;
            }

            // Phase 3: compress (under GPU lock; re-extract mantissa since b may have been
            // reused by another worker while we waited for the anchor table)
            {
                std::lock_guard<std::mutex> gpu_lock(shared_bufs_mtx_);
                SharedGPUBufs& b = shared_bufs_;

                si = extract_mantissa(w, b, item.d_ptr, item.num_elements, item.dtype);
                const void* compress_src2 = (si.bytes_per_elem > 0)
                    ? static_cast<const void*>(b.d_masked_buf)
                    : item.d_ptr;

                ct = compress_tensor(
                    item.name, compress_src2, item.byte_size,
                    item.dtype, item.num_elements,
                    table, owns_table, anchor_name,
                    w, b);
            }
            ct.hist_ms  = hist_ms;
            ct.table_ms = table_ms;

            // ---- Accumulate stats ----
            const double tensor_compute = ct.hist_ms + ct.table_ms + ct.encode_ms + ct.copy_ms;
            {
                std::lock_guard<std::mutex> lk(stats_mtx);
                total_compute_ms += tensor_compute;
                done_count++;
            }

            if (logger_) {
                const size_t sidecar_bytes = si.data.size();
                // ans_ratio = original / ANS-only bytes; > 1 means compression gain.
                // The sidecar and table are NOT counted in the denominator.
                const double ans_ratio = (ct.data.size() > 0)
                    ? (double)item.byte_size / (double)ct.data.size() : 1.0;
                // disk_ratio = original / (ANS + sidecar); shows true storage factor.
                const double disk_ratio = (item.byte_size > 0)
                    ? (double)item.byte_size / (double)(ct.data.size() + sidecar_bytes) : 1.0;
                logger_->log_message(
                    lp("tensor") + "name=" + item.name +
                    " original_bytes=" + std::to_string(item.byte_size) +
                    " ans_bytes=" + std::to_string(ct.data.size()) +
                    " sidecar_bytes=" + std::to_string(sidecar_bytes) +
                    " ans_ratio=" + std::to_string(ans_ratio) +
                    " disk_ratio=" + std::to_string(disk_ratio) +
                    " owns_table=" + (owns_table ? "1" : "0") +
                    " hist_ms=" + std::to_string(ct.hist_ms) +
                    " table_ms=" + std::to_string(ct.table_ms) +
                    " encode_ms=" + std::to_string(ct.encode_ms) +
                    " copy_ms=" + std::to_string(ct.copy_ms));
            }

            // ---- Deposit result (no mutex needed; each slot owned by one worker) ----
            results[work_idx].ct    = std::move(ct);
            results[work_idx].si    = std::move(si);
            results[work_idx].table = table;   // copy: table may be needed by non-anchor peers
            results[work_idx].valid = true;

            } catch (const std::bad_alloc& ba) {
                if (logger_) logger_->log_message(
                    lp("bad_alloc") + "worker" + std::to_string(wid) +
                    " tensor='" + item.name +
                    "' bytes=" + std::to_string(item.byte_size) +
                    " n_elems=" + std::to_string(item.num_elements) +
                    " what=" + std::string(ba.what()));
                throw;
            } catch (const std::exception& ex) {
                if (logger_) logger_->log_message(
                    lp("error") + "worker" + std::to_string(wid) +
                    " tensor='" + item.name + "': " + ex.what());
                throw;
            }
        } // while(true)
    }; // worker_fn

    // ---- Launch & join worker threads ----
    std::vector<std::thread> threads;
    threads.reserve(n_workers_);
    {
        std::lock_guard<std::mutex> lk(queue_mtx);
        all_enqueued = true;
    }
    for (size_t wid = 0; wid < n_workers_; wid++)
        threads.emplace_back(worker_fn, (int)wid);
    queue_cv.notify_all();
    for (auto& t : threads) t.join();

    // ---- Assemble per-rank output files ----
    // Collect results in work_item order (preserves tensor ordering).
    std::vector<CompressedTensor>  rank_cts;
    std::vector<SidecarEntry>      rank_sidecars;
    std::vector<TableEntry>        rank_tables;
    rank_cts.reserve(done_count);
    rank_sidecars.reserve(done_count);
    rank_tables.reserve(done_count);

    for (size_t i = 0; i < n_items; i++) {
        if (!results[i].valid) continue;
        auto& r = results[i];
        const auto& item = work_items[i];

        if (r.ct.owns_table) {
            rank_tables.push_back({r.ct.name, r.table});
        }
        if (r.si.bytes_per_elem > 0) {
            rank_sidecars.push_back({r.ct.name, r.ct.dtype, item.num_elements, std::move(r.si)});
        }
        rank_cts.push_back(std::move(r.ct));
    }

    // ---- Write the three per-rank files ----
    const std::string rank_tag   = "rank_" + std::to_string(rank_ < 0 ? 0 : rank_);
    const std::string states_path = output_dir + "/" + rank_tag + "_states.ansc";
    const std::string mant_path   = output_dir + "/mantissa/" + rank_tag + "_mantissa.mant";
    const std::string dict_path   = output_dir + "/tables/"   + rank_tag + "_tables.dict";

    const auto t_io0 = now_ms();
    write_rank_states_file(states_path, rank_, rank_cts);
    write_rank_mantissa_file(mant_path, rank_, rank_sidecars);
    write_rank_tables_file(dict_path,   rank_, rank_tables);
    const double io_ms = elapsed_ms(t_io0);

    const double total_wall_ms = elapsed_ms(total_t0);

    if (logger_) {
        logger_->log_message(
            lp("summary") + "tensors=" + std::to_string(done_count) +
            " total_wall_ms="    + std::to_string(total_wall_ms) +
            " total_compute_ms=" + std::to_string(total_compute_ms) +
            " io_ms="            + std::to_string(io_ms) +
            " states="           + states_path +
            " mantissa="         + mant_path +
            " tables="           + dict_path);

        logger_->log_message("[TIMING] compress_and_save_total_ms="   + std::to_string(total_wall_ms));
        logger_->log_message("[TIMING] compress_computation_ms="      + std::to_string(total_compute_ms));
        logger_->log_message("[TIMING] compress_io_ms="               + std::to_string(io_ms));
    }
}

} // namespace checkers
