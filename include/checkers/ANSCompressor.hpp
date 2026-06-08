#pragma once
// include/checkers/ANSCompressor.hpp
//
// GPU-accelerated Asymmetric Numeral Systems (rANS) checkpoint compressor.
//
// Design overview
// ---------------
//  * rANS with 256-byte alphabet, M = 4096 (12-bit scale), state fits in uint32_t.
//  * 256 independent interleaved lanes per tensor (one lane per GPU thread).
//    Lane t encodes bytes at strided positions t, t+256, t+512, ... so all 256
//    lanes can run concurrently in one 256-thread GPU block.
//  * Cluster-aware: the first (anchor) tensor in a cluster builds the frequency
//    table; every other tensor in the same cluster reuses that same table.
//    Singletons always build their own table.
//  * Background CPU thread pool drives per-cluster compression and I/O so that
//    multiple clusters are processed concurrently using independent GPU streams.
//
// Usage (from api.cpp)
// --------------------
//  ANSCompressor comp(logger, /*n_workers=*/4);
//  comp.compress_and_save_all(mgr.get_cluster_info(), mgr, output_dir);
//  (ClusterInfo is written by TensorAnalyzer::build_knn_clusters)

#include "checkers/MemoryManager.hpp"
#include "checkers/logging.hpp"

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  include <hip/hip_runtime.h>
#else
#  include <cuda_runtime.h>
#endif

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace checkers {

// ===================================================================
// rANS constants
// -------------------------------------------------------------------
// Scale:   M  = 2^12 = 4096  (table slots)
// State:   L  = 2^23         (lower bound; state lives in [L, 256·L))
// Renorm:  NORM_BASE = L/M·B = 2^11·2^8 = 2^19
//          Per-symbol threshold: x_max[s] = NORM_BASE * norm_freq[s]
//          Condition to emit one byte: state >= x_max[s]
// ===================================================================
static constexpr uint32_t ANS_SCALE_BITS = 12;
static constexpr uint32_t ANS_M          = 1u << ANS_SCALE_BITS;  // 4096
static constexpr uint32_t ANS_L          = 1u << 23;               // 8388608
static constexpr uint32_t ANS_NORM_BASE  = (ANS_L >> ANS_SCALE_BITS) << 8; // 2^19
static constexpr uint32_t ANS_LANES      = 256;  // threads per block, one lane each

// ===================================================================
// ANSFreqTable
// -------------------------------------------------------------------
// Holds the complete encode + decode lookup tables for one "grammar".
// Small enough (< 40 KB) to upload to every GPU kernel as a plain
// device pointer without using constant memory.
// ===================================================================
struct alignas(64) ANSFreqTable {
    // ---- encode side ----
    uint32_t norm_freq[256];     // normalized frequencies; sum == ANS_M
    uint32_t cum_freq[257];      // prefix sums; cum_freq[0]=0, cum_freq[256]=ANS_M
    uint32_t x_max[256];         // per-symbol renorm threshold: NORM_BASE * norm_freq[s]

    // ---- decode side ----
    uint8_t  decode_sym [ANS_M]; // slot → symbol
    uint32_t decode_freq[ANS_M]; // slot → norm_freq[sym]
    uint32_t decode_cum [ANS_M]; // slot → cum_freq[sym]
};

// ===================================================================
// ANSStreamHeader – per-lane metadata written by the GPU encode kernel
// ===================================================================
struct ANSStreamHeader {
    uint32_t final_state;    // rANS state at end of encoding (start state for decoder)
    uint32_t byte_count;     // compressed bytes produced by this lane
    uint32_t buf_offset;     // offset from start of lane's raw buffer where data begins
    uint32_t pad;            // align to 16 bytes
};

// ===================================================================
// CompressedTensor – CPU-side result after one compress_tensor() call
// ===================================================================
struct CompressedTensor {
    std::string  name;
    DataType     dtype;
    size_t       original_bytes;
    size_t       num_elements;

    bool         owns_table;   // true  → table was built from THIS tensor (anchor / singleton)
                               // false → table comes from the cluster anchor
    std::string  anchor_name;  // meaningful only when !owns_table

    ANSStreamHeader streams[ANS_LANES]; // per-lane metadata

    // Compacted compressed bytes (all 256 lane streams packed contiguously).
    // streams[t].buf_offset holds the start byte of lane t inside this vector.
    std::vector<uint8_t> data;

    // ---- Timing ----
    double hist_ms   = 0.0;  // GPU histogram kernel + copy
    double table_ms  = 0.0;  // CPU table normalization
    double encode_ms = 0.0;  // GPU encode kernel
    double copy_ms   = 0.0;  // GPU→CPU copy + CPU compaction
    double io_ms     = 0.0;  // file write
};

// ===================================================================
// ANSCompressor
// ===================================================================
class ANSCompressor {
public:
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    using gpuStream_t = hipStream_t;
#else
    using gpuStream_t = cudaStream_t;
#endif

    explicit ANSCompressor(std::shared_ptr<RankLogger> logger,
                           size_t n_worker_threads = 4);
    ~ANSCompressor();

    int rank() const { return rank_; }

    ANSCompressor(const ANSCompressor&)            = delete;
    ANSCompressor& operator=(const ANSCompressor&) = delete;

    // -----------------------------------------------------------
    // Build a byte-frequency histogram from a device buffer.
    // Launches a multi-block 256-thread histogram kernel and copies
    // the result (h_hist_out[256]) to the caller's host array.
    // -----------------------------------------------------------
    void build_byte_histogram(const void* d_data,
                              size_t      n_bytes,
                              uint32_t    h_hist_out[256],
                              gpuStream_t stream = nullptr);

    // -----------------------------------------------------------
    // CPU-side: normalize a raw byte histogram into a full ANSFreqTable.
    // Symbols with zero count get norm_freq = 0 (cannot be encoded).
    // If ensure_full_alphabet is true every symbol gets >= 1 slot,
    // which is required for cluster (non-anchor) tensors.
    // -----------------------------------------------------------
    ANSFreqTable normalize_histogram(const uint32_t h_raw_hist[256],
                                     bool ensure_full_alphabet = false) const;

    // -----------------------------------------------------------
    // High-level: compress every tensor tracked in the MemoryManager,
    // applying the cluster-aware grammar sharing described above.
    // -----------------------------------------------------------
    void compress_and_save_all(const ClusterInfo& clusters,
                               MemoryManager&     mgr,
                               const std::string& output_dir);

private:
    std::shared_ptr<RankLogger> logger_;
    size_t n_workers_;

    // ===================================================================
    // SidecarInfo – mantissa bits extracted from one tensor before ANS.
    // For bf16/fp16 (2B/elem): byte[0] of each element (7-10 mantissa bits)
    //   is zeroed in the masked copy; saved here = n_elements bytes.
    // For fp32    (4B/elem): bytes[0,1] of each element (16 mantissa bits)
    //   are zeroed in the masked copy; saved here = 2*n_elements bytes.
    // Reconstruction (lossless): original_elem = masked_elem | sidecar_elem
    // ===================================================================
    struct SidecarInfo {
        std::vector<uint8_t> data;       // extracted low bytes on host
        uint32_t bytes_per_elem = 0;     // 0 = dtype not supported (no sidecar)
    };

    // ---------------------------------------------------------------
    // Per-worker GPU state: only fixed-size small buffers live here.
    // The four large dynamic buffers (enc, compact, masked, sidecar)
    // are shared across all workers in SharedGPUBufs to prevent OOM
    // when multiple workers run on the same physical GPU.
    // ---------------------------------------------------------------
    struct WorkerGPU {
        int          device_id  = 0;       // GPU device this worker is bound to
        gpuStream_t  stream      = nullptr;
        uint32_t*    d_hist      = nullptr;  // 256 uint32_t  (fixed)
        ANSStreamHeader* d_streams = nullptr; // ANS_LANES entries (fixed)
        // Prefix-sum offsets for stream compaction (ANS_LANES+1 uint32_t).
        uint32_t*    d_offsets   = nullptr;  // fixed
        // Device-side encode tables (uploaded per-tensor from host)
        uint32_t*    d_norm_freq = nullptr;  // 256 uint32_t  (fixed)
        uint32_t*    d_cum_freq  = nullptr;  // 257 uint32_t  (fixed)
        uint32_t*    d_x_max     = nullptr;  // 256 uint32_t  (fixed)
    };

    // ---------------------------------------------------------------
    // Shared large dynamic GPU buffers.
    // All workers contend for this single set under shared_bufs_mtx.
    // Because each GPU kernel call is fully synchronous (stream sync
    // before release), only one worker actually uses GPU memory at a
    // time, so sharing eliminates the N-worker memory amplification.
    // ---------------------------------------------------------------
    struct SharedGPUBufs {
        uint8_t*  d_enc_buf     = nullptr;  size_t d_enc_cap     = 0;
        uint8_t*  d_compact_buf = nullptr;  size_t d_compact_cap = 0;
        uint8_t*  d_masked_buf  = nullptr;  size_t d_masked_cap  = 0;
        uint8_t*  d_sidecar_buf = nullptr;  size_t d_sidecar_cap = 0;
    };
    SharedGPUBufs   shared_bufs_;
    std::mutex      shared_bufs_mtx_;

    int rank_ = -1;  // global_rank from logger
    std::vector<WorkerGPU> workers_;

    // Helper: prefix string for log messages carrying rank and context.
    std::string lp(const std::string& tag) const {
        return "[ANS][rank=" + std::to_string(rank_) + "][" + tag + "] ";
    }

    void init_worker_gpu(WorkerGPU& w);
    void free_worker_gpu(WorkerGPU& w);
    void free_shared_bufs();
    void ensure_enc_buf    (SharedGPUBufs& b, size_t required);
    void ensure_compact_buf(SharedGPUBufs& b, size_t required);
    void ensure_masked_buf (SharedGPUBufs& b, size_t masked_cap, size_t sidecar_cap);

    // Compress one device tensor. Caller must hold shared_bufs_mtx_.
    CompressedTensor compress_tensor(const std::string&  name,
                                     const void*         d_data,
                                     size_t              byte_size,
                                     DataType            dtype,
                                     size_t              num_elements,
                                     const ANSFreqTable& table,
                                     bool                owns_table,
                                     const std::string&  anchor_name,
                                     WorkerGPU&          w,
                                     SharedGPUBufs&      b);

    // Extract low mantissa bytes. Caller must hold shared_bufs_mtx_.
    SidecarInfo extract_mantissa(WorkerGPU& w, SharedGPUBufs& b,
                                 const void* d_data,
                                 size_t      n_elements,
                                 DataType    dtype);

    // ---------------------------------------------------------------
    // Per-rank combined file writers (called after all tensors are done).
    // ---------------------------------------------------------------

    // Per-rank ANS states file: all CompressedTensors concatenated.
    // One table entry per tensor in `cts`; one owned table entry in `tables`
    // keyed by anchor_name (or the tensor's own name for singletons).
    struct SidecarEntry {
        std::string  name;
        DataType     dtype;
        size_t       n_elements;
        SidecarInfo  si;
    };
    struct TableEntry {
        std::string  anchor_name;  // key used by non-anchor tensors to look up
        ANSFreqTable table;
    };

    void write_rank_states_file(const std::string&                path,
                                int                               rank,
                                const std::vector<CompressedTensor>& cts) const;

    void write_rank_mantissa_file(const std::string&                  path,
                                  int                                 rank,
                                  const std::vector<SidecarEntry>&    sidecars) const;

    void write_rank_tables_file(const std::string&               path,
                                int                              rank,
                                const std::vector<TableEntry>&   tables) const;
};

} // namespace checkers
