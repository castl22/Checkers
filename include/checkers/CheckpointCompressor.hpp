#pragma once
// Orchestrates one online checkpoint: critical-path device snapshot into the
// StagingPool, then asynchronous per-stream compression + I/O on a background
// worker pool.  One process-wide instance, created by initialize_context.

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "checkers/api.hpp"            // CompressionConfig, CodecId
#include "checkers/StagingPool.hpp"
#include "checkers/TensorResource.hpp" // TensorCategory, tracked_tensor_category_count

namespace checkers {

class MemoryManager;
class RankLogger;

// ---- metrics (plain C++, converted to a py::dict on the Python thread) ----
struct StreamMetrics {
    uint64_t n_tensors        = 0;
    uint64_t orig_bytes       = 0;
    uint64_t compressed_bytes = 0;
    double   sq_err           = 0.0;   // SNR accumulators (lossy codecs only)
    double   signal           = 0.0;
    bool     lossless         = true;
    double   compute_ms       = 0.0;   // GPU transform/quant (+ entropy stage in Phase A)
    double   d2h_ms           = 0.0;
    double   io_ms            = 0.0;
};

struct CompressionMetrics {
    bool     pending          = false;
    double   critical_path_ms = 0.0;   // the only part that stalls training
    double   compute_ms       = 0.0;
    double   d2h_ms           = 0.0;
    double   io_ms            = 0.0;
    double   wall_ms          = 0.0;
    uint64_t n_tensors        = 0;
    uint64_t orig_bytes       = 0;
    uint64_t compressed_bytes = 0;
    double   sq_err           = 0.0;
    double   signal           = 0.0;
    bool     lossless         = true;
    std::array<StreamMetrics, tracked_tensor_category_count> per_stream{};

    // extra memory this feature costs (reported so error/convergence studies
    // can weigh ratio against footprint)
    size_t  staging_device_bytes        = 0;
    size_t  pinned_host_bytes           = 0;
    size_t  codec_scratch_device_bytes  = 0;
    size_t  peak_device_overhead_bytes  = 0;
    int64_t measured_free_vram_delta_b  = 0;   // free-VRAM drop measured at allocate()
    std::string staging_mode;
    std::string output_dir;
    uint32_t    ring_depth   = 0;
    uint32_t    workers      = 0;
    double      error_bound  = 0.0;   // representative (first targeted stream)
    std::array<double, tracked_tensor_category_count> error_bounds{};  // per stream
    uint32_t    targets_mask = 0;
};

class CheckpointCompressor {
public:
    static CheckpointCompressor& instance();

    void configure(const CompressionConfig& cfg,
                   std::shared_ptr<RankLogger> logger,
                   int global_rank);

    // size + allocate the staging pool from the largest tensor across the
    // staged categories currently registered in `mgr`.
    void allocate(MemoryManager& mgr);

    // critical-path snapshot + launch (blocking joins the background job).
    // error_bounds is indexed by tensor_category_index (per-stream eps).
    CompressionMetrics run(MemoryManager&     mgr,
                           const std::string& output_dir,
                           const std::array<double, tracked_tensor_category_count>& error_bounds,
                           uint32_t           targets,
                           bool               blocking);

    CompressionMetrics join();        // wait for a pending non-blocking run
    bool has_pending() const { return pending_.load(); }

    const CompressionConfig& config() const { return cfg_; }
    void reset();

private:
    CheckpointCompressor() = default;
    ~CheckpointCompressor();
    CheckpointCompressor(const CheckpointCompressor&)            = delete;
    CheckpointCompressor& operator=(const CheckpointCompressor&) = delete;

    struct WorkItem;                  // defined in the .cpp
    void finalize_and_store(std::shared_ptr<std::vector<WorkItem>> work,
                            std::shared_ptr<CompressionMetrics>    metrics,
                            std::string                            output_dir);

    // Uniform scalar quant codec.
    //   PerGroupRms : linear values, step D_g = 2*eps*RMS_g per 256-group
    //                 -- Phase B, weights (near-Gaussian; adapts to local scale).
    //   GlobalRms   : linear values, one step D = 2*eps*RMS(tensor)
    //                 -- Phase C aggressive/opt-in, exp_avg_sq.  Heavy tail sets
    //                 RMS; the concentrated bulk collapses to index 0 (~25x) but
    //                 the restart loss then hits a divergence cliff.
    //   LogPwRel    : quantize y = log(max(v, 1e-30)) with one global step
    //                 D = 2*log(1+eps); decoder does v_hat = exp(q*D).  Hard
    //                 per-element bound |v_hat/v - 1| <= eps, no zeroing
    //                 -- Phase C default, exp_avg_sq (restart converges).
    // Fully populates wi and returns true; false => caller falls back to
    // raw+zstd.  GPU work is serialized on codec_mtx_.
    enum class PgStepMode { PerGroupRms, GlobalRms, LogPwRel };
    bool codec_pergroup(WorkItem& wi, StagingPool::Slot& sl,
                        int worker_id, double error_bound, PgStepMode mode);

    // exp_avg: capped TT-SVD (TtLinalg) -> EQ error-bounded quant of the
    // truncation residual (single global step from error_bound) -> zstd.
    // Cores are stored fp16 in the aux blob behind a TtHeader.  GPU work is
    // serialized on codec_mtx_ (shares scratch_ + the rocBLAS handle).
    bool codec_tt_eq(WorkItem& wi, StagingPool::Slot& sl,
                     int worker_id, double error_bound);

    // Shared device scratch for the lossy codecs (one set, mutex-guarded --
    // the GPU is one device, so serializing the kernels costs nothing and the
    // per-tensor zstd stays parallel across workers).
    struct CodecScratch {
        float*    fp32       = nullptr;   // [expand_elems]  bf16/fp16 -> fp32
        int16_t*  q          = nullptr;   // [max_elems]     quant idx (then zigzag u16)
        float*    rms        = nullptr;   // [max_groups]
        float*    step       = nullptr;   // [max_groups]
        uint32_t* over_idx   = nullptr;   // [over_cap]
        float*    over_val   = nullptr;   // [over_cap]
        // GPU bit-pack stage (replaces the CPU zstd payload).  packed is sized
        // for the 16-bits-per-code worst case, i.e. the same bytes the u16 code
        // array takes, so a dense tensor can never overrun it.
        uint8_t*  w          = nullptr;   // [max_blocks]    per-block bit width
        uint64_t* off        = nullptr;   // [max_blocks]    per-block byte offset
        uint8_t*  packed     = nullptr;   // [max_elems*2]   packed bitstream
        size_t    max_blocks = 0;
        uint32_t* over_count = nullptr;   // [1]
        size_t    max_elems  = 0;
        size_t    max_groups = 0;
        size_t    over_cap   = 0;
        size_t    bytes      = 0;         // total device footprint
    };
    void alloc_codec_scratch(size_t max_elems, size_t expand_elems, uint32_t group);
    void free_codec_scratch();

    CompressionConfig            cfg_{};
    std::shared_ptr<RankLogger>  logger_;
    int                          global_rank_ = -1;
    int                          device_id_   = 0;

    StagingPool                  pool_;
    CodecScratch                 scratch_;
    std::mutex                   codec_mtx_;
    size_t                       baseline_free_vram_ = 0;
    int64_t                      staging_vram_delta_ = 0;
    std::string                  staging_mode_;

    std::mutex                   job_mtx_;
    std::thread                  finalizer_;
    std::atomic<bool>            pending_{false};
    CompressionMetrics           last_metrics_{};
    // A non-blocking save's failure, raised by the next join() or run()
    // (2026-11-05: it used to be logged and dropped).
    std::exception_ptr           finalize_error_;
};

// Build the nested Python dict from a metrics snapshot (call with the GIL held).
py::dict metrics_to_pydict(const CompressionMetrics& m);

// Read compressed stream files back from disk and refill the live tensors
// registered in `mgr`.  Streams whose .cch file is absent are reported under
// "missing" so the caller can fill them from a framework-native checkpoint.
// Defined in src/Restart.cpp.
py::dict restart_apply(MemoryManager&              mgr,
                       const std::string&          input_dir,
                       uint32_t                    targets,
                       std::shared_ptr<RankLogger> logger,
                       int                         global_rank);

} // namespace checkers
