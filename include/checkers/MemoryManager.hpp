#pragma once
// include/checkers/MemoryManager.hpp
//
// Batch-Pipelined Memory Tracker.
//
// Foreground Process (CPU):
//   Discovers active model weight tensors from the engine and registers them.
//   Accumulates metadata in a local queue until a batch threshold (e.g., 10 tensors) 
//   is hit, then hands off processing to the background system.
//
// Background Pipeline Process (GPU Stream Async):
//   Processes tensors in grouped clusters (batches of 10) to minimize memory fragmentation.
//   * Pass 1 (Scan & Plan) evaluates contiguity and calculates stats tracking byte layouts.
//   * Pass 2 (Allocate & Map) allocates ONE dense local memory slab per batch, carving
//     out structural tracking offsets instantly.
//   * Instantly dispatches a backend RAJA kernel to clear moments/minmax spaces and 
//     pokes the VRAM address to lock the hardware allocation pages safely from driver unmapping.
//
// Nothing computes statistics here -- that is the job of the analysis
// kernel layer that comes later.
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "TensorResource.hpp"
#include "logging.hpp"
#include "TensorFingerprint.hpp"

#include <RAJA/RAJA.hpp>
#include <umpire/Umpire.hpp>

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>

#define CHECKERS_EXPORT __attribute__((visibility("default")))

namespace checkers {

inline void device_synchronize() {
#if defined(RAJA_ENABLE_HIP)
    (void)hipDeviceSynchronize();
#elif defined(RAJA_ENABLE_CUDA)
    (void)cudaDeviceSynchronize();
#endif
}

// ------------------------------------------------------------------ //
//  Device-side metadata record
//  Mirrors TensorMetadata for tensors that are laid out on the GPU so
//  RAJA kernels can read them without host round-trips.
// ------------------------------------------------------------------ //
struct DeviceTensorRecord {
    const char* name;
    void* d_ptr;
    uint32_t* d_histogram;
    float* d_raw_moments;
    TensorStatistics* d_statistics;
    TensorFingerprint* d_fingerprint;
    float* d_minmax;
    size_t num_elements;
    uint32_t histogram_bins;
    uint8_t data_type;
    uint8_t contiguity;
};

// ------------------------------------------------------------------ //
//  Per-tensor buffer plan (host side, computed dynamically per batch)
// ------------------------------------------------------------------ //
struct BufferPlan {
    size_t histogram_bytes;   // bins * sizeof(uint32_t)
    size_t moments_bytes;     // 4 * sizeof(float)
    size_t minmax_bytes;      // 2 * sizeof(float)
    size_t fingerprint_bytes; // sizeof(TensorFingerprint)
    size_t total_bytes;       // aligned sum
    size_t slab_offset;       // byte offset into its local batch slab allocation
};

// ------------------------------------------------------------------ //
//  MemoryManager
// ------------------------------------------------------------------ //
class MemoryManager {
public:
    // Singleton
    static MemoryManager& instance();
    std::shared_ptr<RankLogger> get_logger() { return logger_; }
    void set_logger(std::shared_ptr<RankLogger> logger) { logger_ = logger; }

    void reset();
    void begin_pipeline(std::shared_ptr<RankLogger> logger,
                        size_t background_threads = 1,
                        size_t batch_size = 1);
    void finalize_pipeline();

    // ---- Registration & Pipelined Ingestion ---- //

    // Core pipelined entry-point: Logs metadata and appends to the current 10-tensor batch tracking queue.
    void submit_tensor(const std::string&  name,
                       void* existing_device_ptr,
                       size_t              byte_size,
                       TensorMetadata      meta);

    // Forces immediate execution of any trailing, unsubmitted tensors left in the batch queue.
    void flush_pipeline();

    // Remove a tensor from tracking (does NOT free the original tensor data).
    void untrack_tensor(const std::string& name);


    // ---- Queries ---- //
    size_t tensor_count() const;
    size_t discovered_count() const;
    size_t skipped_count() const;
    size_t get_slab_count() const;          // Returns the total number of allocated batch slabs
    size_t get_total_tensor_bytes() const;   // Sum of all tracked model weights sizes
    size_t get_total_buffer_bytes() const;   // Sum of all active analysis metrics sizes
    double get_total_pass1_ms() const;
    double get_total_allocation_ms() const;
    double get_total_kernel_ms() const;
    size_t batch_count() const;
    std::array<TensorCategoryStats, tracked_tensor_category_count> get_category_stats() const;
    DeviceTensorRecord* device_records() const { return d_records_; }
    DeviceTensorRecord* d_records_;
    size_t record_count() const;

    // Reporting
    void report_memory_usage() const;
    void set_skipped_count(size_t skipped_count);
    void note_discovery(const TensorMetadata& meta, double discovery_ms);

    // ---- Configuration ---- //
    void set_default_histogram_bins(size_t bins) { default_histogram_bins_ = bins; }
    void set_alignment_bytes(size_t align)        { alignment_bytes_ = align; }
    void build_global_descriptor_array();
    void register_param(const std::string& name,
                        py::object param,
                        bool allow_pointer_refresh = true);
    void register_frozen_param(const std::string& name);
    py::object get_param(const std::string& name) const;
    bool has_param(const std::string& name) const;
    bool can_refresh_param(const std::string& name) const;

    std::vector<DeviceTensorRecord>& host_records() {
        return global_host_records_;
    }

    const std::vector<DeviceTensorRecord>& host_records() const {
        return global_host_records_;
    }
    const std::string& get_name_from_index(size_t i) const;
    const std::vector<Fingerprint>& get_fingerprints() const {
        return fingerprint_stage_;
    }

    std::vector<Fingerprint>& get_fingerprints_mut() {
        return fingerprint_stage_;
    }

private:

    MemoryManager();
    ~MemoryManager();

    struct PendingTensor {
        std::string name;
        void* device_ptr = nullptr;
        size_t byte_size = 0;
        TensorMetadata meta;
    };

    // Internal Pipeline Processing Engine
    void worker_loop();
    void process_batch(std::vector<PendingTensor> items);
    void drain_pending_queue_locked(std::vector<PendingTensor>& items);

    // Helpers
    BufferPlan compute_buffer_plan(const TensorMetadata& meta) const;
    void       check_and_fix_contiguity(TensorResource& tr) const;
    size_t     align_up(size_t v, size_t align) const {
        return (v + align - 1) & ~(align - 1);
    }

    // Registry & Thread Safety Structures
    mutable std::shared_mutex                                         registry_mutex_;
    mutable std::mutex            gpu_ops_mutex_;
    std::unordered_map<std::string, std::unique_ptr<TensorResource>> registry_;
    std::vector<std::string>                                         ordered_names_;   // Global stable iteration tracking order

    mutable std::mutex            queue_mutex_;
    std::condition_variable       queue_cv_;
    std::deque<PendingTensor>     pending_tensors_;
    std::vector<std::thread>      worker_threads_;
    bool                          accepting_submissions_ = false;
    bool                          stop_worker_ = false;

    // Batch Allocation Pipelines
    std::vector<BufferPlan>         buffer_plans_;            // Shared repository of calculated tensor layouts
    std::vector<void*>              active_slabs_;            // Managed list of all allocated local device slabs

    // Device-side lookup cache tracking properties for legacy downstream hooks
    size_t              record_count_ = 0;

    // Umpire Resources
    umpire::ResourceManager& rm_;
    umpire::Allocator        device_allocator_;   // GPU device pool allocation hook
    umpire::Allocator        host_allocator_;     // Pinned host pool hook
    std::shared_ptr<RankLogger> logger_;

    // Constants Configurations
    size_t default_histogram_bins_ = 256;
    size_t alignment_bytes_        = 256; // 256-byte alignment requirement for target GPU memory access optimization
    size_t discovered_count_       = 0;
    size_t skipped_count_          = 0;
    size_t processed_batch_count_  = 0;
    size_t background_thread_count_ = 1;
    size_t batch_size_             = 1;
    std::array<TensorCategoryStats, tracked_tensor_category_count> category_stats_{};
    double total_pass1_ms_         = 0.0;
    double total_allocation_ms_    = 0.0;
    double total_kernel_ms_        = 0.0;

    // Disallow duplication patterns
    bool global_descriptor_initialized_ = false;
    bool global_descriptor_built_ = false;
    MemoryManager(const MemoryManager&)            = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    void update_global_descriptor_entry(const std::string& name);

    std::vector<DeviceTensorRecord> global_host_records_;
    size_t                          global_capacity_ = 0;

    struct ParamBinding {
        py::object object;
        bool allow_pointer_refresh = false;
    };

    std::unordered_map<std::string, ParamBinding> param_registry_;
    std::vector<Fingerprint> fingerprint_stage_;
    Fingerprint* d_fingerprint_buffer_ = nullptr;
    size_t fingerprint_capacity_ = 0;
};

} // namespace checkers