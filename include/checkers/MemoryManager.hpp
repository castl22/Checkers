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

#include "TensorResource.hpp"
#include "logging.hpp"

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

// ------------------------------------------------------------------ //
//  Device-side metadata record
//  Mirrors TensorMetadata for tensors that are laid out on the GPU so
//  RAJA kernels can read them without host round-trips.
// ------------------------------------------------------------------ //
struct alignas(16) DeviceTensorRecord {
    float* d_data;          // Base data pointer (or nullptr if fragmented)
    uint32_t* d_histogram;     // Pre-allocated histogram buffer
    float* d_moments;       // Pre-allocated moments buffer [mean,var,skew,kurt]
    float* d_minmax;        // Pre-allocated minmax buffer [min, max]
    size_t    num_elements;
    size_t    histogram_bins;
    uint8_t   data_type;       // cast of DataType enum
    uint8_t   contiguity;      // cast of Contiguity enum
    uint8_t   _pad[6];
};

// ------------------------------------------------------------------ //
//  Per-tensor buffer plan (host side, computed dynamically per batch)
// ------------------------------------------------------------------ //
struct BufferPlan {
    size_t histogram_bytes; // bins * sizeof(uint32_t)
    size_t moments_bytes;   // 4 * sizeof(float)
    size_t minmax_bytes;    // 2 * sizeof(float)
    size_t total_bytes;     // aligned sum
    size_t slab_offset;     // byte offset into its local batch slab allocation
};

// ------------------------------------------------------------------ //
//  MemoryManager
// ------------------------------------------------------------------ //
class MemoryManager {
public:
    // Singleton
    static MemoryManager& instance();

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

    // Reporting
    void report_memory_usage() const;
    void set_skipped_count(size_t skipped_count);
    void note_discovery(const TensorMetadata& meta, double discovery_ms);

    // ---- Configuration ---- //
    void set_default_histogram_bins(size_t bins) { default_histogram_bins_ = bins; }
    void set_alignment_bytes(size_t align)        { alignment_bytes_ = align; }

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
    std::vector<DeviceTensorRecord*> batch_device_records_;   // Tracks device array block addresses per batch segment

    // Device-side lookup cache tracking properties for legacy downstream hooks
    DeviceTensorRecord* d_records_    = nullptr;
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
    MemoryManager(const MemoryManager&)            = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
};

} // namespace checkers