#include "checkers/MemoryManager.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#if defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

// Use visibility attribute for GCC/Clang/HIPCC
#define CHECKERS_API __attribute__ ((visibility ("default")))

namespace checkers {

MemoryManager::MemoryManager() 
    : rm_(umpire::ResourceManager::getInstance()),
      device_allocator_(rm_.getAllocator("DEVICE")),
      host_allocator_(rm_.getAllocator("HOST")) 
{
    // Initialize your allocator hooks here
}

MemoryManager& MemoryManager::instance() {
    static MemoryManager inst; // This calls the constructor above
    return inst;
}

// Destructor cleans up any dynamic records arrays or batch slabs
MemoryManager::~MemoryManager() {
    finalize_pipeline();
    reset();
}

void MemoryManager::reset() {
    finalize_pipeline();

    std::unique_lock queue_lock(queue_mutex_);
    pending_tensors_.clear();
    queue_lock.unlock();

    std::unique_lock registry_lock(registry_mutex_);
    for (void* slab_ptr : active_slabs_) {
        if (slab_ptr) device_allocator_.deallocate(slab_ptr);
    }

    registry_.clear();
    ordered_names_.clear();
    buffer_plans_.clear();
    active_slabs_.clear();
    d_records_ = nullptr;
    record_count_ = 0;
    discovered_count_ = 0;
    skipped_count_ = 0;
    processed_batch_count_ = 0;
    background_thread_count_ = 1;
    batch_size_ = 1;
    category_stats_ = {};
    total_pass1_ms_ = 0.0;
    total_allocation_ms_ = 0.0;
    total_kernel_ms_ = 0.0;
    param_registry_.clear();
    fingerprint_stage_.clear();
    logger_.reset();
}

void MemoryManager::build_global_descriptor_array()
{
    // ------------------------------------------------------------
    // GUARD: ensure we only build once per context
    // ------------------------------------------------------------
    if (global_descriptor_built_) {
        return;
    }

    std::unique_lock lock(registry_mutex_);

    // ------------------------------------------------------------
    // Determine number of tensors
    // ------------------------------------------------------------
    record_count_ = registry_.size();

    if (record_count_ == 0) {
        global_descriptor_built_ = true;
        return;
    }

    ordered_names_.clear();
    ordered_names_.reserve(registry_.size());

    for (const auto& kv : registry_) {
        ordered_names_.push_back(kv.first);
    }

    std::sort(ordered_names_.begin(), ordered_names_.end());

    // ------------------------------------------------------------
    // Allocate GPU persistent descriptor array
    // ------------------------------------------------------------
    d_records_ = static_cast<DeviceTensorRecord*>(
        device_allocator_.allocate(
            record_count_ * sizeof(DeviceTensorRecord)
        )
    );

    // IMPORTANT: zero-init avoids garbage pointers on GPU
    auto& rm = umpire::ResourceManager::getInstance();
    rm.memset(d_records_, 0, record_count_ * sizeof(DeviceTensorRecord));

    // ------------------------------------------------------------
    // Build host-side staging buffer (ONE TIME ONLY)
    // ------------------------------------------------------------
    std::vector<DeviceTensorRecord> host_records;
    host_records.reserve(record_count_);

    for (const auto& name : ordered_names_) {

        auto it = registry_.find(name);
        if (it == registry_.end()) {
            continue;
        }

        const TensorResource& tr = *it->second;
        const TensorMetadata& meta = tr.get_meta();

        DeviceTensorRecord rec{};

        rec.name = name.c_str();
        rec.d_ptr = tr.get_ptr();
        rec.d_histogram  = meta.d_histogram;
        rec.d_raw_moments = meta.d_raw_moments;
        rec.d_minmax     = meta.d_minmax;

        rec.num_elements  = meta.num_elements;
        rec.histogram_bins = meta.histogram_bins;
        rec.data_type     = static_cast<uint8_t>(meta.data_type);
        rec.contiguity    = static_cast<uint8_t>(meta.contiguity);

        host_records.push_back(rec);
    }

    lock.unlock();

    // ------------------------------------------------------------
    // Copy to GPU
    // ------------------------------------------------------------
 #if defined(RAJA_ENABLE_HIP)

    hipError_t err_init = hipMemcpy(
        d_records_,
        host_records.data(),
        record_count_ * sizeof(DeviceTensorRecord),
        hipMemcpyHostToDevice);
    if (err_init != hipSuccess) {
        throw std::runtime_error(
            std::string("hipMemcpy failed for global descriptors: ") + hipGetErrorString(err_init));
    }

#elif defined(RAJA_ENABLE_CUDA)

    cudaError_t err_init = cudaMemcpy(
        d_records_,
        host_records.data(),
        record_count_ * sizeof(DeviceTensorRecord),
        cudaMemcpyHostToDevice);
    if (err_init != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy failed for global descriptors: ") + cudaGetErrorString(err_init));
    }

#else

    throw std::runtime_error("No GPU backend enabled for descriptor copy");

#endif
}

void MemoryManager::begin_pipeline(std::shared_ptr<RankLogger> logger,
                                   size_t background_threads,
                                   size_t batch_size) {
    finalize_pipeline();
    logger_ = std::move(logger);
    {
        std::lock_guard lock(queue_mutex_);
        accepting_submissions_ = true;
        stop_worker_ = false;
        background_thread_count_ = std::max<size_t>(1, background_threads);
        batch_size_ = std::max<size_t>(1, batch_size);
    }
    worker_threads_.clear();
    worker_threads_.reserve(background_thread_count_);
    for (size_t worker_index = 0; worker_index < background_thread_count_; ++worker_index) {
        worker_threads_.emplace_back(&MemoryManager::worker_loop, this);
    }
}

void MemoryManager::finalize_pipeline() {
    {
        std::lock_guard lock(queue_mutex_);
        if (!accepting_submissions_ && worker_threads_.empty()) {
            return;
        }
        accepting_submissions_ = false;
        stop_worker_ = true;
    }
    queue_cv_.notify_all();
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();
}

void MemoryManager::submit_tensor(const std::string& name, void* existing_device_ptr, size_t byte_size, TensorMetadata meta) {
    // 1. Core structural population (Same host logic)
    meta.name         = name;
    if (meta.element_size == 0) {
        meta.element_size = dtype_element_size(meta.data_type);
    }
    if (meta.element_size > 0) {
        meta.num_elements = byte_size / meta.element_size;
    }
    meta.byte_size    = byte_size;

    if (meta.strides.empty() && !meta.shape.empty()) {
        meta.strides.resize(meta.shape.size());
        meta.strides.back() = 1;
        for (int i = static_cast<int>(meta.shape.size()) - 2; i >= 0; --i) {
            meta.strides[i] = meta.strides[i + 1] * static_cast<ptrdiff_t>(meta.shape[i + 1]);
        }
    }
    meta.contiguity = meta.shape.empty() ? Contiguity::Scalar : Contiguity::Contiguous;

    PendingTensor item;
    item.name = name;
    item.device_ptr = existing_device_ptr;
    item.byte_size = byte_size;
    item.meta = std::move(meta);

    {
        std::lock_guard lock(queue_mutex_);
        pending_tensors_.push_back(std::move(item));
        ++discovered_count_;
    }
    queue_cv_.notify_one();
}

void MemoryManager::note_discovery(const TensorMetadata& meta, double discovery_ms) {
    const size_t category_idx = tensor_category_index(meta.category);
    if (category_idx >= tracked_tensor_category_count) {
        return;
    }

    std::unique_lock lock(registry_mutex_);
    TensorCategoryStats& stats = category_stats_[category_idx];
    ++stats.tensor_count;
    stats.tensor_bytes += meta.byte_size;
    stats.discovery_ms += discovery_ms;
    if (!stats.has_sample) {
        stats.sample_name = meta.name;
        stats.sample_dtype = meta.data_type;
        stats.sample_shape = meta.shape;
        stats.sample_bytes = meta.byte_size;
        stats.has_sample = true;
    }
}

void MemoryManager::flush_pipeline() {
    finalize_pipeline();
}

void MemoryManager::worker_loop() {
    while (true) {
        std::vector<PendingTensor> ready_items;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] {
                return !pending_tensors_.empty() || !accepting_submissions_ || stop_worker_;
            });

            drain_pending_queue_locked(ready_items);

            if (ready_items.empty() && !accepting_submissions_) {
                break;
            }
        }

        if (!ready_items.empty()) {
            process_batch(std::move(ready_items));
        }
    }
}

void MemoryManager::drain_pending_queue_locked(std::vector<PendingTensor>& items) {
    const size_t target_batch_size = std::max<size_t>(1, batch_size_);
    while (!pending_tensors_.empty() && items.size() < target_batch_size) {
        items.push_back(std::move(pending_tensors_.front()));
        pending_tensors_.pop_front();
    }
}

void MemoryManager::process_batch(std::vector<PendingTensor> items) {
    if (items.empty()) {
        return;
    }

    const auto pass1_start = std::chrono::steady_clock::now();

    const size_t batch_size = items.size();

    std::vector<BufferPlan> batch_plans;

    // In C++, std::vector::reserve()` pre-allocates memory for a vector to hold at 
    // least a specified number of elements, preventing frequent, costly reallocations.
    batch_plans.reserve(batch_size);

    size_t batch_slab_bytes = 0;

    std::vector<std::string> batch_names;
    batch_names.reserve(batch_size);

    std::vector<TensorResource*> batch_resources;
    batch_resources.reserve(batch_size);

    // ==========================================================
    // STEP 1: MOVE TENSORS INTO REGISTRY
    // ==========================================================
    { // lock the stuff
        std::unique_lock lock(registry_mutex_);
        for (auto& item : items) {
            
            auto resource = std::make_unique<TensorResource>(
                item.device_ptr, 
                item.byte_size, 
                std::move(item.meta), 
                device_allocator_);

            TensorResource* resource_ptr = resource.get();

            ordered_names_.push_back(item.name);

            batch_names.push_back(item.name);

            registry_[item.name] = std::move(resource);

            batch_resources.push_back(resource_ptr);
        }
    } // release lock

    for (size_t index = 0; index < batch_size; ++index) {
        TensorResource& tr = *batch_resources[index];
        TensorMetadata& meta = tr.get_meta_mut(); // mutable/writable, no lock

        const auto contiguity_start = std::chrono::steady_clock::now();

        check_and_fix_contiguity(tr);
        if (meta.histogram_bins == 0) meta.histogram_bins = default_histogram_bins_;

        if (meta.contiguity == Contiguity::NonContiguous) {
            tr.make_contiguous();
        }
        const auto contiguity_end = std::chrono::steady_clock::now();

        const auto planning_start = contiguity_end;

        BufferPlan plan = compute_buffer_plan(meta);

        const auto planning_end = std::chrono::steady_clock::now();

        meta.buffer_histogram_bytes = plan.histogram_bytes;
        meta.buffer_moments_bytes   = plan.moments_bytes;
        meta.buffer_minmax_bytes    = plan.minmax_bytes;
        meta.buffer_total_bytes     = plan.total_bytes;

        const size_t category_idx = tensor_category_index(meta.category);
        if (category_idx < tracked_tensor_category_count) {
            std::unique_lock lock(registry_mutex_);
            TensorCategoryStats& stats = category_stats_[category_idx];
            stats.contiguous_ms += std::chrono::duration<double, std::milli>(contiguity_end - contiguity_start).count();
            stats.planning_ms += std::chrono::duration<double, std::milli>(planning_end - planning_start).count();
            stats.planned_buffer_bytes += plan.total_bytes;
        }

        batch_plans.push_back(plan);
        batch_slab_bytes += plan.total_bytes;
    }

    if (batch_slab_bytes == 0) {
        return;
    }

    const auto pass1_end = std::chrono::steady_clock::now();

    // ==========================================================
    // PASS 2: ALLOCATE ONE SLAB & PARCEL OUT OFFSETS FOR THIS BATCH
    // ==========================================================

    // ==========================================================
    // STEP 3: ALLOCATE SLAB
    // ==========================================================
    // 1. Removed heavy global gpu_ops_mutex_ to prevent multi-rank driver stalling.

    const auto alloc_start = std::chrono::steady_clock::now();

    // 2. Allocation happens concurrently across ranks now.
    void* batch_slab_base = device_allocator_.allocate(batch_slab_bytes);

    {
        // 3. Isolated lock scope strictly for registering the active slab tracking.
        std::unique_lock lock(registry_mutex_);
        active_slabs_.push_back(batch_slab_base);
    }
    // Heavy synchronous Umpire memset completely deleted.
    // Zeroing is now handled out of the critical path by the parallel RAJA kernel below.

    size_t cursor = 0;
    // turn it into a byte-by-byte addressable pointer.
    char* slab_bytes_ptr = static_cast<char*>(batch_slab_base);

    auto& rm = umpire::ResourceManager::getInstance();
    rm.memset(batch_slab_base, 0, batch_slab_bytes);

    // size_t cursor = 0;
    // // tunr it into a byte-by-byte addressable pointer.
    // char* slab_bytes_ptr = static_cast<char*>(batch_slab_base);

    // ==========================================================
    // STEP 4: BUILD HOST DESCRIPTORS (TEMPORARY)
    // ==========================================================
    std::vector<DeviceTensorRecord> host_records(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        TensorResource& tr = *batch_resources[i];
        TensorMetadata& meta = tr.get_meta_mut();
        BufferPlan& plan = batch_plans[i];

        // You add your current byte offset to the start address of the master slab. This calculates the exact physical GPU memory address where the histogram data should live
        plan.slab_offset = cursor;
        // Treat the GPU memory address at this exact location as an array of 32-bit unsigned integers.
        // it is 32 because we want a swwt-spot between how many numbers we can represent, how mnay atomicAdds (moden GPUs can handle 32) and memory bandwidth.
        // If 32 threads in a warp atomicAdd to the same memory location, the operation is serialized and takes 32 times longer than an uncontended write
        meta.d_histogram = reinterpret_cast<uint32_t*>(slab_bytes_ptr + cursor);
        // Once you've allocated space for the histogram, you need to slide the cursor forward so the next buffer (the moments buffer) doesn't overwrite it.
        // However, you don't just add the raw size (plan.histogram_bytes). Instead, you pass it through align_up
        // GPUs are hyper-optimized for speed, but they have a strict rule: they want data arrays to start at specific hardware boundaries (usually multiples of 64, 128, or 256 bytes). If an array starts at an unaligned address, the GPU has to make multiple slow, inefficient memory trips to read a single value.
        // If your histogram takes up 100 bytes, and your hardware alignment requirement (alignment_bytes_) is 64 bytes.
        //
        // The next perfect hardware boundary is 128 bytes.
        //
        // align_up(100, 64) will return 128.
        //
        // This leaves 28 bytes of empty "padding" space, ensuring that the next buffer (d_raw_moments) starts exactly at a clean 128-byte boundary where the GPU can access it at maximum speed.
        cursor += align_up(plan.histogram_bytes, alignment_bytes_);
        
        meta.d_raw_moments   = reinterpret_cast<float*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.moments_bytes, alignment_bytes_);
        
        meta.d_minmax    = reinterpret_cast<float*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.minmax_bytes, alignment_bytes_);

        meta.d_fingerprint = reinterpret_cast<TensorFingerprint*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.fingerprint_bytes, alignment_bytes_);

        DeviceTensorRecord& rec = host_records[i];
        rec.d_ptr          = tr.get_ptr();
        // Populate the device tensor record with the allocated GPU memory addresses
        rec.d_histogram    = meta.d_histogram;
        rec.d_raw_moments  = meta.d_raw_moments;
        rec.d_minmax       = meta.d_minmax;
        rec.d_fingerprint  = meta.d_fingerprint;

        rec.num_elements   = meta.num_elements;
        rec.histogram_bins = meta.histogram_bins;
        rec.data_type      = static_cast<uint8_t>(meta.data_type);
        rec.contiguity     = static_cast<uint8_t>(meta.contiguity);

    }

    {
        std::unique_lock lock(registry_mutex_);
        for (const auto& plan : batch_plans) {
            buffer_plans_.push_back(plan);
        }
    }

    // ==========================================================
    // STEP 5: ALLOCATE & UPLOAD BATCH DESCRIPTOR ARRAY
    // ==========================================================
    DeviceTensorRecord* d_batch_recs = static_cast<DeviceTensorRecord*>(
        device_allocator_.allocate(batch_size * sizeof(DeviceTensorRecord)));

    // Portable, error-checked host-to-device descriptor upload
#if defined(RAJA_ENABLE_HIP)
    hipError_t copy_status = hipMemcpy(
        d_batch_recs,
        host_records.data(),
        batch_size * sizeof(DeviceTensorRecord),
        hipMemcpyHostToDevice);
    if (copy_status != hipSuccess) {
        throw std::runtime_error(
            std::string("hipMemcpy failed for batch descriptors: ") + hipGetErrorString(copy_status));
    }
#elif defined(RAJA_ENABLE_CUDA)
    cudaError_t copy_status = cudaMemcpy(
        d_batch_recs,
        host_records.data(),
        batch_size * sizeof(DeviceTensorRecord),
        cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy failed for batch descriptors: ") + cudaGetErrorString(copy_status));
    }
#else
    throw std::runtime_error("No GPU backend enabled for descriptor copy");
#endif

    // ==========================================================
    // STEP 6: NO GPU INIT KERNEL NEEDED.
    // rm.memset (above) already zeroed the entire slab synchronously.
    // tensor_stats_kernel overwrites d_histogram, d_raw_moments,
    // d_minmax, and d_fingerprint entirely during analysis, so there
    // is nothing to pre-initialize here.
    //
    // The previous async RAJA init kernel caused a ~22-second stall:
    // it ran with only `batch_size` threads (4) each looping over
    // 4096 histogram bins sequentially (terrible utilization), was
    // launched asynchronously on the default stream, and the first
    // synchronous hipMemcpy in analyze_tensors had to drain all 42
    // queued instances before the DMA could start.
    // ==========================================================
    const auto alloc_end    = std::chrono::steady_clock::now();
    const auto kernel_start = alloc_end;   // nothing to time
    const auto kernel_end   = alloc_end;

    // ==========================================================
    // STEP 7: COMPUTE TIME DELTAS FOR REPORTING
    // ==========================================================
    const double pass1_ms = std::chrono::duration<double, std::milli>(pass1_end - pass1_start).count();
    const double allocation_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    const double kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    size_t batch_index = 0;

    // ==========================================================
    // STEP 8: CRITICAL SECTION - UNIFIED GLOBAL TABLE & TELEMETRY
    // ==========================================================
    {
        // Using registry_mutex_ from your class definition to guard all shared state
        std::unique_lock lock(registry_mutex_);

        // --- Part A: Amortized O(1) Global Table Management ---
        const size_t old_size = global_host_records_.size();
        const size_t new_size = old_size + batch_size;

        // Maintain the host-side historical vector mirror
        global_host_records_.insert(
            global_host_records_.end(),
            host_records.begin(),
            host_records.end());

        record_count_ = new_size;

        fingerprint_stage_.resize(record_count_);

        // Check if GPU master buffer needs to grow exponentially
        if (new_size > global_capacity_) {
            size_t new_capacity = global_capacity_ == 0 ? 128 : global_capacity_ * 2;
            if (new_capacity < new_size) {
                new_capacity = new_size;
            }

            DeviceTensorRecord* d_new_records = static_cast<DeviceTensorRecord*>(
                device_allocator_.allocate(new_capacity * sizeof(DeviceTensorRecord)));

            // High-speed VRAM-to-VRAM migration bypasses the slow PCIe bus entirely
            if (d_records_ && old_size > 0) {
#if defined(RAJA_ENABLE_HIP)
                hipError_t err_migrate = hipMemcpy(d_new_records, d_records_, old_size * sizeof(DeviceTensorRecord), hipMemcpyDeviceToDevice);
                if (err_migrate != hipSuccess) {
                    throw std::runtime_error(
                        std::string("hipMemcpy failed for device-to-device migration: ") + hipGetErrorString(err_migrate));
                }
#elif defined(RAJA_ENABLE_CUDA)
                cudaError_t err_migrate = cudaMemcpy(d_new_records, d_records_, old_size * sizeof(DeviceTensorRecord), cudaMemcpyDeviceToDevice);
                if (err_migrate != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaMemcpy failed for device-to-device migration: ") + cudaGetErrorString(err_migrate));
                }
#endif
                device_allocator_.deallocate(d_records_);
            }

            d_records_ = d_new_records;
            global_capacity_ = new_capacity;
        }

        if (new_size > fingerprint_capacity_) {

    size_t new_capacity =
        fingerprint_capacity_ == 0 ? 128 : fingerprint_capacity_ * 2;

    if (new_capacity < new_size)
        new_capacity = new_size;

    Fingerprint* new_buf =
        static_cast<Fingerprint*>(
            device_allocator_.allocate(new_capacity * sizeof(Fingerprint)));

    if (d_fingerprint_buffer_) {
        #if defined(RAJA_ENABLE_HIP)
                hipError_t err_finger = hipMemcpy(new_buf,
                        d_fingerprint_buffer_,
                        fingerprint_capacity_ * sizeof(Fingerprint),
                        hipMemcpyDeviceToDevice);
                if (err_finger != hipSuccess) {
                    throw std::runtime_error(
                        std::string("hipMemcpy failed for device-to-device migration: ") + hipGetErrorString(err_finger));
                }
        #else
                cudaError_t err_finger = cudaMemcpy(new_buf,
                        d_fingerprint_buffer_,
                        fingerprint_capacity_ * sizeof(Fingerprint),
                        cudaMemcpyDeviceToDevice);
                if (err_finger != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaMemcpy failed for device-to-device migration: ") + cudaGetErrorString(err_finger));
                }
        #endif
                device_allocator_.deallocate(d_fingerprint_buffer_);
            }

            d_fingerprint_buffer_ = new_buf;
            fingerprint_capacity_ = new_capacity;
        }

        // NOTE: d_records_ (GPU) is intentionally NOT populated here.
        // analyze_tensors() does one bulk hipMemcpy of global_host_records_ → d_records_
        // just before launching kernels. Keeping GPU copies out of this critical section
        // avoids: (a) serializing all worker threads through a synchronous GPU transfer,
        // and (b) the implicit hipDeviceSynchronize that a synchronous hipMemcpy implies
        // (which would add ~42 full GPU syncs during initialization for 42 batches).

        // --- Part B: Telemetry & Pro-Rated Category Bookkeeping ---
        std::array<size_t, tracked_tensor_category_count> category_tensor_counts{};
        std::array<size_t, tracked_tensor_category_count> category_buffer_bytes{};
        
        for (size_t i = 0; i < batch_size; ++i) {
            const TensorMetadata& meta = batch_resources[i]->get_meta();
            const size_t category_idx = tensor_category_index(meta.category);
            if (category_idx >= tracked_tensor_category_count) {
                continue;
            }
            ++category_tensor_counts[category_idx];
            category_buffer_bytes[category_idx] += batch_plans[i].total_bytes;
        }

        for (size_t category_idx = 0; category_idx < tracked_tensor_category_count; ++category_idx) {
            if (category_tensor_counts[category_idx] == 0) {
                continue;
            }
            TensorCategoryStats& stats = category_stats_[category_idx];
            ++stats.batch_count;
            
            if (batch_slab_bytes > 0) {
                const double buffer_share = static_cast<double>(category_buffer_bytes[category_idx]) / static_cast<double>(batch_slab_bytes);
                stats.allocation_ms += allocation_ms * buffer_share;
            }
            stats.kernel_ms += kernel_ms * (static_cast<double>(category_tensor_counts[category_idx]) / static_cast<double>(batch_size));
        }

        // --- Part C: Update Pipeline Global Totals ---
        total_pass1_ms_ += pass1_ms;
        total_allocation_ms_ += allocation_ms;
        total_kernel_ms_ += kernel_ms;
        
        ++processed_batch_count_;
        batch_index = processed_batch_count_;
    } 
    // registry_mutex_ is automatically and safely dropped here

    // ==========================================================
    // STEP 9: CLEANUP TEMPORARY BATCH DESCRIPTORS (No VRAM Leaks)
    // ==========================================================
    device_allocator_.deallocate(d_batch_recs);

    (void)batch_index;
}

BufferPlan MemoryManager::compute_buffer_plan(const TensorMetadata& meta) const {
    BufferPlan plan{};
    const size_t bins = meta.histogram_bins > 0 ? meta.histogram_bins : default_histogram_bins_;
    plan.histogram_bytes   = bins * sizeof(uint32_t);
    plan.moments_bytes     = 4 * sizeof(float);
    plan.minmax_bytes      = 2 * sizeof(float);
    plan.fingerprint_bytes = sizeof(TensorFingerprint);
    size_t total = 0;
    total += align_up(plan.histogram_bytes,   alignment_bytes_);
    total += align_up(plan.moments_bytes,     alignment_bytes_);
    total += align_up(plan.minmax_bytes,      alignment_bytes_);
    total += align_up(plan.fingerprint_bytes, alignment_bytes_);
    plan.total_bytes = total;
    plan.slab_offset = 0;
    return plan;
}

void MemoryManager::check_and_fix_contiguity(TensorResource& tr) const {
    TensorMetadata& meta = tr.get_meta_mut();
    const auto& shape   = meta.shape;
    const auto& strides = meta.strides;
    if (shape.empty()) { meta.contiguity = Contiguity::Scalar; return; }
    if (strides.empty()) { meta.contiguity = Contiguity::Contiguous; return; }

    bool is_contiguous = true;
    ptrdiff_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (strides[i] != expected) { is_contiguous = false; break; }
        expected *= static_cast<ptrdiff_t>(shape[i]);
    }
    if (is_contiguous) { meta.contiguity = Contiguity::Contiguous; return; }

    meta.contiguity = Contiguity::NonContiguous;
    char* base = static_cast<char*>(tr.get_ptr());
    const size_t elem_size = meta.element_size;
    size_t num_rows = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d) num_rows *= shape[d];
    size_t inner_elems = shape.back();
    ptrdiff_t outer_stride = strides[0]; 

    for (size_t row = 0; row < num_rows; ++row) {
        TensorFragment frag{};
        frag.device_ptr   = base + row * outer_stride * elem_size;
        frag.byte_offset  = row * outer_stride * elem_size;
        frag.num_elements = inner_elems;
        frag.byte_size    = inner_elems * elem_size;
        tr.add_fragment(frag);
    }
}

// ------------------------------------------------------------------ //
//  Queries
// ------------------------------------------------------------------ //
size_t MemoryManager::tensor_count() const { 
    std::shared_lock lock(registry_mutex_); 
    return registry_.size(); 
}

size_t MemoryManager::discovered_count() const {
    std::lock_guard lock(queue_mutex_);
    return discovered_count_;
}

size_t MemoryManager::skipped_count() const {
    std::shared_lock lock(registry_mutex_);
    return skipped_count_;
}

size_t MemoryManager::get_slab_count() const { 
    std::shared_lock lock(registry_mutex_); 
    return active_slabs_.size(); 
}

size_t MemoryManager::get_total_tensor_bytes() const {
    std::shared_lock lock(registry_mutex_);
    size_t total = 0;
    for (const auto& [name, tr] : registry_) total += tr->get_byte_size();
    return total;
}

size_t MemoryManager::get_total_buffer_bytes() const {
    std::shared_lock lock(registry_mutex_);
    size_t total = 0;
    for (const auto& plan : buffer_plans_) total += plan.total_bytes;
    return total;
}

double MemoryManager::get_total_pass1_ms() const {
    std::shared_lock lock(registry_mutex_);
    return total_pass1_ms_;
}

double MemoryManager::get_total_allocation_ms() const {
    std::shared_lock lock(registry_mutex_);
    return total_allocation_ms_;
}

double MemoryManager::get_total_kernel_ms() const {
    std::shared_lock lock(registry_mutex_);
    return total_kernel_ms_;
}

size_t MemoryManager::batch_count() const {
    std::shared_lock lock(registry_mutex_);
    return processed_batch_count_;
}

std::array<TensorCategoryStats, tracked_tensor_category_count> MemoryManager::get_category_stats() const {
    std::shared_lock lock(registry_mutex_);
    return category_stats_;
}

void MemoryManager::set_skipped_count(size_t skipped_count) {
    std::unique_lock lock(registry_mutex_);
    skipped_count_ = skipped_count;
}

size_t MemoryManager::record_count() const {
    return record_count_;
}

void MemoryManager::register_param(const std::string& name,
                                   py::object param,
                                   bool allow_pointer_refresh) {
    std::unique_lock lock(registry_mutex_);
    param_registry_[name] = ParamBinding{std::move(param), allow_pointer_refresh};
}

void MemoryManager::register_frozen_param(const std::string& name) {
    std::unique_lock lock(registry_mutex_);
    param_registry_[name] = ParamBinding{py::none(), false};
}

py::object MemoryManager::get_param(const std::string& name) const {
    std::shared_lock lock(registry_mutex_);
    return param_registry_.at(name).object;
}

bool MemoryManager::has_param(
    const std::string& name) const
{
    std::shared_lock lock(registry_mutex_);
    auto it = param_registry_.find(name);

    if (it == param_registry_.end()) {
        std::cout << "[checkers] NOT FOUND: " << name << std::endl;

        std::cout << "[checkers] Registry contains "
                  << param_registry_.size()
                  << " entries" << std::endl;

        if (!param_registry_.empty()) {
            std::cout << "[checkers] Example key: "
                      << param_registry_.begin()->first
                      << std::endl;
        }

        return false;
    }

    return true;
}

bool MemoryManager::can_refresh_param(const std::string& name) const {
    std::shared_lock lock(registry_mutex_);
    auto it = param_registry_.find(name);
    if (it == param_registry_.end()) {
        return false;
    }
    return it->second.allow_pointer_refresh;
}

const std::string& MemoryManager::get_name_from_index(size_t i) const {
    return ordered_names_.at(i);
}

// ------------------------------------------------------------------ //
//  Reporting
// ------------------------------------------------------------------ //
void MemoryManager::report_memory_usage() const {
    std::shared_lock lock(registry_mutex_);
 
    std::cout << "========== MemoryManager Report ==========\n";
    std::cout << "  Tensors tracked : " << registry_.size() << "\n";
    std::cout << "  Active Batches  : " << active_slabs_.size() << "\n";
 
    size_t total_tensor = 0, total_buf = 0;
    for (size_t i = 0; i < ordered_names_.size(); ++i) {
        const auto& name = ordered_names_[i];
        auto it = registry_.find(name);
        if (it == registry_.end()) continue;
 
        const TensorMetadata& meta = it->second->get_meta();
        const char* cty = meta.contiguity == Contiguity::Contiguous
                          ? "contiguous"
                          : meta.contiguity == Contiguity::NonContiguous
                            ? "non-contiguous" : "scalar";
 
        std::cout << "  [" << name << "]\n"
                  << "      elements  : " << meta.num_elements << "\n"
                  << "      bytes     : " << meta.byte_size << "\n"
                  << "      layout    : " << cty << "\n";
 
        if (i < buffer_plans_.size()) {
            std::cout << "      buf(hist) : " << buffer_plans_[i].histogram_bytes << " B\n"
                      << "      buf(moms) : " << buffer_plans_[i].moments_bytes << " B\n"
                      << "      buf(mm)   : " << buffer_plans_[i].minmax_bytes << " B\n"
                      << "      buf(total): " << buffer_plans_[i].total_bytes << " B\n";
            total_buf += buffer_plans_[i].total_bytes;
        }
        total_tensor += meta.byte_size;
    }
 
    std::cout << "------------------------------------------\n"
              << "  Total tensor data : " << total_tensor << " B ("
              << (total_tensor >> 20) << " MiB)\n"
              << "  Total slab alloc  : " << total_buf << " B ("
              << (total_buf >> 10) << " KiB)\n"
              << "==========================================\n";
}
 
} // namespace checkers