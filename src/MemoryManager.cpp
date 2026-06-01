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

namespace {

void device_synchronize() {
#if defined(RAJA_ENABLE_HIP)
    (void)hipDeviceSynchronize();
#elif defined(RAJA_ENABLE_CUDA)
    (void)cudaDeviceSynchronize();
#endif
}

} // namespace

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
    for (void* d_rec : batch_device_records_) {
        if (d_rec) device_allocator_.deallocate(d_rec);
    }
    registry_.clear();
    ordered_names_.clear();
    buffer_plans_.clear();
    active_slabs_.clear();
    batch_device_records_.clear();
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
    logger_.reset();
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
    batch_plans.reserve(batch_size);
    size_t batch_slab_bytes = 0;
    std::vector<std::string> batch_names;
    batch_names.reserve(batch_size);
    std::vector<TensorResource*> batch_resources;
    batch_resources.reserve(batch_size);

    {
        std::unique_lock lock(registry_mutex_);
        for (auto& item : items) {
            auto resource = std::make_unique<TensorResource>(item.device_ptr, item.byte_size, std::move(item.meta), device_allocator_);
            TensorResource* resource_ptr = resource.get();
            ordered_names_.push_back(item.name);
            batch_names.push_back(item.name);
            registry_[item.name] = std::move(resource);
            batch_resources.push_back(resource_ptr);
        }
    }

    for (size_t index = 0; index < batch_size; ++index) {
        TensorResource& tr = *batch_resources[index];
        TensorMetadata& meta = tr.get_meta_mut();

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
    std::lock_guard gpu_lock(gpu_ops_mutex_);
    const auto alloc_start = std::chrono::steady_clock::now();
    void* batch_slab_base = device_allocator_.allocate(batch_slab_bytes);
    {
        std::unique_lock lock(registry_mutex_);
        active_slabs_.push_back(batch_slab_base);
    }

    auto& rm = umpire::ResourceManager::getInstance();
    rm.memset(batch_slab_base, 0, batch_slab_bytes);

    size_t cursor = 0;
    char* slab_bytes_ptr = static_cast<char*>(batch_slab_base);

    std::vector<DeviceTensorRecord> host_records(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        TensorResource& tr = *batch_resources[i];
        TensorMetadata& meta = tr.get_meta_mut();
        BufferPlan& plan = batch_plans[i];

        plan.slab_offset = cursor;
        meta.d_histogram = reinterpret_cast<uint32_t*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.histogram_bytes, alignment_bytes_);
        
        meta.d_moments   = reinterpret_cast<float*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.moments_bytes, alignment_bytes_);
        
        meta.d_minmax    = reinterpret_cast<float*>(slab_bytes_ptr + cursor);
        cursor += align_up(plan.minmax_bytes, alignment_bytes_);

        DeviceTensorRecord& rec = host_records[i];
        rec.d_data         = static_cast<float*>(tr.get_ptr());
        rec.d_histogram    = meta.d_histogram;
        rec.d_moments      = meta.d_moments;
        rec.d_minmax       = meta.d_minmax;
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

    // Allocate an array descriptor for this block on the device
    DeviceTensorRecord* d_batch_recs = static_cast<DeviceTensorRecord*>(
        device_allocator_.allocate(batch_size * sizeof(DeviceTensorRecord)));
    {
        std::unique_lock lock(registry_mutex_);
        batch_device_records_.push_back(d_batch_recs);
    }

    // Umpire does not own the raw STL buffer backing host_records, so use the device
    // runtime directly for this host-to-device descriptor upload.
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
    // BACKEND GPU WORKER LAUNCH
    // Initializes the tracking values for all 10 tensors in parallel
    // ==========================================================
    const auto alloc_end = std::chrono::steady_clock::now();
    const auto kernel_start = std::chrono::steady_clock::now();
    RAJA::forall<DeviceExecPolicy>(
        RAJA::RangeSegment(0, static_cast<int>(batch_size)),
        [=] __device__ (int i) {
            if (d_batch_recs[i].d_moments) {
                for(int j = 0; j < 4; ++j) d_batch_recs[i].d_moments[j] = 0.0f;
            }
            if (d_batch_recs[i].d_minmax) {
                d_batch_recs[i].d_minmax[0] =  3.402823466e+38f;
                d_batch_recs[i].d_minmax[1] = -3.402823466e+38f;
            }
        }
    );
    device_synchronize();
    const auto kernel_end = std::chrono::steady_clock::now();

    const double pass1_ms = std::chrono::duration<double, std::milli>(pass1_end - pass1_start).count();
    const double allocation_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    const double kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    size_t batch_index = 0;
    {
        std::unique_lock lock(registry_mutex_);
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

        total_pass1_ms_ += pass1_ms;
        total_allocation_ms_ += allocation_ms;
        total_kernel_ms_ += kernel_ms;
        ++processed_batch_count_;
        batch_index = processed_batch_count_;
    }
    (void)batch_index;
}

BufferPlan MemoryManager::compute_buffer_plan(const TensorMetadata& meta) const {
    BufferPlan plan{};
    const size_t bins = meta.histogram_bins > 0 ? meta.histogram_bins : default_histogram_bins_;
    plan.histogram_bytes = bins * sizeof(uint32_t);
    plan.moments_bytes   = 4 * sizeof(float);   
    plan.minmax_bytes    = 2 * sizeof(float);    
    size_t total = 0;
    total += align_up(plan.histogram_bytes, alignment_bytes_);
    total += align_up(plan.moments_bytes,   alignment_bytes_);
    total += align_up(plan.minmax_bytes,    alignment_bytes_);
    plan.total_bytes  = total;
    plan.slab_offset  = 0; 
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