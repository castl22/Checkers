// Implements TensorResource -- owns or borrows a GPU tensor pointer,
// tracks fragments for non-contiguous views, and can make a
// contiguous copy via a RAJA HIP kernel.
 
#include "checkers/TensorResource.hpp"
 
#include <RAJA/RAJA.hpp>
#include <stdexcept>
#include <numeric>
 
namespace checkers {
 
// ------------------------------------------------------------------ //
//  Constructor: borrow an existing device pointer (most common path)
// ------------------------------------------------------------------ //
TensorResource::TensorResource(void*             existing_device_ptr,
                               size_t            byte_size,
                               TensorMetadata    meta,
                               umpire::Allocator analysis_allocator)
    : ptr_(existing_device_ptr)
    , owns_ptr_(false)
    , byte_size_(byte_size)
    , meta_(std::move(meta))
    , allocator_(analysis_allocator)
{
    if (!ptr_) {
        throw std::invalid_argument("TensorResource: null device pointer");
    }
    meta_.byte_size = byte_size_;
    if (meta_.num_elements == 0 && meta_.element_size > 0) {
        meta_.num_elements = byte_size_ / meta_.element_size;
    }
}
 
// ------------------------------------------------------------------ //
//  Constructor: allocate fresh GPU memory via Umpire
// ------------------------------------------------------------------ //
TensorResource::TensorResource(size_t            byte_size,
                               TensorMetadata    meta,
                               umpire::Allocator allocator)
    : ptr_(nullptr)
    , owns_ptr_(true)
    , byte_size_(byte_size)
    , meta_(std::move(meta))
    , allocator_(allocator)
{
    ptr_ = allocator_.allocate(byte_size_);
    if (!ptr_) {
        throw std::runtime_error("TensorResource: Umpire allocation failed");
    }
    meta_.byte_size = byte_size_;
    if (meta_.num_elements == 0 && meta_.element_size > 0) {
        meta_.num_elements = byte_size_ / meta_.element_size;
    }
}
 
// ------------------------------------------------------------------ //
//  Destructor -- only free if we own the memory
// ------------------------------------------------------------------ //
TensorResource::~TensorResource() {
    if (owns_ptr_ && ptr_) {
        try {
            allocator_.deallocate(ptr_);
        } catch (...) {
            // Never throw from destructor
        }
        ptr_ = nullptr;
    }
    // If !owns_ptr_ the framework manages the original tensor memory.
}
 
// ------------------------------------------------------------------ //
//  Total byte size across all fragments
// ------------------------------------------------------------------ //
size_t TensorResource::total_fragment_bytes() const {
    size_t total = 0;
    for (const auto& f : fragments_) {
        total += f.byte_size;
    }
    return total;
}
 
// ------------------------------------------------------------------ //
//  make_contiguous
//
//  Allocates a new flat buffer and packs all fragments into it using a
//  RAJA HIP kernel -- element by element, no host involvement.
//  Sets ptr_ to the new buffer and clears the fragment list.
//  Sets owns_ptr_ = true so the destructor will free the new buffer.
// ------------------------------------------------------------------ //
void TensorResource::make_contiguous() {
    if (meta_.contiguity == Contiguity::Contiguous) return;
    if (fragments_.empty()) {
        throw std::logic_error("Run MemoryManager::scan_and_plan() first.");
    }

    const size_t element_size = meta_.element_size;
    const size_t total_bytes  = total_fragment_bytes();

    // 1. Umpire Allocation
    void* new_ptr = allocator_.allocate(total_bytes);
    if (!new_ptr) throw std::runtime_error("Allocation failed");

    // 2. Upload fragment descriptors using Umpire ResourceManager
    auto& rm = umpire::ResourceManager::getInstance();
    size_t num_frags = fragments_.size();
    
    // Allocate temporary fragment storage via our allocator
    TensorFragment* d_frags = static_cast<TensorFragment*>(
        allocator_.allocate(num_frags * sizeof(TensorFragment)));
    
    // Umpire handles copy regardless of backend (CUDA/HIP)
    rm.copy(d_frags, fragments_.data(), num_frags * sizeof(TensorFragment));

    char* d_dst    = static_cast<char*>(new_ptr);
    size_t total_elements = total_bytes / element_size;

    // 3. Portable RAJA Execution
    // Using the previously defined alias for the execution policy
    RAJA::forall<DeviceExecPolicy>(
        RAJA::RangeSegment(0, static_cast<int>(total_elements)),
        [=] __device__ (size_t global_idx) {
            size_t remaining = global_idx;
            for (size_t f = 0; f < num_frags; ++f) {
                if (remaining < d_frags[f].num_elements) {
                    const char* src = static_cast<const char*>(d_frags[f].device_ptr)
                                      + (remaining * element_size);
                    char* dst = d_dst + (global_idx * element_size);
                    for (size_t b = 0; b < element_size; ++b) {
                        dst[b] = src[b];
                    }
                    return;
                }
                remaining -= d_frags[f].num_elements;
            }
        }
    );

    // 4. Cleanup: No explicit synchronization needed; Umpire ops order correctly.
    allocator_.deallocate(d_frags);

    if (owns_ptr_ && ptr_) {
        allocator_.deallocate(ptr_);
    }

    ptr_ = new_ptr;
    owns_ptr_ = true;
    byte_size_ = total_bytes;
    meta_.contiguity   = Contiguity::Contiguous;
    meta_.num_elements = total_elements;
    meta_.byte_size    = total_bytes;
    fragments_.clear();
}
 
} // namespace checkers
