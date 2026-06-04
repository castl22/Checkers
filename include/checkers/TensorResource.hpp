#pragma once
// include/checkers/TensorResource.hpp
//
// Owns GPU memory for a single tensor (via Umpire).
// Tracks non-contiguous fragments so we never need to copy unless
// the caller explicitly requests contiguous().
// Also carries pre-computed buffer layout so MemoryManager knows
// how much space to reserve for downstream statistics (histograms,
// skewness, kurtosis, etc.) -- but does NOT compute any of those.
 
#include <umpire/Umpire.hpp>
#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

#include <RAJA/RAJA.hpp>
#if defined(RAJA_ENABLE_HIP)
   using DeviceExecPolicy = RAJA::hip_exec<256>;
#else
   using DeviceExecPolicy = RAJA::cuda_exec<256>;
#endif
 
namespace checkers {
 
// ------------------------------------------------------------------ //
//  Data type tag
// ------------------------------------------------------------------ //
enum class DataType : uint8_t {
    Float32   = 0,
    BFloat16  = 1,
    Float16   = 2,
    Int8      = 3,
    Int32     = 4,
    Unknown   = 255
};
 
inline size_t dtype_element_size(DataType dt) {
    switch (dt) {
        case DataType::Float32:  return 4;
        case DataType::BFloat16: return 2;
        case DataType::Float16:  return 2;
        case DataType::Int8:     return 1;
        case DataType::Int32:    return 4;
        default:                 return 0;
    }
}
 
// ------------------------------------------------------------------ //
//  Contiguity
// ------------------------------------------------------------------ //
enum class Contiguity : uint8_t {
    Contiguous     = 0,  // Single flat block, strides are row-major
    NonContiguous  = 1,  // Strided / sliced view; fragments list is populated
    Scalar         = 2   // 0-dim tensor
};

enum class TensorCategory : uint8_t {
    ModelState = 0,
    MasterWeights = 1,
    OptimizerExpAvg = 2,
    OptimizerExpAvgSq = 3,
    Unknown = 255
};

inline constexpr size_t tracked_tensor_category_count = 4;

inline size_t tensor_category_index(TensorCategory category) {
    switch (category) {
        case TensorCategory::ModelState:
            return 0;
        case TensorCategory::MasterWeights:
            return 1;
        case TensorCategory::OptimizerExpAvg:
            return 2;
        case TensorCategory::OptimizerExpAvgSq:
            return 3;
        default:
            return tracked_tensor_category_count;
    }
}

inline const char* tensor_category_name(TensorCategory category) {
    switch (category) {
        case TensorCategory::ModelState:
            return "model_state";
        case TensorCategory::MasterWeights:
            return "master_weights";
        case TensorCategory::OptimizerExpAvg:
            return "exp_avg";
        case TensorCategory::OptimizerExpAvgSq:
            return "exp_avg_sq";
        default:
            return "unknown";
    }
}

struct TensorCategoryStats {
    size_t tensor_count = 0;
    size_t tensor_bytes = 0;
    size_t batch_count = 0;
    double discovery_ms = 0.0;
    double contiguous_ms = 0.0;
    double planning_ms = 0.0;
    size_t planned_buffer_bytes = 0;
    double allocation_ms = 0.0;
    double kernel_ms = 0.0;
    std::string sample_name;
    DataType sample_dtype = DataType::Unknown;
    std::vector<size_t> sample_shape;
    size_t sample_bytes = 0;
    bool has_sample = false;
};

struct TensorStatistics {
    float mean;
    float median;
    float variance;
    float skewness;
    float kurtosis;
};

struct TensorFingerprint {
    float values[5];
};
 
// ------------------------------------------------------------------ //
//  One fragment of a non-contiguous tensor
//  (e.g., a strided slice that points into an existing allocation)
// ------------------------------------------------------------------ //
struct TensorFragment {
    void*  device_ptr;   // Raw GPU pointer to this fragment
    size_t byte_offset;  // Offset from the owning allocation base (if known)
    size_t num_elements; // Number of logical elements in this fragment
    size_t byte_size;    // num_elements * element_size
};
 
// ------------------------------------------------------------------ //
//  Metadata -- everything we know about a tensor BEFORE doing stats
// ------------------------------------------------------------------ //
struct TensorMetadata {
    std::string            name;
    std::string            layer_id;
    TensorCategory         category = TensorCategory::Unknown;
    std::vector<size_t>    shape;         // e.g. {768, 3072}
    std::vector<size_t>    logical_shape; // Full logical shape before sharding
    std::vector<ptrdiff_t> strides;       // In ELEMENTS (not bytes)
    DataType               data_type     = DataType::Float32;
    size_t                 element_size  = 4;   // bytes per element
    size_t                 num_elements  = 0;   // total logical elements
    size_t                 byte_size     = 0;   // logical byte size
    size_t                 logical_num_elements = 0;
    size_t                 logical_byte_size = 0;
    Contiguity             contiguity    = Contiguity::Contiguous;
    int64_t                memory_pool_id = -1;
 
    // ------- Buffer layout (filled by MemoryManager::plan_buffers) ------- //
    // These fields describe how much space the stats-kernel will need.
    // Nothing is allocated here; MemoryManager uses these to compute
    // the total slab size in Pass 1, then writes back actual device
    // pointers in Pass 2.
    size_t histogram_bins      = 256;   // configurable per tensor
    size_t buffer_histogram_bytes  = 0; // histogram_bins * sizeof(uint32_t)
    size_t buffer_moments_bytes    = 0; // 4 floats: mean,var,skew,kurt
    size_t buffer_minmax_bytes     = 0; // 2 floats: min, max
    size_t buffer_total_bytes      = 0; // sum of all the above (+ alignment)
 
    // Device pointers written by MemoryManager AFTER allocation (Pass 2)
    uint32_t* d_histogram = nullptr;
    float* d_raw_moments = nullptr;

    TensorStatistics* d_statistics = nullptr;

    TensorFingerprint* d_fingerprint = nullptr;
    float* d_minmax = nullptr;
};
 
// ------------------------------------------------------------------ //
//  TensorResource
// ------------------------------------------------------------------ //
class TensorResource {
public:
    // Construct from an EXISTING device pointer (zero-copy; we do NOT own ptr).
    // This is the normal path: the framework already allocated the tensor.
    TensorResource(void*             existing_device_ptr,
                   size_t            byte_size,
                   TensorMetadata    meta,
                   umpire::Allocator analysis_allocator);
 
    // Construct and let Umpire allocate fresh GPU memory (e.g., for contiguous copies).
    TensorResource(size_t            byte_size,
                   TensorMetadata    meta,
                   umpire::Allocator allocator);
 
    ~TensorResource();
 
    // Non-copyable (owns or borrows raw GPU memory)
    TensorResource(const TensorResource&)            = delete;
    TensorResource& operator=(const TensorResource&) = delete;
    TensorResource(TensorResource&&)                 = default;
 
    // ---- Accessors ---- //
    void*                     get_ptr()       const { return ptr_; }
    size_t                    get_byte_size() const { return byte_size_; }
    size_t                    get_num_elements() const { return meta_.num_elements; }
    const TensorMetadata&     get_meta()      const { return meta_; }
    TensorMetadata&           get_meta_mut()        { return meta_; }
    Contiguity                get_contiguity() const { return meta_.contiguity; }
 
    // Non-contiguous fragment list (populated during Pass 1 scan)
    const std::vector<TensorFragment>& fragments() const { return fragments_; }
    void add_fragment(TensorFragment f) { fragments_.push_back(f); }
    bool has_fragments() const { return !fragments_.empty(); }
 
    // ---- Contiguous copy ---- //
    // Allocates a new flat buffer (via contiguous_allocator_) and copies
    // all fragments into it using a RAJA kernel.  Sets ptr_ to the new
    // buffer and marks contiguity = Contiguous.
    // This is the ONLY place we copy data -- and only when the caller
    // explicitly requests it.
    void make_contiguous();
 
    // Returns total byte size of all fragments (for non-contiguous tensors)
    size_t total_fragment_bytes() const;
 
private:
    void*                    ptr_;        // Device pointer (borrowed or owned)
    bool                     owns_ptr_;   // true if we allocated via Umpire
    size_t                   byte_size_;
    TensorMetadata           meta_;
    umpire::Allocator        allocator_;  // used for contiguous copies
    std::vector<TensorFragment> fragments_;
};
 
} // namespace checkers
