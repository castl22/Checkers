#pragma once

#include "checkers/TensorResource.hpp"
#include <vector>
#include <string>
#include <memory>

// 1. Resolve the cudaStream_t error by detecting the AMD/ROCm vs NVIDIA platform
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
namespace checkers {
    // Alias hipStream_t to a unified type name
    using gpuStream_t = hipStream_t;
}
#else
#include <cuda_runtime.h>
namespace checkers {
    using gpuStream_t = cudaStream_t;
}
#endif

#include <string>
#include <vector>
#include "checkers/TensorResource.hpp"
#include "checkers/api.hpp" 

namespace checkers {

// Forward declaration of the record struct used by MemoryManager
struct DeviceTensorRecord;

class TensorAnalyzer {
public:
    void compute_stats(TensorResource& tr, gpuStream_t stream = 0);
    float calculate_component_similarity(float anchor, float candidate);

    // Updated to accept the two arguments passed from api.cpp
    void compute_histograms_and_moments(DeviceTensorRecord* d_records, 
                                        size_t record_count, 
                                        gpuStream_t stream = 0);
    
    void finalize_statistics(DeviceTensorRecord* d_records, 
                             size_t record_count, 
                             gpuStream_t stream = 0);
    
    void compute_fingerprints(DeviceTensorRecord* d_records, 
                              size_t record_count, 
                              gpuStream_t stream = 0);
};

} // namespace checkers