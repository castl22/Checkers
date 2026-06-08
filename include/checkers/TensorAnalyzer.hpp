#pragma once

#include "TensorResource.hpp"
#include "MemoryManager.hpp"
#include "TensorFingerprint.hpp"
#include "checkers/api.hpp"
#include "checkers/logging.hpp"   // RankLogger
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace checkers {

// -----------------------------
// TensorAnalyzer
// -----------------------------
class TensorAnalyzer {

private:
    FingerprintWeights m_fp_weights;

    int* d_cluster_ids = nullptr;
    int* d_knn_indices = nullptr;
    float* d_knn_distances = nullptr;
    std::shared_ptr<RankLogger> m_logger;

public:

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    using gpuStream_t = hipStream_t;
#else
    using gpuStream_t = cudaStream_t;
#endif

    void set_logger(std::shared_ptr<RankLogger> logger);
    // -----------------------------
    // Core API
    // -----------------------------
    void compute_stats(TensorResource& tr, gpuStream_t stream = nullptr);

    float calculate_component_similarity(float anchor, float candidate);

    void compute_histograms_and_moments(DeviceTensorRecord* d_records,
                                         size_t record_count,
                                         gpuStream_t stream = nullptr);

    void finalize_statistics(DeviceTensorRecord* d_records,
                             size_t record_count,
                             gpuStream_t stream = nullptr);

    void compute_fingerprints(DeviceTensorRecord* d_records,
                              size_t record_count,
                              gpuStream_t stream = nullptr);

    void launch_tensor_analysis(DeviceTensorRecord* d_records,
                                size_t record_count,
                                DataType dtype,
                                gpuStream_t stream);

    void compute_fingerprint_weights(const MemoryManager& mgr,
                                     DeviceTensorRecord* d_records,
                                     size_t record_count,
                                     std::shared_ptr<RankLogger> logger);

    double compute_weighted_fingerprint(const Fingerprint& fp) const;

    void apply_fingerprint_weights(DeviceTensorRecord* d_records,
                                   size_t record_count,
                                   gpuStream_t stream = nullptr);

    void allocate_cluster_buffers(size_t record_count);

    void build_knn_clusters(DeviceTensorRecord* d_records,
                            size_t record_count,
                            int k = 4,
                            gpuStream_t stream = nullptr);
    void cluster_sanity(
        const std::vector<int>& h_clusters,
        size_t N,
        const std::unordered_map<int, std::vector<float>>& cluster_fp,
        std::shared_ptr<RankLogger> logger);
};

} // namespace checkers