#include "checkers/TensorAnalyzer.hpp"
#include "Utils.hpp"
#include <RAJA/RAJA.hpp>
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    #include <hip/hip_fp16.h>
    #include <hip/hip_bfloat16.h>
#elif defined(__CUDACC__)
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

namespace checkers {

// -------------------------------------------------------------
// 1. Setup RAJA Execution Policies based on Target Backend
// -------------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    using analyzer_exec_policy = RAJA::hip_exec<256>;
#else
    using analyzer_exec_policy = RAJA::cuda_exec<256>;
#endif

// -------------------------------------------------------------
// 2. Device Type-Unpacking Helpers (Device Scope)
// -------------------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    // AMD ROCm Implementation
    template <> __device__ __forceinline__ float to_float(half v) { return __half2float(v); }
    template <> __device__ __forceinline__ float to_float(hip_bfloat16 v) { return static_cast<float>(v); }
#elif defined(__CUDACC__)
    // NVIDIA CUDA Implementation
    template <> __device__ __forceinline__ float to_float(half v) { return __half2float(v); }
    #if __CUDA_ARCH__ >= 800
        template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    #endif
#endif

// -------------------------------------------------------------
// 3. Core Statistics: Device Helper + Universal Kernel
//
// compute_tensor_stats<T> is a __device__ function with all
// per-tensor math.  tensor_stats_kernel dispatches to the correct
// T at runtime from each record's data_type field.  Since one
// block processes one tensor, all threads in the block take the
// same branch — zero warp divergence.
//
// Launch pattern: tensor_stats_kernel<<<N_tensors, 256>>>()
// The GPU schedules all N_tensors blocks concurrently across CUs,
// saturating memory bandwidth instead of serialising records.
//
// Histogram note: shared memory holds exactly 256 slots.  bins is
// clamped to 256; values are still mapped linearly over the full
// [min, max] range, so the histogram is correct and complete.
// -------------------------------------------------------------
template <typename T>
__device__ void compute_tensor_stats(DeviceTensorRecord& rec)
{
    T* data_ptr = static_cast<T*>(rec.d_ptr);
    if (!data_ptr || rec.num_elements == 0) return;

    const int tid = threadIdx.x;
    // Clamp to shared-memory capacity; values are still mapped over
    // the full [min, max] range so the histogram is correct.
    const uint32_t bins = (rec.histogram_bins < 256u) ? rec.histogram_bins : 256u;
    const size_t N = rec.num_elements;

    __shared__ float s_min, s_max, s_mean;
    __shared__ uint32_t s_hist[256];
    __shared__ float s_reduce_min[256];
    __shared__ float s_reduce_max[256];
    __shared__ float s_reduce_sum[256];

    if (tid < bins) s_hist[tid] = 0;
    __syncthreads();

    // PASS 1: Bounds & Sum
    float local_min =  1e37f;
    float local_max = -1e37f;
    float local_sum = 0.0f;

    for (size_t idx = tid; idx < N; idx += blockDim.x) {
        float x = to_float(data_ptr[idx]);
        local_min = fminf(local_min, x);
        local_max = fmaxf(local_max, x);
        local_sum += x;
    }

    s_reduce_min[tid] = local_min;
    s_reduce_max[tid] = local_max;
    s_reduce_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce_min[tid] = fminf(s_reduce_min[tid], s_reduce_min[tid + s]);
            s_reduce_max[tid] = fmaxf(s_reduce_max[tid], s_reduce_max[tid + s]);
            s_reduce_sum[tid] += s_reduce_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_min  = s_reduce_min[0];
        s_max  = s_reduce_max[0];
        s_mean = s_reduce_sum[0] / static_cast<float>(N);
    }
    __syncthreads();

    float mean  = s_mean;
    float range = s_max - s_min;
    float scale = (range > 1e-9f) ? (static_cast<float>(bins) - 1.0f) / range : 0.0f;

    // PASS 2: Central Moments & Histogram
    float m2 = 0.0f;
    float m3 = 0.0f;
    float m4 = 0.0f;

    for (size_t idx = tid; idx < N; idx += blockDim.x) {
        float x = to_float(data_ptr[idx]);

        if (range > 1e-9f) {
            int b = static_cast<int>((x - s_min) * scale);
            b = max(0, min(b, static_cast<int>(bins - 1)));
            atomicAdd(&s_hist[b], 1u);
        } else {
            if (bins > 0) atomicAdd(&s_hist[0], 1u);
        }

        float d  = x - mean;
        float d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    __syncthreads();

    s_reduce_sum[tid] = m2;
    s_reduce_min[tid] = m3;
    s_reduce_max[tid] = m4;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce_sum[tid] += s_reduce_sum[tid + s];
            s_reduce_min[tid] += s_reduce_min[tid + s];
            s_reduce_max[tid] += s_reduce_max[tid + s];
        }
        __syncthreads();
    }

    if (tid < bins && rec.d_histogram) {
        rec.d_histogram[tid] = s_hist[tid];
    }

    if (tid == 0) {
        float M2 = s_reduce_sum[0] / static_cast<float>(N);
        float M3 = s_reduce_min[0] / static_cast<float>(N);
        float M4 = s_reduce_max[0] / static_cast<float>(N);

        float var  = M2;
        float skew = 0.0f;
        float kurt = 0.0f;

        if (var > 1e-9f) {
            float std_dev = sqrtf(var);
            skew = M3 / (var * std_dev);
            kurt = M4 / (var * var);
        }

        if (rec.d_raw_moments) {
            rec.d_raw_moments[0] = mean;
            rec.d_raw_moments[1] = var;
            rec.d_raw_moments[2] = skew;
            rec.d_raw_moments[3] = kurt;
        }

        if (rec.d_minmax) {
            rec.d_minmax[0] = s_min;
            rec.d_minmax[1] = s_max;
        }

        if (rec.d_statistics) {
            rec.d_statistics->mean     = mean;
            rec.d_statistics->variance = var;
            rec.d_statistics->skewness = skew;
            rec.d_statistics->kurtosis = kurt;
        }
    }
}

// Universal kernel — one block per tensor record, dtype dispatched per-record.
// All N tensors are submitted in a single launch; the GPU schedules them
// concurrently across its CUs, saturating memory bandwidth.
__global__ void tensor_stats_kernel(DeviceTensorRecord* recs, size_t Nrecs)
{
    const int i = blockIdx.x;
    if (i >= static_cast<int>(Nrecs)) return;

    // All 256 threads in this block process the same record, so every
    // thread takes the same branch — no warp divergence.
    switch (static_cast<DataType>(recs[i].data_type)) {
        case DataType::Float32:
            compute_tensor_stats<float>(recs[i]);        break;
        case DataType::Float16:
            compute_tensor_stats<half>(recs[i]);         break;
        case DataType::BFloat16:
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
            compute_tensor_stats<hip_bfloat16>(recs[i]);
#elif defined(__CUDACC__)
            compute_tensor_stats<__nv_bfloat16>(recs[i]);
#endif
            break;
        case DataType::Int8:
            compute_tensor_stats<int8_t>(recs[i]);       break;
        case DataType::Int32:
            compute_tensor_stats<int32_t>(recs[i]);      break;
        default:
            break;
    }
}

// Helper: safe division to avoid NaNs
static float safe_divide(float num, float den) {
    return (std::abs(den) < 1e-9f) ? 0.0f : (num / den);
}

// -------------------------------------------------------------
// 4. Class Method Implementations
// -------------------------------------------------------------

void TensorAnalyzer::compute_stats(TensorResource& tr, gpuStream_t stream) {
    // Process single standalone TensorResource profiles if bypassed by batch API
}

// ...

void TensorAnalyzer::compute_histograms_and_moments(DeviceTensorRecord* d_records, 
                                                    size_t record_count, 
                                                    gpuStream_t stream) 
{
    if (record_count == 0 || !d_records) return;

    DataType dtype = static_cast<DataType>(d_records[0].data_type);
    launch_tensor_analysis(d_records, record_count, dtype, stream);
}

void TensorAnalyzer::finalize_statistics(DeviceTensorRecord* d_records, 
                                         size_t record_count, 
                                         gpuStream_t stream) 
{
    RAJA::forall<analyzer_exec_policy>(RAJA::RangeSegment(0, record_count), 
    [=] __device__ (RAJA::Index_type i) {
        DeviceTensorRecord& rec = d_records[i];
        if (rec.d_statistics) {
            if (std::isnan(rec.d_statistics->variance)) {
                rec.d_statistics->variance = 0.0f;
            }
        }
    });
}

void TensorAnalyzer::compute_fingerprints(DeviceTensorRecord* d_records, 
                                          size_t record_count, 
                                          gpuStream_t stream) 
{
    RAJA::forall<analyzer_exec_policy>(
        RAJA::RangeSegment(0, record_count),
        [=] __device__ (RAJA::Index_type i) 
    {
        DeviceTensorRecord& rec = d_records[i];

        if (rec.d_fingerprint && rec.d_raw_moments && rec.d_minmax)
        {
            // d_raw_moments layout: [0]=mean [1]=variance [2]=skewness [3]=kurtosis
            // d_statistics is never allocated in the slab; use d_raw_moments instead.
            rec.d_fingerprint->values[0] = rec.d_raw_moments[0];  // mean
            rec.d_fingerprint->values[1] = rec.d_raw_moments[1];  // variance
            rec.d_fingerprint->values[2] = rec.d_raw_moments[2];  // skewness
            rec.d_fingerprint->values[3] = rec.d_raw_moments[3];  // kurtosis
            rec.d_fingerprint->values[4] = rec.d_minmax[1] - rec.d_minmax[0]; // range
        }
    });
}

void TensorAnalyzer::launch_tensor_analysis(DeviceTensorRecord* d_records,
                                            size_t record_count,
                                            DataType /* dtype — each record carries its own */,
                                            gpuStream_t stream)
{
    if (record_count == 0) return;

    // One block per tensor; dtype dispatched per-record inside the kernel.
    // All record_count blocks are scheduled concurrently by the GPU.
    tensor_stats_kernel<<<dim3(record_count), dim3(256), 0, stream>>>(d_records, record_count);

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        throw std::runtime_error(
            std::string("tensor_stats_kernel launch failed (HIP): ")
            + hipGetErrorString(launch_status));
    }
#elif defined(__CUDACC__)
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("tensor_stats_kernel launch failed (CUDA): ")
            + cudaGetErrorString(launch_status));
    }
#endif
}

void TensorAnalyzer::compute_fingerprint_weights(
    const MemoryManager& mgr,
    DeviceTensorRecord* d_records,
    size_t record_count,
    std::shared_ptr<RankLogger> logger)
{
    struct Stats {
        double m[5] = {0,0,0,0,0};
        double m2[5] = {0,0,0,0,0};
        size_t count = 0;
    };

    const auto& fps = mgr.get_fingerprints();

    auto accumulate = [](Stats& s, const Fingerprint& fp) {
        for (int i = 0; i < 5; i++) {
            s.m[i]  += fp.values[i];
            s.m2[i] += fp.values[i] * fp.values[i];
        }
        s.count++;
    };

    // Compute total variance across ALL tensors for each fingerprint dimension.
    //
    // Within-category variance was previously used but always collapsed to zero:
    // tensors in the same category (e.g. all model layers) are initialized from
    // the same distribution, so their per-category statistics are nearly identical.
    //
    // Total variance captures the large spread between categories (e.g. fp16 model
    // weights vs fp32 optimizer second moments) and correctly identifies which
    // fingerprint dimensions are most discriminative across the full tensor set.
    Stats all;
    for (size_t i = 0; i < record_count; i++) {
        accumulate(all, fps[i]);
    }

    auto variance = [](const Stats& s, int i) {
        if (s.count == 0) return 0.0;
        double mean = s.m[i] / s.count;
        double m2   = s.m2[i] / s.count;
        return m2 - mean * mean;
    };

    double importance[5] = {0};
    for (int i = 0; i < 5; i++) {
        importance[i] = variance(all, i);
    }

    double sum = 0.0;
    for (int i = 0; i < 5; i++)
        sum += importance[i] + 1e-8;

    m_fp_weights.w_mean     = importance[0] / sum;
    m_fp_weights.w_variance = importance[1] / sum;
    m_fp_weights.w_skewness = importance[2] / sum;
    m_fp_weights.w_kurtosis = importance[3] / sum;
    m_fp_weights.w_range    = importance[4] / sum;

    logger->log_message(
        "[FP-WEIGHTS] mean=" + std::to_string(m_fp_weights.w_mean) +
        " var=" + std::to_string(m_fp_weights.w_variance) +
        " skew=" + std::to_string(m_fp_weights.w_skewness) +
        " kurt=" + std::to_string(m_fp_weights.w_kurtosis) +
        " range=" + std::to_string(m_fp_weights.w_range)
    );
}

void TensorAnalyzer::apply_fingerprint_weights(
    DeviceTensorRecord* d_records,
    size_t record_count,
    gpuStream_t stream)
{
    FingerprintWeights w_host = m_fp_weights;

    RAJA::forall<analyzer_exec_policy>(
        RAJA::RangeSegment(0, record_count),
        [=] __device__ (RAJA::Index_type i)
    {
        DeviceTensorRecord& rec = d_records[i];

        if (rec.d_fingerprint)
        {
            float fp =
                rec.d_fingerprint->values[0] * w_host.w_mean +
                rec.d_fingerprint->values[1] * w_host.w_variance +
                rec.d_fingerprint->values[2] * w_host.w_skewness +
                rec.d_fingerprint->values[3] * w_host.w_kurtosis +
                rec.d_fingerprint->values[4] * w_host.w_range;

            rec.d_fingerprint->values[0] = fp;
            rec.d_fingerprint->values[1] = 0;
            rec.d_fingerprint->values[2] = 0;
            rec.d_fingerprint->values[3] = 0;
            rec.d_fingerprint->values[4] = 0;
        }
    });
}

void TensorAnalyzer::allocate_cluster_buffers(size_t /*N*/)
{
    // GPU cluster buffers removed; clustering runs on CPU.
}


void TensorAnalyzer::set_logger(std::shared_ptr<RankLogger> logger)
{
    m_logger = logger;
}

void TensorAnalyzer::cluster_sanity(
    const std::vector<int>& /*h_clusters*/,
    size_t /*N*/,
    const std::unordered_map<int, std::vector<float>>& /*cluster_fp*/,
    std::shared_ptr<RankLogger> /*logger*/)
{
    // Superseded by per-category logging inside build_knn_clusters.
}

void TensorAnalyzer::build_knn_clusters(
    DeviceTensorRecord* /*d_records*/,
    size_t N,
    int    /*K*/,
    gpuStream_t /*stream*/)
{
    if (N == 0 || !m_logger) return;

    // Use the CPU-side fingerprint copy made in analyze_tensors (step 4),
    // which holds the raw 5D values (mean, var, skew, kurt, range) computed
    // BEFORE apply_fingerprint_weights collapsed them to a scalar on the GPU.
    const auto& fps = MemoryManager::instance().get_fingerprints();
    const auto& mgr = MemoryManager::instance();

    // ---- 1. Group tensor indices by category via name prefix ----------------
    auto category_of = [](const std::string& name) -> std::string {
        if (name.rfind("optimizer.exp_avg_sq::", 0) == 0) return "exp_avg_sq";
        if (name.rfind("optimizer.exp_avg::", 0) == 0)    return "exp_avg";
        if (name.rfind("master_weights::", 0) == 0)       return "master_weights";
        return "model_state";
    };

    // std::map keeps categories in a stable order for deterministic logging.
    std::map<std::string, std::vector<size_t>> by_cat;
    for (size_t i = 0; i < N; i++)
        by_cat[category_of(mgr.get_name_from_index(i))].push_back(i);

    // ---- 2. Union-Find (path-compressed, union-by-rank) --------------------
    struct UF {
        std::vector<int> p, r;
        explicit UF(int n) : p(n), r(n, 0) { std::iota(p.begin(), p.end(), 0); }
        int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
        void unite(int a, int b) {
            a = find(a); b = find(b);
            if (a == b) return;
            if (r[a] < r[b]) std::swap(a, b);
            p[b] = a;
            if (r[a] == r[b]) ++r[a];
        }
    };

    size_t grand_total = 0, grand_clustered = 0, grand_singleton = 0;

    // ---- 3. Per-category clustering ----------------------------------------
    for (const auto& [cat, indices] : by_cat) {
        const int n = static_cast<int>(indices.size());
        grand_total += n;
        if (n == 0) continue;

        // Normalize each fingerprint dimension to unit variance within category
        // so no single dimension (e.g. large raw variance) dominates the distance.
        double feat_mean[5] = {}, feat_std[5] = {};
        for (int a = 0; a < n; a++)
            for (int d = 0; d < 5; d++)
                feat_mean[d] += fps[indices[a]].values[d];
        for (int d = 0; d < 5; d++) feat_mean[d] /= n;

        for (int a = 0; a < n; a++)
            for (int d = 0; d < 5; d++) {
                double diff = fps[indices[a]].values[d] - feat_mean[d];
                feat_std[d] += diff * diff;
            }
        for (int d = 0; d < 5; d++)
            feat_std[d] = (feat_std[d] / n < 1e-30) ? 1.0 : std::sqrt(feat_std[d] / n);

        // Normalized Euclidean distance between tensor positions a, b in `indices`
        auto dist = [&](int a, int b) -> double {
            double s = 0;
            for (int d = 0; d < 5; d++) {
                double diff = (fps[indices[a]].values[d] - fps[indices[b]].values[d])
                              / feat_std[d];
                s += diff * diff;
            }
            return std::sqrt(s);
        };

        // Collect all pairwise distances and use the lower-quartile as the
        // merge threshold: pairs distinctly closer than typical are genuinely
        // similar; pairs near the median are ambiguous.
        std::vector<double> all_dists;
        all_dists.reserve(static_cast<size_t>(n) * (n - 1) / 2);
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
                all_dists.push_back(dist(a, b));

        double threshold = 0.0;
        if (!all_dists.empty()) {
            std::sort(all_dists.begin(), all_dists.end());
            threshold = all_dists[all_dists.size() / 4]; // lower quartile
        }

        // Merge pairs whose normalized distance is below the threshold
        UF uf(n);
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
                if (dist(a, b) < threshold)
                    uf.unite(a, b);

        // Collect clusters (map root → local member indices)
        std::map<int, std::vector<int>> clusters;
        for (int a = 0; a < n; a++)
            clusters[uf.find(a)].push_back(a);

        size_t cat_clustered = 0, cat_singleton = 0;
        for (const auto& [root, members] : clusters) {
            if (static_cast<int>(members.size()) > 1)
                cat_clustered += members.size();
            else
                ++cat_singleton;
        }
        grand_clustered += cat_clustered;
        grand_singleton += cat_singleton;

        // Category summary
        m_logger->log_message(
            "[CLUSTER][" + cat + "]"
            " tensors="    + std::to_string(n) +
            " clusters="   + std::to_string(clusters.size()) +
            " clustered="  + std::to_string(cat_clustered) +
            " singletons=" + std::to_string(cat_singleton) +
            " threshold="  + std::to_string(threshold));

        // Log each multi-member cluster with names and member count
        for (const auto& [root, members] : clusters) {
            if (static_cast<int>(members.size()) < 2) continue;
            std::string msg =
                "[CLUSTER][" + cat + "][size=" + std::to_string(members.size()) + "]";
            for (int local : members)
                msg += "\n  " + mgr.get_name_from_index(indices[local]);
            m_logger->log_message(msg);
        }
    }

    // Grand-total summary across all categories
    m_logger->log_message(
        "[CLUSTER][totals]"
        " total="      + std::to_string(grand_total) +
        " clustered="  + std::to_string(grand_clustered) +
        " singletons=" + std::to_string(grand_singleton));
}

} // namespace checkers