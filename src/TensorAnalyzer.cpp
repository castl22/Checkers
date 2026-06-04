#include "checkers/TensorAnalyzer.hpp"
#include <RAJA/RAJA.hpp>
#include <cmath>
#include <algorithm>

namespace checkers {

// Helper: safe division to avoid NaNs
static float safe_divide(float num, float den) {
    return (std::abs(den) < 1e-9f) ? 0.0f : (num / den);
}

void TensorAnalyzer::compute_stats(TensorResource& tr, gpuStream_t stream) {
    
}

float TensorAnalyzer::calculate_component_similarity(float anchor, float candidate) {
    return 0.0f;
}

void TensorAnalyzer::compute_histograms_and_moments(DeviceTensorRecord* d_records, 
                                                    size_t record_count, 
                                                    gpuStream_t stream) {
    RAJA::forall<DeviceExecPolicy>(
        RAJA::RangeSegment(0, static_cast<int>(record_count)),
        [=] __device__ (int i) {
            auto& rec = d_records[i];

            const float* data = rec.d_data;
            uint32_t* hist = rec.d_histogram;
            size_t N = rec.num_elements;

            if (!data || !hist || N == 0) return;

            // -------------------------------
            // register accumulators
            // -------------------------------
            float mean = 0.0f;
            float M2 = 0.0f;
            float M3 = 0.0f;
            float M4 = 0.0f;

            float min_v =  1e30f;
            float max_v = -1e30f;

            // -------------------------------
            // tile size (register streaming)
            // -------------------------------
            constexpr int TILE = 32;

            __shared__ uint32_t local_hist[1024]; // adjust to bins

            // init shared histogram
            for (int t = threadIdx.x; t < rec.histogram_bins; t += blockDim.x) {
                local_hist[t] = 0;
            }
            __syncthreads();

            for (size_t base = threadIdx.x; base < N; base += blockDim.x * TILE)
            {
                #pragma unroll
                for (int t = 0; t < TILE; t++)
                {
                    size_t j = base + t * blockDim.x;
                    if (j >= N) break;

                    float x = data[j];

                    // -------------------------
                    // histogram (shared first)
                    // -------------------------
                    int bin = (int)(x * rec.histogram_bins);
                    bin = max(0, min(bin, rec.histogram_bins - 1));

                    atomicAdd(&local_hist[bin], 1);

                    // -------------------------
                    // min/max
                    // -------------------------
                    min_v = fminf(min_v, x);
                    max_v = fmaxf(max_v, x);

                    // -------------------------
                    // higher-order moments (parallel Welford variant)
                    // -------------------------
                    float delta = x - mean;
                    float delta2 = delta * delta;

                    float new_mean = mean + delta / (j + 1);
                    float term1 = delta * (x - new_mean);

                    M4 += term1 * delta2 * ( (j*j - 3*j + 3) / ((j+1)*(j+1)) )
                        + 6 * term1 * M2
                        - 4 * delta * M3;

                    M3 += term1 * delta - 3 * delta * M2;
                    M2 += term1;
                    mean = new_mean;
                }
            }

            __syncthreads();

            // -------------------------------
            // flush shared histogram → global
            // -------------------------------
            for (int t = threadIdx.x; t < rec.histogram_bins; t += blockDim.x) {
                atomicAdd(&hist[t], local_hist[t]);
            }

            // -------------------------------
            // finalize moments
            // -------------------------------
            float variance = M2 / N;
            float skewness = (sqrtf(N) * M3) / powf(M2, 1.5f);
            float kurtosis = (N * M4) / (M2 * M2);

            rec.d_raw_moments[0] = mean;
            rec.d_raw_moments[1] = variance;
            rec.d_raw_moments[2] = skewness;
            rec.d_raw_moments[3] = kurtosis;

            rec.d_minmax[0] = min_v;
            rec.d_minmax[1] = max_v;
        }
    );

}

void TensorAnalyzer::finalize_statistics(DeviceTensorRecord* d_records, 
                                         size_t record_count, 
                                         gpuStream_t stream) {
    // 2. Write the statistics reduction logic here
}

void TensorAnalyzer::compute_fingerprints(DeviceTensorRecord* d_records, 
                                          size_t record_count, 
                                          gpuStream_t stream) {
    // 3. Write the fingerprinting/hash generation logic here
}

} // namespace checkers