#include "Utils.hpp"
#include "checkers/logging.hpp"

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace checkers {

void sync_or_throw(const std::string& stage,
                   size_t index,
                   const std::string& name,
                   std::shared_ptr<RankLogger> logger)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
    hipError_t status = hipDeviceSynchronize();
    if (status == hipSuccess) return;

    std::string msg = "[checkers] sync failed stage=" + stage +
                      " error=" + hipGetErrorString(status);
#else
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return;

    std::string msg = "[checkers] sync failed stage=" + stage +
                      " error=" + cudaGetErrorString(status);
#endif

    if (index != static_cast<size_t>(-1)) {
        msg += " index=" + std::to_string(index) + " name=" + name;
    }

    if (logger) {
        logger->log_message(msg);
    }

    throw std::runtime_error(msg);
}

} // namespace checkers