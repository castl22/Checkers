#pragma once
// Backend-agnostic thin wrappers over the CUDA / HIP runtime.
// Everything lives in namespace checkers::gpu so call sites read the same
// on both backends:  gpu::MemcpyAsync(...), gpu::StreamCreate(...), etc.

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(RAJA_ENABLE_HIP)
#  include <hip/hip_runtime.h>
#  define CK_GPU_HIP 1
#else
#  include <cuda_runtime.h>
#  define CK_GPU_CUDA 1
#endif

namespace checkers {
namespace gpu {

#if defined(CK_GPU_HIP)
using Stream_t = hipStream_t;
using Event_t  = hipEvent_t;
using Error_t  = hipError_t;
inline constexpr Error_t Success = hipSuccess;
enum class Copy { D2D = hipMemcpyDeviceToDevice,
                  D2H = hipMemcpyDeviceToHost,
                  H2D = hipMemcpyHostToDevice,
                  Any = hipMemcpyDefault };
inline const char* ErrorString(Error_t e)       { return hipGetErrorString(e); }
inline Error_t LastError()                       { return hipGetLastError(); }
inline Error_t DeviceSynchronize()              { return hipDeviceSynchronize(); }
inline Error_t GetDevice(int* d)                { return hipGetDevice(d); }
inline Error_t SetDevice(int d)                 { return hipSetDevice(d); }
inline Error_t MemGetInfo(size_t* f, size_t* t) { return hipMemGetInfo(f, t); }
inline Error_t Malloc(void** p, size_t n)       { return hipMalloc(p, n); }
inline Error_t Free(void* p)                     { return hipFree(p); }
inline Error_t HostAlloc(void** p, size_t n)    { return hipHostMalloc(p, n, hipHostMallocDefault); }
inline Error_t HostFree(void* p)                 { return hipHostFree(p); }
inline Error_t StreamCreate(Stream_t* s)        { return hipStreamCreateWithFlags(s, hipStreamNonBlocking); }
inline Error_t StreamCreatePrio(Stream_t* s, int prio) { return hipStreamCreateWithPriority(s, hipStreamNonBlocking, prio); }
inline Error_t StreamDestroy(Stream_t s)        { return hipStreamDestroy(s); }
inline Error_t StreamSynchronize(Stream_t s)   { return hipStreamSynchronize(s); }
inline Error_t StreamWaitEvent(Stream_t s, Event_t e) { return hipStreamWaitEvent(s, e, 0); }
inline Error_t EventCreate(Event_t* e)          { return hipEventCreateWithFlags(e, hipEventDisableTiming); }
inline Error_t EventDestroy(Event_t e)          { return hipEventDestroy(e); }
inline Error_t EventRecord(Event_t e, Stream_t s) { return hipEventRecord(e, s); }
inline Error_t EventSynchronize(Event_t e)      { return hipEventSynchronize(e); }
inline Error_t MemcpyAsync(void* d, const void* s, size_t n, Copy k, Stream_t st) {
    return hipMemcpyAsync(d, s, n, static_cast<hipMemcpyKind>(k), st);
}
// Synchronous copy.  MI300A APU quirks:
//   * explicit hipMemcpyHostToDevice from *pageable* host memory is rejected
//     ("invalid argument")            -> use hipMemcpyDefault for H2D;
//   * hipMemcpyDefault mis-copies a large device->*pageable* D2H (broadcasts
//     element 0)                       -> use the explicit kind for D2H / D2D.
inline Error_t Memcpy(void* d, const void* s, size_t n, Copy k) {
    const hipMemcpyKind kind = (k == Copy::H2D) ? hipMemcpyDefault
                                                : static_cast<hipMemcpyKind>(k);
    return hipMemcpy(d, s, n, kind);
}
inline Error_t MemsetAsync(void* p, int v, size_t n, Stream_t st) {
    return hipMemsetAsync(p, v, n, st);
}
inline Error_t PointerValid(const void* p) {
    hipPointerAttribute_t a{}; return hipPointerGetAttributes(&a, p);
}
inline Error_t StreamPriorityRange(int* lo, int* hi) { return hipDeviceGetStreamPriorityRange(lo, hi); }
#else
using Stream_t = cudaStream_t;
using Event_t  = cudaEvent_t;
using Error_t  = cudaError_t;
inline constexpr Error_t Success = cudaSuccess;
enum class Copy { D2D = cudaMemcpyDeviceToDevice,
                  D2H = cudaMemcpyDeviceToHost,
                  H2D = cudaMemcpyHostToDevice,
                  Any = cudaMemcpyDefault };
inline const char* ErrorString(Error_t e)       { return cudaGetErrorString(e); }
inline Error_t LastError()                       { return cudaGetLastError(); }
inline Error_t DeviceSynchronize()              { return cudaDeviceSynchronize(); }
inline Error_t GetDevice(int* d)                { return cudaGetDevice(d); }
inline Error_t SetDevice(int d)                 { return cudaSetDevice(d); }
inline Error_t MemGetInfo(size_t* f, size_t* t) { return cudaMemGetInfo(f, t); }
inline Error_t Malloc(void** p, size_t n)       { return cudaMalloc(p, n); }
inline Error_t Free(void* p)                     { return cudaFree(p); }
inline Error_t HostAlloc(void** p, size_t n)    { return cudaHostAlloc(p, n, cudaHostAllocDefault); }
inline Error_t HostFree(void* p)                 { return cudaFreeHost(p); }
inline Error_t StreamCreate(Stream_t* s)        { return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking); }
inline Error_t StreamCreatePrio(Stream_t* s, int prio) { return cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, prio); }
inline Error_t StreamDestroy(Stream_t s)        { return cudaStreamDestroy(s); }
inline Error_t StreamSynchronize(Stream_t s)   { return cudaStreamSynchronize(s); }
inline Error_t StreamWaitEvent(Stream_t s, Event_t e) { return cudaStreamWaitEvent(s, e, 0); }
inline Error_t EventCreate(Event_t* e)          { return cudaEventCreateWithFlags(e, cudaEventDisableTiming); }
inline Error_t EventDestroy(Event_t e)          { return cudaEventDestroy(e); }
inline Error_t EventRecord(Event_t e, Stream_t s) { return cudaEventRecord(e, s); }
inline Error_t EventSynchronize(Event_t e)      { return cudaEventSynchronize(e); }
inline Error_t MemcpyAsync(void* d, const void* s, size_t n, Copy k, Stream_t st) {
    return cudaMemcpyAsync(d, s, n, static_cast<cudaMemcpyKind>(k), st);
}
// Synchronous copy.  Mirrors the HIP wrapper: Default for H2D (pageable-safe),
// the explicit kind otherwise.
inline Error_t Memcpy(void* d, const void* s, size_t n, Copy k) {
    const cudaMemcpyKind kind = (k == Copy::H2D) ? cudaMemcpyDefault
                                                 : static_cast<cudaMemcpyKind>(k);
    return cudaMemcpy(d, s, n, kind);
}
inline Error_t MemsetAsync(void* p, int v, size_t n, Stream_t st) {
    return cudaMemsetAsync(p, v, n, st);
}
inline Error_t PointerValid(const void* p) {
    cudaPointerAttributes a{}; return cudaPointerGetAttributes(&a, p);
}
inline Error_t StreamPriorityRange(int* lo, int* hi) { return cudaDeviceGetStreamPriorityRange(lo, hi); }
#endif

} // namespace gpu
} // namespace checkers

// Throwing check -- use around every runtime call that must not fail silently.
#define CK_GPU_CHECK(call)                                                       \
    do {                                                                        \
        ::checkers::gpu::Error_t _ck_e = (call);                               \
        if (_ck_e != ::checkers::gpu::Success) {                               \
            throw std::runtime_error(std::string("[checkers][gpu] ") +         \
                ::checkers::gpu::ErrorString(_ck_e) + "  (" #call ")");        \
        }                                                                     \
    } while (0)
