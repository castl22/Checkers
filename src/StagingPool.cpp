#include "checkers/StagingPool.hpp"

#include <algorithm>

namespace checkers {

StagingPool::~StagingPool() {
    release();
}

void StagingPool::allocate(size_t   slot_bytes,
                           uint32_t ring_depth,
                           uint32_t n_compute_streams) {
    release();

    slot_bytes_ = std::max<size_t>(slot_bytes, 256);
    ring_depth  = std::max<uint32_t>(ring_depth, 1);
    n_compute_streams = std::max<uint32_t>(n_compute_streams, 1);

    // High-priority copy stream so the critical-path snapshot is not starved
    // by background codec kernels sharing the device.
    int prio_lo = 0, prio_hi = 0;
    if (gpu::StreamPriorityRange(&prio_lo, &prio_hi) == gpu::Success) {
        CK_GPU_CHECK(gpu::StreamCreatePrio(&copy_stream_, prio_hi));
    } else {
        CK_GPU_CHECK(gpu::StreamCreate(&copy_stream_));
    }

    compute_streams_.resize(n_compute_streams);
    for (auto& s : compute_streams_) {
        CK_GPU_CHECK(gpu::StreamCreate(&s));
    }

    slots_.resize(ring_depth);
    free_list_.clear();
    free_list_.reserve(ring_depth);
    for (uint32_t i = 0; i < ring_depth; ++i) {
        Slot& sl = slots_[i];
        CK_GPU_CHECK(gpu::Malloc(&sl.d_buf, slot_bytes_));
        CK_GPU_CHECK(gpu::HostAlloc(&sl.h_buf, slot_bytes_));
        CK_GPU_CHECK(gpu::EventCreate(&sl.copied));
        free_list_.push_back(static_cast<int>(i));
    }
}

void StagingPool::release() {
    // Make sure nothing is still in flight before tearing down.
    if (copy_stream_) {
        (void)gpu::StreamSynchronize(copy_stream_);
    }
    for (auto s : compute_streams_) {
        if (s) (void)gpu::StreamSynchronize(s);
    }

    for (auto& sl : slots_) {
        if (sl.copied) (void)gpu::EventDestroy(sl.copied);
        if (sl.d_buf)  (void)gpu::Free(sl.d_buf);
        if (sl.h_buf)  (void)gpu::HostFree(sl.h_buf);
        sl = Slot{};
    }
    slots_.clear();
    free_list_.clear();

    for (auto s : compute_streams_) {
        if (s) (void)gpu::StreamDestroy(s);
    }
    compute_streams_.clear();

    if (copy_stream_) {
        (void)gpu::StreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }
    slot_bytes_ = 0;
}

int StagingPool::acquire() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&] { return !free_list_.empty(); });
    const int idx = free_list_.back();
    free_list_.pop_back();
    return idx;
}

void StagingPool::give_back(int slot) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        free_list_.push_back(slot);
    }
    cv_.notify_one();
}

} // namespace checkers
