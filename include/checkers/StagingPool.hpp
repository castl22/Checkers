#pragma once
// Persistent staging pool for the online checkpoint compressor.
//
// Allocated ONCE in initialize_context so nothing on the checkpoint
// critical path ever calls the allocator:
//
//   * a ring of `ring_depth` slots, each = { device buffer, pinned host
//     buffer, "copied" event }, every buffer `slot_bytes` big
//     (slot_bytes = largest tensor across the staged categories);
//   * one high-priority copy stream for the critical-path D2D snapshot;
//   * `n_compute_streams` background streams for the codec + D2H.
//
// The critical path acquires a slot (blocking when the ring is full -- that
// backpressure is the memory/throughput knob), enqueues the D2D copy on the
// copy stream, records the slot's event, and hands the slot to a worker.
// The worker waits on the event, runs the codec on a compute stream, D2Hs
// the result, then returns the slot.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "checkers/GpuDefs.hpp"

namespace checkers {

class StagingPool {
public:
    struct Slot {
        void*        d_buf = nullptr;   // device staging buffer  [slot_bytes]
        void*        h_buf = nullptr;   // pinned host buffer      [slot_bytes]
        gpu::Event_t copied{};          // recorded on copy_stream after the D2D
    };

    StagingPool() = default;
    ~StagingPool();
    StagingPool(const StagingPool&)            = delete;
    StagingPool& operator=(const StagingPool&) = delete;

    void allocate(size_t   slot_bytes,
                  uint32_t ring_depth,
                  uint32_t n_compute_streams);
    void release();
    bool allocated() const { return !slots_.empty(); }

    int  acquire();              // blocks until a slot is free; returns its index
    void give_back(int slot);    // return a slot to the free list

    Slot&         slot(int i)              { return slots_[static_cast<size_t>(i)]; }
    gpu::Stream_t copy_stream() const      { return copy_stream_; }
    gpu::Stream_t compute_stream(int i) const {
        return compute_streams_[static_cast<size_t>(i) % compute_streams_.size()];
    }
    uint32_t ring_depth() const            { return static_cast<uint32_t>(slots_.size()); }
    uint32_t n_compute_streams() const     { return static_cast<uint32_t>(compute_streams_.size()); }
    size_t   slot_bytes() const            { return slot_bytes_; }

    // ---- memory accounting ----
    size_t device_bytes() const { return slot_bytes_ * slots_.size(); }
    size_t pinned_bytes() const { return slot_bytes_ * slots_.size(); }

private:
    std::vector<Slot>          slots_;
    std::vector<gpu::Stream_t> compute_streams_;
    gpu::Stream_t              copy_stream_ = nullptr;
    size_t                     slot_bytes_  = 0;

    std::mutex               mtx_;
    std::condition_variable  cv_;
    std::vector<int>         free_list_;
};

} // namespace checkers
