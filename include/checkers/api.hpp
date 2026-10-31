#ifndef CHECKERS_API_HPP
#define CHECKERS_API_HPP

// Public entry points for the Checkers online checkpoint compressor.
//
//   initialize_context(...)        one-time: discover this rank's shard,
//                                  allocate the persistent staging pool,
//                                  create streams + worker pool.
//   compress_and_save(...)         per checkpoint: snapshot selected streams
//                                  on the critical path, then compress +
//                                  write asynchronously.  Returns a metrics
//                                  dict (ratio / SNR / timing / memory,
//                                  overall and per stream).
//   restart_from_checkpoint(...)   read compressed streams back from disk,
//                                  decompress, and refill the live tensors.
//                                  (Built now, exercised later.)
//
// A "stream" here is one optimizer/​model category:
//   model_state (bf16/fp16), master_weights (fp32),
//   exp_avg (fp32), exp_avg_sq (fp32).
// Each stream is written to its own per-rank file so a downstream
// perturbation study can touch one stream in isolation.

#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>

#include "TensorResource.hpp"   // TensorCategory, DataType

namespace py = pybind11;

namespace checkers {

// ------------------------------------------------------------------ //
//  Stream selection knob
// ------------------------------------------------------------------ //
// Bit i corresponds to tensor_category_index(category).
enum class CompressTargets : uint32_t {
    None          = 0u,
    ModelState    = 1u << 0,   // tensor_category_index(TensorCategory::ModelState)
    MasterWeights = 1u << 1,
    ExpAvg        = 1u << 2,
    ExpAvgSq      = 1u << 3,
    All           = ModelState | MasterWeights | ExpAvg | ExpAvgSq,
};

inline uint32_t target_bit(TensorCategory c) {
    return 1u << tensor_category_index(c);
}
inline bool target_selected(uint32_t mask, TensorCategory c) {
    return (mask & target_bit(c)) != 0u;
}

// Parse a Python selector into a bitmask.
//   None / "all"                     -> All
//   "master_weights" / "exp_avg" ... -> that single bit
//   ["exp_avg", "exp_avg_sq"]        -> OR of the bits
uint32_t parse_targets(const py::object& selector);

// ------------------------------------------------------------------ //
//  Codec identifiers (on-disk + metrics)
// ------------------------------------------------------------------ //
enum class CodecId : uint32_t {
    RawZstd           = 0,   // lossless: zstd(raw bytes)                       [Phase A]
    RmsRelPgQuant     = 1,   // per-group RMS-relative uniform quant + zstd     [Phase B]
    LogPwRelQuant     = 2,   // log-domain, hard pointwise-relative quant + zstd [Phase C default, exp_avg_sq]
    TtEqResidual      = 3,   // capped TT-SVD -> EQ error-bounded residual quant [Phase D, exp_avg]
    RmsRelGlobalQuant = 4,   // single global RMS-relative uniform quant + zstd [Phase C aggressive, opt-in]
};

// ------------------------------------------------------------------ //
//  Configuration (set once, at initialize_context)
// ------------------------------------------------------------------ //
struct CompressionConfig {
    uint32_t stage_targets      = static_cast<uint32_t>(CompressTargets::All);
    int64_t  staging_budget_mb  = 0;      // 0 => auto (~25% of free VRAM at init)
    uint32_t staging_ring_depth = 3;      // device staging buffers in flight
    uint32_t group_size         = 256;    // per-group quant grouping
    int32_t  zstd_level         = 3;      // default entropy-stage effort
    uint32_t compute_streams    = 4;      // background GPU streams / workers
    bool     measure_snr        = true;   // dequant + on-GPU error reduction
    bool     per_group          = true;   // RMS_REL_PG vs single global step

    // exp_avg_sq codec:
    //   false (default) -> LogPwRelQuant: log-domain, hard per-element relative
    //                      bound, no zeroing (~4-7x).  Adam's 1/(sqrt(v)+eps)
    //                      is preserved for every weight -> restart converges.
    //   true            -> RmsRelGlobalQuant: single global RMS step, the bulk
    //                      collapses to 0 (~20-30x).  Aggressive; for the
    //                      "is the second moment disposable" study only --
    //                      it drives the restart loss to a divergence cliff.
    bool     exp_avg_sq_aggressive = false;

    // exp_avg codec (TtEqResidual): capped TensorTrain-SVD then an EQ
    // error-bounded quantizer on the truncation residual.
    uint32_t tt_ndim        = 3;      // reshape the 1-D shard to this many modes (2 or 3)
    uint32_t tt_max_rank    = 16;     // hard cap on every TT bond ("artificially low-rank")
    double   tt_energy_keep = 0.999;  // within the cap, smallest rank keeping this energy
    int32_t  tt_eq_mode     = 1;      // residual EQ: 0 ABS, 1 REL (eps*range), 2 PW_REL
};

// ------------------------------------------------------------------ //
//  Public API
// ------------------------------------------------------------------ //
void initialize_context(py::object model_engine,
                        py::object optimizer,
                        py::object stage_targets,
                        int64_t    staging_budget_mb,
                        uint32_t   group_size,
                        int32_t    zstd_level,
                        uint32_t   compute_streams,
                        bool       measure_snr,
                        bool       per_group,
                        bool       exp_avg_sq_aggressive,
                        uint32_t   tt_ndim,
                        uint32_t   tt_max_rank,
                        double     tt_energy_keep,
                        int32_t    tt_eq_mode);

// Returns a nested dict:  { "overall": {...}, "per_stream": {...}, "memory": {...} }
// error_bound may be a float (applied to every stream) or a dict keyed by
// stream name ("model_state" / "master_weights" / "exp_avg" / "exp_avg_sq"),
// with an optional "default" key for the rest.  allow_lossless=true lets an
// explicit 0.0 bound select the lossless raw+zstd path for that stream.
py::dict compress_and_save(const std::string& output_dir,
                           py::object         error_bound,
                           py::object         targets,
                           bool               blocking,
                           bool               allow_lossless);

// Block until the most recent non-blocking compress_and_save has finished
// its background compression + I/O.  Returns the same metrics dict.
py::dict checkpoint_join();

py::dict restart_from_checkpoint(py::object         model_engine,
                                 const std::string& input_dir,
                                 py::object         optimizer,
                                 py::object         targets);

// Release the staging pool, join background threads, and drop all held Python
// tensor refs.  Registered with Python's atexit so this runs while the
// interpreter is still alive (GIL valid) instead of at static-destructor time.
void shutdown();

} // namespace checkers

#endif // CHECKERS_API_HPP
