#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

// Defined in api.cpp
namespace checkers {

// Defined in SelfTest.cpp
py::dict codec_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                        const std::string& dtype_str,
                        double eps, uint32_t group, int zstd_level,
                        bool global_step, bool log_pw_rel, int zstd_workers,
                        bool legacy_fwd, int entropy);

py::dict tt_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                     uint32_t rank_cap, uint32_t ndim, double energy_keep);

py::dict tt_eq_selftest(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                        double eps, uint32_t rank_cap, uint32_t ndim,
                        double energy_keep, int eq_mode, int zstd_level,
                        int zstd_workers, int entropy);

py::tuple quant_roundtrip(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                          double eps, const std::string& step_mode,
                          uint32_t bucket, int zstd_level);

py::dict expavg_probe(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                      const std::string& variant, double eps,
                      const std::string& step_mode, uint32_t bucket,
                      uint32_t rank_cap, uint32_t rank_f16, uint32_t ndim,
                      double energy_keep, int zstd_level,
                      uint32_t n1, uint32_t n2, uint32_t n3);

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

py::dict compress_and_save(const std::string& output_dir,
                           py::object         error_bound,
                           py::object         targets,
                           bool               blocking,
                           bool               allow_lossless);

py::dict checkpoint_join();

py::dict restart_from_checkpoint(py::object         model_engine,
                                 const std::string& input_dir,
                                 py::object         optimizer,
                                 py::object         targets);

void shutdown();

} // namespace checkers

PYBIND11_MODULE(checkers_py, m) {
    m.doc() = "Checkers -- online checkpoint data compressor";

    m.def("initialize_context",
          &checkers::initialize_context,
          py::arg("model_engine"),
          py::arg("optimizer")         = py::none(),
          py::arg("stage_targets")     = py::none(),   // None => all four streams
          py::arg("staging_budget_mb") = static_cast<int64_t>(0),
          py::arg("group_size")        = static_cast<uint32_t>(256),
          py::arg("zstd_level")        = static_cast<int32_t>(3),
          py::arg("compute_streams")   = static_cast<uint32_t>(4),
          py::arg("measure_snr")       = true,
          py::arg("per_group")         = true,
          py::arg("exp_avg_sq_aggressive") = false,
          py::arg("tt_ndim")           = static_cast<uint32_t>(3),
          py::arg("tt_max_rank")       = static_cast<uint32_t>(16),
          py::arg("tt_energy_keep")    = 0.999,
          py::arg("tt_eq_mode")        = static_cast<int32_t>(1),
          "Discover this rank's shard and allocate the persistent staging pool "
          "(device ring buffers, streams, worker pool). Call once after the "
          "first real optimizer step. exp_avg_sq_aggressive=True switches the "
          "exp_avg_sq codec from log-domain pointwise-relative (safe, ~4-7x) to "
          "single global RMS (aggressive, ~25x, restart diverges).");

    m.def("compress_and_save",
          &checkers::compress_and_save,
          py::arg("output_dir"),
          py::arg("error_bound"),
          py::arg("targets")  = py::none(),            // None => all staged streams
          py::arg("blocking") = true,
          py::arg("allow_lossless") = false,
          "Snapshot the selected streams on the critical path, then compress + "
          "write them (per-stream .cch files). Returns a metrics dict "
          "{overall, per_stream, memory}. blocking=False returns after the "
          "snapshot; call checkpoint_join() for the full metrics.\n"
          "error_bound: float (all streams) OR dict keyed by stream name "
          "('model_state'/'master_weights'/'exp_avg'/'exp_avg_sq'), optional "
          "'default' key for the rest, e.g. {'master_weights':0.02,'exp_avg':0.3}.\n"
          "allow_lossless: False (default) rejects a 0 bound on a targeted stream. "
          "True lets an EXPLICIT 0 select the lossless raw+zstd path for that stream.");

    m.def("checkpoint_join",
          &checkers::checkpoint_join,
          "Block until the most recent non-blocking compress_and_save has "
          "finished its background compression + I/O; returns its metrics dict.");

    m.def("restart_from_checkpoint",
          &checkers::restart_from_checkpoint,
          py::arg("model_engine"),
          py::arg("input_dir"),
          py::arg("optimizer") = py::none(),
          py::arg("targets")   = py::none(),
          "Read compressed streams back from disk and refill the live tensors. "
          "Streams with no .cch file are reported under 'missing'.");

    m.def("shutdown", &checkers::shutdown,
          "Release the staging pool, join background threads, and drop held "
          "Python tensor refs. Auto-registered with atexit; safe to call twice.");

    m.def("codec_selftest", &checkers::codec_selftest,
          py::arg("arr"),
          py::arg("dtype")       = std::string("float32"),
          py::arg("eps")         = 0.05,
          py::arg("group")       = static_cast<uint32_t>(256),
          py::arg("zstd_level")  = 3,
          py::arg("global_step") = false,
          py::arg("log_pw_rel")  = false,
          py::arg("zstd_workers") = 0,
          py::arg("legacy_fwd")   = false,
          py::arg("entropy")      = 0,   // 0 = zstd (CPU), 1 = GPU bit-pack
          "Run the uniform quant codec on one host array (GPU round-trip); "
          "returns ratio / SNR / rmse / max_rel_err / n_over. "
          "global_step=False = per-group step (Phase B, weights); "
          "global_step=True = one tensor-wide RMS step (Phase C aggressive). "
          "log_pw_rel=True = quantize log(v), one step D=2*log(1+eps), "
          "v_hat=exp(q*D) (Phase C default for exp_avg_sq). "
          "Not used by the training path.");

    m.def("tt_selftest", &checkers::tt_selftest,
          py::arg("arr"),
          py::arg("rank_cap")    = static_cast<uint32_t>(16),
          py::arg("ndim")        = static_cast<uint32_t>(3),
          py::arg("energy_keep") = 0.999,
          "Randomized TT-SVD decompose+reconstruct round trip (no quantization) "
          "on one host array; returns ranks chosen, low-rank fraction, rel_l2, "
          "SNR. Validates TtLinalg in isolation.");

    m.def("tt_eq_selftest", &checkers::tt_eq_selftest,
          py::arg("arr"),
          py::arg("eps")         = 0.05,
          py::arg("rank_cap")    = static_cast<uint32_t>(16),
          py::arg("ndim")        = static_cast<uint32_t>(3),
          py::arg("energy_keep") = 0.999,
          py::arg("eq_mode")     = 1,           // 0 ABS, 1 REL (eps*range)
          py::arg("zstd_level")  = 3,
          py::arg("zstd_workers") = 0,
          py::arg("entropy")      = 0,   // 0 = zstd (CPU), 1 = GPU bit-pack
          "Full exp_avg codec (capped TT-SVD -> EQ error-bounded residual quant "
          "-> zstd) forward+inverse on one host array; returns ratio / SNR / "
          "rel_l2 / max_abs_err / ranks / core bytes.");

    m.def("quant_roundtrip", &checkers::quant_roundtrip,
          py::arg("arr"), py::arg("eps") = 0.05,
          py::arg("step_mode") = std::string("bucket_rms"),
          py::arg("bucket") = static_cast<uint32_t>(4096),
          py::arg("zstd_level") = 3,
          "Quantize an array with Checkers' uniform quantizer; returns "
          "(dequantized array, serialized bytes, n_overflow).  Measurement "
          "only -- lives in SelfTest.cpp, off the production path.");

    m.def("expavg_probe", &checkers::expavg_probe,
          py::arg("arr"),
          py::arg("variant")     = std::string("tt_resid"),
          py::arg("eps")         = 0.05,
          py::arg("step_mode")   = std::string("global_range"),
          py::arg("bucket")      = static_cast<uint32_t>(0),
          py::arg("rank_cap")    = static_cast<uint32_t>(16),
          py::arg("rank_f16")    = static_cast<uint32_t>(16),
          py::arg("ndim")        = static_cast<uint32_t>(3),
          py::arg("energy_keep") = 0.999,
          py::arg("zstd_level")  = 3,
          py::arg("n1")          = static_cast<uint32_t>(0),
          py::arg("n2")          = static_cast<uint32_t>(0),
          py::arg("n3")          = static_cast<uint32_t>(0),
          "Measurement-only exp_avg codec variants.  MEASUREMENT ONLY: lives in "
          "SelfTest.cpp, which the compressor never calls, so none of these "
          "options can reach the production path.  The defaults reproduce the "
          "shipped codec (TT cores fp16 + element-wise residual, one global "
          "range step, rank cap 16).\n"
          "  variant   nott | tt_resid | tt_rankquant\n"
          "  step_mode global_range | bucket_rms | bucket_range\n"
          "Returns SNR, cosine, sign agreement, zero fractions and the "
          "compression ratio for the stream.");

    // Run cleanup while the interpreter is still alive (GIL valid) rather than
    // letting static destructors free py::object refs after Py_Finalize.
    py::module_::import("atexit").attr("register")(
        py::cpp_function([]() { checkers::shutdown(); }));
}
