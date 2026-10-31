#include "checkers/api.hpp"
#include "checkers/TensorResource.hpp"
#include "checkers/CheckpointCompressor.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/logging.hpp"
#include <pybind11/pybind11.h>     // Include pybind11 BEFORE referencing py::...
#include <pybind11/eval.h>         // Often needed for py::str
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace checkers {

// TensorExtraction is defined in checkers/TensorResource.hpp

static DataType parse_dtype(const std::string& s);

namespace {

int extract_rank(py::object model_engine, const char* attr_name, const char* env_name) {
    if (py::hasattr(model_engine, attr_name)) {
        return model_engine.attr(attr_name).cast<int>();
    }
    if (const char* value = std::getenv(env_name)) {
        return std::atoi(value);
    }
    return -1;
}

void fill_shape_and_stride(py::handle handle,
                           py::object py_list_func,
                           TensorMetadata& meta) {
    if (py::hasattr(handle, "shape")) {
        py::list standard_shape_list = py_list_func(handle.attr("shape"));
        for (auto item : standard_shape_list) {
            meta.shape.push_back(item.cast<size_t>());
        }
    }
    if (py::hasattr(handle, "stride")) {
        py::list standard_stride_list = py_list_func(handle.attr("stride")());
        for (auto item : standard_stride_list) {
            meta.strides.push_back(item.cast<ptrdiff_t>());
        }
    }
}

bool try_tensor_like(py::handle handle,
                     const std::string& source,
                     py::object py_list_func,
                     TensorExtraction& extraction);

int extract_group_rank(py::object param, int fallback_rank);

bool try_zero_partitioned_tensor(py::object param,
                                 py::handle handle,
                                 int fallback_rank,
                                 py::object py_list_func,
                                 TensorExtraction& extraction);

py::object resolve_optimizer(py::object model_engine, py::object optimizer) {
    if (!optimizer.is_none()) {
        return optimizer;
    }
    if (py::hasattr(model_engine, "optimizer")) {
        py::object engine_optimizer = model_engine.attr("optimizer");
        if (!engine_optimizer.is_none()) {
            return engine_optimizer;
        }
    }
    return py::none();
}

py::object resolve_optimizer_state_mapping(py::object optimizer) {
    if (optimizer.is_none()) {
        return py::none();
    }
    if (py::hasattr(optimizer, "state")) {
        py::object state = optimizer.attr("state");
        if (!state.is_none()) {
            return state;
        }
    }
    if (py::hasattr(optimizer, "optimizer")) {
        py::object base_optimizer = optimizer.attr("optimizer");
        if (!base_optimizer.is_none() && py::hasattr(base_optimizer, "state")) {
            py::object state = base_optimizer.attr("state");
            if (!state.is_none()) {
                return state;
            }
        }
    }
    return py::none();
}

std::vector<std::string> collect_optimizer_state_keys(py::object optimizer_state) {
    std::vector<std::string> keys;
    std::unordered_set<std::string> unique_keys;
    if (optimizer_state.is_none()) {
        return keys;
    }

    py::dict state_dict = optimizer_state.cast<py::dict>();
    for (auto item : state_dict) {
        py::handle state_value = item.second;
        if (!py::isinstance<py::dict>(state_value)) {
            continue;
        }
        py::dict per_param_state = state_value.cast<py::dict>();
        for (auto state_item : per_param_state) {
            const std::string key = py::str(state_item.first).cast<std::string>();
            if (!py::hasattr(state_item.second, "data_ptr")) {
                continue;
            }
            if (unique_keys.insert(key).second) {
                keys.push_back(key);
            }
        }
        if (!keys.empty()) {
            break;
        }
    }

    return keys;
}

bool submit_extraction(MemoryManager& mgr,
                       const std::string& name,
                       TensorCategory category,
                       TensorExtraction& extraction,
                       size_t histogram_bins,
                       double discovery_ms) {
    extraction.meta.name = name;
    extraction.meta.category = category;
    extraction.meta.histogram_bins = histogram_bins;   // legacy field; unused by the compressor
    TensorMetadata submitted_meta = extraction.meta;
    mgr.register_shard_tensor(name, extraction.ptr, extraction.byte_size, std::move(extraction.meta));
    mgr.note_discovery(submitted_meta, discovery_ms);
    return true;
}

bool has_zero3_flat_groups(py::object optimizer) {
    return !optimizer.is_none()
        && py::hasattr(optimizer, "fp16_partitioned_groups_flat")
        && py::hasattr(optimizer, "fp32_partitioned_groups_flat");
}

bool try_submit_tensor_handle(py::handle handle,
                              const std::string& name,
                              const std::string& source,
                              TensorCategory category,
                                 py::object py_list_func,
                                 size_t histogram_bins,
                                 MemoryManager& mgr,
                                 std::vector<py::object>& lifetime_guard) {
    const auto discovery_start = std::chrono::steady_clock::now();
    TensorExtraction extraction;
    if (!try_tensor_like(handle, source, py_list_func, extraction)) {
        return false;
    }

    lifetime_guard.push_back(py::reinterpret_borrow<py::object>(handle));
    const double discovery_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - discovery_start).count();
    return submit_extraction(mgr, name, category, extraction, histogram_bins, discovery_ms);
}

size_t resident_partition_numel(py::object param, int fallback_rank) {
    size_t logical_numel = 0;
    if (py::hasattr(param, "ds_numel")) {
        logical_numel = param.attr("ds_numel").cast<size_t>();
    }

    size_t partition_numel = 0;
    if (py::hasattr(param, "partition_numel")) {
        partition_numel = param.attr("partition_numel")().cast<size_t>();
    }

    if (logical_numel == 0 || partition_numel == 0) {
        return 0;
    }

    const int shard_rank = extract_group_rank(param, fallback_rank);
    const size_t shard_offset = shard_rank < 0 ? 0 : static_cast<size_t>(shard_rank) * partition_numel;
    if (shard_offset >= logical_numel) {
        return 0;
    }

    return std::min(partition_numel, logical_numel - shard_offset);
}

void apply_logical_metadata(py::object param, TensorMetadata& meta, py::object py_list_func) {
    if (py::hasattr(param, "ds_shape")) {
        py::list logical_shape_list = py_list_func(param.attr("ds_shape"));
        meta.logical_shape.clear();
        for (auto item : logical_shape_list) {
            meta.logical_shape.push_back(item.cast<size_t>());
        }
    }
    if (py::hasattr(param, "ds_numel")) {
        meta.logical_num_elements = param.attr("ds_numel").cast<size_t>();
        meta.logical_byte_size = meta.logical_num_elements * meta.element_size;
    }
}

bool submit_partition_slice(py::object flat_tensor,
                           py::object param,
                           const std::string& name,
                           const std::string& source,
                           TensorCategory category,
                           size_t offset,
                           size_t resident_numel,
                           py::object py_list_func,
                           size_t histogram_bins,
                           MemoryManager& mgr,
                           std::vector<py::object>& lifetime_guard) {
    if (resident_numel == 0) {
        return false;
    }

    py::object slice = flat_tensor.attr("narrow")(0, offset, resident_numel);
    const auto discovery_start = std::chrono::steady_clock::now();
    TensorExtraction extraction;
    if (!try_tensor_like(slice, source, py_list_func, extraction)) {
        return false;
    }

    extraction.meta.shape = {resident_numel};
    extraction.meta.strides = {1};
    extraction.meta.num_elements = resident_numel;
    extraction.meta.byte_size = resident_numel * extraction.meta.element_size;
    apply_logical_metadata(param, extraction.meta, py_list_func);

    lifetime_guard.push_back(slice);
    const double discovery_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - discovery_start).count();
    return submit_extraction(mgr, name, category, extraction, histogram_bins, discovery_ms);
}

// bool submit_local_tensor(py::handle tensor_handle,
//                          py::object param,
//                          const std::string& name,
//                          const std::string& source,
//                          TensorCategory category,
//                          py::object py_list_func,
//                          size_t histogram_bins,
//                          MemoryManager& mgr,
//                          std::vector<py::object>& lifetime_guard) {
//     const auto discovery_start = std::chrono::steady_clock::now();
//     TensorExtraction extraction;
//     if (!try_tensor_like(tensor_handle, source, py_list_func, extraction)) {
//         return false;
//     }

//     apply_logical_metadata(param, extraction.meta, py_list_func);
//     lifetime_guard.push_back(py::reinterpret_borrow<py::object>(tensor_handle));
//     const double discovery_ms = std::chrono::duration<double, std::milli>(
//         std::chrono::steady_clock::now() - discovery_start).count();
//     return submit_extraction(mgr, name, category, extraction, histogram_bins, discovery_ms);
// }

bool submit_local_tensor(py::handle tensor_handle,
                         py::object param,
                         const std::string& name,
                         const std::string& source,
                         TensorCategory category,
                         py::object py_list_func,
                         size_t histogram_bins,
                         MemoryManager& mgr,
                         std::vector<py::object>& lifetime_guard) {
    py::object tensor_obj = py::reinterpret_borrow<py::object>(tensor_handle);
    size_t resident_numel = 0;
    if (py::hasattr(tensor_obj, "numel")) {
        resident_numel = tensor_obj.attr("numel")().cast<size_t>();
    }
    const size_t expected_numel = resident_partition_numel(param, /*fallback_rank=*/0);
    if (expected_numel != 0) {
        resident_numel = resident_numel == 0 ? expected_numel : std::min(resident_numel, expected_numel);
    }
    if (resident_numel == 0) {
        return false;
    }

    py::object shard = tensor_obj;
    if (py::hasattr(tensor_obj, "numel")
        && tensor_obj.attr("numel")().cast<size_t>() > resident_numel
        && py::hasattr(tensor_obj, "narrow")) {
        shard = tensor_obj.attr("narrow")(0, 0, resident_numel);
    }

    const auto discovery_start = std::chrono::steady_clock::now();
    TensorExtraction extraction;
    if (!try_tensor_like(shard, source, py_list_func, extraction)) {
        return false;
    }

    extraction.meta.num_elements = resident_numel;
    extraction.meta.byte_size = resident_numel * extraction.meta.element_size;
    extraction.meta.shape = {resident_numel};
    extraction.meta.strides = {1};
    apply_logical_metadata(param, extraction.meta, py_list_func);

    lifetime_guard.push_back(shard);

    const double discovery_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - discovery_start).count();
    return submit_extraction(mgr, name, category, extraction, histogram_bins, discovery_ms);
}

py::object get_local_zero3_optimizer_state(py::object z3_optimizer,
                                           py::object param,
                                           const char* state_key,
                                           py::object py_list_func,
                                           std::string* failure_reason = nullptr) {
    if (z3_optimizer.is_none()) {
        if (failure_reason) {
            *failure_reason = "z3 optimizer is none";
        }
        return py::none();
    }

    if (py::hasattr(z3_optimizer, "get_local_fp32_param")) {
        try {
            return z3_optimizer.attr("get_local_fp32_param")(param, py::str(state_key));
        } catch (const py::error_already_set& e) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = e.what();
            }
        } catch (...) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = "get_local_fp32_param threw a non-python exception";
            }
        }
    }

    if (!py::hasattr(z3_optimizer, "grad_position")
        || !py::hasattr(z3_optimizer, "get_param_id")
        || !py::hasattr(z3_optimizer, "fp32_partitioned_groups_flat")
        || !py::hasattr(z3_optimizer, "optimizer")) {
        if (failure_reason && failure_reason->empty()) {
            *failure_reason = "z3 optimizer missing grad_position/get_param_id/fp32_partitioned_groups_flat/optimizer";
        }
        return py::none();
    }

    try {
        py::object param_id = z3_optimizer.attr("get_param_id")(param);
        py::object grad_position = z3_optimizer.attr("grad_position");
        py::tuple location = grad_position[param_id].cast<py::tuple>();
        const size_t group_index = location[0].cast<size_t>();
        const size_t dest_offset = location[1].cast<size_t>();
        const size_t num_elements = location[2].cast<size_t>();

        py::list fp32_groups = py_list_func(z3_optimizer.attr("fp32_partitioned_groups_flat"));
        if (group_index >= fp32_groups.size()) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = "group index out of range for fp32_partitioned_groups_flat";
            }
            return py::none();
        }

        py::object fp32_param = py::reinterpret_borrow<py::object>(fp32_groups[group_index]);
        py::object base_optimizer = z3_optimizer.attr("optimizer");
        if (base_optimizer.is_none() || !py::hasattr(base_optimizer, "state")) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = "base optimizer missing state";
            }
            return py::none();
        }

        py::object state_entry_obj = base_optimizer.attr("state").attr("get")(fp32_param, py::none());
        if (state_entry_obj.is_none() || !py::isinstance<py::dict>(state_entry_obj)) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = "base optimizer state has no entry for fp32 flat group";
            }
            return py::none();
        }

        py::dict state_entry = state_entry_obj.cast<py::dict>();
        py::object key = py::str(state_key);
        if (!state_entry.contains(key)) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = std::string("state entry missing key ") + state_key;
            }
            return py::none();
        }

        py::object state_tensor = state_entry[key];
        if (state_tensor.is_none()) {
            if (failure_reason && failure_reason->empty()) {
                *failure_reason = std::string("state tensor for key ") + state_key + " is none";
            }
            return py::none();
        }

        return state_tensor.attr("narrow")(0, dest_offset, num_elements);
    } catch (const py::error_already_set& e) {
        if (failure_reason && failure_reason->empty()) {
            *failure_reason = e.what();
        }
        return py::none();
    } catch (...) {
        if (failure_reason && failure_reason->empty()) {
            *failure_reason = "fallback optimizer-state lookup threw a non-python exception";
        }
        return py::none();
    }
}

size_t discover_zero3_flat_states(py::object optimizer,
                                  const py::list& named_params,
                                  int global_rank,
                                  py::object py_list_func,
                                  size_t histogram_bins,
                                  MemoryManager& mgr,
                                  const std::shared_ptr<RankLogger>& logger,
                                  std::vector<py::object>& lifetime_guard,
                                  size_t& skipped_tensors) {
    // ------------------------------------------------------------------
    // GLOBAL FLAT VIEW
    //
    // ZeRO-3 avoids replication by assigning each rank responsibility for
    // a contiguous slice of the globally-concatenated parameter space.
    // We compute cumulative ds_numel offsets for every parameter and only
    // track those whose logical range overlaps with this rank's slice:
    //
    //   rank_start = global_rank * flat_group.numel()
    //   rank_end   = (global_rank + 1) * flat_group.numel()
    //
    // For overlapping parameters we use the grad_position offsets to
    // access the rank's actual local shard in the flat buffer, so the
    // data is always physically correct. Different ranks will therefore
    // track different sets of parameters, matching the on-disk checkpoint
    // sharding structure.
    // ------------------------------------------------------------------

    // Collect flat buffer groups
    py::list fp32_groups, fp16_groups;
    if (py::hasattr(optimizer, "fp32_partitioned_groups_flat"))
        fp32_groups = py_list_func(optimizer.attr("fp32_partitioned_groups_flat"));
    if (py::hasattr(optimizer, "fp16_partitioned_groups_flat"))
        fp16_groups = py_list_func(optimizer.attr("fp16_partitioned_groups_flat"));

    // Optimizer state dict (keyed by fp32 flat tensors -> {"exp_avg": ..., ...}).
    // Non-empty only after at least one real optimizer step.
    py::dict optimizer_state_dict;
    bool has_opt_state = false;
    if (py::hasattr(optimizer, "optimizer")) {
        py::object base_opt = optimizer.attr("optimizer");
        if (!base_opt.is_none() && py::hasattr(base_opt, "state")) {
            py::object raw_state = base_opt.attr("state");
            if (!raw_state.is_none() && py::isinstance<py::dict>(raw_state)) {
                optimizer_state_dict = raw_state.cast<py::dict>();
                has_opt_state = !optimizer_state_dict.empty();
            }
        }
    }

    const bool has_grad_pos = py::hasattr(optimizer, "grad_position")
                              && py::hasattr(optimizer, "get_param_id");

    if (logger) {
        logger->log_message("[checkers][zero3] fp32_groups="
            + std::to_string(fp32_groups.size())
            + " fp16_groups=" + std::to_string(fp16_groups.size())
            + " optimizer_state_entries=" + std::to_string(optimizer_state_dict.size())
            + " has_grad_position=" + (has_grad_pos ? "true" : "false"));
    }

    const size_t num_groups = fp32_groups.size();
    if (num_groups == 0) return 0;

    // ------------------------------------------------------------------
    // Build per-group param info sorted by flat_offset so we can walk
    // them in order and accumulate the cumulative ds_numel global offset.
    // ------------------------------------------------------------------
    struct ParamInfo {
        std::string name;
        py::object  param;
        size_t      group_idx;
        size_t      flat_offset;  // grad_position offset (partition_numel-based)
        size_t      flat_numel;   // partition_numel (elements in this rank's flat buffer)
        size_t      ds_numel;     // full logical element count for this parameter
    };
    std::vector<std::vector<ParamInfo>> group_params(num_groups);

    if (has_grad_pos) {
        for (auto item : named_params) {
            auto tuple_item = item.cast<py::tuple>();
            const std::string pname = tuple_item[0].cast<std::string>();
            py::object param = tuple_item[1];

            if (!py::hasattr(param, "requires_grad")
                || !param.attr("requires_grad").cast<bool>()
                || !py::hasattr(param, "ds_numel")) {
                continue;
            }
            try {
                py::object param_id = optimizer.attr("get_param_id")(param);
                py::object loc_obj  = optimizer.attr("grad_position").attr("get")(param_id, py::none());
                if (loc_obj.is_none()) continue;

                py::list loc = py_list_func(loc_obj);
                const size_t gidx   = py::reinterpret_borrow<py::object>(loc[0]).cast<size_t>();
                const size_t foff   = py::reinterpret_borrow<py::object>(loc[1]).cast<size_t>();
                const size_t fnumel = py::reinterpret_borrow<py::object>(loc[2]).cast<size_t>();
                const size_t dsn    = param.attr("ds_numel").cast<size_t>();

                if (gidx < num_groups && fnumel > 0 && dsn > 0)
                    group_params[gidx].push_back({pname, param, gidx, foff, fnumel, dsn});
            } catch (...) {}
        }
        // Sort each group's params by their position inside the flat buffer
        for (auto& params : group_params) {
            std::sort(params.begin(), params.end(),
                [](const ParamInfo& a, const ParamInfo& b) {
                    return a.flat_offset < b.flat_offset;
                });
        }
    }

    size_t discovered = 0;
    std::vector<std::string> tracked_names;  // for per-rank logging

    for (size_t g = 0; g < num_groups; ++g) {
        if (g >= fp32_groups.size() || group_params[g].empty()) continue;

        py::object fp32_flat = py::reinterpret_borrow<py::object>(fp32_groups[g]);
        const size_t flat_size = fp32_flat.attr("numel")().cast<size_t>();
        if (flat_size == 0) continue;

        // group_params[g] already holds EVERY param with a grad_position entry
        // for this rank's partition of group g; (flat_offset, flat_numel) is
        // exactly that param's slice inside fp32_flat, and those slices tile
        // [0, flat_size).  grad_position IS the authoritative ZeRO-3 ownership
        // map -- so there is no further filtering to do here.  (An earlier
        // version additionally filtered by cumulative full-model ds_numel
        // territory, comparing full-model offsets against per-rank flat-buffer
        // bounds; that dropped ~7/8 of every rank's shard -> only 13% of the
        // model was ever captured.)

        // bf16 flat buffer for model state (same layout as fp32, different dtype)
        py::object fp16_flat = py::none();
        if (g < fp16_groups.size())
            fp16_flat = py::reinterpret_borrow<py::object>(fp16_groups[g]);

        // Optimizer state tensors for this group
        py::object exp_avg_flat = py::none(), exp_avg_sq_flat = py::none();
        if (has_opt_state && optimizer_state_dict.contains(fp32_flat)) {
            py::dict st = optimizer_state_dict[fp32_flat].cast<py::dict>();
            const py::str ea_key("exp_avg"), easq_key("exp_avg_sq");
            if (st.contains(ea_key))   exp_avg_flat   = st[ea_key];
            if (st.contains(easq_key)) exp_avg_sq_flat = st[easq_key];
        }

        // Every param slice grad_position reported for this rank -> register it.
        for (const auto& pinfo : group_params[g]) {
            // grad_position offsets index directly into this rank's flat buffer.
            const size_t dest_offset  = pinfo.flat_offset;
            const size_t num_elements = pinfo.flat_numel;
            size_t n_for_param = 0;

            // Model state (bf16): prefer fp16_partitioned_groups_flat, fall back to ds_tensor
            if (!fp16_flat.is_none()) {
                if (submit_partition_slice(fp16_flat, pinfo.param,
                                           pinfo.name,
                                           "zero3.fp16_partitioned_groups_flat",
                                           TensorCategory::ModelState,
                                           dest_offset, num_elements,
                                           py_list_func, histogram_bins, mgr,
                                           lifetime_guard)) {
                    ++discovered; ++n_for_param;
                    tracked_names.push_back(pinfo.name);
                    mgr.register_frozen_param(pinfo.name);
                }
            } else if (py::hasattr(pinfo.param, "ds_tensor")
                       && !pinfo.param.attr("ds_tensor").is_none()) {
                py::object ds_t = pinfo.param.attr("ds_tensor");
                const auto t0 = std::chrono::steady_clock::now();
                TensorExtraction ext;
                if (try_zero_partitioned_tensor(pinfo.param, ds_t, global_rank,
                                                py_list_func, ext)) {
                    apply_logical_metadata(pinfo.param, ext.meta, py_list_func);
                    lifetime_guard.push_back(ds_t);
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    if (submit_extraction(mgr, pinfo.name, TensorCategory::ModelState,
                                          ext, histogram_bins, ms)) {
                        ++discovered; ++n_for_param;
                        tracked_names.push_back(pinfo.name);
                        mgr.register_frozen_param(pinfo.name);
                    }
                }
            }

            // Master weights (fp32 shard)
            if (submit_partition_slice(fp32_flat, pinfo.param,
                                       "master_weights::" + pinfo.name,
                                       "zero3.fp32_partitioned_groups_flat",
                                       TensorCategory::MasterWeights,
                                       dest_offset, num_elements,
                                       py_list_func, histogram_bins, mgr,
                                       lifetime_guard)) {
                ++discovered; ++n_for_param;
                tracked_names.push_back("master_weights::" + pinfo.name);
                mgr.register_frozen_param("master_weights::" + pinfo.name);
            }

            // Optimizer states – only available after the first real optimizer step
            if (has_opt_state && !exp_avg_flat.is_none()) {
                if (submit_partition_slice(exp_avg_flat, pinfo.param,
                                           "optimizer.exp_avg::" + pinfo.name,
                                           "zero3.optimizer.exp_avg",
                                           TensorCategory::OptimizerExpAvg,
                                           dest_offset, num_elements,
                                           py_list_func, histogram_bins, mgr,
                                           lifetime_guard)) {
                    ++discovered; ++n_for_param;
                    tracked_names.push_back("optimizer.exp_avg::" + pinfo.name);
                    mgr.register_frozen_param("optimizer.exp_avg::" + pinfo.name);
                }
            }
            if (has_opt_state && !exp_avg_sq_flat.is_none()) {
                if (submit_partition_slice(exp_avg_sq_flat, pinfo.param,
                                           "optimizer.exp_avg_sq::" + pinfo.name,
                                           "zero3.optimizer.exp_avg_sq",
                                           TensorCategory::OptimizerExpAvgSq,
                                           dest_offset, num_elements,
                                           py_list_func, histogram_bins, mgr,
                                           lifetime_guard)) {
                    ++discovered; ++n_for_param;
                    tracked_names.push_back("optimizer.exp_avg_sq::" + pinfo.name);
                    mgr.register_frozen_param("optimizer.exp_avg_sq::" + pinfo.name);
                }
            }

            if (n_for_param == 0) ++skipped_tensors;
        }
    }

    // Log the tensor names tracked on this rank so operators can verify
    // that different ranks are tracking different parameter sets.
    // if (logger) {
    //     std::string msg = "[checkers][zero3][rank_tensors] rank="
    //         + std::to_string(global_rank)
    //         + " count=" + std::to_string(tracked_names.size());
    //     for (const auto& n : tracked_names)
    //         msg += "\n  " + n;
    //     logger->log_message(msg);
    // }
    if (logger) {
        std::string msg =
            "[checkers][zero3][tracked_tensors] rank="
            + std::to_string(global_rank)
            + " discovered="
            + std::to_string(discovered)
            + " names="
            + std::to_string(tracked_names.size());

        for (const auto& n : tracked_names)
            msg += "\n  " + n;

        logger->log_message(msg);
    }

    return discovered;
}




int extract_group_rank(py::object param, int fallback_rank) {
    if (!py::hasattr(param, "ds_process_group")) {
        return fallback_rank;
    }

    try {
        py::object dist = py::module_::import("torch.distributed");
        if (py::hasattr(dist, "is_initialized") && !dist.attr("is_initialized")().cast<bool>()) {
            return fallback_rank;
        }

        py::object group = param.attr("ds_process_group");
        if (group.is_none()) {
            return fallback_rank;
        }

        return dist.attr("get_rank")(py::arg("group") = group).cast<int>();
    } catch (...) {
        return fallback_rank;
    }
}

bool try_zero_partitioned_tensor(py::object param,
                                 py::handle handle,
                                 int fallback_rank,
                                 py::object py_list_func,
                                 TensorExtraction& extraction) {
    if (!py::hasattr(handle, "data_ptr") || !py::hasattr(handle, "element_size")) {
        return false;
    }

    void* ptr = reinterpret_cast<void*>(handle.attr("data_ptr")().cast<size_t>());
    if (ptr == nullptr) {
        return false;
    }

    const size_t element_size = handle.attr("element_size")().cast<size_t>();
    if (element_size == 0) {
        return false;
    }

    size_t logical_numel = 0;
    if (py::hasattr(param, "ds_numel")) {
        logical_numel = param.attr("ds_numel").cast<size_t>();
    } else if (py::hasattr(handle, "numel")) {
        logical_numel = handle.attr("numel")().cast<size_t>();
    }

    size_t partition_numel = 0;
    if (py::hasattr(param, "partition_numel")) {
        partition_numel = param.attr("partition_numel")().cast<size_t>();
    } else if (py::hasattr(handle, "ds_numel")) {
        partition_numel = handle.attr("ds_numel").cast<size_t>();
    }

    if (logical_numel == 0 || partition_numel == 0) {
        return false;
    }

    const int shard_rank = extract_group_rank(param, fallback_rank);
    const size_t shard_offset = shard_rank < 0 ? 0 : static_cast<size_t>(shard_rank) * partition_numel;
    if (shard_offset >= logical_numel) {
        return false;
    }

    const size_t resident_numel = std::min(partition_numel, logical_numel - shard_offset);
    if (resident_numel == 0) {
        return false;
    }

    TensorMetadata meta;
    if (py::hasattr(param, "ds_shape")) {
        py::list logical_shape_list = py_list_func(param.attr("ds_shape"));
        for (auto item : logical_shape_list) {
            meta.logical_shape.push_back(item.cast<size_t>());
        }
    }

    std::string dtype_str = "unknown";
    if (py::hasattr(handle, "dtype")) {
        dtype_str = py::str(handle.attr("dtype")).cast<std::string>();
    }
    meta.data_type = parse_dtype(dtype_str);
    meta.element_size = element_size;
    meta.shape = {resident_numel};
    meta.logical_shape = meta.logical_shape.empty() ? meta.shape : meta.logical_shape;
    meta.strides = {1};
    meta.num_elements = resident_numel;
    meta.byte_size = resident_numel * element_size;
    meta.logical_num_elements = logical_numel;
    meta.logical_byte_size = logical_numel * element_size;

    extraction.ptr = ptr;
    extraction.byte_size = meta.byte_size;
    extraction.meta = std::move(meta);
    extraction.source = "parameter.ds_tensor.partition";
    return true;
}

bool try_tensor_like(py::handle handle,
                     const std::string& source,
                     py::object py_list_func,
                     TensorExtraction& extraction) {
    if (!py::hasattr(handle, "data_ptr")) {
        return false;
    }

    void* ptr = reinterpret_cast<void*>(handle.attr("data_ptr")().cast<size_t>());

    size_t byte_size = 0;
    if (py::hasattr(handle, "numel") && py::hasattr(handle, "element_size")) {
        byte_size = handle.attr("numel")().cast<size_t>() * handle.attr("element_size")().cast<size_t>();
    }
    if (byte_size == 0 && py::hasattr(handle, "untyped_storage")) {
        auto storage = handle.attr("untyped_storage")();
        if (py::hasattr(storage, "nbytes")) {
            byte_size = storage.attr("nbytes")().cast<size_t>();
        }
        if (ptr == nullptr && py::hasattr(storage, "data_ptr")) {
            ptr = reinterpret_cast<void*>(storage.attr("data_ptr")().cast<size_t>());
        }
    }
    if (byte_size == 0 && py::hasattr(handle, "_typed_storage")) {
        auto storage = handle.attr("_typed_storage")();
        if (py::hasattr(storage, "nbytes")) {
            byte_size = storage.attr("nbytes")().cast<size_t>();
        }
        if (ptr == nullptr && py::hasattr(storage, "data_ptr")) {
            ptr = reinterpret_cast<void*>(storage.attr("data_ptr")().cast<size_t>());
        }
    }

    if (ptr == nullptr || byte_size == 0) {
        return false;
    }

    TensorMetadata meta;
    fill_shape_and_stride(handle, py_list_func, meta);

    std::string dtype_str = "unknown";
    if (py::hasattr(handle, "dtype")) {
        dtype_str = py::str(handle.attr("dtype")).cast<std::string>();
    }
    meta.data_type = parse_dtype(dtype_str);
    if (py::hasattr(handle, "element_size")) {
        meta.element_size = handle.attr("element_size")().cast<size_t>();
    } else {
        meta.element_size = dtype_element_size(meta.data_type);
    }
    meta.logical_shape = meta.shape;
    meta.num_elements = meta.element_size == 0 ? 0 : (byte_size / meta.element_size);
    meta.byte_size = byte_size;
    meta.logical_num_elements = meta.num_elements;
    meta.logical_byte_size = meta.byte_size;

    extraction.ptr = ptr;
    extraction.byte_size = byte_size;
    extraction.meta = std::move(meta);
    extraction.source = source;
    return true;
}

bool extract_tensor(py::object param,
                    int fallback_rank,
                    py::object py_list_func,
                    TensorExtraction& extraction) {
    if (py::hasattr(param, "ds_tensor") && !param.attr("ds_tensor").is_none()) {
        if (try_zero_partitioned_tensor(param, param.attr("ds_tensor"), fallback_rank, py_list_func, extraction)) {
            return true;
        }
        if (try_tensor_like(param.attr("ds_tensor"), "parameter.ds_tensor", py_list_func, extraction)) {
            return true;
        }
    }

    if (try_tensor_like(param, "parameter", py_list_func, extraction)) {
        return true;
    }

    if (py::hasattr(param, "data") && try_tensor_like(param.attr("data"), "parameter.data", py_list_func, extraction)) {
        return true;
    }

    if (py::hasattr(param, "_hp_mapping") && !param.attr("_hp_mapping").is_none()) {
        auto hp_map = param.attr("_hp_mapping");
        if (py::hasattr(hp_map, "hp_tensor") && !hp_map.attr("hp_tensor").is_none()) {
            if (try_tensor_like(hp_map.attr("hp_tensor"), "parameter._hp_mapping.hp_tensor", py_list_func, extraction)) {
                return true;
            }
        }
    }

    return false;
}

} // namespace

static DataType parse_dtype(const std::string& s) {
    // NOTE: check "bfloat16" before "float16" -- the latter is a substring of
    // the former, so the old order silently tagged every bf16 tensor as fp16.
    if (s.find("bfloat16") != std::string::npos) return DataType::BFloat16;
    if (s.find("float32")  != std::string::npos) return DataType::Float32;
    if (s.find("float16")  != std::string::npos) return DataType::Float16;
    if (s.find("int8")     != std::string::npos) return DataType::Int8;
    return DataType::Unknown;
}

// ------------------------------------------------------------------ //
// Public API implementation
// ------------------------------------------------------------------ //
namespace {

// Parse the Python target selector (None | "all" | name | [names] | int mask)
// into a CompressTargets bitmask.
uint32_t targets_from_object(const py::object& sel) {
    using T = CompressTargets;
    auto name_bit = [](const std::string& s) -> uint32_t {
        if (s == "all")            return static_cast<uint32_t>(T::All);
        if (s == "model_state")    return static_cast<uint32_t>(T::ModelState);
        if (s == "master_weights") return static_cast<uint32_t>(T::MasterWeights);
        if (s == "exp_avg")        return static_cast<uint32_t>(T::ExpAvg);
        if (s == "exp_avg_sq")     return static_cast<uint32_t>(T::ExpAvgSq);
        throw std::runtime_error("unknown compression target: '" + s + "'");
    };

    if (sel.is_none()) return static_cast<uint32_t>(T::All);
    if (py::isinstance<py::str>(sel)) return name_bit(sel.cast<std::string>());
    if (py::isinstance<py::int_>(sel)) {
        const uint32_t v = sel.cast<uint32_t>();
        return v ? v : static_cast<uint32_t>(T::All);
    }
    if (py::isinstance<py::list>(sel) || py::isinstance<py::tuple>(sel) || py::isinstance<py::set>(sel)) {
        uint32_t mask = 0;
        for (auto item : sel) mask |= name_bit(py::str(item).cast<std::string>());
        return mask ? mask : static_cast<uint32_t>(T::All);
    }
    throw std::runtime_error("targets must be None, a str, an int mask, or a list/tuple/set of str");
}

// Discover this rank's shard and register every tensor with the MemoryManager.
// Returns the number of skipped tensors.
size_t run_discovery(py::object                         model_engine,
                     py::object                         optimizer,
                     MemoryManager&                     mgr,
                     const std::shared_ptr<RankLogger>& logger,
                     int                                global_rank) {
    py::object py_list_func = py::module_::import("builtins").attr("list");
    py::list   named_params = py_list_func(model_engine.attr("module").attr("named_parameters")());
    py::object resolved_optimizer = resolve_optimizer(model_engine, optimizer);

    size_t skipped_tensors = 0;
    std::vector<py::object>& lifetime_guard = mgr.lifetime_guard();

    if (has_zero3_flat_groups(resolved_optimizer)) {
        discover_zero3_flat_states(resolved_optimizer, named_params, global_rank,
                                   py_list_func, /*histogram_bins=*/256, mgr, logger,
                                   lifetime_guard, skipped_tensors);
    } else {
        for (auto item : named_params) {
            auto tuple_item = item.cast<py::tuple>();
            std::string name = tuple_item[0].cast<std::string>();
            py::object  param = tuple_item[1];

            const auto t0 = std::chrono::steady_clock::now();
            TensorExtraction extraction;
            if (!extract_tensor(param, global_rank, py_list_func, extraction)) {
                ++skipped_tensors;
                if (logger) logger->log_tensor_skipped(name, "no local device storage on this rank");
                continue;
            }
            lifetime_guard.push_back(param);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (submit_extraction(mgr, name, TensorCategory::ModelState, extraction, 256, ms)) {
                mgr.register_param(name, param, true);
            }
        }
    }
    return skipped_tensors;
}

} // namespace

uint32_t parse_targets(const py::object& selector) {
    return targets_from_object(selector);
}

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
                        int32_t    tt_eq_mode) {
    try {
        const auto total_start = std::chrono::steady_clock::now();
        if (group_size == 0)      throw std::runtime_error("group_size must be > 0");
        if (compute_streams == 0) throw std::runtime_error("compute_streams must be > 0");

        const int global_rank = extract_rank(model_engine, "global_rank", "RANK");
        const int local_rank  = extract_rank(model_engine, "local_rank",  "LOCAL_RANK");
        auto logger = RankLogger::create(global_rank, local_rank);

        auto& mgr = MemoryManager::instance();
        mgr.reset();
        mgr.set_logger(logger);

        logger->log_message("[checkers] initialize_context begin");

        CompressionConfig cfg;
        cfg.stage_targets     = targets_from_object(stage_targets);
        cfg.staging_budget_mb = staging_budget_mb;
        cfg.group_size        = group_size;
        cfg.zstd_level        = zstd_level;
        cfg.compute_streams   = compute_streams;
        cfg.measure_snr       = measure_snr;
        cfg.per_group         = per_group;
        cfg.exp_avg_sq_aggressive = exp_avg_sq_aggressive;
        if (tt_ndim)              cfg.tt_ndim = tt_ndim;
        if (tt_max_rank)          cfg.tt_max_rank = tt_max_rank;
        if (tt_energy_keep > 0.0) cfg.tt_energy_keep = tt_energy_keep;
        cfg.tt_eq_mode = tt_eq_mode;

        logger->log_message(
            "[checkers] config stage_targets=" + std::to_string(cfg.stage_targets) +
            " group_size="        + std::to_string(cfg.group_size) +
            " zstd_level="        + std::to_string(cfg.zstd_level) +
            " compute_streams="   + std::to_string(cfg.compute_streams) +
            " staging_budget_mb=" + std::to_string(cfg.staging_budget_mb) +
            " per_group="         + std::to_string(cfg.per_group) +
            " exp_avg_sq_aggressive=" + std::to_string(cfg.exp_avg_sq_aggressive) +
            " tt_ndim=" + std::to_string(cfg.tt_ndim) +
            " tt_max_rank=" + std::to_string(cfg.tt_max_rank) +
            " tt_energy_keep=" + std::to_string(cfg.tt_energy_keep) +
            " tt_eq_mode=" + std::to_string(cfg.tt_eq_mode) +
            " measure_snr="       + std::to_string(cfg.measure_snr));

        const auto disc_start = std::chrono::steady_clock::now();
        const size_t skipped = run_discovery(model_engine, optimizer, mgr, logger, global_rank);
        mgr.set_skipped_count(skipped);
        const double disc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - disc_start).count();

        const auto alloc_start = std::chrono::steady_clock::now();
        auto& comp = CheckpointCompressor::instance();
        comp.configure(cfg, logger, global_rank);
        comp.allocate(mgr);
        const double alloc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - alloc_start).count();

        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start).count();
        logger->log_message("[TIMING] discovery_ms="        + std::to_string(disc_ms));
        logger->log_message("[TIMING] staging_alloc_ms="    + std::to_string(alloc_ms));
        logger->log_message("[TIMING] initialize_context_total_ms=" + std::to_string(total_ms));
        logger->log_summary(mgr.get_category_stats(), mgr.skipped_count(), 0, 0);

    } catch (const std::exception& e) {
        throw std::runtime_error("[checkers] Context initialization failed: " + std::string(e.what()));
    }
}

// ------------------------------------------------------------------ //
py::dict compress_and_save(const std::string& output_dir,
                           py::object         error_bound,
                           py::object         targets,
                           bool               blocking,
                           bool               allow_lossless) {
    try {
        auto& mgr = MemoryManager::instance();
        auto logger = mgr.get_logger();
        if (!logger) {
            throw std::runtime_error("Context not initialized: call initialize_context first.");
        }

        const uint32_t mask = targets_from_object(targets);

        // error_bound: a float applies to every stream; a dict keys per stream
        // name ("model_state"/"master_weights"/"exp_avg"/"exp_avg_sq") with an
        // optional "default" for the rest.
        std::array<double, tracked_tensor_category_count> ebs{};
        ebs.fill(0.0);
        // given[c]: the caller named this stream's bound explicitly (dict key or
        // a single float).  allow_lossless only accepts an explicit 0.0, so a
        // stream missing from the dict is still an error, as before.
        std::array<bool, tracked_tensor_category_count> given{};
        given.fill(false);
        if (py::isinstance<py::dict>(error_bound)) {
            py::dict d = error_bound.cast<py::dict>();
            const double def = d.contains("default") ? d["default"].cast<double>() : -1.0;
            for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
                const char* nm = tensor_category_name(static_cast<TensorCategory>(c));
                if (d.contains(nm))      { ebs[c] = d[py::str(nm)].cast<double>(); given[c] = true; }
                else if (def > 0.0)      ebs[c] = def;
            }
        } else {
            const double v = error_bound.cast<double>();
            ebs.fill(v);
            given.fill(true);
        }
        for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c) {
            // allow_lossless: an explicit 0.0 selects the compressor's existing
            // no-bound path (raw + zstd, CodecId::RawZstd) for that stream.
            if (allow_lossless && given[c] && ebs[c] == 0.0) continue;
            if ((mask & (1u << c)) && !(ebs[c] > 0.0))
                throw std::runtime_error(std::string("compress_and_save: no error_bound for targeted "
                    "stream '") + tensor_category_name(static_cast<TensorCategory>(c)) +
                    "' (pass a float, or a dict with that key or \"default\")");
        }

        {
            std::string dbg = "[checkers] compress_and_save begin output=" + output_dir +
                              " targets=" + std::to_string(mask) +
                              " blocking=" + std::to_string(blocking) + " error_bounds=";
            for (int c = 0; c < static_cast<int>(tracked_tensor_category_count); ++c)
                dbg += std::string(tensor_category_name(static_cast<TensorCategory>(c))) + ":" +
                       std::to_string(ebs[c]) + " ";
            logger->log_message(dbg);
        }

        CompressionMetrics m;
        {
            py::gil_scoped_release release;
            m = CheckpointCompressor::instance().run(mgr, output_dir, ebs, mask, blocking);
        }
        return metrics_to_pydict(m);

    } catch (const std::exception& e) {
        std::cerr << "[CHECKERS CRITICAL ERROR] compress_and_save: " << e.what() << std::endl;
        throw;
    }
}

// ------------------------------------------------------------------ //
py::dict checkpoint_join() {
    CompressionMetrics m;
    {
        py::gil_scoped_release release;
        m = CheckpointCompressor::instance().join();
    }
    return metrics_to_pydict(m);
}

// ------------------------------------------------------------------ //
py::dict restart_from_checkpoint(py::object         model_engine,
                                 const std::string& input_dir,
                                 py::object         optimizer,
                                 py::object         targets) {
    try {
        const int global_rank = extract_rank(model_engine, "global_rank", "RANK");
        const int local_rank  = extract_rank(model_engine, "local_rank",  "LOCAL_RANK");

        auto& mgr = MemoryManager::instance();
        std::shared_ptr<RankLogger> logger = mgr.get_logger();
        if (!logger) {
            logger = RankLogger::create(global_rank, local_rank);
            mgr.set_logger(logger);
        }

        // Ensure the registry holds live destination pointers even if this
        // process never called initialize_context.
        if (mgr.collect_tensors(static_cast<uint32_t>(CompressTargets::All)).empty()) {
            mgr.reset();
            mgr.set_logger(logger);
            const size_t skipped = run_discovery(model_engine, optimizer, mgr, logger, global_rank);
            mgr.set_skipped_count(skipped);
        }

        const uint32_t mask = targets_from_object(targets);
        logger->log_message("[checkers] restart_from_checkpoint input=" + input_dir +
                            " targets=" + std::to_string(mask));
        return restart_apply(mgr, input_dir, mask, logger, global_rank);

    } catch (const std::exception& e) {
        std::cerr << "[CHECKERS CRITICAL ERROR] restart_from_checkpoint: " << e.what() << std::endl;
        throw;
    }
}

// ------------------------------------------------------------------ //
void shutdown() {
    // Runs via Python atexit (GIL held, interpreter still alive).  Joining the
    // background finalizer and dropping held py::object tensor refs here avoids
    // doing either from a static destructor after Py_Finalize.
    try { CheckpointCompressor::instance().reset(); } catch (...) {}
    try { MemoryManager::instance().reset();        } catch (...) {}
}

} // namespace checkers
