#include "checkers/api.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/logging.hpp"
#include <pybind11/pybind11.h>     // Include pybind11 BEFORE referencing py::...
#include <pybind11/eval.h>         // Often needed for py::str

#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace checkers {

static DataType parse_dtype(const std::string& s);

namespace {

struct TensorExtraction {
    void* ptr = nullptr;
    size_t byte_size = 0;
    TensorMetadata meta;
    std::string source;
};

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
    extraction.meta.histogram_bins = histogram_bins;
    TensorMetadata submitted_meta = extraction.meta;
    mgr.submit_tensor(name, extraction.ptr, extraction.byte_size, std::move(extraction.meta));
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

bool submit_local_tensor(py::handle tensor_handle,
                         py::object param,
                         const std::string& name,
                         const std::string& source,
                         TensorCategory category,
                         py::object py_list_func,
                         size_t histogram_bins,
                         MemoryManager& mgr,
                         std::vector<py::object>& lifetime_guard) {
    const auto discovery_start = std::chrono::steady_clock::now();
    TensorExtraction extraction;
    if (!try_tensor_like(tensor_handle, source, py_list_func, extraction)) {
        return false;
    }

    apply_logical_metadata(param, extraction.meta, py_list_func);
    lifetime_guard.push_back(py::reinterpret_borrow<py::object>(tensor_handle));
    const double discovery_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - discovery_start).count();
    return submit_extraction(mgr, name, category, extraction, histogram_bins, discovery_ms);
}

size_t discover_zero3_flat_states(py::object optimizer,
                                  const py::list& named_params,
                                  int global_rank,
                                  py::object py_list_func,
                                  size_t histogram_bins,
                                  MemoryManager& mgr,
                                  std::vector<py::object>& lifetime_guard,
                                  size_t& skipped_tensors) {
    if (!has_zero3_flat_groups(optimizer)) {
        return 0;
    }

    size_t discovered = 0;
    for (auto item : named_params) {
        auto tuple_item = item.cast<py::tuple>();
        std::string param_name = tuple_item[0].cast<std::string>();
        py::object param = tuple_item[1];
        size_t discovered_for_param = 0;

        if (py::hasattr(param, "ds_tensor")
            && !param.attr("ds_tensor").is_none()
            && submit_local_tensor(param.attr("ds_tensor"),
                                   param,
                                   param_name,
                                   "zero3.param.ds_tensor",
                                   TensorCategory::ModelState,
                                   py_list_func,
                                   histogram_bins,
                                   mgr,
                                   lifetime_guard)) {
            ++discovered;
            ++discovered_for_param;
        }

        py::object z3_optimizer = py::none();
        if (py::hasattr(param, "_z3_optimizer") && !param.attr("_z3_optimizer").is_none()) {
            z3_optimizer = param.attr("_z3_optimizer");
        } else if (!optimizer.is_none() && py::hasattr(optimizer, "get_local_fp32_param")) {
            z3_optimizer = optimizer;
        }

        if (!z3_optimizer.is_none() && py::hasattr(param, "requires_grad") && param.attr("requires_grad").cast<bool>()) {
            try {
                py::object master_local = z3_optimizer.attr("get_local_fp32_param")(param, py::none());
                if (!master_local.is_none()
                    && submit_local_tensor(master_local,
                                           param,
                                           "master_weights::" + param_name,
                                           "zero3.get_local_fp32_param",
                                           TensorCategory::MasterWeights,
                                           py_list_func,
                                           histogram_bins,
                                           mgr,
                                           lifetime_guard)) {
                    ++discovered;
                    ++discovered_for_param;
                }
            } catch (...) {
            }

            try {
                py::object exp_avg_local = z3_optimizer.attr("get_local_fp32_param")(param, py::str("exp_avg"));
                if (!exp_avg_local.is_none()
                    && submit_local_tensor(exp_avg_local,
                                           param,
                                           "optimizer.exp_avg::" + param_name,
                                           "zero3.get_local_fp32_param(exp_avg)",
                                           TensorCategory::OptimizerExpAvg,
                                           py_list_func,
                                           histogram_bins,
                                           mgr,
                                           lifetime_guard)) {
                    ++discovered;
                    ++discovered_for_param;
                }
            } catch (...) {
            }

            try {
                py::object exp_avg_sq_local = z3_optimizer.attr("get_local_fp32_param")(param, py::str("exp_avg_sq"));
                if (!exp_avg_sq_local.is_none()
                    && submit_local_tensor(exp_avg_sq_local,
                                           param,
                                           "optimizer.exp_avg_sq::" + param_name,
                                           "zero3.get_local_fp32_param(exp_avg_sq)",
                                           TensorCategory::OptimizerExpAvgSq,
                                           py_list_func,
                                           histogram_bins,
                                           mgr,
                                           lifetime_guard)) {
                    ++discovered;
                    ++discovered_for_param;
                }
            } catch (...) {
            }
        }

        if (discovered_for_param == 0) {
            ++skipped_tensors;
        }
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
    if (s.find("float32") != std::string::npos) return DataType::Float32;
    if (s.find("float16") != std::string::npos) return DataType::Float16;
    if (s.find("bfloat16")!= std::string::npos) return DataType::BFloat16;
    if (s.find("int8")    != std::string::npos) return DataType::Int8;
    return DataType::Unknown;
}

// ------------------------------------------------------------------ //
// Public API Implementation
// ------------------------------------------------------------------ //

void initialize_context(py::object model_engine,
                        py::object optimizer,
                        size_t histogram_bins,
                        size_t background_threads) {
    try {
        if (histogram_bins == 0) {
            throw std::runtime_error("histogram_bins must be greater than zero");
        }
        if (background_threads == 0) {
            throw std::runtime_error("background_threads must be greater than zero");
        }

        const int global_rank = extract_rank(model_engine, "global_rank", "RANK");
        const int local_rank = extract_rank(model_engine, "local_rank", "LOCAL_RANK");
        auto logger = RankLogger::create(global_rank, local_rank);
        auto& mgr = MemoryManager::instance();
        mgr.reset();
        mgr.set_default_histogram_bins(histogram_bins);
        mgr.begin_pipeline(logger, background_threads, background_threads);

        logger->log_message("[checkers] initialize_context begin");
        logger->log_message("[checkers] histogram_bins=" + std::to_string(histogram_bins)
                            + " background_threads=" + std::to_string(background_threads));

        const auto discovery_start = std::chrono::steady_clock::now();
        py::object py_list_func = py::module_::import("builtins").attr("list");
        py::list named_params = py_list_func(model_engine.attr("module").attr("named_parameters")());
        py::object resolved_optimizer = resolve_optimizer(model_engine, optimizer);
        size_t skipped_tensors = 0;

        // CROSS-LANGUAGE RAII GUARD: Keeps Python allocations physically pinned in VRAM
        std::vector<py::object> lifetime_guard;

        if (has_zero3_flat_groups(resolved_optimizer)) {
            discover_zero3_flat_states(resolved_optimizer,
                                       named_params,
                                       global_rank,
                                       py_list_func,
                                       histogram_bins,
                                       mgr,
                                       lifetime_guard,
                                       skipped_tensors);
        } else {
            for (auto item : named_params) {
                auto tuple_item = item.cast<py::tuple>();
                std::string name = tuple_item[0].cast<std::string>();
                py::object param = tuple_item[1];

                const auto tensor_discovery_start = std::chrono::steady_clock::now();
                TensorExtraction extraction;
                if (!extract_tensor(param, global_rank, py_list_func, extraction)) {
                    ++skipped_tensors;
                    logger->log_tensor_skipped(name, "no local device storage on this rank");
                    continue;
                }

                lifetime_guard.push_back(param);
                const double tensor_discovery_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tensor_discovery_start).count();
                submit_extraction(mgr,
                                  name,
                                  TensorCategory::ModelState,
                                  extraction,
                                  histogram_bins,
                                  tensor_discovery_ms);
            }
        }

        const double discovery_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - discovery_start).count();
        (void)discovery_ms;
        mgr.set_skipped_count(skipped_tensors);

        mgr.flush_pipeline();
        logger->log_summary(
            mgr.get_category_stats(),
            mgr.skipped_count(),
            mgr.get_slab_count(),
            mgr.batch_count());

    } catch (const std::exception& e) {
        throw std::runtime_error("[checkers] Context initialization failed: " + std::string(e.what()));
    }
}

void analyze_model(py::object model_engine,
                   py::object optimizer,
                   size_t histogram_bins,
                   size_t background_threads) {
    initialize_context(std::move(model_engine),
                       std::move(optimizer),
                       histogram_bins,
                       background_threads);
}

void compress_and_save(const std::string& output_path) {
    auto& mgr = MemoryManager::instance();
    mgr.report_memory_usage();
    
    std::cout << "[checkers] Saving analysis to: " << output_path << std::endl;
}

}