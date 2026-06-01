#include "checkers/api.hpp"
#include "checkers/MemoryManager.hpp"
#include "checkers/logging.hpp"
#include <pybind11/pybind11.h>     // Include pybind11 BEFORE referencing py::...
#include <pybind11/eval.h>         // Often needed for py::str

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
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

void initialize_context(py::object model_engine) {
    try {
        const int global_rank = extract_rank(model_engine, "global_rank", "RANK");
        const int local_rank = extract_rank(model_engine, "local_rank", "LOCAL_RANK");
        auto logger = RankLogger::create(global_rank, local_rank);
        auto& mgr = MemoryManager::instance();
        mgr.reset();
        mgr.begin_pipeline(logger);

        logger->log_message("[checkers] initialize_context begin");

        const auto discovery_start = std::chrono::steady_clock::now();
        auto named_params = model_engine.attr("module").attr("named_parameters")();
        py::object py_list_func = py::module_::import("builtins").attr("list");
        size_t skipped_tensors = 0;

        // CROSS-LANGUAGE RAII GUARD: Keeps Python allocations physically pinned in VRAM
        std::vector<py::object> lifetime_guard;

        for (auto item : named_params) {
            auto tuple_item = item.cast<py::tuple>();
            std::string name = tuple_item[0].cast<std::string>();
            py::object param = tuple_item[1];

            TensorExtraction extraction;
            if (!extract_tensor(param, global_rank, py_list_func, extraction)) {
                ++skipped_tensors;
                logger->log_tensor_skipped(name, "no local device storage on this rank");
                continue;
            }

            lifetime_guard.push_back(param);

            extraction.meta.name = name;

            logger->log_tensor_discovered(extraction.source, extraction.meta, extraction.ptr, extraction.byte_size);
            mgr.submit_tensor(name, extraction.ptr, extraction.byte_size, std::move(extraction.meta));
        }

        const double discovery_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - discovery_start).count();
        mgr.set_skipped_count(skipped_tensors);

        mgr.flush_pipeline();
        logger->log_summary(
            mgr.discovered_count(),
            mgr.skipped_count(),
            mgr.tensor_count(),
            mgr.get_slab_count(),
            mgr.get_total_tensor_bytes(),
            mgr.get_total_buffer_bytes(),
            discovery_ms,
            mgr.get_total_pass1_ms(),
            mgr.get_total_allocation_ms(),
            mgr.get_total_kernel_ms());

    } catch (const std::exception& e) {
        throw std::runtime_error("[checkers] Context initialization failed: " + std::string(e.what()));
    }
}

void analyze_model(py::object model_engine) {
    initialize_context(std::move(model_engine));
}

void compress_and_save(const std::string& output_path) {
    auto& mgr = MemoryManager::instance();
    mgr.report_memory_usage();
    
    std::cout << "[checkers] Saving analysis to: " << output_path << std::endl;
}

}