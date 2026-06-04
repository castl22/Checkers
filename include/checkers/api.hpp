#ifndef CHECKERS_API_HPP
#define CHECKERS_API_HPP

// 1. ADD THESE TWO LINES AT THE TOP OF YOUR HEADER:
#include <pybind11/pybind11.h>
#include "TensorResource.hpp"
namespace py = pybind11;

namespace checkers {
    struct TensorExtraction {
        void* ptr = nullptr;
        size_t byte_size = 0;
        TensorMetadata meta;
        std::string source;
    };

    void initialize_context(py::object model_engine,
                            py::object optimizer = py::none(),
                            size_t histogram_bins = 256,
                            size_t background_threads = 1);

    void analyze_model();

    void compress_and_save(const std::string& output_path);
}

#endif // CHECKERS_API_HPP