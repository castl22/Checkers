#ifndef CHECKERS_API_HPP
#define CHECKERS_API_HPP

// 1. ADD THESE TWO LINES AT THE TOP OF YOUR HEADER:
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace checkers {

    // Now 'py::object' is fully recognized here
    void initialize_context(py::object model_engine,
                            py::object optimizer = py::none(),
                            size_t histogram_bins = 256,
                            size_t background_threads = 1);

    void analyze_model(py::object model_engine,
                       py::object optimizer = py::none(),
                       size_t histogram_bins = 256,
                       size_t background_threads = 1);
    void compress_and_save(const std::string& output_path);
}

#endif // CHECKERS_API_HPP