#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 1. Tell this file that these functions exist somewhere else (api.cpp)
namespace checkers {
    void initialize_context(py::object model_engine);
    void analyze_model(py::object model_engine);
    void compress_and_save(const std::string& output_path);
}

// 2. This is the crucial gatekeeper that Python looks for!
PYBIND11_MODULE(checkers_py, m) {
    m.doc() = "Checkers LLM Memory Tracking Extension";

    m.def("initialize_context", &checkers::initialize_context, 
          "Initializes Umpire memory resources and cluster context");

    m.def("analyze_model", &checkers::analyze_model, 
          "Analyzes model engine parameters");

    m.def("compress_and_save", &checkers::compress_and_save, 
          "Compresses and saves tracked data to disk");
}