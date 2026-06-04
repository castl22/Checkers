#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 1. Tell this file that these functions exist somewhere else (api.cpp)
namespace checkers {
      void initialize_context(py::object model_engine,
                                          py::object optimizer,
                                          size_t histogram_bins,
                                          size_t background_threads);
      void analyze_model();
      void compress_and_save(const std::string& output_path);
}

// 2. This is the crucial gatekeeper that Python looks for!
PYBIND11_MODULE(checkers_py, m) {
    m.doc() = "Checkers LLM Memory Tracking Extension";

    m.def("initialize_context",
          &checkers::initialize_context,
          py::arg("model_engine"),
          py::arg("optimizer") = py::none(),
          py::arg("histogram_bins") = 256,
          py::arg("background_threads") = 1,
          "Initializes Umpire memory resources and cluster context");

   m.def("analyze_model",
      &checkers::analyze_model,
      "Analyzes tracked model parameters inside the MemoryManager");

    m.def("compress_and_save", &checkers::compress_and_save, 
          "Compresses and saves tracked data to disk");
}