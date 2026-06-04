cmake_minimum_required(VERSION 3.18)
 
# Suppress pybind11's use of the removed FindPythonLibsNew module
# cmake_policy(SET CMP0144 NEW)
# cmake_policy(SET CMP0148 OLD)
 
project(Checkers LANGUAGES CXX)
 
# ------------------------------------------------------------------ #
#  C++ standard (host)
# ------------------------------------------------------------------ #
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
# ------------------------------------------------------------------ #
#  Backend selection
# ------------------------------------------------------------------ #
set(BACKEND "CUDA" CACHE STRING "Target backend: CUDA or HIP")
 
if(BACKEND STREQUAL "HIP")
    # hipcc drives both host and device compilation
    set(CMAKE_CXX_COMPILER hipcc)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
    # Do NOT add_compile_definitions(RAJA_HIP_ACTIVE) here --
    # RAJA's own config.hpp defines it and would cause a redefinition
    # warning (or error with -Werror).  RAJA's find_package sets it up.
    set(DEVICE_LANG HIP)
 
elseif(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Same reasoning: let RAJA's headers own RAJA_CUDA_ACTIVE.
    set(DEVICE_LANG CUDA)
 
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'. Choose CUDA or HIP.")
endif()
 
# ------------------------------------------------------------------ #
#  Dependencies
# ------------------------------------------------------------------ #
set(RAJA_DIR   "/usr/workspace/kogiou1/RAJA/raja_install/lib/cmake/raja")
set(UMPIRE_DIR "/usr/workspace/kogiou1/Umpire/umpire_install/lib64/cmake/umpire")
 
find_package(RAJA   REQUIRED PATHS "${RAJA_DIR}"   NO_DEFAULT_PATH)
find_package(umpire REQUIRED PATHS "${UMPIRE_DIR}" NO_DEFAULT_PATH)
find_package(pybind11 REQUIRED)
 
# ------------------------------------------------------------------ #
#  Sources
#  Files that contain __device__ / __global__ code must be labelled
#  with the correct language so CMake invokes the right compiler.
# ------------------------------------------------------------------ #
set(CHECKERS_SOURCES
    src/python_bindings.cpp
    src/api.cpp
    src/logging.cpp
    src/MemoryManager.cpp
    src/TensorResource.cpp
    src/TensorAnalyzer.cpp
)
 
# Tell CMake these files contain device code for the chosen backend.
# Without this, CMake compiles them as plain C++ and the device
# intrinsics are invisible to the host compiler.
set_source_files_properties(
    src/MemoryManager.cpp
    src/TensorResource.cpp
    PROPERTIES LANGUAGE ${DEVICE_LANG}
)
 
# ------------------------------------------------------------------ #
#  Python extension module
# ------------------------------------------------------------------ #
pybind11_add_module(checkers_py ${CHECKERS_SOURCES})
 
target_include_directories(checkers_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "/usr/workspace/kogiou1/RAJA/raja_install/include"
    "/usr/workspace/kogiou1/Umpire/umpire_install/include"
)

# 2. Ensure linking is using the full path if target-based linking fails
target_link_directories(checkers_py PRIVATE
    "/usr/workspace/kogiou1/RAJA/raja_install/lib"
    "/usr/workspace/kogiou1/Umpire/umpire_install/lib64"
)

target_link_libraries(checkers_py PRIVATE
    RAJA
    umpire
    pybind11::module
)
 
