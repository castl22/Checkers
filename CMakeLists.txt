cmake_minimum_required(VERSION 3.18)

# Suppress pybind11's use of the removed FindPythonLibsNew module
# cmake_policy(SET CMP0144 NEW)
# cmake_policy(SET CMP0148 OLD)

# C is needed for the vendored zstd static library (third_party/zstd).
project(Checkers LANGUAGES C CXX)

# ------------------------------------------------------------------ #
#  Default build type  -- DO NOT REMOVE
# ------------------------------------------------------------------ #
# Without this, an unset CMAKE_BUILD_TYPE leaves CMAKE_HIP_FLAGS with no -O
# flag at all, so clang compiles every DEVICE KERNEL at -O0.  That is not a
# 10-20% effect: measured 2026-09-16 on the llama-3-8b shard, turning on -O3
# alone took the codec from 47.6 ms to 1.4 ms per 0.5 GiB chunk on compress and
# 15.3 ms to 0.5 ms on decompress (~30x), because at -O0 per-thread arrays spill
# to scratch instead of registers.  Every Checkers performance number taken
# before that date was an -O0 number, and every comparison against cuSZp was
# unfair: cuSZp's own CMakeLists carries exactly this guard, so it was always
# built -O3.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE was unset; defaulting to Release "
                   "(an unoptimized build is ~30x slower on the GPU codec)")
endif()

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
set(RAJA_DIR   "/usr/workspace/kogiou1/software/RAJA/raja_install/lib/cmake/raja")
set(UMPIRE_DIR "/usr/workspace/kogiou1/software/Umpire/umpire_install/lib64/cmake/umpire")

find_package(RAJA   REQUIRED PATHS "${RAJA_DIR}"   NO_DEFAULT_PATH)
find_package(umpire REQUIRED PATHS "${UMPIRE_DIR}" NO_DEFAULT_PATH)
find_package(pybind11 REQUIRED)

# Vendored zstd (static, PIC).  Provides target checkers::zstd.
add_subdirectory(third_party/zstd)

# ------------------------------------------------------------------ #
#  Sources
#  Files that contain __device__ / __global__ code (or that call the
#  GPU runtime directly) must be labelled with the correct language so
#  CMake invokes the right compiler.
# ------------------------------------------------------------------ #
set(CHECKERS_SOURCES
    src/python_bindings.cpp
    src/api.cpp
    src/logging.cpp
    src/MemoryManager.cpp
    src/TensorResource.cpp
    src/StagingPool.cpp
    src/CodecKernels.cpp
    src/TtLinalg.cpp
    src/CheckpointCompressor.cpp
    src/Restart.cpp
    src/SelfTest.cpp
    runtime/Utils.cpp
)

set_source_files_properties(
    src/MemoryManager.cpp
    src/TensorResource.cpp
    src/StagingPool.cpp
    src/CodecKernels.cpp
    src/TtLinalg.cpp
    src/CheckpointCompressor.cpp
    src/Restart.cpp
    src/SelfTest.cpp
    runtime/Utils.cpp
    PROPERTIES LANGUAGE ${DEVICE_LANG}
)

# rocBLAS (GEMM for the TT-SVD exp_avg codec).  Header-optional: TtLinalg.cpp
# compiles to a no-op stub if this is not found.
if(BACKEND STREQUAL "HIP")
    find_library(ROCBLAS_LIB rocblas
        HINTS $ENV{ROCM_PATH}/lib /opt/rocm/lib /opt/rocm-6.4.0/lib)
    if(ROCBLAS_LIB)
        message(STATUS "Checkers: rocBLAS = ${ROCBLAS_LIB}")
    else()
        message(WARNING "Checkers: rocBLAS not found; exp_avg TT codec disabled")
    endif()
endif()

# ------------------------------------------------------------------ #
#  Python extension module
# ------------------------------------------------------------------ #
pybind11_add_module(checkers_py ${CHECKERS_SOURCES})

target_include_directories(checkers_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    "/usr/workspace/kogiou1/software/RAJA/raja_install/include"
    "/usr/workspace/kogiou1/software/Umpire/umpire_install/include"
)

# 2. Ensure linking is using the full path if target-based linking fails
target_link_directories(checkers_py PRIVATE
    "/usr/workspace/kogiou1/software/RAJA/raja_install/lib"
    "/usr/workspace/kogiou1/software/Umpire/umpire_install/lib64"
)

target_link_libraries(checkers_py PRIVATE
    RAJA
    umpire
    pybind11::module
    checkers::zstd
)

if(BACKEND STREQUAL "HIP" AND ROCBLAS_LIB)
    target_link_libraries(checkers_py PRIVATE ${ROCBLAS_LIB})
endif()
