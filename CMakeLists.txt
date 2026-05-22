cmake_minimum_required(VERSION 3.18)
project(TensorAnalyzer LANGUAGES CXX CUDA)

# Force C++20 for both Host and Device compilation
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 1. Locate RAJA installation
set(RAJA_DIR "$ENV{HOME}/raja_install/lib/cmake/raja")
find_package(RAJA REQUIRED PATHS ${RAJA_DIR} NO_DEFAULT_PATH)

# 2. Locate Umpire installation
set(UMPIRE_DIR "$ENV{HOME}/raja_install/lib/cmake/umpire")
find_package(umpire REQUIRED PATHS ${UMPIRE_DIR} NO_DEFAULT_PATH)
find_library(ZSTD_LIBRARY zstd REQUIRED)

set(TENSOR_ANALYZER_RAJA_SOURCE_DIR "$ENV{HOME}/Documents/RAJA")
set(TENSOR_ANALYZER_CAMP_INCLUDE_DIR
    "${TENSOR_ANALYZER_RAJA_SOURCE_DIR}/tpl/camp/include")

get_target_property(RAJA_INTERFACE_LINK_LIBRARIES RAJA INTERFACE_LINK_LIBRARIES)
set(TENSOR_ANALYZER_USE_CUDA OFF)
if(RAJA_INTERFACE_LINK_LIBRARIES)
    foreach(raja_link_lib IN LISTS RAJA_INTERFACE_LINK_LIBRARIES)
        if(raja_link_lib STREQUAL "cuda")
            set(TENSOR_ANALYZER_USE_CUDA ON)
            break()
        endif()
    endforeach()
endif()

# Cleanly disable broken user-defined literal templates in older/system fmt variants
add_compile_definitions(FMT_USE_USER_DEFINED_LITERALS=0)
add_compile_definitions(FMT_USE_NONTYPE_TEMPLATE_ARGS=0)

# Build target
add_executable(tensor_analyzer src/main.cpp)
add_executable(tensor_group_repair src/tensor_group_repair.cpp)
add_executable(tensor_group_codebook src/tensor_group_codebook.cpp)
add_executable(tensor_group_bloom_delta src/tensor_group_bloom_delta.cpp)
add_executable(tensor_group_bitplane_repair src/tensor_group_bitplane_repair.cpp)
add_executable(tensor_group_prefix_mask_repair src/tensor_group_prefix_mask_repair.cpp)

if(EXISTS "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}/camp/concepts.hpp")
    target_include_directories(tensor_analyzer BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
    target_include_directories(tensor_group_repair BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
    target_include_directories(tensor_group_codebook BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
    target_include_directories(tensor_group_bloom_delta BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
    target_include_directories(tensor_group_bitplane_repair BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
    target_include_directories(tensor_group_prefix_mask_repair BEFORE PRIVATE
        "${TENSOR_ANALYZER_CAMP_INCLUDE_DIR}")
endif()

# 3. FIX: Enforce clean CUDA translation unit processing
if(TENSOR_ANALYZER_USE_CUDA)
    # Explicitly tell CMake this target must be compiled using NVCC
    set_target_properties(tensor_analyzer PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(tensor_group_repair PROPERTIES LINKER_LANGUAGE CUDA)
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(src/tensor_group_repair.cpp PROPERTIES LANGUAGE CUDA)
    
    # Pass necessary standard compliance and lambda extensions to NVCC
    target_compile_options(tensor_analyzer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )
    target_compile_options(tensor_group_repair PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )
    
    # Pass host-compiler specific flags through NVCC safely if required
    target_compile_options(tensor_analyzer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fconcepts>
    )
    target_compile_options(tensor_group_repair PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fconcepts>
    )

    # This install gates cuda.hpp contents on RAJA_CUDA_ACTIVE for CUDA TUs.
    target_compile_definitions(tensor_analyzer PRIVATE RAJA_CUDA_ACTIVE)
    target_compile_definitions(tensor_group_repair PRIVATE RAJA_CUDA_ACTIVE)
endif()

# 4. Link dependencies
target_link_libraries(tensor_analyzer PRIVATE 
    RAJA 
    umpire
)

target_link_libraries(tensor_group_repair PRIVATE 
    RAJA 
    umpire
)

target_link_libraries(tensor_group_codebook PRIVATE
    ${ZSTD_LIBRARY}
)

target_link_libraries(tensor_group_bloom_delta PRIVATE
    ${ZSTD_LIBRARY}
)

target_link_libraries(tensor_group_bitplane_repair PRIVATE
)

target_link_libraries(tensor_group_prefix_mask_repair PRIVATE
)