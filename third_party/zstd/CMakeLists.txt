# ------------------------------------------------------------------ #
#  Vendored zstd 1.5.6 -- built as a static library.
#
#  Only the core codec is compiled (common/ compress/ decompress/
#  dictBuilder/).  legacy/ and deprecated/ were removed from the
#  vendored tree; the x86-64 Huffman-decode assembly is disabled via
#  ZSTD_DISABLE_ASM so the build needs no ASM toolchain and stays
#  identical on every node.  See VENDORED.txt for provenance.
# ------------------------------------------------------------------ #
cmake_minimum_required(VERSION 3.18)

set(ZSTD_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB ZSTD_SOURCES CONFIGURE_DEPENDS
    ${ZSTD_LIB_DIR}/common/*.c
    ${ZSTD_LIB_DIR}/compress/*.c
    ${ZSTD_LIB_DIR}/decompress/*.c
    ${ZSTD_LIB_DIR}/dictBuilder/*.c
)

add_library(checkers_zstd STATIC ${ZSTD_SOURCES})
add_library(checkers::zstd ALIAS checkers_zstd)

target_include_directories(checkers_zstd
    PUBLIC  ${ZSTD_LIB_DIR}
    PRIVATE ${ZSTD_LIB_DIR}/common
)

# Parallelism is normally driven by our own worker pool (one zstd per tensor,
# see CheckpointCompressor::run).  ZSTD_MULTITHREAD additionally enables
# ZSTD_c_nbWorkers, which parallelises a SINGLE call -- needed when there is
# only one buffer to compress, as in codec_selftest and hence in the benchmark,
# where the serial entropy stage was 85% of compress time and made Checkers
# look ~7x slower than its GPU codec actually is.
# nbWorkers>0 still emits one standard zstd frame, so the on-disk format and
# the restart/decompress path are unchanged.
# XXH_NAMESPACE keeps the bundled xxHash from clashing with any other
# copy linked into the same process (Umpire / RAJA / torch).
target_compile_definitions(checkers_zstd PRIVATE
    XXH_NAMESPACE=CHECKERS_ZSTD_
    ZSTD_DISABLE_ASM=1
    ZSTD_LEGACY_SUPPORT=0
    ZSTD_MULTITHREAD=1
    DEBUGLEVEL=0
)

find_package(Threads REQUIRED)
target_link_libraries(checkers_zstd PUBLIC Threads::Threads)

set_target_properties(checkers_zstd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(checkers_zstd PRIVATE -O3 -fvisibility=hidden -Wno-unused-function)
endif()
