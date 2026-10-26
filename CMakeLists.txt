cmake_minimum_required(VERSION 3.20)
project(amuquant VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler Optimization Options
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -pthread)
    # CPU SIMD Optimization
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        message(STATUS "Target architecture is ARM64 (NEON enabled)")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Header Include Directories
include_directories(include)

# AMU-QUANT Core Library
set(AMUQUANT_SOURCES
    src/telemetry.cpp
    src/turboquant.cpp
    src/turbovec_ffi.cpp
    src/layer_manager.cpp
    src/kv_surgeon.cpp
    src/controller.cpp
    src/amuquant_engine.cpp
    src/gguf_parser.cpp
)

add_library(amuquant_core STATIC ${AMUQUANT_SOURCES})
target_include_directories(amuquant_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# C-ABI Shared Library for FFI (turbovec_ffi)
add_library(turbovec_ffi SHARED
    src/turboquant.cpp
    src/turbovec_ffi.cpp
)
target_include_directories(turbovec_ffi PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Minimal API Usage Example
add_executable(simple_inference examples/simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE amuquant_core)

# Optional Interactive Terminal Chat (requires external/llama.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/include/llama.h")
    add_executable(amuquant_chat examples/amuquant_chat.cpp)
    target_include_directories(amuquant_chat PRIVATE
        include
        external/llama.cpp/include
        external/llama.cpp/ggml/include
    )
    target_link_directories(amuquant_chat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/build/bin
    )
    target_link_libraries(amuquant_chat PRIVATE
        amuquant_core
        llama
        ggml
        ggml-base
        ggml-cpu
    )
    set_target_properties(amuquant_chat PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/build/bin"
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/build/bin"
    )
    install(TARGETS amuquant_chat RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Unit Tests Suite
enable_testing()

add_executable(test_turboquant tests/test_turboquant.cpp)
target_link_libraries(test_turboquant PRIVATE amuquant_core)
add_test(NAME test_turboquant COMMAND test_turboquant)

add_executable(test_controller tests/test_controller.cpp)
target_link_libraries(test_controller PRIVATE amuquant_core)
add_test(NAME test_controller COMMAND test_controller)

add_executable(test_amuquant_pipeline tests/test_amuquant_pipeline.cpp)
target_link_libraries(test_amuquant_pipeline PRIVATE amuquant_core)
add_test(NAME test_amuquant_pipeline COMMAND test_amuquant_pipeline)

# Installation Rules for Package Distribution
include(GNUInstallDirs)
install(TARGETS amuquant_core turbovec_ffi simple_inference
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/amuquant DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
