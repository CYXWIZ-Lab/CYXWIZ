cmake_minimum_required(VERSION 3.20)

find_package(Catch2 CONFIG REQUIRED)

# Unit tests
add_executable(cyxwiz-tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_optimizer.cpp
    unit/test_sequence_layers.cpp
    unit/test_data_loader.cpp
)

target_link_libraries(cyxwiz-tests PRIVATE
    Catch2::Catch2WithMain
    cyxwiz-backend
)

# DuckDB compile definition for data loader tests
if(CYXWIZ_HAS_DUCKDB)
    target_compile_definitions(cyxwiz-tests PRIVATE CYXWIZ_HAS_DUCKDB)
endif()

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(cyxwiz-tests)

# ONNX Export Tests (conditional on ONNX export support)
if(CYXWIZ_HAS_ONNX_EXPORT)
    # vcpkg provides ONNX targets with ONNX:: namespace
    find_package(ONNX CONFIG QUIET)
    find_package(protobuf CONFIG QUIET)

    if(ONNX_FOUND)
        add_executable(test_onnx_export
            unit/test_onnx_export.cpp
        )

        # Use namespaced targets: ONNX::onnx, ONNX::onnx_proto
        # Note: ONNX::onnx already includes ONNX::onnx_proto as dependency
        # and sets ONNX_ML=1 compile definition
        target_link_libraries(test_onnx_export PRIVATE
            Catch2::Catch2WithMain
            cyxwiz-backend
            ONNX::onnx
        )

        target_compile_definitions(test_onnx_export PRIVATE
            CYXWIZ_HAS_ONNX_EXPORT
        )

        # Register ONNX export tests
        catch_discover_tests(test_onnx_export)

        message(STATUS "ONNX Export tests enabled")
    else()
        message(STATUS "ONNX Export tests disabled - ONNX package not found")
    endif()
endif()
