cmake_minimum_required(VERSION 3.20)

find_package(Catch2 CONFIG REQUIRED)

# Unit tests
add_executable(cyxwiz-tests
    unit/test_tensor.cpp
    unit/test_engine.cpp
    unit/test_c_api.cpp
    unit/test_activations.cpp
    unit/test_loss.cpp
    unit/test_device.cpp
    unit/test_optimizer.cpp
    unit/test_linear_layer.cpp
    unit/test_linear_algebra.cpp
    unit/test_dense_layer.cpp
    unit/test_sequence_layers.cpp
    unit/test_data_loader.cpp
    unit/test_training_dataloader.cpp
    unit/test_cyxmodel_format.cpp
    unit/test_text_inference_input.cpp
    unit/test_language_model_training.cpp
    unit/test_sequence_batcher.cpp
    unit/test_pipeline_runtime_capabilities.cpp
    ../cyxwiz-engine/src/core/formats/cyxmodel_format.cpp
    ../cyxwiz-engine/src/core/language_model_training.cpp
    ../cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp
    ../cyxwiz-engine/src/inference/text_inference_input.cpp
)

target_link_libraries(cyxwiz-tests PRIVATE
    Catch2::Catch2WithMain
    cyxwiz-backend
)

add_executable(test_tensor_dot_benchmark
    benchmarks/test_tensor_dot_benchmark.cpp
)

target_link_libraries(test_tensor_dot_benchmark PRIVATE
    cyxwiz-backend
)

add_executable(test_tensor_fan_in_benchmark
    benchmarks/test_tensor_fan_in_benchmark.cpp
)

target_link_libraries(test_tensor_fan_in_benchmark PRIVATE
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
