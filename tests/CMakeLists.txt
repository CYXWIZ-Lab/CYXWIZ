cmake_minimum_required(VERSION 3.20)

find_package(Catch2 CONFIG REQUIRED)

# Unit tests
add_executable(cyxwiz-tests
    unit/test_main.cpp
    unit/test_tensor.cpp
    unit/test_tensor_ownership.cpp
    unit/test_tensor_shape_contract.cpp
    unit/test_tensor_permute_contract.cpp
    unit/test_tensor_indexing_contract.cpp
    unit/test_tensor_concat_contract.cpp
    unit/test_tensor_reduction_contract.cpp
    unit/test_tensor_elementwise_contract.cpp
    unit/test_tensor_mask_contract.cpp
    unit/test_tensor_broadcast_contract.cpp
    unit/test_tensor_factory_contract.cpp
    unit/test_engine.cpp
    unit/test_c_api.cpp
    unit/test_activations.cpp
    unit/test_arrayfire_fallback_source_scan.cpp
    unit/test_arrayfire_backend_utils.cpp
    unit/test_arrayfire_backend_smoke.cpp
    unit/test_loss.cpp
    unit/test_device.cpp
    unit/test_optimizer.cpp
    unit/test_scheduler.cpp
    unit/test_flatten_layer.cpp
    unit/test_dropout_layer.cpp
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
    ../cyxwiz-engine/src/core/compute_runtime_config.cpp
    ../cyxwiz-engine/src/core/route_qualification_snapshot.cpp
    ../cyxwiz-engine/src/core/route_qualification_service.cpp
    ../cyxwiz-engine/src/core/route_recommendation.cpp
    ../cyxwiz-engine/src/inference/text_inference_input.cpp
)

target_link_libraries(cyxwiz-tests PRIVATE
    Catch2::Catch2
    cyxwiz-backend
    nlohmann_json::nlohmann_json
)

set(CYXWIZ_TRAINING_CORE_FIXTURE_PATH
    "${CMAKE_SOURCE_DIR}/cyxwiz-engine/tests/computation_truth/fixtures/training_core_pytorch.json"
)
set_source_files_properties(unit/test_scheduler.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_flatten_layer.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_shape_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_permute_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_indexing_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_concat_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_reduction_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_elementwise_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_mask_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_broadcast_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_tensor_factory_contract.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)
set_source_files_properties(unit/test_dropout_layer.cpp PROPERTIES
    COMPILE_DEFINITIONS
        "CYXWIZ_TRAINING_CORE_FIXTURE_PATH=\"${CYXWIZ_TRAINING_CORE_FIXTURE_PATH}\""
)

if(CYXWIZ_BUILD_ENGINE AND WIN32)
    target_sources(cyxwiz-tests PRIVATE
        ../cyxwiz-engine/src/core/backend_pack_qualification_adapter.cpp
    )
    target_link_libraries(cyxwiz-tests PRIVATE
        cyxwiz-backend-pack-service
    )
    target_compile_definitions(cyxwiz-tests PRIVATE
        CYXWIZ_HAS_BACKEND_PACK_QUALIFICATION_ADAPTER
    )
endif()

add_executable(cyxwiz-device-inventory
    smoke/device_inventory_report.cpp
)

target_link_libraries(cyxwiz-device-inventory PRIVATE
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

if(CYXWIZ_BUILD_ENGINE)
    add_custom_target(test_oneapi_operation_matrix
        COMMAND ${CMAKE_COMMAND}
            "-DPROBE_EXE=$<TARGET_FILE:cyxwiz-route-probe>"
            "-DREPORT_PATH=$<TARGET_FILE_DIR:cyxwiz-route-probe>/oneapi-operation-matrix.txt"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_oneapi_operation_matrix.cmake"
        DEPENDS cyxwiz-route-probe
        USES_TERMINAL
        VERBATIM
    )

    add_custom_target(test_arrayfire_route_matrix
        COMMAND ${CMAKE_COMMAND}
            "-DPROBE_EXE=$<TARGET_FILE:cyxwiz-route-probe>"
            "-DREPORT_PATH=$<TARGET_FILE_DIR:cyxwiz-route-probe>/arrayfire-route-matrix.txt"
            "-DSUMMARY_PATH=$<TARGET_FILE_DIR:cyxwiz-route-probe>/arrayfire-route-qualification.json"
            "-DROUTES=cpu:0|cuda:0|opencl:0|opencl:1|opencl:2|oneapi:0|oneapi:1"
            "-DMATRIX_ID=cyxwiz-route-qualification-v1"
            "-DPACK_ID=local-arrayfire-installation"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/smoke/run_arrayfire_route_matrix.cmake"
        DEPENDS cyxwiz-route-probe
        USES_TERMINAL
        VERBATIM
    )

    add_executable(test_route_qualification_process
        ../cyxwiz-engine/tests/test_route_qualification_process.cpp
        ../cyxwiz-engine/src/core/route_qualification_service.cpp
        ../cyxwiz-engine/src/core/route_qualification_snapshot.cpp
    )
    add_executable(test_route_probe_environment_child
        ../cyxwiz-engine/tests/test_route_probe_environment_child.cpp
    )
    target_include_directories(test_route_qualification_process PRIVATE
        ${CMAKE_SOURCE_DIR}/cyxwiz-engine/src
        ${CMAKE_SOURCE_DIR}/cyxwiz-backend/include
    )
    target_link_libraries(test_route_qualification_process PRIVATE
        cyxwiz-backend
        nlohmann_json::nlohmann_json
    )
    set_target_properties(test_route_qualification_process PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    set_target_properties(test_route_probe_environment_child PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(test_route_qualification_process
        cyxwiz-route-probe
        test_route_probe_environment_child
    )
    add_test(
        NAME RouteQualificationProcessContracts
        COMMAND test_route_qualification_process
                $<TARGET_FILE:cyxwiz-route-probe>
                $<TARGET_FILE:test_route_probe_environment_child>
    )
endif()

# DuckDB compile definition for data loader tests
if(CYXWIZ_HAS_DUCKDB)
    target_compile_definitions(cyxwiz-tests PRIVATE CYXWIZ_HAS_DUCKDB)
endif()

# Register tests with CTest
include(CTest)
# Register the core suite as one process. Per-case Catch2 discovery made CTest
# restart ArrayFire and hardware detection hundreds of times, turning an
# 18-second suite into an impractical validation command.
add_test(NAME CyxWizCoreTests COMMAND cyxwiz-tests "~[oneapi]")

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
