# The GUI-free compile-and-train path shared by every host (TOFIX118 P2).
#
# cyxwiz-training-hooks: the plugin training-hook manager and plugin node
# registry, shared by the plugin SDK and the core (linked once per exe).
# cyxwiz-training-core: graph loader and compiler, launch preparation, model
# builder, training/test executors, Arrow/Parquet/sequence/sparse batchers,
# checkpoints, trace, route qualification and the headless graph training job.
#
# Included by the Engine and the Server Node (whichever is configured first
# defines the targets). Also sets CYXWIZ_TRAINING_EXECUTOR_HARNESS_SOURCES
# (absolute paths) for the Engine's harness tests.

set(_cyxwiz_engine_src "${CMAKE_SOURCE_DIR}/cyxwiz-engine/src")

set(CYXWIZ_TRAINING_EXECUTOR_HARNESS_SOURCES
    ${_cyxwiz_engine_src}/core/training_executor.cpp
    ${_cyxwiz_engine_src}/core/training_generation_preview.cpp
    ${_cyxwiz_engine_src}/core/language_model_generation.cpp
    ${_cyxwiz_engine_src}/core/training_scheduler_controller.cpp
    ${_cyxwiz_engine_src}/core/regression_target_transform.cpp
    ${_cyxwiz_engine_src}/core/preprocessing_state.cpp
    ${_cyxwiz_engine_src}/core/training_batcher_setup.cpp
    ${_cyxwiz_engine_src}/core/sparse_training_batcher_setup.cpp
    ${_cyxwiz_engine_src}/core/sparse_feature_dataset.cpp
    ${_cyxwiz_engine_src}/core/sparse_feature_dataset_batcher.cpp
    ${_cyxwiz_engine_src}/core/sparse_feature_dataset_batcher_balancing.cpp
    ${_cyxwiz_engine_src}/core/classification_decision.cpp
    ${_cyxwiz_engine_src}/core/arrow_dataset.cpp
    ${_cyxwiz_engine_src}/core/arrow_dataset_batcher.cpp
    ${_cyxwiz_engine_src}/core/model_builder.cpp
    ${_cyxwiz_engine_src}/core/spatial_batch_layout.cpp
    ${_cyxwiz_engine_src}/core/executable_model.cpp
    ${_cyxwiz_engine_src}/core/graph_executable_model.cpp
    ${_cyxwiz_engine_src}/core/parquet_backed_dataset.cpp
    ${_cyxwiz_engine_src}/core/parquet_arrow_batcher.cpp
    ${_cyxwiz_engine_src}/core/dataset_base.cpp
    ${_cyxwiz_engine_src}/core/checkpoint_manager.cpp
    ${_cyxwiz_engine_src}/core/crash_run_recorder.cpp
    ${_cyxwiz_engine_src}/core/training_trace_collector.cpp
    ${_cyxwiz_engine_src}/core/route_qualification_snapshot.cpp
    ${_cyxwiz_engine_src}/core/runtime_log_store.cpp
)

if(TARGET cyxwiz-training-core)
    return()
endif()

find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
# Definitions only: node_metadata_registry lists data-convert formats by the
# same flags as the Engine (the adapters stay in their own targets).
find_package(OpenXLSX CONFIG QUIET)
find_package(HighFive CONFIG QUIET)

add_library(cyxwiz-training-hooks STATIC
    ${_cyxwiz_engine_src}/plugin/registries/plugin_training_hook_manager.cpp
    ${_cyxwiz_engine_src}/plugin/registries/plugin_node_registry.cpp
)
target_include_directories(cyxwiz-training-hooks PUBLIC ${_cyxwiz_engine_src})
target_link_libraries(cyxwiz-training-hooks PUBLIC spdlog::spdlog)
set_target_properties(cyxwiz-training-hooks PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    target_compile_definitions(cyxwiz-training-hooks PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Hosts adapt their own datasets to IBatchers and install a dataset catalog
# (graph_compiler_dataset_hooks.h); RunGraphTrainingJob installs its own.
add_library(cyxwiz-training-core STATIC
    ${CYXWIZ_TRAINING_EXECUTOR_HARNESS_SOURCES}
    ${_cyxwiz_engine_src}/core/graph_compiler.cpp
    ${_cyxwiz_engine_src}/core/graph_compiler_dataset_hooks.cpp
    ${_cyxwiz_engine_src}/core/graph_node_factory.cpp
    ${_cyxwiz_engine_src}/core/graph_document.cpp
    ${_cyxwiz_engine_src}/core/graph_training_prep.cpp
    ${_cyxwiz_engine_src}/core/graph_training_job.cpp
    ${_cyxwiz_engine_src}/core/compiled_graph_plan.cpp
    ${_cyxwiz_engine_src}/core/graph_topology_utils.cpp
    ${_cyxwiz_engine_src}/core/node_metadata_registry.cpp
    ${_cyxwiz_engine_src}/core/pipeline_runtime_capabilities.cpp
    ${_cyxwiz_engine_src}/core/sequence_arrow_batcher.cpp
    ${_cyxwiz_engine_src}/core/sha256_digest.cpp
    ${_cyxwiz_engine_src}/core/compute_runtime_config.cpp
    ${_cyxwiz_engine_src}/core/training_benchmark.cpp
    ${_cyxwiz_engine_src}/core/machine_capability.cpp
)
target_include_directories(cyxwiz-training-core PUBLIC ${_cyxwiz_engine_src})
target_link_libraries(cyxwiz-training-core PUBLIC
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
    cyxwiz-training-hooks
    OpenSSL::Crypto
)
set_target_properties(cyxwiz-training-core PROPERTIES CXX_STANDARD 20)
if(WIN32)
    target_compile_definitions(cyxwiz-training-core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
if(TARGET OpenXLSX::OpenXLSX)
    target_compile_definitions(cyxwiz-training-core PRIVATE CYXWIZ_HAS_XLSX)
endif()
if(HighFive_FOUND)
    target_compile_definitions(cyxwiz-training-core PRIVATE CYXWIZ_HAS_HDF5)
endif()
