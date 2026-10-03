if(DEFINED CYXWIZ_INSTALLER_TARGETS_INCLUDED)
    return()
endif()
set(CYXWIZ_INSTALLER_TARGETS_INCLUDED ON)
include("${CMAKE_CURRENT_LIST_DIR}/CyxWizLibArchive.cmake")

set(_cyxwiz_installer_engine_dir "${CMAKE_SOURCE_DIR}/cyxwiz-engine")
set(_cyxwiz_installer_backend_dir "${CMAKE_SOURCE_DIR}/cyxwiz-backend")
set(CYXWIZ_INSTALLER_CATALOG_URL "" CACHE STRING
    "Default HTTPS URL for the signed CyxWiz backend-pack catalog")

# Installer-only builds consume the backend's public device data types without
# building the backend library that normally generates its export header.
set(_cyxwiz_installer_generated_include "")
if(NOT TARGET cyxwiz-backend)
    set(_cyxwiz_installer_generated_include
        "${CMAKE_BINARY_DIR}/installer-generated/include")
    file(MAKE_DIRECTORY
        "${_cyxwiz_installer_generated_include}/cyxwiz")
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/cyxwiz_installer_export.h"
        "${_cyxwiz_installer_generated_include}/cyxwiz/cyxwiz_export.h"
        COPYONLY
    )
endif()

if(TARGET cyxwiz-backend)
    add_executable(cyxwiz-route-probe
        "${CMAKE_SOURCE_DIR}/tests/smoke/test_oneapi_operation_probe.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/arrayfire_backend_discovery_isolation.cpp"
        "${CMAKE_SOURCE_DIR}/tests/smoke/route_probe_flatten_contract.cpp"
        "${CMAKE_SOURCE_DIR}/tests/smoke/route_probe_dropout_contract.cpp"
    )
    target_link_libraries(cyxwiz-route-probe PRIVATE cyxwiz-backend)
    target_include_directories(cyxwiz-route-probe PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(cyxwiz-route-probe PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Keep ArrayFire backends inside the packaged runtime (see source).
        target_sources(cyxwiz-route-probe PRIVATE
            "${_cyxwiz_installer_engine_dir}/src/core/arrayfire_load_guard_linux.cpp")
        target_link_options(cyxwiz-route-probe PRIVATE
            "LINKER:--export-dynamic-symbol=dlopen")
        target_link_libraries(cyxwiz-route-probe PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(cyxwiz-backend-pack-installer
    "${_cyxwiz_installer_engine_dir}/src/backend_pack_installer_main.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_cancellation_channel.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_helper_session.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_progress_channel.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_qualification_adapter.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/route_qualification_service.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/route_qualification_snapshot.cpp"
)
target_include_directories(cyxwiz-backend-pack-installer PRIVATE
    "${_cyxwiz_installer_engine_dir}/src"
    "${_cyxwiz_installer_backend_dir}/include"
    "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
    "${_cyxwiz_installer_generated_include}"
    "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
)
target_link_libraries(cyxwiz-backend-pack-installer PRIVATE
    cyxwiz-backend-pack-service
    cyxwiz-runtime-bootstrap
    nlohmann_json::nlohmann_json
)
if(TARGET cyxwiz-route-probe)
    add_dependencies(cyxwiz-backend-pack-installer cyxwiz-route-probe)
endif()
if(WIN32)
    target_link_libraries(cyxwiz-backend-pack-installer PRIVATE
        advapi32 ole32 shell32
    )
    if(MSVC)
        # The pack worker is an internal process controlled by the graphical
        # installer. Keep its wmain argument contract without allocating or
        # flashing a console window when Windows starts it directly.
        set_target_properties(cyxwiz-backend-pack-installer PROPERTIES
            WIN32_EXECUTABLE TRUE
        )
        target_link_options(cyxwiz-backend-pack-installer PRIVATE
            /ENTRY:wmainCRTStartup
        )
    endif()
endif()
set_target_properties(cyxwiz-backend-pack-installer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS cyxwiz-backend-pack-installer
    RUNTIME_DEPENDENCY_SET cyxwiz-installer-runtime-dependencies
    RUNTIME DESTINATION .
)

if(CYXWIZ_BUILD_TESTS)
    add_executable(test_arrayfire_backend_discovery_isolation
        "${_cyxwiz_installer_engine_dir}/tests/test_arrayfire_backend_discovery_isolation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/arrayfire_backend_discovery_isolation.cpp"
    )
    target_include_directories(test_arrayfire_backend_discovery_isolation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_arrayfire_backend_discovery_isolation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME arrayfire_backend_discovery_isolation_contract
        COMMAND test_arrayfire_backend_discovery_isolation
    )

    add_executable(test_backend_pack_manager_model
        "${_cyxwiz_installer_engine_dir}/tests/test_backend_pack_manager_model.cpp"
        "${_cyxwiz_installer_engine_dir}/tests/backend_pack_catalog_acceptance.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/installer_pack_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_operation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_cancellation_channel.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_helper_session.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_cuda_prerequisite.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_progress_channel.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_transaction_journal.cpp"
    )
    target_include_directories(test_backend_pack_manager_model PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${_cyxwiz_installer_backend_dir}/include"
        "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
        "${_cyxwiz_installer_generated_include}"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    target_link_libraries(test_backend_pack_manager_model PRIVATE
        cyxwiz-backend-pack-service
        nlohmann_json::nlohmann_json
        ${CMAKE_DL_LIBS}
    )
    set_target_properties(test_backend_pack_manager_model PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_installer_verification_summary
        "${_cyxwiz_installer_engine_dir}/tests/test_installer_verification_summary.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_decision_reconciliation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/installer_verification_summary.cpp"
    )
    target_include_directories(test_installer_verification_summary PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${_cyxwiz_installer_backend_dir}/include"
        "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
        "${_cyxwiz_installer_generated_include}"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    set_target_properties(test_installer_verification_summary PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_compute_device_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_compute_device_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/compute_device_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/route_qualification_snapshot.cpp"
    )
    target_include_directories(test_compute_device_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${_cyxwiz_installer_backend_dir}/include"
        "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
        "${_cyxwiz_installer_generated_include}"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    # Same sources and dependencies as cyxwiz-installer, which installer-only
    # builds produce without the backend library.
    target_link_libraries(test_compute_device_presentation PRIVATE
        cyxwiz-backend-pack-service
        cyxwiz-runtime-bootstrap
        nlohmann_json::nlohmann_json
    )
    set_target_properties(test_compute_device_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_python_repl_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_python_repl_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/python_repl_presentation.cpp"
    )
    target_include_directories(test_python_repl_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_python_repl_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME python_repl_presentation_contract
        COMMAND test_python_repl_presentation
    )

    add_executable(test_runtime_log_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_runtime_log_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/runtime_log_presentation.cpp"
    )
    target_include_directories(test_runtime_log_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_runtime_log_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME runtime_log_presentation_contract
        COMMAND test_runtime_log_presentation
    )

    add_executable(test_console_commands_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_console_commands_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/console_commands_presentation.cpp"
    )
    target_include_directories(test_console_commands_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_console_commands_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME console_commands_presentation_contract
        COMMAND test_console_commands_presentation
    )

    add_executable(test_appearance_options
        "${_cyxwiz_installer_engine_dir}/tests/test_appearance_options.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/appearance_options.cpp"
    )
    target_include_directories(test_appearance_options PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_appearance_options PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME appearance_options_contract
        COMMAND test_appearance_options
    )

    # Menu bar presentation model (TOFIX129): menus, palette, shortcuts.
    add_executable(test_menu_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_menu_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/menu_presentation.cpp"
    )
    target_include_directories(test_menu_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_menu_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME menu_presentation_contract
        COMMAND test_menu_presentation
    )

    # Design tokens (TOFIX129): status vocabulary, light and dark sets.
    add_executable(test_ui_tokens
        "${_cyxwiz_installer_engine_dir}/tests/test_ui_tokens.cpp"
        "${_cyxwiz_installer_engine_dir}/src/gui/ui_tokens.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/appearance_options.cpp"
    )
    target_include_directories(test_ui_tokens PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    target_link_libraries(test_ui_tokens PRIVATE imgui::imgui)
    set_target_properties(test_ui_tokens PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME ui_tokens_contract
        COMMAND test_ui_tokens
    )

    # Start page and Create Project model (TOFIX129 piece A2).
    add_executable(test_start_page_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_start_page_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/start_page_presentation.cpp"
    )
    target_include_directories(test_start_page_presentation PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(test_start_page_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME start_page_presentation_contract
        COMMAND test_start_page_presentation
    )

    # Python for scripting: start page chip and Python dialog (TOFIX129 A2-3).
    add_executable(test_python_setup_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_python_setup_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/python_setup_presentation.cpp"
    )
    set_target_properties(test_python_setup_presentation PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME python_setup_presentation_contract
        COMMAND test_python_setup_presentation
    )

    # Python scan cache: a start skips the scan when nothing changed.
    add_executable(test_python_scan_cache
        "${_cyxwiz_installer_engine_dir}/tests/test_python_scan_cache.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/python_scan_cache.cpp"
    )
    target_link_libraries(test_python_scan_cache PRIVATE nlohmann_json::nlohmann_json)
    set_target_properties(test_python_scan_cache PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME python_scan_cache_contract
        COMMAND test_python_scan_cache
    )

    # Script Editor file rules (TOFIX133 P0 items 1-3).
    add_executable(test_script_text_file
        "${_cyxwiz_installer_engine_dir}/tests/test_script_text_file.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/script_text_file.cpp"
    )
    set_target_properties(test_script_text_file PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME script_text_file_contract
        COMMAND test_script_text_file
    )

    # Script Editor Python colouring and palette byte order (TOFIX133 P0 items 4-5).
    add_executable(test_python_tokenizer
        "${_cyxwiz_installer_engine_dir}/tests/test_python_tokenizer.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/python_tokenizer.cpp"
    )
    add_executable(test_argb_colour
        "${_cyxwiz_installer_engine_dir}/tests/test_argb_colour.cpp"
    )
    # Find/Replace and editor columns (TOFIX133 P0 item 13).
    add_executable(test_text_search
        "${_cyxwiz_installer_engine_dir}/tests/test_text_search.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/text_search.cpp"
    )
    # .cyx notebook format and Python literals for the debugger (TOFIX133 P0 items 15-16).
    add_executable(test_cyx_format
        "${_cyxwiz_installer_engine_dir}/tests/test_cyx_format.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/cyx_format.cpp"
    )
    add_executable(test_python_literal
        "${_cyxwiz_installer_engine_dir}/tests/test_python_literal.cpp"
    )
    # Script Editor run/debug keys (TOFIX133 P0 item 7).
    add_executable(test_script_keys
        "${_cyxwiz_installer_engine_dir}/tests/test_script_keys.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/script_keys.cpp"
    )
    # Script Editor text model (TOFIX133 P1, decision D1).
    add_executable(test_text_document
        "${_cyxwiz_installer_engine_dir}/tests/test_text_document.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/text_document.cpp"
    )
    # Plot model: kinds, spec JSON, data labels, stats (TOFIX134 P1).
    add_executable(test_plot_model
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_model.cpp"
    )
    target_link_libraries(test_plot_model PRIVATE nlohmann_json::nlohmann_json)
    # Plot data preparation for every P1 kind (TOFIX134 P1).
    add_executable(test_plot_prepare
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_prepare.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_presets.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_prepare.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_image.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/series_decimation.cpp"
    )
    target_link_libraries(test_plot_prepare PRIVATE nlohmann_json::nlohmann_json)
    # Plot exports: CSV and SVG (TOFIX134 P1).
    add_executable(test_plot_export
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_export.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_export.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_prepare.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_image.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/series_decimation.cpp"
    )
    target_link_libraries(test_plot_export PRIVATE nlohmann_json::nlohmann_json)
    # Plot source from a DataTable: numeric columns, aligned rows (TOFIX134 P1).
    add_executable(test_plot_table_source
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_table_source.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_table_source.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_prepare.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_image.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/plot_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/series_decimation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/data/data_table.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/csv_records.cpp"
    )
    target_link_libraries(test_plot_table_source PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog fmt::fmt)
    if(MSVC)
        target_compile_options(test_plot_table_source PRIVATE /utf-8)
        target_compile_definitions(test_plot_table_source PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()
    # Plot node result lane plan: closure only, staleness, unavailable (TOFIX134 P2).
    add_executable(test_node_result_plan
        "${_cyxwiz_installer_engine_dir}/tests/test_node_result_plan.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot/node_result_plan.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/pipeline_type_names.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/pipeline_runtime_capabilities.cpp"
    )
    target_link_libraries(test_node_result_plan PRIVATE nlohmann_json::nlohmann_json)
    # Training Dashboard draws reduced series (TOFIX134 P0 item 8).
    add_executable(test_series_decimation
        "${_cyxwiz_installer_engine_dir}/tests/test_series_decimation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/series_decimation.cpp"
    )
    # Plot Output takes every published figure once (TOFIX134 P0 item 6).
    add_executable(test_plot_inbox
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_inbox.cpp"
    )
    # "Plot with Python" writes valid Python (TOFIX134 P0 item 4).
    add_executable(test_plot_script
        "${_cyxwiz_installer_engine_dir}/tests/test_plot_script.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/plot_script.cpp"
    )
    # Scatter/correlation pairs from one row (TOFIX134 P0 item 3).
    add_executable(test_paired_columns
        "${_cyxwiz_installer_engine_dir}/tests/test_paired_columns.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/paired_columns.cpp"
    )
    # Breakpoints follow their lines (TOFIX133 P6).
    add_executable(test_breakpoint_lines
        "${_cyxwiz_installer_engine_dir}/tests/test_breakpoint_lines.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/breakpoint_lines.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/text_document.cpp"
    )
    add_executable(test_editor_highlight_folding
        "${_cyxwiz_installer_engine_dir}/tests/test_editor_highlight_folding.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/text_document.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/python_highlight.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/folding.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/editor/outline.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/python_tokenizer.cpp"
    )
    # Jupyter .ipynb read/write (TOFIX133 P4, decision D3).
    add_executable(test_notebook_format
        "${_cyxwiz_installer_engine_dir}/tests/test_notebook_format.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/notebook_format.cpp"
    )
    target_link_libraries(test_notebook_format PRIVATE nlohmann_json::nlohmann_json)
    add_executable(test_notebook_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_notebook_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/notebook_presentation.cpp"
    )
    add_executable(test_markdown_blocks
        "${_cyxwiz_installer_engine_dir}/tests/test_markdown_blocks.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/markdown_blocks.cpp"
    )
    add_executable(test_html_table
        "${_cyxwiz_installer_engine_dir}/tests/test_html_table.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/html_table.cpp"
    )
    # CSV records for the Table Viewer (quoted commas, line breaks).
    add_executable(test_csv_records
        "${_cyxwiz_installer_engine_dir}/tests/test_csv_records.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/csv_records.cpp"
    )
    # Script Editor language tool results (TOFIX133 P3).
    add_executable(test_language_results
        "${_cyxwiz_installer_engine_dir}/tests/test_language_results.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/language_results.cpp"
    )
    target_link_libraries(test_language_results PRIVATE nlohmann_json::nlohmann_json)
    # Variable Explorer presentation (TOFIX133 P5).
    add_executable(test_variables_presentation
        "${_cyxwiz_installer_engine_dir}/tests/test_variables_presentation.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/variables_presentation.cpp"
    )
    target_link_libraries(test_variables_presentation PRIVATE nlohmann_json::nlohmann_json)
    foreach(_cyxwiz_p0_test test_python_tokenizer test_argb_colour test_text_search test_cyx_format
            test_python_literal test_script_keys test_text_document test_editor_highlight_folding
            test_notebook_format test_notebook_presentation
            test_markdown_blocks test_html_table test_csv_records
            test_language_results test_variables_presentation test_breakpoint_lines
            test_paired_columns test_plot_script test_plot_inbox test_series_decimation
            test_plot_model test_plot_prepare test_plot_export
            test_plot_table_source test_node_result_plan)
        set_target_properties(${_cyxwiz_p0_test} PROPERTIES
            CXX_STANDARD 20
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        add_test(NAME ${_cyxwiz_p0_test}_contract COMMAND ${_cyxwiz_p0_test})
    endforeach()
    if(Python3_EXECUTABLE)
        # The generated plot scripts are compiled by Python too.
        set_tests_properties(test_plot_script_contract PROPERTIES ENVIRONMENT "CYXWIZ_TEST_PYTHON=${Python3_EXECUTABLE}")
        set_tests_properties(test_plot_export_contract PROPERTIES ENVIRONMENT "CYXWIZ_TEST_PYTHON=${Python3_EXECUTABLE}")
    endif()

    add_executable(test_installer_product_removal
        "${_cyxwiz_installer_engine_dir}/tests/test_installer_product_removal.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_product_removal.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_external_session.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_external_session_platform.cpp"
    )
    target_include_directories(test_installer_product_removal PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    target_link_libraries(test_installer_product_removal PRIVATE
        cyxwiz-runtime-bootstrap
        LibArchive::LibArchive
        ${CMAKE_DL_LIBS}
    )
    set_target_properties(test_installer_product_removal PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME installer_product_removal_contract
        COMMAND test_installer_product_removal
    )
    add_test(
        NAME installer_helper_dependency_smoke
        COMMAND cyxwiz-backend-pack-installer --dependency-smoke
    )

    add_executable(test_product_registration
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_registration.cpp"
    )
    target_include_directories(test_product_registration PRIVATE
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    target_link_libraries(test_product_registration PRIVATE
        cyxwiz-runtime-bootstrap
    )
    if(WIN32)
        target_link_libraries(test_product_registration PRIVATE
            advapi32 ole32 shell32
        )
    endif()
    set_target_properties(test_product_registration PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_registration_contract
        COMMAND test_product_registration
    )

    add_executable(test_product_installation_receipt
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_installation_receipt.cpp"
    )
    target_link_libraries(test_product_installation_receipt PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_installation_receipt PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_installation_receipt_contract
        COMMAND test_product_installation_receipt
    )

    add_executable(test_product_removal_authorization
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_authorization.cpp"
    )
    target_link_libraries(test_product_removal_authorization PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_authorization PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_removal_authorization_contract
        COMMAND test_product_removal_authorization
    )

    add_executable(test_product_removal_request
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_request.cpp"
    )
    target_link_libraries(test_product_removal_request PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_request PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_removal_request_contract
        COMMAND test_product_removal_request
    )

    add_executable(test_product_removal_quarantine
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_quarantine.cpp"
    )
    target_link_libraries(test_product_removal_quarantine PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_quarantine PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_removal_quarantine_contract
        COMMAND test_product_removal_quarantine
    )

    add_executable(test_product_removal_finalizer
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_finalizer.cpp"
    )
    target_link_libraries(test_product_removal_finalizer PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_finalizer PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(test_product_removal_finalizer
        cyxwiz-product-removal-finalizer
    )
    add_test(
        NAME product_removal_finalizer_contract
        COMMAND test_product_removal_finalizer
    )

    add_executable(test_product_removal_finalizer_child
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_finalizer_child.cpp"
    )
    target_link_libraries(test_product_removal_finalizer_child PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_finalizer_child PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_product_removal_handoff
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_handoff.cpp"
    )
    target_link_libraries(test_product_removal_handoff PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_handoff PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(test_product_removal_handoff
        test_product_removal_finalizer_child
    )
    add_test(
        NAME product_removal_handoff_contract
        COMMAND test_product_removal_handoff
    )

    add_executable(test_product_removal_cleanup
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_cleanup.cpp"
    )
    target_link_libraries(test_product_removal_cleanup PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_cleanup PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_removal_cleanup_contract
        COMMAND test_product_removal_cleanup
    )

    add_executable(test_product_removal_transaction
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper/test_product_removal_transaction.cpp"
    )
    target_link_libraries(test_product_removal_transaction PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_product_removal_transaction PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME product_removal_transaction_contract
        COMMAND test_product_removal_transaction
    )
endif()

find_package(OpenGL REQUIRED)
set(_cyxwiz_installer_sources
    "${_cyxwiz_installer_engine_dir}/src/backend_pack_manager_main.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/backend_pack_installer_platform.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_cancellation_channel.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_helper_session.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_cuda_prerequisite.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_frame_pacing.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_progress_channel.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_transaction_journal.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_theme.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_view.cpp"
    "${_cyxwiz_installer_engine_dir}/src/gui/ui_buttons.cpp"
    "${_cyxwiz_installer_engine_dir}/src/gui/ui_tokens.cpp"
    "${_cyxwiz_installer_engine_dir}/src/gui/ui_widgets.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/appearance_options.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_operation.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_product_removal.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_external_session.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_external_session_platform.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_removal_view.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_decision_reconciliation.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/installer_pack_presentation.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/installer_verification_summary.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/compute_device_presentation.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/route_qualification_snapshot.cpp"
)
if(WIN32)
    list(APPEND _cyxwiz_installer_sources
        "${_cyxwiz_installer_engine_dir}/resources/installer_icon.rc")
    add_executable(cyxwiz-installer WIN32 ${_cyxwiz_installer_sources})
else()
    add_executable(cyxwiz-installer ${_cyxwiz_installer_sources})
endif()
target_include_directories(cyxwiz-installer PRIVATE
    "${_cyxwiz_installer_engine_dir}/src"
    "${_cyxwiz_installer_backend_dir}/include"
    "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
    "${_cyxwiz_installer_generated_include}"
    "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
)
target_link_libraries(cyxwiz-installer PRIVATE
    LibArchive::LibArchive
    imgui::imgui
    glfw
    glad::glad
    OpenGL::GL
    cyxwiz-backend-pack-service
    cyxwiz-runtime-bootstrap
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)
target_compile_definitions(cyxwiz-installer PRIVATE
    CYXWIZ_INSTALLER_DEFAULT_CATALOG_URL="${CYXWIZ_INSTALLER_CATALOG_URL}"
)
if(WIN32)
    target_link_libraries(cyxwiz-installer PRIVATE shell32)
endif()
add_dependencies(cyxwiz-installer cyxwiz-backend-pack-installer)
set_target_properties(cyxwiz-installer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_custom_command(TARGET cyxwiz-installer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "$<TARGET_FILE_DIR:cyxwiz-installer>/resources/fonts"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_cyxwiz_installer_engine_dir}/resources/cyxwiz.png"
        "$<TARGET_FILE_DIR:cyxwiz-installer>/resources/cyxwiz.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_cyxwiz_installer_engine_dir}/resources/fonts/Inter-Regular.ttf"
        "${_cyxwiz_installer_engine_dir}/resources/fonts/Inter-Bold.ttf"
        "${_cyxwiz_installer_engine_dir}/resources/fonts/fa-solid-900.ttf"
        "$<TARGET_FILE_DIR:cyxwiz-installer>/resources/fonts"
    COMMENT "Staging CyxWiz Installer visual resources"
)
install(TARGETS cyxwiz-installer
    RUNTIME_DEPENDENCY_SET cyxwiz-installer-runtime-dependencies
    RUNTIME DESTINATION .
)
install(FILES
    "${_cyxwiz_installer_engine_dir}/resources/cyxwiz.png"
    DESTINATION resources
)
install(FILES
    "${_cyxwiz_installer_engine_dir}/resources/fonts/Inter-Regular.ttf"
    "${_cyxwiz_installer_engine_dir}/resources/fonts/Inter-Bold.ttf"
    "${_cyxwiz_installer_engine_dir}/resources/fonts/fa-solid-900.ttf"
    DESTINATION resources/fonts
)

if(CYXWIZ_INSTALLER_BOOTSTRAP_METADATA_DIR)
    get_filename_component(
        _cyxwiz_installer_bootstrap_metadata_dir
        "${CYXWIZ_INSTALLER_BOOTSTRAP_METADATA_DIR}"
        ABSOLUTE
        BASE_DIR "${CMAKE_SOURCE_DIR}"
    )
    install(
        DIRECTORY "${_cyxwiz_installer_bootstrap_metadata_dir}/"
        DESTINATION runtime
    )
    message(STATUS
        "Installer bootstrap metadata: ${_cyxwiz_installer_bootstrap_metadata_dir}")
endif()

if(MSVC)
    # The dependency scans below exclude C:\Windows, so the MSVC runtime must
    # be installed explicitly for both the installer and the first-stage setup;
    # a clean machine may not have the VC++ redistributable.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
    include(InstallRequiredSystemLibraries)
    install(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION .)
    install(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
        DESTINATION . COMPONENT cyxwiz-setup)
endif()

set(_cyxwiz_installer_runtime_directories)
if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
    list(APPEND _cyxwiz_installer_runtime_directories
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
    )
elseif(VCPKG_INSTALLED_DIR)
    file(GLOB _cyxwiz_installer_candidate_runtime_directories
        LIST_DIRECTORIES TRUE
        "${VCPKG_INSTALLED_DIR}/*/bin"
    )
    list(APPEND _cyxwiz_installer_runtime_directories
        ${_cyxwiz_installer_candidate_runtime_directories}
    )
endif()
set(_cyxwiz_installer_runtime_directory_args)
if(_cyxwiz_installer_runtime_directories)
    list(APPEND _cyxwiz_installer_runtime_directory_args
        DIRECTORIES ${_cyxwiz_installer_runtime_directories}
    )
endif()
install(RUNTIME_DEPENDENCY_SET cyxwiz-installer-runtime-dependencies
    PRE_EXCLUDE_REGEXES
        "api-ms-win-.*"
        "ext-ms-.*"
        "azureattest.*"
        "hvsifiletrust\\.dll"
        "pdmutilities\\.dll"
        "wpaxholder\\.dll"
    POST_EXCLUDE_REGEXES
        ".*[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*"
        "^/lib/.*"
        "^/lib64/.*"
        "^/usr/lib/.*"
        "^/System/Library/.*"
    ${_cyxwiz_installer_runtime_directory_args}
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
    FRAMEWORK DESTINATION Frameworks
)
install(RUNTIME_DEPENDENCY_SET cyxwiz-setup-runtime-dependencies
    PRE_EXCLUDE_REGEXES
        "api-ms-win-.*"
        "ext-ms-.*"
        "azureattest.*"
        "hvsifiletrust\\.dll"
        "pdmutilities\\.dll"
        "wpaxholder\\.dll"
    POST_EXCLUDE_REGEXES
        ".*[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*"
        "^/lib/.*"
        "^/lib64/.*"
        "^/usr/lib/.*"
        "^/System/Library/.*"
    ${_cyxwiz_installer_runtime_directory_args}
    RUNTIME DESTINATION . COMPONENT cyxwiz-setup
    LIBRARY DESTINATION . COMPONENT cyxwiz-setup
    FRAMEWORK DESTINATION Frameworks COMPONENT cyxwiz-setup
)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION .)

if(TARGET cyxwiz-engine)
    if(TARGET cyxwiz-route-probe)
        add_dependencies(cyxwiz-engine cyxwiz-route-probe)
    endif()
    add_dependencies(cyxwiz-engine
        cyxwiz-backend-pack-installer
        cyxwiz-installer
    )
    if(MSVC AND CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        # package_release.py collects every DLL beside the Engine; ship the
        # MSVC runtime app-locally so the base needs no VC++ redistributable.
        add_custom_command(TARGET cyxwiz-engine POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
                "$<TARGET_FILE_DIR:cyxwiz-engine>"
            VERBATIM
        )
    endif()
endif()

unset(_cyxwiz_installer_sources)
unset(_cyxwiz_installer_candidate_runtime_directories)
unset(_cyxwiz_installer_runtime_directory_args)
unset(_cyxwiz_installer_runtime_directories)
unset(_cyxwiz_installer_bootstrap_metadata_dir)
unset(_cyxwiz_installer_generated_include)
unset(_cyxwiz_installer_backend_dir)
unset(_cyxwiz_installer_engine_dir)
