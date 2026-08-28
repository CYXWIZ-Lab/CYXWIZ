if(DEFINED CYXWIZ_INSTALLER_TARGETS_INCLUDED)
    return()
endif()
set(CYXWIZ_INSTALLER_TARGETS_INCLUDED ON)

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
    )
    target_link_libraries(cyxwiz-route-probe PRIVATE cyxwiz-backend)
    target_include_directories(cyxwiz-route-probe PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
    )
    set_target_properties(cyxwiz-route-probe PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

add_executable(cyxwiz-backend-pack-installer
    "${_cyxwiz_installer_engine_dir}/src/backend_pack_installer_main.cpp"
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
    add_executable(test_backend_pack_manager_model
        "${_cyxwiz_installer_engine_dir}/tests/test_backend_pack_manager_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_operation.cpp"
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
    )
    set_target_properties(test_backend_pack_manager_model PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_installer_verification_summary
        "${_cyxwiz_installer_engine_dir}/tests/test_installer_verification_summary.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/installer_verification_summary.cpp"
    )
    target_include_directories(test_installer_verification_summary PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${_cyxwiz_installer_backend_dir}/include"
        "${CMAKE_BINARY_DIR}/cyxwiz-backend/include"
        "${_cyxwiz_installer_generated_include}"
    )
    set_target_properties(test_installer_verification_summary PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_installer_product_removal
        "${_cyxwiz_installer_engine_dir}/tests/test_installer_product_removal.cpp"
        "${_cyxwiz_installer_engine_dir}/src/installer/installer_product_removal.cpp"
    )
    target_include_directories(test_installer_product_removal PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    target_link_libraries(test_installer_product_removal PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_installer_product_removal PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME installer_product_removal_contract
        COMMAND test_installer_product_removal
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
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_theme.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_view.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_operation.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_product_removal.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/installer_removal_view.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/installer_verification_summary.cpp"
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
    imgui::imgui
    glfw
    glad::glad
    OpenGL::GL
    cyxwiz-backend-pack-service
    cyxwiz-runtime-bootstrap
    nlohmann_json::nlohmann_json
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
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
    include(InstallRequiredSystemLibraries)
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
endif()

unset(_cyxwiz_installer_sources)
unset(_cyxwiz_installer_candidate_runtime_directories)
unset(_cyxwiz_installer_runtime_directory_args)
unset(_cyxwiz_installer_runtime_directories)
unset(_cyxwiz_installer_bootstrap_metadata_dir)
unset(_cyxwiz_installer_generated_include)
unset(_cyxwiz_installer_backend_dir)
unset(_cyxwiz_installer_engine_dir)
