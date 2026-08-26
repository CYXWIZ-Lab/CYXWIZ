find_package(LibArchive REQUIRED)
if(NOT WIN32)
    find_package(httplib CONFIG REQUIRED)
endif()

add_library(cyxwiz-runtime-bootstrap STATIC
    atomic_file_publisher.cpp
    backend_pack_state_service.cpp
    product_installation_receipt.cpp
    product_removal_authorization.cpp
    product_removal_finalizer.cpp
    product_removal_quarantine.cpp
    product_removal_request.cpp
    runtime_layout.cpp
)
if(WIN32)
    target_sources(cyxwiz-runtime-bootstrap PRIVATE windows_dll_search.cpp)
endif()
target_include_directories(cyxwiz-runtime-bootstrap PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cyxwiz-runtime-bootstrap PUBLIC
    nlohmann_json::nlohmann_json
)

add_library(cyxwiz-backend-pack-maintenance STATIC
    backend_pack_maintenance_request.cpp
    backend_pack_remover.cpp
)
target_include_directories(cyxwiz-backend-pack-maintenance PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cyxwiz-backend-pack-maintenance PUBLIC
    cyxwiz-runtime-bootstrap
)

add_library(cyxwiz-backend-pack-service STATIC
    backend_pack_acquisition.cpp
    backend_pack_archive_extractor.cpp
    base_launcher_publisher.cpp
    backend_pack_hash.cpp
    backend_pack_https_source.cpp
    backend_pack_installer.cpp
    backend_pack_lifecycle_service.cpp
    backend_pack_metadata_cache.cpp
    backend_pack_metadata_refresh.cpp
    backend_pack_metadata_verifier.cpp
    backend_pack_path.cpp
)
target_include_directories(cyxwiz-backend-pack-service PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cyxwiz-backend-pack-service PRIVATE
    cyxwiz-backend-pack-maintenance
    cyxwiz-runtime-bootstrap
    LibArchive::LibArchive
    OpenSSL::Crypto
)
if(WIN32)
    target_link_libraries(cyxwiz-backend-pack-service PRIVATE winhttp)
else()
    target_link_libraries(cyxwiz-backend-pack-service PRIVATE
        httplib::httplib
        OpenSSL::SSL
    )
    if(APPLE)
        find_library(CYXWIZ_COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(CYXWIZ_CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
        target_link_libraries(cyxwiz-backend-pack-service PRIVATE
            "${CYXWIZ_COREFOUNDATION_FRAMEWORK}"
            "${CYXWIZ_CFNETWORK_FRAMEWORK}"
        )
    endif()
endif()

if(WIN32)
    set(_cyxwiz_runtime_bootstrapper_source main.cpp)
else()
    set(_cyxwiz_runtime_bootstrapper_source main_posix.cpp)
endif()
add_executable(cyxwiz-runtime-bootstrapper
    ${_cyxwiz_runtime_bootstrapper_source}
)
target_link_libraries(cyxwiz-runtime-bootstrapper PRIVATE
    cyxwiz-backend-pack-maintenance
    cyxwiz-runtime-bootstrap
)
set_target_properties(cyxwiz-runtime-bootstrapper PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if(NOT CYXWIZ_BUILD_INSTALLER_ONLY)
    install(TARGETS cyxwiz-runtime-bootstrapper RUNTIME DESTINATION .)
endif()
unset(_cyxwiz_runtime_bootstrapper_source)

# This executable is copied outside the installation before destructive
# cleanup. Compile its small validation closure directly so Windows can use a
# static CRT and the finalizer never depends on DLLs inside the product root.
add_executable(cyxwiz-product-removal-finalizer
    product_removal_finalizer_main.cpp
    product_removal_finalizer.cpp
    product_removal_request.cpp
    product_removal_authorization.cpp
    product_installation_receipt.cpp
    atomic_file_publisher.cpp
    runtime_layout.cpp
)
target_include_directories(cyxwiz-product-removal-finalizer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cyxwiz-product-removal-finalizer PRIVATE
    nlohmann_json::nlohmann_json
)
set_target_properties(cyxwiz-product-removal-finalizer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if(MSVC)
    set_property(TARGET cyxwiz-product-removal-finalizer PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
if(NOT CYXWIZ_BUILD_INSTALLER_ONLY)
    install(TARGETS cyxwiz-product-removal-finalizer RUNTIME DESTINATION .)
endif()

if(CYXWIZ_BUILD_TESTS)
    enable_testing()
    add_executable(generate_installer_metadata_fixture
        generate_installer_metadata_fixture.cpp
    )
    target_link_libraries(generate_installer_metadata_fixture PRIVATE
        nlohmann_json::nlohmann_json
        OpenSSL::Crypto
    )
    set_target_properties(generate_installer_metadata_fixture PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    if(WIN32)
        set(_cyxwiz_runtime_child_source test_runtime_child.cpp)
        set(_cyxwiz_runtime_contract_source test_runtime_layout.cpp)
    else()
        set(_cyxwiz_runtime_child_source test_runtime_child_posix.cpp)
        set(_cyxwiz_runtime_contract_source test_runtime_layout_posix.cpp)
    endif()
    add_executable(test_runtime_bootstrapper_child
        ${_cyxwiz_runtime_child_source}
    )
    target_link_libraries(test_runtime_bootstrapper_child PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_runtime_bootstrapper_child PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_runtime_bootstrapper
        ${_cyxwiz_runtime_contract_source}
    )
    target_link_libraries(test_runtime_bootstrapper PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_runtime_bootstrapper PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(test_runtime_bootstrapper
        cyxwiz-runtime-bootstrapper
        test_runtime_bootstrapper_child
    )
    add_test(NAME runtime_bootstrapper_contract COMMAND test_runtime_bootstrapper)
    unset(_cyxwiz_runtime_child_source)
    unset(_cyxwiz_runtime_contract_source)

    add_executable(test_backend_pack_state_service
        test_backend_pack_state_service.cpp
    )
    target_link_libraries(test_backend_pack_state_service PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_backend_pack_state_service PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_state_service_contract
        COMMAND test_backend_pack_state_service
    )

    add_executable(test_backend_pack_installer
        test_backend_pack_installer.cpp
    )
    target_link_libraries(test_backend_pack_installer PRIVATE
        cyxwiz-backend-pack-service
    )
    set_target_properties(test_backend_pack_installer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_installer_contract
        COMMAND test_backend_pack_installer
    )

    add_executable(test_backend_pack_metadata_verifier
        test_backend_pack_metadata_verifier.cpp
    )
    target_link_libraries(test_backend_pack_metadata_verifier PRIVATE
        cyxwiz-backend-pack-service
        OpenSSL::Crypto
    )
    set_target_properties(test_backend_pack_metadata_verifier PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_metadata_verifier_contract
        COMMAND test_backend_pack_metadata_verifier
    )

    add_executable(test_backend_pack_delivery
        test_backend_pack_delivery.cpp
    )
    target_link_libraries(test_backend_pack_delivery PRIVATE
        cyxwiz-backend-pack-service
        LibArchive::LibArchive
    )
    set_target_properties(test_backend_pack_delivery PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_delivery_contract
        COMMAND test_backend_pack_delivery
    )

    add_executable(test_backend_pack_maintenance
        test_backend_pack_maintenance.cpp
    )
    target_link_libraries(test_backend_pack_maintenance PRIVATE
        cyxwiz-backend-pack-service
    )
    set_target_properties(test_backend_pack_maintenance PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_maintenance_contract
        COMMAND test_backend_pack_maintenance
    )

    add_executable(test_backend_pack_maintenance_request
        test_backend_pack_maintenance_request.cpp
    )
    target_link_libraries(test_backend_pack_maintenance_request PRIVATE
        cyxwiz-backend-pack-maintenance
    )
    set_target_properties(test_backend_pack_maintenance_request PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_maintenance_request_contract
        COMMAND test_backend_pack_maintenance_request
    )

    add_executable(test_backend_pack_lifecycle_service
        test_backend_pack_lifecycle_service.cpp
    )
    target_link_libraries(test_backend_pack_lifecycle_service PRIVATE
        cyxwiz-backend-pack-service
        LibArchive::LibArchive
        OpenSSL::Crypto
    )
    set_target_properties(test_backend_pack_lifecycle_service PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(
        NAME backend_pack_lifecycle_service_contract
        COMMAND test_backend_pack_lifecycle_service
    )
endif()
