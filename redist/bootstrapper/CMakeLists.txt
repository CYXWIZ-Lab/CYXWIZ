if(NOT WIN32)
    return()
endif()

find_package(LibArchive REQUIRED)

add_library(cyxwiz-runtime-bootstrap STATIC
    backend_pack_state_service.cpp
    runtime_layout.cpp
    windows_dll_search.cpp
)
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
    backend_pack_hash.cpp
    backend_pack_https_source.cpp
    backend_pack_installer.cpp
    backend_pack_lifecycle_service.cpp
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
    winhttp
)

add_executable(cyxwiz-runtime-bootstrapper main.cpp)
target_link_libraries(cyxwiz-runtime-bootstrapper PRIVATE
    cyxwiz-backend-pack-maintenance
    cyxwiz-runtime-bootstrap
)
set_target_properties(cyxwiz-runtime-bootstrapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS cyxwiz-runtime-bootstrapper RUNTIME DESTINATION .)

if(CYXWIZ_BUILD_TESTS)
    enable_testing()
    add_executable(test_runtime_bootstrapper_child test_runtime_child.cpp)
    target_link_libraries(test_runtime_bootstrapper_child PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_runtime_bootstrapper_child PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_runtime_bootstrapper test_runtime_layout.cpp)
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
