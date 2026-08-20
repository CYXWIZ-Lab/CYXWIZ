if(DEFINED CYXWIZ_INSTALLER_TARGETS_INCLUDED)
    return()
endif()
set(CYXWIZ_INSTALLER_TARGETS_INCLUDED ON)

set(_cyxwiz_installer_engine_dir "${CMAKE_SOURCE_DIR}/cyxwiz-engine")
set(_cyxwiz_installer_backend_dir "${CMAKE_SOURCE_DIR}/cyxwiz-backend")

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
set_target_properties(cyxwiz-backend-pack-installer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS cyxwiz-backend-pack-installer RUNTIME DESTINATION .)

if(CYXWIZ_BUILD_TESTS)
    add_executable(test_backend_pack_manager_model
        "${_cyxwiz_installer_engine_dir}/tests/test_backend_pack_manager_model.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
        "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
    )
    target_include_directories(test_backend_pack_manager_model PRIVATE
        "${_cyxwiz_installer_engine_dir}/src"
        "${CMAKE_SOURCE_DIR}/redist/bootstrapper"
    )
    set_target_properties(test_backend_pack_manager_model PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

find_package(OpenGL REQUIRED)
set(_cyxwiz_installer_sources
    "${_cyxwiz_installer_engine_dir}/src/backend_pack_manager_main.cpp"
    "${_cyxwiz_installer_engine_dir}/src/installer/backend_pack_installer_platform.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_catalog_adapter.cpp"
    "${_cyxwiz_installer_engine_dir}/src/core/backend_pack_manager_model.cpp"
)
if(WIN32)
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
)
if(WIN32)
    target_link_libraries(cyxwiz-installer PRIVATE shell32)
endif()
add_dependencies(cyxwiz-installer cyxwiz-backend-pack-installer)
set_target_properties(cyxwiz-installer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS cyxwiz-installer RUNTIME DESTINATION .)

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
unset(_cyxwiz_installer_generated_include)
unset(_cyxwiz_installer_backend_dir)
unset(_cyxwiz_installer_engine_dir)
