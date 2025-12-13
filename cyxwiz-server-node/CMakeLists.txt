cmake_minimum_required(VERSION 3.20)

project(cyxwiz-server-node)

# ========== Options ==========
option(CYXWIZ_BUILD_GUI "Build GUI mode (ImGui-based)" ON)
option(CYXWIZ_BUILD_TUI "Build TUI mode (FTXUI-based)" ON)
option(CYXWIZ_BUILD_DAEMON "Build server daemon" ON)
option(CYXWIZ_BUILD_GUI_CLIENT "Build GUI/TUI client" ON)

# ========== Required Packages ==========
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# Optional: Find ArrayFire for GPU support
find_package(ArrayFire QUIET)
if(ArrayFire_FOUND)
    message(STATUS "ArrayFire found - GPU support enabled")
    # Note: CYXWIZ_HAS_ARRAYFIRE is added per-target, not globally
else()
    message(WARNING "ArrayFire not found - GPU metrics will be limited")
endif()

# ========== GUI Dependencies (Conditional) ==========
if(CYXWIZ_BUILD_GUI)
    find_package(imgui CONFIG REQUIRED)
    find_package(implot CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(Stb REQUIRED)
    find_package(OpenGL REQUIRED)
    message(STATUS "GUI mode enabled - ImGui dependencies found")
endif()

# ========== TUI Dependencies (Conditional) ==========
if(CYXWIZ_BUILD_TUI)
    find_package(ftxui CONFIG REQUIRED)
    message(STATUS "TUI mode enabled - FTXUI found")
endif()

# ========== Shared Core Sources (Used by both daemon and client) ==========
set(SHARED_CORE_SOURCES
    src/core/backend_manager.cpp
    src/core/state_manager.cpp
    src/core/config_manager.cpp
    src/core/metrics_collector.cpp
    src/core/metrics_storage.cpp
    src/core/device_pool.cpp
    src/api/api_key_manager.cpp
)

# ========== IPC Sources ==========
# Daemon service (server-side) - only for daemon
set(IPC_SERVER_SOURCES
    src/ipc/daemon_service.cpp
)

# Daemon client (client-side) - only for GUI
set(IPC_CLIENT_SOURCES
    src/ipc/daemon_client.cpp
)

# ========== Daemon-Specific Sources ==========
set(DAEMON_SOURCES
    src/daemon_main.cpp
    src/deployment_handler.cpp
    src/deployment_manager.cpp
    src/model_loader.cpp
    src/terminal_handler.cpp
    src/node_client.cpp
    src/node_service.cpp
    src/job_execution_service.cpp
    src/node_server.cpp
    src/job_executor.cpp
    # API layer
    src/api/api_key_manager.cpp
    src/api/rate_limiter.cpp
    src/api/endpoint_manager.cpp
    # HTTP API server
    src/http/openai_api_server.cpp
    src/http/routes/models_route.cpp
    # gRPC InferenceService
    src/inference_handler.cpp
    src/http/routes/chat_route.cpp
    src/http/routes/completions_route.cpp
    src/http/routes/embeddings_route.cpp
    # Security
    src/security/api_key_validator.cpp
    src/security/tls_config.cpp
    src/security/audit_logger.cpp
    src/security/docker_manager.cpp
    src/security/wallet_manager.cpp
)

# ========== GUI Client Sources ==========
set(GUI_CLIENT_SOURCES
    src/gui_main.cpp
)

# ========== GUI Source Files (Conditional) ==========
set(GUI_SOURCES "")
if(CYXWIZ_BUILD_GUI)
    set(GUI_SOURCES
        src/gui/server_application.cpp
        src/gui/server_main_window.cpp
        src/gui/server_panel.cpp
        src/gui/theme.cpp
        src/gui/dock_style.cpp
        src/gui/panels/dashboard_panel.cpp
        src/gui/panels/hardware_panel.cpp
        src/gui/panels/allocation_panel.cpp
        src/gui/panels/job_monitor_panel.cpp
        src/gui/panels/model_browser_panel.cpp
        src/gui/panels/deployment_panel.cpp
        src/gui/panels/api_keys_panel.cpp
        src/gui/panels/settings_panel.cpp
        src/gui/panels/logs_panel.cpp
        src/gui/panels/pool_mining_panel.cpp
        src/gui/panels/marketplace_panel.cpp
        src/gui/panels/wallet_panel.cpp
        src/gui/panels/analytics_panel.cpp
        src/gui/panels/fine_tuning_panel.cpp
        src/gui/panels/login_panel.cpp
        src/gui/panels/account_settings_panel.cpp
        # Authentication
        src/auth/auth_manager.cpp
        # External: imnodes from engine
        ${CMAKE_SOURCE_DIR}/cyxwiz-engine/external/imnodes/imnodes.cpp
    )
endif()

# ========== TUI Source Files (Conditional) ==========
set(TUI_SOURCES "")
if(CYXWIZ_BUILD_TUI)
    set(TUI_SOURCES
        src/tui/tui_application.cpp
        src/tui/components/resource_gauges.cpp
    )
endif()

# ========================================
# TARGET 1: cyxwiz-server-daemon (Headless Service)
# ========================================
if(CYXWIZ_BUILD_DAEMON)
    add_executable(cyxwiz-server-daemon
        ${DAEMON_SOURCES}
        ${SHARED_CORE_SOURCES}
        ${IPC_SERVER_SOURCES}
    )

    target_include_directories(cyxwiz-server-daemon PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
    )

    target_link_libraries(cyxwiz-server-daemon PRIVATE
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        cyxwiz-backend
        cyxwiz-protocol
        gRPC::grpc++
        gRPC::grpc++_reflection
        protobuf::libprotobuf
        Threads::Threads
        httplib::httplib
        yaml-cpp::yaml-cpp
        OpenSSL::SSL
        OpenSSL::Crypto
        unofficial::sqlite3::sqlite3
    )

    # Platform-specific libraries for daemon
    if(WIN32)
        target_link_libraries(cyxwiz-server-daemon PRIVATE ws2_32 pdh iphlpapi)
        target_compile_definitions(cyxwiz-server-daemon PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NOGDI
            PROTOBUF_USE_DLLS
            GOOGLE_PROTOBUF_NO_THREADLOCAL
        )
    else()
        target_link_libraries(cyxwiz-server-daemon PRIVATE util pthread)
    endif()

    # ArrayFire linkage
    if(ArrayFire_FOUND)
        target_link_libraries(cyxwiz-server-daemon PRIVATE ArrayFire::af)
        target_compile_definitions(cyxwiz-server-daemon PRIVATE CYXWIZ_HAS_ARRAYFIRE)
    endif()

    # Mark as daemon build (for conditional includes in shared code)
    target_compile_definitions(cyxwiz-server-daemon PRIVATE CYXWIZ_IS_DAEMON)

    target_compile_features(cyxwiz-server-daemon PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(cyxwiz-server-daemon PRIVATE /W4)
    else()
        target_compile_options(cyxwiz-server-daemon PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    install(TARGETS cyxwiz-server-daemon RUNTIME DESTINATION bin)
    message(STATUS "Building cyxwiz-server-daemon (headless service)")
endif()

# ========================================
# TARGET 2: cyxwiz-server-gui (GUI/TUI Client)
# ========================================
if(CYXWIZ_BUILD_GUI_CLIENT AND (CYXWIZ_BUILD_GUI OR CYXWIZ_BUILD_TUI))
    add_executable(cyxwiz-server-gui
        ${GUI_CLIENT_SOURCES}
        ${SHARED_CORE_SOURCES}
        ${IPC_CLIENT_SOURCES}
        ${GUI_SOURCES}
        ${TUI_SOURCES}
    )

    target_include_directories(cyxwiz-server-gui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-engine/external/imnodes
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-engine/src
    )

    if(CYXWIZ_BUILD_GUI)
        target_include_directories(cyxwiz-server-gui PRIVATE ${Stb_INCLUDE_DIR})
    endif()

    # Core libraries
    target_link_libraries(cyxwiz-server-gui PRIVATE
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        cyxwiz-backend
        cyxwiz-protocol
        gRPC::grpc++
        protobuf::libprotobuf
        Threads::Threads
        yaml-cpp::yaml-cpp
        unofficial::sqlite3::sqlite3
        httplib::httplib
    )

    # GUI Libraries (Conditional)
    if(CYXWIZ_BUILD_GUI)
        target_link_libraries(cyxwiz-server-gui PRIVATE
            imgui::imgui
            implot::implot
            glfw
            glad::glad
            OpenGL::GL
        )
        target_compile_definitions(cyxwiz-server-gui PRIVATE CYXWIZ_HAS_GUI)
    endif()

    # TUI Libraries (Conditional)
    if(CYXWIZ_BUILD_TUI)
        target_link_libraries(cyxwiz-server-gui PRIVATE
            ftxui::screen
            ftxui::dom
            ftxui::component
        )
        target_compile_definitions(cyxwiz-server-gui PRIVATE CYXWIZ_HAS_TUI)
    endif()

    # Platform-specific libraries for GUI client
    if(WIN32)
        target_link_libraries(cyxwiz-server-gui PRIVATE ws2_32 pdh iphlpapi)
        target_compile_definitions(cyxwiz-server-gui PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            # Note: Don't use NOGDI here - GUI needs GDI types for dwmapi
            GOOGLE_PROTOBUF_NO_THREADLOCAL
        )
    else()
        target_link_libraries(cyxwiz-server-gui PRIVATE pthread)
    endif()

    # ArrayFire linkage (for device pool)
    if(ArrayFire_FOUND)
        target_link_libraries(cyxwiz-server-gui PRIVATE ArrayFire::af)
        target_compile_definitions(cyxwiz-server-gui PRIVATE CYXWIZ_HAS_ARRAYFIRE)
    endif()

    target_compile_features(cyxwiz-server-gui PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(cyxwiz-server-gui PRIVATE /W4)
    else()
        target_compile_options(cyxwiz-server-gui PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    install(TARGETS cyxwiz-server-gui RUNTIME DESTINATION bin)
    message(STATUS "Building cyxwiz-server-gui (GUI/TUI client)")
endif()

# ========================================
# Legacy: Single executable (for backward compatibility)
# Can be removed once dual-process is fully adopted
# ========================================
# Keeping the old main.cpp target commented out for reference
# add_executable(cyxwiz-server-node src/main.cpp ...)

# ========== Tests ==========
if(CYXWIZ_BUILD_TESTS)
    find_package(Catch2 CONFIG REQUIRED)
    enable_testing()

    # P2P Service Tests
    add_executable(test_job_execution_service
        tests/test_job_execution_service.cpp
        src/job_execution_service.cpp
        src/job_executor.cpp
        src/node_client.cpp
    )

    target_include_directories(test_job_execution_service PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
    )

    target_link_libraries(test_job_execution_service PRIVATE
        Catch2::Catch2
        Catch2::Catch2WithMain
        fmt::fmt
        spdlog::spdlog
        cyxwiz-backend
        cyxwiz-protocol
        gRPC::grpc++
        protobuf::libprotobuf
        Threads::Threads
    )

    if(WIN32)
        target_compile_definitions(test_job_execution_service PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NOGDI
            PROTOBUF_USE_DLLS
            GOOGLE_PROTOBUF_NO_THREADLOCAL
        )
    endif()

    # Mock Engine Client (standalone test tool)
    add_executable(mock_engine_client
        tests/mock_engine_client.cpp
    )

    target_include_directories(mock_engine_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
    )

    target_link_libraries(mock_engine_client PRIVATE
        cyxwiz-protocol
        gRPC::grpc++
        protobuf::libprotobuf
        Threads::Threads
    )

    if(WIN32)
        target_compile_definitions(mock_engine_client PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NOGDI
            PROTOBUF_USE_DLLS
            GOOGLE_PROTOBUF_NO_THREADLOCAL
        )
    endif()

    # Standalone P2P Server (for manual testing)
    add_executable(standalone_p2p_server
        tests/standalone_p2p_server.cpp
        src/job_execution_service.cpp
        src/job_executor.cpp
        src/node_client.cpp
    )

    target_include_directories(standalone_p2p_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
    )

    target_link_libraries(standalone_p2p_server PRIVATE
        cyxwiz-backend
        cyxwiz-protocol
        gRPC::grpc++
        protobuf::libprotobuf
        spdlog::spdlog
        fmt::fmt
        Threads::Threads
    )

    if(WIN32)
        target_compile_definitions(standalone_p2p_server PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NOGDI
            PROTOBUF_USE_DLLS
            GOOGLE_PROTOBUF_NO_THREADLOCAL
        )
    endif()

    # Security Tests
    add_executable(test_security
        tests/test_security.cpp
        src/api/api_key_manager.cpp
        src/api/rate_limiter.cpp
        src/security/tls_config.cpp
        src/security/audit_logger.cpp
        src/security/wallet_manager.cpp
        src/security/docker_manager.cpp
    )

    target_include_directories(test_security PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_security PRIVATE
        Catch2::Catch2
        Catch2::Catch2WithMain
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenSSL::SSL
        OpenSSL::Crypto
        httplib::httplib
        gRPC::grpc++
        Threads::Threads
    )

    if(WIN32)
        target_compile_definitions(test_security PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NOGDI
        )
    endif()

    # Register tests with CTest
    add_test(NAME P2PServiceTests COMMAND test_job_execution_service)
    add_test(NAME SecurityTests COMMAND test_security)

    message(STATUS "Server Node P2P tests enabled")
    message(STATUS "Server Node Security tests enabled")
endif()
