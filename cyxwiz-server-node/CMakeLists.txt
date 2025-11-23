cmake_minimum_required(VERSION 3.20)

project(cyxwiz-server-node)

# Find packages
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Optional: Find ArrayFire for GPU support
find_package(ArrayFire QUIET)
if(ArrayFire_FOUND)
    message(STATUS "ArrayFire found - GPU support enabled")
    add_definitions(-DCYXWIZ_HAS_ARRAYFIRE)
else()
    message(WARNING "ArrayFire not found - GPU metrics will be limited")
endif()

# Source files
set(SERVER_NODE_SOURCES
    src/main.cpp
    src/deployment_handler.cpp
    src/deployment_manager.cpp
    src/model_loader.cpp
    src/terminal_handler.cpp
    src/metrics_collector.cpp
    src/node_client.cpp
    src/node_service.cpp
    src/job_execution_service.cpp  # P2P service
    # Legacy files (to be refactored)
    src/node_server.cpp
    src/job_executor.cpp
)

# Create executable
add_executable(cyxwiz-server-node
    ${SERVER_NODE_SOURCES}
)

target_include_directories(cyxwiz-server-node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../cyxwiz-protocol
)

target_link_libraries(cyxwiz-server-node PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    cyxwiz-backend
    cyxwiz-protocol
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
    Threads::Threads
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(cyxwiz-server-node PRIVATE
        ws2_32  # Windows Sockets
    )
else()
    target_link_libraries(cyxwiz-server-node PRIVATE
        util    # For PTY (pseudo-terminal) on Unix/Linux
        pthread
    )
endif()

# ArrayFire linkage
if(ArrayFire_FOUND)
    target_link_libraries(cyxwiz-server-node PRIVATE ArrayFire::af)
endif()

# Compiler options
target_compile_features(cyxwiz-server-node PRIVATE cxx_std_20)

# Windows: Prevent min/max macro conflicts with C++ standard library
if(WIN32)
    target_compile_definitions(cyxwiz-server-node PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NOGDI  # Exclude GDI (fixes DeviceCapabilities conflict)
        # Protobuf compatibility flags to fix arena allocation issue
        PROTOBUF_USE_DLLS              # Use DLL linkage
        GOOGLE_PROTOBUF_NO_THREADLOCAL # Disable thread-local optimization
    )
endif()

if(MSVC)
    target_compile_options(cyxwiz-server-node PRIVATE /W4)
else()
    target_compile_options(cyxwiz-server-node PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install
install(TARGETS cyxwiz-server-node
    RUNTIME DESTINATION bin
)

# ========== Tests ==========
if(CYXWIZ_BUILD_TESTS)
    find_package(Catch2 CONFIG REQUIRED)
    enable_testing()

    # P2P Service Tests
    add_executable(test_job_execution_service
        tests/test_job_execution_service.cpp
        src/job_execution_service.cpp
        src/job_executor.cpp
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

    # Register tests with CTest
    add_test(NAME P2PServiceTests COMMAND test_job_execution_service)

    message(STATUS "Server Node P2P tests enabled")
endif()
