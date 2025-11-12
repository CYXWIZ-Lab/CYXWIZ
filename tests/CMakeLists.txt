cmake_minimum_required(VERSION 3.20)

find_package(Catch2 CONFIG REQUIRED)

# Unit tests
add_executable(cyxwiz-tests
    unit/test_tensor.cpp
    unit/test_device.cpp
    unit/test_optimizer.cpp
)

target_link_libraries(cyxwiz-tests PRIVATE
    Catch2::Catch2WithMain
    cyxwiz-backend
)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(cyxwiz-tests)
