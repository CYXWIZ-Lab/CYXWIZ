cmake_minimum_required(VERSION 3.20)
project(cyxwiz_assistant_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cyxwiz_assistant SHARED
    knowledge_pack_backend.cpp
    cyxwiz_assistant_plugin.cpp
)

find_package(httplib CONFIG REQUIRED)

target_link_libraries(cyxwiz_assistant PRIVATE
    cyxwiz-plugin-sdk
    httplib::httplib
)

set_target_properties(cyxwiz_assistant PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
