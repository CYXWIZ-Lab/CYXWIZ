cmake_minimum_required(VERSION 3.20)
project(image_nodes_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(image_nodes SHARED
    image_nodes_plugin.cpp
)

# Link against the plugin SDK (provides PluginContext, registries, security)
target_link_libraries(image_nodes PRIVATE cyxwiz-plugin-sdk)

# Output to bin/ subdirectory
set_target_properties(image_nodes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
