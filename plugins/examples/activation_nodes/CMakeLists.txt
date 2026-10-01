cmake_minimum_required(VERSION 3.20)
project(activation_nodes_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Trainable example nodes (TOFIX125): plugin API 1.1, ITensorNodeProvider.
add_library(activation_nodes SHARED
    activation_nodes_plugin.cpp
)

# The plugin SDK (PluginContext, interfaces) and the backend (Tensor, Module):
# a trainable node's module is a real cyxwiz::Module.
target_link_libraries(activation_nodes PRIVATE cyxwiz-plugin-sdk cyxwiz-backend)

# Output to bin/ subdirectory, beside plugin.json
set_target_properties(activation_nodes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
