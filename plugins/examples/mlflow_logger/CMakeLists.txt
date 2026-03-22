cmake_minimum_required(VERSION 3.20)
project(mlflow_logger_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(mlflow_logger SHARED
    mlflow_logger_plugin.cpp
)

# Link against the plugin SDK (provides PluginContext, registries, security)
target_link_libraries(mlflow_logger PRIVATE cyxwiz-plugin-sdk)

# Output to bin/ subdirectory
set_target_properties(mlflow_logger PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
