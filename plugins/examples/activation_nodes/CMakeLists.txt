cmake_minimum_required(VERSION 3.20)
project(activation_nodes_plugin LANGUAGES CXX)

# Inside the CyxWiz source tree the SDK targets already exist; elsewhere,
# configure with -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>.
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

# Trainable example nodes (TOFIX125): ITensorNodeProvider.
add_library(activation_nodes SHARED
    activation_nodes_plugin.cpp
)

# The plugin headers and the backend (Tensor, Module): a trainable node's
# module is a real cyxwiz::Module.
target_link_libraries(activation_nodes PRIVATE CyxWiz::PluginSDK)

# Inside the tree the library goes to bin/ beside plugin.json, where the
# Engine and its tests find it; against the SDK, to <build>/activation_nodes.
if(TARGET cyxwiz-backend)
    cyxwiz_plugin_folder(activation_nodes FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    cyxwiz_plugin_folder(activation_nodes)
endif()
