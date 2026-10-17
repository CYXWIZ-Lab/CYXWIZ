cmake_minimum_required(VERSION 3.20)
project(my_nodes LANGUAGES CXX)

# Inside the CyxWiz source tree the SDK targets already exist; elsewhere,
# configure with -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>.
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

add_library(my_nodes SHARED my_nodes_plugin.cpp)
target_link_libraries(my_nodes PRIVATE CyxWiz::PluginSDK)

# <build>/my_nodes: plugin.json and bin/my_nodes.dll, ready for the Engine's
# Plugin Manager > Install...
cyxwiz_plugin_folder(my_nodes)
