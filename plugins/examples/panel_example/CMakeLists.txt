cmake_minimum_required(VERSION 3.20)
project(panel_example_plugin LANGUAGES CXX)

# Inside the CyxWiz source tree the SDK targets already exist; elsewhere,
# configure with -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>.
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()
if(NOT TARGET CyxWiz::PluginImGui)
    message(FATAL_ERROR "This CyxWiz SDK has no ImGui (CyxWiz::PluginImGui): a plugin with panels "
                        "needs an SDK from an Engine release that ships it.")
endif()

# A plugin with a panel (IPanelProvider), drawn with the Engine's widgets.
add_library(panel_example SHARED
    panel_example_plugin.cpp
)

# The plugin headers, and the ImGui the Engine was built with.
target_link_libraries(panel_example PRIVATE CyxWiz::PluginSDK CyxWiz::PluginImGui)

if(TARGET cyxwiz-backend)
    cyxwiz_plugin_folder(panel_example FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    cyxwiz_plugin_folder(panel_example)
endif()
