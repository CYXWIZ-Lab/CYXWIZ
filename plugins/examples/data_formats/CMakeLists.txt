cmake_minimum_required(VERSION 3.20)
project(data_formats_plugin LANGUAGES CXX)

# Inside the CyxWiz source tree the SDK targets already exist; elsewhere,
# configure with -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>.
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

# A dataset format and a dataset analytic (TOFIX181): IDataProvider and
# IAnalyticsProvider, on the Arrow C stream (no Arrow library needed).
add_library(data_formats SHARED
    data_formats_plugin.cpp
)
target_link_libraries(data_formats PRIVATE CyxWiz::PluginSDK)

# Inside the tree the library goes to bin/ beside plugin.json, where the
# Engine and its tests find it; against the SDK, to <build>/data_formats.
if(TARGET cyxwiz-backend)
    cyxwiz_plugin_folder(data_formats FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    cyxwiz_plugin_folder(data_formats)
endif()
