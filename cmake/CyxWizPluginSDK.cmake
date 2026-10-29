# The CyxWiz plugin SDK package (TOFIX125 P5 step 5.3, Part D9).
#
# Included by cyxwiz-engine/CMakeLists.txt after the cyxwiz-plugin-sdk
# target. Inside the tree it defines CyxWiz::PluginSDK and the plugin
# helpers, so a plugin's CMakeLists builds the same in the tree and against
# the installed SDK. It also installs the SDK, only when asked for:
#
#   cmake --install <build> --config Release --component plugin_sdk --prefix <folder>
#
# <folder>/include          plugin headers, backend headers (Tensor, Module)
# <folder>/lib              the backend's import library
# <folder>/lib/cmake/CyxWizSDK  find_package(CyxWizSDK) -> CyxWiz::PluginSDK
# <folder>/template         the template plugin (plugins/template)
# <folder>/examples/python-nodes  ten built-in layers as Python nodes (examples/python_nodes)
# <folder>/examples/plugins/data-formats  a dataset format and analytic (plugins/examples/data_formats),
#                                         source and built/ (ready to install)
# <folder>/LICENSE, NOTICE  Apache-2.0 for every file in the SDK

include(CMakePackageConfigHelpers)

# The SDK version is the backend's (cyxwiz/version.h), the version the build
# fingerprint names.
file(STRINGS "${CMAKE_SOURCE_DIR}/cyxwiz-backend/include/cyxwiz/version.h" _cyxwiz_sdk_version_lines
     REGEX "#define CYXWIZ_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(_line ${_cyxwiz_sdk_version_lines})
    string(REGEX MATCH "CYXWIZ_VERSION_([A-Z]+) ([0-9]+)" _ "${_line}")
    set(CYXWIZ_SDK_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
endforeach()
set(CYXWIZ_SDK_VERSION "${CYXWIZ_SDK_VERSION_MAJOR}.${CYXWIZ_SDK_VERSION_MINOR}.${CYXWIZ_SDK_VERSION_PATCH}")

file(STRINGS "${CMAKE_SOURCE_DIR}/cyxwiz-engine/src/plugin/plugin_types.h" _cyxwiz_sdk_api_lines
     REGEX "PLUGIN_API_VERSION_(MAJOR|MINOR) = ")
foreach(_line ${_cyxwiz_sdk_api_lines})
    string(REGEX MATCH "PLUGIN_API_VERSION_([A-Z]+) = ([0-9]+)" _ "${_line}")
    set(_cyxwiz_sdk_api_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
endforeach()
set(CYXWIZ_SDK_PLUGIN_API "${_cyxwiz_sdk_api_MAJOR}.${_cyxwiz_sdk_api_MINOR}")
set(CYXWIZ_SDK_BUILT_WITH "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on ${CMAKE_SYSTEM_NAME}")

# Inside the tree: the same names a plugin uses against the installed SDK.
add_library(cyxwiz-plugin-sdk-backend INTERFACE)
target_link_libraries(cyxwiz-plugin-sdk-backend INTERFACE cyxwiz-plugin-sdk cyxwiz-backend)
add_library(CyxWiz::PluginSDK ALIAS cyxwiz-plugin-sdk-backend)
include("${CMAKE_SOURCE_DIR}/cmake/sdk/CyxWizPluginHelpers.cmake")

set(_cyxwiz_sdk_cmake "${CMAKE_BINARY_DIR}/sdk/cmake")
configure_file("${CMAKE_SOURCE_DIR}/cmake/sdk/CyxWizSDKConfig.cmake.in"
               "${_cyxwiz_sdk_cmake}/CyxWizSDKConfig.cmake" @ONLY)
write_basic_package_version_file("${_cyxwiz_sdk_cmake}/CyxWizSDKConfigVersion.cmake"
    VERSION ${CYXWIZ_SDK_VERSION} COMPATIBILITY SameMinorVersion)

install(DIRECTORY "${CYXWIZ_PLUGIN_SDK_INCLUDE_DIR}/"
        DESTINATION include COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/cyxwiz-backend/include/cyxwiz"
        DESTINATION include COMPONENT plugin_sdk EXCLUDE_FROM_ALL
        PATTERN "distributed" EXCLUDE)
install(FILES "${CMAKE_BINARY_DIR}/cyxwiz-backend/include/cyxwiz/cyxwiz_export.h"
        DESTINATION include/cyxwiz COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
# The library a plugin links. Windows: the import library. Elsewhere the
# linker file is the lib<name>.so / .dylib symlink, which would point at a
# file the SDK does not have (and packaging keeps real files only): install
# the real shared library under that name instead.
if(WIN32)
    install(FILES "$<TARGET_LINKER_FILE:cyxwiz-backend>"
            DESTINATION lib COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
else()
    install(FILES "$<TARGET_FILE:cyxwiz-backend>"
            DESTINATION lib COMPONENT plugin_sdk EXCLUDE_FROM_ALL
            RENAME "${CMAKE_SHARED_LIBRARY_PREFIX}cyxwiz-backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
install(FILES
            "${_cyxwiz_sdk_cmake}/CyxWizSDKConfig.cmake"
            "${_cyxwiz_sdk_cmake}/CyxWizSDKConfigVersion.cmake"
            "${CMAKE_SOURCE_DIR}/cmake/sdk/CyxWizPluginHelpers.cmake"
        DESTINATION lib/cmake/CyxWizSDK COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
# Python example nodes: ten built-in layers written as a Python node bundle,
# an example graph and their test (examples/python_nodes/builtin_layers).
set(_cyxwiz_sdk_python_example "${CMAKE_SOURCE_DIR}/examples/python_nodes/builtin_layers")
install(FILES
            "${_cyxwiz_sdk_python_example}/README.md"
            "${_cyxwiz_sdk_python_example}/builtin_layers.py"
            "${_cyxwiz_sdk_python_example}/test_builtin_layers.py"
            "${_cyxwiz_sdk_python_example}/blobs_python_layers.cyxgraph"
            "${_cyxwiz_sdk_python_example}/blobs.csv"
            "${CMAKE_SOURCE_DIR}/cyxwiz-backend/python/tests/pycyxwiz_test_env.py"
        DESTINATION examples/python-nodes/builtin-layers COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
install(DIRECTORY "${_cyxwiz_sdk_python_example}/nodes/builtin-layers.cyxnode"
        DESTINATION examples/python-nodes/builtin-layers/nodes COMPONENT plugin_sdk EXCLUDE_FROM_ALL
        PATTERN "__pycache__" EXCLUDE)
# The data plugin example (TOFIX181): its source to build against the SDK,
# and built/, the same plugin built here, ready for Plugin Manager > Install.
# The library is installed as a file (as the backend above), so the target
# plugins/examples/data_formats defines later needs no install rule there.
set(_cyxwiz_sdk_data_example "${CMAKE_SOURCE_DIR}/plugins/examples/data_formats")
install(FILES
            "${_cyxwiz_sdk_data_example}/CMakeLists.txt"
            "${_cyxwiz_sdk_data_example}/plugin.json"
            "${_cyxwiz_sdk_data_example}/data_formats_plugin.cpp"
            "${_cyxwiz_sdk_data_example}/README.md"
            "${_cyxwiz_sdk_data_example}/weather.arff"
        DESTINATION examples/plugins/data-formats COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
install(FILES "${_cyxwiz_sdk_data_example}/plugin.json"
        DESTINATION examples/plugins/data-formats/built COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
install(FILES "$<TARGET_FILE:data_formats>"
        DESTINATION examples/plugins/data-formats/built/bin COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
# Apache-2.0 (owner 2026-10-17, the plugin-SDK part of TOFIX142).
install(FILES "${CMAKE_SOURCE_DIR}/cmake/sdk/LICENSE" "${CMAKE_SOURCE_DIR}/cmake/sdk/NOTICE"
        DESTINATION . COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
install(FILES
            "${CMAKE_SOURCE_DIR}/plugins/template/CMakeLists.txt"
            "${CMAKE_SOURCE_DIR}/plugins/template/plugin.json"
            "${CMAKE_SOURCE_DIR}/plugins/template/my_nodes_plugin.cpp"
            "${CMAKE_SOURCE_DIR}/plugins/template/README.md"
        DESTINATION template COMPONENT plugin_sdk EXCLUDE_FROM_ALL)
