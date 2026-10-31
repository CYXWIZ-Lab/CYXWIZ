# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
# Helpers for CyxWiz C++ plugins (TOFIX125 P5 step 5.3). Used by plugins
# built inside the CyxWiz tree and, installed beside CyxWizSDKConfig.cmake,
# by plugins built against the SDK.

# cyxwiz_plugin_folder(<target> [FOLDER <dir>])
#
# Makes <dir> (default: <build>/<target>) a plugin folder the Engine can
# install: the library goes to <dir>/bin and plugin.json (from the current
# source folder) beside it. Point the Engine's Plugin Manager > Install... at
# <dir>.
#
# The plugin must be built the way the Engine was (its build fingerprint is
# checked at load). The Engine is a Release build, so with MSVC every
# configuration of the plugin uses the release C++ runtime: a Debug build
# keeps its debug information and no optimisation, and still loads.
function(cyxwiz_plugin_folder target)
    cmake_parse_arguments(PARSE_ARGV 1 _cyxwiz "" "FOLDER" "")
    if(NOT _cyxwiz_FOLDER)
        set(_cyxwiz_FOLDER "${CMAKE_BINARY_DIR}/${target}")
    endif()
    foreach(_config "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY${_config} "${_cyxwiz_FOLDER}/bin"
            LIBRARY_OUTPUT_DIRECTORY${_config} "${_cyxwiz_FOLDER}/bin")
    endforeach()
    if(MSVC)
        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
    # plugin.json's api_version tells an Engine, before it loads the library,
    # which plugin API the plugin needs. Declaring an older one than this SDK's
    # lets an older Engine load the library and fail on a missing function.
    if(DEFINED CyxWizSDK_PLUGIN_API_VERSION)
        set(_cyxwiz_sdk_api "${CyxWizSDK_PLUGIN_API_VERSION}")
    else()
        set(_cyxwiz_sdk_api "${CYXWIZ_SDK_PLUGIN_API}")
    endif()
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/plugin.json" _cyxwiz_manifest)
    string(JSON _cyxwiz_api ERROR_VARIABLE _cyxwiz_json_error GET "${_cyxwiz_manifest}" api_version)
    if(_cyxwiz_json_error)
        set(_cyxwiz_api "1.0.0")
    endif()
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _cyxwiz_api_major_minor "${_cyxwiz_api}")
    if(_cyxwiz_sdk_api AND _cyxwiz_api_major_minor VERSION_LESS _cyxwiz_sdk_api)
        message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/plugin.json: api_version is ${_cyxwiz_api}, "
                            "but this SDK has plugin API ${_cyxwiz_sdk_api}. Set \"api_version\": "
                            "\"${_cyxwiz_sdk_api}.0\" (an older Engine then refuses the plugin "
                            "instead of failing to load it).")
    endif()
    get_filename_component(_cyxwiz_source "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    get_filename_component(_cyxwiz_folder "${_cyxwiz_FOLDER}" ABSOLUTE)
    if(NOT _cyxwiz_source STREQUAL _cyxwiz_folder)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/plugin.json" "${_cyxwiz_FOLDER}/plugin.json" COPYONLY)
    endif()
endfunction()
