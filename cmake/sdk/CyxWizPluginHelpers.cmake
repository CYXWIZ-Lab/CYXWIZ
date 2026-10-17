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
    get_filename_component(_cyxwiz_source "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    get_filename_component(_cyxwiz_folder "${_cyxwiz_FOLDER}" ABSOLUTE)
    if(NOT _cyxwiz_source STREQUAL _cyxwiz_folder)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/plugin.json" "${_cyxwiz_FOLDER}/plugin.json" COPYONLY)
    endif()
endfunction()
