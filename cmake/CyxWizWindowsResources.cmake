# Windows version information (publisher, product, description, version)
# and, for user-facing programs, the CyxWiz icon, for every program CyxWiz
# ships. Without them an executable is anonymous to Windows: Explorer and the
# Start Menu show no icon, and security scanners trust it less.
#
#   cyxwiz_windows_resources(<target> DESCRIPTION "<text>" [ICON])
#
# The version comes from cyxwiz/version.h, as the Engine reports it.

set(_cyxwiz_windows_resources_dir "${CMAKE_CURRENT_LIST_DIR}")

function(cyxwiz_windows_resources target)
    if(NOT WIN32)
        return()
    endif()
    cmake_parse_arguments(ARG "ICON" "DESCRIPTION" "" ${ARGN})
    if(NOT ARG_DESCRIPTION)
        message(FATAL_ERROR "cyxwiz_windows_resources(${target}) needs a DESCRIPTION")
    endif()

    set(root "${_cyxwiz_windows_resources_dir}/..")
    file(STRINGS "${root}/cyxwiz-backend/include/cyxwiz/version.h" version_lines
         REGEX "#define CYXWIZ_VERSION_(MAJOR|MINOR|PATCH) ")
    foreach(part MAJOR MINOR PATCH)
        string(REGEX MATCH "CYXWIZ_VERSION_${part} ([0-9]+)" _ "${version_lines}")
        set(CYXWIZ_RC_${part} "${CMAKE_MATCH_1}")
    endforeach()
    set(CYXWIZ_RC_VERSION "${CYXWIZ_RC_MAJOR}.${CYXWIZ_RC_MINOR}.${CYXWIZ_RC_PATCH}")
    set(CYXWIZ_RC_TARGET "${target}")
    set(CYXWIZ_RC_DESCRIPTION "${ARG_DESCRIPTION}")
    if(ARG_ICON)
        get_filename_component(icon "${root}/cyxwiz-engine/resources/cyxwiz.ico" ABSOLUTE)
        set(CYXWIZ_RC_ICON_LINE "101 ICON \"${icon}\"")
    else()
        set(CYXWIZ_RC_ICON_LINE "")
    endif()

    set(rc "${CMAKE_CURRENT_BINARY_DIR}/${target}_windows_resources.rc")
    configure_file("${_cyxwiz_windows_resources_dir}/windows_resources.rc.in" "${rc}" @ONLY)
    target_sources(${target} PRIVATE "${rc}")
endfunction()
