include_guard(GLOBAL)

if(ArrayFire_DIR)
    get_filename_component(_cyxwiz_default_arrayfire_root
        "${ArrayFire_DIR}/.." ABSOLUTE)
else()
    set(_cyxwiz_default_arrayfire_root "")
endif()

set(CYXWIZ_PACKAGE_OUTPUT_DIR
    "${CMAKE_SOURCE_DIR}/redist/output"
    CACHE PATH "Output directory for validated CyxWiz release packages")
set(CYXWIZ_PACKAGE_ARRAYFIRE_DIR
    "${_cyxwiz_default_arrayfire_root}"
    CACHE PATH "ArrayFire distribution root used by the full release package")
set(CYXWIZ_PACKAGE_PYTHON_DIR
    "${CMAKE_SOURCE_DIR}/redist/output/dependencies/python-3.12.8-embed-amd64"
    CACHE PATH "Embeddable Python 3.12 runtime used by the full release package")
set(CYXWIZ_PACKAGE_INTEL_NOTICES_DIR
    "${CMAKE_SOURCE_DIR}/redist/output/dependencies/intel-runtime-notices-2025.2"
    CACHE PATH "Intel redistribution notices used by the full release package")
set(CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR
    ""
    CACHE PATH "NVIDIA redistribution notices required when the CUDA pack is selected")
set(CYXWIZ_PACKAGE_FULL_BACKENDS
    "cpu,oneapi"
    CACHE STRING "Comma-separated ArrayFire backend packs for the full release package")

function(cyxwiz_add_release_packaging_targets)
    if(NOT TARGET cyxwiz-engine)
        message(FATAL_ERROR "Release packaging requires the cyxwiz-engine target")
    endif()

    find_package(Python3 3.8 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        message(STATUS
            "CyxWiz release packaging targets disabled; Python 3.8+ was not found")
        return()
    endif()
    set(_script "${CMAKE_SOURCE_DIR}/redist/scripts/package_release.py")
    set(_release_guard
        "${CMAKE_SOURCE_DIR}/cmake/RequireReleasePackageConfig.cmake")
    if(NOT EXISTS "${_script}")
        message(FATAL_ERROR "Release packaging script not found: ${_script}")
    endif()

    set(_common_args
        --build-dir "$<TARGET_FILE_DIR:cyxwiz-engine>"
        --resources-dir "${CMAKE_SOURCE_DIR}/cyxwiz-engine/resources"
        --output-root "${CYXWIZ_PACKAGE_OUTPUT_DIR}")

    add_custom_target(cyxwiz-package-minimal
        COMMAND "${CMAKE_COMMAND}"
                -DCYXWIZ_PACKAGE_CONFIG=$<CONFIG>
                -P "${_release_guard}"
        COMMAND "${Python3_EXECUTABLE}" "${_script}" minimal ${_common_args}
        DEPENDS cyxwiz-engine
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Building validated minimal CyxWiz package and manifest"
        USES_TERMINAL
        VERBATIM)

    set(_full_missing)
    foreach(_entry IN ITEMS
            CYXWIZ_PACKAGE_ARRAYFIRE_DIR
            CYXWIZ_PACKAGE_PYTHON_DIR
            CYXWIZ_PACKAGE_INTEL_NOTICES_DIR)
        if(NOT IS_DIRECTORY "${${_entry}}")
            list(APPEND _full_missing "${_entry}")
        endif()
    endforeach()
    if(CYXWIZ_PACKAGE_FULL_BACKENDS MATCHES "(^|,)cuda(,|$)" AND
       NOT IS_DIRECTORY "${CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR}")
        list(APPEND _full_missing "CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR")
    endif()

    if(_full_missing)
        list(JOIN _full_missing ", " _missing_text)
        message(STATUS
            "cyxwiz-package-full disabled; configure existing paths for: ${_missing_text}")
        return()
    endif()

    set(_full_args
        --arrayfire-dir "${CYXWIZ_PACKAGE_ARRAYFIRE_DIR}"
        --python-dir "${CYXWIZ_PACKAGE_PYTHON_DIR}"
        --intel-runtime-license-dir "${CYXWIZ_PACKAGE_INTEL_NOTICES_DIR}"
        --backends "${CYXWIZ_PACKAGE_FULL_BACKENDS}")
    if(CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR)
        list(APPEND _full_args
            --nvidia-runtime-license-dir "${CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR}")
    endif()

    add_custom_target(cyxwiz-package-full
        COMMAND "${CMAKE_COMMAND}"
                -DCYXWIZ_PACKAGE_CONFIG=$<CONFIG>
                -P "${_release_guard}"
        COMMAND "${Python3_EXECUTABLE}" "${_script}" full
                ${_common_args} ${_full_args}
        DEPENDS cyxwiz-engine
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Building validated full CyxWiz package and manifest"
        USES_TERMINAL
        VERBATIM)
endfunction()
