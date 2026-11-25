# Test, benchmark and tool executables run in a scratch user folder of their
# own, <temp>/cyxwiz-tests/<executable>-<pid>, never in the user's real
# settings, logs and caches: cyxwiz-engine/tests/support/scratch_user_folders.cpp
# points the user folders (cyxwiz-engine/src/core/user_folders.h) there before
# main(). Many test executables run outside ctest, so a CTest ENVIRONMENT would
# not be enough.
#
# Call cyxwiz_scratch_user_folders_for_tests() at the end of every CMakeLists
# that adds such executables: it adds the file to each executable of that
# directory named test_*, bench_*, benchmark_* or tool_*, and to the test tools
# listed below. A program that ships (redist/scripts/package_release.py) never
# gets it: it must use the user's real folders.

set(CYXWIZ_SCRATCH_TEST_TOOLS
    cyxwiz-tests
    cyxwiz-bench-out-of-core
    cyxwiz-bench-training-data
    cyxwiz-bench-training-loop
    cyxwiz-graph-compile-check
    cyxwiz-rl-stability-gate
    cyxwiz-training-profile
    generate_installer_metadata_fixture
    CACHE INTERNAL "Test tools that run in a scratch user folder")

set(CYXWIZ_SHIPPED_EXECUTABLES
    cyxwiz-engine
    cyxwiz-installer
    cyxwiz-setup
    cyxwiz-plugin-check
    cyxwiz-route-probe
    cyxwiz-backend-pack-installer
    cyxwiz-runtime-bootstrapper
    cyxwiz-cli
    cyxwiz-product-removal-finalizer
    cyxwiz-device-inventory
    CACHE INTERNAL "Programs that ship and use the user's real folders")

function(cyxwiz_scratch_user_folders_for_tests)
    set(_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/tests/support/scratch_user_folders.cpp")
    cmake_path(NORMAL_PATH _source)
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        get_target_property(_imported ${_target} IMPORTED)
        if(NOT _type STREQUAL "EXECUTABLE" OR _imported)
            continue()
        endif()
        if(_target MATCHES "^(test|bench|benchmark|tool)_" OR _target IN_LIST CYXWIZ_SCRATCH_TEST_TOOLS)
            if(_target IN_LIST CYXWIZ_SHIPPED_EXECUTABLES)
                message(FATAL_ERROR "${_target} ships: it must not get the test scratch user folder")
            endif()
            target_sources(${_target} PRIVATE "${_source}")
        endif()
    endforeach()
endfunction()
