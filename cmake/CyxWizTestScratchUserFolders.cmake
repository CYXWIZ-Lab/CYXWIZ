# Test, benchmark and tool executables run in a scratch user folder of their
# own, <temp>/cyxwiz-tests/<executable>-<pid>, never in the user's real
# settings, logs and caches: cyxwiz-engine/tests/support/scratch_user_folders.cpp
# points the user folders (cyxwiz-engine/src/core/user_folders.h) there before
# main(). Many test executables run outside ctest, so a CTest ENVIRONMENT would
# not be enough.
#
# Call cyxwiz_scratch_user_folders_for_tests() at the end of every CMakeLists
# that adds such executables: it adds the file to each executable of that
# directory named test_*, bench_* or tool_* (and the cyxwiz-tests suite).

function(cyxwiz_scratch_user_folders_for_tests)
    set(_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/tests/support/scratch_user_folders.cpp")
    cmake_path(NORMAL_PATH _source)
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        get_target_property(_imported ${_target} IMPORTED)
        if(_type STREQUAL "EXECUTABLE" AND NOT _imported AND
           (_target MATCHES "^(test|bench|tool)_" OR _target STREQUAL "cyxwiz-tests"))
            target_sources(${_target} PRIVATE "${_source}")
        endif()
    endforeach()
endfunction()
