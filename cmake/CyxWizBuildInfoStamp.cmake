# Script mode (cmake -P), run at every build by the cyxwiz-build-info-stamp
# target: writes the commit the build is made from, and whether tracked files
# differ from it, into OUTPUT. The file is rewritten only when that changes, so
# an unchanged checkout recompiles and relinks nothing.
#
# In: GIT_EXECUTABLE (may be empty), SOURCE_DIR, OUTPUT.

set(commit "unknown")
set(modified 0)
if(GIT_EXECUTABLE)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                    WORKING_DIRECTORY "${SOURCE_DIR}"
                    OUTPUT_VARIABLE head RESULT_VARIABLE head_result
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(head_result EQUAL 0 AND head)
        set(commit "${head}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD --
                        WORKING_DIRECTORY "${SOURCE_DIR}"
                        RESULT_VARIABLE diff_result OUTPUT_QUIET ERROR_QUIET)
        if(diff_result EQUAL 1)
            set(modified 1)
        endif()
    endif()
endif()

set(content "// Generated at build time by cmake/CyxWizBuildInfoStamp.cmake: do not edit.
extern \"C\" const char* cyxwiz_build_commit() { return \"${commit}\"; }
extern \"C\" int cyxwiz_build_modified() { return ${modified}; }
")
set(previous "")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" previous)
endif()
if(NOT previous STREQUAL content)
    file(WRITE "${OUTPUT}" "${content}")
endif()
