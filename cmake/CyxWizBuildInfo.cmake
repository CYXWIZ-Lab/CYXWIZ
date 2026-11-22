# cyxwiz-build-info: the commit a binary is built from (Engine 2.0 core
# Phase 1, ORA-01). A test that links it prints the commit beside its result,
# so a result from an old build tree cannot pass for today's code.
#
# The source is regenerated at every build (cyxwiz-build-info-stamp) but
# rewritten only when the commit or the modified flag changes: a new commit
# recompiles one file and relinks the targets that link the library.

find_package(Git QUIET)
set(CYXWIZ_BUILD_INFO_SOURCE "${CMAKE_BINARY_DIR}/generated/cyxwiz_build_info.cpp")
set(CYXWIZ_BUILD_INFO_STAMP_ARGS
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DOUTPUT=${CYXWIZ_BUILD_INFO_SOURCE}"
    -P "${CMAKE_CURRENT_LIST_DIR}/CyxWizBuildInfoStamp.cmake")

# The first configure writes it, so the library has a source to list.
execute_process(COMMAND "${CMAKE_COMMAND}" ${CYXWIZ_BUILD_INFO_STAMP_ARGS})

add_custom_target(cyxwiz-build-info-stamp
    COMMAND "${CMAKE_COMMAND}" ${CYXWIZ_BUILD_INFO_STAMP_ARGS}
    BYPRODUCTS "${CYXWIZ_BUILD_INFO_SOURCE}"
    COMMENT "Build commit stamp"
    VERBATIM)

add_library(cyxwiz-build-info STATIC "${CYXWIZ_BUILD_INFO_SOURCE}")
add_dependencies(cyxwiz-build-info cyxwiz-build-info-stamp)
