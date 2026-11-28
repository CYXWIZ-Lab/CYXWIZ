# The neural provider table of a configure: which optional providers this
# build makes, and why one is missing. cyxwiz-backend records each provider
# where it decides it; the configuration summary prints the table.
#
# A provider that was asked for and whose toolkit is missing warns and is left
# out; the configure does not fail (the gpu-dev preset relies on this).

# cyxwiz_record_provider(<name> <BUILT|NOT_BUILT> <why>)
function(cyxwiz_record_provider name state why)
    if(NOT state STREQUAL "BUILT" AND NOT state STREQUAL "NOT_BUILT")
        message(FATAL_ERROR "cyxwiz_record_provider: state must be BUILT or NOT_BUILT, not '${state}'")
    endif()
    set_property(GLOBAL APPEND PROPERTY CYXWIZ_PROVIDER_TABLE "${name}|${state}|${why}")
endfunction()

function(cyxwiz_print_provider_table)
    get_property(rows GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE)
    message(STATUS "Neural providers (optional accelerators; ArrayFire runs whatever they do not):")
    if(NOT rows)
        message(STATUS "  none recorded")
        return()
    endif()
    foreach(row IN LISTS rows)
        string(REPLACE "|" ";" fields "${row}")
        list(GET fields 0 name)
        list(GET fields 1 state)
        list(GET fields 2 why)
        if(state STREQUAL "BUILT")
            message(STATUS "  ${name}: built (${why})")
        else()
            message(STATUS "  ${name}: not built (${why})")
        endif()
    endforeach()
endfunction()
