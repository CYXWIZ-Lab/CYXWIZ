# The neural provider table of a configure: which optional providers this
# build makes, and why one is missing. cyxwiz-backend records each provider
# where it decides it; the configuration summary prints the table.
#
# A provider that was asked for and whose toolkit is missing warns and is left
# out; the configure does not fail (the gpu-dev preset relies on this).
#
# Each row is kept in global properties of its own (name, state, why by row
# number), never joined into one string, so a "why" may hold any text,
# including "|" and ";" (a list of toolkits, a path list).

# cyxwiz_record_provider(<name> <BUILT|NOT_BUILT> <why>)
function(cyxwiz_record_provider name state why)
    if(NOT state STREQUAL "BUILT" AND NOT state STREQUAL "NOT_BUILT")
        message(FATAL_ERROR "cyxwiz_record_provider: state must be BUILT or NOT_BUILT, not '${state}'")
    endif()
    get_property(count GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_COUNT)
    if(NOT count)
        set(count 0)
    endif()
    set_property(GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${count}_NAME" "${name}")
    set_property(GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${count}_STATE" "${state}")
    set_property(GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${count}_WHY" "${why}")
    math(EXPR count "${count} + 1")
    set_property(GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_COUNT "${count}")
endfunction()

function(cyxwiz_print_provider_table)
    get_property(count GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_COUNT)
    message(STATUS "Neural providers (optional accelerators; ArrayFire runs whatever they do not):")
    if(NOT count)
        message(STATUS "  none recorded")
        return()
    endif()
    math(EXPR last "${count} - 1")
    foreach(row RANGE 0 ${last})
        get_property(name GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${row}_NAME")
        get_property(state GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${row}_STATE")
        get_property(why GLOBAL PROPERTY "CYXWIZ_PROVIDER_TABLE_${row}_WHY")
        if(state STREQUAL "BUILT")
            message(STATUS "  ${name}: built (${why})")
        else()
            message(STATUS "  ${name}: not built (${why})")
        endif()
    endforeach()
endfunction()
