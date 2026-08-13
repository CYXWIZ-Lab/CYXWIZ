if(NOT DEFINED PROBE_EXE OR NOT EXISTS "${PROBE_EXE}")
    message(FATAL_ERROR "PROBE_EXE must name the built oneAPI operation probe")
endif()

if(NOT DEFINED REPORT_PATH)
    set(REPORT_PATH "${CMAKE_CURRENT_BINARY_DIR}/oneapi-operation-matrix.txt")
endif()

set(operations
    device_info
    constant
    randu
    randu_scaled
    abs
    exp
    log
    maximum
    sum
    mean
    sigmoid
    matmul
    identity
    transpose
    bce_forward_expression
    bce_backward_expression
    tensor_row_major
    cyxwiz_bce_forward
    cyxwiz_bce_backward
    linear_init
)

set(report "oneAPI isolated operation matrix\nprobe=${PROBE_EXE}\n")
set(pass_count 0)
set(skip_count 0)
set(failure_count 0)

foreach(operation IN LISTS operations)
    message(STATUS "oneAPI probe: ${operation}")
    execute_process(
        COMMAND "${PROBE_EXE}" "${operation}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        TIMEOUT 20
    )

    if("${result}" STREQUAL "0")
        set(verdict "pass")
        math(EXPR pass_count "${pass_count} + 1")
    elseif("${result}" STREQUAL "77")
        set(verdict "skip")
        math(EXPR skip_count "${skip_count} + 1")
    else()
        set(verdict "failed")
        math(EXPR failure_count "${failure_count} + 1")
    endif()

    string(APPEND report
        "\n=== ${operation} verdict=${verdict} result=${result} ===\n"
        "${output}${error}")
    message(STATUS "oneAPI probe: ${operation} -> ${verdict} (${result})")
endforeach()

string(APPEND report
    "\nsummary pass=${pass_count} skip=${skip_count} failed=${failure_count}\n")
file(WRITE "${REPORT_PATH}" "${report}")

message(STATUS
    "oneAPI operation matrix complete: pass=${pass_count} skip=${skip_count} "
    "failed=${failure_count}")
message(STATUS "oneAPI operation report: ${REPORT_PATH}")
