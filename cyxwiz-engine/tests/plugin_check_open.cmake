# cyxwiz-plugin-check --open on the data plugin example (TOFIX181): a good
# file opens through the plugin's format and its analytic answers; a bad row,
# a file of another format and a file the format's CanLoad refuses each fail
# with their reason. Run by ctest:
#   cmake -DCHECK=<cyxwiz-plugin-check> -DPLUGIN=<data_formats folder>
#         -DWORK=<scratch folder> -P plugin_check_open.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

execute_process(COMMAND "${CHECK}" --open "${PLUGIN}/weather.arff" "${PLUGIN}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
message("${output}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "cyxwiz-plugin-check --open weather.arff failed (${result})")
endif()
foreach(expected
        "ok   open weather.arff as ARFF (Weka): 14 rows, 5 columns in 1 batch"
        "outlook (string), temperature (double), humidity (double), windy (string), play (string)"
        "ok   analytic Coefficient of variation on weather.arff (temperature, humidity): temperature 0.0893, humidity 0.1260"
        "all plugins pass")
    string(FIND "${output}" "${expected}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "missing from the output: ${expected}")
    endif()
endforeach()

file(WRITE "${WORK}/bad.arff" "@relation bad\n@attribute a numeric\n@attribute b numeric\n@data\n1,2\n3\n")
file(WRITE "${WORK}/data.csv" "x,y\n1,2\n")
file(WRITE "${WORK}/fake.arff" "not an arff file\n")
execute_process(COMMAND "${CHECK}" --open "${WORK}/bad.arff" --open "${WORK}/data.csv" --open "${WORK}/fake.arff"
                        "${PLUGIN}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
message("${output}")
if(NOT result EQUAL 1)
    message(FATAL_ERROR "files the plugin cannot read should fail the check (exit 1), got ${result}")
endif()
foreach(expected
        "FAIL open bad.arff: batch 1: line 6: 1 values, the header declares 2"
        "FAIL open data.csv: no dataset format of this plugin reads .csv files"
        "FAIL open fake.arff: ARFF (Weka) does not take the file (its CanLoad returned false)")
    string(FIND "${output}" "${expected}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "missing from the output: ${expected}")
    endif()
endforeach()
