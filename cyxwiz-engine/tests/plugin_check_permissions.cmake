# cyxwiz-plugin-check on a plugin that declares a dangerous permission
# (TOFIX125 acceptance walk): the check grants it for the run instead of
# waiting for the Engine's approval dialog, says so, and still checks the
# nodes. Run by ctest:
#   cmake -DCHECK=<cyxwiz-plugin-check> -DPLUGIN=<activation_nodes folder>
#         -DWORK=<scratch folder> -P plugin_check_permissions.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}/activation_nodes")
file(COPY "${PLUGIN}/bin" DESTINATION "${WORK}/activation_nodes")
file(READ "${PLUGIN}/plugin.json" manifest)
string(REPLACE "\"permissions\": [\"Training\"]" "\"permissions\": [\"Training\", \"FileSystem\"]"
       manifest "${manifest}")
if(NOT manifest MATCHES "\"FileSystem\"")
    message(FATAL_ERROR "could not add FileSystem to the example's plugin.json")
endif()
file(WRITE "${WORK}/activation_nodes/plugin.json" "${manifest}")

execute_process(COMMAND "${CHECK}" "${WORK}/activation_nodes"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
message("${output}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "cyxwiz-plugin-check failed (${result})")
endif()
foreach(expected "permission FileSystem: granted for this check only" "ok   node " "all plugins pass")
    string(FIND "${output}" "${expected}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "missing from the output: ${expected}")
    endif()
endforeach()
