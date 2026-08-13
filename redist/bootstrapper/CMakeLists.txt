if(NOT WIN32)
    return()
endif()

add_library(cyxwiz-runtime-bootstrap STATIC
    runtime_layout.cpp
    windows_dll_search.cpp
)
target_include_directories(cyxwiz-runtime-bootstrap PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cyxwiz-runtime-bootstrap PUBLIC
    nlohmann_json::nlohmann_json
)

add_executable(cyxwiz-runtime-bootstrapper main.cpp)
target_link_libraries(cyxwiz-runtime-bootstrapper PRIVATE
    cyxwiz-runtime-bootstrap
)
set_target_properties(cyxwiz-runtime-bootstrapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS cyxwiz-runtime-bootstrapper RUNTIME DESTINATION .)

if(CYXWIZ_BUILD_TESTS)
    enable_testing()
    add_executable(test_runtime_bootstrapper_child test_runtime_child.cpp)
    target_link_libraries(test_runtime_bootstrapper_child PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_runtime_bootstrapper_child PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(test_runtime_bootstrapper test_runtime_layout.cpp)
    target_link_libraries(test_runtime_bootstrapper PRIVATE
        cyxwiz-runtime-bootstrap
    )
    set_target_properties(test_runtime_bootstrapper PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(test_runtime_bootstrapper
        cyxwiz-runtime-bootstrapper
        test_runtime_bootstrapper_child
    )
    add_test(NAME runtime_bootstrapper_contract COMMAND test_runtime_bootstrapper)
endif()
