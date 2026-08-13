if(NOT CYXWIZ_PACKAGE_CONFIG STREQUAL "Release")
    message(FATAL_ERROR
        "CyxWiz release packages require --config Release; received '${CYXWIZ_PACKAGE_CONFIG}'")
endif()
