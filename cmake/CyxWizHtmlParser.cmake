option(CYXWIZ_ENABLE_HTML_PARSER "Enable optional Lexbor parsed HTML in Text Clean" OFF)
if(CYXWIZ_ENABLE_HTML_PARSER)
    find_path(CYXWIZ_LEXBOR_INCLUDE_DIR lexbor/html/parser.h REQUIRED)
    get_filename_component(_lexbor_prefix "${CYXWIZ_LEXBOR_INCLUDE_DIR}" DIRECTORY)
    find_library(CYXWIZ_LEXBOR_RELEASE NAMES lexbor HINTS "${_lexbor_prefix}/lib" REQUIRED)
    find_library(CYXWIZ_LEXBOR_DEBUG NAMES lexbor HINTS "${_lexbor_prefix}/debug/lib" NO_DEFAULT_PATH)
    # vcpkg's multi-config search may select debug/lib for an ordinary find_library.
    # Bind both known Windows variants explicitly, as with the archive adapter.
    if(WIN32 AND EXISTS "${_lexbor_prefix}/lib/lexbor.lib"
             AND EXISTS "${_lexbor_prefix}/debug/lib/lexbor.lib")
        set(CYXWIZ_LEXBOR_RELEASE "${_lexbor_prefix}/lib/lexbor.lib" CACHE FILEPATH "Lexbor Release library" FORCE)
        set(CYXWIZ_LEXBOR_DEBUG "${_lexbor_prefix}/debug/lib/lexbor.lib" CACHE FILEPATH "Lexbor Debug library" FORCE)
    endif()
    add_library(cyxwiz-lexbor UNKNOWN IMPORTED)
    set_target_properties(cyxwiz-lexbor PROPERTIES
        IMPORTED_LOCATION "${CYXWIZ_LEXBOR_RELEASE}"
        INTERFACE_INCLUDE_DIRECTORIES "${CYXWIZ_LEXBOR_INCLUDE_DIR}")
    if(CYXWIZ_LEXBOR_DEBUG)
        set_target_properties(cyxwiz-lexbor PROPERTIES
            IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
            IMPORTED_LOCATION_DEBUG "${CYXWIZ_LEXBOR_DEBUG}"
            IMPORTED_LOCATION_RELEASE "${CYXWIZ_LEXBOR_RELEASE}"
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
            MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE)
    endif()
endif()
function(cyxwiz_target_html_parser target_name)
    if(NOT TARGET cyxwiz-html-document)
        add_library(cyxwiz-html-document STATIC
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/src/core/html_document_text.cpp")
        target_link_libraries(cyxwiz-html-document PRIVATE Arrow::arrow_shared nlohmann_json::nlohmann_json)
        if(CYXWIZ_ENABLE_HTML_PARSER)
            target_link_libraries(cyxwiz-html-document PRIVATE cyxwiz-lexbor)
            target_compile_definitions(cyxwiz-html-document PRIVATE CYXWIZ_HAS_HTML_PARSER)
        endif()
    endif()
    target_link_libraries(${target_name} PRIVATE cyxwiz-html-document)
endfunction()
