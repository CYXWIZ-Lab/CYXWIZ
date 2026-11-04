# Attach the same XLSX / HDF5 adapters to every executable compiling DataConvertService.
# Found here so every directory that includes this file sees the same features.
find_package(OpenXLSX CONFIG QUIET)
find_package(HighFive CONFIG QUIET)
if(TARGET OpenXLSX::OpenXLSX)
    include("${CMAKE_CURRENT_LIST_DIR}/CyxWizLibArchive.cmake")
endif()
function(cyxwiz_target_data_convert target_name)
    target_sources(${target_name} PRIVATE
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/src/core/excel_table_adapter.cpp"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/src/core/hdf5_table_adapter.cpp")
    if(TARGET OpenXLSX::OpenXLSX)
        target_sources(${target_name} PRIVATE
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cyxwiz-engine/src/core/xlsx_archive_guard.cpp")
        target_link_libraries(${target_name} PRIVATE OpenXLSX::OpenXLSX LibArchive::LibArchive)
        target_compile_definitions(${target_name} PRIVATE CYXWIZ_HAS_XLSX)
    endif()
    if(HighFive_FOUND)
        target_link_libraries(${target_name} PRIVATE HighFive)
        target_compile_definitions(${target_name} PRIVATE CYXWIZ_HAS_HDF5)
    endif()
endfunction()
