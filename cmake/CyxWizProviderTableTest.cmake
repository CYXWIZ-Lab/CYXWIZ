# cmake -P cmake/CyxWizProviderTableTest.cmake (ctest CMakeProviderTable)
# The provider table keeps a "why" whole, whatever it holds ("|" and ";"
# included), and prints every row.
include("${CMAKE_CURRENT_LIST_DIR}/CyxWizProviderTable.cmake")

set(why_with_separators "CUDA Toolkit 12.6 | cuBLAS 12; searched C:/a;C:/b")
cyxwiz_record_provider("NVIDIA (test)" BUILT "${why_with_separators}")
cyxwiz_record_provider("oneAPI (test)" NOT_BUILT "no DPC++ compiler")

get_property(count GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_COUNT)
if(NOT count EQUAL 2)
    message(FATAL_ERROR "provider table: 2 rows recorded, ${count} kept")
endif()
get_property(why GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_0_WHY)
if(NOT why STREQUAL why_with_separators)
    message(FATAL_ERROR "provider table: the why '${why_with_separators}' came back as '${why}'")
endif()
get_property(state GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_1_STATE)
get_property(name GLOBAL PROPERTY CYXWIZ_PROVIDER_TABLE_1_NAME)
if(NOT state STREQUAL "NOT_BUILT" OR NOT name STREQUAL "oneAPI (test)")
    message(FATAL_ERROR "provider table: row 1 is '${name}' '${state}'")
endif()
cyxwiz_print_provider_table()
