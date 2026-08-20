cmake_minimum_required(VERSION 3.20)
project(libvterm VERSION 0.3.3 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(vterm STATIC
    src/encoding.c
    src/keyboard.c
    src/mouse.c
    src/parser.c
    src/pen.c
    src/screen.c
    src/state.c
    src/unicode.c
    src/vterm.c
)
add_library(libvterm::vterm ALIAS vterm)

target_compile_features(vterm PUBLIC c_std_99)
target_include_directories(vterm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(vterm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME vterm
)

install(TARGETS vterm
    EXPORT libvtermTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT libvtermTargets
    FILE libvtermTargets.cmake
    NAMESPACE libvterm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libvterm
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/libvtermConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libvtermConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libvterm
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/libvtermConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libvtermConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/libvtermConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libvterm
)
