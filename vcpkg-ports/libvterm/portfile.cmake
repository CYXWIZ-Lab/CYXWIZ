vcpkg_download_distfile(
    ARCHIVE
    URLS "https://github.com/neovim/libvterm/archive/refs/tags/v${VERSION}.tar.gz"
    FILENAME "libvterm-neovim-v${VERSION}.tar.gz"
    SHA512 400955c0e602177465034d3fd8e77779e883371fb889f52bc3620062563434523dd9f598d947086b31ba194e9df3af2d86721848b0a651b89960cea643c9cab9
)

vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
)

file(COPY
    "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_LIST_DIR}/libvtermConfig.cmake.in"
    DESTINATION "${SOURCE_PATH}"
)
file(COPY
    "${CMAKE_CURRENT_LIST_DIR}/generated/DECdrawing.inc"
    "${CMAKE_CURRENT_LIST_DIR}/generated/uk.inc"
    DESTINATION "${SOURCE_PATH}/src/encoding"
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME libvterm CONFIG_PATH lib/cmake/libvterm)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
