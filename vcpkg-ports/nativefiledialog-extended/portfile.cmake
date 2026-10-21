# CyxWiz overlay of the vcpkg port (1.2.1): on Windows a dialog opens in the
# folder the Engine passes (IFileDialog::SetFolder), not in the folder Windows
# last remembered for cyxwiz-engine.exe (SetDefaultFolder, which only uses the
# given folder when nothing was remembered).
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO btzy/nativefiledialog-extended
    REF v${VERSION}
    SHA512 4ec3e174a90354c524d9be2776422740f80b73021df94e1942e60ab4310995245554f83097b9b2dcca04d016a8548d3fc0760f73daf724c5c3d72c15cf776bed
    HEAD_REF master
    PATCHES
        use-set-folder.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DNFD_BUILD_TESTS=OFF
        -DNFD_PORTAL=ON
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME nfd CONFIG_PATH lib/cmake/nfd)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_copy_pdbs()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
