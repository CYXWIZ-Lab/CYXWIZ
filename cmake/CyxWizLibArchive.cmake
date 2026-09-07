# CMake's FindLibArchive exposes one IMPORTED_LOCATION. With vcpkg's
# multi-config search paths that can select debug/lib even for Release.
# Preserve upstream discovery/transitive dependencies, then map both variants
# only for the recognized Windows vcpkg layout of the selected installation.
find_package(LibArchive REQUIRED)
if(WIN32 AND TARGET LibArchive::LibArchive)
    get_filename_component(_cyxwiz_archive_prefix "${LibArchive_INCLUDE_DIR}" DIRECTORY)
    set(_cyxwiz_archive_release "${_cyxwiz_archive_prefix}/lib/archive.lib")
    set(_cyxwiz_archive_debug "${_cyxwiz_archive_prefix}/debug/lib/archive.lib")
    if(EXISTS "${_cyxwiz_archive_prefix}/share/libarchive/vcpkg-cmake-wrapper.cmake"
       AND EXISTS "${_cyxwiz_archive_release}" AND EXISTS "${_cyxwiz_archive_debug}"
       AND (LibArchive_LIBRARY STREQUAL _cyxwiz_archive_release
            OR LibArchive_LIBRARY STREQUAL _cyxwiz_archive_debug))
        set_target_properties(LibArchive::LibArchive PROPERTIES
            IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
            IMPORTED_LOCATION "${_cyxwiz_archive_release}"
            IMPORTED_LOCATION_DEBUG "${_cyxwiz_archive_debug}"
            IMPORTED_LOCATION_RELEASE "${_cyxwiz_archive_release}"
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
            MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE)
        # Keep legacy consumers and cache diagnostics truthful too.
        set(LibArchive_LIBRARY "${_cyxwiz_archive_release}" CACHE FILEPATH
            "libarchive default library (target selects per configuration)" FORCE)
        set(LibArchive_LIBRARIES LibArchive::LibArchive)
    endif()
    unset(_cyxwiz_archive_prefix)
    unset(_cyxwiz_archive_release)
    unset(_cyxwiz_archive_debug)
endif()
