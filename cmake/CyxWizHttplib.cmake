# cpp-httplib from vcpkg exports CPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO.
# Its Windows branch (GetAddrInfoExW, still so in 0.65.0) casts the
# ADDRINFOEXW list to addrinfo, so ai_next reads ADDRINFOEXW::ai_blob: only
# the first address is ever tried (localhost -> ::1, no IPv4 fallback; a
# host whose IPv6 is broken never falls back to IPv4) and the list is freed
# with freeaddrinfo instead of FreeAddrInfoExW. Plain getaddrinfo returns a
# real list and the connection timeout still applies to each address.
#
# Included at the top level before any add_subdirectory, so every target
# below links the corrected imported target (their own find_package of
# httplib finds it already defined). QUIET: installer-only builds may not
# have httplib.
find_package(httplib CONFIG QUIET)
if(WIN32 AND TARGET httplib::httplib)
    get_target_property(_cyxwiz_httplib_defs httplib::httplib INTERFACE_COMPILE_DEFINITIONS)
    if(_cyxwiz_httplib_defs)
        list(FILTER _cyxwiz_httplib_defs EXCLUDE REGEX "CPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO")
        set_target_properties(httplib::httplib PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_cyxwiz_httplib_defs}")
    endif()
    unset(_cyxwiz_httplib_defs)
endif()
