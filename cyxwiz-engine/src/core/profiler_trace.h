#pragma once

#if defined(CYXWIZ_HAS_TRACY) && __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#define CYXWIZ_PROFILE_ZONE(name) ZoneScopedN(name)
#else
#define CYXWIZ_PROFILE_ZONE(name) ((void)0)
#endif
