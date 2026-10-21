// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// This header includes the CMake-generated export header
// and provides the CYXWIZ_API macro for backward compatibility

#include "cyxwiz/cyxwiz_export.h"

// Alias CYXWIZ_EXPORT to CYXWIZ_API for backward compatibility
#ifndef CYXWIZ_API
    #define CYXWIZ_API CYXWIZ_EXPORT
#endif
