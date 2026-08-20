#pragma once

// Installer-only builds use backend value types but do not export backend APIs.
#define CYXWIZ_EXPORT
#define CYXWIZ_NO_EXPORT
#define CYXWIZ_DEPRECATED
#define CYXWIZ_DEPRECATED_EXPORT
#define CYXWIZ_DEPRECATED_NO_EXPORT
