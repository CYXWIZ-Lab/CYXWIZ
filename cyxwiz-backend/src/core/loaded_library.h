#pragma once

#include "cyxwiz/api_export.h"

namespace cyxwiz::detail {

// The address of `symbol` in a library this process has already loaded,
// named by its stem: "afopencl" is afopencl.dll on Windows, and
// libafopencl.so, .so.3, .dylib or .3.10.0.dylib elsewhere, whatever name or
// folder it was loaded from. Never loads anything. Null when the library is
// not loaded or has no such symbol.
CYXWIZ_API void* FindLoadedSymbol(const char* stem, const char* symbol);

// Whether that library is loaded.
CYXWIZ_API bool IsLibraryLoaded(const char* stem);

} // namespace cyxwiz::detail
