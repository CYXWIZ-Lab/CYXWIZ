// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
//
// The Arrow C data interface and C stream interface: the structs of the
// Apache Arrow specification (https://arrow.apache.org/docs/format/CDataInterface.html,
// CStreamInterface.html), unchanged. They are the stable C ABI through which
// a plugin's data reaches the Engine (IDataProvider) and the Engine's data
// reaches a plugin (IAnalyticsProvider). A plugin needs no Arrow library to
// use them, and any Arrow library (Arrow C++, nanoarrow, arrow-rs, ...) of
// any version produces and reads them. The guards are the specification's,
// so this header and arrow/c/abi.h can be included together.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
    // Array type description
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    // Release callback
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data
    void* private_data;
};

struct ArrowArray {
    // Array data description
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    // Release callback
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
    // Callbacks providing stream functionality
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
    const char* (*get_last_error)(struct ArrowArrayStream*);

    // Release callback
    void (*release)(struct ArrowArrayStream*);

    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

#ifdef __cplusplus
}
#endif
