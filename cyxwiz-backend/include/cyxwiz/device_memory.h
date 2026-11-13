// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Device memory, told in two parts that are measured differently:
// - what this process holds on its compute device (ArrayFire's memory
//   manager: buffers it has allocated, and the part of them in use);
// - what the device itself has free, for every program (asked from the
//   driver; today the NVIDIA management library answers, others do not).
// A figure nobody measured is "not known", never a guess.

#include "api_export.h"

#include <cstdint>
#include <string>

namespace cyxwiz {

// Bytes this process holds on the compute device.
struct CYXWIZ_API DeviceMemoryHeld {
    bool known = false;
    std::uint64_t allocated_bytes = 0;  // held by the memory manager (in use + kept for reuse)
    std::uint64_t in_use_bytes = 0;     // of those, in use by live arrays
    std::uint64_t buffers = 0;
    std::uint64_t buffers_in_use = 0;
    // From LastDeviceMemoryHeld: how long ago that reading was taken. A
    // watcher treats an old reading as not known rather than as current.
    double age_seconds = 0.0;
};

// Reads the calling thread's ArrayFire device and remembers the answer.
// ArrayFire's device is per thread: call this on a thread that computes (the
// training thread, the thread that initialised the library), not on a
// watcher thread. Not known before Initialize().
CYXWIZ_API DeviceMemoryHeld ReadDeviceMemoryHeld();

// The last answer of ReadDeviceMemoryHeld on any thread. No ArrayFire call:
// safe from a watcher thread.
CYXWIZ_API DeviceMemoryHeld LastDeviceMemoryHeld();

// The device's own memory, for every program.
struct CYXWIZ_API DeviceMemoryFree {
    bool known = false;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::string source;  // who answered ("NVIDIA management library"), or why nobody did
};

// Free memory of the device with this hardware UUID (DeviceInfo::hardware_uuid,
// 32 hex digits). Safe from any thread and needs no device context. An empty
// UUID matches when the machine has exactly one device the source knows.
CYXWIZ_API DeviceMemoryFree ReadDeviceMemoryFree(const std::string& hardware_uuid);

// What FreeUnusedDeviceMemory gave back.
struct CYXWIZ_API DeviceMemoryReleased {
    bool ran = false;  // false before Initialize()
    std::uint64_t allocated_before = 0;
    std::uint64_t allocated_after = 0;
    std::uint64_t freed_bytes() const {
        return allocated_before > allocated_after ? allocated_before - allocated_after : 0;
    }
};

// Returns the buffers the memory manager keeps for reuse to the driver
// (ArrayFire garbage collection) on the calling thread's device. Buffers in
// use stay. Safe to call at any time. (cyxwiz.h's ReleaseUnusedDeviceMemory
// is the same call without the answer.)
CYXWIZ_API DeviceMemoryReleased FreeUnusedDeviceMemory();

}  // namespace cyxwiz
