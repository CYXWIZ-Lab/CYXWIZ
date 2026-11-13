#include "bindings_system.h"

#include <pybind11/stl.h>

#include "cyxwiz/device_memory.h"
#include "cyxwiz/memory_manager.h"

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pycyxwiz_bindings {

void BindSystem(py::module_& m) {
    auto s = m.def_submodule(
        "system", "The Engine's memory: what it holds on the compute device, what the device has free, host tensors");

    s.def(
        "device_memory_held",
        [] {
            const cyxwiz::DeviceMemoryHeld held = cyxwiz::ReadDeviceMemoryHeld();
            py::dict d;
            d["known"] = held.known;
            d["allocated_bytes"] = held.allocated_bytes;
            d["in_use_bytes"] = held.in_use_bytes;
            d["buffers"] = held.buffers;
            d["buffers_in_use"] = held.buffers_in_use;
            return d;
        },
        "What this process holds on its compute device (ArrayFire's memory manager): known, allocated_bytes "
        "(in use plus kept for reuse), in_use_bytes, buffers, buffers_in_use. known is False before initialize().");

    s.def(
        "device_memory_free",
        [](const std::string& hardware_uuid) {
            const cyxwiz::DeviceMemoryFree free_memory = cyxwiz::ReadDeviceMemoryFree(hardware_uuid);
            py::dict d;
            d["known"] = free_memory.known;
            d["total_bytes"] = free_memory.total_bytes;
            d["free_bytes"] = free_memory.free_bytes;
            d["source"] = free_memory.source;
            return d;
        },
        "hardware_uuid"_a = "",
        "The device's own memory for every program: known, total_bytes, free_bytes, source. Asked from the NVIDIA "
        "management library by DeviceInfo.hardware_uuid ('' = the only card); a driver that does not answer gives "
        "known False and source 'Not reported by this driver'. Never a guess.");

    s.def(
        "release_unused_device_memory",
        [] {
            const cyxwiz::DeviceMemoryReleased released = cyxwiz::FreeUnusedDeviceMemory();
            py::dict d;
            d["ran"] = released.ran;
            d["allocated_before"] = released.allocated_before;
            d["allocated_after"] = released.allocated_after;
            d["freed_bytes"] = released.freed_bytes();
            return d;
        },
        "Returns the device buffers kept for reuse to the driver; buffers in use stay. Answers ran, "
        "allocated_before, allocated_after and freed_bytes. ran is False before initialize().");

    s.def(
        "host_tensor_bytes", [] { return static_cast<unsigned long long>(cyxwiz::MemoryManager::GetAllocatedBytes()); },
        "Bytes of main memory held by host tensors (the tensor allocator; not the whole process).");
    s.def(
        "host_tensor_peak_bytes", [] { return static_cast<unsigned long long>(cyxwiz::MemoryManager::GetPeakBytes()); },
        "The highest host_tensor_bytes since the process started, since reset_host_tensor_peak(), or since the "
        "last training run started (a run starts its own peak).");
    s.def(
        "reset_host_tensor_peak", [] { cyxwiz::MemoryManager::ResetPeak(); },
        "Starts the host tensor peak again from the current value.");
}

}  // namespace pycyxwiz_bindings
