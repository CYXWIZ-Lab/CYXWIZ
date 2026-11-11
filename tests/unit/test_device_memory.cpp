#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device_memory.h>

#include <arrayfire.h>

#include <thread>

// The CPU backend, whatever the machine's default is: these checks are about
// the bookkeeping, not about a graphics card.
namespace {
void UseCpu() {
    cyxwiz::Initialize();
    af::setBackend(AF_BACKEND_CPU);
}
}  // namespace

TEST_CASE("Held device memory follows what ArrayFire allocates", "[device_memory]") {
    UseCpu();
    af::deviceGC();
    const auto before = cyxwiz::ReadDeviceMemoryHeld();
    REQUIRE(before.known);
    {
        af::array block = af::constant(1.0f, 1 << 20);  // 4 MB of float32
        block.eval();
        const auto during = cyxwiz::ReadDeviceMemoryHeld();
        REQUIRE(during.known);
        REQUIRE(during.in_use_bytes >= before.in_use_bytes + (4u << 20));
        REQUIRE(during.allocated_bytes >= during.in_use_bytes);
        REQUIRE(during.buffers_in_use >= before.buffers_in_use + 1);
    }
    // The array is gone: its buffer is no longer in use, but is kept for reuse.
    const auto after = cyxwiz::ReadDeviceMemoryHeld();
    REQUIRE(after.in_use_bytes + (4u << 20) <= after.allocated_bytes);
}

TEST_CASE("Releasing unused device memory reports what it gave back", "[device_memory]") {
    UseCpu();
    {
        af::array block = af::constant(2.0f, 1 << 21);  // 8 MB
        block.eval();
    }
    const auto released = cyxwiz::FreeUnusedDeviceMemory();
    REQUIRE(released.ran);
    REQUIRE(released.freed_bytes() >= (8u << 20));
    REQUIRE(released.allocated_after == cyxwiz::ReadDeviceMemoryHeld().allocated_bytes);
    // Nothing left to give back.
    REQUIRE(cyxwiz::FreeUnusedDeviceMemory().freed_bytes() == 0);
    // The call without an answer (the SDK's first form) frees the same way.
    {
        af::array block = af::constant(4.0f, 1 << 20);
        block.eval();
    }
    cyxwiz::ReleaseUnusedDeviceMemory();
    REQUIRE(cyxwiz::FreeUnusedDeviceMemory().freed_bytes() == 0);
}

TEST_CASE("Another thread sees the last reading without touching ArrayFire", "[device_memory]") {
    UseCpu();
    af::array block = af::constant(3.0f, 1 << 18);
    block.eval();
    const auto read = cyxwiz::ReadDeviceMemoryHeld();
    cyxwiz::DeviceMemoryHeld seen;
    std::thread watcher([&seen] { seen = cyxwiz::LastDeviceMemoryHeld(); });
    watcher.join();
    REQUIRE(seen.known);
    REQUIRE(seen.allocated_bytes == read.allocated_bytes);
    REQUIRE(seen.in_use_bytes == read.in_use_bytes);
    REQUIRE(seen.buffers == read.buffers);
    REQUIRE(seen.buffers_in_use == read.buffers_in_use);
    REQUIRE(seen.age_seconds >= 0.0);
    REQUIRE(seen.age_seconds < 5.0);  // just taken
    REQUIRE(read.age_seconds == 0.0);  // a fresh reading has no age
}

TEST_CASE("The device's free memory is known or says who did not answer", "[device_memory]") {
    // No such device: never a guess.
    const auto nobody = cyxwiz::ReadDeviceMemoryFree("00000000000000000000000000000000");
    REQUIRE(!nobody.known);
    REQUIRE(nobody.total_bytes == 0);
    REQUIRE(nobody.source == "Not reported by this driver");
    // The only device, when the machine has a source: free never exceeds total.
    const auto only = cyxwiz::ReadDeviceMemoryFree("");
    if (only.known) {
        REQUIRE(only.total_bytes > 0);
        REQUIRE(only.free_bytes <= only.total_bytes);
        REQUIRE(only.source == "NVIDIA management library");
    } else {
        REQUIRE(only.source == "Not reported by this driver");
    }
}

TEST_CASE("A listed CUDA device carries its free memory when the driver's library answers", "[device_memory][devices]") {
    cyxwiz::Initialize();
    if (!cyxwiz::ReadDeviceMemoryFree("").known) return;  // no NVIDIA library, or several cards
    bool cuda_seen = false;
    for (const auto& device : cyxwiz::Device::GetAvailableDevices()) {
        if (device.type != cyxwiz::DeviceType::CUDA || !device.hardware_uuid_known) continue;
        cuda_seen = true;
        const auto free_memory = cyxwiz::ReadDeviceMemoryFree(device.hardware_uuid);
        REQUIRE(free_memory.known);  // the CUDA UUID names the same card to the library
        REQUIRE(device.memory_available_known);
        REQUIRE(device.memory_total_known);
        REQUIRE(device.memory_available <= device.memory_total);
        // Two sources for the total (CUDA and the management library) agree within 1 %.
        const double ratio = static_cast<double>(free_memory.total_bytes) / static_cast<double>(device.memory_total);
        REQUIRE(ratio > 0.99);
        REQUIRE(ratio < 1.01);
    }
    if (!cuda_seen) SUCCEED("no CUDA device listed");
}
