// A lent GPU registers with the runtime that drives it, not "CUDA" for every
// GPU (found on the Dell, 2026-10-21: an Intel Iris Xe stored as cuda).
#include "core/allocation_device_type.h"

#include <catch2/catch_test_macros.hpp>

using namespace cyxwiz::servernode::core;

namespace {

std::vector<KnownDevice> ThisPc() {
    // The backend's enumeration on CJ_PC: CPU, a CUDA GPU, the Intel GPU on oneAPI.
    return {{cyxwiz::DeviceType::CPU, 0, "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz"},
            {cyxwiz::DeviceType::CUDA, 0, "NVIDIA GeForce GTX 1050 Ti"},
            {cyxwiz::DeviceType::ONEAPI, 0, "Intel(R) UHD Graphics 630"}};
}

std::vector<KnownDevice> Dell() {
    return {{cyxwiz::DeviceType::CPU, 0, "11th Gen Intel(R) Core(TM) i7"},
            {cyxwiz::DeviceType::ONEAPI, 0, "Intel(R) Iris(R) Xe Graphics"}};
}

}  // namespace

TEST_CASE("backend device types map to allocation types", "[allocation]") {
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::CPU) == kAllocCpu);
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::CUDA) == kAllocCuda);
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::OPENCL) == kAllocOpenCl);
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::ONEAPI) == kAllocOpenCl);  // until the protocol has oneAPI
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::METAL) == kAllocMetal);
    CHECK(AllocationTypeOf(cyxwiz::DeviceType::VULKAN) == kAllocNone);
}

TEST_CASE("a lent GPU is resolved by name, then by position", "[allocation]") {
    CHECK(ResolveGpuAllocationType("NVIDIA GeForce GTX 1050 Ti", 0, ThisPc()) == kAllocCuda);
    CHECK(ResolveGpuAllocationType("Intel(R) UHD Graphics 630", 1, ThisPc()) == kAllocOpenCl);
    CHECK(ResolveGpuAllocationType("Intel(R) Iris(R) Xe Graphics", 0, Dell()) == kAllocOpenCl);

    SECTION("an unknown name falls back to the GPU at the same position") {
        CHECK(ResolveGpuAllocationType("GPU 0", 0, ThisPc()) == kAllocCuda);
        CHECK(ResolveGpuAllocationType("GPU 1", 1, ThisPc()) == kAllocOpenCl);
        CHECK(ResolveGpuAllocationType("GPU 0", 0, Dell()) == kAllocOpenCl);
    }
    SECTION("nothing known keeps the old behaviour") {
        CHECK(ResolveGpuAllocationType("GPU 0", 0, {}) == kAllocCuda);
        CHECK(ResolveGpuAllocationType("GPU 5", 5, Dell()) == kAllocCuda);
    }
    SECTION("the CPU never answers for a GPU") {
        CHECK(ResolveGpuAllocationType("Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz", 0, ThisPc()) == kAllocCuda);
    }
}
