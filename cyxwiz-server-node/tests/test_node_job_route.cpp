// The route a node trains on comes from what the process chose, on any thread
// (found 2026-11-11: a gRPC thread reported ArrayFire's per-thread default,
// oneAPI, while the daemon had activated and verified OpenCL).
#include "node_job_route.h"

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <thread>

using cyxwiz::Device;
using cyxwiz::DeviceType;
using cyxwiz::ProcessDeviceSelection;
using cyxwiz::servernode::NodeJobRoute;

namespace {

std::optional<ProcessDeviceSelection> RouteOnANewThread() {
    std::optional<ProcessDeviceSelection> route;
    std::thread worker([&route] { route = NodeJobRoute(); });
    worker.join();
    return route;
}

}  // namespace

TEST_CASE("the job route is the process's device on every thread", "[node_job_route]") {
    // The daemon activated OpenCL at startup; no machine preference is saved.
    Device::RecordProcessDevice(DeviceType::OPENCL, 0);
    REQUIRE_FALSE(cyxwiz::GetSavedExecutionDeviceSelection().has_value());

    const auto here = NodeJobRoute();
    REQUIRE(here.has_value());
    CHECK(here->type == DeviceType::OPENCL);
    CHECK(here->device_id == 0);

    // A thread that never activated a backend gets the same route.
    const auto there = RouteOnANewThread();
    REQUIRE(there.has_value());
    CHECK(there->type == DeviceType::OPENCL);
    CHECK(there->device_id == 0);

    // A saved machine preference wins.
    cyxwiz::CommitExecutionDeviceSelectionState({DeviceType::CPU, 0, ""});
    const auto preferred = RouteOnANewThread();
    REQUIRE(preferred.has_value());
    CHECK(preferred->type == DeviceType::CPU);
    CHECK(preferred->device_id == 0);
}
