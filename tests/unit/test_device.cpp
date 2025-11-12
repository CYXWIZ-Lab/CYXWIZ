#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/device.h>

TEST_CASE("Device enumeration", "[device]") {
    auto devices = cyxwiz::Device::GetAvailableDevices();
    REQUIRE(devices.size() >= 1); // At least CPU
}

// TODO: Add more tests
