#pragma once

#include "../src/core/route_qualification_snapshot.h"

#include <cyxwiz/device.h>

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::test {

inline RouteQualificationSnapshot MakeQualifiedRouteSnapshot(
    const std::vector<DeviceInfo>& inventory,
    std::string matrix_id = "test-qualified-routes") {
    RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = std::move(matrix_id);
    snapshot.pack_id = "test-runtime";
    snapshot.routes.reserve(inventory.size());
    for (const auto& device : inventory) {
        RouteQualificationRecord record;
        record.type = device.type;
        record.device_id = device.device_id;
        if (device.physical_fingerprint_known) {
            record.physical_fingerprint = device.physical_fingerprint;
        }
        if (device.provider_known) {
            record.provider = device.provider;
        }
        if (device.driver_version_known) {
            record.driver_version = device.driver_version;
        }
        record.operation_count = kRouteQualificationOperationCount;
        record.pass_count = kRouteQualificationOperationCount;
        record.certified = true;
        snapshot.routes.push_back(std::move(record));
    }
    return snapshot;
}

inline void InstallQualifiedRouteSnapshot(
    const std::vector<DeviceInfo>& inventory = Device::GetAvailableDevices()) {
    InstallRouteQualificationSnapshot(
        MakeQualifiedRouteSnapshot(inventory));
}

}  // namespace cyxwiz::test
