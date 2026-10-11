#pragma once

// Host values in and out of ArrayFire for the host-vector analysis APIs (linear
// algebra, data transforms): values cross to the active device once, in float64
// where the device has it and float32 otherwise, and come back as double.

#ifdef CYXWIZ_HAS_ARRAYFIRE

#include "arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <algorithm>
#include <string>
#include <vector>

namespace cyxwiz {

inline bool DeviceHasDouble() {
    return af::isDoubleAvailable(af::getDevice());
}

// `values` as a device column vector in the compute precision.
inline af::array HostValuesToDevice(const std::vector<double>& values) {
    const dim_t count = static_cast<dim_t>(values.size());
    if (DeviceHasDouble()) return af::array(count, values.data());
    std::vector<float> narrow(values.size());
    std::transform(values.begin(), values.end(), narrow.begin(), [](double v) { return static_cast<float>(v); });
    return af::array(count, narrow.data());
}

// Every element of `source`, column-major, as double.
inline std::vector<double> DeviceToHostValues(const af::array& source, const char* operation) {
    std::vector<double> values(static_cast<size_t>(source.elements()));
    if (values.empty()) return values;
    if (source.type() == f64) {
        af::array data = source;
        data.eval();
        MaterializeArrayFireToHost(data, values.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   operation, "arrayfire_column_major");
        return values;
    }
    af::array data = source.as(f32);
    data.eval();
    std::vector<float> narrow(values.size());
    MaterializeArrayFireToHost(data, narrow.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                               operation, "arrayfire_column_major");
    std::copy(narrow.begin(), narrow.end(), values.begin());
    return values;
}

inline std::string DeviceError(const char* operation, const af::exception& e) {
    return std::string(operation) + " failed on the ArrayFire device: " + e.what();
}

}  // namespace cyxwiz

#endif
