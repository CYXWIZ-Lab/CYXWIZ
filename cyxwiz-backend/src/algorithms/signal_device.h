#pragma once

// The signal backend's device helpers (TOFIX129 D7): values cross to the
// ArrayFire device once, in float64 where the device has it and float32
// otherwise, the arithmetic happens there, and the answers come back as
// double. Shared by the signal_processing_*.cpp files only.

#include "arrayfire_host_values.h"

#include <arrayfire.h>

#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace cyxwiz::signal_detail {

inline af::dtype ComputeType() { return DeviceHasDouble() ? f64 : f32; }
inline af::dtype ComplexType() { return DeviceHasDouble() ? c64 : c32; }

inline af::array ToDevice(const std::vector<double>& values) { return HostValuesToDevice(values).as(ComputeType()); }

inline std::vector<double> ToHost(const af::array& values, const char* operation) { return DeviceToHostValues(values, operation); }

// A complex device array, element by element, as std::complex<double>.
inline std::vector<std::complex<double>> ComplexToHost(const af::array& values, const char* operation) {
    const std::vector<double> re = ToHost(af::real(values), operation), im = ToHost(af::imag(values), operation);
    std::vector<std::complex<double>> out(re.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = {re[i], i < im.size() ? im[i] : 0.0};
    return out;
}

inline af::array ComplexToDevice(const std::vector<std::complex<double>>& values) {
    std::vector<double> re(values.size()), im(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        re[i] = values[i].real();
        im[i] = values[i].imag();
    }
    return af::complex(ToDevice(re), ToDevice(im));
}

// "" when the signal can be worked on, else why not (too short, a missing
// or infinite value).
inline std::string Refusal(const std::vector<double>& data, size_t min_n, const char* what) {
    if (data.size() < min_n)
        return std::string(what) + " needs at least " + std::to_string(min_n) + " values (" + std::to_string(data.size()) + " given)";
    for (const double v : data)
        if (!std::isfinite(v)) return std::string(what) + ": the signal has a missing or infinite value";
    return "";
}

inline double SumOf(const std::vector<double>& values) {
    double s = 0.0;
    for (const double v : values) s += v;
    return s;
}

inline double SumOfSquares(const std::vector<double>& values) {
    double s = 0.0;
    for (const double v : values) s += v * v;
    return s;
}

}  // namespace cyxwiz::signal_detail
