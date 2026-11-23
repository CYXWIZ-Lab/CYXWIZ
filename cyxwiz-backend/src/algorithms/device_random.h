#pragma once

// Seeded ArrayFire draws that are the same on every device (TOFIX205). Probed
// with scripts/arrayfire_probes/random_engine_portability (ArrayFire 3.10):
// Philox randu is bit-identical on CPU, CUDA and OpenCL and on Windows and
// Linux; Threefry and Mersenne differ between CPU and GPU, and randn differs
// everywhere. So every engine here is Philox, and normal values are made from
// Philox uniforms on the device (Box-Muller), equal to float rounding.

#include <arrayfire.h>

#include <limits>

namespace cyxwiz {

// A Philox engine seeded with `seed`.
inline af::randomEngine DeviceRandomEngine(unsigned long long seed) {
    return af::randomEngine(AF_RANDOM_ENGINE_PHILOX_4X32_10, seed);
}

// Box-Muller on two uniform arrays. randu returns exact 0 and exact 1 (in
// 2^28 Philox f32 draws on the CPU: 8 zeros, 13 ones), so 1 - u is kept above
// 0 and log never sees 0: a value is at most about 13 (f32) from the mean.
inline af::array NormalFromUniforms(const af::array& u_first, const af::array& u_second) {
    const double tiny = u_first.type() == f64 ? (std::numeric_limits<double>::min)()
                                              : static_cast<double>((std::numeric_limits<float>::min)());
    const af::array u1 = (af::max)(1.0 - u_first, tiny);  // (0, 1]
    return af::sqrt(-2.0 * af::log(u1)) * af::cos(6.283185307179586 * u_second);
}

// Standard normal values of `type` (f32 or f64) from two uniform draws of
// `dims`; in place of af::randn, which differs between devices.
inline af::array DeviceNormal(const af::dim4& dims, af::dtype type, af::randomEngine& engine) {
    const af::dtype work = type == f64 ? f64 : f32;
    const af::array u_first = af::randu(dims, work, engine);
    const af::array u_second = af::randu(dims, work, engine);
    const af::array z = NormalFromUniforms(u_first, u_second);
    return work == type ? z : z.as(type);
}

// The same on the process's default engine (Philox, seeded by
// SeedCurrentArrayFireRandomEngine).
inline af::array DeviceNormal(const af::dim4& dims, af::dtype type) {
    af::randomEngine engine = af::getDefaultRandomEngine();
    return DeviceNormal(dims, type, engine);
}

}  // namespace cyxwiz
