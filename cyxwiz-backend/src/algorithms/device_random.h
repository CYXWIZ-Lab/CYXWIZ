#pragma once

// Seeded ArrayFire draws that are the same on every device (TOFIX205). Probed
// with scripts/arrayfire_probes/random_engine_portability (ArrayFire 3.10):
// Philox randu is bit-identical on CPU, CUDA and OpenCL and on Windows and
// Linux; Threefry and Mersenne differ between CPU and GPU, and randn differs
// everywhere. So every engine here is Philox, and normal values are made from
// Philox uniforms on the device (Box-Muller), equal to float rounding.

#include <arrayfire.h>

namespace cyxwiz {

// A Philox engine seeded with `seed`.
inline af::randomEngine DeviceRandomEngine(unsigned long long seed) {
    return af::randomEngine(AF_RANDOM_ENGINE_PHILOX_4X32_10, seed);
}

// Standard normal values of `type` (f32 or f64) by Box-Muller on two uniform
// draws of `dims`; in place of af::randn, which differs between devices.
inline af::array DeviceNormal(const af::dim4& dims, af::dtype type, af::randomEngine& engine) {
    const af::dtype work = type == f64 ? f64 : f32;
    const af::array u1 = 1.0 - af::randu(dims, work, engine);  // (0, 1]
    const af::array u2 = af::randu(dims, work, engine);
    af::array z = af::sqrt(-2.0 * af::log(u1)) * af::cos(6.283185307179586 * u2);
    return work == type ? z : z.as(type);
}

// The same on the process's default engine (Philox, seeded by
// SeedCurrentArrayFireRandomEngine).
inline af::array DeviceNormal(const af::dim4& dims, af::dtype type) {
    af::randomEngine engine = af::getDefaultRandomEngine();
    return DeviceNormal(dims, type, engine);
}

}  // namespace cyxwiz
