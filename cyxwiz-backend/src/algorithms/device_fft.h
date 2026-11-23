#pragma once

// FFTs of any length on every ArrayFire backend (ORA-13 / ORA-14).
//
// ArrayFire's OpenCL backend runs clFFT, which plans only lengths whose prime
// factors are small: a prime length such as 257 throws "Invalid input
// argument" (202), and some mixed lengths give wrong values. On OpenCL a
// length with a prime factor above 7 goes through Bluestein's algorithm (the
// chirp-z transform), which computes the same DFT with power-of-two FFTs on
// the same device. Every other backend, and every other length, calls
// ArrayFire's FFT directly.
//
// Semantics are ArrayFire's: along dimension 0 (each column of a matrix is
// one signal), the forward transform unnormalised, the inverse scaled by the
// given factor.

#include <arrayfire.h>

namespace cyxwiz {

af::array DeviceFft(const af::array& signal);
af::array DeviceIfftNorm(const af::array& spectrum, double scale);
af::array DeviceFft2(const af::array& image);
af::array DeviceIfft2Norm(const af::array& spectrum, double scale);

// Whether ArrayFire's own FFT plans this length on the active backend.
bool NativeFftLength(dim_t length);

} // namespace cyxwiz
