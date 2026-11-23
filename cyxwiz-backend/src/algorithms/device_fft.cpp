#include "device_fft.h"

#include <cmath>
#include <vector>

namespace cyxwiz {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool SmallPrimeFactorsOnly(dim_t n) {
    for (const dim_t p : {2, 3, 5, 7})
        while (n > 1 && n % p == 0) n /= p;
    return n == 1;
}

dim_t PowerOfTwoAtLeast(dim_t n) {
    dim_t p = 1;
    while (p < n) p *= 2;
    return p;
}

bool DoubleDevice() {
    return af::isDoubleAvailable(af::getDevice());
}

// exp(-i pi n^2 / N) for n = 0..count-1, n^2 taken mod 2N so the angle stays
// exact for long signals.
af::array Chirp(dim_t n_total, dim_t count) {
    std::vector<double> re(static_cast<size_t>(count)), im(static_cast<size_t>(count));
    const long long two_n = 2LL * static_cast<long long>(n_total);
    for (dim_t n = 0; n < count; ++n) {
        const long long k = static_cast<long long>(n);
        const long long sq = (k % two_n) * (k % two_n) % two_n;
        const double angle = -kPi * static_cast<double>(sq) / static_cast<double>(n_total);
        re[static_cast<size_t>(n)] = std::cos(angle);
        im[static_cast<size_t>(n)] = std::sin(angle);
    }
    if (DoubleDevice()) return af::complex(af::array(count, re.data()), af::array(count, im.data()));
    // A float32-only device (Intel Iris Xe) cannot hold a double array.
    std::vector<float> re32(re.size()), im32(im.size());
    for (size_t i = 0; i < re.size(); ++i) {
        re32[i] = static_cast<float>(re[i]);
        im32[i] = static_cast<float>(im[i]);
    }
    return af::complex(af::array(count, re32.data()), af::array(count, im32.data()));
}

// The DFT along dimension 0 by Bluestein's identity
//   X_k = w_k * sum_n (x_n w_n) conj(w_{k-n}),  w_n = exp(-i pi n^2 / N),
// the sum a circular convolution of length M >= 2N - 1 (a power of two).
af::array Bluestein(const af::array& signal) {
    const dim_t n = signal.dims(0);
    const dim_t columns = signal.elements() / n;
    const dim_t m = PowerOfTwoAtLeast(2 * n - 1);
    const af::dtype complex_type = DoubleDevice() ? c64 : c32;
    const af::array x = af::moddims(signal, n, columns).as(complex_type);
    const af::array w = Chirp(n, n);

    af::array a = af::constant(0, m, columns, complex_type);
    a(af::seq(0, static_cast<double>(n - 1)), af::span) = x * af::tile(w, 1, static_cast<unsigned>(columns));

    af::array b = af::constant(0, m, complex_type);
    const af::array conj_w = af::conjg(w);
    b(af::seq(0, static_cast<double>(n - 1))) = conj_w;
    if (n > 1) {
        // b_{M-j} = conj(w_j), j = 1..N-1
        b(af::seq(static_cast<double>(m - n + 1), static_cast<double>(m - 1))) =
            af::flip(conj_w(af::seq(1, static_cast<double>(n - 1))), 0);
    }

    const af::array product = af::fft(a) * af::tile(af::fft(b), 1, static_cast<unsigned>(columns));
    const af::array convolution = af::ifftNorm(product, 1.0 / static_cast<double>(m));
    af::array out = convolution(af::seq(0, static_cast<double>(n - 1)), af::span) *
                    af::tile(w, 1, static_cast<unsigned>(columns));
    return af::moddims(out, signal.dims());
}

}  // namespace

bool NativeFftLength(dim_t length) {
    return af::getActiveBackend() != AF_BACKEND_OPENCL || SmallPrimeFactorsOnly(length);
}

af::array DeviceFft(const af::array& signal) {
    if (NativeFftLength(signal.dims(0))) return af::fft(signal);
    return Bluestein(signal);
}

af::array DeviceIfftNorm(const af::array& spectrum, double scale) {
    if (NativeFftLength(spectrum.dims(0))) return af::ifftNorm(spectrum, scale);
    // ifft(X) = conj(fft(conj(X))), then the caller's scale.
    return af::conjg(Bluestein(af::conjg(spectrum))) * scale;
}

af::array DeviceFft2(const af::array& image) {
    if (NativeFftLength(image.dims(0)) && NativeFftLength(image.dims(1))) return af::fft2(image);
    // Along the columns, then along the rows.
    return af::transpose(DeviceFft(af::transpose(DeviceFft(image))));
}

af::array DeviceIfft2Norm(const af::array& spectrum, double scale) {
    if (NativeFftLength(spectrum.dims(0)) && NativeFftLength(spectrum.dims(1)))
        return af::ifft2Norm(spectrum, scale);
    return af::conjg(DeviceFft2(af::conjg(spectrum))) * scale;
}

} // namespace cyxwiz
