#ifdef _WIN32
#define NOMINMAX
#endif

#include <cyxwiz/signal_processing.h>

#include "signal_device.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

using namespace signal_detail;

// ============================================================================
// Wavelet Transform: pywt.wavedec / waverec, mode "symmetric", on the
// ArrayFire device (TOFIX129 D7). The index rules are asserted against
// PyWavelets in tests/computation_truth/reference/generate_signal_fixtures.py.
// ============================================================================

namespace {

// pywt's symmetric extension by `pad` values on each side: ... x2 x1 x0 | x0
// x1 ... xN-1 | xN-1 xN-2 ..., reflecting again past the ends (numpy.pad
// "symmetric"). A gather by index on the device.
af::array SymmetricExtend(const af::array& x, dim_t pad) {
    const dim_t n = x.dims(0);
    std::vector<unsigned> index(static_cast<size_t>(n + 2 * pad));
    const dim_t period = 2 * n;
    for (dim_t p = -pad; p < n + pad; ++p) {
        dim_t q = ((p % period) + period) % period;
        if (q >= n) q = period - 1 - q;
        index[static_cast<size_t>(p + pad)] = static_cast<unsigned>(q);
    }
    return af::lookup(x, af::array(static_cast<dim_t>(index.size()), index.data()), 0);
}

// One analysis level: the extended signal convolved with each filter, every
// other sample from F on; (n + F - 1) / 2 coefficients each.
void Analyse(const af::array& x, const af::array& dec_lo, const af::array& dec_hi, af::array& approx, af::array& detail) {
    const dim_t n = x.dims(0), f = dec_lo.dims(0);
    const dim_t out_len = (n + f - 1) / 2;
    const af::array ext = SymmetricExtend(x, f - 1);
    const af::seq pick(static_cast<double>(f), static_cast<double>(f + 2 * (out_len - 1)), 2.0);
    approx = af::convolve1(ext, dec_lo, AF_CONV_EXPAND)(pick);
    detail = af::convolve1(ext, dec_hi, AF_CONV_EXPAND)(pick);
}

// One synthesis level: the coefficients with zeros between them, convolved
// with the reconstruction filters, added; 2 n - F + 2 samples from F - 2 on.
af::array Synthesise(const af::array& approx, const af::array& detail, const af::array& rec_lo, const af::array& rec_hi) {
    const dim_t n = approx.dims(0), f = rec_lo.dims(0);
    af::array up_a = af::constant(0.0, 2 * n, approx.type());
    af::array up_d = af::constant(0.0, 2 * n, detail.type());
    const af::seq even(0.0, static_cast<double>(2 * n - 2), 2.0);
    up_a(even) = approx;
    up_d(even) = detail;
    const af::array rec = af::convolve1(up_a, rec_lo, AF_CONV_EXPAND) + af::convolve1(up_d, rec_hi, AF_CONV_EXPAND);
    const dim_t rec_len = 2 * n - f + 2;
    return rec(af::seq(static_cast<double>(f - 2), static_cast<double>(f - 2 + rec_len - 1)));
}

}  // namespace

WaveletResult SignalProcessing::DWT(const std::vector<double>& signal, const std::string& wavelet, int levels) {
    WaveletResult result;
    result.wavelet_name = wavelet;
    result.error_message = Refusal(signal, 2, "DWT");
    if (!result.error_message.empty()) return result;
    std::vector<double> dec_lo, dec_hi, rec_lo, rec_hi;
    if (!GetWaveletFilters(wavelet, dec_lo, dec_hi, rec_lo, rec_hi)) {
        result.error_message = "DWT: unknown wavelet '" + wavelet + "' (haar, db1..db8, sym2..sym8, coif1..coif5)";
        return result;
    }
    const int n = static_cast<int>(signal.size());
    result.original_size = n;
    result.max_levels = WaveletMaxLevel(n, wavelet);
    if (levels < 1) {
        result.error_message = "DWT: levels must be at least 1";
        return result;
    }
    if (levels > result.max_levels) {
        result.error_message = "DWT: " + std::to_string(n) + " samples give " + wavelet + " at most " +
                               std::to_string(result.max_levels) + " level" + (result.max_levels == 1 ? "" : "s") + " (" +
                               std::to_string(levels) + " asked)";
        return result;
    }
    try {
        const af::array lo = ToDevice(dec_lo), hi = ToDevice(dec_hi);
        af::array approx = ToDevice(signal);
        result.details.resize(static_cast<size_t>(levels));
        for (int level = 0; level < levels; ++level) {  // sequential by level: each reads the one before
            af::array next, detail;
            Analyse(approx, lo, hi, next, detail);
            result.details[static_cast<size_t>(level)] = ToHost(detail, "SignalProcessing::DWT");
            approx = next;
        }
        result.approximation = ToHost(approx, "SignalProcessing::DWT");
        result.levels = levels;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::DWT", e);
    }
    return result;
}

std::vector<double> SignalProcessing::IDWT(const WaveletResult& coeffs) {
    if (!coeffs.success || coeffs.approximation.empty() || coeffs.levels < 1 ||
        coeffs.details.size() != static_cast<size_t>(coeffs.levels))
        return {};
    std::vector<double> dec_lo, dec_hi, rec_lo, rec_hi;
    if (!GetWaveletFilters(coeffs.wavelet_name, dec_lo, dec_hi, rec_lo, rec_hi)) return {};
    try {
        const af::array lo = ToDevice(rec_lo), hi = ToDevice(rec_hi);
        af::array approx = ToDevice(coeffs.approximation);
        for (int level = coeffs.levels - 1; level >= 0; --level) {
            const std::vector<double>& detail = coeffs.details[static_cast<size_t>(level)];
            // pywt.waverec: a rebuilt level one sample longer than the next
            // details (an odd length below) loses its last sample first.
            if (approx.dims(0) == static_cast<dim_t>(detail.size()) + 1)
                approx = approx(af::seq(0.0, static_cast<double>(detail.size() - 1)));
            if (approx.dims(0) != static_cast<dim_t>(detail.size())) return {};
            approx = Synthesise(approx, ToDevice(detail), lo, hi);
        }
        std::vector<double> out = ToHost(approx, "SignalProcessing::IDWT");
        if (coeffs.original_size > 0 && static_cast<int>(out.size()) > coeffs.original_size) out.resize(static_cast<size_t>(coeffs.original_size));
        return out;
    } catch (const af::exception& e) {
        static_cast<void>(DeviceError("SignalProcessing::IDWT", e));  // the vector API has no place for the message
        return {};
    }
}

DenoiseResult SignalProcessing::WaveletDenoise(const std::vector<double>& signal, const std::string& wavelet, int levels,
                                               double threshold, bool soft) {
    DenoiseResult result;
    WaveletResult coeffs = DWT(signal, wavelet, levels);
    if (!coeffs.success) {
        result.error_message = coeffs.error_message;
        return result;
    }
    try {
        // The noise level from the finest details (Donoho and Johnstone): median |cD_1| / 0.6745.
        const af::array finest = ToDevice(coeffs.details[0]);
        result.sigma = af::median<double>(af::abs(finest)) / 0.6745;
        result.threshold = threshold > 0.0 ? threshold : result.sigma * std::sqrt(2.0 * std::log(static_cast<double>(signal.size())));
        const double t = result.threshold;
        for (auto& detail : coeffs.details) {
            const af::array d = ToDevice(detail);
            const af::array magnitude = af::abs(d);
            // Soft: shrink towards zero by t (pywt.threshold "soft"); hard: drop what is at or under t.
            // (af::sign is 1 for negative values and 0 otherwise, so the sign is a select.)
            const af::array signs = af::select(d < 0.0, af::constant(-1.0, d.dims(), d.type()), af::constant(1.0, d.dims(), d.type()));
            const af::array kept = soft ? signs * af::max(magnitude - t, 0.0) : af::select(magnitude > t, d, 0.0);
            detail = ToHost(kept, "SignalProcessing::WaveletDenoise");
        }
        result.denoised = IDWT(coeffs);
        if (result.denoised.empty()) {
            result.error_message = "WaveletDenoise: the signal could not be rebuilt";
            return result;
        }
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::WaveletDenoise", e);
    }
    return result;
}

} // namespace cyxwiz
