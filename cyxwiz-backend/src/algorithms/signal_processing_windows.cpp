#include <cyxwiz/signal_processing.h>

#include <cmath>
#include <string>
#include <vector>

namespace cyxwiz {
namespace {
constexpr double PI = 3.14159265358979323846;

// A generalised cosine window of `size` points, symmetric (size - 1 in the
// denominator); a single point is 1. These are coefficient tables, not data
// arithmetic, so they are made on the host.
std::vector<double> Cosine(int size, double a0, double a1, double a2) {
    if (size <= 0) return {};
    if (size == 1) return {1.0};
    std::vector<double> window(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        const double phase = 2.0 * PI * i / (size - 1);
        window[static_cast<size_t>(i)] = a0 - a1 * std::cos(phase) + a2 * std::cos(2.0 * phase);
    }
    return window;
}
}  // namespace

// ============================================================================
// Window Functions (scipy.signal.get_window)
// ============================================================================

std::vector<double> SignalProcessing::Window(const std::string& name, int size, bool periodic) {
    if (size <= 0) return {};
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    if (name == "hann" || name == "hanning") {
        a0 = 0.5; a1 = 0.5;
    } else if (name == "hamming") {
        a0 = 0.54; a1 = 0.46;
    } else if (name == "blackman") {
        a0 = 0.42; a1 = 0.5; a2 = 0.08;
    } else if (name == "boxcar" || name == "rectangular" || name == "rect") {
        return std::vector<double>(static_cast<size_t>(size), 1.0);
    } else {
        return {};
    }
    // scipy fftbins=True: the symmetric window of size + 1 points without its last one.
    if (periodic && size > 1) {
        std::vector<double> longer = Cosine(size + 1, a0, a1, a2);
        longer.pop_back();
        return longer;
    }
    return Cosine(size, a0, a1, a2);
}

std::vector<double> SignalProcessing::HammingWindow(int size) { return Window("hamming", size, false); }

std::vector<double> SignalProcessing::HannWindow(int size) { return Window("hann", size, false); }

std::vector<double> SignalProcessing::BlackmanWindow(int size) { return Window("blackman", size, false); }

std::vector<double> SignalProcessing::RectangularWindow(int size) { return Window("boxcar", size, false); }

} // namespace cyxwiz
