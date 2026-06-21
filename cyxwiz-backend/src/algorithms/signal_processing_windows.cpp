#include <cyxwiz/signal_processing.h>

#include <cmath>

namespace cyxwiz {
namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
} // namespace

// ============================================================================
// Window Functions
// ============================================================================

std::vector<double> SignalProcessing::HammingWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.54 - 0.46 * std::cos(TWO_PI * i / (size - 1));
    }
    return window;
}

std::vector<double> SignalProcessing::HannWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1.0 - std::cos(TWO_PI * i / (size - 1)));
    }
    return window;
}

std::vector<double> SignalProcessing::BlackmanWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.42 - 0.5 * std::cos(TWO_PI * i / (size - 1))
                    + 0.08 * std::cos(4.0 * PI * i / (size - 1));
    }
    return window;
}

std::vector<double> SignalProcessing::RectangularWindow(int size) {
    return std::vector<double>(size, 1.0);
}

} // namespace cyxwiz