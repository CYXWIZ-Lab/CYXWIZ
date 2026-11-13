// cyxwiz::SignalProcessing against numpy.fft, scipy.signal and PyWavelets
// (TOFIX129 D7): fixtures/signal_scipy.json from
// reference/generate_signal_fixtures.py. Every function the Signal tab and the
// signal nodes use: FFT and its inverse, the one-sided spectrum, Welch, the
// spectrogram, windows, firwin filters with their response and zero-phase
// apply, numpy.convolve's three modes, wavedec / waverec, denoising at the
// universal threshold, and find_peaks by prominence and distance.

#include "test_device_selection.h"

#include <cyxwiz/signal_processing.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;
using cyxwiz::SignalProcessing;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double actual, double expected, double tolerance, const std::string& what) {
    Check(std::fabs(actual - expected) <= tolerance * (1.0 + std::fabs(expected)),
          what + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

std::vector<double> Doubles(const json& j) {
    std::vector<double> out;
    for (const auto& v : j) out.push_back(v.get<double>());
    return out;
}

void NearAll(const std::vector<double>& actual, const json& expected, double tolerance, const std::string& what) {
    const std::vector<double> e = Doubles(expected);
    Check(actual.size() == e.size(), what + ": size " + std::to_string(actual.size()) + " vs " + std::to_string(e.size()));
    for (size_t i = 0; i < e.size(); ++i) Near(actual[i], e[i], tolerance, what + "[" + std::to_string(i) + "]");
}

// Phases agree modulo 2 pi, and mean nothing where the magnitude is nil.
void NearPhases(const std::vector<double>& actual, const json& expected, const std::vector<double>& magnitude, double tolerance,
                const std::string& what) {
    const std::vector<double> e = Doubles(expected);
    Check(actual.size() == e.size(), what + ": size");
    double largest = 0.0;
    for (const double m : magnitude)
        if (m > largest) largest = m;
    for (size_t i = 0; i < e.size(); ++i) {
        if (magnitude[i] < 1e-6 * largest) continue;
        double d = std::fabs(actual[i] - e[i]);
        d = std::fmin(d, std::fabs(d - 2.0 * 3.14159265358979323846));
        Check(d <= tolerance, what + "[" + std::to_string(i) + "]: " + std::to_string(actual[i]) + " vs " + std::to_string(e[i]));
    }
}

double Tol(double f64, double f32) { return cyxwiz::test::DeviceTolerance(f64, f32); }

std::map<std::string, std::vector<double>> g_signals;

const std::vector<double>& Signal(const json& c) {
    Check(c.contains("signal"), "case " + c.value("op", std::string()) + "/" + c.value("label", std::string()) + " names no signal");
    const std::string name = c.at("signal").get<std::string>();
    Check(g_signals.count(name) == 1, "unknown signal " + name);
    return g_signals[name];
}

void RunCase(const json& c) {
    const std::string op = c.at("op").get<std::string>();
    const std::string name = op + "/" + c.at("label").get<std::string>();
    if (std::getenv("CYXWIZ_TRACE_CASES")) std::cerr << "case " << name << std::endl;  // set it to see which case a crash is in
    if (op == "fft") {
        const auto r = SignalProcessing::FFT(Signal(c), c["fs"].get<double>());
        Check(r.success, name + ": " + r.error_message);
        NearAll(r.magnitude, c["magnitude"], Tol(1e-9, 1e-4), name + " magnitude");
        NearPhases(r.phase, c["phase"], r.magnitude, Tol(1e-7, 1e-2), name + " phase");
        NearAll(r.frequencies, c["frequencies"], 1e-12, name + " frequencies");
        return;
    }
    if (op == "ifft") {
        const std::vector<double> re = Doubles(c["real"]), im = Doubles(c["imag"]);
        std::vector<std::complex<double>> spectrum(re.size());
        for (size_t i = 0; i < re.size(); ++i) spectrum[i] = {re[i], im[i]};
        NearAll(SignalProcessing::IFFT(spectrum), json(Signal(c)), Tol(1e-9, 1e-4), name);
        return;
    }
    if (op == "irfft") {
        const std::vector<double> re = Doubles(c["real"]), im = Doubles(c["imag"]);
        std::vector<std::complex<double>> half(re.size());
        for (size_t i = 0; i < re.size(); ++i) half[i] = {re[i], im[i]};
        NearAll(SignalProcessing::InverseSpectrum(half, c["samples"].get<int>()), c["y"], Tol(1e-9, 1e-4), name);
        // Spectrum's own one-sided output goes back the same way.
        const auto sp = SignalProcessing::Spectrum(Signal(c), 1.0, "hann", true);
        Check(sp.success && sp.complex_output.size() == half.size(), name + ": Spectrum complex output");
        NearAll(SignalProcessing::InverseSpectrum(sp.complex_output, sp.n), c["y"], Tol(1e-9, 1e-4), name + " from Spectrum");
        Check(SignalProcessing::InverseSpectrum(half, static_cast<int>(re.size()) * 3).empty(), name + ": a sample count that does not fit is refused");
        return;
    }
    if (op == "spectrum") {
        const auto r = SignalProcessing::Spectrum(Signal(c), c["fs"].get<double>(), "hann", true);
        Check(r.success, name + ": " + r.error_message);
        NearAll(r.frequencies, c["frequencies"], 1e-12, name + " frequencies");
        NearAll(r.amplitude, c["amplitude"], Tol(1e-9, 1e-4), name + " amplitude");
        NearAll(r.psd, c["psd"], Tol(1e-9, 1e-4), name + " psd");
        return;
    }
    if (op == "welch") {
        const auto r = SignalProcessing::Welch(Signal(c), c["fs"].get<double>(), c["nperseg"].get<int>(), "hann");
        Check(r.success, name + ": " + r.error_message);
        NearAll(r.frequencies, c["frequencies"], 1e-12, name + " frequencies");
        NearAll(r.psd, c["psd"], Tol(1e-9, 1e-4), name + " psd");
        return;
    }
    if (op == "spectrogram") {
        const auto r = SignalProcessing::ComputeSpectrogram(Signal(c), c["nperseg"].get<int>(), c["hop"].get<int>(), c["fs"].get<double>(), "hann");
        Check(r.success, name + ": " + r.error_message);
        NearAll(r.frequencies, c["frequencies"], 1e-12, name + " frequencies");
        NearAll(r.times, c["times"], 1e-12, name + " times");
        const json& sxx = c["sxx"];
        Check(r.spectrogram.size() == sxx.size() && r.num_frames == static_cast<int>(sxx.size()), name + ": frames");
        for (size_t f = 0; f < sxx.size(); ++f) NearAll(r.spectrogram[f], sxx[f], Tol(1e-9, 1e-4), name + " frame " + std::to_string(f));
        return;
    }
    if (op == "window") {
        const std::string w = c["name"].get<std::string>();
        const int size = c["size"].get<int>();
        NearAll(SignalProcessing::Window(w, size, true), c["periodic"], 1e-12, name + " periodic");
        NearAll(SignalProcessing::Window(w, size, false), c["symmetric"], 1e-12, name + " symmetric");
        return;
    }
    if (op == "firwin") {
        const std::vector<double> cutoff = Doubles(c["cutoff"]);
        const bool pass_zero = c["pass_zero"].get<bool>();
        const double fs = c["fs"].get<double>();
        const int taps = c["taps"].get<int>();
        cyxwiz::FilterCoefficients f;
        if (cutoff.size() == 1) f = pass_zero ? SignalProcessing::DesignLowpass(cutoff[0], fs, taps) : SignalProcessing::DesignHighpass(cutoff[0], fs, taps);
        else f = pass_zero ? SignalProcessing::DesignBandstop(cutoff[0], cutoff[1], fs, taps) : SignalProcessing::DesignBandpass(cutoff[0], cutoff[1], fs, taps);
        Check(f.success, name + ": " + f.error_message);
        Check(f.order == taps && static_cast<int>(f.b.size()) == taps, name + ": taps mean taps");
        NearAll(f.b, c["b"], 1e-12, name + " b");
        NearAll(f.freq_response_mag, c["response_mag"], Tol(1e-9, 1e-4), name + " |H|");
        NearPhases(f.freq_response_phase, c["response_phase"], f.freq_response_mag, Tol(1e-7, 1e-2), name + " angle H");
        NearAll(SignalProcessing::ApplyFilter(Signal(c), f), c["filtered"], Tol(1e-9, 1e-4), name + " filtered");
        return;
    }
    if (op == "convolve") {
        const auto r = SignalProcessing::Convolve1D(Doubles(c["x"]), Doubles(c["h"]), c["mode"].get<std::string>());
        Check(r.success, name + ": " + r.error_message);
        NearAll(r.output, c["y"], Tol(1e-10, 1e-5), name);
        return;
    }
    if (op == "dwt") {
        const std::string wavelet = c["wavelet"].get<std::string>();
        const int level = c["level"].get<int>();
        const auto r = SignalProcessing::DWT(Signal(c), wavelet, level);
        Check(r.success, name + ": " + r.error_message);
        Check(r.max_levels == c["max_level"].get<int>(), name + ": max level " + std::to_string(r.max_levels));
        NearAll(r.approximation, c["approximation"], Tol(1e-9, 1e-4), name + " cA");
        const json& details = c["details"];  // coarsest first (pywt), ours finest first
        Check(details.size() == static_cast<size_t>(level) && r.details.size() == static_cast<size_t>(level), name + ": levels");
        for (int k = 0; k < level; ++k)
            NearAll(r.details[static_cast<size_t>(level - 1 - k)], details[static_cast<size_t>(k)], Tol(1e-9, 1e-4), name + " cD" + std::to_string(level - k));
        NearAll(SignalProcessing::IDWT(r), c["reconstruction"], Tol(1e-9, 1e-4), name + " waverec");
        return;
    }
    if (op == "denoise") {
        const auto r = SignalProcessing::WaveletDenoise(Signal(c), c["wavelet"].get<std::string>(), c["level"].get<int>(), 0.0,
                                                        c["kind"].get<std::string>() == "soft");
        Check(r.success, name + ": " + r.error_message);
        Near(r.sigma, c["sigma"].get<double>(), Tol(1e-9, 1e-4), name + " sigma");
        Near(r.threshold, c["threshold"].get<double>(), Tol(1e-9, 1e-4), name + " threshold");
        NearAll(r.denoised, c["denoised"], Tol(1e-9, 1e-4), name + " denoised");
        return;
    }
    if (op == "peaks") {
        const auto peaks = SignalProcessing::FindPeaksByProminence(Doubles(c["x"]), c["prominence"].get<double>(), c["distance"].get<int>());
        const json& idx = c["indices"];
        Check(peaks.size() == idx.size(), name + ": " + std::to_string(peaks.size()) + " peaks vs " + std::to_string(idx.size()));
        for (size_t i = 0; i < peaks.size(); ++i) {
            Check(peaks[i].index == idx[i].get<int>(), name + " index " + std::to_string(i));
            Near(peaks[i].prominence, c["prominences"][i].get<double>(), 1e-12, name + " prominence " + std::to_string(i));
            Check(peaks[i].left_base == c["left_bases"][i].get<int>() && peaks[i].right_base == c["right_bases"][i].get<int>(),
                  name + " bases " + std::to_string(i));
        }
        return;
    }
    Check(false, "unknown op " + op);
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    std::ifstream in(CYXWIZ_SIGNAL_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_SIGNAL_FIXTURE);
    const json fixture = json::parse(in);
    for (const auto& [label, s] : fixture.at("signals").items()) g_signals[label] = Doubles(s.at("x"));
    // Refusals the tab relies on.
    {
        const auto fft = SignalProcessing::FFT({});
        Check(!fft.success && fft.error_message.find("at least 1") != std::string::npos, "empty signal refused");
        const auto bad = SignalProcessing::DesignLowpass(600.0, 1000.0, 31);
        Check(!bad.success && bad.error_message.find("Nyquist") != std::string::npos, "a cutoff past Nyquist refused");
        const auto even = SignalProcessing::DesignHighpass(100.0, 1000.0, 30);
        Check(!even.success && even.error_message.find("odd") != std::string::npos, "an even highpass refused");
        const auto deep = SignalProcessing::DWT(g_signals["odd"], "db8", 9);
        Check(!deep.success && deep.error_message.find("at most") != std::string::npos, "too many levels refused: " + deep.error_message);
        Check(SignalProcessing::WaveletNames().size() == 21, "21 wavelets");
        Check(SignalProcessing::Window("nope", 8, true).empty(), "unknown window");
    }
    size_t cases = 0;
    for (const auto& c : fixture.at("cases")) {
        RunCase(c);
        ++cases;
    }
    std::cout << "signal processing matches numpy / scipy / PyWavelets: " << cases << " cases, " << checks << " checks\n";
    return 0;
}
