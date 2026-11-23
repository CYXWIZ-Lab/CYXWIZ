// Data Studio's Explore tab Signal (TOFIX129 D7): reading a signal over its
// time through the session query service (the sample rate from the step, the
// unit that fits), the four answers on a synthetic sensor (1,440 rows at 10 s
// with a 34-minute cycle, the MetroPT-3 shape) and on the airline passengers,
// the peaks' choices, the hop rule for long series, the kernels, the parts
// brought back to the rows, the node parameters, and the words. The figures
// themselves are the backend's, checked against numpy / scipy / PyWavelets in
// test_computation_truth_signal.

#include "../src/core/signal_explore.h"
#include "../src/core/signal_presentation.h"
#include "../src/core/session_query_engine.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double a, double b, double tol, const std::string& what) {
    Check(std::fabs(a - b) <= tol * (1.0 + std::fabs(b)), what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

void Contains(const std::string& text, const std::string& part, const std::string& what) {
    Check(text.find(part) != std::string::npos, what + ": \"" + part + "\" not in \"" + text + "\"");
}

constexpr double kPi = 3.14159265358979323846;

// A sensor sampled every 10 s for 4 hours (1,440 rows): a 34-minute cycle of
// amplitude 1.9, its harmonic, a 20-minute cycle, an offset and a little
// noise; row 700 has no value, rows 10 and 11 are swapped.
std::shared_ptr<arrow::Table> SensorTable() {
    arrow::TimestampBuilder when(arrow::timestamp(arrow::TimeUnit::SECOND), arrow::default_memory_pool());
    arrow::DoubleBuilder value, other;
    std::vector<int> order(1440);
    for (int i = 0; i < 1440; ++i) order[static_cast<size_t>(i)] = i;
    std::swap(order[10], order[11]);
    unsigned seed = 7;
    for (const int i : order) {
        const double t = 10.0 * i;
        seed = seed * 1103515245u + 12345u;
        const double noise = 0.05 * (static_cast<double>((seed >> 8) % 1000) / 500.0 - 1.0);
        const double v = 4.0 + 1.9 * std::sin(2.0 * kPi * t / 2040.0) + 0.5 * std::sin(2.0 * kPi * t / 1020.0 + 0.3) + 0.8 * std::sin(2.0 * kPi * t / 1200.0) + noise;
        (void)when.Append(1580515200 + static_cast<int64_t>(t));  // 2020-02-01 00:00 UTC
        if (i == 700) (void)value.AppendNull();
        else (void)value.Append(v);
        (void)other.Append(static_cast<double>(i % 7));
    }
    std::shared_ptr<arrow::Array> wa, va, oa;
    (void)when.Finish(&wa);
    (void)value.Finish(&va);
    (void)other.Finish(&oa);
    return arrow::Table::Make(arrow::schema({arrow::field("timestamp", arrow::timestamp(arrow::TimeUnit::SECOND)),
                                             arrow::field("Motor_current", arrow::float64()), arrow::field("other", arrow::float64())}),
                              {wa, va, oa});
}

std::shared_ptr<arrow::Table> AirlineTable() {
    std::ifstream in(CYXWIZ_AIRLINE_CSV);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_AIRLINE_CSV);
    arrow::StringBuilder month;
    arrow::DoubleBuilder value;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        (void)month.Append(line.substr(0, comma));
        (void)value.Append(std::stod(line.substr(comma + 1)));
    }
    std::shared_ptr<arrow::Array> ma, va;
    (void)month.Finish(&ma);
    (void)value.Finish(&va);
    return arrow::Table::Make(arrow::schema({arrow::field("Month", arrow::utf8()), arrow::field("Passengers", arrow::float64())}), {ma, va});
}
}  // namespace

int main() {
    // ---- pure helpers: the unit that fits the step
    Check(signal::UnitFor(series::TimeKind::Dates, 1000.0).name == "Hz", "1 kHz reads in Hz");
    Check(signal::UnitFor(series::TimeKind::Dates, 0.1).name == "per hour", "10 s steps read per hour");
    Check(signal::UnitFor(series::TimeKind::Dates, 1.0 / 86400.0).name == "per month", "daily rows read per month");
    Check(signal::UnitFor(series::TimeKind::Dates, 12.0 / (365.2425 * 86400.0)).name == "per year", "monthly rows read per year");
    Check(signal::UnitFor(series::TimeKind::Numbers, 2.0).name == "per unit" && signal::UnitFor(series::TimeKind::Order, 1.0).name == "per row", "numbers and order units");
    Check(signal::Kernel(signal::KernelPreset::MovingAverage, 4, {}) == std::vector<double>({0.25, 0.25, 0.25, 0.25}), "moving average kernel");
    const auto gauss = signal::Kernel(signal::KernelPreset::Gaussian, 5, {});
    Near(gauss[0] + gauss[1] + gauss[2] + gauss[3] + gauss[4], 1.0, 1e-12, "gaussian sums to 1");
    Check(gauss[2] > gauss[1] && gauss[1] == gauss[3], "gaussian is a symmetric bump");
    Check(signal::Kernel(signal::KernelPreset::Difference, 9, {}).size() == 3 && signal::Kernel(signal::KernelPreset::Typed, 0, {1, 2}).size() == 2, "difference and typed kernels");
    Near(signal::Rms({3.0, 4.0}), std::sqrt(12.5), 1e-12, "rms");

    // ---- reading
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_signal_explore";
    std::error_code ec;
    fs::remove_all(root, ec);
    SessionQueryEngine engine(root.string());
    std::string error;
    Check(engine.AttachArrow("sensor", 1, SensorTable(), &error), "attach sensor: " + error);
    Check(engine.AttachArrow("airline", 2, AirlineTable(), &error), "attach airline: " + error);
    const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };

    signal::SignalRequest base{"timestamp", "Motor_current", 0.0};
    const auto read = signal::ReadSignal("sensor", base, run, {});
    Check(read.ok(), "read: " + read.error);
    const auto& d = read.data;
    Check(d.series.rows == 1440 && d.series.used == 1439 && d.series.unordered == 1, "1,439 of 1,440 rows, one swap");
    Near(d.sample_rate, 0.1, 1e-9, "0.1 Hz from the 10 s step");
    Check(d.rate_from_step && d.unit.name == "per hour" && d.series.step.even, "unit per hour, even step");
    Contains(signal_view::RateWords(d), "0.1 Hz = 360 per hour (from the step: every 10 s, even)", "rate words: " + signal_view::RateWords(d));
    Check(signal_view::RowsWords(d) == "1,439 of 1,440 rows (1 has no value or time); the rows were not in time order and were sorted", signal_view::RowsWords(d));
    Check(signal_view::PeriodWords(1.0 / 2040.0, d) == "every 34 min" && signal_view::SpanWords(9720.0, d) == "2.7 h" && signal_view::SpanWords(10.0, d) == "10 s",
          "period and span words: " + signal_view::PeriodWords(1.0 / 2040.0, d) + " / " + signal_view::SpanWords(9720.0, d));
    signal::SignalRequest typed = base;
    typed.sample_rate = 0.2;
    const auto typed_read = signal::ReadSignal("sensor", typed, run, {});
    Check(typed_read.ok() && !typed_read.data.rate_from_step && typed_read.data.unit.name == "per hour", "a typed rate");
    Contains(signal_view::RateWords(typed_read.data), "(typed)", "typed words");
    signal::SignalRequest same = base;
    same.time = "Motor_current";
    Check(!signal::ReadSignal("sensor", same, run, {}).ok(), "time = value refused");

    // ---- Spectrum: the 34-minute cycle is the strongest, the 17-minute one its harmonic
    signal::SpectrumRequest spectrum;
    spectrum.base = base;
    spectrum.peak_gap_rows = 60;  // 10 minutes
    const auto sp = signal::RunSpectrum("sensor", spectrum, run, {});
    Check(sp.ok(), "spectrum: " + sp.error);
    Check(!sp.peaks.empty(), "peaks found");
    Near(1.0 / sp.peaks[0].frequency, 2040.0, 0.08, "the strongest cycle is the 34-minute one: " + std::to_string(1.0 / sp.peaks[0].frequency));
    Near(sp.peaks[0].amplitude, 1.9, 0.2, "its amplitude reads near 1.9 (7.05 cycles in the window: Hann scalloping): " + std::to_string(sp.peaks[0].amplitude));
    Check(sp.row_peaks.size() >= 5 && sp.row_peaks.size() <= 9 && sp.row_gap == 60, "row peaks: " + std::to_string(sp.row_peaks.size()));
    Near(sp.median_gap_seconds, 2040.0, 0.2, "median gap near the cycle: " + std::to_string(sp.median_gap_seconds));
    const std::string words = signal_view::SpectrumSentence(sp);
    Contains(words, "Motor_current cycles every 34 min (1.751 per hour, amplitude 1.9", "spectrum words: " + words);
    Contains(words, "peaks stand out in the rows", "row peak words: " + words);
    Contains(signal_view::SpectrumTitle(sp), "Amplitude spectrum (Hann, mean removed), per hour", "title: " + signal_view::SpectrumTitle(sp));
    Contains(signal_view::SpectrumCsv(sp), "frequency_per hour,period,amplitude,power_density\n", "spectrum csv");
    Contains(signal_view::RowPeaksCsv(sp), "row,time,value,stands_out_by\n", "row peaks csv");
    spectrum.welch_segment = 512;
    const auto welch = signal::RunSpectrum("sensor", spectrum, run, {});
    Check(welch.ok() && welch.spectrum.segments == 4 && welch.spectrum.amplitude.empty() && !welch.peaks.empty(), "Welch: 4 segments, power only: " + welch.error);
    Near(1.0 / welch.peaks[0].frequency, 2560.0, 0.01, "Welch's strongest bin at 512-row segments (bins 5,120 s apart in period: 2,560 s is the nearest to the 2,040 s cycle): " + std::to_string(1.0 / welch.peaks[0].frequency));
    Contains(signal_view::SpectrumTitle(welch), "Welch power density (Hann, segments of 512 rows, 4 averaged), per hour", signal_view::SpectrumTitle(welch));
    const auto node = signal::NodeParameters(welch.request);
    Check(node.at("signal_col") == "Motor_current" && node.at("window") == "hann" && node.at("sample_rate") == "0.1" && node.at("remove_trend") == "false",
          "FFT node parameters carry the resolved rate: " + node.at("sample_rate"));

    // ---- Spectrogram: 128-row window, hop 32 (42 frames); a tiny frame budget raises the hop and says so
    signal::SpectrogramRequest spec;
    spec.base = base;
    spec.window_rows = 128;
    spec.hop_rows = 32;
    const auto sg = signal::RunSpectrogram("sensor", spec, run, {});
    Check(sg.ok() && sg.spectrogram.num_bins == 65 && sg.spectrogram.num_frames == 41 && !sg.hop_raised, "spectrogram 65 x 41: " + sg.error + " " + std::to_string(sg.spectrogram.num_frames));
    Near(1.0 / sg.loudest_frequency, 1280.0, 0.01, "loudest in the first bin (1,280 s at a 128-row window; the 2,040 s cycle lives there): " + std::to_string(1.0 / sg.loudest_frequency));
    Contains(signal_view::SpectrogramSentence(sg), "65 frequency bins x 41 frames (window 21 min, hop 5.3 min). Resolution 2.812 per hour x 5.3 min. Loudest", signal_view::SpectrogramSentence(sg));
    spec.max_frames = 10;
    const auto few = signal::RunSpectrogram("sensor", spec, run, {});
    Check(few.ok() && few.hop_raised && few.spectrogram.num_frames <= 10 && few.hop_used == 132, "the hop is raised to stay within 10 frames: " + std::to_string(few.hop_used));
    Contains(signal_view::SpectrogramSentence(few), "raised from 32 rows to stay within 10 frames", "hop words");
    spec.window_rows = 5000;
    Check(!signal::RunSpectrogram("sensor", spec, run, {}).ok(), "a window longer than the series is refused");
    Contains(signal_view::SpectrogramCsv(sg), "time,frequency_per hour,power_density,db\n", "spectrogram csv");

    // ---- Filter: a 10-minute low-pass keeps the cycles, a moving average too; kernels in Kernel mode
    signal::FilterRequest filter;
    filter.base = base;
    filter.cut = 1.0 / 600.0;
    filter.taps = 101;
    const auto lp = signal::RunFilter("sensor", filter, run, {});
    Check(lp.ok() && lp.kernel.size() == 101 && lp.filtered.size() == 1439 && lp.edge_rows == 50, "low-pass: " + lp.error);
    Check(lp.rms_removed < 0.2 && lp.rms_kept > 3.5, "the cycles pass, the noise goes: removed " + std::to_string(lp.rms_removed));
    Check(std::isfinite(lp.half_power) && 1.0 / lp.half_power > 600.0 && 1.0 / lp.half_power < 1000.0, "half power a little past the cut: " + std::to_string(1.0 / lp.half_power));
    Contains(signal_view::FilterSentence(lp), "A low-pass of 101 taps = 17 min of rows keeps the cycles of 10 min and longer (half power at", signal_view::FilterSentence(lp));
    Contains(signal_view::FilterSentence(lp), "first and last 50 rows lean on the edge values", "edge words");
    Check(signal::FilterNodeType(filter) == std::string("FilterDesigner"), "Design -> Filter Designer node");
    const auto fp = signal::NodeParameters(lp.request);
    Check(fp.at("filter_type") == "lowpass" && fp.at("taps") == "101" && std::fabs(std::stod(fp.at("cutoff")) - 1.0 / 600.0) < 1e-12 && fp.at("sample_rate") == "0.1",
          "filter node parameters: " + fp.at("cutoff") + " at " + fp.at("sample_rate"));
    filter.kind = signal::FilterKind::Highpass;
    filter.taps = 100;
    Check(!signal::RunFilter("sensor", filter, run, {}).ok(), "an even high-pass is refused");
    filter.mode = signal::FilterMode::Kernel;
    filter.preset = signal::KernelPreset::MovingAverage;
    filter.kernel_size = 30;
    const auto ma = signal::RunFilter("sensor", filter, run, {});
    Check(ma.ok() && ma.kernel.size() == 30 && ma.edge_rows == 14 && ma.rms_removed < 0.3, "moving average: " + ma.error);
    Contains(signal_view::FilterSentence(ma), "A 30-row moving average (5 min) removes", signal_view::FilterSentence(ma));
    Check(signal::FilterNodeType(filter) == std::string("Convolution1D") && signal::NodeParameters(filter).at("kernel").rfind("0.0333333333333,", 0) == 0,
          "Kernel -> Convolution 1D node: " + signal::NodeParameters(filter).at("kernel"));
    filter.preset = signal::KernelPreset::Typed;
    filter.kernel_typed = {};
    Check(!signal::RunFilter("sensor", filter, run, {}).ok(), "an empty typed kernel is refused");
    Contains(signal_view::FilterCsv(lp), "row,time,Motor_current,Motor_current_filtered\n", "filter csv");

    // ---- Wavelet: db4, the default levels; the parts add up to the column; the approximation holds most of the energy
    signal::WaveletRequest wavelet;
    wavelet.base = base;
    const auto wv = signal::RunWavelet("sensor", wavelet, run, {});
    Check(wv.ok() && wv.request.levels == 4 && wv.parts.size() == 5 && wv.parts[0].name == "A4" && wv.parts[1].name == "D4" && wv.parts[4].name == "D1", "wavelet parts: " + wv.error);
    Check(wv.coefficients.max_levels == 7, "7 levels at most for 1,439 rows with db4: " + std::to_string(wv.coefficients.max_levels));
    double share = 0.0;
    for (const auto& part : wv.parts) share += part.energy_share;
    Near(share, 1.0, 1e-9, "the energy shares sum to 1");
    Check(wv.parts[0].energy_share > 0.9, "the approximation holds most of the energy: " + std::to_string(wv.parts[0].energy_share));
    Check(wv.reconstruction_error < 1e-9, "perfect reconstruction: " + std::to_string(wv.reconstruction_error));
    const auto& x = wv.data.series.values;
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double sum = 0.0;
        for (const auto& part : wv.parts) sum += part.rows[i];
        worst = std::max(worst, std::fabs(sum - x[i]));
    }
    Check(worst < 1e-9, "the parts brought back to the rows add up to the column: " + std::to_string(worst));
    Check(signal_view::PartScale(wv.parts[0], wv.data) == "A4: 5.3 min and longer" && signal_view::PartScale(wv.parts[4], wv.data) == "D1: 20 s .. 40 s",
          "part scales: " + signal_view::PartScale(wv.parts[0], wv.data) + " / " + signal_view::PartScale(wv.parts[4], wv.data));
    Contains(signal_view::WaveletSentence(wv), "Most of Motor_current is slow: the approximation over 5.3 min and longer holds", signal_view::WaveletSentence(wv));
    wavelet.denoise = signal::DenoiseMode::Soft;
    const auto dn = signal::RunWavelet("sensor", wavelet, run, {});
    Check(dn.ok() && dn.denoised.success && dn.denoised.denoised.size() == x.size() && dn.denoised.threshold > 0.0, "denoised: " + dn.error);
    Contains(signal_view::WaveletSentence(dn), "soft thresholding at", "denoise words");
    const auto wp = signal::NodeParameters(dn.request);
    Check(wp.at("wavelet") == "db4" && wp.at("level") == "4" && wp.at("denoise") == "soft", "wavelet node parameters carry the levels used: " + wp.at("level"));
    wavelet.levels = 9;
    Check(!signal::RunWavelet("sensor", wavelet, run, {}).ok(), "too many levels refused");
    Contains(signal_view::WaveletCsv(dn), "row,time,Motor_current,Motor_current_A4,Motor_current_D4,Motor_current_D3,Motor_current_D2,Motor_current_D1,Motor_current_denoised\n", "wavelet csv");

    // ---- the airline passengers: monthly rows read per year, the yearly cycle on top
    signal::SpectrumRequest yearly;
    yearly.base = {"Month", "Passengers", 0.0};
    const auto trending = signal::RunSpectrum("airline", yearly, run, {});
    Check(trending.ok() && !trending.peaks.empty() && trending.peaks[0].frequency * trending.data.unit.seconds < 0.1,
          "with the trend in, the whole length tops the spectrum");
    yearly.remove_trend = true;
    const auto air = signal::RunSpectrum("airline", yearly, run, {});
    Contains(signal_view::SpectrumTitle(air), "trend removed", "title says the trend is out");
    Check(air.ok() && air.data.unit.name == "per year", "airline unit: " + air.error + " " + air.data.unit.name);
    Check(!air.peaks.empty(), "airline peaks");
    Near(air.peaks[0].frequency * air.data.unit.seconds, 1.0, 0.05, "the strongest cycle is yearly: " + std::to_string(air.peaks[0].frequency * air.data.unit.seconds));
    Contains(signal_view::SpectrumSentence(air), "Passengers cycles every 12 months (1 per year", signal_view::SpectrumSentence(air));

    std::cout << "signal explore: units, reading, spectrum (whole and Welch), spectrogram (hop rule), filter (design and kernels), wavelet parts, "
                 "nodes, words, airline. OK\n";
    return 0;
}
