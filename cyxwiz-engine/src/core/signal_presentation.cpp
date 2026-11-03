#include "signal_presentation.h"

#include "explore_presentation.h"
#include "time_steps.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::signal_view {

using explore_view::Figure;
using explore_view::Number;
using series::TimeKind;

namespace {

std::string Percent(double share) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), share >= 0.1 ? "%.0f%%" : "%.1f%%", 100.0 * share);
    return buf;
}

std::string Fixed(double v, int decimals) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

// A count of time units in a few digits: "34", "2.5".
std::string Count(double v) {
    const double r = std::round(v);
    if (std::fabs(v - r) < 0.05 * std::max(1.0, std::fabs(v))) return Number(static_cast<size_t>(std::llround(std::fabs(r))));
    return Fixed(v, v < 10.0 ? 1 : 0);
}

std::string Cell(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (const char c : s) out += c == '"' ? "\"\"" : std::string(1, c);
    return out + "\"";
}

std::string TimeLabel(const signal::SignalData& d, size_t row) {
    if (d.series.kind == TimeKind::Dates && row < d.series.labels.size()) return d.series.labels[row];
    return row < d.series.t.size() ? Figure(d.series.t[row]) : "";
}

}  // namespace

std::string Num(double v) { return Figure(v); }

std::string SpanWords(double seconds, const signal::SignalData& d) {
    if (!std::isfinite(seconds) || seconds <= 0.0) return "";
    if (d.series.kind == TimeKind::Numbers) return Figure(seconds) + (std::fabs(seconds - 1.0) < 1e-9 ? " unit" : " units");
    if (d.series.kind == TimeKind::Order) return Count(seconds) + (std::fabs(seconds - 1.0) < 1e-9 ? " row" : " rows");
    const double day = series::kDaySeconds;
    if (seconds >= 2.0 * 365.2425 * day) return Count(seconds / (365.2425 * day)) + " years";
    if (seconds >= 60.0 * day) return Count(seconds / (30.436875 * day)) + " months";
    if (seconds >= 2.0 * day) return Count(seconds / day) + " days";
    if (seconds >= 2.0 * 3600.0) return Count(seconds / 3600.0) + " h";
    if (seconds >= 120.0) return Count(seconds / 60.0) + " min";
    if (seconds >= 1.0) return Count(seconds) + " s";
    return Figure(seconds * 1000.0) + " ms";
}

std::string FrequencyWords(double frequency, const signal::FrequencyUnit& unit) {
    const double in_unit = frequency * unit.seconds;
    return Figure(in_unit) + " " + unit.name;
}

std::string PeriodWords(double frequency, const signal::SignalData& d) {
    if (!(frequency > 0.0)) return "";
    return "every " + SpanWords(1.0 / frequency, d);
}

std::string RateWords(const signal::SignalData& d) {
    const signal::FrequencyUnit& u = d.unit;
    std::string out = FrequencyWords(d.sample_rate, u);
    if (d.series.kind == TimeKind::Dates && u.name != "Hz") out += " (" + Figure(d.sample_rate) + " Hz)";
    if (!d.rate_from_step) return out + " (typed)";
    out += " (from the step: every " + SpanWords(d.step_seconds, d);
    if (d.series.kind != TimeKind::Order) out += d.series.step.even ? ", even" : ", uneven: the rows are taken in order";
    return out + ")";
}

std::string RowsWords(const signal::SignalData& d) {
    const auto& s = d.series;
    const int64_t missing = s.rows - s.used;
    std::string out = Number(static_cast<size_t>(s.used)) + " rows";
    if (missing > 0)
        out = Number(static_cast<size_t>(s.used)) + " of " + Number(static_cast<size_t>(s.rows)) + " rows (" + Number(static_cast<size_t>(missing)) +
              (missing == 1 ? " has" : " have") + " no value or time)";
    else out += ", none missing";
    if (s.unordered > 0) out += "; the rows were not in time order and were sorted";
    return out;
}

std::string SpectrumTitle(const signal::SpectrumAnswer& a) {
    const std::string window = a.request.window == "boxcar" ? "no window" : (a.request.window == "hann" ? "Hann" : a.request.window == "hamming" ? "Hamming" : "Blackman");
    if (a.request.welch_segment > 0)
        return "Welch power density (" + window + ", segments of " + Number(static_cast<size_t>(a.spectrum.n)) + " rows, " +
               Number(static_cast<size_t>(a.spectrum.segments)) + " averaged), " + a.data.unit.name;
    return std::string("Amplitude spectrum (") + window + (a.request.remove_trend ? ", trend removed" : a.request.remove_mean ? ", mean removed" : "") +
           "), " + a.data.unit.name;
}

std::string SpectrumSentence(const signal::SpectrumAnswer& a) {
    const std::string& col = a.request.base.value;
    std::string out;
    if (a.peaks.empty()) {
        out = col + " has no cycle that stands out: no bin of the spectrum rises a twentieth of the tallest above its surroundings.";
    } else {
        const auto& top = a.peaks[0];
        out = col + " cycles " + PeriodWords(top.frequency, a.data) + " (" + FrequencyWords(top.frequency, a.data.unit);
        out += a.request.welch_segment > 0 ? ", power density " + Figure(top.amplitude) : ", amplitude " + Figure(top.amplitude);
        out += ")";
        if (a.peaks.size() > 1) {
            const auto& second = a.peaks[1];
            const double ratio = second.frequency / top.frequency;
            const bool harmonic = std::fabs(ratio - std::round(ratio)) < 0.03 && std::round(ratio) >= 2.0;
            out += harmonic ? ", with its harmonic " + PeriodWords(second.frequency, a.data)
                            : ", then a cycle " + PeriodWords(second.frequency, a.data) + " (" + FrequencyWords(second.frequency, a.data.unit) + ")";
        }
        out += ".";
    }
    const double resolution = a.data.sample_rate / std::max(1, a.spectrum.n);
    out += " Resolution " + FrequencyWords(resolution, a.data.unit) + " (" + SpanWords(a.spectrum.n * a.data.step_seconds, a.data) + " of rows";
    out += a.request.welch_segment > 0 ? " per segment)." : ").";
    if (a.row_sd > 0.0) {
        if (a.row_peaks.empty()) {
            out += " No row stands out by " + Figure(a.request.peak_prominence_sd) + " SD above its surroundings.";
        } else {
            out += " " + Number(a.row_peaks.size()) + (a.row_peaks.size() == 1 ? " peak stands" : " peaks stand") + " out in the rows (by " +
                   Figure(a.request.peak_prominence_sd) + " SD = " + Figure(a.request.peak_prominence_sd * a.row_sd) + ", at least " +
                   SpanWords(a.row_gap * a.data.step_seconds, a.data) + " apart)";
            if (a.row_peaks.size() >= 2) out += ", " + SpanWords(a.median_gap_seconds, a.data) + " apart on the median";
            out += ".";
        }
    }
    return out;
}

std::string SpectrogramSentence(const signal::SpectrogramAnswer& a) {
    const auto& s = a.spectrogram;
    std::string out = Number(static_cast<size_t>(s.num_bins)) + " frequency bins x " + Number(static_cast<size_t>(s.num_frames)) + " frames";
    out += " (window " + SpanWords(a.request.window_rows * a.data.step_seconds, a.data) + ", hop " + SpanWords(a.hop_used * a.data.step_seconds, a.data);
    if (a.hop_raised) out += ", raised from " + Number(static_cast<size_t>(a.request.hop_rows)) + " rows to stay within " + Number(static_cast<size_t>(a.request.max_frames)) + " frames";
    out += "). Resolution " + FrequencyWords(a.data.sample_rate / std::max(1, a.request.window_rows), a.data.unit) + " x " +
           SpanWords(a.hop_used * a.data.step_seconds, a.data) + ".";
    out += " Loudest " + Fixed(a.loudest_db, 0) + " dB at " + FrequencyWords(a.loudest_frequency, a.data.unit);
    if (a.loudest_frequency > 0.0) out += " (" + PeriodWords(a.loudest_frequency, a.data) + ")";
    out += "; the floor is " + Fixed(a.request.range_db, 0) + " dB below it.";
    return out;
}

std::string FilterSentence(const signal::FilterAnswer& a) {
    const auto& r = a.request;
    const std::string& col = r.base.value;
    std::string out;
    if (r.mode == signal::FilterMode::Design) {
        const std::string taps = Number(static_cast<size_t>(r.taps)) + " taps = " + SpanWords(r.taps * a.data.step_seconds, a.data) + " of rows";
        switch (r.kind) {
            case signal::FilterKind::Lowpass: out = "A low-pass of " + taps + " keeps the cycles of " + SpanWords(1.0 / r.cut, a.data) + " and longer"; break;
            case signal::FilterKind::Highpass: out = "A high-pass of " + taps + " keeps the cycles shorter than " + SpanWords(1.0 / r.cut, a.data); break;
            case signal::FilterKind::Bandpass:
                out = "A band-pass of " + taps + " keeps the cycles between " + SpanWords(1.0 / r.cut_high, a.data) + " and " + SpanWords(1.0 / r.cut, a.data);
                break;
            case signal::FilterKind::Bandstop:
                out = "A band-stop of " + taps + " removes the cycles between " + SpanWords(1.0 / r.cut_high, a.data) + " and " + SpanWords(1.0 / r.cut, a.data);
                break;
        }
        if (std::isfinite(a.half_power)) out += " (half power at " + SpanWords(1.0 / a.half_power, a.data) + ")";
    } else {
        const std::string size = Number(a.kernel.size()) + "-row";
        switch (r.preset) {
            case signal::KernelPreset::MovingAverage: out = "A " + size + " moving average (" + SpanWords(a.kernel.size() * a.data.step_seconds, a.data) + ")"; break;
            case signal::KernelPreset::Gaussian: out = "A " + size + " Gaussian smoothing (" + SpanWords(a.kernel.size() * a.data.step_seconds, a.data) + ")"; break;
            case signal::KernelPreset::Difference: out = "The centred difference (the change per row, halved)"; break;
            case signal::KernelPreset::SecondDifference: out = "The second difference (the curvature)"; break;
            case signal::KernelPreset::Typed: out = "The typed kernel of " + Number(a.kernel.size()) + " taps"; break;
        }
    }
    out += " removes " + Figure(a.rms_removed) + " of the " + Figure(a.rms_original) + " RMS of " + col + " and keeps " + Figure(a.rms_kept) + ".";
    out += " The filtered column is centred on the original (no delay)";
    if (a.edge_rows > 0) out += ", so its first and last " + Number(static_cast<size_t>(a.edge_rows)) + " rows lean on the edge values";
    out += ".";
    return out;
}

std::string PartScale(const signal::WaveletPart& part, const signal::SignalData& d) {
    if (part.approximation) return part.name + ": " + SpanWords(part.scale_low, d) + " and longer";
    return part.name + ": " + SpanWords(part.scale_low, d) + " .. " + SpanWords(part.scale_high, d);
}

std::string WaveletSentence(const signal::WaveletAnswer& a) {
    const std::string& col = a.request.base.value;
    if (a.parts.empty()) return "";
    const auto& approx = a.parts[0];
    std::string out = (approx.energy_share >= 0.5 ? "Most of " : "Part of ") + col + " is slow: the approximation over " +
                      SpanWords(approx.scale_low, a.data) + " and longer holds " + Percent(approx.energy_share) + " of the energy";
    const signal::WaveletPart* largest_detail = nullptr;
    for (size_t i = 1; i < a.parts.size(); ++i)
        if (!largest_detail || a.parts[i].energy_share > largest_detail->energy_share) largest_detail = &a.parts[i];
    if (largest_detail)
        out += "; the " + SpanWords(largest_detail->scale_low, a.data) + "-to-" + SpanWords(largest_detail->scale_high, a.data) + " detail (" +
               largest_detail->name + ") carries " + Percent(largest_detail->energy_share) + ".";
    else out += ".";
    out += " Rebuilt from its " + Number(a.parts.size()) + " parts the column comes back to " + Figure(a.reconstruction_error) + " at most.";
    if (a.denoised.success) {
        double change = 0.0;
        const auto& x = a.data.series.values;
        if (a.denoised.denoised.size() == x.size()) {
            double s = 0.0;
            for (size_t i = 0; i < x.size(); ++i) s += (x[i] - a.denoised.denoised[i]) * (x[i] - a.denoised.denoised[i]);
            change = std::sqrt(s / std::max<size_t>(1, x.size()));
        }
        out += " The sample-to-sample noise reads " + Figure(a.denoised.sigma) + " (from D1); " +
               (a.request.denoise == signal::DenoiseMode::Soft ? "soft" : "hard") + " thresholding at " + Figure(a.denoised.threshold) +
               (a.request.threshold > 0.0 ? " (typed)" : " (the universal threshold)") + " changes the column by " + Figure(change) + " RMS.";
    }
    return out;
}

// ---- CSV

std::string SpectrumCsv(const signal::SpectrumAnswer& a) {
    const bool welch = a.request.welch_segment > 0;
    std::string out = "frequency_" + Cell(a.data.unit.name) + ",period," + (welch ? "power_density" : "amplitude") + (welch ? "" : ",power_density") + "\n";
    for (size_t k = 0; k < a.spectrum.frequencies.size(); ++k) {
        const double f = a.spectrum.frequencies[k];
        out += Figure(f * a.data.unit.seconds) + "," + (f > 0.0 ? Figure(1.0 / f) : "") + ",";
        if (welch) out += Figure(a.spectrum.psd[k]);
        else out += Figure(a.spectrum.amplitude[k]) + "," + Figure(a.spectrum.psd[k]);
        out += "\n";
    }
    return out;
}

std::string RowPeaksCsv(const signal::SpectrumAnswer& a) {
    std::string out = "row,time,value,stands_out_by\n";
    for (const auto& p : a.row_peaks)
        out += std::to_string(p.index + 1) + "," + Cell(TimeLabel(a.data, static_cast<size_t>(p.index))) + "," + Figure(p.value) + "," + Figure(p.prominence) + "\n";
    return out;
}

std::string SpectrogramCsv(const signal::SpectrogramAnswer& a) {
    std::string out = "time,frequency_" + Cell(a.data.unit.name) + ",power_density,db\n";
    const auto& s = a.spectrogram;
    for (size_t f = 0; f < s.spectrogram.size(); ++f)
        for (size_t b = 0; b < s.spectrogram[f].size(); ++b) {
            const double p = s.spectrogram[f][b];
            out += Figure(s.times[f]) + "," + Figure(s.frequencies[b] * a.data.unit.seconds) + "," + Figure(p) + "," +
                   (p > 0.0 ? Fixed(10.0 * std::log10(p), 2) : "") + "\n";
        }
    return out;
}

std::string FilterCsv(const signal::FilterAnswer& a) {
    std::string out = "row,time," + Cell(a.request.base.value) + "," + Cell(a.request.base.value + "_filtered") + "\n";
    const auto& x = a.data.series.values;
    for (size_t i = 0; i < x.size() && i < a.filtered.size(); ++i)
        out += std::to_string(i + 1) + "," + Cell(TimeLabel(a.data, i)) + "," + Figure(x[i]) + "," + Figure(a.filtered[i]) + "\n";
    return out;
}

std::string WaveletCsv(const signal::WaveletAnswer& a) {
    std::string out = "row,time," + Cell(a.request.base.value);
    for (const auto& part : a.parts) out += "," + Cell(a.request.base.value + "_" + part.name);
    if (a.denoised.success) out += "," + Cell(a.request.base.value + "_denoised");
    out += "\n";
    const auto& x = a.data.series.values;
    for (size_t i = 0; i < x.size(); ++i) {
        out += std::to_string(i + 1) + "," + Cell(TimeLabel(a.data, i)) + "," + Figure(x[i]);
        for (const auto& part : a.parts) out += "," + (i < part.rows.size() ? Figure(part.rows[i]) : std::string());
        if (a.denoised.success) out += "," + (i < a.denoised.denoised.size() ? Figure(a.denoised.denoised[i]) : std::string());
        out += "\n";
    }
    return out;
}

}  // namespace cyxwiz::signal_view
