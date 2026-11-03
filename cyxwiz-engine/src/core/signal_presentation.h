#pragma once

// Words of Data Studio's Explore tab Signal (TOFIX129 D7, approved mockup
// d7_signal_mockup.html): the sample rate and its unit, frequencies and
// periods in time words, what the spectrum, the spectrogram, the filter and
// the wavelet parts found, and the CSV exports. Answers in, text out, so the
// tab and its test agree. No ImGui, no queries.

#include "signal_explore.h"

#include <string>
#include <vector>

namespace cyxwiz::signal_view {

// "0.1 Hz (from the step: every 10 s, even)" / "12 per year (typed)" / "1 per row".
std::string RateWords(const signal::SignalData& d);
// A frequency (per second / unit / row) in the display unit: "1.75 per hour".
std::string FrequencyWords(double frequency, const signal::FrequencyUnit& unit);
// The period of a frequency in time words: "every 34 min", "every 2.5 h", "every 12 months"; "every 10 rows" / "every 2.5 units".
std::string PeriodWords(double frequency, const signal::SignalData& d);
// A span of seconds (or units / rows) in a fitting unit: "34 min", "2.7 h", "10 s", "3 rows".
std::string SpanWords(double seconds, const signal::SignalData& d);
// A number in a few digits (explore_view::Figure).
std::string Num(double v);

// "1,440 rows, none missing" / "1,438 of 1,440 rows (2 have no value); the rows were not in time order and were sorted".
std::string RowsWords(const signal::SignalData& d);

std::string SpectrumSentence(const signal::SpectrumAnswer& a);
std::string SpectrogramSentence(const signal::SpectrogramAnswer& a);
std::string FilterSentence(const signal::FilterAnswer& a);
std::string WaveletSentence(const signal::WaveletAnswer& a);

// Labels: "Amplitude spectrum (Hann, mean removed), per hour" / "Welch power density (Hann, segments of 1,024 rows)".
std::string SpectrumTitle(const signal::SpectrumAnswer& a);
// "D4: 80 s .. 2.7 min" / "A4: 2.7 min and longer".
std::string PartScale(const signal::WaveletPart& part, const signal::SignalData& d);

// CSV exports.
std::string SpectrumCsv(const signal::SpectrumAnswer& a);
std::string RowPeaksCsv(const signal::SpectrumAnswer& a);
std::string SpectrogramCsv(const signal::SpectrogramAnswer& a);
std::string FilterCsv(const signal::FilterAnswer& a);
std::string WaveletCsv(const signal::WaveletAnswer& a);

}  // namespace cyxwiz::signal_view
