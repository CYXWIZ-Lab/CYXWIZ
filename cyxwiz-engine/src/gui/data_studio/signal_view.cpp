#include "signal_view.h"

#include "explore_plots.h"
#include "series_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/signal_presentation.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cyxwiz {

namespace {

namespace sv = signal_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

constexpr size_t kWelchAbove = 100000;  // rows: Welch averaging on by default (D7-4)

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// A faint note after the item before it, on the same line.
void FaintAfter(const std::string& text) {
    if (text.empty()) return;
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Heading(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

float TableHeight(size_t rows) {
    return static_cast<float>(rows) * (ImGui::GetTextLineHeight() + 2.0f * ImGui::GetStyle().CellPadding.y) + 4.0f;
}

// A setting after the one before it, on a new line when it would not fit.
void Setting(const char* label, float width) {
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(label).x + width) ImGui::NewLine();
    Label(label);
}

std::string Round(double v, const char* format = "%.3g") {
    if (!std::isfinite(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

float PlotHeight(float w) { return std::clamp(w * 0.3f, 140.0f, 240.0f); }

// The axis ends of a signal: "2020-02-01 00:00:00" .. , numbers, or row numbers.
std::string AxisLabel(const series::Series& s, size_t i) {
    if (s.kind == series::TimeKind::Dates && i < s.labels.size()) return s.labels[i];
    if (s.kind == series::TimeKind::Numbers && i < s.t.size()) return explore_view::Figure(s.t[i]);
    return std::to_string(i + 1);
}

// The time of a row in a few words: the date label (its time of day when the
// day is the same as the first row's), the number, or the row number.
std::string RowLabel(const series::Series& s, size_t i) {
    if (s.kind == series::TimeKind::Dates && i < s.labels.size()) {
        const std::string& l = s.labels[i];
        if (!s.labels.empty() && l.size() > 11 && s.labels.front().size() > 11 && l.compare(0, 10, s.labels.front(), 0, 10) == 0) return l.substr(11);
        return l;
    }
    return AxisLabel(s, i);
}

const char* const kWindows[] = {"Hann", "Hamming", "Blackman", "None"};
const char* const kWindowNames[] = {"hann", "hamming", "blackman", "boxcar"};
int WindowIndex(const std::string& name) {
    for (int i = 0; i < 4; ++i)
        if (name == kWindowNames[i]) return i;
    return 0;
}

std::vector<double> ParseKernel(const char* text) {
    std::vector<double> out;
    std::string item;
    std::stringstream in(text);
    while (std::getline(in, item, ',')) {
        const size_t a = item.find_first_not_of(" \t"), b = item.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        item = item.substr(a, b - a + 1);
        char* end = nullptr;
        const double v = std::strtod(item.c_str(), &end);
        out.push_back(end && *end == '\0' ? v : NAN);
    }
    return out;
}

}  // namespace

void SignalView::Message(const std::string& text, bool error) {
    message_ = text;
    message_error_ = error;
}

void SignalView::SelectLook(Look look, bool kernel) {
    look_ = look;
    if (look == Look::Filter) filter_.mode = kernel ? signal::FilterMode::Kernel : signal::FilterMode::Design;
}

template <typename T>
bool SignalView::Waiting(ExploreJob<T>& job, const std::string& key, const char* text) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (key == stopped_key_) {
        Faint("Stopped. Change a setting to run again.");
        return false;
    }
    if (!job.Error().empty()) {
        Wrapped(t.error, job.Error());
        return false;
    }
    Faint(text);
    if (ui::LinkButton("Stop")) {
        job.Cancel();
        stopped_key_ = key;
    }
    return true;
}

std::string SignalView::BaseKey(const std::string& dataset) const {
    return dataset + "|" + base_.time + "|" + base_.value + "|" + Round(base_.sample_rate, "%.17g");
}

void SignalView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to read one of its columns as a signal over time.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        spectrum_job_.Reset();
        spectrogram_job_.Reset();
        filter_job_.Reset();
        wavelet_job_.Reset();
        message_.clear();
        stopped_key_.clear();
        if (!welch_chosen_) welch_ = p.rows > kWelchAbove;  // a long series: segments by default (D7-4)
    }
    RenderControls(context, p);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (base_.value.empty()) {
        Faint("Pick the number column to read as the signal.");
        return;
    }
    switch (look_) {
        case Look::Spectrum: RenderSpectrum(context); break;
        case Look::Spectrogram: RenderSpectrogram(context); break;
        case Look::Filter: RenderFilter(context); break;
        case Look::Wavelet: RenderWavelet(context); break;
    }
}

void SignalView::RenderControls(const ExploreContext& context, const DatasetProfile& p) {
    series::SeriesRequest pick;
    pick.time = base_.time;
    pick.value = base_.value;
    RenderSeriesPickers("##signal", context, p, picker_, pick, false, "Signal");
    base_.time = pick.time;
    base_.value = pick.value;
    Setting("Sample rate", 150.0f);
    ui::NumberFieldOptions rate;
    rate.min = 0.0;
    rate.max = 1e12;
    rate.format = "%g";
    rate.width = 72.0f;
    rate.hint = "Samples per second (dates), per unit (a number time column) or per row. 0: from the time column's step";
    ui::NumberField("##signal_rate", &base_.sample_rate, rate);
    FaintAfter((rate_unit_.empty() ? std::string() : rate_unit_ + "; ") + (base_.sample_rate > 0.0 ? "typed; 0: from the step" : "0: 1 / the step"));
    static const char* const kLooks[] = {"Spectrum", "Spectrogram", "Filter", "Wavelet"};
    int look = static_cast<int>(look_);
    ui::FlowRow looks;
    looks.Next(ImGui::CalcTextSize("Look atSpectrumSpectrogramFilterWavelet").x + 110.0f);
    Label("Look at");
    if (ExploreChoices("##signal_look", kLooks, 4, &look)) look_ = static_cast<Look>(look);
    RenderLookSettings(p);
}

void SignalView::RenderLookSettings(const DatasetProfile& p) {
    switch (look_) {
        case Look::Spectrum: {
            int window = WindowIndex(spectrum_.window);
            Setting("Window", 230.0f);
            if (ExploreChoices("##signal_window", kWindows, 4, &window)) spectrum_.window = kWindowNames[window];
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The taper before the FFT: Hann for most signals; None for a signal that is already periodic in its length");
            Setting("Remove", 220.0f);
            ImGui::Checkbox("the mean##signal_mean", &spectrum_.remove_mean);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The mean taken out first, so the 0 bin does not top the spectrum");
            ImGui::SameLine(0.0f, 10.0f);
            ImGui::Checkbox("the trend##signal_trend", &spectrum_.remove_trend);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("A least-squares line taken out first: a trending series otherwise puts its whole length on top");
            static const char* const kAverage[] = {"whole series", "Welch segments of"};
            int average = welch_ ? 1 : 0;
            Setting("Average", 260.0f);
            if (ExploreChoices("##signal_average", kAverage, 2, &average)) {
                welch_ = average == 1;
                welch_chosen_ = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Welch: the spectrum averaged over segments (half overlapping): smoother, coarser bins; on by default above %s rows", explore_view::Number(kWelchAbove).c_str());
            if (welch_) {
                ImGui::SameLine(0.0f, 6.0f);
                ui::NumberField("##signal_segment", &welch_segment_, 8, 1 << 24, 0, 64.0f, "Rows per segment (the bins are the sample rate over this)");
                FaintAfter("rows");
            }
            static const char* const kScale[] = {"amplitude", "power", "dB"};
            int scale = static_cast<int>(scale_);
            Setting("Scale", 200.0f);
            if (ExploreChoices("##signal_scale", kScale, 3, &scale)) scale_ = static_cast<Scale>(scale);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("amplitude: the height of a tone in the column's units (whole series only); power: the density (scipy periodogram / welch); dB: 10 log10 of it");
            break;
        }
        case Look::Spectrogram: {
            Setting("Window", 150.0f);
            ui::NumberField("##signal_swin", &spectrogram_.window_rows, 2, 1 << 24, 0, 64.0f, "Rows per frame: longer gives finer frequencies and coarser times");
            FaintAfter("rows");
            Setting("Hop", 150.0f);
            ui::NumberField("##signal_hop", &spectrogram_.hop_rows, 1, 1 << 24, 0, 64.0f, "Rows between frames; raised when the series would give more than 4,000 frames");
            FaintAfter("rows");
            int window = WindowIndex(spectrogram_.window);
            Setting("Shape", 230.0f);
            if (ExploreChoices("##signal_sshape", kWindows, 4, &window)) spectrogram_.window = kWindowNames[window];
            static const char* const kScale[] = {"dB", "power"};
            int scale = spectrogram_db_ ? 0 : 1;
            Setting("Scale", 130.0f);
            if (ExploreChoices("##signal_sscale", kScale, 2, &scale)) spectrogram_db_ = scale == 0;
            Setting("Range", 120.0f);
            ui::NumberFieldOptions range;
            range.min = 1.0;
            range.max = 300.0;
            range.format = "%g";
            range.width = 56.0f;
            range.hint = "The floor, this many dB below the loudest bin; cells below it take the background";
            ui::NumberField("##signal_range", &spectrogram_.range_db, range);
            FaintAfter("dB");
            break;
        }
        case Look::Filter: {
            static const char* const kMode[] = {"Design", "Kernel"};
            int mode = filter_.mode == signal::FilterMode::Design ? 0 : 1;
            Setting("Filter", 150.0f);
            if (ExploreChoices("##signal_fmode", kMode, 2, &mode)) filter_.mode = mode == 0 ? signal::FilterMode::Design : signal::FilterMode::Kernel;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Design: a windowed-sinc FIR (scipy firwin). Kernel: a list of taps run over the column (the Convolution Calculator)");
            if (filter_.mode == signal::FilterMode::Design) {
                static const char* const kKeep[] = {"below", "above", "between", "outside"};
                int keep = static_cast<int>(filter_.kind);
                Setting("Keep", 250.0f);
                if (ExploreChoices("##signal_keep", kKeep, 4, &keep)) filter_.kind = static_cast<signal::FilterKind>(keep);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("below: low-pass; above: high-pass; between: band-pass; outside: band-stop");
                const bool band = filter_.kind == signal::FilterKind::Bandpass || filter_.kind == signal::FilterKind::Bandstop;
                Setting(band ? "From" : "Cut", 160.0f);
                CutField("##signal_cut", &filter_.cut, "The cut frequency; 0: a tenth of the Nyquist frequency");
                FaintAfter(unit_.name + (filter_.cut > 0.0 ? "" : " (0: a tenth of Nyquist)"));
                if (band) {
                    Setting("to", 160.0f);
                    CutField("##signal_cut_high", &filter_.cut_high, "The upper cut frequency, above the lower one and below Nyquist");
                    FaintAfter(unit_.name);
                }
                Setting("Taps", 110.0f);
                if (ui::NumberField("##signal_taps", &filter_.taps, 3, 100001, 2, 56.0f, "Coefficients: more gives a sharper cut and more edge rows; odd")) {
                    if (filter_.taps % 2 == 0) ++filter_.taps;
                }
                static const char* const kShape[] = {"Hamming", "Hann", "Blackman"};
                static const char* const kShapeNames[] = {"hamming", "hann", "blackman"};
                int shape = filter_.window == "hann" ? 1 : filter_.window == "blackman" ? 2 : 0;
                Setting("Shape", 200.0f);
                if (ExploreChoices("##signal_fshape", kShape, 3, &shape)) filter_.window = kShapeNames[shape];
            } else {
                static const char* const kPreset[] = {"Moving average", "Gaussian", "Difference", "Second difference", "Typed"};
                int preset = static_cast<int>(filter_.preset);
                Setting("Kernel", 320.0f);
                if (ExploreChoices("##signal_preset", kPreset, 5, &preset)) filter_.preset = static_cast<signal::KernelPreset>(preset);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Moving average 1/n; Gaussian with sigma = size / 6; difference [-1, 0, 1] / 2 (the change per row); second difference [1, -2, 1]; or the taps typed");
                if (filter_.preset == signal::KernelPreset::MovingAverage || filter_.preset == signal::KernelPreset::Gaussian) {
                    Setting("of", 100.0f);
                    ui::NumberField("##signal_ksize", &filter_.kernel_size, 2, 1 << 20, 1, 56.0f, "Taps");
                    FaintAfter("rows");
                }
                if (filter_.preset == signal::KernelPreset::Typed) {
                    Setting("Taps", 260.0f);
                    ImGui::SetNextItemWidth(220.0f);
                    if (ImGui::InputText("##signal_ktyped", kernel_text_, sizeof(kernel_text_))) filter_.kernel_typed = ParseKernel(kernel_text_);
                    if (filter_.kernel_typed.empty()) filter_.kernel_typed = ParseKernel(kernel_text_);
                    double sum = 0.0;
                    for (const double v : filter_.kernel_typed) sum += v;
                    FaintAfter(filter_.kernel_typed.empty() ? "comma-separated" : "sum " + Round(sum));
                }
            }
            break;
        }
        case Look::Wavelet: {
            Setting("Wavelet", 120.0f);
            ImGui::SetNextItemWidth(84.0f);
            if (ImGui::BeginCombo("##signal_wavelet", wavelet_.wavelet.c_str())) {
                for (const auto& name : SignalProcessing::WaveletNames())
                    if (ImGui::Selectable(name.c_str(), name == wavelet_.wavelet)) wavelet_.wavelet = name;
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("haar; Daubechies db1..db8; symlets sym2..sym8; coiflets coif1..coif5 (PyWavelets' filters)");
            Setting("Levels", 150.0f);
            ui::NumberField("##signal_levels", &wavelet_.levels, 0, 40, 1, 48.0f, "How many times the signal is split in two; 0: 4 or the maximum for the rows");
            if (p.rows > 0) {
                const int most = SignalProcessing::WaveletMaxLevel(static_cast<int>(p.rows), wavelet_.wavelet);
                if (most >= 1) FaintAfter("up to " + std::to_string(most) + " for " + explore_view::Number(p.rows) + " rows");
            }
            static const char* const kDenoise[] = {"off", "soft", "hard"};
            int denoise = static_cast<int>(wavelet_.denoise);
            Setting("Denoise", 200.0f);
            if (ExploreChoices("##signal_denoise", kDenoise, 3, &denoise)) wavelet_.denoise = static_cast<signal::DenoiseMode>(denoise);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The details thresholded: soft shrinks every coefficient by the threshold; hard zeroes those below it");
            if (wavelet_.denoise != signal::DenoiseMode::Off) {
                Setting("threshold", 150.0f);
                ui::NumberFieldOptions threshold;
                threshold.min = 0.0;
                threshold.max = 1e300;
                threshold.format = "%g";
                threshold.width = 72.0f;
                threshold.hint = "0: the universal threshold, noise SD (from D1) times sqrt(2 ln n)";
                ui::NumberField("##signal_threshold", &wavelet_.threshold, threshold);
                FaintAfter(wavelet_.threshold > 0.0 ? "typed" : "0: universal");
            }
            break;
        }
    }
}

void SignalView::RenderSignalLine(const signal::SignalData& d) {
    unit_ = d.unit;
    rate_unit_ = d.series.kind == series::TimeKind::Dates ? "Hz" : d.unit.name;
    Faint("Sample rate " + sv::RateWords(d) + "; " + sv::RowsWords(d) + ".");
}

bool SignalView::CutField(const char* id, double* cut, const char* hint) {
    ui::NumberFieldOptions o;
    o.min = 0.0;
    o.max = 1e12;
    o.format = "%g";
    o.width = 72.0f;
    o.hint = hint;
    double shown = *cut * unit_.seconds;
    if (!ui::NumberField(id, &shown, o)) return false;
    *cut = shown / unit_.seconds;
    return true;
}

// ---- Spectrum

void SignalView::RenderSpectrum(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    spectrum_.base = base_;
    spectrum_.welch_segment = welch_ ? welch_segment_ : 0;
    const std::string key = BaseKey(context.dataset) + "|spectrum|" + spectrum_.window + (spectrum_.remove_mean ? "|m" : "") + (spectrum_.remove_trend ? "|t" : "") +
                            "|" + std::to_string(spectrum_.welch_segment) + "|" + Round(spectrum_.peak_prominence_sd, "%.17g") + "|" + std::to_string(spectrum_.peak_gap_rows);
    const std::string dataset = context.dataset;
    const signal::SpectrumRequest req = spectrum_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_) spectrum_job_.Ask("Signal: spectrum", key, [dataset, req, texts](LambdaTask&) { return signal::RunSpectrum(dataset, req, RunNow, texts); });
    const signal::SpectrumAnswer* a = spectrum_job_.Answer(key);
    if (!a) {
        Waiting(spectrum_job_, key, "Computing the spectrum on the device...");
        return;
    }
    const signal::SignalData& d = a->data;
    const auto& s = d.series;
    const bool welch = a->request.welch_segment > 0;
    const double resolution = d.sample_rate / std::max(1, a->spectrum.n);
    std::vector<std::pair<std::string, std::string>> figures;
    if (!a->peaks.empty()) {
        figures.push_back({"strongest cycle", sv::PeriodWords(a->peaks[0].frequency, d) + " (" + sv::FrequencyWords(a->peaks[0].frequency, d.unit) + ")"});
        figures.push_back({welch ? "its power density" : "its amplitude", sv::Num(a->peaks[0].amplitude)});
    } else {
        figures.push_back({"strongest cycle", "none stands out"});
    }
    figures.push_back({"resolution", sv::FrequencyWords(resolution, d.unit)});
    figures.push_back({"rows", explore_view::Number(static_cast<size_t>(s.used))});
    ExploreFigures("##signal_spectrum_figures", figures);
    Wrapped(t.text, sv::SpectrumSentence(*a));
    RenderSignalLine(d);
    ImGui::Spacing();
    // The spectrum in the display unit; amplitude only without Welch.
    const Scale scale = scale_ == Scale::Amplitude && welch ? Scale::Power : scale_;
    std::vector<double> xs(a->spectrum.frequencies.size());
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = a->spectrum.frequencies[i] * d.unit.seconds;
    std::vector<double> ys(xs.size());
    for (size_t i = 0; i < ys.size(); ++i) {
        const double v = scale == Scale::Amplitude ? a->spectrum.amplitude[i] : a->spectrum.psd[i];
        ys[i] = scale == Scale::Decibel ? (v > 0.0 ? 10.0 * std::log10(v) : NAN) : v;
    }
    const float avail = ImGui::GetContentRegionAvail().x;
    bool split = avail >= 820.0f;
    // split stays true only when the table began (EndTable must not run otherwise).
    split = split && ImGui::BeginTable("##signal_spectrum_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings);
    if (split) {
        ImGui::TableSetupColumn("plots", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("peaks", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = ImGui::GetContentRegionAvail().x;
        std::string title = sv::SpectrumTitle(*a);
        std::transform(title.begin(), title.end(), title.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        Heading(title.c_str());
        const char* y_name = scale == Scale::Amplitude ? "amplitude" : scale == Scale::Power ? "power density" : "dB";
        explore_plot::Lines(xs, {{y_name, ys}}, w, PlotHeight(w), false, a->peaks.empty() ? NAN : a->peaks[0].frequency * d.unit.seconds,
                            a->peaks.empty() ? "" : sv::PeriodWords(a->peaks[0].frequency, d), d.unit.name.c_str());
        Heading("THE ROWS, WITH THE PEAKS FOUND");
        std::vector<double> marks(s.values.size(), NAN);
        for (const auto& p : a->row_peaks)
            if (p.index >= 0 && static_cast<size_t>(p.index) < marks.size()) marks[static_cast<size_t>(p.index)] = p.value;
        explore_plot::LinesOver(s.t, {{base_.value, s.values}, {"peaks", marks}}, w, PlotHeight(w), false, NAN, "", "", AxisLabel(s, 0), AxisLabel(s, s.t.size() - 1));
    }
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
    {
        Heading("PEAKS IN THE SPECTRUM");
        if (a->peaks.empty()) Faint("No bin rises a twentieth of the tallest above its surroundings.");
        else if (ImGui::BeginTable("##signal_peaks", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings,
                                   ImVec2(0.0f, TableHeight(a->peaks.size() + 1)))) {
            ImGui::TableSetupColumn("cycle", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("frequency", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("height", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("note", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Heading("cycle");
            NumberCell(d.unit.name, &t.text_dim);
            NumberCell(welch ? "power density" : "amplitude", &t.text_dim);
            ImGui::TableNextColumn();
            Heading("");
            const double top = a->peaks[0].frequency;
            for (size_t i = 0; i < a->peaks.size(); ++i) {
                const auto& p = a->peaks[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(i == 0 ? t.text : t.text_dim, "%s", sv::PeriodWords(p.frequency, d).c_str());
                NumberCell(sv::Num(p.frequency * d.unit.seconds), i == 0 ? nullptr : &t.text_dim);
                NumberCell(sv::Num(p.amplitude), i == 0 ? nullptr : &t.text_dim);
                ImGui::TableNextColumn();
                const double ratio = top > 0.0 ? p.frequency / top : 0.0;
                if (i == 0) ImGui::TextColored(t.success, "%s", "strongest");
                else if (std::fabs(ratio - std::round(ratio)) < 0.03 && std::round(ratio) >= 2.0) ImGui::TextColored(t.text_faint, "harmonic %d of it", static_cast<int>(std::round(ratio)));
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
        Heading("PEAKS IN THE ROWS");
        Label("Stand out by");
        ui::NumberFieldOptions prominence;
        prominence.min = 0.01;
        prominence.max = 1000.0;
        prominence.format = "%g";
        prominence.width = 56.0f;
        prominence.hint = "A row is a peak when it rises this many SDs above the higher of the valleys around it (scipy find_peaks prominence)";
        ui::NumberField("##signal_prominence", &spectrum_.peak_prominence_sd, prominence);
        FaintAfter("SD (" + sv::Num(spectrum_.peak_prominence_sd * a->row_sd) + ")");
        Setting("at least", 140.0f);
        int gap = spectrum_.peak_gap_rows > 0 ? spectrum_.peak_gap_rows : a->row_gap;
        if (ui::NumberField("##signal_gap", &gap, 1, 1 << 24, 0, 56.0f, "Peaks closer than this keep the taller one")) spectrum_.peak_gap_rows = gap;
        FaintAfter("rows apart (" + sv::SpanWords(gap * d.step_seconds, d) + ")");
        if (a->row_peaks.empty()) Faint("No row stands out by that much.");
        else {
            const size_t shown = std::min<size_t>(a->row_peaks.size(), 12);
            if (ImGui::BeginTable("##signal_row_peaks", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings,
                                  ImVec2(0.0f, TableHeight(shown + 1)))) {
                ImGui::TableSetupColumn("time", ImGuiTableColumnFlags_WidthStretch, 1.2f);
                ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                ImGui::TableSetupColumn("gap", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                Heading("time");
                NumberCell("value", &t.text_dim);
                NumberCell("gap", &t.text_dim);
                for (size_t i = 0; i < shown; ++i) {
                    const auto& p = a->row_peaks[i];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(RowLabel(s, static_cast<size_t>(p.index)).c_str());
                    NumberCell(sv::Num(p.value));
                    NumberCell(i == 0 ? "" : sv::SpanWords((p.index - a->row_peaks[i - 1].index) * d.step_seconds, d));
                }
                ImGui::EndTable();
            }
            std::string foot = explore_view::Number(a->row_peaks.size()) + (a->row_peaks.size() == 1 ? " peak" : " peaks");
            if (shown < a->row_peaks.size()) foot += " (the first " + explore_view::Number(shown) + " shown; Export CSV has them all)";
            if (a->row_peaks.size() >= 2) foot += "; median gap " + sv::SpanWords(a->median_gap_seconds, d);
            Faint(foot + ".");
        }
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::SecondaryButton("Keep as a node", !welch, "The FFT node has no Welch averaging; set Average to whole series to keep this spectrum as a node"))
            Message(add_to_graph("FFTNode", "Spectrum (" + base_.value + ")", signal::NodeParameters(a->request), false), false);
        if (!welch && ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", (explore_view::BranchTip("FFT") + " Its table has one row per frequency bin: frequency, period, amplitude, phase, power density, real, imag.").c_str());
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("signal_spectrum.csv", sv::SpectrumCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One row per bin: frequency, period, amplitude, power density");
    if (!a->row_peaks.empty()) {
        ExploreNextButton("Export the row peaks...");
        if (ui::SecondaryButton("Export the row peaks...")) Export("signal_row_peaks.csv", sv::RowPeaksCsv(*a));
    }
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::SpectrumSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Spectrogram

void SignalView::RenderSpectrogram(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    spectrogram_.base = base_;
    const std::string key = BaseKey(context.dataset) + "|spectrogram|" + std::to_string(spectrogram_.window_rows) + "|" + std::to_string(spectrogram_.hop_rows) + "|" +
                            spectrogram_.window + "|" + Round(spectrogram_.range_db, "%.17g");
    const std::string dataset = context.dataset;
    const signal::SpectrogramRequest req = spectrogram_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_) spectrogram_job_.Ask("Signal: spectrogram", key, [dataset, req, texts](LambdaTask&) { return signal::RunSpectrogram(dataset, req, RunNow, texts); });
    const signal::SpectrogramAnswer* a = spectrogram_job_.Answer(key);
    if (!a) {
        Waiting(spectrogram_job_, key, "Computing the frames on the device...");
        return;
    }
    const signal::SignalData& d = a->data;
    const auto& sg = a->spectrogram;
    ExploreFigures("##signal_spectrogram_figures",
                   {{"frames x bins", explore_view::Number(static_cast<size_t>(sg.num_frames)) + " x " + explore_view::Number(static_cast<size_t>(sg.num_bins))},
                    {"resolution", sv::FrequencyWords(d.sample_rate / std::max(1, a->request.window_rows), d.unit) + " x " + sv::SpanWords(a->hop_used * d.step_seconds, d)},
                    {"loudest", Round(a->loudest_db, "%.0f") + " dB at " + sv::FrequencyWords(a->loudest_frequency, d.unit)},
                    {"rows", explore_view::Number(static_cast<size_t>(d.series.used))}});
    Wrapped(t.text, sv::SpectrogramSentence(*a));
    RenderSignalLine(d);
    ImGui::Spacing();
    // The cells in dB (or power) with the floor; the frequency axis cropped to
    // the highest bin that rises above the floor (the rest is background).
    const double floor_power = std::pow(10.0, a->floor_db / 10.0);
    size_t highest = 0;
    for (const auto& frame : sg.spectrogram)
        for (size_t b = frame.size(); b-- > highest + 1;)
            if (frame[b] > floor_power) {
                highest = b;
                break;
            }
    const size_t bins = std::min(sg.frequencies.size(), highest + 2);
    std::vector<std::vector<double>> cells(sg.spectrogram.size());
    for (size_t f = 0; f < cells.size(); ++f) {
        cells[f].resize(bins);
        for (size_t b = 0; b < bins && b < sg.spectrogram[f].size(); ++b) {
            const double v = sg.spectrogram[f][b];
            cells[f][b] = spectrogram_db_ ? (v > 0.0 ? 10.0 * std::log10(v) : -INFINITY) : v;
        }
    }
    const double lo = spectrogram_db_ ? a->floor_db : floor_power, hi = spectrogram_db_ ? a->loudest_db : std::pow(10.0, a->loudest_db / 10.0);
    const float w = ImGui::GetContentRegionAvail().x;
    const std::string top_f = sv::FrequencyWords(bins > 0 ? sg.frequencies[bins - 1] : 0.0, d.unit);
    Heading(("SPECTROGRAM (" + std::string(kWindows[WindowIndex(a->request.window)]) + " " + explore_view::Number(static_cast<size_t>(a->request.window_rows)) + ", HOP " +
             explore_view::Number(static_cast<size_t>(a->hop_used)) + "), " + (spectrogram_db_ ? "DB" : "POWER DENSITY") + "; DRAWN UP TO " + top_f + ", THE REST IS BELOW THE FLOOR")
                .c_str());
    const series::Series& s = d.series;
    explore_plot::Heatmap(cells, bins, lo, hi, w, std::clamp(w * 0.4f, 180.0f, 360.0f), AxisLabel(s, 0), "", AxisLabel(s, s.t.size() - 1), "0", top_f);
    Faint(std::string(spectrogram_db_ ? "dB " : "power ") + Round(lo, "%.3g") + " .. " + Round(hi, "%.3g") + " (" + Round(a->request.range_db, "%.0f") +
          " dB range); darker is quieter, the frequency axis grows upwards.");
    ImGui::Spacing();
    if (ui::SecondaryButton("Export CSV...")) Export("signal_spectrogram.csv", sv::SpectrogramCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One row per frame and bin: time, frequency, power density");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::SpectrogramSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Filter

void SignalView::RenderFilter(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    filter_.base = base_;
    std::string key = BaseKey(context.dataset) + "|filter|";
    if (filter_.mode == signal::FilterMode::Design) {
        key += std::string("d|") + signal::KindName(filter_.kind) + "|" + Round(filter_.cut, "%.17g") + "|" + Round(filter_.cut_high, "%.17g") + "|" +
               std::to_string(filter_.taps) + "|" + filter_.window;
    } else {
        key += std::string("k|") + signal::PresetName(filter_.preset) + "|" + std::to_string(filter_.kernel_size);
        if (filter_.preset == signal::KernelPreset::Typed)
            for (const double v : filter_.kernel_typed) key += "|" + Round(v, "%.17g");
    }
    const std::string dataset = context.dataset;
    const signal::FilterRequest req = filter_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_) filter_job_.Ask("Signal: filter", key, [dataset, req, texts](LambdaTask&) { return signal::RunFilter(dataset, req, RunNow, texts); });
    const signal::FilterAnswer* a = filter_job_.Answer(key);
    if (!a) {
        Waiting(filter_job_, key, "Filtering on the device...");
        return;
    }
    const signal::SignalData& d = a->data;
    const series::Series& s = d.series;
    const bool design = a->request.mode == signal::FilterMode::Design;
    std::vector<std::pair<std::string, std::string>> figures;
    if (design && std::isfinite(a->half_power))
        figures.push_back({"half power (-3 dB) at", sv::PeriodWords(a->half_power, d) + " (" + sv::FrequencyWords(a->half_power, d.unit) + ")"});
    figures.push_back({"removed RMS", sv::Num(a->rms_removed) + " of " + sv::Num(a->rms_original)});
    figures.push_back({"kept RMS", sv::Num(a->rms_kept)});
    figures.push_back({"taps", explore_view::Number(a->kernel.size()) + " = " + sv::SpanWords(a->kernel.size() * d.step_seconds, d) + " of rows, zero-phase"});
    ExploreFigures("##signal_filter_figures", figures);
    Wrapped(t.text, sv::FilterSentence(*a));
    RenderSignalLine(d);
    ImGui::Spacing();
    const float avail = ImGui::GetContentRegionAvail().x;
    bool split = avail >= 820.0f;
    // split stays true only when the table began (EndTable must not run otherwise).
    split = split && ImGui::BeginTable("##signal_filter_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings);
    if (split) {
        ImGui::TableSetupColumn("series", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("response", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading((base_.value + " AND THE FILTERED COLUMN").c_str());
        explore_plot::LinesOver(s.t, {{base_.value, s.values}, {"filtered", a->filtered}}, w, PlotHeight(w), false, NAN, "", "", AxisLabel(s, 0), AxisLabel(s, s.t.size() - 1));
    }
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading(("FREQUENCY RESPONSE, DB, " + d.unit.name).c_str());
        std::vector<double> xs(a->filter.freq_axis.size()), ys(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            xs[i] = a->filter.freq_axis[i] * d.unit.seconds;
            const double m = i < a->filter.freq_response_mag.size() ? a->filter.freq_response_mag[i] : 0.0;
            ys[i] = m > 0.0 ? std::max(-120.0, 20.0 * std::log10(m)) : -120.0;
        }
        explore_plot::Lines(xs, {{"|H| dB", ys}}, w, PlotHeight(w), false, design && std::isfinite(a->half_power) ? a->half_power * d.unit.seconds : NAN, "-3 dB",
                            d.unit.name.c_str(), 2);
        if (!design) {
            std::string taps;
            for (size_t i = 0; i < a->kernel.size() && i < 12; ++i) taps += (i ? ", " : "") + Round(a->kernel[i], "%.4g");
            if (a->kernel.size() > 12) taps += ", ...";
            double sum = 0.0;
            for (const double v : a->kernel) sum += v;
            Faint("Kernel [" + taps + "], sum " + Round(sum) + ".");
        }
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        const char* type = signal::FilterNodeType(a->request);
        const std::string node = design ? "Filter Designer" : "Convolution 1D";
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph(type, (design ? "Filter " : "Convolve ") + base_.value, signal::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", explore_view::StepTip(node, "the column " + base_.value + "_filtered beside the others").c_str());
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("signal_filtered.csv", sv::FilterCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One row per row: time, the column, the filtered column");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::FilterSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Wavelet

void SignalView::RenderWavelet(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    wavelet_.base = base_;
    const std::string key = BaseKey(context.dataset) + "|wavelet|" + wavelet_.wavelet + "|" + std::to_string(wavelet_.levels) + "|" +
                            std::to_string(static_cast<int>(wavelet_.denoise)) + "|" + Round(wavelet_.threshold, "%.17g");
    const std::string dataset = context.dataset;
    const signal::WaveletRequest req = wavelet_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_) wavelet_job_.Ask("Signal: wavelet", key, [dataset, req, texts](LambdaTask&) { return signal::RunWavelet(dataset, req, RunNow, texts); });
    const signal::WaveletAnswer* a = wavelet_job_.Answer(key);
    if (!a) {
        Waiting(wavelet_job_, key, "Splitting the signal by scale on the device...");
        return;
    }
    const signal::SignalData& d = a->data;
    const series::Series& s = d.series;
    std::vector<std::pair<std::string, std::string>> figures;
    for (const auto& part : a->parts) figures.push_back({sv::PartScale(part, d), Round(100.0 * part.energy_share, part.energy_share >= 0.1 ? "%.0f" : "%.1f") + "% of the energy"});
    figures.push_back({"reconstruction error", sv::Num(a->reconstruction_error)});
    if (a->denoised.success) figures.push_back({"threshold", sv::Num(a->denoised.threshold) + " (noise SD " + sv::Num(a->denoised.sigma) + ")"});
    ExploreFigures("##signal_wavelet_figures", figures);
    Wrapped(t.text, sv::WaveletSentence(*a));
    RenderSignalLine(d);
    ImGui::Spacing();
    const float avail = ImGui::GetContentRegionAvail().x;
    bool split = avail >= 820.0f;
    // split stays true only when the table began (EndTable must not run otherwise).
    split = split && ImGui::BeginTable("##signal_wavelet_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings);
    if (split) {
        ImGui::TableSetupColumn("parts", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("energy", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = ImGui::GetContentRegionAvail().x;
        const std::string lo = AxisLabel(s, 0), hi = AxisLabel(s, s.t.size() - 1);
        Heading(("THE COLUMN AND ITS APPROXIMATION (" + a->request.wavelet + ", " + std::to_string(a->request.levels) + " LEVELS)").c_str());
        if (!a->parts.empty())
            explore_plot::LinesOver(s.t, {{base_.value, s.values}, {a->parts[0].name, a->parts[0].rows}}, w, PlotHeight(w), false, NAN, "", "", lo, hi);
        for (size_t i = 1; i < a->parts.size(); ++i) {
            Heading((sv::PartScale(a->parts[i], d)).c_str());
            explore_plot::LinesOver(s.t, {{a->parts[i].name, a->parts[i].rows}}, w, PlotHeight(w) * 0.5f, false, NAN, "", "", lo, hi, 1 + (i % 5));
        }
    }
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
    {
        Heading("ENERGY BY SCALE");
        if (ImGui::BeginTable("##signal_parts", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings,
                              ImVec2(0.0f, TableHeight(a->parts.size() + 1)))) {
            ImGui::TableSetupColumn("part", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("scale", ImGuiTableColumnFlags_WidthStretch, 1.4f);
            ImGui::TableSetupColumn("coefficients", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("energy", ImGuiTableColumnFlags_WidthStretch, 0.7f);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Heading("part");
            ImGui::TableNextColumn();
            Heading("scale");
            NumberCell("coefficients", &t.text_dim);
            NumberCell("energy", &t.text_dim);
            for (const auto& part : a->parts) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(part.approximation ? t.text : t.text_dim, "%s", part.name.c_str());
                ImGui::TableNextColumn();
                const std::string scale = sv::PartScale(part, d);
                ImGui::TextUnformatted(scale.substr(scale.find(": ") == std::string::npos ? 0 : scale.find(": ") + 2).c_str());
                NumberCell(explore_view::Number(part.coefficients));
                NumberCell(Round(100.0 * part.energy_share, part.energy_share >= 0.1 ? "%.0f" : "%.1f") + "%");
            }
            ImGui::EndTable();
        }
        if (a->denoised.success) {
            ImGui::Spacing();
            const float w = ImGui::GetContentRegionAvail().x;
            Heading((std::string("DENOISED (") + (a->request.denoise == signal::DenoiseMode::Soft ? "SOFT" : "HARD") + ", THRESHOLD " + sv::Num(a->denoised.threshold) + ")").c_str());
            explore_plot::LinesOver(s.t, {{base_.value, s.values}, {"denoised", a->denoised.denoised}}, w, PlotHeight(w), false, NAN, "", "", AxisLabel(s, 0),
                                    AxisLabel(s, s.t.size() - 1), 2);
        }
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph("WaveletTransform", "Wavelet parts of " + base_.value, signal::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered()) {
            std::string columns = base_.value + "_A" + std::to_string(a->request.levels) + ", " + base_.value + "_D" + std::to_string(a->request.levels) + " .. " + base_.value + "_D1";
            if (a->request.denoise != signal::DenoiseMode::Off) columns += " and " + base_.value + "_denoised";
            ImGui::SetTooltip("%s", explore_view::StepTip("Wavelet Transform", "the columns " + columns + " beside the others").c_str());
        }
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("signal_wavelet.csv", sv::WaveletCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One row per row: time, the column, each part, the denoised column");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::WaveletSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

void SignalView::Export(const std::string& name, const std::string& csv) {
    if (const auto path = FileDialogs::SaveFile("Export signal", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << csv;
        Message(out ? "Saved " + *path : "Could not write " + *path, !out);
    }
}

}  // namespace cyxwiz
