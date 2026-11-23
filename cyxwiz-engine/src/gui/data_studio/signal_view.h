#pragma once

// Data Studio > Explore > Signal (TOFIX129 D7, approved mockup boards 1-4;
// replaces the FFT, Spectrogram, Filter Designer, Convolution Calculator and
// Wavelet Transform windows). One signal column over the time column D6's
// reader gives, its sample rate from the step or typed, in four looks:
// Spectrum (amplitude or Welch power, the peaks in the spectrum and in the
// rows), Spectrogram (STFT heatmap), Filter (a designed FIR or a kernel,
// applied zero-phase; the Convolution window folded in as Kernel mode) and
// Wavelet (the parts by scale, energy, denoising). Worked out in the
// background (core/signal_explore, core/signal_presentation).

#include "explore_tab.h"
#include "../../core/signal_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

class SignalView {
public:
    enum class Look { Spectrum, Spectrogram, Filter, Wavelet };

    void Render(const ExploreContext& context);
    // Opens on this look (Tools > Signal Processing); the Convolution item opens Filter in Kernel mode.
    void SelectLook(Look look, bool kernel = false);
    bool IsRunning() const { return spectrum_job_.Running() || spectrogram_job_.Running() || filter_job_.Running() || wavelet_job_.Running(); }

    // Into the graph: a node of `type` (its import name, "FilterDesigner")
    // named `name` with `params`, after the data in use (`step`) or as a
    // branch reading it; returns what happened.
    std::function<std::string(const std::string& type, const std::string& name, std::map<std::string, std::string> params, bool step)> add_to_graph;

private:
    enum class Scale { Amplitude, Power, Decibel };

    void RenderControls(const ExploreContext& context, const DatasetProfile& profile);
    void RenderLookSettings(const DatasetProfile& profile);
    void RenderSpectrum(const ExploreContext& context);
    void RenderSpectrogram(const ExploreContext& context);
    void RenderFilter(const ExploreContext& context);
    void RenderWavelet(const ExploreContext& context);
    void RenderSignalLine(const signal::SignalData& d);
    // A cut frequency typed in the display unit (kept per second / unit / row).
    bool CutField(const char* id, double* cut, const char* hint);
    void Message(const std::string& text, bool error);
    // What shows while a look runs (the text, and Stop); false when stopped.
    template <typename T> bool Waiting(ExploreJob<T>& job, const std::string& key, const char* text);
    void Export(const std::string& name, const std::string& csv);
    std::string BaseKey(const std::string& dataset) const;

    std::shared_ptr<const DatasetProfile> seen_;
    Look look_ = Look::Spectrum;
    signal::SignalRequest base_;
    signal::FrequencyUnit unit_;      // the display unit of the last answer (the cut is typed in it)
    std::string rate_unit_;           // "Hz", "per unit", "per row": the sample rate's unit, from the last answer
    signal::SpectrumRequest spectrum_;
    bool welch_ = false;            // Welch over segments (spectrum_.welch_segment is the size)
    int welch_segment_ = 1024;
    bool welch_chosen_ = false;     // the user picked; else on above 100,000 rows
    Scale scale_ = Scale::Amplitude;
    signal::SpectrogramRequest spectrogram_;
    bool spectrogram_db_ = true;
    signal::FilterRequest filter_;
    char kernel_text_[256] = "0.25, 0.5, 0.25";
    signal::WaveletRequest wavelet_;
    std::string message_;
    bool message_error_ = false;
    std::string stopped_key_;
    plot::ColumnPicker picker_;
    ExploreJob<signal::SpectrumAnswer> spectrum_job_;
    ExploreJob<signal::SpectrogramAnswer> spectrogram_job_;
    ExploreJob<signal::FilterAnswer> filter_job_;
    ExploreJob<signal::WaveletAnswer> wavelet_job_;
};

}  // namespace cyxwiz
