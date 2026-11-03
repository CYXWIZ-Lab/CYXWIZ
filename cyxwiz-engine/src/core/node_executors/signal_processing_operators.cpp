#include "signal_processing_operators.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/signal_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    double progress,
                    uint64_t processed = 0,
                    uint64_t total = 0,
                    uint64_t memory = 0) {
    if (!callback) {
        return;
    }
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.status = "running";
    event.progress = static_cast<float>(progress);
    event.processed_items = processed;
    event.total_items = total;
    event.estimated_memory_bytes = memory;
    callback(event);
}

bool ParseCsvDoubles(const std::string& s, std::vector<double>& out,
                     std::string& bad_token) {
    out.clear();
    std::vector<std::string> tokens;
    ParseCommaList(s, tokens);
    for (const auto& t : tokens) {
        try {
            out.push_back(std::stod(t));
        } catch (...) {
            bad_token = t;
            return false;
        }
    }
    return true;
}

// One numeric column as doubles, exactly (an Arrow cast to float64, no
// float32 hop), nulls as NaN; `out_col_idx` is its index.
arrow::Status ReadColumnAsDouble(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<double>& out,
    int& out_col_idx) {
    out_col_idx = input->schema()->GetFieldIndex(col_name);
    if (out_col_idx < 0) {
        return arrow::Status::KeyError(
            op_name + ": signal column '" + col_name + "' not found");
    }
    const auto col = input->column(out_col_idx);
    if (!arrow::is_numeric(col->type()->id()) && col->type()->id() != arrow::Type::BOOL) {
        return arrow::Status::TypeError(
            op_name + ": signal column '" + col_name +
            "' must be numeric (got '" + col->type()->ToString() + "')");
    }
    ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(col, arrow::float64()));
    const auto doubles = cast.chunked_array();
    out.clear();
    out.reserve(static_cast<size_t>(doubles->length()));
    for (const auto& chunk : doubles->chunks()) {
        const auto array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        for (int64_t i = 0; i < array->length(); ++i)
            out.push_back(array->IsNull(i) ? std::numeric_limits<double>::quiet_NaN() : array->Value(i));
    }
    return arrow::Status::OK();
}

arrow::Result<int64_t> CheckedSizeForArrow(size_t size, const std::string& context) {
    if (size > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
        return arrow::Status::CapacityError(context + ": row count exceeds Arrow int64 capacity");
    }
    return static_cast<int64_t>(size);
}

// A float64 column of `values` (NaN as null) appended to the table under `name`.
arrow::Result<std::shared_ptr<arrow::Table>> AddDoubleColumn(
    const std::shared_ptr<arrow::Table>& table, const std::string& name, const std::vector<double>& values) {
    if (table->schema()->GetFieldIndex(name) >= 0) {
        return arrow::Status::Invalid("the table already has a column '" + name + "'");
    }
    arrow::DoubleBuilder builder(arrow::default_memory_pool());
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (const double v : values) {
        if (std::isfinite(v)) builder.UnsafeAppend(v);
        else builder.UnsafeAppendNull();
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return table->AddColumn(table->num_columns(), arrow::field(name, arrow::float64()), std::make_shared<arrow::ChunkedArray>(array));
}

std::string PreflightMessage(const std::string& op_name,
                             const MaterializationMemoryEstimate& estimate,
                             const MaterializationMemoryDecision& decision) {
    std::ostringstream ss;
    ss << op_name << " memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", samples=" << estimate.rows
       << ", planned_columns=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: reduce signal rows, or window or sample the signal first.";
    return ss.str();
}

// The memory preflight every signal operator runs: `columns` float64 columns
// of `rows`; the estimate is reported and a blocked plan stops the operator.
arrow::Result<uint64_t> Preflight(const std::string& op_name, uint64_t rows, uint64_t columns,
                                  const PipelineOperatorProgressCallback& callback, const MaterializationMemoryContext& context) {
    const auto estimate = EstimateDenseMaterializationMemory(rows, columns, static_cast<uint64_t>(sizeof(double)));
    const auto decision = EvaluateMaterializationMemory(estimate, context);
    const std::string message = PreflightMessage(op_name, estimate, decision);
    uint64_t cells = 0;
    if (!CheckedMulU64(rows, columns, cells)) cells = std::numeric_limits<uint64_t>::max();
    if (callback) {
        PipelineOperatorProgress event;
        event.stage = op_name + " memory preflight";
        event.message = message;
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        event.processed_items = 0;
        event.total_items = cells;
        callback(event);
    }
    if (decision.blocked) return arrow::Status::CapacityError("Materialization blocked: " + message);
    return estimate.estimated_peak_bytes;
}

// The signal column's checks shared by the operators: present, numeric, not empty.
arrow::Result<uint64_t> SignalRows(const std::shared_ptr<arrow::Table>& input, const std::string& signal_col, const std::string& op_name) {
    if (!input) return arrow::Status::Invalid(op_name + ": input table is null");
    const int col_idx = input->schema()->GetFieldIndex(signal_col);
    if (col_idx < 0) return arrow::Status::KeyError(op_name + ": signal column '" + signal_col + "' not found");
    const auto column = input->column(col_idx);
    if (!IsNumericChunked(column)) {
        const std::string got = column && column->num_chunks() > 0 ? column->chunk(0)->type()->ToString() : "<empty>";
        return arrow::Status::TypeError(op_name + ": signal column '" + signal_col + "' must be numeric (got '" + got + "')");
    }
    if (column->length() == 0) return arrow::Status::Invalid(op_name + ": signal column is empty");
    return static_cast<uint64_t>(column->length());
}

bool ReadDouble(const std::map<std::string, std::string>& params, const char* name, double& value, const std::string& op, std::string& error) {
    const auto it = params.find(name);
    if (it == params.end() || it->second.empty()) return true;
    try {
        value = std::stod(it->second);
    } catch (...) {
        error = op + ": '" + name + "' is not a valid float: " + it->second;
        return false;
    }
    return true;
}

bool ReadInt(const std::map<std::string, std::string>& params, const char* name, int& value, const std::string& op, std::string& error) {
    const auto it = params.find(name);
    if (it == params.end() || it->second.empty()) return true;
    try {
        value = std::stoi(it->second);
    } catch (...) {
        error = op + ": '" + name + "' is not a valid integer: " + it->second;
        return false;
    }
    return true;
}

bool KnownWindow(const std::string& name) {
    return name == "hann" || name == "hamming" || name == "blackman" || name == "boxcar";
}

} // namespace

// ============================================================================
// FFTOperator: the one-sided spectrum as a table (a branch)
// ============================================================================

bool FFTOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    sample_rate_ = 1.0;
    window_ = "hann";
    remove_mean_ = true;
    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "FFT: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;
    if (!ReadDouble(params, "sample_rate", sample_rate_, "FFT", error)) return false;
    if (sample_rate_ <= 0.0) {
        error = "FFT: sample_rate must be > 0 (got " + std::to_string(sample_rate_) + ")";
        return false;
    }
    if (const auto w = params.find("window"); w != params.end() && !w->second.empty()) {
        window_ = NormalizeTimeSeriesParameterChoice(w->second);
        if (!KnownWindow(window_)) {
            error = "FFT: 'window' must be hann, hamming, blackman or boxcar (got '" + window_ + "')";
            return false;
        }
    }
    if (const auto m = params.find("remove_mean"); m != params.end() && !m->second.empty()) {
        const std::string v = NormalizeTimeSeriesParameterChoice(m->second);
        remove_mean_ = v == "true" || v == "1" || v == "yes";
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
FFTOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz FFT Materializer");
    ARROW_ASSIGN_OR_RAISE(const uint64_t rows, SignalRows(input, signal_col_, "FFT"));
    ARROW_ASSIGN_OR_RAISE(const uint64_t memory, Preflight("FFT", rows / 2 + 1, 7, progress_callback_, GetMaterializationMemoryContext()));
    ReportProgress(progress_callback_, "Reading signal", "Reading FFT signal column '" + signal_col_ + "'", 0.10, 0, rows, memory);
    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "FFT", signal, col_idx));
    ReportProgress(progress_callback_, "Computing spectrum", "One-sided spectrum of " + std::to_string(signal.size()) + " samples", 0.35, 0, rows, memory);
    const SpectrumResult spectrum = SignalProcessing::Spectrum(signal, sample_rate_, window_, remove_mean_);
    if (!spectrum.success) {
        return arrow::Status::ExecutionError("FFT: " + spectrum.error_message);
    }
    const size_t bins = spectrum.frequencies.size();
    if (bins == 0 || spectrum.amplitude.size() != bins || spectrum.psd.size() != bins || spectrum.complex_output.size() != bins) {
        return arrow::Status::ExecutionError("FFT: the backend returned inconsistent bin counts");
    }
    ARROW_ASSIGN_OR_RAISE(const int64_t nbins, CheckedSizeForArrow(bins, "FFT"));
    ReportProgress(progress_callback_, "Building table", "Building the spectrum table", 0.70, 0, static_cast<uint64_t>(nbins), memory);
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::DoubleBuilder frequency(pool), period(pool), amplitude(pool), phase(pool), power(pool), real(pool), imag(pool);
    arrow::DoubleBuilder* builders[7] = {&frequency, &period, &amplitude, &phase, &power, &real, &imag};
    for (arrow::DoubleBuilder* b : builders) ARROW_RETURN_NOT_OK(b->Reserve(nbins));
    for (size_t k = 0; k < bins; ++k) {
        const double f = spectrum.frequencies[k];
        frequency.UnsafeAppend(f);
        if (f > 0.0) period.UnsafeAppend(1.0 / f);
        else period.UnsafeAppendNull();
        amplitude.UnsafeAppend(spectrum.amplitude[k]);
        phase.UnsafeAppend(std::arg(spectrum.complex_output[k]));
        power.UnsafeAppend(spectrum.psd[k]);
        real.UnsafeAppend(spectrum.complex_output[k].real());
        imag.UnsafeAppend(spectrum.complex_output[k].imag());
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(7);
    for (size_t i = 0; i < 7; ++i) ARROW_RETURN_NOT_OK(builders[i]->Finish(&arrays[i]));
    auto schema = arrow::schema({
        arrow::field("frequency", arrow::float64()),
        arrow::field("period", arrow::float64()),
        arrow::field("amplitude", arrow::float64()),
        arrow::field("phase", arrow::float64()),
        arrow::field("power_density", arrow::float64()),
        arrow::field("real", arrow::float64()),
        arrow::field("imag", arrow::float64()),
    });
    auto table = arrow::Table::Make(schema, arrays, nbins);
    spdlog::info("FFT: {} samples -> {} one-sided bins (sample_rate={}, window={}, mean {})", signal.size(), nbins, sample_rate_, window_,
                 remove_mean_ ? "removed" : "kept");
    ReportProgress(progress_callback_, "Complete", "FFT materialization complete", 1.0, static_cast<uint64_t>(nbins), static_cast<uint64_t>(nbins), memory);
    return table;
}

// ============================================================================
// IFFTOperator: the spectrum table back to a signal
// ============================================================================

bool IFFTOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    samples_ = 0;
    if (!ReadInt(params, "samples", samples_, "IFFT", error)) return false;
    if (samples_ < 0) {
        error = "IFFT: samples must be 0 (even length from the bins) or the signal's length";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> IFFTOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz IFFT Materializer");
    if (!input) return arrow::Status::Invalid("IFFT: input table is null");
    for (const char* name : {"real", "imag"}) {
        if (input->schema()->GetFieldIndex(name) < 0)
            return arrow::Status::KeyError(std::string("IFFT: the table has no '") + name + "' column; it reads the FFT node's table (real, imag)");
    }
    const uint64_t bins = static_cast<uint64_t>(input->num_rows());
    if (bins == 0) return arrow::Status::Invalid("IFFT: the spectrum table is empty");
    const uint64_t samples = samples_ > 0 ? static_cast<uint64_t>(samples_) : 2 * (bins - 1);
    ARROW_ASSIGN_OR_RAISE(const uint64_t memory, Preflight("IFFT", samples, 1, progress_callback_, GetMaterializationMemoryContext()));
    std::vector<double> re, im;
    int idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, "real", "IFFT", re, idx));
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, "imag", "IFFT", im, idx));
    std::vector<std::complex<double>> half(re.size());
    for (size_t k = 0; k < re.size(); ++k) {
        if (!std::isfinite(re[k]) || !std::isfinite(im[k])) return arrow::Status::Invalid("IFFT: bin " + std::to_string(k) + " is empty");
        half[k] = {re[k], im[k]};
    }
    ReportProgress(progress_callback_, "Inverting spectrum", "Rebuilding " + std::to_string(samples) + " samples from " + std::to_string(bins) + " bins", 0.4, 0, samples, memory);
    const std::vector<double> signal = SignalProcessing::InverseSpectrum(half, samples_);
    if (signal.empty()) {
        return arrow::Status::Invalid("IFFT: " + std::to_string(bins) + " bins do not fit a signal of " + std::to_string(samples) +
                                      " samples (a one-sided spectrum has samples / 2 + 1 bins); set 'samples' to the FFT node's row count");
    }
    arrow::DoubleBuilder builder(arrow::default_memory_pool());
    ARROW_RETURN_NOT_OK(builder.AppendValues(signal));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    auto table = arrow::Table::Make(arrow::schema({arrow::field("signal", arrow::float64())}), {array}, static_cast<int64_t>(signal.size()));
    spdlog::info("IFFT: {} bins -> {} samples", bins, signal.size());
    ReportProgress(progress_callback_, "Complete", "IFFT materialization complete", 1.0, samples, samples, memory);
    return table;
}

// ============================================================================
// Convolve1DOperator: a kernel applied zero-phase, a column added
// ============================================================================

bool Convolve1DOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    kernel_.clear();
    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "Convolve1D: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;
    auto k = params.find("kernel");
    if (k == params.end() || k->second.empty()) {
        error = "Convolve1D: 'kernel' parameter is required "
                "(comma-sep floats, e.g. '0.25,0.5,0.25')";
        return false;
    }
    std::string bad;
    if (!ParseCsvDoubles(k->second, kernel_, bad)) {
        error = "Convolve1D: 'kernel' token '" + bad + "' is not a valid float";
        return false;
    }
    if (kernel_.empty()) {
        error = "Convolve1D: 'kernel' parsed to empty list";
        return false;
    }
    for (const double v : kernel_) {
        if (!std::isfinite(v)) {
            error = "Convolve1D: 'kernel' has a value that is not a finite number";
            return false;
        }
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
Convolve1DOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz Convolve1D Materializer");
    ARROW_ASSIGN_OR_RAISE(const uint64_t rows, SignalRows(input, signal_col_, "Convolve1D"));
    ARROW_ASSIGN_OR_RAISE(const uint64_t memory, Preflight("Convolve1D", rows, 3, progress_callback_, GetMaterializationMemoryContext()));
    if (kernel_.size() > rows) {
        return arrow::Status::Invalid("Convolve1D: the kernel has " + std::to_string(kernel_.size()) + " taps but the column only " +
                                      std::to_string(rows) + " rows");
    }
    ReportProgress(progress_callback_, "Reading signal", "Reading Convolve1D signal column '" + signal_col_ + "'", 0.10, 0, rows, memory);
    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "Convolve1D", signal, col_idx));
    ReportProgress(progress_callback_, "Convolving signal", "Applying the kernel of " + std::to_string(kernel_.size()) + " taps, centred", 0.45, 0, rows, memory);
    FilterCoefficients filter;
    filter.b = kernel_;
    filter.a = {1.0};
    const std::vector<double> filtered = SignalProcessing::ApplyFilter(signal, filter);
    if (filtered.size() != signal.size()) {
        return arrow::Status::ExecutionError("Convolve1D: the backend returned " + std::to_string(filtered.size()) + " samples, expected " +
                                             std::to_string(signal.size()));
    }
    const std::string name = signal_col_ + "_filtered";
    ReportProgress(progress_callback_, "Adding column", "Adding " + name, 0.85, rows, rows, memory);
    ARROW_ASSIGN_OR_RAISE(auto out, AddDoubleColumn(input, name, filtered));
    spdlog::info("Convolve1D: {} samples, kernel of {} taps, column '{}' added", signal.size(), kernel_.size(), name);
    ReportProgress(progress_callback_, "Complete", "Convolve1D materialization complete", 1.0, rows, rows, memory);
    return out;
}

// ============================================================================
// FilterDesignerOperator: a firwin FIR designed and applied, a column added
// ============================================================================

bool FilterDesignerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    filter_type_ = "lowpass";
    cutoff_ = 0.1;
    cutoff_high_ = 0.0;
    sample_rate_ = 1.0;
    order_ = 15;
    taps_ = 0;
    window_ = "hamming";
    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "FilterDesigner: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;
    if (const auto ft = params.find("filter_type"); ft != params.end() && !ft->second.empty()) {
        filter_type_ = NormalizeTimeSeriesParameterChoice(ft->second);
        if (filter_type_ != "lowpass" && filter_type_ != "highpass" && filter_type_ != "bandpass" && filter_type_ != "bandstop") {
            error = "FilterDesigner: 'filter_type' must be lowpass/highpass/bandpass/bandstop (got '" + filter_type_ + "')";
            return false;
        }
    }
    if (!ReadDouble(params, "cutoff", cutoff_, "FilterDesigner", error)) return false;
    if (!ReadDouble(params, "cutoff_high", cutoff_high_, "FilterDesigner", error)) return false;
    if (!ReadDouble(params, "sample_rate", sample_rate_, "FilterDesigner", error)) return false;
    if (!ReadInt(params, "order", order_, "FilterDesigner", error)) return false;
    // `taps` is the number of coefficients (scipy.signal.firwin); without it
    // the old `order` still means 2 * order + 1 taps.
    if (!ReadInt(params, "taps", taps_, "FilterDesigner", error)) return false;
    if (const auto w = params.find("window"); w != params.end() && !w->second.empty()) {
        window_ = NormalizeTimeSeriesParameterChoice(w->second);
        if (!KnownWindow(window_)) {
            error = "FilterDesigner: 'window' must be hamming, hann, blackman or boxcar (got '" + window_ + "')";
            return false;
        }
    }
    if (sample_rate_ <= 0.0) {
        error = "FilterDesigner: sample_rate must be > 0";
        return false;
    }
    if (cutoff_ <= 0.0) {
        error = "FilterDesigner: cutoff must be > 0";
        return false;
    }
    if (cutoff_ >= sample_rate_ / 2.0) {
        error = "FilterDesigner: cutoff must be below the Nyquist frequency " + std::to_string(sample_rate_ / 2.0);
        return false;
    }
    if (order_ < 1) {
        error = "FilterDesigner: order must be >= 1 (got " + std::to_string(order_) + ")";
        return false;
    }
    if (taps_ == 0) taps_ = 2 * order_ + 1;
    if (taps_ < 3) {
        error = "FilterDesigner: taps must be at least 3 (got " + std::to_string(taps_) + ")";
        return false;
    }
    if ((filter_type_ == "highpass" || filter_type_ == "bandstop") && taps_ % 2 == 0) {
        error = "FilterDesigner: a " + filter_type_ + " filter needs an odd number of taps (got " + std::to_string(taps_) + ")";
        return false;
    }
    if ((filter_type_ == "bandpass" || filter_type_ == "bandstop") && cutoff_high_ <= cutoff_) {
        error = "FilterDesigner: '" + filter_type_ + "' requires cutoff_high (" + std::to_string(cutoff_high_) + ") > cutoff (" +
                std::to_string(cutoff_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
FilterDesignerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz FilterDesigner Materializer");
    ARROW_ASSIGN_OR_RAISE(const uint64_t rows, SignalRows(input, signal_col_, "FilterDesigner"));
    ARROW_ASSIGN_OR_RAISE(const uint64_t memory, Preflight("FilterDesigner", rows, 3, progress_callback_, GetMaterializationMemoryContext()));
    if (static_cast<uint64_t>(taps_) > rows) {
        return arrow::Status::Invalid("FilterDesigner: " + std::to_string(taps_) + " taps but the column only has " + std::to_string(rows) + " rows");
    }
    ReportProgress(progress_callback_, "Reading signal", "Reading FilterDesigner signal column '" + signal_col_ + "'", 0.10, 0, rows, memory);
    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "FilterDesigner", signal, col_idx));
    ReportProgress(progress_callback_, "Designing filter", "Designing the " + filter_type_ + " filter", 0.35, 0, rows, memory);
    const FilterCoefficients filter = SignalProcessing::DesignFIR(filter_type_, cutoff_, cutoff_high_, sample_rate_, taps_, window_);
    if (!filter.success) {
        return arrow::Status::ExecutionError("FilterDesigner: filter design failed: " + filter.error_message);
    }
    ReportProgress(progress_callback_, "Applying filter", "Applying the filter, centred", 0.60, 0, rows, memory);
    const std::vector<double> filtered = SignalProcessing::ApplyFilter(signal, filter);
    if (filtered.size() != signal.size()) {
        return arrow::Status::ExecutionError("FilterDesigner: ApplyFilter returned " + std::to_string(filtered.size()) + " samples, expected " +
                                             std::to_string(signal.size()));
    }
    const std::string name = signal_col_ + "_filtered";
    ReportProgress(progress_callback_, "Adding column", "Adding " + name, 0.85, rows, rows, memory);
    ARROW_ASSIGN_OR_RAISE(auto out, AddDoubleColumn(input, name, filtered));
    spdlog::info("FilterDesigner: type={}, cutoff={}{}, taps={}, window={}, sample_rate={}, {} samples, column '{}' added",
                 filter_type_, cutoff_, (filter_type_ == "bandpass" || filter_type_ == "bandstop") ? ("/" + std::to_string(cutoff_high_)) : std::string(),
                 taps_, window_, sample_rate_, signal.size(), name);
    ReportProgress(progress_callback_, "Complete", "FilterDesigner materialization complete", 1.0, rows, rows, memory);
    return out;
}

// ============================================================================
// WaveletTransformOperator: the parts (and the denoised signal) as columns
// ============================================================================

bool WaveletTransformOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    wavelet_ = "db4";
    level_ = 3;
    denoise_ = "off";
    threshold_ = 0.0;
    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "WaveletTransform: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;
    if (const auto w = params.find("wavelet"); w != params.end() && !w->second.empty()) {
        wavelet_ = NormalizeTimeSeriesParameterChoice(w->second);
        const auto names = SignalProcessing::WaveletNames();
        if (std::find(names.begin(), names.end(), wavelet_) == names.end()) {
            error = "WaveletTransform: unknown wavelet '" + wavelet_ + "' (haar, db1..db8, sym2..sym8, coif1..coif5)";
            return false;
        }
    }
    if (!ReadInt(params, "level", level_, "WaveletTransform", error)) return false;
    if (level_ < 1) {
        error = "WaveletTransform: level must be at least 1 (got " + std::to_string(level_) + ")";
        return false;
    }
    if (const auto d = params.find("denoise"); d != params.end() && !d->second.empty()) {
        denoise_ = NormalizeTimeSeriesParameterChoice(d->second);
        if (denoise_ != "off" && denoise_ != "soft" && denoise_ != "hard") {
            error = "WaveletTransform: 'denoise' must be off, soft or hard (got '" + denoise_ + "')";
            return false;
        }
    }
    if (!ReadDouble(params, "threshold", threshold_, "WaveletTransform", error)) return false;
    if (threshold_ < 0.0) {
        error = "WaveletTransform: threshold must be 0 (the universal threshold) or positive";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> WaveletTransformOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz WaveletTransform Materializer");
    ARROW_ASSIGN_OR_RAISE(const uint64_t rows, SignalRows(input, signal_col_, "WaveletTransform"));
    const uint64_t columns = static_cast<uint64_t>(level_) + 2 + (denoise_ == "off" ? 0 : 1);
    ARROW_ASSIGN_OR_RAISE(const uint64_t memory, Preflight("WaveletTransform", rows, columns, progress_callback_, GetMaterializationMemoryContext()));
    ReportProgress(progress_callback_, "Reading signal", "Reading WaveletTransform signal column '" + signal_col_ + "'", 0.10, 0, rows, memory);
    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "WaveletTransform", signal, col_idx));
    ReportProgress(progress_callback_, "Decomposing", wavelet_ + " over " + std::to_string(level_) + " levels", 0.30, 0, rows, memory);
    const WaveletResult coefficients = SignalProcessing::DWT(signal, wavelet_, level_);
    if (!coefficients.success) {
        return arrow::Status::ExecutionError("WaveletTransform: " + coefficients.error_message);
    }
    // Each part alone, back to the rows: the approximation, then the coarsest detail to the finest.
    const auto alone = [&](int keep_level) {
        WaveletResult only = coefficients;
        if (keep_level >= 0) std::fill(only.approximation.begin(), only.approximation.end(), 0.0);
        for (int level = 0; level < level_; ++level)
            if (level != keep_level) std::fill(only.details[static_cast<size_t>(level)].begin(), only.details[static_cast<size_t>(level)].end(), 0.0);
        return SignalProcessing::IDWT(only);
    };
    std::shared_ptr<arrow::Table> out = input;
    const std::string approximation = signal_col_ + "_A" + std::to_string(level_);
    ReportProgress(progress_callback_, "Adding parts", "Adding " + approximation, 0.5, 0, rows, memory);
    std::vector<double> part = alone(-1);
    if (part.size() != signal.size()) return arrow::Status::ExecutionError("WaveletTransform: the approximation could not be brought back to the rows");
    ARROW_ASSIGN_OR_RAISE(out, AddDoubleColumn(out, approximation, part));
    for (int level = level_; level >= 1; --level) {
        const std::string name = signal_col_ + "_D" + std::to_string(level);
        part = alone(level - 1);
        if (part.size() != signal.size()) return arrow::Status::ExecutionError("WaveletTransform: " + name + " could not be brought back to the rows");
        ARROW_ASSIGN_OR_RAISE(out, AddDoubleColumn(out, name, part));
    }
    if (denoise_ != "off") {
        ReportProgress(progress_callback_, "Denoising", "Thresholding the details", 0.85, 0, rows, memory);
        const DenoiseResult denoised = SignalProcessing::WaveletDenoise(signal, wavelet_, level_, threshold_, denoise_ == "soft");
        if (!denoised.success) return arrow::Status::ExecutionError("WaveletTransform: " + denoised.error_message);
        ARROW_ASSIGN_OR_RAISE(out, AddDoubleColumn(out, signal_col_ + "_denoised", denoised.denoised));
        spdlog::info("WaveletTransform: {} {} levels, {} denoising at {} (noise {}), {} samples", wavelet_, level_, denoise_, denoised.threshold,
                     denoised.sigma, signal.size());
    } else {
        spdlog::info("WaveletTransform: {} {} levels, {} samples, {} part columns added", wavelet_, level_, signal.size(), level_ + 1);
    }
    ReportProgress(progress_callback_, "Complete", "WaveletTransform materialization complete", 1.0, rows, rows, memory);
    return out;
}

} // namespace cyxwiz
