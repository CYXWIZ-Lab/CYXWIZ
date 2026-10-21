#include "signal_processing_operators.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/signal_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
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

// Read one numeric column and return it as std::vector<double> suitable
// for the signal-processing backend (which uses double internally).
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
    auto col = input->column(out_col_idx);
    std::vector<float> floats;
    std::string bad;
    if (!ReadColumnAsFloat(col, floats, bad)) {
        return arrow::Status::TypeError(
            op_name + ": signal column '" + col_name +
            "' must be numeric (got '" + bad + "')");
    }
    out.assign(floats.begin(), floats.end());
    return arrow::Status::OK();
}

arrow::Result<int64_t> CheckedSizeForArrow(size_t size, const std::string& context) {
    if (size > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
        return arrow::Status::CapacityError(context + ": row count exceeds Arrow int64 capacity");
    }
    return static_cast<int64_t>(size);
}

std::string BuildFftMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision) {
    std::ostringstream ss;
    ss << "FFT memory preflight: risk="
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
       << ". Suggestion: reduce signal rows, window the signal first, "
          "or use a future chunked/sampled FFT materialization path.";
    return ss.str();
}

std::string BuildSignalReplacementMemoryPreflightMessage(
    const std::string& op_name,
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
       << ". Suggestion: reduce signal rows, filter a sampled/windowed "
          "signal first, or use a future chunked signal materialization path.";
    return ss.str();
}

} // namespace

// ============================================================================
// FFTOperator
// ============================================================================

bool FFTOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    signal_col_.clear();
    sample_rate_ = 1.0;

    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "FFT: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;

    auto sr = params.find("sample_rate");
    if (sr != params.end() && !sr->second.empty()) {
        try { sample_rate_ = std::stod(sr->second); }
        catch (...) {
            error = "FFT: 'sample_rate' is not a valid float: " + sr->second;
            return false;
        }
    }
    if (sample_rate_ <= 0.0) {
        error = "FFT: sample_rate must be > 0 (got " +
                std::to_string(sample_rate_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
FFTOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz FFT Materializer");

    if (!input) return arrow::Status::Invalid("FFT: input table is null");

    int col_idx = input->schema()->GetFieldIndex(signal_col_);
    if (col_idx < 0) {
        return arrow::Status::KeyError(
            "FFT: signal column '" + signal_col_ + "' not found");
    }
    auto signal_column = input->column(col_idx);
    if (!IsNumericChunked(signal_column)) {
        std::string got = signal_column && signal_column->num_chunks() > 0
            ? signal_column->chunk(0)->type()->ToString()
            : "<empty>";
        return arrow::Status::TypeError(
            "FFT: signal column '" + signal_col_ +
            "' must be numeric (got '" + got + "')");
    }
    const uint64_t planned_samples =
        static_cast<uint64_t>(std::max<int64_t>(0, signal_column->length()));
    if (planned_samples == 0) {
        return arrow::Status::Invalid("FFT: signal column is empty");
    }
    const uint64_t planned_columns = 4;
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_samples, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message = BuildFftMemoryPreflightMessage(
        preflight_estimate, preflight_decision);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_samples, planned_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "FFT memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading FFT signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    int read_col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "FFT", signal, read_col_idx));
    col_idx = read_col_idx;

    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Computing FFT",
                   "Computing FFT over " + std::to_string(signal.size()) +
                   " samples",
                   0.35,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);

    auto result = SignalProcessing::FFT(signal, sample_rate_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "FFT: backend FFT failed: " + result.error_message);
    }

    ARROW_ASSIGN_OR_RAISE(
        const int64_t nbins,
        CheckedSizeForArrow(result.magnitude.size(), "FFT"));
    if (nbins == 0 ||
        result.phase.size() != result.magnitude.size() ||
        result.frequencies.size() != result.magnitude.size()) {
        return arrow::Status::ExecutionError(
            "FFT: backend returned inconsistent bin counts "
            "(mag=" + std::to_string(result.magnitude.size()) +
            ", phase=" + std::to_string(result.phase.size()) +
            ", freq=" + std::to_string(result.frequencies.size()) + ")");
    }

    const auto output_memory_plan = EstimateDenseMaterializationMemory(
        static_cast<uint64_t>(nbins), 3, static_cast<uint64_t>(sizeof(double)));
    const uint64_t estimated_output_bytes =
        output_memory_plan.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Building Arrow table",
                   "Building FFT frequency-domain output table",
                   0.70,
                   0,
                   static_cast<uint64_t>(nbins),
                   estimated_output_bytes);
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::DoubleBuilder freq_builder(pool);
    arrow::DoubleBuilder mag_builder(pool);
    arrow::DoubleBuilder phase_builder(pool);
    ARROW_RETURN_NOT_OK(freq_builder.Reserve(nbins));
    ARROW_RETURN_NOT_OK(mag_builder.Reserve(nbins));
    ARROW_RETURN_NOT_OK(phase_builder.Reserve(nbins));
    for (int64_t i = 0; i < nbins; ++i) {
        ARROW_RETURN_NOT_OK(freq_builder.Append(result.frequencies[i]));
        ARROW_RETURN_NOT_OK(mag_builder.Append(result.magnitude[i]));
        ARROW_RETURN_NOT_OK(phase_builder.Append(result.phase[i]));
    }

    std::shared_ptr<arrow::Array> freq_arr, mag_arr, phase_arr;
    ARROW_RETURN_NOT_OK(freq_builder.Finish(&freq_arr));
    ARROW_RETURN_NOT_OK(mag_builder.Finish(&mag_arr));
    ARROW_RETURN_NOT_OK(phase_builder.Finish(&phase_arr));

    auto schema = arrow::schema({
        arrow::field("frequency", arrow::float64()),
        arrow::field("magnitude", arrow::float64()),
        arrow::field("phase", arrow::float64()),
    });
    auto table = arrow::Table::Make(
        schema, {freq_arr, mag_arr, phase_arr}, nbins);

    spdlog::info("FFT: {} samples -> {} bins, sample_rate={}",
                 signal.size(), nbins, sample_rate_);
    ReportProgress(progress_callback_, "Complete",
                   "FFT materialization complete",
                   1.0,
                   static_cast<uint64_t>(nbins),
                   static_cast<uint64_t>(nbins),
                   estimated_output_bytes);
    return table;
}

// ============================================================================
// Convolve1DOperator
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
        error = "Convolve1D: 'kernel' token '" + bad +
                "' is not a valid float";
        return false;
    }
    if (kernel_.empty()) {
        error = "Convolve1D: 'kernel' parsed to empty list";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
Convolve1DOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz Convolve1D Materializer");

    if (!input) return arrow::Status::Invalid("Convolve1D: input table is null");

    int col_idx = input->schema()->GetFieldIndex(signal_col_);
    if (col_idx < 0) {
        return arrow::Status::KeyError(
            "Convolve1D: signal column '" + signal_col_ + "' not found");
    }
    auto signal_column = input->column(col_idx);
    if (!IsNumericChunked(signal_column)) {
        std::string got = signal_column && signal_column->num_chunks() > 0
            ? signal_column->chunk(0)->type()->ToString()
            : "<empty>";
        return arrow::Status::TypeError(
            "Convolve1D: signal column '" + signal_col_ +
            "' must be numeric (got '" + got + "')");
    }
    const uint64_t planned_samples =
        static_cast<uint64_t>(std::max<int64_t>(0, signal_column->length()));
    if (planned_samples == 0) {
        return arrow::Status::Invalid("Convolve1D: signal column is empty");
    }
    const uint64_t planned_columns = 3;
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_samples, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message =
        BuildSignalReplacementMemoryPreflightMessage(
            "Convolve1D", preflight_estimate, preflight_decision);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_samples, planned_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "Convolve1D memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading Convolve1D signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    int read_col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(
        input, signal_col_, "Convolve1D", signal, read_col_idx));
    col_idx = read_col_idx;

    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;

    // "same" mode keeps row count aligned with all other input columns.
    ReportProgress(progress_callback_, "Convolving signal",
                   "Applying Convolve1D kernel with " +
                   std::to_string(kernel_.size()) + " taps",
                   0.45,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    auto result = SignalProcessing::Convolve1D(signal, kernel_, "same");
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "Convolve1D: backend failed: " + result.error_message);
    }

    if (result.output.size() != signal.size()) {
        return arrow::Status::ExecutionError(
            "Convolve1D: 'same' mode produced " +
            std::to_string(result.output.size()) +
            " samples, expected " + std::to_string(signal.size()));
    }

    ARROW_ASSIGN_OR_RAISE(
        const int64_t row_count,
        CheckedSizeForArrow(signal.size(), "Convolve1D"));
    std::vector<float> out_floats;
    out_floats.reserve(result.output.size());
    for (double value : result.output) {
        out_floats.push_back(static_cast<float>(value));
    }
    ReportProgress(progress_callback_, "Replacing column",
                   "Writing Convolve1D output column",
                   0.85,
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()) * sizeof(float));
    ARROW_ASSIGN_OR_RAISE(
        auto new_table,
        ReplaceColumnWithFloat(input, col_idx, out_floats, row_count));

    spdlog::info("Convolve1D: {} samples, kernel size {}, column '{}' replaced",
                 signal.size(), kernel_.size(), signal_col_);
    ReportProgress(progress_callback_, "Complete",
                   "Convolve1D materialization complete",
                   1.0,
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()) * sizeof(float));
    return new_table;
}

// ============================================================================
// FilterDesignerOperator
// ============================================================================

bool FilterDesignerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    signal_col_.clear();
    filter_type_ = "lowpass";
    cutoff_ = 0.5;
    cutoff_high_ = 0.0;
    sample_rate_ = 1.0;
    order_ = 4;

    auto it = params.find("signal_col");
    if (it == params.end() || it->second.empty()) {
        error = "FilterDesigner: 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = it->second;

    auto ft = params.find("filter_type");
    if (ft != params.end() && !ft->second.empty()) {
        filter_type_ = NormalizeTimeSeriesParameterChoice(ft->second);
        if (filter_type_ != "lowpass" && filter_type_ != "highpass" &&
            filter_type_ != "bandpass" && filter_type_ != "bandstop") {
            error = "FilterDesigner: 'filter_type' must be lowpass/highpass/"
                    "bandpass/bandstop (got '" + filter_type_ + "')";
            return false;
        }
    }

    auto c = params.find("cutoff");
    if (c != params.end() && !c->second.empty()) {
        try { cutoff_ = std::stod(c->second); }
        catch (...) {
            error = "FilterDesigner: 'cutoff' is not a valid float: " + c->second;
            return false;
        }
    }

    auto ch = params.find("cutoff_high");
    if (ch != params.end() && !ch->second.empty()) {
        try { cutoff_high_ = std::stod(ch->second); }
        catch (...) {
            error = "FilterDesigner: 'cutoff_high' is not a valid float: " + ch->second;
            return false;
        }
    }

    auto sr = params.find("sample_rate");
    if (sr != params.end() && !sr->second.empty()) {
        try { sample_rate_ = std::stod(sr->second); }
        catch (...) {
            error = "FilterDesigner: 'sample_rate' is not a valid float: " + sr->second;
            return false;
        }
    }

    auto o = params.find("order");
    if (o != params.end() && !o->second.empty()) {
        try { order_ = std::stoi(o->second); }
        catch (...) {
            error = "FilterDesigner: 'order' is not a valid integer: " + o->second;
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
    if (order_ < 1) {
        error = "FilterDesigner: order must be >= 1 (got " +
                std::to_string(order_) + ")";
        return false;
    }
    if ((filter_type_ == "bandpass" || filter_type_ == "bandstop") &&
        cutoff_high_ <= cutoff_) {
        error = "FilterDesigner: '" + filter_type_ +
                "' requires cutoff_high (" + std::to_string(cutoff_high_) +
                ") > cutoff (" + std::to_string(cutoff_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
FilterDesignerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz FilterDesigner Materializer");

    if (!input) return arrow::Status::Invalid("FilterDesigner: input table is null");

    int col_idx = input->schema()->GetFieldIndex(signal_col_);
    if (col_idx < 0) {
        return arrow::Status::KeyError(
            "FilterDesigner: signal column '" + signal_col_ + "' not found");
    }
    auto signal_column = input->column(col_idx);
    if (!IsNumericChunked(signal_column)) {
        std::string got = signal_column && signal_column->num_chunks() > 0
            ? signal_column->chunk(0)->type()->ToString()
            : "<empty>";
        return arrow::Status::TypeError(
            "FilterDesigner: signal column '" + signal_col_ +
            "' must be numeric (got '" + got + "')");
    }
    const uint64_t planned_samples =
        static_cast<uint64_t>(std::max<int64_t>(0, signal_column->length()));
    if (planned_samples == 0) {
        return arrow::Status::Invalid("FilterDesigner: signal column is empty");
    }
    const uint64_t planned_columns = 3;
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_samples, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message =
        BuildSignalReplacementMemoryPreflightMessage(
            "FilterDesigner", preflight_estimate, preflight_decision);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_samples, planned_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "FilterDesigner memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading FilterDesigner signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    int read_col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(
        input, signal_col_, "FilterDesigner", signal, read_col_idx));
    col_idx = read_col_idx;

    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;

    ReportProgress(progress_callback_, "Designing filter",
                   "Designing " + filter_type_ + " filter",
                   0.35,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    FilterCoefficients filter;
    if (filter_type_ == "lowpass") {
        filter = SignalProcessing::DesignLowpass(cutoff_, sample_rate_, order_);
    } else if (filter_type_ == "highpass") {
        filter = SignalProcessing::DesignHighpass(cutoff_, sample_rate_, order_);
    } else if (filter_type_ == "bandpass") {
        filter = SignalProcessing::DesignBandpass(
            cutoff_, cutoff_high_, sample_rate_, order_);
    } else {
        filter = SignalProcessing::DesignBandstop(
            cutoff_, cutoff_high_, sample_rate_, order_);
    }
    if (!filter.success) {
        return arrow::Status::ExecutionError(
            "FilterDesigner: filter design failed: " + filter.error_message);
    }

    ReportProgress(progress_callback_, "Applying filter",
                   "Applying designed filter to signal",
                   0.60,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    auto filtered = SignalProcessing::ApplyFilter(signal, filter);
    if (filtered.size() != signal.size()) {
        return arrow::Status::ExecutionError(
            "FilterDesigner: ApplyFilter returned " +
            std::to_string(filtered.size()) + " samples, expected " +
            std::to_string(signal.size()));
    }

    ARROW_ASSIGN_OR_RAISE(
        const int64_t row_count,
        CheckedSizeForArrow(signal.size(), "FilterDesigner"));
    std::vector<float> out_floats;
    out_floats.reserve(filtered.size());
    for (double value : filtered) {
        out_floats.push_back(static_cast<float>(value));
    }
    ReportProgress(progress_callback_, "Replacing column",
                   "Writing filtered signal column",
                   0.85,
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()) * sizeof(float));
    ARROW_ASSIGN_OR_RAISE(
        auto new_table,
        ReplaceColumnWithFloat(input, col_idx, out_floats, row_count));

    spdlog::info("FilterDesigner: type={}, cutoff={}{}, order={}, "
                 "sample_rate={}, {} samples filtered",
                 filter_type_, cutoff_,
                 (filter_type_ == "bandpass" || filter_type_ == "bandstop")
                    ? ("/" + std::to_string(cutoff_high_)) : std::string(),
                 order_, sample_rate_, signal.size());
    ReportProgress(progress_callback_, "Complete",
                   "FilterDesigner materialization complete",
                   1.0,
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()),
                   static_cast<uint64_t>(out_floats.size()) * sizeof(float));
    return new_table;
}

} // namespace cyxwiz
