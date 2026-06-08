#include "signal_processing_operators.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/signal_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

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
    if (!input) return arrow::Status::Invalid("FFT: input table is null");

    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(input, signal_col_, "FFT", signal, col_idx));

    if (signal.empty()) {
        return arrow::Status::Invalid("FFT: signal column is empty");
    }

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
    if (!input) return arrow::Status::Invalid("Convolve1D: input table is null");

    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(
        input, signal_col_, "Convolve1D", signal, col_idx));

    if (signal.empty()) {
        return arrow::Status::Invalid("Convolve1D: signal column is empty");
    }

    // "same" mode keeps row count aligned with all other input columns.
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
    ARROW_ASSIGN_OR_RAISE(
        auto new_table,
        ReplaceColumnWithFloat(input, col_idx, out_floats, row_count));

    spdlog::info("Convolve1D: {} samples, kernel size {}, column '{}' replaced",
                 signal.size(), kernel_.size(), signal_col_);
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
        filter_type_ = ft->second;
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
    if (!input) return arrow::Status::Invalid("FilterDesigner: input table is null");

    std::vector<double> signal;
    int col_idx = -1;
    ARROW_RETURN_NOT_OK(ReadColumnAsDouble(
        input, signal_col_, "FilterDesigner", signal, col_idx));

    if (signal.empty()) {
        return arrow::Status::Invalid("FilterDesigner: signal column is empty");
    }

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
    ARROW_ASSIGN_OR_RAISE(
        auto new_table,
        ReplaceColumnWithFloat(input, col_idx, out_floats, row_count));

    spdlog::info("FilterDesigner: type={}, cutoff={}{}, order={}, "
                 "sample_rate={}, {} samples filtered",
                 filter_type_, cutoff_,
                 (filter_type_ == "bandpass" || filter_type_ == "bandstop")
                    ? ("/" + std::to_string(cutoff_high_)) : std::string(),
                 order_, sample_rate_, signal.size());
    return new_table;
}

} // namespace cyxwiz
