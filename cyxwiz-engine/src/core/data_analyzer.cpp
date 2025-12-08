#include "data_analyzer.h"
#include "../data/data_table.h"
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ========== DataProfile Methods ==========

const ColumnProfile* DataProfile::GetColumn(const std::string& name) const {
    for (const auto& col : columns) {
        if (col.name == name) {
            return &col;
        }
    }
    return nullptr;
}

ColumnProfile* DataProfile::GetColumn(const std::string& name) {
    for (auto& col : columns) {
        if (col.name == name) {
            return &col;
        }
    }
    return nullptr;
}

// ========== CorrelationMatrix Methods ==========

double CorrelationMatrix::Get(const std::string& col1, const std::string& col2) const {
    size_t i = 0, j = 0;
    bool found_i = false, found_j = false;

    for (size_t k = 0; k < column_names.size(); k++) {
        if (column_names[k] == col1) { i = k; found_i = true; }
        if (column_names[k] == col2) { j = k; found_j = true; }
    }

    if (found_i && found_j) {
        return Get(i, j);
    }
    return 0.0;
}

// ========== Statistical Functions ==========

double DataAnalyzer::Mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double DataAnalyzer::Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

double DataAnalyzer::StdDev(const std::vector<double>& values, double mean) {
    return std::sqrt(Variance(values, mean));
}

double DataAnalyzer::Variance(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(values.size() - 1);  // Sample variance
}

double DataAnalyzer::Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(values.begin(), values.end());
    if (p >= 100.0) return *std::max_element(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    double index = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper || upper >= values.size()) {
        return values[lower];
    }

    double fraction = index - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

double DataAnalyzer::Skewness(const std::vector<double>& values, double mean, double std_dev) {
    if (values.size() < 3 || std_dev == 0.0) return 0.0;

    double n = static_cast<double>(values.size());
    double sum_cubed = 0.0;
    for (double v : values) {
        double diff = (v - mean) / std_dev;
        sum_cubed += diff * diff * diff;
    }

    // Adjusted Fisher-Pearson standardized moment coefficient
    double skew = (n / ((n - 1.0) * (n - 2.0))) * sum_cubed;
    return skew;
}

double DataAnalyzer::Kurtosis(const std::vector<double>& values, double mean, double std_dev) {
    if (values.size() < 4 || std_dev == 0.0) return 0.0;

    double n = static_cast<double>(values.size());
    double sum_fourth = 0.0;
    for (double v : values) {
        double diff = (v - mean) / std_dev;
        double sq = diff * diff;
        sum_fourth += sq * sq;
    }

    // Excess kurtosis (Fisher's definition)
    double kurt = ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth
                  - (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));
    return kurt;
}

double DataAnalyzer::MAD(std::vector<double> values, double median) {
    if (values.empty()) return 0.0;

    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double v : values) {
        deviations.push_back(std::abs(v - median));
    }

    return Median(deviations);
}

double DataAnalyzer::PearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;

    double n = static_cast<double>(x.size());
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;

    for (size_t i = 0; i < x.size(); i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

// ========== Utility Functions ==========

bool DataAnalyzer::IsNull(const std::variant<std::string, double, int64_t, std::monostate>& value) {
    if (std::holds_alternative<std::monostate>(value)) return true;
    if (std::holds_alternative<std::string>(value)) {
        const auto& str = std::get<std::string>(value);
        return str.empty() || str == "NA" || str == "N/A" || str == "null" ||
               str == "NULL" || str == "nan" || str == "NaN" || str == "None";
    }
    if (std::holds_alternative<double>(value)) {
        return std::isnan(std::get<double>(value));
    }
    return false;
}

std::optional<double> DataAnalyzer::ToDouble(const std::variant<std::string, double, int64_t, std::monostate>& value) {
    if (std::holds_alternative<double>(value)) {
        double d = std::get<double>(value);
        if (!std::isnan(d) && !std::isinf(d)) return d;
        return std::nullopt;
    }
    if (std::holds_alternative<int64_t>(value)) {
        return static_cast<double>(std::get<int64_t>(value));
    }
    if (std::holds_alternative<std::string>(value)) {
        const auto& str = std::get<std::string>(value);
        if (str.empty()) return std::nullopt;
        try {
            size_t pos = 0;
            double d = std::stod(str, &pos);
            if (pos == str.size() && !std::isnan(d) && !std::isinf(d)) {
                return d;
            }
        } catch (...) {}
    }
    return std::nullopt;
}

std::vector<double> DataAnalyzer::GetNumericValues(const DataTable& table, size_t column_index) {
    std::vector<double> values;
    values.reserve(table.GetRowCount());

    for (size_t i = 0; i < table.GetRowCount(); i++) {
        auto opt = ToDouble(table.GetCell(i, column_index));
        if (opt.has_value()) {
            values.push_back(opt.value());
        }
    }

    return values;
}

ColumnDataType DataAnalyzer::DetectColumnType(const DataTable& table, size_t column_index) {
    size_t numeric_count = 0;
    size_t integer_count = 0;
    size_t string_count = 0;
    size_t bool_count = 0;
    size_t valid_count = 0;

    for (size_t i = 0; i < std::min(table.GetRowCount(), size_t(1000)); i++) {
        auto cell = table.GetCell(i, column_index);

        if (IsNull(cell)) continue;
        valid_count++;

        if (std::holds_alternative<double>(cell)) {
            double d = std::get<double>(cell);
            if (d == std::floor(d) && std::abs(d) < 1e15) {
                integer_count++;
            }
            numeric_count++;
        } else if (std::holds_alternative<int64_t>(cell)) {
            integer_count++;
            numeric_count++;
        } else if (std::holds_alternative<std::string>(cell)) {
            const auto& str = std::get<std::string>(cell);
            // Check if it's a boolean
            if (str == "true" || str == "false" || str == "True" || str == "False" ||
                str == "TRUE" || str == "FALSE" || str == "0" || str == "1") {
                bool_count++;
            }
            // Check if it's numeric
            try {
                size_t pos = 0;
                double d = std::stod(str, &pos);
                if (pos == str.size()) {
                    if (d == std::floor(d) && std::abs(d) < 1e15) {
                        integer_count++;
                    }
                    numeric_count++;
                } else {
                    string_count++;
                }
            } catch (...) {
                string_count++;
            }
        }
    }

    if (valid_count == 0) return ColumnDataType::Unknown;

    double numeric_ratio = static_cast<double>(numeric_count) / valid_count;
    double integer_ratio = static_cast<double>(integer_count) / valid_count;
    double string_ratio = static_cast<double>(string_count) / valid_count;
    double bool_ratio = static_cast<double>(bool_count) / valid_count;

    if (bool_ratio > 0.9) return ColumnDataType::Boolean;
    if (numeric_ratio > 0.9) {
        if (integer_ratio > 0.9) return ColumnDataType::Integer;
        return ColumnDataType::Numeric;
    }
    if (string_ratio > 0.5) return ColumnDataType::Categorical;
    if (numeric_ratio > 0.5 || string_ratio > 0.5) return ColumnDataType::Mixed;

    return ColumnDataType::Unknown;
}

// ========== Histogram and Top Values ==========

std::vector<HistogramBin> DataAnalyzer::BuildHistogram(const std::vector<double>& values,
                                                        double min_val, double max_val,
                                                        int num_bins) {
    std::vector<HistogramBin> histogram(num_bins);

    if (values.empty() || min_val >= max_val) {
        return histogram;
    }

    double bin_width = (max_val - min_val) / num_bins;

    // Initialize bins
    for (int i = 0; i < num_bins; i++) {
        histogram[i].low = min_val + i * bin_width;
        histogram[i].high = min_val + (i + 1) * bin_width;
        histogram[i].count = 0;
    }

    // Count values in each bin
    for (double v : values) {
        int bin_index = static_cast<int>((v - min_val) / bin_width);
        if (bin_index < 0) bin_index = 0;
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        histogram[bin_index].count++;
    }

    // Calculate percentages
    double total = static_cast<double>(values.size());
    for (auto& bin : histogram) {
        bin.percentage = static_cast<float>(bin.count) / static_cast<float>(total) * 100.0f;
    }

    return histogram;
}

std::vector<TopValue> DataAnalyzer::GetTopValues(const std::vector<std::string>& values, int top_n) {
    std::unordered_map<std::string, size_t> counts;
    for (const auto& v : values) {
        counts[v]++;
    }

    std::vector<std::pair<std::string, size_t>> sorted_counts(counts.begin(), counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<TopValue> result;
    double total = static_cast<double>(values.size());

    for (int i = 0; i < std::min(static_cast<int>(sorted_counts.size()), top_n); i++) {
        TopValue tv;
        tv.value = sorted_counts[i].first;
        tv.count = sorted_counts[i].second;
        tv.percentage = static_cast<float>(tv.count) / static_cast<float>(total) * 100.0f;
        result.push_back(tv);
    }

    return result;
}

// ========== Column Profiling ==========

ColumnProfile DataAnalyzer::ProfileColumn(const DataTable& table, size_t column_index,
                                           int histogram_bins, int top_n) {
    ColumnProfile profile;

    if (column_index >= table.GetColumnCount()) {
        return profile;
    }

    const auto& headers = table.GetHeaders();
    profile.name = column_index < headers.size() ? headers[column_index] : "Column " + std::to_string(column_index);
    profile.total_count = table.GetRowCount();

    // Detect type
    profile.dtype = DetectColumnType(table, column_index);

    // Collect values
    std::vector<double> numeric_values;
    std::vector<std::string> string_values;
    std::unordered_map<std::string, size_t> unique_values;

    for (size_t i = 0; i < table.GetRowCount(); i++) {
        auto cell = table.GetCell(i, column_index);

        if (IsNull(cell)) {
            profile.null_count++;
            continue;
        }

        profile.non_null_count++;

        // Track unique values
        std::string str_val = table.GetCellAsString(i, column_index);
        unique_values[str_val]++;

        // Collect numeric values
        auto opt = ToDouble(cell);
        if (opt.has_value()) {
            numeric_values.push_back(opt.value());
        }

        // Collect string values for categorical
        if (profile.dtype == ColumnDataType::Categorical) {
            string_values.push_back(str_val);
        }
    }

    profile.unique_count = unique_values.size();
    profile.null_percentage = profile.total_count > 0 ?
        static_cast<float>(profile.null_count) / profile.total_count * 100.0f : 0.0f;

    // Compute numeric statistics
    if (!numeric_values.empty() && profile.IsNumeric()) {
        profile.min = *std::min_element(numeric_values.begin(), numeric_values.end());
        profile.max = *std::max_element(numeric_values.begin(), numeric_values.end());
        profile.sum = std::accumulate(numeric_values.begin(), numeric_values.end(), 0.0);
        profile.mean = Mean(numeric_values);
        profile.median = Median(numeric_values);
        profile.std_dev = StdDev(numeric_values, profile.mean);
        profile.variance = Variance(numeric_values, profile.mean);
        profile.q1 = Percentile(numeric_values, 25.0);
        profile.q3 = Percentile(numeric_values, 75.0);
        profile.iqr = profile.q3 - profile.q1;

        if (profile.std_dev > 0) {
            profile.skewness = Skewness(numeric_values, profile.mean, profile.std_dev);
            profile.kurtosis = Kurtosis(numeric_values, profile.mean, profile.std_dev);
        }

        // Build histogram
        profile.histogram = BuildHistogram(numeric_values, profile.min, profile.max, histogram_bins);
    }

    // Get top values for categorical
    if (profile.dtype == ColumnDataType::Categorical && !string_values.empty()) {
        profile.top_values = GetTopValues(string_values, top_n);
    }

    // Estimate memory usage
    profile.memory_estimate = profile.non_null_count * sizeof(double);  // Rough estimate

    return profile;
}

ColumnProfile DataAnalyzer::ProfileColumn(const DataTable& table, const std::string& column_name,
                                           int histogram_bins, int top_n) {
    const auto& headers = table.GetHeaders();
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == column_name) {
            return ProfileColumn(table, i, histogram_bins, top_n);
        }
    }
    return ColumnProfile();
}

// ========== Table Profiling ==========

DataProfile DataAnalyzer::ProfileTable(const DataTable& table, int histogram_bins, int top_n) {
    DataProfile profile;
    profile.source_name = table.GetName();
    profile.row_count = table.GetRowCount();
    profile.column_count = table.GetColumnCount();

    spdlog::debug("Profiling table '{}' with {} rows and {} columns",
                  profile.source_name, profile.row_count, profile.column_count);

    // Profile each column
    for (size_t i = 0; i < table.GetColumnCount(); i++) {
        auto col_profile = ProfileColumn(table, i, histogram_bins, top_n);
        profile.total_nulls += col_profile.null_count;
        profile.memory_estimate += col_profile.memory_estimate;
        profile.columns.push_back(std::move(col_profile));
    }

    // Calculate overall null percentage
    size_t total_cells = profile.row_count * profile.column_count;
    profile.null_percentage = total_cells > 0 ?
        static_cast<float>(profile.total_nulls) / total_cells * 100.0f : 0.0f;

    return profile;
}

// ========== Correlation Analysis ==========

CorrelationMatrix DataAnalyzer::ComputeCorrelationMatrix(const DataTable& table) {
    CorrelationMatrix result;
    result.row_count = table.GetRowCount();

    // Find numeric columns
    std::vector<size_t> numeric_cols;
    const auto& headers = table.GetHeaders();

    for (size_t i = 0; i < table.GetColumnCount(); i++) {
        auto dtype = DetectColumnType(table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_cols.push_back(i);
            result.column_names.push_back(i < headers.size() ? headers[i] : "Col" + std::to_string(i));
        }
    }

    if (numeric_cols.empty()) {
        spdlog::warn("No numeric columns found for correlation matrix");
        return result;
    }

    // Get numeric values for each column
    std::vector<std::vector<double>> column_values;
    for (size_t col : numeric_cols) {
        column_values.push_back(GetNumericValues(table, col));
    }

    // Compute correlation matrix
    size_t n = numeric_cols.size();
    result.matrix.resize(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
        result.matrix[i][i] = 1.0;  // Diagonal is always 1

        for (size_t j = i + 1; j < n; j++) {
            // Align values (only use rows where both have values)
            std::vector<double> x, y;
            const auto& vals_i = column_values[i];
            const auto& vals_j = column_values[j];

            // Get paired values (assumes same row indices)
            size_t min_len = std::min(vals_i.size(), vals_j.size());
            for (size_t k = 0; k < min_len; k++) {
                x.push_back(vals_i[k]);
                y.push_back(vals_j[k]);
            }

            double corr = PearsonCorrelation(x, y);
            result.matrix[i][j] = corr;
            result.matrix[j][i] = corr;  // Symmetric
        }
    }

    return result;
}

CorrelationResult DataAnalyzer::ComputeCorrelation(const DataTable& table,
                                                    const std::string& col1,
                                                    const std::string& col2) {
    CorrelationResult result;
    result.col1 = col1;
    result.col2 = col2;

    const auto& headers = table.GetHeaders();
    size_t idx1 = SIZE_MAX, idx2 = SIZE_MAX;

    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == col1) idx1 = i;
        if (headers[i] == col2) idx2 = i;
    }

    if (idx1 == SIZE_MAX || idx2 == SIZE_MAX) {
        spdlog::warn("Column not found for correlation: {} or {}", col1, col2);
        return result;
    }

    auto vals1 = GetNumericValues(table, idx1);
    auto vals2 = GetNumericValues(table, idx2);

    // Align by using paired indices
    std::vector<double> x, y;
    size_t min_len = std::min(vals1.size(), vals2.size());
    for (size_t i = 0; i < min_len; i++) {
        x.push_back(vals1[i]);
        y.push_back(vals2[i]);
    }

    result.sample_count = x.size();
    result.pearson = PearsonCorrelation(x, y);

    return result;
}

// ========== Missing Value Analysis ==========

MissingValueAnalysis DataAnalyzer::AnalyzeMissingValues(const DataTable& table,
                                                         size_t max_indices_per_column) {
    MissingValueAnalysis result;
    result.source_name = table.GetName();
    result.total_cells = table.GetRowCount() * table.GetColumnCount();

    const auto& headers = table.GetHeaders();

    // Analyze each column
    for (size_t col = 0; col < table.GetColumnCount(); col++) {
        MissingValueAnalysis::ColumnMissing cm;
        cm.name = col < headers.size() ? headers[col] : "Column " + std::to_string(col);

        for (size_t row = 0; row < table.GetRowCount(); row++) {
            if (IsNull(table.GetCell(row, col))) {
                cm.missing_count++;
                if (cm.missing_indices.size() < max_indices_per_column) {
                    cm.missing_indices.push_back(row);
                }
            }
        }

        cm.missing_percentage = table.GetRowCount() > 0 ?
            static_cast<float>(cm.missing_count) / table.GetRowCount() * 100.0f : 0.0f;

        result.total_missing += cm.missing_count;
        result.columns.push_back(std::move(cm));
    }

    // Count rows with any missing values
    for (size_t row = 0; row < table.GetRowCount(); row++) {
        bool has_missing = false;
        for (size_t col = 0; col < table.GetColumnCount(); col++) {
            if (IsNull(table.GetCell(row, col))) {
                has_missing = true;
                break;
            }
        }
        if (has_missing) {
            result.rows_with_missing++;
        }
    }

    result.complete_rows = table.GetRowCount() - result.rows_with_missing;
    result.missing_percentage = result.total_cells > 0 ?
        static_cast<float>(result.total_missing) / result.total_cells * 100.0f : 0.0f;
    result.rows_with_missing_percentage = table.GetRowCount() > 0 ?
        static_cast<float>(result.rows_with_missing) / table.GetRowCount() * 100.0f : 0.0f;

    return result;
}

// ========== Outlier Detection ==========

OutlierResult DataAnalyzer::DetectOutliers(const DataTable& table, size_t column_index,
                                            OutlierMethod method, double parameter) {
    OutlierResult result;

    if (column_index >= table.GetColumnCount()) {
        return result;
    }

    const auto& headers = table.GetHeaders();
    result.column_name = column_index < headers.size() ?
        headers[column_index] : "Column " + std::to_string(column_index);
    result.method = method;

    // Get numeric values with their original indices
    std::vector<std::pair<size_t, double>> indexed_values;
    for (size_t i = 0; i < table.GetRowCount(); i++) {
        auto opt = ToDouble(table.GetCell(i, column_index));
        if (opt.has_value()) {
            indexed_values.push_back({i, opt.value()});
        }
    }

    if (indexed_values.empty()) {
        return result;
    }

    result.total_valid = indexed_values.size();

    // Extract just values for statistics
    std::vector<double> values;
    values.reserve(indexed_values.size());
    for (const auto& iv : indexed_values) {
        values.push_back(iv.second);
    }

    // Compute base statistics
    result.mean = Mean(values);
    result.std_dev = StdDev(values, result.mean);
    result.median = Median(values);
    result.mad = MAD(values, result.median);
    result.q1 = Percentile(values, 25.0);
    result.q3 = Percentile(values, 75.0);

    double iqr = result.q3 - result.q1;

    // Set default parameters and compute bounds
    switch (method) {
        case OutlierMethod::IQR: {
            result.parameter = (parameter <= 0) ? 1.5 : parameter;
            result.lower_bound = result.q1 - result.parameter * iqr;
            result.upper_bound = result.q3 + result.parameter * iqr;
            break;
        }
        case OutlierMethod::ZScore: {
            result.parameter = (parameter <= 0) ? 3.0 : parameter;
            result.lower_bound = result.mean - result.parameter * result.std_dev;
            result.upper_bound = result.mean + result.parameter * result.std_dev;
            break;
        }
        case OutlierMethod::ModifiedZScore: {
            result.parameter = (parameter <= 0) ? 3.5 : parameter;
            // Modified Z-score uses MAD
            double k = 0.6745;  // Constant for normal distribution
            if (result.mad > 0) {
                double scaled_mad = result.mad / k;
                result.lower_bound = result.median - result.parameter * scaled_mad;
                result.upper_bound = result.median + result.parameter * scaled_mad;
            } else {
                result.lower_bound = result.median;
                result.upper_bound = result.median;
            }
            break;
        }
    }

    // Find outliers
    for (const auto& iv : indexed_values) {
        double value = iv.second;
        bool is_outlier = false;
        double score = 0.0;
        bool is_low = false;

        if (value < result.lower_bound) {
            is_outlier = true;
            is_low = true;
            if (method == OutlierMethod::ZScore) {
                score = std::abs((value - result.mean) / result.std_dev);
            } else if (method == OutlierMethod::ModifiedZScore && result.mad > 0) {
                score = std::abs((value - result.median) / (result.mad / 0.6745));
            } else {
                score = (result.q1 - value) / (iqr > 0 ? iqr : 1.0);
            }
        } else if (value > result.upper_bound) {
            is_outlier = true;
            if (method == OutlierMethod::ZScore) {
                score = std::abs((value - result.mean) / result.std_dev);
            } else if (method == OutlierMethod::ModifiedZScore && result.mad > 0) {
                score = std::abs((value - result.median) / (result.mad / 0.6745));
            } else {
                score = (value - result.q3) / (iqr > 0 ? iqr : 1.0);
            }
        }

        if (is_outlier) {
            OutlierEntry entry;
            entry.row_index = iv.first;
            entry.value = value;
            entry.score = score;
            entry.is_low = is_low;
            result.outliers.push_back(entry);
        }
    }

    result.outlier_count = result.outliers.size();
    result.outlier_percentage = result.total_valid > 0 ?
        static_cast<float>(result.outlier_count) / result.total_valid * 100.0f : 0.0f;

    // Sort outliers by score (highest first)
    std::sort(result.outliers.begin(), result.outliers.end(),
              [](const OutlierEntry& a, const OutlierEntry& b) {
                  return a.score > b.score;
              });

    return result;
}

OutlierResult DataAnalyzer::DetectOutliers(const DataTable& table, const std::string& column_name,
                                            OutlierMethod method, double parameter) {
    const auto& headers = table.GetHeaders();
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == column_name) {
            return DetectOutliers(table, i, method, parameter);
        }
    }
    return OutlierResult();
}

// ========== Phase 4: Statistical Methods ==========

double DataAnalyzer::Mode(const std::vector<double>& data) {
    if (data.empty()) return 0.0;

    // Round to 6 decimal places for grouping
    std::unordered_map<int64_t, size_t> counts;
    for (double v : data) {
        int64_t key = static_cast<int64_t>(v * 1000000.0);
        counts[key]++;
    }

    int64_t mode_key = 0;
    size_t max_count = 0;
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            mode_key = pair.first;
        }
    }

    return static_cast<double>(mode_key) / 1000000.0;
}

double DataAnalyzer::SEM(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double mean = Mean(data);
    double sd = StdDev(data, mean);
    return sd / std::sqrt(static_cast<double>(data.size()));
}

DescriptiveStats DataAnalyzer::ComputeDescriptiveStats(const std::vector<double>& data) {
    DescriptiveStats stats;

    if (data.empty()) return stats;

    stats.count = static_cast<double>(data.size());
    stats.sum = std::accumulate(data.begin(), data.end(), 0.0);
    stats.mean = stats.sum / stats.count;
    stats.median = Median(std::vector<double>(data));
    stats.mode = Mode(data);
    stats.min = *std::min_element(data.begin(), data.end());
    stats.max = *std::max_element(data.begin(), data.end());
    stats.range = stats.max - stats.min;
    stats.variance = Variance(data, stats.mean);
    stats.std_dev = std::sqrt(stats.variance);
    stats.q1 = Percentile(std::vector<double>(data), 25.0);
    stats.q3 = Percentile(std::vector<double>(data), 75.0);
    stats.iqr = stats.q3 - stats.q1;
    stats.skewness = Skewness(data, stats.mean, stats.std_dev);
    stats.kurtosis = Kurtosis(data, stats.mean, stats.std_dev);
    stats.sem = SEM(data);
    stats.cv = stats.mean != 0 ? (stats.std_dev / std::abs(stats.mean)) * 100.0 : 0.0;

    // Calculate multiple percentiles
    stats.percentiles = {
        Percentile(std::vector<double>(data), 5.0),
        Percentile(std::vector<double>(data), 10.0),
        Percentile(std::vector<double>(data), 25.0),
        Percentile(std::vector<double>(data), 50.0),
        Percentile(std::vector<double>(data), 75.0),
        Percentile(std::vector<double>(data), 90.0),
        Percentile(std::vector<double>(data), 95.0)
    };

    return stats;
}

// ========== Hypothesis Testing Implementations ==========

// Approximation of the error function
static double erf_approx(double x) {
    // Horner form coefficients for erf
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::abs(x);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return sign * y;
}

double DataAnalyzer::NormalCDF(double x, double mu, double sigma) {
    if (sigma <= 0) return 0.0;
    double z = (x - mu) / sigma;
    return 0.5 * (1.0 + erf_approx(z / std::sqrt(2.0)));
}

// Regularized incomplete beta function approximation
static double betaInc(double a, double b, double x) {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;

    // Simple continued fraction approximation
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                         a * std::log(x) + b * std::log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        // Use forward recursion
        double sum = 1.0;
        double term = 1.0;
        for (int n = 1; n <= 100; n++) {
            double an = (a + n) * (a + b + n) * x / ((a + 2.0 * n) * (a + 2.0 * n + 1.0));
            term *= -an;
            sum += term;
            if (std::abs(term) < 1e-10) break;
        }
        return bt * sum / a;
    } else {
        // Use backward recursion
        return 1.0 - betaInc(b, a, 1.0 - x);
    }
}

double DataAnalyzer::StudentTCDF(double t, double df) {
    if (df <= 0) return 0.5;

    double x = df / (df + t * t);
    double prob = 0.5 * betaInc(df / 2.0, 0.5, x);

    return t < 0 ? prob : 1.0 - prob;
}

double DataAnalyzer::FCDF(double f, double df1, double df2) {
    if (f <= 0 || df1 <= 0 || df2 <= 0) return 0.0;

    double x = df1 * f / (df1 * f + df2);
    return betaInc(df1 / 2.0, df2 / 2.0, x);
}

double DataAnalyzer::ChiSquareCDF(double x, double df) {
    if (x <= 0 || df <= 0) return 0.0;

    // Chi-square is a special case of Gamma distribution
    // Using incomplete gamma function approximation
    double a = df / 2.0;
    double z = x / 2.0;

    // Simple series approximation for lower incomplete gamma
    double sum = 0.0;
    double term = 1.0 / a;
    sum = term;

    for (int n = 1; n <= 200; n++) {
        term *= z / (a + n);
        sum += term;
        if (std::abs(term) < 1e-10) break;
    }

    return std::min(1.0, sum * std::exp(-z + a * std::log(z) - std::lgamma(a)));
}

double DataAnalyzer::TInv(double alpha, double df) {
    // Newton-Raphson to find t such that P(T < t) = 1 - alpha/2
    // For two-tailed test
    double target = 1.0 - alpha / 2.0;
    double t = 1.96;  // Initial guess (normal approx)

    for (int i = 0; i < 50; i++) {
        double p = StudentTCDF(t, df);
        double error = p - target;

        if (std::abs(error) < 1e-8) break;

        // Derivative approximation
        double dp = (StudentTCDF(t + 0.001, df) - p) / 0.001;
        if (std::abs(dp) < 1e-10) break;

        t -= error / dp;
    }

    return t;
}

HypothesisTestResult DataAnalyzer::OneSampleTTest(const std::vector<double>& sample, double mu0, double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::OneSampleTTest;

    if (sample.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2)";
        return result;
    }

    double n = static_cast<double>(sample.size());
    double mean = Mean(sample);
    double sd = StdDev(sample, mean);
    double se = sd / std::sqrt(n);

    if (se == 0) {
        result.interpretation = "Standard error is zero (no variance in sample)";
        return result;
    }

    result.test_statistic = (mean - mu0) / se;
    result.df = n - 1;
    result.mean_diff = mean - mu0;
    result.se_diff = se;

    // Two-tailed p-value
    result.p_value = 2.0 * (1.0 - StudentTCDF(std::abs(result.test_statistic), result.df));

    // Confidence interval
    double t_crit = TInv(alpha, result.df);
    result.confidence_interval_low = mean - t_crit * se;
    result.confidence_interval_high = mean + t_crit * se;

    // Effect size (Cohen's d)
    result.effect_size = (mean - mu0) / sd;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Sample mean significantly differs from " +
                               std::to_string(mu0) + " (p = " +
                               std::to_string(result.p_value) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant difference from " +
                               std::to_string(mu0) + " (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::TwoSampleTTest(const std::vector<double>& sample1,
                                                   const std::vector<double>& sample2,
                                                   bool equal_variance,
                                                   double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::TwoSampleTTest;

    if (sample1.size() < 2 || sample2.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2 in one or both groups)";
        return result;
    }

    double n1 = static_cast<double>(sample1.size());
    double n2 = static_cast<double>(sample2.size());
    double mean1 = Mean(sample1);
    double mean2 = Mean(sample2);
    double var1 = Variance(sample1, mean1);
    double var2 = Variance(sample2, mean2);

    double se, df;

    if (equal_variance) {
        // Pooled variance
        double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
        se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
        df = n1 + n2 - 2;
    } else {
        // Welch's approximation
        se = std::sqrt(var1 / n1 + var2 / n2);
        double num = std::pow(var1 / n1 + var2 / n2, 2);
        double den = std::pow(var1 / n1, 2) / (n1 - 1) + std::pow(var2 / n2, 2) / (n2 - 1);
        df = num / den;
    }

    if (se == 0) {
        result.interpretation = "Standard error is zero";
        return result;
    }

    result.test_statistic = (mean1 - mean2) / se;
    result.df = df;
    result.mean_diff = mean1 - mean2;
    result.se_diff = se;

    result.p_value = 2.0 * (1.0 - StudentTCDF(std::abs(result.test_statistic), result.df));

    double t_crit = TInv(alpha, result.df);
    result.confidence_interval_low = (mean1 - mean2) - t_crit * se;
    result.confidence_interval_high = (mean1 - mean2) + t_crit * se;

    // Cohen's d
    double pooled_sd = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
    result.effect_size = pooled_sd > 0 ? (mean1 - mean2) / pooled_sd : 0;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Significant difference between groups (p = " +
                               std::to_string(result.p_value) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant difference (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::PairedTTest(const std::vector<double>& sample1,
                                                const std::vector<double>& sample2,
                                                double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::PairedTTest;

    if (sample1.size() != sample2.size()) {
        result.interpretation = "Sample sizes must be equal for paired test";
        return result;
    }

    if (sample1.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2)";
        return result;
    }

    // Calculate differences
    std::vector<double> diff(sample1.size());
    for (size_t i = 0; i < sample1.size(); i++) {
        diff[i] = sample1[i] - sample2[i];
    }

    // One-sample t-test on differences
    result = OneSampleTTest(diff, 0.0, alpha);
    result.test_type = TestType::PairedTTest;

    return result;
}

HypothesisTestResult DataAnalyzer::OneWayANOVA(const std::vector<std::vector<double>>& groups,
                                                double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::OneWayANOVA;

    if (groups.size() < 2) {
        result.interpretation = "Need at least 2 groups for ANOVA";
        return result;
    }

    size_t k = groups.size();  // Number of groups
    size_t N = 0;              // Total sample size
    for (const auto& g : groups) {
        if (g.empty()) {
            result.interpretation = "Empty group detected";
            return result;
        }
        N += g.size();
    }

    // Calculate grand mean
    double grand_sum = 0.0;
    for (const auto& g : groups) {
        grand_sum += std::accumulate(g.begin(), g.end(), 0.0);
    }
    double grand_mean = grand_sum / static_cast<double>(N);

    // Calculate SSB (between groups) and SSW (within groups)
    double SSB = 0.0;
    double SSW = 0.0;

    for (const auto& g : groups) {
        double group_mean = Mean(g);
        double n = static_cast<double>(g.size());

        SSB += n * std::pow(group_mean - grand_mean, 2);

        for (double v : g) {
            SSW += std::pow(v - group_mean, 2);
        }
    }

    double df_between = static_cast<double>(k - 1);
    double df_within = static_cast<double>(N - k);

    if (df_within <= 0) {
        result.interpretation = "Insufficient degrees of freedom";
        return result;
    }

    double MSB = SSB / df_between;
    double MSW = SSW / df_within;

    if (MSW == 0) {
        result.interpretation = "Within-group variance is zero";
        return result;
    }

    result.test_statistic = MSB / MSW;  // F-statistic
    result.df = df_between;
    result.df2 = df_within;

    result.p_value = 1.0 - FCDF(result.test_statistic, df_between, df_within);

    // Effect size: eta-squared
    double SST = SSB + SSW;
    result.effect_size = SST > 0 ? SSB / SST : 0;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Significant differences among groups (p = " +
                               std::to_string(result.p_value) + ", eta^2 = " +
                               std::to_string(result.effect_size) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant differences (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::ChiSquareTest(const std::vector<std::vector<double>>& contingency,
                                                  double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::ChiSquare;

    if (contingency.empty() || contingency[0].empty()) {
        result.interpretation = "Empty contingency table";
        return result;
    }

    size_t rows = contingency.size();
    size_t cols = contingency[0].size();

    // Calculate row totals, column totals, and grand total
    std::vector<double> row_totals(rows, 0.0);
    std::vector<double> col_totals(cols, 0.0);
    double grand_total = 0.0;

    for (size_t i = 0; i < rows; i++) {
        if (contingency[i].size() != cols) {
            result.interpretation = "Inconsistent column counts";
            return result;
        }
        for (size_t j = 0; j < cols; j++) {
            row_totals[i] += contingency[i][j];
            col_totals[j] += contingency[i][j];
            grand_total += contingency[i][j];
        }
    }

    if (grand_total == 0) {
        result.interpretation = "Total count is zero";
        return result;
    }

    // Calculate chi-square statistic
    double chi2 = 0.0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double expected = (row_totals[i] * col_totals[j]) / grand_total;
            if (expected > 0) {
                chi2 += std::pow(contingency[i][j] - expected, 2) / expected;
            }
        }
    }

    result.test_statistic = chi2;
    result.df = static_cast<double>((rows - 1) * (cols - 1));

    result.p_value = 1.0 - ChiSquareCDF(chi2, result.df);

    // Effect size: Cramér's V
    double min_dim = std::min(rows - 1, cols - 1);
    if (min_dim > 0 && grand_total > 0) {
        result.effect_size = std::sqrt(chi2 / (grand_total * min_dim));
    }

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Variables are associated (p = " +
                               std::to_string(result.p_value) + ", Cramer's V = " +
                               std::to_string(result.effect_size) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant association (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

// ========== Distribution Fitting ==========

DistributionFitResult DataAnalyzer::FitNormal(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Normal;
    result.name = "Normal";

    if (data.size() < 3) return result;

    double mu = Mean(data);
    double sigma = StdDev(data, mu);

    result.parameters["mu"] = mu;
    result.parameters["sigma"] = sigma;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double ll = 0.0;
    for (double x : data) {
        double z = (x - mu) / sigma;
        ll += -0.5 * z * z - std::log(sigma) - 0.5 * std::log(2.0 * M_PI);
    }
    result.log_likelihood = ll;

    // AIC and BIC
    result.aic = -2.0 * ll + 2.0 * 2.0;  // 2 parameters
    result.bic = -2.0 * ll + 2.0 * std::log(n);

    // KS test
    auto ks = KolmogorovSmirnovTest(data, DistributionType::Normal, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitUniform(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Uniform;
    result.name = "Uniform";

    if (data.empty()) return result;

    double a = *std::min_element(data.begin(), data.end());
    double b = *std::max_element(data.begin(), data.end());

    result.parameters["a"] = a;
    result.parameters["b"] = b;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double range = b - a;
    if (range > 0) {
        result.log_likelihood = -n * std::log(range);
    }

    result.aic = -2.0 * result.log_likelihood + 2.0 * 2.0;
    result.bic = -2.0 * result.log_likelihood + 2.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::Uniform, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitExponential(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Exponential;
    result.name = "Exponential";

    if (data.empty()) return result;

    // Check for positive values
    for (double x : data) {
        if (x < 0) {
            result.good_fit = false;
            return result;
        }
    }

    double lambda = 1.0 / Mean(data);
    result.parameters["lambda"] = lambda;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    result.log_likelihood = n * std::log(lambda) - lambda * sum;

    result.aic = -2.0 * result.log_likelihood + 2.0 * 1.0;
    result.bic = -2.0 * result.log_likelihood + 1.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::Exponential, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitLogNormal(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::LogNormal;
    result.name = "Log-Normal";

    if (data.empty()) return result;

    // Check for positive values
    std::vector<double> log_data;
    for (double x : data) {
        if (x <= 0) {
            result.good_fit = false;
            return result;
        }
        log_data.push_back(std::log(x));
    }

    double mu = Mean(log_data);
    double sigma = StdDev(log_data, mu);

    result.parameters["mu"] = mu;
    result.parameters["sigma"] = sigma;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double ll = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double z = (log_data[i] - mu) / sigma;
        ll += -0.5 * z * z - std::log(sigma * data[i]) - 0.5 * std::log(2.0 * M_PI);
    }
    result.log_likelihood = ll;

    result.aic = -2.0 * ll + 2.0 * 2.0;
    result.bic = -2.0 * ll + 2.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::LogNormal, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

std::vector<DistributionFitResult> DataAnalyzer::FitAllDistributions(const std::vector<double>& data) {
    std::vector<DistributionFitResult> results;

    results.push_back(FitNormal(data));
    results.push_back(FitUniform(data));

    // Only fit positive-value distributions if data is positive
    bool all_positive = std::all_of(data.begin(), data.end(), [](double x) { return x > 0; });
    if (all_positive) {
        results.push_back(FitExponential(data));
        results.push_back(FitLogNormal(data));
    }

    // Sort by AIC (lower is better)
    std::sort(results.begin(), results.end(), [](const DistributionFitResult& a, const DistributionFitResult& b) {
        return a.aic < b.aic;
    });

    return results;
}

std::pair<double, double> DataAnalyzer::KolmogorovSmirnovTest(const std::vector<double>& data,
                                                               DistributionType dist_type,
                                                               const std::map<std::string, double>& params) {
    if (data.empty()) return {0.0, 1.0};

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    double n = static_cast<double>(data.size());
    double D = 0.0;

    for (size_t i = 0; i < sorted_data.size(); i++) {
        double F_empirical = static_cast<double>(i + 1) / n;
        double F_theoretical = 0.0;

        switch (dist_type) {
            case DistributionType::Normal: {
                double mu = params.count("mu") ? params.at("mu") : 0.0;
                double sigma = params.count("sigma") ? params.at("sigma") : 1.0;
                F_theoretical = NormalCDF(sorted_data[i], mu, sigma);
                break;
            }
            case DistributionType::Uniform: {
                double a = params.count("a") ? params.at("a") : 0.0;
                double b = params.count("b") ? params.at("b") : 1.0;
                if (b > a) {
                    F_theoretical = (sorted_data[i] - a) / (b - a);
                    F_theoretical = std::max(0.0, std::min(1.0, F_theoretical));
                }
                break;
            }
            case DistributionType::Exponential: {
                double lambda = params.count("lambda") ? params.at("lambda") : 1.0;
                F_theoretical = 1.0 - std::exp(-lambda * sorted_data[i]);
                break;
            }
            case DistributionType::LogNormal: {
                double mu = params.count("mu") ? params.at("mu") : 0.0;
                double sigma = params.count("sigma") ? params.at("sigma") : 1.0;
                if (sorted_data[i] > 0) {
                    F_theoretical = NormalCDF(std::log(sorted_data[i]), mu, sigma);
                }
                break;
            }
            default:
                F_theoretical = 0.5;
        }

        double d1 = std::abs(F_empirical - F_theoretical);
        double d2 = std::abs((static_cast<double>(i) / n) - F_theoretical);
        D = std::max(D, std::max(d1, d2));
    }

    // Approximate p-value using Kolmogorov distribution
    // Using asymptotic formula
    double sqrt_n = std::sqrt(n);
    double lambda_ks = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D;
    double p_value = 2.0 * std::exp(-2.0 * lambda_ks * lambda_ks);
    p_value = std::max(0.0, std::min(1.0, p_value));

    return {D, p_value};
}

std::vector<double> DataAnalyzer::TheoreticalQuantiles(size_t n, DistributionType type,
                                                        const std::map<std::string, double>& params) {
    std::vector<double> quantiles;
    quantiles.reserve(n);

    for (size_t i = 1; i <= n; i++) {
        double p = (static_cast<double>(i) - 0.5) / static_cast<double>(n);

        double q = 0.0;
        switch (type) {
            case DistributionType::Normal: {
                double mu = params.count("mu") ? params.at("mu") : 0.0;
                double sigma = params.count("sigma") ? params.at("sigma") : 1.0;
                // Inverse normal approximation (Abramowitz and Stegun)
                double t = std::sqrt(-2.0 * std::log(p < 0.5 ? p : 1.0 - p));
                double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
                double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
                double z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
                if (p < 0.5) z = -z;
                q = mu + sigma * z;
                break;
            }
            case DistributionType::Uniform: {
                double a = params.count("a") ? params.at("a") : 0.0;
                double b = params.count("b") ? params.at("b") : 1.0;
                q = a + p * (b - a);
                break;
            }
            case DistributionType::Exponential: {
                double lambda = params.count("lambda") ? params.at("lambda") : 1.0;
                q = -std::log(1.0 - p) / lambda;
                break;
            }
            default:
                q = p;
        }

        quantiles.push_back(q);
    }

    return quantiles;
}

// ========== Regression Analysis ==========

RegressionResult DataAnalyzer::LinearRegression(const std::vector<double>& x,
                                                 const std::vector<double>& y) {
    RegressionResult result;
    result.type = RegressionType::Linear;

    if (x.size() != y.size() || x.size() < 3) {
        return result;
    }

    size_t n = x.size();
    result.n = n;
    result.df_model = 1;
    result.df_resid = n - 2;

    double mean_x = Mean(x);
    double mean_y = Mean(y);

    // Calculate slope and intercept
    double ss_xy = 0.0, ss_xx = 0.0;
    for (size_t i = 0; i < n; i++) {
        ss_xy += (x[i] - mean_x) * (y[i] - mean_y);
        ss_xx += (x[i] - mean_x) * (x[i] - mean_x);
    }

    if (ss_xx == 0) return result;

    double b1 = ss_xy / ss_xx;  // Slope
    double b0 = mean_y - b1 * mean_x;  // Intercept

    result.coefficients = {b0, b1};
    result.predictor_names = {"intercept", "x"};

    // Calculate predictions and residuals
    result.predicted.resize(n);
    result.residuals.resize(n);
    double ss_res = 0.0, ss_tot = 0.0;
    double mae = 0.0;

    for (size_t i = 0; i < n; i++) {
        result.predicted[i] = b0 + b1 * x[i];
        result.residuals[i] = y[i] - result.predicted[i];
        ss_res += result.residuals[i] * result.residuals[i];
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        mae += std::abs(result.residuals[i]);
    }

    result.r_squared = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
    result.adjusted_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - 2);
    result.mse = ss_res / (n - 2);
    result.rmse = std::sqrt(result.mse);
    result.mae = mae / n;

    // Standard errors
    double se_b1 = std::sqrt(result.mse / ss_xx);
    double se_b0 = std::sqrt(result.mse * (1.0 / n + mean_x * mean_x / ss_xx));

    result.std_errors = {se_b0, se_b1};

    // t-values and p-values
    result.t_values = {b0 / se_b0, b1 / se_b1};
    result.p_values = {
        2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[0]), n - 2)),
        2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[1]), n - 2))
    };

    // F-statistic
    double ss_reg = ss_tot - ss_res;
    result.f_statistic = result.mse > 0 ? (ss_reg / 1) / result.mse : 0;
    result.f_p_value = 1.0 - FCDF(result.f_statistic, 1, n - 2);

    return result;
}

RegressionResult DataAnalyzer::PolynomialRegression(const std::vector<double>& x,
                                                     const std::vector<double>& y,
                                                     int degree) {
    RegressionResult result;
    result.type = RegressionType::Polynomial;

    if (x.size() != y.size() || x.size() < static_cast<size_t>(degree + 2)) {
        return result;
    }

    // Build design matrix X = [1, x, x^2, ..., x^degree]
    size_t n = x.size();
    size_t p = degree + 1;

    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    for (size_t i = 0; i < n; i++) {
        X[i][0] = 1.0;
        for (int j = 1; j <= degree; j++) {
            X[i][j] = std::pow(x[i], j);
        }
    }

    // Build predictor names
    std::vector<std::string> names = {"intercept"};
    for (int j = 1; j <= degree; j++) {
        names.push_back("x^" + std::to_string(j));
    }

    return MultipleLinearRegression(X, y, names);
}

RegressionResult DataAnalyzer::MultipleLinearRegression(const std::vector<std::vector<double>>& X,
                                                         const std::vector<double>& y,
                                                         const std::vector<std::string>& predictor_names) {
    RegressionResult result;
    result.type = RegressionType::Multiple;

    size_t n = X.size();
    if (n == 0 || X[0].empty() || y.size() != n) {
        return result;
    }

    size_t p = X[0].size();  // Number of predictors including intercept
    result.n = n;
    result.df_model = p - 1;
    result.df_resid = n - p;

    if (n <= p) return result;

    // Normal equations: (X'X)β = X'y
    // Using Gauss-Jordan elimination

    // Compute X'X
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            for (size_t k = 0; k < n; k++) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
    }

    // Compute X'y
    std::vector<double> Xty(p, 0.0);
    for (size_t i = 0; i < p; i++) {
        for (size_t k = 0; k < n; k++) {
            Xty[i] += X[k][i] * y[k];
        }
    }

    // Solve using Gauss-Jordan with pivoting
    std::vector<std::vector<double>> augmented(p, std::vector<double>(p + 1));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            augmented[i][j] = XtX[i][j];
        }
        augmented[i][p] = Xty[i];
    }

    // Forward elimination with partial pivoting
    for (size_t col = 0; col < p; col++) {
        // Find pivot
        size_t max_row = col;
        for (size_t row = col + 1; row < p; row++) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[max_row][col])) {
                max_row = row;
            }
        }
        std::swap(augmented[col], augmented[max_row]);

        if (std::abs(augmented[col][col]) < 1e-10) {
            // Matrix is singular
            return result;
        }

        // Eliminate
        for (size_t row = 0; row < p; row++) {
            if (row != col) {
                double factor = augmented[row][col] / augmented[col][col];
                for (size_t j = col; j <= p; j++) {
                    augmented[row][j] -= factor * augmented[col][j];
                }
            }
        }
    }

    // Extract solution
    result.coefficients.resize(p);
    for (size_t i = 0; i < p; i++) {
        result.coefficients[i] = augmented[i][p] / augmented[i][i];
    }

    // Set predictor names
    if (predictor_names.size() == p) {
        result.predictor_names = predictor_names;
    } else {
        result.predictor_names.push_back("intercept");
        for (size_t i = 1; i < p; i++) {
            result.predictor_names.push_back("x" + std::to_string(i));
        }
    }

    // Calculate predictions and residuals
    double mean_y = Mean(y);
    result.predicted.resize(n);
    result.residuals.resize(n);
    double ss_res = 0.0, ss_tot = 0.0;
    double mae = 0.0;

    for (size_t i = 0; i < n; i++) {
        result.predicted[i] = 0.0;
        for (size_t j = 0; j < p; j++) {
            result.predicted[i] += result.coefficients[j] * X[i][j];
        }
        result.residuals[i] = y[i] - result.predicted[i];
        ss_res += result.residuals[i] * result.residuals[i];
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        mae += std::abs(result.residuals[i]);
    }

    result.r_squared = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
    result.adjusted_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - p);
    result.mse = ss_res / (n - p);
    result.rmse = std::sqrt(result.mse);
    result.mae = mae / n;

    // Compute (X'X)^-1 for standard errors
    // We already have the result of Gauss-Jordan, extract diagonal
    std::vector<std::vector<double>> XtX_inv(p, std::vector<double>(p, 0.0));

    // Re-solve with identity matrix to get inverse
    std::vector<std::vector<double>> aug2(p, std::vector<double>(2 * p));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            aug2[i][j] = XtX[i][j];
            aug2[i][p + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t col = 0; col < p; col++) {
        size_t max_row = col;
        for (size_t row = col + 1; row < p; row++) {
            if (std::abs(aug2[row][col]) > std::abs(aug2[max_row][col])) {
                max_row = row;
            }
        }
        std::swap(aug2[col], aug2[max_row]);

        if (std::abs(aug2[col][col]) > 1e-10) {
            double pivot = aug2[col][col];
            for (size_t j = 0; j < 2 * p; j++) {
                aug2[col][j] /= pivot;
            }
            for (size_t row = 0; row < p; row++) {
                if (row != col) {
                    double factor = aug2[row][col];
                    for (size_t j = 0; j < 2 * p; j++) {
                        aug2[row][j] -= factor * aug2[col][j];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            XtX_inv[i][j] = aug2[i][p + j];
        }
    }

    // Standard errors
    result.std_errors.resize(p);
    result.t_values.resize(p);
    result.p_values.resize(p);

    for (size_t i = 0; i < p; i++) {
        result.std_errors[i] = std::sqrt(result.mse * XtX_inv[i][i]);
        if (result.std_errors[i] > 0) {
            result.t_values[i] = result.coefficients[i] / result.std_errors[i];
            result.p_values[i] = 2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[i]), n - p));
        }
    }

    // F-statistic
    double ss_reg = ss_tot - ss_res;
    double ms_reg = ss_reg / (p - 1);
    result.f_statistic = result.mse > 0 ? ms_reg / result.mse : 0;
    result.f_p_value = 1.0 - FCDF(result.f_statistic, p - 1, n - p);

    return result;
}

} // namespace cyxwiz
