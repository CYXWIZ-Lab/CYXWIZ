#include "plot_dataset.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <spdlog/spdlog.h>

namespace cyxwiz::plotting {

// ============================================================================
// PlotDataset Implementation
// ============================================================================

void PlotDataset::AddSeries(const std::string& name) {
    if (HasSeries(name)) {
        spdlog::warn("Series '{}' already exists", name);
        return;
    }

    Series series;
    series.name = name;
    series_index_[name] = series_.size();
    series_.push_back(std::move(series));

    if (series_.size() == 1) {
        default_series_idx_ = 0;
    }
}

bool PlotDataset::HasSeries(const std::string& name) const {
    return series_index_.find(name) != series_index_.end();
}

PlotDataset::Series* PlotDataset::GetSeries(const std::string& name) {
    auto it = series_index_.find(name);
    if (it == series_index_.end()) {
        return nullptr;
    }
    return &series_[it->second];
}

const PlotDataset::Series* PlotDataset::GetSeries(const std::string& name) const {
    auto it = series_index_.find(name);
    if (it == series_index_.end()) {
        return nullptr;
    }
    return &series_[it->second];
}

std::vector<std::string> PlotDataset::GetSeriesNames() const {
    std::vector<std::string> names;
    names.reserve(series_.size());
    for (const auto& series : series_) {
        names.push_back(series.name);
    }
    return names;
}

void PlotDataset::AddPoint(double x, double y) {
    if (series_.empty()) {
        AddSeries("default");
    }
    series_[default_series_idx_].AddPoint(x, y);
}

void PlotDataset::AddPoint(double y) {
    if (series_.empty()) {
        AddSeries("default");
    }
    auto& series = series_[default_series_idx_];
    double x = static_cast<double>(series.Size());
    series.AddPoint(x, y);
}

void PlotDataset::Clear() {
    series_.clear();
    series_index_.clear();
    default_series_idx_ = 0;
}

bool PlotDataset::SaveToJSON(const std::string& filepath) const {
    try {
        nlohmann::json j;
        j["version"] = "1.0";
        j["series"] = nlohmann::json::array();

        for (const auto& series : series_) {
            nlohmann::json series_json;
            series_json["name"] = series.name;
            series_json["x_data"] = series.x_data;
            series_json["y_data"] = series.y_data;
            j["series"].push_back(series_json);
        }

        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(2);  // Pretty print with 2-space indentation
        file.close();

        spdlog::info("Saved plot data to {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to save plot data: {}", e.what());
        return false;
    }
}

bool PlotDataset::LoadFromJSON(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for reading: {}", filepath);
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        Clear();

        for (const auto& series_json : j["series"]) {
            std::string name = series_json["name"];
            AddSeries(name);

            auto* series = GetSeries(name);
            if (series) {
                series->x_data = series_json["x_data"].get<std::vector<double>>();
                series->y_data = series_json["y_data"].get<std::vector<double>>();
            }
        }

        spdlog::info("Loaded plot data from {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load plot data: {}", e.what());
        return false;
    }
}

// ============================================================================
// CircularBuffer Implementation
// ============================================================================

CircularBuffer::CircularBuffer(size_t capacity)
    : capacity_(capacity)
    , size_(0)
    , head_(0)
{
    x_buffer_.resize(capacity);
    y_buffer_.resize(capacity);
}

void CircularBuffer::AddPoint(double x, double y) {
    x_buffer_[head_] = x;
    y_buffer_[head_] = y;

    head_ = (head_ + 1) % capacity_;

    if (size_ < capacity_) {
        size_++;
    }
}

void CircularBuffer::Clear() {
    size_ = 0;
    head_ = 0;
}

void CircularBuffer::SetCapacity(size_t capacity) {
    if (capacity == capacity_) {
        return;
    }

    std::vector<double> new_x_buffer(capacity);
    std::vector<double> new_y_buffer(capacity);

    size_t new_size = std::min(size_, capacity);

    // Copy data from old buffer to new buffer
    for (size_t i = 0; i < new_size; ++i) {
        size_t old_idx = (head_ + capacity_ - size_ + i) % capacity_;
        new_x_buffer[i] = x_buffer_[old_idx];
        new_y_buffer[i] = y_buffer_[old_idx];
    }

    x_buffer_ = std::move(new_x_buffer);
    y_buffer_ = std::move(new_y_buffer);
    capacity_ = capacity;
    size_ = new_size;
    head_ = new_size % capacity;
}

double CircularBuffer::GetMinY() const {
    if (size_ == 0) return 0.0;
    return *std::min_element(y_buffer_.begin(), y_buffer_.begin() + size_);
}

double CircularBuffer::GetMaxY() const {
    if (size_ == 0) return 0.0;
    return *std::max_element(y_buffer_.begin(), y_buffer_.begin() + size_);
}

double CircularBuffer::GetMeanY() const {
    if (size_ == 0) return 0.0;
    double sum = std::accumulate(y_buffer_.begin(), y_buffer_.begin() + size_, 0.0);
    return sum / static_cast<double>(size_);
}

} // namespace cyxwiz::plotting
