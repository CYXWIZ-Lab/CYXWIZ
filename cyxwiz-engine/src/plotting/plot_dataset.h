#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace cyxwiz::plotting {

/**
 * PlotDataset - Container for plot data
 * Supports multiple series in one plot
 */
class PlotDataset {
public:
    struct Series {
        std::string name;
        std::vector<double> x_data;
        std::vector<double> y_data;

        void AddPoint(double x, double y) {
            x_data.push_back(x);
            y_data.push_back(y);
        }

        void Clear() {
            x_data.clear();
            y_data.clear();
        }

        size_t Size() const { return x_data.size(); }
    };

    PlotDataset() = default;

    // Series management
    void AddSeries(const std::string& name);
    bool HasSeries(const std::string& name) const;
    Series* GetSeries(const std::string& name);
    const Series* GetSeries(const std::string& name) const;

    std::vector<std::string> GetSeriesNames() const;
    size_t GetSeriesCount() const { return series_.size(); }

    // Convenience for single-series plots
    void AddPoint(double x, double y);
    void AddPoint(double y);  // Auto x = index

    // Data access
    const std::vector<Series>& GetAllSeries() const { return series_; }

    // Utility
    void Clear();
    bool IsEmpty() const { return series_.empty(); }

    // Serialization
    bool SaveToJSON(const std::string& filepath) const;
    bool LoadFromJSON(const std::string& filepath);

private:
    std::vector<Series> series_;
    std::unordered_map<std::string, size_t> series_index_;
    size_t default_series_idx_ = 0;
};

/**
 * CircularBuffer - Fixed-size buffer for real-time plotting
 * Efficiently maintains last N points
 */
class CircularBuffer {
public:
    CircularBuffer(size_t capacity);

    void AddPoint(double x, double y);
    void Clear();
    void SetCapacity(size_t capacity);

    // Data access for ImPlot
    const double* GetXData() const { return x_buffer_.data(); }
    const double* GetYData() const { return y_buffer_.data(); }
    size_t GetSize() const { return size_; }
    size_t GetCapacity() const { return capacity_; }

    // Statistics
    double GetMinY() const;
    double GetMaxY() const;
    double GetMeanY() const;

private:
    std::vector<double> x_buffer_;
    std::vector<double> y_buffer_;
    size_t capacity_;
    size_t size_;
    size_t head_;  // Circular buffer head
};

} // namespace cyxwiz::plotting
