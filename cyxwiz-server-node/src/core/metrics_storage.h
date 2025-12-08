// metrics_storage.h - SQLite persistence for metrics history
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <mutex>

// Forward declaration for SQLite
struct sqlite3;

namespace cyxwiz::servernode::core {

// Metric types for storage and querying
enum class MetricType {
    CPU_USAGE,
    GPU_USAGE,
    RAM_USAGE,
    VRAM_USAGE,
    NETWORK_RX,
    NETWORK_TX,
    TEMPERATURE,
    POWER_WATTS
};

// Aggregation levels for different time ranges
enum class AggregationLevel {
    RAW,        // 1-second samples (default)
    MINUTE,     // 1-minute averages
    HOUR,       // 1-hour averages
    DAY         // Daily averages
};

// Single metric data point
struct MetricPoint {
    int64_t timestamp = 0;  // Unix timestamp (seconds)
    double value = 0.0;
    double min = 0.0;       // For aggregated data
    double max = 0.0;
    double avg = 0.0;
};

// System metrics snapshot for storage
struct SystemMetricsSnapshot {
    int64_t timestamp = 0;
    float cpu_usage = 0.0f;
    float gpu_usage = 0.0f;
    float ram_usage = 0.0f;
    float vram_usage = 0.0f;
    float network_rx_mbps = 0.0f;
    float network_tx_mbps = 0.0f;
    float temperature = 0.0f;
    float power_watts = 0.0f;
};

// Job metrics for correlation analysis
struct JobMetricsRecord {
    std::string job_id;
    int64_t timestamp = 0;
    int epoch = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    float learning_rate = 0.0f;
    float gpu_usage = 0.0f;
    float vram_usage = 0.0f;
};

class MetricsStorage {
public:
    explicit MetricsStorage(const std::string& db_path);
    ~MetricsStorage();

    // Initialization
    bool Initialize();
    bool IsInitialized() const { return initialized_; }

    // Store metrics
    bool StoreSystemMetrics(const SystemMetricsSnapshot& metrics);
    bool StoreJobMetrics(const JobMetricsRecord& record);

    // Query historical data
    std::vector<MetricPoint> GetMetricsHistory(
        MetricType type,
        int64_t start_time,
        int64_t end_time,
        AggregationLevel level = AggregationLevel::RAW);

    // Get job-correlated metrics
    std::vector<JobMetricsRecord> GetJobMetrics(
        const std::string& job_id,
        int64_t start_time = 0,
        int64_t end_time = 0);

    // Get latest N samples
    std::vector<MetricPoint> GetRecentMetrics(
        MetricType type,
        int count);

    // Statistics
    struct MetricsStats {
        double average = 0.0;
        double min = 0.0;
        double max = 0.0;
        double std_dev = 0.0;
        int64_t sample_count = 0;
    };

    MetricsStats GetMetricsStats(
        MetricType type,
        int64_t start_time,
        int64_t end_time);

    // Retention management
    void SetRetentionDays(int days) { retention_days_ = days; }
    int GetRetentionDays() const { return retention_days_; }
    bool Cleanup();  // Remove old data based on retention policy
    bool AggregateOldData();  // Compress old raw data to aggregates

    // Database info
    int64_t GetDatabaseSize();  // Size in bytes
    int64_t GetTotalRecords();

private:
    bool CreateTables();
    bool ExecuteSQL(const std::string& sql);

    // Aggregation helpers
    bool AggregateToMinute(int64_t before_timestamp);
    bool AggregateToHour(int64_t before_timestamp);
    bool AggregateToDay(int64_t before_timestamp);

    // Query helpers
    std::string GetTableForLevel(AggregationLevel level);
    std::string GetColumnForMetric(MetricType type);

    sqlite3* db_ = nullptr;
    std::string db_path_;
    bool initialized_ = false;
    int retention_days_ = 30;  // Default: keep 30 days of data
    std::mutex mutex_;
};

// Singleton for global access
class MetricsStorageSingleton {
public:
    static MetricsStorage& Instance();
    static bool Initialize(const std::string& db_path);
    static bool IsInitialized();

private:
    MetricsStorageSingleton() = default;
    static std::unique_ptr<MetricsStorage> instance_;
    static std::mutex init_mutex_;
};

} // namespace cyxwiz::servernode::core
