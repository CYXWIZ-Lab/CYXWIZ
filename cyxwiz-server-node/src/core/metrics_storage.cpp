// metrics_storage.cpp - SQLite persistence for metrics history
#include "core/metrics_storage.h"
#include <sqlite3.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <chrono>

namespace cyxwiz::servernode::core {

// Static members
std::unique_ptr<MetricsStorage> MetricsStorageSingleton::instance_;
std::mutex MetricsStorageSingleton::init_mutex_;

MetricsStorage::MetricsStorage(const std::string& db_path)
    : db_path_(db_path) {
}

MetricsStorage::~MetricsStorage() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool MetricsStorage::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return true;
    }

    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to open metrics database: {}", sqlite3_errmsg(db_));
        return false;
    }

    // Enable WAL mode for better concurrent performance
    ExecuteSQL("PRAGMA journal_mode=WAL;");
    ExecuteSQL("PRAGMA synchronous=NORMAL;");

    if (!CreateTables()) {
        spdlog::error("Failed to create metrics tables");
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }

    initialized_ = true;
    spdlog::info("Metrics storage initialized: {}", db_path_);
    return true;
}

bool MetricsStorage::CreateTables() {
    // Raw metrics table (1-second granularity)
    const char* create_raw = R"(
        CREATE TABLE IF NOT EXISTS metrics_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            cpu_usage REAL,
            gpu_usage REAL,
            ram_usage REAL,
            vram_usage REAL,
            network_rx REAL,
            network_tx REAL,
            temperature REAL,
            power_watts REAL
        );
        CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON metrics_raw(timestamp);
    )";

    // Minute aggregates
    const char* create_minute = R"(
        CREATE TABLE IF NOT EXISTS metrics_minute (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            cpu_avg REAL, cpu_min REAL, cpu_max REAL,
            gpu_avg REAL, gpu_min REAL, gpu_max REAL,
            ram_avg REAL, ram_min REAL, ram_max REAL,
            vram_avg REAL, vram_min REAL, vram_max REAL,
            network_rx_avg REAL, network_tx_avg REAL,
            sample_count INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_minute_timestamp ON metrics_minute(timestamp);
    )";

    // Hour aggregates
    const char* create_hour = R"(
        CREATE TABLE IF NOT EXISTS metrics_hour (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            cpu_avg REAL, cpu_min REAL, cpu_max REAL,
            gpu_avg REAL, gpu_min REAL, gpu_max REAL,
            ram_avg REAL, ram_min REAL, ram_max REAL,
            vram_avg REAL, vram_min REAL, vram_max REAL,
            network_rx_avg REAL, network_tx_avg REAL,
            sample_count INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_hour_timestamp ON metrics_hour(timestamp);
    )";

    // Day aggregates
    const char* create_day = R"(
        CREATE TABLE IF NOT EXISTS metrics_day (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            cpu_avg REAL, cpu_min REAL, cpu_max REAL,
            gpu_avg REAL, gpu_min REAL, gpu_max REAL,
            ram_avg REAL, ram_min REAL, ram_max REAL,
            vram_avg REAL, vram_min REAL, vram_max REAL,
            network_rx_avg REAL, network_tx_avg REAL,
            sample_count INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_day_timestamp ON metrics_day(timestamp);
    )";

    // Job metrics table
    const char* create_job = R"(
        CREATE TABLE IF NOT EXISTS job_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            epoch INTEGER,
            loss REAL,
            accuracy REAL,
            learning_rate REAL,
            gpu_usage REAL,
            vram_usage REAL
        );
        CREATE INDEX IF NOT EXISTS idx_job_id ON job_metrics(job_id);
        CREATE INDEX IF NOT EXISTS idx_job_timestamp ON job_metrics(timestamp);
    )";

    return ExecuteSQL(create_raw) &&
           ExecuteSQL(create_minute) &&
           ExecuteSQL(create_hour) &&
           ExecuteSQL(create_day) &&
           ExecuteSQL(create_job);
}

bool MetricsStorage::ExecuteSQL(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        spdlog::error("SQL error: {}", err_msg);
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

bool MetricsStorage::StoreSystemMetrics(const SystemMetricsSnapshot& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    const char* sql = R"(
        INSERT INTO metrics_raw (timestamp, cpu_usage, gpu_usage, ram_usage, vram_usage,
                                 network_rx, network_tx, temperature, power_watts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare insert statement: {}", sqlite3_errmsg(db_));
        return false;
    }

    int64_t timestamp = metrics.timestamp;
    if (timestamp == 0) {
        timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }

    sqlite3_bind_int64(stmt, 1, timestamp);
    sqlite3_bind_double(stmt, 2, metrics.cpu_usage);
    sqlite3_bind_double(stmt, 3, metrics.gpu_usage);
    sqlite3_bind_double(stmt, 4, metrics.ram_usage);
    sqlite3_bind_double(stmt, 5, metrics.vram_usage);
    sqlite3_bind_double(stmt, 6, metrics.network_rx_mbps);
    sqlite3_bind_double(stmt, 7, metrics.network_tx_mbps);
    sqlite3_bind_double(stmt, 8, metrics.temperature);
    sqlite3_bind_double(stmt, 9, metrics.power_watts);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

bool MetricsStorage::StoreJobMetrics(const JobMetricsRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    const char* sql = R"(
        INSERT INTO job_metrics (job_id, timestamp, epoch, loss, accuracy,
                                 learning_rate, gpu_usage, vram_usage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    int64_t timestamp = record.timestamp;
    if (timestamp == 0) {
        timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }

    sqlite3_bind_text(stmt, 1, record.job_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, timestamp);
    sqlite3_bind_int(stmt, 3, record.epoch);
    sqlite3_bind_double(stmt, 4, record.loss);
    sqlite3_bind_double(stmt, 5, record.accuracy);
    sqlite3_bind_double(stmt, 6, record.learning_rate);
    sqlite3_bind_double(stmt, 7, record.gpu_usage);
    sqlite3_bind_double(stmt, 8, record.vram_usage);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return rc == SQLITE_DONE;
}

std::string MetricsStorage::GetColumnForMetric(MetricType type) {
    switch (type) {
        case MetricType::CPU_USAGE: return "cpu";
        case MetricType::GPU_USAGE: return "gpu";
        case MetricType::RAM_USAGE: return "ram";
        case MetricType::VRAM_USAGE: return "vram";
        case MetricType::NETWORK_RX: return "network_rx";
        case MetricType::NETWORK_TX: return "network_tx";
        case MetricType::TEMPERATURE: return "temperature";
        case MetricType::POWER_WATTS: return "power_watts";
        default: return "cpu";
    }
}

std::string MetricsStorage::GetTableForLevel(AggregationLevel level) {
    switch (level) {
        case AggregationLevel::RAW: return "metrics_raw";
        case AggregationLevel::MINUTE: return "metrics_minute";
        case AggregationLevel::HOUR: return "metrics_hour";
        case AggregationLevel::DAY: return "metrics_day";
        default: return "metrics_raw";
    }
}

std::vector<MetricPoint> MetricsStorage::GetMetricsHistory(
    MetricType type,
    int64_t start_time,
    int64_t end_time,
    AggregationLevel level) {

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MetricPoint> result;

    if (!initialized_ || !db_) return result;

    std::string col = GetColumnForMetric(type);
    std::string table = GetTableForLevel(level);

    std::string sql;
    if (level == AggregationLevel::RAW) {
        // Raw table has different column naming
        std::string raw_col;
        switch (type) {
            case MetricType::CPU_USAGE: raw_col = "cpu_usage"; break;
            case MetricType::GPU_USAGE: raw_col = "gpu_usage"; break;
            case MetricType::RAM_USAGE: raw_col = "ram_usage"; break;
            case MetricType::VRAM_USAGE: raw_col = "vram_usage"; break;
            case MetricType::NETWORK_RX: raw_col = "network_rx"; break;
            case MetricType::NETWORK_TX: raw_col = "network_tx"; break;
            case MetricType::TEMPERATURE: raw_col = "temperature"; break;
            case MetricType::POWER_WATTS: raw_col = "power_watts"; break;
            default: raw_col = "cpu_usage"; break;
        }
        sql = "SELECT timestamp, " + raw_col + " FROM " + table +
              " WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp;";
    } else {
        sql = "SELECT timestamp, " + col + "_avg, " + col + "_min, " + col + "_max FROM " + table +
              " WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp;";
    }

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return result;
    }

    sqlite3_bind_int64(stmt, 1, start_time);
    sqlite3_bind_int64(stmt, 2, end_time);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MetricPoint point;
        point.timestamp = sqlite3_column_int64(stmt, 0);

        if (level == AggregationLevel::RAW) {
            point.value = sqlite3_column_double(stmt, 1);
            point.avg = point.value;
            point.min = point.value;
            point.max = point.value;
        } else {
            point.avg = sqlite3_column_double(stmt, 1);
            point.min = sqlite3_column_double(stmt, 2);
            point.max = sqlite3_column_double(stmt, 3);
            point.value = point.avg;
        }

        result.push_back(point);
    }

    sqlite3_finalize(stmt);
    return result;
}

std::vector<MetricPoint> MetricsStorage::GetRecentMetrics(MetricType type, int count) {
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // Query last N seconds of raw data
    return GetMetricsHistory(type, now - count, now, AggregationLevel::RAW);
}

std::vector<JobMetricsRecord> MetricsStorage::GetJobMetrics(
    const std::string& job_id,
    int64_t start_time,
    int64_t end_time) {

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<JobMetricsRecord> result;

    if (!initialized_ || !db_) return result;

    std::string sql = "SELECT job_id, timestamp, epoch, loss, accuracy, "
                      "learning_rate, gpu_usage, vram_usage FROM job_metrics "
                      "WHERE job_id = ?";

    if (start_time > 0) {
        sql += " AND timestamp >= " + std::to_string(start_time);
    }
    if (end_time > 0) {
        sql += " AND timestamp <= " + std::to_string(end_time);
    }
    sql += " ORDER BY timestamp;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return result;
    }

    sqlite3_bind_text(stmt, 1, job_id.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        JobMetricsRecord record;
        record.job_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        record.timestamp = sqlite3_column_int64(stmt, 1);
        record.epoch = sqlite3_column_int(stmt, 2);
        record.loss = static_cast<float>(sqlite3_column_double(stmt, 3));
        record.accuracy = static_cast<float>(sqlite3_column_double(stmt, 4));
        record.learning_rate = static_cast<float>(sqlite3_column_double(stmt, 5));
        record.gpu_usage = static_cast<float>(sqlite3_column_double(stmt, 6));
        record.vram_usage = static_cast<float>(sqlite3_column_double(stmt, 7));
        result.push_back(record);
    }

    sqlite3_finalize(stmt);
    return result;
}

MetricsStorage::MetricsStats MetricsStorage::GetMetricsStats(
    MetricType type,
    int64_t start_time,
    int64_t end_time) {

    std::lock_guard<std::mutex> lock(mutex_);
    MetricsStats stats;

    if (!initialized_ || !db_) return stats;

    std::string col;
    switch (type) {
        case MetricType::CPU_USAGE: col = "cpu_usage"; break;
        case MetricType::GPU_USAGE: col = "gpu_usage"; break;
        case MetricType::RAM_USAGE: col = "ram_usage"; break;
        case MetricType::VRAM_USAGE: col = "vram_usage"; break;
        case MetricType::NETWORK_RX: col = "network_rx"; break;
        case MetricType::NETWORK_TX: col = "network_tx"; break;
        case MetricType::TEMPERATURE: col = "temperature"; break;
        case MetricType::POWER_WATTS: col = "power_watts"; break;
        default: col = "cpu_usage"; break;
    }

    std::string sql = "SELECT AVG(" + col + "), MIN(" + col + "), MAX(" + col + "), COUNT(*) "
                      "FROM metrics_raw WHERE timestamp >= ? AND timestamp <= ?;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return stats;
    }

    sqlite3_bind_int64(stmt, 1, start_time);
    sqlite3_bind_int64(stmt, 2, end_time);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        stats.average = sqlite3_column_double(stmt, 0);
        stats.min = sqlite3_column_double(stmt, 1);
        stats.max = sqlite3_column_double(stmt, 2);
        stats.sample_count = sqlite3_column_int64(stmt, 3);
    }

    sqlite3_finalize(stmt);

    // Calculate standard deviation in a second pass
    if (stats.sample_count > 0) {
        std::string std_sql = "SELECT AVG((" + col + " - ?) * (" + col + " - ?)) "
                              "FROM metrics_raw WHERE timestamp >= ? AND timestamp <= ?;";

        rc = sqlite3_prepare_v2(db_, std_sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK) {
            sqlite3_bind_double(stmt, 1, stats.average);
            sqlite3_bind_double(stmt, 2, stats.average);
            sqlite3_bind_int64(stmt, 3, start_time);
            sqlite3_bind_int64(stmt, 4, end_time);

            if (sqlite3_step(stmt) == SQLITE_ROW) {
                double variance = sqlite3_column_double(stmt, 0);
                stats.std_dev = std::sqrt(variance);
            }
            sqlite3_finalize(stmt);
        }
    }

    return stats;
}

bool MetricsStorage::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    int64_t cutoff = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count() - (retention_days_ * 24 * 60 * 60);

    // Delete old raw data (keep aggregates longer)
    std::string sql = "DELETE FROM metrics_raw WHERE timestamp < " + std::to_string(cutoff) + ";";
    if (!ExecuteSQL(sql)) return false;

    // Delete old job metrics
    sql = "DELETE FROM job_metrics WHERE timestamp < " + std::to_string(cutoff) + ";";
    if (!ExecuteSQL(sql)) return false;

    // VACUUM to reclaim space
    ExecuteSQL("VACUUM;");

    spdlog::info("Metrics cleanup complete - removed data older than {} days", retention_days_);
    return true;
}

bool MetricsStorage::AggregateOldData() {
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // Aggregate data older than 1 hour to minute level
    int64_t hour_ago = now - 3600;
    if (!AggregateToMinute(hour_ago)) return false;

    // Aggregate data older than 1 day to hour level
    int64_t day_ago = now - 86400;
    if (!AggregateToHour(day_ago)) return false;

    // Aggregate data older than 7 days to day level
    int64_t week_ago = now - 604800;
    if (!AggregateToDay(week_ago)) return false;

    return true;
}

bool MetricsStorage::AggregateToMinute(int64_t before_timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    // Group raw data by minute and insert into minute table
    const char* sql = R"(
        INSERT INTO metrics_minute (timestamp, cpu_avg, cpu_min, cpu_max,
                                    gpu_avg, gpu_min, gpu_max,
                                    ram_avg, ram_min, ram_max,
                                    vram_avg, vram_min, vram_max,
                                    network_rx_avg, network_tx_avg, sample_count)
        SELECT (timestamp / 60) * 60,
               AVG(cpu_usage), MIN(cpu_usage), MAX(cpu_usage),
               AVG(gpu_usage), MIN(gpu_usage), MAX(gpu_usage),
               AVG(ram_usage), MIN(ram_usage), MAX(ram_usage),
               AVG(vram_usage), MIN(vram_usage), MAX(vram_usage),
               AVG(network_rx), AVG(network_tx), COUNT(*)
        FROM metrics_raw
        WHERE timestamp < ?
        GROUP BY (timestamp / 60) * 60;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, before_timestamp);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) return false;

    // Delete aggregated raw data
    std::string del_sql = "DELETE FROM metrics_raw WHERE timestamp < " +
                          std::to_string(before_timestamp) + ";";
    return ExecuteSQL(del_sql);
}

bool MetricsStorage::AggregateToHour(int64_t before_timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    const char* sql = R"(
        INSERT INTO metrics_hour (timestamp, cpu_avg, cpu_min, cpu_max,
                                  gpu_avg, gpu_min, gpu_max,
                                  ram_avg, ram_min, ram_max,
                                  vram_avg, vram_min, vram_max,
                                  network_rx_avg, network_tx_avg, sample_count)
        SELECT (timestamp / 3600) * 3600,
               AVG(cpu_avg), MIN(cpu_min), MAX(cpu_max),
               AVG(gpu_avg), MIN(gpu_min), MAX(gpu_max),
               AVG(ram_avg), MIN(ram_min), MAX(ram_max),
               AVG(vram_avg), MIN(vram_min), MAX(vram_max),
               AVG(network_rx_avg), AVG(network_tx_avg), SUM(sample_count)
        FROM metrics_minute
        WHERE timestamp < ?
        GROUP BY (timestamp / 3600) * 3600;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, before_timestamp);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) return false;

    std::string del_sql = "DELETE FROM metrics_minute WHERE timestamp < " +
                          std::to_string(before_timestamp) + ";";
    return ExecuteSQL(del_sql);
}

bool MetricsStorage::AggregateToDay(int64_t before_timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return false;

    const char* sql = R"(
        INSERT INTO metrics_day (timestamp, cpu_avg, cpu_min, cpu_max,
                                 gpu_avg, gpu_min, gpu_max,
                                 ram_avg, ram_min, ram_max,
                                 vram_avg, vram_min, vram_max,
                                 network_rx_avg, network_tx_avg, sample_count)
        SELECT (timestamp / 86400) * 86400,
               AVG(cpu_avg), MIN(cpu_min), MAX(cpu_max),
               AVG(gpu_avg), MIN(gpu_min), MAX(gpu_max),
               AVG(ram_avg), MIN(ram_min), MAX(ram_max),
               AVG(vram_avg), MIN(vram_min), MAX(vram_max),
               AVG(network_rx_avg), AVG(network_tx_avg), SUM(sample_count)
        FROM metrics_hour
        WHERE timestamp < ?
        GROUP BY (timestamp / 86400) * 86400;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, before_timestamp);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) return false;

    std::string del_sql = "DELETE FROM metrics_hour WHERE timestamp < " +
                          std::to_string(before_timestamp) + ";";
    return ExecuteSQL(del_sql);
}

int64_t MetricsStorage::GetDatabaseSize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return 0;

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, "PRAGMA page_count;", -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return 0;

    int64_t page_count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        page_count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);

    rc = sqlite3_prepare_v2(db_, "PRAGMA page_size;", -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return 0;

    int64_t page_size = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        page_size = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);

    return page_count * page_size;
}

int64_t MetricsStorage::GetTotalRecords() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !db_) return 0;

    int64_t total = 0;
    const char* tables[] = {"metrics_raw", "metrics_minute", "metrics_hour", "metrics_day", "job_metrics"};

    for (const char* table : tables) {
        std::string sql = "SELECT COUNT(*) FROM " + std::string(table) + ";";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                total += sqlite3_column_int64(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
    }

    return total;
}

// Singleton implementation
MetricsStorage& MetricsStorageSingleton::Instance() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (!instance_) {
        throw std::runtime_error("MetricsStorage not initialized. Call Initialize() first.");
    }
    return *instance_;
}

bool MetricsStorageSingleton::Initialize(const std::string& db_path) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (instance_) {
        return true;  // Already initialized
    }
    instance_ = std::make_unique<MetricsStorage>(db_path);
    return instance_->Initialize();
}

bool MetricsStorageSingleton::IsInitialized() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    return instance_ && instance_->IsInitialized();
}

} // namespace cyxwiz::servernode::core
