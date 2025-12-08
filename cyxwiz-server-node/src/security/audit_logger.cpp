// audit_logger.cpp - Security event logging implementation
#include "security/audit_logger.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace cyxwiz::servernode::security {

const char* GetAuditEventName(AuditEvent event) {
    switch (event) {
        case AuditEvent::API_KEY_CREATED: return "API_KEY_CREATED";
        case AuditEvent::API_KEY_REVOKED: return "API_KEY_REVOKED";
        case AuditEvent::API_KEY_USED: return "API_KEY_USED";
        case AuditEvent::API_KEY_REJECTED: return "API_KEY_REJECTED";
        case AuditEvent::API_KEY_INVALID: return "API_KEY_INVALID";
        case AuditEvent::MODEL_ACCESS_GRANTED: return "MODEL_ACCESS_GRANTED";
        case AuditEvent::MODEL_ACCESS_DENIED: return "MODEL_ACCESS_DENIED";
        case AuditEvent::RATE_LIMIT_EXCEEDED: return "RATE_LIMIT_EXCEEDED";
        case AuditEvent::MODEL_DEPLOYED: return "MODEL_DEPLOYED";
        case AuditEvent::MODEL_UNDEPLOYED: return "MODEL_UNDEPLOYED";
        case AuditEvent::MODEL_UPDATED: return "MODEL_UPDATED";
        case AuditEvent::WALLET_CONNECTED: return "WALLET_CONNECTED";
        case AuditEvent::WALLET_DISCONNECTED: return "WALLET_DISCONNECTED";
        case AuditEvent::TRANSACTION_INITIATED: return "TRANSACTION_INITIATED";
        case AuditEvent::TRANSACTION_COMPLETED: return "TRANSACTION_COMPLETED";
        case AuditEvent::TRANSACTION_FAILED: return "TRANSACTION_FAILED";
        case AuditEvent::TLS_ENABLED: return "TLS_ENABLED";
        case AuditEvent::TLS_DISABLED: return "TLS_DISABLED";
        case AuditEvent::INVALID_CERTIFICATE: return "INVALID_CERTIFICATE";
        case AuditEvent::UNAUTHORIZED_ACCESS: return "UNAUTHORIZED_ACCESS";
        case AuditEvent::DAEMON_STARTED: return "DAEMON_STARTED";
        case AuditEvent::DAEMON_STOPPED: return "DAEMON_STOPPED";
        case AuditEvent::CONFIG_CHANGED: return "CONFIG_CHANGED";
        case AuditEvent::DOCKER_CONTAINER_CREATED: return "DOCKER_CONTAINER_CREATED";
        case AuditEvent::DOCKER_CONTAINER_STARTED: return "DOCKER_CONTAINER_STARTED";
        case AuditEvent::DOCKER_CONTAINER_STOPPED: return "DOCKER_CONTAINER_STOPPED";
        default: return "UNKNOWN";
    }
}

AuditLogger::AuditLogger() {
    // Pre-allocate some space
    entries_.reserve(1000);
}

AuditLogger& AuditLogger::Instance() {
    static AuditLogger instance;
    return instance;
}

void AuditLogger::Log(AuditEvent event,
                       const std::string& actor,
                       const std::string& resource,
                       bool success,
                       const std::string& details,
                       const std::string& ip_address) {
    AuditEntry entry;
    entry.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry.event = event;
    entry.actor = actor;
    entry.resource = resource;
    entry.success = success;
    entry.details = details;
    entry.ip_address = ip_address;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.push_back(entry);
        TrimMemoryEntries();
    }

    // Write to file if configured
    WriteToFile(entry);

    // Call callback if set
    if (callback_) {
        callback_(entry);
    }

    // Also log via spdlog for debugging
    if (success) {
        spdlog::info("AUDIT: {} actor={} resource={} ip={}",
                    GetAuditEventName(event), actor, resource, ip_address);
    } else {
        spdlog::warn("AUDIT: {} FAILED actor={} resource={} ip={} details={}",
                    GetAuditEventName(event), actor, resource, ip_address, details);
    }
}

void AuditLogger::LogKeyUsed(const std::string& key_id, const std::string& endpoint,
                              const std::string& ip) {
    Log(AuditEvent::API_KEY_USED, key_id, endpoint, true, "", ip);
}

void AuditLogger::LogKeyRejected(const std::string& key_prefix, const std::string& reason,
                                  const std::string& ip) {
    Log(AuditEvent::API_KEY_REJECTED, key_prefix, "", false, reason, ip);
}

void AuditLogger::LogRateLimitExceeded(const std::string& key_id, const std::string& endpoint,
                                        const std::string& ip) {
    Log(AuditEvent::RATE_LIMIT_EXCEEDED, key_id, endpoint, false, "", ip);
}

void AuditLogger::LogModelDeployed(const std::string& model_id, const std::string& actor) {
    Log(AuditEvent::MODEL_DEPLOYED, actor, model_id, true);
}

void AuditLogger::LogModelUndeployed(const std::string& model_id, const std::string& actor) {
    Log(AuditEvent::MODEL_UNDEPLOYED, actor, model_id, true);
}

std::vector<AuditEntry> AuditLogger::GetEntries(const AuditQuery& query) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AuditEntry> results;
    int skipped = 0;

    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        const auto& entry = *it;

        // Apply filters
        if (query.since > 0 && entry.timestamp < query.since) continue;
        if (query.until > 0 && entry.timestamp > query.until) continue;
        if (!query.actor.empty() && entry.actor != query.actor) continue;
        if (!query.resource.empty() && entry.resource != query.resource) continue;
        if (query.filter_event && entry.event != query.event_type) continue;

        // Pagination
        if (skipped < query.offset) {
            skipped++;
            continue;
        }

        results.push_back(entry);

        if (static_cast<int>(results.size()) >= query.limit) {
            break;
        }
    }

    return results;
}

std::vector<AuditEntry> AuditLogger::GetRecentEntries(int limit) {
    AuditQuery query;
    query.limit = limit;
    return GetEntries(query);
}

std::vector<AuditEntry> AuditLogger::GetEntriesSince(int64_t since, int limit) {
    AuditQuery query;
    query.since = since;
    query.limit = limit;
    return GetEntries(query);
}

void AuditLogger::SetLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_file_ = path;
    spdlog::info("AuditLogger: Log file set to {}", path);
}

void AuditLogger::SetMaxMemoryEntries(size_t max) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_memory_entries_ = max;
    TrimMemoryEntries();
}

void AuditLogger::SetCallback(AuditCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
}

size_t AuditLogger::GetEntryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

int64_t AuditLogger::GetOldestEntryTimestamp() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.empty()) return 0;
    return entries_.front().timestamp;
}

std::string AuditLogger::ExportAsJSON(const AuditQuery& query) {
    auto entries = GetEntries(query);

    nlohmann::json j = nlohmann::json::array();
    for (const auto& entry : entries) {
        j.push_back({
            {"timestamp", entry.timestamp},
            {"event", GetAuditEventName(entry.event)},
            {"actor", entry.actor},
            {"resource", entry.resource},
            {"success", entry.success},
            {"details", entry.details},
            {"ip_address", entry.ip_address}
        });
    }

    return j.dump(2);
}

bool AuditLogger::ExportToFile(const std::string& path, const AuditQuery& query) {
    try {
        std::string json = ExportAsJSON(query);
        std::ofstream file(path);
        if (!file) {
            spdlog::error("AuditLogger: Failed to open file for export: {}", path);
            return false;
        }
        file << json;
        return true;
    } catch (const std::exception& e) {
        spdlog::error("AuditLogger: Export failed: {}", e.what());
        return false;
    }
}

void AuditLogger::WriteToFile(const AuditEntry& entry) {
    if (log_file_.empty()) return;

    try {
        std::ofstream file(log_file_, std::ios::app);
        if (!file) {
            spdlog::warn("AuditLogger: Failed to open log file: {}", log_file_);
            return;
        }

        // Write as JSON Lines format
        nlohmann::json j = {
            {"timestamp", entry.timestamp},
            {"event", GetAuditEventName(entry.event)},
            {"actor", entry.actor},
            {"resource", entry.resource},
            {"success", entry.success},
            {"details", entry.details},
            {"ip_address", entry.ip_address}
        };
        file << j.dump() << "\n";

    } catch (const std::exception& e) {
        spdlog::warn("AuditLogger: Failed to write to log file: {}", e.what());
    }
}

void AuditLogger::TrimMemoryEntries() {
    // Must be called with mutex held
    while (entries_.size() > max_memory_entries_) {
        entries_.erase(entries_.begin());
    }
}

} // namespace cyxwiz::servernode::security
