// audit_logger.h - Security event logging
#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <cstdint>
#include <functional>

namespace cyxwiz::servernode::security {

// Audit event types
enum class AuditEvent {
    // Authentication
    API_KEY_CREATED,
    API_KEY_REVOKED,
    API_KEY_USED,
    API_KEY_REJECTED,
    API_KEY_INVALID,

    // Access control
    MODEL_ACCESS_GRANTED,
    MODEL_ACCESS_DENIED,
    RATE_LIMIT_EXCEEDED,

    // Deployment
    MODEL_DEPLOYED,
    MODEL_UNDEPLOYED,
    MODEL_UPDATED,

    // Wallet operations
    WALLET_CONNECTED,
    WALLET_DISCONNECTED,
    TRANSACTION_INITIATED,
    TRANSACTION_COMPLETED,
    TRANSACTION_FAILED,

    // Security events
    TLS_ENABLED,
    TLS_DISABLED,
    INVALID_CERTIFICATE,
    UNAUTHORIZED_ACCESS,

    // System events
    DAEMON_STARTED,
    DAEMON_STOPPED,
    CONFIG_CHANGED,
    DOCKER_CONTAINER_CREATED,
    DOCKER_CONTAINER_STARTED,
    DOCKER_CONTAINER_STOPPED
};

// Get string name for event type
const char* GetAuditEventName(AuditEvent event);

// Audit log entry
struct AuditEntry {
    int64_t timestamp = 0;       // Unix timestamp
    AuditEvent event;
    std::string actor;           // API key ID, wallet address, or "system"
    std::string resource;        // Model ID, endpoint, container ID, etc.
    std::string details;         // JSON extra info
    std::string ip_address;      // Client IP if applicable
    bool success = true;
};

// Audit log query parameters
struct AuditQuery {
    int64_t since = 0;           // Start timestamp (0 = no filter)
    int64_t until = 0;           // End timestamp (0 = no filter)
    std::string actor;           // Filter by actor (empty = all)
    std::string resource;        // Filter by resource (empty = all)
    AuditEvent event_type;       // Filter by event type
    bool filter_event = false;   // Whether to filter by event type
    int limit = 100;             // Max entries to return
    int offset = 0;              // Pagination offset
};

// Callback for real-time audit events
using AuditCallback = std::function<void(const AuditEntry&)>;

class AuditLogger {
public:
    static AuditLogger& Instance();

    // Log an audit event
    void Log(AuditEvent event,
             const std::string& actor,
             const std::string& resource,
             bool success,
             const std::string& details = "",
             const std::string& ip_address = "");

    // Convenience methods for common events
    void LogKeyUsed(const std::string& key_id, const std::string& endpoint,
                    const std::string& ip = "");
    void LogKeyRejected(const std::string& key_prefix, const std::string& reason,
                        const std::string& ip = "");
    void LogRateLimitExceeded(const std::string& key_id, const std::string& endpoint,
                               const std::string& ip = "");
    void LogModelDeployed(const std::string& model_id, const std::string& actor);
    void LogModelUndeployed(const std::string& model_id, const std::string& actor);

    // Query audit log
    std::vector<AuditEntry> GetEntries(const AuditQuery& query);
    std::vector<AuditEntry> GetRecentEntries(int limit = 100);

    // Get entries since timestamp (simpler API)
    std::vector<AuditEntry> GetEntriesSince(int64_t since, int limit = 100);

    // Configuration
    void SetLogFile(const std::string& path);
    void SetMaxMemoryEntries(size_t max);
    void SetCallback(AuditCallback callback);

    // Statistics
    size_t GetEntryCount() const;
    int64_t GetOldestEntryTimestamp() const;

    // Export
    std::string ExportAsJSON(const AuditQuery& query);
    bool ExportToFile(const std::string& path, const AuditQuery& query);

private:
    AuditLogger();

    mutable std::mutex mutex_;
    std::vector<AuditEntry> entries_;
    std::string log_file_;
    size_t max_memory_entries_ = 10000;
    AuditCallback callback_;

    void WriteToFile(const AuditEntry& entry);
    void TrimMemoryEntries();
};

} // namespace cyxwiz::servernode::security
