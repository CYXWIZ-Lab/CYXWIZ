// api_key_manager.h - API key management for deployed models
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <chrono>

namespace cyxwiz::servernode::api {

struct APIKey {
    std::string id;
    std::string name;
    std::string key_hash;       // SHA-256 hash of full key (never store plaintext)
    std::string key_prefix;     // First 12 chars for identification (cyx_sk_live_)
    std::vector<std::string> allowed_models;  // Empty = all models allowed
    int64_t created_at = 0;
    int64_t last_used_at = 0;
    int64_t request_count = 0;
    int rate_limit_rpm = 60;    // Requests per minute
    bool is_active = true;
};

// Result type for CreateKey operation
struct CreateKeyResult {
    std::string key_id;      // UUID for managing the key
    std::string full_key;    // Full API key (only shown once!)
    bool success = false;
    std::string error;
};

class APIKeyManager {
public:
    APIKeyManager() = default;
    explicit APIKeyManager(const std::string& storage_path);
    ~APIKeyManager() = default;

    // Key management
    // Returns the full key only once during creation
    CreateKeyResult CreateKey(const std::string& name,
                               int rate_limit_rpm = 60,
                               const std::vector<std::string>& allowed_models = {});
    bool RevokeKey(const std::string& key_id);
    bool UpdateKey(const std::string& key_id, int rate_limit_rpm,
                   const std::vector<std::string>& allowed_models);

    // Key validation
    // Returns true if key is valid and active
    bool ValidateKey(const std::string& api_key);

    // Validate key with model access check
    bool ValidateKeyForModel(const std::string& api_key, const std::string& model_id);

    // Get key ID from API key (for rate limiting, logging)
    std::string GetKeyIdFromApiKey(const std::string& api_key);

    // Key lookup
    std::vector<APIKey> ListKeys() const;
    const APIKey* GetKey(const std::string& key_id) const;
    size_t GetKeyCount() const;

    // Rate limiting (built-in token bucket)
    bool CheckRateLimit(const std::string& key_id);
    int GetRemainingQuota(const std::string& key_id);
    void RecordRequest(const std::string& key_id);

    // Persistence
    bool Load();  // Uses storage_path_
    bool Save() const;  // Uses storage_path_

    // Crypto helpers (public for testing)
    static std::string GenerateRandomString(size_t length);
    static std::string SHA256Hash(const std::string& input);
    static std::string GenerateUUID();

private:
    mutable std::mutex mutex_;
    std::vector<APIKey> keys_;
    std::string storage_path_;

    // Rate limiting state: key_id -> request timestamps
    std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>> rate_buckets_;

    // Find key by hash
    APIKey* FindKeyByHash(const std::string& hash);
    const APIKey* FindKeyByHash(const std::string& hash) const;
};

} // namespace cyxwiz::servernode::api
