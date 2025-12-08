// api_key_manager.cpp - API key management implementation
#include "api/api_key_manager.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <openssl/sha.h>
#include <openssl/rand.h>

namespace cyxwiz::servernode::api {

APIKeyManager::APIKeyManager(const std::string& storage_path)
    : storage_path_(storage_path) {
    // Create directory if needed
    if (!storage_path_.empty()) {
        std::filesystem::path dir = std::filesystem::path(storage_path_).parent_path();
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
    }
}

std::string APIKeyManager::GenerateRandomString(size_t length) {
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    std::string result;
    result.resize(length);

    // Use OpenSSL for cryptographically secure random bytes
    std::vector<unsigned char> random_bytes(length);
    if (RAND_bytes(random_bytes.data(), static_cast<int>(length)) != 1) {
        // Fallback to std::random_device if OpenSSL fails
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, sizeof(charset) - 2);
        for (size_t i = 0; i < length; ++i) {
            result[i] = charset[dist(gen)];
        }
        return result;
    }

    for (size_t i = 0; i < length; ++i) {
        result[i] = charset[random_bytes[i] % (sizeof(charset) - 1)];
    }

    return result;
}

std::string APIKeyManager::SHA256Hash(const std::string& input) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()),
           input.size(), hash);

    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::string APIKeyManager::GenerateUUID() {
    // Generate a UUID v4 (random)
    unsigned char uuid_bytes[16];
    if (RAND_bytes(uuid_bytes, 16) != 1) {
        // Fallback
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dist(0, 255);
        for (int i = 0; i < 16; ++i) {
            uuid_bytes[i] = static_cast<unsigned char>(dist(gen));
        }
    }

    // Set version (4) and variant bits
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40;  // Version 4
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80;  // Variant 1

    // Format as UUID string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            oss << '-';
        }
        oss << std::setw(2) << static_cast<int>(uuid_bytes[i]);
    }
    return oss.str();
}

CreateKeyResult APIKeyManager::CreateKey(const std::string& name,
                                          int rate_limit_rpm,
                                          const std::vector<std::string>& allowed_models) {
    CreateKeyResult result;

    if (name.empty()) {
        result.error = "Key name cannot be empty";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate key format: cyx_sk_live_[32 random chars]
    std::string random_part = GenerateRandomString(32);
    std::string full_key = "cyx_sk_live_" + random_part;

    // Hash for storage
    std::string key_hash = SHA256Hash(full_key);
    std::string key_prefix = full_key.substr(0, 16);  // "cyx_sk_live_" + 4 chars

    // Create key record
    APIKey key;
    key.id = GenerateUUID();
    key.name = name;
    key.key_hash = key_hash;
    key.key_prefix = key_prefix;
    key.rate_limit_rpm = rate_limit_rpm;
    key.allowed_models = allowed_models;
    key.created_at = std::time(nullptr);
    key.last_used_at = 0;
    key.request_count = 0;
    key.is_active = true;

    keys_.push_back(key);

    // Save to file
    Save();

    result.key_id = key.id;
    result.full_key = full_key;  // Only returned once!
    result.success = true;

    spdlog::info("APIKeyManager: Created key '{}' with ID {}", name, key.id);
    spdlog::info("APIKeyManager: Key prefix: {}...", key_prefix);

    return result;
}

bool APIKeyManager::RevokeKey(const std::string& key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& key : keys_) {
        if (key.id == key_id) {
            key.is_active = false;
            Save();
            spdlog::info("APIKeyManager: Revoked key '{}' ({})", key.name, key_id);
            return true;
        }
    }

    spdlog::warn("APIKeyManager: Key not found for revocation: {}", key_id);
    return false;
}

bool APIKeyManager::UpdateKey(const std::string& key_id, int rate_limit_rpm,
                               const std::vector<std::string>& allowed_models) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& key : keys_) {
        if (key.id == key_id) {
            key.rate_limit_rpm = rate_limit_rpm;
            key.allowed_models = allowed_models;
            Save();
            spdlog::info("APIKeyManager: Updated key '{}' ({})", key.name, key_id);
            return true;
        }
    }

    return false;
}

bool APIKeyManager::ValidateKey(const std::string& api_key) {
    if (api_key.empty()) {
        return false;
    }

    // Check format
    if (!api_key.starts_with("cyx_sk_")) {
        spdlog::debug("APIKeyManager: Invalid key format");
        return false;
    }

    std::string hash = SHA256Hash(api_key);

    std::lock_guard<std::mutex> lock(mutex_);
    const APIKey* key = FindKeyByHash(hash);

    if (!key) {
        spdlog::debug("APIKeyManager: Key not found");
        return false;
    }

    if (!key->is_active) {
        spdlog::debug("APIKeyManager: Key is revoked");
        return false;
    }

    return true;
}

bool APIKeyManager::ValidateKeyForModel(const std::string& api_key,
                                         const std::string& model_id) {
    if (!ValidateKey(api_key)) {
        return false;
    }

    std::string hash = SHA256Hash(api_key);

    std::lock_guard<std::mutex> lock(mutex_);
    const APIKey* key = FindKeyByHash(hash);

    if (!key) {
        return false;
    }

    // If no model restrictions, allow all
    if (key->allowed_models.empty()) {
        return true;
    }

    // Check if model is in allowed list
    auto it = std::find(key->allowed_models.begin(), key->allowed_models.end(), model_id);
    if (it == key->allowed_models.end()) {
        spdlog::debug("APIKeyManager: Model '{}' not allowed for key '{}'",
                     model_id, key->name);
        return false;
    }

    return true;
}

std::string APIKeyManager::GetKeyIdFromApiKey(const std::string& api_key) {
    if (api_key.empty()) {
        return "";
    }

    std::string hash = SHA256Hash(api_key);

    std::lock_guard<std::mutex> lock(mutex_);
    const APIKey* key = FindKeyByHash(hash);

    return key ? key->id : "";
}

std::vector<APIKey> APIKeyManager::ListKeys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keys_;
}

const APIKey* APIKeyManager::GetKey(const std::string& key_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& key : keys_) {
        if (key.id == key_id) {
            return &key;
        }
    }
    return nullptr;
}

size_t APIKeyManager::GetKeyCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keys_.size();
}

bool APIKeyManager::CheckRateLimit(const std::string& key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find the key to get rate limit
    const APIKey* key = nullptr;
    for (const auto& k : keys_) {
        if (k.id == key_id) {
            key = &k;
            break;
        }
    }

    if (!key) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    auto& bucket = rate_buckets_[key_id];

    // Clean old timestamps (older than 1 minute)
    auto cutoff = now - std::chrono::minutes(1);
    while (!bucket.empty() && bucket.front() < cutoff) {
        bucket.pop_front();
    }

    // Check if under limit
    if (static_cast<int>(bucket.size()) >= key->rate_limit_rpm) {
        spdlog::debug("APIKeyManager: Rate limit exceeded for key '{}'", key->name);
        return false;
    }

    return true;
}

int APIKeyManager::GetRemainingQuota(const std::string& key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find the key
    const APIKey* key = nullptr;
    for (const auto& k : keys_) {
        if (k.id == key_id) {
            key = &k;
            break;
        }
    }

    if (!key) {
        return 0;
    }

    // Clean and count
    auto now = std::chrono::steady_clock::now();
    auto& bucket = rate_buckets_[key_id];
    auto cutoff = now - std::chrono::minutes(1);
    while (!bucket.empty() && bucket.front() < cutoff) {
        bucket.pop_front();
    }

    return std::max(0, key->rate_limit_rpm - static_cast<int>(bucket.size()));
}

void APIKeyManager::RecordRequest(const std::string& key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Record timestamp for rate limiting
    auto now = std::chrono::steady_clock::now();
    rate_buckets_[key_id].push_back(now);

    // Update key stats
    for (auto& key : keys_) {
        if (key.id == key_id) {
            key.last_used_at = std::time(nullptr);
            key.request_count++;
            break;
        }
    }

    // Periodically save (every 100 requests)
    static int save_counter = 0;
    if (++save_counter >= 100) {
        save_counter = 0;
        Save();
    }
}

bool APIKeyManager::Load() {
    if (storage_path_.empty()) {
        spdlog::debug("APIKeyManager: No storage path set, skipping load");
        return true;
    }

    if (!std::filesystem::exists(storage_path_)) {
        spdlog::debug("APIKeyManager: Storage file does not exist: {}", storage_path_);
        return true;  // Not an error, just no saved keys
    }

    try {
        std::ifstream file(storage_path_);
        if (!file) {
            spdlog::error("APIKeyManager: Failed to open storage file: {}", storage_path_);
            return false;
        }

        nlohmann::json j = nlohmann::json::parse(file);

        std::lock_guard<std::mutex> lock(mutex_);
        keys_.clear();

        for (const auto& [id, data] : j.items()) {
            APIKey key;
            key.id = id;
            key.name = data.value("name", "");
            key.key_hash = data.value("key_hash", "");
            key.key_prefix = data.value("key_prefix", "");
            key.rate_limit_rpm = data.value("rate_limit_rpm", 60);
            key.created_at = data.value("created_at", 0);
            key.last_used_at = data.value("last_used_at", 0);
            key.request_count = data.value("request_count", 0);
            key.is_active = data.value("is_active", true);

            if (data.contains("allowed_models")) {
                key.allowed_models = data["allowed_models"].get<std::vector<std::string>>();
            }

            keys_.push_back(key);
        }

        spdlog::info("APIKeyManager: Loaded {} keys from {}", keys_.size(), storage_path_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("APIKeyManager: Failed to load keys: {}", e.what());
        return false;
    }
}

bool APIKeyManager::Save() const {
    if (storage_path_.empty()) {
        spdlog::debug("APIKeyManager: No storage path set, skipping save");
        return true;
    }

    try {
        nlohmann::json j;

        for (const auto& key : keys_) {
            j[key.id] = {
                {"name", key.name},
                {"key_hash", key.key_hash},
                {"key_prefix", key.key_prefix},
                {"rate_limit_rpm", key.rate_limit_rpm},
                {"allowed_models", key.allowed_models},
                {"created_at", key.created_at},
                {"last_used_at", key.last_used_at},
                {"request_count", key.request_count},
                {"is_active", key.is_active}
            };
        }

        std::ofstream file(storage_path_);
        if (!file) {
            spdlog::error("APIKeyManager: Failed to open file for writing: {}", storage_path_);
            return false;
        }

        file << j.dump(2);
        spdlog::debug("APIKeyManager: Saved {} keys to {}", keys_.size(), storage_path_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("APIKeyManager: Failed to save keys: {}", e.what());
        return false;
    }
}

APIKey* APIKeyManager::FindKeyByHash(const std::string& hash) {
    for (auto& key : keys_) {
        if (key.key_hash == hash) {
            return &key;
        }
    }
    return nullptr;
}

const APIKey* APIKeyManager::FindKeyByHash(const std::string& hash) const {
    for (const auto& key : keys_) {
        if (key.key_hash == hash) {
            return &key;
        }
    }
    return nullptr;
}

} // namespace cyxwiz::servernode::api
