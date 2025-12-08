// rate_limiter.h - Token bucket rate limiting
#pragma once

#include <string>
#include <unordered_map>
#include <deque>
#include <chrono>
#include <mutex>

namespace cyxwiz::servernode::api {

// Rate limit result with detailed info
struct RateLimitResult {
    bool allowed = false;
    int remaining = 0;         // Requests remaining in current window
    int limit = 0;             // Total limit per window
    int64_t reset_at = 0;      // Unix timestamp when window resets
    int retry_after = 0;       // Seconds until request might succeed
};

class RateLimiter {
public:
    RateLimiter() = default;
    ~RateLimiter() = default;

    // Check if request is allowed for given key
    // rpm_limit: requests per minute limit
    bool Allow(const std::string& key, int rpm_limit);

    // Check with detailed result
    RateLimitResult Check(const std::string& key, int rpm_limit);

    // Record a request (call after Allow() returns true)
    void Record(const std::string& key);

    // Get remaining quota for a key
    int GetRemainingQuota(const std::string& key, int rpm_limit);

    // Clear all rate limit data (for testing)
    void Clear();

    // Clear rate limit data for specific key
    void ClearKey(const std::string& key);

    // Get current request count for key (in last minute)
    int GetCurrentCount(const std::string& key);

private:
    std::mutex mutex_;

    // key -> request timestamps
    std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>> buckets_;

    // Clean old entries from a bucket
    void CleanBucket(std::deque<std::chrono::steady_clock::time_point>& bucket);
};

// Global rate limiter singleton
class GlobalRateLimiter {
public:
    static GlobalRateLimiter& Instance();

    RateLimiter& GetLimiter() { return limiter_; }

    // Convenience methods
    bool Allow(const std::string& key, int rpm_limit) {
        return limiter_.Allow(key, rpm_limit);
    }

    RateLimitResult Check(const std::string& key, int rpm_limit) {
        return limiter_.Check(key, rpm_limit);
    }

    void Record(const std::string& key) {
        limiter_.Record(key);
    }

private:
    GlobalRateLimiter() = default;
    RateLimiter limiter_;
};

} // namespace cyxwiz::servernode::api
