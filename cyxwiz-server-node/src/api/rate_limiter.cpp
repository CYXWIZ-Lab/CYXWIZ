// rate_limiter.cpp - Token bucket rate limiting implementation
#include "api/rate_limiter.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz::servernode::api {

void RateLimiter::CleanBucket(std::deque<std::chrono::steady_clock::time_point>& bucket) {
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::minutes(1);

    while (!bucket.empty() && bucket.front() < cutoff) {
        bucket.pop_front();
    }
}

bool RateLimiter::Allow(const std::string& key, int rpm_limit) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& bucket = buckets_[key];
    CleanBucket(bucket);

    // Check if under limit
    if (static_cast<int>(bucket.size()) >= rpm_limit) {
        spdlog::debug("RateLimiter: Limit exceeded for key '{}' ({}/{})",
                     key, bucket.size(), rpm_limit);
        return false;
    }

    // Record this request
    bucket.push_back(std::chrono::steady_clock::now());
    return true;
}

RateLimitResult RateLimiter::Check(const std::string& key, int rpm_limit) {
    std::lock_guard<std::mutex> lock(mutex_);

    RateLimitResult result;
    result.limit = rpm_limit;

    auto& bucket = buckets_[key];
    CleanBucket(bucket);

    int current_count = static_cast<int>(bucket.size());
    result.remaining = std::max(0, rpm_limit - current_count);
    result.allowed = current_count < rpm_limit;

    // Calculate reset time
    auto now = std::chrono::system_clock::now();
    result.reset_at = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count() + 60;

    // Calculate retry after (if rate limited)
    if (!result.allowed && !bucket.empty()) {
        auto oldest = bucket.front();
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - oldest).count();
        result.retry_after = std::max(1, 60 - static_cast<int>(age));
    }

    return result;
}

void RateLimiter::Record(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_[key].push_back(std::chrono::steady_clock::now());
}

int RateLimiter::GetRemainingQuota(const std::string& key, int rpm_limit) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& bucket = buckets_[key];
    CleanBucket(bucket);

    return std::max(0, rpm_limit - static_cast<int>(bucket.size()));
}

void RateLimiter::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_.clear();
}

void RateLimiter::ClearKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_.erase(key);
}

int RateLimiter::GetCurrentCount(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = buckets_.find(key);
    if (it == buckets_.end()) {
        return 0;
    }

    auto& bucket = it->second;
    CleanBucket(bucket);
    return static_cast<int>(bucket.size());
}

// GlobalRateLimiter singleton
GlobalRateLimiter& GlobalRateLimiter::Instance() {
    static GlobalRateLimiter instance;
    return instance;
}

} // namespace cyxwiz::servernode::api
