#pragma once

#include <string>
#include <random>

namespace cyxwiz {
namespace protocol {

// Get current Unix timestamp
int64_t GetCurrentTimestamp();

// Generate a UUID v4
std::string GenerateUUID();

// Format bytes to human-readable string
std::string FormatBytes(int64_t bytes);

} // namespace protocol
} // namespace cyxwiz
