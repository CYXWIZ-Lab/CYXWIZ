#include "support_redaction.h"

#include <algorithm>
#include <cctype>
#include <vector>

namespace cyxwiz {
namespace {

std::string ToLower(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char current) {
            return static_cast<char>(std::tolower(current));
        });
    return value;
}

} // namespace

bool SupportRedaction::IsSensitiveKey(const std::string& key) {
    const std::string lower = ToLower(key);
    return lower.find("path") != std::string::npos ||
           lower.find("file") != std::string::npos ||
           lower.find("dataset") != std::string::npos ||
           lower.find("raw") != std::string::npos ||
           lower.find("preview") != std::string::npos ||
           lower.find("token") != std::string::npos ||
           lower.find("password") != std::string::npos ||
           lower.find("secret") != std::string::npos ||
           lower.find("credential") != std::string::npos;
}

nlohmann::json SupportRedaction::RedactJson(const nlohmann::json& value) {
    if (value.is_object()) {
        nlohmann::json output = nlohmann::json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
            output[it.key()] = IsSensitiveKey(it.key())
                ? nlohmann::json("[REDACTED]")
                : RedactJson(it.value());
        }
        return output;
    }

    if (value.is_array()) {
        nlohmann::json output = nlohmann::json::array();
        for (const auto& item : value) output.push_back(RedactJson(item));
        return output;
    }

    if (value.is_string()) {
        return RedactString(value.get<std::string>());
    }
    return value;
}

std::string SupportRedaction::RedactString(const std::string& value) {
    std::string output = value;
    static const std::vector<std::string> markers = {
        "token=", "password=", "secret=", "credential="};

    for (const auto& marker : markers) {
        const auto position = ToLower(output).find(marker);
        if (position != std::string::npos) {
            output = output.substr(0, position + marker.size()) + "[REDACTED]";
        }
    }
    return output;
}

} // namespace cyxwiz
