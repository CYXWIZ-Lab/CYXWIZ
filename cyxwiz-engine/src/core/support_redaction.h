#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace cyxwiz {

class SupportRedaction {
public:
    static bool IsSensitiveKey(const std::string& key);
    static nlohmann::json RedactJson(const nlohmann::json& value);
    static std::string RedactString(const std::string& value);
};

} // namespace cyxwiz
