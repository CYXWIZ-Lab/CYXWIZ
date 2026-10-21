#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cyxwiz::artifact_json {

using Json = nlohmann::json;

inline void SetError(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

inline bool WriteJsonFile(const std::string& path,
                          const Json& document,
                          std::string* error) {
    try {
        const std::filesystem::path out_path(path);
        if (out_path.empty()) {
            SetError(error, "model artifact path is empty");
            return false;
        }
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            SetError(error, "failed to open model artifact for writing: " + path);
            return false;
        }
        out << document.dump(2);
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

inline bool ReadJsonFile(const std::string& path,
                         Json& document,
                         std::string* error) {
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            SetError(error, "failed to open model artifact for reading: " + path);
            return false;
        }
        document = Json::parse(in);
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

inline Json MakeEnvelope(const std::string& format,
                         const std::string& model_type,
                         const Json& model) {
    return {
        {"format", format},
        {"version", 1},
        {"model_type", model_type},
        {"model", model},
    };
}

inline bool ValidateEnvelope(const Json& document,
                             const std::string& format,
                             const std::string& model_type,
                             std::string* error) {
    if (document.value("format", "") != format) {
        SetError(error, "model artifact format is not " + format);
        return false;
    }
    if (document.value("version", 0) != 1) {
        SetError(error, "unsupported model artifact version");
        return false;
    }
    if (document.value("model_type", "") != model_type) {
        SetError(error, "model artifact type mismatch; expected " + model_type);
        return false;
    }
    if (!document.contains("model")) {
        SetError(error, "model artifact is missing model payload");
        return false;
    }
    return true;
}

inline std::string ReadArtifactType(const std::string& path,
                                    const std::string& format,
                                    std::string* error) {
    Json document;
    if (!ReadJsonFile(path, document, error)) {
        return {};
    }
    if (document.value("format", "") != format) {
        SetError(error, "model artifact format is not " + format);
        return {};
    }
    if (document.value("version", 0) != 1) {
        SetError(error, "unsupported model artifact version");
        return {};
    }
    const std::string model_type = document.value("model_type", "");
    if (model_type.empty()) {
        SetError(error, "model artifact is missing model_type");
        return {};
    }
    if (!document.contains("model")) {
        SetError(error, "model artifact is missing model payload");
        return {};
    }
    return model_type;
}

} // namespace cyxwiz::artifact_json
