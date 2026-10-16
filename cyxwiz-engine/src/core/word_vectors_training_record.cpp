#include "word_vectors_training_record.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace {

bool FileIdentity(const std::string& path, uint64_t& bytes, int64_t& time) {
    std::error_code error;
    bytes = std::filesystem::file_size(path, error);
    if (error) return false;
    const auto written = std::filesystem::last_write_time(path, error);
    if (error) return false;
    time = static_cast<int64_t>(written.time_since_epoch().count());
    return true;
}

}  // namespace

std::string WordVectorsTrainingRecordPath(const std::string& vectors_path) {
    return vectors_path + ".training.json";
}

std::string FastTextModelPath(const std::string& vectors_path) {
    std::filesystem::path path(vectors_path);
    if (path.extension() == ".bin") return vectors_path + ".model.bin";
    return path.replace_extension(".bin").string();
}

bool WriteWordVectorsTrainingRecord(const std::string& vectors_path, WordVectorsTrainingRecord record,
                                    std::string* error) {
    if (!FileIdentity(vectors_path, record.vectors_bytes, record.vectors_time)) {
        if (error) *error = "cannot read " + vectors_path;
        return false;
    }
    if (record.fasttext_model &&
        !FileIdentity(FastTextModelPath(vectors_path), record.model_bytes, record.model_time)) {
        if (error) *error = "cannot read " + FastTextModelPath(vectors_path);
        return false;
    }
    nlohmann::json json = {{"version", record.version},          {"settings", record.settings},
                                 {"text_sha256", record.text_sha256},  {"vocab_sha256", record.vocab_sha256},
                                 {"trained_at", record.trained_at},    {"seconds", record.seconds},
                                 {"summary", record.summary},          {"coverage", record.coverage},
                                 {"vectors_bytes", record.vectors_bytes}, {"vectors_time", record.vectors_time}};
    if (record.fasttext_model)
        json["fasttext_model"] = {{"bytes", record.model_bytes}, {"time", record.model_time}};
    const std::string path = WordVectorsTrainingRecordPath(vectors_path);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << json.dump(2) << '\n';
    if (!out) {
        if (error) *error = "cannot write " + path;
        return false;
    }
    return true;
}

std::optional<WordVectorsTrainingRecord> ReadWordVectorsTrainingRecord(const std::string& vectors_path) {
    std::ifstream in(WordVectorsTrainingRecordPath(vectors_path), std::ios::binary);
    if (!in) return std::nullopt;
    const auto json = nlohmann::json::parse(in, nullptr, false);
    if (json.is_discarded() || !json.is_object()) return std::nullopt;
    WordVectorsTrainingRecord record;
    try {
        record.version = json.at("version").get<int>();
        record.settings = json.at("settings").get<std::string>();
        record.text_sha256 = json.at("text_sha256").get<std::string>();
        record.vocab_sha256 = json.at("vocab_sha256").get<std::string>();
        record.trained_at = json.at("trained_at").get<std::string>();
        record.seconds = json.at("seconds").get<double>();
        record.summary = json.at("summary").get<std::string>();
        record.coverage = json.at("coverage").get<std::string>();
        record.vectors_bytes = json.at("vectors_bytes").get<uint64_t>();
        record.vectors_time = json.at("vectors_time").get<int64_t>();
        if (json.contains("fasttext_model")) {
            record.fasttext_model = true;
            record.model_bytes = json.at("fasttext_model").at("bytes").get<uint64_t>();
            record.model_time = json.at("fasttext_model").at("time").get<int64_t>();
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
    uint64_t bytes = 0;
    int64_t time = 0;
    if (record.version != kWordVectorsTrainingVersion || !FileIdentity(vectors_path, bytes, time) ||
        bytes != record.vectors_bytes || time != record.vectors_time)
        return std::nullopt;
    if (record.fasttext_model && (!FileIdentity(FastTextModelPath(vectors_path), bytes, time) ||
                                  bytes != record.model_bytes || time != record.model_time))
        return std::nullopt;
    return record;
}

}  // namespace cyxwiz
