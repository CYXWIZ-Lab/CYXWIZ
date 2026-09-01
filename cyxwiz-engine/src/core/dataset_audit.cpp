#include "dataset_audit.h"

#include "arrow_dataset.h"
#include "image_utils.h"
#include "parquet_backed_dataset.h"

#include <cyxwiz/audio_processing.h>

#include <arrow/api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace cyxwiz {

namespace {

constexpr int64_t kMaxLabelRowsToScan = 100000;
constexpr size_t kMaxImageAuditSamples = 100;
constexpr size_t kMaxAudioAuditSamples = 100;
constexpr size_t kMaxTextAuditSamples = 100;
constexpr size_t kMaxTextFileBytesToRead = 1024 * 1024;
constexpr size_t kMaxIssueExamples = 5;
constexpr float kBlackPixelEpsilon = 1.0f / 255.0f;
constexpr float kWhitePixelEpsilon = 1.0f - (1.0f / 255.0f);
constexpr float kNearSilentRms = 0.001f;  // -60 dBFS

namespace fs = std::filesystem;

bool IsNumericType(arrow::Type::type id) {
    switch (id) {
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::HALF_FLOAT:
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
            return true;
        default:
            return false;
    }
}

template <typename ArrowType>
bool IsTypedColumnConstant(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    const std::function<bool()>& should_cancel,
    bool& cancelled) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using ValueType = typename ArrowType::c_type;

    bool saw_value = false;
    ValueType first_value{};
    int64_t visited = 0;
    for (const auto& chunk : column->chunks()) {
        auto values = std::static_pointer_cast<ArrayType>(chunk);
        for (int64_t row = 0; row < values->length(); ++row, ++visited) {
            if ((visited & 4095) == 0 && should_cancel && should_cancel()) {
                cancelled = true;
                return false;
            }
            if (values->IsNull(row)) continue;
            const ValueType value = values->Value(row);
            if (!saw_value) {
                first_value = value;
                saw_value = true;
            } else if (value != first_value) {
                return false;
            }
        }
    }
    return saw_value;
}

bool IsNumericColumnConstant(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    const std::function<bool()>& should_cancel,
    bool& cancelled) {
    switch (column->type()->id()) {
        case arrow::Type::HALF_FLOAT:
            return IsTypedColumnConstant<arrow::HalfFloatType>(
                column, should_cancel, cancelled);
        case arrow::Type::FLOAT:
            return IsTypedColumnConstant<arrow::FloatType>(
                column, should_cancel, cancelled);
        case arrow::Type::DOUBLE:
            return IsTypedColumnConstant<arrow::DoubleType>(
                column, should_cancel, cancelled);
        case arrow::Type::INT8:
            return IsTypedColumnConstant<arrow::Int8Type>(
                column, should_cancel, cancelled);
        case arrow::Type::INT16:
            return IsTypedColumnConstant<arrow::Int16Type>(
                column, should_cancel, cancelled);
        case arrow::Type::INT32:
            return IsTypedColumnConstant<arrow::Int32Type>(
                column, should_cancel, cancelled);
        case arrow::Type::INT64:
            return IsTypedColumnConstant<arrow::Int64Type>(
                column, should_cancel, cancelled);
        case arrow::Type::UINT8:
            return IsTypedColumnConstant<arrow::UInt8Type>(
                column, should_cancel, cancelled);
        case arrow::Type::UINT16:
            return IsTypedColumnConstant<arrow::UInt16Type>(
                column, should_cancel, cancelled);
        case arrow::Type::UINT32:
            return IsTypedColumnConstant<arrow::UInt32Type>(
                column, should_cancel, cancelled);
        case arrow::Type::UINT64:
            return IsTypedColumnConstant<arrow::UInt64Type>(
                column, should_cancel, cancelled);
        default:
            return false;
    }
}

struct NumericHealthSample {
    bool saw_nan = false;
    bool saw_inf = false;
};

template <typename ArrowType>
NumericHealthSample SampleTypedColumnHealth(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    int64_t sample_count,
    const std::function<bool()>& should_cancel,
    bool& cancelled) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using ValueType = typename ArrowType::c_type;

    NumericHealthSample health;
    const int64_t length = column->length();
    int chunk_index = 0;
    int64_t chunk_start = 0;
    for (int64_t sample = 0; sample < sample_count; ++sample) {
        if ((sample & 255) == 0 && should_cancel && should_cancel()) {
            cancelled = true;
            return health;
        }
        const int64_t row = sample_count == length
            ? sample
            : (sample_count <= 1
                ? 0
                : static_cast<int64_t>(
                    (static_cast<long double>(sample) * (length - 1)) /
                    (sample_count - 1)));
        while (chunk_index < column->num_chunks() &&
               row >= chunk_start + column->chunk(chunk_index)->length()) {
            chunk_start += column->chunk(chunk_index)->length();
            ++chunk_index;
        }
        if (chunk_index >= column->num_chunks()) break;

        auto values = std::static_pointer_cast<ArrayType>(
            column->chunk(chunk_index));
        const int64_t local_row = row - chunk_start;
        if (values->IsNull(local_row)) continue;
        if constexpr (std::is_floating_point_v<ValueType>) {
            const double value = static_cast<double>(values->Value(local_row));
            health.saw_nan |= std::isnan(value);
            health.saw_inf |= !std::isfinite(value) && !std::isnan(value);
        }
    }
    return health;
}

NumericHealthSample SampleNumericColumnHealth(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    int64_t sample_count,
    const std::function<bool()>& should_cancel,
    bool& cancelled) {
    switch (column->type()->id()) {
        case arrow::Type::FLOAT:
            return SampleTypedColumnHealth<arrow::FloatType>(
                column, sample_count, should_cancel, cancelled);
        case arrow::Type::DOUBLE:
            return SampleTypedColumnHealth<arrow::DoubleType>(
                column, sample_count, should_cancel, cancelled);
        default:
            return {};
    }
}

std::string LowerExtension(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return ext;
}

bool IsArrowNativeTableExtension(const std::string& path) {
    const std::string ext = LowerExtension(path);
    return ext == ".parquet" || ext == ".pq" ||
           ext == ".feather" || ext == ".fea" ||
           ext == ".arrow" || ext == ".ipc";
}

void AuditBasicCounts(DatasetAuditResult& result,
                      bool labels_expected,
                      bool class_counts_available) {
    if (result.sample_count <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "empty_dataset",
                   "Dataset has no samples.");
    }

    if (labels_expected && result.class_count <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_classes",
                   "Dataset has no classes or labels.");
    } else if (labels_expected && result.class_count == 1) {
        result.Add(DatasetAuditSeverity::Warning,
                   "single_class",
                   "Dataset has only one class; classification training will not learn a useful boundary.");
    }

    if (class_counts_available && result.class_count > result.sample_count &&
        result.sample_count > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "class_count_exceeds_samples",
                   "Dataset reports more classes than samples.");
    }
}

void AuditClassNameList(DatasetAuditResult& result,
                        size_t reported_class_count,
                        const std::vector<std::string>& class_names) {
    if (!class_names.empty() && reported_class_count != class_names.size()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "class_name_count_mismatch",
                   "Reported class count does not match the class name list.");
    }
}

bool IsSupportedImageExtension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    static const std::vector<std::string> kExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tga", ".tiff", ".webp"
    };
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

bool IsSupportedAudioExtension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    static const std::vector<std::string> kExtensions = {
        ".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"
    };
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

bool IsSupportedTextExtension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    static const std::vector<std::string> kExtensions = {
        ".txt", ".text", ".md"
    };
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

bool IsWhitespaceOnly(const std::string& value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
}

bool HasUtf8ReplacementMarker(const std::string& value) {
    return value.find("\xEF\xBF\xBD") != std::string::npos;
}

bool LooksBinaryText(const std::string& value) {
    return std::find(value.begin(), value.end(), '\0') != value.end();
}

bool ReadCSVRowForAudit(std::istream& in,
                        char delimiter,
                        std::vector<std::string>& out_fields) {
    out_fields.clear();
    std::string field;
    bool in_quotes = false;
    bool any_char_read = false;
    int c;

    while ((c = in.get()) != EOF) {
        any_char_read = true;
        if (in_quotes) {
            if (c == '"') {
                if (in.peek() == '"') {
                    field.push_back('"');
                    in.get();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(static_cast<char>(c));
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == delimiter) {
                out_fields.push_back(std::move(field));
                field.clear();
            } else if (c == '\r') {
                if (in.peek() == '\n') in.get();
                out_fields.push_back(std::move(field));
                return true;
            } else if (c == '\n') {
                out_fields.push_back(std::move(field));
                return true;
            } else {
                field.push_back(static_cast<char>(c));
            }
        }
    }

    if (any_char_read) {
        out_fields.push_back(std::move(field));
        return true;
    }
    return false;
}

std::vector<fs::path> CollectImagePaths(const std::string& folder) {
    std::vector<fs::path> paths;
    if (folder.empty()) return paths;

    std::error_code ec;
    if (!fs::exists(folder, ec) || !fs::is_directory(folder, ec)) {
        return paths;
    }

    for (const auto& entry : fs::recursive_directory_iterator(folder, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        if (IsSupportedImageExtension(entry.path().extension().string())) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<fs::path> CollectTextPaths(const std::string& folder) {
    std::vector<fs::path> paths;
    if (folder.empty()) return paths;

    std::error_code ec;
    if (!fs::exists(folder, ec) || !fs::is_directory(folder, ec)) {
        return paths;
    }

    for (const auto& entry : fs::recursive_directory_iterator(folder, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        if (IsSupportedTextExtension(entry.path().extension().string())) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<fs::path> CollectAudioPaths(const std::string& folder) {
    std::vector<fs::path> paths;
    if (folder.empty()) return paths;

    std::error_code ec;
    if (!fs::exists(folder, ec) || !fs::is_directory(folder, ec)) {
        return paths;
    }

    for (const auto& entry : fs::recursive_directory_iterator(folder, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        if (IsSupportedAudioExtension(entry.path().extension().string())) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string ClassNameFromPath(const fs::path& root, const fs::path& file) {
    std::error_code ec;
    const fs::path rel = fs::relative(file.parent_path(), root, ec);
    if (ec || rel.empty() || rel == ".") {
        return {};
    }
    return (*rel.begin()).string();
}

std::string ExamplePath(const fs::path& root, const fs::path& file) {
    std::error_code ec;
    const fs::path rel = fs::relative(file, root, ec);
    return (ec || rel.empty()) ? file.filename().generic_string()
                               : rel.generic_string();
}

void AddIssueExample(std::vector<std::string>& examples,
                     const fs::path& root,
                     const fs::path& file) {
    if (examples.size() >= kMaxIssueExamples) {
        return;
    }
    examples.push_back(ExamplePath(root, file));
}

void AuditClassBadRatios(DatasetAuditResult& result,
                         const std::map<std::string, size_t>& sampled_by_class,
                         const std::map<std::string, size_t>& bad_by_class,
                         const std::string& issue_code,
                         const std::string& item_name) {
    for (const auto& [class_name, bad_count] : bad_by_class) {
        auto sampled_it = sampled_by_class.find(class_name);
        if (sampled_it == sampled_by_class.end() || sampled_it->second == 0) {
            continue;
        }

        const size_t sampled = sampled_it->second;
        if (bad_count * 5 <= sampled) {
            continue;
        }

        result.Add(DatasetAuditSeverity::Error,
                   issue_code,
                   "Class '" + class_name + "' has " +
                       std::to_string(bad_count) + "/" +
                       std::to_string(sampled) + " suspicious " +
                       item_name + " in the audit sample, exceeding the 20% refusal threshold.");
    }
}

void AuditImageSampleFiles(DatasetAuditResult& result,
                           const std::string& folder,
                           bool class_subdirs) {
    const auto paths = CollectImagePaths(folder);
    if (paths.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "no_image_files_found",
                   "Image audit could not find decodable image file candidates in the source folder.");
        return;
    }

    const size_t samples_to_check = std::min(paths.size(), kMaxImageAuditSamples);
    size_t zero_byte = 0;
    size_t decode_failed = 0;
    size_t tiny = 0;
    size_t blank_black = 0;
    size_t blank_white = 0;
    std::map<std::string, size_t> sampled_by_class;
    std::map<std::string, size_t> bad_by_class;
    std::vector<std::string> examples;
    const fs::path root(folder);

    for (size_t i = 0; i < samples_to_check; ++i) {
        const size_t idx = paths.size() == samples_to_check
            ? i
            : (i * paths.size()) / samples_to_check;
        const auto& path = paths[idx];
        const std::string class_name = class_subdirs
            ? ClassNameFromPath(root, path)
            : std::string();
        if (!class_name.empty()) {
            ++sampled_by_class[class_name];
        }
        bool suspicious = false;

        std::error_code ec;
        if (fs::file_size(path, ec) == 0 && !ec) {
            ++zero_byte;
            suspicious = true;
        } else {
            std::vector<float> data;
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!ImageUtils::LoadImage(path.string(), data, width, height, channels)) {
                ++decode_failed;
                suspicious = true;
            } else if (width <= 1 || height <= 1 || channels <= 0 || data.empty()) {
                ++tiny;
                suspicious = true;
            } else {
                bool all_black = true;
                bool all_white = true;
                for (float value : data) {
                    if (value > kBlackPixelEpsilon) all_black = false;
                    if (value < kWhitePixelEpsilon) all_white = false;
                    if (!all_black && !all_white) break;
                }
                if (all_black) {
                    ++blank_black;
                    suspicious = true;
                } else if (all_white) {
                    ++blank_white;
                    suspicious = true;
                }
            }
        }

        if (suspicious && !class_name.empty()) {
            ++bad_by_class[class_name];
        }
        if (suspicious) {
            AddIssueExample(examples, root, path);
        }
    }

    const size_t bad = zero_byte + decode_failed + tiny + blank_black + blank_white;
    if (bad == 0) {
        return;
    }

    std::string message = "Image sample audit found " + std::to_string(bad) +
        "/" + std::to_string(samples_to_check) + " suspicious files";
    message += " (zero-byte=" + std::to_string(zero_byte);
    message += ", decode-failed=" + std::to_string(decode_failed);
    message += ", tiny=" + std::to_string(tiny);
    message += ", black=" + std::to_string(blank_black);
    message += ", white=" + std::to_string(blank_white) + ").";

    result.Add(DatasetAuditSeverity::Warning,
               bad * 5 > samples_to_check ? "high_bad_image_sample_ratio"
                                           : "bad_image_samples",
               std::move(message),
               std::move(examples));

    AuditClassBadRatios(result, sampled_by_class, bad_by_class,
                        "high_bad_image_class_ratio", "images");
}

void AuditAudioSampleFiles(DatasetAuditResult& result,
                           const std::string& folder,
                           int target_sample_rate,
                           bool class_subdirs) {
    const auto paths = CollectAudioPaths(folder);
    if (paths.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "no_audio_files_found",
                   "Audio audit could not find audio file candidates in the source folder.");
        return;
    }

    const size_t samples_to_check = std::min(paths.size(), kMaxAudioAuditSamples);
    size_t zero_byte = 0;
    size_t decode_failed = 0;
    size_t all_zero = 0;
    size_t near_silent = 0;
    std::map<std::string, size_t> sampled_by_class;
    std::map<std::string, size_t> bad_by_class;
    std::vector<std::string> examples;
    const fs::path root(folder);

    for (size_t i = 0; i < samples_to_check; ++i) {
        const size_t idx = paths.size() == samples_to_check
            ? i
            : (i * paths.size()) / samples_to_check;
        const auto& path = paths[idx];
        const std::string class_name = class_subdirs
            ? ClassNameFromPath(root, path)
            : std::string();
        if (!class_name.empty()) {
            ++sampled_by_class[class_name];
        }
        bool suspicious = false;

        std::error_code ec;
        if (fs::file_size(path, ec) == 0 && !ec) {
            ++zero_byte;
            suspicious = true;
        } else {
            AudioData audio = AudioProcessing::LoadAudio(path.string(), target_sample_rate);
            if (!audio.valid || audio.samples.empty()) {
                ++decode_failed;
                suspicious = true;
            } else {
                double sum_squares = 0.0;
                bool any_nonzero = false;
                for (float sample : audio.samples) {
                    if (sample != 0.0f) any_nonzero = true;
                    sum_squares += static_cast<double>(sample) * static_cast<double>(sample);
                }

                if (!any_nonzero) {
                    ++all_zero;
                    suspicious = true;
                } else {
                    const double rms = std::sqrt(
                        sum_squares / static_cast<double>(audio.samples.size()));
                    if (rms < kNearSilentRms) {
                        ++near_silent;
                        suspicious = true;
                    }
                }
            }
        }

        if (suspicious && !class_name.empty()) {
            ++bad_by_class[class_name];
        }
        if (suspicious) {
            AddIssueExample(examples, root, path);
        }
    }

    const size_t bad = zero_byte + decode_failed + all_zero + near_silent;
    if (bad == 0) {
        return;
    }

    std::string message = "Audio sample audit found " + std::to_string(bad) +
        "/" + std::to_string(samples_to_check) + " suspicious files";
    message += " (zero-byte=" + std::to_string(zero_byte);
    message += ", decode-failed=" + std::to_string(decode_failed);
    message += ", all-zero=" + std::to_string(all_zero);
    message += ", below -60 dBFS RMS=" + std::to_string(near_silent) + ").";

    result.Add(DatasetAuditSeverity::Warning,
               bad * 5 > samples_to_check ? "high_low_energy_audio_sample_ratio"
                                           : "low_energy_audio_samples",
               std::move(message),
               std::move(examples));

    AuditClassBadRatios(result, sampled_by_class, bad_by_class,
                        "high_low_energy_audio_class_ratio", "audio files");
}

void AuditTextValues(DatasetAuditResult& result,
                     size_t sampled,
                     size_t empty,
                     size_t single_char,
                     size_t replacement_markers,
                     size_t binary_like,
                     std::vector<std::string> examples) {
    if (sampled == 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "no_text_samples_read",
                   "Text audit could not read any source text samples.");
        return;
    }

    if (empty > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "empty_text_samples",
                   "Text sample audit found " + std::to_string(empty) +
                       "/" + std::to_string(sampled) + " empty samples.",
                   examples);
    }
    if (single_char > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "single_character_text_samples",
                   "Text sample audit found " + std::to_string(single_char) +
                       "/" + std::to_string(sampled) + " single-character samples.");
    }
    if (replacement_markers > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "text_encoding_replacement_markers",
                   "Text sample audit found UTF-8 replacement markers in " +
                       std::to_string(replacement_markers) + "/" +
                       std::to_string(sampled) + " samples.");
    }
    if (binary_like > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "binary_like_text_samples",
                   "Text sample audit found NUL bytes in " +
                       std::to_string(binary_like) + "/" +
                       std::to_string(sampled) + " samples.");
    }
}

void CountTextSample(const std::string& text,
                     const std::string& example,
                     size_t& sampled,
                     size_t& empty,
                     size_t& single_char,
                     size_t& replacement_markers,
                     size_t& binary_like,
                     std::vector<std::string>& examples) {
    ++sampled;
    const bool empty_text = text.empty() || IsWhitespaceOnly(text);
    if (empty_text) {
        ++empty;
        if (examples.size() < kMaxIssueExamples) examples.push_back(example);
    } else if (text.size() == 1) {
        ++single_char;
    }
    if (HasUtf8ReplacementMarker(text)) {
        ++replacement_markers;
    }
    if (LooksBinaryText(text)) {
        ++binary_like;
    }
}

void AuditDelimitedTextSource(DatasetAuditResult& result,
                              const std::string& path,
                              const std::string& text_column) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "text_source_unreadable",
                   "Text audit could not open source file.");
        return;
    }

    const char delimiter = fs::path(path).extension() == ".tsv" ? '\t' : ',';
    std::vector<std::string> headers;
    if (!ReadCSVRowForAudit(file, delimiter, headers)) {
        result.Add(DatasetAuditSeverity::Warning,
                   "empty_text_source_file",
                   "Text source file appears empty.");
        return;
    }

    int text_idx = -1;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        if (headers[i] == text_column) {
            text_idx = i;
            break;
        }
    }
    if (text_idx < 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "text_column_not_found_in_source_sample",
                   "Configured text column was not found in the sampled source header.");
        return;
    }

    size_t sampled = 0;
    size_t empty = 0;
    size_t single_char = 0;
    size_t replacement_markers = 0;
    size_t binary_like = 0;
    std::vector<std::string> examples;
    std::vector<std::string> fields;
    while (sampled < kMaxTextAuditSamples &&
           ReadCSVRowForAudit(file, delimiter, fields)) {
        const std::string text = text_idx < static_cast<int>(fields.size())
            ? fields[text_idx]
            : std::string();
        CountTextSample(text, "row " + std::to_string(sampled + 2),
                        sampled, empty, single_char, replacement_markers,
                        binary_like, examples);
    }

    AuditTextValues(result, sampled, empty, single_char, replacement_markers,
                    binary_like, std::move(examples));
}

void AuditJsonTextSource(DatasetAuditResult& result,
                         const std::string& path,
                         const std::string& text_column) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "text_source_unreadable",
                   "Text audit could not open source file.");
        return;
    }

    size_t sampled = 0;
    size_t empty = 0;
    size_t single_char = 0;
    size_t replacement_markers = 0;
    size_t binary_like = 0;
    std::vector<std::string> examples;

    auto count_json_object = [&](const nlohmann::json& obj,
                                 const std::string& example) {
        std::string text;
        if (obj.is_object() && obj.contains(text_column) &&
            obj[text_column].is_string()) {
            text = obj[text_column].get<std::string>();
        }
        CountTextSample(text, example, sampled, empty, single_char,
                        replacement_markers, binary_like, examples);
    };

    std::string lowered_ext = fs::path(path).extension().string();
    std::transform(lowered_ext.begin(), lowered_ext.end(), lowered_ext.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });

    if (lowered_ext == ".jsonl") {
        std::string line;
        while (sampled < kMaxTextAuditSamples && std::getline(file, line)) {
            if (line.empty()) continue;
            try {
                auto obj = nlohmann::json::parse(line);
                count_json_object(obj, "line " + std::to_string(sampled + 1));
            } catch (...) {
                result.Add(DatasetAuditSeverity::Warning,
                           "malformed_json_text_sample",
                           "Text sample audit found malformed JSONL content.");
                return;
            }
        }
    } else {
        std::string content;
        content.assign(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
        if (content.size() > kMaxTextFileBytesToRead) {
            content.resize(kMaxTextFileBytesToRead);
        }
        try {
            auto data = nlohmann::json::parse(content);
            if (data.is_array()) {
                for (const auto& obj : data) {
                    if (sampled >= kMaxTextAuditSamples) break;
                    count_json_object(obj, "item " + std::to_string(sampled + 1));
                }
            } else {
                count_json_object(data, "item 1");
            }
        } catch (...) {
            result.Add(DatasetAuditSeverity::Warning,
                       "malformed_json_text_sample",
                       "Text sample audit found malformed JSON content.");
            return;
        }
    }

    AuditTextValues(result, sampled, empty, single_char, replacement_markers,
                    binary_like, std::move(examples));
}

void AuditPlainTextSource(DatasetAuditResult& result,
                          const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "text_source_unreadable",
                   "Text audit could not open source file.");
        return;
    }

    size_t sampled = 0;
    size_t empty = 0;
    size_t single_char = 0;
    size_t replacement_markers = 0;
    size_t binary_like = 0;
    size_t bytes_read = 0;
    std::vector<std::string> examples;
    std::string line;
    while (sampled < kMaxTextAuditSamples &&
           bytes_read < kMaxTextFileBytesToRead &&
           std::getline(file, line)) {
        bytes_read += line.size() + 1;
        CountTextSample(line, "line " + std::to_string(sampled + 1),
                        sampled, empty, single_char, replacement_markers,
                        binary_like, examples);
    }

    AuditTextValues(result, sampled, empty, single_char, replacement_markers,
                    binary_like, std::move(examples));
}

void AuditTextFolderSource(DatasetAuditResult& result,
                           const std::string& folder) {
    const auto paths = CollectTextPaths(folder);
    if (paths.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "no_text_files_found",
                   "Text audit could not find text files in the source folder.");
        return;
    }

    const size_t samples_to_check = std::min(paths.size(), kMaxTextAuditSamples);
    size_t sampled = 0;
    size_t empty = 0;
    size_t single_char = 0;
    size_t replacement_markers = 0;
    size_t binary_like = 0;
    std::vector<std::string> examples;
    const fs::path root(folder);

    for (size_t i = 0; i < samples_to_check; ++i) {
        const size_t idx = paths.size() == samples_to_check
            ? i
            : (i * paths.size()) / samples_to_check;
        const auto& path = paths[idx];
        std::ifstream file(path, std::ios::binary);
        std::string content;
        if (file.is_open()) {
            content.assign(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
            if (content.size() > kMaxTextFileBytesToRead) {
                content.resize(kMaxTextFileBytesToRead);
            }
        }
        CountTextSample(content, ExamplePath(root, path),
                        sampled, empty, single_char, replacement_markers,
                        binary_like, examples);
    }

    AuditTextValues(result, sampled, empty, single_char, replacement_markers,
                    binary_like, std::move(examples));
}

void AuditTextSourceSample(DatasetAuditResult& result,
                           const DataRegistry::TextDatasetEntry& entry) {
    if (entry.source_path.empty()) {
        return;
    }

    std::error_code ec;
    if (fs::is_directory(entry.source_path, ec)) {
        AuditTextFolderSource(result, entry.source_path);
        return;
    }

    const std::string lowered = LowerExtension(entry.source_path);
    if (lowered == ".csv" || lowered == ".tsv") {
        AuditDelimitedTextSource(result, entry.source_path, entry.text_column);
    } else if (lowered == ".json" || lowered == ".jsonl") {
        AuditJsonTextSource(result, entry.source_path, entry.text_column);
    } else {
        AuditPlainTextSource(result, entry.source_path);
    }
}

std::string ScalarKey(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return "<null>";
    }
    return scalar->ToString();
}

void AuditArrowColumns(DatasetAuditResult& result,
                       const std::shared_ptr<arrow::Table>& table,
                       const std::string& label_column,
                       const DatasetAuditOptions& options,
                       bool enforce_degenerate_threshold = true) {
    if (!table) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_arrow_table",
                   "Arrow table is not available.");
        return;
    }

    int feature_columns = 0;
    int degenerate_feature_columns = 0;
    bool sampled_numeric_values = false;

    const auto should_cancel = [&]() {
        return options.should_cancel && options.should_cancel();
    };
    if (options.report_progress) {
        options.report_progress(0.0f, "Auditing columns");
    }

    for (int i = 0; i < table->num_columns(); ++i) {
        if (should_cancel()) {
            result.cancelled = true;
            return;
        }

        auto column = table->column(i);
        if (!column) continue;

        const std::string name = table->field(i)->name();
        const bool is_label = name == label_column;
        if (!is_label) {
            ++feature_columns;
        }

        bool degenerate = false;
        if (column->length() > 0 && column->null_count() == column->length()) {
            result.Add(DatasetAuditSeverity::Warning,
                       "all_null_column",
                       "Column '" + name + "' contains only null values.");
            degenerate = !is_label;
        }

        auto type = column->type();
        if (!type || !IsNumericType(type->id())) {
            if (degenerate) {
                ++degenerate_feature_columns;
            }
            if (options.report_progress) {
                options.report_progress(
                    0.9f * static_cast<float>(i + 1) /
                        static_cast<float>(std::max(1, table->num_columns())),
                    "Auditing column " + std::to_string(i + 1) + "/" +
                        std::to_string(table->num_columns()));
            }
            continue;
        }

        bool constant_scan_cancelled = false;
        const bool exact_constant = IsNumericColumnConstant(
            column, options.should_cancel, constant_scan_cancelled);
        if (constant_scan_cancelled) {
            result.cancelled = true;
            return;
        }

        const int64_t length = column->length();
        const int64_t configured_limit =
            options.max_numeric_samples_per_column;
        const int64_t sample_count = configured_limit <= 0
            ? length
            : std::min(length, configured_limit);
        sampled_numeric_values |= sample_count < length;

        bool health_scan_cancelled = false;
        const auto health = SampleNumericColumnHealth(
            column, sample_count, options.should_cancel,
            health_scan_cancelled);
        if (health_scan_cancelled) {
            result.cancelled = true;
            return;
        }

        if (health.saw_nan) {
            result.Add(DatasetAuditSeverity::Warning,
                       "nan_values",
                       "Column '" + name + "' contains NaN values.");
        }
        if (health.saw_inf) {
            result.Add(DatasetAuditSeverity::Warning,
                       "infinite_values",
                       "Column '" + name + "' contains infinite values.");
        }
        if (exact_constant && name != label_column) {
            result.Add(DatasetAuditSeverity::Warning,
                       "constant_column",
                       "Column '" + name + "' is constant.");
            degenerate = true;
        }

        if (degenerate) {
            ++degenerate_feature_columns;
        }

        if (options.report_progress) {
            options.report_progress(
                0.9f * static_cast<float>(i + 1) /
                    static_cast<float>(std::max(1, table->num_columns())),
                "Auditing column " + std::to_string(i + 1) + "/" +
                    std::to_string(table->num_columns()));
        }
    }

    if (sampled_numeric_values) {
        result.Add(
            DatasetAuditSeverity::Info,
            "numeric_health_scan_sampled",
            "Numeric NaN/infinity checks sampled up to " +
                std::to_string(options.max_numeric_samples_per_column) +
                " evenly spaced values per column; constant checks used the complete columns.");
    }

    if (enforce_degenerate_threshold &&
        feature_columns > 0 &&
        degenerate_feature_columns * 2 > feature_columns) {
        result.Add(DatasetAuditSeverity::Error,
                   "too_many_degenerate_columns",
                   std::to_string(degenerate_feature_columns) + "/" +
                       std::to_string(feature_columns) +
                       " feature columns are all-null or constant, exceeding the 50% refusal threshold.");
    }
}

void AuditArrowLabels(DatasetAuditResult& result,
                      const std::shared_ptr<arrow::Table>& table,
                      const std::string& label_column,
                      const DatasetAuditOptions& options) {
    if (label_column.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "missing_label_column",
                   "No label column is configured.");
        return;
    }
    if (!table) return;

    auto column = table->GetColumnByName(label_column);
    if (!column) {
        result.Add(DatasetAuditSeverity::Error,
                   "label_column_not_found",
                   "Label column '" + label_column + "' was not found.");
        return;
    }

    std::unordered_map<std::string, int64_t> counts;
    int64_t scanned = 0;
    int64_t null_labels = 0;
    const int64_t scan_limit = std::min<int64_t>(column->length(), kMaxLabelRowsToScan);

    for (const auto& chunk : column->chunks()) {
        if (!chunk || scanned >= scan_limit) continue;
        for (int64_t row = 0; row < chunk->length() && scanned < scan_limit; ++row, ++scanned) {
            if ((scanned & 255) == 0 && options.should_cancel &&
                options.should_cancel()) {
                result.cancelled = true;
                return;
            }
            if (chunk->IsNull(row)) {
                ++null_labels;
                continue;
            }
            auto scalar_result = chunk->GetScalar(row);
            if (!scalar_result.ok()) continue;
            ++counts[ScalarKey(*scalar_result)];
            if (options.report_progress && (scanned & 1023) == 0) {
                options.report_progress(
                    0.9f + 0.1f * static_cast<float>(scanned + 1) /
                        static_cast<float>(std::max<int64_t>(1, scan_limit)),
                    "Auditing labels");
            }
        }
    }

    result.class_count = static_cast<int64_t>(counts.size());
    if (null_labels > 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "null_labels",
                   "Label column contains null labels.");
    }
    if (result.class_count <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "no_label_values",
                   "Label column has no non-null values.");
        return;
    }
    if (result.class_count == 1) {
        result.Add(DatasetAuditSeverity::Warning,
                   "single_class",
                   "Label column contains only one class.");
    }

    int64_t min_count = std::numeric_limits<int64_t>::max();
    int64_t max_count = 0;
    for (const auto& [label, count] : counts) {
        (void)label;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
    }
    if (min_count > 0 && max_count >= min_count * 20 && result.class_count > 1) {
        result.Add(DatasetAuditSeverity::Warning,
                   "severe_class_imbalance",
                   "Largest class has at least 20x more samples than the smallest class.");
    }
    if (column->length() > kMaxLabelRowsToScan) {
        result.Add(DatasetAuditSeverity::Info,
                   "label_scan_sampled",
                   "Label distribution was estimated from the first 100000 rows.");
    }
}

}  // namespace

bool DatasetAuditResult::HasErrors() const {
    return std::any_of(issues.begin(), issues.end(), [](const DatasetAuditIssue& issue) {
        return issue.severity == DatasetAuditSeverity::Error;
    });
}

bool DatasetAuditResult::HasWarnings() const {
    return std::any_of(issues.begin(), issues.end(), [](const DatasetAuditIssue& issue) {
        return issue.severity == DatasetAuditSeverity::Warning;
    });
}

int DatasetAuditResult::ErrorCount() const {
    return static_cast<int>(std::count_if(
        issues.begin(), issues.end(), [](const DatasetAuditIssue& issue) {
            return issue.severity == DatasetAuditSeverity::Error;
        }));
}

int DatasetAuditResult::WarningCount() const {
    return static_cast<int>(std::count_if(
        issues.begin(), issues.end(), [](const DatasetAuditIssue& issue) {
            return issue.severity == DatasetAuditSeverity::Warning;
        }));
}

void DatasetAuditResult::Add(DatasetAuditSeverity severity,
                             std::string code,
                             std::string message,
                             std::vector<std::string> examples) {
    issues.push_back({severity, std::move(code), std::move(message),
                      std::move(examples)});
}

DatasetAuditResult DatasetAudit::AuditTabular(
    const std::string& dataset_name,
    const std::shared_ptr<ArrowDataset>& dataset,
    const std::string& label_column,
    const DatasetAuditOptions& options) {
    DatasetAuditResult result;
    result.dataset_name = dataset_name;
    result.domain = "tabular";

    if (!dataset) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_dataset",
                   "Tabular dataset is not registered.");
        return result;
    }

    result.sample_count = dataset->GetNumRows();
    result.feature_count = dataset->GetNumColumns();
    AuditBasicCounts(result, false, false);

    if (result.feature_count <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "empty_schema",
                   "Tabular dataset has no columns.");
    }

    auto table = dataset->GetArrowTable();
    AuditArrowColumns(result, table, label_column, options);
    if (result.cancelled) return result;
    AuditArrowLabels(result, table, label_column, options);
    if (result.cancelled) return result;
    if (options.report_progress) {
        options.report_progress(1.0f, "Audit complete");
    }
    return result;
}

DatasetAuditResult DatasetAudit::AuditParquet(
    const std::string& dataset_name,
    const std::shared_ptr<ParquetBackedDataset>& dataset,
    const std::string& label_column) {
    DatasetAuditResult result;
    result.dataset_name = dataset_name;
    result.domain = "tabular";

    if (!dataset) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_dataset",
                   "Parquet-backed dataset is not registered.");
        return result;
    }

    result.sample_count = dataset->GetNumRows();
    result.feature_count = dataset->GetNumColumns();
    AuditBasicCounts(result, false, false);
    if (result.feature_count <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "empty_schema",
                   "Parquet-backed dataset has no columns.");
    }

    if (dataset->GetFileSizeBytes() == 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "empty_parquet_file",
                   "Parquet-backed dataset file has zero bytes on disk.");
    }

    auto schema = dataset->GetSchema();
    if (!schema) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_schema",
                   "Parquet-backed dataset schema is not available.");
    } else {
        if (schema->num_fields() != result.feature_count) {
            result.Add(DatasetAuditSeverity::Warning,
                       "schema_column_count_mismatch",
                       "Parquet schema field count does not match dataset column metadata.");
        }

        std::unordered_map<std::string, int> name_counts;
        for (int i = 0; i < schema->num_fields(); ++i) {
            auto field = schema->field(i);
            if (!field) continue;
            ++name_counts[field->name()];
            if (!field->type() || field->type()->id() == arrow::Type::NA) {
                result.Add(DatasetAuditSeverity::Warning,
                           "null_typed_column",
                           "Column '" + field->name() + "' has null/unknown type.");
            }
        }
        for (const auto& [name, count] : name_counts) {
            if (count > 1) {
                result.Add(DatasetAuditSeverity::Error,
                           "duplicate_column_name",
                           "Column name '" + name + "' appears more than once.");
            }
        }
    }

    if (label_column.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "missing_label_column",
                   "No label column is configured.");
    } else {
        const auto columns = dataset->GetColumnNames();
        if (std::find(columns.begin(), columns.end(), label_column) == columns.end()) {
            result.Add(DatasetAuditSeverity::Error,
                       "label_column_not_found",
                       "Label column '" + label_column + "' was not found.");
        }
    }

    const int row_groups = dataset->GetNumRowGroups();
    if (row_groups <= 0) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_row_groups",
                   "Parquet-backed dataset has no row groups.");
        return result;
    }
    for (int i = 0; i < row_groups; ++i) {
        if (dataset->GetRowGroupSize(i) <= 0) {
            result.Add(DatasetAuditSeverity::Warning,
                       "empty_row_group",
                       "Parquet row group " + std::to_string(i) + " has no rows.");
        }
    }
    return result;
}

DatasetAuditResult DatasetAudit::AuditImage(
    const std::string& dataset_name,
    const DataRegistry::ImageDatasetEntry& entry) {
    DatasetAuditResult result;
    result.dataset_name = dataset_name;
    result.domain = "image";
    result.sample_count = static_cast<int64_t>(entry.num_images);
    result.class_count = static_cast<int64_t>(entry.num_classes);
    AuditBasicCounts(result, true, true);
    AuditClassNameList(result, entry.num_classes, entry.class_names);
    if (entry.folder_path.empty()) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_source_path",
                   "Image dataset has no source folder.");
        return result;
    }
    AuditImageSampleFiles(result, entry.folder_path, entry.layout == 0);
    return result;
}

DatasetAuditResult DatasetAudit::AuditAudio(
    const std::string& dataset_name,
    const DataRegistry::AudioDatasetEntry& entry) {
    DatasetAuditResult result;
    result.dataset_name = dataset_name;
    result.domain = "audio";
    result.sample_count = static_cast<int64_t>(entry.num_samples);
    result.class_count = static_cast<int64_t>(entry.num_classes);
    result.feature_count = entry.feature_rows > 0 && entry.feature_cols > 0
        ? static_cast<int64_t>(entry.feature_rows) * entry.feature_cols
        : 0;
    AuditBasicCounts(result, true, true);
    AuditClassNameList(result, entry.num_classes, entry.class_names);
    if (entry.folder_path.empty()) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_source_path",
                   "Audio dataset has no source folder.");
    }
    if (!entry.labeled_subdirs && entry.csv_path.empty()) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_label_csv",
                   "Flat audio layout requires a labels CSV.");
    }
    if (entry.folder_path.empty()) {
        return result;
    }
    AuditAudioSampleFiles(result, entry.folder_path, entry.target_sr,
                          entry.labeled_subdirs);
    return result;
}

DatasetAuditResult DatasetAudit::AuditText(
    const std::string& dataset_name,
    const DataRegistry::TextDatasetEntry& entry) {
    DatasetAuditResult result;
    result.dataset_name = dataset_name;
    result.domain = "text";
    result.sample_count = static_cast<int64_t>(entry.num_samples);
    result.class_count = static_cast<int64_t>(entry.num_classes);
    result.feature_count = entry.max_length;
    AuditBasicCounts(result, entry.has_labels, true);
    AuditClassNameList(result, entry.num_classes, entry.class_names);
    if (entry.source_path.empty()) {
        result.Add(DatasetAuditSeverity::Error,
                   "missing_source_path",
                   "Text dataset has no source path.");
    }
    if (entry.has_labels && entry.label_column.empty()) {
        result.Add(DatasetAuditSeverity::Warning,
                   "missing_label_column",
                   "Text dataset is marked labeled but has no label column.");
    }
    if (entry.vocab_size == 0) {
        result.Add(DatasetAuditSeverity::Warning,
                   "empty_vocabulary",
                   "Text vocabulary is empty.");
    }
    if (!IsArrowNativeTableExtension(entry.source_path)) {
        AuditTextSourceSample(result, entry);
    }
    return result;
}

const char* ToString(DatasetAuditSeverity severity) {
    switch (severity) {
        case DatasetAuditSeverity::Info: return "info";
        case DatasetAuditSeverity::Warning: return "warning";
        case DatasetAuditSeverity::Error: return "error";
    }
    return "unknown";
}

std::string FormatAuditSummary(const DatasetAuditResult& result) {
    const int errors = result.ErrorCount();
    const int warnings = result.WarningCount();
    if (errors == 0 && warnings == 0) {
        return {};
    }

    std::string summary = "Audit:";
    if (errors > 0) {
        summary += " " + std::to_string(errors) + " error";
        if (errors != 1) summary += "s";
    }
    if (warnings > 0) {
        if (errors > 0) summary += ",";
        summary += " " + std::to_string(warnings) + " warning";
        if (warnings != 1) summary += "s";
    }
    summary += ".";
    return summary;
}

std::vector<std::string> FormatAuditIssueLines(const DatasetAuditResult& result) {
    std::vector<std::string> lines;
    lines.reserve(result.issues.size());
    for (const auto& issue : result.issues) {
        std::string line = std::string(ToString(issue.severity)) + " [" +
            issue.code + "]: " + issue.message;
        if (!issue.examples.empty()) {
            line += " Examples: ";
            for (size_t i = 0; i < issue.examples.size(); ++i) {
                if (i > 0) line += ", ";
                line += issue.examples[i];
            }
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

}  // namespace cyxwiz
