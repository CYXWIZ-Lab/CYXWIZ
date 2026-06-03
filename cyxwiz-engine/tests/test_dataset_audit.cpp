#include "../src/core/arrow_dataset.h"
#include "../src/core/dataset_audit.h"
#include "../src/core/image_utils.h"

#include <arrow/api.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

namespace fs = std::filesystem;

template <typename T>
bool HasIssue(const DatasetAuditResult& result,
              DatasetAuditSeverity severity,
              T code) {
    for (const auto& issue : result.issues) {
        if (issue.severity == severity && issue.code == code) {
            return true;
        }
    }
    return false;
}

bool FormattedIssuesContain(const DatasetAuditResult& result,
                            const std::string& text) {
    const auto lines = FormatAuditIssueLines(result);
    for (const auto& line : lines) {
        if (line.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<ArrowDataset> MakeArrowDataset() {
    arrow::DoubleBuilder feature_builder;
    arrow::DoubleBuilder varying_builder;
    arrow::StringBuilder label_builder;

    assert(feature_builder.Append(1.0).ok());
    assert(feature_builder.Append(1.0).ok());
    assert(feature_builder.Append(1.0).ok());
    assert(feature_builder.Append(1.0).ok());

    assert(varying_builder.Append(0.0).ok());
    assert(varying_builder.Append(1.0).ok());
    assert(varying_builder.Append(0.0).ok());
    assert(varying_builder.Append(1.0).ok());

    assert(label_builder.Append("A").ok());
    assert(label_builder.Append("A").ok());
    assert(label_builder.Append("A").ok());
    assert(label_builder.Append("B").ok());

    std::shared_ptr<arrow::Array> feature_array;
    std::shared_ptr<arrow::Array> varying_array;
    std::shared_ptr<arrow::Array> label_array;
    assert(feature_builder.Finish(&feature_array).ok());
    assert(varying_builder.Finish(&varying_array).ok());
    assert(label_builder.Finish(&label_array).ok());

    auto schema = arrow::schema({
        arrow::field("feature", arrow::float64()),
        arrow::field("varying", arrow::float64()),
        arrow::field("label", arrow::utf8())
    });
    auto table = arrow::Table::Make(schema, {feature_array, varying_array, label_array});
    return std::make_shared<ArrowDataset>(table, "audit_arrow");
}

std::shared_ptr<ArrowDataset> MakeDegenerateArrowDataset() {
    arrow::DoubleBuilder constant_a_builder;
    arrow::DoubleBuilder constant_b_builder;
    arrow::StringBuilder label_builder;

    for (int i = 0; i < 4; ++i) {
        assert(constant_a_builder.Append(1.0).ok());
        assert(constant_b_builder.Append(2.0).ok());
    }
    assert(label_builder.Append("A").ok());
    assert(label_builder.Append("A").ok());
    assert(label_builder.Append("B").ok());
    assert(label_builder.Append("B").ok());

    std::shared_ptr<arrow::Array> constant_a_array;
    std::shared_ptr<arrow::Array> constant_b_array;
    std::shared_ptr<arrow::Array> label_array;
    assert(constant_a_builder.Finish(&constant_a_array).ok());
    assert(constant_b_builder.Finish(&constant_b_array).ok());
    assert(label_builder.Finish(&label_array).ok());

    auto schema = arrow::schema({
        arrow::field("constant_a", arrow::float64()),
        arrow::field("constant_b", arrow::float64()),
        arrow::field("label", arrow::utf8())
    });
    auto table = arrow::Table::Make(
        schema, {constant_a_array, constant_b_array, label_array});
    return std::make_shared<ArrowDataset>(table, "degenerate_arrow");
}

void WriteLe16(std::ofstream& out, int16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}

void WriteLe32(std::ofstream& out, uint32_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
    out.put(static_cast<char>((value >> 16) & 0xff));
    out.put(static_cast<char>((value >> 24) & 0xff));
}

void WriteMonoWav16(const fs::path& path,
                    const std::vector<int16_t>& samples,
                    int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    assert(out.is_open());

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

    out.write("RIFF", 4);
    WriteLe32(out, 36 + data_bytes);
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    WriteLe32(out, 16);
    WriteLe16(out, 1);
    WriteLe16(out, channels);
    WriteLe32(out, static_cast<uint32_t>(sample_rate));
    WriteLe32(out, byte_rate);
    WriteLe16(out, block_align);
    WriteLe16(out, bits_per_sample);

    out.write("data", 4);
    WriteLe32(out, data_bytes);
    for (int16_t sample : samples) {
        WriteLe16(out, sample);
    }
}

void TestTabularAudit() {
    auto dataset = MakeArrowDataset();
    auto result = DatasetAudit::AuditTabular("audit_arrow", dataset, "label");

    assert(!result.HasErrors());
    assert(result.sample_count == 4);
    assert(result.feature_count == 3);
    assert(result.class_count == 2);
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "constant_column"));
    assert(FormatAuditSummary(result) == "Audit: 1 warning.");

    auto missing_label = DatasetAudit::AuditTabular("audit_arrow", dataset, "missing");
    assert(missing_label.HasErrors());
    assert(HasIssue(missing_label, DatasetAuditSeverity::Error, "label_column_not_found"));
}

void TestTabularDegenerateRefusal() {
    auto dataset = MakeDegenerateArrowDataset();
    auto result = DatasetAudit::AuditTabular("degenerate_arrow", dataset, "label");

    assert(result.HasErrors());
    assert(HasIssue(result, DatasetAuditSeverity::Error,
                    "too_many_degenerate_columns"));
}

void TestImageAudit() {
    DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = "images";
    entry.num_images = 8;
    entry.num_classes = 1;
    entry.class_names = {"cat"};

    auto result = DatasetAudit::AuditImage("images", entry);
    assert(!result.HasErrors());
    assert(result.HasWarnings());
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "single_class"));
}

void TestImageSampleAudit() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_dataset_audit_image_test";
    const fs::path class_dir = root / "cat";

    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(class_dir);

    std::vector<float> pixels(4 * 4 * 3, 0.5f);
    assert(ImageUtils::SaveImage((class_dir / "valid.png").string(), pixels, 4, 4, 3));

    {
        std::ofstream bad(class_dir / "bad.png", std::ios::binary);
        bad << "not an image";
    }

    DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = root.string();
    entry.num_images = 2;
    entry.num_classes = 1;
    entry.class_names = {"cat"};

    auto result = DatasetAudit::AuditImage("images", entry);
    assert(result.HasWarnings());
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "bad_image_samples") ||
           HasIssue(result, DatasetAuditSeverity::Warning, "high_bad_image_sample_ratio"));
    assert(HasIssue(result, DatasetAuditSeverity::Error, "high_bad_image_class_ratio"));
    assert(FormattedIssuesContain(result, "cat/bad.png") ||
           FormattedIssuesContain(result, "cat\\bad.png"));

    fs::remove_all(root, ec);
}

void TestAudioSampleAudit() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_dataset_audit_audio_test";
    const fs::path class_dir = root / "drone";

    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(class_dir);

    WriteMonoWav16(class_dir / "signal.wav", {0, 12000, -12000, 6000, -6000}, 16000);
    WriteMonoWav16(class_dir / "silent.wav", {0, 0, 0, 0, 0}, 16000);

    DataRegistry::AudioDatasetEntry entry;
    entry.folder_path = root.string();
    entry.labeled_subdirs = true;
    entry.target_sr = 16000;
    entry.num_samples = 2;
    entry.num_classes = 1;
    entry.class_names = {"drone"};

    auto result = DatasetAudit::AuditAudio("audio", entry);
    assert(result.HasWarnings());
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "low_energy_audio_samples") ||
           HasIssue(result, DatasetAuditSeverity::Warning, "high_low_energy_audio_sample_ratio"));
    assert(HasIssue(result, DatasetAuditSeverity::Error,
                    "high_low_energy_audio_class_ratio"));
    assert(FormattedIssuesContain(result, "drone/silent.wav") ||
           FormattedIssuesContain(result, "drone\\silent.wav"));

    fs::remove_all(root, ec);
}

void TestTextAudit() {
    DataRegistry::TextDatasetEntry entry;
    entry.has_labels = true;
    entry.label_column = "";
    entry.num_samples = 0;
    entry.num_classes = 0;
    entry.vocab_size = 0;

    auto result = DatasetAudit::AuditText("reviews", entry);
    assert(result.HasErrors());
    assert(HasIssue(result, DatasetAuditSeverity::Error, "empty_dataset"));
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "missing_label_column"));
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "empty_vocabulary"));
}

void TestTextCsvSampleAudit() {
    const fs::path path =
        fs::temp_directory_path() / "cyxwiz_dataset_audit_text.csv";
    std::error_code ec;
    fs::remove(path, ec);

    {
        std::ofstream out(path);
        out << "text,label\n";
        out << "hello world,A\n";
        out << ",A\n";
        out << "x,B\n";
        out << "\xEF\xBF\xBD bad encoding,B\n";
    }

    DataRegistry::TextDatasetEntry entry;
    entry.source_path = path.string();
    entry.text_column = "text";
    entry.label_column = "label";
    entry.has_labels = true;
    entry.num_samples = 4;
    entry.num_classes = 2;
    entry.class_names = {"A", "B"};
    entry.vocab_size = 8;

    auto result = DatasetAudit::AuditText("reviews", entry);
    assert(!result.HasErrors());
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "empty_text_samples"));
    assert(HasIssue(result, DatasetAuditSeverity::Warning,
                    "single_character_text_samples"));
    assert(HasIssue(result, DatasetAuditSeverity::Warning,
                    "text_encoding_replacement_markers"));
    assert(FormattedIssuesContain(result, "row 3"));

    fs::remove(path, ec);
}

void TestTextFolderSampleAudit() {
    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_dataset_audit_text_folder";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "positive");

    {
        std::ofstream out(root / "positive" / "good.txt");
        out << "useful sample";
    }
    {
        std::ofstream out(root / "positive" / "empty.txt");
        out << "   ";
    }

    DataRegistry::TextDatasetEntry entry;
    entry.source_path = root.string();
    entry.text_column = "text";
    entry.has_labels = true;
    entry.num_samples = 2;
    entry.num_classes = 1;
    entry.class_names = {"positive"};
    entry.vocab_size = 4;

    auto result = DatasetAudit::AuditText("folder_reviews", entry);
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "empty_text_samples"));
    assert(FormattedIssuesContain(result, "positive/empty.txt") ||
           FormattedIssuesContain(result, "positive\\empty.txt"));

    fs::remove_all(root, ec);
}

void TestTextJsonSampleAudit() {
    const fs::path path =
        fs::temp_directory_path() / "cyxwiz_dataset_audit_text.json";
    std::error_code ec;
    fs::remove(path, ec);

    {
        std::ofstream out(path);
        out << R"([{"text":"hello","label":"A"},{"text":"","label":"B"}])";
    }

    DataRegistry::TextDatasetEntry entry;
    entry.source_path = path.string();
    entry.text_column = "text";
    entry.label_column = "label";
    entry.has_labels = true;
    entry.num_samples = 2;
    entry.num_classes = 2;
    entry.class_names = {"A", "B"};
    entry.vocab_size = 4;

    auto result = DatasetAudit::AuditText("json_reviews", entry);
    assert(!result.HasErrors());
    assert(HasIssue(result, DatasetAuditSeverity::Warning, "empty_text_samples"));
    assert(FormattedIssuesContain(result, "item 2"));

    fs::remove(path, ec);
}

}  // namespace

int main() {
    TestTabularAudit();
    TestTabularDegenerateRefusal();
    TestImageAudit();
    TestImageSampleAudit();
    TestAudioSampleAudit();
    TestTextAudit();
    TestTextCsvSampleAudit();
    TestTextFolderSampleAudit();
    TestTextJsonSampleAudit();
    std::cout << "Dataset audit tests passed\n";
    return 0;
}
