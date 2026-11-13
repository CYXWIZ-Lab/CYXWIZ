// The weight file reader against files written by PyTorch and safetensors
// (TOFIX178 1.9c): every tensor's name, shape, type and values, and the files
// that must be refused. Fixtures: tests/computation_truth/reference/
// generate_weight_file_fixtures.py.
#include "core/weight_file.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void Check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

const std::filesystem::path kFixtures = CYXWIZ_WEIGHT_FILE_FIXTURES;

// Every tensor of a good file equals what torch reported when it wrote it.
void CheckFile(const nlohmann::json& expected, const std::string& name, const std::string& format) {
    cyxwiz::weight_file::WeightFile file;
    std::string error;
    const bool read = cyxwiz::weight_file::Read(kFixtures / name, file, error);
    Check(read, name + " is read: " + error);
    if (!read) return;
    Check(file.format == format, name + " is a " + format);
    const auto& entries = expected.at("files").at(name);
    Check(file.entries.size() == entries.size(), name + ": " + std::to_string(entries.size()) + " tensors, got " +
                                                     std::to_string(file.entries.size()));
    for (size_t i = 0; i < entries.size() && i < file.entries.size(); ++i) {
        const auto& want = entries[i];
        const auto& got = file.entries[i];
        const std::string label = name + " " + want.at("name").get<std::string>();
        Check(got.name == want.at("name").get<std::string>(), label + ": name and file order (got " + got.name + ")");
        Check(got.shape == want.at("shape").get<std::vector<size_t>>(), label + ": shape");
        Check(got.dtype == want.at("dtype").get<std::string>(), label + ": type (got " + got.dtype + ")");
        if (!want.contains("values")) {
            Check(!got.is_float && got.values.empty(), label + ": an integer tensor carries no float values");
            continue;
        }
        const auto values = want.at("values").get<std::vector<float>>();
        Check(got.is_float && got.values.size() == values.size(), label + ": value count");
        if (got.values.size() != values.size()) continue;
        bool same = true;
        for (size_t v = 0; v < values.size(); ++v) same = same && got.values[v] == values[v];
        Check(same, label + ": every value, bit for bit");
    }
}

// A file that must not be used says why, and nothing is returned.
void CheckRefused(const std::string& name, const std::string& reason_part) {
    cyxwiz::weight_file::WeightFile file;
    file.format = "left over";
    std::string error;
    const bool read = cyxwiz::weight_file::Read(kFixtures / name, file, error);
    Check(!read, name + " is refused");
    Check(error.find(reason_part) != std::string::npos, name + " says why (\"" + reason_part + "\"): " + error);
    Check(file.entries.empty() && file.format.empty(), name + ": nothing is returned");
}

}  // namespace

// With a path: lists that file's tensors (name, type, shape, sum of values),
// to compare a real published file with what torch.load reports.
int Inspect(const std::filesystem::path& path) {
    cyxwiz::weight_file::WeightFile file;
    std::string error;
    if (!cyxwiz::weight_file::Read(path, file, error)) {
        std::cerr << "refused: " << error << "\n";
        return 1;
    }
    std::cout << file.format << ", " << file.entries.size() << " tensors\n";
    for (const auto& entry : file.entries) {
        double sum = 0.0;
        for (const float value : entry.values) sum += value;
        std::cout << entry.name << " " << entry.dtype << " [";
        for (size_t d = 0; d < entry.shape.size(); ++d) std::cout << (d ? "," : "") << entry.shape[d];
        std::cout << "] " << sum << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1) return Inspect(argv[1]);
    std::ifstream in(kFixtures / "expected.json");
    if (!in.good()) {
        std::cerr << "FAIL: cannot open " << (kFixtures / "expected.json").string() << "\n";
        return 1;
    }
    const auto expected = nlohmann::json::parse(in);

    // What torch.save writes: a module's state_dict with its metadata, a
    // transposed matrix, a slice inside shared storage, float64, float16, an
    // int64 counter and a scalar; the same with pickle protocol 4; and a
    // training checkpoint that keeps its state-dict under "state_dict".
    CheckFile(expected, "tiny_state_dict.pth", "PyTorch state-dict");
    CheckFile(expected, "protocol4.pth", "PyTorch state-dict");
    CheckFile(expected, "checkpoint.pth", "PyTorch state-dict");
    // The same dict in the format before PyTorch 1.6 (older published weights use it).
    CheckFile(expected, "legacy.pth", "PyTorch state-dict (format before PyTorch 1.6)");
    CheckFile(expected, "tiny.safetensors", "safetensors");
    {
        cyxwiz::weight_file::WeightFile file;
        std::string error;
        Check(cyxwiz::weight_file::Read(kFixtures / "tiny.safetensors", file, error) &&
                  file.metadata.count("model_name") && file.metadata.at("model_name") == "tiny",
              "safetensors metadata is kept");
    }

    // Refused, each with its reason. hostile.pth asks the reader to run
    // os.system: the name is only ever text here, so nothing runs.
    CheckRefused("hostile.pth", "'os.system' object, not a plain state-dict");
    CheckRefused("whole_model.pth", "plain state-dict");
    CheckRefused("truncated.pth", "ZIP directory was not found");
    CheckRefused("missing.pth", "not found");
    // The old format at the size of a network: the same tensors as the current format.
    {
        cyxwiz::weight_file::WeightFile old_format, new_format;
        std::string error;
        Check(cyxwiz::weight_file::Read(kFixtures / "mobilenet_legacy.pth", old_format, error),
              "a MobileNetV2 state-dict in the old format is read: " + error);
        Check(cyxwiz::weight_file::Read(kFixtures / "mobilenet_new.pth", new_format, error),
              "the same state-dict in the current format is read: " + error);
        bool same = !old_format.entries.empty() && old_format.entries.size() == new_format.entries.size();
        for (size_t i = 0; same && i < old_format.entries.size(); ++i) {
            const auto& a = old_format.entries[i];
            const auto& b = new_format.entries[i];
            same = a.name == b.name && a.shape == b.shape && a.dtype == b.dtype && a.values == b.values;
        }
        Check(same, "both formats give the same " + std::to_string(new_format.entries.size()) + " tensors, value for value");
    }
    CheckRefused("truncated_legacy.pth", "ends before");
    // Shapes meant to exhaust the stack or the memory.
    CheckRefused("deep.pth", "no state-dict");
    CheckRefused("expanded.pth", "repeats its stored values");
    CheckRefused("overlap.safetensors", "overlap");
    {
        // Names, shapes and types without the values (what the dialog reads).
        cyxwiz::weight_file::WeightFile full, names;
        std::string error;
        Check(cyxwiz::weight_file::Read(kFixtures / "tiny_state_dict.pth", full, error) &&
                  cyxwiz::weight_file::Read(kFixtures / "tiny_state_dict.pth", names, error, false),
              "the file is read with and without its values");
        bool same = full.entries.size() == names.entries.size();
        for (size_t i = 0; same && i < full.entries.size(); ++i) {
            same = full.entries[i].name == names.entries[i].name && full.entries[i].shape == names.entries[i].shape &&
                   full.entries[i].dtype == names.entries[i].dtype && names.entries[i].values.empty();
        }
        Check(same, "without values the names, shapes and types are the same and no value is held");
    }
    {
        const auto other = std::filesystem::temp_directory_path() / "cyxwiz_weight_file_other.bin";
        std::ofstream(other, std::ios::binary) << "not a weight file at all";
        cyxwiz::weight_file::WeightFile file;
        std::string error;
        Check(!cyxwiz::weight_file::Read(other, file, error) && error.find("neither") != std::string::npos,
              "an unrelated file is refused: " + error);
        // A safetensors header that claims more than the file holds.
        const auto bad = std::filesystem::temp_directory_path() / "cyxwiz_weight_file_bad.safetensors";
        {
            const std::string header = R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
            const uint64_t size = header.size();
            std::ofstream out(bad, std::ios::binary);
            out.write(reinterpret_cast<const char*>(&size), 8);
            out << header << "12345678";  // 8 bytes of data, 16 claimed
        }
        Check(!cyxwiz::weight_file::Read(bad, file, error) && error.find("do not fit") != std::string::npos,
              "safetensors offsets past the end are refused: " + error);
        std::filesystem::remove(other);
        std::filesystem::remove(bad);
    }

    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "Weight files read as PyTorch wrote them: " << checks << " checks\n";
    return 0;
}
