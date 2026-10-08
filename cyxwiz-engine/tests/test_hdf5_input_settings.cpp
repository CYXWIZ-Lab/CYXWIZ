#include "../src/core/hdf5_input_settings.h"

#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

using Params = std::map<std::string, std::string>;

Params Canonical(std::string data = "/nested/data",
                 std::string labels = "/nested/labels",
                 std::string policy = "preserve",
                 std::string budget = "4096") {
    return {
        {"hdf5_selection_version", "1"},
        {"hdf5_data_path", std::move(data)},
        {"hdf5_label_path", std::move(labels)},
        {"hdf5_import_mode", "numeric_table"},
        {"hdf5_numeric_policy", std::move(policy)},
        {"hdf5_max_materialized_bytes", std::move(budget)},
    };
}

void CheckSettings(const cyxwiz::Hdf5InputSettings& settings,
                   const std::string& data,
                   const std::string& labels,
                   cyxwiz::Hdf5NumericPolicy policy,
                   uint64_t budget) {
    Check(settings.selection.data_path == data, "data path");
    Check(settings.selection.label_path == labels, "label path");
    Check(settings.numeric_policy == policy, "numeric policy");
    Check(settings.max_materialized_bytes == budget, "materialization budget");
}

void CheckReadFailure(const Params& params, const std::string& context) {
    const auto result = cyxwiz::ReadHdf5InputSettings(params);
    Check(!result.ok && !result.error.empty(), context + ": read should fail");
}

void CheckWriteFailure(const cyxwiz::Hdf5InputSettings& settings,
                       const Params& before,
                       const std::string& context) {
    auto params = before;
    std::string error;
    Check(!cyxwiz::WriteHdf5InputSettings(settings, params, error) &&
              !error.empty(),
          context + ": write should fail");
    Check(params == before, context + ": write failure must not mutate params");
}

void TestDefaultsAndCanonicalRoundTrip() {
    auto defaults = cyxwiz::ReadHdf5InputSettings({});
    Check(defaults.ok && defaults.error.empty() && !defaults.migrated_legacy,
          "missing HDF5 keys should read defaults");
    CheckSettings(defaults.settings, "/data", "",
                  cyxwiz::Hdf5NumericPolicy::Preserve, 256ULL * 1024 * 1024);

    Params params = {{"unrelated", "keep"}, {"hdf5_dataset", "legacy"}};
    cyxwiz::Hdf5InputSettings settings;
    settings.selection = {"/nested/data", "/nested/labels"};
    settings.numeric_policy = cyxwiz::Hdf5NumericPolicy::Float64;
    settings.max_materialized_bytes = 4096;
    std::string error;
    Check(cyxwiz::WriteHdf5InputSettings(settings, params, error) &&
              error.empty(),
          "canonical write should succeed");
    Check(params.at("unrelated") == "keep", "canonical write preserves unrelated keys");
    Check(params.find("hdf5_dataset") == params.end(),
          "canonical write erases legacy hdf5_dataset");
    Check(params.at("hdf5_selection_version") == "1", "version key");
    Check(params.at("hdf5_data_path") == "/nested/data", "data path key");
    Check(params.at("hdf5_label_path") == "/nested/labels", "label path key");
    Check(params.at("hdf5_import_mode") == "numeric_table", "import mode key");
    Check(params.at("hdf5_numeric_policy") == "float64", "policy key");
    Check(params.at("hdf5_max_materialized_bytes") == "4096", "budget key");

    const auto read = cyxwiz::ReadHdf5InputSettings(params);
    Check(read.ok && !read.migrated_legacy && read.error.empty(),
          "canonical read should succeed");
    CheckSettings(read.settings, "/nested/data", "/nested/labels",
                  cyxwiz::Hdf5NumericPolicy::Float64, 4096);
}

void TestLegacyMigrationAndCanonicalPrecedence() {
    auto legacy = cyxwiz::ReadHdf5InputSettings({{"hdf5_dataset", "group/data"}});
    Check(legacy.ok && legacy.migrated_legacy, "legacy relative path should migrate");
    CheckSettings(legacy.settings, "/group/data", "",
                  cyxwiz::Hdf5NumericPolicy::Preserve, 256ULL * 1024 * 1024);

    legacy = cyxwiz::ReadHdf5InputSettings({{"hdf5_dataset", "/already/absolute"}});
    Check(legacy.ok && legacy.migrated_legacy, "legacy absolute path should migrate");
    CheckSettings(legacy.settings, "/already/absolute", "",
                  cyxwiz::Hdf5NumericPolicy::Preserve, 256ULL * 1024 * 1024);

    CheckReadFailure({{"hdf5_dataset", ""}}, "empty legacy path");

    auto params = Canonical("/canonical/data", "", "preserve", "123");
    params["hdf5_dataset"] = "legacy/data";
    auto read = cyxwiz::ReadHdf5InputSettings(params);
    Check(read.ok && !read.migrated_legacy, "canonical keys win over legacy");
    CheckSettings(read.settings, "/canonical/data", "",
                  cyxwiz::Hdf5NumericPolicy::Preserve, 123);

    Params malformed = {{"hdf5_selection_version", "2"},
                        {"hdf5_dataset", "legacy/data"}};
    CheckReadFailure(malformed, "malformed canonical must not fallback to legacy");
}

void TestCanonicalMalformedKeys() {
    for (const char* missing : {"hdf5_selection_version", "hdf5_data_path",
                                "hdf5_label_path", "hdf5_import_mode",
                                "hdf5_numeric_policy",
                                "hdf5_max_materialized_bytes"}) {
        auto params = Canonical();
        params.erase(missing);
        CheckReadFailure(params, std::string("missing ") + missing);
    }

    auto bad_version = Canonical();
    bad_version["hdf5_selection_version"] = "01";
    CheckReadFailure(bad_version, "bad version");

    auto bad_mode = Canonical();
    bad_mode["hdf5_import_mode"] = "tensor";
    CheckReadFailure(bad_mode, "bad import mode");

    auto bad_policy = Canonical();
    bad_policy["hdf5_numeric_policy"] = "native";
    CheckReadFailure(bad_policy, "bad policy");

    for (const auto& [budget, label] : {
             std::pair<std::string, std::string>{"", "empty budget"},
             {"0", "zero budget"},
             {"268435457", "over max budget"},
             {"-1", "signed budget"},
             {"+1", "plus budget"},
             {"1x", "trailing budget"},
             {"18446744073709551616", "overflow budget"},
         }) {
        auto params = Canonical("/data", "", "preserve", budget);
        CheckReadFailure(params, label);
    }
}

void TestPathValidation() {
    std::string error;
    cyxwiz::Hdf5InputSettings valid;
    valid.selection = {"/data", ""};
    Check(cyxwiz::ValidateHdf5InputSettings(valid, error) && error.empty(),
          "blank label path is valid");

    valid.selection.label_path = "/labels";
    Check(cyxwiz::ValidateHdf5InputSettings(valid, error), "absolute label path is valid");

    for (const auto& [path, label] : {
             std::pair<std::string, std::string>{"", "empty"},
             {"/", "root"},
             {"relative", "relative"},
             {"/trailing/", "trailing slash"},
             {"/double//slash", "empty segment"},
             {"/dot/./segment", "dot segment"},
             {"/dotdot/../segment", "dotdot segment"},
             {"/" + std::string(4096, 'a'), "too long"},
             {std::string("/nul\0suffix", 11), "nul"},
         }) {
        cyxwiz::Hdf5InputSettings settings;
        settings.selection.data_path = path;
        Check(!cyxwiz::ValidateHdf5InputSettings(settings, error) &&
                  !error.empty(),
              label + " data path should be invalid");
        auto params = Canonical(path, "", "preserve", "1024");
        CheckReadFailure(params, label + " canonical data path");
    }

    for (const auto& label_path : {"relative", "/", "/bad/../label"}) {
        cyxwiz::Hdf5InputSettings settings;
        settings.selection = {"/data", label_path};
        Check(!cyxwiz::ValidateHdf5InputSettings(settings, error) &&
                  !error.empty(),
              std::string(label_path) + " label path should be invalid");
    }
}

void TestTransactionalWriteFailure() {
    const Params before = {{"unrelated", "keep"}, {"hdf5_dataset", "legacy"}};

    cyxwiz::Hdf5InputSettings invalid_path;
    invalid_path.selection.data_path = "relative";
    CheckWriteFailure(invalid_path, before, "relative data path");

    cyxwiz::Hdf5InputSettings invalid_label;
    invalid_label.selection = {"/data", "/bad/../label"};
    CheckWriteFailure(invalid_label, before, "bad label path");

    cyxwiz::Hdf5InputSettings invalid_budget;
    invalid_budget.selection.data_path = "/data";
    invalid_budget.max_materialized_bytes = 0;
    CheckWriteFailure(invalid_budget, before, "zero budget");

    invalid_budget.max_materialized_bytes = 256ULL * 1024 * 1024 + 1;
    CheckWriteFailure(invalid_budget, before, "oversized budget");
}

} // namespace

int main() try {
    TestDefaultsAndCanonicalRoundTrip();
    TestLegacyMigrationAndCanonicalPrecedence();
    TestCanonicalMalformedKeys();
    TestPathValidation();
    TestTransactionalWriteFailure();

    std::cout << "HDF5 input settings: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
