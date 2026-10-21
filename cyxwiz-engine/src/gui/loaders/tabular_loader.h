#pragma once

#include "data_loader.h"
#include "../../core/data_input_formats.h"

#include <string>
#include <utility>

namespace cyxwiz::loaders {

inline std::string NormalizeTabularFileType(std::string file_type) {
    return NormalizeDataInputFormat(std::move(file_type));
}

inline std::string ResolveTabularFileType(
    const std::string& file_type,
    const std::string& source_path) {
    return data_input::ResolveFormat(file_type, source_path);
}

inline bool IsSupportedTabularFileType(const std::string& file_type) {
    return data_input::IsExecutable(file_type);
}

inline bool IsUnsupportedTabularFileType(const std::string& file_type) {
    return !IsSupportedTabularFileType(file_type);
}

inline std::string UnsupportedTabularFileTypeMessage(
    const std::string& file_type) {
    return data_input::UnsupportedReason(file_type, data_input::kHdf5BuildAvailable);
}

inline bool ValidateTabularApplyContext(const ApplyContext& ctx, std::string& err) {
    if (ctx.source_path.empty()) {
        err = "Tabular load needs a file path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
        return false;
    }
    const auto file_type = ResolveTabularFileType(ctx.detected_file_type, ctx.source_path);
    if (file_type == "zip_text" && ctx.archive_member.empty()) {
        err = "ZIP text requires an exact member path inside the archive";
        return false;
    }
    if (file_type == "zip_text" && ctx.force_disk_backed) {
        err = "ZIP text uses a bounded in-memory document table; disable Force disk-backed";
        return false;
    }
    if (IsUnsupportedTabularFileType(file_type)) {
        err = UnsupportedTabularFileTypeMessage(file_type);
        return false;
    }
    const char delimiter = file_type == "tsv" ? '\t' : ctx.delimiter;
    if ((file_type == "csv" || file_type == "tsv") &&
        (delimiter == '\0' || ctx.decimal_point == '\0')) {
        err = "CSV delimiter and decimal separator must each be one character";
        return false;
    }
    if ((file_type == "csv" || file_type == "tsv") && delimiter == ctx.decimal_point) {
        err = "CSV delimiter and decimal separator must be different";
        return false;
    }
    err.clear();
    return true;
}

// Handles FileCategory::Tabular (and FileCategory::TimeSeries, which
// shares the same load path). Storage backend (Arrow in-memory vs
// Parquet disk-backed) is chosen inside LaunchAsyncLoad based on file
// size and ApplyContext::force_disk_backed — the worker overrides
// BackendTag() from 1 to 2 when it ends up on the disk-backed path.
class TabularLoader : public DataLoader {
public:
    FileCategory Category() const override { return FileCategory::Tabular; }
    const char* CategoryName() const override { return "Tabular"; }
    int BackendTag() const override { return 1; }
    bool IsLazyLoaded() const override { return false; }

    bool ValidateApplyContext(const ApplyContext& ctx,
                              std::string& err) const override;
    uint64_t LaunchAsyncLoad(const ApplyContext& ctx,
                             std::shared_ptr<AsyncLoadState> state) override;

    bool IsRegistered(const std::string& name) const override;
    void Unregister(const std::string& name) override;

    bool RestoreFromRegistry(const std::string& name,
                             const gui::MLNode& node,
                             RestoreState& out) const override;
    CompletedLoadDescription DescribeCompletedLoad(
        const AsyncLoadState& state) const override;

    bool LaunchTraining(
        TrainingConfiguration config,
        const std::string& dataset_name,
        const std::string& label_column,
        int epochs,
        int batch_size,
        std::weak_ptr<TrainingPlotPanel> plot_panel,
        std::function<void(bool)> node_editor_callback) override;

    cyxwiz::PreprocessingDomain Domain(
        const std::string& file_category) const override;
    bool LabelsFromStructure() const override { return false; }

    std::vector<ParamSchema> NodeParams() const override;
    SyntheticBatch MakeSynthetic(
        const cyxwiz::TrainingConfiguration& config,
        uint32_t seed) const override;
};

}  // namespace cyxwiz::loaders
