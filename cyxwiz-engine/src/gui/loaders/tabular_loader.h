#pragma once

#include "data_loader.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace cyxwiz::loaders {

inline std::string NormalizeTabularFileType(std::string file_type) {
    std::transform(file_type.begin(), file_type.end(), file_type.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return file_type.empty() ? "auto" : file_type;
}

inline bool IsSupportedTabularFileType(const std::string& file_type) {
    const std::string normalized = NormalizeTabularFileType(file_type);
    return normalized == "auto" || normalized == "csv" ||
           normalized == "tsv" || normalized == "parquet" ||
           normalized == "feather" || normalized == "arrow" ||
           normalized == "ipc";
}

inline bool IsUnsupportedTabularFileType(const std::string& file_type) {
    return !IsSupportedTabularFileType(file_type);
}

inline std::string UnsupportedTabularFileTypeMessage(
    const std::string& file_type) {
    const std::string normalized = NormalizeTabularFileType(file_type);
    if (normalized == "json") {
        return "Tabular JSON loading is not supported yet";
    }
    if (normalized == "excel") {
        return "Tabular Excel loading is not supported yet";
    }
    if (normalized == "hdf5") {
        return "Tabular HDF5 loading is not supported yet";
    }
    if (normalized == "txt") {
        return "Tabular TXT loading is not supported on this path; use Text source for text files";
    }
    if (normalized == "arff") {
        return "Tabular ARFF loading is not supported yet";
    }
    return "Tabular file type is not supported yet";
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
