#pragma once

#include "data_loader.h"

namespace cyxwiz::loaders {

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
