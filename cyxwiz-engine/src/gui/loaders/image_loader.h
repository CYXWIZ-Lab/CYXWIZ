#pragma once

#include "data_loader.h"

namespace cyxwiz::loaders {

// Handles FileCategory::Image. Two layouts, picked by
// ApplyContext::image_layout:
//   0 (ClassSubdirs)  — folder/<class>/*.jpg, ImageNet style
//   1 (FlatWithCSV)   — folder/*.jpg + labels CSV mapping filename→label
// Pixels are loaded lazily by ImageDatasetBatcher at training time;
// this loader only scans the folder to populate num_samples /
// num_classes and surface invalid-layout errors early.
class ImageLoader : public DataLoader {
public:
    FileCategory Category() const override { return FileCategory::Image; }
    const char* CategoryName() const override { return "Image"; }
    int BackendTag() const override { return 3; }
    bool IsLazyLoaded() const override { return true; }

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
        TrainingPlotPanel* plot_panel,
        std::function<void(bool)> node_editor_callback) override;

    cyxwiz::PreprocessingDomain Domain(
        const std::string& file_category) const override;
    bool LabelsFromStructure() const override { return true; }

    std::vector<ParamSchema> NodeParams() const override;
    SyntheticBatch MakeSynthetic(
        const cyxwiz::TrainingConfiguration& config,
        uint32_t seed) const override;
};

}  // namespace cyxwiz::loaders
