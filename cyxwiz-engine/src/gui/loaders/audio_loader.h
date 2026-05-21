#pragma once

#include "data_loader.h"

namespace cyxwiz::loaders {

// Handles FileCategory::Audio. Two layouts picked by
// ApplyContext::audio_labeled_subdirs:
//   true  — folder/<class>/*.wav (ClassSubdirs in the dialog)
//   false — folder/*.wav + CSV with filename + label columns
// Feature extraction (MelSpectrogram by default) runs lazily inside
// AudioDataset::GetItem at training time; this loader only scans
// the folder + optional labels CSV to populate num_samples /
// num_classes / feature_shape and surface invalid layouts.
class AudioLoader : public DataLoader {
public:
    FileCategory Category() const override { return FileCategory::Audio; }
    const char* CategoryName() const override { return "Audio"; }
    int BackendTag() const override { return 4; }
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
        std::weak_ptr<TrainingPlotPanel> plot_panel,
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
