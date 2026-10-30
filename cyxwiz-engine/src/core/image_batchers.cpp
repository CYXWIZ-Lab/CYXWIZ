#include "image_batchers.h"

#include "classification_decision.h"
#include "graph_compiler.h"
#include "prepared_image_set.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

namespace cyxwiz {

std::unique_ptr<ImageDatasetBatcher> MakeImageBatcher(const DataRegistry::ImageDatasetEntry& entry,
                                                      const TrainingConfiguration& run, int batch_size,
                                                      const SampleSplitSettings& split, std::function<bool()> stop) {
    // The leading transforms that are the same every epoch and the OpenCV
    // steps are done once with the images (TOFIX178 1.3c, 1.3f); the rest run
    // on each batch.
    const auto transforms = SplitFixedImageTransforms(run.image_augmentation, run.image_cv_steps);
    size_t reported = 0;
    const auto prepared = PrepareImageSet(
        entry, PreparedImageSpecFrom(run.image_preprocessing), transforms.fixed, 0,
        [&](size_t done, size_t total) {
            if (done * 10 / total != reported * 10 / total || done == total) {
                reported = done;
                spdlog::info("Preparing images: {} of {} decoded", done, total);
            }
            return !(stop && stop());
        });
    if (!prepared.set)
        throw std::runtime_error("the images in '" + entry.folder_path + "' could not be prepared: " + prepared.error);
    if (prepared.from_cache)
        spdlog::info("Reading the {} prepared images from {}", prepared.set->Count(), prepared.set->Path().string());

    auto batcher = std::make_unique<ImageDatasetBatcher>(prepared.set, batch_size, split, run.shuffle,
                                                         static_cast<uint32_t>(run.dataloader_seed));
    batcher->SetDropLast(run.drop_last);
    batcher->SetImageTransforms(transforms.per_batch);
    batcher->ExcludeFiles(run.image_excluded_files);
    if (run.preprocessing.has_normalization) {
        batcher->SetNormalization(run.preprocessing.norm_mean, run.preprocessing.norm_std);
        if (!run.preprocessing.norm_channel_mean.empty())
            batcher->SetChannelNormalization(run.preprocessing.norm_channel_mean, run.preprocessing.norm_channel_std);
    }
    if (UsesScalarBinaryTargets(run.loss_type)) {
        batcher->SetScalarLabelMode(true);
    } else if (run.preprocessing.has_onehot && run.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(run.preprocessing.num_classes);
    }
    return batcher;
}

SampleSplitSettings ImageSplitOf(const TrainingConfiguration& run) {
    return {run.train_ratio, run.val_ratio, run.test_ratio, run.stratified, static_cast<uint32_t>(run.split_seed)};
}

ExternalTestSource MakeImageTestSource(const std::string& dataset_name, const DataRegistry::ImageDatasetEntry& entry,
                                       TestDatasetScope scope) {
    ExternalTestSource source;
    source.class_names = entry.ShownClassNames();
    source.source_dataset = DataRegistry::ImageIndexName(dataset_name);
    // Row ids are image numbers: rows of the dataset's index table, files of the set.
    auto files = std::make_shared<std::vector<std::string>>();
    source.row_file = [files](int64_t row) {
        return row >= 0 && static_cast<size_t>(row) < files->size() ? (*files)[static_cast<size_t>(row)]
                                                                     : std::string();
    };
    source.make_batcher = [entry, scope, files](const TrainingConfiguration& config,
                                                int batch_size) -> std::unique_ptr<IBatcher> {
        // A dataset given whole is all test; otherwise the same split training made.
        const SampleSplitSettings split = scope == TestDatasetScope::EntireProvidedDataset
            ? SampleSplitSettings{0.0f, 0.0f, 1.0f, false, static_cast<uint32_t>(config.split_seed)}
            : ImageSplitOf(config);
        auto batcher = MakeImageBatcher(entry, config, batch_size, split);
        batcher->SetPhase(BatcherPhase::Test);
        batcher->Reset();
        *files = batcher->Files();
        if (batcher->GetNumSamples() == 0)
            throw std::runtime_error("no test images: the Data Split's Test ratio is 0 and no Test Dataset is wired");
        spdlog::info("Run Test: {} test images of '{}'", batcher->GetNumSamples(), entry.folder_path);
        return batcher;
    };
    return source;
}

}  // namespace cyxwiz
