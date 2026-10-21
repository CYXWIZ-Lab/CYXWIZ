#pragma once

#include "dataset_batcher.h"
#include "data_registry.h"
#include "datasets/image_folder_dataset.h"
#include "datasets/image_csv_dataset.h"
#include "../preprocessing/preprocessing_config.h"
#include <cyxwiz/image_augmentation.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <random>

namespace cyxwiz {

class ImageDatasetBatcher : public IBatcher {
public:
    ImageDatasetBatcher(const DataRegistry::ImageDatasetEntry& entry,
                        const ImagePreprocessingConfig& preprocess_config,
                        int batch_size,
                        float train_split = 0.8f,
                        bool shuffle = true,
                        int num_workers = 0,
                        uint32_t seed = 42);

    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetScalarLabelMode(bool enable) override { scalar_label_mode_ = enable; }
    void SetFlatten(bool flatten) override;
    void SetDropLast(bool drop_last) override { drop_last_ = drop_last; }
    void SetPhase(BatcherPhase phase) override;
    // The compiled image transforms and batch mix (TOFIX140). They and
    // Normalize run on the ArrayFire device on each whole batch; random ones
    // and the mix on Train batches only.
    void SetImageTransforms(const image::ImageAugmentation& compiled);
    // Leaves these files out (the Quality Analyzer's rejects) and splits the
    // rest again, as if the dataset never had them.
    void ExcludeFiles(const std::vector<std::string>& files);

    size_t GetNumValSamples() const { return val_indices_.size(); }

private:
    // Shuffled train/val split of these dataset indices.
    void Split(std::vector<size_t> indices);

    std::shared_ptr<Dataset> dataset_;
    float train_split_ = 0.8f;
    uint32_t seed_ = 42;

    int batch_size_;
    bool shuffle_;
    int num_workers_ = 0;
    bool drop_last_ = false;
    bool flatten_ = false;  // output [batch, H, W, C] — let graph's Flatten node handle it

    // Transforms, then Normalize, on the device; empty = rows as decoded.
    image::ImageAugmentation augmentation_;
    std::mt19937 augmentation_rng_;

    size_t num_classes_ = 0;
    bool do_onehot_ = false;
    bool scalar_label_mode_ = false;

    // Separate train/val index sets (shuffled split at construction).
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    BatcherPhase current_phase_ = BatcherPhase::Train;

    std::vector<size_t> epoch_order_;
    size_t current_idx_ = 0;

    int target_width_ = 224;
    int target_height_ = 224;
    // The decoded [H, W, 3] image and the sample the model receives.
    image::ImageShape decoded_shape_;
    image::ImageShape sample_shape_;

    std::mt19937 rng_;
};

} // namespace cyxwiz
