#pragma once

#include "dataset_batcher.h"
#include "data_registry.h"
#include "datasets/image_folder_dataset.h"
#include "datasets/image_csv_dataset.h"
#include "../preprocessing/image_transform.h"
#include "../preprocessing/preprocessing_config.h"
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
                        bool shuffle = true);

    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetFlatten(bool flatten) override;

private:
    std::shared_ptr<Dataset> dataset_;
    std::unique_ptr<ImageTransform> transform_;

    int batch_size_;
    bool shuffle_;
    bool flatten_ = true;   // output [batch, H*W*C] — pre-flattened for Dense layers

    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool do_normalize_ = false;

    size_t num_classes_ = 0;
    bool do_onehot_ = false;

    std::vector<size_t> train_indices_;
    std::vector<size_t> epoch_order_;
    size_t current_idx_ = 0;

    int target_width_ = 224;
    int target_height_ = 224;
    int channels_ = 3;

    std::mt19937 rng_;
};

} // namespace cyxwiz
