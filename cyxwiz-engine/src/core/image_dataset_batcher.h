#pragma once

#include "dataset_batcher.h"
#include "prepared_image_set.h"
#include <cyxwiz/image_augmentation.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <random>

namespace cyxwiz {

// Batches of an image Data Input from its prepared image set (TOFIX178 1.3b):
// the images were decoded once (PrepareImageSet); a batch gathers its uint8
// rows, uploads them once and turns them into float [0, 1] on the device,
// then the compiled image transforms and Normalize run there. When the whole
// set fits the device budget it is uploaded once and a batch is a lookup on
// the device.
class ImageDatasetBatcher : public IBatcher {
public:
    ~ImageDatasetBatcher() override;
    ImageDatasetBatcher(std::shared_ptr<const PreparedImageSet> images,
                        int batch_size,
                        float train_split = 0.8f,
                        bool shuffle = true,
                        uint32_t seed = 42);

    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    // One mean and std per channel of the batches (TOFIX178 1.4), instead of SetNormalization's pair.
    void SetChannelNormalization(const std::vector<float>& mean, const std::vector<float>& std_dev);
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
    // True once the whole set is on the device (from the first batch on).
    bool ImagesOnDevice() const { return resident_on_; }

private:
    // Shuffled train/val split of these image indices.
    void Split(std::vector<size_t> indices);
    // The batch's images as float rows [n, H * W * 3] in [0, 1] on the device.
    Tensor GatherRows(const size_t* indices, size_t n);

    std::shared_ptr<const PreparedImageSet> images_;
    float train_split_ = 0.8f;
    uint32_t seed_ = 42;

    int batch_size_;
    bool shuffle_;
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

    // The decoded [H, W, 3] image and the sample the model receives.
    image::ImageShape decoded_shape_;
    image::ImageShape sample_shape_;

    // The whole set on the device when it fits (no ArrayFire header here:
    // it brings windows.h, whose LoadImage macro renames ImageUtils::LoadImage).
    struct Resident;
    std::unique_ptr<Resident> resident_;
    bool resident_on_ = false;
    bool resident_tried_ = false;

    std::mt19937 rng_;
};

} // namespace cyxwiz
