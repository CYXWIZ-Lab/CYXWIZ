#include "image_dataset_batcher.h"
#include "device_data_budget.h"
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace cyxwiz {

namespace {

// The images the split is made over: every one except `excluded` (a
// Quality node's rejects and the images the index found unreadable, as the
// compile counts), in set order. Files that fail to decode here are taken
// out of their part after the split (Split), so the parts stay the counted ones.
std::vector<size_t> UsableImages(const PreparedImageSet& images, const std::vector<std::string>& excluded) {
    const std::unordered_set<std::string> left_out(excluded.begin(), excluded.end());
    std::vector<size_t> kept;
    kept.reserve(images.Count());
    for (size_t i = 0; i < images.Count(); ++i)
        if (!left_out.count(images.Files()[i])) kept.push_back(i);
    return kept;
}

}  // namespace

// uint8, one column of H * W * 3 bytes per image (the file's order, no transpose).
struct ImageDatasetBatcher::Resident {
    af::array images;
};

ImageDatasetBatcher::~ImageDatasetBatcher() = default;

ImageDatasetBatcher::ImageDatasetBatcher(
    std::shared_ptr<const PreparedImageSet> images,
    int batch_size,
    const SampleSplitSettings& split,
    bool shuffle,
    uint32_t shuffle_seed)
    : images_(std::move(images)), split_(split), batch_size_(batch_size),
      shuffle_(shuffle), augmentation_rng_(shuffle_seed ^ 0x9E3779B9u), rng_(shuffle_seed)
{
    if (!images_ || images_->Count() == 0) {
        spdlog::error("ImageDatasetBatcher: no prepared images");
        return;
    }
    decoded_shape_ = {images_->Height(), images_->Width(), images_->Channels()};
    sample_shape_ = decoded_shape_;
    num_classes_ = images_->ClassNames().size();
    if (!images_->Unreadable().empty())
        spdlog::warn("ImageDatasetBatcher: {} of {} images cannot be read and are left out (first: {})",
                     images_->Unreadable().size(), images_->Count(), images_->Unreadable().front());
    // Until ExcludeFiles hands the compile's list, the files that do not
    // decode are left out before the split (no index to follow).
    Split(UsableImages(*images_, images_->Unreadable()));
    Reset();
    spdlog::info("ImageDatasetBatcher: {} train / {} validation / {} test images at {}x{}, {} classes, "
                 "batch_size={}", train_indices_.size(), val_indices_.size(), test_indices_.size(), images_->Width(),
                 images_->Height(), num_classes_, batch_size_);
}

void ImageDatasetBatcher::Split(const std::vector<size_t>& usable) {
    auto parts = SplitSamples(usable, images_->Labels(), split_);
    // A file the index read but that does not decode (a truncated JPEG):
    // out of its own part, so the other images stay where the split put them.
    const std::unordered_set<std::string> broken(images_->Unreadable().begin(), images_->Unreadable().end());
    size_t dropped = 0;
    for (auto* part : {&parts.train, &parts.val, &parts.test}) {
        const auto end = std::remove_if(part->begin(), part->end(),
                                        [&](size_t i) { return broken.count(images_->Files()[i]) > 0; });
        dropped += static_cast<size_t>(part->end() - end);
        part->erase(end, part->end());
    }
    if (dropped > 0)
        spdlog::warn("ImageDatasetBatcher: {} image(s) the index read cannot be decoded; left out of their part", dropped);
    train_indices_ = std::move(parts.train);
    val_indices_ = std::move(parts.val);
    test_indices_ = std::move(parts.test);
}

const std::vector<std::string>& ImageDatasetBatcher::Files() const {
    static const std::vector<std::string> none;
    return images_ ? images_->Files() : none;
}

const std::vector<size_t>& ImageDatasetBatcher::Part(BatcherPhase phase) const {
    return phase == BatcherPhase::Val ? val_indices_ : phase == BatcherPhase::Test ? test_indices_ : train_indices_;
}

void ImageDatasetBatcher::ExcludeFiles(const std::vector<std::string>& files) {
    if (!images_ || files.empty()) return;
    const auto kept = UsableImages(*images_, files);
    const size_t left_out = images_->Count() - kept.size();
    Split(kept);
    Reset();
    spdlog::info("ImageDatasetBatcher: {} of {} images left out before the split (a Quality node's, or unreadable in "
                 "the image index); split: {} train / {} "
                 "validation / {} test", left_out, images_->Count(), train_indices_.size(), val_indices_.size(),
                 test_indices_.size());
}

Tensor ImageDatasetBatcher::GatherRows(const size_t* indices, size_t n) {
    const size_t row_bytes = images_->RowBytes();
    if (!resident_tried_) {
        resident_tried_ = true;
        // ArrayFire's CPU backend reads the mapped file directly; a GPU keeps
        // the whole set when it fits the device budget.
        const uint64_t bytes = static_cast<uint64_t>(images_->Count()) * row_bytes;
        const uint64_t budget = ResidentDataBudgetBytes();
        if (af::getActiveBackend() != AF_BACKEND_CPU && bytes <= budget &&
            images_->Count() <= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            try {
                auto resident = std::make_unique<Resident>();
                resident->images = af::array(static_cast<dim_t>(row_bytes),
                                             static_cast<dim_t>(images_->Count()), images_->Row(0));
                resident->images.eval();
                resident_ = std::move(resident);
                resident_on_ = true;
                spdlog::info("ImageDatasetBatcher: {} images ({:.1f} MB) on the device for the run",
                             images_->Count(), bytes / 1048576.0);
            } catch (const std::exception& e) {
                resident_.reset();
                spdlog::warn("ImageDatasetBatcher: the images could not stay on the device ({}); each batch is "
                             "copied and uploaded", e.what());
            }
        } else if (af::getActiveBackend() != AF_BACKEND_CPU) {
            spdlog::info("ImageDatasetBatcher: {} images ({:.1f} MB) are over the device budget ({:.1f} MB); each "
                         "batch is copied and uploaded", images_->Count(), bytes / 1048576.0, budget / 1048576.0);
        }
    }

    af::array rows;
    if (resident_on_) {
        std::vector<int32_t> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int32_t>(indices[i]);
        rows = af::lookup(resident_->images, af::array(static_cast<dim_t>(n), ids.data()), 1);
    } else {
        std::vector<uint8_t> host(n * row_bytes);
        for (size_t i = 0; i < n; ++i) std::memcpy(host.data() + i * row_bytes, images_->Row(indices[i]), row_bytes);
        rows = af::array(static_cast<dim_t>(row_bytes), static_cast<dim_t>(n), host.data());
    }
    // One column per image is the row-major [n, H * W * 3] order.
    return Tensor::FromRowMajorFlatArray(af::flat(rows.as(f32) / 255.0f), {n, row_bytes});
}

Batch ImageDatasetBatcher::GetNextBatch() {
    Batch batch;
    if (!images_ || IsEpochComplete()) return batch;

    size_t actual_size = std::min(static_cast<size_t>(batch_size_),
                                   epoch_order_.size() - current_idx_);
    if (actual_size == 0) return batch;

    const size_t* indices = epoch_order_.data() + current_idx_;
    const bool onehot = !scalar_label_mode_ && do_onehot_ && num_classes_ > 0;
    std::vector<float> batch_labels(onehot ? actual_size * num_classes_ : actual_size, 0.0f);
    for (size_t i = 0; i < actual_size; ++i) {
        const int label = images_->Labels()[indices[i]];
        if (onehot) {
            if (label >= 0 && static_cast<size_t>(label) < num_classes_)
                batch_labels[i * num_classes_ + static_cast<size_t>(label)] = 1.0f;
        } else {
            batch_labels[i] = static_cast<float>(label);
        }
    }
    if (onehot) {
        batch.labels = Tensor({actual_size, num_classes_}, batch_labels.data(), DataType::Float32);
    } else if (scalar_label_mode_) {
        batch.labels = Tensor({actual_size, 1}, batch_labels.data(), DataType::Float32);
    } else {
        batch.labels = Tensor({actual_size}, batch_labels.data(), DataType::Float32);
    }

    // The transforms, the batch mix (one-hot labels mixed alongside) and
    // Normalize on the device; the batch stays there for the model.
    Tensor rows = GatherRows(indices, actual_size);
    if (!augmentation_.Empty()) {
        rows = augmentation_.Apply(rows, decoded_shape_, current_phase_ == BatcherPhase::Train,
                                   augmentation_rng_, onehot ? &batch.labels : nullptr);
    }
    batch.data = flatten_
        ? rows
        : rows.Reshape({actual_size, sample_shape_.height, sample_shape_.width, sample_shape_.channels});

    // Image numbers in the prepared set (= rows of the dataset's index table):
    // test results name the files with them.
    batch.row_ids.assign(indices, indices + actual_size);
    batch.size = actual_size;
    current_idx_ += actual_size;

    return batch;
}

void ImageDatasetBatcher::Reset() {
    // The current phase's part; only train batches are shuffled.
    epoch_order_ = Part(current_phase_);
    if (current_phase_ == BatcherPhase::Train && shuffle_) std::shuffle(epoch_order_.begin(), epoch_order_.end(), rng_);
    current_idx_ = 0;
}

void ImageDatasetBatcher::SetPhase(BatcherPhase phase) {
    current_phase_ = phase;
    // Caller should Reset() afterwards; training_executor's
    // RunValidationArrow does that via batcher.Reset().
}

bool ImageDatasetBatcher::IsEpochComplete() const {
    if (current_idx_ >= epoch_order_.size()) return true;
    return drop_last_ && batch_size_ > 0 &&
           current_phase_ == BatcherPhase::Train &&
           epoch_order_.size() - current_idx_ < static_cast<size_t>(batch_size_);
}

size_t ImageDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    if (drop_last_ && current_phase_ == BatcherPhase::Train) {
        return epoch_order_.size() / static_cast<size_t>(batch_size_);
    }
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t ImageDatasetBatcher::GetNumSamples() const {
    return Part(current_phase_).size();
}

void ImageDatasetBatcher::SetNormalization(float mean, float std_dev) {
    augmentation_.normalize = true;
    augmentation_.mean = mean;
    augmentation_.std_dev = (std_dev > 0.0f) ? std_dev : 1.0f;
}

void ImageDatasetBatcher::SetChannelNormalization(const std::vector<float>& mean, const std::vector<float>& std_dev) {
    augmentation_.normalize = true;
    augmentation_.channel_mean = mean;
    augmentation_.channel_std = std_dev;
}

void ImageDatasetBatcher::SetImageTransforms(const image::ImageAugmentation& compiled) {
    augmentation_.ops = compiled.ops;
    augmentation_.mix = compiled.mix;
    augmentation_.mix_alpha = compiled.mix_alpha;
    augmentation_.mix_probability = compiled.mix_probability;
    sample_shape_ = augmentation_.ShapeAfter(decoded_shape_);
}

void ImageDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    do_onehot_ = true;
}

void ImageDatasetBatcher::SetFlatten(bool flatten) {
    flatten_ = flatten;
}

} // namespace cyxwiz
