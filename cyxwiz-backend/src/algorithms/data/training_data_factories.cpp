#include "cyxwiz/dataloader.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cyxwiz {
TrainingDataLoader LoadMNIST(const std::string& path,
                             size_t batch_size,
                             bool train,
                             bool shuffle) {
    Split split = train ? Split::Train : Split::Test;
    auto dataset = std::make_shared<MNISTDataset>(path, split, true, true);
    return TrainingDataLoader(dataset, batch_size, shuffle);
}

TrainingDataLoader CreateTrainingDataLoader(const Tensor& data,
                                            const Tensor& labels,
                                            size_t batch_size,
                                            bool shuffle) {
    // Create a wrapper dataset from tensors
    class TensorDataset : public DatasetBase {
    public:
        TensorDataset(const Tensor& data, const Tensor& labels)
            : data_(data.Clone()), labels_(labels.Clone()) {
            const auto& shape = data.Shape();
            num_samples_ = shape[0];
            sample_size_ = data.NumElements() / num_samples_;

            // Store shape without batch dimension
            for (size_t i = 1; i < shape.size(); i++) {
                sample_shape_.push_back(shape[i]);
            }
            if (sample_shape_.empty()) {
                sample_shape_.push_back(sample_size_);
            }
        }

        size_t Size() const override { return num_samples_; }

        std::pair<std::vector<float>, int> GetItem(size_t index) const override {
            const float* data_ptr = data_.Data<float>();
            const int32_t* label_ptr = labels_.Data<int32_t>();

            size_t offset = index * sample_size_;
            std::vector<float> sample(data_ptr + offset, data_ptr + offset + sample_size_);
            int label = static_cast<int>(label_ptr[index]);

            return {sample, label};
        }

        std::vector<size_t> GetShape() const override { return sample_shape_; }
        size_t NumClasses() const override { return 10; }  // Default

    private:
        Tensor data_;
        Tensor labels_;
        size_t num_samples_;
        size_t sample_size_;
        std::vector<size_t> sample_shape_;
    };

    auto dataset = std::make_shared<TensorDataset>(data, labels);
    return TrainingDataLoader(dataset, batch_size, shuffle);
}

}  // namespace cyxwiz

