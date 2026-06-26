#pragma once

#include "executable_model.h"
#include "metric_learning_batch.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct PairEmbeddings {
    Tensor embedding_a;
    Tensor embedding_b;
};

struct TripletEmbeddings {
    Tensor anchor;
    Tensor positive;
    Tensor negative;
};

struct PairBranchGradients {
    Tensor input_a;
    Tensor input_b;
};

struct TripletBranchGradients {
    Tensor anchor;
    Tensor positive;
    Tensor negative;
};

class SharedEncoderRuntime {
public:
    explicit SharedEncoderRuntime(std::unique_ptr<IExecutableModel> encoder)
        : encoder_(std::move(encoder)) {
        if (!encoder_) {
            throw std::invalid_argument(
                "SharedEncoderRuntime requires an encoder");
        }
    }

    PairEmbeddings ForwardPair(const PairBatch& batch) {
        if (!batch.IsValid()) {
            throw std::invalid_argument(
                "SharedEncoderRuntime requires a valid PairBatch");
        }
        ClearAccumulatedGradients();
        CacheSequentialPairInputs(batch);
        return {
            encoder_->Forward(batch.input_a),
            encoder_->Forward(batch.input_b),
        };
    }

    TripletEmbeddings ForwardTriplet(const TripletBatch& batch) {
        if (!batch.IsValid()) {
            throw std::invalid_argument(
                "SharedEncoderRuntime requires a valid TripletBatch");
        }
        ClearAccumulatedGradients();
        CacheSequentialTripletInputs(batch);
        return {
            encoder_->Forward(batch.anchor),
            encoder_->Forward(batch.positive),
            encoder_->Forward(batch.negative),
        };
    }

    PairBranchGradients BackwardPair(const Tensor& grad_a,
                                     const Tensor& grad_b) {
        if (encoder_->AsSequentialModel() != nullptr) {
            return BackwardSequentialPair(grad_a, grad_b);
        }
        return {
            encoder_->Backward(grad_a),
            encoder_->Backward(grad_b),
        };
    }

    TripletBranchGradients BackwardTriplet(const Tensor& anchor_grad,
                                           const Tensor& positive_grad,
                                           const Tensor& negative_grad) {
        if (encoder_->AsSequentialModel() != nullptr) {
            return BackwardSequentialTriplet(
                anchor_grad, positive_grad, negative_grad);
        }
        return {
            encoder_->Backward(anchor_grad),
            encoder_->Backward(positive_grad),
            encoder_->Backward(negative_grad),
        };
    }

    void SetTraining(bool training) { encoder_->SetTraining(training); }

    std::map<std::string, Tensor> GetParameters() {
        return encoder_->GetParameters();
    }

    void SetParameters(const std::map<std::string, Tensor>& params) {
        encoder_->SetParameters(params);
    }

    std::map<std::string, Tensor> GetGradients() {
        if (has_accumulated_gradients_) {
            return accumulated_gradients_;
        }
        return encoder_->GetGradients();
    }

    void UpdateParameters(Optimizer* optimizer) {
        if (has_accumulated_gradients_) {
            if (optimizer == nullptr) {
                throw std::invalid_argument(
                    "SharedEncoderRuntime accumulated gradient update requires an optimizer");
            }
            auto params = encoder_->GetParameters();
            optimizer->Step(params, accumulated_gradients_);
            encoder_->SetParameters(params);
            ClearAccumulatedGradients();
            return;
        }
        encoder_->UpdateParameters(optimizer);
    }

    IExecutableModel* Encoder() { return encoder_.get(); }
    const IExecutableModel* Encoder() const { return encoder_.get(); }

private:
    static bool IsReplayUnsafeTrainingModule(const Module& module) {
        if (!module.IsTraining()) {
            return false;
        }
        const std::string name = module.GetName();
        return name.rfind("Dropout", 0) == 0 ||
               name.rfind("BatchNorm", 0) == 0;
    }

    void EnsureSequentialReplaySupported(const char* mode) const {
        const auto* model = encoder_->AsSequentialModel();
        if (model == nullptr) {
            return;
        }
        for (size_t i = 0; i < model->Size(); ++i) {
            const Module* module = model->GetModule(i);
            if (module != nullptr && IsReplayUnsafeTrainingModule(*module)) {
                throw std::logic_error(
                    std::string("SharedEncoderRuntime cannot replay ") +
                    mode +
                    " branches through training-mode stateful module " +
                    module->GetName() +
                    "; activation snapshots are required");
            }
        }
    }

    void CacheSequentialPairInputs(const PairBatch& batch) {
        if (encoder_->AsSequentialModel() == nullptr) {
            has_pair_cache_ = false;
            return;
        }
        EnsureSequentialReplaySupported("pair");
        pair_input_a_ = batch.input_a.Clone();
        pair_input_b_ = batch.input_b.Clone();
        has_pair_cache_ = true;
        has_triplet_cache_ = false;
    }

    void CacheSequentialTripletInputs(const TripletBatch& batch) {
        if (encoder_->AsSequentialModel() == nullptr) {
            has_triplet_cache_ = false;
            return;
        }
        EnsureSequentialReplaySupported("triplet");
        triplet_anchor_ = batch.anchor.Clone();
        triplet_positive_ = batch.positive.Clone();
        triplet_negative_ = batch.negative.Clone();
        has_triplet_cache_ = true;
        has_pair_cache_ = false;
    }

    static Tensor SumGradientTensor(const Tensor& left, const Tensor& right) {
        if (left.Shape() != right.Shape() ||
            left.GetDataType() != right.GetDataType()) {
            throw std::logic_error(
                "SharedEncoderRuntime cannot accumulate mismatched gradients");
        }
        if (left.GetDataType() != DataType::Float32) {
            throw std::logic_error(
                "SharedEncoderRuntime accumulated gradients require Float32 tensors");
        }

        std::vector<float> values(left.NumElements(), 0.0f);
        const float* left_data = left.Data<float>();
        const float* right_data = right.Data<float>();
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = left_data[i] + right_data[i];
        }
        return Tensor(left.Shape(), values.data(), DataType::Float32);
    }

    static std::map<std::string, Tensor> SumGradientMaps(
        const std::vector<std::map<std::string, Tensor>>& gradient_maps) {
        std::map<std::string, Tensor> summed;
        for (const auto& gradients : gradient_maps) {
            for (const auto& [name, gradient] : gradients) {
                auto it = summed.find(name);
                if (it == summed.end()) {
                    summed[name] = gradient.Clone();
                } else {
                    it->second = SumGradientTensor(it->second, gradient);
                }
            }
        }
        return summed;
    }

    PairBranchGradients BackwardSequentialPair(const Tensor& grad_a,
                                               const Tensor& grad_b) {
        EnsureSequentialReplaySupported("pair");
        if (!has_pair_cache_) {
            throw std::logic_error(
                "SharedEncoderRuntime pair backward requires ForwardPair first");
        }

        (void)encoder_->Forward(pair_input_a_);
        Tensor input_grad_a = encoder_->Backward(grad_a);
        auto grad_map_a = encoder_->GetGradients();

        (void)encoder_->Forward(pair_input_b_);
        Tensor input_grad_b = encoder_->Backward(grad_b);
        auto grad_map_b = encoder_->GetGradients();

        accumulated_gradients_ = SumGradientMaps({grad_map_a, grad_map_b});
        has_accumulated_gradients_ = true;
        return {input_grad_a, input_grad_b};
    }

    TripletBranchGradients BackwardSequentialTriplet(
        const Tensor& anchor_grad,
        const Tensor& positive_grad,
        const Tensor& negative_grad) {
        EnsureSequentialReplaySupported("triplet");
        if (!has_triplet_cache_) {
            throw std::logic_error(
                "SharedEncoderRuntime triplet backward requires ForwardTriplet first");
        }

        (void)encoder_->Forward(triplet_anchor_);
        Tensor input_grad_anchor = encoder_->Backward(anchor_grad);
        auto grad_map_anchor = encoder_->GetGradients();

        (void)encoder_->Forward(triplet_positive_);
        Tensor input_grad_positive = encoder_->Backward(positive_grad);
        auto grad_map_positive = encoder_->GetGradients();

        (void)encoder_->Forward(triplet_negative_);
        Tensor input_grad_negative = encoder_->Backward(negative_grad);
        auto grad_map_negative = encoder_->GetGradients();

        accumulated_gradients_ = SumGradientMaps(
            {grad_map_anchor, grad_map_positive, grad_map_negative});
        has_accumulated_gradients_ = true;
        return {input_grad_anchor, input_grad_positive, input_grad_negative};
    }

    void ClearAccumulatedGradients() {
        accumulated_gradients_.clear();
        has_accumulated_gradients_ = false;
    }

    std::unique_ptr<IExecutableModel> encoder_;
    Tensor pair_input_a_;
    Tensor pair_input_b_;
    Tensor triplet_anchor_;
    Tensor triplet_positive_;
    Tensor triplet_negative_;
    bool has_pair_cache_ = false;
    bool has_triplet_cache_ = false;
    bool has_accumulated_gradients_ = false;
    std::map<std::string, Tensor> accumulated_gradients_;
};

}  // namespace cyxwiz
