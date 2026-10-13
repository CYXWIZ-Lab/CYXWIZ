#pragma once

#include <cyxwiz/sequential.h>

namespace cyxwiz {

// Engine adapter for a graph Flatten at the spatial-to-row boundary. Unlike
// FlattenModule/PermuteModule, spatial input has its batch axis last. Reuses
// the shared row conversion and retains metadata only for backward validation.
class SpatialFlattenModule final : public Module {
public:
  explicit SpatialFlattenModule(std::vector<size_t> sample_shape);
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SpatialFlatten"; }

private:
  std::vector<size_t> sample_shape_;
  size_t features_ = 0;
  size_t batch_size_ = 0;
};

// The 1-D (Conv1D) section's layout boundaries (TOFIX140). Conv1D runs on
// [L,C,N] (batch last); torch's Conv1d runs on [N,C,L].

// Opens the section on the model's input rows: [N, C*L] (or [N,C,L]) in
// torch's channel-major order -> [L,C,N] (torch x.view(N, C, L)).
class SequenceRowsModule final : public Module {
public:
  SequenceRowsModule(size_t length, size_t channels);
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SequenceRows"; }

private:
  size_t length_ = 0;
  size_t channels_ = 0;
  std::vector<size_t> input_shape_;
};

// Opens the section after an Embedding: [N,L,E] -> [L,E,N]
// (torch x.transpose(1, 2) before Conv1d).
class SequenceChannelsLastModule final : public Module {
public:
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SequenceChannelsLast"; }
};

// Ends the section with Flatten: [L,C,N] -> [N, C*L] rows in torch's
// channel-major order (torch.flatten of [N,C,L]).
class SequenceFlattenModule final : public Module {
public:
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SequenceFlatten"; }

private:
  std::vector<size_t> input_shape_;
};

// Ends the section with Global Max Pool: [L,C,N] -> [N,C], each channel's
// maximum over L (torch adaptive_max_pool1d(x, 1).flatten(1)); the backend's
// 2-D global max pool on the [L,1,C,N] view.
class SequenceGlobalMaxPoolModule final : public Module {
public:
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SequenceGlobalMaxPool"; }

private:
  GlobalMaxPool2DModule pool_;
  std::vector<size_t> input_shape_;
};

// Ends the section with Global Avg Pool: [L,C,N] -> [N,C], each channel's
// mean over L (torch adaptive_avg_pool1d(x, 1).flatten(1)).
class SequenceGlobalAvgPoolModule final : public Module {
public:
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &gradient) override;
  std::string GetName() const override { return "SequenceGlobalAvgPool"; }

private:
  std::vector<size_t> input_shape_;
};

} // namespace cyxwiz
