#include "spatial_batch_layout.h"
#include "spatial_head_module.h"

#include <stdexcept>
#include <utility>
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
Tensor SpatialBatchFromRows(const Tensor &rows,
                            const std::vector<size_t> &sample_shape) {
  const size_t features = SpatialSampleElements(sample_shape);
  // The image batcher emits [N,H,W,C] (row-major, the same bytes as N rows of
  // HWC); tabular batchers emit [N,H*W*C] rows. Both enter here.
  const auto &in = rows.Shape();
  const bool image_batch = in.size() == 4 && in[1] == sample_shape[0] &&
                           in[2] == sample_shape[1] && in[3] == sample_shape[2];
  if (rows.GetDataType() != DataType::Float32 || (in.size() != 2 && !image_batch))
    throw std::invalid_argument(
        "Spatial batch ingress requires Float32 [N,H*W*C] rows or [N,H,W,C] images");
  if (!image_batch && in[1] != features)
    throw std::invalid_argument(
        "Spatial row feature count does not match [H,W,C]");
  const auto shape = SpatialRuntimeShape(sample_shape, in[0]);
#ifdef CYXWIZ_HAS_ARRAYFIRE
  // Host-origin rows enter the selected device here; device-current rows reuse
  // the canonical semantic view. Avoid Reshape's host-copy compatibility path.
  (void)rows.GetSemanticArray();
#endif
  return rows.Reshape({shape[3], shape[0], shape[1], shape[2]})
      .Permute({1, 2, 3, 0});
}

Tensor SpatialBatchToRows(const Tensor &spatial) {
  if (spatial.GetDataType() != DataType::Float32 || spatial.Shape().size() != 4)
    throw std::invalid_argument(
        "Spatial batch egress requires Float32 [H,W,C,N]");
  const auto &shape = spatial.Shape();
  const std::vector<size_t> sample{shape[0], shape[1], shape[2]};
  const size_t features = SpatialSampleElements(sample);
  SpatialRuntimeShape(sample, shape[3]);
  return spatial.Permute({3, 0, 1, 2}).Reshape({shape[3], features});
}

SpatialFlattenModule::SpatialFlattenModule(std::vector<size_t> sample_shape)
    : sample_shape_(std::move(sample_shape)),
      features_(SpatialSampleElements(sample_shape_)) {}

Tensor SpatialFlattenModule::Forward(const Tensor &input) {
  batch_size_ = 0; // Failed forward invalidates the previous backward contract.
  const auto &shape = input.Shape();
  if (shape.size() != 4 || shape[0] != sample_shape_[0] ||
      shape[1] != sample_shape_[1] || shape[2] != sample_shape_[2])
    throw std::invalid_argument(
        "SpatialFlatten input does not match compiled [H,W,C]");
  Tensor output = SpatialBatchToRows(input);
  batch_size_ = shape[3];
  return output;
}

Tensor SpatialFlattenModule::Backward(const Tensor &gradient) {
  if (batch_size_ == 0)
    throw std::logic_error(
        "SpatialFlatten backward requires a successful forward");
  if (gradient.Shape() != std::vector<size_t>{batch_size_, features_})
    throw std::invalid_argument(
        "SpatialFlatten gradient must match the last [N,F] output");
  return SpatialBatchFromRows(gradient, sample_shape_);
}

SequenceRowsModule::SequenceRowsModule(size_t length, size_t channels)
    : length_(length), channels_(channels) {
  if (length_ == 0 || channels_ == 0)
    throw std::invalid_argument("SequenceRows needs a non-empty [L, C] sample");
}

Tensor SequenceRowsModule::Forward(const Tensor &input) {
  input_shape_.clear();
  const auto &in = input.Shape();
  const bool rows = in.size() == 2 && in[1] == channels_ * length_;
  const bool cube = in.size() == 3 && in[1] == channels_ && in[2] == length_;
  if (input.GetDataType() != DataType::Float32 || (!rows && !cube))
    throw std::invalid_argument(
        "Conv1D input rows must be Float32 [N, C*L] or [N, C, L] with C=" +
        std::to_string(channels_) + ", L=" + std::to_string(length_));
  Tensor output = input.Reshape({in[0], channels_, length_}).Permute({2, 1, 0});
  input_shape_ = in;
  return output;
}

Tensor SequenceRowsModule::Backward(const Tensor &gradient) {
  if (input_shape_.empty())
    throw std::logic_error("SequenceRows backward requires a successful forward");
  return gradient.Permute({2, 1, 0}).Reshape(input_shape_);
}

Tensor SequenceChannelsLastModule::Forward(const Tensor &input) {
  if (input.Shape().size() != 3)
    throw std::invalid_argument("Conv1D after an Embedding needs its [N, L, E] output");
  return input.Permute({1, 2, 0});
}

Tensor SequenceChannelsLastModule::Backward(const Tensor &gradient) {
  return gradient.Permute({2, 0, 1});
}

Tensor SequenceFlattenModule::Forward(const Tensor &input) {
  input_shape_.clear();
  const auto &in = input.Shape();
  if (in.size() != 3)
    throw std::invalid_argument("SequenceFlatten needs an [L, C, N] input");
  Tensor output = input.Permute({2, 1, 0}).Reshape({in[2], in[1] * in[0]});
  input_shape_ = in;
  return output;
}

Tensor SequenceFlattenModule::Backward(const Tensor &gradient) {
  if (input_shape_.empty())
    throw std::logic_error("SequenceFlatten backward requires a successful forward");
  return gradient.Reshape({input_shape_[2], input_shape_[1], input_shape_[0]})
      .Permute({2, 1, 0});
}

Tensor SequenceGlobalMaxPoolModule::Forward(const Tensor &input) {
  input_shape_.clear();
  const auto &in = input.Shape();
  if (in.size() != 3)
    throw std::invalid_argument("Global Max Pool over a sequence needs an [L, C, N] input");
  Tensor output = pool_.Forward(input.Reshape({in[0], 1, in[1], in[2]}));
  input_shape_ = in;
  return output;
}

Tensor SequenceGlobalMaxPoolModule::Backward(const Tensor &gradient) {
  if (input_shape_.empty())
    throw std::logic_error("SequenceGlobalMaxPool backward requires a successful forward");
  return pool_.Backward(gradient).Reshape(input_shape_);
}

Tensor SequenceGlobalAvgPoolModule::Forward(const Tensor &input) {
  input_shape_.clear();
  const auto &in = input.Shape();
  if (in.size() != 3)
    throw std::invalid_argument("Global Avg Pool over a sequence needs an [L, C, N] input");
  Tensor output = input.Mean(0).Reshape({in[1], in[2]}).Transpose();
  input_shape_ = in;
  return output;
}

Tensor SequenceGlobalAvgPoolModule::Backward(const Tensor &gradient) {
  if (input_shape_.empty())
    throw std::logic_error("SequenceGlobalAvgPool backward requires a successful forward");
  const size_t length = input_shape_[0];
  return gradient.Transpose()
             .Reshape({1, input_shape_[1], input_shape_[2]})
             .BroadcastTo(input_shape_) *
         (1.0f / static_cast<float>(length));
}

} // namespace cyxwiz
