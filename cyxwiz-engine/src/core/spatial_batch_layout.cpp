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
  if (rows.GetDataType() != DataType::Float32 || rows.Shape().size() != 2)
    throw std::invalid_argument(
        "Spatial batch ingress requires Float32 [N,H*W*C] rows");
  if (rows.Shape()[1] != features)
    throw std::invalid_argument(
        "Spatial row feature count does not match [H,W,C]");
  const auto shape = SpatialRuntimeShape(sample_shape, rows.Shape()[0]);
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

} // namespace cyxwiz
