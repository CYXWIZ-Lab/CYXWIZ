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

} // namespace cyxwiz
