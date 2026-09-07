#pragma once
#include "algorithms/arrayfire_backend_utils.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>
#include <string>
#include <vector>
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::test::convolution {
inline void CheckValues(const cyxwiz::Tensor &actual,
                        const std::vector<size_t> &expected_shape,
                        const std::vector<float> &expected_values) {
  REQUIRE(actual.Shape() == expected_shape);
  REQUIRE(actual.GetDataType() == cyxwiz::DataType::Float32);
  REQUIRE(actual.NumElements() == expected_values.size());
  const float *data = actual.ReadData<float>();
  for (size_t index = 0; index < expected_values.size(); ++index) {
    CHECK(data[index] == Catch::Approx(expected_values[index]));
  }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
inline size_t conv_host_sync_count = 0;
inline uint64_t conv_host_sync_bytes = 0;
inline size_t conv_fallback_count = 0;
inline bool saw_conv_cpu_path = false;
inline cyxwiz::ArrayFireNativeCpuFallbackEvent last_conv_fallback;

inline void CountConvHostSync(const cyxwiz::ArrayFireHostSyncEvent &event) {
  ++conv_host_sync_count;
  conv_host_sync_bytes += event.bytes;
  saw_conv_cpu_path |= event.attribution_category == "layer_cpu_path";
}

inline void
CountConvFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent &event) {
  ++conv_fallback_count;
  last_conv_fallback = event;
}

inline void ResetConvObservations() {
  conv_host_sync_count = 0;
  conv_host_sync_bytes = 0;
  conv_fallback_count = 0;
  saw_conv_cpu_path = false;
  last_conv_fallback = {};
}

inline af::dim4 SemanticDims(const std::vector<size_t> &shape) {
  REQUIRE_FALSE(shape.empty());
  REQUIRE(shape.size() <= 4);
  af::dim4 dims(1, 1, 1, 1);
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    dims[static_cast<unsigned>(axis)] = static_cast<dim_t>(shape[axis]);
  }
  return dims;
}

inline cyxwiz::Tensor DeviceOnlyTensor(const std::vector<size_t> &shape,
                                       const std::vector<float> &values) {
  const cyxwiz::Tensor host(shape, values.data(), cyxwiz::DataType::Float32);
  af::array semantic = host.GetSemanticArray();
  semantic.eval();
  return cyxwiz::Tensor::FromSemanticArray(semantic, shape);
}

inline cyxwiz::Tensor DeviceOnlyOnes(const std::vector<size_t> &shape) {
  af::array values = af::constant(1.0f, SemanticDims(shape), af::dtype::f32);
  values.eval();
  return cyxwiz::Tensor::FromSemanticArray(values, shape);
}

inline const char *ExpectedBackendName(cyxwiz::DeviceType type) {
  switch (type) {
  case cyxwiz::DeviceType::CPU:
    return "cpu";
  case cyxwiz::DeviceType::CUDA:
    return "cuda";
  case cyxwiz::DeviceType::OPENCL:
    return "opencl";
  case cyxwiz::DeviceType::ONEAPI:
    return "oneapi";
  default:
    return "unsupported";
  }
}

inline void SetEnvVar(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline void ClearEnvVar(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *previous = std::getenv(name);
    if (previous != nullptr) {
      had_previous_ = true;
      previous_ = previous;
    }
    SetEnvVar(name_, value);
  }

  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

  ~ScopedEnvVar() {
    if (had_previous_) {
      SetEnvVar(name_, previous_.c_str());
    } else {
      ClearEnvVar(name_);
    }
  }

private:
  const char *name_;
  bool had_previous_ = false;
  std::string previous_;
};

#endif

} // namespace cyxwiz::test::convolution
