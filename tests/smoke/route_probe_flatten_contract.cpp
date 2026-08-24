#include "route_probe_flatten_contract.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <arrayfire.h>

#include <cyxwiz/memory_manager.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cyxwiz::route_probe {
namespace {

size_t host_sync_count = 0;
size_t fallback_count = 0;

void CountHostSync(const ArrayFireHostSyncEvent&) {
    ++host_sync_count;
}

void CountFallback(const ArrayFireNativeCpuFallbackEvent&) {
    ++fallback_count;
}

Tensor DeviceRowMajor3D(const std::vector<float>& values) {
    af::array row_major_storage(4, 3, 2, values.data());
    return Tensor::FromArrayRowMajor3D(
        af::reorder(row_major_storage, 2, 1, 0));
}

Tensor DeviceRowMajor2D(const std::vector<float>& values) {
    af::array row_major_storage(12, 2, values.data());
    return Tensor::FromArrayRowMajor2D(af::transpose(row_major_storage));
}

void RequireShape(
    const Tensor& tensor,
    const std::vector<size_t>& expected,
    const char* value_name) {
    if (tensor.Shape() != expected) {
        throw std::runtime_error(
            std::string("Flatten ") + value_name + " shape mismatch");
    }
    if (tensor.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            std::string("Flatten ") + value_name + " dtype mismatch");
    }
}

void RequireValues(
    const Tensor& tensor,
    const std::vector<float>& expected,
    const char* value_name) {
    const float* actual = tensor.ReadData<float>();
    if (actual == nullptr || tensor.NumElements() != expected.size()) {
        throw std::runtime_error(
            std::string("Flatten ") + value_name + " payload mismatch");
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index]) || actual[index] != expected[index]) {
            throw std::runtime_error(
                std::string("Flatten ") + value_name +
                " numerical mismatch at index " + std::to_string(index));
        }
    }
}

} // namespace

void RunFlattenForwardBackwardContract(
    const std::string& operation,
    StageReporter report_stage) {
    std::vector<float> values(24);
    std::vector<float> gradients(24);
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index) - 7.5f;
        gradients[index] = static_cast<float>(index) * 0.25f - 1.25f;
    }

    report_stage(operation, "flatten_input_create_begin");
    Tensor input = DeviceRowMajor3D(values);
    Tensor grad_output = DeviceRowMajor2D(gradients);
    report_stage(operation, "flatten_input_create_complete");

    FlattenModule module(1);
    Tensor output;
    Tensor grad_input;
    host_sync_count = 0;
    fallback_count = 0;
    const size_t host_bytes_before = MemoryManager::GetAllocatedBytes();
    {
        const ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const ScopedArrayFireHostSyncObserver host_sync_observer(
            &CountHostSync);
        const ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFallback);

        report_stage(operation, "flatten_forward_begin");
        output = module.Forward(input);
        report_stage(operation, "flatten_forward_complete");
        report_stage(operation, "flatten_backward_begin");
        grad_input = module.Backward(grad_output);
        report_stage(operation, "flatten_backward_complete");

        RequireShape(output, {2, 12}, "forward");
        RequireShape(grad_input, {2, 3, 4}, "backward");
        output.GetArrayRowMajor2D().eval();
        grad_input.GetArrayRowMajor3D().eval();
        af::sync();
        report_stage(operation, "flatten_device_sync_complete");
    }

    if (host_sync_count != 0) {
        throw std::runtime_error(
            "Flatten performed an undeclared host synchronization");
    }
    if (fallback_count != 0) {
        throw std::runtime_error("Flatten used native CPU fallback");
    }
    if (MemoryManager::GetAllocatedBytes() != host_bytes_before) {
        throw std::runtime_error("Flatten materialized host-owned tensor bytes");
    }

    report_stage(operation, "flatten_numerical_read_begin");
    RequireValues(output, values, "forward");
    RequireValues(grad_input, gradients, "backward");
    report_stage(operation, "flatten_numerical_read_complete");
    std::cout << "flatten_contract schema=1 pytorch_case=rank3_start1"
              << " forward_shape=2x12 backward_shape=2x3x4"
              << " dtype=float32 host_sync_count=" << host_sync_count
              << " native_fallback_count=" << fallback_count << std::endl;
}

} // namespace cyxwiz::route_probe
