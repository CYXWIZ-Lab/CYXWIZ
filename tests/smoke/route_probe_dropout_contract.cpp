#include "route_probe_dropout_contract.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <arrayfire.h>

#include <cyxwiz/layers/dropout.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>

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

} // namespace

void RunDropoutForwardBackwardContract(
    const std::string& operation,
    DropoutStageReporter report_stage) {
    report_stage(operation, "dropout_input_create_begin");
    Tensor input = Tensor::FromArrayRowMajor2D(
        af::constant(1.0f, 128, 64, f32));
    Tensor grad_output = Tensor::FromArrayRowMajor2D(
        af::constant(1.0f, 128, 64, f32));
    report_stage(operation, "dropout_input_create_complete");

    DropoutLayer dropout(0.5f);
    Tensor output;
    Tensor grad_input;
    host_sync_count = 0;
    fallback_count = 0;
    const size_t host_bytes_before = MemoryManager::GetAllocatedBytes();
    {
        const ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const ScopedArrayFireHostSyncObserver host_observer(&CountHostSync);
        const ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFallback);

        af::setSeed(3953);
        report_stage(operation, "dropout_forward_begin");
        output = dropout.Forward(input);
        report_stage(operation, "dropout_forward_complete");
        report_stage(operation, "dropout_backward_begin");
        grad_input = dropout.Backward(grad_output);
        report_stage(operation, "dropout_backward_complete");
        output.GetArrayRowMajor2D().eval();
        grad_input.GetArrayRowMajor2D().eval();
        af::sync();
        report_stage(operation, "dropout_device_sync_complete");
    }

    if (output.Shape() != input.Shape() ||
        grad_input.Shape() != input.Shape() ||
        output.GetDataType() != DataType::Float32 ||
        grad_input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dropout shape or dtype contract failed");
    }
    if (host_sync_count != 0) {
        throw std::runtime_error(
            "Dropout performed an undeclared host synchronization");
    }
    if (fallback_count != 0) {
        throw std::runtime_error("Dropout used native CPU fallback");
    }
    if (MemoryManager::GetAllocatedBytes() != host_bytes_before) {
        throw std::runtime_error("Dropout materialized host-owned tensor bytes");
    }

    report_stage(operation, "dropout_numerical_read_begin");
    const float* output_data = output.ReadData<float>();
    const float* grad_data = grad_input.ReadData<float>();
    size_t kept = 0;
    size_t dropped = 0;
    size_t invalid = 0;
    size_t backward_mismatches = 0;
    for (size_t index = 0; index < output.NumElements(); ++index) {
        if (output_data[index] == 0.0f) {
            ++dropped;
        } else if (output_data[index] == 2.0f) {
            ++kept;
        } else {
            ++invalid;
        }
        backward_mismatches +=
            grad_data[index] == output_data[index] ? 0 : 1;
    }
    if (kept == 0 || dropped == 0 || invalid != 0 ||
        backward_mismatches != 0) {
        throw std::runtime_error("Dropout numerical mask contract failed");
    }
    report_stage(operation, "dropout_numerical_read_complete");
    std::cout << "dropout_contract schema=1 pytorch_semantics=inverted_dropout"
              << " probability=0.5 dtype=float32 kept=" << kept
              << " dropped=" << dropped
              << " host_sync_count=" << host_sync_count
              << " native_fallback_count=" << fallback_count << std::endl;
}

} // namespace cyxwiz::route_probe
