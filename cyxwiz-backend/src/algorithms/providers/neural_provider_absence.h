#pragma once

// Why no provider serves a device family: the decision, as a pure function of
// the facts of a process, so every outcome can be tested on any machine. The
// registry gathers the facts (neural_provider_registry.cpp) and
// WhyNoNeuralProviderServes answers with this.

#include "cyxwiz/neural_provider.h"

namespace cyxwiz::provider_absence {

// A provider's module or kernel library, as the process found it.
enum class ModuleState {
    NotFound,  // not on the library path and not beside the backend
    WrongAbi,  // found, with another ABI number or without its entry points
    Present,   // found and usable
};

struct Facts {
    bool disabled = false;  // CYXWIZ_DISABLE_NEURAL_PROVIDERS, or a test
    // CYXWIZ_HAS_*_DNN_PROVIDER: what this build made.
    bool nvidia_built = false;
    bool opencl_built = false;
    bool oneapi_built = false;
    ModuleState nvidia_module = ModuleState::NotFound;
    ModuleState oneapi_kernels = ModuleState::NotFound;
};

inline BackendFallbackReason Reason(DeviceType platform, NeuralOp op, const Facts& facts) {
    using R = BackendFallbackReason;
    if (!NeuralProviderOfferedFor(platform, op)) return R::ProviderNotOffered;
    if (facts.disabled) return R::ProviderDisabled;
    switch (platform) {
        case DeviceType::CUDA:
            // A module that is found is used whatever this build made, so the
            // build flag decides only when none is found.
            if (facts.nvidia_module == ModuleState::WrongAbi) return R::ProviderAbiMismatch;
            if (facts.nvidia_module == ModuleState::Present) return R::NvidiaProviderUnavailable;
            return facts.nvidia_built ? R::ProviderNotInstalled : R::ProviderNotBuilt;
        case DeviceType::OPENCL:
            // Compiled into the backend: it cannot be missing at run time.
            return facts.opencl_built ? R::OpenclProviderUnavailable : R::ProviderNotBuilt;
        case DeviceType::ONEAPI:
            if (!facts.oneapi_built) return R::ProviderNotBuilt;
            if (facts.oneapi_kernels == ModuleState::WrongAbi) return R::ProviderAbiMismatch;
            if (facts.oneapi_kernels == ModuleState::Present) return R::OneapiProviderUnavailable;
            return R::ProviderNotInstalled;
        default:
            return R::ProviderNotOffered;
    }
}

}  // namespace cyxwiz::provider_absence
