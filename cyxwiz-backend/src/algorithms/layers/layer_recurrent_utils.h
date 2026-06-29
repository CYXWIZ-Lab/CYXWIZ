#pragma once

#include "../arrayfire_backend_utils.h"
#include "cyxwiz/recurrent_cuda_placement.h"

#include <cstddef>
#include <string>

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
std::string BuildRecurrentFormalParameterOverflowFallbackMessage(const char* layer_name);
void DisableArrayFireCudaRecurrentAfterFailure(
    RecurrentLayerKind kind,
    const char* layer_name,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional,
    const char* error_message);
bool ShouldUseArrayFireRecurrentForward(
    RecurrentLayerKind kind,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional);
#endif

} // namespace cyxwiz
