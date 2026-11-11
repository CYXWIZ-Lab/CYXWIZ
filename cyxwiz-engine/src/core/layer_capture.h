#pragma once

// What every layer held during one training step, for the Studio Debugger's
// layer views (TOFIX178 6.1): each layer's input and output, its parameters,
// the state it keeps beside them (BatchNorm's running statistics) and the
// gradients after backward. Tensors are kept where they live: a copy of a
// device Tensor shares its ArrayFire array, so capture copies nothing to the
// host; a view reads the slice it shows. A byte budget bounds what is held;
// tensors past it are skipped and counted. No capture object, no cost.

#include "image_channel_layout.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

class IExecutableModel;
class Loss;
struct BuiltModuleProvenance;

struct LayerCaptureOptions {
    bool activations = true;  // each layer's input and output
    bool parameters = true;   // weights and kept state
    bool gradients = true;    // the parameters' gradients after backward
    size_t byte_budget = size_t{256} << 20;
    // The model takes rank-4 tensors batch first [N, H, W, C]: its first
    // layer is not spatial (a Flatten-first image MLP; UsesSpatialSequentialInput
    // false). The capture keeps every image as [H, W, C, N], as a spatial
    // model takes it, so the debugger reads one layout (TOFIX207).
    bool images_batch_first = false;
    // What the model input's channels are (image models; empty otherwise):
    // the debugger draws an input that is neither grey nor RGB as a sheet.
    ImageChannelLayout input_channels;
};

// The debugger's image layout [H, W, C, N] from a batch-first [N, H, W, C]
// tensor, and back; any other rank comes back as it is (TOFIX207).
Tensor ImagesBatchLast(const Tensor& tensor);
Tensor ImagesBatchFirst(const Tensor& tensor);

enum class CapturedTensorRole { Input, Output, Parameter, State, Gradient };

const char* CapturedTensorRoleName(CapturedTensorRole role);

struct CapturedTensor {
    CapturedTensorRole role = CapturedTensorRole::Output;
    std::string name;  // "input", "output", or the module's parameter name
    Tensor tensor;
};

struct CapturedLayer {
    size_t module_index = 0;
    int node_id = -1;
    std::string node_name;
    std::vector<CapturedTensor> tensors;

    // The first tensor of this role (and name, when given), or nullptr.
    const CapturedTensor* Find(CapturedTensorRole role, std::string_view name = {}) const;
};

class LayerCapture {
public:
    // `provenance` names the layers (node id and name by module index).
    LayerCapture(LayerCaptureOptions options, const std::vector<BuiltModuleProvenance>& provenance);
    // A capture read back from the run store (layer_capture_store.h).
    LayerCapture(LayerCaptureOptions options, std::vector<CapturedLayer> layers, size_t skipped_tensors,
                 size_t skipped_bytes);

    const LayerCaptureOptions& Options() const { return options_; }

    // One layer's forward, from IExecutableModel's LayerForwardObserver.
    void RecordForward(size_t module_index, const Tensor& input, const Tensor& output);
    // Every layer's parameters, kept state and gradients; call after backward.
    void RecordModuleTensors(IExecutableModel& model);

    // In run order; layers that ran no forward follow, in module order.
    const std::vector<CapturedLayer>& Layers() const { return layers_; }
    const CapturedLayer* FindLayer(int node_id) const;

    // Bytes held counts every tensor in full, also where two share storage
    // (a layer's output is the next one's input), so it can overstate.
    size_t BytesHeld() const { return bytes_held_; }
    size_t SkippedTensors() const { return skipped_tensors_; }
    size_t SkippedBytes() const { return skipped_bytes_; }
    bool Truncated() const { return skipped_tensors_ > 0; }

private:
    CapturedLayer& LayerFor(size_t module_index);
    void Hold(CapturedLayer& layer, CapturedTensorRole role, std::string name, const Tensor& tensor);

    LayerCaptureOptions options_;
    struct LayerName {
        size_t module_index = 0;
        int node_id = -1;
        std::string name;
    };
    std::vector<LayerName> names_;
    std::vector<CapturedLayer> layers_;
    size_t bytes_held_ = 0;
    size_t skipped_tensors_ = 0;
    size_t skipped_bytes_ = 0;
};

// A training snapshot (TOFIX178 6.1b): one forward and backward of the
// model-ready `input` with the model in eval mode (BatchNorm on its running
// statistics, no dropout), so the running statistics and the random state
// training uses are left as they were; no optimizer step, and the next
// training backward replaces the gradients it leaves. The model is left in
// training mode. Throws what the model or the loss throws.
std::shared_ptr<LayerCapture> CaptureStep(IExecutableModel& model, Loss& loss, const Tensor& input,
                                          const Tensor& labels, LayerCaptureOptions options,
                                          const std::vector<BuiltModuleProvenance>& provenance);

} // namespace cyxwiz
