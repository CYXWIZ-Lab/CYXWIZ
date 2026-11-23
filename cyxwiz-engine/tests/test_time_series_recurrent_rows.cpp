// A recurrent layer first on time-series window rows (TOFIX223 0.20, audit
// SEQ-01). A window row holds input_width values per feature, feature after
// feature (torch [N, C, L] flattened); LSTM / GRU / RNN read [N, L, C] time
// steps. ModelBuilder turns the rows into steps as torch's
// x.view(N, C, L).transpose(1, 2) does. Before, the rows reached the layer as
// [N, L*C] and it threw on the first batch.
//
// Oracle: the built model on the rows against the same recurrent layer (its
// own PyTorch parity is tested elsewhere) on the [N, L, C] tensor built here
// by hand, with the same weights; then a backward pass gives the rows'
// gradient shape.
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "core/model_builder.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/sequential.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

constexpr size_t kBatch = 2;
constexpr size_t kSteps = 4;     // input_width
constexpr size_t kFeatures = 2;  // the target and one more column
constexpr size_t kHidden = 3;

std::unique_ptr<cyxwiz::Module> Standalone(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::LSTM: return std::make_unique<cyxwiz::LSTMModule>(kFeatures, kHidden, 1, false, false);
        case gui::NodeType::GRU: return std::make_unique<cyxwiz::GRUModule>(kFeatures, kHidden, 1, false, false);
        default: return std::make_unique<cyxwiz::RNNModule>(kFeatures, kHidden, 1, false, "tanh", false);
    }
}

void CheckRecurrentFirst(gui::NodeType type, const std::string& name) {
    cyxwiz::TrainingConfiguration config;
    config.is_time_series = true;
    config.input_size = kSteps * kFeatures;
    config.input_shape = {kSteps * kFeatures};
    config.sequence_input_shape = {kSteps, kFeatures};
    config.output_size = 1;
    config.loss_type = gui::NodeType::MSELoss;
    config.optimizer_type = gui::NodeType::SGD;
    cyxwiz::CompiledLayer recurrent;
    recurrent.type = type;
    recurrent.parameters = {{"hidden_size", std::to_string(kHidden)}, {"num_layers", "1"},
                            {"bidirectional", "false"}, {"return_sequences", "false"}};
    config.layers.push_back(recurrent);
    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 1;
    config.layers.push_back(dense);

    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": the model builds: " + built.error_message);
    if (!built.ok() || !built.model->AsSequentialModel()) return;
    auto& model = *built.model->AsSequentialModel();

    // Rows: feature c's window at [c * kSteps, (c + 1) * kSteps).
    std::vector<float> rows(kBatch * kSteps * kFeatures);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = std::sin(0.37f * static_cast<float>(i)) * 0.8f;
    const cyxwiz::Tensor input({kBatch, kSteps * kFeatures}, rows.data(), cyxwiz::DataType::Float32);
    // The same values as time steps: steps[b][l][c] = rows[b][c * kSteps + l].
    std::vector<float> steps(rows.size());
    for (size_t b = 0; b < kBatch; ++b)
        for (size_t l = 0; l < kSteps; ++l)
            for (size_t c = 0; c < kFeatures; ++c)
                steps[(b * kSteps + l) * kFeatures + c] = rows[b * kSteps * kFeatures + c * kSteps + l];
    const cyxwiz::Tensor step_input({kBatch, kSteps, kFeatures}, steps.data(), cyxwiz::DataType::Float32);

    cyxwiz::Module* layer = nullptr;
    for (size_t i = 0; i < model.Size(); ++i)
        if (model.GetModule(i)->HasParameters()) {
            layer = model.GetModule(i);
            break;
        }
    Check(layer != nullptr, name + ": the model has the recurrent module first among its weighted modules");
    if (!layer) return;
    auto reference = Standalone(type);
    reference->SetParameters(layer->GetParameters());

    cyxwiz::Tensor output;
    try {
        model.Forward(input);
        output = layer->Forward(model.GetModule(1)->Forward(model.GetModule(0)->Forward(input)));
    } catch (const std::exception& e) {
        Check(false, name + ": the rows run through the model: " + e.what());
        return;
    }
    const cyxwiz::Tensor expected = reference->Forward(step_input);
    Check(output.Shape() == expected.Shape(), name + ": the recurrent output has the reference's shape");
    if (output.Shape() == expected.Shape()) {
        const float* a = output.ReadData<float>();
        const float* e = expected.ReadData<float>();
        size_t wrong = 0;
        for (size_t i = 0; i < expected.NumElements(); ++i)
            if (std::fabs(a[i] - e[i]) > 1e-5f) ++wrong;
        Check(wrong == 0, name + ": " + std::to_string(wrong) + " values differ from the layer on [N, L, C] steps");
    }

    try {
        const cyxwiz::Tensor prediction = model.Forward(input);
        std::vector<float> ones(prediction.NumElements(), 1.0f);
        const cyxwiz::Tensor grad = model.Backward(cyxwiz::Tensor(prediction.Shape(), ones.data(),
                                                                  cyxwiz::DataType::Float32));
        Check(grad.Shape() == std::vector<size_t>({kBatch, kSteps * kFeatures}),
              name + ": the gradient comes back in the rows' shape");
    } catch (const std::exception& e) {
        Check(false, name + ": backward through the rows: " + e.what());
    }
    std::cout << "  " << name << " checked\n";
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) {
        std::cerr << "the requested device could not be activated\n";
        return 1;
    }
    CheckRecurrentFirst(gui::NodeType::LSTM, "LSTM");
    CheckRecurrentFirst(gui::NodeType::GRU, "GRU");
    CheckRecurrentFirst(gui::NodeType::RNN, "RNN");
    if (failures == 0) std::cout << "test_time_series_recurrent_rows: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
