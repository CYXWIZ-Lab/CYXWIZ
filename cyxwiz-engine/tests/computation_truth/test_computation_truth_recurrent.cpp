// LSTMModule / GRUModule / RNNModule and bidirectional LSTMLayer / GRULayer /
// RNNLayer against PyTorch nn.LSTM / nn.GRU / nn.RNN (TOFIX140).
//
// fixtures/recurrent_pytorch.json (generate_recurrent_fixtures.py): batch-first,
// 1-2 layers, uni- and bidirectional, full sequence or last step, hidden up to 64.
// Each case checks the output, dL/dx and every weight / bias gradient on the
// active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu),
// once through the neural providers where they serve the case and once with
// the providers off (the ArrayFire path).
#include "test_device_selection.h"

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/sequential.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

cyxwiz::Tensor ReadTensor(const json& j) {
    const auto shape = j.at("shape").get<std::vector<size_t>>();
    const auto values = j.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckTensor(const cyxwiz::Tensor& actual, const json& expected, const std::string& what) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<float>>();
    Check(actual.Shape() == shape, what + ": shape");
    const float* data = actual.ReadData<float>();
    double worst = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double diff = std::abs(static_cast<double>(data[i]) - values[i]);
        const double allowed = 2e-4 + 2e-4 * std::abs(values[i]);
        worst = (std::max)(worst, diff);
        Check(diff <= allowed, what + ": element " + std::to_string(i) + " is " + std::to_string(data[i]) +
                                   ", PyTorch " + std::to_string(values[i]));
    }
    (void)worst;
}

std::unique_ptr<cyxwiz::Module> MakeModule(const json& c) {
    const size_t in = c.at("input_size").get<size_t>(), hidden = c.at("hidden_size").get<size_t>();
    const size_t layers = c.at("num_layers").get<size_t>();
    const bool bi = c.at("bidirectional").get<bool>(), seq = c.at("return_sequences").get<bool>();
    const std::string kind = c.at("kind").get<std::string>();
    if (kind == "LSTM") return std::make_unique<cyxwiz::LSTMModule>(in, hidden, layers, bi, seq);
    if (kind == "GRU") return std::make_unique<cyxwiz::GRUModule>(in, hidden, layers, bi, seq);
    return std::make_unique<cyxwiz::RNNModule>(in, hidden, layers, seq, c.at("nonlinearity").get<std::string>(), bi);
}

// A bidirectional layer keys the module's layer{L}.forward.X as layer{L}_X
// and layer{L}.reverse.X as layer{L}_X_reverse (gradients: layer{L}_grad_X
// and layer{L}_grad_X_reverse).
std::string LayerKey(const std::string& module_key, bool gradient) {
    const size_t dot1 = module_key.find('.'), dot2 = module_key.find('.', dot1 + 1);
    const bool reverse = module_key.substr(dot1 + 1, dot2 - dot1 - 1) == "reverse";
    return module_key.substr(0, dot1) + (gradient ? "_grad_" : "_") + module_key.substr(dot2 + 1) +
           (reverse ? "_reverse" : "");
}

// The fixture's last-step data as full-sequence data for a layer, which
// always returns the whole sequence: the layer output's last step is the
// module output, and the last-step gradient expands with zeros elsewhere.
cyxwiz::Tensor LastStep(const cyxwiz::Tensor& sequence) {
    const auto& shape = sequence.Shape();
    std::vector<float> values(shape[0] * shape[2]);
    const float* data = sequence.ReadData<float>();
    for (size_t b = 0; b < shape[0]; ++b)
        for (size_t f = 0; f < shape[2]; ++f)
            values[b * shape[2] + f] = data[(b * shape[1] + shape[1] - 1) * shape[2] + f];
    return cyxwiz::Tensor({shape[0], shape[2]}, values.data(), cyxwiz::DataType::Float32);
}

cyxwiz::Tensor ExpandLastStep(const cyxwiz::Tensor& last, size_t seq) {
    const auto& shape = last.Shape();
    std::vector<float> values(shape[0] * seq * shape[1], 0.0f);
    const float* data = last.ReadData<float>();
    for (size_t b = 0; b < shape[0]; ++b)
        for (size_t f = 0; f < shape[1]; ++f)
            values[(b * seq + seq - 1) * shape[1] + f] = data[b * shape[1] + f];
    return cyxwiz::Tensor({shape[0], seq, shape[1]}, values.data(), cyxwiz::DataType::Float32);
}

// A bidirectional LSTMLayer / GRULayer / RNNLayer (batch_first) directly against the
// fixture: output, dx and every forward and reverse gradient.
void CheckBidirectionalLayer(const json& c, const std::string& name) {
    const int in = c.at("input_size").get<int>(), hidden = c.at("hidden_size").get<int>();
    const int layers = c.at("num_layers").get<int>();
    const std::string kind = c.at("kind").get<std::string>();
    std::unique_ptr<cyxwiz::Layer> layer;
    if (kind == "LSTM") layer = std::make_unique<cyxwiz::LSTMLayer>(in, hidden, layers, true, true, 0.0f);
    else if (kind == "GRU") layer = std::make_unique<cyxwiz::GRULayer>(in, hidden, layers, true, true, 0.0f);
    else layer = std::make_unique<cyxwiz::RNNLayer>(in, hidden, layers, true, true,
                                                    c.at("nonlinearity").get<std::string>());
    std::map<std::string, cyxwiz::Tensor> parameters;
    for (const auto& [k, v] : c.at("parameters").items()) parameters[LayerKey(k, false)] = ReadTensor(v);
    layer->SetParameters(parameters);

    const cyxwiz::Tensor input = ReadTensor(c.at("input"));
    const bool seq = c.at("return_sequences").get<bool>();
    const cyxwiz::Tensor output = layer->Forward(input);
    CheckTensor(seq ? output : LastStep(output), c.at("output"), name + " output");
    const cyxwiz::Tensor grad_output = ReadTensor(c.at("grad_output"));
    CheckTensor(layer->Backward(seq ? grad_output : ExpandLastStep(grad_output, input.Shape()[1])),
                c.at("grad_input"), name + " dx");
    const auto all = layer->GetParameters();
    for (const auto& [k, expected] : c.at("parameter_gradients").items()) {
        const auto it = all.find(LayerKey(k, true));
        Check(it != all.end(), name + ": no gradient " + LayerKey(k, true));
        CheckTensor(it->second, expected, name + " d" + k);
    }
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "recurrent_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_RECURRENT_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the recurrent fixture");
    const json fixture = json::parse(in);
    size_t passed = 0, layer_passed = 0;
    for (const bool providers : {true, false}) {
    cyxwiz::SetNeuralProvidersDisabledForTesting(!providers);
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name").get<std::string>() + (providers ? "" : " [ArrayFire]");
        try {
            auto module = MakeModule(c);
            std::map<std::string, cyxwiz::Tensor> parameters;
            for (const auto& [k, v] : c.at("parameters").items()) parameters[k] = ReadTensor(v);
            module->SetParameters(parameters);

            CheckTensor(module->Forward(ReadTensor(c.at("input"))), c.at("output"), name + " output");
            CheckTensor(module->Backward(ReadTensor(c.at("grad_output"))), c.at("grad_input"), name + " dx");
            // Gradients come keyed like the parameters they belong to.
            const auto gradients = module->GetGradients();
            for (const auto& [k, expected] : c.at("parameter_gradients").items()) {
                const auto it = gradients.find(k);
                Check(it != gradients.end(), name + ": no gradient " + k);
                CheckTensor(it->second, expected, name + " d" + k);
            }
            std::cout << "  ok " << name << " (" << module->GetName() << ")" << std::endl;
            ++passed;
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
    }
    for (const auto& c : fixture.at("cases")) {
        if (!c.at("bidirectional").get<bool>()) continue;
        const std::string name = c.at("name").get<std::string>() + " layer" + (providers ? "" : " [ArrayFire]");
        try {
            CheckBidirectionalLayer(c, name);
            std::cout << "  ok " << name << std::endl;
            ++layer_passed;
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
    }
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    // Large shapes on the ArrayFire path: no reference, but every step must
    // stay on the device (CUDA once refused GRU and large LSTM for generated-
    // kernel parameter overflow) and stay finite.
    const auto make = [](const std::string& kind, size_t in, size_t hidden, size_t layers, bool bi,
                         bool seq) -> std::unique_ptr<cyxwiz::Module> {
        if (kind == "LSTM") return std::make_unique<cyxwiz::LSTMModule>(in, hidden, layers, bi, seq);
        if (kind == "GRU") return std::make_unique<cyxwiz::GRUModule>(in, hidden, layers, bi, seq);
        return std::make_unique<cyxwiz::RNNModule>(in, hidden, layers, seq, "tanh", bi);
    };
    for (const std::string kind : {"LSTM", "GRU", "RNN"}) {
        for (const bool bi : {false, true}) {
            const size_t batch = 4, seq = 100, features = 32, hidden = 256;
            const auto module = make(kind, features, hidden, 2, bi, true);
            const std::string name = kind + (bi ? " bi" : "") + " h256 l2 s100 [ArrayFire]";
            try {
                const cyxwiz::Tensor x = cyxwiz::Tensor::Random({batch, seq, features}, cyxwiz::DataType::Float32);
                const cyxwiz::Tensor y = module->Forward(x);
                const cyxwiz::Tensor dx = module->Backward(cyxwiz::Tensor::Ones(y.Shape(), cyxwiz::DataType::Float32));
                for (const cyxwiz::Tensor* t : {&y, &dx}) {
                    const float* data = t->ReadData<float>();
                    for (size_t i = 0; i < t->NumElements(); ++i) Check(std::isfinite(data[i]), name + ": not finite");
                }
                std::cout << "  ok " << name << std::endl;
            } catch (const std::exception& error) {
                Check(false, name + ": " + error.what());
            }
        }
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    std::cout << "recurrent modules match PyTorch: " << passed << " cases" << std::endl;
    std::cout << "bidirectional recurrent layers match PyTorch: " << layer_passed << " cases" << std::endl;

    // Training through SequentialModel: the optimizer pairs each parameter
    // with the gradient of the same name, so 30 SGD steps on a fixed target
    // must at least halve the loss (it stalled when GetGradients returned the
    // weights themselves).
    for (const std::string kind : {"LSTM", "GRU", "RNN"}) {
        for (const bool bi : {false, true}) {
            const std::string name = kind + (bi ? " bi" : "") + " trains";
            try {
                cyxwiz::SequentialModel model;
                model.AddModule(make(kind, 3, 8, 1, bi, false));
                // A vanilla RNN overshoots at 0.5 (PyTorch nn.RNN diverges
                // there too) and converges at 0.1.
                auto optimizer = cyxwiz::CreateOptimizer(cyxwiz::OptimizerType::SGD, kind == "RNN" ? 0.1 : 0.5);
                const cyxwiz::Tensor x = cyxwiz::Tensor::Random({4, 5, 3}, cyxwiz::DataType::Float32);
                const size_t width = bi ? 16 : 8;
                const cyxwiz::Tensor target = cyxwiz::Tensor::Random({4, width}, cyxwiz::DataType::Float32) * 0.5f;
                double first = 0.0, last = 0.0;
                for (int step = 0; step < 30; ++step) {
                    const cyxwiz::Tensor y = model.Forward(x);
                    const cyxwiz::Tensor diff = y - target;
                    const float* d = diff.ReadData<float>();
                    double loss = 0.0;
                    for (size_t i = 0; i < diff.NumElements(); ++i) loss += 0.5 * d[i] * d[i];
                    if (step == 0) first = loss;
                    last = loss;
                    model.Backward(diff);
                    model.UpdateParameters(optimizer.get());
                }
                Check(last < 0.5 * first, name + ": loss " + std::to_string(first) + " -> " + std::to_string(last));
                std::cout << "  ok " << name << " (loss " << first << " -> " << last << ")" << std::endl;
            } catch (const std::exception& error) {
                Check(false, name + ": " + error.what());
            }
        }
    }
    return 0;
}
