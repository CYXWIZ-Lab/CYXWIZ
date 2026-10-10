// LSTMModule / GRUModule against PyTorch nn.LSTM / nn.GRU (TOFIX140).
//
// fixtures/recurrent_pytorch.json (generate_recurrent_fixtures.py): batch-first,
// 1-2 layers, uni- and bidirectional, full sequence or last step, hidden up to 64.
// Each case checks the output, dL/dx and every weight / bias gradient on the
// active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu),
// once through the neural providers where they serve the case and once with
// the providers off (the ArrayFire path).
#include "test_device_selection.h"

#include <cyxwiz/neural_provider.h>
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
    if (c.at("kind").get<std::string>() == "LSTM") return std::make_unique<cyxwiz::LSTMModule>(in, hidden, layers, bi, seq);
    return std::make_unique<cyxwiz::GRUModule>(in, hidden, layers, bi, seq);
}

// Unidirectional modules report gradients as layer{L}_grad_X; split
// bidirectional ones under the parameter key itself.
std::string GradientKey(const std::string& parameter_key, bool bidirectional) {
    if (bidirectional) return parameter_key;
    const size_t underscore = parameter_key.find('_');
    return parameter_key.substr(0, underscore + 1) + "grad_" + parameter_key.substr(underscore + 1);
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
    size_t passed = 0;
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
            const auto gradients = module->GetGradients();
            const bool bi = c.at("bidirectional").get<bool>();
            for (const auto& [k, expected] : c.at("parameter_gradients").items()) {
                const auto it = gradients.find(GradientKey(k, bi));
                Check(it != gradients.end(), name + ": no gradient " + GradientKey(k, bi));
                CheckTensor(it->second, expected, name + " d" + k);
            }
            std::cout << "  ok " << name << " (" << module->GetName() << ")" << std::endl;
            ++passed;
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
    }
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    std::cout << "recurrent modules match PyTorch: " << passed << " cases" << std::endl;
    return 0;
}
