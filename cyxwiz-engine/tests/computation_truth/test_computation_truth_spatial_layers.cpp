// The spatial (CNN) layers against PyTorch (TOFIX140 A1): every case of
// fixtures/spatial_layers_pytorch.json is replayed through the backend's
// SequentialModel modules on [H,W,C,N] tensors - SetParameters, Forward,
// Backward, GetGradients - and compared within the fixture's tolerance.
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

struct Tolerance {
    float absolute = 1e-4f;
    float relative = 1e-4f;
};

cyxwiz::Tensor ReadTensor(const json& fixture) {
    const auto shape = fixture.at("shape").get<std::vector<size_t>>();
    const auto values = fixture.at("values").get<std::vector<float>>();
    size_t count = 1;
    for (size_t d : shape) count *= d;
    Check(count == values.size(), "fixture tensor size mismatch");
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckTensor(const cyxwiz::Tensor& actual, const json& expected, const Tolerance& tol,
                 const std::string& context) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<float>>();
    Check(actual.Shape() == shape, context + ": shape differs");
    const float* data = actual.ReadData<float>();
    float worst = 0.0f;
    size_t worst_index = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const float difference = std::fabs(data[i] - values[i]);
        const float allowed = tol.absolute + tol.relative * std::fabs(values[i]);
        if (difference - allowed > worst) {
            worst = difference - allowed;
            worst_index = i;
        }
    }
    if (worst > 0.0f) {
        std::ostringstream ss;
        ss << context << ": element " << worst_index << " expected=" << values[worst_index]
           << " actual=" << data[worst_index] << " (atol " << tol.absolute << ", rtol " << tol.relative << ")";
        Check(false, ss.str());
    }
}

std::unique_ptr<cyxwiz::Module> MakeModule(const std::string& layer, const json& g, size_t channels_in) {
    using namespace cyxwiz;
    if (layer == "Conv2D")
        return std::make_unique<Conv2DModule>(static_cast<int>(channels_in), g.at("filters").get<int>(),
                                              g.at("kernel_size").get<int>(), g.at("stride").get<int>(),
                                              g.at("padding").get<int>(), true);
    if (layer == "MaxPool2D")
        return std::make_unique<MaxPool2DModule>(g.at("pool_size").get<int>(), g.at("stride").get<int>(),
                                                 g.at("padding").get<int>());
    if (layer == "AvgPool2D")
        return std::make_unique<AvgPool2DModule>(g.at("pool_size").get<int>(), g.at("stride").get<int>(),
                                                 g.at("padding").get<int>());
    if (layer == "ConvTranspose2D")
        return std::make_unique<ConvTranspose2DModule>(
            static_cast<int>(channels_in), g.at("out_channels").get<int>(), g.at("kernel_size").get<int>(),
            g.at("stride").get<int>(), g.at("padding").get<int>(), g.at("output_padding").get<int>(), true);
    if (layer == "GroupNorm")
        return std::make_unique<GroupNormModule>(g.at("num_groups").get<int>(), static_cast<int>(channels_in),
                                                 g.at("eps").get<float>(), g.at("affine").get<bool>());
    if (layer == "InstanceNorm")
        return std::make_unique<InstanceNorm2DModule>(static_cast<int>(channels_in), g.at("eps").get<float>(),
                                                      g.at("affine").get<bool>());
    if (layer == "Upsample")
        return std::make_unique<Upsample2DModule>(
            g.at("scale_factor").get<int>(),
            g.at("mode").get<int>() == 0 ? UpsampleMode::Nearest : UpsampleMode::Bilinear);
    if (layer == "PixelShuffle")
        return std::make_unique<PixelShuffleModule>(g.at("upscale_factor").get<int>());
    Check(false, "unknown layer in fixture: " + layer);
    return nullptr;
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "spatial_layers_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path("tests/computation_truth/fixtures/spatial_layers_pytorch.json");
}

}  // namespace

int main(int, char** argv) {
    const auto path = FixturePath(argv[0]);
    std::ifstream in(path);
    Check(static_cast<bool>(in), "cannot open " + path.string());
    json fixture;
    in >> fixture;
    Check(fixture.at("schema_version").get<int>() == 1, "fixture schema");

    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        const std::string layer = c.at("layer").get<std::string>();
        const Tolerance tol{c.at("tolerance").at("atol").get<float>(), c.at("tolerance").at("rtol").get<float>()};
        const cyxwiz::Tensor input = ReadTensor(c.at("input"));
        const size_t channels_in = input.Shape()[2];

        auto module = MakeModule(layer, c.at("geometry"), channels_in);
        std::map<std::string, cyxwiz::Tensor> parameters;
        for (const auto& [key, value] : c.at("parameters").items()) parameters[key] = ReadTensor(value);
        if (!parameters.empty()) module->SetParameters(parameters);

        const cyxwiz::Tensor output = module->Forward(input);
        CheckTensor(output, c.at("output"), tol, name + " forward");

        const cyxwiz::Tensor grad_input = module->Backward(ReadTensor(c.at("grad_output")));
        CheckTensor(grad_input, c.at("grad_input"), tol, name + " grad_input");

        if (!parameters.empty()) {
            const auto gradients = module->GetGradients();
            for (const auto& [key, expected] : c.at("parameter_gradients").items()) {
                const auto it = gradients.find(key);
                Check(it != gradients.end(), name + ": no gradient for " + key);
                CheckTensor(it->second, expected, tol, name + " grad " + key);
            }
        }
        std::cout << "  ok " << name << " (" << module->GetName() << ")\n";
        ++passed;
    }
    std::cout << "spatial layers match PyTorch: " << passed << " cases\n";
    return 0;
}
