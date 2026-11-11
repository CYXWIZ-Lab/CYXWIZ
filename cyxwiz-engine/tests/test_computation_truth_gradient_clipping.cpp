// Gradient clipping by the global norm against PyTorch: the host clip
// (ClipGradientsByGlobalNorm, the optimizer nodes' "Gradient clip norm"),
// the device clip (ClipGradientsByGlobalNormOnDevice, PPO's per-tensor path)
// and the fused Adam step with its clip (AdamOptimizer::StepTogether, PPO on
// a provider) all match torch.nn.utils.clip_grad_norm_ (its default,
// error_if_nonfinite=False) followed by torch.optim.Adam: a clip that bites,
// one that does not, a NaN norm and an infinite norm. Fixture:
// computation_truth/fixtures/gradient_clipping_pytorch.json
// (reference/generate_gradient_clipping_fixtures.py).
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device; the fused
// Adam is checked where a provider steps Adam (CUDA with the NVIDIA provider).

#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/gradient_clipping.h>
#include <cyxwiz/tensor.h>

#include "computation_truth/test_device_selection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::Tensor;
using Params = std::map<std::string, Tensor>;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside =
        std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "gradient_clipping_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_GRADIENT_CLIPPING_FIXTURE;
}

// A number of the fixture: a JSON number, or "nan" / "inf" / "-inf"
float Number(const json& v) {
    if (v.is_number()) return v.get<float>();
    const std::string s = v.get<std::string>();
    if (s == "nan") return std::numeric_limits<float>::quiet_NaN();
    return s == "inf" ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
}

std::vector<float> Numbers(const json& f) {
    std::vector<float> out;
    for (const auto& v : f.at("values")) out.push_back(Number(v));
    return out;
}

Tensor FromFixture(const json& f) {
    const std::vector<size_t> shape = f.at("shape").get<std::vector<size_t>>();
    const std::vector<float> values = Numbers(f);
    return Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

Params Group(const json& g) {
    Params out;
    for (const auto& [name, f] : g.items()) out[name] = FromFixture(f);
    return out;
}

std::vector<float> Values(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

// The same non-finite positions, and the finite values within tol of max(1, |want|)
bool Matches(const std::vector<float>& got, const std::vector<float>& want, double tol, std::string& why) {
    if (got.size() != want.size()) {
        why = "size " + std::to_string(got.size()) + " vs " + std::to_string(want.size());
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = got[i], w = want[i];
        if (std::isnan(w) || std::isnan(g)) {
            if (std::isnan(w) != std::isnan(g)) {
                why = "element " + std::to_string(i) + ": " + std::to_string(g) + " vs " + std::to_string(w);
                return false;
            }
            continue;
        }
        if (std::isinf(w) || std::isinf(g)) {
            if (g != w) {
                why = "element " + std::to_string(i) + ": " + std::to_string(g) + " vs " + std::to_string(w);
                return false;
            }
            continue;
        }
        if (std::fabs(static_cast<double>(g) - w) > tol * std::max(1.0, std::fabs(static_cast<double>(w)))) {
            why = "element " + std::to_string(i) + ": " + std::to_string(g) + " vs " + std::to_string(w);
            return false;
        }
    }
    return true;
}

bool GroupMatches(const Params& got, const json& want, double tol, std::string& why) {
    for (const auto& [name, f] : want.items()) {
        const auto it = got.find(name);
        if (it == got.end()) {
            why = "no " + name;
            return false;
        }
        if (!Matches(Values(it->second), Numbers(f), tol, why)) {
            why = name + ": " + why;
            return false;
        }
    }
    return true;
}

void CheckClips(const json& cases) {
    for (const auto& c : cases) {
        const std::string name = c.at("name").get<std::string>();
        const float max_norm = c.at("max_norm").get<float>();
        const float want_norm = Number(c.at("norm"));
        for (const bool device : {false, true}) {
            const std::string what = std::string(device ? "device clip " : "host clip ") + name;
            Params g = Group(c.at("gradients"));
            float norm = 0.0f;
            try {
                norm = device ? Values(cyxwiz::ClipGradientsByGlobalNormOnDevice(g, max_norm)).at(0)
                              : cyxwiz::ClipGradientsByGlobalNorm(g, max_norm);
            } catch (const std::exception& e) {
                Check(false, what + ": refused (" + e.what() + ")");
                continue;
            }
            std::string why;
            Check(Matches({norm}, {want_norm}, 1e-5, why), what + ": the norm (" + why + ")");
            Check(GroupMatches(g, c.at("clipped"), 1e-5, why), what + ": the clipped gradients (" + why + ")");
        }
    }
}

// Three ways to take the Adam steps: the host clip then Step, the device clip
// then Step, and the fused StepTogether (where a provider runs it)
enum class Way { HostClip, DeviceClip, Fused };

void CheckAdam(const json& cases, bool fused_available) {
    for (const auto& c : cases) {
        const std::string name = c.at("name").get<std::string>();
        const float max_norm = c.at("max_norm").get<float>();
        for (const Way way : {Way::HostClip, Way::DeviceClip, Way::Fused}) {
            if (way == Way::Fused && !fused_available) continue;
            const std::string what = std::string(way == Way::HostClip     ? "Adam after the host clip "
                                                 : way == Way::DeviceClip ? "Adam after the device clip "
                                                                          : "fused Adam ") +
                                     name;
            Params p = Group(c.at("initial"));
            cyxwiz::AdamOptimizer adam(c.at("lr").get<double>(), c.at("beta1").get<double>(),
                                       c.at("beta2").get<double>(), c.at("eps").get<double>());
            int step = 0;
            for (const auto& s : c.at("steps")) {
                ++step;
                Params g = Group(s.at("gradients"));
                try {
                    if (way == Way::Fused) {
                        std::string why;
                        Check(cyxwiz::AdamOptimizer::StepTogether({{&adam, &p, &g}}, max_norm, why),
                              what + ": StepTogether runs (" + why + ")");
                    } else {
                        if (way == Way::HostClip) cyxwiz::ClipGradientsByGlobalNorm(g, max_norm);
                        else cyxwiz::ClipGradientsByGlobalNormOnDevice(g, max_norm);
                        adam.Step(p, g);
                    }
                } catch (const std::exception& e) {
                    Check(false, what + ": step " + std::to_string(step) + " refused (" + e.what() + ")");
                    break;
                }
                std::string why;
                Check(GroupMatches(p, s.at("parameters"), 1e-5, why),
                      what + ": the parameters after step " + std::to_string(step) + " (" + why + ")");
            }
        }
    }
}

}  // namespace

int main(int, char** argv) {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    if (!in) {
        std::printf("FAIL: no fixture at %s\n", FixturePath(argv[0]).string().c_str());
        return EXIT_FAILURE;
    }
    const json fixture = json::parse(in);
    // Does a provider step Adam on this device?
    Params probe{{"w", Tensor({1}, std::vector<float>{1.0f}.data(), cyxwiz::DataType::Float32)}};
    const Params probe_grad{{"w", Tensor({1}, std::vector<float>{1.0f}.data(), cyxwiz::DataType::Float32)}};
    cyxwiz::AdamOptimizer probe_adam(1e-3);
    std::string why;
    const bool fused = cyxwiz::AdamOptimizer::StepTogether({{&probe_adam, &probe, &probe_grad}}, 0.0f, why);
    CheckClips(fixture.at("clip"));
    CheckAdam(fixture.at("adam"), fused);
    if (failures == 0)
        std::printf("Gradient clipping matches PyTorch (torch %s): %zu clips on the host and the device, %zu Adam cases%s\n",
                    fixture.at("torch_version").get<std::string>().c_str(), fixture.at("clip").size(),
                    fixture.at("adam").size(), fused ? " including the fused Adam" : "");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
