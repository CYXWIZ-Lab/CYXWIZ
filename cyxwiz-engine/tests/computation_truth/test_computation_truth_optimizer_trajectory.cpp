// 50-step optimizer trajectories against PyTorch, in which epsilon matters
// (TOFIX228 1.4, ORA-03). The 3-step fixtures in training_core_pytorch.json
// use gradients of 0.1 to 1: there a 1000x wrong epsilon passes. Here every
// optimizer runs 50 steps on a 4x5 parameter with gradients from 1e-8 to 1e-4
// ("tiny") and with exact zeros (whole steps, a column that never gets a
// gradient: "zeros"), in float32 and float64, and is compared at steps 1, 5,
// 10, 25 and 50. Its state is exported after step 25 and imported into a new
// optimizer, which finishes the run (the state carries all of it).
//
// Each case with an epsilon is also run with epsilon x10 and must then miss
// the fixture: the case can catch a wrong epsilon on this Engine, not only in
// the generator. Float64 cases run where the device has float64.
//
// Fixture: computation_truth/fixtures/optimizer_trajectory_pytorch.json
// (reference/generate_optimizer_trajectory_fixtures.py).

#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/adaptive.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/tensor.h>

#include "../oracle/oracle_test.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
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
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "optimizer_trajectory_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_OPTIMIZER_TRAJECTORY_FIXTURE;
}

std::unique_ptr<cyxwiz::Optimizer> MakeOptimizer(const std::string& name, const json& h, double epsilon_scale) {
    const auto get = [&h](const char* key) { return h.at(key).get<double>(); };
    if (name == "sgd_momentum") return std::make_unique<cyxwiz::SGDOptimizer>(get("learning_rate"), get("momentum"));
    if (name == "sgd_nesterov_decay")
        return std::make_unique<cyxwiz::SGDOptimizer>(get("learning_rate"), get("momentum"), get("weight_decay"), true);
    if (name == "adam")
        return std::make_unique<cyxwiz::AdamOptimizer>(get("learning_rate"), get("beta1"), get("beta2"),
                                                       get("epsilon") * epsilon_scale);
    if (name == "adamw")
        return std::make_unique<cyxwiz::AdamWOptimizer>(get("learning_rate"), get("beta1"), get("beta2"),
                                                        get("epsilon") * epsilon_scale, get("weight_decay"));
    if (name == "nadam")
        return std::make_unique<cyxwiz::NAdamOptimizer>(get("learning_rate"), get("beta1"), get("beta2"),
                                                        get("epsilon") * epsilon_scale);
    if (name == "rmsprop")
        return std::make_unique<cyxwiz::RMSpropOptimizer>(get("learning_rate"), get("alpha"),
                                                          get("epsilon") * epsilon_scale, get("momentum"));
    if (name == "adagrad")
        return std::make_unique<cyxwiz::AdaGradOptimizer>(get("learning_rate"), get("epsilon") * epsilon_scale);
    if (name == "adadelta") {
        auto optimizer = std::make_unique<cyxwiz::AdadeltaOptimizer>(get("rho"), get("epsilon") * epsilon_scale);
        optimizer->SetLearningRate(get("learning_rate"));
        return optimizer;
    }
    return nullptr;
}

Tensor MakeTensor(const std::vector<size_t>& shape, const std::vector<double>& values, bool float64) {
    if (float64) return Tensor(shape, values.data(), cyxwiz::DataType::Float64);
    std::vector<float> narrow(values.size());
    std::transform(values.begin(), values.end(), narrow.begin(), [](double v) { return static_cast<float>(v); });
    return Tensor(shape, narrow.data(), cyxwiz::DataType::Float32);
}

std::vector<double> Read(const Tensor& tensor) {
    std::vector<double> out(tensor.NumElements());
    if (tensor.GetDataType() == cyxwiz::DataType::Float64) {
        const double* data = tensor.ReadData<double>();
        std::copy(data, data + out.size(), out.begin());
    } else {
        const float* data = tensor.ReadData<float>();
        std::copy(data, data + out.size(), out.begin());
    }
    return out;
}

// max over elements of |a - e| / (atol + rtol |e|); above 1 is a miss.
double Excess(const std::vector<double>& actual, const std::vector<double>& expected, double atol, double rtol) {
    double worst = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::abs(actual[i] - expected[i]);
        const double excess = std::isfinite(actual[i]) ? diff / (atol + rtol * std::abs(expected[i])) : 1e300;
        worst = std::max(worst, excess);
    }
    return worst;
}

struct Trajectory {
    double worst = 0.0;    // over every checkpoint, in tolerances
    int worst_step = 0;
    std::string error;
};

// Runs a case to the end. With resume_after > 0 the state is exported after
// that step and a new optimizer imports it and finishes.
Trajectory Run(const json& c, const json& stream, double epsilon_scale, int resume_after) {
    Trajectory result;
    const bool float64 = c.at("dtype").get<std::string>() == "float64";
    const auto shape = stream.at("shape").get<std::vector<size_t>>();
    const double atol = c.at("tolerance").at("atol").get<double>();
    const double rtol = c.at("tolerance").at("rtol").get<double>();
    const std::string name = c.at("optimizer").get<std::string>();
    auto optimizer = MakeOptimizer(name, c.at("hyperparameters"), epsilon_scale);
    if (!optimizer) {
        result.error = "no Engine optimizer for " + name;
        return result;
    }
    Params parameters{{"weight", MakeTensor(shape, stream.at("initial_parameter").get<std::vector<double>>(), float64)}};
    const auto& gradients = stream.at("gradients");
    const auto& expected = c.at("expected");
    for (size_t index = 0; index < gradients.size(); ++index) {
        const int step = static_cast<int>(index) + 1;
        const Params gradient{{"weight", MakeTensor(shape, gradients.at(index).get<std::vector<double>>(), float64)}};
        optimizer->ZeroGrad();
        optimizer->Step(parameters, gradient);
        const Tensor& weight = parameters.at("weight");
        if (step == 1 && weight.GetDataType() != (float64 ? cyxwiz::DataType::Float64 : cyxwiz::DataType::Float32)) {
            result.error = std::string("the step changed the parameter's type from ") + (float64 ? "float64" : "float32");
            return result;
        }
        const std::string key = std::to_string(step);
        if (expected.contains(key)) {
            const double excess = Excess(Read(weight), expected.at(key).get<std::vector<double>>(), atol, rtol);
            if (excess > result.worst) {
                result.worst = excess;
                result.worst_step = step;
            }
        }
        if (step == resume_after) {
            cyxwiz::OptimizerState state;
            std::string error;
            if (!optimizer->ExportState(state, error)) {
                result.error = "state export after step " + key + ": " + error;
                return result;
            }
            optimizer = MakeOptimizer(name, c.at("hyperparameters"), epsilon_scale);
            if (!optimizer->ImportState(state, error)) {
                result.error = "state import after step " + key + ": " + error;
                return result;
            }
        }
    }
    return result;
}

}  // namespace

int RunAll(int argc, char** argv) {
    if (!cyxwiz::test::BeginOracleTest("test_computation_truth_optimizer_trajectory")) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    if (!in) {
        std::printf("FAIL: no fixture at %s\n", FixturePath(argv[0]).string().c_str());
        return EXIT_FAILURE;
    }
    const json fixture = json::parse(in);
    const bool has_double = cyxwiz::test::DeviceComputesInDouble();
    // One case only: test_computation_truth_optimizer_trajectory <case name>
    const std::string only = argc > 1 ? argv[1] : "";
    size_t checked = 0, skipped = 0, refused = 0, epsilon_checked = 0;
    for (const auto& [case_name, c] : fixture.at("cases").items()) {
        if (!only.empty() && case_name != only) continue;
        if (c.at("dtype").get<std::string>() == "float64" && !has_double) {
            ++skipped;
            continue;
        }
        const json& stream = fixture.at("streams").at(c.at("stream").get<std::string>());
        Trajectory run;
        try {
            run = Run(c, stream, 1.0, c.at("resume_after").get<int>());
        } catch (const std::exception& e) {
            // The Engine's optimizers train float32 parameters and refuse
            // float64 by name; that is reported, not passed. Anything else fails.
            const std::string what = e.what();
            if (c.at("dtype").get<std::string>() == "float64" && what.find("requires Float32") != std::string::npos) {
                std::printf("  %-36s refused float64: %s\n", case_name.c_str(), what.c_str());
                ++refused;
                continue;
            }
            run.error = what;
        }
        if (!run.error.empty()) {
            Check(false, case_name + ": " + run.error);
            continue;
        }
        Check(run.worst <= 1.0, case_name + ": step " + std::to_string(run.worst_step) + " is " +
                                    std::to_string(run.worst) + " tolerances from PyTorch");
        std::printf("  %-36s worst %.3g tolerances (step %d)\n", case_name.c_str(), run.worst, run.worst_step);
        std::fflush(stdout);
        ++checked;
        if (c.contains("epsilon_x10_tolerances")) {
            const Trajectory wrong = Run(c, stream, 10.0, 0);
            Check(wrong.error.empty() && wrong.worst > 10.0,
                  case_name + ": with epsilon x10 the run stays within " + std::to_string(wrong.worst) +
                      " tolerances; the case cannot catch a wrong epsilon here");
            ++epsilon_checked;
        }
    }
    Check(checked > 0, "no case ran");
    std::string skipped_note =
        skipped ? ", " + std::to_string(skipped) + " float64 cases skipped: no float64 here" : std::string();
    if (refused) skipped_note += ", " + std::to_string(refused) + " float64 cases refused by the Engine";
    if (failures == 0)
        std::printf("Optimizer trajectories match PyTorch %s: %zu cases of %d steps (%zu also caught a 10x "
                    "epsilon)%s\n",
                    fixture.at("oracle").at("version").get<std::string>().c_str(), checked,
                    fixture.at("steps").get<int>(), epsilon_checked, skipped_note.c_str());
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
    try {
        return RunAll(argc, argv);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
