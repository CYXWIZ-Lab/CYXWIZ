// BatchNormModule (BatchNorm 1D on [batch, features]) against
// torch.nn.BatchNorm1d (TOFIX228 1.4: it had no oracle). Every case trains 4
// batches in a row and compares, after each, the output, the input gradient,
// the gamma / beta gradients and both running statistics (so the running
// statistics are checked across batches, not only after one); then it
// evaluates a fifth batch with them. Batch sizes 2, 7 and 64, a case with eps
// 1e-3 and momentum 0.3, and one whose features have a variance near eps: that
// case is also run with eps x10 and must then miss, so a wrong epsilon shows.
//
// Fixture: computation_truth/fixtures/batchnorm1d_pytorch.json
// (reference/generate_batchnorm1d_fixtures.py).
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cpu|cuda|opencl|oneapi picks the device.

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include "../oracle/oracle_test.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::Tensor;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside =
        std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "batchnorm1d_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_BATCHNORM1D_FIXTURE;
}

Tensor FromFixture(const json& f) {
    const auto shape = f.at("shape").get<std::vector<size_t>>();
    const auto values = f.at("values").get<std::vector<float>>();
    return Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

// max over elements of |a - e| / (atol + rtol |e|); also a shape mismatch.
double Excess(const Tensor& actual, const json& expected, double atol, double rtol, std::string& why) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    if (actual.Shape() != shape) {
        why = "shape";
        return 1e300;
    }
    const auto values = expected.at("values").get<std::vector<double>>();
    const float* data = actual.ReadData<float>();
    double worst = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double diff = std::abs(static_cast<double>(data[i]) - values[i]);
        worst = std::max(worst, std::isfinite(data[i]) ? diff / (atol + rtol * std::abs(values[i])) : 1e300);
    }
    return worst;
}

struct Result {
    double worst = 0.0;
    std::string where;
};

void Note(Result& result, double excess, const std::string& where) {
    if (excess > result.worst) {
        result.worst = excess;
        result.where = where;
    }
}

Result RunCase(const json& c, double eps_scale) {
    Result result;
    const double atol = c.at("tolerance").at("atol").get<double>();
    const double rtol = c.at("tolerance").at("rtol").get<double>();
    cyxwiz::BatchNormModule bn(c.at("num_features").get<size_t>(),
                               static_cast<float>(c.at("eps").get<double>() * eps_scale),
                               static_cast<float>(c.at("momentum").get<double>()));
    bn.SetParameters({{"gamma", FromFixture(c.at("gamma"))}, {"beta", FromFixture(c.at("beta"))}});
    bn.SetTraining(true);
    std::string why;
    const auto& train = c.at("train");
    for (size_t i = 0; i < train.size(); ++i) {
        const auto& s = train.at(i);
        const std::string at = "batch " + std::to_string(i + 1) + " ";
        const Tensor output = bn.Forward(FromFixture(s.at("input")));
        Note(result, Excess(output, s.at("output"), atol, rtol, why), at + "output");
        const Tensor grad_input = bn.Backward(FromFixture(s.at("grad_output")));
        Note(result, Excess(grad_input, s.at("grad_input"), atol, rtol, why), at + "input gradient");
        auto gradients = bn.GetGradients();
        Note(result, Excess(gradients.at("gamma"), s.at("grad_gamma"), atol, rtol, why), at + "gamma gradient");
        Note(result, Excess(gradients.at("beta"), s.at("grad_beta"), atol, rtol, why), at + "beta gradient");
        auto parameters = bn.GetParameters();
        Note(result, Excess(parameters.at("running_mean"), s.at("running_mean"), atol, rtol, why), at + "running mean");
        Note(result, Excess(parameters.at("running_var"), s.at("running_var"), atol, rtol, why), at + "running var");
    }
    bn.SetTraining(false);
    const Tensor evaluated = bn.Forward(FromFixture(c.at("eval").at("input")));
    Note(result, Excess(evaluated, c.at("eval").at("output"), atol, rtol, why), "evaluation output");
    if (!why.empty()) result.where += " (" + why + " mismatch)";
    return result;
}

}  // namespace

int main(int, char** argv) {
    if (!cyxwiz::test::BeginOracleTest("test_computation_truth_batchnorm1d")) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    if (!in) {
        std::printf("FAIL: no fixture at %s\n", FixturePath(argv[0]).string().c_str());
        return EXIT_FAILURE;
    }
    const json fixture = json::parse(in);
    size_t checked = 0, sensitive = 0;
    for (const auto& [name, c] : fixture.at("cases").items()) {
        const Result run = RunCase(c, 1.0);
        Check(run.worst <= 1.0, name + ": " + run.where + " is " + std::to_string(run.worst) + " tolerances from PyTorch");
        std::printf("  %-32s worst %.3g tolerances (%s)\n", name.c_str(), run.worst, run.where.c_str());
        ++checked;
        if (name.rfind("variance_near_eps", 0) == 0) {
            const Result wrong = RunCase(c, 10.0);
            Check(wrong.worst > 10.0, name + ": with eps x10 the module stays within " + std::to_string(wrong.worst) +
                                          " tolerances; the case cannot catch a wrong epsilon here");
            ++sensitive;
        }
    }
    Check(sensitive == 1, "the variance-near-eps case is missing");
    if (failures == 0)
        std::printf("BatchNorm 1D matches PyTorch %s: %zu cases of 4 training batches and an evaluation (one "
                    "also caught a 10x eps)\n",
                    fixture.at("oracle").at("version").get<std::string>().c_str(), checked);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
