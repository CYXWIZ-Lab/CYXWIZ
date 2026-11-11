// The whole small-MLP kernels (TOFIX206 3.2) against the model's own layers:
// FusedMlp's forward, and its gradients after a backward, equal
// SequentialModel's Forward / Backward / GetGradients for every fused
// activation and a spread of widths and row counts. Also: what FusedMlp
// refuses (a layer it does not fuse, too wide, two activations in a row),
// and the time of one forward + backward both ways.
//
// Runs on the device CYXWIZ_TEST_ARRAYFIRE_BACKEND names. Without the NVIDIA
// provider (another device), it checks the refusals and says the fused
// comparison was not run.

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include "computation_truth/test_device_selection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

cyxwiz::Tensor Random(const std::vector<size_t>& shape, unsigned seed, float scale = 1.0f) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& f : v) f = dist(rng);
    return cyxwiz::Tensor(shape, v.data(), cyxwiz::DataType::Float32);
}

// The largest difference relative to the reference's largest magnitude
double Difference(const cyxwiz::Tensor& got, const cyxwiz::Tensor& want) {
    if (got.Shape() != want.Shape()) return 1e30;
    const float* a = got.ReadData<float>();
    const float* b = want.ReadData<float>();
    double diff = 0.0, scale = 1e-6;
    for (size_t i = 0; i < want.NumElements(); ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(a[i] - b[i])));
        scale = std::max(scale, static_cast<double>(std::fabs(b[i])));
    }
    return diff / scale;
}

using AddActivation = std::function<void(cyxwiz::SequentialModel&)>;

struct Activation {
    const char* name;
    AddActivation add;
};

const std::vector<Activation>& Activations() {
    static const std::vector<Activation> all = {
        {"Tanh", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::TanhModule>(); }},
        {"ReLU", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::ReLUModule>(); }},
        {"LeakyReLU", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::LeakyReLUModule>(0.2f); }},
        {"Sigmoid", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::SigmoidModule>(); }},
        {"ELU", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::ELUModule>(0.7f); }},
        {"SELU", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::SELUModule>(); }},
        {"GELU", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::GELUModule>(); }},
        {"Swish", [](cyxwiz::SequentialModel& m) { m.Add<cyxwiz::SwishModule>(); }},
    };
    return all;
}

// inputs -> hidden... -> outputs, `act` after every hidden layer
cyxwiz::SequentialModel Chain(size_t inputs, const std::vector<size_t>& hidden, size_t outputs, const AddActivation& act) {
    cyxwiz::SequentialModel m;
    size_t in = inputs;
    for (size_t h : hidden) {
        m.Add<cyxwiz::LinearModule>(in, h, true);
        act(m);
        in = h;
    }
    m.Add<cyxwiz::LinearModule>(in, outputs, true);
    return m;
}

// Forward and gradients of the fused chain equal the model's
void Compare(cyxwiz::SequentialModel& model, const std::string& what, size_t rows, size_t inputs, size_t outputs) {
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    Check(fused != nullptr, what + ": fused (" + why + ")");
    if (!fused) return;
    model.SetTraining(true);
    const auto x = Random({rows, inputs}, 11);
    const auto dy = Random({rows, outputs}, 12);
    const cyxwiz::Tensor want = model.Forward(x);
    model.Backward(dy);
    const auto want_grads = model.GetGradients();
    const cyxwiz::Tensor got = fused->Forward(x, true);
    const auto got_grads = fused->Backward(dy);
    const double fd = Difference(got, want);
    Check(fd < 1e-5, what + ": forward differs by " + std::to_string(fd));
    Check(got_grads.size() == want_grads.size(), what + ": the same gradients");
    double gd = 0.0;
    for (const auto& [name, g] : want_grads) {
        const auto it = got_grads.find(name);
        if (it == got_grads.end()) {
            Check(false, what + ": no gradient " + name);
            continue;
        }
        gd = std::max(gd, Difference(it->second, g));
    }
    Check(gd < 1e-4, what + ": gradients differ by " + std::to_string(gd));
    // Without keep: the same forward
    Check(Difference(fused->Forward(x, false), want) < 1e-5, what + ": forward without keep");
}

void CheckParity() {
    struct Shape { size_t inputs; std::vector<size_t> hidden; size_t outputs; size_t rows; };
    const Shape shapes[] = {
        {4, {64, 64}, 2, 8},      // CartPole's networks, one rollout step
        {4, {64, 64}, 1, 64},     // the value network, one minibatch
        {17, {256, 256}, 6, 64},  // HalfCheetah's, one minibatch
        {348, {256}, 17, 5},      // a wide observation (Humanoid-like), rows not a multiple of 4
        {7, {1, 33, 129}, 3, 513},
        {3, {}, 2, 1},            // a single Dense layer, one row
    };
    for (const auto& act : Activations())
        for (const auto& s : shapes) {
            auto model = Chain(s.inputs, s.hidden, s.outputs, act.add);
            std::string what = std::string(act.name) + " " + std::to_string(s.inputs);
            for (size_t h : s.hidden) what += "-" + std::to_string(h);
            what += "-" + std::to_string(s.outputs) + " x" + std::to_string(s.rows);
            try {
                Compare(model, what, s.rows, s.inputs, s.outputs);
            } catch (const std::exception& e) {
                Check(false, what + ": " + e.what());
            }
            std::fflush(stdout);
        }
    // A plain Dense output after an activation on the last hidden layer only
    cyxwiz::SequentialModel last_act;
    last_act.Add<cyxwiz::LinearModule>(5, 9, true);
    last_act.Add<cyxwiz::LinearModule>(9, 3, true);
    last_act.Add<cyxwiz::TanhModule>();
    Compare(last_act, "Dense-Dense-Tanh", 6, 5, 3);
}

// Inputs that are column runs of a wider table (a minibatch's columns, TOFIX206
// 3.2 B2) are read in place from their offset: the same results as owned copies
void CheckViews() {
    const size_t rows = 64, inputs = 4, outputs = 2;
    auto model = Chain(inputs, {64, 64}, outputs, Activations()[0].add);
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    if (!fused) return;
    const auto table = Random({rows, 3 + inputs + 2}, 21);
    const auto x_view = table.Slice(1, 3, 3 + static_cast<int>(inputs));  // offset 3 * rows
    const auto dy_table = Random({rows, 1 + outputs}, 22);
    const auto dy_view = dy_table.Slice(1, 1, 1 + static_cast<int>(outputs));
    const auto owned = [](const cyxwiz::Tensor& t) {
        const float* data = t.ReadData<float>();
        return cyxwiz::Tensor(t.Shape(), std::vector<float>(data, data + t.NumElements()).data());
    };
    const auto x = owned(x_view), dy = owned(dy_view);
    const cyxwiz::Tensor want = fused->Forward(x, true);
    const auto want_grads = fused->Backward(dy);
    const cyxwiz::Tensor got = fused->Forward(x_view, true);
    const auto got_grads = fused->Backward(dy_view);
    Check(Difference(got, want) == 0.0, "a column view's forward equals its copy's");
    double gd = 0.0;
    for (const auto& [name, g] : want_grads) {
        const auto it = got_grads.find(name);
        gd = it == got_grads.end() ? 1.0 : std::max(gd, Difference(it->second, g));
    }
    Check(gd == 0.0, "a column view's gradients equal its copy's (differ by " + std::to_string(gd) + ")");
}

void CheckRefusals() {
    std::string why;
    auto wide = Chain(4, {300}, 2, Activations()[0].add);
    Check(!cyxwiz::FusedMlp::For(wide, why) && !why.empty(), "a 300-unit layer is refused: " + why);
    cyxwiz::SequentialModel twice;
    twice.Add<cyxwiz::LinearModule>(4, 8, true);
    twice.Add<cyxwiz::TanhModule>();
    twice.Add<cyxwiz::ReLUModule>();
    twice.Add<cyxwiz::LinearModule>(8, 2, true);
    Check(!cyxwiz::FusedMlp::For(twice, why) && why.find("activation without a Dense") != std::string::npos,
          "two activations in a row are refused: " + why);
    cyxwiz::SequentialModel dropout;
    dropout.Add<cyxwiz::LinearModule>(4, 8, true);
    dropout.Add<cyxwiz::DropoutModule>(0.1f);
    dropout.Add<cyxwiz::LinearModule>(8, 2, true);
    Check(!cyxwiz::FusedMlp::For(dropout, why) && why.find("not a Dense layer") != std::string::npos,
          "a Dropout layer is refused: " + why);
    cyxwiz::SequentialModel mish;
    mish.Add<cyxwiz::LinearModule>(4, 8, true);
    mish.Add<cyxwiz::MishModule>();
    mish.Add<cyxwiz::LinearModule>(8, 2, true);
    Check(!cyxwiz::FusedMlp::For(mish, why), "Mish is not fused: " + why);
}

// One forward + backward, both ways (the time includes reading one value back)
void Time(const char* what, size_t inputs, const std::vector<size_t>& hidden, size_t outputs, size_t rows) {
    auto model = Chain(inputs, hidden, outputs, Activations()[0].add);
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    if (!fused) return;
    model.SetTraining(true);
    const auto x = Random({rows, inputs}, 21);
    const auto dy = Random({rows, outputs}, 22);
    const auto sync = [](const cyxwiz::Tensor& t) { volatile float v = t.ReadData<float>()[0]; (void)v; };
    const auto run = [&](bool use_fused, int n) {
        cyxwiz::Tensor last;
        for (int i = 0; i < n; ++i) {
            if (use_fused) {
                fused->Forward(x, true);
                last = fused->Backward(dy).begin()->second;
            } else {
                model.Forward(x);
                model.Backward(dy);
                last = model.GetGradients().begin()->second;
            }
        }
        sync(last);
    };
    double ms[2];
    for (int k = 0; k < 2; ++k) {
        run(k == 1, 5);  // warm-up
        const auto start = std::chrono::steady_clock::now();
        run(k == 1, 200);
        ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 200.0;
    }
    std::printf("%-34s layers %.3f ms, fused %.3f ms per forward+backward (%.1fx)\n", what, ms[0], ms[1],
                ms[1] > 0.0 ? ms[0] / ms[1] : 0.0);
}

// `count <fused|layers> <n> [inputs hidden outputs rows]`: n forward + backward
// passes of one path and nothing else, for a CUDA call counter
// (cyxwiz-cuda-call-counter) to count per pass: run with n and with 0.
int CountMode(int argc, char** argv) {
    const bool use_fused = std::string(argv[2]) == "fused";
    const int n = std::atoi(argv[3]);
    const size_t inputs = argc > 7 ? std::strtoul(argv[4], nullptr, 10) : 4;
    const size_t hidden = argc > 7 ? std::strtoul(argv[5], nullptr, 10) : 64;
    const size_t outputs = argc > 7 ? std::strtoul(argv[6], nullptr, 10) : 2;
    const size_t rows = argc > 7 ? std::strtoul(argv[7], nullptr, 10) : 64;
    auto model = Chain(inputs, {hidden, hidden}, outputs, Activations()[0].add);
    model.SetTraining(true);
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    if (use_fused && !fused) {
        std::printf("not fused: %s\n", why.c_str());
        return EXIT_FAILURE;
    }
    const auto x = Random({rows, inputs}, 31);
    const auto dy = Random({rows, outputs}, 32);
    // Both inputs on the device before counting (a trainer's are)
    volatile float warm = x.ReadData<float>()[0] + dy.ReadData<float>()[0];
    (void)warm;
    x.GetSemanticArray();
    dy.GetSemanticArray();
    cyxwiz::Tensor last;
    for (int i = 0; i < n; ++i) {
        if (use_fused) {
            fused->Forward(x, true);
            last = fused->Backward(dy).begin()->second;
        } else {
            model.Forward(x);
            model.Backward(dy);
            last = model.GetGradients().begin()->second;
        }
    }
    if (n > 0) {
        volatile float v = last.ReadData<float>()[0];
        (void)v;
    }
    std::printf("%s: %d passes of %zu-%zu-%zu-%zu x%zu\n", use_fused ? "fused" : "layers", n, inputs, hidden, hidden,
                outputs, rows);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    if (argc >= 4 && std::string(argv[1]) == "count") return CountMode(argc, argv);
    CheckRefusals();
    auto probe = Chain(4, {64, 64}, 2, Activations()[0].add);
    std::string why;
    if (cyxwiz::FusedMlp::For(probe, why)) {
        CheckParity();
        CheckViews();
        Time("CartPole policy 4-64-64-2 x8", 4, {64, 64}, 2, 8);
        Time("CartPole minibatch 4-64-64-2 x64", 4, {64, 64}, 2, 64);
        Time("HalfCheetah 17-256-256-6 x64", 17, {256, 256}, 6, 64);
    } else {
        std::printf("fused comparison not run: %s\n", why.c_str());
    }
    if (failures == 0) std::printf("test_fused_mlp: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
