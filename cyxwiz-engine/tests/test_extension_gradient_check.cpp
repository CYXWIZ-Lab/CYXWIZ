// The gradient checker (TOFIX125 P2 step 2.4): passes correct hand-written
// gradients, including built-in backend modules as a control, and fails
// wrong ones, for the input and for parameters.
#include "../src/core/extension_gradient_check.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::vector<float> Host(const cyxwiz::Tensor& tensor) {
    const float* data = tensor.ReadData<float>();
    return std::vector<float>(data, data + tensor.NumElements());
}

// y = x / (1 + |x|).
class Softsign : public cyxwiz::Module {
public:
    explicit Softsign(bool wrong) : wrong_(wrong) {}
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        return x / (x.Abs() + 1.0f);
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        const auto d = input_cache_.Abs() + 1.0f;
        return wrong_ ? grad / d : grad / (d * d);  // wrong: forgets the square
    }
    std::string GetName() const override { return "Softsign"; }

private:
    bool wrong_;
};

// y = a * x, learnable a.
class Scale : public cyxwiz::Module {
public:
    explicit Scale(bool wrong) : wrong_(wrong) {
        const float a = 0.7f;
        a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &a);
        grad_a_ = a_;
    }
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        return x * Host(a_)[0];
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        const auto g = Host(grad);
        const auto x = Host(input_cache_);
        double sum = 0.0;
        for (size_t i = 0; i < g.size(); ++i) sum += static_cast<double>(g[i]) * x[i];
        const float value = static_cast<float>(wrong_ ? sum * 2.0 : sum);  // wrong: doubled
        grad_a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &value);
        return grad * Host(a_)[0];
    }
    std::map<std::string, cyxwiz::Tensor> GetParameters() override { return {{"a", a_}}; }
    void SetParameters(const std::map<std::string, cyxwiz::Tensor>& p) override {
        if (p.count("a")) a_ = p.at("a");
    }
    std::map<std::string, cyxwiz::Tensor> GetGradients() override { return {{"a", grad_a_}}; }
    bool HasParameters() const override { return true; }
    std::string GetName() const override { return "Scale"; }

private:
    bool wrong_;
    cyxwiz::Tensor a_;
    cyxwiz::Tensor grad_a_;
};

// y = x + sin^2(a x) / a with a = 0.8: its true gradient 1 + sin(2ax) comes
// close to zero, where 32-bit rounding must not read as a failure.
class SnakeLike : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        const auto s = (x * 0.8f).Sin();
        return x + (s * s) / 0.8f;
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        return grad * ((input_cache_ * 1.6f).Sin() + 1.0f);
    }
    std::string GetName() const override { return "SnakeLike"; }
};

class Throws : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor&) override { throw std::runtime_error("boom in forward"); }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& g) override { return g; }
    std::string GetName() const override { return "Throws"; }
};

}  // namespace

int main() {
    std::cout << "correct gradients pass\n";
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<Softsign>(false); });
        Check(r.passed && r.error.empty(), "Softsign: " + cyxwiz::DescribeGradientCheck(r));
    }
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<Scale>(false); });
        Check(r.passed && r.error.empty(), "Scale (learnable a): " + cyxwiz::DescribeGradientCheck(r));
    }
    {
        cyxwiz::GradientCheckOptions options;
        options.input_shapes = {{4, 16}, {3, 5, 8}};  // many inputs, some near the zero-gradient points
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<SnakeLike>(); }, options);
        Check(r.passed, "a gradient that passes near zero (Snake-like): " + cyxwiz::DescribeGradientCheck(r));
    }
    std::cout << "built-in modules pass (the checker is not too strict)\n";
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<cyxwiz::TanhModule>(); });
        Check(r.passed, "Tanh: " + cyxwiz::DescribeGradientCheck(r));
    }
    {
        cyxwiz::GradientCheckOptions options;
        options.input_shapes = {{2, 3}, {4, 3}};
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>& shape) {
                return std::make_unique<cyxwiz::LinearModule>(shape.back(), 2, true);
            },
            options);
        Check(r.passed, "Linear 3 -> 2 with weights and bias: " + cyxwiz::DescribeGradientCheck(r));
    }
    std::cout << "wrong gradients fail\n";
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<Softsign>(true); });
        Check(!r.passed && r.worst == "input", "a wrong input gradient fails: " + cyxwiz::DescribeGradientCheck(r));
    }
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<Scale>(true); });
        Check(!r.passed && r.worst == "parameter a",
              "a wrong parameter gradient fails and is named: " + cyxwiz::DescribeGradientCheck(r));
    }
    std::cout << "a module that throws is reported, not crashed on\n";
    {
        const auto r = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::make_unique<Throws>(); });
        Check(!r.passed && r.error.find("boom in forward") != std::string::npos,
              cyxwiz::DescribeGradientCheck(r));
        const auto none = cyxwiz::CheckModuleGradients(
            [](const std::vector<size_t>&) { return std::unique_ptr<cyxwiz::Module>(); });
        Check(!none.passed && !none.error.empty(), "a factory that makes nothing: " + cyxwiz::DescribeGradientCheck(none));
    }
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_extension_gradient_check: all checks passed\n";
    return 0;
}
