#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "algorithms/arrayfire_backend_utils.h"
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/tensor.h>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>* g_optimizer_fallback_events =
    nullptr;
std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_optimizer_host_sync_events =
    nullptr;

void CaptureOptimizerFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    if (g_optimizer_fallback_events != nullptr) {
        g_optimizer_fallback_events->push_back(event);
    }
}

void CaptureOptimizerHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_optimizer_host_sync_events != nullptr) {
        g_optimizer_host_sync_events->push_back(event);
    }
}

void SetOptimizerTestEnv(const char* value) {
#ifdef _WIN32
    _putenv_s(kForceFallbackEnv, value);
#else
    setenv(kForceFallbackEnv, value, 1);
#endif
}

void ClearOptimizerTestEnv() {
#ifdef _WIN32
    _putenv_s(kForceFallbackEnv, "");
#else
    unsetenv(kForceFallbackEnv);
#endif
}

class ScopedOptimizerFallbackEnv {
public:
    explicit ScopedOptimizerFallbackEnv(const char* value) {
        const char* previous = std::getenv(kForceFallbackEnv);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        SetOptimizerTestEnv(value);
    }

    ~ScopedOptimizerFallbackEnv() {
        if (had_previous_) {
            SetOptimizerTestEnv(previous_.c_str());
        } else {
            ClearOptimizerTestEnv();
        }
    }

private:
    bool had_previous_ = false;
    std::string previous_;
};

class ScopedOptimizerEventCapture {
public:
    ScopedOptimizerEventCapture(
        std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>& fallback_events,
        std::vector<cyxwiz::ArrayFireHostSyncEvent>& host_sync_events) {
        g_optimizer_fallback_events = &fallback_events;
        g_optimizer_host_sync_events = &host_sync_events;
    }

    ~ScopedOptimizerEventCapture() {
        g_optimizer_fallback_events = nullptr;
        g_optimizer_host_sync_events = nullptr;
    }
};

#ifdef CYXWIZ_HAS_ARRAYFIRE
using OptimizerFactory =
    std::function<std::unique_ptr<cyxwiz::Optimizer>()>;

std::map<std::string, cyxwiz::Tensor> MakeStatefulOptimizerParameters() {
    const float values[] = {1.0f, -2.0f, 0.5f};
    return {{"weight", cyxwiz::Tensor(af::array(3, values))}};
}

std::map<std::string, cyxwiz::Tensor> MakeStatefulOptimizerGradients() {
    const float values[] = {0.25f, -0.5f, 0.125f};
    return {{"weight", cyxwiz::Tensor(af::array(3, values))}};
}

void RequireOptimizerFallbackContract(
    const OptimizerFactory& factory,
    const std::string& operation_name,
    const std::vector<std::string>& state_names) {
    auto strict_parameters = MakeStatefulOptimizerParameters();
    const auto gradients = MakeStatefulOptimizerGradients();
    auto strict_optimizer = factory();
    {
        const ScopedOptimizerFallbackEnv forced(operation_name.c_str());
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        REQUIRE_THROWS_AS(
            strict_optimizer->Step(strict_parameters, gradients),
            std::runtime_error);
    }
    REQUIRE(strict_optimizer->GetStepCount() == 0);
    cyxwiz::OptimizerState strict_state;
    std::string error;
    REQUIRE(strict_optimizer->ExportState(strict_state, error));
    REQUIRE(error.empty());
    REQUIRE(strict_state.tensors.empty());

    auto arrayfire_parameters = MakeStatefulOptimizerParameters();
    auto arrayfire_optimizer = factory();
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        arrayfire_optimizer->Step(arrayfire_parameters, gradients);
    }

    auto native_parameters = MakeStatefulOptimizerParameters();
    auto native_optimizer = factory();
    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
    {
        const ScopedOptimizerEventCapture capture(
            fallback_events, host_sync_events);
        const ScopedOptimizerFallbackEnv forced(operation_name.c_str());
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureOptimizerFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureOptimizerHostSync);
        native_optimizer->Step(native_parameters, gradients);
    }

    REQUIRE(fallback_events.size() == 1);
    REQUIRE(fallback_events.front().operation_name == operation_name);
    REQUIRE(fallback_events.front().reason_code == "gpu_backend_exception");
    REQUIRE_FALSE(fallback_events.front().fallback_forbidden);
    REQUIRE_FALSE(host_sync_events.empty());
    for (const auto& event : host_sync_events) {
        REQUIRE(event.attribution_category == "optimizer_cpu_path");
        REQUIRE(event.attribution_operation == operation_name);
    }

    const float* expected =
        arrayfire_parameters.at("weight").ReadData<float>();
    const float* actual = native_parameters.at("weight").ReadData<float>();
    for (size_t index = 0; index < 3; ++index) {
        REQUIRE(actual[index] == Catch::Approx(expected[index]).margin(1.0e-6f));
    }

    cyxwiz::OptimizerState expected_state;
    cyxwiz::OptimizerState actual_state;
    REQUIRE(arrayfire_optimizer->ExportState(expected_state, error));
    REQUIRE(native_optimizer->ExportState(actual_state, error));
    for (const auto& state_name : state_names) {
        REQUIRE(expected_state.tensors.count(state_name) == 1);
        REQUIRE(actual_state.tensors.count(state_name) == 1);
        const float* expected_values =
            expected_state.tensors.at(state_name).ReadData<float>();
        const float* actual_values =
            actual_state.tensors.at(state_name).ReadData<float>();
        for (size_t index = 0;
             index < expected_state.tensors.at(state_name).NumElements();
             ++index) {
            REQUIRE(actual_values[index] ==
                    Catch::Approx(expected_values[index]).margin(1.0e-6f));
        }
    }
}
#endif

} // namespace

TEST_CASE("SGD optimizer creation", "[optimizer]") {
    auto opt = cyxwiz::CreateOptimizer(cyxwiz::OptimizerType::SGD, 0.01);
    REQUIRE(opt != nullptr);
    REQUIRE(opt->GetLearningRate() == 0.01);
}

TEST_CASE("Adam optimizer creation", "[optimizer]") {
    auto opt = cyxwiz::CreateOptimizer(cyxwiz::OptimizerType::Adam, 0.001);
    REQUIRE(opt != nullptr);
}

TEST_CASE("SGD optimizer updates parameters", "[optimizer]") {
    float param_data[] = {1.0f, -2.0f};
    float grad_data[] = {0.5f, -1.0f};

    std::map<std::string, cyxwiz::Tensor> params;
    std::map<std::string, cyxwiz::Tensor> grads;
    params.emplace("w", cyxwiz::Tensor({2}, param_data, cyxwiz::DataType::Float32));
    grads.emplace("w", cyxwiz::Tensor({2}, grad_data, cyxwiz::DataType::Float32));

    cyxwiz::SGDOptimizer opt(0.1);
    opt.Step(params, grads);

    const float* updated = params.at("w").Data<float>();
    REQUIRE(updated[0] > 0.949f);
    REQUIRE(updated[0] < 0.951f);
    REQUIRE(updated[1] > -1.901f);
    REQUIRE(updated[1] < -1.899f);
}

TEST_CASE("SGD validates the complete step before mutating parameters",
          "[optimizer][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float valid_gradient_values[] = {0.5f, -1.0f};
    const float invalid_gradient_values[] = {0.25f};
    std::map<std::string, cyxwiz::Tensor> params = {
        {"a", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> grads = {
        {"a", cyxwiz::Tensor(
                  {2}, valid_gradient_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {1}, invalid_gradient_values, cyxwiz::DataType::Float32)},
    };

    cyxwiz::SGDOptimizer optimizer(0.1, 0.9);
    REQUIRE_THROWS_AS(optimizer.Step(params, grads), std::invalid_argument);
    REQUIRE(optimizer.GetStepCount() == 0);
    const float* unchanged = params.at("a").ReadData<float>();
    REQUIRE(unchanged[0] == Catch::Approx(1.0f));
    REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
}

TEST_CASE("SGD strict fallback rejects before parameter state or step mutation",
          "[optimizer][arrayfire][fallback][policy][truth]") {
#ifdef NDEBUG
    return;
#else
    const float parameter_values[] = {1.0f, -2.0f};
    const float gradient_values[] = {0.5f, -1.0f};
    std::map<std::string, cyxwiz::Tensor> params = {
        {"weight", cyxwiz::Tensor(
                       {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> grads = {
        {"weight", cyxwiz::Tensor(
                       {2}, gradient_values, cyxwiz::DataType::Float32)},
    };
    cyxwiz::SGDOptimizer optimizer(0.1, 0.9);

    {
        const ScopedOptimizerFallbackEnv forced("SGDOptimizer::Step");
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        REQUIRE_THROWS_AS(optimizer.Step(params, grads), std::runtime_error);
    }

    REQUIRE(optimizer.GetStepCount() == 0);
    const float* unchanged = params.at("weight").ReadData<float>();
    REQUIRE(unchanged[0] == Catch::Approx(1.0f));
    REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
    cyxwiz::OptimizerState state;
    std::string error;
    REQUIRE(optimizer.ExportState(state, error));
    REQUIRE(error.empty());
    REQUIRE(state.tensors.empty());
#endif
}

TEST_CASE("SGD allowed fallback records and attributes native CPU execution",
          "[optimizer][arrayfire][fallback][host_sync][truth]") {
#if !defined(CYXWIZ_HAS_ARRAYFIRE) || defined(NDEBUG)
    return;
#else
    const float parameter_values[] = {1.0f, -2.0f};
    const float gradient_values[] = {0.5f, -1.0f};
    std::map<std::string, cyxwiz::Tensor> params = {
        {"weight", cyxwiz::Tensor(
                       af::array(2, parameter_values))},
    };
    const std::map<std::string, cyxwiz::Tensor> grads = {
        {"weight", cyxwiz::Tensor(
                       af::array(2, gradient_values))},
    };
    cyxwiz::SGDOptimizer optimizer(0.1, 0.9);
    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;

    {
        const ScopedOptimizerEventCapture capture(
            fallback_events, host_sync_events);
        const ScopedOptimizerFallbackEnv forced("SGDOptimizer::Step");
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureOptimizerFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureOptimizerHostSync);
        optimizer.Step(params, grads);
    }

    REQUIRE(fallback_events.size() == 1);
    REQUIRE(fallback_events.front().operation_name == "SGDOptimizer::Step");
    REQUIRE(fallback_events.front().reason_code == "gpu_backend_exception");
    REQUIRE_FALSE(fallback_events.front().fallback_forbidden);
    REQUIRE_FALSE(host_sync_events.empty());
    for (const auto& event : host_sync_events) {
        REQUIRE(event.attribution_category == "optimizer_cpu_path");
        REQUIRE(event.attribution_operation == "SGDOptimizer::Step");
    }

    REQUIRE(optimizer.GetStepCount() == 1);
    const float* updated = params.at("weight").ReadData<float>();
    REQUIRE(updated[0] == Catch::Approx(0.95f));
    REQUIRE(updated[1] == Catch::Approx(-1.9f));
#endif
}

TEST_CASE("Adaptive optimizers reject invalid hyperparameters",
          "[optimizer][truth]") {
    REQUIRE_THROWS_AS(
        cyxwiz::RMSpropOptimizer(-0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::RMSpropOptimizer(0.1, 1.1), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::RMSpropOptimizer(0.1, 0.9, -1.0e-8),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdaGradOptimizer(-0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdaGradOptimizer(0.1, -1.0e-10),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdadeltaOptimizer(1.1), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdadeltaOptimizer(0.9, -1.0e-6),
        std::invalid_argument);
}

TEST_CASE("Adaptive optimizers preflight the complete step before mutation",
          "[optimizer][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float valid_gradient_values[] = {0.5f, -1.0f};
    const float invalid_gradient_values[] = {0.25f};
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"a", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"a", cyxwiz::Tensor(
                  {2}, valid_gradient_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {1}, invalid_gradient_values, cyxwiz::DataType::Float32)},
    };

    cyxwiz::RMSpropOptimizer optimizer(0.01, 0.9, 1.0e-8, 0.5);
    REQUIRE_THROWS_AS(
        optimizer.Step(parameters, gradients), std::invalid_argument);
    REQUIRE(optimizer.GetStepCount() == 0);
    const float* unchanged = parameters.at("a").ReadData<float>();
    REQUIRE(unchanged[0] == Catch::Approx(1.0f));
    REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
    cyxwiz::OptimizerState state;
    std::string error;
    REQUIRE(optimizer.ExportState(state, error));
    REQUIRE(state.tensors.empty());
}

TEST_CASE("Adaptive optimizers declare strict and compatible fallback",
          "[optimizer][arrayfire][fallback][policy][host_sync][truth]") {
#if !defined(CYXWIZ_HAS_ARRAYFIRE) || defined(NDEBUG)
    return;
#else
    SECTION("RMSprop") {
        RequireOptimizerFallbackContract(
            []() {
                return std::make_unique<cyxwiz::RMSpropOptimizer>(
                    0.01, 0.9, 1.0e-8, 0.5);
            },
            "RMSpropOptimizer::Step",
            {"square_average/weight", "momentum_buffer/weight"});
    }
    SECTION("AdaGrad") {
        RequireOptimizerFallbackContract(
            []() {
                return std::make_unique<cyxwiz::AdaGradOptimizer>(
                    0.1, 1.0e-10);
            },
            "AdaGradOptimizer::Step",
            {"sum/weight"});
    }
    SECTION("Adadelta") {
        RequireOptimizerFallbackContract(
            []() {
                auto optimizer =
                    std::make_unique<cyxwiz::AdadeltaOptimizer>(0.9, 1.0e-6);
                optimizer->SetLearningRate(0.5);
                return optimizer;
            },
            "AdadeltaOptimizer::Step",
            {"square_average/weight", "accumulated_delta/weight"});
    }
#endif
}

TEST_CASE("Adaptive optimizer state import is transactional",
          "[optimizer][checkpoint][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float gradient_values[] = {0.5f, -1.0f};
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(
                       {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"weight", cyxwiz::Tensor(
                       {2}, gradient_values, cyxwiz::DataType::Float32)},
    };
    cyxwiz::RMSpropOptimizer source(0.01, 0.9, 1.0e-8, 0.5);
    source.Step(parameters, gradients);
    cyxwiz::OptimizerState valid_state;
    std::string error;
    REQUIRE(source.ExportState(valid_state, error));

    cyxwiz::RMSpropOptimizer target(0.01, 0.9, 1.0e-8, 0.5);
    REQUIRE(target.ImportState(valid_state, error));
    auto invalid_state = valid_state;
    invalid_state.step_count = 99;
    invalid_state.tensors.erase("momentum_buffer/weight");
    REQUIRE_FALSE(target.ImportState(invalid_state, error));
    REQUIRE(error.find("incomplete") != std::string::npos);

    cyxwiz::OptimizerState preserved_state;
    REQUIRE(target.ExportState(preserved_state, error));
    REQUIRE(preserved_state.step_count == valid_state.step_count);
    REQUIRE(preserved_state.tensors.size() == valid_state.tensors.size());
}

TEST_CASE("LAMB rejects invalid hyperparameters", "[optimizer][truth]") {
    REQUIRE_THROWS_AS(
        cyxwiz::LAMBOptimizer(-0.001), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::LAMBOptimizer(0.001, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::LAMBOptimizer(0.001, 0.9, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::LAMBOptimizer(0.001, 0.9, 0.999, 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::LAMBOptimizer(0.001, 0.9, 0.999, 1.0e-6, -0.01),
        std::invalid_argument);
}

TEST_CASE("LAMB preflights the complete step before mutation",
          "[optimizer][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float valid_gradient_values[] = {0.5f, -1.0f};
    const float invalid_gradient_values[] = {0.25f};
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"a", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"a", cyxwiz::Tensor(
                  {2}, valid_gradient_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {1}, invalid_gradient_values, cyxwiz::DataType::Float32)},
    };

    cyxwiz::LAMBOptimizer optimizer;
    REQUIRE_THROWS_AS(
        optimizer.Step(parameters, gradients), std::invalid_argument);
    REQUIRE(optimizer.GetStepCount() == 0);
    const float* unchanged = parameters.at("a").ReadData<float>();
    REQUIRE(unchanged[0] == Catch::Approx(1.0f));
    REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
    cyxwiz::OptimizerState state;
    std::string error;
    REQUIRE(optimizer.ExportState(state, error));
    REQUIRE(state.tensors.empty());
}

TEST_CASE("LAMB declares strict and compatible fallback",
          "[optimizer][arrayfire][fallback][policy][host_sync][truth]") {
#if !defined(CYXWIZ_HAS_ARRAYFIRE) || defined(NDEBUG)
    return;
#else
    RequireOptimizerFallbackContract(
        []() {
            return std::make_unique<cyxwiz::LAMBOptimizer>(
                0.01, 0.9, 0.999, 1.0e-6, 0.02);
        },
        "LAMBOptimizer::Step",
        {"first_moment/weight", "second_moment/weight"});
#endif
}

TEST_CASE("LAMB state import is transactional",
          "[optimizer][checkpoint][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float gradient_values[] = {0.5f, -1.0f};
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(
                       {2}, parameter_values, cyxwiz::DataType::Float32)},
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"weight", cyxwiz::Tensor(
                       {2}, gradient_values, cyxwiz::DataType::Float32)},
    };
    cyxwiz::LAMBOptimizer source(0.01, 0.9, 0.999, 1.0e-6, 0.02);
    source.Step(parameters, gradients);
    cyxwiz::OptimizerState valid_state;
    std::string error;
    REQUIRE(source.ExportState(valid_state, error));

    cyxwiz::LAMBOptimizer target(0.01, 0.9, 0.999, 1.0e-6, 0.02);
    REQUIRE(target.ImportState(valid_state, error));
    auto invalid_state = valid_state;
    invalid_state.step_count = 99;
    invalid_state.tensors.erase("second_moment/weight");
    REQUIRE_FALSE(target.ImportState(invalid_state, error));
    REQUIRE(error.find("incomplete") != std::string::npos);

    cyxwiz::OptimizerState preserved_state;
    REQUIRE(target.ExportState(preserved_state, error));
    REQUIRE(preserved_state.step_count == valid_state.step_count);
    REQUIRE(preserved_state.tensors.size() == valid_state.tensors.size());
}

TEST_CASE("Adam-family optimizers reject invalid hyperparameters",
          "[optimizer][truth]") {
    REQUIRE_THROWS_AS(cyxwiz::AdamOptimizer(-0.001), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdamOptimizer(0.001, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdamOptimizer(0.001, 0.9, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdamOptimizer(0.001, 0.9, 0.999, 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::AdamWOptimizer(0.001, 0.9, 0.999, 1.0e-8, -0.01),
        std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::NAdamOptimizer(-0.002), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::NAdamOptimizer(0.002, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::NAdamOptimizer(0.002, 0.9, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::NAdamOptimizer(0.002, 0.9, 0.999, 0.0),
        std::invalid_argument);
}

TEST_CASE("Adam family preflights the complete step before mutation",
          "[optimizer][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float valid_gradient_values[] = {0.5f, -1.0f};
    const float invalid_gradient_values[] = {0.25f};
    const auto make_parameters = [&]() {
        return std::map<std::string, cyxwiz::Tensor>{
            {"a", cyxwiz::Tensor(
                      {2}, parameter_values, cyxwiz::DataType::Float32)},
            {"b", cyxwiz::Tensor(
                      {2}, parameter_values, cyxwiz::DataType::Float32)},
        };
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"a", cyxwiz::Tensor(
                  {2}, valid_gradient_values, cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  {1}, invalid_gradient_values, cyxwiz::DataType::Float32)},
    };

    SECTION("AdamW rejects before decoupled weight decay") {
        auto parameters = make_parameters();
        cyxwiz::AdamWOptimizer optimizer(0.01, 0.9, 0.999, 1.0e-8, 0.1);
        REQUIRE_THROWS_AS(
            optimizer.Step(parameters, gradients), std::invalid_argument);
        REQUIRE(optimizer.GetStepCount() == 0);
        const float* unchanged = parameters.at("a").ReadData<float>();
        REQUIRE(unchanged[0] == Catch::Approx(1.0f));
        REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
        cyxwiz::OptimizerState state;
        std::string error;
        REQUIRE(optimizer.ExportState(state, error));
        REQUIRE(state.tensors.empty());
    }

    SECTION("NAdam rejects before moment or schedule mutation") {
        auto parameters = make_parameters();
        cyxwiz::NAdamOptimizer optimizer;
        REQUIRE_THROWS_AS(
            optimizer.Step(parameters, gradients), std::invalid_argument);
        REQUIRE(optimizer.GetStepCount() == 0);
        const float* unchanged = parameters.at("a").ReadData<float>();
        REQUIRE(unchanged[0] == Catch::Approx(1.0f));
        REQUIRE(unchanged[1] == Catch::Approx(-2.0f));
        cyxwiz::OptimizerState state;
        std::string error;
        REQUIRE(optimizer.ExportState(state, error));
        REQUIRE(state.tensors.empty());
    }
}

TEST_CASE("Adam family declares strict and compatible fallback",
          "[optimizer][arrayfire][fallback][policy][host_sync][truth]") {
#if !defined(CYXWIZ_HAS_ARRAYFIRE) || defined(NDEBUG)
    return;
#else
    SECTION("Adam") {
        RequireOptimizerFallbackContract(
            []() {
                return std::make_unique<cyxwiz::AdamOptimizer>(
                    0.01, 0.9, 0.999, 1.0e-8);
            },
            "AdamOptimizer::Step",
            {"first_moment/weight", "second_moment/weight"});
    }
    SECTION("AdamW") {
        RequireOptimizerFallbackContract(
            []() {
                return std::make_unique<cyxwiz::AdamWOptimizer>(
                    0.01, 0.9, 0.999, 1.0e-8, 0.1);
            },
            "AdamWOptimizer::Step",
            {"first_moment/weight", "second_moment/weight"});
    }
    SECTION("NAdam") {
        RequireOptimizerFallbackContract(
            []() {
                return std::make_unique<cyxwiz::NAdamOptimizer>(
                    0.01, 0.9, 0.999, 1.0e-8);
            },
            "NAdamOptimizer::Step",
            {"first_moment/weight", "second_moment/weight"});
    }
#endif
}

TEST_CASE("AdamW and NAdam state imports are transactional",
          "[optimizer][checkpoint][truth]") {
    const float parameter_values[] = {1.0f, -2.0f};
    const float gradient_values[] = {0.5f, -1.0f};
    const auto make_parameters = [&]() {
        return std::map<std::string, cyxwiz::Tensor>{
            {"weight", cyxwiz::Tensor(
                           {2}, parameter_values,
                           cyxwiz::DataType::Float32)},
        };
    };
    const std::map<std::string, cyxwiz::Tensor> gradients = {
        {"weight", cyxwiz::Tensor(
                       {2}, gradient_values, cyxwiz::DataType::Float32)},
    };

    SECTION("AdamW") {
        auto parameters = make_parameters();
        cyxwiz::AdamWOptimizer source(0.01, 0.9, 0.999, 1.0e-8, 0.1);
        source.Step(parameters, gradients);
        cyxwiz::OptimizerState valid_state;
        std::string error;
        REQUIRE(source.ExportState(valid_state, error));
        REQUIRE(valid_state.optimizer_type == "AdamW");
        cyxwiz::AdamWOptimizer target(0.01, 0.9, 0.999, 1.0e-8, 0.1);
        REQUIRE(target.ImportState(valid_state, error));
        auto invalid_state = valid_state;
        invalid_state.step_count = 99;
        invalid_state.tensors.erase("second_moment/weight");
        REQUIRE_FALSE(target.ImportState(invalid_state, error));
        cyxwiz::OptimizerState preserved;
        REQUIRE(target.ExportState(preserved, error));
        REQUIRE(preserved.step_count == valid_state.step_count);
    }

    SECTION("NAdam") {
        auto parameters = make_parameters();
        cyxwiz::NAdamOptimizer source(0.01, 0.9, 0.999, 1.0e-8);
        source.Step(parameters, gradients);
        cyxwiz::OptimizerState valid_state;
        std::string error;
        REQUIRE(source.ExportState(valid_state, error));
        REQUIRE(valid_state.optimizer_type == "NAdam");
        cyxwiz::NAdamOptimizer target(0.01, 0.9, 0.999, 1.0e-8);
        REQUIRE(target.ImportState(valid_state, error));
        auto invalid_state = valid_state;
        invalid_state.step_count = 99;
        invalid_state.tensors.erase("mu_product/weight");
        REQUIRE_FALSE(target.ImportState(invalid_state, error));
        cyxwiz::OptimizerState preserved;
        REQUIRE(target.ExportState(preserved, error));
        REQUIRE(preserved.step_count == valid_state.step_count);
    }
}

TEST_CASE("Adam optimizer updates parameters", "[optimizer]") {
    float param_data[] = {1.0f, -2.0f};
    float grad_data[] = {0.5f, -1.0f};

    std::map<std::string, cyxwiz::Tensor> params;
    std::map<std::string, cyxwiz::Tensor> grads;
    params.emplace("w", cyxwiz::Tensor({2}, param_data, cyxwiz::DataType::Float32));
    grads.emplace("w", cyxwiz::Tensor({2}, grad_data, cyxwiz::DataType::Float32));

    cyxwiz::AdamOptimizer opt(0.001);
    opt.Step(params, grads);

    const float* updated = params.at("w").Data<float>();
    REQUIRE(updated[0] > 0.998f);
    REQUIRE(updated[0] < 1.0f);
    REQUIRE(updated[1] > -2.0f);
    REQUIRE(updated[1] < -1.998f);
}

TEST_CASE("Adam optimizer state resumes the exact next step", "[optimizer][checkpoint]") {
    float param_data[] = {1.0f, -2.0f};
    float grad_data[] = {0.5f, -1.0f};

    std::map<std::string, cyxwiz::Tensor> original_params;
    std::map<std::string, cyxwiz::Tensor> grads;
    original_params.emplace(
        "w", cyxwiz::Tensor({2}, param_data, cyxwiz::DataType::Float32));
    grads.emplace(
        "w", cyxwiz::Tensor({2}, grad_data, cyxwiz::DataType::Float32));

    cyxwiz::AdamOptimizer original(0.001, 0.9, 0.999, 1e-8);
    original.Step(original_params, grads);

    cyxwiz::OptimizerState state;
    std::string error;
    REQUIRE(original.ExportState(state, error));
    REQUIRE(error.empty());
    REQUIRE(state.optimizer_type == "Adam");
    REQUIRE(state.step_count == 1);
    REQUIRE(state.tensors.count("first_moment/w") == 1);
    REQUIRE(state.tensors.count("second_moment/w") == 1);
    REQUIRE(state.tensors.count("parameter_step/w") == 1);

    auto resumed_params = original_params;
    cyxwiz::AdamOptimizer resumed(0.001, 0.9, 0.999, 1e-8);
    REQUIRE(resumed.ImportState(state, error));
    REQUIRE(error.empty());
    REQUIRE(resumed.GetStepCount() == 1);

    auto legacy_state = state;
    legacy_state.tensors.erase("parameter_step/w");
    auto legacy_params = original_params;
    cyxwiz::AdamOptimizer legacy_resumed(0.001, 0.9, 0.999, 1e-8);
    REQUIRE(legacy_resumed.ImportState(legacy_state, error));
    REQUIRE(error.empty());

    original.Step(original_params, grads);
    resumed.Step(resumed_params, grads);
    legacy_resumed.Step(legacy_params, grads);

    const float* expected = original_params.at("w").Data<float>();
    const float* actual = resumed_params.at("w").Data<float>();
    REQUIRE(actual[0] == Catch::Approx(expected[0]).margin(1e-7f));
    REQUIRE(actual[1] == Catch::Approx(expected[1]).margin(1e-7f));
    const float* legacy_actual = legacy_params.at("w").Data<float>();
    REQUIRE(legacy_actual[0] == Catch::Approx(expected[0]).margin(1e-7f));
    REQUIRE(legacy_actual[1] == Catch::Approx(expected[1]).margin(1e-7f));
    REQUIRE(resumed.GetStepCount() == original.GetStepCount());

    auto incomplete = state;
    incomplete.tensors.erase("second_moment/w");
    REQUIRE_FALSE(resumed.ImportState(incomplete, error));
    REQUIRE(error.find("incomplete moment tensor pairs") != std::string::npos);
    REQUIRE(resumed.GetStepCount() == 2);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
static bool HasArrayFireDeviceBackend() {
    try {
        af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL;
    } catch (...) {
        return false;
    }
}

TEST_CASE("SGD GPU step keeps parameters device resident until host read", "[optimizer][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    std::map<std::string, cyxwiz::Tensor> params;
    std::map<std::string, cyxwiz::Tensor> grads;
    params.emplace("w", cyxwiz::Tensor(af::constant(1.0f, 2, f32)));
    grads.emplace("w", cyxwiz::Tensor(af::constant(0.5f, 2, f32)));

    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    cyxwiz::SGDOptimizer opt(0.1);
    opt.Step(params, grads);

    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float* updated = params.at("w").Data<float>();
    REQUIRE(updated[0] == Catch::Approx(0.95f));
    REQUIRE(updated[1] == Catch::Approx(0.95f));
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + params.at("w").NumBytes());
}

TEST_CASE("Adam GPU step keeps parameters device resident until host read", "[optimizer][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    std::map<std::string, cyxwiz::Tensor> params;
    std::map<std::string, cyxwiz::Tensor> grads;
    params.emplace("w", cyxwiz::Tensor(af::constant(1.0f, 2, f32)));
    grads.emplace("w", cyxwiz::Tensor(af::constant(0.5f, 2, f32)));

    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    cyxwiz::AdamOptimizer opt(0.001);
    opt.Step(params, grads);

    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float* updated = params.at("w").Data<float>();
    REQUIRE(updated[0] == Catch::Approx(0.999f));
    REQUIRE(updated[1] == Catch::Approx(0.999f));
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + params.at("w").NumBytes());
}

static void RequireOptimizerKeepsParametersDeviceResident(cyxwiz::Optimizer& opt) {
    std::map<std::string, cyxwiz::Tensor> params;
    std::map<std::string, cyxwiz::Tensor> grads;
    params.emplace("w", cyxwiz::Tensor(af::constant(1.0f, 2, f32)));
    grads.emplace("w", cyxwiz::Tensor(af::constant(0.5f, 2, f32)));

    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    opt.Step(params, grads);

    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float* updated = params.at("w").Data<float>();
    REQUIRE(updated[0] < 1.0f);
    REQUIRE(updated[1] < 1.0f);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + params.at("w").NumBytes());
}

TEST_CASE("Adaptive GPU optimizers keep parameters device resident until host read", "[optimizer][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    SECTION("AdamW") {
        cyxwiz::AdamWOptimizer opt(0.001);
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }

    SECTION("RMSprop") {
        cyxwiz::RMSpropOptimizer opt(0.001);
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }

    SECTION("AdaGrad") {
        cyxwiz::AdaGradOptimizer opt(0.01);
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }

    SECTION("NAdam") {
        cyxwiz::NAdamOptimizer opt(0.002);
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }

    SECTION("Adadelta") {
        cyxwiz::AdadeltaOptimizer opt;
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }

    SECTION("LAMB") {
        cyxwiz::LAMBOptimizer opt(0.001);
        RequireOptimizerKeepsParametersDeviceResident(opt);
    }
}
#endif
