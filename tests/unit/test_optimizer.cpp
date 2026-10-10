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
