#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/tensor.h>
#include <map>
#include <string>

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

    auto resumed_params = original_params;
    cyxwiz::AdamOptimizer resumed(0.001, 0.9, 0.999, 1e-8);
    REQUIRE(resumed.ImportState(state, error));
    REQUIRE(error.empty());
    REQUIRE(resumed.GetStepCount() == 1);

    original.Step(original_params, grads);
    resumed.Step(resumed_params, grads);

    const float* expected = original_params.at("w").Data<float>();
    const float* actual = resumed_params.at("w").Data<float>();
    REQUIRE(actual[0] == Catch::Approx(expected[0]).margin(1e-7f));
    REQUIRE(actual[1] == Catch::Approx(expected[1]).margin(1e-7f));
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
