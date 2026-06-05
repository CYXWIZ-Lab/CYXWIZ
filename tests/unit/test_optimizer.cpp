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
