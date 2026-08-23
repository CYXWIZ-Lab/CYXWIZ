#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/optimizers/lr_warmup.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/scheduler.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

json LoadSchedulerFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("device", "") == "cpu");
    REQUIRE(!fixture.at("oracle").value("version", "").empty());
    return fixture.at("cases").at("scheduler_lr_sequences");
}

json LoadOptimizerWarmupFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    return fixture.at("cases").at("optimizer_lr_warmup_sequences");
}

cyxwiz::WarmupType ParseWarmupType(const std::string& value) {
    if (value == "linear") return cyxwiz::WarmupType::Linear;
    if (value == "cosine") return cyxwiz::WarmupType::Cosine;
    if (value == "none") return cyxwiz::WarmupType::None;
    throw std::invalid_argument("unsupported optimizer warmup type: " + value);
}

std::unique_ptr<cyxwiz::LRScheduler> CreateFixtureScheduler(
    const json& test_case,
    cyxwiz::Optimizer& optimizer) {
    const auto type = test_case.at("operation").get<std::string>();
    const auto& parameters = test_case.at("parameters");

    if (type == "torch.optim.lr_scheduler.StepLR") {
        return std::make_unique<cyxwiz::StepLR>(
            &optimizer,
            parameters.at("step_size").get<int>(),
            parameters.at("gamma").get<double>());
    }
    if (type == "torch.optim.lr_scheduler.ExponentialLR") {
        return std::make_unique<cyxwiz::ExponentialLR>(
            &optimizer, parameters.at("gamma").get<double>());
    }
    if (type == "torch.optim.lr_scheduler.CosineAnnealingLR") {
        return std::make_unique<cyxwiz::CosineAnnealingLR>(
            &optimizer,
            parameters.at("T_max").get<int>(),
            parameters.at("eta_min").get<double>());
    }
    if (type == "torch.optim.lr_scheduler.ReduceLROnPlateau") {
        return std::make_unique<cyxwiz::ReduceLROnPlateau>(
            &optimizer,
            parameters.at("mode").get<std::string>(),
            parameters.at("factor").get<double>(),
            parameters.at("patience").get<int>(),
            parameters.at("threshold").get<double>(),
            parameters.at("min_lr").get<double>());
    }
    if (type == "torch.optim.lr_scheduler.LambdaLR(linear_warmup)") {
        return std::make_unique<cyxwiz::LinearWarmupLR>(
            &optimizer,
            parameters.at("warmup_epochs").get<int>(),
            test_case.at("base_learning_rate").get<double>(),
            parameters.at("start_lr").get<double>());
    }
    if (type == "torch.optim.lr_scheduler.OneCycleLR") {
        REQUIRE(parameters.at("anneal_strategy").get<std::string>() == "cos");
        REQUIRE_FALSE(parameters.at("cycle_momentum").get<bool>());
        REQUIRE_FALSE(parameters.at("three_phase").get<bool>());
        return std::make_unique<cyxwiz::OneCycleLR>(
            &optimizer,
            parameters.at("max_lr").get<double>(),
            parameters.at("total_steps").get<int>(),
            parameters.at("pct_start").get<double>(),
            parameters.at("div_factor").get<double>(),
            parameters.at("final_div_factor").get<double>());
    }

    FAIL_CHECK("unsupported scheduler fixture operation: " + type);
    return nullptr;
}

} // namespace

TEST_CASE("Learning-rate schedulers match PyTorch sequences and reset",
          "[scheduler][pytorch]") {
    const auto cases = LoadSchedulerFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        INFO("case=" << test_case.at("name").get<std::string>());
        REQUIRE(test_case.at("call_order").get<std::string>() ==
                "optimizer.step_then_scheduler.step");

        const double base_lr =
            test_case.at("base_learning_rate").get<double>();
        const auto& tolerance = test_case.at("tolerance");
        const double margin = tolerance.at("atol").get<double>();
        cyxwiz::SGDOptimizer optimizer(base_lr);
        auto scheduler = CreateFixtureScheduler(test_case, optimizer);
        REQUIRE(scheduler != nullptr);
        REQUIRE(scheduler->GetLR() == Catch::Approx(
                    test_case.at("expected_initial_learning_rate").get<double>())
                    .margin(margin));

        for (const auto& expected_step : test_case.at("expected_steps")) {
            const int index = expected_step.at("index").get<int>();
            const float metric = expected_step.value("metric", 0.0f);
            scheduler->Step(index, metric);
            const double expected_lr =
                expected_step.at("learning_rate").get<double>();
            CHECK(scheduler->GetLR() ==
                  Catch::Approx(expected_lr).margin(margin));
            CHECK(optimizer.GetLearningRate() ==
                  Catch::Approx(expected_lr).margin(margin));
        }

        if (test_case.contains("expected_error_step")) {
            CHECK_THROWS_AS(
                scheduler->Step(test_case.at("expected_error_step").get<int>()),
                std::out_of_range);
        }

        scheduler->Reset();
        const double expected_reset =
            test_case.at("expected_reset_learning_rate").get<double>();
        CHECK(scheduler->GetLR() ==
              Catch::Approx(expected_reset).margin(margin));
        CHECK(optimizer.GetLearningRate() ==
              Catch::Approx(expected_reset).margin(margin));
    }
}

TEST_CASE("Optimizer LRWarmup matches PyTorch update ordering",
          "[scheduler][pytorch][optimizer]") {
    const auto cases = LoadOptimizerWarmupFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        INFO("case=" << test_case.at("name").get<std::string>());
        REQUIRE(test_case.at("call_order").get<std::string>() ==
                "optimizer.step_then_scheduler.step");
        const double margin =
            test_case.at("tolerance").at("atol").get<double>();
        const double base_lr =
            test_case.at("base_learning_rate").get<double>();
        const int warmup_steps = test_case.at("warmup_steps").get<int>();
        auto optimizer = std::make_unique<cyxwiz::SGDOptimizer>(base_lr);
        cyxwiz::LRWarmup warmup(
            std::move(optimizer), warmup_steps,
            ParseWarmupType(test_case.at("warmup_type").get<std::string>()),
            base_lr);

        CHECK(warmup.GetCurrentLR() == Catch::Approx(
                  test_case.at("expected_initial_learning_rate").get<double>())
                  .margin(margin));

        float parameter_value = test_case.at("initial_parameter").get<float>();
        float gradient_value = test_case.at("gradient").get<float>();
        std::map<std::string, cyxwiz::Tensor> parameters{
            {"w", cyxwiz::Tensor({1}, &parameter_value,
                                 cyxwiz::DataType::Float32)}};
        const std::map<std::string, cyxwiz::Tensor> gradients{
            {"w", cyxwiz::Tensor({1}, &gradient_value,
                                 cyxwiz::DataType::Float32)}};

        for (const auto& expected_step : test_case.at("expected_steps")) {
            warmup.Step(parameters, gradients);
            CHECK(warmup.GetCurrentLR() == Catch::Approx(
                      expected_step.at("learning_rate").get<double>())
                      .margin(margin));
            CHECK(parameters.at("w").ReadData<float>()[0] == Catch::Approx(
                      expected_step.at("parameter").get<float>())
                      .margin(margin));
            CHECK(warmup.GetWarmupProgress() == Catch::Approx(
                      expected_step.at("warmup_progress").get<double>())
                      .margin(margin));
            CHECK(warmup.IsWarmupComplete() ==
                  expected_step.at("warmup_complete").get<bool>());
        }
    }
}

TEST_CASE("Optimizer LRWarmup rejects invalid ownership and step counts",
          "[scheduler][validation]") {
    CHECK_THROWS_AS(cyxwiz::LRWarmup(nullptr, 4), std::invalid_argument);

    CHECK_THROWS_AS(
        cyxwiz::SGDOptimizer(-0.1),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::SGDOptimizer(0.1, -0.1),
        std::invalid_argument);

    auto optimizer = std::make_unique<cyxwiz::SGDOptimizer>(0.1);
    CHECK_THROWS_AS(
        cyxwiz::LRWarmup(std::move(optimizer), -1),
        std::invalid_argument);

    optimizer = std::make_unique<cyxwiz::SGDOptimizer>(0.1);
    CHECK_THROWS_AS(
        cyxwiz::LRWarmup(
            std::move(optimizer), 4, static_cast<cyxwiz::WarmupType>(99)),
        std::invalid_argument);

    optimizer = std::make_unique<cyxwiz::SGDOptimizer>(0.1);
    CHECK_THROWS_AS(
        cyxwiz::LRWarmup(
            std::move(optimizer), 4, cyxwiz::WarmupType::Linear,
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
}

TEST_CASE("Optimizer LRWarmup state resumes its nested SGD update",
          "[scheduler][checkpoint][optimizer]") {
    constexpr double base_lr = 0.1;
    float parameter_value = 1.0f;
    float gradient_value = 0.25f;
    std::map<std::string, cyxwiz::Tensor> source_parameters{
        {"w", cyxwiz::Tensor(
                  {1}, &parameter_value, cyxwiz::DataType::Float32)}};
    const std::map<std::string, cyxwiz::Tensor> gradients{
        {"w", cyxwiz::Tensor(
                  {1}, &gradient_value, cyxwiz::DataType::Float32)}};

    auto source_optimizer =
        std::make_unique<cyxwiz::SGDOptimizer>(base_lr, 0.9);
    cyxwiz::LRWarmup source(
        std::move(source_optimizer), 4, cyxwiz::WarmupType::Linear, base_lr);
    source.Step(source_parameters, gradients);
    source.ZeroGrad();
    source.Step(source_parameters, gradients);
    source.ZeroGrad();
    CHECK(source_parameters.at("w").ReadData<float>()[0] ==
          Catch::Approx(0.988125f).margin(1.0e-7f));

    cyxwiz::LRWarmupState state;
    std::string error;
    REQUIRE(source.ExportState(state, error));
    REQUIRE(error.empty());

    auto resumed_optimizer =
        std::make_unique<cyxwiz::SGDOptimizer>(base_lr, 0.9);
    cyxwiz::LRWarmup resumed(
        std::move(resumed_optimizer), 4, cyxwiz::WarmupType::Linear, base_lr);
    REQUIRE(resumed.ImportState(state, error));
    REQUIRE(error.empty());
    CHECK(resumed.GetCurrentLR() ==
          Catch::Approx(source.GetCurrentLR()).margin(1.0e-12));
    CHECK(resumed.GetOptimizer()->GetStepCount() ==
          source.GetOptimizer()->GetStepCount());

    auto resumed_parameters = source_parameters;
    source.Step(source_parameters, gradients);
    resumed.Step(resumed_parameters, gradients);
    CHECK(resumed_parameters.at("w").ReadData<float>()[0] ==
          Catch::Approx(source_parameters.at("w").ReadData<float>()[0])
              .margin(1.0e-7f));
    CHECK(resumed.GetCurrentLR() ==
          Catch::Approx(source.GetCurrentLR()).margin(1.0e-12));

    const double accepted_lr = resumed.GetCurrentLR();
    const int accepted_steps = resumed.GetOptimizer()->GetStepCount();
    auto invalid = state;
    invalid.current_step = 5;
    CHECK_FALSE(resumed.ImportState(invalid, error));
    CHECK_FALSE(error.empty());
    CHECK(resumed.GetCurrentLR() == Catch::Approx(accepted_lr).margin(1.0e-12));
    CHECK(resumed.GetOptimizer()->GetStepCount() == accepted_steps);

    invalid = state;
    invalid.optimizer_state.hyperparameters.at("momentum") = 0.5;
    CHECK_FALSE(resumed.ImportState(invalid, error));
    CHECK(error.find("wrapped optimizer state import failed") !=
          std::string::npos);
    CHECK(resumed.GetCurrentLR() == Catch::Approx(accepted_lr).margin(1.0e-12));
    CHECK(resumed.GetOptimizer()->GetStepCount() == accepted_steps);
}

TEST_CASE("Optimizer LRWarmup state resumes every PyTorch sequence",
          "[scheduler][pytorch][state][optimizer]") {
    const auto cases = LoadOptimizerWarmupFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const double margin =
            test_case.at("tolerance").at("atol").get<double>();
        const double base_lr =
            test_case.at("base_learning_rate").get<double>();
        const int warmup_steps = test_case.at("warmup_steps").get<int>();
        const auto warmup_type =
            ParseWarmupType(test_case.at("warmup_type").get<std::string>());
        const auto& expected_steps = test_case.at("expected_steps");
        const std::size_t split = expected_steps.size() / 2;

        auto source_optimizer =
            std::make_unique<cyxwiz::SGDOptimizer>(base_lr);
        cyxwiz::LRWarmup source(
            std::move(source_optimizer), warmup_steps, warmup_type, base_lr);
        float parameter_value = test_case.at("initial_parameter").get<float>();
        float gradient_value = test_case.at("gradient").get<float>();
        std::map<std::string, cyxwiz::Tensor> source_parameters{
            {"w", cyxwiz::Tensor(
                      {1}, &parameter_value, cyxwiz::DataType::Float32)}};
        const std::map<std::string, cyxwiz::Tensor> gradients{
            {"w", cyxwiz::Tensor(
                      {1}, &gradient_value, cyxwiz::DataType::Float32)}};
        for (std::size_t index = 0; index < split; ++index) {
            source.Step(source_parameters, gradients);
        }

        cyxwiz::LRWarmupState state;
        std::string error;
        REQUIRE(source.ExportState(state, error));
        auto resumed_optimizer =
            std::make_unique<cyxwiz::SGDOptimizer>(base_lr);
        cyxwiz::LRWarmup resumed(
            std::move(resumed_optimizer), warmup_steps, warmup_type, base_lr);
        REQUIRE(resumed.ImportState(state, error));
        auto resumed_parameters = source_parameters;

        for (std::size_t index = split; index < expected_steps.size(); ++index) {
            const auto& expected = expected_steps.at(index);
            resumed.Step(resumed_parameters, gradients);
            CHECK(resumed.GetCurrentLR() ==
                  Catch::Approx(expected.at("learning_rate").get<double>())
                      .margin(margin));
            CHECK(resumed_parameters.at("w").ReadData<float>()[0] ==
                  Catch::Approx(expected.at("parameter").get<float>())
                      .margin(margin));
            CHECK(resumed.GetWarmupProgress() ==
                  Catch::Approx(expected.at("warmup_progress").get<double>())
                      .margin(margin));
            CHECK(resumed.IsWarmupComplete() ==
                  expected.at("warmup_complete").get<bool>());
        }
    }
}

TEST_CASE("Scheduler state resumes PyTorch sequences transactionally",
          "[scheduler][pytorch][state]") {
    const auto cases = LoadSchedulerFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const auto& expected_steps = test_case.at("expected_steps");
        const std::size_t split = expected_steps.size() / 2;
        cyxwiz::SGDOptimizer source_optimizer(
            test_case.at("base_learning_rate").get<double>());
        auto source = CreateFixtureScheduler(test_case, source_optimizer);

        for (std::size_t index = 0; index < split; ++index) {
            const auto& expected = expected_steps.at(index);
            source->Step(
                expected.at("index").get<int>(),
                expected.value("metric", 0.0f));
        }

        cyxwiz::SchedulerState state;
        std::string error;
        REQUIRE(source->ExportState(state, error));
        REQUIRE(error.empty());

        cyxwiz::SGDOptimizer resumed_optimizer(
            test_case.at("base_learning_rate").get<double>());
        auto resumed = CreateFixtureScheduler(test_case, resumed_optimizer);
        REQUIRE(resumed->ImportState(state, error));
        REQUIRE(error.empty());
        CHECK(resumed->GetLR() == Catch::Approx(source->GetLR()).margin(1.0e-12));
        CHECK(resumed_optimizer.GetLearningRate() ==
              Catch::Approx(source_optimizer.GetLearningRate()).margin(1.0e-12));

        const double margin =
            test_case.at("tolerance").at("atol").get<double>();
        for (std::size_t index = split; index < expected_steps.size(); ++index) {
            const auto& expected = expected_steps.at(index);
            resumed->Step(
                expected.at("index").get<int>(),
                expected.value("metric", 0.0f));
            CHECK(resumed->GetLR() == Catch::Approx(
                      expected.at("learning_rate").get<double>())
                      .margin(margin));
        }

        const double accepted_lr = resumed->GetLR();
        auto invalid_state = state;
        invalid_state.schema_version = 99;
        CHECK_FALSE(resumed->ImportState(invalid_state, error));
        CHECK_FALSE(error.empty());
        CHECK(resumed->GetLR() == Catch::Approx(accepted_lr).margin(1.0e-12));

        invalid_state = state;
        invalid_state.scheduler_type = "WrongScheduler";
        CHECK_FALSE(resumed->ImportState(invalid_state, error));
        CHECK_FALSE(error.empty());
        CHECK(resumed->GetLR() == Catch::Approx(accepted_lr).margin(1.0e-12));

        invalid_state = state;
        invalid_state.current_learning_rate =
            std::numeric_limits<double>::quiet_NaN();
        CHECK_FALSE(resumed->ImportState(invalid_state, error));
        CHECK_FALSE(error.empty());
        CHECK(resumed->GetLR() == Catch::Approx(accepted_lr).margin(1.0e-12));
    }
}

TEST_CASE("Schedulers reject unsafe configurations",
          "[scheduler][validation]") {
    cyxwiz::SGDOptimizer optimizer(0.1);

    CHECK_THROWS_AS(cyxwiz::StepLR(nullptr, 1), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::StepLR(&optimizer, 0), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::ExponentialLR(nullptr, 0.9), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::ExponentialLR(
            &optimizer, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::CosineAnnealingLR(nullptr, 10), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::CosineAnnealingLR(&optimizer, 0), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::ReduceLROnPlateau(nullptr), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::ReduceLROnPlateau(&optimizer, "median"),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::ReduceLROnPlateau(&optimizer, "min", 1.0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::LinearWarmupLR(nullptr, 4, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::LinearWarmupLR(&optimizer, 0, 0.1),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::OneCycleLR(nullptr, 0.1, 10), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::OneCycleLR(&optimizer, 0.1, 0), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::OneCycleLR(&optimizer, 0.1, 10, -0.1),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::OneCycleLR(&optimizer, 0.1, 10, 1.1),
        std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::OneCycleLR(&optimizer, 0.1, 10, 0.3, 0.0),
        std::invalid_argument);
}
