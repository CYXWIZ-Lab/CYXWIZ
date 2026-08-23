#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/scheduler.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
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

        scheduler->Reset();
        const double expected_reset =
            test_case.at("expected_reset_learning_rate").get<double>();
        CHECK(scheduler->GetLR() ==
              Catch::Approx(expected_reset).margin(margin));
        CHECK(optimizer.GetLearningRate() ==
              Catch::Approx(expected_reset).margin(margin));
    }
}
