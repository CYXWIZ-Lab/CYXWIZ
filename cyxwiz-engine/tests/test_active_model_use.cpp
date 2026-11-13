// The active model's turns: Explain, Feature Importance and generation from
// this session's model take turns with Train, Test and each other.
#include "core/active_model_use.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using cyxwiz::ActiveModelBusy;
using cyxwiz::ActiveModelUse;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    Check(ActiveModelBusy(false, false).empty(), "free at the start");
    Check(ActiveModelBusy(true, false) == "Train is running on the model", "Train holds the model");
    Check(ActiveModelBusy(false, true) == "A test is running on the model", "a test holds the model");

    std::string why;
    Check(!ActiveModelUse::TryClaim("Explain", true, false, &why) && why == "Train is running on the model",
          "no claim during Train, and why");
    why.clear();
    Check(!ActiveModelUse::TryClaim("Explain", false, true, &why) && why == "A test is running on the model",
          "no claim during a test, and why");
    Check(ActiveModelBusy(false, false).empty(), "a refused claim leaves the model free");

    {
        auto explain = ActiveModelUse::TryClaim("Explain", false, false, &why);
        Check(explain != nullptr, "Explain claims a free model");
        Check(ActiveModelBusy(false, false) == "Explain is running on the model", "Test sees Explain");
        why.clear();
        Check(!ActiveModelUse::TryClaim("Feature Importance", false, false, &why) &&
                  why == "Explain is running on the model",
              "Feature Importance waits for Explain");
        auto copy = explain;  // A job's state holds a copy.
        explain.reset();
        Check(!ActiveModelBusy(false, false).empty(), "the claim lasts while a copy is held");
        copy.reset();
        Check(ActiveModelBusy(false, false).empty(), "the last copy released frees the model");
    }

    {
        auto unnamed = ActiveModelUse::TryClaim("", false, false, &why);
        Check(unnamed && ActiveModelBusy(false, false) == "A job is running on the model",
              "an unnamed claim reads as a job");
    }
    Check(ActiveModelBusy(false, false).empty(), "the unnamed claim ended with its scope");

    // Many threads racing for a free model: exactly one wins each round.
    for (int round = 0; round < 50; ++round) {
        std::vector<std::shared_ptr<ActiveModelUse>> held(8);
        std::vector<std::thread> threads;
        for (size_t i = 0; i < held.size(); ++i)
            threads.emplace_back([&held, i] { held[i] = ActiveModelUse::TryClaim("Job", false, false, nullptr); });
        for (auto& thread : threads) thread.join();
        int winners = 0;
        for (const auto& claim : held) winners += claim ? 1 : 0;
        if (winners != 1) Check(false, "round " + std::to_string(round) + ": " + std::to_string(winners) + " claims");
        held.clear();
        if (!ActiveModelBusy(false, false).empty()) Check(false, "round " + std::to_string(round) + " left a claim");
    }
    ++checks;

    std::cout << "active model use: " << checks << " checks passed\n";
    return 0;
}
