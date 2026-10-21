#pragma once

#include <string>

namespace cyxwiz::route_probe {

using DropoutStageReporter =
    void (*)(const std::string& operation, const char* stage);

void RunDropoutForwardBackwardContract(
    const std::string& operation,
    DropoutStageReporter report_stage);

} // namespace cyxwiz::route_probe
