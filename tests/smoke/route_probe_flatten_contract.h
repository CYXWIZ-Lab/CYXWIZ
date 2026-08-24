#pragma once

#include <string>

namespace cyxwiz::route_probe {

using StageReporter = void (*)(const std::string& operation, const char* stage);

void RunFlattenForwardBackwardContract(
    const std::string& operation,
    StageReporter report_stage);

} // namespace cyxwiz::route_probe
