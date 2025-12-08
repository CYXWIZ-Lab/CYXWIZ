// resource_gauges.h - TUI resource gauge components
#pragma once

#include <ftxui/dom/elements.hpp>

namespace cyxwiz::servernode::tui {

ftxui::Element CreateCPUGauge(float usage);
ftxui::Element CreateGPUGauge(float usage);
ftxui::Element CreateRAMGauge(float usage);
ftxui::Element CreateVRAMGauge(float usage);

} // namespace cyxwiz::servernode::tui
