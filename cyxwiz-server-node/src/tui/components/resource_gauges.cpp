// resource_gauges.cpp - TUI resource gauge implementations
#include "tui/components/resource_gauges.h"
#include <ftxui/dom/elements.hpp>

namespace cyxwiz::servernode::tui {

using namespace ftxui;

Element CreateCPUGauge(float usage) {
    Color c = usage > 0.9f ? Color::Red : (usage > 0.7f ? Color::Yellow : Color::Blue);
    return hbox({
        text("CPU") | size(WIDTH, EQUAL, 5),
        gauge(usage) | flex | color(c),
        text(" " + std::to_string(static_cast<int>(usage * 100)) + "%") | size(WIDTH, EQUAL, 6),
    });
}

Element CreateGPUGauge(float usage) {
    Color c = usage > 0.9f ? Color::Red : (usage > 0.7f ? Color::Yellow : Color::Green);
    return hbox({
        text("GPU") | size(WIDTH, EQUAL, 5),
        gauge(usage) | flex | color(c),
        text(" " + std::to_string(static_cast<int>(usage * 100)) + "%") | size(WIDTH, EQUAL, 6),
    });
}

Element CreateRAMGauge(float usage) {
    Color c = usage > 0.9f ? Color::Red : (usage > 0.7f ? Color::Yellow : Color::Magenta);
    return hbox({
        text("RAM") | size(WIDTH, EQUAL, 5),
        gauge(usage) | flex | color(c),
        text(" " + std::to_string(static_cast<int>(usage * 100)) + "%") | size(WIDTH, EQUAL, 6),
    });
}

Element CreateVRAMGauge(float usage) {
    Color c = usage > 0.9f ? Color::Red : (usage > 0.7f ? Color::Yellow : Color::Cyan);
    return hbox({
        text("VRAM") | size(WIDTH, EQUAL, 5),
        gauge(usage) | flex | color(c),
        text(" " + std::to_string(static_cast<int>(usage * 100)) + "%") | size(WIDTH, EQUAL, 6),
    });
}

} // namespace cyxwiz::servernode::tui
