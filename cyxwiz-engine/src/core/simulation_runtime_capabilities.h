#pragma once

#include "../gui/node_editor.h"

#include <array>

namespace cyxwiz {

enum class SimulationNodeRole {
    ScalarSource,
    TimeVaryingScalarSource,
    ScalarScope,
};

struct SimulationRuntimeCapability {
    gui::NodeType node_type;
    SimulationNodeRole role;
    const char* runtime_name;
};

inline constexpr std::array<SimulationRuntimeCapability, 6>
    kBuiltInSimulationRuntimeCapabilities{{
        {gui::NodeType::Constant, SimulationNodeRole::ScalarSource, "Constant"},
        {gui::NodeType::SignalSlider, SimulationNodeRole::ScalarSource,
         "SignalSlider"},
        {gui::NodeType::SineWave,
         SimulationNodeRole::TimeVaryingScalarSource, "SineWave"},
        {gui::NodeType::StepSignal,
         SimulationNodeRole::TimeVaryingScalarSource, "StepSignal"},
        {gui::NodeType::RampSignal,
         SimulationNodeRole::TimeVaryingScalarSource, "RampSignal"},
        {gui::NodeType::SignalScope, SimulationNodeRole::ScalarScope,
         "SignalScope"},
    }};

inline const SimulationRuntimeCapability* FindSimulationRuntimeCapability(
    gui::NodeType node_type) {
    for (const auto& capability : kBuiltInSimulationRuntimeCapabilities) {
        if (capability.node_type == node_type) {
            return &capability;
        }
    }
    return nullptr;
}

inline bool IsBuiltInSimulationRuntimeNode(gui::NodeType node_type) {
    return FindSimulationRuntimeCapability(node_type) != nullptr;
}

} // namespace cyxwiz
