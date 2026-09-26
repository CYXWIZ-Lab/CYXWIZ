#pragma once

#include "graph_model.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>

namespace cyxwiz {
namespace recurrent_configuration_policy_detail {

inline std::string TrimLower(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    value = first < last ? std::string(first, last) : std::string{};
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

} // namespace recurrent_configuration_policy_detail

// Parameter-dependent gaps for otherwise supported recurrent model layers.
// Header-only by design: focused model tests compile ModelBuilder directly and
// must receive the same fail-closed policy without linking the full Engine
// runtime-capability catalog.
inline std::optional<std::string>
ResolvePipelineUnsupportedSequentialModelConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    if (node_type != gui::NodeType::LSTM &&
        node_type != gui::NodeType::GRU &&
        node_type != gui::NodeType::RNN) {
        return std::nullopt;
    }

    const char* layer_name = node_type == gui::NodeType::LSTM
        ? "LSTM"
        : (node_type == gui::NodeType::GRU ? "GRU" : "RNN");
    // Bidirectional RNN/LSTM/GRU training all run as split forward/reverse
    // branches (RNNModule/LSTMModule/GRUModule), each a proven
    // single-direction layer, so no directionality fails closed here.

    if (node_type == gui::NodeType::RNN) {
        const auto nonlinearity = parameters.find("nonlinearity");
        if (nonlinearity != parameters.end()) {
            const std::string value =
                recurrent_configuration_policy_detail::TrimLower(
                    nonlinearity->second);
            if (!value.empty() && value != "tanh" && value != "relu") {
                return std::string(
                    "RNN nonlinearity must be \"tanh\" or \"relu\"; the simple "
                    "RNN layer implements no other cell activation.");
            }
        }
    }

    const auto dropout = parameters.find("dropout");
    if (dropout == parameters.end()) {
        return std::nullopt;
    }
    const std::string dropout_text =
        recurrent_configuration_policy_detail::TrimLower(dropout->second);
    if (dropout_text.empty()) return std::nullopt;

    errno = 0;
    char* end = nullptr;
    const double dropout_value = std::strtod(dropout_text.c_str(), &end);
    if (errno == ERANGE || end != dropout_text.c_str() + dropout_text.size() ||
        !std::isfinite(dropout_value)) {
        return std::string(layer_name) +
            " dropout must be a finite number. Engine recurrent training "
            "currently supports only dropout=0.0.";
    }
    if (dropout_value != 0.0) {
        return std::string(layer_name) +
            " dropout is not wired through the Engine sequential module. "
            "Use dropout=0.0 and an explicit Dropout node instead.";
    }

    return std::nullopt;
}

} // namespace cyxwiz
