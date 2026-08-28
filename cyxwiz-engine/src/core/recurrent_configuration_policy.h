#pragma once

#include "../gui/node_editor.h"

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
        node_type != gui::NodeType::GRU) {
        return std::nullopt;
    }

    const char* layer_name = node_type == gui::NodeType::LSTM ? "LSTM" : "GRU";
    const auto bidirectional = parameters.find("bidirectional");
    if (node_type == gui::NodeType::LSTM &&
        bidirectional != parameters.end()) {
        const std::string value =
            recurrent_configuration_policy_detail::TrimLower(
                bidirectional->second);
        if (value == "true" || value == "1" || value == "yes" ||
            value == "on") {
            return std::string(
                "LSTM bidirectional=true is not supported for Engine training "
                "because reverse-direction backward gradients are not implemented. "
                "Use bidirectional=false until bidirectional backward parity is proven.");
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
