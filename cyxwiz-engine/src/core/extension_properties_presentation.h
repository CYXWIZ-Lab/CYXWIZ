#pragma once

// Properties of an installed extension node (TOFIX125 P2 step 2.8, approved
// mockup row 5, "Properties - Snake" and "Properties - failed check"): the
// typed settings the node declares, and for a trainable node the gradient
// check. Pure data in, data out: the renderer runs the check and draws.

#include "extension_node_contract.h"

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class ExtensionSettingEditor { Number, WholeNumber, Toggle, Choice, Text };

struct ExtensionSettingRow {
    std::string key;
    std::string label;
    std::string value;        // the node's value, else the declared default
    ExtensionSettingEditor editor = ExtensionSettingEditor::Text;
    std::vector<std::string> choices;
    std::string description;  // the author's words
    std::string type_note;    // "Number. Default 1.0"
    std::string note;         // the line under the field: description, then type_note
    std::string error;        // the plugin's refusal that names this setting
};

// One run of the gradient check, kept by the renderer per node.
struct ExtensionGradientRecord {
    bool passed = false;
    std::string summary;     // DescribeGradientCheck
    std::string error;       // set when the module could not be checked
    double tolerance = 0.0;
    double max_relative_error = 0.0;
    std::string worst;       // "input", "parameter a"
    double absolute_floor = 0.0;
    std::vector<std::vector<size_t>> input_shapes;
    std::map<std::string, std::string> parameters;  // the settings it ran with
};

// NeedsApproval: a Python bundle node whose code is not approved yet; the
// card's button then opens the approval dialog (TOFIX125 P4 step 4.5).
enum class ExtensionGradientState { NotChecked, Checking, Passed, Failed, CouldNotCheck, Stale, NeedsApproval };

struct ExtensionGradientView {
    bool shown = false;  // trainable nodes only
    ExtensionGradientState state = ExtensionGradientState::NotChecked;
    std::string status_line;
    std::string next_step;     // failures: what it means and what to do
    std::string button_label;  // "Check gradient", "Check again"
    bool button_enabled = true;
    std::string button_disabled_reason;
    std::vector<std::pair<std::string, std::string>> details;
};

struct ExtensionPropertiesInputs {
    const ExtensionNodeDescriptor* descriptor = nullptr;
    std::string provider_name;     // the plugin's display name, else its id
    std::string provider_version;  // "1.0.0", may be empty
    const std::map<std::string, std::string>* parameters = nullptr;
    std::string shape_error;       // the factory's refusal for these settings
    const ExtensionGradientRecord* last_check = nullptr;
    bool checking = false;
    // A node from a Python bundle: "Python node" in the subtitle, "the node's
    // code" and "the bundle's author" where a plugin's say "the plugin".
    bool python_bundle = false;
    bool needs_approval = false;  // its bundle's code is not approved yet
};

struct ExtensionPropertiesView {
    std::string subtitle;          // "Example Activation Nodes 1.0.0 · Activations"
    std::vector<ExtensionSettingRow> settings;
    std::string no_settings_text;  // "This node has no settings." when none
    std::string general_error;     // a refusal that names no setting
    ExtensionGradientView gradient;
    std::string bundled_check;     // Python: "When bundled: gradient passed, largest error 1.3e-03 (2026-10-17)."
};

ExtensionPropertiesView BuildExtensionPropertiesView(const ExtensionPropertiesInputs& inputs);

// The settings with their values, defaults filled in (what a check runs with).
std::map<std::string, std::string> ExtensionSettingValues(
    const ExtensionNodeDescriptor& descriptor, const std::map<std::string, std::string>& parameters);

// "9.0e-04"
std::string FormatGradientError(double value);

}  // namespace cyxwiz
