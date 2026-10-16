#include "extension_properties_presentation.h"

#include <cstdio>

namespace cyxwiz {

namespace {

ExtensionSettingEditor EditorFor(const std::string& type) {
    if (type == "float" || type == "double" || type == "number") return ExtensionSettingEditor::Number;
    if (type == "int" || type == "integer") return ExtensionSettingEditor::WholeNumber;
    if (type == "bool" || type == "boolean") return ExtensionSettingEditor::Toggle;
    if (type == "enum" || type == "choice") return ExtensionSettingEditor::Choice;
    return ExtensionSettingEditor::Text;
}

const char* EditorWord(ExtensionSettingEditor editor) {
    switch (editor) {
        case ExtensionSettingEditor::Number: return "Number";
        case ExtensionSettingEditor::WholeNumber: return "Whole number";
        case ExtensionSettingEditor::Toggle: return "On or off";
        case ExtensionSettingEditor::Choice: return "Choice";
        case ExtensionSettingEditor::Text: return "Text";
    }
    return "Text";
}

std::string Shape(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

std::string Shapes(const std::vector<std::vector<size_t>>& shapes) {
    std::string text;
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i > 0) text += i + 1 == shapes.size() ? " and " : ", ";
        text += Shape(shapes[i]);
    }
    return text;
}

std::string CheckedWith(const ExtensionGradientRecord& record) {
    std::string text;
    for (const auto& [key, value] : record.parameters) {
        if (!text.empty()) text += ", ";
        text += key + " = " + value;
    }
    const std::string shapes = Shapes(record.input_shapes);
    if (!shapes.empty()) {
        if (!text.empty()) text += "; ";
        text += (record.input_shapes.size() == 1 ? "input " : "inputs ") + shapes;
    }
    return text;
}

std::string Method(double floor) {
    char text[96];
    std::snprintf(text, sizeof(text), "Central differences, 32-bit, floor %.2g", floor);
    return text;
}

ExtensionGradientView BuildGradient(const ExtensionPropertiesInputs& in,
                                    const std::map<std::string, std::string>& values) {
    ExtensionGradientView view;
    view.shown = in.descriptor->kind == ExtensionNodeKind::TensorLayer;
    if (!view.shown) return view;

    const ExtensionGradientRecord* record = in.last_check;
    // The plugin refuses these settings: the node cannot be built, so there
    // is nothing to check (an old result would describe other settings).
    if (!in.shape_error.empty()) {
        view.state = ExtensionGradientState::CouldNotCheck;
        view.status_line = "Could not check: " + in.shape_error;
        view.next_step = "Fix the setting the plugin refused, then check the gradient.";
        view.button_label = record ? "Check again" : "Check gradient";
        view.button_enabled = false;
        view.button_disabled_reason = "The plugin refuses the current settings";
        return view;
    }
    if (in.checking) {
        view.state = ExtensionGradientState::Checking;
        view.status_line = "Running the node on small inputs and comparing gradients.";
        view.button_label = "Checking...";
        view.button_enabled = false;
        return view;
    }
    if (!record) {
        view.state = ExtensionGradientState::NotChecked;
        view.status_line = "Checks the node's hand-written gradient against a numerical one before you train.";
        view.button_label = "Check gradient";
        return view;
    }

    view.button_label = "Check again";
    if (!record->error.empty()) {
        view.state = ExtensionGradientState::CouldNotCheck;
        view.status_line = record->summary.empty() ? "Could not check: " + record->error : record->summary;
        view.next_step = "The node could not be run on the check's inputs. Fix the settings, or tell the plugin's author.";
    } else if (record->parameters != values) {
        view.state = ExtensionGradientState::Stale;
        view.status_line = "Settings changed since the last check. Check again before you train.";
    } else if (record->passed) {
        view.state = ExtensionGradientState::Passed;
        view.status_line = record->summary + ".";
    } else {
        view.state = ExtensionGradientState::Failed;
        view.status_line = record->summary + ".";
        view.next_step =
            "The node's backward step does not match what its forward step computes. Training can still run, "
            "but the node may learn wrongly. Tell the plugin's author.";
    }

    if (record->error.empty()) {
        view.details.emplace_back("Largest error", FormatGradientError(record->max_relative_error));
        if (!record->worst.empty()) view.details.emplace_back("Where", record->worst);
        view.details.emplace_back("Tolerance", FormatGradientError(record->tolerance));
    } else {
        view.details.emplace_back("Reason", record->error);
    }
    const std::string with = CheckedWith(*record);
    if (!with.empty()) view.details.emplace_back("Checked with", with);
    view.details.emplace_back("Method", Method(record->absolute_floor));
    return view;
}

}  // namespace

std::map<std::string, std::string> ExtensionSettingValues(
    const ExtensionNodeDescriptor& descriptor, const std::map<std::string, std::string>& parameters) {
    std::map<std::string, std::string> values;
    for (const auto& definition : descriptor.metadata.parameters) {
        const auto it = parameters.find(definition.name);
        values[definition.name] = it != parameters.end() ? it->second : definition.default_value;
    }
    return values;
}

std::string FormatGradientError(double value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1e", value);
    return text;
}

ExtensionPropertiesView BuildExtensionPropertiesView(const ExtensionPropertiesInputs& in) {
    ExtensionPropertiesView view;
    if (!in.descriptor) return view;
    const ExtensionNodeDescriptor& d = *in.descriptor;
    static const std::map<std::string, std::string> kNone;
    const auto& parameters = in.parameters ? *in.parameters : kNone;

    view.subtitle = in.provider_name.empty() ? d.provider_id : in.provider_name;
    if (!in.provider_version.empty()) view.subtitle += " " + in.provider_version;
    if (!d.menu_category.empty()) view.subtitle += " \xC2\xB7 " + d.menu_category;

    const auto values = ExtensionSettingValues(d, parameters);
    bool error_placed = in.shape_error.empty();
    for (const auto& definition : d.metadata.parameters) {
        ExtensionSettingRow row;
        row.key = definition.name;
        row.label = definition.display_name.empty() ? definition.name : definition.display_name;
        row.value = values.at(definition.name);
        row.editor = EditorFor(definition.type);
        row.choices = definition.enum_values;
        if (row.editor == ExtensionSettingEditor::Choice && row.choices.empty()) {
            row.editor = ExtensionSettingEditor::Text;
        }
        row.description = definition.description;
        row.type_note = EditorWord(row.editor);
        if (!definition.default_value.empty()) row.type_note += ". Default " + definition.default_value;
        row.note = row.description.empty() ? row.type_note : row.description + ". " + row.type_note;
        if (!error_placed && in.shape_error.find(definition.name) != std::string::npos) {
            row.error = in.shape_error;
            error_placed = true;
        }
        view.settings.push_back(std::move(row));
    }
    // Saved values the node does not declare stay visible and editable as
    // text ("_meta_" keys are the plugin's own read-only facts).
    for (const auto& [key, value] : parameters) {
        if (values.count(key) > 0 || key.rfind("_meta_", 0) == 0) continue;
        ExtensionSettingRow row;
        row.key = row.label = key;
        row.value = value;
        row.type_note = row.note = "Text";
        view.settings.push_back(std::move(row));
    }
    if (view.settings.empty()) view.no_settings_text = "This node has no settings.";
    if (!error_placed) view.general_error = in.shape_error;

    view.gradient = BuildGradient(in, values);
    return view;
}

}  // namespace cyxwiz
