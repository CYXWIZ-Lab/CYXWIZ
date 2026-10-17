// Properties of an installed extension node (TOFIX125 P2 step 2.8, approved
// mockup row 5): the plugin and version, typed settings with the author's
// words and the plugin's refusals, and for a trainable node the gradient
// check. Wording comes from core/extension_properties_presentation.

#include "properties_extension_node.h"

#include "node_editor.h"
#include "properties_rows.h"
#include <cyxwiz/gradient_check.h>
#include "../core/extension_node_registry.h"
#include "../core/extension_properties_presentation.h"
#include "../plugin/plugin_manager.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace gui::properties_node_editors {

namespace {

// The last gradient check per node id, and the node whose check runs on the
// next frame (one frame shows "Checking..." first).
struct GradientState {
    std::map<int, cyxwiz::ExtensionGradientRecord> records;
    std::set<int> pending;
    std::set<int> details_open;
};

GradientState& Gradients() {
    static GradientState state;
    return state;
}

// The plugin's refusal for these settings, on the shape the check uses.
std::string ShapeError(const std::string& type_id, const std::map<std::string, std::string>& values) {
    const auto factory = cyxwiz::ExtensionNodeRegistry::Instance().TensorFactory(type_id);
    if (!factory) return {};
    const auto result = factory->InferOutputShape({3}, values);  // one sample of the check's [2, 3]
    return result.ok ? std::string() : result.error;
}

cyxwiz::ExtensionGradientRecord RunGradientCheck(const cyxwiz::ExtensionNodeDescriptor& descriptor,
                                                 const std::map<std::string, std::string>& values) {
    cyxwiz::GradientCheckOptions options;
    cyxwiz::ExtensionGradientRecord record;
    record.parameters = values;
    record.input_shapes = options.input_shapes;
    record.absolute_floor = options.absolute_floor;

    const auto factory = cyxwiz::ExtensionNodeRegistry::Instance().TensorFactory(descriptor.type_id);
    if (!factory) {
        record.error = "the plugin no longer provides this node";
        record.summary = "Could not check: " + record.error;
        return record;
    }
    const auto result = cyxwiz::CheckModuleGradients(
        [&](const std::vector<size_t>& shape) {
            // The checker's shapes include the batch; a factory is given the
            // shape of one sample, as the model builder gives it.
            const std::vector<size_t> sample(shape.begin() + (shape.empty() ? 0 : 1), shape.end());
            std::string error;
            auto module = factory->CreateModule(sample, values, error);
            if (!module) throw std::runtime_error(error.empty() ? "the module could not be created" : error);
            return module;
        },
        options);
    record.passed = result.passed;
    record.summary = cyxwiz::DescribeGradientCheck(result);
    record.error = result.error;
    record.tolerance = result.tolerance;
    record.max_relative_error = result.max_relative_error;
    record.worst = result.worst;
    return record;
}

cyxwiz::ui::StatusStyle GradientStatus(cyxwiz::ExtensionGradientState state) {
    using cyxwiz::ui::Status;
    switch (state) {
        case cyxwiz::ExtensionGradientState::Checking: {
            auto s = cyxwiz::ui::StatusStyleFor(Status::Verifying);
            s.label = "Checking...";
            return s;
        }
        case cyxwiz::ExtensionGradientState::Passed: return cyxwiz::ui::StatusStyleFor(Status::Verified);
        case cyxwiz::ExtensionGradientState::Failed: return cyxwiz::ui::StatusStyleFor(Status::Failed);
        case cyxwiz::ExtensionGradientState::CouldNotCheck: {
            auto s = cyxwiz::ui::StatusStyleFor(Status::Failed);
            s.label = "Could not check";
            return s;
        }
        case cyxwiz::ExtensionGradientState::NotChecked:
        case cyxwiz::ExtensionGradientState::Stale: break;
    }
    return cyxwiz::ui::StatusStyleFor(Status::NotVerifiedYet);
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// One setting's editor. Returns true when the value changed.
bool SettingEditor(const cyxwiz::ExtensionSettingRow& row, std::string& value) {
    ImGui::SetNextItemWidth(-1.0f);
    switch (row.editor) {
        case cyxwiz::ExtensionSettingEditor::Number: {
            char buf[64];
            std::strncpy(buf, value.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText("##v", buf, sizeof(buf),
                                 ImGuiInputTextFlags_CharsScientific | ImGuiInputTextFlags_EnterReturnsTrue)) {
                value = buf;
                return true;
            }
            return false;
        }
        case cyxwiz::ExtensionSettingEditor::WholeNumber: {
            int number = 0;
            try { number = std::stoi(value); } catch (...) {}
            if (ImGui::InputInt("##v", &number)) {
                value = std::to_string(number);
                return true;
            }
            return false;
        }
        case cyxwiz::ExtensionSettingEditor::Toggle: {
            bool on = value == "true" || value == "1";
            if (ImGui::Checkbox("##v", &on)) {
                value = on ? "true" : "false";
                return true;
            }
            return false;
        }
        case cyxwiz::ExtensionSettingEditor::Choice: {
            bool changed = false;
            if (ImGui::BeginCombo("##v", value.c_str())) {
                for (const auto& choice : row.choices) {
                    if (ImGui::Selectable(choice.c_str(), choice == value)) {
                        changed = choice != value;
                        value = choice;
                    }
                }
                ImGui::EndCombo();
            }
            return changed;
        }
        case cyxwiz::ExtensionSettingEditor::Text: break;
    }
    char buf[512];
    std::strncpy(buf, value.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    if (ImGui::InputText("##v", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
        value = buf;
        return true;
    }
    return false;
}

void RenderGradientCard(const MLNode& node, const cyxwiz::ExtensionGradientView& gradient) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    auto& state = Gradients();
    cyxwiz::ui::BeginCard("gradient_check");
    const auto status = GradientStatus(gradient.state);
    cyxwiz::ui::CardHeader("Gradient check", nullptr, &status);
    Wrapped(t.text, gradient.status_line);
    Wrapped(t.text_dim, gradient.next_step);
    ImGui::Spacing();
    if (cyxwiz::ui::SecondaryButton(gradient.button_label.c_str(), gradient.button_enabled,
                                    gradient.button_disabled_reason.empty() ? "The check is running"
                                                                            : gradient.button_disabled_reason.c_str())) {
        state.pending.insert(node.id);
    }
    if (!gradient.details.empty()) {
        const bool open = state.details_open.count(node.id) > 0;
        ImGui::SameLine(0.0f, t.space_md);
        if (cyxwiz::ui::LinkButton(open ? "Hide##gradient" : "Details##gradient")) {
            if (open) state.details_open.erase(node.id); else state.details_open.insert(node.id);
        }
        if (open) {
            std::vector<cyxwiz::ui::KeyValue> rows;
            for (const auto& [key, value] : gradient.details) rows.push_back({key, value});
            ImGui::Spacing();
            cyxwiz::ui::KeyValueTable("##gradient_details", rows, 1);
        }
    }
    cyxwiz::ui::EndCard();
}

}  // namespace

void RenderInstalledExtensionProperties(MLNode& node, const cyxwiz::ExtensionNodeDescriptor& descriptor,
                                        RenderNodePropertiesContext& context) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    auto& gradients = Gradients();

    // A check asked for last frame runs now, after "Checking..." was shown.
    const auto values = cyxwiz::ExtensionSettingValues(descriptor, node.parameters);
    if (gradients.pending.count(node.id) > 0 && descriptor.kind == cyxwiz::ExtensionNodeKind::TensorLayer) {
        static int shown_checking_for = -1;
        if (shown_checking_for == node.id) {
            gradients.records[node.id] = RunGradientCheck(descriptor, values);
            gradients.pending.erase(node.id);
            shown_checking_for = -1;
        } else {
            shown_checking_for = node.id;
        }
    }

    cyxwiz::ExtensionPropertiesInputs in;
    in.descriptor = &descriptor;
    if (const auto* plugin = cyxwiz::plugin::PluginManager::Instance().GetLoadedPlugin(descriptor.provider_id)) {
        in.provider_name = plugin->manifest.name;
        in.provider_version = plugin->manifest.version.ToString();
    }
    in.parameters = &node.parameters;
    if (descriptor.kind == cyxwiz::ExtensionNodeKind::TensorLayer) in.shape_error = ShapeError(descriptor.type_id, values);
    const auto record = gradients.records.find(node.id);
    in.last_check = record != gradients.records.end() ? &record->second : nullptr;
    in.checking = gradients.pending.count(node.id) > 0;
    const auto view = cyxwiz::BuildExtensionPropertiesView(in);

    Wrapped(t.text_dim, view.subtitle);
    ImGui::Spacing();

    if (!view.no_settings_text.empty()) Wrapped(t.text_dim, view.no_settings_text);
    Wrapped(t.caution, view.general_error);
    if (!view.settings.empty()) {
        properties_rows::Rows rows("##extension_settings");
        if (rows.ok) {
            for (const auto& row : view.settings) {
                properties_rows::Label(row.label.c_str(), false, row.description.empty() ? nullptr : row.description.c_str());
                ImGui::PushID(row.key.c_str());
                std::string value = row.value;
                if (SettingEditor(row, value)) {
                    node.parameters[row.key] = value;
                    if (node.has_dynamic_pins && row.key == node.dynamic_pin_trigger && context.node_editor) {
                        context.node_editor->ResolveDynamicPins(node.id);
                    }
                    context.invalidate_shapes();
                }
                ImGui::PopID();
                properties_rows::Status(row.key.c_str());
                if (!row.error.empty()) {
                    // The refusal on its own row under the field (Note is muted).
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(1);
                    Wrapped(t.caution, row.error);
                }
                properties_rows::Note(row.note.c_str());
            }
        }
    }

    if (view.gradient.shown) {
        ImGui::Spacing();
        RenderGradientCard(node, view.gradient);
    }
}

}  // namespace gui::properties_node_editors
