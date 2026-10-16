// Properties of an installed extension node (TOFIX125 P2 step 2.8), on the
// example plugin's real nodes (plugins/examples/activation_nodes): Snake with
// its one setting, Softsign with none, and the gradient check results
// measured by test_plugin_tensor_nodes and test_extension_gradient_check.
#include "../src/core/extension_properties_presentation.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Equal(const std::string& actual, const std::string& expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message << "\n  expected: " << expected << "\n  actual:   " << actual << '\n';
        std::exit(1);
    }
}

constexpr const char* kProvider = "com.cyxwiz.examples.activation-nodes";

cyxwiz::ExtensionNodeDescriptor Node(const char* type_name, cyxwiz::ExtensionNodeKind kind) {
    cyxwiz::ExtensionNodeDescriptor d;
    d.provider_id = kProvider;
    d.type_name = type_name;
    d.type_id = cyxwiz::MakeExtensionTypeId(kProvider, type_name);
    d.version = "1.0.0";
    d.kind = kind;
    d.menu_category = "Activations";
    d.metadata.name = type_name;
    return d;
}

cyxwiz::ExtensionNodeDescriptor Snake() {
    auto d = Node("Snake", cyxwiz::ExtensionNodeKind::TensorLayer);
    d.metadata.parameters.emplace_back("initial_a", "float", "1.0",
                                       "Starting value of the frequency a (greater than 0)");
    return d;
}

cyxwiz::ExtensionPropertiesInputs Inputs(const cyxwiz::ExtensionNodeDescriptor& d,
                                         const std::map<std::string, std::string>& parameters) {
    cyxwiz::ExtensionPropertiesInputs in;
    in.descriptor = &d;
    in.provider_name = "Example Activation Nodes";
    in.provider_version = "1.0.0";
    in.parameters = &parameters;
    return in;
}

cyxwiz::ExtensionGradientRecord Record(bool passed, double error, const char* summary,
                                       std::map<std::string, std::string> parameters) {
    cyxwiz::ExtensionGradientRecord r;
    r.passed = passed;
    r.max_relative_error = error;
    r.tolerance = 2e-2;
    r.worst = "input";
    r.absolute_floor = 5e-2;
    r.summary = summary;
    r.input_shapes = {{2, 3}, {3, 4, 5}};
    r.parameters = std::move(parameters);
    return r;
}

}  // namespace

int main() {
    const auto snake = Snake();
    const std::map<std::string, std::string> saved = {{"initial_a", "1.0"}};

    // Snake, never checked: header, one typed setting, the check offered.
    {
        const auto view = cyxwiz::BuildExtensionPropertiesView(Inputs(snake, saved));
        Equal(view.subtitle, "Example Activation Nodes 1.0.0 \xC2\xB7 Activations", "subtitle");
        Check(view.settings.size() == 1, "one setting");
        const auto& row = view.settings[0];
        Equal(row.key, "initial_a", "setting key");
        Equal(row.value, "1.0", "setting value");
        Check(row.editor == cyxwiz::ExtensionSettingEditor::Number, "float edits as a number");
        Equal(row.type_note, "Number. Default 1.0", "type note");
        Equal(row.description, "Starting value of the frequency a (greater than 0)", "description");
        Equal(row.note, "Starting value of the frequency a (greater than 0). Number. Default 1.0", "note");
        Check(row.error.empty() && view.general_error.empty(), "no refusal");
        Check(view.no_settings_text.empty(), "has settings");
        Check(view.gradient.shown, "trainable node shows the check");
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::NotChecked, "not checked yet");
        Equal(view.gradient.button_label, "Check gradient", "first button label");
        Check(view.gradient.button_enabled, "button enabled");
        Check(view.gradient.details.empty(), "no details before a check");
    }

    // A value the node has not saved yet falls back to the declared default.
    {
        const std::map<std::string, std::string> none;
        const auto view = cyxwiz::BuildExtensionPropertiesView(Inputs(snake, none));
        Equal(view.settings[0].value, "1.0", "default fills a missing value");
        const auto values = cyxwiz::ExtensionSettingValues(snake, none);
        Check(values.size() == 1 && values.at("initial_a") == "1.0", "check runs with the default");
    }

    // While the check runs.
    {
        auto in = Inputs(snake, saved);
        in.checking = true;
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::Checking, "checking");
        Equal(view.gradient.button_label, "Checking...", "busy label");
        Check(!view.gradient.button_enabled, "busy button disabled");
    }

    // Passed (Snake measured 9.0e-04 in input).
    {
        const auto record = Record(true, 9.0e-4,
                                   "Passed: largest relative error 9.0e-04 in input (tolerance 2.0e-02)", saved);
        auto in = Inputs(snake, saved);
        in.last_check = &record;
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::Passed, "passed");
        Equal(view.gradient.status_line,
              "Passed: largest relative error 9.0e-04 in input (tolerance 2.0e-02).", "passed line");
        Equal(view.gradient.button_label, "Check again", "again label");
        Check(view.gradient.next_step.empty(), "nothing to do after a pass");
        const auto& d = view.gradient.details;
        Check(d.size() == 5, "five detail rows");
        Equal(d[0].second, "9.0e-04", "largest error");
        Equal(d[1].second, "input", "where");
        Equal(d[2].second, "2.0e-02", "tolerance");
        Equal(d[3].second, "initial_a = 1.0; inputs [2, 3] and [3, 4, 5]", "checked with");
        Equal(d[4].second, "Central differences, 32-bit, floor 0.05", "method");
    }

    // The settings changed after the check: the result no longer applies.
    {
        const auto record = Record(true, 9.0e-4, "Passed: x", saved);
        const std::map<std::string, std::string> changed = {{"initial_a", "2.0"}};
        auto in = Inputs(snake, changed);
        in.last_check = &record;
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::Stale, "stale after a change");
        Equal(view.gradient.button_label, "Check again", "stale label");
    }

    // Failed (the checker's case: Softsign missing its square, 5.0e-01).
    {
        const auto record = Record(false, 5.0e-1,
                                   "Failed: largest relative error 5.0e-01 in input (tolerance 2.0e-02)", saved);
        auto in = Inputs(snake, saved);
        in.last_check = &record;
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::Failed, "failed");
        Check(view.gradient.next_step.find("Tell the plugin's author") != std::string::npos, "failure next step");
        Equal(view.gradient.details[0].second, "5.0e-01", "failed largest error");
    }

    // The module could not run at all.
    {
        auto record = Record(false, 0.0, "Could not check: initial_a must be greater than 0", saved);
        record.error = "initial_a must be greater than 0";
        auto in = Inputs(snake, saved);
        in.last_check = &record;
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::CouldNotCheck, "could not check");
        Equal(view.gradient.details[0].first, "Reason", "reason row");
        Check(!view.gradient.next_step.empty(), "could-not-check next step");
    }

    // The plugin's refusal lands under the setting it names.
    {
        const std::map<std::string, std::string> bad = {{"initial_a", "-1"}};
        auto in = Inputs(snake, bad);
        in.shape_error = "initial_a must be greater than 0";
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Equal(view.settings[0].error, "initial_a must be greater than 0", "refusal under the field");
        Check(view.general_error.empty(), "not repeated above");
        // The check cannot run on refused settings, even after a pass.
        Check(view.gradient.state == cyxwiz::ExtensionGradientState::CouldNotCheck, "refused: could not check");
        Equal(view.gradient.status_line, "Could not check: initial_a must be greater than 0", "refused line");
        Check(!view.gradient.button_enabled && !view.gradient.button_disabled_reason.empty(),
              "refused: button off with a reason");
        const auto passed = Record(true, 9.0e-4, "Passed: x", bad);
        in.last_check = &passed;
        Check(cyxwiz::BuildExtensionPropertiesView(in).gradient.state ==
                  cyxwiz::ExtensionGradientState::CouldNotCheck,
              "a pass on refused settings is not shown as verified");
    }
    // A refusal that names no setting is shown once, above.
    {
        auto in = Inputs(snake, saved);
        in.shape_error = "Snake needs at least one feature";
        const auto view = cyxwiz::BuildExtensionPropertiesView(in);
        Check(view.settings[0].error.empty(), "no field error");
        Equal(view.general_error, "Snake needs at least one feature", "general refusal");
    }

    // Softsign: no settings.
    {
        const auto softsign = Node("Softsign", cyxwiz::ExtensionNodeKind::TensorLayer);
        const std::map<std::string, std::string> none = {{"_meta_note", "plugin fact"}};
        const auto view = cyxwiz::BuildExtensionPropertiesView(Inputs(softsign, none));
        Check(view.settings.empty(), "Softsign has no settings (_meta_ keys are not settings)");
        Equal(view.no_settings_text, "This node has no settings.", "no settings text");
        Check(view.gradient.shown, "Softsign is checkable");
    }

    // A saved value the node does not declare stays visible as text.
    {
        const auto softsign = Node("Softsign", cyxwiz::ExtensionNodeKind::TensorLayer);
        const auto view = cyxwiz::BuildExtensionPropertiesView(Inputs(softsign, saved));
        Check(view.settings.size() == 1 && view.settings[0].key == "initial_a", "undeclared value kept");
        Check(view.settings[0].editor == cyxwiz::ExtensionSettingEditor::Text, "undeclared value edits as text");
        Check(view.no_settings_text.empty(), "not 'no settings' when a value is shown");
    }

    // A simulation node (MuJoCo) has no gradient check.
    {
        auto plant = Node("RewardFunction", cyxwiz::ExtensionNodeKind::Signal);
        plant.metadata.parameters.emplace_back("mode", "enum", "dense", "", std::vector<std::string>{"dense", "sparse"});
        const std::map<std::string, std::string> none;
        const auto view = cyxwiz::BuildExtensionPropertiesView(Inputs(plant, none));
        Check(!view.gradient.shown, "signal node: no check");
        Check(view.settings[0].editor == cyxwiz::ExtensionSettingEditor::Choice, "enum edits as a choice");
        Equal(view.settings[0].type_note, "Choice. Default dense", "choice note");
    }

    std::cout << "test_extension_properties_presentation: all checks passed\n";
    return 0;
}
