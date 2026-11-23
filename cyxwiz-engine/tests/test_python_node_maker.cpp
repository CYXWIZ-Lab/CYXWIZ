#include "../src/core/python_node_maker.h"

#include <iostream>
#include <string>

namespace nm = cyxwiz::node_maker;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

bool HasIssue(const nm::NodeForm& form, const std::string& part) {
    for (const auto& issue : nm::Validate(form)) {
        if (Contains(issue, part)) return true;
    }
    return false;
}

// Generate then Read gives the same form back.
void CheckRoundTrip(const nm::NodeForm& form, const std::string& label) {
    const std::string source = nm::Generate(form, "project.catdog." + nm::FileStemFor(form.class_name));
    nm::NodeForm back;
    std::string error;
    Check(nm::Read(source, back, &error), label + ": reads back (" + error + ")");
    Check(back.kind == form.kind, label + ": kind");
    Check(back.display_name == form.display_name, label + ": display name");
    Check(back.class_name == form.class_name, label + ": class name");
    Check(back.version == form.version, label + ": version");
    Check(back.category == form.category, label + ": category");
    Check(back.description == form.description, label + ": description");
    Check(back.shown_as_activation == form.shown_as_activation, label + ": shown as");
    Check(back.computes_shape == form.computes_shape, label + ": output shape");
    Check(back.continuous_targets == form.continuous_targets, label + ": targets");
    Check(back.settings.size() == form.settings.size(), label + ": setting count");
    for (size_t i = 0; i < form.settings.size() && i < back.settings.size(); ++i) {
        const auto& a = form.settings[i];
        const auto& b = back.settings[i];
        Check(a.name == b.name && a.type == b.type && a.default_value == b.default_value && a.lowest == b.lowest &&
                  a.highest == b.highest && a.description == b.description && a.choices == b.choices,
              label + ": setting " + a.name);
    }
    for (const auto& method : nm::MethodNames(form)) {
        Check(back.bodies[method] == form.bodies.at(method), label + ": body of " + method);
    }
}

}  // namespace

int main() {
    // Every starter is valid and writes the decorator pycyxwiz expects.
    for (auto kind : {nm::Kind::Activation, nm::Kind::Layer, nm::Kind::Loss}) {
        const nm::NodeForm form = nm::Starter(kind);
        const auto issues = nm::Validate(form);
        Check(issues.empty(), std::string("the ") + nm::KindLabel(kind) + " starter is valid" +
                                  (issues.empty() ? "" : ": " + issues.front()));
        CheckRoundTrip(form, std::string(nm::KindLabel(kind)) + " starter");
    }
    const std::string act = nm::Generate(nm::Starter(nm::Kind::Activation), "project.catdog.soft_clip");
    Check(Contains(act, "@nodes.activation(\n    id=\"project.catdog.soft_clip\",\n    name=\"SoftClip\","),
          "an activation uses @nodes.activation with its id and name");
    Check(Contains(act, "class SoftClip:\n    def f(self, x):\n        return x / (x.abs() + 1.0)\n"),
          "f is written with its body indented");
    Check(Contains(act, "    def df(self, x):\n        d = x.abs() + 1.0\n        return 1.0 / (d * d)\n"),
          "df keeps both lines");

    const std::string layer = nm::Generate(nm::Starter(nm::Kind::Layer), "project.catdog.swish");
    Check(Contains(layer, "@nodes.node(") && Contains(layer, "    role=\"activation\",\n"),
          "a Layer shown as Activation is @nodes.node with role activation");
    Check(Contains(layer, "class Swish(cyx.Module):\n    beta = nodes.Param(float, default=1.0, min=0.0, "
                          "description=\"How sharp the curve is\")\n"),
          "a setting is a Param line");
    Check(Contains(layer, "    def backward(self, grad_output):\n"), "a Layer's backward takes grad_output");

    const std::string loss = nm::Generate(nm::Starter(nm::Kind::Loss), "project.catdog.log_cosh");
    Check(Contains(loss, "@nodes.loss(") && Contains(loss, "    target=\"continuous\",\n"),
          "a loss on values is @nodes.loss with target continuous");
    Check(Contains(loss, "class LogCosh(cyx.Loss):") &&
              Contains(loss, "    def backward(self, predictions, targets):\n"),
          "a loss's methods take predictions and targets");

    // Every setting type, quotes and a computed shape survive the round trip.
    nm::NodeForm rich = nm::Starter(nm::Kind::Layer);
    rich.display_name = "My \"quoted\" layer";
    rich.description = "Two\nlines with a \\ backslash";
    rich.shown_as_activation = false;
    rich.computes_shape = true;
    rich.bodies["output_shape"] = "return list(input_shape[:-1]) + [self.units]";
    nm::Setting units;
    units.name = "units";
    units.type = nm::SettingType::WholeNumber;
    units.default_value = "16";
    units.lowest = "1";
    units.highest = "4096";
    nm::Setting bias;
    bias.name = "use_bias";
    bias.type = nm::SettingType::OnOff;
    bias.default_value = "true";
    nm::Setting mode;
    mode.name = "mode";
    mode.type = nm::SettingType::Choice;
    mode.choices = {"fast", "exact, slow"};
    mode.default_value = "exact, slow";
    mode.description = "Which formula";
    nm::Setting tag;
    tag.name = "tag";
    tag.type = nm::SettingType::Text;
    tag.default_value = "a, \"b\"";
    rich.settings.push_back(units);
    rich.settings.push_back(bias);
    rich.settings.push_back(mode);
    rich.settings.push_back(tag);
    Check(nm::Validate(rich).empty(), "the rich layer is valid");
    CheckRoundTrip(rich, "rich layer");
    const std::string rich_source = nm::Generate(rich, "project.catdog.my_quoted_layer");
    Check(Contains(rich_source, "units = nodes.Param(int, default=16, min=1, max=4096)"), "a whole number has int bounds");
    Check(Contains(rich_source, "use_bias = nodes.Param(bool, default=True)"), "on / off is True / False");
    Check(Contains(rich_source, "choices=[\"fast\", \"exact, slow\"]"), "choices are a list");
    Check(Contains(rich_source, "    def output_shape(self, input_shape):\n"), "a computed shape writes output_shape");

    // A file the maker did not write is refused with the way out.
    nm::NodeForm untouched = nm::Starter(nm::Kind::Loss);
    std::string error;
    Check(!nm::Read("import pycyxwiz\n", untouched, &error) && Contains(error, "Script Editor"),
          "a hand-written file opens in the Script Editor: " + error);
    Check(untouched.class_name == "LogCosh", "a refused read leaves the form as it was");

    // Validation in plain words.
    nm::NodeForm bad = nm::Starter(nm::Kind::Layer);
    bad.display_name = " ";
    bad.class_name = "2fast";
    bad.version = "1.0";
    bad.bodies["backward"] = "  ";
    Check(HasIssue(bad, "name for the Nodes panel"), "a blank name is refused");
    Check(HasIssue(bad, "class name starts with a letter"), "a class name starting with a digit is refused");
    Check(HasIssue(bad, "version looks like 1.0.0"), "a two-part version is refused");
    Check(HasIssue(bad, "body of backward"), "an empty body is refused");
    nm::NodeForm settings = nm::Starter(nm::Kind::Layer);
    nm::Setting dup = settings.settings[0];
    dup.default_value = "-1";
    settings.settings.push_back(dup);
    nm::Setting word;
    word.name = "lambda";
    word.default_value = "x";
    settings.settings.push_back(word);
    nm::Setting method;
    method.name = "forward";
    method.default_value = "1";
    settings.settings.push_back(method);
    nm::Setting choice;
    choice.name = "pick";
    choice.type = nm::SettingType::Choice;
    choice.choices = {"a"};
    choice.default_value = "b";
    settings.settings.push_back(choice);
    Check(HasIssue(settings, "Two settings are called beta"), "duplicate settings are refused");
    Check(HasIssue(settings, "outside lowest and highest"), "a default below lowest is refused");
    Check(HasIssue(settings, "lambda is a Python word"), "a Python word is refused");
    Check(HasIssue(settings, "the default must be a number"), "a non-number default is refused");
    Check(HasIssue(settings, "name of a method"), "a method name is refused");
    Check(HasIssue(settings, "one of its choices"), "a default outside the choices is refused");
    nm::NodeForm act_with_setting = nm::Starter(nm::Kind::Activation);
    act_with_setting.settings.push_back(nm::Starter(nm::Kind::Layer).settings[0]);
    Check(HasIssue(act_with_setting, "Layer shown as Activation"), "an activation with a setting is refused");

    // A file changed outside the form is refused, not read in part: the
    // next Check would drop what the form does not write.
    {
        const nm::NodeForm swish = nm::Starter(nm::Kind::Layer);
        const std::string source = nm::Generate(swish, "project.catdog.swish");
        nm::NodeForm back;
        std::string why;
        std::string with_import = source;
        with_import.replace(with_import.find("from pycyxwiz import nodes\n"), 27,
                            "from pycyxwiz import nodes\nimport math\n");
        Check(!nm::Read(with_import, back, &why) && Contains(why, "Create node from script"),
              "an added import refuses the read: " + why);
        Check(!nm::Read(source + "\nHELPER = 2\n", back, &why), "a module-level line after the class refuses the read");
        std::string crlf;
        for (char c : source) crlf += c == '\n' ? std::string("\r\n") : std::string(1, c);
        Check(nm::Read(crlf, back, &why), "a CRLF copy of the file reads (" + why + ")");
        nm::NodeForm tabs = swish;
        tabs.bodies["forward"] = "self.x = x\nif True:\n\tself.s = x\nreturn x\n\n\n";
        const std::string tab_source = nm::Generate(tabs, "project.catdog.swish");
        Check(!Contains(tab_source, "\t") && Contains(tab_source, "            self.s = x\n"),
              "a tab in a body becomes four spaces");
        Check(nm::Read(tab_source, back, &why) && back.bodies["forward"] == "self.x = x\nif True:\n    self.s = x\nreturn x",
              "a body with trailing blank lines reads back without them (" + why + ")");
    }

    // Numbers Python reads as written, and names the class already has.
    {
        nm::NodeForm numbers = nm::Starter(nm::Kind::Layer);
        numbers.settings[0].default_value = "inf";
        Check(HasIssue(numbers, "the default must be a number"), "inf is not a number here");
        numbers.settings[0].default_value = "0x10";
        Check(HasIssue(numbers, "the default must be a number"), "hex is not a number here");
        numbers.settings[0].default_value = "1e-5";
        Check(!HasIssue(numbers, "the default must be a number"), "1e-5 is a number");
        nm::Setting whole;
        whole.name = "units";
        whole.type = nm::SettingType::WholeNumber;
        whole.default_value = "16";
        whole.lowest = "007";
        numbers.settings.push_back(whole);
        Check(HasIssue(numbers, "lowest must be a whole number"), "a leading zero is refused");
        nm::NodeForm hides = nm::Starter(nm::Kind::Layer);
        hides.settings[0].name = "state";
        Check(HasIssue(hides, "name of a method"), "a setting may not hide Module.state");
        nm::NodeForm kept = nm::Starter(nm::Kind::Layer);
        kept.bodies["forward"] = "return x";
        Check(nm::WithKind(kept, nm::Kind::Loss).display_name == "Swish (beta)",
              "a starter whose code was edited keeps its names when the kind changes");
    }

    // Kinds: pins, methods and switching.
    Check(nm::InputPins(nm::Kind::Loss).size() == 2 && nm::InputPins(nm::Kind::Loss)[1].type == "Labels",
          "a loss takes predictions and labels");
    Check(nm::OutputPins(nm::Kind::Layer)[0].name == "Output", "a layer gives Output");
    Check(nm::MethodNames(nm::Starter(nm::Kind::Activation)) == std::vector<std::string>{"f", "df"},
          "an activation writes f and df");
    const nm::NodeForm switched = nm::WithKind(nm::Starter(nm::Kind::Layer), nm::Kind::Loss);
    Check(switched.class_name == "LogCosh", "a starter switched becomes the other starter");
    nm::NodeForm named = nm::Starter(nm::Kind::Layer);
    named.display_name = "Gate";
    named.class_name = "Gate";
    const nm::NodeForm named_act = nm::WithKind(named, nm::Kind::Activation);
    Check(named_act.class_name == "Gate" && named_act.settings.empty() && named_act.bodies.count("f"),
          "a named form keeps its names, drops settings for an activation and gets f");

    // Names.
    Check(nm::ClassNameFrom("Swish (beta)") == "SwishBeta", "class name from a display name");
    Check(nm::ClassNameFrom("3 way mix") == "Node3WayMix", "a class name never starts with a digit");
    Check(nm::FileStemFor("SoftClip") == "soft_clip", "file stem of SoftClip");
    Check(nm::FileStemFor("MLPBlock") == "mlp_block", "file stem of MLPBlock");
    Check(nm::FileStemFor("Layer2Norm") == "layer2_norm", "file stem with a digit");
    Check(nm::BundleIdFor("Cats & Dogs", "soft_clip") == "project.cats-dogs.soft_clip", "bundle id from a project");
    Check(nm::BundleIdFor("", "x") == "project.untitled.x", "bundle id without a project name");

    if (failures > 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_python_node_maker: all checks passed\n";
    return 0;
}
