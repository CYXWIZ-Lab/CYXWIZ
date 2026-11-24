#include "python_node_maker.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>

namespace cyxwiz::node_maker {

namespace {

constexpr const char* kMarker = "# Made with Make a Python node in CyxWiz.";

bool IsIdentifier(const std::string& text) {
    if (text.empty() || !std::isalpha(static_cast<unsigned char>(text[0]))) return false;
    return std::all_of(text.begin(), text.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}

bool IsPythonWord(const std::string& text) {
    static const std::set<std::string> words = {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
        "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
        "with", "yield", "self"};
    return words.count(text) > 0;
}

// A decimal literal Python reads the same way: digits, an optional point and
// exponent; no inf, nan or hex (strtod takes those).
bool IsDecimal(const std::string& text) {
    size_t i = 0;
    const auto digits_at = [&]() {
        size_t n = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i, ++n;
        return n;
    };
    if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
    size_t digits = digits_at();
    if (i < text.size() && text[i] == '.') {
        ++i;
        digits += digits_at();
    }
    if (digits == 0) return false;
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
        if (digits_at() == 0) return false;
    }
    return i == text.size();
}

bool ParseNumber(const std::string& text, double& out) {
    if (!IsDecimal(text)) return false;
    char* end = nullptr;
    out = std::strtod(text.c_str(), &end);
    return end && *end == '\0';
}

bool ParseWhole(const std::string& text, long long& out) {
    if (text.empty()) return false;
    // Python refuses a whole number with a leading zero ("007").
    const size_t first = (text[0] == '-' || text[0] == '+') ? 1 : 0;
    if (text.size() > first + 1 && text[first] == '0') return false;
    char* end = nullptr;
    out = std::strtoll(text.c_str(), &end, 10);
    return end && *end == '\0';
}

std::string Quote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '\\' || c == '"') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// A quoted Python string back to its text; anything else as it is.
std::string Unquote(const std::string& text) {
    if (text.size() < 2 || text.front() != '"' || text.back() != '"') return text;
    std::string out;
    for (size_t i = 1; i + 1 < text.size(); ++i) {
        if (text[i] == '\\' && i + 2 < text.size()) {
            const char next = text[++i];
            out += next == 'n' ? '\n' : next;
        } else {
            out += text[i];
        }
    }
    return out;
}

// "1" -> "1.0": a float setting's literal stays a float in Python.
std::string FloatLiteral(const std::string& text) {
    if (text.find_first_of(".eE") != std::string::npos) return text;
    return text + ".0";
}

// Splits "a, b=[1, 2], c=\"x, y\"" at the commas outside quotes and brackets.
std::vector<std::string> SplitTopLevel(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    bool quoted = false;
    int depth = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            current += c;
            if (c == '\\' && i + 1 < text.size()) {
                current += text[++i];
            } else if (c == '"') {
                quoted = false;
            }
            continue;
        }
        if (c == '"') quoted = true;
        if (c == '[') ++depth;
        if (c == ']') --depth;
        if (c == ',' && depth == 0) {
            parts.push_back(current);
            current.clear();
            continue;
        }
        current += c;
    }
    if (!current.empty()) parts.push_back(current);
    for (auto& part : parts) {
        const size_t first = part.find_first_not_of(' ');
        const size_t last = part.find_last_not_of(' ');
        part = first == std::string::npos ? std::string() : part.substr(first, last - first + 1);
    }
    return parts;
}

// A body as the file holds it: tabs as four spaces (a tab after the
// method's spaces is a TabError), no trailing blank lines (Read drops them).
std::string CleanBody(const std::string& body) {
    std::string out;
    for (char c : body) {
        if (c == '\t') {
            out += "    ";
        } else if (c != '\r') {
            out += c;
        }
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

std::string Indent(const std::string& body, const std::string& pad) {
    std::string out;
    const std::string clean = CleanBody(body);
    std::istringstream in(clean.empty() ? std::string("pass") : clean);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out += line.empty() ? "\n" : pad + line + "\n";
    }
    return out;
}

// Text compared line for line: no carriage returns, no trailing spaces, no
// blank lines at the end.
std::string Normalised(const std::string& text) {
    std::string out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out += line + "\n";
    }
    while (out.size() > 1 && out[out.size() - 1] == '\n' && out[out.size() - 2] == '\n') out.pop_back();
    return out;
}

const char* PythonType(SettingType type) {
    switch (type) {
        case SettingType::Number: return "float";
        case SettingType::WholeNumber: return "int";
        case SettingType::OnOff: return "bool";
        case SettingType::Text:
        case SettingType::Choice: return "str";
    }
    return "float";
}

std::string DefaultLiteral(const Setting& s) {
    switch (s.type) {
        case SettingType::Number: return FloatLiteral(s.default_value);
        case SettingType::WholeNumber: return s.default_value;
        case SettingType::OnOff: return s.default_value == "true" ? "True" : "False";
        case SettingType::Text:
        case SettingType::Choice: return Quote(s.default_value);
    }
    return s.default_value;
}

std::string ParamLine(const Setting& s) {
    std::string line = "    " + s.name + " = nodes.Param(" + PythonType(s.type) + ", default=" + DefaultLiteral(s);
    const bool numeric = s.type == SettingType::Number || s.type == SettingType::WholeNumber;
    const auto bound = [&](const std::string& v) { return s.type == SettingType::Number ? FloatLiteral(v) : v; };
    if (numeric && !s.lowest.empty()) line += ", min=" + bound(s.lowest);
    if (numeric && !s.highest.empty()) line += ", max=" + bound(s.highest);
    if (s.type == SettingType::Choice) {
        line += ", choices=[";
        for (size_t i = 0; i < s.choices.size(); ++i) line += (i ? ", " : "") + Quote(s.choices[i]);
        line += "]";
    }
    if (!s.description.empty()) line += ", description=" + Quote(s.description);
    return line + ")";
}

std::vector<std::string> ParseList(const std::string& text) {
    std::vector<std::string> items;
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') return items;
    for (const auto& part : SplitTopLevel(text.substr(1, text.size() - 2))) {
        if (!part.empty()) items.push_back(Unquote(part));
    }
    return items;
}

// Names a node's class already has (its methods, and what pycyxwiz.Module
// and Loss define): a setting with one of them would hide it.
bool IsModuleName(const std::string& name) {
    static const std::set<std::string> names = {
        "forward", "backward", "build", "output_shape", "f", "df", "add_parameter", "add_state",
        "backward_working_elements_per_sample", "cached_elements_per_sample", "freeze", "unfreeze",
        "get_gradients", "get_name", "get_parameters", "get_state_names", "gradient", "has_parameters",
        "is_trainable", "is_training", "parameter", "parameter_names", "set_gradient", "set_parameters",
        "set_state", "set_training", "state", "state_names", "gradcheck", "gradcheck_loss"};
    return names.count(name) > 0;
}

bool StartsWith(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

const char* KindLabel(Kind kind) {
    switch (kind) {
        case Kind::Activation: return "Activation";
        case Kind::Layer: return "Layer";
        case Kind::Loss: return "Loss";
    }
    return "Layer";
}

const char* KindNote(Kind kind) {
    switch (kind) {
        case Kind::Activation:
            return "Element by element: y = f(x). You write f and its derivative df; no settings.";
        case Kind::Layer:
            return "Input \xE2\x86\x92 Output. You write forward and backward; settings show in Properties.";
        case Kind::Loss:
            return "Predictions + Targets \xE2\x86\x92 Loss. You write the loss and its gradient for the predictions.";
    }
    return "";
}

const char* SettingTypeLabel(SettingType type) {
    switch (type) {
        case SettingType::Number: return "Number";
        case SettingType::WholeNumber: return "Whole number";
        case SettingType::OnOff: return "On / off";
        case SettingType::Text: return "Text";
        case SettingType::Choice: return "Choice";
    }
    return "Number";
}

std::vector<Pin> InputPins(Kind kind) {
    if (kind == Kind::Loss) return {{"Predictions", "Tensor"}, {"Targets", "Labels"}};
    return {{"Input", "Tensor"}};
}

std::vector<Pin> OutputPins(Kind kind) {
    if (kind == Kind::Loss) return {{"Loss", "Loss"}};
    return {{"Output", "Tensor"}};
}

std::vector<std::string> MethodNames(const NodeForm& form) {
    switch (form.kind) {
        case Kind::Activation: return {"f", "df"};
        case Kind::Layer:
            if (form.computes_shape) return {"output_shape", "forward", "backward"};
            return {"forward", "backward"};
        case Kind::Loss: return {"forward", "backward"};
    }
    return {};
}

std::string MethodSignature(const NodeForm& form, const std::string& method) {
    if (form.kind == Kind::Loss) return "def " + method + "(self, predictions, targets):";
    if (method == "output_shape") return "def output_shape(self, input_shape):";
    if (method == "backward") return "def backward(self, grad_output):";
    return "def " + method + "(self, x):";
}

NodeForm Starter(Kind kind) {
    NodeForm form;
    form.kind = kind;
    switch (kind) {
        case Kind::Activation:
            form.display_name = "Soft Clip";
            form.class_name = "SoftClip";
            form.category = "Activation";
            form.description = "x / (1 + |x|): like tanh, cheaper, never saturates to exactly \xC2\xB1" "1";
            form.bodies["f"] = "return x / (x.abs() + 1.0)";
            form.bodies["df"] = "d = x.abs() + 1.0\nreturn 1.0 / (d * d)";
            break;
        case Kind::Layer: {
            form.display_name = "Swish (beta)";
            form.class_name = "Swish";
            form.category = "Activation";
            form.description = "x * sigmoid(beta * x); beta = 1 is SiLU";
            form.shown_as_activation = true;
            Setting beta;
            beta.name = "beta";
            beta.type = SettingType::Number;
            beta.default_value = "1.0";
            beta.lowest = "0.0";
            beta.description = "How sharp the curve is";
            form.settings.push_back(beta);
            form.bodies["forward"] = "self.x = x\nself.s = 1.0 / ((-(x * self.beta)).exp() + 1.0)\nreturn x * self.s";
            form.bodies["backward"] = "s = self.s\nreturn grad_output * (s + self.x * self.beta * s * (1.0 - s))";
            form.bodies["output_shape"] = "return list(input_shape)";
            break;
        }
        case Kind::Loss:
            form.display_name = "Log-cosh";
            form.class_name = "LogCosh";
            form.category = "Training";
            form.description = "mean(log(cosh(prediction - target))): like MSE near 0, like L1 far away";
            form.continuous_targets = true;
            form.bodies["forward"] =
                "d = predictions - targets\ncosh = (d.exp() + (-d).exp()) * 0.5\nreturn cosh.log().mean()";
            form.bodies["backward"] =
                "e2 = ((predictions - targets) * 2.0).exp()\ntanh = (e2 - 1.0) / (e2 + 1.0)\n"
                "return tanh * (1.0 / predictions.num_elements())";
            break;
    }
    return form;
}

NodeForm WithKind(const NodeForm& form, Kind kind) {
    if (form.kind == kind) return form;
    const NodeForm old_starter = Starter(form.kind);
    NodeForm next = Starter(kind);
    // A form still showing the starter becomes the other starter; one the
    // user named keeps its names, group, description and version.
    const bool own = Generate(form, "x") != Generate(old_starter, "x");
    if (own) {
        next.display_name = form.display_name;
        next.class_name = form.class_name;
        next.category = form.category;
        next.description = form.description;
        next.version = form.version;
        if (kind != Kind::Activation && form.kind != Kind::Activation) next.settings = form.settings;
        if (kind != Kind::Activation && form.kind == Kind::Activation) next.settings.clear();
    }
    return next;
}

std::string ClassNameFrom(const std::string& display_name) {
    std::string out;
    bool start = true;
    for (unsigned char c : display_name) {
        if (!std::isalnum(c)) {
            start = true;
            continue;
        }
        out += start ? static_cast<char>(std::toupper(c)) : static_cast<char>(c);
        start = false;
    }
    if (!out.empty() && std::isdigit(static_cast<unsigned char>(out[0]))) out = "Node" + out;
    return out;
}

std::string FileStemFor(const std::string& class_name) {
    std::string out;
    for (size_t i = 0; i < class_name.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(class_name[i]);
        if (std::isupper(c) && i > 0) {
            const unsigned char prev = static_cast<unsigned char>(class_name[i - 1]);
            const bool next_lower = i + 1 < class_name.size() &&
                                    std::islower(static_cast<unsigned char>(class_name[i + 1]));
            if (std::islower(prev) || std::isdigit(prev) || (std::isupper(prev) && next_lower)) out += '_';
        }
        out += std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
    }
    if (out.empty() || !std::isalnum(static_cast<unsigned char>(out[0]))) out = "node_" + out;
    return out;
}

std::string BundleIdFor(const std::string& project_name, const std::string& file_stem) {
    std::string project;
    for (unsigned char c : project_name) {
        if (std::isalnum(c)) {
            project += static_cast<char>(std::tolower(c));
        } else if (!project.empty() && project.back() != '-') {
            project += '-';
        }
    }
    while (!project.empty() && project.back() == '-') project.pop_back();
    if (project.empty()) project = "untitled";
    return "project." + project + "." + file_stem;
}

std::vector<std::string> Validate(const NodeForm& form) {
    std::vector<std::string> issues;
    if (form.display_name.find_first_not_of(" \t") == std::string::npos)
        issues.push_back("Give it a name for the Nodes panel");
    if (form.class_name.empty()) {
        issues.push_back("Give it a class name");
    } else if (!IsIdentifier(form.class_name) || IsPythonWord(form.class_name)) {
        issues.push_back("The class name starts with a letter and holds only letters, digits and _");
    }
    {
        int parts = 0;
        bool ok = !form.version.empty();
        std::string part;
        for (char c : form.version + ".") {
            if (c == '.') {
                ok = ok && !part.empty();
                ++parts;
                part.clear();
            } else if (std::isdigit(static_cast<unsigned char>(c))) {
                part += c;
            } else {
                ok = false;
            }
        }
        if (!ok || parts != 3) issues.push_back("The version looks like 1.0.0");
    }
    if (form.kind == Kind::Activation && !form.settings.empty())
        issues.push_back("An activation has no settings: make it a Layer shown as Activation");

    const auto methods = MethodNames(form);
    std::set<std::string> names;
    for (const auto& s : form.settings) {
        const std::string label = s.name.empty() ? std::string("A setting") : "Setting " + s.name;
        if (s.name.empty()) {
            issues.push_back("A setting needs a name");
            continue;
        }
        if (!IsIdentifier(s.name)) {
            issues.push_back(label + ": a name starts with a letter and holds only letters, digits and _");
        } else if (IsPythonWord(s.name)) {
            issues.push_back(label + ": " + s.name + " is a Python word; pick another name");
        } else if (IsModuleName(s.name)) {
            issues.push_back(label + " has the name of a method; pick another name");
        }
        if (!names.insert(s.name).second) issues.push_back("Two settings are called " + s.name);

        if (s.type == SettingType::Number || s.type == SettingType::WholeNumber) {
            const bool whole = s.type == SettingType::WholeNumber;
            const auto read = [&](const std::string& text, double& out) {
                long long w = 0;
                if (whole) {
                    if (!ParseWhole(text, w)) return false;
                    out = static_cast<double>(w);
                    return true;
                }
                return ParseNumber(text, out);
            };
            const char* what = whole ? "a whole number" : "a number";
            double value = 0, low = 0, high = 0;
            const bool has_value = read(s.default_value, value);
            if (!has_value) issues.push_back(label + ": the default must be " + std::string(what));
            const bool has_low = !s.lowest.empty();
            const bool has_high = !s.highest.empty();
            if (has_low && !read(s.lowest, low)) issues.push_back(label + ": lowest must be " + std::string(what));
            if (has_high && !read(s.highest, high)) issues.push_back(label + ": highest must be " + std::string(what));
            if (has_low && has_high && low > high) issues.push_back(label + ": lowest is above highest");
            if (has_value && ((has_low && value < low) || (has_high && value > high)))
                issues.push_back(label + ": the default is outside lowest and highest");
        } else if (s.type == SettingType::OnOff) {
            if (s.default_value != "true" && s.default_value != "false")
                issues.push_back(label + ": the default is on or off");
        } else if (s.type == SettingType::Choice) {
            if (s.choices.empty()) {
                issues.push_back(label + ": add at least one choice");
            } else if (std::find(s.choices.begin(), s.choices.end(), s.default_value) == s.choices.end()) {
                issues.push_back(label + ": the default must be one of its choices");
            }
        }
    }
    for (const auto& method : methods) {
        const auto it = form.bodies.find(method);
        if (it == form.bodies.end() || it->second.find_first_not_of(" \t\r\n") == std::string::npos)
            issues.push_back("Write the body of " + method);
    }
    return issues;
}

std::string Generate(const NodeForm& form, const std::string& bundle_id) {
    std::string out;
    out += std::string(kMarker) + "\n";
    out += "# The decorator and the settings follow the form; the method bodies are yours.\n";
    out += "import pycyxwiz as cyx\nfrom pycyxwiz import nodes\n\n\n";
    const char* decorator = form.kind == Kind::Activation ? "activation" : form.kind == Kind::Loss ? "loss" : "node";
    out += std::string("@nodes.") + decorator + "(\n";
    out += "    id=" + Quote(bundle_id) + ",\n";
    out += "    name=" + Quote(form.class_name) + ",\n";
    out += "    version=" + Quote(form.version) + ",\n";
    if (form.kind == Kind::Layer) out += std::string("    role=") + (form.shown_as_activation ? "\"activation\"" : "\"layer\"") + ",\n";
    if (form.kind == Kind::Loss) out += std::string("    target=") + (form.continuous_targets ? "\"continuous\"" : "\"categorical\"") + ",\n";
    out += "    category=" + Quote(form.category.empty() ? std::string("Custom") : form.category) + ",\n";
    out += "    display_name=" + Quote(form.display_name) + ",\n";
    out += "    description=" + Quote(form.description) + ",\n";
    out += ")\n";
    out += "class " + form.class_name;
    out += form.kind == Kind::Activation ? ":\n" : form.kind == Kind::Loss ? "(cyx.Loss):\n" : "(cyx.Module):\n";
    if (form.kind != Kind::Activation) {
        for (const auto& s : form.settings) out += ParamLine(s) + "\n";
        if (!form.settings.empty()) out += "\n";
    }
    const auto methods = MethodNames(form);
    for (size_t i = 0; i < methods.size(); ++i) {
        if (i) out += "\n";
        out += "    " + MethodSignature(form, methods[i]) + "\n";
        const auto it = form.bodies.find(methods[i]);
        out += Indent(it == form.bodies.end() ? std::string() : it->second, "        ");
    }
    return out;
}

bool Read(const std::string& source, NodeForm& form, std::string* error) {
    const auto fail = [error](const std::string& why) {
        if (error) *error = why;
        return false;
    };
    std::vector<std::string> lines;
    {
        std::istringstream in(source);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
    }
    if (lines.empty() || lines[0] != kMarker)
        return fail("it was not made with Make a Python node; open it in the Script Editor");

    NodeForm read;
    read.version.clear();
    std::string bundle_id;
    size_t i = 0;
    while (i < lines.size() && !StartsWith(lines[i], "@nodes.")) ++i;
    if (i == lines.size()) return fail("it has no node");
    const std::string& head = lines[i];
    if (StartsWith(head, "@nodes.activation(")) {
        read.kind = Kind::Activation;
    } else if (StartsWith(head, "@nodes.loss(")) {
        read.kind = Kind::Loss;
    } else if (StartsWith(head, "@nodes.node(")) {
        read.kind = Kind::Layer;
    } else {
        return fail("its decorator is not one Make a Python node writes");
    }
    for (++i; i < lines.size() && lines[i] != ")"; ++i) {
        std::string line = lines[i];
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        key.erase(0, key.find_first_not_of(' '));
        std::string value = line.substr(eq + 1);
        if (!value.empty() && value.back() == ',') value.pop_back();
        value = Unquote(value);
        if (key == "id") bundle_id = value;
        else if (key == "name") read.class_name = value;
        else if (key == "version") read.version = value;
        else if (key == "category") read.category = value;
        else if (key == "display_name") read.display_name = value;
        else if (key == "description") read.description = value;
        else if (key == "role") read.shown_as_activation = value == "activation";
        else if (key == "target") read.continuous_targets = value == "continuous";
    }
    if (i == lines.size()) return fail("its decorator does not end");
    ++i;  // the class line
    if (i >= lines.size() || !StartsWith(lines[i], "class ")) return fail("the class line is missing");

    const std::string param_mark = " = nodes.Param(";
    std::string method;
    std::string body;
    const auto close_method = [&]() {
        if (method.empty()) return;
        while (!body.empty() && body.back() == '\n') body.pop_back();
        read.bodies[method] = body;
        method.clear();
        body.clear();
    };
    for (++i; i < lines.size(); ++i) {
        const std::string& line = lines[i];
        if (StartsWith(line, "    def ")) {
            close_method();
            const size_t open = line.find('(');
            method = line.substr(8, open == std::string::npos ? std::string::npos : open - 8);
            if (method == "output_shape") read.computes_shape = true;
            continue;
        }
        if (method.empty()) {
            const size_t mark = line.find(param_mark);
            if (StartsWith(line, "    ") && mark != std::string::npos && line.back() == ')') {
                Setting s;
                s.name = line.substr(4, mark - 4);
                const auto parts =
                    SplitTopLevel(line.substr(mark + param_mark.size(), line.size() - mark - param_mark.size() - 1));
                if (parts.empty()) continue;
                const std::string& type = parts[0];
                s.type = type == "int" ? SettingType::WholeNumber
                         : type == "bool" ? SettingType::OnOff
                         : type == "str" ? SettingType::Text
                                         : SettingType::Number;
                for (size_t p = 1; p < parts.size(); ++p) {
                    const size_t eq = parts[p].find('=');
                    if (eq == std::string::npos) continue;
                    const std::string key = parts[p].substr(0, eq);
                    const std::string value = parts[p].substr(eq + 1);
                    if (key == "default") {
                        s.default_value = value == "True" ? "true" : value == "False" ? "false" : Unquote(value);
                    } else if (key == "min") {
                        s.lowest = value;
                    } else if (key == "max") {
                        s.highest = value;
                    } else if (key == "choices") {
                        s.choices = ParseList(value);
                        s.type = SettingType::Choice;
                    } else if (key == "description") {
                        s.description = Unquote(value);
                    }
                }
                read.settings.push_back(s);
            }
            continue;
        }
        body += (StartsWith(line, "        ") ? line.substr(8) : line) + "\n";
    }
    close_method();
    if (read.class_name.empty()) return fail("its decorator has no name");
    // Only a file this form writes back unchanged: anything else (an import,
    // a helper, a comment) would be lost on the next Check or Create.
    if (Normalised(Generate(read, bundle_id)) != Normalised(source))
        return fail("it has lines Make a Python node does not write (changed in the Script Editor?); open it in the "
                    "Script Editor and make the node with Run > Create node from script");
    form = read;
    return true;
}

}  // namespace cyxwiz::node_maker
