#include "node_presets.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <system_error>

namespace cyxwiz::node_presets {

std::vector<Preset> BuiltinPresets(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Conv2D:
            return {
                {"VGG-style", {{"filters", "64"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}}, true},
                {"ResNet-style", {{"filters", "64"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}, {"activation", "relu"}}, true},
                {"MobileNet-style", {{"filters", "32"}, {"kernel_size", "3"}, {"stride", "2"}, {"padding", "same"}}, true},
            };
        case gui::NodeType::Dense:
            return {
                {"Small (64)", {{"units", "64"}}, true},
                {"Medium (256)", {{"units", "256"}}, true},
                {"Large (1024)", {{"units", "1024"}}, true},
            };
        case gui::NodeType::Adam:
            return {
                {"Default", {{"learning_rate", "0.001"}, {"beta1", "0.9"}, {"beta2", "0.999"}, {"epsilon", "1e-8"}}, true},
                {"Fast learning", {{"learning_rate", "0.01"}, {"beta1", "0.9"}, {"beta2", "0.999"}, {"epsilon", "1e-8"}}, true},
                {"Fine-tuning", {{"learning_rate", "0.0001"}, {"beta1", "0.9"}, {"beta2", "0.999"}, {"epsilon", "1e-8"}}, true},
            };
        default:
            return {};
    }
}

bool Store::Load(const std::filesystem::path& file, std::string* error) {
    presets_.clear();
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) return true;
    std::ifstream in(file);
    if (!in) {
        if (error) *error = "cannot read " + file.string();
        return false;
    }
    nlohmann::json j;
    try {
        in >> j;
        // Bound to locals: items() over a temporary would dangle.
        const nlohmann::json all = j.value("presets", nlohmann::json::object());
        for (const auto& [type_name, list] : all.items()) {
            for (const auto& entry : list) {
                Preset p;
                p.name = entry.value("name", "");
                if (p.name.empty()) continue;
                const nlohmann::json parameters = entry.value("parameters", nlohmann::json::object());
                for (const auto& [k, v] : parameters.items()) {
                    p.parameters[k] = v.is_string() ? v.get<std::string>() : v.dump();
                }
                presets_[type_name].push_back(std::move(p));
            }
        }
    } catch (const std::exception& e) {
        presets_.clear();
        if (error) *error = std::string("node_presets.json is not readable: ") + e.what();
        return false;
    }
    return true;
}

bool Store::Save(const std::filesystem::path& file, std::string* error) const {
    nlohmann::ordered_json j;
    j["version"] = 1;
    nlohmann::ordered_json all = nlohmann::ordered_json::object();
    for (const auto& [type_name, list] : presets_) {
        nlohmann::ordered_json arr = nlohmann::ordered_json::array();
        for (const auto& p : list) {
            nlohmann::ordered_json entry;
            entry["name"] = p.name;
            entry["parameters"] = p.parameters;
            arr.push_back(std::move(entry));
        }
        all[type_name] = std::move(arr);
    }
    j["presets"] = std::move(all);
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    std::ofstream out(file);
    if (!out) {
        if (error) *error = "cannot write " + file.string();
        return false;
    }
    out << j.dump(2) << '\n';
    return static_cast<bool>(out);
}

std::vector<Preset> Store::PresetsFor(const std::string& type_name) const {
    const auto it = presets_.find(type_name);
    return it == presets_.end() ? std::vector<Preset>{} : it->second;
}

void Store::Put(const std::string& type_name, const std::string& name,
                const std::map<std::string, std::string>& parameters) {
    auto& list = presets_[type_name];
    for (auto& p : list) {
        if (p.name == name) {
            p.parameters = parameters;
            return;
        }
    }
    list.push_back(Preset{name, parameters, false});
}

bool Store::Remove(const std::string& type_name, const std::string& name) {
    const auto it = presets_.find(type_name);
    if (it == presets_.end()) return false;
    auto& list = it->second;
    for (auto p = list.begin(); p != list.end(); ++p) {
        if (p->name == name) {
            list.erase(p);
            if (list.empty()) presets_.erase(it);
            return true;
        }
    }
    return false;
}

size_t Store::Size() const {
    size_t n = 0;
    for (const auto& [_, list] : presets_) n += list.size();
    return n;
}

std::vector<Preset> AllPresets(gui::NodeType type, const std::string& type_name, const Store& store) {
    std::vector<Preset> all = BuiltinPresets(type);
    for (const auto& saved : store.PresetsFor(type_name)) {
        bool replaced = false;
        for (auto& p : all) {
            if (p.name == saved.name) {
                p = saved;
                replaced = true;
            }
        }
        if (!replaced) all.push_back(saved);
    }
    return all;
}

size_t Apply(const Preset& preset, gui::MLNode& node) {
    for (const auto& [k, v] : preset.parameters) node.parameters[k] = v;
    return preset.parameters.size();
}

std::map<std::string, std::string> ParametersToSave(const gui::MLNode& node) {
    std::map<std::string, std::string> out;
    for (const auto& [k, v] : node.parameters) {
        if (k.rfind("_meta_", 0) == 0 || k == "plot_spec" || k == "dashboard_spec") continue;
        out[k] = v;
    }
    return out;
}

std::filesystem::path DefaultStoreFile() {
#ifdef _WIN32
    const char* base = std::getenv("APPDATA");
    if (base && *base) return std::filesystem::path(base) / "CyxWiz" / "node_presets.json";
    return std::filesystem::path(".") / "node_presets.json";
#else
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home && *home ? home : ".") / ".cyxwiz" / "node_presets.json";
#endif
}

}  // namespace cyxwiz::node_presets
