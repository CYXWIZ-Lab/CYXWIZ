#include "about_dialog.h"

#include "../../core/engine_config.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device.h>
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

#ifndef CYXWIZ_GIT_COMMIT
#define CYXWIZ_GIT_COMMIT "unknown"
#endif

namespace cyxwiz {

namespace {
constexpr const char* kTitle = "About CyxWiz###about_cyxwiz";
constexpr const char* kSite = "https://github.com/CYXWIZ-Lab/CYXWIZ";
constexpr const char* kIssues = "https://github.com/CYXWIZ-Lab/CYXWIZ/issues";
constexpr const char* kLicence = "https://github.com/CYXWIZ-Lab/CYXWIZ/blob/master/LICENSE";

std::string Gigabytes(size_t bytes) {
    if (bytes == 0) return {};
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

const char* BackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::METAL: return "Metal";
        case DeviceType::VULKAN: return "Vulkan";
        case DeviceType::ONEAPI: return "oneAPI";
    }
    return "Other";
}

// "Intel(R)_Core(TM)_i7-8750H_CPU_@ 2.20GHz" -> "Intel Core i7-8750H CPU"
std::string CleanDeviceName(std::string name) {
    for (char& c : name)
        if (c == '_') c = ' ';
    for (const char* mark : {"(R)", "(TM)", "(r)", "(tm)"}) {
        for (size_t at; (at = name.find(mark)) != std::string::npos;) name.erase(at, std::strlen(mark));
    }
    if (const size_t at = name.find(" @"); at != std::string::npos) name.erase(at);
    if (const size_t at = name.find('@'); at != std::string::npos) name.erase(at);
    std::string out;
    for (char c : name) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out += c;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string DeviceKey(const std::string& name) {
    std::string key;
    for (char c : CleanDeviceName(name)) {
        if (std::isalnum(static_cast<unsigned char>(c))) key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (key.size() > 3 && key.compare(key.size() - 3, 3, "cpu") == 0) key.resize(key.size() - 3);
    return key;
}
}  // namespace

void AboutDialog::Open() {
    Gather();
    copied_note_.clear();
    request_open_ = true;
}

void AboutDialog::Gather() {
    rows_.clear();
    rows_.push_back({"Version", GetVersionString()});
    rows_.push_back({"Build", std::string(__DATE__) + ", commit " + CYXWIZ_GIT_COMMIT});

    // One line per physical device, with every backend that reaches it
    // (the backend lists the same GPU once per CUDA, OpenCL and oneAPI).
    std::string devices;
    try {
        struct Physical {
            std::string name;
            std::vector<std::string> backends;
            size_t memory = 0;
        };
        std::vector<Physical> list;
        for (const auto& d : Device::GetAvailableDevices()) {
            if (!d.backend_available || d.name.empty()) continue;
            const std::string key = DeviceKey(d.name);
            auto it = std::find_if(list.begin(), list.end(), [&](const Physical& p) { return DeviceKey(p.name) == key; });
            if (it == list.end()) {
                list.push_back({CleanDeviceName(d.name), {}, 0});
                it = list.end() - 1;
            }
            const char* backend = BackendName(d.type);
            if (std::find(it->backends.begin(), it->backends.end(), backend) == it->backends.end()) it->backends.push_back(backend);
            if (d.type == DeviceType::CUDA && d.memory_total > 0) it->memory = d.memory_total;
        }
        for (const auto& p : list) {
            std::string line = p.name + " (";
            for (size_t i = 0; i < p.backends.size(); ++i) line += (i ? ", " : "") + p.backends[i];
            line += ")";
            if (p.memory > 0) line += ", " + Gigabytes(p.memory);
            devices += (devices.empty() ? "" : "\n") + line;
        }
    } catch (...) {
        devices = "Could not list the compute devices; see Preferences > Devices.";
    }
    rows_.push_back({"Compute", devices.empty() ? "CPU" : devices});

    const std::string python = core::EngineConfig::Instance().GetSystemPythonPath();
    rows_.push_back({"Python", python.empty() ? "Not configured" : python});
    rows_.push_back({"Built with", "C++20, Dear ImGui, ImPlot, ImNodes, ArrayFire, gRPC, pybind11"});
    rows_.push_back({"Licence", "See LICENSE in the installation folder, or the licence link below."});
}

void AboutDialog::Render() {
    using namespace ui;
    if (request_open_) {
        ImGui::OpenPopup(kTitle);
        request_open_ = false;
    }
    DialogOptions options;
    options.size = ImVec2(620.0f, 460.0f);
    options.min_size = ImVec2(480.0f, 360.0f);
    if (!BeginDialog(kTitle, options)) return;
    const Tokens& t = CurrentTokens();

    ImGui::TextColored(t.accent_text, "%s", ICON_FA_DIAGRAM_PROJECT);
    ImGui::SameLine(0.0f, t.space_md);
    {
        FontScope heading(Font::Heading);
        ImGui::TextColored(t.text_bright, "CyxWiz Engine");
    }
    ImGui::TextColored(t.text_dim, "Decentralized ML compute platform");
    ImGui::Spacing();

    std::vector<KeyValue> rows;
    for (const auto& [k, v] : rows_) rows.push_back({k, v});
    KeyValueTable("##about_rows", rows, 1, true);
    ImGui::Spacing();

    if (LinkButton("Website and source")) OpenUrl(kSite);
    ImGui::SameLine(0.0f, t.space_xl);
    if (LinkButton("Report an issue")) OpenUrl(kIssues);
    ImGui::SameLine(0.0f, t.space_xl);
    if (LinkButton("Licence text")) OpenUrl(kLicence);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "%s", copied_note_.empty() ? "Click a value to copy it, or copy every detail for a bug report."
                                                              : copied_note_.c_str());

    if (SecondaryButton("Copy details", true, nullptr, ButtonSize::Regular)) {
        std::string all;
        for (const auto& [k, v] : rows_) all += k + ": " + v + "\n";
        ImGui::SetClipboardText(all.c_str());
        copied_note_ = "Details copied to the clipboard.";
    }
    EndDialog("OK", nullptr);
}

}  // namespace cyxwiz
