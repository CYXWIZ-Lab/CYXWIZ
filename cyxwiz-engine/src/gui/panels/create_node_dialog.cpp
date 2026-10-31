#include "create_node_dialog.h"

#include "../python_bundle_ui.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../extension/python/python_node_provider.h"
#include "../../scripting/scripting_engine.h"

#include <imgui.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_PYTHON
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

constexpr const char* kPopup = "Create node from script###CreateNodeFromScript";

using CallResult = std::pair<bool, std::string>;

// pycyxwiz.nodes.<function>(script, extra...) on a worker thread, under the
// interpreter lock (the Engine releases it after starting Python).
std::future<CallResult> CallNodes(const char* function, std::string script, std::string folder = {},
                                  std::string name = {}) {
    return std::async(std::launch::async, [function, script, folder, name]() -> CallResult {
#ifdef CYXWIZ_HAS_PYTHON
        if (!Py_IsInitialized()) return {false, "Python is not running"};
        py::gil_scoped_acquire gil;
        try {
            py::object nodes = py::module_::import("pycyxwiz.nodes");
            py::object out = folder.empty() ? nodes.attr(function)(script)
                                            : nodes.attr(function)(script, folder, name);
            return {true, py::str(out).cast<std::string>()};
        } catch (py::error_already_set& e) {
            return {false, e.what()};
        } catch (const std::exception& e) {
            return {false, e.what()};
        }
#else
        (void)function;
        return {false, "Python is not available in this build of CyxWiz"};
#endif
    });
}

std::vector<CreateNodeFound> ParseFound(const std::string& text) {
    std::vector<CreateNodeFound> found;
    const auto list = nlohmann::json::parse(text);
    for (const auto& node : list) {
        CreateNodeFound f;
        f.name = node.value("name", "");
        f.version = node.value("version", "");
        f.category = node.value("category", "");
        const std::string tier = node.value("tier", "");
        f.tier = tier == "activation" ? "f and df" : tier == "loss" ? "loss" : "module";
        std::string in;
        std::string out;
        for (const auto& pin : node.value("inputs", nlohmann::json::array())) in += (in.empty() ? "" : ", ") + pin.value("name", "");
        for (const auto& pin : node.value("outputs", nlohmann::json::array())) out += (out.empty() ? "" : ", ") + pin.value("name", "");
        f.pins = in + " -> " + out;
        for (const auto& p : node.value("parameters", nlohmann::json::array())) {
            std::vector<std::string> choices;
            for (const auto& c : p.value("choices", nlohmann::json::array())) {
                if (c.is_string()) choices.push_back(c.get<std::string>());
            }
            f.settings.push_back(CreateNodeSettingText(p.value("name", ""), p.value("type", ""), p.value("default", ""),
                                                       p.value("validation", ""), choices));
        }
        found.push_back(std::move(f));
    }
    return found;
}

}  // namespace

void CreateNodeDialog::Open(const fs::path& script, std::shared_ptr<scripting::ScriptingEngine> engine) {
    if (pending_.valid()) pending_.wait();  // a call left running by Cancel
    pending_ = {};
    call_ = Call::None;
    script_ = script;
    engine_ = std::move(engine);
    inputs_ = CreateNodeInputs{};
    inputs_.script_name = script.filename().string();
    inputs_.bundle_name = SuggestBundleName(inputs_.script_name);
    inputs_.project_nodes = python_ui::CurrentRoots().project_nodes;
    std::snprintf(name_buf_, sizeof(name_buf_), "%s", inputs_.bundle_name.c_str());
    popup_requested_ = true;

    std::string error;
    if (!engine_ || !engine_->StartPython(&error)) {
        inputs_.stage = CreateNodeStage::Failed;
        inputs_.error = error.empty() ? std::string("Python did not start") : error;
        return;
    }
    StartInspect();
}

void CreateNodeDialog::StartInspect() {
    inputs_.stage = CreateNodeStage::Inspecting;
    call_ = Call::Inspect;
    pending_ = CallNodes("_engine_inspect", script_.string());
}

void CreateNodeDialog::StartCheck() {
    inputs_.stage = CreateNodeStage::Checking;
    call_ = Call::Check;
    pending_ = CallNodes("_engine_check", script_.string());
}

void CreateNodeDialog::StartWrite() {
    inputs_.stage = CreateNodeStage::Writing;
    call_ = Call::Write;
    std::error_code ec;
    fs::create_directories(inputs_.project_nodes, ec);
    pending_ = CallNodes("_engine_bundle", script_.string(), inputs_.project_nodes.string(), inputs_.bundle_name);
}

void CreateNodeDialog::Collect() {
    if (!pending_.valid() || pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    const auto [ok, text] = pending_.get();
    const Call call = call_;
    call_ = Call::None;
    if (!ok) {
        inputs_.stage = CreateNodeStage::Failed;
        inputs_.error = text;
        spdlog::warn("Create node from script ({}): {}", inputs_.script_name, text.substr(0, text.find('\n')));
        return;
    }
    try {
        if (call == Call::Inspect) {
            inputs_.nodes = ParseFound(text);
            inputs_.stage = inputs_.nodes.empty() ? CreateNodeStage::NoNodes : CreateNodeStage::Found;
        } else if (call == Call::Check) {
            for (const auto& r : nlohmann::json::parse(text)) {
                const std::string name = r.value("name", "");
                for (auto& node : inputs_.nodes) {
                    if (node.name != name) continue;
                    node.checked = true;
                    node.check = PythonCheckLine(r.value("status", ""), r.value("max_relative_error", 0.0),
                                                 r.value("where", ""), r.value("error", ""), node.kind);
                }
            }
            inputs_.stage = CreateNodeStage::Checked;
        } else if (call == Call::Write) {
            PythonBundle bundle;
            std::string error;
            if (!ReadPythonBundle(fs::path(text), bundle, error)) {
                inputs_.stage = CreateNodeStage::Failed;
                inputs_.error = "The bundle was written but cannot be read: " + error;
                return;
            }
            // The user wrote this code: approved as created (decision D6).
            PythonNodeApprovals::User().Approve(MakePythonNodeApproval(bundle, "created", TodayIsoDate()), &error);
            if (auto* provider = python_ui::Provider()) provider->Rescan();
            inputs_.written_folder = text;
            inputs_.stage = CreateNodeStage::Written;
            spdlog::info("Created Python node bundle {} from {}", text, inputs_.script_name);
        }
    } catch (const std::exception& e) {
        inputs_.stage = CreateNodeStage::Failed;
        inputs_.error = std::string("Unexpected answer from pycyxwiz: ") + e.what();
    }
}

void CreateNodeDialog::Render() {
    if (popup_requested_) {
        ImGui::OpenPopup(kPopup);
        popup_requested_ = false;
        open_ = true;
    }
    Collect();
    ui::DialogOptions options;
    options.size = ImVec2(620.0f, 520.0f);
    if (!ui::BeginDialog(kPopup, options)) {
        open_ = false;
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    inputs_.bundle_name = name_buf_;
    const CreateNodeView view = BuildCreateNodeView(inputs_);

    ImGui::TextUnformatted(view.title.c_str());
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(view.lead.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    for (const auto& node : inputs_.nodes) {
        if (inputs_.stage == CreateNodeStage::Written || inputs_.stage == CreateNodeStage::Failed) break;
        ImGui::PushID(node.name.c_str());
        ui::BeginCard(node.name.c_str());
        ImGui::TextUnformatted(node.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s \xC2\xB7 %s", node.version.c_str(), node.category.c_str(),
                           node.tier.c_str());
        const std::string check = node.checked ? node.check : std::string("Not checked yet");
        const ImVec4 colour = !node.checked ? ui::StatusStyleFor(ui::Status::NotVerifiedYet, t).colour
                              : node.kind == PythonCheckKind::Passed ? ui::StatusStyleFor(ui::Status::Verified, t).colour
                              : node.kind == PythonCheckKind::Failed ? ui::StatusStyleFor(ui::Status::Failed, t).colour
                                                                     : ui::StatusStyleFor(ui::Status::NotVerifiedYet, t).colour;
        ui::SameLineRight(ImGui::CalcTextSize(check.c_str()).x);
        ImGui::TextColored(colour, "%s", check.c_str());
        std::string detail = node.pins;
        if (node.settings.empty()) {
            detail += " \xC2\xB7 no settings";
        } else {
            for (const auto& s : node.settings) detail += " \xC2\xB7 " + s;
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", detail.c_str());
        ImGui::PopTextWrapPos();
        ui::EndCard();
        ImGui::PopID();
    }

    if (view.show_name) {
        ImGui::Spacing();
        ui::FieldLabel("Bundle name");
        ImGui::BeginDisabled(view.busy);
        ImGui::InputText("##bundle_name", name_buf_, sizeof(name_buf_));
        ImGui::EndDisabled();
        if (!view.name_error.empty()) ImGui::TextColored(t.caution, "%s", view.name_error.c_str());
        if (!view.target.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "Writes to %s", view.target.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    if (!view.note.empty()) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(inputs_.stage == CreateNodeStage::Failed ? t.caution : t.text_dim, "%s", view.note.c_str());
        ImGui::PopTextWrapPos();
    }

    const bool closes = view.primary_label == "Close";
    const char* reason = view.primary_disabled_reason.empty() ? nullptr : view.primary_disabled_reason.c_str();
    const auto result = ui::EndDialog(view.primary_label.c_str(), closes ? nullptr : "Cancel", view.primary_enabled,
                                      reason, false, /*close_on_primary=*/closes);
    if (result == ui::DialogResult::Primary && !closes) {
        if (inputs_.stage == CreateNodeStage::Found) {
            StartCheck();
        } else if (inputs_.stage == CreateNodeStage::Checked) {
            StartWrite();
        }
    }
    if (result != ui::DialogResult::None && (closes || result == ui::DialogResult::Secondary)) open_ = false;
}

}  // namespace cyxwiz
