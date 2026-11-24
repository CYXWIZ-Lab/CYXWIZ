#include "python_node_maker_panel.h"

#include "../editor_fonts.h"
#include "../icons.h"
#include "../python_node_calls.h"
#include "../separate_windows.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"

#include <imgui.h>
#include <imgui_stdlib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace cyxwiz {

namespace fs = std::filesystem;
namespace nm = node_maker;

namespace {

constexpr const char* kWindow = "Make a Python node";
constexpr const char* kConfirm = "Drop the changes?###MakePythonNodeConfirm";

std::string ReadText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// Text without carriage returns, to compare a file with what was written.
std::string WithoutCR(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

std::string FormatNumber(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.12g", value);
    return buffer;
}

// A check error in two significant figures: 8.5e-05.
std::string FormatError(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2g", value);
    return buffer;
}

// The source text before the first method: the decorator and the settings,
// which follow the form.
std::string HeaderOf(const std::string& source) {
    const size_t def = source.find("\n    def ");
    return def == std::string::npos ? source : source.substr(0, def + 1);
}

// A multi-line code field over a std::string.
bool CodeField(const char* id, std::string& text, int min_lines) {
    const ui::Tokens& t = ui::CurrentTokens();
    const int lines = std::max(min_lines, static_cast<int>(std::count(text.begin(), text.end(), '\n')) + 2);
    ImFont* code = cyxwiz::gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_input);
    const float height = ImGui::GetTextLineHeight() * static_cast<float>(lines) + ImGui::GetStyle().FramePadding.y * 2.0f;
    const bool changed = ImGui::InputTextMultiline(id, &text, ImVec2(-1.0f, height), ImGuiInputTextFlags_AllowTabInput);
    ImGui::PopStyleColor();
    if (code) ImGui::PopFont();
    return changed;
}

void Section(const char* title) { ui::SettingsSection(title); }

// A pin with its dot, as the canvas colours them.
void PinLine(const nm::Pin& pin, bool dot_after) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec4 colour = pin.type == "Labels" ? t.series[3] : pin.type == "Loss" ? t.accent : t.info;
    const float r = ImGui::GetTextLineHeight() * 0.28f;
    const auto dot = [&]() {
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(at.x + r, at.y + ImGui::GetTextLineHeight() * 0.5f), r,
                                                    ImGui::GetColorU32(colour));
        ImGui::Dummy(ImVec2(r * 2.0f, ImGui::GetTextLineHeight()));
    };
    if (!dot_after) {
        dot();
        ImGui::SameLine();
    }
    ImGui::TextColored(t.text, "%s", pin.name.c_str());
    if (dot_after) {
        ImGui::SameLine();
        dot();
    }
}

}  // namespace

PythonNodeMakerPanel::PythonNodeMakerPanel() : Panel(kWindow, false) {
    Reset(nm::Starter(nm::Kind::Layer), {});
}

void PythonNodeMakerPanel::Open() {
    Show();
    bring_to_front_ = true;
}

bool PythonNodeMakerPanel::HasProject() const { return ProjectManager::Instance().HasActiveProject(); }

std::string PythonNodeMakerPanel::ProjectName() const { return ProjectManager::Instance().GetProjectName(); }

// The source in the project's nodes_src; empty without a project (Python
// starts only with one, so nothing is written or run before).
fs::path PythonNodeMakerPanel::SourcePath() const {
    if (form_.class_name.empty() || !HasProject()) return {};
    return fs::path(ProjectManager::Instance().GetProjectRoot()) / "nodes_src" /
           (nm::FileStemFor(form_.class_name) + ".py");
}

std::string PythonNodeMakerPanel::ConflictReason() const {
    const fs::path path = SourcePath();
    if (path.empty()) return {};
    std::error_code ec;
    if (fs::exists(path, ec) && !(!file_.empty() && fs::equivalent(path, file_, ec)))
        return "A node source " + path.filename().string() +
               " is already in the project: open it, or pick another class name";
    const std::string stem = path.stem().string();
    const fs::path manifest = fs::path(ProjectManager::Instance().GetProjectRoot()) / "nodes" / (stem + ".cyxnode") /
                              "node.json";
    if (fs::exists(manifest, ec)) {
        std::ifstream in(manifest);
        const auto j = nlohmann::json::parse(in, nullptr, false);
        const std::string id = j.is_object() ? j.value("id", std::string()) : std::string();
        if (id != nm::BundleIdFor(ProjectName(), stem))
            return "The project's nodes folder already has a bundle " + stem + ".cyxnode" +
                   (id.empty() ? std::string() : " (" + id + ")") + " made another way: pick another class name";
    }
    return {};
}

void PythonNodeMakerPanel::ConfirmIfEdited(const std::string& question, const char* confirm_label,
                                           std::function<void()> action) {
    if (!Edited()) {
        action();
        return;
    }
    confirm_question_ = question;
    confirm_label_ = confirm_label;
    confirm_action_ = std::move(action);
    confirm_requested_ = true;
}

void PythonNodeMakerPanel::Reset(const nm::NodeForm& form, const fs::path& file) {
    form_ = form;
    file_ = file;
    class_follows_name_ = file.empty() && nm::ClassNameFrom(form.display_name) == form.class_name;
    saved_source_ = file.empty() ? std::string() : ReadText(file);
    created_ = false;
    result_ = Result::None;
    result_title_.clear();
    result_detail_.clear();
}

bool PythonNodeMakerPanel::Edited() const {
    if (!file_.empty()) return Dirty();
    // A new form the user has not touched is the starter of its kind.
    return nm::Generate(form_, "x") != nm::Generate(nm::Starter(form_.kind), "x");
}

bool PythonNodeMakerPanel::Dirty() const {
    if (file_.empty()) return true;
    return nm::Generate(form_, nm::BundleIdFor(ProjectName(), nm::FileStemFor(form_.class_name))) != saved_source_;
}

bool PythonNodeMakerPanel::WriteSource(fs::path& written, std::string& error) {
    if (!HasProject()) {
        error = "Open a project first: Python runs in the project's environment";
        return false;
    }
    written = SourcePath();
    if (written.empty()) {
        error = "Give it a class name";
        return false;
    }
    std::error_code ec;
    fs::create_directories(written.parent_path(), ec);
    if (ec) {
        error = "Could not create " + written.parent_path().string() + ": " + ec.message();
        return false;
    }
    // The file this form opened, changed on disk since (in the Script Editor):
    // writing now would throw that change away.
    if (!file_.empty() && fs::exists(written, ec) && fs::equivalent(written, file_, ec) &&
        WithoutCR(ReadText(written)) != WithoutCR(saved_source_)) {
        error = written.filename().string() +
                " was changed on disk since it was opened here (in the Script Editor?). Open it again to load the "
                "change, or make it with Run > Create node from script";
        return false;
    }
    const std::string source = nm::Generate(form_, nm::BundleIdFor(ProjectName(), written.stem().string()));
    std::ofstream out(written, std::ios::binary | std::ios::trunc);
    out << source;
    out.close();
    if (!out) {
        error = "Could not write " + written.string();
        return false;
    }
    file_ = written;
    saved_source_ = source;
    return true;
}

void PythonNodeMakerPanel::StartCheck() {
    std::string error;
    if (!engine_ || !engine_->StartPython(&error)) {
        result_ = Result::CheckError;
        result_title_ = "The check could not run";
        result_detail_ = error.empty() ? std::string("Python did not start") : error;
        return;
    }
    fs::path written;
    if (!WriteSource(written, error)) {
        result_ = Result::Error;
        result_title_ = "Could not write the source";
        result_detail_ = error;
        return;
    }
    ran_source_ = saved_source_.empty() ? ReadText(written) : saved_source_;
    call_ = Call::Check;
    result_ = Result::Busy;
    result_title_ = "Checking the gradients...";
    result_detail_ = "backward against small changes of the input and of every learned value";
    pending_ = python_calls::CallNodes("_engine_check", written.string());
}

void PythonNodeMakerPanel::StartCreate() {
    std::string error;
    if (!engine_ || !engine_->StartPython(&error)) {
        result_ = Result::Error;
        result_title_ = "Could not create the node";
        result_detail_ = error.empty() ? std::string("Python did not start") : error;
        return;
    }
    fs::path written;
    if (!WriteSource(written, error)) {
        result_ = Result::Error;
        result_title_ = "Could not write the source";
        result_detail_ = error;
        return;
    }
    const fs::path nodes = fs::path(ProjectManager::Instance().GetProjectRoot()) / "nodes";
    std::error_code ec;
    fs::create_directories(nodes, ec);
    ran_source_ = saved_source_.empty() ? ReadText(written) : saved_source_;
    call_ = Call::Create;
    result_ = Result::Busy;
    result_title_ = "Creating the node...";
    result_detail_ = "checking it and writing " + (nodes / (written.stem().string() + ".cyxnode")).string();
    created_group_ = form_.category.empty() ? std::string("Custom") : form_.category;
    pending_ = python_calls::CallNodes("_engine_bundle", written.string(), nodes.string(), written.stem().string());
}

void PythonNodeMakerPanel::Collect() {
    if (!pending_.valid() || pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    const auto [ok, text] = pending_.get();
    const Call call = call_;
    call_ = Call::None;
    const std::string first_line = text.substr(0, text.find('\n'));
    if (!ok) {
        result_ = call == Call::Check ? Result::CheckError : Result::Error;
        result_title_ = call == Call::Check ? "The check could not run" : "Could not create the node";
        result_detail_ = first_line;
        spdlog::warn("Make a Python node ({}): {}", form_.class_name, first_line);
        return;
    }
    if (call == Call::Check) {
        try {
            for (const auto& entry : nlohmann::json::parse(text)) {
                if (entry.value("name", "") != form_.class_name) continue;
                const std::string status = entry.value("status", "");
                const double error = entry.value("max_relative_error", 0.0);
                if (status == "passed") {
                    result_ = Result::Passed;
                    result_title_ = "Gradient check passed";
                    result_detail_ = "Largest relative error " + FormatError(error) + " (limit 0.02).";
                } else if (status == "failed") {
                    const std::string where = entry.value("where", "");
                    result_ = Result::Failed;
                    result_title_ = "Gradient check failed" + (where.empty() ? std::string() : " at " + where);
                    result_detail_ = "Relative error " + FormatError(error) +
                                     " (limit 0.02): backward does not match forward. You can still create it; "
                                     "the result is kept with the node.";
                } else {
                    result_ = Result::CheckError;
                    result_title_ = "The check could not run";
                    result_detail_ = entry.value("error", std::string("no result"));
                }
                return;
            }
            result_ = Result::CheckError;
            result_title_ = "The check could not run";
            result_detail_ = "pycyxwiz found no node called " + form_.class_name + " in the source";
        } catch (const std::exception& e) {
            result_ = Result::CheckError;
            result_title_ = "The check could not run";
            result_detail_ = std::string("Unexpected answer from pycyxwiz: ") + e.what();
        }
        return;
    }
    std::string error;
    if (!python_calls::FinishCreatedBundle(fs::path(text), &error)) {
        result_ = Result::Error;
        result_title_ = "Could not create the node";
        result_detail_ = error;
        return;
    }
    created_ = true;
    result_ = Result::Created;
    result_title_ = "Created " + form_.display_name;
    result_detail_ = "Written to " + text + " (source " + file_.string() +
                     "), approved as yours. It is in the Nodes panel under Plugins > " + created_group_ + ".";
    spdlog::info("Make a Python node: created {} in {}", form_.display_name, text);
}

void PythonNodeMakerPanel::Render() {
    Collect();
    // A result belongs to the source it ran on: an edit since puts it away.
    if (result_ != Result::None && result_ != Result::Busy) {
        const fs::path path = SourcePath();
        const std::string now =
            path.empty() ? std::string() : nm::Generate(form_, nm::BundleIdFor(ProjectName(), path.stem().string()));
        if (WithoutCR(now) != WithoutCR(ran_source_)) result_ = Result::None;
    }
    if (!visible_) return;
    if (bring_to_front_) {
        ::gui::BringToFront(kWindow);
        bring_to_front_ = false;
    }
    ImGui::SetNextWindowSize(ImVec2(1040.0f, 780.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(kWindow, &visible_)) {
        ImGui::End();
        return;
    }
    const float footer = ImGui::GetFrameHeightWithSpacing() * 2.6f;
    RenderTopRow();
    // The form stays as it is while Check or Create runs on its source.
    ImGui::BeginDisabled(call_ != Call::None);
    ImGui::BeginChild("##maker_body", ImVec2(0.0f, -footer));
    RenderKind();
    ImGui::Spacing();
    if (ImGui::BeginTable("##maker_columns", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("form", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("code", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderDetails();
        RenderSettings();
        ImGui::TableNextColumn();
        RenderPins();
        RenderCode();
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::EndDisabled();
    RenderFooter(footer);
    RenderConfirm();
    ImGui::End();
}

void PythonNodeMakerPanel::RenderTopRow() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool busy = call_ != Call::None;
    if (ui::SecondaryButton("Open " ICON_FA_CHEVRON_DOWN, !busy, "Wait for the running step")) ImGui::OpenPopup("##open_source");
    if (ImGui::BeginPopup("##open_source")) {
        const fs::path folder = HasProject() ? fs::path(ProjectManager::Instance().GetProjectRoot()) / "nodes_src" : fs::path();
        if (ImGui::IsWindowAppearing()) {
            // Read the folder once per opening, not every frame.
            sources_.clear();
            std::vector<fs::path> files;
            std::error_code ec;
            for (fs::directory_iterator it(folder, ec), end; !folder.empty() && !ec && it != end; it.increment(ec)) {
                if (it->path().extension() == ".py") files.push_back(it->path());
            }
            std::sort(files.begin(), files.end());
            for (const auto& file : files) {
                SourceEntry entry;
                entry.file = file;
                std::string error;
                entry.ours = nm::Read(ReadText(file), entry.form, &error);
                entry.label = file.filename().string() +
                              (entry.ours ? "  \xC2\xB7  " + entry.form.display_name + ", " + nm::KindLabel(entry.form.kind)
                                          : std::string("  \xC2\xB7  opens in the Script Editor"));
                sources_.push_back(std::move(entry));
            }
        }
        if (folder.empty()) {
            ImGui::TextColored(t.text_dim, "Open a project to see its node sources");
        } else {
            ImGui::TextColored(t.text_dim, "Node sources in %s/nodes_src", ProjectName().c_str());
            if (sources_.empty()) ImGui::TextColored(t.text_faint, "None yet: Create node writes one here");
            for (const auto& entry : sources_) {
                if (!ImGui::MenuItem(entry.label.c_str())) continue;
                if (!entry.ours) {
                    if (open_in_script_editor_) open_in_script_editor_(entry.file.string());
                } else {
                    const SourceEntry chosen = entry;
                    ConfirmIfEdited("Your changes to this node are not created yet. Open " + chosen.file.filename().string() +
                                        " instead?",
                                    "Open it", [this, chosen]() { Reset(chosen.form, chosen.file); });
                }
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("New", !busy, "Wait for the running step")) {
        ConfirmIfEdited("Your changes to this node are not created yet. Start a new one?", "Start new",
                        [this]() { Reset(nm::Starter(nm::Kind::Layer), {}); });
    }
    ImGui::SameLine();
    const std::string file = form_.class_name.empty() ? std::string() : nm::FileStemFor(form_.class_name) + ".py";
    const std::string state = created_ && !Dirty() ? "created" : file_.empty() || Dirty() ? "not created yet" : "saved";
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Editing %s \xC2\xB7 %s", file.empty() ? "a new node" : file.c_str(), state.c_str());
}

void PythonNodeMakerPanel::RenderConfirm() {
    if (confirm_requested_) {
        ImGui::OpenPopup(kConfirm);
        confirm_requested_ = false;
    }
    const auto result = ui::ConfirmDialog(kConfirm, confirm_question_.c_str(), confirm_label_.c_str(), true);
    if (result == ui::DialogResult::Primary && confirm_action_) confirm_action_();
    if (result != ui::DialogResult::None) confirm_action_ = nullptr;
}

void PythonNodeMakerPanel::RenderKind() {
    const ui::Tokens& t = ui::CurrentTokens();
    Section("Kind");
    const char* kinds[] = {"Activation", "Layer", "Loss"};
    int selected = static_cast<int>(form_.kind);
    if (ui::SegmentedControl("##kind", kinds, 3, &selected)) {
        const auto kind = static_cast<nm::Kind>(selected);
        ConfirmIfEdited(std::string("A ") + nm::KindLabel(kind) + " starts its code over: the method bodies you wrote "
                                                                  "go. Switch?",
                        "Switch", [this, kind]() {
                            form_ = nm::WithKind(form_, kind);
                            class_follows_name_ = nm::ClassNameFrom(form_.display_name) == form_.class_name;
                        });
    }
    ImGui::TextColored(t.text_dim, "%s", nm::KindNote(form_.kind));
}

void PythonNodeMakerPanel::RenderDetails() {
    Section("Details");
    ui::FieldLabel("Name in the Nodes panel");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputText("##display_name", &form_.display_name) && class_follows_name_)
        form_.class_name = nm::ClassNameFrom(form_.display_name);

    if (ImGui::BeginTable("##names", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("class", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("version", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ui::FieldLabel("Class name");
        ImGui::SetNextItemWidth(-1.0f);
        ImFont* code = cyxwiz::gui::GetCodeFont();
        if (code) ImGui::PushFont(code);
        if (ImGui::InputText("##class_name", &form_.class_name)) class_follows_name_ = false;
        if (code) ImGui::PopFont();
        ImGui::TableNextColumn();
        ui::FieldLabel("Version");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##version", &form_.version);
        ImGui::EndTable();
    }

    ui::FieldLabel("Group in the Nodes panel (under Plugins)");
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##group", "Custom", &form_.category);
    if (ImGui::BeginPopupContextItem("##group_suggestions", ImGuiPopupFlags_MouseButtonLeft)) {
        for (const char* group : {"Activation", "ML Layers", "Normalization", "Training", "Custom"}) {
            if (ImGui::MenuItem(group)) form_.category = group;
        }
        ImGui::EndPopup();
    }

    ui::FieldLabel("Description");
    ImGui::InputTextMultiline("##description", &form_.description,
                              ImVec2(-1.0f, ImGui::GetTextLineHeight() * 2.0f + ImGui::GetStyle().FramePadding.y * 2.0f));

    if (form_.kind == nm::Kind::Layer) {
        ImGui::Spacing();
        if (ImGui::BeginTable("##layer_options", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ui::FieldLabel("Shown as");
            const char* roles[] = {"Layer", "Activation"};
            int role = form_.shown_as_activation ? 1 : 0;
            if (ui::SegmentedControl("##role", roles, 2, &role)) form_.shown_as_activation = role == 1;
            ImGui::TableNextColumn();
            ui::FieldLabel("Output shape");
            const char* shapes[] = {"Same as input", "I compute it"};
            int shape = form_.computes_shape ? 1 : 0;
            if (ui::SegmentedControl("##shape", shapes, 2, &shape)) {
                form_.computes_shape = shape == 1;
                if (form_.computes_shape && form_.bodies["output_shape"].empty())
                    form_.bodies["output_shape"] = "return list(input_shape)";
            }
            ImGui::EndTable();
        }
    } else if (form_.kind == nm::Kind::Loss) {
        ImGui::Spacing();
        ui::FieldLabel("Targets");
        const char* targets[] = {"Classes (one-hot)", "Values"};
        int target = form_.continuous_targets ? 1 : 0;
        if (ui::SegmentedControl("##targets", targets, 2, &target)) form_.continuous_targets = target == 1;
    }
}

void PythonNodeMakerPanel::RenderSettings() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    Section("Settings");
    if (form_.kind == nm::Kind::Activation) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim,
                           "An activation has no settings: you write f(x) and its derivative, and the Engine does "
                           "the rest. Need a setting?");
        ImGui::PopTextWrapPos();
        if (ui::LinkButton("Make it a Layer shown as Activation")) {
            ConfirmIfEdited("A Layer starts its code over: f and df become forward and backward. Switch?", "Switch",
                            [this]() {
                                nm::NodeForm layer = nm::WithKind(form_, nm::Kind::Layer);
                                layer.shown_as_activation = true;
                                form_ = layer;
                            });
        }
        return;
    }
    if (form_.settings.empty()) ImGui::TextColored(t.text_faint, "No settings. They show in Properties when you add them.");
    size_t remove_at = form_.settings.size();
    for (size_t i = 0; i < form_.settings.size(); ++i) {
        bool remove = false;
        RenderSetting(form_.settings[i], i, remove);
        if (remove) remove_at = i;
    }
    if (remove_at < form_.settings.size()) form_.settings.erase(form_.settings.begin() + static_cast<std::ptrdiff_t>(remove_at));
    if (ui::LinkButton("+ Add setting")) {
        nm::Setting s;
        s.name = "setting" + std::to_string(form_.settings.size() + 1);
        s.default_value = "1.0";
        form_.settings.push_back(s);
    }
}

void PythonNodeMakerPanel::RenderSetting(nm::Setting& s, size_t index, bool& remove) {
    ImGui::PushID(static_cast<int>(index));
    ui::BeginCard("##setting");
    if (ImGui::BeginTable("##row", 4, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("type", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("default", ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableSetupColumn("remove", ImGuiTableColumnFlags_WidthFixed, ui::ButtonWidth("Remove", ui::ButtonSize::Small));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ui::FieldLabel("Name");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##name", &s.name);
        ImGui::TableNextColumn();
        ui::FieldLabel("Type");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("##type", nm::SettingTypeLabel(s.type))) {
            for (auto type : {nm::SettingType::Number, nm::SettingType::WholeNumber, nm::SettingType::OnOff,
                              nm::SettingType::Text, nm::SettingType::Choice}) {
                if (ImGui::Selectable(nm::SettingTypeLabel(type), type == s.type) && type != s.type) {
                    s.type = type;
                    s.lowest.clear();
                    s.highest.clear();
                    s.default_value = type == nm::SettingType::Number        ? "1.0"
                                      : type == nm::SettingType::WholeNumber ? "1"
                                      : type == nm::SettingType::OnOff       ? "true"
                                                                             : std::string();
                    if (type == nm::SettingType::Choice) {
                        if (s.choices.empty()) s.choices = {"first", "second"};
                        s.default_value = s.choices.front();
                    }
                }
            }
            ImGui::EndCombo();
        }
        ImGui::TableNextColumn();
        ui::FieldLabel("Default");
        switch (s.type) {
            case nm::SettingType::Number: {
                double value = std::strtod(s.default_value.c_str(), nullptr);
                ui::NumberFieldOptions options;
                options.step = 0.1;
                options.format = "%g";
                options.width = 56.0f;
                if (ui::NumberField("##default", &value, options)) s.default_value = FormatNumber(value);
                break;
            }
            case nm::SettingType::WholeNumber: {
                int value = std::atoi(s.default_value.c_str());
                if (ui::NumberField("##default", &value, -1000000000, 1000000000, 1, 56.0f)) s.default_value = std::to_string(value);
                break;
            }
            case nm::SettingType::OnOff: {
                bool on = s.default_value == "true";
                if (ImGui::Checkbox("On##default", &on)) s.default_value = on ? "true" : "false";
                break;
            }
            case nm::SettingType::Text:
                ImGui::SetNextItemWidth(-1.0f);
                ImGui::InputText("##default", &s.default_value);
                break;
            case nm::SettingType::Choice:
                ImGui::SetNextItemWidth(-1.0f);
                if (ImGui::BeginCombo("##default", s.default_value.c_str())) {
                    for (const auto& choice : s.choices) {
                        if (ImGui::Selectable(choice.c_str(), choice == s.default_value)) s.default_value = choice;
                    }
                    ImGui::EndCombo();
                }
                break;
        }
        ImGui::TableNextColumn();
        ui::FieldLabel(" ");
        if (ui::DangerButton("Remove")) remove = true;
        ImGui::EndTable();
    }
    const bool numeric = s.type == nm::SettingType::Number || s.type == nm::SettingType::WholeNumber;
    if (ImGui::BeginTable("##more", numeric ? 3 : 1, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableNextRow();
        if (numeric) {
            ImGui::TableNextColumn();
            ui::FieldLabel("Lowest");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputTextWithHint("##lowest", "none", &s.lowest, ImGuiInputTextFlags_CharsScientific);
            ImGui::TableNextColumn();
            ui::FieldLabel("Highest");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputTextWithHint("##highest", "none", &s.highest, ImGuiInputTextFlags_CharsScientific);
        }
        ImGui::TableNextColumn();
        ui::FieldLabel("Description");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##description", "Shown in Properties", &s.description);
        ImGui::EndTable();
    }
    if (s.type == nm::SettingType::Choice) {
        ui::FieldLabel("Choices");
        size_t drop = s.choices.size();
        for (size_t c = 0; c < s.choices.size(); ++c) {
            ImGui::PushID(static_cast<int>(c));
            ImGui::SetNextItemWidth(-ui::ButtonWidth("Remove", ui::ButtonSize::Small) - ImGui::GetStyle().ItemSpacing.x);
            const std::string before = s.choices[c];
            if (ImGui::InputText("##choice", &s.choices[c]) && s.default_value == before) s.default_value = s.choices[c];
            ImGui::SameLine();
            if (ui::DangerButton("Remove", s.choices.size() > 1, "A choice setting keeps at least one choice")) drop = c;
            ImGui::PopID();
        }
        if (drop < s.choices.size()) {
            const bool was_default = s.choices[drop] == s.default_value;
            s.choices.erase(s.choices.begin() + static_cast<std::ptrdiff_t>(drop));
            if (was_default && !s.choices.empty()) s.default_value = s.choices.front();
        }
        if (ui::LinkButton("+ Add choice")) s.choices.push_back("choice" + std::to_string(s.choices.size() + 1));
    }
    ui::EndCard();
    ImGui::PopID();
}

void PythonNodeMakerPanel::RenderPins() {
    const ui::Tokens& t = ui::CurrentTokens();
    Section("Pins (set by the kind)");
    ui::BeginCard("##pins");
    if (ImGui::BeginTable("##pin_row", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("in", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("node", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("out", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        for (const auto& pin : nm::InputPins(form_.kind)) {
            const float w = ImGui::CalcTextSize(pin.name.c_str()).x + ImGui::GetTextLineHeight();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
            PinLine(pin, true);
        }
        ImGui::TableNextColumn();
        const std::string title = form_.display_name.empty() ? std::string("(no name)") : form_.display_name;
        const std::string sub = (form_.category.empty() ? std::string("Custom") : form_.category) + " \xC2\xB7 Python";
        ImGui::TextColored(t.text_bright, "%s", title.c_str());
        ImGui::TextColored(t.text_dim, "%s", sub.c_str());
        ImGui::TableNextColumn();
        for (const auto& pin : nm::OutputPins(form_.kind)) PinLine(pin, false);
        ImGui::EndTable();
    }
    ui::EndCard();
}

void PythonNodeMakerPanel::RenderCode() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    Section("Code");
    const float link = ui::ButtonWidth("Open in Script Editor", ui::ButtonSize::Small);
    ui::SameLineRight(link);
    if (ui::LinkButton("Open in Script Editor") && open_in_script_editor_) {
        fs::path written;
        std::string error;
        if (WriteSource(written, error)) {
            open_in_script_editor_(written.string());
        } else {
            result_ = Result::Error;
            result_title_ = "Could not write the source";
            result_detail_ = error;
        }
    }
    // The decorator and settings follow the form; shown, not edited here.
    const std::string source = nm::Generate(form_, nm::BundleIdFor(ProjectName(), nm::FileStemFor(form_.class_name)));
    std::string header = HeaderOf(source);
    while (!header.empty() && header.back() == '\n') header.pop_back();
    ImFont* code = cyxwiz::gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_bar);
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    const int header_lines = static_cast<int>(std::count(header.begin(), header.end(), '\n')) + 1;
    ImGui::InputTextMultiline("##header", &header,
                              ImVec2(-1.0f, ImGui::GetTextLineHeight() * static_cast<float>(header_lines) +
                                                ImGui::GetStyle().FramePadding.y * 2.0f),
                              ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor(2);
    if (code) ImGui::PopFont();
    for (const auto& method : nm::MethodNames(form_)) {
        ImGui::PushID(method.c_str());
        if (code) ImGui::PushFont(code);
        ImGui::TextColored(t.accent_text, "    %s", nm::MethodSignature(form_, method).c_str());
        if (code) ImGui::PopFont();
        // The body sits under its def, as it does in the file.
        const float indent = (code ? code->CalcTextSizeA(code->FontSize, FLT_MAX, 0.0f, "        ").x
                                   : ImGui::CalcTextSize("        ").x);
        ImGui::Indent(indent);
        CodeField("##body", form_.bodies[method], 2);
        ImGui::Unindent(indent);
        ImGui::PopID();
    }
}

void PythonNodeMakerPanel::RenderFooter(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto issues = nm::Validate(form_);
    const bool busy = call_ != Call::None;
    std::string blocker = issues.empty() ? ConflictReason() : issues.front();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::BeginChild("##maker_footer", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    const float buttons = ui::ButtonWidth("Check", ui::ButtonSize::Regular) +
                          ui::ButtonWidth("Create node", ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.x * 3.0f;

    ImVec4 dot = t.pending;
    std::string title = "Not checked yet";
    std::string detail = "Check compares backward with small changes of the input (and of every learned value) "
                         "before you create the node.";
    if (!blocker.empty() && result_ != Result::Busy) {
        dot = t.warning;
        title = "Not ready yet";
        detail = blocker;
    } else if (result_ != Result::None) {
        title = result_title_;
        detail = result_detail_;
        dot = result_ == Result::Passed || result_ == Result::Created ? t.success
              : result_ == Result::Busy                              ? t.running
              : result_ == Result::Failed                            ? t.caution
                                                                     : t.error;
    } else if (!HasProject()) {
        title = "Open a project to check and create a node";
        detail = "Python runs in the project's environment, and the node is written into its nodes folder. "
                 "You can write it now.";
    }
    if (ImGui::BeginTable("##footer_row", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("result", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("actions", ImGuiTableColumnFlags_WidthFixed, buttons);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const float r = ImGui::GetTextLineHeight() * 0.3f;
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(at.x + r, at.y + ImGui::GetTextLineHeight() * 0.5f), r,
                                                    ImGui::GetColorU32(dot));
        ImGui::Indent(r * 2.0f + ImGui::GetStyle().ItemSpacing.x);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(result_ == Result::None && blocker.empty() ? t.text : dot, "%s", title.c_str());
        ImGui::TextColored(t.text_dim, "%s", detail.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Unindent(r * 2.0f + ImGui::GetStyle().ItemSpacing.x);
        ImGui::TableNextColumn();
        const char* check_reason = busy               ? "Wait for the running step"
                                   : !blocker.empty() ? blocker.c_str()
                                   : !HasProject()    ? "Open a project first: Python runs in the project's environment"
                                                      : nullptr;
        if (ui::SecondaryButton("Check", check_reason == nullptr, check_reason, ui::ButtonSize::Regular))
            StartCheck();
        ImGui::SameLine();
        const char* reason = busy                ? "Wait for the running step"
                             : !blocker.empty()  ? blocker.c_str()
                             : !HasProject()     ? "Open a project first: the node goes into its nodes folder"
                                                 : nullptr;
        if (ui::PrimaryButton("Create node", reason == nullptr, reason)) StartCreate();
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

}  // namespace cyxwiz
