// The Variables view (TOFIX133 P5, approved board 9). See variables_view.h.

#include "variables_view.h"

#include "../core/file_dialogs.h"
#include "../scripting/scripting_engine.h"
#include "editor_fonts.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <utility>

namespace cyxwiz {

namespace {
}  // namespace

std::string ClockNow() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &local);
    return buf;
}

namespace {

// The JSON string a cyxwiz_vars function returned ("" when it is not a string).
std::string JsonString(const std::string& text) {
    const auto doc = nlohmann::json::parse(text, nullptr, false);
    return !doc.is_discarded() && doc.is_string() ? doc.get<std::string>() : std::string();
}

ImVec4 KindColour(const std::string& kind) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (kind == "number") return t.info;
    if (kind == "text") return t.warning;
    if (kind == "collection") return t.caution;
    if (kind == "table") return t.success;
    if (kind == "array") return t.accent_text;
    if (kind == "module" || kind == "function" || kind == "class") return t.text_dim;
    return t.text;
}
}  // namespace

std::map<std::uint64_t, VariablesView::Route>& VariablesView::Routes() {
    static std::map<std::uint64_t, Route> routes;
    return routes;
}

std::uint64_t VariablesView::Read(scripting::ScriptingEngine* engine, scripting::VariablesService::Request request,
                                  const void* owner, std::function<void(const scripting::VariablesService::Result&)> done) {
    if (!engine) return 0;
    const std::uint64_t id = engine->Variables().Submit(std::move(request));
    if (id != 0) Routes()[id] = Route{owner, std::move(done)};
    return id;
}

void VariablesView::CancelReads(const void* owner) {
    auto& routes = Routes();
    for (auto it = routes.begin(); it != routes.end();) {
        if (it->second.owner == owner) it = routes.erase(it);
        else ++it;
    }
}

VariablesView::VariablesView() = default;

VariablesView::~VariablesView() { CancelReads(this); }

void VariablesView::SetScope(const Scope& scope) {
    if (scope.key == scope_.key) {  // same namespace: only the words change
        scope_.label = scope.label;
        scope_.detail = scope.detail;
        return;
    }
    scope_ = scope;
    vars_.clear();
    children_.clear();
    open_.clear();
    changed_.clear();
    selected_.clear();
    have_read_ = false;
    list_request_ = 0;
    Invalidate("on opening");
}

void VariablesView::Invalidate(const std::string& reason) {
    read_wanted_ = true;
    read_reason_ = reason;
}

void VariablesView::Forget(const std::string& reason) {
    vars_.clear();
    children_.clear();
    open_.clear();
    changed_.clear();
    digests_.erase(scope_.key);
    selected_.clear();
    Invalidate(reason);
}

std::vector<std::string> VariablesView::Names() const {
    std::vector<std::string> names;
    for (const auto& v : vars_)
        if (v.more == 0) names.push_back(v.name);
    return names;
}

void VariablesView::Submit(scripting::VariablesService::Request request, Pending pending) {
    if (!engine_) return;
    request.scope = scope_.key;
    pending.scope = scope_.key;
    const bool list = pending.kind == Kind::List;
    const std::uint64_t id = Read(engine_, std::move(request), this,
                                  [this, pending](const scripting::VariablesService::Result& r) { OnResult(r, pending); });
    if (id != 0 && list) list_request_ = id;
}

void VariablesView::PollAll(scripting::ScriptingEngine* engine) {
    if (!engine) return;
    auto& routes = Routes();
    if (routes.empty()) return;  // nothing asked: no need to start the worker
    for (const auto& result : engine->Variables().Poll()) {
        auto it = routes.find(result.id);
        if (it == routes.end()) continue;  // whoever asked is gone
        auto done = std::move(it->second.done);
        routes.erase(it);
        if (done) done(result);
    }
}

void VariablesView::OnResult(const scripting::VariablesService::Result& result, const Pending& pending) {
    const bool this_scope = pending.scope == scope_.key;
    switch (result.kind) {
        case Kind::List: {
            if (result.id != list_request_ || !this_scope) return;
            list_request_ = 0;
            if (result.busy) {  // a run started: read when it ends
                read_wanted_ = true;
                return;
            }
            if (result.json.empty()) return;
            vars_ = vars::ParseVariables(result.json);
            auto& previous = digests_[scope_.key];
            changed_ = vars::ChangedNames(previous, vars_);
            previous = vars::Digests(vars_);
            have_read_ = true;
            status_ = vars::ReadStatus(pending.name, ClockNow());
            children_.clear();
            // Open rows stay open: read their children again.
            for (const auto& path : open_) {
                scripting::VariablesService::Request r;
                r.kind = Kind::Children;
                r.path_json = path;
                Submit(std::move(r), Pending{Kind::Children, path, {}, {}});
            }
            break;
        }
        case Kind::Children:
            if (!this_scope || result.busy) return;
            children_[pending.path] = vars::ParseVariables(result.json.empty() ? "[]" : result.json);
            break;
        case Kind::Value:
            if (result.busy) Note("A run is active; copy when it finishes.");
            else if (!result.json.empty()) {
                ImGui::SetClipboardText(JsonString(result.json).c_str());
                Note("Copied the value of " + pending.name + ".");
            }
            break;
        case Kind::Delete: {
            if (result.busy) {
                Note("A run is active; delete when it finishes.");
                break;
            }
            const std::string why = JsonString(result.json);
            Note(why.empty() ? "Deleted " + pending.name + "." : why);
            if (why.empty() && this_scope) Invalidate("after deleting " + pending.name);
            break;
        }
        case Kind::SaveCsv: {
            if (result.busy) {
                Note("A run is active; save when it finishes.");
                break;
            }
            const std::string why = result.json.empty() ? std::string("Could not save (see the log).") : JsonString(result.json);
            Note(why.empty() ? "Saved " + pending.name + "." : why);
            break;
        }
        case Kind::Table:
            if (result.busy) Note("A run is active; view the data when it finishes.");
            else if (!result.error.empty() || !result.table) Note(result.error.empty() ? "Could not read the data." : result.error);
            else if (on_open_table) {
                OpenRequest open;
                open.name = pending.name;
                open.path = pending.path;
                open.scope = scope_;
                if (!this_scope) open.scope.key = pending.scope;
                open.plot = pending.plot;
                on_open_table(open, result);
            }
            break;
    }
}

void VariablesView::Note(const std::string& text) {
    note_ = text;
    note_until_ = ImGui::GetTime() + 4.0;
}

void VariablesView::ReadIfNeeded() {
    const bool running = engine_ && ((engine_->IsScriptRunning() && !engine_->IsDebugPaused()) || engine_->IsCommandRunning());
    if (was_running_ && !running) Invalidate(engine_->IsDebugPaused() ? "while paused" : "after the run finished");
    was_running_ = running;
    const int frame = ImGui::GetFrameCount();
    if (last_frame_ >= 0 && frame > last_frame_ + 1 && have_read_) Invalidate("on opening");
    last_frame_ = frame;
    if (!read_wanted_ || running || list_request_ != 0 || !engine_ || !engine_->IsInitialized()) return;
    read_wanted_ = false;
    scripting::VariablesService::Request r;
    r.kind = Kind::List;
    r.show_all = show_all_;
    Submit(std::move(r), Pending{Kind::List, {}, read_reason_, {}});
}

void VariablesView::Render(float height) {
    ReadIfNeeded();
    ImGui::PushID(this);
    ImGui::BeginChild("##variables_view", ImVec2(0.0f, height), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    RenderHeader();
    RenderChips();
    const float footer_h = ImGui::GetFrameHeight() + 4.0f;
    RenderTable(std::max(40.0f, ImGui::GetContentRegionAvail().y - footer_h));
    RenderFooter();
    ImGui::EndChild();
    ImGui::PopID();
}

void VariablesView::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    if (!title_.empty()) {
        {
            ui::FontScope bold(ui::Font::Bold);
            ImGui::TextUnformatted(title_.c_str());
        }
        ImGui::SameLine(0.0f, 12.0f);
    }
    if (scopes) {
        const std::string text = scope_.label + (scope_.detail.empty() ? "" : "  \xC2\xB7 " + scope_.detail) + "  " ICON_FA_CHEVRON_DOWN;
        if (ui::StatusPill("##scope", text.c_str(), engine_ && engine_->IsInitialized() ? t.success : t.pending))
            ImGui::OpenPopup("##scopes");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Whose variables: the Python session or an open notebook");
        if (ImGui::BeginPopup("##scopes")) {
            for (const auto& s : scopes()) {
                const std::string label = s.label + (s.detail.empty() ? "" : "  \xC2\xB7 " + s.detail);
                if (ImGui::Selectable(label.c_str(), s.key == scope_.key)) SetScope(s);
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine(0.0f, 12.0f);
    } else if (!title_.empty()) {
        ImGui::TextColored(t.text_dim, "%s", scope_.label.c_str());
        ImGui::SameLine(0.0f, 12.0f);
    }
    const bool running = engine_ && ((engine_->IsScriptRunning() && !engine_->IsDebugPaused()) || engine_->IsCommandRunning());
    const char* status = !engine_ || !engine_->IsInitialized() ? "Python has not started"
                         : running                            ? "Running... values from before the run"
                         : list_request_ != 0 && !have_read_  ? "Reading..."
                                                              : status_.c_str();
    ImGui::TextColored(t.text_dim, "%s", status);

    const float refresh_w = ui::ButtonWidth(ICON_FA_ROTATE " Refresh", ui::ButtonSize::Small);
    const float filter_w = std::min(240.0f, std::max(120.0f, ImGui::GetContentRegionAvail().x * 0.3f));
    const float right = ImGui::GetContentRegionMax().x;
    (void)right;
    ui::SameLineRight(filter_w + refresh_w + 8.0f);
    ui::SearchField("##variables_filter", filter_, sizeof(filter_), "Filter by name, type or value", filter_w);
    ImGui::SameLine(0.0f, 8.0f);
    if (ui::GhostButton(ICON_FA_ROTATE " Refresh", engine_ && engine_->IsInitialized() && !running,
                        running ? "A run is active; the view reads when it finishes" : "Python has not started"))
        Invalidate("on Refresh");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Read the variables again");
}

void VariablesView::RenderChips() {
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    for (int i = 0; i < vars::kChipCount; ++i) {
        const auto chip = static_cast<vars::Chip>(i);
        if (i > 0) ImGui::SameLine(0.0f, 6.0f);
        const std::string count = std::to_string(vars::ChipCount(vars_, chip));
        const std::string id = std::string("##chip") + std::to_string(i);
        if (ui::FilterChip(id.c_str(), vars::ChipLabel(chip), count.c_str(), chip_ == chip)) chip_ = chip;
    }
    const char* label = "Show modules, functions and classes";
    const float w = ImGui::GetFrameHeight() + ImGui::CalcTextSize(label).x + 8.0f;
    ui::SameLineRight(w);
    if (ImGui::Checkbox(label, &show_all_)) Invalidate("on Refresh");
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
}

void VariablesView::RenderTable(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!engine_ || !engine_->IsInitialized()) {
        ImGui::BeginChild("##empty", ImVec2(0.0f, height));
        ui::EmptyState(ICON_FA_LIST_UL, "Python has not started yet.",
                       "Run a script, a cell or a Console command: its variables show here.");
        ImGui::EndChild();
        return;
    }
    std::vector<vars::Variable> top;
    for (const auto& v : vars_)
        if (vars::Matches(v, filter_, chip_)) top.push_back(v);
    vars::Sort(top, sort_column_, sort_ascending_);
    const auto rows = vars::Flatten(top, open_, children_);

    if (vars_.empty() || top.empty()) {
        ImGui::BeginChild("##empty", ImVec2(0.0f, height));
        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        if (!have_read_) ImGui::TextColored(t.text_dim, "%s", "Reading...");
        else if (vars_.empty()) ImGui::TextColored(t.text_dim, "%s", "No variables yet. Run some code: what it makes shows here.");
        else ImGui::TextColored(t.text_dim, "No variable matches \"%s\" in %s.", filter_, vars::ChipLabel(chip_));
        ImGui::EndChild();
        return;
    }

    // The rows to draw: tree rows, plus a "Reading" line under an open row
    // whose children have not arrived.
    struct Line {
        const vars::TreeRow* row;
        bool placeholder;
    };
    std::vector<Line> lines;
    lines.reserve(rows.size());
    for (const auto& r : rows) {
        lines.push_back({&r, false});
        if (r.loading) lines.push_back({&r, true});
    }

    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                                  ImGuiTableFlags_Sortable | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody;
    ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
    bool open_menu = false;
    if (ImGui::BeginTable("##vars", 6, flags, ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 220.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 80.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##view", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort |
                                              ImGuiTableColumnFlags_NoResize, 26.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
            if (specs->SpecsDirty && specs->SpecsCount > 0) {
                const int column = specs->Specs[0].ColumnIndex;
                if (column <= 4) sort_column_ = static_cast<vars::Column>(column);
                sort_ascending_ = specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
            }
            specs->SpecsDirty = false;
        }

        ImFont* mono = gui::GetCodeFont();
        const float row_h = ImGui::GetFrameHeight();
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(lines.size()), row_h);
        while (clipper.Step()) {
            for (int li = clipper.DisplayStart; li < clipper.DisplayEnd; ++li) {
                const Line& line = lines[static_cast<size_t>(li)];
                const vars::TreeRow& row = *line.row;
                const vars::Variable& v = *row.var;
                ImGui::TableNextRow(ImGuiTableRowFlags_None, row_h);
                ImGui::TableSetColumnIndex(0);
                const float indent = 18.0f * static_cast<float>(row.depth + (line.placeholder ? 1 : 0));
                if (line.placeholder || v.more != 0) {
                    ImGui::AlignTextToFramePadding();
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + indent + 26.0f);
                    if (line.placeholder) ImGui::TextColored(t.text_faint, "%s", "Reading...");
                    else ImGui::TextColored(t.text_faint, "+ %lld more (the first 200 are shown)", v.more);
                    continue;
                }
                ImGui::PushID(row.path.c_str());
                const bool selected = selected_ == row.path;
                const ImVec2 cell = ImGui::GetCursorScreenPos();
                if (ImGui::Selectable("##row", selected,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap |
                                          ImGuiSelectableFlags_AllowDoubleClick,
                                      ImVec2(0.0f, row_h))) {
                    selected_ = row.path;
                    const bool on_chevron = ImGui::GetIO().MousePos.x < cell.x + indent + 22.0f;
                    if (v.expandable && (on_chevron || (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && !v.viewable)))
                        ToggleOpen(row.path);
                    else if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && v.viewable)
                        ViewData(row.path, v.name);
                }
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    selected_ = row.path;
                    menu_path_ = row.path;
                    open_menu = true;
                }
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const float mid = cell.y + row_h * 0.5f;
                float x = cell.x + indent;
                if (changed_.count(v.name) && row.depth == 0)
                    dl->AddCircleFilled(ImVec2(x + 3.0f, mid), 3.0f, ui::ToU32(t.accent_text));
                x += 10.0f;
                if (v.expandable) {
                    const ImU32 c = ui::ToU32(t.text_dim);
                    if (row.open)
                        dl->AddTriangleFilled(ImVec2(x + 1.0f, mid - 2.5f), ImVec2(x + 9.0f, mid - 2.5f), ImVec2(x + 5.0f, mid + 2.5f), c);
                    else
                        dl->AddTriangleFilled(ImVec2(x + 2.5f, mid - 4.0f), ImVec2(x + 7.5f, mid), ImVec2(x + 2.5f, mid + 4.0f), c);
                }
                x += 16.0f;
                const float font = ImGui::GetFontSize();
                ImFont* name_font = mono ? mono : ImGui::GetFont();
                dl->AddText(name_font, font, ImVec2(x, mid - font * 0.5f), ui::ToU32(t.text_bright), v.name.c_str());

                ImGui::TableSetColumnIndex(1);
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(KindColour(v.kind), "%s", v.type.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_dim, "%s", v.size.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::AlignTextToFramePadding();
                const std::string mem = vars::MemoryText(v.memory);
                if (!mem.empty()) {
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(mem.c_str()).x - 6.0f);
                    ImGui::TextColored(t.text_dim, "%s", mem.c_str());
                }
                ImGui::TableSetColumnIndex(4);
                ImGui::AlignTextToFramePadding();
                if (mono) ImGui::PushFont(mono);
                ImGui::TextColored(ui::Mix(t.text, t.text_dim, 0.4f), "%s", v.value.c_str());
                if (mono) ImGui::PopFont();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 36.0f);
                    ImGui::TextUnformatted(v.value.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::TableSetColumnIndex(5);
                if (v.viewable) {
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    if (ImGui::InvisibleButton("##view", ImVec2(22.0f, row_h))) {
                        selected_ = row.path;
                        ViewData(row.path, v.name);
                    }
                    const bool hovered = ImGui::IsItemHovered();
                    if (hovered) dl->AddRectFilled(p, ImVec2(p.x + 22.0f, p.y + row_h), ui::ToU32(t.hover), 4.0f);
                    const ImVec2 ts = ImGui::CalcTextSize(ICON_FA_TABLE);
                    dl->AddText(ImVec2(p.x + (22.0f - ts.x) * 0.5f, mid - ts.y * 0.5f), ui::ToU32(t.accent_text), ICON_FA_TABLE);
                    if (hovered) ImGui::SetTooltip("View data (double-click or Enter)");
                }
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleColor(3);

    if (open_menu) ImGui::OpenPopup("##row_menu");
    if (ImGui::BeginPopup("##row_menu")) {
        for (const auto& r : rows)
            if (r.path == menu_path_ && r.var->more == 0) {
                RenderRowMenu(r);
                break;
            }
        ImGui::EndPopup();
    }
    HandleKeys(rows);
}

void VariablesView::RenderRowMenu(const vars::TreeRow& row) {
    const ui::Tokens& t = ui::CurrentTokens();
    const vars::Variable& v = *row.var;
    auto gap = []() { ImGui::Dummy(ImVec2(0.0f, 4.0f)); };
    if (v.viewable && ImGui::MenuItem("View data", "Enter")) ViewData(row.path, v.name);
    if (v.viewable && (v.kind == "table" || v.kind == "array") && ImGui::MenuItem("Plot"))
        ViewData(row.path, v.name, true);
    if (v.expandable && ImGui::MenuItem(row.open ? "Collapse" : "Expand", row.open ? "Left" : "Right")) ToggleOpen(row.path);
    gap();
    if (ImGui::MenuItem("Copy name")) ImGui::SetClipboardText(v.name.c_str());
    if (ImGui::MenuItem("Copy value", "Ctrl+C")) CopyValue(row.path);
    if (row.depth == 0 && on_insert_name && ImGui::MenuItem("Insert name in the editor")) on_insert_name(v.name);
    if (v.viewable && ImGui::MenuItem("Save as CSV...")) SaveCsv(row.path, v.name);
    if (row.depth == 0) {
        gap();
        ImGui::PushStyleColor(ImGuiCol_Text, t.error);
        if (ImGui::MenuItem("Delete variable", "Delete")) Delete(row.path, v.name);
        ImGui::PopStyleColor();
    }
}

void VariablesView::HandleKeys(const std::vector<vars::TreeRow>& rows) {
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) || ImGui::IsAnyItemActive() || selected_.empty()) return;
    int index = -1;
    for (size_t i = 0; i < rows.size(); ++i)
        if (rows[i].path == selected_ && rows[i].var->more == 0) index = static_cast<int>(i);
    if (index < 0) return;
    const vars::TreeRow& row = rows[static_cast<size_t>(index)];
    const ImGuiIO& io = ImGui::GetIO();
    auto move = [&](int step) {
        for (int i = index + step; i >= 0 && i < static_cast<int>(rows.size()); i += step)
            if (rows[static_cast<size_t>(i)].var->more == 0) {
                selected_ = rows[static_cast<size_t>(i)].path;
                return;
            }
    };
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) move(1);
    else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) move(-1);
    else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false) && row.var->expandable && !row.open) ToggleOpen(row.path);
    else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false) && row.open) ToggleOpen(row.path);
    else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) {
        if (row.var->viewable) ViewData(row.path, row.var->name);
        else if (row.var->expandable) ToggleOpen(row.path);
    } else if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) && row.depth == 0) {
        Delete(row.path, row.var->name);
    } else if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C, false)) {
        CopyValue(row.path);
    }
}

void VariablesView::RenderFooter() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", vars::FooterText(vars_).c_str());
    if (!changed_.empty()) {
        ImGui::SameLine(0.0f, 16.0f);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float mid = p.y + ImGui::GetFrameHeight() * 0.5f;
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + 3.0f, mid), 3.0f, ui::ToU32(t.accent_text));
        ImGui::SetCursorScreenPos(ImVec2(p.x + 10.0f, p.y));
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", "changed in the last run");
    }
    const bool noting = !note_.empty() && ImGui::GetTime() < note_until_;
    const char* hint = noting ? note_.c_str() : "Double-click or Enter: view data \xC2\xB7 right-click: more";
    const float w = ImGui::CalcTextSize(hint).x;
    if (ui::SameLineRight(w, 16.0f) || noting) ImGui::TextColored(noting ? t.text : t.text_faint, "%s", hint);
}

void VariablesView::ToggleOpen(const std::string& path) {
    if (open_.erase(path)) return;
    open_.insert(path);
    if (children_.count(path)) return;
    scripting::VariablesService::Request r;
    r.kind = Kind::Children;
    r.path_json = path;
    Submit(std::move(r), Pending{Kind::Children, path, {}, {}});
}

void VariablesView::ViewData(const std::string& path, const std::string& name, bool plot) {
    scripting::VariablesService::Request r;
    r.kind = Kind::Table;
    r.path_json = path;
    Submit(std::move(r), Pending{Kind::Table, path, name, {}, plot});
    Note("Reading " + name + "...");
}

void VariablesView::CopyValue(const std::string& path) {
    std::string name;
    for (const auto& v : vars_)
        if (vars::PathJson("", v.step) == path) name = v.name;
    scripting::VariablesService::Request r;
    r.kind = Kind::Value;
    r.path_json = path;
    Submit(std::move(r), Pending{Kind::Value, path, name.empty() ? std::string("the value") : name, {}});
}

void VariablesView::SaveCsv(const std::string& path, const std::string& name) {
    const std::string file_name = name + ".csv";
    const auto file = FileDialogs::SaveFile("Save as CSV", {{"CSV files", "csv"}}, nullptr, file_name.c_str());
    if (!file) return;
    scripting::VariablesService::Request r;
    r.kind = Kind::SaveCsv;
    r.path_json = path;
    r.file = *file;
    Submit(std::move(r), Pending{Kind::SaveCsv, path, name, {}});
}

void VariablesView::Delete(const std::string& path, const std::string& name) {
    scripting::VariablesService::Request r;
    r.kind = Kind::Delete;
    r.path_json = path;
    Submit(std::move(r), Pending{Kind::Delete, path, name, {}});
}

}  // namespace cyxwiz
