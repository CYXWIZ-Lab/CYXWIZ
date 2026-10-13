#include "dashboard_window.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/column_role_store.h"
#include "../../core/data_registry.h"
#include "../../core/dashboard/sparse_summary.h"
#include "../../core/dashboard/text_words.h"
#include "../../core/project_manager.h"
#include "../../core/dataset_catalog.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../plot/plot_view.h"
#include "dashboard_links.h"
#include "../plot/plot_window.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../separate_windows.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <optional>

namespace cyxwiz::dashboard {

namespace {

std::string Thousands(double v) {
    if (!std::isfinite(v)) return "";
    const long long n = std::llround(v);
    std::string s = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + s : s;
}

std::string Short(double v) {
    if (!std::isfinite(v)) return "n/a";
    char buf[32];
    if (std::fabs(v) >= 1e6) std::snprintf(buf, sizeof(buf), "%.3g", v);
    else if (std::fabs(v - std::round(v)) < 1e-9 && std::fabs(v) >= 100) return Thousands(v);
    else std::snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

ImVec4 RoleColour(ColumnRole role) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (role) {
        case ColumnRole::Target: return t.success;
        case ColumnRole::Numeric: return t.series[1];
        case ColumnRole::Category: return t.series[0];
        case ColumnRole::DateTime: return t.series[2];
        case ColumnRole::Id:
        case ColumnRole::Ignore: return t.text_faint;
        default: return t.text_dim;
    }
}

bool NumberType(const std::string& type) {
    return type == "int" || type == "float";
}

std::string WidgetTitle(const WidgetSpec& w) {
    if (!w.title.empty()) return w.title;
    switch (w.type) {
        case WidgetType::Kpi: return std::string(MeasureLabel(w.measure)) + (w.field.empty() ? "" : " of " + w.field);
        case WidgetType::Table: return "Table";
        case WidgetType::Missing: return "Missing values";
        case WidgetType::Plot: {
            if (w.IsText()) return std::string(TextViewLabel(w.text_view)) + " \xC2\xB7 " + w.text_field;
            const std::string col = !w.plot.x_column.empty() ? w.plot.x_column : !w.plot.y_columns.empty() ? w.plot.y_columns.front() : "";
            return col.empty() ? KindOf(w).label : col;
        }
    }
    return "Widget";
}

}  // namespace

DashboardWindow::DashboardWindow(std::string id, Mode mode) : id_(std::move(id)), mode_(mode) {}

void DashboardWindow::Select(const std::string& widget_id) {
    selected_ = widget_id;
    focus_ = true;
    scroll_to_selected_ = true;
}

void DashboardWindow::RemoveWidget(const std::string& id) {
    spec_.filters.ClearWidget(id);
    spec_.widgets.erase(std::remove_if(spec_.widgets.begin(), spec_.widgets.end(), [&](const WidgetSpec& x) { return x.id == id; }),
                        spec_.widgets.end());
    views_.erase(id);
    if (selected_ == id) selected_.clear();
}

DashboardWindow::~DashboardWindow() {
    if (profile_task_) AsyncTaskManager::Instance().Cancel(profile_task_);
}

void DashboardWindow::SetSpecJson(const std::string& json) {
    if (json.empty()) return;
    DashboardSpec spec;
    std::string problem;
    if (DashboardFromJson(json, spec, &problem)) {
        spec_ = std::move(spec);
        saved_json_ = json;
    } else {
        spdlog::warn("Dashboard: the saved layout could not be read ({}); starting a new one", problem);
    }
}

void DashboardWindow::SetData(const std::string& dataset_name, const std::string& title) {
    title_ = title;
    message_.clear();
    if (dataset_name != dataset_) {
        dataset_ = dataset_name;
        session_.SetDataset(dataset_);
        profile_.reset();
        profiled_generation_ = ~0ull;
        views_.clear();
        view_versions_.clear();
        // Sparse features: their own summary (no SQL).
        sparse_ = DataRegistry::Instance().IsSparseFeatureDataset(dataset_);
        sparse_summary_.reset();
        sparse_views_.clear();
        sparse_dirty_ = true;
        sparse_error_.clear();
        words_started_.clear();
        words_generation_ = ~0ull;
    }
}

std::string DashboardWindow::ShownName() const {
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_);
    return entry ? entry->Shown() : dataset_;
}

void DashboardWindow::FinishExport() {
    if (!capture_ || !capture_->ready) return;
    const std::vector<unsigned char> png = std::move(capture_->png);
    capture_.reset();
    const plot::ViewHooks& h = plot::Hooks();
    if (png.empty()) {
        note_ = "Could not read the dashboard image.";
    } else if (h.save_path) {
        std::string name = "dashboard";
        if (!title_.empty()) name += "_" + title_;
        for (char& c : name)
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') c = '_';
        if (auto path = h.save_path("Save dashboard as PNG", "png", name + ".png")) {
            std::ofstream(*path, std::ios::binary).write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
            note_ = "Saved " + *path;
        } else {
            note_.clear();
        }
    }
    note_until_ = ImGui::GetTime() + 4.0;
    spdlog::info("Dashboard '{}': {} ({} bytes of PNG)", id_, note_, png.size());
}

void DashboardWindow::ClearData(const std::string& message) {
    message_ = message;
}

namespace {

// The catalog's entry with its generation stamped: a dataset registered since
// the catalog last looked still reads generation 0, and a profile started on
// 0 was started again when the real generation came (the double profile).
std::optional<DatasetEntry> ResolveStamped(const std::string& name) {
    auto entry = DatasetCatalog::Instance().Resolve(name);
    if (entry && entry->generation == 0) {
        DatasetCatalog::Instance().Pump(std::chrono::milliseconds(0));
        entry = DatasetCatalog::Instance().Resolve(name);
    }
    return entry;
}

}  // namespace

void DashboardWindow::EnsureWords() {
    const auto entry = ResolveStamped(dataset_);
    if (!entry) return;
    if (entry->generation != words_generation_) {  // new data: its words again
        words_generation_ = entry->generation;
        words_started_.clear();
    }
    for (const auto& field : DashboardSession::TextFields(spec_)) {
        if (!words_started_.insert(field).second) continue;
        session_.SetWordsTable(field, std::string());  // its widgets wait for the words
        const std::string key = WordsCacheKey(entry->source_path, dataset_, entry->generation, field);
        const std::string root = ProjectManager::Instance().GetProjectRoot();
        const std::filesystem::path dir = root.empty() ? std::filesystem::temp_directory_path() / "cyxwiz" / "dashboard_words"
                                                       : std::filesystem::path(root) / "cache" / "dashboard_words";
        auto result = std::make_shared<WordsTable>();
        const std::string table = dataset_;
        std::weak_ptr<int> alive = alive_;
        AsyncTaskManager::Instance().RunAsync(
            "Dashboard: words of " + field,
            [table, field, key, dir, result](LambdaTask&) {
                *result = EnsureWordsTable(table, field, key, dir, [](const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); });
            },
            nullptr,
            [this, alive, result, field](bool, const std::string&) {
                if (alive.expired()) return;
                if (!result->error.empty()) {
                    spdlog::warn("Dashboard: the words of '{}' were not saved ({}); each text widget splits them", field, result->error);
                    session_.ForgetWords(field);
                    return;
                }
                SessionQueryService::Instance().SetSideTable(result->name, result->path, result->rows);
                session_.SetWordsTable(field, result->name);
                spdlog::info("Dashboard: words of '{}' {} ({} rows, {})", field, result->loaded ? "loaded" : "split and saved", result->rows,
                             result->path);
            },
            alive_);
    }
}

void DashboardWindow::EnsureProfile() {
    if (dataset_.empty() || profile_task_) return;
    const auto entry = ResolveStamped(dataset_);
    if (!entry || entry->generation == profiled_generation_) return;
    profiled_generation_ = entry->generation;
    ProfileOptions options;
    if (const auto* s = ProjectColumnRoles().Find(RoleSourceKey(entry->source_path, entry->Shown()))) options.missing_text = s->missing_text;
    auto result = std::make_shared<DatasetProfile>();
    const std::string table = dataset_;
    std::weak_ptr<int> alive = alive_;
    profile_task_ = AsyncTaskManager::Instance().RunAsync(
        "Profile for the dashboard",
        [table, options, result](LambdaTask& task) {
            ProfileOptions opts = options;
            opts.should_stop = [&task] { return task.ShouldStop(); };
            opts.progress = [&task](float f, const std::string& what) { task.ReportProgress(f, what); };
            *result = ProfileTable(table, [](const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }, opts);
            if (!result->ok()) task.MarkFailed(result->error);
            else spdlog::info("Dashboard: profiled '{}' in {:.0f} ms ({} columns)", table, result->elapsed_ms, result->columns.size());
        },
        nullptr,
        [this, alive, result](bool, const std::string&) {
            if (alive.expired()) return;
            profile_task_ = 0;
            if (!result->ok()) {
                profile_error_ = result->error;
                return;
            }
            profile_error_.clear();
            profile_ = result;
            RebuildContract();
            // A new dashboard starts with the automatic layout (before widgets added from Data Studio).
            if (mode_ == Mode::Dashboard && !spec_.automatic_done) {
                auto autos = AutomaticWidgets(*profile_, contract_, spec_);
                autos.insert(autos.end(), spec_.widgets.begin(), spec_.widgets.end());
                spec_.widgets = std::move(autos);
                spec_.automatic_done = true;
                SaveIfChanged();
            }
        },
        alive_);
}

void DashboardWindow::RebuildContract() {
    if (!profile_) return;
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_);
    target_ = entry ? entry->target_column : std::string();
    std::map<std::string, ColumnRole> contract_roles;
    if (!target_.empty()) contract_roles[target_] = ColumnRole::Target;
    const std::string key = entry ? RoleSourceKey(entry->source_path, entry->Shown()) : dataset_;
    contract_ = BuildContract(key, profile_->Facts(), contract_roles, ProjectColumnRoles().RolesFor(key));
}

void DashboardWindow::SaveIfChanged() {
    const std::string json = DashboardToJson(spec_);
    if (json == saved_json_) return;
    saved_json_ = json;
    if (on_spec_changed) on_spec_changed(json);
}

std::string DashboardWindow::FirstField(FieldNeed need, const std::string& other) const {
    const auto fits = [&](const ColumnContract& c) {
        if (c.name == other || c.role == ColumnRole::Id || c.role == ColumnRole::Ignore) return false;
        if (need == FieldNeed::Number && !NumberType(c.type)) return false;
        if (need == FieldNeed::Category && !(c.role == ColumnRole::Category || (c.role == ColumnRole::Target && !NumberType(c.type)))) return false;
        return true;
    };
    // The target first when it fits, then the columns in order.
    for (const auto& c : contract_.columns)
        if (c.role == ColumnRole::Target && fits(c)) return c.name;
    for (const auto& c : contract_.columns)
        if (fits(c)) return c.name;
    return {};
}

void DashboardWindow::AddWidget(const std::string& kind_id) {
    const WidgetKind* kind = FindWidgetKind(kind_id);
    if (!kind) return;
    WidgetSpec w;
    w.id = spec_.NewId();
    w.type = kind->type;
    w.at = {0, 1000, kind->type == WidgetType::Kpi ? 2 : 4, kind->type == WidgetType::Kpi ? 1 : 3};
    if (kind->type == WidgetType::Plot) {
        w.plot.kind = kind->plot_kind;
        w.plot.legend = false;  // one series; Colour by turns it on in the Plot window
        const auto& info = plot::Info(kind->plot_kind);
        if (info.required & plot::kEncX) w.plot.x_column = FirstField(kind->x_need);
        if (info.required & plot::kEncY) {
            const std::string y = FirstField(kind->y_need, w.plot.x_column);
            if (!y.empty()) w.plot.y_columns = {y};
        }
    } else if (kind->type == WidgetType::Table) {
        for (const auto& c : contract_.columns)
            if (w.columns.size() < 6) w.columns.push_back(c.name);
    }
    spec_.widgets.push_back(w);
    selected_ = w.id;
    SaveIfChanged();
}

void DashboardWindow::Render() {
    if (editor_) editor_->Render();
    FinishExport();
    if (!visible) return;
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::SetNextWindowSize(ImVec2(1360, 860), ImGuiCond_FirstUseEver);
    if (focus_) {
        ImGui::SetNextWindowFocus();
        focus_ = false;
    }
    const std::string title = "Dashboard" + (title_.empty() ? std::string() : " \xC2\xB7 " + title_) + "###" + id_;
    ::gui::NextWindowMayLeave(title.c_str());
    const bool expanded = ImGui::Begin(title.c_str(), &visible);
    ::gui::TabMenu(title.c_str(), &visible);
    if (!expanded) {
        ImGui::End();
        return;
    }
    if (draw_header) draw_header();
    if (dataset_.empty() || !message_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "%s", message_.empty() ? "No data yet." : message_.c_str());
        ImGui::End();
        return;
    }
    if (sparse_) {
        DrawSparse();
        ImGui::End();
        return;
    }
    EnsureProfile();
    if (profile_) {
        // Roles may have changed in Data Studio or in the graph.
        if (ImGui::GetTime() - roles_at_ > 1.0) {
            roles_at_ = ImGui::GetTime();
            RebuildContract();
        }
        EnsureWords();
        session_.Poll(spec_, contract_, *profile_);
        SaveIfChanged();  // known types learned while binding
    }
    DrawToolbar();
    DrawFilters();
    if (!profile_) {
        ImGui::TextColored(profile_error_.empty() ? t.info : t.error, "%s",
                           profile_error_.empty() ? ICON_FA_SPINNER " Looking at the data (profile in Task View)..." : profile_error_.c_str());
        ImGui::End();
        return;
    }
    const float fields_w = 210.0f;
    const float settings_w = selected_.empty() ? 0.0f : 270.0f;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float centre_w = ImGui::GetContentRegionAvail().x - fields_w - gap - (settings_w > 0 ? settings_w + gap : 0.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##fields", ImVec2(fields_w, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    DrawFields(fields_w);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##centre", ImVec2(centre_w, 0), ImGuiChildFlags_None);
    DrawStrip();
    ImGui::Spacing();
    DrawGrid(ImGui::GetContentRegionAvail().x);
    ImGui::EndChild();
    centre_min_ = ImGui::GetItemRectMin();
    centre_max_ = ImGui::GetItemRectMax();
    // Export PNG: read the centre (strip and widgets as shown) after the click's frame.
    if (capture_frame_ > 0 && ImGui::GetFrameCount() >= capture_frame_) {
        capture_frame_ = 0;
        auto sink = capture_ = std::make_shared<Capture>();
        spdlog::info("Dashboard '{}': reading the image ({:.0f} x {:.0f})", id_, centre_max_.x - centre_min_.x, centre_max_.y - centre_min_.y);
        if (plot::Hooks().capture_png)
            plot::Hooks().capture_png(centre_min_, centre_max_, [sink](std::vector<unsigned char> png) {
                sink->png = std::move(png);
                sink->ready = true;
            });
    }
    if (settings_w > 0) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
        ImGui::BeginChild("##settings", ImVec2(settings_w, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        DrawSettings(settings_w);
        ImGui::EndChild();
    }
    ImGui::End();
}

void DashboardWindow::DrawToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ui::SecondaryButton(ICON_FA_PLUS " Add widget")) ImGui::OpenPopup("##add_widget");
    if (ImGui::BeginPopup("##add_widget")) {
        std::string group;
        for (const auto& k : WidgetKinds()) {
            if (k.group != group) {
                if (!group.empty()) ImGui::Separator();
                group = k.group;
                ImGui::TextColored(t.text_dim, "%s", group.c_str());
            }
            if (ImGui::MenuItem(k.label.c_str())) AddWidget(k.id);
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_WAND_MAGIC_SPARKLES " Regenerate", profile_ != nullptr, "Profiling first")) {
        spec_.RemoveAutomatic();
        auto autos = AutomaticWidgets(*profile_, contract_, spec_);
        // The automatic widgets first, the hand-made ones after them.
        autos.insert(autos.end(), spec_.widgets.begin(), spec_.widgets.end());
        spec_.widgets = std::move(autos);
        spec_.automatic_done = true;
        selected_.clear();
        SaveIfChanged();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rebuilds the automatic widgets from the current data and roles; the widgets you added stay.");
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_ROTATE " Re-query")) session_.RefreshAll();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Runs every widget's query again (the data itself is read again with Refresh above).");
    ImGui::SameLine();
    const plot::ViewHooks& h = plot::Hooks();
    if (ui::SecondaryButton(ICON_FA_IMAGE " Export PNG", h.capture_png && h.save_path && profile_ && !capture_, "Nothing to export yet")) {
        capture_frame_ = ImGui::GetFrameCount() + 1;  // after this frame's hover state is gone
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Saves the summary and widgets as shown (scrolled-away widgets are not in the image).");
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_TABLE " Open in Data Studio", static_cast<bool>(on_open_query), "Data Studio is not available")) {
        on_open_query(dataset_, FilteredRowsSql(ShownName(), spec_.filters));
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Opens the Query tab on these rows: the dataset with the current filters as SQL.");
    ImGui::SameLine();
    if (!note_.empty() && ImGui::GetTime() < note_until_) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", note_.c_str());
    } else {
        ImGui::TextColored(t.text_faint, session_.Busy() ? ICON_FA_SPINNER " updating" : "");
    }
}

void DashboardWindow::DrawFilters() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "FILTERS");
    ImGui::SameLine();
    if (spec_.filters.Empty()) {
        ImGui::TextColored(t.text_faint, "none - click a bar, a slice or a bin in a widget to filter the others");
        return;
    }
    int remove = -1;
    for (size_t i = 0; i < spec_.filters.predicates.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const std::string chip = spec_.filters.predicates[i].Text() + "  " ICON_FA_XMARK;
        if (ui::ToggleChip("##filter", chip.c_str(), "", true, ImGui::ColorConvertFloat4ToU32(t.accent_text))) remove = static_cast<int>(i);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove this filter");
        ImGui::PopID();
        ImGui::SameLine();
    }
    if (remove >= 0) spec_.filters.predicates.erase(spec_.filters.predicates.begin() + remove);
    if (ui::LinkButton("Clear all")) spec_.filters.Clear();
    const StripResult& s = session_.Strip();
    if (s.ready) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "\xC2\xB7 %s of %s rows", Thousands(s.rows_now).c_str(), Thousands(s.rows_all).c_str());
    }
    SaveIfChanged();
}

void DashboardWindow::DrawFields(float width) {
    (void)width;
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "FIELDS");
    for (const auto& c : contract_.columns) {
        ImGui::TextUnformatted(c.name.c_str());
        const char* role = RoleLabel(c.role);
        const float rw = ImGui::CalcTextSize(role).x;
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 4.0f, ImGui::GetContentRegionMax().x - rw));
        ImGui::TextColored(RoleColour(c.role), "%s", role);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s (%s)", c.reason.c_str(), RoleSourceLabel(c.source));
    }
    ImGui::Spacing();
    if (on_edit_roles && ui::LinkButton("Edit roles in Data Studio")) on_edit_roles(dataset_);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Roles: the Data Input label (contract) first, then roles set in Data Studio, then inferred.");
    ImGui::PopTextWrapPos();
}

void DashboardWindow::DrawStrip() {
    const ui::Tokens& t = ui::CurrentTokens();
    const StripResult& s = session_.Strip();
    struct Kpi {
        std::string label, value, sub;
    };
    std::vector<Kpi> kpis;
    kpis.push_back({"Rows", s.ready ? Thousands(s.rows_now) : "...", s.ready && s.rows_now != s.rows_all ? "of " + Thousands(s.rows_all) : ""});
    kpis.push_back({"Columns", std::to_string(profile_->columns.size()), ""});
    if (s.ready && std::isfinite(s.missing_now) && s.rows_now > 0) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * s.missing_now / (s.rows_now * static_cast<double>(profile_->columns.size())));
        kpis.push_back({"Missing cells", buf, ""});
    }
    kpis.push_back({"Duplicate rows", profile_->duplicates_known ? Thousands(static_cast<double>(profile_->duplicate_rows)) : "not counted", ""});
    if (!target_.empty() && s.ready) {
        if (std::isfinite(s.target_now))
            kpis.push_back({target_ + " (target)", "mean " + Short(s.target_now), s.target_now != s.target_all ? "all " + Short(s.target_all) : ""});
        else if (!s.target_text_now.empty())
            kpis.push_back({target_ + " (target)", s.target_text_now, s.target_text_now != s.target_text_all ? "all " + s.target_text_all : ""});
    }
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float w = (ImGui::GetContentRegionAvail().x - gap * static_cast<float>(kpis.size() - 1)) / static_cast<float>(kpis.size());
    for (size_t i = 0; i < kpis.size(); ++i) {
        if (i) ImGui::SameLine();
        ImGui::PushID(static_cast<int>(i));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
        ImGui::BeginChild("##kpi", ImVec2(w, ImGui::GetTextLineHeight() * 3.2f), ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        ImGui::TextColored(t.text_dim, "%s", kpis[i].label.c_str());
        {
            ui::FontScope big(ui::Font::Heading);
            ImGui::TextUnformatted(kpis[i].value.c_str());
        }
        if (!kpis[i].sub.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", kpis[i].sub.c_str());
        }
        ImGui::EndChild();
        ImGui::PopID();
    }
}

void DashboardWindow::DrawGrid(float width) {
    // A flow layout on 12 columns, in the widgets' order (y, then x).
    std::vector<WidgetSpec*> order;
    for (auto& w : spec_.widgets) order.push_back(&w);
    std::stable_sort(order.begin(), order.end(), [](const WidgetSpec* a, const WidgetSpec* b) {
        return a->at.y != b->at.y ? a->at.y < b->at.y : a->at.x < b->at.x;
    });
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float cell = (width - gap * (DashboardSpec::kColumns - 1)) / DashboardSpec::kColumns;
    const float unit = 105.0f;
    int x = 0;
    bool first_in_row = true;
    for (WidgetSpec* w : order) {
        const int span = std::clamp(w->at.w, 1, DashboardSpec::kColumns);
        if (x + span > DashboardSpec::kColumns) {
            x = 0;
            first_in_row = true;
        }
        if (!first_in_row) ImGui::SameLine();
        DrawCard(*w, cell * static_cast<float>(span) + gap * static_cast<float>(span - 1), unit * static_cast<float>(std::clamp(w->at.h, 1, 8)));
        x += span;
        first_in_row = false;
    }
    if (spec_.widgets.empty()) ImGui::TextColored(ui::CurrentTokens().text_dim, "No widgets: Add widget, or Regenerate for the automatic layout.");
}

void DashboardWindow::DrawCard(WidgetSpec& w, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool selected = w.id == selected_;
    ImGui::PushID(w.id.c_str());
    if (selected && scroll_to_selected_) {  // a widget just added from Data Studio
        ImGui::SetScrollHereY(0.0f);
        scroll_to_selected_ = false;
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, selected && mode_ == Mode::Dashboard ? ui::Mix(t.plot_bg, t.accent, 0.10f) : t.plot_bg);
    ImGui::BeginChild("##card", ImVec2(width, height), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleColor();
    const WidgetResult& r = session_.ResultOf(w.id);
    // Header: the title selects the widget; the state at the right.
    {
        ui::FontScope medium(ui::Font::Medium);
        if (mode_ == Mode::Visualize) ImGui::TextUnformatted(WidgetTitle(w).c_str());
        else if (ImGui::Selectable(WidgetTitle(w).c_str(), false, 0, ImVec2(width * 0.6f, 0))) selected_ = selected ? std::string() : w.id;
    }
    if (mode_ == Mode::Dashboard && ImGui::IsItemHovered()) ImGui::SetTooltip("Click to edit this widget");
    // The state after the title, on its line (the body starts below).
    if (r.state == WidgetResult::State::Running) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, ICON_FA_SPINNER);
    } else if (r.sampled) {
        ImGui::SameLine();
        ImGui::TextColored(t.info, "sampled");
    }
    const ImVec2 body(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
    switch (r.state) {
        case WidgetResult::State::Unbound: {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.warning, "%s", r.message.c_str());
            ImGui::PopTextWrapPos();
            if (!r.binding.rename_candidate.empty()) {
                const std::string label = "Rebind to " + r.binding.rename_candidate;
                if (ui::PrimaryButton(label.c_str(), true, nullptr, ui::ButtonSize::Small)) {
                    w.RenameField(r.binding.field, r.binding.rename_candidate);
                    SaveIfChanged();
                }
                ImGui::SameLine();
            }
            if (ui::DangerButton("Remove")) {
                const std::string id = w.id;
                spec_.filters.ClearWidget(id);
                spec_.widgets.erase(std::remove_if(spec_.widgets.begin(), spec_.widgets.end(), [&](const WidgetSpec& x) { return x.id == id; }),
                                    spec_.widgets.end());
                ImGui::EndChild();
                ImGui::PopID();
                SaveIfChanged();
                return;
            }
            break;
        }
        case WidgetResult::State::Failed: {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.error, "%s", r.message.c_str());
            ImGui::PopTextWrapPos();
            break;
        }
        case WidgetResult::State::Waiting:
        case WidgetResult::State::Running:
            if (!r.prepared && !r.table && !std::isfinite(r.value)) {
                ImGui::TextColored(t.text_faint, "Reading...");
                break;
            }
            [[fallthrough]];
        case WidgetResult::State::Ready: {
            if (w.type == WidgetType::Kpi) {
                {
                    ui::FontScope big(ui::Font::Heading);
                    ImGui::TextUnformatted((w.measure == Measure::MissingPct ? Short(r.value) + "%" : Short(r.value)).c_str());
                }
                if (std::isfinite(r.all) && r.all != r.value) ImGui::TextColored(t.text_dim, "all rows: %s", Short(r.all).c_str());
            } else if (w.type == WidgetType::Table && r.table) {
                const int cols = std::min(r.table->num_columns(), 24);
                if (cols > 0 && ImGui::BeginTable("##rows", cols, ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg |
                                                                    ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody, body)) {
                    for (int c = 0; c < cols; ++c) ImGui::TableSetupColumn(r.table->field(c)->name().c_str());
                    ImGui::TableHeadersRow();
                    for (int64_t row = 0; row < r.table->num_rows(); ++row) {
                        ImGui::TableNextRow();
                        for (int c = 0; c < cols; ++c) {
                            ImGui::TableSetColumnIndex(c);
                            auto s = r.table->column(c)->GetScalar(row);
                            ImGui::TextUnformatted(s.ok() && (*s)->is_valid ? (*s)->ToString().c_str() : "");
                        }
                    }
                    ImGui::EndTable();
                }
            } else if (w.type == WidgetType::Missing && r.table) {
                // Columns with missing values: a bar of their share each.
                const double rows = r.table->num_rows() ? std::max(1.0, [&] {
                    auto s = r.table->GetColumnByName("rows")->GetScalar(0);
                    return s.ok() ? std::stod((*s)->ToString()) : 1.0;
                }()) : 1.0;
                int shown = 0;
                const float bar_w = ImGui::GetContentRegionAvail().x * 0.45f;
                for (size_t i = 0; i < profile_->columns.size(); ++i) {
                    auto col = r.table->GetColumnByName("m" + std::to_string(i));
                    if (!col) continue;
                    auto s = col->GetScalar(0);
                    const double m = s.ok() ? std::stod((*s)->ToString()) : 0.0;
                    if (m <= 0) continue;
                    ++shown;
                    ImGui::TextColored(t.text_dim, "%s", profile_->columns[i].facts.name.c_str());
                    ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.42f);
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const float h = ImGui::GetTextLineHeight();
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, at.y + h * 0.25f), ImVec2(at.x + std::max(2.0f, bar_w * static_cast<float>(m / rows)), at.y + h * 0.75f),
                                                              ui::ToU32(t.warning), 2.0f);
                    ImGui::Dummy(ImVec2(bar_w, h));
                    ImGui::SameLine();
                    ImGui::Text("%.1f%%", 100.0 * m / rows);
                }
                if (shown == 0) ImGui::TextColored(t.text_dim, "No missing values in these rows.");
            } else if (w.type == WidgetType::Plot && r.prepared) {
                auto& view = views_[w.id];
                if (!view) view = std::make_unique<plot::PlotView>(id_ + "_" + w.id);
                if (view_versions_[w.id] != r.version) {
                    view->SetData(*r.prepared);
                    view_versions_[w.id] = r.version;
                }
                view->SetBackground(r.all_rows);
                // What this widget selected: its bars or bins in full colour.
                std::vector<std::string> picked;
                double lo = NAN, hi = NAN;
                for (const auto& q : spec_.filters.predicates) {
                    // A text widget filters the text (or class) column, not its query's.
                    const bool own = q.source_widget == w.id && (w.IsText() || q.field == w.plot.x_column);
                    if (!own) continue;
                    if (q.op == FilterPredicate::Op::In || q.op == FilterPredicate::Op::Contains) picked = q.values;
                    else if (q.op == FilterPredicate::Op::Range) {
                        lo = q.lo;
                        hi = q.hi;
                    }
                }
                view->SetHighlight(picked, lo, hi);
                plot::PlotView::Options o;
                o.toolbar = mode_ == Mode::Visualize;  // Visualize: Fit and Export on its one plot
                o.own_window_button = false;
                view->Draw(body, o);
                // A click on a bar, slice or bin filters the other widgets (again: clears).
                auto click = view->TakeClick();
                if (mode_ == Mode::Visualize || w.IsQuery()) click.reset();  // no shared filter here / the query's own columns
                FilterPredicate p;
                if (click) {
                    p.field = click->field;
                    p.source_widget = w.id;
                    p.bucket = w.bucket;  // a date widget's bins are years
                    if (click->what == plot::PlotView::Click::What::Category) p.values = {click->value};
                    else {
                        p.op = FilterPredicate::Op::Range;
                        p.lo = click->lo;
                        p.hi = click->hi;
                    }
                    // A text widget: a word or phrase keeps the texts that have it, a
                    // length bin their word count, a class its rows.
                    if (w.IsText()) {
                        if (w.text_view == TextView::WordsByClass) {
                            p.field = w.label_field;
                        } else if (w.text_view == TextView::Length) {
                            p.field = w.text_field;
                            p.bucket = "words";
                        } else {
                            p.field = w.text_field;
                            p.op = FilterPredicate::Op::Contains;
                        }
                    }
                    if (p.field.empty()) click.reset();  // a class view with no class column
                }
                if (click) {
                    bool same = false;
                    for (const auto& q : spec_.filters.predicates)
                        same = same || (q.source_widget == w.id && q.field == p.field && q.Text() == p.Text());
                    if (same) spec_.filters.ClearWidget(w.id);
                    else spec_.filters.Set(p);
                    SaveIfChanged();
                }
            }
            break;
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void DashboardWindow::DrawSettings(float width) {
    (void)width;
    const ui::Tokens& t = ui::CurrentTokens();
    WidgetSpec* w = spec_.Find(selected_);
    if (!w) {
        selected_.clear();
        return;
    }
    ImGui::TextColored(t.text_dim, mode_ == Mode::Visualize ? "PLOT" : "WIDGET");
    ImGui::SameLine();
    ImGui::TextUnformatted(WidgetTitle(*w).c_str());
    if (title_for_ != w->id) {
        title_for_ = w->id;
        std::snprintf(title_buf_, sizeof(title_buf_), "%s", w->title.c_str());
    }
    ImGui::TextColored(t.text_dim, "Title (empty: from the content)");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##title", title_buf_, sizeof(title_buf_))) w->title = title_buf_;
    const auto column_combo = [&](const char* id, std::string& value, FieldNeed need, bool allow_none) {
        bool changed = false;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo(id, value.empty() ? "(none)" : value.c_str())) {
            if (allow_none && ImGui::Selectable("(none)", value.empty())) {
                value.clear();
                changed = true;
            }
            for (const auto& c : contract_.columns) {
                if (need == FieldNeed::Number && !NumberType(c.type)) continue;
                const std::string label = c.name + "  \xC2\xB7  " + RoleLabel(c.role);
                if (ImGui::Selectable(label.c_str(), value == c.name)) {
                    value = c.name;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        return changed;
    };
    switch (w->type) {
        case WidgetType::Plot: {
            ImGui::TextColored(t.text_dim, "Plot type");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##kind", plot::Info(w->plot.kind).label)) {
                std::string group;
                for (const auto& k : WidgetKinds()) {
                    if (k.type != WidgetType::Plot) continue;
                    if (k.group != group) {
                        group = k.group;
                        ImGui::TextColored(t.text_dim, "%s", group.c_str());
                    }
                    if (ImGui::Selectable(k.label.c_str(), k.plot_kind == w->plot.kind)) w->plot.kind = k.plot_kind;
                }
                ImGui::EndCombo();
            }
            if (w->IsText()) {
                // A text widget: what it shows of which text column.
                ImGui::TextColored(t.text_dim, "Shows");
                ImGui::SetNextItemWidth(-1);
                if (ImGui::BeginCombo("##text_view", TextViewLabel(w->text_view))) {
                    for (TextView v : {TextView::Length, TextView::Words, TextView::Phrases, TextView::WordsByClass})
                        if (ImGui::Selectable(TextViewLabel(v), v == w->text_view) && v != w->text_view) {
                            w->text_view = v;
                            // Its plot follows the query's columns.
                            w->plot = plot::PlotSpec{};
                            if (v == TextView::Length) {
                                w->plot.kind = plot::Kind::Histogram;
                                w->plot.x_column = "words";
                                w->plot.x_label = "words per text (the longest 1% in the last bin)";
                            } else if (v == TextView::WordsByClass) {
                                w->plot.kind = plot::Kind::Heatmap;
                                w->plot.x_column = "class";
                                w->plot.y_columns = {"word"};
                                w->plot.value_column = "share";
                            } else {
                                w->plot.kind = plot::Kind::Bar;
                                w->plot.x_column = v == TextView::Words ? "word" : "phrase";
                                w->plot.y_columns = {"count"};
                                w->plot.bar_horizontal = true;
                            }
                            w->title.clear();
                        }
                    ImGui::EndCombo();
                }
                ImGui::TextColored(t.text_dim, "Text column");
                std::string text_field = w->text_field;
                if (column_combo("##text_field", text_field, FieldNeed::Any, false)) w->text_field = text_field;
                if (w->text_view == TextView::WordsByClass) {
                    ImGui::TextColored(t.text_dim, "Class column");
                    std::string label = w->label_field;
                    if (column_combo("##label_field", label, FieldNeed::Category, true)) w->label_field = label;
                }
                if (w->text_view != TextView::Length) {
                    ImGui::Checkbox("Keep common words (the, and, ...)", &w->keep_stop_words);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off: English stop words are left out of the top words and phrases.");
                }
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_faint, "Words: lower case, letters and digits. Click a word or phrase to keep the texts that have it.");
                ImGui::PopTextWrapPos();
                break;
            }
            if (w->IsQuery()) {
                // A query widget: its columns are the query's; the query is edited in Data Studio.
                std::string cols = w->plot.x_column;
                for (const auto& y : w->plot.y_columns) cols += (cols.empty() ? "" : ", ") + y;
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "From a query over the rows shown here (columns %s):", cols.c_str());
                ImGui::PopTextWrapPos();
                ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
                ImGui::BeginChild("##query", ImVec2(-1, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
                ImGui::PopStyleColor();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(w->query.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndChild();
                if (ui::SecondaryButton("Edit in Query tab", static_cast<bool>(on_open_query), "Data Studio is not available", ui::ButtonSize::Small))
                    on_open_query(dataset_, w->query);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Opens the query in Data Studio; Add to Dashboard from there adds the changed one.");
                break;
            }
            const WidgetKind& kind = KindOf(*w);
            const auto& info = plot::Info(w->plot.kind);
            if ((info.required | info.optional) & plot::kEncX) {
                ImGui::TextColored(t.text_dim, "%s", info.x_hint);
                column_combo("##x", w->plot.x_column, kind.x_need, !(info.required & plot::kEncX));
            }
            if ((info.required | info.optional) & plot::kEncY) {
                ImGui::TextColored(t.text_dim, "%s", info.y_hint);
                std::string y = w->plot.y_columns.empty() ? std::string() : w->plot.y_columns.front();
                if (column_combo("##y", y, kind.y_need, !(info.required & plot::kEncY))) {
                    if (y.empty()) w->plot.y_columns.clear();
                    else if (w->plot.y_columns.empty()) w->plot.y_columns = {y};
                    else w->plot.y_columns.front() = y;
                }
                if (w->plot.y_columns.size() > 1) ImGui::TextColored(t.text_faint, "+%zu more (Plot window)", w->plot.y_columns.size() - 1);
            }
            if (info.optional & plot::kEncColor) {
                ImGui::TextColored(t.text_dim, "Colour by");
                column_combo("##colour", w->plot.color_column, FieldNeed::Any, true);
            }
            if (ui::SecondaryButton("All settings in the Plot window", true, nullptr, ui::ButtonSize::Small, -1)) OpenInPlotWindow(*w);
            break;
        }
        case WidgetType::Kpi: {
            ImGui::TextColored(t.text_dim, "Measure");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##measure", MeasureLabel(w->measure))) {
                for (Measure m : {Measure::Count, Measure::Sum, Measure::Mean, Measure::Median, Measure::Min, Measure::Max, Measure::Distinct,
                                  Measure::MissingPct})
                    if (ImGui::Selectable(MeasureLabel(m), w->measure == m)) w->measure = m;
                ImGui::EndCombo();
            }
            if (w->measure != Measure::Count) {
                ImGui::TextColored(t.text_dim, "Field");
                column_combo("##field", w->field, w->measure == Measure::Distinct || w->measure == Measure::MissingPct ? FieldNeed::Any : FieldNeed::Number,
                             false);
            }
            break;
        }
        case WidgetType::Missing:
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "The share of missing values per column under the filters; texts marked as missing in Data Studio count.");
            ImGui::PopTextWrapPos();
            break;
        case WidgetType::Table: {
            ImGui::TextColored(t.text_dim, "Columns");
            for (const auto& c : contract_.columns) {
                bool on = std::find(w->columns.begin(), w->columns.end(), c.name) != w->columns.end();
                if (ImGui::Checkbox(c.name.c_str(), &on)) {
                    if (on) w->columns.push_back(c.name);
                    else w->columns.erase(std::remove(w->columns.begin(), w->columns.end(), c.name), w->columns.end());
                }
            }
            ImGui::TextColored(t.text_dim, "Rows");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputInt("##rows", &w->rows, 0, 0)) w->rows = std::clamp(w->rows, 1, 10000);
            break;
        }
    }
    // The query behind the widget, as text to read or run in Data Studio.
    ImGui::Spacing();
    const bool show_sql = sql_for_ == w->id;
    if (ui::LinkButton(show_sql ? "Hide SQL" : "View SQL")) sql_for_ = show_sql ? std::string() : w->id;
    if (sql_for_ == w->id && profile_) {
        const QueryRequest q = session_.RequestFor(*w, *profile_, spec_.filters, ShownName());
        const std::string sql = InlineParams(q.sql, q.params);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
        ImGui::BeginChild("##sql", ImVec2(-1, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(sql.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
        if (ui::SecondaryButton("Open in Query tab", static_cast<bool>(on_open_query), "Data Studio is not available", ui::ButtonSize::Small))
            on_open_query(dataset_, sql);
        ImGui::SameLine();
        if (ui::SecondaryButton("Copy", true, nullptr, ui::ButtonSize::Small)) ImGui::SetClipboardText(sql.c_str());
    }
    if (mode_ == Mode::Visualize) {
        SaveIfChanged();
        return;
    }
    // Size and place.
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "SIZE AND PLACE");
    ImGui::SetNextItemWidth(110);
    if (ImGui::SliderInt("##w", &w->at.w, 1, DashboardSpec::kColumns, "width %d")) w->at.w = std::clamp(w->at.w, 1, DashboardSpec::kColumns);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("##h", &w->at.h, 1, 8, "height %d")) w->at.h = std::clamp(w->at.h, 1, 8);
    // Earlier / later: swap order with the neighbour.
    auto it = std::find_if(spec_.widgets.begin(), spec_.widgets.end(), [&](const WidgetSpec& x) { return x.id == selected_; });
    const auto swap_place = [&](std::vector<WidgetSpec>::iterator other) {
        std::swap(it->at.x, other->at.x);
        std::swap(it->at.y, other->at.y);
        std::iter_swap(it, other);
    };
    std::stable_sort(spec_.widgets.begin(), spec_.widgets.end(),
                     [](const WidgetSpec& a, const WidgetSpec& b) { return a.at.y != b.at.y ? a.at.y < b.at.y : a.at.x < b.at.x; });
    it = std::find_if(spec_.widgets.begin(), spec_.widgets.end(), [&](const WidgetSpec& x) { return x.id == selected_; });
    if (ui::SecondaryButton(ICON_FA_ARROW_LEFT " Earlier", it != spec_.widgets.begin())) swap_place(it - 1);
    ImGui::SameLine();
    if (ui::SecondaryButton("Later " ICON_FA_ARROW_RIGHT, it + 1 != spec_.widgets.end())) swap_place(it + 1);
    // Positions in order (so the flow keeps the order after edits).
    for (size_t i = 0; i < spec_.widgets.size(); ++i) {
        spec_.widgets[i].at.y = static_cast<int>(i);
        spec_.widgets[i].at.x = 0;
    }
    ImGui::Spacing();
    if (ui::DangerButton("Remove widget")) RemoveWidget(selected_);
    ImGui::SameLine();
    if (ui::LinkButton("Done")) selected_.clear();
    SaveIfChanged();
}

void DashboardWindow::RenderEmbedded() {
    if (editor_) editor_->Render();
    const ui::Tokens& t = ui::CurrentTokens();
    if (dataset_.empty() || !message_.empty()) {
        ImGui::TextColored(t.text_dim, "%s", message_.empty() ? "Pick a dataset above to plot it." : message_.c_str());
        return;
    }
    EnsureProfile();
    if (profile_) {
        if (ImGui::GetTime() - roles_at_ > 1.0) {
            roles_at_ = ImGui::GetTime();
            RebuildContract();
        }
        session_.Poll(spec_, contract_, *profile_);
    }
    DrawVisualizeToolbar();
    if (!profile_) {
        ImGui::TextColored(profile_error_.empty() ? t.info : t.error, "%s",
                           profile_error_.empty() ? ICON_FA_SPINNER " Looking at the data (profile in Task View)..." : profile_error_.c_str());
        return;
    }
    if (spec_.widgets.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "No plots yet: New plot offers every plot type; its columns start from the data's roles.");
        return;
    }
    if (!spec_.Find(selected_)) selected_ = spec_.widgets.front().id;
    const float list_w = 200.0f, settings_w = 270.0f;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float centre_w = ImGui::GetContentRegionAvail().x - list_w - settings_w - 2 * gap;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##plots", ImVec2(list_w, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    DrawPlotList();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##plot", ImVec2(centre_w, 0), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    if (WidgetSpec* w = spec_.Find(selected_)) DrawCard(*w, ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##plot_settings", ImVec2(settings_w, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    DrawSettings(settings_w);
    ImGui::EndChild();
}

void DashboardWindow::DrawVisualizeToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ui::SecondaryButton(ICON_FA_PLUS " New plot", profile_ != nullptr, "Looking at the data first")) ImGui::OpenPopup("##new_plot");
    if (ImGui::BeginPopup("##new_plot")) {
        std::string group;
        for (const auto& k : WidgetKinds()) {
            if (k.type != WidgetType::Plot) continue;
            if (k.group != group) {
                if (!group.empty()) ImGui::Separator();
                group = k.group;
                ImGui::TextColored(t.text_dim, "%s", group.c_str());
            }
            if (ImGui::MenuItem(k.label.c_str())) AddWidget(k.id);
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    const WidgetSpec* selected = spec_.Find(selected_);
    if (ui::SecondaryButton(ICON_FA_TABLE_COLUMNS " Add to Dashboard", selected != nullptr, "Make a plot first")) ImGui::OpenPopup("##add_to_dashboard");
    if (ImGui::IsItemHovered() && selected) ImGui::SetTooltip("Copies this plot into a dashboard on the same data.");
    if (ImGui::BeginPopup("##add_to_dashboard")) {
        if (selected) {
            ImGui::TextColored(t.text_dim, "Add \"%s\" to", WidgetTitle(*selected).c_str());
            if (DrawAddToDashboardItems(dataset_, *selected)) {
                note_ = "Added to the dashboard.";
                note_until_ = ImGui::GetTime() + 4.0;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Remove", selected != nullptr, "No plot selected")) RemoveWidget(selected_);
    ImGui::SameLine();
    if (ui::SecondaryButton("Clear all", !spec_.widgets.empty(), "No plots")) {
        spec_.widgets.clear();
        views_.clear();
        selected_.clear();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    const size_t n = spec_.widgets.size();
    if (!note_.empty() && ImGui::GetTime() < note_until_) ImGui::TextColored(t.text_dim, "%s", note_.c_str());
    else ImGui::TextColored(t.text_dim, "%zu %s \xC2\xB7 this session", n, n == 1 ? "plot" : "plots");
    if (const auto entry = DatasetCatalog::Instance().Resolve(dataset_)) {
        const std::string data = "Data at " + entry->Shown() + " \xC2\xB7 " + Thousands(static_cast<double>(entry->rows)) + " rows";
        const float w = ImGui::CalcTextSize(data.c_str()).x;
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 8.0f, ImGui::GetContentRegionMax().x - w));
        ImGui::TextColored(t.text_faint, "%s", data.c_str());
    }
}

void DashboardWindow::DrawPlotList() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "PLOTS");
    for (const auto& w : spec_.widgets) {
        ImGui::PushID(w.id.c_str());
        const bool sel = w.id == selected_;
        const float line = ImGui::GetTextLineHeight();
        const ImVec2 at = ImGui::GetCursorPos();
        if (ImGui::Selectable("##plot", sel, 0, ImVec2(0, line * 2 + 2))) selected_ = w.id;
        ImGui::SetCursorPos(ImVec2(at.x + 4, at.y));
        ImGui::TextUnformatted(WidgetTitle(w).c_str());
        ImGui::SetCursorPos(ImVec2(at.x + 4, at.y + line + 2));
        ImGui::TextColored(t.text_dim, "%s", KindOf(w).label.c_str());
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Click a plot to edit it; the list keeps every plot made here.");
    ImGui::PopTextWrapPos();
}

void DashboardWindow::OpenInPlotWindow(WidgetSpec& w) {
    auto ds = DataRegistry::Instance().GetArrowDataset(dataset_);
    if (!ds || !ds->GetArrowTable()) return;
    if (!editor_) editor_ = std::make_unique<plot::PlotWindow>(id_ + "_editor");
    editing_ = w.id;
    editor_->SetSpec(w.plot);
    editor_->on_spec_changed = [this](const plot::PlotSpec& spec) {
        if (WidgetSpec* x = spec_.Find(editing_)) {
            x->plot = spec;
            SaveIfChanged();
        }
    };
    editor_->SetArrowTable(title_.empty() ? dataset_ : title_, ds->GetArrowTable());
    editor_->visible = true;
}

}  // namespace cyxwiz::dashboard
