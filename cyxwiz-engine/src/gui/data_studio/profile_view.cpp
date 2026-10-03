#include "profile_view.h"

#include "../../core/async_task_manager.h"
#include "../../core/column_role_store.h"
#include "../../core/dataset_catalog.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <sstream>

namespace cyxwiz {

namespace {

std::string Thousands(size_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

std::string Short(double v) {
    if (!std::isfinite(v)) return "";
    char buf[32];
    if (std::fabs(v) >= 1e6) std::snprintf(buf, sizeof(buf), "%.3g", v);
    else if (std::fabs(v - std::round(v)) < 1e-9) std::snprintf(buf, sizeof(buf), "%.0f", v);
    else std::snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

std::string Percent(size_t part, size_t whole) {
    if (whole == 0) return "0%";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buf;
}

ImVec4 RoleColour(ColumnRole role) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (role) {
        case ColumnRole::Target: return t.success;
        case ColumnRole::Numeric: return t.series[1];
        case ColumnRole::Category: return t.series[0];
        case ColumnRole::DateTime: return t.series[2];
        case ColumnRole::Id: return t.text_faint;
        case ColumnRole::Ignore: return t.text_faint;
        default: return t.text_dim;
    }
}

// A role as a filled pill in its colour (no outline); true when clicked.
bool RoleChip(const char* label, const ImVec4& colour) {
    const ImVec2 ts = ImGui::CalcTextSize(label);
    const ImVec2 size(ts.x + 12.0f, ts.y + 2.0f);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##chip", size);
    const bool hot = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ui::ToU32(ui::WithAlpha(colour, hot ? 0.30f : 0.16f)), size.y * 0.5f);
    dl->AddText(ImVec2(at.x + 6.0f, at.y + 1.0f), ui::ToU32(colour), label);
    return clicked;
}

constexpr ColumnRole kRoles[] = {ColumnRole::Id,   ColumnRole::Target,   ColumnRole::Numeric, ColumnRole::Category, ColumnRole::DateTime,
                                 ColumnRole::Text, ColumnRole::FilePath, ColumnRole::Weight,  ColumnRole::Ignore};

}  // namespace

ProfileView::ProfileView() = default;

ProfileView::~ProfileView() {
    if (task_id_) AsyncTaskManager::Instance().Cancel(task_id_);
}

std::string ProfileView::SourceKey() const {
    return RoleSourceKey(source_path_, shown_.empty() ? dataset_ : shown_);
}

void ProfileView::SetActiveDataset(const std::string& dataset_name) {
    if (dataset_name == dataset_ && profile_) return;
    dataset_ = dataset_name;
    profile_.reset();
    contract_ = {};
    selected_ = -1;
    error_.clear();
    generation_ = 0;
    if (!dataset_.empty()) Reprofile();
}

void ProfileView::Reprofile() {
    if (dataset_.empty()) return;
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_);
    if (!entry) {
        error_ = "'" + dataset_ + "' is no longer loaded.";
        return;
    }
    if (entry->storage != DatasetStorageKind::InMemoryArrow && entry->storage != DatasetStorageKind::DiskBackedParquet) {
        error_ = std::string("This dataset is ") + StorageText(entry->storage) + ": its profile comes with its dashboard (file counts, classes).";
        return;
    }
    if (task_id_) AsyncTaskManager::Instance().Cancel(task_id_);
    shown_ = entry->Shown();
    source_path_ = entry->source_path;
    target_column_ = entry->target_column;
    generation_ = entry->generation;
    error_.clear();
    ProfileOptions options;
    if (const auto* s = ProjectColumnRoles().Find(SourceKey())) options.missing_text = s->missing_text;
    const std::string table = dataset_;
    std::weak_ptr<int> alive = alive_;
    auto result = std::make_shared<DatasetProfile>();
    task_id_ = AsyncTaskManager::Instance().RunAsync(
        "Profile " + shown_,
        [table, options, result](LambdaTask& task) {
            ProfileOptions opts = options;
            opts.should_stop = [&task] { return task.ShouldStop(); };
            opts.progress = [&task](float f, const std::string& what) { task.ReportProgress(f, what); };
            *result = ProfileTable(table, [](const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }, opts);
            if (!result->ok()) task.MarkFailed(result->error);
        },
        nullptr,
        [this, alive, result](bool /*success*/, const std::string& /*message*/) {
            if (alive.expired()) return;
            task_id_ = 0;
            if (!result->ok()) {
                error_ = result->error == "Cancelled." ? std::string() : result->error;
                return;
            }
            profile_ = result;
            std::time_t now = std::time(nullptr);
            char buf[16];
            std::strftime(buf, sizeof(buf), "%H:%M", std::localtime(&now));
            profiled_at_ = buf;
            RebuildContract();
            spdlog::info("[Data Studio] Profiled '{}': {} rows, {} columns in {:.0f} ms{}", shown_, profile_->rows, profile_->columns.size(),
                         profile_->elapsed_ms, profile_->exact ? "" : " (approximate)");
        },
        alive_);
}

void ProfileView::RebuildContract() {
    if (!profile_) return;
    std::map<std::string, ColumnRole> contract_roles;
    if (!target_column_.empty()) contract_roles[target_column_] = ColumnRole::Target;
    contract_ = BuildContract(SourceKey(), profile_->Facts(), contract_roles, ProjectColumnRoles().RolesFor(SourceKey()));
}

void ProfileView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    // The data changed (re-loaded) or the graph's target changed: profile again / rebuild.
    if (!dataset_.empty() && !IsRunning()) {
        if (const auto entry = DatasetCatalog::Instance().Resolve(dataset_)) {
            if (profile_ && entry->generation != generation_ && entry->generation != 0) Reprofile();
            else if (entry->target_column != target_column_) {
                target_column_ = entry->target_column;
                RebuildContract();
            }
        }
    }
    if (dataset_.empty()) {
        ImGui::TextColored(t.text_dim, "Pick a dataset above to profile it.");
        return;
    }
    RenderOverview();
    if (!profile_) return;
    ImGui::Spacing();
    const float detail_w = std::min(380.0f, ImGui::GetContentRegionAvail().x * 0.34f);
    const float table_w = ImGui::GetContentRegionAvail().x - detail_w - ImGui::GetStyle().ItemSpacing.x;
    ImGui::BeginChild("##profile_columns", ImVec2(table_w, 0), ImGuiChildFlags_None);
    RenderColumns(table_w);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##profile_detail", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    RenderDetail();
    ImGui::EndChild();
}

void ProfileView::RenderOverview() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(shown_.empty() ? dataset_.c_str() : shown_.c_str());
    if (profile_) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s rows \xC2\xB7 %zu columns \xC2\xB7 profiled %s in %.1f s (%s)", Thousands(profile_->rows).c_str(),
                           profile_->columns.size(), profiled_at_.c_str(), profile_->elapsed_ms / 1000.0,
                           profile_->exact ? "exact" : "approximate distinct counts and quartiles");
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_ROTATE " Profile again", !IsRunning(), "Profiling now")) Reprofile();
    if (IsRunning()) {
        ImGui::SameLine();
        ImGui::TextColored(t.info, ICON_FA_SPINNER " Profiling (Task View shows the steps)");
    }
    if (!error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!profile_) return;
    // Role counts, missing cells, duplicates, quality (kept from Analyze).
    std::map<ColumnRole, int> counts;
    for (const auto& c : contract_.columns) ++counts[c.role];
    std::string roles;
    for (ColumnRole r : kRoles)
        if (counts[r]) roles += (roles.empty() ? "" : " \xC2\xB7 ") + std::string(RoleLabel(r)) + " " + std::to_string(counts[r]);
    const size_t cells = profile_->rows * profile_->columns.size();
    const size_t missing = profile_->MissingCells();
    const double missing_pct = cells ? 100.0 * static_cast<double>(missing) / static_cast<double>(cells) : 0.0;
    ImGui::TextColored(t.text_dim, "%s", roles.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_faint, "|");
    ImGui::SameLine();
    ImGui::TextColored(missing ? t.warning : t.text_dim, "missing cells %s", Percent(missing, cells).c_str());
    ImGui::SameLine();
    if (profile_->duplicates_known) ImGui::TextColored(profile_->duplicate_rows ? t.warning : t.text_dim, "\xC2\xB7 duplicate rows %s", Thousands(profile_->duplicate_rows).c_str());
    else ImGui::TextColored(t.text_dim, "\xC2\xB7 duplicate rows not counted (too many columns)");
    ImGui::SameLine();
    const double quality = std::max(0.0, 100.0 - 2.0 * missing_pct);
    ImGui::TextColored(quality > 80 ? t.success : quality > 60 ? t.warning : t.error, "\xC2\xB7 data quality %.0f%%", quality);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("100 minus twice the share of missing cells (as Analyze computed it).");
    // Warnings: a role here that contradicts the graph; saved roles for columns that are gone.
    for (const auto& c : contract_.columns)
        if (c.user_conflict)
            ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION " %s: set to %s here, but the graph says %s. Training keeps the Data Input label; change it there.",
                               c.name.c_str(), RoleLabel(*c.user_conflict), RoleLabel(c.role));
    if (!contract_.unmatched.empty()) {
        std::string names;
        for (const auto& n : contract_.unmatched) names += (names.empty() ? "" : ", ") + n;
        ImGui::TextColored(t.text_dim, "Saved roles for columns not in this table (kept for when they return): %s", names.c_str());
    }
    if (!save_error_.empty()) ImGui::TextColored(t.error, "%s", save_error_.c_str());
}

void ProfileView::RenderColumns(float /*width*/) {
    const ui::Tokens& t = ui::CurrentTokens();
    auto& store = ProjectColumnRoles();
    const std::string key = SourceKey();
    if (!ImGui::BeginTable("##profile", 10,
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit |
                               ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings))
        return;
    ImGui::TableSetupScrollFreeze(1, 1);
    for (const char* h : {"column", "type", "role", "from", "values"}) ImGui::TableSetupColumn(h);
    ImGui::TableSetupColumn("missing", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("888,888 (88.8%)").x);
    for (const char* h : {"mean", "std", "median", "outliers"}) ImGui::TableSetupColumn(h);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(profile_->columns.size()));
    while (clipper.Step())
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const ProfiledColumn& c = profile_->columns[static_cast<size_t>(i)];
            const ColumnContract* cc = contract_.Find(c.facts.name);
            ImGui::TableNextRow();
            ImGui::PushID(i);
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(c.facts.name.c_str(), selected_ == i, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                selected_ = i;
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(t.text_dim, "%s", cc ? cc->type.c_str() : c.sql_type.c_str());
            ImGui::TableSetColumnIndex(2);
            if (cc) {
                // The role as a chip; click to change (contract roles cannot be changed here).
                const ImVec4 col = RoleColour(cc->role);
                const std::string chip = std::string(RoleLabel(cc->role)) + (cc->source == RoleSource::Contract ? "" : " " ICON_FA_CARET_DOWN);
                if (RoleChip(chip.c_str(), col) && cc->source != RoleSource::Contract) ImGui::OpenPopup("##role");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(cc->source == RoleSource::Contract ? "%s (the graph sets it: change the Data Input label to change it)"
                                                                         : "%s. Click to set the role.", cc->reason.c_str());
                if (ImGui::BeginPopup("##role")) {
                    for (ColumnRole r : kRoles) {
                        if (r == ColumnRole::Target && !target_column_.empty()) continue;  // the graph has a target
                        if (ImGui::MenuItem(RoleLabel(r), nullptr, cc->role == r)) {
                            store.SetRole(key, c.facts.name, r);
                            store.SetSchema(key, contract_.schema_fingerprint);
                            save_error_.clear();
                            if (!store.Save(&save_error_)) spdlog::warn("Column roles: {}", save_error_);
                            RebuildContract();
                        }
                    }
                    if (cc->source == RoleSource::User) {
                        ImGui::Separator();
                        if (ImGui::MenuItem("Back to inferred")) {
                            store.ClearRole(key, c.facts.name);
                            save_error_.clear();
                            store.Save(&save_error_);
                            RebuildContract();
                        }
                    }
                    ImGui::EndPopup();
                }
            }
            ImGui::TableSetColumnIndex(3);
            if (cc) {
                const ImVec4 from = cc->source == RoleSource::Contract ? t.success : cc->source == RoleSource::User ? t.accent_text : t.text_faint;
                ImGui::TextColored(from, "%s", RoleSourceLabel(cc->source));
            }
            ImGui::TableSetColumnIndex(4);
            if (c.numeric) ImGui::TextColored(t.text_dim, "%s to %s", Short(c.min).c_str(), Short(c.max).c_str());
            else if (c.facts.type == ColumnFacts::Type::Temporal || (cc && cc->role == ColumnRole::DateTime && !c.min_text.empty()))
                ImGui::TextColored(t.text_dim, "%s to %s", c.min_text.c_str(), c.max_text.c_str());
            else ImGui::TextColored(t.text_dim, "%s distinct%s", Thousands(c.facts.distinct).c_str(), profile_->exact ? "" : " (about)");
            ImGui::TableSetColumnIndex(5);
            if (c.missing) ImGui::TextColored(t.warning, "%s (%s)", Thousands(c.missing).c_str(), Percent(c.missing, profile_->rows).c_str());
            else ImGui::TextColored(t.text_faint, "0");
            ImGui::TableSetColumnIndex(6);
            if (c.numeric) ImGui::TextUnformatted(Short(c.mean).c_str());
            ImGui::TableSetColumnIndex(7);
            if (c.numeric) ImGui::TextUnformatted(Short(c.std).c_str());
            ImGui::TableSetColumnIndex(8);
            if (c.numeric) ImGui::TextUnformatted(Short(c.median).c_str());
            ImGui::TableSetColumnIndex(9);
            if (c.numeric && c.outliers) ImGui::TextColored(t.text_dim, "%s", Thousands(c.outliers).c_str());
            ImGui::PopID();
        }
    ImGui::EndTable();
}

void ProfileView::RenderDetail() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (selected_ < 0 || selected_ >= static_cast<int>(profile_->columns.size())) {
        ImGui::TextColored(t.text_dim, "COLUMN");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Click a column for its distribution and values.");
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "ROLES");
        ImGui::TextColored(t.text_faint, "contract = from the graph (the Data Input label is the target); you = set here, saved with the "
                                         "project for this file and used by dashboards and plots; inferred = guessed from the values. "
                                         "Contract first, then yours, then inferred.");
        ImGui::PopTextWrapPos();
    } else {
        const ProfiledColumn& c = profile_->columns[static_cast<size_t>(selected_)];
        const ColumnContract* cc = contract_.Find(c.facts.name);
        ImGui::TextUnformatted(c.facts.name.c_str());
        if (cc) {
            ImGui::SameLine();
            ImGui::TextColored(RoleColour(cc->role), "%s", RoleLabel(cc->role));
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "(%s: %s)", RoleSourceLabel(cc->source), cc->reason.c_str());
        }
        // Distribution: histogram for numbers, top values otherwise.
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float w = ImGui::GetContentRegionAvail().x, h = 110.0f;
        const ImVec2 at = ImGui::GetCursorScreenPos();
        if (!c.hist_counts.empty()) {
            size_t most = 1;
            for (size_t n : c.hist_counts) most = std::max(most, n);
            const float bw = w / static_cast<float>(c.hist_counts.size());
            for (size_t b = 0; b < c.hist_counts.size(); ++b) {
                const float bh = (h - 16) * static_cast<float>(c.hist_counts[b]) / static_cast<float>(most);
                dl->AddRectFilled(ImVec2(at.x + b * bw + 1, at.y + h - 14 - bh), ImVec2(at.x + (b + 1) * bw - 1, at.y + h - 14), ui::ToU32(t.series[1]), 1.5f);
            }
            dl->AddText(ImVec2(at.x, at.y + h - 12), ui::ToU32(t.text_dim), Short(c.min).c_str());
            const std::string hi = Short(c.max);
            dl->AddText(ImVec2(at.x + w - ImGui::CalcTextSize(hi.c_str()).x, at.y + h - 12), ui::ToU32(t.text_dim), hi.c_str());
            ImGui::Dummy(ImVec2(w, h));
        } else if (!c.top.empty()) {
            size_t most = 1;
            for (const auto& [v, n] : c.top) most = std::max(most, n);
            const float row = ImGui::GetTextLineHeightWithSpacing();
            for (size_t i = 0; i < c.top.size(); ++i) {
                const float y = at.y + row * static_cast<float>(i);
                const std::string label = c.top[i].first.size() > 18 ? c.top[i].first.substr(0, 17) + ".." : c.top[i].first;
                dl->AddText(ImVec2(at.x, y), ui::ToU32(t.text_dim), label.c_str());
                const float x0 = at.x + w * 0.42f, bw = (w * 0.58f - 50.0f) * static_cast<float>(c.top[i].second) / static_cast<float>(most);
                dl->AddRectFilled(ImVec2(x0, y + 3), ImVec2(x0 + bw, y + row - 3), ui::ToU32(t.series[0]), 2.0f);
                dl->AddText(ImVec2(x0 + bw + 4, y), ui::ToU32(t.text_dim), Thousands(c.top[i].second).c_str());
            }
            ImGui::Dummy(ImVec2(w, row * static_cast<float>(c.top.size())));
            if (c.facts.distinct > c.top.size())
                ImGui::TextColored(t.text_faint, "the %zu most frequent of %s values", c.top.size(), Thousands(c.facts.distinct).c_str());
        }
        // The figures.
        if (ImGui::BeginTable("##figures", 2, ImGuiTableFlags_SizingStretchSame)) {
            const auto row = [&](const char* name, const std::string& value) {
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", name);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            };
            row("Non-missing", Thousands(c.facts.non_null));
            row("Missing", Thousands(c.missing) + (c.missing_text ? " (" + Thousands(c.missing_text) + " as text)" : std::string()));
            row("Distinct", Thousands(c.facts.distinct) + (profile_->exact ? "" : " (about)"));
            if (c.numeric) {
                row("Min", Short(c.min));
                row("Q1", Short(c.q1));
                row("Median", Short(c.median));
                row("Q3", Short(c.q3));
                row("Max", Short(c.max));
                row("Mean", Short(c.mean));
                row("Std", Short(c.std));
                row("Outliers (1.5 IQR)", Thousands(c.outliers));
            } else if (c.facts.type == ColumnFacts::Type::Text) {
                row("Mean length", Short(c.facts.avg_length));
                if (c.facts.date_share > 0) row("Read as dates", Percent(static_cast<size_t>(c.facts.date_share * 100), 100));
            }
            row("Type", c.sql_type);
            ImGui::EndTable();
        }
        // Texts that mean missing (saved with the project; the profile counts them as missing).
        if (!c.numeric) {
            if (missing_for_ != c.facts.name) {
                missing_for_ = c.facts.name;
                std::string joined;
                if (const auto* s = ProjectColumnRoles().Find(SourceKey())) {
                    auto it = s->missing_text.find(c.facts.name);
                    if (it != s->missing_text.end())
                        for (const auto& v : it->second) joined += (joined.empty() ? "" : ", ") + v;
                }
                std::snprintf(missing_buf_, sizeof(missing_buf_), "%s", joined.c_str());
            }
            ImGui::Spacing();
            ImGui::TextColored(t.text_dim, "Treat as missing (comma separated, e.g. N/A, -, none)");
            ImGui::SetNextItemWidth(-90.0f);
            ImGui::InputText("##missing_text", missing_buf_, sizeof(missing_buf_));
            ImGui::SameLine();
            if (ui::SecondaryButton("Apply")) {
                std::vector<std::string> texts;
                std::stringstream ss(missing_buf_);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    const size_t a = item.find_first_not_of(' '), b = item.find_last_not_of(' ');
                    if (a != std::string::npos) texts.push_back(item.substr(a, b - a + 1));
                }
                auto& store = ProjectColumnRoles();
                store.SetMissingText(SourceKey(), c.facts.name, texts);
                save_error_.clear();
                store.Save(&save_error_);
                Reprofile();
            }
            // A suggestion: a frequent value that looks like a missing marker.
            for (const auto& [v, n] : c.top) {
                std::string lower = v;
                for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                if ((lower == "n/a" || lower == "na" || lower == "none" || lower == "null" || lower == "-" || lower == "?") &&
                    std::string(missing_buf_).find(v) == std::string::npos) {
                    ImGui::TextColored(t.warning, "'%s' appears %s times: treat it as missing?", v.c_str(), Thousands(n).c_str());
                    break;
                }
            }
        }
    }
    // Correlations (kept from Analyze, now computed).
    if (!profile_->correlations.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "STRONGEST CORRELATIONS (Pearson)");
        for (const auto& [a, b, r] : profile_->correlations)
            ImGui::TextColored(std::fabs(r) > 0.5 ? t.text : t.text_dim, "%+.2f  %s \xC2\xB7 %s", r, a.c_str(), b.c_str());
    }
}

}  // namespace cyxwiz
