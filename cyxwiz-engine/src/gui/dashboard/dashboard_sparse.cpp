// The Dashboard on sparse features (TOFIX134 P3, approved board 19): the
// Count / TF-IDF Vectorizer's matrix read directly (no SQL), summarised off
// the UI thread (core/dashboard/sparse_summary), exact. A click on the label
// bar keeps that class's rows in every card; again: all rows.

#include "dashboard_window.h"

#include "../../core/async_task_manager.h"
#include "../../core/dashboard/sparse_summary.h"
#include "../../core/data_registry.h"
#include "../../core/plot/plot_prepare.h"
#include "../../core/sparse_feature_dataset.h"
#include "../icons.h"
#include "../plot/plot_view.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <cstdio>

namespace cyxwiz::dashboard {

namespace {

std::vector<double> AsDoubles(const std::vector<size_t>& v) {
    std::vector<double> out;
    out.reserve(v.size());
    for (size_t x : v) out.push_back(static_cast<double>(x));
    return out;
}

std::string Count(size_t n) {
    const std::string d = std::to_string(n);
    std::string out;
    for (size_t i = 0; i < d.size(); ++i) {
        if (i > 0 && (d.size() - i) % 3 == 0) out += ',';
        out += d[i];
    }
    return out;
}

plot::SourceColumn Text(const std::string& name, std::vector<std::string> values) {
    plot::SourceColumn c;
    c.name = name;
    c.numeric = false;
    c.text = std::move(values);
    return c;
}

plot::SourceColumn Numbers(const std::string& name, std::vector<double> values) {
    plot::SourceColumn c;
    c.name = name;
    c.numeric = true;
    c.numbers = std::move(values);
    return c;
}

plot::Prepared Bars(const std::string& x, std::vector<std::string> cats, std::vector<double> values, const std::string& y, bool horizontal) {
    plot::PlotSpec s;
    s.kind = plot::Kind::Bar;
    s.x_column = x;
    s.y_columns = {y};
    s.bar_horizontal = horizontal;
    s.legend = false;
    plot::Source src;
    src.columns = {Text(x, std::move(cats)), Numbers(y, std::move(values))};
    return plot::Prepare(s, src);
}

plot::Prepared Histogram(const std::string& x, std::vector<double> values, const std::string& label, bool log_y) {
    plot::PlotSpec s;
    s.kind = plot::Kind::Histogram;
    s.x_column = x;
    s.x_label = label;
    s.log_y = log_y;
    s.legend = false;
    plot::Source src;
    src.columns = {Numbers(x, std::move(values))};
    return plot::Prepare(s, src);
}

}  // namespace

void DashboardWindow::StartSparseSummary() {
    if (sparse_task_ || dataset_.empty()) return;
    auto result = std::make_shared<SparseSummary>();
    auto error = std::make_shared<std::string>();
    const std::string name = dataset_;
    const std::set<std::string> keep = sparse_keep_;
    std::weak_ptr<int> alive = alive_;
    sparse_dirty_ = false;
    sparse_task_ = AsyncTaskManager::Instance().RunAsync(
        "Sparse features summary",
        [name, keep, result, error](LambdaTask& task) {
            auto ds = DataRegistry::Instance().GetSparseFeatureDataset(name);
            if (!ds) {
                *error = "The sparse features '" + name + "' are no longer loaded: refresh.";
                return;
            }
            SparseInput in;
            in.rows = static_cast<size_t>(ds->GetNumRows());
            in.features = static_cast<size_t>(ds->GetNumFeatures());
            in.offsets = ds->GetRowOffsets().data();
            in.indices = ds->GetColumnIndices().data();
            in.values = ds->GetValues().data();
            in.feature_names = ds->GetFeatureNames();
            in.bytes = ds->GetFeatureStorageBytes() + ds->GetLabelStorageBytes();
            if (const auto& labels = ds->GetLabels()) {
                // Codes as their class names when the vectorizer kept them.
                const auto& names = ds->GetClassNames();
                in.labels.reserve(in.rows);
                for (const auto& chunk : labels->chunks()) {
                    auto codes = std::dynamic_pointer_cast<arrow::Int32Array>(chunk);
                    for (int64_t i = 0; i < chunk->length(); ++i) {
                        if (codes && codes->IsValid(i)) {
                            const int32_t c = codes->Value(i);
                            in.labels.push_back(c >= 0 && static_cast<size_t>(c) < names.size() ? names[static_cast<size_t>(c)] : std::to_string(c));
                        } else {
                            auto scalar = chunk->GetScalar(i);
                            in.labels.push_back(scalar.ok() && (*scalar)->is_valid ? (*scalar)->ToString() : std::string());
                        }
                    }
                }
                if (in.labels.size() != in.rows) in.labels.clear();
            }
            task.ReportProgress(0.3f, "Counting features");
            *result = SummarizeSparse(in, keep);
        },
        nullptr,
        [this, alive, result, error](bool, const std::string& message) {
            if (alive.expired()) return;
            sparse_task_ = 0;
            sparse_error_ = !error->empty() ? *error : message;
            if (error->empty()) {
                sparse_error_.clear();
                sparse_summary_ = result;
                sparse_views_.clear();
            }
        },
        alive_);
}

void DashboardWindow::DrawSparse() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (sparse_dirty_) StartSparseSummary();
    if (!sparse_summary_) {
        ImGui::TextColored(sparse_error_.empty() ? t.info : t.error, "%s",
                           sparse_error_.empty() ? ICON_FA_SPINNER " Reading the sparse features (Task View)..." : sparse_error_.c_str());
        return;
    }
    const SparseSummary& s = *sparse_summary_;
    ImGui::TextColored(t.text_dim, "%s", ICON_FA_TABLE_CELLS);
    ImGui::SameLine();
    ImGui::Text("Sparse features \xC2\xB7 %s rows \xC3\x97 %s features \xC2\xB7 exact", Count(s.rows_all).c_str(), Count(s.features).c_str());
    if (sparse_task_) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, ICON_FA_SPINNER " updating");
    }
    // The label filter, in words, with Clear.
    if (!sparse_keep_.empty()) {
        ImGui::TextColored(t.text_dim, "FILTERS");
        ImGui::SameLine();
        std::string text;
        for (const auto& k : sparse_keep_) text += (text.empty() ? "" : ", ") + k;
        ImGui::Text("label = %s", text.c_str());
        ImGui::SameLine();
        if (ui::LinkButton("Clear all")) {
            sparse_keep_.clear();
            sparse_dirty_ = true;
        }
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "\xC2\xB7 %s of %s rows", Count(s.rows).c_str(), Count(s.rows_all).c_str());
    }

    // The KPI strip.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f%%", s.density * 100.0);
    const std::string density = buf;
    std::snprintf(buf, sizeof(buf), "%.1f MB", s.memory_mb);
    const std::string memory = buf;
    const std::pair<const char*, std::string> kpis[] = {
        {"Rows", Count(s.rows) + (s.rows != s.rows_all ? " of " + Count(s.rows_all) : std::string())},
        {"Features", Count(s.features)},
        {"Non-zero values", Count(s.nnz)},
        {"Density", density},
        {"Memory", memory},
        {"Labels", s.classes.empty() ? std::string("none") : std::to_string(s.classes.size()) + (s.classes.size() == 1 ? " class" : " classes")},
    };
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float kpi_w = (ImGui::GetContentRegionAvail().x - gap * 5) / 6.0f;
    for (size_t i = 0; i < 6; ++i) {
        if (i) ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
        ImGui::BeginChild(("##skpi" + std::to_string(i)).c_str(), ImVec2(kpi_w, ImGui::GetTextLineHeight() * 3.2f),
                          ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleColor();
        ImGui::TextColored(t.text_dim, "%s", kpis[i].first);
        ImGui::TextColored(t.text_bright, "%s", kpis[i].second.c_str());
        ImGui::EndChild();
    }

    // The cards, three across.
    ImGui::BeginChild("##sparse_cards", ImVec2(0, 0));
    const float card_w = (ImGui::GetContentRegionAvail().x - gap * 2) / 3.0f;
    const float card_h = 330.0f;
    int slot = 0;
    const auto card = [&](const char* title, const char* note, auto&& body) {
        if (slot % 3) ImGui::SameLine();
        ++slot;
        ImGui::PushID(title);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
        ImGui::BeginChild("##card", ImVec2(card_w, card_h), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleColor();
        ImGui::TextUnformatted(title);
        if (note && *note) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_faint, "%s", note);
        }
        body(ImVec2(-1, ImGui::GetContentRegionAvail().y));
        ImGui::EndChild();
        ImGui::PopID();
    };
    const auto view_of = [&](const std::string& key, const std::function<plot::Prepared()>& make) -> plot::PlotView& {
        auto& v = sparse_views_[key];
        if (!v) {
            v = std::make_unique<plot::PlotView>(id_ + "_sparse_" + key);
            v->SetData(make());
        }
        return *v;
    };
    plot::PlotView::Options o;
    o.toolbar = false;
    o.own_window_button = false;

    if (!s.classes.empty()) {
        card("Labels", "click a bar to filter", [&](ImVec2 size) {
            auto& v = view_of("labels", [&] {
                return Bars("label", s.classes, AsDoubles(s.class_rows), "rows", false);
            });
            if (!sparse_keep_.empty()) {
                v.SetBackground(std::make_shared<plot::Prepared>(Bars("label", s.classes, AsDoubles(s.class_rows_all), "rows", false)));
            } else {
                v.SetBackground(nullptr);
            }
            v.SetHighlight(std::vector<std::string>(sparse_keep_.begin(), sparse_keep_.end()), NAN, NAN);
            v.Draw(size, o);
            if (auto click = v.TakeClick(); click && !click->value.empty()) {
                if (sparse_keep_.size() == 1 && sparse_keep_.count(click->value)) sparse_keep_.clear();
                else sparse_keep_ = {click->value};
                sparse_dirty_ = true;
            }
        });
    }
    card("Top features", "total weight", [&](ImVec2 size) {
        auto& v = view_of("top", [&] {
            std::vector<std::string> names;
            std::vector<double> weights;
            for (const auto& [n, w] : s.top_features) {
                names.push_back(n);
                weights.push_back(w);
            }
            return Bars("feature", names, weights, "weight", true);
        });
        v.Draw(size, o);
    });
    card("Features per row", "", [&](ImVec2 size) {
        view_of("per_row", [&] { return Histogram("features", s.features_per_row, "features used per row", false); }).Draw(size, o);
    });
    card("Feature spread", "rows that use each feature", [&](ImVec2 size) {
        view_of("spread", [&] { return Histogram("rows", s.rows_per_feature, "rows per feature", true); }).Draw(size, o);
    });
    if (!s.by_class_features.empty()) {
        card("Top features by class", "mean weight", [&](ImVec2 size) {
            auto& v = view_of("by_class", [&] {
                plot::PlotSpec spec;
                spec.kind = plot::Kind::Heatmap;
                spec.x_column = "class";
                spec.y_columns = {"feature"};
                spec.value_column = "weight";
                spec.legend = false;
                std::vector<std::string> cls, feat;
                std::vector<double> weight;
                for (size_t f = 0; f < s.by_class_features.size(); ++f)
                    for (size_t c = 0; c < s.classes.size(); ++c) {
                        if (s.class_rows[c] == 0) continue;  // a class the filter left out
                        cls.push_back(s.classes[c]);
                        feat.push_back(s.by_class_features[f]);
                        weight.push_back(s.by_class_weight[f * s.classes.size() + c]);
                    }
                plot::Source src;
                src.columns = {Text("class", cls), Text("feature", feat), Numbers("weight", weight)};
                return plot::Prepare(spec, src);
            });
            v.Draw(size, o);
        });
    }
    card("Rows", s.classes.empty() ? "" : "the kept rows", [&](ImVec2 size) {
        const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("##rows", s.classes.empty() ? 3 : 4, flags, size)) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            if (!s.classes.empty()) ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 90.0f);
            ImGui::TableSetupColumn("features", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("strongest features");
            ImGui::TableHeadersRow();
            for (const auto& r : s.sample) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%zu", r.row);
                if (!s.classes.empty()) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(r.label.c_str());
                }
                ImGui::TableNextColumn();
                ImGui::Text("%zu", r.used);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.strongest.c_str());
            }
            ImGui::EndTable();
        }
    });
    ImGui::EndChild();
}

}  // namespace cyxwiz::dashboard
