#include "bar_chart_dialog.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"

#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#include <arrow/api.h>
#include <arrow/array.h>

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_set>

namespace gui::visualization {

namespace {
constexpr int64_t kMaxScanRows = 1'000'000;

// Stringify a single row from a chunked Arrow column. Returns empty
// string for nulls. Supports the types users typically categorize on —
// strings, ints. Floats are supported but tend to be noisy for a bar
// chart (infinite distinct values); flagged with a truncated_ warning
// when the unique count hits the cap.
std::string ValueAsString(const std::shared_ptr<arrow::Array>& chunk,
                          int64_t idx) {
    if (!chunk || chunk->IsNull(idx)) return "";
    switch (chunk->type_id()) {
        case arrow::Type::STRING: {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            return arr->GetString(idx);
        }
        case arrow::Type::LARGE_STRING: {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            return arr->GetString(idx);
        }
        case arrow::Type::INT8: {
            auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
            return std::to_string(static_cast<int>(arr->Value(idx)));
        }
        case arrow::Type::UINT8: {
            auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
            return std::to_string(static_cast<int>(arr->Value(idx)));
        }
        case arrow::Type::INT16: {
            auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::UINT16: {
            auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::INT32: {
            auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::UINT32: {
            auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::INT64: {
            auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::UINT64: {
            auto arr = std::static_pointer_cast<arrow::UInt64Array>(chunk);
            return std::to_string(arr->Value(idx));
        }
        case arrow::Type::BOOL: {
            auto arr = std::static_pointer_cast<arrow::BooleanArray>(chunk);
            return arr->Value(idx) ? "true" : "false";
        }
        default:
            // Floats / dates / etc. fall here. For a bar chart those
            // rarely make sense — the user will see the "unsupported"
            // message and re-pick a categorical column.
            return {};
    }
}
}  // namespace

BarChartDialog::BarChartDialog(MLNode* node)
    : NodeConfigDialog("Bar Chart", node) {
    Reset();
}

std::string BarChartDialog::FindUpstreamDatasetName() const {
    // BFS upstream from this BarChart node through the graph's link
    // topology, following to_node → from_node edges. Stop at the first
    // ancestor with NodeType::DataInput or DatasetInput and return its
    // dataset_name parameter. Bounded by visited set so cycles (which
    // the compile gate already rejects) can't infinite-loop.
    if (!node_editor_ || !node_) return {};

    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();

    // node_id → node* for O(1) lookup.
    std::unordered_map<int, const MLNode*> node_by_id;
    node_by_id.reserve(nodes.size());
    for (const auto& n : nodes) node_by_id[n.id] = &n;

    // Reverse adjacency: to_node → list of from_node IDs.
    std::unordered_map<int, std::vector<int>> upstream;
    for (const auto& link : links) {
        upstream[link.to_node].push_back(link.from_node);
    }

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(node_->id);
    visited.insert(node_->id);

    while (!queue.empty()) {
        int id = queue.front();
        queue.pop();

        auto it = upstream.find(id);
        if (it == upstream.end()) continue;
        for (int up_id : it->second) {
            if (!visited.insert(up_id).second) continue;
            auto n_it = node_by_id.find(up_id);
            if (n_it == node_by_id.end()) continue;
            const MLNode* n = n_it->second;
            if (n->type == NodeType::DataInput ||
                n->type == NodeType::DatasetInput) {
                auto p = n->parameters.find("dataset_name");
                if (p != n->parameters.end() && !p->second.empty()) {
                    return p->second;
                }
            }
            queue.push(up_id);
        }
    }
    return {};
}

void BarChartDialog::Apply() {
    if (!node_) return;
    node_->parameters["chart_type"] = chart_type_;
    node_->parameters["column"]     = column_;
    node_->parameters["title"]      = title_;
    node_->parameters["max_bars"]   = std::to_string(max_bars_);
    node_->parameters["dataset_name"] = dataset_name_;
}

void BarChartDialog::Reset() {
    if (!node_) return;
    auto get = [&](const char* key, const std::string& fallback) {
        auto it = node_->parameters.find(key);
        return (it != node_->parameters.end()) ? it->second : fallback;
    };
    chart_type_   = get("chart_type",   "bar");
    column_       = get("column",       "");
    title_        = get("title",        "Bar Chart");
    dataset_name_ = get("dataset_name", "");
    try {
        max_bars_ = std::stoi(get("max_bars", "20"));
    } catch (...) { max_bars_ = 20; }
    if (max_bars_ < 1) max_bars_ = 1;

    counts_.clear();
    cache_key_.clear();
    total_rows_ = 0;
    truncated_ = false;
    last_error_.clear();
}

void BarChartDialog::RenderContent() {
    // First render after Open: if the node has no dataset hint but
    // there's an upstream DataInput we can see, auto-populate. Users
    // who explicitly wired DataInput → BarChart get the expected
    // "dataset already selected" feel without needing to pick in the
    // dropdown. first_open_ is cleared by the base Render() after
    // the first pass.
    if (first_open_ && dataset_name_.empty()) {
        std::string upstream = FindUpstreamDatasetName();
        if (!upstream.empty()) {
            dataset_name_ = upstream;
            cache_key_.clear();  // force recompute with new dataset
            has_changes_ = true;
            spdlog::info("[BarChartDialog] auto-populated dataset "
                         "'{}' from upstream DataInput", upstream);
        }
    }

    RenderConfigPanel();
    ImGui::Separator();
    RenderChart();
}

void BarChartDialog::RenderConfigPanel() {
    auto& reg = cyxwiz::DataRegistry::Instance();

    // Dataset picker — dropdown of every Arrow dataset currently loaded.
    // Until runtime pin-walking lands, the user picks manually; the
    // BarChart node's wired input serves as documentation for now.
    std::vector<std::string> datasets;
    for (const auto& name : reg.GetDatasetNames()) {
        if (reg.IsArrowDataset(name)) datasets.push_back(name);
    }

    ImGui::Text("Dataset:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(260);
    const char* dataset_preview =
        dataset_name_.empty() ? "(select a dataset)" : dataset_name_.c_str();
    if (ImGui::BeginCombo("##dataset", dataset_preview)) {
        for (const auto& name : datasets) {
            bool sel = (name == dataset_name_);
            if (ImGui::Selectable(name.c_str(), sel)) {
                if (name != dataset_name_) {
                    dataset_name_ = name;
                    column_.clear();  // columns change per dataset
                    has_changes_ = true;
                }
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // Column picker — populated from the selected dataset's schema.
    std::vector<std::string> columns;
    if (!dataset_name_.empty() && reg.IsArrowDataset(dataset_name_)) {
        auto ds = reg.GetArrowDataset(dataset_name_);
        if (ds) columns = ds->GetColumnNames();
    }

    ImGui::Text("Column:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(260);
    const char* col_preview =
        column_.empty() ? "(pick a column)" : column_.c_str();
    if (ImGui::BeginCombo("##column", col_preview)) {
        for (const auto& col : columns) {
            bool sel = (col == column_);
            if (ImGui::Selectable(col.c_str(), sel)) {
                if (col != column_) {
                    column_ = col;
                    has_changes_ = true;
                }
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // Title input. Free-text; written to node param on Apply.
    char title_buf[256];
    std::snprintf(title_buf, sizeof(title_buf), "%s", title_.c_str());
    ImGui::Text("Title:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(260);
    if (ImGui::InputText("##title", title_buf, sizeof(title_buf))) {
        title_ = title_buf;
        has_changes_ = true;
    }

    // Chart-type selector (only bar / horizontal_bar wired for now).
    const char* types[] = {"bar", "horizontal_bar"};
    int type_idx = (chart_type_ == "horizontal_bar") ? 1 : 0;
    ImGui::Text("Orient:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(260);
    if (ImGui::Combo("##chart_type", &type_idx, types, IM_ARRAYSIZE(types))) {
        chart_type_ = types[type_idx];
        has_changes_ = true;
    }

    // Max-bars cap.
    ImGui::Text("Max bars:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##max_bars", &max_bars_)) {
        if (max_bars_ < 1) max_bars_ = 1;
        if (max_bars_ > 200) max_bars_ = 200;
        has_changes_ = true;
    }
}

void BarChartDialog::ComputeDistribution() {
    std::string key = dataset_name_ + "|" + column_;
    if (key == cache_key_) return;

    counts_.clear();
    total_rows_ = 0;
    truncated_ = false;
    last_error_.clear();
    cache_key_ = key;

    if (dataset_name_.empty() || column_.empty()) return;

    auto& reg = cyxwiz::DataRegistry::Instance();
    if (!reg.IsArrowDataset(dataset_name_)) {
        last_error_ = "Dataset is not loaded as Arrow (disk-backed "
                      "Parquet not yet supported in this dialog).";
        return;
    }
    auto ds = reg.GetArrowDataset(dataset_name_);
    if (!ds) { last_error_ = "Dataset handle null."; return; }
    auto table = ds->GetArrowTable();
    if (!table) { last_error_ = "Arrow table null."; return; }

    int col_idx = -1;
    auto names = ds->GetColumnNames();
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == column_) { col_idx = static_cast<int>(i); break; }
    }
    if (col_idx < 0) {
        last_error_ = "Column '" + column_ + "' not in dataset schema.";
        return;
    }

    auto chunked = table->column(col_idx);
    std::map<std::string, int> tally;
    int64_t scanned = 0;
    bool type_unsupported = false;
    for (int c = 0; c < chunked->num_chunks(); ++c) {
        auto chunk = chunked->chunk(c);
        for (int64_t i = 0; i < chunk->length(); ++i) {
            if (scanned >= kMaxScanRows) { truncated_ = true; break; }
            std::string v = ValueAsString(chunk, i);
            if (v.empty()) {
                if (!chunk->IsNull(i)) type_unsupported = true;
            } else {
                tally[v]++;
            }
            ++scanned;
        }
        if (scanned >= kMaxScanRows) break;
    }
    total_rows_ = scanned;
    if (type_unsupported) {
        last_error_ = "Column type not supported for bar chart (try a "
                      "string or integer column).";
    }

    counts_.reserve(tally.size());
    for (auto& kv : tally) counts_.emplace_back(kv.first, kv.second);
    std::sort(counts_.begin(), counts_.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (static_cast<int>(counts_.size()) > max_bars_) {
        counts_.resize(max_bars_);
        truncated_ = true;
    }
}

void BarChartDialog::RenderChart() {
    ComputeDistribution();

    if (!last_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s",
                           last_error_.c_str());
        return;
    }
    if (counts_.empty()) {
        ImGui::TextDisabled("Pick a dataset and column above to render "
                            "the distribution.");
        return;
    }

    ImGui::Text("%lld rows scanned%s  |  %zu bars%s",
                static_cast<long long>(total_rows_),
                truncated_ ? " (truncated)" : "",
                counts_.size(),
                truncated_ ? " (top-N)" : "");

    // Flag imbalance — catches the common class-imbalance trap.
    if (counts_.size() >= 2) {
        int max_c = counts_.front().second;
        int min_c = counts_.back().second;
        if (min_c > 0 && max_c / min_c >= 10) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                               "  imbalance %.1fx",
                               static_cast<float>(max_c) /
                               static_cast<float>(min_c));
        }
    }

    std::vector<double> values;
    std::vector<const char*> labels;
    values.reserve(counts_.size());
    labels.reserve(counts_.size());
    for (const auto& kv : counts_) {
        values.push_back(static_cast<double>(kv.second));
        labels.push_back(kv.first.c_str());
    }

    const bool horizontal = (chart_type_ == "horizontal_bar");
    int n = static_cast<int>(values.size());
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float plot_h = std::max(240.0f, avail.y - 20.0f);

    if (ImPlot::BeginPlot(title_.c_str(), ImVec2(-1, plot_h),
                          ImPlotFlags_NoMouseText)) {
        if (horizontal) {
            ImPlot::SetupAxes("count", nullptr,
                              ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisTicks(ImAxis_Y1, 0, n - 1, n,
                                   labels.data(), false);
            ImPlot::PlotBars("##counts", values.data(), n, 0.7,
                             0.0, ImPlotBarsFlags_Horizontal);
        } else {
            ImPlot::SetupAxes(nullptr, "count",
                              ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n,
                                   labels.data(), false);
            ImPlot::PlotBars("##counts", values.data(), n, 0.7);
        }
        ImPlot::EndPlot();
    }
}

} // namespace gui::visualization
