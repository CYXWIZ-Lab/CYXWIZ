#pragma once

// The Dashboard window (TOFIX134 P3.8, approved board 13): a Dashboard node's
// data at its position, shown as a summary strip and a grid of widgets (KPIs,
// tables, every plot type) that start from an automatic layout. One shared
// filter: clicking a bar, slice or bin filters every other widget. Fields are
// listed with their roles (edited in Data Studio). The layout and filters are
// saved in the node (on_spec_changed). UI thread only.

#include "../../core/dashboard/dashboard_model.h"
#include "../../core/dashboard/dashboard_session.h"
#include "../../core/dataset_contract.h"
#include "../../core/dataset_profiler.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

namespace cyxwiz::plot {
class PlotView;
class PlotWindow;
}

namespace cyxwiz::dashboard {

class DashboardWindow {
public:
    explicit DashboardWindow(std::string id);
    ~DashboardWindow();

    bool visible = false;
    std::function<void()> draw_header;                               // the node's data state (shared with Plot nodes)
    std::function<void(const std::string& json)> on_spec_changed;    // save the layout in the node
    std::function<void(const std::string& dataset)> on_edit_roles;   // open Data Studio's Profile on the dataset
    // Open Data Studio's Query tab on the dataset with this SQL (Open in Data Studio, Open in Query tab).
    std::function<void(const std::string& dataset, const std::string& sql)> on_open_query;

    void SetSpecJson(const std::string& json);
    // The data at the node: a registry dataset (catalog name) and how to call it.
    void SetData(const std::string& dataset_name, const std::string& title);
    void ClearData(const std::string& message);
    void Render();

private:
    void EnsureProfile();
    void RebuildContract();
    void SaveIfChanged();
    void AddWidget(const std::string& kind_id);
    std::string FirstField(FieldNeed need, const std::string& other = {}) const;
    void DrawToolbar();
    void DrawFilters();
    void DrawFields(float width);
    void DrawStrip();
    void DrawGrid(float width);
    void DrawCard(WidgetSpec& w, float width, float height);
    void DrawSettings(float width);
    void OpenInPlotWindow(WidgetSpec& w);
    // The table as people call it (queries shown or opened elsewhere name it so).
    std::string ShownName() const;
    void FinishExport();

    std::string id_;
    std::string title_;
    std::string dataset_;
    std::string message_ = "Not read yet: refresh to read the data at this node.";
    DashboardSpec spec_;
    std::string saved_json_;
    DashboardSession session_;
    std::shared_ptr<DatasetProfile> profile_;
    uint64_t profiled_generation_ = ~0ull;
    uint64_t profile_task_ = 0;
    std::string profile_error_;
    DatasetContract contract_;
    std::string target_;
    std::map<std::string, std::unique_ptr<plot::PlotView>> views_;
    std::map<std::string, uint64_t> view_versions_;
    std::string selected_;
    std::unique_ptr<plot::PlotWindow> editor_;
    std::string editing_;
    char title_buf_[128] = {};
    std::string title_for_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    // Export PNG: the centre is read back the frame after the click.
    struct Capture {
        bool ready = false;
        std::vector<unsigned char> png;
    };
    std::shared_ptr<Capture> capture_;
    int capture_frame_ = 0;
    ImVec2 centre_min_, centre_max_;
    std::string note_;
    double note_until_ = 0;
    std::string sql_for_;   // the widget whose SQL is shown (View SQL)
};

}  // namespace cyxwiz::dashboard
