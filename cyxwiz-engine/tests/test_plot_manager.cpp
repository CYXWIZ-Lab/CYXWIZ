// PlotManager (TOFIX134 P0 item 5): the plot type and config reach the
// backend, the config starts with defined values, and an unknown plot id
// does not crash.
#include "plotting/plot_manager.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz::plotting;

namespace {
int failures = 0;
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Records what the manager asked for.
struct FakeBackend : PlotBackend {
    std::vector<std::string> calls;
    bool Initialize(int, int) override { return true; }
    void Shutdown() override {}
    void BeginPlot(const char* title) override { calls.push_back(std::string("begin:") + title); }
    void EndPlot() override { calls.push_back("end"); }
    void PlotLine(const char* l, const double*, const double*, int n) override { calls.push_back("line:" + std::string(l) + ":" + std::to_string(n)); }
    void PlotScatter(const char* l, const double*, const double*, int n) override { calls.push_back("scatter:" + std::string(l) + ":" + std::to_string(n)); }
    void PlotBars(const char* l, const double*, const double*, int n) override { calls.push_back("bars:" + std::string(l) + ":" + std::to_string(n)); }
    void PlotHistogram(const char* l, const double*, int n, int) override { calls.push_back("hist:" + std::string(l) + ":" + std::to_string(n)); }
    void PlotHeatmap(const char*, const double*, int, int) override {}
    void PlotBoxPlot(const char* l, const double*, int n) override { calls.push_back("box:" + std::string(l) + ":" + std::to_string(n)); }
    void PlotStems(const char*, const double*, const double*, int) override {}
    void PlotStairs(const char*, const double*, const double*, int) override {}
    void PlotPieChart(const char*, const double*, const char* const*, int) override {}
    void PlotPolarLine(const char*, const double*, const double*, int) override {}
    void SetAxisLabel(int axis, const char* label) override { calls.push_back("axis" + std::to_string(axis) + ":" + label); }
    void SetAxisLimits(int, double, double) override {}
    void SetAxisAutoFit(int, bool) override {}
    void SetTitle(const char*) override {}
    void SetLegendVisible(bool v) override { calls.push_back(v ? "legend:on" : "legend:off"); }
    void SetGridVisible(bool v) override { calls.push_back(v ? "grid:on" : "grid:off"); }
    bool SaveToFile(const char*) override { return true; }
    const char* GetBackendName() const override { return "fake"; }
    bool IsRealtime() const override { return true; }
};

bool Has(const std::vector<std::string>& calls, const std::string& what) {
    for (const auto& c : calls)
        if (c == what) return true;
    return false;
}
}  // namespace

int main() {
    const PlotManager::PlotConfig defaults;
    Check(defaults.type == PlotManager::PlotType::Line && defaults.backend == PlotManager::BackendType::ImPlot,
          "a default config has defined type and backend");

    std::unordered_map<std::string, PlotDataset> datasets;
    PlotDataset ds;
    ds.AddSeries("loss");
    ds.GetSeries("loss")->AddPoint(1, 0.9);
    ds.GetSeries("loss")->AddPoint(2, 0.5);
    datasets["d"] = ds;

    PlotManager::PlotConfig cfg;
    cfg.title = "Training";
    cfg.x_label = "epoch";
    cfg.y_label = "loss";
    cfg.show_legend = false;
    cfg.show_grid = true;
    cfg.type = PlotManager::PlotType::Scatter;
    FakeBackend scatter;
    PlotManager::DrawPlot(scatter, cfg, datasets);
    Check(Has(scatter.calls, "scatter:loss:2") && !Has(scatter.calls, "line:loss:2"), "a scatter plot draws a scatter");
    Check(Has(scatter.calls, "axis0:epoch") && Has(scatter.calls, "axis1:loss"), "axis labels from the config");
    Check(Has(scatter.calls, "legend:off") && Has(scatter.calls, "grid:on"), "legend and grid from the config");
    Check(Has(scatter.calls, "begin:Training") && scatter.calls.back() == "end", "begin with the title, end last");

    cfg.type = PlotManager::PlotType::Histogram;
    FakeBackend hist;
    PlotManager::DrawPlot(hist, cfg, datasets);
    Check(Has(hist.calls, "hist:loss:2"), "a histogram draws a histogram");
    cfg.type = PlotManager::PlotType::Bar;
    FakeBackend bars;
    PlotManager::DrawPlot(bars, cfg, datasets);
    Check(Has(bars.calls, "bars:loss:2"), "a bar plot draws bars");

    Check(!PlotManager::GetInstance().UpdateRealtimePlot("no such plot", 1.0, 2.0, "s"),
          "an unknown plot id is refused, not dereferenced");
    if (failures) return 1;
    std::cout << "plot manager: defaults, type and config reach the backend, unknown id refused. OK\n";
    return 0;
}
