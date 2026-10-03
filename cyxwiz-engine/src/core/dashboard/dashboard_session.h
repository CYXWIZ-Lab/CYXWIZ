#pragma once

// A dashboard's live state (TOFIX134 P3.7, dashboard_architecture.md L5):
// each frame Poll() checks the bindings, and every widget whose data, spec
// or filters changed gets its query run in the background (the session query
// service, then the plot preparation, off the UI thread). A filter change
// waits 150 ms for the next click before querying; a superseded query is
// cancelled; results are kept by fingerprint (data version, widget, filters)
// so going back to an earlier selection is instant. UI thread only.

#include "../dataset_profiler.h"
#include "../plot/plot_model.h"
#include "dashboard_runtime.h"

#include <arrow/api.h>

#include <chrono>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz::dashboard {

struct WidgetResult {
    enum class State { Waiting, Running, Ready, Failed, Unbound };
    State state = State::Waiting;
    std::string message;                              // Failed / Unbound: in words
    Binding binding;                                  // Unbound: what to rebind
    std::shared_ptr<const plot::Prepared> prepared;   // plot widgets
    std::shared_ptr<const plot::Prepared> all_rows;   // the same over all rows when filters apply (bars, histograms)
    std::shared_ptr<arrow::Table> table;              // table widgets
    double value = NAN, all = NAN;                    // KPI widgets (filtered, all rows)
    bool sampled = false;                             // drawn from a sample of the rows
    uint64_t version = 0;                             // bumps each time the result changes
    std::string fingerprint;
};

struct StripResult {
    bool ready = false;
    double rows_now = 0, rows_all = 0, missing_now = NAN;
    double target_now = NAN, target_all = NAN;        // a number target: its mean
    std::string target_text_now, target_text_all;      // a category target: its most frequent value
    std::string fingerprint;
};

class DashboardSession {
public:
    DashboardSession();
    ~DashboardSession();

    void SetDataset(const std::string& catalog_name);
    const std::string& Dataset() const { return dataset_; }

    // UI thread, each frame. Updates spec.known_types when every widget is bound.
    void Poll(DashboardSpec& spec, const DatasetContract& contract, const DatasetProfile& profile);
    // Queries everything again (the dashboard's Refresh).
    void RefreshAll();
    const WidgetResult& ResultOf(const std::string& widget_id) const;
    const StripResult& Strip() const { return strip_; }
    bool Busy() const { return !running_.empty(); }

    static constexpr size_t kRowCap = 1000000;  // plot widgets sample beyond this many rows

private:
    struct Running {
        uint64_t task = 0;
        std::string fingerprint;
    };
    void Start(const WidgetSpec& w, const DatasetProfile& profile, const std::string& fingerprint, const FilterState& filters);
    void StartStrip(const DashboardSpec& spec, const DatasetContract& contract, const DatasetProfile& profile, const std::string& fp);
    void Remember(const WidgetResult& r);

    std::string dataset_;
    uint64_t generation_ = 0;
    uint64_t epoch_ = 0;                              // RefreshAll bumps it
    std::map<std::string, WidgetResult> results_;
    std::map<std::string, Running> running_;          // widget id (or "#strip") -> its query
    std::list<WidgetResult> cache_;                   // most recent first
    StripResult strip_;
    std::string filter_text_;
    std::chrono::steady_clock::time_point filter_changed_{};
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    WidgetResult empty_;
    static constexpr size_t kCacheSize = 64;
};

}  // namespace cyxwiz::dashboard
