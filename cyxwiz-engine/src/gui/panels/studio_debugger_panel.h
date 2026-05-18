#pragma once

#include "../panel.h"
#include "../../core/debug_executor.h"
#include "../icons.h"
#include <cstddef>
#include <imgui.h>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct StudioDebuggerSnapshot {
    bool success = false;
    bool has_debug_result = false;

    uint64_t graph_hash = 0;
    size_t node_count = 0;
    size_t link_count = 0;

    std::string graph_summary;
    std::string sample_summary = "Synthetic sample 0 (POC)";
    std::string failure_summary;

    std::vector<ValidationIssue> issues;
    DebugResult debug_result;
};

class StudioDebuggerPanel : public Panel {
public:
    StudioDebuggerPanel();
    ~StudioDebuggerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_BUG; }

    using RunDebugCallback = std::function<StudioDebuggerSnapshot()>;
    using FocusNodeCallback = std::function<void(int)>;

    void SetRunDebugCallback(RunDebugCallback callback) { run_debug_callback_ = std::move(callback); }
    void SetFocusNodeCallback(FocusNodeCallback callback) { focus_node_callback_ = std::move(callback); }

    void SetSession(const StudioDebuggerSnapshot& session);
    void Clear();

    bool HasSession() const { return has_session_; }

private:
    void RenderToolbar();
    void RenderOverview();
    void RenderTraceTimeline();
    void RenderSelectedTraceDetails();
    void RenderIssueList();
    static std::string FormatShape(const std::vector<size_t>& shape);

    RunDebugCallback run_debug_callback_;
    FocusNodeCallback focus_node_callback_;

    StudioDebuggerSnapshot session_;
    bool has_session_ = false;
    int selected_trace_index_ = -1;
};

} // namespace cyxwiz
