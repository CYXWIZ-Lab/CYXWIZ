#pragma once

#include <string>
#include <vector>
#include <memory>
#include <deque>

namespace cyxwiz {

class InferenceServer;
class SequentialModel;
struct RequestLogEntry;

// Panel for controlling and monitoring the inference server
class ServingPanel {
public:
    ServingPanel();
    ~ServingPanel();

    void Render();
    void Toggle() { visible_ = !visible_; }
    bool IsVisible() const { return visible_; }
    void Show() { visible_ = true; }
    void Hide() { visible_ = false; }

    // Model management
    void SetModel(SequentialModel* model, const std::string& name);
    void ClearModel();

    // Get server instance (for external control)
    InferenceServer* GetServer() { return server_.get(); }

private:
    void RenderServerControl();
    void RenderMetricsDashboard();
    void RenderRequestLog();
    void RenderTestInference();
    void RenderLatencyChart();

    // UI state
    bool visible_ = false;
    int selected_tab_ = 0;

    // Server control
    std::unique_ptr<InferenceServer> server_;
    int port_ = 8080;
    char port_buffer_[16] = "8080";

    // Test inference
    char test_input_[4096] = "[1.0, 2.0, 3.0, 4.0]";
    std::string test_output_;
    std::string test_error_;
    float test_latency_ = 0.0f;

    // Latency history for chart
    std::deque<float> latency_history_;
    static constexpr size_t MAX_LATENCY_POINTS = 100;

    // Status message
    std::string status_message_;
    bool status_is_error_ = false;
};

} // namespace cyxwiz
