// Windows header order fix - must come before httplib.h
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "serving_panel.h"
#include "../../serving/inference_server.h"
#include <cyxwiz/sequential.h>
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cstring>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace cyxwiz {

using json = nlohmann::json;

ServingPanel::ServingPanel()
    : server_(std::make_unique<InferenceServer>()) {

    // Set up status callback
    server_->SetStatusCallback([this](bool running, const std::string& message) {
        status_message_ = message;
        status_is_error_ = !running && message.find("Failed") != std::string::npos;
    });
}

ServingPanel::~ServingPanel() {
    if (server_ && server_->IsRunning()) {
        server_->Stop();
    }
}

void ServingPanel::SetModel(SequentialModel* model, const std::string& name) {
    if (server_) {
        server_->SetModel(model);
        server_->SetModelName(name);
        spdlog::info("ServingPanel: Model set - {}", name);
    }
}

void ServingPanel::ClearModel() {
    if (server_) {
        if (server_->IsRunning()) {
            server_->Stop();
        }
        server_->SetModel(nullptr);
        server_->SetModelName("No Model");
    }
}

void ServingPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Model Serving", &visible_)) {
        // Tab bar
        if (ImGui::BeginTabBar("ServingTabs")) {
            if (ImGui::BeginTabItem("Server Control")) {
                RenderServerControl();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Metrics")) {
                RenderMetricsDashboard();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Request Log")) {
                RenderRequestLog();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Test Inference")) {
                RenderTestInference();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void ServingPanel::RenderServerControl() {
    bool running = server_ && server_->IsRunning();
    auto model = server_ ? server_->GetModel() : nullptr;

    // Status indicator
    ImGui::Text("Status:");
    ImGui::SameLine();
    if (running) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "RUNNING");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "STOPPED");
    }

    ImGui::Separator();

    // Model info
    ImGui::Text("Model:");
    ImGui::SameLine();
    if (model) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s",
            server_->GetModelName().c_str());
        ImGui::Text("Layers: %zu", model->Size());
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f), "No model loaded");
    }

    ImGui::Separator();

    // Port configuration
    ImGui::BeginDisabled(running);
    ImGui::Text("Port:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputText("##port", port_buffer_, sizeof(port_buffer_),
        ImGuiInputTextFlags_CharsDecimal)) {
        port_ = std::atoi(port_buffer_);
        if (port_ < 1 || port_ > 65535) {
            port_ = 8080;
            std::strcpy(port_buffer_, "8080");
        }
    }
    ImGui::EndDisabled();

    ImGui::Spacing();

    // Start/Stop button
    float button_width = 120;
    if (running) {
        if (ImGui::Button("Stop Server", ImVec2(button_width, 0))) {
            server_->Stop();
        }
    } else {
        ImGui::BeginDisabled(!model);
        if (ImGui::Button("Start Server", ImVec2(button_width, 0))) {
            port_ = std::atoi(port_buffer_);
            if (server_->Start(port_)) {
                status_message_ = "Server started on port " + std::to_string(port_);
                status_is_error_ = false;
            } else {
                status_message_ = "Failed to start server on port " + std::to_string(port_);
                status_is_error_ = true;
            }
        }
        ImGui::EndDisabled();
        if (!model) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "(Load a model first)");
        }
    }

    // Server URL
    if (running) {
        ImGui::Spacing();
        ImGui::Text("Server URL:");
        ImGui::SameLine();
        std::string url = server_->GetServerUrl();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", url.c_str());

        if (ImGui::Button("Copy URL")) {
            ImGui::SetClipboardText(url.c_str());
        }
    }

    // Status message
    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        if (status_is_error_) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", status_message_.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", status_message_.c_str());
        }
    }

    // API Documentation
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("API Endpoints")) {
        ImGui::Indent();
        ImGui::BulletText("GET  /health  - Health check");
        ImGui::BulletText("GET  /info    - Model information");
        ImGui::BulletText("POST /predict - Run inference");
        ImGui::BulletText("GET  /metrics - Server metrics");
        ImGui::Spacing();
        ImGui::TextWrapped("POST /predict expects JSON: {\"input\": [...], \"shape\": [batch, features]}");
        ImGui::Unindent();
    }
}

void ServingPanel::RenderMetricsDashboard() {
    if (!server_) return;

    const auto& metrics = server_->GetMetrics();
    bool running = server_->IsRunning();

    // Running status
    if (!running) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Server not running");
        return;
    }

    // Metrics cards
    float card_width = 150;
    float card_height = 60;

    ImGui::BeginGroup();

    // Total requests
    ImGui::BeginChild("TotalReq", ImVec2(card_width, card_height), true);
    ImGui::Text("Total Requests");
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%llu",
        static_cast<unsigned long long>(metrics.total_requests.load()));
    ImGui::EndChild();

    ImGui::SameLine();

    // Success rate
    ImGui::BeginChild("SuccessRate", ImVec2(card_width, card_height), true);
    ImGui::Text("Success Rate");
    uint64_t total = metrics.total_requests.load();
    uint64_t success = metrics.successful_requests.load();
    float rate = total > 0 ? (static_cast<float>(success) / total * 100.0f) : 100.0f;
    ImVec4 rate_color = rate >= 95 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
                       rate >= 80 ? ImVec4(0.8f, 0.8f, 0.2f, 1.0f) :
                                    ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
    ImGui::TextColored(rate_color, "%.1f%%", rate);
    ImGui::EndChild();

    ImGui::SameLine();

    // Requests per second
    ImGui::BeginChild("RPS", ImVec2(card_width, card_height), true);
    ImGui::Text("Requests/sec");
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "%.1f",
        metrics.requests_per_second.load());
    ImGui::EndChild();

    ImGui::SameLine();

    // Average latency
    ImGui::BeginChild("AvgLatency", ImVec2(card_width, card_height), true);
    ImGui::Text("Avg Latency");
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "%.2f ms",
        metrics.avg_latency_ms.load());
    ImGui::EndChild();

    ImGui::EndGroup();

    ImGui::Spacing();

    // Detailed stats
    ImGui::Columns(2, "MetricsColumns", false);

    ImGui::Text("Successful: %llu", static_cast<unsigned long long>(metrics.successful_requests.load()));
    ImGui::Text("Failed: %llu", static_cast<unsigned long long>(metrics.failed_requests.load()));

    ImGui::NextColumn();

    ImGui::Text("Min Latency: %.2f ms", metrics.min_latency_ms.load());
    ImGui::Text("Max Latency: %.2f ms", metrics.max_latency_ms.load());

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Latency chart
    RenderLatencyChart();

    // Reset button
    ImGui::Spacing();
    if (ImGui::Button("Reset Metrics")) {
        server_->ResetMetrics();
    }
}

void ServingPanel::RenderRequestLog() {
    if (!server_) return;

    // Get recent requests
    auto requests = server_->GetRecentRequests(100);

    // Toolbar
    if (ImGui::Button("Clear Log")) {
        server_->ClearRequestLog();
    }
    ImGui::SameLine();
    ImGui::Text("Showing %zu requests", requests.size());

    ImGui::Separator();

    // Request table
    if (ImGui::BeginTable("RequestLogTable", 5,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
        ImVec2(0, 300))) {

        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Method", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Endpoint", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Latency", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (const auto& req : requests) {
            ImGui::TableNextRow();

            // Time
            ImGui::TableNextColumn();
            auto time_t = std::chrono::system_clock::to_time_t(req.timestamp);
            char time_buf[32];
            std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&time_t));
            ImGui::Text("%s", time_buf);

            // Method
            ImGui::TableNextColumn();
            ImVec4 method_color = req.method == "POST" ?
                ImVec4(0.2f, 0.6f, 1.0f, 1.0f) : ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            ImGui::TextColored(method_color, "%s", req.method.c_str());

            // Endpoint
            ImGui::TableNextColumn();
            ImGui::Text("%s", req.endpoint.c_str());

            // Status
            ImGui::TableNextColumn();
            ImVec4 status_color = req.status_code == 200 ?
                ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
            ImGui::TextColored(status_color, "%d", req.status_code);

            // Latency
            ImGui::TableNextColumn();
            ImGui::Text("%.2f ms", req.latency_ms);
        }

        ImGui::EndTable();
    }
}

void ServingPanel::RenderTestInference() {
    if (!server_) return;

    bool running = server_->IsRunning();

    if (!running) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Start the server first");
        return;
    }

    ImGui::Text("Test Input (JSON array):");
    ImGui::InputTextMultiline("##test_input", test_input_, sizeof(test_input_),
        ImVec2(-1, 100), ImGuiInputTextFlags_AllowTabInput);

    if (ImGui::Button("Run Inference")) {
        test_output_.clear();
        test_error_.clear();

        try {
            // Make HTTP request to local server
            httplib::Client cli("localhost", server_->GetPort());
            cli.set_connection_timeout(5);
            cli.set_read_timeout(30);

            json request_body;
            request_body["input"] = json::parse(test_input_);

            auto start = std::chrono::high_resolution_clock::now();
            auto res = cli.Post("/predict", request_body.dump(), "application/json");
            auto end = std::chrono::high_resolution_clock::now();

            test_latency_ = std::chrono::duration<float, std::milli>(end - start).count();

            if (res) {
                if (res->status == 200) {
                    auto response = json::parse(res->body);
                    test_output_ = response.dump(2);
                } else {
                    auto response = json::parse(res->body);
                    test_error_ = response.value("error", "Unknown error");
                }
            } else {
                test_error_ = "Connection failed";
            }
        } catch (const std::exception& e) {
            test_error_ = std::string("Error: ") + e.what();
        }
    }

    ImGui::SameLine();
    ImGui::Text("Latency: %.2f ms", test_latency_);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Output
    if (!test_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", test_error_.c_str());
    } else if (!test_output_.empty()) {
        ImGui::Text("Output:");
        ImGui::InputTextMultiline("##test_output", const_cast<char*>(test_output_.c_str()),
            test_output_.size() + 1, ImVec2(-1, 150), ImGuiInputTextFlags_ReadOnly);
    }

    // Example inputs
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Example Inputs")) {
        ImGui::Indent();
        if (ImGui::Button("1D Array")) {
            std::strcpy(test_input_, "[1.0, 2.0, 3.0, 4.0, 5.0]");
        }
        ImGui::SameLine();
        if (ImGui::Button("2D Batch")) {
            std::strcpy(test_input_, "[[1.0, 2.0], [3.0, 4.0]]");
        }
        ImGui::SameLine();
        if (ImGui::Button("MNIST-like")) {
            std::strcpy(test_input_, "[0.0, 0.0, 0.3, 0.8, 0.9, 0.5, 0.0, 0.0]");
        }
        ImGui::Unindent();
    }
}

void ServingPanel::RenderLatencyChart() {
    if (!server_) return;

    // Get recent requests for latency data
    auto requests = server_->GetRecentRequests(MAX_LATENCY_POINTS);

    // Update latency history
    latency_history_.clear();
    for (auto it = requests.rbegin(); it != requests.rend(); ++it) {
        latency_history_.push_back(it->latency_ms);
    }

    if (latency_history_.empty()) {
        ImGui::Text("No latency data yet");
        return;
    }

    // Plot
    if (ImPlot::BeginPlot("Request Latency", ImVec2(-1, 150))) {
        ImPlot::SetupAxes("Request", "Latency (ms)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, MAX_LATENCY_POINTS, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImGuiCond_Once);

        std::vector<float> x_data(latency_history_.size());
        for (size_t i = 0; i < x_data.size(); ++i) {
            x_data[i] = static_cast<float>(i);
        }

        ImPlot::PlotLine("Latency", x_data.data(),
            const_cast<float*>(&latency_history_[0]),
            static_cast<int>(latency_history_.size()));

        ImPlot::EndPlot();
    }
}

} // namespace cyxwiz
