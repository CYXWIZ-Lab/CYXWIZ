#include "properties_node_editors.h"
#include "node_editor.h"
#include <imgui.h>
#include <implot.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gui::properties_node_editors {

void ScopeBuffer::Push(float t, float v) {
    times.push_back(t);
    values.push_back(v);
    while (static_cast<int>(times.size()) > max_samples) {
        times.pop_front();
        values.pop_front();
    }
}

void ScopeBuffer::Clear() {
    times.clear();
    values.clear();
}
void RenderNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    // Render editable parameters based on node type
    switch (node.type) {
        case NodeType::Dense: {
            // Units
            std::string& units = node.parameters["units"];
            if (units.empty()) units = "64";
            char u_buffer[16];
            strncpy(u_buffer, units.c_str(), sizeof(u_buffer) - 1);
            u_buffer[sizeof(u_buffer) - 1] = '\0';

            ImGui::Text("Units:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##units", u_buffer, sizeof(u_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                units = u_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::Conv2D: {
            // Filters
            std::string& filters = node.parameters["filters"];
            if (filters.empty()) filters = "32";
            char f_buffer[16];
            strncpy(f_buffer, filters.c_str(), sizeof(f_buffer) - 1);
            f_buffer[sizeof(f_buffer) - 1] = '\0';

            ImGui::Text("Filters:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##filters", f_buffer, sizeof(f_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                filters = f_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Kernel Size
            std::string& kernel = node.parameters["kernel_size"];
            if (kernel.empty()) kernel = "3";
            char k_buffer[16];
            strncpy(k_buffer, kernel.c_str(), sizeof(k_buffer) - 1);
            k_buffer[sizeof(k_buffer) - 1] = '\0';

            ImGui::Text("Kernel Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##kernel", k_buffer, sizeof(k_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                kernel = k_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "1";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Padding
            std::string& padding = node.parameters["padding"];
            if (padding.empty()) padding = "same";

            const char* paddings[] = { "same", "valid" };
            int current_padding = (padding == "valid") ? 1 : 0;

            ImGui::Text("Padding:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##padding", &current_padding, paddings, 2)) {
                padding = paddings[current_padding];
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation_conv", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::MaxPool2D: {
            // Pool Size
            std::string& pool_size = node.parameters["pool_size"];
            if (pool_size.empty()) pool_size = "2";
            char p_buffer[16];
            strncpy(p_buffer, pool_size.c_str(), sizeof(p_buffer) - 1);
            p_buffer[sizeof(p_buffer) - 1] = '\0';

            ImGui::Text("Pool Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##pool_size", p_buffer, sizeof(p_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                pool_size = p_buffer;
                context.invalidate_shapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "2";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride_pool", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                context.invalidate_shapes();
            }
            break;
        }

        case NodeType::Dropout: {
            std::string& rate_str = node.parameters["rate"];
            if (rate_str.empty()) rate_str = "0.5";

            float rate = std::stof(rate_str);
            ImGui::Text("Drop Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##rate", &rate, 0.0f, 0.9f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", rate);
                rate_str = buf;
            }
            break;
        }

        case NodeType::BatchNorm: {
            // Momentum
            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.99";

            float momentum = std::stof(momentum_str);
            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum", &momentum, 0.0f, 1.0f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", momentum);
                momentum_str = buf;
            }

            ImGui::Spacing();

            // Epsilon
            std::string& epsilon_str = node.parameters["epsilon"];
            if (epsilon_str.empty()) epsilon_str = "0.001";

            float epsilon = std::stof(epsilon_str);
            ImGui::Text("Epsilon:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##epsilon", &epsilon, 0.0001f, 0.01f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", epsilon);
                epsilon_str = buf;
            }
            break;
        }

        case NodeType::Output: {
            std::string& classes = node.parameters["classes"];
            if (classes.empty()) classes = "10";
            char c_buffer[16];
            strncpy(c_buffer, classes.c_str(), sizeof(c_buffer) - 1);
            c_buffer[sizeof(c_buffer) - 1] = '\0';

            ImGui::Text("Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##classes", c_buffer, sizeof(c_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                classes = c_buffer;
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Number of output classes");
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput:
        case NodeType::DataLoader:
        case NodeType::Augmentation:
        case NodeType::DataSplit:
            RenderDataPipelineNodeProperties(node, context);
            break;
        case NodeType::TensorReshape: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Reshape Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Shape
            std::string& shape = node.parameters["shape"];
            if (shape.empty()) shape = "-1,28,28,1";
            char shape_buffer[64];
            strncpy(shape_buffer, shape.c_str(), sizeof(shape_buffer) - 1);
            shape_buffer[sizeof(shape_buffer) - 1] = '\0';

            ImGui::Text("Target Shape:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##reshape", shape_buffer, sizeof(shape_buffer))) {
                shape = shape_buffer;
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use -1 for batch dimension");
            break;
        }

        case NodeType::Normalize: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Normalize Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Mean
            std::string& mean_str = node.parameters["mean"];
            if (mean_str.empty()) mean_str = "0.0";
            char mean_buffer[32];
            strncpy(mean_buffer, mean_str.c_str(), sizeof(mean_buffer) - 1);
            mean_buffer[sizeof(mean_buffer) - 1] = '\0';

            ImGui::Text("Mean:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##mean", mean_buffer, sizeof(mean_buffer))) {
                mean_str = mean_buffer;
            }

            ImGui::Spacing();

            // Std
            std::string& std_str = node.parameters["std"];
            if (std_str.empty()) std_str = "1.0";
            char std_buffer[32];
            strncpy(std_buffer, std_str.c_str(), sizeof(std_buffer) - 1);
            std_buffer[sizeof(std_buffer) - 1] = '\0';

            ImGui::Text("Standard Deviation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##std", std_buffer, sizeof(std_buffer))) {
                std_str = std_buffer;
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Common values:");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  MNIST: mean=0.1307, std=0.3081");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  ImageNet: mean=0.485,0.456,0.406");
            break;
        }

        case NodeType::OneHotEncode: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "One-Hot Encode Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Num classes
            std::string& num_classes = node.parameters["num_classes"];
            if (num_classes.empty()) num_classes = "10";
            char classes_buffer[16];
            strncpy(classes_buffer, num_classes.c_str(), sizeof(classes_buffer) - 1);
            classes_buffer[sizeof(classes_buffer) - 1] = '\0';

            ImGui::Text("Number of Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_classes", classes_buffer, sizeof(classes_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                num_classes = classes_buffer;
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "MNIST=10, CIFAR-10=10, ImageNet=1000");
            break;
        }

        // ========== Activation Functions ==========
        case NodeType::ReLU:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "ReLU Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(0, x)");
            break;

        case NodeType::Sigmoid:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Sigmoid Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = 1 / (1 + exp(-x))");
            break;

        case NodeType::Tanh:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Tanh Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = tanh(x)");
            break;

        case NodeType::Softmax:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Softmax Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x_i) = exp(x_i) / sum(exp(x))");
            break;

        case NodeType::LeakyReLU: {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Leaky ReLU Activation");

            std::string& slope_str = node.parameters["negative_slope"];
            if (slope_str.empty()) slope_str = "0.01";
            float slope = std::stof(slope_str);

            ImGui::Text("Negative Slope:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##neg_slope", &slope, 0.001f, 0.3f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", slope);
                slope_str = buf;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(slope*x, x)");
            break;
        }

        // ========== Loss Functions ==========
        case NodeType::MSELoss:
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Mean Squared Error Loss");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "L = mean((y - y_hat)^2)");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use for: Regression tasks");
            break;

        case NodeType::CrossEntropyLoss:
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Cross Entropy Loss");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "L = -sum(y * log(y_hat))");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use for: Classification tasks");
            break;

        // ========== Optimizers ==========
        case NodeType::SGD: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "SGD Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.01";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_sgd", &lr, 0.0001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", lr);
                lr_str = buf;
            }

            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.9";
            float momentum = std::stof(momentum_str);

            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum_sgd", &momentum, 0.0f, 0.99f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", momentum);
                momentum_str = buf;
            }
            break;
        }

        case NodeType::Adam: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "Adam Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adam", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }
            break;
        }

        case NodeType::AdamW: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "AdamW Optimizer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Adam with decoupled weight decay");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adamw", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1_w", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2_w", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }

            std::string& wd_str = node.parameters["weight_decay"];
            if (wd_str.empty()) wd_str = "0.01";
            float wd = std::stof(wd_str);

            ImGui::Text("Weight Decay:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##wd", &wd, 0.0f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", wd);
                wd_str = buf;
            }
            break;
        }

        case NodeType::Flatten:
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Flatten Layer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Flattens input to 1D vector");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[H, W, C] -> [H * W * C]");
            break;

        case NodeType::SignalSlider: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Slider");
            ImGui::Spacing();

            std::string& val_str = node.parameters["value"];
            std::string& min_str = node.parameters["min"];
            std::string& max_str = node.parameters["max"];
            float val = std::stof(val_str.empty() ? "0" : val_str);
            float mn = std::stof(min_str.empty() ? "-1" : min_str);
            float mx = std::stof(max_str.empty() ? "1" : max_str);

            ImGui::Text("Value:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##slider_val", &val, mn, mx)) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.4f", val);
                val_str = buf;
            }

            ImGui::Text("Range:");
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_min", &mn, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mn);
                min_str = buf;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_max", &mx, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mx);
                max_str = buf;
            }
            break;
        }

        case NodeType::SineWave: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Sine Wave Generator");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "A*sin(2*pi*f*t + phase) + offset");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key, float step = 0.1f) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##sine_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, step, step * 10, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Amplitude", "amplitude");
            floatParam("Frequency", "frequency");
            floatParam("Phase", "phase");
            floatParam("Offset", "offset");
            break;
        }

        case NodeType::StepSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Step Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##step_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Step Time", "step_time");
            floatParam("Initial Value", "initial_value");
            floatParam("Final Value", "final_value");
            break;
        }

        case NodeType::RampSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Ramp Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##ramp_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Start Value", "start_value");
            floatParam("End Value", "end_value");
            floatParam("Duration", "duration");
            break;
        }

        case NodeType::SignalScope: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Scope");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Plots incoming signal values in real-time");
            ImGui::Spacing();

            std::string& ws = node.parameters["window_size"];
            int win = std::stoi(ws.empty() ? "500" : ws);
            ImGui::Text("Window Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputInt("##scope_win", &win)) {
                if (win < 10) win = 10;
                ws = std::to_string(win);
            }

            std::string& as = node.parameters["auto_scale"];
            bool auto_s = (as == "true");
            if (ImGui::Checkbox("Auto Scale", &auto_s)) {
                as = auto_s ? "true" : "false";
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Real-time signal plot
            auto& buf = context.scope_buffers[node.id];
            buf.max_samples = win;

            // Generate demo data when no simulation is running
            // (will be replaced by real simulation data when connected)
            context.scope_demo_time += ImGui::GetIO().DeltaTime;
            buf.Push(context.scope_demo_time, std::sin(2.0f * 3.14159f * 0.5f * context.scope_demo_time));

            if (!buf.times.empty()) {
                // Copy deque to contiguous arrays for ImPlot
                std::vector<float> t_arr(buf.times.begin(), buf.times.end());
                std::vector<float> v_arr(buf.values.begin(), buf.values.end());

                ImPlotFlags plot_flags = ImPlotFlags_NoTitle;
                if (ImPlot::BeginPlot("##scope_plot", ImVec2(-1, 200), plot_flags)) {
                    ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
                    ImPlotAxisFlags y_flags = auto_s ? (ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel) : ImPlotAxisFlags_NoLabel;
                    ImPlot::SetupAxes("Time (s)", "Value", x_flags, y_flags);

                    // Auto-scroll X axis to follow latest data
                    if (!t_arr.empty()) {
                        float t_max = t_arr.back();
                        float t_window = win * 0.016f;  // Approximate window in seconds
                        if (t_window < 2.0f) t_window = 2.0f;
                        ImPlot::SetupAxisLimits(ImAxis_X1, t_max - t_window, t_max, ImGuiCond_Always);
                    }

                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 0.9f, 0.8f, 1.0f));
                    ImPlot::PlotLine("Signal", t_arr.data(), v_arr.data(), static_cast<int>(t_arr.size()));
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
            }

            // Controls
            if (ImGui::Button("Clear")) {
                buf.Clear();
                context.scope_demo_time = 0.0f;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Samples: %d", static_cast<int>(buf.times.size()));

            break;
        }

        case NodeType::PluginCustom:
            RenderPluginCustomNodeProperties(node, context);
            break;
        // ========== Smart I/O Nodes (Dialog-only configuration) ==========
        case NodeType::DataInput:
        case NodeType::DataOutput:
            // These nodes are configured via the Open Dialog button only
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Use 'Open Dialog' to configure");
            break;

        default: {
            // Generic parameter editor for nodes that don't have a
            // custom case above (e.g. the new Image Transform nodes).
            // Renders each parameter as an editable text field. Nodes
            // with no parameters at all show a "no parameters" hint.
            if (node.parameters.empty()) {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                   "No editable parameters for this node type");
            } else {
                for (auto& [key, value] : node.parameters) {
                    char buf[256] = {};
                    strncpy(buf, value.c_str(), sizeof(buf) - 1);

                    ImGui::Text("%s:", key.c_str());
                    ImGui::SameLine(140);
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10);
                    std::string label = "##param_" + key;
                    if (ImGui::InputText(label.c_str(), buf, sizeof(buf))) {
                        value = buf;
                    }
                }
            }
            break;
        }
    }
}
} // namespace gui::properties_node_editors
