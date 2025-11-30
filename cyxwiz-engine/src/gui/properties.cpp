#include "properties.h"
#include "node_editor.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cmath>

namespace gui {

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::SetSelectedNode(MLNode* node) {
    selected_node_ = node;
}

void Properties::ClearSelection() {
    selected_node_ = nullptr;
}

void Properties::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Properties", &show_window_)) {
        ImGui::Text("Node Properties");
        ImGui::Separator();

        if (!selected_node_) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No node selected");
            ImGui::Text("Click on a node in the Node Editor to view its properties");
        } else {
            // Display selected node info
            ImGui::Text("Node: %s", selected_node_->name.c_str());
            ImGui::Text("ID: %d", selected_node_->id);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Render node-specific properties
            RenderNodeProperties(*selected_node_);
        }
    }
    ImGui::End();
}

void Properties::RenderNodeProperties(MLNode& node) {
    // Render editable parameters based on node type
    switch (node.type) {
        case NodeType::Input: {
            ImGui::Text("Input Shape:");
            std::string& shape = node.parameters["shape"];
            if (shape.empty()) shape = "28,28,1";

            char buffer[64];
            strncpy(buffer, shape.c_str(), sizeof(buffer) - 1);
            buffer[sizeof(buffer) - 1] = '\0';

            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##shape", buffer, sizeof(buffer))) {
                shape = buffer;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Format: height,width,channels");
            break;
        }

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
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Number of output classes");
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Dataset Input Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Dataset name
            std::string& dataset_name = node.parameters["dataset_name"];
            char name_buffer[128];
            strncpy(name_buffer, dataset_name.c_str(), sizeof(name_buffer) - 1);
            name_buffer[sizeof(name_buffer) - 1] = '\0';

            ImGui::Text("Dataset Name:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##dataset_name", name_buffer, sizeof(name_buffer))) {
                dataset_name = name_buffer;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Name in DataRegistry");

            ImGui::Spacing();

            // Split selection
            std::string& split = node.parameters["split"];
            if (split.empty()) split = "train";

            const char* splits[] = { "train", "val", "test" };
            int current_split = 0;
            for (int i = 0; i < 3; i++) {
                if (split == splits[i]) {
                    current_split = i;
                    break;
                }
            }

            ImGui::Text("Split:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##split", &current_split, splits, 3)) {
                split = splits[current_split];
            }
            break;
        }

        case NodeType::DataLoader: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Data Loader Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Batch size
            std::string& batch_size = node.parameters["batch_size"];
            if (batch_size.empty()) batch_size = "32";
            char batch_buffer[16];
            strncpy(batch_buffer, batch_size.c_str(), sizeof(batch_buffer) - 1);
            batch_buffer[sizeof(batch_buffer) - 1] = '\0';

            ImGui::Text("Batch Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##batch_size", batch_buffer, sizeof(batch_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                batch_size = batch_buffer;
            }

            ImGui::Spacing();

            // Shuffle
            std::string& shuffle = node.parameters["shuffle"];
            if (shuffle.empty()) shuffle = "true";
            bool shuffle_val = (shuffle == "true");
            if (ImGui::Checkbox("Shuffle", &shuffle_val)) {
                shuffle = shuffle_val ? "true" : "false";
            }

            // Drop last
            std::string& drop_last = node.parameters["drop_last"];
            if (drop_last.empty()) drop_last = "false";
            bool drop_last_val = (drop_last == "true");
            if (ImGui::Checkbox("Drop Last Batch", &drop_last_val)) {
                drop_last = drop_last_val ? "true" : "false";
            }

            ImGui::Spacing();

            // Num workers
            std::string& num_workers = node.parameters["num_workers"];
            if (num_workers.empty()) num_workers = "4";
            char workers_buffer[16];
            strncpy(workers_buffer, num_workers.c_str(), sizeof(workers_buffer) - 1);
            workers_buffer[sizeof(workers_buffer) - 1] = '\0';

            ImGui::Text("Num Workers:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_workers", workers_buffer, sizeof(workers_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                num_workers = workers_buffer;
            }
            break;
        }

        case NodeType::Augmentation: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Augmentation Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Transforms
            std::string& transforms = node.parameters["transforms"];
            if (transforms.empty()) transforms = "RandomFlip,Normalize";
            char transform_buffer[256];
            strncpy(transform_buffer, transforms.c_str(), sizeof(transform_buffer) - 1);
            transform_buffer[sizeof(transform_buffer) - 1] = '\0';

            ImGui::Text("Transforms:");
            ImGui::SetNextItemWidth(250.0f);
            if (ImGui::InputText("##transforms", transform_buffer, sizeof(transform_buffer))) {
                transforms = transform_buffer;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Comma-separated list");

            ImGui::Spacing();

            // Flip probability
            std::string& flip_prob_str = node.parameters["flip_prob"];
            if (flip_prob_str.empty()) flip_prob_str = "0.5";
            float flip_prob = std::stof(flip_prob_str);

            ImGui::Text("Flip Probability:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##flip_prob", &flip_prob, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", flip_prob);
                flip_prob_str = buf;
            }

            ImGui::Spacing();

            // Normalize mean
            std::string& mean = node.parameters["normalize_mean"];
            if (mean.empty()) mean = "0.0";
            char mean_buffer[32];
            strncpy(mean_buffer, mean.c_str(), sizeof(mean_buffer) - 1);
            mean_buffer[sizeof(mean_buffer) - 1] = '\0';

            ImGui::Text("Normalize Mean:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##norm_mean", mean_buffer, sizeof(mean_buffer))) {
                mean = mean_buffer;
            }

            // Normalize std
            std::string& std_val = node.parameters["normalize_std"];
            if (std_val.empty()) std_val = "1.0";
            char std_buffer[32];
            strncpy(std_buffer, std_val.c_str(), sizeof(std_buffer) - 1);
            std_buffer[sizeof(std_buffer) - 1] = '\0';

            ImGui::Text("Normalize Std:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##norm_std", std_buffer, sizeof(std_buffer))) {
                std_val = std_buffer;
            }
            break;
        }

        case NodeType::DataSplit: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Data Split Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Train ratio
            std::string& train_ratio_str = node.parameters["train_ratio"];
            if (train_ratio_str.empty()) train_ratio_str = "0.8";
            float train_ratio = std::stof(train_ratio_str);

            ImGui::Text("Train Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##train_ratio", &train_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", train_ratio);
                train_ratio_str = buf;
            }

            // Validation ratio
            std::string& val_ratio_str = node.parameters["val_ratio"];
            if (val_ratio_str.empty()) val_ratio_str = "0.1";
            float val_ratio = std::stof(val_ratio_str);

            ImGui::Text("Validation Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##val_ratio", &val_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", val_ratio);
                val_ratio_str = buf;
            }

            // Test ratio
            std::string& test_ratio_str = node.parameters["test_ratio"];
            if (test_ratio_str.empty()) test_ratio_str = "0.1";
            float test_ratio = std::stof(test_ratio_str);

            ImGui::Text("Test Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##test_ratio", &test_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", test_ratio);
                test_ratio_str = buf;
            }

            // Show total
            float total = train_ratio + val_ratio + test_ratio;
            ImVec4 total_color = (std::abs(total - 1.0f) < 0.01f) ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            ImGui::TextColored(total_color, "Total: %.2f (should be 1.0)", total);

            ImGui::Spacing();

            // Stratified
            std::string& stratified = node.parameters["stratified"];
            if (stratified.empty()) stratified = "true";
            bool stratified_val = (stratified == "true");
            if (ImGui::Checkbox("Stratified Split", &stratified_val)) {
                stratified = stratified_val ? "true" : "false";
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Maintain class distribution");

            ImGui::Spacing();

            // Seed
            std::string& seed = node.parameters["seed"];
            if (seed.empty()) seed = "42";
            char seed_buffer[16];
            strncpy(seed_buffer, seed.c_str(), sizeof(seed_buffer) - 1);
            seed_buffer[sizeof(seed_buffer) - 1] = '\0';

            ImGui::Text("Random Seed:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##seed", seed_buffer, sizeof(seed_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                seed = seed_buffer;
            }
            break;
        }

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
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "MNIST=10, CIFAR-10=10, ImageNet=1000");
            break;
        }

        default:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No editable parameters for this node type");
            break;
    }
}

} // namespace gui
