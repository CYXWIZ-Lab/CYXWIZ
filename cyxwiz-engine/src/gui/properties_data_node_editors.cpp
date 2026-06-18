// Properties panel editors for data pipeline node types.

#include "properties_node_editors.h"
#include "node_editor.h"
#include "../core/data_registry.h"
#include "../core/worker_defaults.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace gui::properties_node_editors {

void RenderDataPipelineNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    switch (node.type) {
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
                context.invalidate_shapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Name in DataRegistry");

            ImGui::Spacing();

            // Show loaded dataset info if available
            auto& registry = cyxwiz::DataRegistry::Instance();
            if (registry.HasDataset(dataset_name)) {
                auto handle = registry.GetDataset(dataset_name);
                if (handle.IsValid()) {
                    auto info = handle.GetInfo();
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Dataset loaded!");
                    ImGui::Text("Samples: %zu", info.num_samples);
                    ImGui::Text("Classes: %zu", info.num_classes);
                    ImGui::Text("Shape: %s", info.GetShapeString().c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Dataset not loaded");
            }

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
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                               "Owns all training-loop hyperparameters");
            ImGui::Separator();
            ImGui::Spacing();

            // ---- Training loop ----
            ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Training Loop");

            // Epochs
            std::string& epochs = node.parameters["epochs"];
            if (epochs.empty()) epochs = "10";
            char epochs_buffer[16];
            strncpy(epochs_buffer, epochs.c_str(), sizeof(epochs_buffer) - 1);
            epochs_buffer[sizeof(epochs_buffer) - 1] = '\0';
            ImGui::Text("Epochs:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##epochs", epochs_buffer, sizeof(epochs_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                epochs = epochs_buffer;
            }

            // Batch size
            std::string& batch_size = node.parameters["batch_size"];
            if (batch_size.empty()) batch_size = "32";
            char batch_buffer[16];
            strncpy(batch_buffer, batch_size.c_str(), sizeof(batch_buffer) - 1);
            batch_buffer[sizeof(batch_buffer) - 1] = '\0';
            ImGui::Text("Batch Size:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##batch_size", batch_buffer, sizeof(batch_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                batch_size = batch_buffer;
            }

            // Gradient accumulation steps (effective batch = batch_size * this)
            std::string& grad_accum = node.parameters["grad_accum_steps"];
            if (grad_accum.empty()) grad_accum = "1";
            char grad_accum_buffer[16];
            strncpy(grad_accum_buffer, grad_accum.c_str(), sizeof(grad_accum_buffer) - 1);
            grad_accum_buffer[sizeof(grad_accum_buffer) - 1] = '\0';
            ImGui::Text("Grad Accum:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##grad_accum", grad_accum_buffer, sizeof(grad_accum_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                grad_accum = grad_accum_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Future field: current TrainingExecutor does not yet accumulate gradients.\n"
                                  "Effective batch size is still the configured batch_size.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "  Partial: not consumed by the current training loop.");

            ImGui::Spacing();

            // ---- Iteration order ----
            ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Iteration Order");

            std::string& shuffle = node.parameters["shuffle"];
            if (shuffle.empty()) shuffle = "true";
            bool shuffle_val = (shuffle == "true");
            if (ImGui::Checkbox("Shuffle", &shuffle_val)) {
                shuffle = shuffle_val ? "true" : "false";
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reshuffle samples at the start of every epoch.");
            }

            std::string& drop_last = node.parameters["drop_last"];
            if (drop_last.empty()) drop_last = "false";
            bool drop_last_val = (drop_last == "true");
            if (ImGui::Checkbox("Drop Last Batch", &drop_last_val)) {
                drop_last = drop_last_val ? "true" : "false";
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Skip the final batch if it has fewer than batch_size samples.");
            }

            // Seed for reproducibility
            std::string& seed = node.parameters["seed"];
            if (seed.empty()) seed = "42";
            char seed_buffer[16];
            strncpy(seed_buffer, seed.c_str(), sizeof(seed_buffer) - 1);
            seed_buffer[sizeof(seed_buffer) - 1] = '\0';
            ImGui::Text("Seed:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##seed", seed_buffer, sizeof(seed_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                seed = seed_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Partial field: DataSplit.seed controls split reproducibility.\n"
                                  "Current DataLoader training shuffle does not consume this seed directly.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "  Partial: split seed is runtime-owned; loader shuffle seed is not yet wired.");

            ImGui::Spacing();

            // ---- Checkpointing ----
            ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Checkpointing");

            std::string& save_best_checkpoint = node.parameters["save_best_checkpoint"];
            if (save_best_checkpoint.empty()) save_best_checkpoint = "true";
            bool save_best_checkpoint_val = (save_best_checkpoint == "true");
            if (ImGui::Checkbox("Save Best Checkpoint", &save_best_checkpoint_val)) {
                save_best_checkpoint = save_best_checkpoint_val ? "true" : "false";
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Keep the best validation epoch instead of only the last epoch.");
            }

            std::string& early_stop_patience = node.parameters["early_stopping_patience"];
            if (early_stop_patience.empty()) early_stop_patience = "5";
            char patience_buffer[16];
            strncpy(patience_buffer, early_stop_patience.c_str(), sizeof(patience_buffer) - 1);
            patience_buffer[sizeof(patience_buffer) - 1] = '\0';
            ImGui::Text("Early Stop Patience:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##early_stopping_patience", patience_buffer, sizeof(patience_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                early_stop_patience = patience_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Stop after this many epochs with no validation improvement.\n"
                                  "Set 0 to disable early stopping.");
            }

            std::string& checkpoint_dir = node.parameters["checkpoint_dir"];
            if (checkpoint_dir.empty()) checkpoint_dir = "";
            char checkpoint_dir_buffer[260];
            strncpy(checkpoint_dir_buffer, checkpoint_dir.c_str(), sizeof(checkpoint_dir_buffer) - 1);
            checkpoint_dir_buffer[sizeof(checkpoint_dir_buffer) - 1] = '\0';
            ImGui::Text("Checkpoint Dir:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(220.0f);
            if (ImGui::InputText("##checkpoint_dir", checkpoint_dir_buffer, sizeof(checkpoint_dir_buffer))) {
                checkpoint_dir = checkpoint_dir_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Optional checkpoint root. Empty uses the default run-local folder.");
            }

            ImGui::Spacing();

            // ---- Performance ----
            ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Performance");

            std::string& num_workers = node.parameters["num_workers"];
            if (num_workers.empty()) num_workers = std::to_string(cyxwiz::GetDefaultNumWorkers());
            char workers_buffer[16];
            strncpy(workers_buffer, num_workers.c_str(), sizeof(workers_buffer) - 1);
            workers_buffer[sizeof(workers_buffer) - 1] = '\0';
            ImGui::Text("Num Workers:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_workers", workers_buffer, sizeof(workers_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                num_workers = workers_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Synchronous per-batch workers where supported. 0 = single-threaded.\n"
                                  "Empty uses a hardware-based default. Async prefetch is controlled by Prefetch.");
            }

            std::string& prefetch = node.parameters["prefetch_factor"];
            if (prefetch.empty()) prefetch = "2";
            char prefetch_buffer[16];
            strncpy(prefetch_buffer, prefetch.c_str(), sizeof(prefetch_buffer) - 1);
            prefetch_buffer[sizeof(prefetch_buffer) - 1] = '\0';
            ImGui::Text("Prefetch:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##prefetch_factor", prefetch_buffer, sizeof(prefetch_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                prefetch = prefetch_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Bounded async queue depth for supported Arrow and Parquet batchers.\n"
                                  "0 disables prefetch. Positive values overlap batch construction with model compute.");
            }

            std::string& pin_memory = node.parameters["pin_memory"];
            if (pin_memory.empty()) pin_memory = "false";
            bool pin_memory_val = (pin_memory == "true");
            if (ImGui::Checkbox("Pin Memory (CUDA)", &pin_memory_val)) {
                pin_memory = pin_memory_val ? "true" : "false";
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reserved for future pinned host-memory transfers. Current batchers ignore this field.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "  Future: current batchers ignore this field.");

            ImGui::Spacing();

            // ---- Logging ----
            ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Logging");

            std::string& log_interval = node.parameters["log_interval"];
            if (log_interval.empty()) log_interval = "10";
            char log_buffer[16];
            strncpy(log_buffer, log_interval.c_str(), sizeof(log_buffer) - 1);
            log_buffer[sizeof(log_buffer) - 1] = '\0';
            ImGui::Text("Log every N:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##log_interval", log_buffer, sizeof(log_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                log_interval = log_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Future field: current TrainingExecutor logging cadence is hardcoded in the loop.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "  Partial: logging cadence is not runtime-configurable yet.");

            std::string& val_freq = node.parameters["validation_freq"];
            if (val_freq.empty()) val_freq = "1";
            char val_freq_buffer[16];
            strncpy(val_freq_buffer, val_freq.c_str(), sizeof(val_freq_buffer) - 1);
            val_freq_buffer[sizeof(val_freq_buffer) - 1] = '\0';
            ImGui::Text("Validate every:");
            ImGui::SameLine(140.0f);
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##validation_freq", val_freq_buffer, sizeof(val_freq_buffer),
                                 ImGuiInputTextFlags_CharsDecimal)) {
                val_freq = val_freq_buffer;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Future field: current TrainingExecutor validates every epoch when validation is available.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "  Partial: current runtime validates every epoch.");
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
        default:
            break;
    }
}

} // namespace gui::properties_node_editors
