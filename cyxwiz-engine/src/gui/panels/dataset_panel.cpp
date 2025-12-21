#include "dataset_panel.h"
#include "training_plot_panel.h"
#include "wallet_panel.h"
#include "../node_editor.h"
#include "../../network/job_manager.h"
#include "../../core/texture_manager.h"
#include "../../core/training_manager.h"
#include "../../core/graph_compiler.h"
#include "job.pb.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>
#include <nlohmann/json.hpp>
#include <filesystem>

// cyxwiz-backend includes for local training
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shobjidl.h>  // For IFileDialog (folder browser)
#include <shellapi.h>  // For ShellExecuteA (open URLs)
#endif

namespace gui {

DatasetPanel::DatasetPanel() : cyxwiz::Panel("Dataset Manager", false) {
    // Hidden by default - shown when user clicks Dataset > Import Dataset
}

DatasetPanel::~DatasetPanel() {
    // Training is now managed by TrainingManager singleton - no cleanup needed here
}

const cyxwiz::DatasetInfo& DatasetPanel::GetDatasetInfo() const {
    return cached_info_;
}

const std::vector<size_t>& DatasetPanel::GetTrainIndices() const {
    static std::vector<size_t> empty;
    if (!current_dataset_.IsValid()) return empty;
    return current_dataset_.GetTrainIndices();
}

bool DatasetPanel::IsLocalTrainingRunning() const {
    return cyxwiz::TrainingManager::Instance().IsTrainingActive();
}

void DatasetPanel::StopLocalTraining() {
    cyxwiz::TrainingManager::Instance().StopTraining();
}

void DatasetPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 750), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(name_.c_str(), &visible_)) {

        // Header
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Dataset Management");
        ImGui::Spacing();

        // Show loading task indicator when async loading is in progress
        if (is_loading_.load()) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.15f, 0.2f, 1.0f));
            ImGui::BeginChild("LoadingTask", ImVec2(0, 60), true);

            // Animated loading spinner using time
            float time = static_cast<float>(ImGui::GetTime());
            const char* spinner_chars[] = {"|", "/", "-", "\\"};
            int spinner_idx = static_cast<int>(time * 8) % 4;

            ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%s LOADING DATASET...", spinner_chars[spinner_idx]);

            // Status message
            if (!loading_status_message_.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(%s)", loading_status_message_.c_str());
            }

            // Progress bar
            float progress = loading_progress_.load();
            ImGui::ProgressBar(progress, ImVec2(-80, 0), progress > 0 ? "" : "Preparing...");

            // Cancel button
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(70, 0))) {
                CancelLoading();
            }

            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::Spacing();
        }

        // Show notification (fades out after 2 seconds)
        if (show_notification_) {
            float elapsed = static_cast<float>(ImGui::GetTime()) - notification_time_;
            const float NOTIFICATION_DURATION = 2.0f;

            if (elapsed < NOTIFICATION_DURATION) {
                // Fade out in the last 0.5 seconds
                float alpha = (elapsed > NOTIFICATION_DURATION - 0.5f) ?
                    (NOTIFICATION_DURATION - elapsed) / 0.5f : 1.0f;

                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.3f, 0.15f, alpha * 0.9f));
                ImGui::BeginChild("Notification", ImVec2(0, 30), true);
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, alpha), "%s", notification_message_.c_str());
                ImGui::EndChild();
                ImGui::PopStyleColor();
                ImGui::Spacing();
            } else {
                show_notification_ = false;
            }
        }

        // Tab bar for different views
        if (ImGui::BeginTabBar("DatasetTabs")) {
            if (ImGui::BeginTabItem("Load Dataset")) {
                ImGui::BeginChild("LoadPanel", ImVec2(0, 0), false);

                // Single column clean layout
                RenderDatasetSelection();

                // Show loaded dataset info below in a card-style section
                if (IsDatasetLoaded()) {
                    ImGui::Spacing();
                    ImGui::Spacing();

                    // Dataset Info Card
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                    ImGui::BeginChild("DatasetInfoCard", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysAutoResize);

                    RenderDatasetInfo();
                    ImGui::Spacing();
                    ImGui::Spacing();
                    RenderSplitConfiguration();
                    ImGui::Spacing();
                    ImGui::Spacing();
                    RenderStatistics();

                    ImGui::EndChild();
                    ImGui::PopStyleColor();
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Loaded Datasets")) {
                RenderLoadedDatasets();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Preview")) {
                if (IsDatasetLoaded()) {
                    RenderDataPreview();
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a dataset to preview samples");
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Augmentation")) {
                RenderAugmentationTab();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Training")) {
                if (IsDatasetLoaded()) {
                    RenderTrainingSection();
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a dataset to train a model");
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void DatasetPanel::RenderDatasetSelection() {
    // Dataset type selection card
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
    ImGui::BeginChild("SelectionCard", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "SELECT DATASET TYPE");
    ImGui::Spacing();

    // Dataset type selection (merged Images + CSV into single "Images" option)
    const char* types[] = {"CSV", "Images", "MNIST", "CIFAR-10", "HuggingFace", "Kaggle", "Custom"};
    int current_type = 0;
    if (selected_type_ == cyxwiz::DatasetType::CSV) current_type = 0;
    else if (selected_type_ == cyxwiz::DatasetType::ImageFolder || selected_type_ == cyxwiz::DatasetType::ImageCSV) current_type = 1;
    else if (selected_type_ == cyxwiz::DatasetType::MNIST) current_type = 2;
    else if (selected_type_ == cyxwiz::DatasetType::CIFAR10) current_type = 3;
    else if (selected_type_ == cyxwiz::DatasetType::HuggingFace) current_type = 4;
    else if (selected_type_ == cyxwiz::DatasetType::Kaggle) current_type = 5;
    else if (selected_type_ == cyxwiz::DatasetType::Custom) current_type = 6;

    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##Type", &current_type, types, IM_ARRAYSIZE(types))) {
        switch (current_type) {
            case 0: selected_type_ = cyxwiz::DatasetType::CSV; break;
            case 1: selected_type_ = cyxwiz::DatasetType::ImageCSV; break;  // Unified Images type
            case 2: selected_type_ = cyxwiz::DatasetType::MNIST; break;
            case 3: selected_type_ = cyxwiz::DatasetType::CIFAR10; break;
            case 4: selected_type_ = cyxwiz::DatasetType::HuggingFace; break;
            case 5: selected_type_ = cyxwiz::DatasetType::Kaggle; break;
            case 6: selected_type_ = cyxwiz::DatasetType::Custom; break;
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Show different input based on type
    if (selected_type_ == cyxwiz::DatasetType::HuggingFace) {
        // HuggingFace dataset name input
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "DATASET NAME");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##HFName", "e.g., mnist, cifar10, imdb", hf_dataset_name_, sizeof(hf_dataset_name_));

        ImGui::Spacing();
        ImGui::Spacing();

        // Load button for HuggingFace
        if (ImGui::Button("Load from HuggingFace", ImVec2(-1, 30))) {
            std::string name = hf_dataset_name_;
            if (!name.empty()) {
                LoadHuggingFaceDatasetAsync(name);
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::Kaggle) {
        // Kaggle dataset slug input
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "DATASET SLUG");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##KaggleSlug", "e.g., titanic, uciml/iris", kaggle_dataset_slug_, sizeof(kaggle_dataset_slug_));

        ImGui::Spacing();
        ImGui::Spacing();

        // Load button for Kaggle
        if (ImGui::Button("Load from Kaggle", ImVec2(-1, 30))) {
            std::string slug = kaggle_dataset_slug_;
            if (!slug.empty()) {
                LoadKaggleDatasetAsync(slug);
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::Custom) {
        // Custom dataset configuration
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "DATA PATH");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        ImGui::InputText("##CustomPath", file_path_buffer_, sizeof(file_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##Custom", ImVec2(-1, 0))) {
            ShowFileBrowser();
        }

        ImGui::Spacing();

        // Format selection
        static int format_idx = 0;
        const char* formats[] = {"Auto-detect", "JSON", "CSV", "TSV", "Binary", "Folder"};
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "FORMAT");
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##CustomFormat", &format_idx, formats, IM_ARRAYSIZE(formats));

        ImGui::Spacing();

        // Schema configuration (collapsible)
        if (ImGui::CollapsingHeader("Schema Configuration")) {
            static char data_key[64] = "data";
            static char labels_key[64] = "labels";
            static int label_column = -1;
            static bool has_header = false;
            static bool normalize = true;
            static float scale = 1.0f;

            ImGui::SetNextItemWidth(-1);
            ImGui::InputTextWithHint("##DataKey", "Data Key (JSON)", data_key, sizeof(data_key));
            ImGui::SetNextItemWidth(-1);
            ImGui::InputTextWithHint("##LabelsKey", "Labels Key (JSON)", labels_key, sizeof(labels_key));
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("Label Column", &label_column);
            ImGui::Checkbox("Has Header", &has_header);
            ImGui::SameLine();
            ImGui::Checkbox("Normalize", &normalize);
            ImGui::SetNextItemWidth(120);
            ImGui::InputFloat("Scale", &scale, 0.001f, 0.01f, "%.4f");
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // Load button for Custom
        if (ImGui::Button("Load Custom Dataset", ImVec2(-1, 30))) {
            std::string path = file_path_buffer_;
            if (!path.empty()) {
                // Build config
                cyxwiz::CustomConfig config;
                config.data_path = path;

                // Set format based on selection
                switch (format_idx) {
                    case 0: config.format = ""; break;  // Auto-detect
                    case 1: config.format = "json"; break;
                    case 2: config.format = "csv"; break;
                    case 3: config.format = "tsv"; break;
                    case 4: config.format = "binary"; break;
                    case 5: config.format = "folder"; break;
                }

                LoadCustomDatasetAsync(config);
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::ImageCSV || selected_type_ == cyxwiz::DatasetType::ImageFolder) {
        // Unified image dataset UI
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "IMAGE FOLDER");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        ImGui::InputText("##ImageFolder", file_path_buffer_, sizeof(file_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##ImgFolder", ImVec2(-1, 0))) {
            ShowFolderBrowser(file_path_buffer_, sizeof(file_path_buffer_));
        }

        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "LABELS CSV (Optional)");
        float btn_width = 65;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - btn_width * 2 - 8);
        ImGui::InputText("##CSVPath", csv_path_buffer_, sizeof(csv_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##CSV", ImVec2(btn_width, 0))) {
            ShowCSVFileBrowser();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear##ClearCSV", ImVec2(-1, 0))) {
            csv_path_buffer_[0] = '\0';
        }

        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "No CSV: subfolders = classes | CSV: filename,label columns");

        ImGui::Spacing();

        // Image size and memory options
        if (ImGui::CollapsingHeader("Image Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("Width", &image_target_width_);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("Height", &image_target_height_);
            image_target_width_ = std::max(1, std::min(2048, image_target_width_));
            image_target_height_ = std::max(1, std::min(2048, image_target_height_));

            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "CACHE");
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("##CacheSize", &image_cache_size_);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "images");
            image_cache_size_ = std::max(1, std::min(10000, image_cache_size_));

            // Calculate estimated memory usage
            int channels = 3;  // Assume RGB
            size_t bytes_per_image = static_cast<size_t>(image_target_width_) *
                                     static_cast<size_t>(image_target_height_) *
                                     channels * sizeof(float);
            size_t estimated_cache_mb = (bytes_per_image * image_cache_size_) / (1024 * 1024);

            ImGui::TextColored(ImVec4(0.4f, 0.7f, 0.4f, 1.0f),
                "Est. memory: ~%zu MB (lazy loading)", estimated_cache_mb);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // Load button
        if (ImGui::Button("Load Image Dataset", ImVec2(-1, 30))) {
            std::string img_folder = file_path_buffer_;
            std::string csv_file = csv_path_buffer_;  // Can be empty
            if (!img_folder.empty()) {
                // Use async loading with loading indicator
                if (is_loading_.load()) {
                    spdlog::warn("Already loading a dataset, please wait...");
                } else {
                    is_loading_.store(true);
                    loading_progress_.store(0.0f);
                    loading_status_message_ = "Starting...";

                    // Capture values for lambda
                    int target_w = image_target_width_;
                    int target_h = image_target_height_;
                    int cache_size = image_cache_size_;

                    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
                        "Loading Images: " + std::filesystem::path(img_folder).filename().string(),
                        [this, img_folder, csv_file, target_w, target_h, cache_size](cyxwiz::LambdaTask& task) {
                            task.ReportProgress(0.1f, "Scanning directory...");

                            auto& registry = cyxwiz::DataRegistry::Instance();

                            task.ReportProgress(0.3f, "Loading images...");
                            if (task.ShouldStop()) return;

                            auto handle = registry.LoadImageCSV(img_folder, csv_file, "",
                                target_w, target_h, cache_size);

                            task.ReportProgress(0.9f, "Finalizing...");
                            if (task.ShouldStop()) return;

                            if (handle.IsValid()) {
                                current_dataset_ = handle;
                                cached_info_ = handle.GetInfo();
                                if (csv_file.empty()) {
                                    spdlog::info("Loaded image dataset from folder: {} samples, {} classes (cache: {} images)",
                                        cached_info_.num_samples, cached_info_.num_classes, cache_size);
                                } else {
                                    spdlog::info("Loaded image dataset with CSV: {} samples, {} classes (cache: {} images)",
                                        cached_info_.num_samples, cached_info_.num_classes, cache_size);
                                }
                                ApplySplit();
                                UpdateClassCounts();
                                task.MarkCompleted();
                            } else {
                                task.MarkFailed("Failed to load image dataset");
                            }
                        },
                        [this](float progress, const std::string& msg) {
                            loading_progress_.store(progress);
                            loading_status_message_ = msg;
                        },
                        [this](bool success, const std::string& error) {
                            is_loading_.store(false);
                            loading_status_message_.clear();
                            if (!success) {
                                spdlog::error("Async image load failed: {}", error);
                            }
                        }
                    );
                }
            }
        }
    } else {
        // File path input for other types (CSV, MNIST, CIFAR-10)
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "DATASET PATH");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        ImGui::InputText("##Path", file_path_buffer_, sizeof(file_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse", ImVec2(-1, 0))) {
            ShowFileBrowser();
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // Load button
        if (ImGui::Button("Load Dataset", ImVec2(-1, 30))) {
            std::string path = file_path_buffer_;
            if (!path.empty()) {
                LoadDatasetAsync(path);
            }
        }
    }

    // Clear button
    if (IsDatasetLoaded()) {
        ImGui::Spacing();
        if (ImGui::Button("Clear Dataset", ImVec2(-1, 0))) {
            ClearDataset();
        }
    }

    ImGui::Spacing();

    // Config Export/Import section (collapsible)
    if (ImGui::CollapsingHeader("Configuration")) {
        if (IsDatasetLoaded()) {
            if (ImGui::Button("Export Config", ImVec2(-1, 0))) {
                #ifdef _WIN32
                OPENFILENAMEA ofn;
                char filename[MAX_PATH] = "dataset_config.json";
                ZeroMemory(&ofn, sizeof(ofn));
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = NULL;
                ofn.lpstrFilter = "JSON Files\0*.json\0All Files\0*.*\0";
                ofn.lpstrFile = filename;
                ofn.nMaxFile = MAX_PATH;
                ofn.lpstrDefExt = "json";
                ofn.Flags = OFN_OVERWRITEPROMPT;
                if (GetSaveFileNameA(&ofn)) {
                    auto& registry = cyxwiz::DataRegistry::Instance();
                    if (registry.ExportConfig(cached_info_.name, filename, split_config_)) {
                        spdlog::info("Exported config to: {}", filename);
                    }
                }
                #endif
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Export dataset configuration to JSON file");
            }
        }

        if (ImGui::Button("Import Config", ImVec2(-1, 0))) {
            #ifdef _WIN32
            OPENFILENAMEA ofn;
            char filename[MAX_PATH] = "";
            ZeroMemory(&ofn, sizeof(ofn));
            ofn.lStructSize = sizeof(ofn);
            ofn.hwndOwner = NULL;
            ofn.lpstrFilter = "JSON Files\0*.json\0All Files\0*.*\0";
            ofn.lpstrFile = filename;
            ofn.nMaxFile = MAX_PATH;
            ofn.Flags = OFN_FILEMUSTEXIST;
            if (GetOpenFileNameA(&ofn)) {
                auto& registry = cyxwiz::DataRegistry::Instance();
                std::string loaded_name;
                cyxwiz::SplitConfig imported_split;
                if (registry.ImportConfig(filename, loaded_name, imported_split)) {
                    current_dataset_ = registry.GetDataset(loaded_name);
                    if (current_dataset_.IsValid()) {
                        cached_info_ = current_dataset_.GetInfo();
                        split_config_ = imported_split;
                        cached_info_.train_count = current_dataset_.GetTrainIndices().size();
                        cached_info_.val_count = current_dataset_.GetValIndices().size();
                        cached_info_.test_count = current_dataset_.GetTestIndices().size();
                        spdlog::info("Imported config: {} (split: {:.0f}/{:.0f}/{:.0f})",
                            loaded_name, split_config_.train_ratio * 100,
                            split_config_.val_ratio * 100, split_config_.test_ratio * 100);
                    }
                }
            }
            #endif
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Import dataset configuration from JSON file");
        }

        // Versioning (only when dataset loaded)
        if (IsDatasetLoaded()) {
            ImGui::Spacing();
            if (ImGui::Button("Save Version", ImVec2(-1, 0))) {
                auto& registry = cyxwiz::DataRegistry::Instance();
                registry.SaveVersion(cached_info_.name, "Manual save");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Save current dataset state as a version");
            }

            auto& registry = cyxwiz::DataRegistry::Instance();
            auto versions = registry.GetVersionHistory(cached_info_.name);
            if (!versions.empty()) {
                if (ImGui::TreeNode("Version History")) {
                    for (const auto& ver : versions) {
                        ImGui::BulletText("%s - %s", ver.version_id.c_str(), ver.timestamp.c_str());
                        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  %s (%zu samples)",
                            ver.description.c_str(), ver.num_samples);
                    }
                    ImGui::TreePop();
                }
            }
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Popular Datasets section
    if (ImGui::CollapsingHeader("Quick Load Popular Datasets")) {
        ImGui::Spacing();

        // Source and dataset on same row
        ImGui::SetNextItemWidth(100);
        const char* sources[] = {"HuggingFace", "Kaggle"};
        ImGui::Combo("##Source", &popular_dataset_source_, sources, IM_ARRAYSIZE(sources));
        ImGui::SameLine();

        ImGui::SetNextItemWidth(-1);
        if (popular_dataset_source_ == 0) {
            const char* hf_datasets[] = {"mnist", "cifar10", "imdb", "fashion_mnist", "ag_news"};
            ImGui::Combo("##Dataset", &popular_dataset_index_, hf_datasets, IM_ARRAYSIZE(hf_datasets));
        } else {
            const char* kaggle_datasets[] = {"titanic", "uciml/iris", "digits", "zalando-research/fashionmnist", "mnist"};
            ImGui::Combo("##Dataset", &popular_dataset_index_, kaggle_datasets, IM_ARRAYSIZE(kaggle_datasets));
        }

        if (ImGui::Button("Download & Load", ImVec2(-1, 0))) {
            if (popular_dataset_source_ == 0) {
                const char* hf_datasets[] = {"mnist", "cifar10", "imdb", "fashion_mnist", "ag_news"};
                LoadHuggingFaceDatasetAsync(hf_datasets[popular_dataset_index_]);
            } else {
                const char* kaggle_datasets[] = {"titanic", "uciml/iris", "digits", "zalando-research/fashionmnist", "mnist"};
                LoadKaggleDatasetAsync(kaggle_datasets[popular_dataset_index_]);
            }
        }

        ImGui::Spacing();

        // Search for more datasets
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
        ImGui::InputTextWithHint("##SearchDataset", "Search online...", dataset_search_buffer_, sizeof(dataset_search_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Go", ImVec2(-1, 0))) {
            std::string query = dataset_search_buffer_;
            if (!query.empty()) {
                std::string url;
                if (popular_dataset_source_ == 0) {
                    url = "https://huggingface.co/datasets?search=" + query;
                } else {
                    url = "https://www.kaggle.com/search?q=" + query;
                }
                #ifdef _WIN32
                ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
                #endif
            }
        }
    }

    ImGui::Spacing();

    // Memory usage section (collapsible)
    if (ImGui::CollapsingHeader("Memory Usage")) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        cyxwiz::MemoryStats stats = registry.GetMemoryStats();

        // Add texture memory
        auto& texture_mgr = cyxwiz::TextureManager::Instance();
        stats.texture_memory = texture_mgr.GetMemoryUsage();

        // Progress bar with overlay text and color coding
        float usage_percent = stats.GetUsagePercent();
        ImVec4 bar_color = usage_percent > 90 ? ImVec4(1.0f, 0.3f, 0.3f, 1.0f) :
                           usage_percent > 75 ? ImVec4(1.0f, 0.6f, 0.2f, 1.0f) :
                                                ImVec4(0.3f, 0.8f, 0.3f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
        char overlay[64];
        snprintf(overlay, sizeof(overlay), "%s / %s (%.0f%%)",
            stats.FormatBytes(stats.total_allocated).c_str(),
            stats.FormatBytes(stats.memory_limit).c_str(),
            usage_percent);
        ImGui::ProgressBar(usage_percent / 100.0f, ImVec2(-1, 0), overlay);
        ImGui::PopStyleColor();

        // Memory pressure warning
        if (registry.IsMemoryPressure()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Memory pressure!");
        }

        // Details
        ImGui::Text("Datasets: %zu | Peak: %s", stats.datasets_count, stats.FormatBytes(stats.peak_usage).c_str());
        ImGui::Text("Textures: %s | Cache: %.1f%%", stats.FormatBytes(stats.texture_memory).c_str(), stats.GetCacheHitRate());

        // Buttons
        bool can_trim = stats.datasets_count > 0;
        if (!can_trim) ImGui::BeginDisabled();
        if (ImGui::Button("Trim", ImVec2(60, 0))) {
            size_t before = registry.GetTotalMemoryUsage();
            registry.TrimMemory();
            size_t after = registry.GetTotalMemoryUsage();
            if (before > after) {
                spdlog::info("Trim: freed {} bytes", before - after);
            }
        }
        if (!can_trim) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(60, 0))) {
            registry.ResetCacheStats();
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void DatasetPanel::RenderLoadedDatasets() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    auto datasets = registry.ListDatasets();

    ImGui::Text("Loaded Datasets: %zu", datasets.size());
    ImGui::Separator();
    ImGui::Spacing();

    if (datasets.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No datasets loaded");
        return;
    }

    // Dataset list
    ImGui::BeginChild("DatasetList", ImVec2(250, 0), true);
    for (size_t i = 0; i < datasets.size(); ++i) {
        const auto& info = datasets[i];
        bool is_selected = (selected_dataset_index_ == static_cast<int>(i));
        bool is_active = current_dataset_.IsValid() && current_dataset_.GetName() == info.name;

        // Highlight active dataset with green background
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.1f, 0.4f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.15f, 0.5f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
        }

        std::string label = info.name + " (" + cyxwiz::DataRegistry::TypeToString(info.type) + ")";
        if (is_active) {
            label = "[Active] " + label;
        }

        if (ImGui::Selectable(label.c_str(), is_selected || is_active)) {
            selected_dataset_index_ = static_cast<int>(i);
        }

        if (is_active) {
            ImGui::PopStyleColor(3);
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Selected dataset details
    ImGui::BeginChild("DatasetDetails", ImVec2(0, 0), true);
    if (selected_dataset_index_ >= 0 && selected_dataset_index_ < static_cast<int>(datasets.size())) {
        const auto& info = datasets[selected_dataset_index_];

        ImGui::Text("Name: %s", info.name.c_str());
        ImGui::Text("Type: %s", cyxwiz::DataRegistry::TypeToString(info.type).c_str());
        ImGui::Text("Path: %s", info.path.c_str());
        ImGui::Separator();
        ImGui::Text("Samples: %zu", info.num_samples);
        ImGui::Text("Classes: %zu", info.num_classes);
        ImGui::Text("Shape: %s", info.GetShapeString().c_str());
        ImGui::Separator();
        ImGui::Text("Train: %zu", info.train_count);
        ImGui::Text("Val: %zu", info.val_count);
        ImGui::Text("Test: %zu", info.test_count);
        ImGui::Separator();
        ImGui::Text("Memory: %.2f MB", info.memory_usage / (1024.0 * 1024.0));

        ImGui::Spacing();

        if (ImGui::Button("Set as Active", ImVec2(-1, 0))) {
            current_dataset_ = registry.GetDataset(info.name);
            if (current_dataset_.IsValid()) {
                cached_info_ = current_dataset_.GetInfo();
                spdlog::info("Set active dataset: {}", info.name);
                // Show notification
                show_notification_ = true;
                notification_time_ = static_cast<float>(ImGui::GetTime());
                notification_message_ = "Active dataset: " + info.name;
            }
        }

        if (ImGui::Button("Unload", ImVec2(-1, 0))) {
            if (current_dataset_.IsValid() && current_dataset_.GetName() == info.name) {
                current_dataset_ = cyxwiz::DatasetHandle();
                cached_info_ = cyxwiz::DatasetInfo();
            }
            registry.UnloadDataset(info.name);
            selected_dataset_index_ = -1;
        }
    }
    ImGui::EndChild();
}

void DatasetPanel::RenderDatasetInfo() {
    ImGui::Text("Dataset Information");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Name: %s", cached_info_.name.c_str());
    ImGui::Text("Type: %s", cyxwiz::DataRegistry::TypeToString(cached_info_.type).c_str());
    ImGui::Text("Total Samples: %zu", cached_info_.num_samples);
    ImGui::Text("Number of Classes: %zu", cached_info_.num_classes);
    ImGui::Text("Shape: %s", cached_info_.GetShapeString().c_str());

    // Memory usage section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Memory Usage");

    // Refresh cached_info_ to get current memory stats
    if (current_dataset_.IsValid()) {
        cached_info_ = current_dataset_.GetInfo();
    }

    // Display memory in appropriate units
    auto formatMemory = [](size_t bytes) -> std::string {
        if (bytes >= 1024 * 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024 * 1024)) + "." +
                   std::to_string((bytes / (1024 * 1024 * 100)) % 10) + " GB";
        } else if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + "." +
                   std::to_string((bytes / (1024 * 100)) % 10) + " MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        }
        return std::to_string(bytes) + " B";
    };

    ImGui::Text("Current: %s", formatMemory(cached_info_.memory_usage).c_str());

    // For streaming/lazy-loaded datasets, show cache info
    if (cached_info_.is_streaming) {
        ImGui::Text("Cache: %s", formatMemory(cached_info_.cache_usage).c_str());

        // Cache hit ratio
        size_t total_accesses = cached_info_.cache_hits + cached_info_.cache_misses;
        if (total_accesses > 0) {
            float hit_ratio = static_cast<float>(cached_info_.cache_hits) / total_accesses * 100.0f;
            ImGui::Text("Cache Hit Ratio: %.1f%%", hit_ratio);
            ImGui::Text("  Hits: %zu, Misses: %zu", cached_info_.cache_hits, cached_info_.cache_misses);
        }

        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "(Lazy loading enabled)");
    } else {
        // For fully loaded datasets, calculate total memory
        size_t element_size = sizeof(float);
        size_t sample_size = 1;
        for (size_t dim : cached_info_.shape) sample_size *= dim;
        size_t total_data_bytes = cached_info_.num_samples * sample_size * element_size;
        size_t label_bytes = cached_info_.num_samples * sizeof(int);
        size_t total_bytes = total_data_bytes + label_bytes;

        ImGui::Text("Data: %s", formatMemory(total_data_bytes).c_str());
        ImGui::Text("Labels: %s", formatMemory(label_bytes).c_str());
        ImGui::Text("Total: %s", formatMemory(total_bytes).c_str());
    }
}

void DatasetPanel::RenderSplitConfiguration() {
    ImGui::Text("Train/Val/Test Split");
    ImGui::Separator();
    ImGui::Spacing();

    bool changed = false;
    changed |= ImGui::SliderFloat("Train", &split_config_.train_ratio, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Val", &split_config_.val_ratio, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Test", &split_config_.test_ratio, 0.0f, 1.0f, "%.2f");

    // Normalize to sum to 1.0
    float total = split_config_.train_ratio + split_config_.val_ratio + split_config_.test_ratio;
    if (total > 0.0f && std::abs(total - 1.0f) > 0.01f) {
        split_config_.train_ratio /= total;
        split_config_.val_ratio /= total;
        split_config_.test_ratio /= total;
    }

    ImGui::Spacing();
    ImGui::Text("Split sizes:");
    ImGui::Text("  Train: %zu samples", cached_info_.train_count);
    ImGui::Text("  Val:   %zu samples", cached_info_.val_count);
    ImGui::Text("  Test:  %zu samples", cached_info_.test_count);

    ImGui::Spacing();
    if (ImGui::Button("Apply Split", ImVec2(-1, 0))) {
        ApplySplit();
    }
}

void DatasetPanel::RenderStatistics() {
    ImGui::Text("Dataset Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    if (!class_counts_.empty()) {
        ImGui::Text("Class Distribution:");
        ImGui::Spacing();

        // Find max count for normalization
        int max_count = *std::max_element(class_counts_.begin(), class_counts_.end());

        for (size_t i = 0; i < class_counts_.size() && i < 10; ++i) {
            float ratio = max_count > 0 ? static_cast<float>(class_counts_[i]) / max_count : 0.0f;

            std::string class_label = i < class_names_.size() ? class_names_[i] : std::to_string(i);
            ImGui::Text("%s:", class_label.c_str());
            ImGui::SameLine(80);
            ImGui::ProgressBar(ratio, ImVec2(-1, 0), std::to_string(class_counts_[i]).c_str());
        }

        if (class_counts_.size() > 10) {
            ImGui::Text("... and %zu more classes", class_counts_.size() - 10);
        }
    }
}

void DatasetPanel::RenderDataPreview() {
    if (!current_dataset_.IsValid()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No dataset loaded");
        ImGui::TextWrapped("Load a dataset to preview samples");
        return;
    }

    size_t dataset_size = current_dataset_.Size();
    if (dataset_size == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Dataset is empty");
        return;
    }

    // Check if this is image data
    bool is_image_data = (cached_info_.type == cyxwiz::DatasetType::MNIST ||
                          cached_info_.type == cyxwiz::DatasetType::CIFAR10 ||
                          cached_info_.type == cyxwiz::DatasetType::ImageFolder ||
                          cached_info_.type == cyxwiz::DatasetType::ImageCSV);

    // View mode toggle
    ImGui::Text("View Mode:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Single", preview_view_mode_ == 0)) preview_view_mode_ = 0;
    ImGui::SameLine();
    if (ImGui::RadioButton("Grid", preview_view_mode_ == 1)) preview_view_mode_ = 1;
    if (!is_image_data) {
        ImGui::SameLine();
        if (ImGui::RadioButton("Table", preview_view_mode_ == 2)) preview_view_mode_ = 2;
    }

    ImGui::Spacing();

    if (preview_view_mode_ == 0) {
        // Single sample view
        RenderSingleSamplePreview(dataset_size, is_image_data);
    } else if (preview_view_mode_ == 1) {
        // Grid view
        RenderGridPreview(dataset_size, is_image_data);
    } else {
        // Table view (tabular data only)
        RenderTablePreview(dataset_size);
    }
}

void DatasetPanel::RenderImagePreview(const float* image_data, int width, int height, int channels) {
    ImGui::Text("Image: %dx%d, %d channels", width, height, channels);

    int total_pixels = width * height * channels;
    float min_val = *std::min_element(image_data, image_data + total_pixels);
    float max_val = *std::max_element(image_data, image_data + total_pixels);
    ImGui::Text("Value range: [%.3f, %.3f]", min_val, max_val);

    ImGui::Spacing();

    // Use TextureManager for GPU-accelerated image preview
    cyxwiz::RenderImageWithTexture(image_data, width, height, channels);
}

void DatasetPanel::RenderSingleSamplePreview(size_t dataset_size, bool is_image_data) {
    // Two-column layout: controls on left, preview on right
    ImGui::Columns(2, "SinglePreviewColumns", true);
    ImGui::SetColumnWidth(0, 200);

    // Left column: Navigation and info
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Navigation");
    ImGui::Spacing();

    // Sample navigation
    ImGui::Text("Sample %d / %zu", preview_sample_idx_ + 1, dataset_size);

    if (ImGui::Button("Prev", ImVec2(60, 0))) {
        preview_sample_idx_ = (preview_sample_idx_ - 1 + static_cast<int>(dataset_size)) % static_cast<int>(dataset_size);
    }
    ImGui::SameLine();
    if (ImGui::Button("Next", ImVec2(60, 0))) {
        preview_sample_idx_ = (preview_sample_idx_ + 1) % static_cast<int>(dataset_size);
    }

    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##SampleNav", &preview_sample_idx_, 0, static_cast<int>(dataset_size) - 1, "%d");

    // Random sample button
    if (ImGui::Button("Random", ImVec2(-1, 0))) {
        preview_sample_idx_ = rand() % static_cast<int>(dataset_size);
    }

    ImGui::Spacing();

    // Get sample info
    auto [sample, label] = current_dataset_.GetSample(preview_sample_idx_);

    // Label info
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Sample Info");
    ImGui::Spacing();

    ImGui::Text("Label: %d", label);
    if (label >= 0 && label < static_cast<int>(class_names_.size())) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Class: %s", class_names_[label].c_str());
    }

    ImGui::Text("Features: %zu", sample.size());

    // Statistics for this sample
    if (!sample.empty()) {
        float sum = 0, min_v = sample[0], max_v = sample[0];
        for (float v : sample) {
            sum += v;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        float mean = sum / sample.size();

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Statistics");
        ImGui::Text("Min: %.4f", min_v);
        ImGui::Text("Max: %.4f", max_v);
        ImGui::Text("Mean: %.4f", mean);
    }

    // Right column: Preview
    ImGui::NextColumn();

    if (is_image_data && !cached_info_.shape.empty() && cached_info_.shape.size() >= 2) {
        int width = static_cast<int>(cached_info_.shape[0]);
        int height = static_cast<int>(cached_info_.shape[1]);
        int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;

        if (sample.size() == static_cast<size_t>(width * height * channels)) {
            // Image zoom control
            ImGui::Text("Zoom:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("##Zoom", &preview_zoom_, 0.5f, 4.0f, "%.1fx");
            ImGui::SameLine();
            if (ImGui::Button("Reset##Zoom")) preview_zoom_ = 1.0f;

            ImGui::Spacing();

            // Render image with zoom
            float display_w = width * preview_zoom_;
            float display_h = height * preview_zoom_;

            // Scrollable region for zoomed image
            ImGui::BeginChild("ImagePreview", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
            cyxwiz::RenderImageWithTexture(sample.data(), width, height, channels, display_w, display_h);
            ImGui::EndChild();
        }
    } else {
        // Tabular data preview
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Feature Values");
        ImGui::Spacing();

        ImGui::BeginChild("FeatureValues", ImVec2(0, 0), true);

        // Show features in a nice format
        int num_cols = 4;
        if (ImGui::BeginTable("FeaturesTable", num_cols, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < sample.size(); i += 2) {
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("[%zu]", i);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", sample[i]);

                if (i + 1 < sample.size()) {
                    ImGui::TableNextColumn();
                    ImGui::Text("[%zu]", i + 1);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", sample[i + 1]);
                }
            }
            ImGui::EndTable();
        }
        ImGui::EndChild();
    }

    ImGui::Columns(1);
}

void DatasetPanel::RenderGridPreview(size_t dataset_size, bool is_image_data) {
    // Grid settings
    ImGui::Text("Grid Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputInt("##GridCols", &preview_grid_cols_, 0, 0);
    preview_grid_cols_ = std::clamp(preview_grid_cols_, 2, 8);
    ImGui::SameLine();
    ImGui::Text("x");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputInt("##GridRows", &preview_grid_rows_, 0, 0);
    preview_grid_rows_ = std::clamp(preview_grid_rows_, 2, 8);

    ImGui::SameLine();
    ImGui::Text("Start:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    int start_idx = preview_sample_idx_;
    if (ImGui::InputInt("##StartIdx", &start_idx, 0, 0)) {
        preview_sample_idx_ = std::clamp(start_idx, 0, static_cast<int>(dataset_size) - 1);
    }

    ImGui::SameLine();
    if (ImGui::Button("Random Page")) {
        int page_size = preview_grid_cols_ * preview_grid_rows_;
        int max_start = std::max(0, static_cast<int>(dataset_size) - page_size);
        preview_sample_idx_ = rand() % (max_start + 1);
    }

    ImGui::Spacing();

    // Render grid with horizontal scrollbar for wide grids
    ImGui::BeginChild("GridPreview", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

    int item_size = is_image_data ? 80 : 120;
    int idx = preview_sample_idx_;

    for (int row = 0; row < preview_grid_rows_ && idx < static_cast<int>(dataset_size); ++row) {
        for (int col = 0; col < preview_grid_cols_ && idx < static_cast<int>(dataset_size); ++col) {
            if (col > 0) ImGui::SameLine();

            auto [sample, label] = current_dataset_.GetSample(idx);

            ImGui::BeginGroup();
            ImGui::PushID(idx);

            if (is_image_data && !cached_info_.shape.empty() && cached_info_.shape.size() >= 2) {
                int width = static_cast<int>(cached_info_.shape[0]);
                int height = static_cast<int>(cached_info_.shape[1]);
                int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;

                if (sample.size() == static_cast<size_t>(width * height * channels)) {
                    // Use cached texture with unique key per sample index
                    auto& tm = cyxwiz::TextureManager::Instance();
                    std::string cache_key = "grid_sample_" + std::to_string(idx) + "_" +
                                           current_dataset_.GetName() + "_" +
                                           std::to_string(width) + "x" + std::to_string(height);
                    uint32_t tex_id = tm.GetOrCreateCachedTexture(cache_key, sample.data(),
                                                                   width, height, channels);
                    if (tex_id != 0) {
                        ImGui::Image((ImTextureID)(intptr_t)tex_id,
                                    ImVec2(static_cast<float>(item_size), static_cast<float>(item_size)));
                    }
                }
            } else {
                // Mini feature display for tabular
                ImGui::BeginChild("MiniFeature", ImVec2(static_cast<float>(item_size), static_cast<float>(item_size)), true);
                for (size_t i = 0; i < std::min(size_t(4), sample.size()); ++i) {
                    ImGui::Text("%.2f", sample[i]);
                }
                if (sample.size() > 4) ImGui::Text("...");
                ImGui::EndChild();
            }

            // Label below
            std::string label_text = label >= 0 && label < static_cast<int>(class_names_.size())
                ? class_names_[label] : std::to_string(label);
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", label_text.c_str());

            ImGui::PopID();
            ImGui::EndGroup();

            ++idx;
        }
    }

    ImGui::EndChild();
}

void DatasetPanel::RenderTablePreview(size_t dataset_size) {
    // Table settings
    ImGui::Text("Rows per page:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputInt("##TableRows", &preview_table_rows_, 0, 0);
    preview_table_rows_ = std::clamp(preview_table_rows_, 5, 100);

    ImGui::SameLine();
    ImGui::Text("Page:");
    ImGui::SameLine();
    int page = preview_sample_idx_ / preview_table_rows_;
    int total_pages = static_cast<int>((dataset_size + preview_table_rows_ - 1) / preview_table_rows_);
    ImGui::SetNextItemWidth(60);
    if (ImGui::InputInt("##Page", &page, 0, 0)) {
        page = std::clamp(page, 0, total_pages - 1);
        preview_sample_idx_ = page * preview_table_rows_;
    }
    ImGui::SameLine();
    ImGui::Text("/ %d", total_pages);

    ImGui::SameLine();
    if (ImGui::Button("<##PrevPage") && page > 0) {
        preview_sample_idx_ = (page - 1) * preview_table_rows_;
    }
    ImGui::SameLine();
    if (ImGui::Button(">##NextPage") && page < total_pages - 1) {
        preview_sample_idx_ = (page + 1) * preview_table_rows_;
    }

    ImGui::Spacing();

    // Get feature count from first sample
    auto [first_sample, _] = current_dataset_.GetSample(0);
    int num_features = static_cast<int>(first_sample.size());
    int display_cols = std::min(num_features, 10);  // Limit columns displayed

    ImGui::BeginChild("TablePreview", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

    if (ImGui::BeginTable("DataTable", display_cols + 2,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
            ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable)) {

        // Header
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40);
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 60);
        for (int i = 0; i < display_cols; ++i) {
            char col_name[16];
            snprintf(col_name, sizeof(col_name), "F%d", i);
            ImGui::TableSetupColumn(col_name, ImGuiTableColumnFlags_WidthFixed, 70);
        }
        ImGui::TableHeadersRow();

        // Data rows
        int start = preview_sample_idx_;
        int end = std::min(start + preview_table_rows_, static_cast<int>(dataset_size));

        for (int i = start; i < end; ++i) {
            auto [sample, label] = current_dataset_.GetSample(i);

            ImGui::TableNextRow();

            // Index
            ImGui::TableNextColumn();
            ImGui::Text("%d", i);

            // Label
            ImGui::TableNextColumn();
            if (label >= 0 && label < static_cast<int>(class_names_.size())) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", class_names_[label].c_str());
            } else {
                ImGui::Text("%d", label);
            }

            // Features
            for (int j = 0; j < display_cols && j < static_cast<int>(sample.size()); ++j) {
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", sample[j]);
            }
        }

        ImGui::EndTable();
    }

    if (num_features > display_cols) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Showing %d of %d features", display_cols, num_features);
    }

    ImGui::EndChild();
}

void DatasetPanel::ShowFileBrowser() {
#ifdef _WIN32
    // For ImageCSV and ImageFolder, show folder browser
    if (selected_type_ == cyxwiz::DatasetType::ImageCSV ||
        selected_type_ == cyxwiz::DatasetType::ImageFolder) {
        ShowFolderBrowser(file_path_buffer_, sizeof(file_path_buffer_));
    } else {
        // Show file browser for other types
        OPENFILENAMEA ofn;
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL;
        ofn.lpstrFile = file_path_buffer_;
        ofn.nMaxFile = sizeof(file_path_buffer_);

        if (selected_type_ == cyxwiz::DatasetType::CSV) {
            ofn.lpstrFilter = "CSV Files\0*.csv\0All Files\0*.*\0";
        } else {
            ofn.lpstrFilter = "All Files\0*.*\0";
        }

        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

        if (GetOpenFileNameA(&ofn)) {
            spdlog::info("Selected file: {}", file_path_buffer_);
        }
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

void DatasetPanel::ShowFolderBrowser(char* buffer, size_t buffer_size) {
#ifdef _WIN32
    // Use IFileDialog for modern folder selection
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    IFileDialog* pfd = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                                   IID_IFileDialog, reinterpret_cast<void**>(&pfd));

    if (SUCCEEDED(hr)) {
        // Set options to pick folders
        DWORD dwOptions;
        pfd->GetOptions(&dwOptions);
        pfd->SetOptions(dwOptions | FOS_PICKFOLDERS);

        // Set title
        pfd->SetTitle(L"Select Image Folder");

        // Show the dialog
        hr = pfd->Show(NULL);
        if (SUCCEEDED(hr)) {
            IShellItem* psi = nullptr;
            hr = pfd->GetResult(&psi);
            if (SUCCEEDED(hr)) {
                PWSTR pszPath = nullptr;
                hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
                if (SUCCEEDED(hr)) {
                    // Convert wide string to narrow
                    size_t converted = 0;
                    wcstombs_s(&converted, buffer, buffer_size, pszPath, _TRUNCATE);
                    spdlog::info("Selected folder: {}", buffer);
                    CoTaskMemFree(pszPath);
                }
                psi->Release();
            }
        }
        pfd->Release();
    }

    CoUninitialize();
#else
    spdlog::warn("Folder browser not implemented for this platform");
#endif
}

void DatasetPanel::ShowCSVFileBrowser() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = csv_path_buffer_;
    ofn.nMaxFile = sizeof(csv_path_buffer_);
    ofn.lpstrFilter = "CSV Files\0*.csv\0Text Files\0*.txt\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        spdlog::info("Selected CSV file: {}", csv_path_buffer_);
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

bool DatasetPanel::LoadDataset(const std::string& path) {
    auto& registry = cyxwiz::DataRegistry::Instance();

    // Auto-detect type if not specified
    cyxwiz::DatasetType detected_type = selected_type_;
    if (detected_type == cyxwiz::DatasetType::None) {
        detected_type = cyxwiz::DataRegistry::DetectType(path);
    }

    bool success = false;
    switch (detected_type) {
        case cyxwiz::DatasetType::CSV:
            success = LoadCSVDataset(path);
            break;
        case cyxwiz::DatasetType::ImageFolder:
            success = LoadImageDataset(path);
            break;
        case cyxwiz::DatasetType::MNIST:
            success = LoadMNISTDataset(path);
            break;
        case cyxwiz::DatasetType::CIFAR10:
            success = LoadCIFAR10Dataset(path);
            break;
        default:
            // Try generic load
            current_dataset_ = registry.LoadDataset(path);
            success = current_dataset_.IsValid();
    }

    if (success && current_dataset_.IsValid()) {
        cached_info_ = current_dataset_.GetInfo();
        spdlog::info("Dataset loaded successfully: {}", cached_info_.name);
    }

    return success;
}

bool DatasetPanel::LoadCSVDataset(const std::string& path) {
    spdlog::info("Loading CSV dataset from: {}", path);

    auto& registry = cyxwiz::DataRegistry::Instance();
    current_dataset_ = registry.LoadCSV(path);

    if (!current_dataset_.IsValid()) {
        spdlog::error("Failed to load CSV dataset");
        return false;
    }

    cached_info_ = current_dataset_.GetInfo();
    ApplySplit();

    // Update class counts
    UpdateClassCounts();

    return true;
}

bool DatasetPanel::LoadImageDataset(const std::string& path) {
    spdlog::warn("Image dataset loading not yet implemented");
    auto& registry = cyxwiz::DataRegistry::Instance();
    current_dataset_ = registry.LoadImageFolder(path);
    return current_dataset_.IsValid();
}

bool DatasetPanel::LoadMNISTDataset(const std::string& path) {
    spdlog::info("Loading MNIST dataset from: {}", path);

    auto& registry = cyxwiz::DataRegistry::Instance();
    current_dataset_ = registry.LoadMNIST(path);

    if (!current_dataset_.IsValid()) {
        spdlog::error("Failed to load MNIST dataset");
        return false;
    }

    cached_info_ = current_dataset_.GetInfo();
    class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    ApplySplit();
    UpdateClassCounts();

    return true;
}

bool DatasetPanel::LoadCIFAR10Dataset(const std::string& path) {
    spdlog::info("Loading CIFAR-10 dataset from: {}", path);

    auto& registry = cyxwiz::DataRegistry::Instance();
    current_dataset_ = registry.LoadCIFAR10(path);

    if (!current_dataset_.IsValid()) {
        spdlog::error("Failed to load CIFAR-10 dataset");
        return false;
    }

    cached_info_ = current_dataset_.GetInfo();
    class_names_ = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    ApplySplit();
    UpdateClassCounts();

    return true;
}

bool DatasetPanel::LoadHuggingFaceDataset(const std::string& dataset_name) {
    spdlog::info("Loading HuggingFace dataset: {}", dataset_name);

    auto& registry = cyxwiz::DataRegistry::Instance();

    // Create HuggingFace config
    cyxwiz::HuggingFaceConfig config;
    config.dataset_name = dataset_name;
    config.split = "train";
    config.cache_dir = "./data/huggingface_cache";

    current_dataset_ = registry.LoadHuggingFace(config);

    if (!current_dataset_.IsValid()) {
        spdlog::error("Failed to load HuggingFace dataset: {}", dataset_name);
        return false;
    }

    cached_info_ = current_dataset_.GetInfo();
    class_names_ = cached_info_.class_names;
    ApplySplit();
    UpdateClassCounts();

    spdlog::info("Loaded HuggingFace dataset '{}': {} samples, {} classes",
                dataset_name, cached_info_.num_samples, cached_info_.num_classes);
    return true;
}

bool DatasetPanel::LoadKaggleDataset(const std::string& dataset_slug) {
    spdlog::info("Loading Kaggle dataset: {}", dataset_slug);

    auto& registry = cyxwiz::DataRegistry::Instance();

    // Create Kaggle config
    cyxwiz::KaggleConfig config;
    config.dataset_slug = dataset_slug;
    config.cache_dir = "./data/kaggle_cache";

    current_dataset_ = registry.LoadKaggle(config);

    if (!current_dataset_.IsValid()) {
        spdlog::error("Failed to load Kaggle dataset: {}", dataset_slug);
        return false;
    }

    cached_info_ = current_dataset_.GetInfo();
    class_names_ = cached_info_.class_names;
    ApplySplit();
    UpdateClassCounts();

    spdlog::info("Loaded Kaggle dataset '{}': {} samples, {} classes",
                dataset_slug, cached_info_.num_samples, cached_info_.num_classes);
    return true;
}

// ============================================================================
// Async Dataset Loading Methods
// ============================================================================

void DatasetPanel::LoadCSVDatasetAsync(const std::string& path) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading CSV: " + std::filesystem::path(path).filename().string(),
        [this, path](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            task.ReportProgress(0.3f, "Parsing CSV data...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadCSV(path);

            task.ReportProgress(0.8f, "Processing dataset...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                ApplySplit();
                UpdateClassCounts();
                task.ReportProgress(1.0f, "Complete");
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load CSV file");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async CSV load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadImageDatasetAsync(const std::string& path) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading Images: " + std::filesystem::path(path).filename().string(),
        [this, path](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Scanning directory...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            task.ReportProgress(0.3f, "Loading images...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadImageFolder(path);

            task.ReportProgress(0.9f, "Finalizing...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                ApplySplit();
                UpdateClassCounts();
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load image folder");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async image load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadMNISTDatasetAsync(const std::string& path) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading MNIST",
        [this, path](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening MNIST files...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            task.ReportProgress(0.3f, "Reading training data...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadMNIST(path);

            task.ReportProgress(0.7f, "Processing labels...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
                ApplySplit();

                task.ReportProgress(0.9f, "Computing statistics...");
                UpdateClassCounts();

                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load MNIST dataset");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async MNIST load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadCIFAR10DatasetAsync(const std::string& path) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading CIFAR-10",
        [this, path](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening CIFAR-10 files...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            task.ReportProgress(0.3f, "Reading image data...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadCIFAR10(path);

            task.ReportProgress(0.7f, "Processing labels...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                class_names_ = {
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"
                };
                ApplySplit();

                task.ReportProgress(0.9f, "Computing statistics...");
                UpdateClassCounts();

                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load CIFAR-10 dataset");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async CIFAR-10 load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadHuggingFaceDatasetAsync(const std::string& dataset_name) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading HuggingFace: " + dataset_name,
        [this, dataset_name](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Connecting to HuggingFace...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            cyxwiz::HuggingFaceConfig config;
            config.dataset_name = dataset_name;
            config.split = "train";
            config.cache_dir = "./data/huggingface_cache";

            task.ReportProgress(0.3f, "Downloading dataset...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadHuggingFace(config);

            task.ReportProgress(0.8f, "Processing data...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                class_names_ = cached_info_.class_names;
                ApplySplit();
                UpdateClassCounts();
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load HuggingFace dataset");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async HuggingFace load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadKaggleDatasetAsync(const std::string& dataset_slug) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading Kaggle: " + dataset_slug,
        [this, dataset_slug](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Authenticating with Kaggle...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            cyxwiz::KaggleConfig config;
            config.dataset_slug = dataset_slug;
            config.cache_dir = "./data/kaggle_cache";

            task.ReportProgress(0.3f, "Downloading dataset...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadKaggle(config);

            task.ReportProgress(0.8f, "Processing data...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                class_names_ = cached_info_.class_names;
                ApplySplit();
                UpdateClassCounts();
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load Kaggle dataset");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async Kaggle load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadCustomDatasetAsync(const cyxwiz::CustomConfig& config) {
    if (is_loading_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    is_loading_.store(true);
    loading_progress_.store(0.0f);
    loading_status_message_ = "Starting...";

    loading_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading Custom Dataset",
        [this, config](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Analyzing file format...");

            auto& registry = cyxwiz::DataRegistry::Instance();

            task.ReportProgress(0.3f, "Reading data...");
            if (task.ShouldStop()) return;

            auto handle = registry.LoadCustom(config);

            task.ReportProgress(0.8f, "Processing data...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                current_dataset_ = handle;
                cached_info_ = current_dataset_.GetInfo();
                class_names_ = cached_info_.class_names;
                ApplySplit();
                UpdateClassCounts();
                task.MarkCompleted();
                spdlog::info("Loaded custom dataset: {} samples", cached_info_.num_samples);
            } else {
                task.MarkFailed("Failed to load custom dataset");
            }
        },
        [this](float progress, const std::string& msg) {
            loading_progress_.store(progress);
            loading_status_message_ = msg;
        },
        [this](bool success, const std::string& error) {
            is_loading_.store(false);
            loading_status_message_.clear();
            if (!success) {
                spdlog::error("Async custom dataset load failed: {}", error);
            }
        }
    );
}

void DatasetPanel::LoadDatasetAsync(const std::string& path) {
    // Auto-detect dataset type and use appropriate async loader
    std::filesystem::path fs_path(path);
    std::string ext = fs_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".csv") {
        LoadCSVDatasetAsync(path);
    } else if (std::filesystem::is_directory(fs_path)) {
        // Check for MNIST/CIFAR-10 patterns
        if (std::filesystem::exists(fs_path / "train-images-idx3-ubyte") ||
            std::filesystem::exists(fs_path / "train-images.idx3-ubyte")) {
            LoadMNISTDatasetAsync(path);
        } else if (std::filesystem::exists(fs_path / "data_batch_1.bin") ||
                   std::filesystem::exists(fs_path / "cifar-10-batches-bin")) {
            LoadCIFAR10DatasetAsync(path);
        } else {
            LoadImageDatasetAsync(path);
        }
    } else {
        spdlog::warn("Unknown dataset format: {}", path);
    }
}

void DatasetPanel::CancelLoading() {
    if (is_loading_.load() && loading_task_id_ != 0) {
        cyxwiz::AsyncTaskManager::Instance().Cancel(loading_task_id_);
        spdlog::info("Cancelled dataset loading task");
    }
}

void DatasetPanel::UpdateClassCounts() {
    if (!current_dataset_.IsValid()) {
        class_counts_.clear();
        return;
    }

    class_counts_.resize(cached_info_.num_classes, 0);
    std::fill(class_counts_.begin(), class_counts_.end(), 0);

    // Sample labels to compute distribution
    size_t sample_count = std::min(current_dataset_.Size(), size_t(10000));
    for (size_t i = 0; i < sample_count; ++i) {
        auto [_, label] = current_dataset_.GetSample(i);
        if (label >= 0 && label < static_cast<int>(class_counts_.size())) {
            class_counts_[label]++;
        }
    }
}

void DatasetPanel::ApplySplit() {
    if (!current_dataset_.IsValid()) return;

    spdlog::info("Applying train/val/test split: {:.2f}/{:.2f}/{:.2f}",
                 split_config_.train_ratio, split_config_.val_ratio, split_config_.test_ratio);

    current_dataset_.ApplySplit(split_config_);
    cached_info_ = current_dataset_.GetInfo();

    spdlog::info("Split complete: {} train, {} val, {} test",
        cached_info_.train_count, cached_info_.val_count, cached_info_.test_count);
}

void DatasetPanel::ClearDataset() {
    if (current_dataset_.IsValid()) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        registry.UnloadDataset(current_dataset_.GetName());
    }

    current_dataset_ = cyxwiz::DatasetHandle();
    cached_info_ = cyxwiz::DatasetInfo();
    class_counts_.clear();
    class_names_.clear();
    preview_sample_idx_ = 0;

    spdlog::info("Dataset cleared");
}

bool DatasetPanel::GetPreviewSamples(int count, std::vector<float>& out_images, std::vector<int>& out_labels) {
    if (!current_dataset_.IsValid() || count <= 0) return false;

    count = std::min(count, static_cast<int>(current_dataset_.Size()));

    out_images.clear();
    out_labels.clear();

    for (int i = 0; i < count; ++i) {
        auto [sample, label] = current_dataset_.GetSample(i);
        out_images.insert(out_images.end(), sample.begin(), sample.end());
        out_labels.push_back(label);
    }

    return true;
}

void DatasetPanel::RenderTrainingSection() {
    // Check training state from centralized TrainingManager
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    bool training_running = training_mgr.IsTrainingActive();

    // Check if node editor is connected and graph is valid
    bool has_valid_graph = node_editor_ && node_editor_->IsGraphValid();

    // Status bar at top
    if (training_running) {
        auto metrics = training_mgr.GetCurrentMetrics();
        float progress = static_cast<float>(metrics.current_epoch) / std::max(1, metrics.total_epochs);

        // Training status card
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.35f, 0.15f, 1.0f));
        ImGui::BeginChild("TrainingStatus", ImVec2(-1, 100), true);

        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Training in Progress");

        ImGui::Spacing();
        ImGui::ProgressBar(progress, ImVec2(-1, 20));

        ImGui::Columns(4, "MetricsRow", false);
        ImGui::Text("Epoch");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "%d / %d", metrics.current_epoch, metrics.total_epochs);
        ImGui::NextColumn();
        ImGui::Text("Loss");
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "%.4f", metrics.train_loss);
        ImGui::NextColumn();
        ImGui::Text("Accuracy");
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%.2f%%", metrics.train_accuracy * 100.0f);
        ImGui::NextColumn();
        ImGui::Text("Val Loss");
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.4f", metrics.val_loss);
        ImGui::Columns(1);

        ImGui::EndChild();
        ImGui::PopStyleColor();

        // Stop button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button("Stop Training", ImVec2(-1, 30))) {
            StopLocalTraining();
        }
        ImGui::PopStyleColor(2);

        ImGui::Spacing();
    }

    // Show warning if no valid graph
    if (!has_valid_graph && !training_running) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.3f, 0.2f, 0.1f, 1.0f));
        ImGui::BeginChild("GraphWarning", ImVec2(-1, 50), true);
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Model Required");
        ImGui::TextWrapped("Create a model in Node Editor: Input -> Layers -> Loss -> Optimizer");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // Hyperparameters section
    if (ImGui::CollapsingHeader("Hyperparameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (training_running) ImGui::BeginDisabled();

        // Basic parameters
        ImGui::Columns(2, "HyperparamsBasic", false);

        ImGui::Text("Epochs");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputInt("##Epochs", &train_epochs_, 1, 10);
        train_epochs_ = std::clamp(train_epochs_, 1, 1000);

        ImGui::NextColumn();

        ImGui::Text("Batch Size");
        ImGui::SetNextItemWidth(-1);
        const char* batch_sizes[] = {"8", "16", "32", "64", "128", "256"};
        int batch_idx = 0;
        if (train_batch_size_ == 8) batch_idx = 0;
        else if (train_batch_size_ == 16) batch_idx = 1;
        else if (train_batch_size_ == 32) batch_idx = 2;
        else if (train_batch_size_ == 64) batch_idx = 3;
        else if (train_batch_size_ == 128) batch_idx = 4;
        else if (train_batch_size_ == 256) batch_idx = 5;
        if (ImGui::Combo("##BatchSize", &batch_idx, batch_sizes, IM_ARRAYSIZE(batch_sizes))) {
            const int sizes[] = {8, 16, 32, 64, 128, 256};
            train_batch_size_ = sizes[batch_idx];
        }

        ImGui::Columns(1);

        ImGui::Spacing();

        // Optimizer settings
        ImGui::Text("Optimizer");
        ImGui::SetNextItemWidth(150);
        const char* optimizers[] = {"SGD", "Adam", "AdamW", "RMSprop"};
        ImGui::Combo("##Optimizer", &selected_optimizer_, optimizers, IM_ARRAYSIZE(optimizers));

        ImGui::SameLine();
        ImGui::Text("Learning Rate");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::InputFloat("##LR", &train_learning_rate_, 0.0f, 0.0f, "%.6f");

        // Advanced options (collapsible)
        if (ImGui::TreeNode("Advanced Options")) {
            ImGui::Columns(2, "AdvancedParams", false);

            ImGui::Text("Weight Decay");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputFloat("##WeightDecay", &train_weight_decay_, 0.0f, 0.0f, "%.6f");

            ImGui::NextColumn();

            ImGui::Text("Gradient Clip");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputFloat("##GradClip", &train_grad_clip_, 0.0f, 0.0f, "%.2f");

            ImGui::Columns(1);

            ImGui::Checkbox("Early Stopping", &train_early_stopping_);
            if (train_early_stopping_) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Patience##ES", &train_early_stopping_patience_, 0, 0);
            }

            ImGui::Checkbox("Learning Rate Scheduler", &train_lr_scheduler_);
            if (train_lr_scheduler_) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120);
                const char* schedulers[] = {"StepLR", "CosineAnnealing", "ReduceOnPlateau"};
                ImGui::Combo("##Scheduler", &train_scheduler_type_, schedulers, IM_ARRAYSIZE(schedulers));
            }

            ImGui::TreePop();
        }

        if (training_running) ImGui::EndDisabled();
    }

    ImGui::Spacing();

    // Local Training section
    if (ImGui::CollapsingHeader("Local Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool can_train_local = IsDatasetLoaded() && has_valid_graph && !training_running;

        // Dataset info
        if (IsDatasetLoaded()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "Dataset: %s (%zu samples)", cached_info_.name.c_str(), cached_info_.num_samples);
        }

        if (!can_train_local) ImGui::BeginDisabled();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 1.0f));
        if (ImGui::Button("Start Training", ImVec2(-1, 35))) {
            StartLocalTraining();
        }
        ImGui::PopStyleColor(2);

        if (!can_train_local) ImGui::EndDisabled();

        // Requirements checklist
        if (!can_train_local && !training_running) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Requirements:");
            ImGui::BulletText("%s Dataset loaded", IsDatasetLoaded() ? "[OK]" : "[  ]");
            ImGui::BulletText("%s Valid model graph", has_valid_graph ? "[OK]" : "[  ]");
        }
    }

    ImGui::Spacing();

    // P2P Network Training section
    if (ImGui::CollapsingHeader("P2P Network Training")) {
        bool can_train_p2p = job_manager_ != nullptr && job_manager_->IsConnected() && IsDatasetLoaded();

        // Connection status
        if (!job_manager_) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Status: Not configured");
        } else if (!job_manager_->IsConnected()) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Status: Disconnected");
            if (ImGui::Button("Connect to Network", ImVec2(-1, 25))) {
                // TODO: Trigger connection
            }
        } else {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Status: Connected");
        }

        ImGui::Spacing();

        if (!can_train_p2p) ImGui::BeginDisabled();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.6f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
        if (ImGui::Button("Submit to Network", ImVec2(-1, 35))) {
            if (SubmitTrainingJob()) {
                spdlog::info("Training job submitted successfully");
            }
        }
        ImGui::PopStyleColor(2);

        if (!can_train_p2p) ImGui::EndDisabled();

        // Last submitted job
        if (!last_submitted_job_id_.empty()) {
            ImGui::Spacing();
            ImGui::Text("Last Job:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", last_submitted_job_id_.c_str());
        }
    }
}

bool DatasetPanel::SubmitTrainingJob() {
    if (!job_manager_ || !job_manager_->IsConnected()) {
        spdlog::error("Cannot submit job: not connected to server");
        return false;
    }

    if (!IsDatasetLoaded()) {
        spdlog::error("Cannot submit job: no dataset loaded");
        return false;
    }

    spdlog::info("Preparing training job submission...");

    // Create model definition JSON
    nlohmann::json model_def;
    model_def["type"] = "mlp";

    // Calculate input size from dataset
    size_t input_size = 1;
    for (size_t dim : cached_info_.shape) {
        input_size *= dim;
    }

    model_def["input_size"] = input_size;
    model_def["output_size"] = cached_info_.num_classes;
    model_def["hidden_layers"] = nlohmann::json::array({256, 128});
    model_def["activation"] = "relu";
    model_def["output_activation"] = "softmax";

    std::string model_definition = model_def.dump();

    // Create dataset URI
    std::string dataset_uri;
    switch (cached_info_.type) {
        case cyxwiz::DatasetType::MNIST:
            dataset_uri = "file://mnist/" + cached_info_.path;
            break;
        case cyxwiz::DatasetType::CIFAR10:
            dataset_uri = "file://cifar10/" + cached_info_.path;
            break;
        case cyxwiz::DatasetType::CSV:
            dataset_uri = "file://csv/" + cached_info_.path;
            break;
        default:
            dataset_uri = "mock://random";
    }

    // Create job config
    cyxwiz::protocol::JobConfig config;
    config.set_job_type(cyxwiz::protocol::JOB_TYPE_TRAINING);
    config.set_priority(cyxwiz::protocol::PRIORITY_NORMAL);
    config.set_model_definition(model_definition);
    config.set_dataset_uri(dataset_uri);
    config.set_batch_size(train_batch_size_);
    config.set_epochs(train_epochs_);
    config.set_required_device(cyxwiz::protocol::DEVICE_CUDA);

    // Calculate estimated memory
    int64_t estimated_memory = static_cast<int64_t>(input_size) * train_batch_size_ * 4 * 10;
    config.set_estimated_memory(std::max(estimated_memory, static_cast<int64_t>(512 * 1024 * 1024)));

    // Estimate duration
    int estimated_duration = (train_epochs_ * static_cast<int>(cached_info_.num_samples) / 1000) + 60;
    config.set_estimated_duration(estimated_duration);

    config.set_payment_amount(0.1);

    // Set wallet address from WalletPanel for blockchain escrow
    if (wallet_panel_ && wallet_panel_->IsConnected()) {
        config.set_payment_address(wallet_panel_->GetWalletAddress());
        spdlog::info("Using wallet address for job payment: {}", wallet_panel_->GetWalletAddress());
    } else {
        spdlog::error("Cannot submit job: wallet not connected. Please connect your wallet first.");
        return false;
    }

    // Add hyperparameters
    auto* hyperparams = config.mutable_hyperparameters();
    (*hyperparams)["learning_rate"] = std::to_string(train_learning_rate_);
    (*hyperparams)["optimizer"] = (selected_optimizer_ == 0) ? "sgd" :
                                  (selected_optimizer_ == 1) ? "adam" : "adamw";

    // Submit with P2P workflow
    std::string job_id;
    if (!job_manager_->SubmitJobWithP2P(config, job_id)) {
        spdlog::error("Failed to submit training job");
        return false;
    }

    last_submitted_job_id_ = job_id;
    spdlog::info("Training job submitted: {}", job_id);

    if (training_start_callback_) {
        training_start_callback_(job_id);
    }

    return true;
}

void DatasetPanel::StartLocalTraining() {
    if (!IsDatasetLoaded()) {
        spdlog::error("Cannot start local training: no dataset loaded");
        return;
    }

    if (!node_editor_) {
        spdlog::error("Cannot start local training: no node editor connected");
        return;
    }

    // Check if the graph is valid for training
    if (!node_editor_->IsGraphValid()) {
        spdlog::error("Cannot start local training: node graph is not valid for training");
        spdlog::error("Need: DatasetInput -> Model layers -> Loss -> Optimizer");
        return;
    }

    spdlog::info("Starting local training via TrainingManager...");
    spdlog::info("  Dataset: {} ({} samples)", cached_info_.name, cached_info_.num_samples);
    spdlog::info("  Epochs: {}, Batch Size: {}, LR: {}", train_epochs_, train_batch_size_, train_learning_rate_);

    // Compile the node graph to get the model architecture
    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(
        node_editor_->GetNodes(),
        node_editor_->GetLinks()
    );

    if (!config.is_valid) {
        spdlog::error("Graph compilation failed: {}", config.error_message);
        return;
    }

    // Update config with dataset info
    config.dataset_name = cached_info_.name;

    // Calculate input size from shape (product of dimensions)
    size_t input_size = 1;
    for (auto dim : cached_info_.shape) {
        input_size *= dim;
    }
    config.input_size = input_size;
    config.output_size = cached_info_.num_classes;

    // Override with UI settings
    config.learning_rate = train_learning_rate_;
    if (selected_optimizer_ == 0) {
        config.optimizer_type = gui::NodeType::SGD;
    } else if (selected_optimizer_ == 1) {
        config.optimizer_type = gui::NodeType::Adam;
    } else {
        config.optimizer_type = gui::NodeType::AdamW;
    }

    spdlog::info("  Compiled graph: {} layers, optimizer={}",
                 config.layers.size(), config.GetOptimizerName());

    // Create callback to update node editor training animation
    auto node_editor_callback = [this](bool active) {
        if (node_editor_) {
            node_editor_->SetTrainingActive(active);
        }
    };

    // Delegate to TrainingManager with compiled configuration
    bool started = cyxwiz::TrainingManager::Instance().StartTraining(
        std::move(config),
        current_dataset_,
        train_epochs_,
        train_batch_size_,
        training_plot_panel_,
        node_editor_callback
    );

    if (!started) {
        spdlog::warn("Could not start training - another training session may be in progress");
    }
}

// ============================================================================
// Augmentation Tab Implementation
// ============================================================================

void DatasetPanel::RenderAugmentationTab() {
    using namespace cyxwiz::transforms;

    ImGui::BeginChild("AugmentationPanel", ImVec2(0, 0), false);

    // Vertical layout: Pipeline on top, Preview below
    RenderAugmentationPipeline();

    ImGui::Spacing();

    // Preview section below pipeline
    RenderAugmentationPreview();

    ImGui::EndChild();
}

void DatasetPanel::RenderAugmentationPipeline() {
    using namespace cyxwiz::transforms;

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Augmentation Pipeline");
    ImGui::Spacing();

    // Preset selection
    ImGui::Text("Preset:");
    ImGui::SameLine();
    const char* presets[] = {
        "None",
        "ImageNet Train",
        "CIFAR-10 Train",
        "Medical Imaging",
        "RandAugment",
        "AutoAugment",
        "Aggressive",
        "Light",
        "Object Detection",
        "Satellite/Aerial",
        "Text/OCR",
        "Grayscale (MNIST)",
        "Custom"
    };
    if (ImGui::Combo("##AugPreset", &augmentation_preset_, presets, IM_ARRAYSIZE(presets))) {
        // Apply preset
        switch (augmentation_preset_) {
            case 0:  // None
                augmentation_pipeline_.reset();
                break;
            case 1:  // ImageNet Train
                augmentation_pipeline_ = TransformFactory::createImageNetTrain(224);
                break;
            case 2:  // CIFAR-10 Train
                augmentation_pipeline_ = TransformFactory::createCIFAR10Train();
                break;
            case 3:  // Medical Imaging
                augmentation_pipeline_ = TransformFactory::createMedicalTrain(224);
                break;
            case 4:  // RandAugment - strong augmentation
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<RandomResizedCrop>(224));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<RandAugment>(2, 9));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 5:  // AutoAugment - learned policies
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<RandomResizedCrop>(224));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<AutoAugment>(AutoAugmentPolicy::ImageNet));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 6:  // Aggressive - heavy regularization
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<RandomResizedCrop>(224, 0.08f, 1.0f));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<RandomVerticalFlip>(0.2f));
                    pipeline->add(std::make_unique<RandomRotation>(30.0f));
                    pipeline->add(std::make_unique<ColorJitter>(0.5f, 0.5f, 0.5f, 0.2f));
                    pipeline->add(std::make_unique<RandomGrayscale>(0.2f));
                    pipeline->add(std::make_unique<GaussianBlur>(3, 0.5f));
                    pipeline->add(std::make_unique<RandomErasing>(0.4f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 7:  // Light - minimal transforms
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<Resize>(224, 224));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<ColorJitter>(0.1f, 0.1f, 0.0f, 0.0f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 8:  // Object Detection
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<Resize>(416, 416));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.2f, 0.1f));
                    pipeline->add(std::make_unique<RandomRotation>(10.0f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 9:  // Satellite/Aerial
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<Resize>(256, 256));
                    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                    pipeline->add(std::make_unique<RandomVerticalFlip>(0.5f));
                    pipeline->add(std::make_unique<RandomRotation>(90.0f));  // Any orientation
                    pipeline->add(std::make_unique<ColorJitter>(0.2f, 0.2f, 0.1f, 0.05f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 10:  // Text/OCR
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<Resize>(224, 224));
                    // Conservative for text - no flips, minimal distortion
                    pipeline->add(std::make_unique<RandomRotation>(5.0f));
                    pipeline->add(std::make_unique<ColorJitter>(0.2f, 0.3f, 0.0f, 0.0f));
                    pipeline->add(std::make_unique<GaussianBlur>(3, 0.3f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.485f, 0.456f, 0.406f},
                        std::vector<float>{0.229f, 0.224f, 0.225f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 11:  // Grayscale (MNIST-style)
                {
                    auto pipeline = std::make_unique<Compose>();
                    pipeline->add(std::make_unique<Resize>(28, 28));
                    pipeline->add(std::make_unique<Grayscale>(1));
                    pipeline->add(std::make_unique<RandomRotation>(15.0f));
                    pipeline->add(std::make_unique<Normalize>(
                        std::vector<float>{0.1307f},
                        std::vector<float>{0.3081f}));
                    augmentation_pipeline_ = std::move(pipeline);
                }
                break;
            case 12:  // Custom - start empty
                augmentation_pipeline_ = std::make_unique<Compose>();
                break;
        }
        preview_needs_update_ = true;

        // Update UI state
        if (augmentation_pipeline_) {
            transform_ui_states_.resize(augmentation_pipeline_->size());
            for (auto& state : transform_ui_states_) {
                state.enabled = true;
                state.expanded = false;
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Transform list
    if (augmentation_pipeline_ && augmentation_pipeline_->size() > 0) {
        ImGui::Text("Transforms (%zu):", augmentation_pipeline_->size());
        ImGui::Spacing();

        // Ensure UI state vector matches
        while (transform_ui_states_.size() < augmentation_pipeline_->size()) {
            transform_ui_states_.push_back({true, false});
        }

        for (size_t i = 0; i < augmentation_pipeline_->size(); ++i) {
            auto* transform = augmentation_pipeline_->get(i);
            if (!transform) continue;

            ImGui::PushID(static_cast<int>(i));

            // Enable/disable checkbox
            bool enabled = transform->isEnabled();
            if (ImGui::Checkbox("##Enabled", &enabled)) {
                transform->setEnabled(enabled);
                preview_needs_update_ = true;
            }
            ImGui::SameLine();

            // Transform header
            ImGui::Text("[%zu] %s", i + 1, transform->name().c_str());

            // Category badge
            ImGui::SameLine();
            ImVec4 badge_color;
            if (transform->category() == "Geometric") badge_color = ImVec4(0.2f, 0.6f, 0.8f, 1.0f);
            else if (transform->category() == "Color") badge_color = ImVec4(0.8f, 0.4f, 0.2f, 1.0f);
            else if (transform->category() == "Noise") badge_color = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
            else badge_color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);  // Advanced

            ImGui::TextColored(badge_color, "(%s)", transform->category().c_str());

            // Parameters (if expanded)
            if (transform_ui_states_[i].expanded) {
                ImGui::Indent();
                auto params = transform->getParams();
                for (const auto& [key, value] : params) {
                    if (std::holds_alternative<float>(value)) {
                        ImGui::Text("  %s: %.3f", key.c_str(), std::get<float>(value));
                    } else if (std::holds_alternative<int>(value)) {
                        ImGui::Text("  %s: %d", key.c_str(), std::get<int>(value));
                    } else if (std::holds_alternative<bool>(value)) {
                        ImGui::Text("  %s: %s", key.c_str(), std::get<bool>(value) ? "true" : "false");
                    }
                }
                ImGui::Unindent();
            }

            ImGui::PopID();
        }
    } else if (augmentation_preset_ == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No augmentation selected");
        ImGui::Text("Choose a preset to get started");
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Pipeline is empty");
    }

    // Add transform button (for custom mode)
    if (augmentation_preset_ == 4 && augmentation_pipeline_) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("+ Add Transform")) {
            ImGui::OpenPopup("AddTransformPopup");
        }

        if (ImGui::BeginPopup("AddTransformPopup")) {
            ImGui::Text("Geometric");
            ImGui::Separator();
            if (ImGui::Selectable("Random Horizontal Flip")) {
                augmentation_pipeline_->add(std::make_unique<RandomHorizontalFlip>(0.5f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Random Vertical Flip")) {
                augmentation_pipeline_->add(std::make_unique<RandomVerticalFlip>(0.5f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Random Rotation (15°)")) {
                augmentation_pipeline_->add(std::make_unique<RandomRotation>(15.0f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Random Resized Crop (224)")) {
                augmentation_pipeline_->add(std::make_unique<RandomResizedCrop>(224));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }

            ImGui::Spacing();
            ImGui::Text("Color");
            ImGui::Separator();
            if (ImGui::Selectable("Color Jitter")) {
                augmentation_pipeline_->add(std::make_unique<ColorJitter>(0.4f, 0.4f, 0.4f, 0.1f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Random Grayscale")) {
                augmentation_pipeline_->add(std::make_unique<RandomGrayscale>(0.1f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Normalize (ImageNet)")) {
                augmentation_pipeline_->add(std::make_unique<Normalize>(
                    std::vector<float>{0.485f, 0.456f, 0.406f},
                    std::vector<float>{0.229f, 0.224f, 0.225f}));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }

            ImGui::Spacing();
            ImGui::Text("Noise/Blur");
            ImGui::Separator();
            if (ImGui::Selectable("Gaussian Blur")) {
                augmentation_pipeline_->add(std::make_unique<RandomGaussianBlur>());
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Random Erasing")) {
                augmentation_pipeline_->add(std::make_unique<RandomErasing>(0.5f));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("Cutout")) {
                augmentation_pipeline_->add(std::make_unique<Cutout>(1, 16));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }

            ImGui::Spacing();
            ImGui::Text("Advanced");
            ImGui::Separator();
            if (ImGui::Selectable("RandAugment")) {
                augmentation_pipeline_->add(std::make_unique<RandAugment>(2, 9));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }
            if (ImGui::Selectable("AutoAugment (ImageNet)")) {
                augmentation_pipeline_->add(std::make_unique<AutoAugment>(AutoAugmentPolicy::ImageNet));
                transform_ui_states_.push_back({true, false});
                preview_needs_update_ = true;
            }

            ImGui::EndPopup();
        }
    }
}

void DatasetPanel::RenderAugmentationPreview() {
    using namespace cyxwiz::transforms;

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Augmentation Preview");
    ImGui::Spacing();

    if (!IsDatasetLoaded()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a dataset to preview augmentations");
        return;
    }

    // Get sample from dataset
    auto info = cached_info_;
    if (info.num_samples == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Dataset is empty");
        return;
    }

    // Sample selector (using preview_sample_idx_ member)
    int max_idx = static_cast<int>(info.num_samples) - 1;
    ImGui::Text("Sample:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##SampleIdx", &preview_sample_idx_)) {
        preview_sample_idx_ = std::clamp(preview_sample_idx_, 0, max_idx);
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("<")) {
        preview_sample_idx_ = std::max(0, preview_sample_idx_ - 1);
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(">")) {
        preview_sample_idx_ = std::min(max_idx, preview_sample_idx_ + 1);
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Random")) {
        preview_sample_idx_ = rand() % (max_idx + 1);
        preview_needs_update_ = true;
    }

    ImGui::Spacing();

    // Refresh button
    if (ImGui::Button("Apply Augmentation")) {
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Auto-refresh", &show_augmented_preview_);

    ImGui::Spacing();

    // Load sample and apply augmentation
    if (preview_needs_update_ || show_augmented_preview_) {
        auto [sample_data, label] = current_dataset_.GetSample(preview_sample_idx_);

        if (!sample_data.empty() && info.shape.size() >= 2) {
            int width = static_cast<int>(info.shape[0]);
            int height = static_cast<int>(info.shape[1]);
            int channels = info.shape.size() >= 3 ? static_cast<int>(info.shape[2]) : 1;

            // Create original image
            preview_original_ = Image(sample_data, width, height, channels);

            // Apply augmentation
            if (augmentation_pipeline_) {
                preview_augmented_ = augmentation_pipeline_->apply(preview_original_);
            } else {
                preview_augmented_ = preview_original_;
            }
        }

        preview_needs_update_ = false;
    }

    // Display vertically (original on top, augmented below)
    if (preview_original_.isValid()) {
        float preview_size = 150.0f;

        // Calculate aspect ratio preserved display size
        float aspect = static_cast<float>(preview_original_.width) / static_cast<float>(preview_original_.height);
        float display_w = preview_size;
        float display_h = preview_size;
        if (aspect > 1.0f) {
            display_h = preview_size / aspect;
        } else {
            display_w = preview_size * aspect;
        }

        // Create/update textures for preview
        auto& tm = cyxwiz::TextureManager::Instance();

        // Check if original texture needs recreation (size changed)
        bool orig_size_changed = (preview_original_.width != preview_tex_orig_w_ ||
                                   preview_original_.height != preview_tex_orig_h_ ||
                                   preview_original_.channels != preview_tex_orig_c_);

        if (preview_texture_original_ == 0 || orig_size_changed) {
            if (preview_texture_original_ != 0) {
                tm.DeleteTexture(preview_texture_original_);
            }
            preview_texture_original_ = tm.CreateTextureFromFloatData(
                preview_original_.data.data(),
                preview_original_.width,
                preview_original_.height,
                preview_original_.channels
            );
            preview_tex_orig_w_ = preview_original_.width;
            preview_tex_orig_h_ = preview_original_.height;
            preview_tex_orig_c_ = preview_original_.channels;
        } else {
            tm.UpdateTexture(preview_texture_original_,
                preview_original_.data.data(),
                preview_original_.width,
                preview_original_.height,
                preview_original_.channels
            );
        }

        // Check if augmented texture needs recreation (size changed)
        bool aug_size_changed = (preview_augmented_.width != preview_tex_aug_w_ ||
                                  preview_augmented_.height != preview_tex_aug_h_ ||
                                  preview_augmented_.channels != preview_tex_aug_c_);

        if (preview_texture_augmented_ == 0 || aug_size_changed) {
            if (preview_texture_augmented_ != 0) {
                tm.DeleteTexture(preview_texture_augmented_);
            }
            preview_texture_augmented_ = tm.CreateTextureFromFloatData(
                preview_augmented_.data.data(),
                preview_augmented_.width,
                preview_augmented_.height,
                preview_augmented_.channels
            );
            preview_tex_aug_w_ = preview_augmented_.width;
            preview_tex_aug_h_ = preview_augmented_.height;
            preview_tex_aug_c_ = preview_augmented_.channels;
        } else {
            tm.UpdateTexture(preview_texture_augmented_,
                preview_augmented_.data.data(),
                preview_augmented_.width,
                preview_augmented_.height,
                preview_augmented_.channels
            );
        }

        // Augmented might have different size, recalculate
        float aug_aspect = static_cast<float>(preview_augmented_.width) / static_cast<float>(preview_augmented_.height);
        float aug_w = preview_size;
        float aug_h = preview_size;
        if (aug_aspect > 1.0f) {
            aug_h = preview_size / aug_aspect;
        } else {
            aug_w = preview_size * aug_aspect;
        }

        // Side-by-side layout: Original on left, Augmented on right
        ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "ORIGINAL (%dx%d)", preview_original_.width, preview_original_.height);
        if (preview_texture_original_ != 0) {
            ImGui::Image((ImTextureID)(intptr_t)preview_texture_original_, ImVec2(display_w, display_h));
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(No texture)");
        }
        ImGui::EndGroup();

        ImGui::SameLine(0.0f, 20.0f);  // 20px spacing between images

        ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "AUGMENTED (%dx%d)", preview_augmented_.width, preview_augmented_.height);
        if (preview_texture_augmented_ != 0) {
            ImGui::Image((ImTextureID)(intptr_t)preview_texture_augmented_, ImVec2(aug_w, aug_h));
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(No texture)");
        }
        ImGui::EndGroup();

        // Show label
        ImGui::Spacing();
        auto [_, label] = current_dataset_.GetSample(preview_sample_idx_);
        if (label >= 0 && label < static_cast<int>(info.class_names.size())) {
            ImGui::Text("Label: %d (%s)", label, info.class_names[label].c_str());
        } else {
            ImGui::Text("Label: %d", label);
        }

        // Show pipeline info
        if (augmentation_pipeline_ && augmentation_pipeline_->size() > 0) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Pipeline: %zu transforms active", augmentation_pipeline_->size());
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No augmentation pipeline");
        }
    }
}

} // namespace gui
