#include "dataset_panel.h"
#include "training_plot_panel.h"
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
#endif

namespace gui {

DatasetPanel::DatasetPanel() : cyxwiz::Panel("Dataset Manager", true) {
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
        ImGui::Separator();
        ImGui::Spacing();

        // Tab bar for different views
        if (ImGui::BeginTabBar("DatasetTabs")) {
            if (ImGui::BeginTabItem("Load Dataset")) {
                ImGui::BeginChild("LoadPanel", ImVec2(0, 0), false);

                // Two column layout
                ImGui::Columns(2, "LoadColumns", true);
                ImGui::SetColumnWidth(0, 280);

                RenderDatasetSelection();

                ImGui::NextColumn();

                if (IsDatasetLoaded()) {
                    RenderDatasetInfo();
                    ImGui::Spacing();
                    RenderSplitConfiguration();
                    ImGui::Spacing();
                    RenderStatistics();
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No dataset loaded");
                    ImGui::Text("Select a dataset type and load data");
                }

                ImGui::Columns(1);
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
    ImGui::Text("Dataset Type");
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
    ImGui::Separator();
    ImGui::Spacing();

    // Show different input based on type
    if (selected_type_ == cyxwiz::DatasetType::HuggingFace) {
        // HuggingFace dataset name input
        ImGui::Text("Dataset Name");
        ImGui::InputText("##HFName", hf_dataset_name_, sizeof(hf_dataset_name_));
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "e.g., mnist, cifar10, imdb");

        ImGui::Spacing();

        // Load button for HuggingFace
        if (ImGui::Button("Load from HuggingFace", ImVec2(-1, 0))) {
            std::string name = hf_dataset_name_;
            if (!name.empty()) {
                LoadHuggingFaceDatasetAsync(name);
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::Kaggle) {
        // Kaggle dataset slug input
        ImGui::Text("Dataset Slug");
        ImGui::InputText("##KaggleSlug", kaggle_dataset_slug_, sizeof(kaggle_dataset_slug_));
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "e.g., titanic, uciml/iris, zalando-research/fashionmnist");

        ImGui::Spacing();

        // Load button for Kaggle
        if (ImGui::Button("Load from Kaggle", ImVec2(-1, 0))) {
            std::string slug = kaggle_dataset_slug_;
            if (!slug.empty()) {
                LoadKaggleDatasetAsync(slug);
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::Custom) {
        // Custom dataset configuration
        ImGui::Text("Data Path");
        ImGui::InputText("##CustomPath", file_path_buffer_, sizeof(file_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...##Custom")) {
            ShowFileBrowser();
        }

        ImGui::Spacing();

        // Format selection
        static int format_idx = 0;
        const char* formats[] = {"Auto-detect", "JSON", "CSV", "TSV", "Binary", "Folder"};
        ImGui::Text("Format");
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

            ImGui::InputText("Data Key (JSON)", data_key, sizeof(data_key));
            ImGui::InputText("Labels Key (JSON)", labels_key, sizeof(labels_key));
            ImGui::InputInt("Label Column (CSV)", &label_column);
            ImGui::Checkbox("Has Header (CSV)", &has_header);
            ImGui::Checkbox("Normalize", &normalize);
            ImGui::InputFloat("Scale Factor", &scale, 0.001f, 0.01f, "%.4f");
        }

        ImGui::Spacing();

        // Load button for Custom
        if (ImGui::Button("Load Custom Dataset", ImVec2(-1, 0))) {
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

                auto& registry = cyxwiz::DataRegistry::Instance();
                auto handle = registry.LoadCustom(config);
                if (handle.IsValid()) {
                    current_dataset_ = handle;
                    cached_info_ = handle.GetInfo();
                    spdlog::info("Loaded custom dataset: {} samples", cached_info_.num_samples);
                }
            }
        }
    } else if (selected_type_ == cyxwiz::DatasetType::ImageCSV || selected_type_ == cyxwiz::DatasetType::ImageFolder) {
        // Unified image dataset UI
        ImGui::Text("Image Folder");
        ImGui::InputText("##ImageFolder", file_path_buffer_, sizeof(file_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...##ImgFolder")) {
            ShowFolderBrowser(file_path_buffer_, sizeof(file_path_buffer_));
        }

        ImGui::Spacing();

        ImGui::Text("Labels CSV (Optional)");
        ImGui::InputText("##CSVPath", csv_path_buffer_, sizeof(csv_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...##CSV")) {
            ShowCSVFileBrowser();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear##ClearCSV")) {
            csv_path_buffer_[0] = '\0';
        }

        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Without CSV: uses subfolders as class names");
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "With CSV: expects filename,label columns");

        ImGui::Spacing();

        // Image size and memory options
        if (ImGui::CollapsingHeader("Image Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputInt("Target Width", &image_target_width_);
            ImGui::InputInt("Target Height", &image_target_height_);
            image_target_width_ = std::max(1, std::min(2048, image_target_width_));
            image_target_height_ = std::max(1, std::min(2048, image_target_height_));
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Images will be resized to this size");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Memory Management (Lazy Loading)");
            ImGui::InputInt("Cache Size (images)", &image_cache_size_);
            image_cache_size_ = std::max(1, std::min(10000, image_cache_size_));

            // Calculate estimated memory usage
            int channels = 3;  // Assume RGB
            size_t bytes_per_image = static_cast<size_t>(image_target_width_) *
                                     static_cast<size_t>(image_target_height_) *
                                     channels * sizeof(float);
            size_t estimated_cache_mb = (bytes_per_image * image_cache_size_) / (1024 * 1024);

            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "Max memory: ~%zu MB (%d images in cache)", estimated_cache_mb, image_cache_size_);
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f),
                "Images loaded on-demand, not all at once");
        }

        ImGui::Spacing();

        // Load button
        if (ImGui::Button("Load Image Dataset", ImVec2(-1, 0))) {
            std::string img_folder = file_path_buffer_;
            std::string csv_file = csv_path_buffer_;  // Can be empty
            if (!img_folder.empty()) {
                // Load with configurable cache size (CSV is optional)
                auto& registry = cyxwiz::DataRegistry::Instance();
                auto handle = registry.LoadImageCSV(img_folder, csv_file, "",
                    image_target_width_, image_target_height_, image_cache_size_);
                if (handle.IsValid()) {
                    current_dataset_ = handle;
                    cached_info_ = handle.GetInfo();
                    if (csv_file.empty()) {
                        spdlog::info("Loaded image dataset from folder: {} samples, {} classes (cache: {} images)",
                            cached_info_.num_samples, cached_info_.num_classes, image_cache_size_);
                    } else {
                        spdlog::info("Loaded image dataset with CSV: {} samples, {} classes (cache: {} images)",
                            cached_info_.num_samples, cached_info_.num_classes, image_cache_size_);
                    }
                }
            }
        }
    } else {
        // File path input for other types
        ImGui::Text("Dataset Path");
        ImGui::InputText("##Path", file_path_buffer_, sizeof(file_path_buffer_));

        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            ShowFileBrowser();
        }

        ImGui::Spacing();

        // Load button
        if (ImGui::Button("Load Dataset", ImVec2(-1, 0))) {
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
    ImGui::Separator();
    ImGui::Spacing();

    // Config Export/Import section
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Configuration");
    ImGui::Spacing();

    if (IsDatasetLoaded()) {
        if (ImGui::Button("Export Config", ImVec2(-1, 0))) {
            // Show save dialog
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
                // Pass the current split_config_ to export with user's modified ratios
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
        // Show open dialog
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
                    // Update split_config_ from the imported config so sliders reflect new values
                    split_config_ = imported_split;
                    // Update cached_info_ counts from the actual dataset
                    cached_info_.train_count = current_dataset_.GetTrainIndices().size();
                    cached_info_.val_count = current_dataset_.GetValIndices().size();
                    cached_info_.test_count = current_dataset_.GetTestIndices().size();
                    spdlog::info("Imported config and applied to dataset: {} (split: {:.0f}/{:.0f}/{:.0f})",
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

    // Versioning
    if (IsDatasetLoaded()) {
        ImGui::Spacing();
        if (ImGui::Button("Save Version", ImVec2(-1, 0))) {
            auto& registry = cyxwiz::DataRegistry::Instance();
            registry.SaveVersion(cached_info_.name, "Manual save");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Save current dataset state as a version");
        }

        // Show version history if available
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

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Quick load buttons for built-in datasets
    ImGui::Text("Quick Load");
    ImGui::Spacing();

    if (ImGui::Button("MNIST (Default)", ImVec2(-1, 0))) {
        LoadMNISTDatasetAsync("./data/mnist");
    }

    if (ImGui::Button("CIFAR-10 (Default)", ImVec2(-1, 0))) {
        LoadCIFAR10DatasetAsync("./data/cifar10");
    }

    ImGui::Spacing();

    // HuggingFace quick load buttons
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "HuggingFace Datasets");
    ImGui::Spacing();

    if (ImGui::Button("HF: MNIST", ImVec2(-1, 0))) {
        LoadHuggingFaceDatasetAsync("mnist");
    }

    if (ImGui::Button("HF: CIFAR-10", ImVec2(-1, 0))) {
        LoadHuggingFaceDatasetAsync("cifar10");
    }

    if (ImGui::Button("HF: IMDB", ImVec2(-1, 0))) {
        LoadHuggingFaceDatasetAsync("imdb");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Kaggle Datasets section
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 1.0f, 1.0f), "Kaggle Datasets");
    ImGui::Spacing();

    if (ImGui::Button("Kaggle: Titanic", ImVec2(-1, 0))) {
        LoadKaggleDatasetAsync("titanic");
    }

    if (ImGui::Button("Kaggle: Iris", ImVec2(-1, 0))) {
        LoadKaggleDatasetAsync("uciml/iris");
    }

    if (ImGui::Button("Kaggle: Digits", ImVec2(-1, 0))) {
        LoadKaggleDatasetAsync("digits");
    }

    if (ImGui::Button("Kaggle: Fashion-MNIST", ImVec2(-1, 0))) {
        LoadKaggleDatasetAsync("zalando-research/fashionmnist");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Memory usage section
    auto& registry = cyxwiz::DataRegistry::Instance();
    cyxwiz::MemoryStats stats = registry.GetMemoryStats();

    // Add texture memory
    auto& texture_mgr = cyxwiz::TextureManager::Instance();
    stats.texture_memory = texture_mgr.GetMemoryUsage();

    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "Memory");

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

    // Collapsible details
    if (ImGui::TreeNode("Details##Memory")) {
        ImGui::Text("Datasets: %zu", stats.datasets_count);
        ImGui::Text("Peak: %s", stats.FormatBytes(stats.peak_usage).c_str());
        ImGui::Text("Textures: %s", stats.FormatBytes(stats.texture_memory).c_str());
        ImGui::Text("Cache Hit: %.1f%% (%zu evictions)", stats.GetCacheHitRate(), stats.cache_evictions);

        ImGui::Spacing();

        // Disable button if no datasets to evict
        bool can_trim = stats.datasets_count > 0;
        if (!can_trim) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Trim Memory", ImVec2(100, 0))) {
            size_t before = registry.GetTotalMemoryUsage();
            registry.TrimMemory();
            size_t after = registry.GetTotalMemoryUsage();

            if (before > after) {
                spdlog::info("Trim Memory: freed {} bytes ({} -> {})",
                           before - after, stats.FormatBytes(before), stats.FormatBytes(after));
            } else {
                spdlog::info("Trim Memory: nothing evicted (usage {} is under limit {})",
                           stats.FormatBytes(before), stats.FormatBytes(registry.GetMemoryLimit()));
            }
        }

        if (!can_trim) {
            ImGui::EndDisabled();
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!can_trim) {
                ImGui::SetTooltip("No datasets loaded to evict");
            } else if (stats.total_allocated <= registry.GetMemoryLimit()) {
                ImGui::SetTooltip("Memory under limit - nothing to evict.\nTrim only works when over the memory limit.");
            } else {
                ImGui::SetTooltip("Evict least-used datasets to free memory");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Stats", ImVec2(100, 0))) {
            registry.ResetCacheStats();
        }
        ImGui::TreePop();
    }
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

        std::string label = info.name + " (" + cyxwiz::DataRegistry::TypeToString(info.type) + ")";
        if (ImGui::Selectable(label.c_str(), is_selected)) {
            selected_dataset_index_ = static_cast<int>(i);
            // Switch to this dataset
            current_dataset_ = registry.GetDataset(info.name);
            if (current_dataset_.IsValid()) {
                cached_info_ = current_dataset_.GetInfo();
            }
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
    ImGui::Text("Data Preview");
    ImGui::Separator();
    ImGui::Spacing();

    if (!current_dataset_.IsValid()) {
        ImGui::Text("No dataset loaded");
        return;
    }

    size_t dataset_size = current_dataset_.Size();
    if (dataset_size == 0) {
        ImGui::Text("Dataset is empty");
        return;
    }

    // Navigation
    ImGui::Text("Sample %d / %zu", preview_sample_idx_ + 1, dataset_size);

    if (ImGui::Button("<< Prev")) {
        preview_sample_idx_ = (preview_sample_idx_ - 1 + static_cast<int>(dataset_size)) % static_cast<int>(dataset_size);
    }
    ImGui::SameLine();
    if (ImGui::Button("Next >>")) {
        preview_sample_idx_ = (preview_sample_idx_ + 1) % static_cast<int>(dataset_size);
    }
    ImGui::SameLine();
    ImGui::SliderInt("##SampleIdx", &preview_sample_idx_, 0, static_cast<int>(dataset_size) - 1);

    ImGui::Spacing();

    // Get and display the sample
    auto [sample, label] = current_dataset_.GetSample(preview_sample_idx_);

    ImGui::Text("Label: %d", label);
    if (label >= 0 && label < static_cast<int>(class_names_.size())) {
        ImGui::SameLine();
        ImGui::Text("(%s)", class_names_[label].c_str());
    }
    ImGui::Spacing();

    // Render based on dataset type
    if (cached_info_.type == cyxwiz::DatasetType::MNIST ||
        cached_info_.type == cyxwiz::DatasetType::CIFAR10) {
        if (!cached_info_.shape.empty() && cached_info_.shape.size() >= 2) {
            int width = static_cast<int>(cached_info_.shape[0]);
            int height = static_cast<int>(cached_info_.shape[1]);
            int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;

            if (sample.size() == static_cast<size_t>(width * height * channels)) {
                RenderImagePreview(sample.data(), width, height, channels);
            }
        }
    } else {
        // CSV/tabular data - show values
        ImGui::Text("Features:");
        int cols = std::min(8, static_cast<int>(sample.size()));
        for (int i = 0; i < cols; ++i) {
            ImGui::Text("  [%d] = %.4f", i, sample[i]);
        }
        if (sample.size() > static_cast<size_t>(cols)) {
            ImGui::Text("  ... (%zu more)", sample.size() - cols);
        }
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
            if (!success) {
                spdlog::error("Async Kaggle load failed: {}", error);
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
    ImGui::Text("Train Model");
    ImGui::Separator();
    ImGui::Spacing();

    // Check training state from centralized TrainingManager
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    bool training_running = training_mgr.IsTrainingActive();

    // Check if node editor is connected and graph is valid
    bool has_valid_graph = node_editor_ && node_editor_->IsGraphValid();

    // Show warning if no valid graph
    if (!has_valid_graph) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
            "Node graph required for training");
        ImGui::TextWrapped("Create a valid model in the Node Editor with: "
            "DatasetInput -> Model layers -> Loss -> Optimizer");
        ImGui::Spacing();
    }

    // Training hyperparameters
    ImGui::Text("Hyperparameters:");

    // Disable hyperparameters if training is running
    if (training_running) {
        ImGui::BeginDisabled();
    }

    ImGui::SliderInt("Epochs", &train_epochs_, 1, 100);
    ImGui::SliderInt("Batch Size", &train_batch_size_, 1, 256);
    ImGui::InputFloat("Learning Rate", &train_learning_rate_, 0.0001f, 0.01f, "%.6f");

    const char* optimizers[] = {"SGD", "Adam", "AdamW"};
    ImGui::Combo("Optimizer", &selected_optimizer_, optimizers, IM_ARRAYSIZE(optimizers));

    if (training_running) {
        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ============ LOCAL TRAINING ============
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Local Training");
    ImGui::Text("Train on this machine using the Node Graph model");

    bool can_train_local = IsDatasetLoaded() && has_valid_graph && !training_running;

    if (!can_train_local) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Train Locally", ImVec2(-1, 35))) {
        StartLocalTraining();
    }

    if (!can_train_local) {
        ImGui::EndDisabled();
    }

    // Show training progress (both Dataset Panel and Node Editor now use same code path)
    if (training_running) {
        ImGui::Spacing();
        auto metrics = training_mgr.GetCurrentMetrics();
        float progress = static_cast<float>(metrics.current_epoch) / std::max(1, metrics.total_epochs);

        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        ImGui::Text("Epoch %d/%d | Loss: %.4f | Acc: %.2f%%",
            metrics.current_epoch, metrics.total_epochs,
            metrics.train_loss, metrics.train_accuracy * 100.0f);

        if (ImGui::Button("Stop Training", ImVec2(-1, 25))) {
            StopLocalTraining();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ============ P2P TRAINING ============
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "P2P Network Training");
    ImGui::Text("Submit job to decentralized compute network");

    // P2P training doesn't conflict with local training (runs on remote nodes)
    bool can_train_p2p = job_manager_ != nullptr && job_manager_->IsConnected() && IsDatasetLoaded();

    if (!job_manager_) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Not configured");
    } else if (!job_manager_->IsConnected()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Not connected to server");
    }

    if (!can_train_p2p) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Train on Network", ImVec2(-1, 35))) {
        if (SubmitTrainingJob()) {
            spdlog::info("Training job submitted successfully");
        }
    }

    if (!can_train_p2p) {
        ImGui::EndDisabled();
    }

    // Show last submitted job
    if (!last_submitted_job_id_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Last Job ID:");
        ImGui::SameLine();
        ImGui::Text("%s", last_submitted_job_id_.c_str());
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

    // Two column layout: Pipeline on left, Preview on right
    ImGui::Columns(2, "AugmentColumns", true);
    ImGui::SetColumnWidth(0, 350);

    // Left column: Pipeline configuration
    RenderAugmentationPipeline();

    ImGui::NextColumn();

    // Right column: Preview
    RenderAugmentationPreview();

    ImGui::Columns(1);
    ImGui::EndChild();
}

void DatasetPanel::RenderAugmentationPipeline() {
    using namespace cyxwiz::transforms;

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Augmentation Pipeline");
    ImGui::Separator();
    ImGui::Spacing();

    // Preset selection
    ImGui::Text("Preset:");
    ImGui::SameLine();
    const char* presets[] = {"None", "ImageNet Train", "CIFAR-10 Train", "Medical Train", "Custom"};
    if (ImGui::Combo("##AugPreset", &augmentation_preset_, presets, IM_ARRAYSIZE(presets))) {
        // Apply preset
        switch (augmentation_preset_) {
            case 0:  // None
                augmentation_pipeline_.reset();
                break;
            case 1:  // ImageNet
                augmentation_pipeline_ = TransformFactory::createImageNetTrain(224);
                break;
            case 2:  // CIFAR-10
                augmentation_pipeline_ = TransformFactory::createCIFAR10Train();
                break;
            case 3:  // Medical
                augmentation_pipeline_ = TransformFactory::createMedicalTrain(224);
                break;
            case 4:  // Custom - start empty
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
    ImGui::Separator();
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
    ImGui::Separator();
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

    // Display side by side
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

        ImGui::Text("Original (%dx%d)", preview_original_.width, preview_original_.height);
        ImGui::SameLine(preview_size + 30);
        ImGui::Text("Augmented (%dx%d)", preview_augmented_.width, preview_augmented_.height);

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

        // Render original image
        ImGui::BeginChild("OriginalPreview", ImVec2(preview_size + 10, preview_size + 10), true);
        if (preview_texture_original_ != 0) {
            ImGui::Image((ImTextureID)(intptr_t)preview_texture_original_, ImVec2(display_w, display_h));
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(No texture)");
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Render augmented image
        ImGui::BeginChild("AugmentedPreview", ImVec2(preview_size + 10, preview_size + 10), true);
        if (preview_texture_augmented_ != 0) {
            // Augmented might have different size, recalculate
            float aug_aspect = static_cast<float>(preview_augmented_.width) / static_cast<float>(preview_augmented_.height);
            float aug_w = preview_size;
            float aug_h = preview_size;
            if (aug_aspect > 1.0f) {
                aug_h = preview_size / aug_aspect;
            } else {
                aug_w = preview_size * aug_aspect;
            }
            ImGui::Image((ImTextureID)(intptr_t)preview_texture_augmented_, ImVec2(aug_w, aug_h));
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(No texture)");
        }
        ImGui::EndChild();

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
