// Data Input/Output Dialog Implementations
// Part of CyxWiz Studio smart data loading system
// Comprehensive support for files, ML datasets, databases, and cloud storage

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#ifdef CreateDialog
#undef CreateDialog
#endif

#include "node_config_dialog.h"
#include "node_editor.h"
#include "../core/data_registry.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace gui {

// ==================== DataInputDialog ====================

DataInputDialog::DataInputDialog(MLNode* node)
    : NodeConfigDialog("Data Input", node)
{
    if (node_) {
        // Restore from parameters
        if (node_->parameters.count("source_type")) {
            std::string st = node_->parameters["source_type"];
            if (st == "file") source_type_ = SourceType::File;
            else if (st == "ml_dataset") source_type_ = SourceType::MLDataset;
            else if (st == "database") source_type_ = SourceType::Database;
            else if (st == "cloud") source_type_ = SourceType::Cloud;
        }
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
            DetectFileType();
            DetectFileCategory();
        }
        if (node_->parameters.count("folder_path")) {
            strncpy(folder_path_, node_->parameters["folder_path"].c_str(), sizeof(folder_path_) - 1);
        }
        if (node_->parameters.count("dataset_name")) {
            strncpy(dataset_name_, node_->parameters["dataset_name"].c_str(), sizeof(dataset_name_) - 1);
        }
        if (node_->parameters.count("memory_policy")) {
            memory_policy_ = (node_->parameters["memory_policy"] == "disc")
                ? MemoryPolicy::WriteToDisc : MemoryPolicy::CacheInMemory;
        }
    }
}

void DataInputDialog::Apply() {
    if (!node_) return;

    // Source type
    const char* source_types[] = {"file", "ml_dataset", "database", "cloud"};
    node_->parameters["source_type"] = source_types[static_cast<int>(source_type_)];

    // File category
    const char* categories[] = {"tabular", "image", "audio", "video"};
    node_->parameters["file_category"] = categories[static_cast<int>(file_category_)];

    // Common parameters
    node_->parameters["file_path"] = file_path_;
    node_->parameters["folder_path"] = folder_path_;
    node_->parameters["memory_policy"] = (memory_policy_ == MemoryPolicy::WriteToDisc) ? "disc" : "memory";
    node_->parameters["configured"] = "true";

    // Source-specific parameters
    if (source_type_ == SourceType::File) {
        const char* types[] = {"auto", "csv", "tsv", "json", "parquet", "excel", "hdf5", "feather", "arrow", "txt", "arff"};
        node_->parameters["type"] = types[detected_type_];
        node_->parameters["has_header"] = has_header_ ? "true" : "false";
        node_->parameters["delimiter"] = custom_delimiter_;
        node_->parameters["skip_rows"] = std::to_string(skip_rows_);
        node_->parameters["max_rows"] = std::to_string(max_rows_);
        node_->parameters["encoding"] = std::to_string(encoding_idx_);

        if (file_category_ == FileCategory::Image) {
            node_->parameters["target_width"] = std::to_string(target_width_);
            node_->parameters["target_height"] = std::to_string(target_height_);
            node_->parameters["normalize"] = normalize_images_ ? "true" : "false";
            node_->parameters["rgb"] = rgb_mode_ ? "true" : "false";
            node_->parameters["labels_csv"] = labels_csv_;
        } else if (file_category_ == FileCategory::Audio) {
            node_->parameters["sample_rate"] = std::to_string(sample_rate_);
            node_->parameters["mono"] = mono_ ? "true" : "false";
        }

        // Excel specific
        if (detected_type_ == 5) {
            node_->parameters["sheet_idx"] = std::to_string(sheet_idx_);
            node_->parameters["sheet_range"] = sheet_range_;
        }
        // HDF5 specific
        if (detected_type_ == 6) {
            node_->parameters["hdf5_dataset"] = hdf5_dataset_;
        }
        // JSON specific
        if (detected_type_ == 3) {
            node_->parameters["json_lines"] = json_lines_ ? "true" : "false";
            node_->parameters["json_path"] = json_path_;
        }

        // Auto-generate description
        if (strlen(file_path_) > 0) {
            fs::path p(file_path_);
            node_->description = "Reading " + p.filename().string();
        } else if (strlen(folder_path_) > 0) {
            fs::path p(folder_path_);
            node_->description = "Loading from " + p.filename().string();
        }
    }
    else if (source_type_ == SourceType::MLDataset) {
        const char* ml_types[] = {"mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet", "image_folder", "huggingface", "kaggle", "custom"};
        node_->parameters["ml_dataset_type"] = ml_types[static_cast<int>(ml_dataset_type_)];
        node_->parameters["dataset_name"] = dataset_name_;
        node_->parameters["dataset_subset"] = dataset_subset_;
        node_->parameters["cache_dir"] = cache_dir_;

        if (ml_dataset_type_ == MLDatasetType::HuggingFace) {
            node_->parameters["hf_token"] = hf_token_;
        } else if (ml_dataset_type_ == MLDatasetType::Kaggle) {
            node_->parameters["kaggle_slug"] = kaggle_slug_;
        }

        node_->description = std::string("Loading ") + dataset_name_;
    }
    else if (source_type_ == SourceType::Database) {
        const char* db_types[] = {"sqlite", "postgresql", "mysql", "duckdb"};
        node_->parameters["database_type"] = db_types[static_cast<int>(database_type_)];
        node_->parameters["db_host"] = db_host_;
        node_->parameters["db_port"] = std::to_string(db_port_);
        node_->parameters["db_name"] = db_name_;
        node_->parameters["db_user"] = db_user_;
        node_->parameters["db_file"] = db_file_;
        node_->parameters["sql_query"] = sql_query_;

        node_->description = "Query from " + std::string(db_name_);
    }
    else if (source_type_ == SourceType::Cloud) {
        node_->parameters["cloud_bucket"] = cloud_bucket_;
        node_->parameters["cloud_path"] = cloud_path_;
        node_->parameters["cloud_credentials"] = cloud_credentials_;

        node_->description = "Loading from " + std::string(cloud_bucket_);
    }

    has_changes_ = false;
    spdlog::info("DataInputDialog: Applied settings");
}

void DataInputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    preview_loaded_ = false;
    has_changes_ = false;
}

void DataInputDialog::RenderContent() {
    // KNIME-style tab bar at TOP based on source type
    if (source_type_ == SourceType::File) {
        if (ImGui::BeginTabBar("DataInputTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Settings")) {
                RenderFileSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Transformation")) {
                RenderTransformationTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Limit Rows")) {
                RenderLimitRowsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Encoding")) {
                RenderEncodingTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Memory")) {
                RenderMemoryTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::MLDataset) {
        if (ImGui::BeginTabBar("MLDatasetTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Dataset")) {
                RenderMLDatasetSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Options")) {
                RenderMLDatasetOptions();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::Database) {
        if (ImGui::BeginTabBar("DatabaseTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Connection")) {
                RenderDatabaseSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Query")) {
                RenderSQLQuery();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::Cloud) {
        RenderCloudSource();
    }

    // Preview section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderPreviewPanel();

    // Source type selector at BOTTOM
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderSourceSelector();
}

void DataInputDialog::RenderSourceSelector() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATA SOURCE");
    ImGui::Spacing();

    // Source type radio buttons in a row
    int source_idx = static_cast<int>(source_type_);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(20, 4));
    if (ImGui::RadioButton("File", &source_idx, 0)) {
        source_type_ = SourceType::File;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("ML Dataset", &source_idx, 1)) {
        source_type_ = SourceType::MLDataset;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Database", &source_idx, 2)) {
        source_type_ = SourceType::Database;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Cloud", &source_idx, 3)) {
        source_type_ = SourceType::Cloud;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::PopStyleVar();
}

void DataInputDialog::RenderFileSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    ImVec4 info_color = style.Colors[ImGuiCol_TextDisabled];

    // File category selector
    ImGui::TextColored(accent, "File Type");
    ImGui::Spacing();

    int cat_idx = static_cast<int>(file_category_);
    if (ImGui::RadioButton("Tabular", &cat_idx, 0)) {
        file_category_ = FileCategory::Tabular;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Image", &cat_idx, 1)) {
        file_category_ = FileCategory::Image;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Audio", &cat_idx, 2)) {
        file_category_ = FileCategory::Audio;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Video", &cat_idx, 3)) {
        file_category_ = FileCategory::Video;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // File/Folder path section
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.FrameRounding);
    if (ImGui::BeginChild("FileSelection", ImVec2(0, 100), true)) {
        ImGui::TextColored(accent, "Input File");
        ImGui::Separator();
        ImGui::Spacing();

        // File path input
        ImGui::Text("Path:");
        ImGui::SameLine(60);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##filepath", file_path_, sizeof(file_path_))) {
            DetectFileType();
            DetectFileCategory();
            has_changes_ = true;
            preview_loaded_ = false;
        }
        ImGui::PopStyleVar();
        ImGui::SameLine();
        if (ImGui::Button("Browse...", ImVec2(80, 0))) {
            BrowseFile();
        }

        // File info
        if (strlen(file_path_) > 0) {
            ImGui::TextColored(info_color, "Type: %s", GetFileTypeName());
            ImGui::SameLine(150);
            if (file_size_ > 0) {
                if (file_size_ >= 1024 * 1024)
                    ImGui::TextColored(info_color, "Size: %.1f MB", file_size_ / (1024.0f * 1024.0f));
                else if (file_size_ >= 1024)
                    ImGui::TextColored(info_color, "Size: %.1f KB", file_size_ / 1024.0f);
                else
                    ImGui::TextColored(info_color, "Size: %zu B", file_size_);
            }
        } else {
            ImGui::TextDisabled("No file selected - click Browse to select a data file");
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::Spacing();

    // Category-specific options
    switch (file_category_) {
        case FileCategory::Tabular: RenderTabularOptions(); break;
        case FileCategory::Image:   RenderImageOptions(); break;
        case FileCategory::Audio:   RenderAudioOptions(); break;
        case FileCategory::Video:   RenderVideoOptions(); break;
    }
}

void DataInputDialog::RenderTabularOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    if (ImGui::CollapsingHeader("Format Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Format auto-detect or manual selection
        ImGui::Text("Format:");
        ImGui::SameLine(100);
        const char* formats[] = {"Auto", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5", "Feather", "Arrow", "TXT", "ARFF"};
        ImGui::SetNextItemWidth(120);
        if (ImGui::Combo("##format", &detected_type_, formats, 11)) {
            has_changes_ = true;
            preview_loaded_ = false;
        }

        // CSV/TSV/TXT/ARFF specific options (delimiter-based)
        if (detected_type_ <= 2 || detected_type_ == 9 || detected_type_ == 10) {
            ImGui::Spacing();

            ImGui::Text("Delimiter:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(50);
            if (ImGui::InputText("##delim", custom_delimiter_, sizeof(custom_delimiter_))) {
                has_changes_ = true;
                preview_loaded_ = false;
            }

            ImGui::SameLine(180);
            if (ImGui::Checkbox("Has header", &has_header_)) {
                has_changes_ = true;
                preview_loaded_ = false;
            }

            ImGui::Text("Quote char:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(30);
            ImGui::InputText("##quote", quote_char_, sizeof(quote_char_));
        }
        // Excel specific
        else if (detected_type_ == 5) {
            ImGui::Text("Sheet index:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(60);
            ImGui::InputInt("##sheet", &sheet_idx_);

            ImGui::Text("Cell range:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(100);
            ImGui::InputText("##range", sheet_range_, sizeof(sheet_range_));
            ImGui::SameLine();
            ImGui::TextDisabled("(e.g., A1:D100)");
        }
        // HDF5 specific
        else if (detected_type_ == 6) {
            ImGui::Text("Dataset key:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(150);
            ImGui::InputText("##hdf5key", hdf5_dataset_, sizeof(hdf5_dataset_));
        }
        // JSON specific
        else if (detected_type_ == 3) {
            if (ImGui::Checkbox("JSON Lines format", &json_lines_)) {
                has_changes_ = true;
            }
            ImGui::Text("JSON Path:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(150);
            ImGui::InputText("##jsonpath", json_path_, sizeof(json_path_));
            ImGui::SameLine();
            ImGui::TextDisabled("($.data.rows)");
        }
    }

}

void DataInputDialog::RenderImageOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Image Loading Options");
    ImGui::Spacing();

    // Option to use folder or file list
    static int image_mode = 0;
    ImGui::RadioButton("Image folder", &image_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("File list (CSV)", &image_mode, 1);

    ImGui::Spacing();

    if (image_mode == 0) {
        // Image folder mode
        ImGui::Text("Folder:");
        ImGui::SameLine(70);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        if (ImGui::InputText("##imgfolder", folder_path_, sizeof(folder_path_))) {
            has_changes_ = true;
            preview_loaded_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("...##imgbrowse", ImVec2(70, 0))) {
            BrowseFolder();
        }
        ImGui::TextDisabled("Subfolders become class labels");
    } else {
        // CSV with image paths
        ImGui::Text("Labels CSV:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        ImGui::InputText("##labelscsv", labels_csv_, sizeof(labels_csv_));
        ImGui::SameLine();
        if (ImGui::Button("...##csvbrowse", ImVec2(70, 0))) {
            std::string path;
            if (FileSelector("Select Labels CSV:", path, "CSV Files\0*.csv\0All Files\0*.*\0")) {
                strncpy(labels_csv_, path.c_str(), sizeof(labels_csv_) - 1);
                has_changes_ = true;
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Resize options
    if (ImGui::CollapsingHeader("Resize", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Target size:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##width", &target_width_);
        ImGui::SameLine();
        ImGui::Text("x");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##height", &target_height_);

        // Preset sizes
        ImGui::Text("Presets:");
        ImGui::SameLine(100);
        if (ImGui::Button("224x224")) { target_width_ = target_height_ = 224; has_changes_ = true; }
        ImGui::SameLine();
        if (ImGui::Button("256x256")) { target_width_ = target_height_ = 256; has_changes_ = true; }
        ImGui::SameLine();
        if (ImGui::Button("512x512")) { target_width_ = target_height_ = 512; has_changes_ = true; }
    }

    if (ImGui::CollapsingHeader("Normalization")) {
        if (ImGui::Checkbox("Normalize to [0, 1]", &normalize_images_)) {
            has_changes_ = true;
        }
        if (ImGui::Checkbox("Convert to RGB", &rgb_mode_)) {
            has_changes_ = true;
        }
    }
}

void DataInputDialog::RenderAudioOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Audio Loading Options");
    ImGui::Spacing();

    ImGui::Text("Sample rate:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##samplerate", &sample_rate_);
    ImGui::SameLine();
    ImGui::TextDisabled("Hz");

    if (ImGui::Checkbox("Convert to mono", &mono_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: WAV, MP3, FLAC, OGG");
}

void DataInputDialog::RenderVideoOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Video Loading Options");
    ImGui::Spacing();

    static int frame_extraction = 0;
    ImGui::RadioButton("Extract all frames", &frame_extraction, 0);
    ImGui::RadioButton("Extract N frames", &frame_extraction, 1);
    ImGui::RadioButton("Sample at FPS", &frame_extraction, 2);

    if (frame_extraction == 1) {
        static int n_frames = 16;
        ImGui::Text("Frames:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##nframes", &n_frames);
    } else if (frame_extraction == 2) {
        static float fps = 1.0f;
        ImGui::Text("FPS:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##fps", &fps, 0.1f, 1.0f, "%.1f");
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: MP4, AVI, MOV, WebM");
}

void DataInputDialog::RenderTransformationTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Column Transformations");
    ImGui::Separator();
    ImGui::Spacing();

    // Column selection
    if (ImGui::Checkbox("Include all columns", &select_all_columns_)) {
        has_changes_ = true;
        for (size_t i = 0; i < selected_columns_.size(); i++) {
            selected_columns_[i] = select_all_columns_;
        }
    }

    if (!select_all_columns_ && !available_columns_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Select columns to include:");
        ImGui::BeginChild("ColumnList", ImVec2(0, 200), true);
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); i++) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
                has_changes_ = true;
            }
        }
        ImGui::EndChild();
    }

    if (available_columns_.empty() && strlen(file_path_) > 0) {
        ImGui::TextDisabled("Load preview to see column list");
    }
}

void DataInputDialog::RenderLimitRowsTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Row Filtering Options");
    ImGui::Separator();
    ImGui::Spacing();

    // Skip rows
    ImGui::Text("Skip first N rows:");
    ImGui::SameLine(180);
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##skip", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(excluding header)");

    ImGui::Spacing();

    // Limit rows
    static bool limit_enabled = false;
    ImGui::Checkbox("Limit number of rows", &limit_enabled);
    if (limit_enabled) {
        ImGui::SameLine(180);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##maxrows", &max_rows_)) {
            if (max_rows_ < 1) max_rows_ = 1;
            has_changes_ = true;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // WHERE clause filter
    ImGui::Text("Filter rows (SQL WHERE clause):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##where", where_clause_, sizeof(where_clause_), ImVec2(0, 60))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("Example: age > 18 AND status = 'active'");
}

void DataInputDialog::RenderEncodingTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Character Encoding");
    ImGui::Separator();
    ImGui::Spacing();

    const char* encodings[] = {"Auto-detect", "UTF-8", "UTF-16", "ASCII", "ISO-8859-1", "Windows-1252", "UTF-8 with BOM"};
    ImGui::Text("File encoding:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##encoding", &encoding_idx_, encodings, 7)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Auto-detect will scan the first bytes of the file to determine encoding.");
}

void DataInputDialog::RenderMemoryTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Memory Management");
    ImGui::Separator();
    ImGui::Spacing();

    // Memory policy
    int policy = (memory_policy_ == MemoryPolicy::CacheInMemory) ? 0 : 1;
    ImGui::Text("Memory policy:");
    ImGui::Spacing();
    if (ImGui::RadioButton("Cache in memory (fast access)", &policy, 0)) {
        memory_policy_ = MemoryPolicy::CacheInMemory;
        has_changes_ = true;
    }
    ImGui::TextDisabled("   Keep data cached in RAM for fast repeated access");
    ImGui::Spacing();
    if (ImGui::RadioButton("Write to temp file (low memory)", &policy, 1)) {
        memory_policy_ = MemoryPolicy::WriteToDisc;
        has_changes_ = true;
    }
    ImGui::TextDisabled("   Store data in temporary file, load chunks on demand");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Streaming options
    ImGui::Text("Streaming options (when using disk mode):");
    ImGui::Spacing();

    if (ImGui::Checkbox("Auto chunk size", &auto_chunk_size_)) {
        has_changes_ = true;
    }

    if (!auto_chunk_size_) {
        ImGui::Text("Chunk size:");
        ImGui::SameLine(150);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##chunk", &chunk_size_kb_)) {
            if (chunk_size_kb_ < 64) chunk_size_kb_ = 64;
            if (chunk_size_kb_ > 65536) chunk_size_kb_ = 65536;
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("KB");
    }

    ImGui::Text("LRU cache chunks:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##lru", &lru_chunks_)) {
        if (lru_chunks_ < 1) lru_chunks_ = 1;
        if (lru_chunks_ > 64) lru_chunks_ = 64;
        has_changes_ = true;
    }

    if (ImGui::Checkbox("Enable prefetch", &prefetch_enabled_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(preload next chunk while processing current)");

    // RAM estimate
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    UpdateRAMEstimate();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Estimated RAM usage: %.1f MB", estimated_ram_mb_);
}

void DataInputDialog::RenderMLDatasetOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Dataset Options");
    ImGui::Separator();
    ImGui::Spacing();

    // Split options
    ImGui::Text("Data Split:");
    ImGui::Spacing();

    static int split_mode = 0;
    ImGui::RadioButton("Use default split", &split_mode, 0);
    ImGui::RadioButton("Custom split", &split_mode, 1);

    if (split_mode == 1) {
        static float train_ratio = 0.8f;
        static float val_ratio = 0.1f;
        ImGui::Text("Train:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##train", &train_ratio, 0.1f, 0.9f, "%.0f%%");
        ImGui::Text("Val:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##val", &val_ratio, 0.0f, 0.5f, "%.0f%%");
        ImGui::Text("Test:");
        ImGui::SameLine(80);
        float test_ratio = 1.0f - train_ratio - val_ratio;
        ImGui::Text("%.0f%%", test_ratio * 100);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Preprocessing
    ImGui::TextColored(accent, "Preprocessing");
    ImGui::Spacing();

    if (ImGui::Checkbox("Normalize to [0, 1]", &normalize_images_)) {
        has_changes_ = true;
    }

    static bool shuffle = true;
    if (ImGui::Checkbox("Shuffle on load", &shuffle)) {
        has_changes_ = true;
    }
}

void DataInputDialog::RenderMLDatasetSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "ML DATASET");
    ImGui::Spacing();

    // Dataset source selector
    int ml_idx = static_cast<int>(ml_dataset_type_);
    const char* ml_items[] = {
        "MNIST", "CIFAR-10", "CIFAR-100", "Fashion-MNIST",
        "ImageNet", "Image Folder", "HuggingFace", "Kaggle", "Custom"
    };

    ImGui::Text("Dataset:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##mldataset", &ml_idx, ml_items, 9)) {
        ml_dataset_type_ = static_cast<MLDatasetType>(ml_idx);
        has_changes_ = true;
        preview_loaded_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Dataset-specific configuration
    switch (ml_dataset_type_) {
        case MLDatasetType::MNIST:
        case MLDatasetType::CIFAR10:
        case MLDatasetType::CIFAR100:
        case MLDatasetType::FashionMNIST:
            RenderBuiltinDatasets();
            break;
        case MLDatasetType::ImageNet:
        case MLDatasetType::ImageFolder:
            RenderImageFolderPicker();
            break;
        case MLDatasetType::HuggingFace:
            RenderHuggingFaceConfig();
            break;
        case MLDatasetType::Kaggle:
            RenderKaggleConfig();
            break;
        case MLDatasetType::Custom:
            // Custom uses file source
            ImGui::TextDisabled("Use File source for custom datasets");
            break;
    }
}

void DataInputDialog::RenderBuiltinDatasets() {
    const char* dataset_names[] = {"mnist", "cifar10", "cifar100", "fashion_mnist"};
    const char* descriptions[] = {
        "Handwritten digits (28x28, 10 classes, 60K train)",
        "Color images (32x32x3, 10 classes, 50K train)",
        "Color images (32x32x3, 100 classes, 50K train)",
        "Fashion items (28x28, 10 classes, 60K train)"
    };

    int idx = static_cast<int>(ml_dataset_type_);
    if (idx < 4) {
        strncpy(dataset_name_, dataset_names[idx], sizeof(dataset_name_) - 1);
        ImGui::TextWrapped("%s", descriptions[idx]);
    }

    ImGui::Spacing();

    // Cache directory
    ImGui::Text("Cache dir:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    ImGui::InputText("##cachedir", cache_dir_, sizeof(cache_dir_));
    ImGui::SameLine();
    if (ImGui::Button("...##cachebrowse", ImVec2(80, 0))) {
        BrowseFolder();
        strncpy(cache_dir_, folder_path_, sizeof(cache_dir_) - 1);
    }

    if (strlen(cache_dir_) == 0) {
        ImGui::TextDisabled("Default: ~/.cyxwiz/datasets/");
    }

    ImGui::Spacing();

    // Download button
    if (is_downloading_) {
        ImGui::ProgressBar(download_progress_, ImVec2(-1, 0), "Downloading...");
    } else {
        if (ImGui::Button("Download Dataset", ImVec2(-1, 30))) {
            DownloadMLDataset();
        }
    }

    if (!status_message_.empty()) {
        ImVec4 color = status_message_.find("Error") != std::string::npos
            ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
            : ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
        ImGui::TextColored(color, "%s", status_message_.c_str());
    }
}

void DataInputDialog::RenderImageFolderPicker() {
    ImGui::Text("Root folder:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##imgroot", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("...##rootbrowse", ImVec2(90, 0))) {
        BrowseFolder();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Structure: root/class_name/image.jpg");

    // Show detected classes
    if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        std::vector<std::string> classes;
        for (const auto& entry : fs::directory_iterator(folder_path_)) {
            if (entry.is_directory()) {
                classes.push_back(entry.path().filename().string());
            }
        }

        if (!classes.empty()) {
            ImGui::Text("Detected %zu classes:", classes.size());
            ImGui::BeginChild("ClassList", ImVec2(0, 100), true);
            for (const auto& c : classes) {
                ImGui::BulletText("%s", c.c_str());
            }
            ImGui::EndChild();
        }
    }
}

void DataInputDialog::RenderHuggingFaceConfig() {
    ImGui::Text("Dataset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##hfname", dataset_name_, sizeof(dataset_name_))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("e.g., mnist, imdb, squad, coco");

    ImGui::Spacing();

    ImGui::Text("Subset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##hfsubset", dataset_subset_, sizeof(dataset_subset_));

    ImGui::Text("Split:");
    ImGui::SameLine(90);
    static int split_idx = 0;
    const char* splits[] = {"train", "validation", "test"};
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##hfsplit", &split_idx, splits, 3);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Auth token:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    ImGui::InputText("##hftoken", hf_token_, sizeof(hf_token_), ImGuiInputTextFlags_Password);
    ImGui::TextDisabled("Required for private/gated datasets");

    ImGui::Spacing();

    if (ImGui::Button("Load from HuggingFace", ImVec2(-1, 30))) {
        DownloadMLDataset();
    }
}

void DataInputDialog::RenderKaggleConfig() {
    ImGui::Text("Dataset slug:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##kaggleslug", kaggle_slug_, sizeof(kaggle_slug_));
    ImGui::TextDisabled("e.g., zalando-research/fashionmnist");

    ImGui::Spacing();

    ImGui::Text("or Competition:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    static char competition[128] = "";
    ImGui::InputText("##kagglecomp", competition, sizeof(competition));
    ImGui::TextDisabled("e.g., titanic, digit-recognizer");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("API credentials from ~/.kaggle/kaggle.json");

    if (ImGui::Button("Download from Kaggle", ImVec2(-1, 30))) {
        DownloadMLDataset();
    }
}

void DataInputDialog::RenderDatabaseSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATABASE CONNECTION");
    ImGui::Spacing();

    // Database type
    int db_idx = static_cast<int>(database_type_);
    const char* db_types[] = {"SQLite", "PostgreSQL", "MySQL", "DuckDB"};

    ImGui::Text("Type:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##dbtype", &db_idx, db_types, 4)) {
        database_type_ = static_cast<DatabaseType>(db_idx);
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderDatabaseConnection();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderSQLQuery();
}

void DataInputDialog::RenderDatabaseConnection() {
    if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
        // File-based database
        ImGui::Text("Database file:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
        ImGui::InputText("##dbfile", db_file_, sizeof(db_file_));
        ImGui::SameLine();
        if (ImGui::Button("...##dbbrowse", ImVec2(100, 0))) {
            std::string path;
            const char* filter = (database_type_ == DatabaseType::SQLite)
                ? "SQLite\0*.db;*.sqlite;*.sqlite3\0All Files\0*.*\0"
                : "DuckDB\0*.duckdb;*.db\0All Files\0*.*\0";
            if (FileSelector("Select Database:", path, filter)) {
                strncpy(db_file_, path.c_str(), sizeof(db_file_) - 1);
                has_changes_ = true;
            }
        }
    } else {
        // Network database
        ImGui::Text("Host:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbhost", db_host_, sizeof(db_host_));
        ImGui::SameLine();
        ImGui::Text("Port:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##dbport", &db_port_);

        ImGui::Text("Database:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbname", db_name_, sizeof(db_name_));

        ImGui::Text("User:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(120);
        ImGui::InputText("##dbuser", db_user_, sizeof(db_user_));
        ImGui::SameLine();
        ImGui::Text("Password:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##dbpass", db_password_, sizeof(db_password_), ImGuiInputTextFlags_Password);
    }

    ImGui::Spacing();

    // Connect button
    ImVec4 btn_color = db_connected_
        ? ImVec4(0.2f, 0.7f, 0.2f, 1.0f)
        : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, btn_color);
    if (ImGui::Button(db_connected_ ? "Connected" : "Test Connection", ImVec2(150, 0))) {
        TestDatabaseConnection();
    }
    ImGui::PopStyleColor();
}

void DataInputDialog::RenderSQLQuery() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "SQL Query");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##sqlquery", sql_query_, sizeof(sql_query_), ImVec2(0, 100))) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    // Quick templates
    if (ImGui::Button("SELECT *")) {
        strncpy(sql_query_, "SELECT * FROM table_name LIMIT 1000", sizeof(sql_query_) - 1);
    }
    ImGui::SameLine();
    if (ImGui::Button("SHOW TABLES")) {
        if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
            strncpy(sql_query_, "SELECT name FROM sqlite_master WHERE type='table'", sizeof(sql_query_) - 1);
        } else if (database_type_ == DatabaseType::PostgreSQL) {
            strncpy(sql_query_, "SELECT tablename FROM pg_tables WHERE schemaname='public'", sizeof(sql_query_) - 1);
        } else {
            strncpy(sql_query_, "SHOW TABLES", sizeof(sql_query_) - 1);
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Execute Query", ImVec2(-1, 25))) {
        LoadPreview();
    }
}

void DataInputDialog::RenderCloudSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "CLOUD STORAGE");
    ImGui::Spacing();

    // Cloud provider
    static int cloud_provider = 0;
    ImGui::RadioButton("AWS S3", &cloud_provider, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Google Cloud", &cloud_provider, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Azure Blob", &cloud_provider, 2);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Bucket:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("##bucket", cloud_bucket_, sizeof(cloud_bucket_));

    ImGui::Text("Path:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudpath", cloud_path_, sizeof(cloud_path_));

    ImGui::Spacing();

    ImGui::Text("Credentials:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudcreds", cloud_credentials_, sizeof(cloud_credentials_));
    ImGui::SameLine();
    if (ImGui::Button("...##credbrowse", ImVec2(100, 0))) {
        std::string path;
        if (FileSelector("Select Credentials:", path, "JSON\0*.json\0All Files\0*.*\0")) {
            strncpy(cloud_credentials_, path.c_str(), sizeof(cloud_credentials_) - 1);
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("AWS: ~/.aws/credentials or JSON key");
    ImGui::TextDisabled("GCS: service-account.json");
    ImGui::TextDisabled("Azure: connection string or SAS token");

    ImGui::Spacing();

    if (ImGui::Button("Connect & List Files", ImVec2(-1, 30))) {
        LoadPreview();
    }
}

void DataInputDialog::RenderPreviewPanel() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "PREVIEW");
    ImGui::Separator();
    ImGui::Spacing();

    if (!preview_loaded_) {
        if (ImGui::Button("Load Preview", ImVec2(-1, 30))) {
            LoadPreview();
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Configure source and click\nLoad Preview to see data");
    } else if (!preview_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", preview_error_.c_str());
    } else {
        // Render based on category
        if (source_type_ == SourceType::File) {
            switch (file_category_) {
                case FileCategory::Tabular: RenderTabularPreview(); break;
                case FileCategory::Image:   RenderImagePreview(); break;
                case FileCategory::Audio:   RenderAudioPreview(); break;
                default: RenderTabularPreview(); break;
            }
        } else {
            // Default to tabular preview for other sources
            RenderTabularPreview();
        }
    }

    // RAM estimate
    if (preview_loaded_ && estimated_ram_mb_ > 0) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Est. RAM: %.1f MB", estimated_ram_mb_);
    }
}

void DataInputDialog::RenderTabularPreview() {
    if (preview_columns_.empty()) {
        ImGui::TextDisabled("No data to preview");
        return;
    }

    // Column info
    ImGui::Text("%zu columns, %zu rows", preview_columns_.size(), preview_data_.size());
    ImGui::Spacing();

    // Table
    if (ImGui::BeginTable("Preview", static_cast<int>(preview_columns_.size()),
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        // Headers
        for (const auto& col : preview_columns_) {
            ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 80.0f);
        }
        ImGui::TableHeadersRow();

        // Data rows (limit to 20)
        int row_idx = 0;
        for (const auto& row : preview_data_) {
            if (row_idx >= 20) break;
            ImGui::TableNextRow();
            int col_idx = 0;
            for (const auto& cell : row) {
                if (col_idx < static_cast<int>(preview_columns_.size())) {
                    ImGui::TableSetColumnIndex(col_idx);
                    ImGui::TextUnformatted(cell.c_str());
                }
                col_idx++;
            }
            row_idx++;
        }
        ImGui::EndTable();
    }
}

void DataInputDialog::RenderImagePreview() {
    if (preview_image_textures_.empty()) {
        ImGui::TextDisabled("No images to preview");
        ImGui::TextDisabled("(Image preview coming soon)");
        return;
    }

    ImGui::Text("%zu images", preview_image_textures_.size());
    ImGui::Spacing();

    // Grid of thumbnails
    float thumb_size = 64.0f;
    float avail_width = ImGui::GetContentRegionAvail().x;
    int cols = std::max(1, static_cast<int>(avail_width / (thumb_size + 8)));

    int idx = 0;
    for (ImTextureID tex_id : preview_image_textures_) {
        if (idx > 0 && (idx % cols) != 0) {
            ImGui::SameLine();
        }

        ImGui::BeginGroup();
        ImGui::Image(tex_id, ImVec2(thumb_size, thumb_size));
        if (idx < static_cast<int>(preview_image_labels_.size())) {
            ImGui::TextDisabled("%s", preview_image_labels_[idx].c_str());
        }
        ImGui::EndGroup();

        idx++;
        if (idx >= 12) break;  // Limit preview images
    }
}

void DataInputDialog::RenderAudioPreview() {
    ImGui::TextDisabled("Audio preview not yet implemented");
    ImGui::TextDisabled("Waveform visualization coming soon");
}

// ==================== Helper Functions ====================

const char* DataInputDialog::GetFileTypeName() const {
    const char* names[] = {"Auto", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5", "Feather", "Arrow", "TXT", "ARFF"};
    if (detected_type_ >= 0 && detected_type_ < 11) {
        return names[detected_type_];
    }
    return "Unknown";
}

void DataInputDialog::DetectFileType() {
    std::string path(file_path_);
    if (path.empty()) {
        detected_type_ = 0;
        return;
    }

    // Get file size
    try {
        file_size_ = static_cast<size_t>(fs::file_size(path));
    } catch (...) {
        file_size_ = 0;
    }

    // Detect by extension
    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == "csv") detected_type_ = 1;
    else if (ext == "tsv" || ext == "tab") detected_type_ = 2;
    else if (ext == "json" || ext == "jsonl") detected_type_ = 3;
    else if (ext == "parquet" || ext == "pq") detected_type_ = 4;
    else if (ext == "xlsx" || ext == "xls") detected_type_ = 5;
    else if (ext == "h5" || ext == "hdf5" || ext == "hdf") detected_type_ = 6;
    else if (ext == "feather" || ext == "fea") detected_type_ = 7;
    else if (ext == "arrow" || ext == "ipc") detected_type_ = 8;
    else if (ext == "txt") detected_type_ = 9;
    else if (ext == "arff") detected_type_ = 10;
    else detected_type_ = 0;
}

void DataInputDialog::DetectFileCategory() {
    std::string path(file_path_);
    if (path.empty()) return;

    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    // Image extensions
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "gif" || ext == "tiff" || ext == "webp") {
        file_category_ = FileCategory::Image;
    }
    // Audio extensions
    else if (ext == "wav" || ext == "mp3" || ext == "flac" || ext == "ogg" ||
             ext == "m4a" || ext == "aac") {
        file_category_ = FileCategory::Audio;
    }
    // Video extensions
    else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
             ext == "webm" || ext == "wmv") {
        file_category_ = FileCategory::Video;
    }
    // Default to tabular
    else {
        file_category_ = FileCategory::Tabular;
    }
}

void DataInputDialog::LoadPreview() {
    preview_error_.clear();
    preview_columns_.clear();
    preview_data_.clear();

    if (source_type_ == SourceType::File && file_category_ == FileCategory::Tabular) {
        LoadColumnList();
    }
    // TODO: Add image, audio, database preview loading

    preview_loaded_ = true;
    UpdateRAMEstimate();
}

void DataInputDialog::LoadColumnList() {
    std::string path(file_path_);
    if (path.empty()) return;

    std::ifstream file(path);
    if (!file.is_open()) {
        preview_error_ = "Cannot open file";
        return;
    }

    char delim = custom_delimiter_[0];
    if (delim == '\0') delim = ',';
    if (detected_type_ == 2) delim = '\t';

    std::string line;
    int line_count = 0;

    while (std::getline(file, line) && line_count < 25) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delim)) {
            // Trim whitespace and quotes
            size_t start = cell.find_first_not_of(" \t\r\n\"");
            size_t end = cell.find_last_not_of(" \t\r\n\"");
            if (start != std::string::npos && end != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            cells.push_back(cell);
        }

        if (line_count == 0 && has_header_) {
            preview_columns_ = cells;
            available_columns_ = cells;
            selected_columns_.resize(cells.size(), true);
        } else {
            preview_data_.push_back(cells);
            if (!has_header_ && line_count == 0) {
                for (size_t i = 0; i < cells.size(); i++) {
                    preview_columns_.push_back("Column" + std::to_string(i + 1));
                    available_columns_.push_back("Column" + std::to_string(i + 1));
                }
                selected_columns_.resize(cells.size(), true);
            }
        }
        line_count++;
    }
}

void DataInputDialog::UpdateRAMEstimate() {
    if (file_size_ <= 0) {
        estimated_ram_mb_ = 0;
        return;
    }
    double multiplier = (memory_policy_ == MemoryPolicy::CacheInMemory) ? 2.0 : 0.1;
    estimated_ram_mb_ = static_cast<float>((file_size_ / (1024.0 * 1024.0)) * multiplier);
}

void DataInputDialog::BrowseFile() {
#ifdef _WIN32
    const char* filter = nullptr;

    switch (file_category_) {
        case FileCategory::Tabular:
            filter = "All Data Files\0*.csv;*.tsv;*.xlsx;*.xls;*.json;*.parquet;*.h5;*.hdf5;*.feather;*.arrow;*.txt;*.arff\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0Excel\0*.xlsx;*.xls\0JSON\0*.json\0Parquet\0*.parquet\0"
                     "HDF5\0*.h5;*.hdf5\0Feather\0*.feather\0Arrow\0*.arrow;*.ipc\0Text\0*.txt\0ARFF\0*.arff\0All Files\0*.*\0";
            break;
        case FileCategory::Image:
            filter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp\0All Files\0*.*\0";
            break;
        case FileCategory::Audio:
            filter = "Audio Files\0*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac\0All Files\0*.*\0";
            break;
        case FileCategory::Video:
            filter = "Video Files\0*.mp4;*.avi;*.mov;*.mkv;*.webm;*.wmv\0All Files\0*.*\0";
            break;
    }

    OPENFILENAMEA ofn = {};
    char file[512] = {};
    strncpy(file, file_path_, sizeof(file) - 1);

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof(file);
    ofn.lpstrTitle = "Select Data File";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        strncpy(file_path_, file, sizeof(file_path_) - 1);
        DetectFileType();
        DetectFileCategory();
        has_changes_ = true;
        preview_loaded_ = false;
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

void DataInputDialog::BrowseFolder() {
#ifdef _WIN32
    BROWSEINFOA bi = {};
    bi.lpszTitle = "Select Folder";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderA(&bi);
    if (pidl) {
        char path[MAX_PATH];
        if (SHGetPathFromIDListA(pidl, path)) {
            strncpy(folder_path_, path, sizeof(folder_path_) - 1);
            has_changes_ = true;
            preview_loaded_ = false;
        }
        CoTaskMemFree(pidl);
    }
#else
    // TODO: Implement for other platforms
    spdlog::warn("Folder browser not implemented for this platform");
#endif
}

void DataInputDialog::TestDatabaseConnection() {
    // TODO: Implement actual database connection test
    db_connected_ = true;
    status_message_ = "Connected successfully";
    spdlog::info("Database connection test: simulated success");
}

void DataInputDialog::DownloadMLDataset() {
    is_downloading_ = true;
    download_progress_ = 0.0f;

    // TODO: Implement actual download
    // For now, simulate progress
    is_downloading_ = false;
    download_progress_ = 1.0f;
    status_message_ = "Dataset ready (simulated)";

    spdlog::info("ML dataset download: {}", dataset_name_);
}

// ==================== DataOutputDialog ====================

DataOutputDialog::DataOutputDialog(MLNode* node)
    : NodeConfigDialog("Data Output", node)
{
    if (node_) {
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
        }
        if (node_->parameters.count("file_type")) {
            std::string type = node_->parameters["file_type"];
            if (type == "csv") output_type_ = 0;
            else if (type == "tsv") output_type_ = 1;
            else if (type == "json") output_type_ = 2;
            else if (type == "parquet") output_type_ = 3;
            else if (type == "excel") output_type_ = 4;
            else if (type == "hdf5") output_type_ = 5;
        }
    }
}

void DataOutputDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;
    const char* types[] = {"csv", "tsv", "json", "parquet", "excel", "hdf5"};
    node_->parameters["file_type"] = types[output_type_];
    node_->parameters["overwrite"] = overwrite_ ? "true" : "false";
    node_->parameters["include_header"] = include_header_ ? "true" : "false";
    const char* compressions[] = {"none", "gzip", "snappy", "zstd"};
    node_->parameters["compression"] = compressions[compression_];
    node_->parameters["configured"] = "true";

    if (strlen(file_path_) > 0) {
        fs::path p(file_path_);
        node_->description = "Writing to " + p.filename().string();
    }

    has_changes_ = false;
}

void DataOutputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    has_changes_ = false;
}

void DataOutputDialog::RenderContent() {
    if (ImGui::BeginTabBar("DataOutputTabs")) {
        if (ImGui::BeginTabItem("Settings")) {
            RenderSettingsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Advanced")) {
            RenderAdvancedTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataOutputDialog::RenderSettingsTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Output Location");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("File:");
    ImGui::SameLine(60);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    if (ImGui::InputText("##outpath", file_path_, sizeof(file_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse", ImVec2(70, 0))) {
        // TODO: Save file dialog
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Format");
    ImGui::Separator();
    ImGui::Spacing();

    const char* formats[] = {"CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5"};
    ImGui::Text("Format:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##format", &output_type_, formats, 6)) {
        has_changes_ = true;
    }

    if (output_type_ <= 1) {
        if (ImGui::Checkbox("Include header", &include_header_)) {
            has_changes_ = true;
        }
    }

    if (ImGui::Checkbox("Overwrite existing", &overwrite_)) {
        has_changes_ = true;
    }
}

void DataOutputDialog::RenderAdvancedTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Compression");
    ImGui::Separator();
    ImGui::Spacing();

    const char* compressions[] = {"None", "gzip", "Snappy", "zstd"};
    ImGui::Text("Compression:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##compression", &compression_, compressions, 4)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Encoding");
    ImGui::Separator();
    ImGui::Spacing();

    const char* encodings[] = {"UTF-8", "UTF-8 with BOM", "ASCII", "ISO-8859-1"};
    static int out_encoding = 0;
    ImGui::Text("Encoding:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("##outencoding", &out_encoding, encodings, 4);
}

} // namespace gui
