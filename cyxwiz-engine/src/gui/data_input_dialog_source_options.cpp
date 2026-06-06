// DataInputDialog source selector and file option rendering.

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

#include "node_config_dialog.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace gui {

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
    if (ImGui::RadioButton("ML Dataset (planned)", &source_idx, 1)) {
        source_type_ = SourceType::MLDataset;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Database (planned)", &source_idx, 2)) {
        source_type_ = SourceType::Database;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Cloud (planned)", &source_idx, 3)) {
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
    if (ImGui::RadioButton("Video (planned)", &cat_idx, 3)) {
        file_category_ = FileCategory::Video;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Text", &cat_idx, 4)) {
        file_category_ = FileCategory::Text;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Time Series", &cat_idx, 5)) {
        file_category_ = FileCategory::TimeSeries;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Single-file path section — shown for Tabular and for Text in
    // SingleFile layout mode. In CorpusSubdirs mode Text uses a folder
    // picker inside RenderTextOptions instead (mirroring image/audio).
    // Image uses its own folder/CSV inputs in RenderImageOptions.
    // Audio/Video are not yet supported and show their own messages.
    // Hiding this for other categories prevents the user from picking
    // a .jpg in a file dialog that's meant for CSVs.
    const bool show_file_picker =
        (file_category_ == FileCategory::Tabular) ||
        (file_category_ == FileCategory::TimeSeries) ||
        (file_category_ == FileCategory::Text &&
         text_layout_ == TextLayout::SingleFile);
    if (show_file_picker) {
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
                LoadColumnList();
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
    }

    ImGui::Spacing();

    // Category-specific options
    switch (file_category_) {
        case FileCategory::Tabular:    RenderTabularOptions(); break;
        case FileCategory::Image:      RenderImageOptions(); break;
        case FileCategory::Audio:      RenderAudioOptions(); break;
        case FileCategory::Video:      RenderVideoOptions(); break;
        case FileCategory::Text:       RenderTextOptions(); break;
        // Phase 4: Time Series shares the Tabular loader entirely — the
        // only semantic difference is the file_category="timeseries" stamp
        // on the DataInput node, which the downstream TimeSeriesWindow
        // operator uses as a sanity hint. All format/delimiter/column
        // config is identical to Tabular.
        case FileCategory::TimeSeries: RenderTabularOptions(); break;
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

    // Column Mapping - Label/Target column selection
    if (ImGui::CollapsingHeader("Column Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
            "For ML training, specify which column contains labels/targets");
        ImGui::Spacing();

        // Label column selector
        ImGui::Text("Label Column:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(200);

        // Build combo items from available columns
        const char* preview = (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(available_columns_.size()))
            ? available_columns_[label_column_idx_].c_str()
            : "(None - select label column)";

        if (ImGui::BeginCombo("##labelcol", preview)) {
            // Option for no label
            if (ImGui::Selectable("(None)", label_column_idx_ == -1)) {
                label_column_idx_ = -1;
                has_changes_ = true;
            }

            // List available columns
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                bool is_selected = (label_column_idx_ == i);
                if (ImGui::Selectable(available_columns_[i].c_str(), is_selected)) {
                    label_column_idx_ = i;
                    has_changes_ = true;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // Hint for common label column names
        if (label_column_idx_ < 0 && !available_columns_.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "!");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Select the target column for training.\nCommon names: 'label', 'class', 'target', 'y'");
            }
        }

        // Auto-detect common label column names
        if (label_column_idx_ < 0 && !available_columns_.empty()) {
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                std::string lower = available_columns_[i];
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower == "label" || lower == "labels" || lower == "class" || lower == "target" || lower == "y") {
                    // Found a common label column - suggest it
                    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f),
                        "Suggestion: '%s' looks like a label column", available_columns_[i].c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Use")) {
                        label_column_idx_ = i;
                        has_changes_ = true;
                    }
                    break;
                }
            }
        }

        ImGui::Spacing();
        ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
            "Features: All other columns will be used as input features");
    }

}

void DataInputDialog::RenderImageOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Image Loading Options");
    ImGui::Spacing();

    // Image dataset layout options. Each entry describes a supported
    // on-disk layout, the combo-box display name, a one-line description,
    // and whether the user needs to pick a Labels CSV. Adding a new
    // layout for Phase 1 means adding one entry here and one case in
    // Apply's dispatch — nothing else in the dialog changes. Kept as a
    // function-local static so we don't need to expose the private
    // ImageLayout enum at file scope.
    struct ImageLayoutOption {
        ImageLayout value;
        const char* display;
        const char* description;
        bool needs_labels_csv;
    };
    static const ImageLayoutOption kImageLayoutOptions[] = {
        {ImageLayout::ClassSubdirs,
         "Class subdirectories (ImageNet-style)",
         "One subfolder per class; folder name becomes the label. "
         "E.g. root/cat/*.jpg, root/dog/*.jpg",
         false},
        {ImageLayout::FlatWithCSV,
         "Flat folder + CSV labels (Kaggle-style)",
         "All images in one folder; a CSV file maps each filename to its label. "
         "E.g. root/img1.jpg + labels.csv",
         true},
        // Phase 1 extends here: Unlabeled, COCO, YOLO, JSONLabels, HDF5Labels.
    };
    constexpr int kImageLayoutOptionsCount =
        static_cast<int>(sizeof(kImageLayoutOptions) / sizeof(kImageLayoutOptions[0]));

    // === Layout strategy combo ===
    // The user explicitly picks how their image dataset is laid out on
    // disk. No auto-detect magic — what they pick is what Apply dispatches
    // to. Adding new layouts is a one-line entry in kImageLayoutOptions.
    int current_idx = 0;
    for (int i = 0; i < kImageLayoutOptionsCount; ++i) {
        if (kImageLayoutOptions[i].value == image_layout_) {
            current_idx = i;
            break;
        }
    }

    ImGui::Text("Dataset layout:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10);
    if (ImGui::BeginCombo("##imglayout", kImageLayoutOptions[current_idx].display)) {
        for (int i = 0; i < kImageLayoutOptionsCount; ++i) {
            bool selected = (i == current_idx);
            if (ImGui::Selectable(kImageLayoutOptions[i].display, selected)) {
                image_layout_ = kImageLayoutOptions[i].value;
                has_changes_ = true;
                preview_loaded_ = false;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // Description line below the combo explains what this layout means.
    ImGui::TextDisabled("   %s", kImageLayoutOptions[current_idx].description);

    ImGui::Spacing();

    // === Folder input (required for all layouts) ===
    ImGui::Text("Folder:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    if (ImGui::InputText("##imgfolder", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##imgbrowse", ImVec2(80, 0))) {
        BrowseFolder();
    }

    // === Labels CSV input (only shown for layouts that need it) ===
    const bool needs_csv = kImageLayoutOptions[current_idx].needs_labels_csv;
    if (needs_csv) {
        ImGui::Text("Labels CSV:");
        ImGui::SameLine(130);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##labelscsv", labels_csv_, sizeof(labels_csv_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse##csvbrowse", ImVec2(80, 0))) {
#ifdef _WIN32
            OPENFILENAMEA ofn = {};
            char file[512] = {};
            strncpy(file, labels_csv_, sizeof(file) - 1);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "CSV Files\0*.csv\0All Files\0*.*\0";
            ofn.lpstrFile = file;
            ofn.nMaxFile = sizeof(file);
            ofn.lpstrTitle = "Select Labels CSV";
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn)) {
                strncpy(labels_csv_, file, sizeof(labels_csv_) - 1);
                has_changes_ = true;
            }
#else
            spdlog::warn("Labels CSV file browser not implemented for this platform");
#endif
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

    // Layout selector — ClassSubdirs vs FlatWithCSV. Mirrors the image dialog.
    static const char* kAudioLayoutNames[] = {
        "Class subdirectories (folder/<class>/*.wav)",
        "Flat folder + Labels CSV"
    };
    int layout_idx = static_cast<int>(audio_layout_);
    ImGui::Text("Layout:");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##audiolayout", &layout_idx, kAudioLayoutNames, IM_ARRAYSIZE(kAudioLayoutNames))) {
        audio_layout_ = static_cast<AudioLayout>(layout_idx);
        has_changes_ = true;
    }

    ImGui::Spacing();

    // Folder picker — same field as image folder. The Apply path
    // dispatches by file_category so wires don't cross.
    ImGui::Text("Folder:");
    ImGui::SetNextItemWidth(-80.0f);
    if (ImGui::InputText("##audiofolder", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##audio")) {
        BrowseFolder();
    }

    // FlatWithCSV: show the CSV picker + optional column overrides
    if (audio_layout_ == AudioLayout::FlatWithCSV) {
        ImGui::TextDisabled("Flat folder layout: all audio files in one directory, labels in CSV.");

        ImGui::Spacing();
        ImGui::Text("Labels CSV:");
        ImGui::SetNextItemWidth(-80.0f);
        if (ImGui::InputText("##audiocsv", audio_labels_csv_, sizeof(audio_labels_csv_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse##audiocsv")) {
            // Reuse the same file browser as everything else
            BrowseFile();
            // BrowseFile writes to file_path_; copy it to the CSV field
            if (strlen(file_path_) > 0) {
                strncpy(audio_labels_csv_, file_path_,
                        sizeof(audio_labels_csv_) - 1);
                audio_labels_csv_[sizeof(audio_labels_csv_) - 1] = '\0';
            }
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Column auto-detect: leave blank to use header names like "
                            "filename / file / id / call_id and label / class / category. "
                            "Override below if your CSV uses unusual column names.");

        ImGui::Text("Filename col:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiofnamecol", audio_filename_col_,
                             sizeof(audio_filename_col_))) {
            has_changes_ = true;
        }

        ImGui::Text("Label col:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiolabelcol", audio_label_col_,
                             sizeof(audio_label_col_))) {
            has_changes_ = true;
        }
    } else {
        ImGui::TextDisabled("Class subdirectories layout: folder/class_a/*.wav, folder/class_b/*.wav");
    }

    // Quick scan preview if folder exists
    if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
        if (audio_layout_ == AudioLayout::ClassSubdirs) {
            int subdir_count = 0;
            int audio_file_count = 0;
            try {
                for (const auto& entry : fs::directory_iterator(folder_path_)) {
                    if (entry.is_directory()) {
                        subdir_count++;
                        for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                            if (sub.is_regular_file()) {
                                std::string ext = sub.path().extension().string();
                                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                                if (ext == ".wav" || ext == ".flac" || ext == ".ogg" ||
                                    ext == ".aiff" || ext == ".aif" || ext == ".mp3") {
                                    audio_file_count++;
                                }
                            }
                        }
                    }
                }
            } catch (...) { /* permission error etc., ignore */ }
            if (subdir_count > 0) {
                ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                                   "Detected: %d classes, %d audio files",
                                   subdir_count, audio_file_count);
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f),
                                   "No class subdirectories found");
            }
        } else {
            // Flat folder count
            int flat_count = 0;
            try {
                for (const auto& entry : fs::recursive_directory_iterator(folder_path_)) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".wav" || ext == ".flac" || ext == ".ogg" ||
                            ext == ".aiff" || ext == ".aif" || ext == ".mp3") {
                            flat_count++;
                        }
                    }
                }
            } catch (...) { }
            ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                               "Detected: %d audio files in folder tree", flat_count);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Sample rate:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##samplerate", &sample_rate_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Hz");

    ImGui::Text("Max duration:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputFloat("##duration", &duration_sec_, 0.5f, 1.0f, "%.1f")) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("seconds (pad/truncate)");

    if (ImGui::Checkbox("Convert to mono", &mono_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Feature extraction defaults to MelSpectrogram (n_mels=128, n_fft=512).");
    ImGui::TextDisabled("Spectrogram / MFCC nodes will override this in a later phase.");
    ImGui::TextDisabled("Supported: WAV, FLAC, OGG, AIFF");
}

void DataInputDialog::RenderVideoOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Video Loading");
    ImGui::Spacing();
    ImGui::TextWrapped("%s", UnsupportedApplyMessage());
    ImGui::Spacing();
    ImGui::TextDisabled("Use tabular, image, audio, text, or time series data for now.");
}

void DataInputDialog::RenderTextOptions() {
    // Phase 3 text options. Two layouts:
    //   1. SingleFile    — CSV / JSON / TXT picked via shared file picker
    //   2. CorpusSubdirs — folder/<class>/*.txt picked via folder picker
    //                      shown inline here. Analogous to image
    //                      ClassSubdirs / audio ClassSubdirs.
    //
    // Column mapping controls only make sense in SingleFile mode — a
    // folder corpus gets its labels from subdirectory names, not CSV
    // columns. We hide the column inputs when CorpusSubdirs is picked
    // so the user can't fill in fields that get ignored at load time.
    //
    // Graph preprocessing nodes (TextTokenizer / TextVocabulary /
    // TextPadding) override the tokenizer defaults at compile time if
    // present, so these are just sensible fallbacks for the common
    // case of "drop a DataInput, point at a CSV, run".
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Text Loading Options");
    ImGui::Spacing();

    // Layout selector — mirrors the image / audio combo at the top of
    // their options panels. Switching between modes preserves whichever
    // path (file vs folder) was previously filled in, so round-tripping
    // doesn't destroy state.
    static const char* kTextLayoutNames[] = {
        "Single file (CSV / JSON / TXT)",
        "Corpus subdirectories (folder/<class>/*.txt)"
    };
    int layout_idx = static_cast<int>(text_layout_);
    ImGui::Text("Layout:");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##textlayout", &layout_idx, kTextLayoutNames, IM_ARRAYSIZE(kTextLayoutNames))) {
        text_layout_ = static_cast<TextLayout>(layout_idx);
        has_changes_ = true;
    }
    ImGui::Spacing();

    if (text_layout_ == TextLayout::SingleFile) {
        ImGui::TextDisabled("Loads from CSV/TSV (text column + label column), "
                            "JSON / JSONL (text field + label field), or plain "
                            "TXT (one unlabeled sample per line).");
    } else {
        ImGui::TextDisabled("Loads text from folder/<class>/*.txt — each "
                            "subdirectory is a class, each .txt file is one "
                            "sample. Mirrors ImageFolder / audio ClassSubdirs.");
    }
    ImGui::Spacing();

    // CorpusSubdirs: folder picker lives inline here (file picker at the
    // top is hidden in this mode). Shows a quick scan of subdirs when
    // the path exists — mirrors the image/audio folder feedback.
    if (text_layout_ == TextLayout::CorpusSubdirs) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(accent, "Folder");
        ImGui::Spacing();

        ImGui::Text("Path:");
        ImGui::SameLine(60);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##textfolder", folder_path_, sizeof(folder_path_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...##textfolder", ImVec2(80, 0))) {
            BrowseFolder();
        }

        if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
            int subdir_count = 0;
            int file_count = 0;
            try {
                for (const auto& entry : fs::directory_iterator(folder_path_)) {
                    if (!entry.is_directory()) continue;
                    subdir_count++;
                    for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                        if (!sub.is_regular_file()) continue;
                        std::string ext = sub.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".txt" || ext == ".text" || ext == ".md") {
                            file_count++;
                        }
                    }
                }
            } catch (...) { /* permission errors etc., ignore */ }
            if (subdir_count > 0) {
                ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                                   "Detected: %d classes, %d text files",
                                   subdir_count, file_count);
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f),
                                   "No class subdirectories found — corpus "
                                   "layout expects folder/<class>/*.txt");
            }
        }
        ImGui::Spacing();
    }

    // Column mapping is only meaningful for SingleFile CSV/JSON mode.
    // In corpus mode the label comes from the subdirectory name, so we
    // hide these fields to avoid confusion.
    if (text_layout_ == TextLayout::SingleFile) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(accent, "Column Mapping");
        ImGui::Spacing();

        ImGui::Text("Text col:");
        ImGui::SameLine(140);
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##textcol", text_column_, sizeof(text_column_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(CSV column / JSON field holding sentences)");

        ImGui::Text("Label col:");
        ImGui::SameLine(140);
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##textlabelcol", text_label_column_, sizeof(text_label_column_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(blank = unlabeled corpus / plain TXT)");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Tokenizer Defaults");
    ImGui::Spacing();
    ImGui::TextDisabled("Overridden by a TextTokenizer node in the graph, if present.");
    ImGui::Spacing();

    static const char* kTokenizerNames[] = {
        "Whitespace (split on spaces)",
        "Word (splits on punctuation too)",
        "Character (one token per char)"
    };
    ImGui::Text("Tokenizer:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(260);
    if (ImGui::Combo("##texttok", &text_tokenizer_type_, kTokenizerNames, IM_ARRAYSIZE(kTokenizerNames))) {
        has_changes_ = true;
    }

    ImGui::Text("Max length:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textmaxlen", &text_max_length_)) {
        if (text_max_length_ < 1) text_max_length_ = 1;
        if (text_max_length_ > 8192) text_max_length_ = 8192;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("tokens (pad/truncate)");

    if (ImGui::Checkbox("Lowercase text", &text_lowercase_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Vocabulary");
    ImGui::Spacing();

    ImGui::Text("Min freq:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textminfreq", &text_min_freq_)) {
        if (text_min_freq_ < 1) text_min_freq_ = 1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(drop words occurring fewer times)");

    ImGui::Text("Max vocab:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textmaxvocab", &text_max_vocab_size_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(-1 = unlimited)");

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: CSV, TSV, JSON, TXT (one line per sample), folder of text files");
}

} // namespace gui
