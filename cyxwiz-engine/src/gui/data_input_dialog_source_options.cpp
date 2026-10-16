// DataInputDialog source selector and file option rendering.

#include "node_config_dialog.h"
#include "../core/file_dialogs.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/data_input_formats.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace gui {

void DataInputDialog::RenderFileSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    ImVec4 info_color = style.Colors[ImGuiCol_TextDisabled];

    // File category selector
    ImGui::TextColored(accent, "File type");
    ImGui::TextColored(info_color,
                       "Pick the data contract this node should expose to the training graph.");
    ImGui::Spacing();

    // Sequence (token tagging) loads as a tabular file and adds the tagging
    // contract, so it is its own radio index rather than a loader category.
    constexpr int kSequenceIndex = 5;
    int cat_idx = sequence_tagging_ ? kSequenceIndex : static_cast<int>(file_category_);
    const auto pick = [this](FileCategory category, bool sequence) {
        file_category_ = category;
        sequence_tagging_ = sequence;
        has_changes_ = true;
        SyncHdf5InspectorSource();
    };
    if (ImGui::RadioButton("Tabular", &cat_idx, 0)) pick(FileCategory::Tabular, false);
    ImGui::SameLine();
    if (ImGui::RadioButton("Image", &cat_idx, 1)) pick(FileCategory::Image, false);
    ImGui::SameLine();
    if (ImGui::RadioButton("Audio", &cat_idx, 2)) pick(FileCategory::Audio, false);
    ImGui::SameLine();
    if (ImGui::RadioButton("Text", &cat_idx, 3)) pick(FileCategory::Text, false);
    ImGui::SameLine();
    if (ImGui::RadioButton("Sequence (token tagging)", &cat_idx, kSequenceIndex)) {
        pick(FileCategory::Tabular, true);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Time Series", &cat_idx, 4)) pick(FileCategory::TimeSeries, false);

    ImGui::Spacing();

    // Single-file path section — shown for Tabular and for Text in
    // SingleFile layout mode. In CorpusSubdirs mode Text uses a folder
    // picker inside RenderTextOptions instead (mirroring image/audio).
    // Image uses its own folder/CSV inputs in RenderImageOptions.
    // Hiding this for other categories prevents the user from picking
    // a .jpg in a file dialog that's meant for CSVs.
    const bool show_file_picker =
        (file_category_ == FileCategory::Tabular) ||
        (file_category_ == FileCategory::TimeSeries) ||
        (file_category_ == FileCategory::Text &&
         text_layout_ == TextLayout::SingleFile);
    if (show_file_picker) {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 12.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 12.0f));
        if (ImGui::BeginChild("FileSelection", ImVec2(0, 118), false,
                              ImGuiWindowFlags_NoScrollbar |
                                  ImGuiWindowFlags_NoScrollWithMouse)) {
            ImGui::TextColored(accent, "Input File");
            ImGui::SameLine();
            ImGui::TextColored(info_color, "Select the source file used by materialization and training.");
            ImGui::Spacing();

            // File path input
            ImGui::Text("Path:");
            ImGui::SameLine(68);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
            if (ImGui::InputText("##filepath", file_path_, sizeof(file_path_))) {
                DetectFileType();
                DetectFileCategory();
                RefreshColumnList();
                has_changes_ = true;
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
        ImGui::PopStyleVar(2);
    }

    ImGui::Spacing();

    // Category-specific options
    switch (file_category_) {
        case FileCategory::Tabular:    RenderTabularOptions(); break;
        case FileCategory::Image:      RenderImageOptions(); break;
        case FileCategory::Audio:      RenderAudioOptions(); break;
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
    if (ImGui::CollapsingHeader("Format Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Format auto-detect or manual selection
        ImGui::Text("Format:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(120);
        if (ImGui::BeginCombo("##format", GetFileTypeName())) {
            for (const auto& option : cyxwiz::data_input::kSourceFormats) {
                if (!option.executable) continue;
                const bool selected = detected_type_ == option.id;
                if (ImGui::Selectable(option.label, selected)) {
                    detected_type_ = option.id;
                    RefreshColumnList();
                    has_changes_ = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        const auto file_type = data_input::FileTypeParam(detected_type_);
        if (!cyxwiz::data_input::IsExecutable(file_type)) {
            ImGui::TextDisabled("%s", cyxwiz::data_input::UnsupportedReason(
                file_type, cyxwiz::data_input::kHdf5BuildAvailable));
        }

        if (detected_type_ == 11) {
            if (ImGui::InputTextMultiline("Member paths", archive_member_, sizeof(archive_member_), ImVec2(-1, 100))) {
                has_changes_ = true;
                RefreshColumnList();
            }
            ImGui::TextWrapped("One exact case-sensitive ZIP path per line; selection order is preserved. No wildcards. Original UTF-8 documents with source hashes. Limits: archive 64 MiB, member 16 MiB, selected text 64 MiB, 4096 selections, 10000 archive entries.");
        }

        // CSV/TSV specific options (delimiter-based)
        if (detected_type_ <= 2) {
            ImGui::Spacing();

            ImGui::Text("Delimiter:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(50);
            if (ImGui::InputText("##delim", custom_delimiter_, sizeof(custom_delimiter_))) {
                RefreshColumnList();
                has_changes_ = true;
            }

            ImGui::SameLine(180);
            if (ImGui::Checkbox("Has header", &has_header_)) {
                RefreshColumnList();
                has_changes_ = true;
            }

            ImGui::Text("Decimal separator:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(110);
            const char* decimal_label =
                decimal_point_ == ',' ? "Comma (,)" : "Dot (.)";
            if (ImGui::BeginCombo("##decimalpoint", decimal_label)) {
                if (ImGui::Selectable("Dot (.)", decimal_point_ == '.')) {
                    decimal_point_ = '.';
                    has_changes_ = true;
                }
                if (ImGui::Selectable("Comma (,)", decimal_point_ == ',')) {
                    decimal_point_ = ',';
                    has_changes_ = true;
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Controls numeric parsing only. It is separate from the field delimiter.");
            }

            ImGui::Text("Quote char:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(30);
            // One character; empty reads quotes as ordinary text.
            if (ImGui::InputText("##quote", quote_char_, 2)) {
                has_changes_ = true;
                preview_loaded_ = false;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Values in this character may hold the delimiter and line breaks. "
                                  "Leave it empty to read quotes as ordinary text.");
            }

            ImGui::Text("Missing tokens:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(220);
            if (ImGui::InputText("##missingtokens", missing_value_tokens_,
                                 sizeof(missing_value_tokens_))) {
                has_changes_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Comma-separated values treated as null before type inference.\n"
                    "Example: na,?,missing. Arrow standard null spellings remain enabled.");
            }
            ImGui::TextDisabled("Reads UTF-8 (a byte order mark is skipped).");
        }
    }

    if (IsHdf5Source()) {
        SyncHdf5InspectorSource();
        hdf5_inspector_.RenderSettings();
        return;
    }

    if (ImGui::CollapsingHeader(
            "Dataset Assignment", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(
            ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
            "This Data Input always produces one role-neutral Dataset.");
        ImGui::TextWrapped(
            "There is deliberately no role selector here. The connection to "
            "Data Split is the single source of truth:");
        ImGui::BulletText("Training Dataset: model fitting source");
        ImGui::BulletText(
            "Validation Dataset: optional external validation/dev source");
        ImGui::BulletText(
            "Test Dataset: optional held-out final evaluation source");
        ImGui::TextWrapped(
            "Open Data Split properties after connecting the sources to "
            "review the resolved Train/Validation/Test manifest.");

        const auto legacy_role = node_->parameters.find("dataset_role");
        if (legacy_role != node_->parameters.end() &&
            !legacy_role->second.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(
                cyxwiz::ui::CurrentTokens().warning,
                "Legacy role hint '%s' is ignored.",
                legacy_role->second.c_str());
            ImGui::TextWrapped(
                "Reconnect this Dataset to the intended named Data Split "
                "input. Apply removes the obsolete hint.");
        }
    }


    // Column Mapping - Label/Target column selection. A token-tagging file
    // maps its token / POS / tag columns instead.
    if (sequence_tagging_) {
        RenderSequenceColumns();
    } else if (ImGui::CollapsingHeader("Column Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
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
                    if (i < static_cast<int>(selected_columns_.size())) {
                        selected_columns_[static_cast<std::size_t>(i)] = true;
                    }
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
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning, "!");
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
                    ImGui::TextColored(cyxwiz::ui::CurrentTokens().accent_text,
                        "Suggestion: '%s' looks like a label column", available_columns_[i].c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Use")) {
                        label_column_idx_ = i;
                        if (i < static_cast<int>(selected_columns_.size())) {
                            selected_columns_[static_cast<std::size_t>(i)] = true;
                        }
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
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select Labels CSV",
                    {{"CSV Files", "csv"}, {"All Files", "*"}},
                    labels_csv_[0] == '\0' ? nullptr : labels_csv_)) {
                strncpy(labels_csv_, selected->c_str(), sizeof(labels_csv_) - 1);
                labels_csv_[sizeof(labels_csv_) - 1] = '\0';
                has_changes_ = true;
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderPreprocessingNodes();
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
            // Its own picker: the shared one wrote the main file path.
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select Labels CSV", {{"CSV Files", "csv"}, {"All Files", "*"}},
                    audio_labels_csv_[0] == '\0' ? nullptr : audio_labels_csv_)) {
                strncpy(audio_labels_csv_, selected->c_str(), sizeof(audio_labels_csv_) - 1);
                audio_labels_csv_[sizeof(audio_labels_csv_) - 1] = '\0';
                has_changes_ = true;
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
                ImGui::TextColored(cyxwiz::ui::CurrentTokens().success,
                                   "Detected: %d classes, %d audio files",
                                   subdir_count, audio_file_count);
            } else {
                ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning,
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
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().success,
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

    ImGui::Text("Length:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputFloat("##duration", &duration_sec_, 0.5f, 1.0f, "%.1f")) {
        if (duration_sec_ < 0.1f) duration_sec_ = 0.1f;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("seconds, padded or cut");
    ImGui::TextDisabled("Audio is mixed to mono (channels averaged). Supported: WAV, FLAC, OGG, AIFF");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderPreprocessingNodes();
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
    // Tokenizing is the Text Tokenizer node's (TOFIX129 B-8): this dialog
    // names the node that does it, or adds one.
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
                ImGui::TextColored(cyxwiz::ui::CurrentTokens().success,
                                   "Detected: %d classes, %d text files",
                                   subdir_count, file_count);
            } else {
                ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning,
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
    RenderPreprocessingNodes();

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: CSV, TSV, JSON, TXT (one line per sample), folder of text files");
}

void DataInputDialog::RenderPreprocessingNodes() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_HeaderActive], "Preprocessing (done by nodes)");
    ImGui::Spacing();
    const bool text = file_category_ == FileCategory::Text;
    const bool image = file_category_ == FileCategory::Image;
    if (!node_editor_ || !node_) {
        ImGui::TextDisabled(text ? "A Text Tokenizer node after this input tokenizes the text."
                            : image ? "A Resize node after the Data Loader sets the image size."
                                    : "A Mel Spectrogram, Spectrogram or MFCC node makes the audio features.");
        return;
    }
    // What this kind of data needs from its nodes (TOFIX129 B-8, B-9).
    const auto wanted = [&](NodeType type) {
        if (text)
            return type == NodeType::TextTokenizer || type == NodeType::TFIDFVectorizer ||
                   type == NodeType::CountVectorizer || type == NodeType::WordEmbeddings;
        if (image) return type == NodeType::Resize || type == NodeType::Normalize;
        return type == NodeType::Spectrogram || type == NodeType::MelSpectrogram || type == NodeType::MFCC;
    };
    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();
    const auto find = [&](int id) {
        const auto it = std::find_if(nodes.begin(), nodes.end(), [id](const MLNode& n) { return n.id == id; });
        return it == nodes.end() ? nullptr : &*it;
    };
    // The wanted nodes after this input, nearest first.
    std::vector<const MLNode*> found;
    std::set<int> seen{node_->id};
    std::vector<int> frontier{node_->id};
    for (size_t i = 0; i < frontier.size(); ++i) {
        for (const auto& link : links) {
            if (link.from_node != frontier[i] || !seen.insert(link.to_node).second) continue;
            const MLNode* next = find(link.to_node);
            if (!next) continue;
            if (wanted(next->type)) found.push_back(next);
            // Text stops at the node that tokenizes it; images and audio go on.
            if (!text || !wanted(next->type)) frontier.push_back(next->id);
        }
    }
    const auto param = [](const MLNode& n, const char* key) {
        const auto it = n.parameters.find(key);
        return it == n.parameters.end() ? std::string() : it->second;
    };
    const auto& factory = NodeConfigDialogFactory::Instance();
    int open_id = -1, focus_id = -1;
    const MLNode* resize = nullptr;
    bool normalized = false;
    for (const MLNode* n : found) {
        ImGui::PushID(n->id);
        std::string role, facts;
        switch (n->type) {
            case NodeType::TextTokenizer: {
                role = "Tokenized by";
                if (const auto vocab = param(*n, "vocab_file"); !vocab.empty())
                    facts = "vocabulary " + fs::path(vocab).filename().string();
                if (const auto length = param(*n, "max_length"); !length.empty())
                    facts += (facts.empty() ? "" : ", ") + length + " ids per row";
                break;
            }
            case NodeType::WordEmbeddings: role = "Word vectors by"; break;
            case NodeType::TFIDFVectorizer:
            case NodeType::CountVectorizer: role = "Vectorized by"; break;
            case NodeType::Resize:
                role = "Resized by";
                if (!resize) resize = n;
                facts = param(*n, "width") + " x " + param(*n, "height");
                break;
            case NodeType::Normalize:
                role = "Normalized by";
                normalized = true;
                facts = "mean " + param(*n, "mean") + ", std " + param(*n, "std");
                break;
            case NodeType::MFCC:
                role = "Features by";
                facts = param(*n, "n_mfcc") + " coefficients";
                break;
            case NodeType::MelSpectrogram:
                role = "Features by";
                facts = param(*n, "n_mels") + " mel bins, n_fft " + param(*n, "n_fft");
                break;
            default:
                role = "Features by";
                facts = "n_fft " + param(*n, "n_fft");
                break;
        }
        ImGui::TextColored(t.text_dim, "%s", role.c_str());
        ImGui::SameLine();
        ImGui::TextUnformatted(n->name.c_str());
        if (!facts.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", facts.c_str());
        }
        // Nodes without a dialog are set in Properties: Show on canvas selects them.
        if (factory.HasDialog(n->type)) {
            ImGui::SameLine();
            if (cyxwiz::ui::LinkButton("Open it")) open_id = n->id;
        }
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Show on canvas")) focus_id = n->id;
        ImGui::PopID();
    }
    if (text && found.empty()) {
        ImGui::TextColored(t.warning, "Nothing after this input tokenizes the text yet.");
        if (cyxwiz::ui::LinkButton("Add a Text Tokenizer"))
            node_editor_->AddTextTokenizerAfter(node_->id, text_column_, text_label_column_);
    } else if (image && !resize) {
        ImGui::TextColored(t.warning, "No Resize yet: images of different sizes cannot share a batch.");
        if (cyxwiz::ui::LinkButton("Add a Resize")) node_editor_->AddModelInputStepAfter(node_->id, NodeType::Resize);
    } else if (!text && !image && found.empty()) {
        ImGui::TextColored(t.warning, "No feature node yet: the model cannot read raw audio.");
        if (cyxwiz::ui::LinkButton("Add Mel Spectrogram"))
            node_editor_->AddModelInputStepAfter(node_->id, NodeType::MelSpectrogram);
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Add Spectrogram"))
            node_editor_->AddModelInputStepAfter(node_->id, NodeType::Spectrogram);
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Add MFCC")) node_editor_->AddModelInputStepAfter(node_->id, NodeType::MFCC);
    }
    if (image && resize && !normalized) {
        ImGui::TextColored(t.text_dim, "Normalized by");
        ImGui::SameLine();
        ImGui::TextDisabled("no Normalize node (pixels are read as 0 to 1)");
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Add Normalize")) node_editor_->RequestStepAfter(resize->id, NodeType::Normalize);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled(text ? "The tokenizer's settings live in its node only."
                             : "These settings live in their nodes only; this dialog only loads the files.");
    ImGui::PopTextWrapPos();
    if (open_id >= 0) node_editor_->RequestOpenNode(open_id);
    if (focus_id >= 0) node_editor_->FocusNode(focus_id);
}

} // namespace gui
