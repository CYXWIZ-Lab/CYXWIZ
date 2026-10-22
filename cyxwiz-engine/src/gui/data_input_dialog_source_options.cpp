// DataInputDialog Source tab: 1. the file or folder, 2. how its rows are used
// (TOFIX129 B-10), and the preprocessing nodes (B-8, B-9).

#include "node_config_dialog.h"
#include "../core/file_dialogs.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/data_input_formats.h"
#include "../core/data_input_presentation.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace gui {

namespace {

constexpr float kLabelWidth = 150.0f;

// A muted label in the left column, the control after it.
void FieldLabel(const char* label) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "%s", label);
    ImGui::SameLine(kLabelWidth);
}

void SectionTitle(const char* title) {
    ImGui::Spacing();
    ImGui::TextUnformatted(title);
    ImGui::Spacing();
}

// One column picked by name from the file's columns (free text until they
// are known). Returns true when it changed.
bool ColumnPicker(const char* id, char* value, size_t size, const std::vector<std::string>& columns,
                  const char* none_label) {
    ImGui::SetNextItemWidth(260.0f);
    if (columns.empty()) return ImGui::InputText(id, value, size);
    bool changed = false;
    if (ImGui::BeginCombo(id, value[0] ? value : none_label)) {
        if (none_label && ImGui::Selectable(none_label, value[0] == '\0')) {
            value[0] = '\0';
            changed = true;
        }
        for (const auto& column : columns) {
            if (ImGui::Selectable(column.c_str(), column == value)) {
                std::strncpy(value, column.c_str(), size - 1);
                value[size - 1] = '\0';
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

std::string JoinFirst(const std::vector<std::string>& names, size_t limit) {
    std::string out;
    for (size_t i = 0; i < names.size() && i < limit; ++i)
        out += (i ? ", " : "") + (names[i].empty() ? std::string("(unnamed)") : names[i]);
    if (names.size() > limit) out += ", and " + std::to_string(names.size() - limit) + " more";
    return out;
}

}  // namespace

void DataInputDialog::SetSource(const std::string& path) {
    std::strncpy(source_buf_, path.c_str(), sizeof(source_buf_) - 1);
    source_buf_[sizeof(source_buf_) - 1] = '\0';
    std::error_code ec;
    if (!path.empty() && fs::is_directory(path, ec)) {
        // A folder: its files say what it is (images, audio, text).
        std::strncpy(folder_path_, path.c_str(), sizeof(folder_path_) - 1);
        folder_path_[sizeof(folder_path_) - 1] = '\0';
        file_path_[0] = '\0';
        file_size_ = 0;
        RescanFolder();
        sequence_tagging_ = false;
        switch (folder_scan_.kind) {
            case data_input::FolderKind::Images:
                file_category_ = FileCategory::Image;
                if (folder_scan_.class_names.empty()) image_layout_ = ImageLayout::FlatWithCSV;
                break;
            case data_input::FolderKind::Audio:
                file_category_ = FileCategory::Audio;
                if (folder_scan_.class_names.empty()) audio_layout_ = AudioLayout::FlatWithCSV;
                break;
            case data_input::FolderKind::Text:
                file_category_ = FileCategory::Text;
                text_layout_ = TextLayout::CorpusSubdirs;
                break;
            case data_input::FolderKind::Unknown: break;
        }
    } else {
        // A file is a table (or a text document set): its rows get a use.
        std::strncpy(file_path_, path.c_str(), sizeof(file_path_) - 1);
        file_path_[sizeof(file_path_) - 1] = '\0';
        folder_path_[0] = '\0';
        folder_scan_ = {};
        folder_scan_path_.clear();
        DetectFileType();
        if (file_category_ == FileCategory::Image || file_category_ == FileCategory::Audio)
            file_category_ = FileCategory::Tabular;
        if (file_category_ == FileCategory::Text) text_layout_ = TextLayout::SingleFile;
        if (data_input::FileTypeParam(detected_type_) == std::string("zip_text")) format_options_open_ = true;
        RefreshColumnList();
    }
    SyncHdf5InspectorSource();
    has_changes_ = true;
    preview_loaded_ = false;
}

void DataInputDialog::RescanFolder() {
    folder_scan_path_ = folder_path_;
    folder_scan_ = data_input::ScanFolder(folder_path_);
}

void DataInputDialog::RenderFileSource() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    if (source_buf_[0] == '\0' && (file_path_[0] || folder_path_[0])) {
        const bool folder = data_input::UsesFolderPath(file_category_, text_layout_) && folder_path_[0];
        std::strncpy(source_buf_, folder ? folder_path_ : file_path_, sizeof(source_buf_) - 1);
        if (folder && folder_scan_path_ != folder_path_) RescanFolder();
    }

    // === 1. Source =========================================================
    SectionTitle("1. Source");
    RenderSourceChoice();
    if (CloudSourceChosen()) {
        RenderCloudSource();
        return;
    }
    FieldLabel("File or folder");
    const float browse_w = cyxwiz::ui::ButtonWidth("Browse file...", cyxwiz::ui::ButtonSize::Small) +
                           cyxwiz::ui::ButtonWidth("Browse folder...", cyxwiz::ui::ButtonSize::Small) +
                           ImGui::GetStyle().ItemSpacing.x * 2.0f;
    ImGui::SetNextItemWidth((std::max)(200.0f, ImGui::GetContentRegionAvail().x - browse_w));
    if (ImGui::InputText("##source", source_buf_, sizeof(source_buf_), ImGuiInputTextFlags_EnterReturnsTrue) ||
        ImGui::IsItemDeactivatedAfterEdit())
        SetSource(source_buf_);
    ImGui::SameLine();
    if (cyxwiz::ui::SecondaryButton("Browse file...", true, nullptr, cyxwiz::ui::ButtonSize::Small)) BrowseFile();
    ImGui::SameLine();
    if (cyxwiz::ui::SecondaryButton("Browse folder...", true, nullptr, cyxwiz::ui::ButtonSize::Small))
        BrowseFolder();

    const bool is_folder = folder_path_[0] != '\0' && file_path_[0] == '\0';
    const bool is_file = file_path_[0] != '\0';
    FieldLabel("Detected");
    if (is_file) {
        ImGui::Text("%s table", GetFileTypeName());
        std::string facts = file_size_ > 0 ? FormatBytes(file_size_) : std::string();
        if (!available_columns_.empty())
            facts += (facts.empty() ? "" : " \xC2\xB7 ") + std::to_string(available_columns_.size()) + " columns: " +
                     JoinFirst(available_columns_, 6);
        if (!facts.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", facts.c_str());
        }
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton(format_options_open_ ? "Hide format options" : "Format options"))
            format_options_open_ = !format_options_open_;
    } else if (is_folder) {
        const auto& scan = folder_scan_;
        const char* kind = scan.kind == data_input::FolderKind::Images ? "Image folder"
                           : scan.kind == data_input::FolderKind::Audio ? "Audio folder"
                           : scan.kind == data_input::FolderKind::Text  ? "Text folder"
                                                                         : "No images, audio or text files found";
        ImGui::TextUnformatted(kind);
        if (scan.kind != data_input::FolderKind::Unknown) {
            const char* noun = scan.kind == data_input::FolderKind::Images ? "images"
                               : scan.kind == data_input::FolderKind::Audio ? "audio files"
                                                                             : "text files";
            std::string facts = scan.class_names.empty()
                ? "flat folder, " + cyxwiz::data_input_view::FormatCount(scan.files) + " " + noun
                : std::to_string(scan.class_names.size()) + " class folders (" + JoinFirst(scan.class_names, 4) +
                      "), " + cyxwiz::data_input_view::FormatCount(scan.files) + " " + noun;
            if (scan.truncated) facts += " (counted up to " + cyxwiz::data_input_view::FormatCount(scan.files) + ")";
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", facts.c_str());
        }
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Scan again")) RescanFolder();
    } else {
        ImGui::TextColored(t.text_faint, "Choose a file (a table) or a folder (images, audio or text).");
    }
    if (is_file && format_options_open_) RenderFormatOptions();
    if (is_file && IsHdf5Source()) {
        SyncHdf5InspectorSource();
        hdf5_inspector_.RenderSettings();
        return;
    }

    // === 2. Use the rows as ================================================
    ImGui::Spacing();
    ImGui::Separator();
    if (is_file) {
        SectionTitle("2. Use the rows as");
        static const char* kUses[] = {"Table", "Text", "Token tagging", "Time series"};
        int use = sequence_tagging_ ? 2
                  : file_category_ == FileCategory::Text ? 1
                  : file_category_ == FileCategory::TimeSeries ? 3
                                                               : 0;
        if (cyxwiz::ui::SegmentedControl("##use_rows_as", kUses, 4, &use)) {
            sequence_tagging_ = use == 2;
            file_category_ = use == 1   ? FileCategory::Text
                             : use == 3 ? FileCategory::TimeSeries
                                        : FileCategory::Tabular;
            if (use == 1) text_layout_ = TextLayout::SingleFile;
            SyncHdf5InspectorSource();
            has_changes_ = true;
        }
        ImGui::TextColored(t.text_faint, "A table file can be used these four ways; a folder of images, audio or "
                                         "text gives its one use.");
        ImGui::Spacing();
        if (sequence_tagging_) RenderSequenceColumns();
        else if (file_category_ == FileCategory::Text) RenderTextColumns();
        else RenderLabelColumn();
        if (file_category_ == FileCategory::TimeSeries) RenderTimeColumn();
    } else if (is_folder && folder_scan_.kind != data_input::FolderKind::Unknown) {
        SectionTitle(folder_scan_.kind == data_input::FolderKind::Images  ? "2. Use as Images"
                     : folder_scan_.kind == data_input::FolderKind::Audio ? "2. Use as Audio"
                                                                          : "2. Use as Text");
        if (folder_scan_.kind == data_input::FolderKind::Images) RenderImageLabels();
        else if (folder_scan_.kind == data_input::FolderKind::Audio) RenderAudioLoading();
        else ImGui::TextColored(t.text_dim, "Each class folder's name is the label; each file is one sample.");
    }
    if (is_file || (is_folder && folder_scan_.kind != data_input::FolderKind::Unknown)) {
        ImGui::Spacing();
        if (file_category_ == FileCategory::Text || file_category_ == FileCategory::Image ||
            file_category_ == FileCategory::Audio)
            RenderPreprocessingNodes();
        RenderDatasetRoleNote();
    }
}

void DataInputDialog::RenderFormatOptions() {
    FieldLabel("Format");
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::BeginCombo("##format", GetFileTypeName())) {
        for (const auto& option : cyxwiz::data_input::kSourceFormats) {
            if (!option.executable) continue;
            const bool selected = detected_type_ == option.id;
            if (ImGui::Selectable(option.label, selected)) {
                detected_type_ = option.id;
                RefreshColumnList();
                has_changes_ = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("detected from the file; change it only when the extension is wrong");

    const auto file_type = data_input::FileTypeParam(detected_type_);
    if (!cyxwiz::data_input::IsExecutable(file_type)) {
        ImGui::TextDisabled("%s", cyxwiz::data_input::UnsupportedReason(
            file_type, cyxwiz::data_input::kHdf5BuildAvailable));
    }

    if (detected_type_ == 11) {
        FieldLabel("Members to read");
        if (ImGui::InputTextMultiline("##members", archive_member_, sizeof(archive_member_), ImVec2(-1, 100))) {
            has_changes_ = true;
            RefreshColumnList();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("One exact, case-sensitive path inside the ZIP per line, read in this order; no "
                            "wildcards. Each document keeps its original UTF-8 text and a source hash. Limits: "
                            "archive 64 MiB, member 16 MiB, selected text 64 MiB, 4,096 members, 10,000 archive "
                            "entries.");
        ImGui::PopTextWrapPos();
    }

    // CSV / TSV options (delimiter-based)
    if (detected_type_ <= 2) {
        FieldLabel("Delimiter");
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputText("##delim", custom_delimiter_, sizeof(custom_delimiter_))) {
            RefreshColumnList();
            has_changes_ = true;
        }
        ImGui::SameLine(0.0f, 24.0f);
        if (ImGui::Checkbox("First row is the header", &has_header_)) {
            RefreshColumnList();
            has_changes_ = true;
        }

        FieldLabel("Decimal separator");
        ImGui::SetNextItemWidth(110);
        const char* decimal_label = decimal_point_ == ',' ? "Comma (,)" : "Dot (.)";
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
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Controls numeric parsing only. It is separate from the field delimiter.");

        FieldLabel("Quote character");
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

        FieldLabel("Missing values");
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##missingtokens", missing_value_tokens_, sizeof(missing_value_tokens_))) {
            has_changes_ = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Comma-separated values treated as null before type inference.\n"
                              "Example: na,?,missing. Arrow standard null spellings remain enabled.");
        }
        ImGui::TextDisabled("Reads UTF-8 (a byte order mark is skipped).");
    }
}

void DataInputDialog::RenderTimeColumn() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    FieldLabel("Time column");
    if (available_columns_.empty()) {
        ImGui::TextColored(t.text_faint, "The columns show once the file is read (Preview or Apply).");
        return;
    }
    if (ColumnPicker("##timecol", time_column_, sizeof(time_column_), available_columns_, "(none: the file's order)")) {
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); ++i)
            if (available_columns_[i] == time_column_) selected_columns_[i] = true;
        has_changes_ = true;
    }
    if (time_column_[0] == '\0') {
        // A column named like a time: offer it.
        for (const auto& name : available_columns_) {
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "date" || lower == "time" || lower == "timestamp" || lower == "datetime" || lower == "ds") {
                ImGui::SameLine();
                ImGui::TextColored(t.text_dim, "'%s' looks like the time", name.c_str());
                ImGui::SameLine();
                if (cyxwiz::ui::LinkButton("Use it")) {
                    std::strncpy(time_column_, name.c_str(), sizeof(time_column_) - 1);
                    has_changes_ = true;
                }
                break;
            }
        }
    }
    ImGui::TextColored(t.text_dim, time_column_[0] != '\0'
                                       ? "Rows are kept in time order, for windowing and a chronological split."
                                       : "Rows are used in the file's order (windows and the split follow it).");
}

void DataInputDialog::RenderLabelColumn() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    FieldLabel("Label column");
    char label[256] = {};
    if (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(available_columns_.size()))
        std::strncpy(label, available_columns_[static_cast<size_t>(label_column_idx_)].c_str(), sizeof(label) - 1);
    if (available_columns_.empty()) {
        ImGui::TextColored(t.text_faint, "The columns show once the file is read (Preview or Apply).");
        return;
    }
    if (ColumnPicker("##labelcol", label, sizeof(label), available_columns_, "(none)")) {
        label_column_idx_ = -1;
        for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
            if (available_columns_[static_cast<size_t>(i)] == label) {
                label_column_idx_ = i;
                if (static_cast<size_t>(i) < selected_columns_.size()) selected_columns_[static_cast<size_t>(i)] = true;
            }
        }
        has_changes_ = true;
    }
    if (label_column_idx_ < 0) {
        // A column named like a label: offer it.
        for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
            std::string lower = available_columns_[static_cast<size_t>(i)];
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "label" || lower == "labels" || lower == "class" || lower == "target" || lower == "y") {
                ImGui::SameLine();
                const std::string& name = available_columns_[static_cast<size_t>(i)];
                ImGui::TextColored(t.text_dim, "'%s' looks like the label", name.c_str());
                ImGui::SameLine();
                if (cyxwiz::ui::LinkButton("Use it")) {
                    label_column_idx_ = i;
                    if (static_cast<size_t>(i) < selected_columns_.size())
                        selected_columns_[static_cast<size_t>(i)] = true;
                    has_changes_ = true;
                }
                break;
            }
        }
    }
    ImGui::TextColored(t.text_dim, "Every other column is an input feature.");
}

void DataInputDialog::RenderTextColumns() {
    FieldLabel("Text column");
    if (ColumnPicker("##textcol", text_column_, sizeof(text_column_), available_columns_, nullptr)) has_changes_ = true;
    FieldLabel("Label column");
    if (ColumnPicker("##textlabelcol", text_label_column_, sizeof(text_label_column_), available_columns_,
                     "(none: unlabeled)"))
        has_changes_ = true;
    ImGui::TextDisabled("CSV / TSV: a text column and a label column; JSON Lines: fields; TXT: one sample per line.");
}

void DataInputDialog::RenderImageLabels() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    FieldLabel("Labels from");
    static const char* kFrom[] = {"Class folders", "Flat folder + labels CSV"};
    int from = image_layout_ == ImageLayout::FlatWithCSV ? 1 : 0;
    if (cyxwiz::ui::SegmentedControl("##image_labels_from", kFrom, 2, &from)) {
        image_layout_ = from == 1 ? ImageLayout::FlatWithCSV : ImageLayout::ClassSubdirs;
        has_changes_ = true;
    }
    ImGui::TextColored(t.text_dim, "Class folders: root/cat/*.jpg, root/dog/*.jpg. Flat: one folder and a CSV "
                                   "mapping each file name to its label.");
    if (image_layout_ == ImageLayout::FlatWithCSV) {
        FieldLabel("Labels CSV");
        const float browse = cyxwiz::ui::ButtonWidth("Browse...", cyxwiz::ui::ButtonSize::Small) + 8.0f;
        ImGui::SetNextItemWidth((std::max)(200.0f, ImGui::GetContentRegionAvail().x - browse));
        if (ImGui::InputText("##labelscsv", labels_csv_, sizeof(labels_csv_))) has_changes_ = true;
        ImGui::SameLine();
        if (cyxwiz::ui::SecondaryButton("Browse...##imagecsv", true, nullptr, cyxwiz::ui::ButtonSize::Small)) {
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select Labels CSV", {{"CSV Files", "csv"}, {"All Files", "*"}},
                    labels_csv_[0] == '\0' ? nullptr : labels_csv_)) {
                std::strncpy(labels_csv_, selected->c_str(), sizeof(labels_csv_) - 1);
                labels_csv_[sizeof(labels_csv_) - 1] = '\0';
                has_changes_ = true;
            }
        }
    }
}

void DataInputDialog::RenderAudioLoading() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    FieldLabel("Labels from");
    static const char* kFrom[] = {"Class folders", "Flat folder + labels CSV"};
    int from = audio_layout_ == AudioLayout::FlatWithCSV ? 1 : 0;
    if (cyxwiz::ui::SegmentedControl("##audio_labels_from", kFrom, 2, &from)) {
        audio_layout_ = from == 1 ? AudioLayout::FlatWithCSV : AudioLayout::ClassSubdirs;
        has_changes_ = true;
    }
    if (audio_layout_ == AudioLayout::FlatWithCSV) {
        FieldLabel("Labels CSV");
        const float browse = cyxwiz::ui::ButtonWidth("Browse...", cyxwiz::ui::ButtonSize::Small) + 8.0f;
        ImGui::SetNextItemWidth((std::max)(200.0f, ImGui::GetContentRegionAvail().x - browse));
        if (ImGui::InputText("##audiocsv", audio_labels_csv_, sizeof(audio_labels_csv_))) has_changes_ = true;
        ImGui::SameLine();
        if (cyxwiz::ui::SecondaryButton("Browse...##audiocsv", true, nullptr, cyxwiz::ui::ButtonSize::Small)) {
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select Labels CSV", {{"CSV Files", "csv"}, {"All Files", "*"}},
                    audio_labels_csv_[0] == '\0' ? nullptr : audio_labels_csv_)) {
                std::strncpy(audio_labels_csv_, selected->c_str(), sizeof(audio_labels_csv_) - 1);
                audio_labels_csv_[sizeof(audio_labels_csv_) - 1] = '\0';
                has_changes_ = true;
            }
        }
        FieldLabel("File name column");
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiofnamecol", audio_filename_col_, sizeof(audio_filename_col_))) has_changes_ = true;
        ImGui::SameLine(0.0f, 24.0f);
        ImGui::TextColored(t.text_dim, "Label column");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiolabelcol", audio_label_col_, sizeof(audio_label_col_))) has_changes_ = true;
        ImGui::TextDisabled("Left empty, they are found by name: filename / file / id / call_id and "
                            "label / class / category.");
    }

    FieldLabel("Sample rate");
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##samplerate", &sample_rate_)) {
        if (sample_rate_ < 1) sample_rate_ = 1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Hz");
    FieldLabel("Length");
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputFloat("##duration", &duration_sec_, 0.5f, 1.0f, "%.1f")) {
        if (duration_sec_ < 0.1f) duration_sec_ = 0.1f;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("seconds, padded or cut");
    ImGui::TextColored(t.text_dim, "Audio is mixed to mono (channels averaged). WAV, FLAC, OGG and AIFF are read.");
}

void DataInputDialog::RenderDatasetRoleNote() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Training, validation and test roles come from the graph's Data Split, not from "
                                     "this node.");
    if (node_) {
        const auto role = node_->parameters.find("dataset_role");
        if (role != node_->parameters.end() && !role->second.empty()) {
            ImGui::TextColored(t.warning, "The old role hint '%s' is ignored; Apply removes it. Connect this input "
                                          "to the Data Split input it is for.", role->second.c_str());
        }
    }
    ImGui::PopTextWrapPos();
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
