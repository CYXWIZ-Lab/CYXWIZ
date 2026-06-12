// Data output, loader, and split node dialog implementations.

#include "node_config_dialog.h"
#include "../core/worker_defaults.h"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>

namespace gui {

namespace {

void CopyToBuffer(char* destination,
                  std::size_t destination_size,
                  const std::string& value) {
    if (destination_size == 0) return;
    std::strncpy(destination, value.c_str(), destination_size - 1);
    destination[destination_size - 1] = '\0';
}

std::string LowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

} // namespace

// ==================== DataOutputDialog ====================

DataOutputDialog::DataOutputDialog(MLNode* node)
    : NodeConfigDialog("Data Output", node)
{
    if (node_) {
        auto file_path_it = node_->parameters.find("file_path");
        if (file_path_it == node_->parameters.end() ||
            file_path_it->second.empty()) {
            file_path_it = node_->parameters.find("path");
        }
        if (file_path_it != node_->parameters.end()) {
            CopyToBuffer(file_path_, sizeof(file_path_), file_path_it->second);
        }

        auto file_type_it = node_->parameters.find("file_type");
        if (file_type_it == node_->parameters.end() ||
            file_type_it->second.empty()) {
            file_type_it = node_->parameters.find("format");
        }
        if (file_type_it != node_->parameters.end()) {
            std::string type = LowerAscii(file_type_it->second);
            if (type == "csv") output_type_ = 0;
            else if (type == "parquet") output_type_ = 1;
        }
    }
}

void DataOutputDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;
    const char* types[] = {"csv", "parquet"};
    node_->parameters["file_type"] = types[output_type_];
    node_->parameters["overwrite"] = overwrite_ ? "true" : "false";
    node_->parameters["include_header"] = include_header_ ? "true" : "false";
    const char* compressions[] = {"none", "gzip", "snappy", "zstd"};
    node_->parameters["compression"] = compressions[compression_];
    node_->parameters["configured"] = "true";

    if (strlen(file_path_) > 0) {
        std::filesystem::path p(file_path_);
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

    const char* formats[] = {"CSV", "Parquet"};
    ImGui::Text("Format:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##format", &output_type_, formats, 2)) {
        has_changes_ = true;
    }

    if (output_type_ == 0) {
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

// ==================== DataLoaderDialog ====================

namespace {
void ReadIntParam(const std::map<std::string, std::string>& params,
                  const char* key, int& out) {
    auto it = params.find(key);
    if (it != params.end() && !it->second.empty()) {
        try { out = std::stoi(it->second); } catch (...) {}
    }
}

void ReadFloatParam(const std::map<std::string, std::string>& params,
                    const char* key, float& out) {
    auto it = params.find(key);
    if (it != params.end() && !it->second.empty()) {
        try { out = std::stof(it->second); } catch (...) {}
    }
}

void ReadBoolParam(const std::map<std::string, std::string>& params,
                   const char* key, bool& out) {
    auto it = params.find(key);
    if (it != params.end()) out = (it->second == "true");
}
}

DataLoaderDialog::DataLoaderDialog(MLNode* node)
    : NodeConfigDialog("Data Loader", node)
{
    if (node_) {
        ReadIntParam(node_->parameters, "batch_size", batch_size_);
        ReadBoolParam(node_->parameters, "shuffle", shuffle_);
        ReadBoolParam(node_->parameters, "drop_last", drop_last_);
        num_workers_ = cyxwiz::GetDefaultNumWorkers();
        ReadIntParam(node_->parameters, "num_workers", num_workers_);
        ReadIntParam(node_->parameters, "prefetch_factor", prefetch_factor_);
        if (prefetch_factor_ < 0) prefetch_factor_ = 0;
    }
}

void DataLoaderDialog::Apply() {
    if (!node_) return;
    node_->parameters["batch_size"] = std::to_string(batch_size_);
    node_->parameters["shuffle"] = shuffle_ ? "true" : "false";
    node_->parameters["drop_last"] = drop_last_ ? "true" : "false";
    node_->parameters["num_workers"] = std::to_string(num_workers_);
    node_->parameters["prefetch_factor"] = std::to_string(prefetch_factor_);
    node_->description = "batch=" + std::to_string(batch_size_) +
                          (shuffle_ ? ", shuffled" : ", ordered");
    has_changes_ = false;
}

void DataLoaderDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    // Re-initialize local UI state from the restored params so sliders match.
    batch_size_ = 32;
    shuffle_ = true;
    drop_last_ = false;
    num_workers_ = cyxwiz::GetDefaultNumWorkers();
    prefetch_factor_ = 2;
    ReadIntParam(original_params_, "batch_size", batch_size_);
    ReadBoolParam(original_params_, "shuffle", shuffle_);
    ReadBoolParam(original_params_, "drop_last", drop_last_);
    ReadIntParam(original_params_, "num_workers", num_workers_);
    ReadIntParam(original_params_, "prefetch_factor", prefetch_factor_);
    if (prefetch_factor_ < 0) prefetch_factor_ = 0;
    has_changes_ = false;
}

void DataLoaderDialog::RenderContent() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Batching");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Batch size:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##batch_size", &batch_size_)) {
        if (batch_size_ < 1) batch_size_ = 1;
        if (batch_size_ > 100000) batch_size_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(samples per gradient step)");

    ImGui::Spacing();
    if (ImGui::Checkbox("Shuffle each epoch", &shuffle_)) has_changes_ = true;
    ImGui::TextDisabled("  Reshuffles training samples at the start of every epoch.");

    ImGui::Spacing();
    if (ImGui::Checkbox("Drop last incomplete batch", &drop_last_)) has_changes_ = true;
    ImGui::TextDisabled("  Discard the final batch if it has fewer than batch_size samples.");

    ImGui::Spacing();
    ImGui::TextColored(accent, "Performance");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Worker threads:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##num_workers", &num_workers_)) {
        if (num_workers_ < 0) num_workers_ = 0;
        if (num_workers_ > 64) num_workers_ = 64;
        has_changes_ = true;
    }
    if (num_workers_ > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                           "  Used as synchronous per-batch workers where supported; Prefetch controls async queuing.");
    } else {
        ImGui::TextDisabled("  0 = load batches on the training thread (current behavior).");
    }

    ImGui::Spacing();
    ImGui::Text("Prefetch:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##prefetch_factor", &prefetch_factor_)) {
        if (prefetch_factor_ < 0) prefetch_factor_ = 0;
        if (prefetch_factor_ > 64) prefetch_factor_ = 64;
        has_changes_ = true;
    }
    if (prefetch_factor_ > 0) {
        ImGui::TextDisabled("  Bounded async queue depth for Arrow/Parquet batchers.");
    } else {
        ImGui::TextDisabled("  0 = disable async prefetch.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(
        "This node is the single source of truth for batching. If batch_size is\n"
        "also set on the optimizer node, the DataLoader value wins and a warning\n"
        "is logged at training start.");
}

// ==================== DataSplitDialog ====================

DataSplitDialog::DataSplitDialog(MLNode* node)
    : NodeConfigDialog("Data Split", node)
{
    if (node_) {
        ReadFloatParam(node_->parameters, "train_ratio", train_ratio_);
        ReadFloatParam(node_->parameters, "val_ratio", val_ratio_);
        ReadFloatParam(node_->parameters, "test_ratio", test_ratio_);
        ReadIntParam(node_->parameters, "seed", seed_);
        ReadBoolParam(node_->parameters, "stratified", stratified_);
    }
}

void DataSplitDialog::Apply() {
    if (!node_) return;
    // Guard against all-zero sliders, then normalize to 1.0 if the user drifted.
    float sum = train_ratio_ + val_ratio_ + test_ratio_;
    if (sum <= 0.001f) {
        train_ratio_ = 0.8f;
        val_ratio_ = 0.1f;
        test_ratio_ = 0.1f;
    } else if (std::abs(sum - 1.0f) > 0.01f) {
        train_ratio_ /= sum;
        val_ratio_ /= sum;
        test_ratio_ /= sum;
    }
    node_->parameters["train_ratio"] = std::to_string(train_ratio_);
    node_->parameters["val_ratio"] = std::to_string(val_ratio_);
    node_->parameters["test_ratio"] = std::to_string(test_ratio_);
    node_->parameters["seed"] = std::to_string(seed_);
    node_->parameters["stratified"] = stratified_ ? "true" : "false";
    char buf[128];
    snprintf(buf, sizeof(buf), "%.0f/%.0f/%.0f",
             train_ratio_ * 100.0f, val_ratio_ * 100.0f, test_ratio_ * 100.0f);
    node_->description = buf;
    has_changes_ = false;
}

void DataSplitDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    // Re-initialize local UI state from the restored params so sliders match.
    train_ratio_ = 0.8f;
    val_ratio_ = 0.1f;
    test_ratio_ = 0.1f;
    seed_ = 42;
    stratified_ = true;
    ReadFloatParam(original_params_, "train_ratio", train_ratio_);
    ReadFloatParam(original_params_, "val_ratio", val_ratio_);
    ReadFloatParam(original_params_, "test_ratio", test_ratio_);
    ReadIntParam(original_params_, "seed", seed_);
    ReadBoolParam(original_params_, "stratified", stratified_);
    has_changes_ = false;
}

void DataSplitDialog::RenderContent() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Split Ratios");
    ImGui::Separator();
    ImGui::Spacing();

    bool changed = false;
    ImGui::Text("Train:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##train", &train_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    ImGui::Text("Validation:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##val", &val_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    ImGui::Text("Test:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##test", &test_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    if (changed) has_changes_ = true;

    float sum = train_ratio_ + val_ratio_ + test_ratio_;
    ImGui::Spacing();
    if (std::abs(sum - 1.0f) > 0.01f) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                           "Sum = %.2f (will be normalized to 1.00 on Apply)", sum);
    } else {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Sum = %.2f", sum);
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Options");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Seed:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::InputInt("##seed", &seed_)) has_changes_ = true;
    ImGui::TextDisabled("  Controls the shuffled order used when partitioning.");

    ImGui::Spacing();
    if (ImGui::Checkbox("Stratified split", &stratified_)) has_changes_ = true;
    ImGui::TextDisabled("  Preserve class distribution across splits (classification only).");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(
        "If no DataSplit node is in the graph, training uses defaults (80/10/10).\n"
        "This node is the single source of truth for dataset partitioning.");
}

} // namespace gui
