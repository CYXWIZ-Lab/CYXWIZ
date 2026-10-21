#include "embedding_dialog.h"

#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include "../core/file_dialogs.h"
#include "../core/model_builder.h"
#include "../core/node_executors/word_embeddings_operator.h"
#include "../core/project_manager.h"
#include "../core/word_embeddings_presentation.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

namespace gui {

namespace {

namespace ui = cyxwiz::ui;
namespace fs = std::filesystem;

constexpr float kLabelWidth = 170.0f;
const char* const kDot = " \xC2\xB7 ";  // " · "

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void Dim(const std::string& text) { Wrapped(ui::CurrentTokens().text_dim, text); }
void Faint(const std::string& text) { Wrapped(ui::CurrentTokens().text_faint, text); }

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::SameLine(kLabelWidth);
}

bool BeginCard(const char* id) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::CurrentTokens().bg_raised);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, ui::CurrentTokens().rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ui::CurrentTokens().card_padding);
    return ImGui::BeginChild(id, ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

std::string Param(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

int IntParam(const MLNode& node, const char* key, int fallback) {
    try {
        const std::string value = Param(node, key);
        return value.empty() ? fallback : std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

std::string Shape(size_t rows, size_t cols) {
    return cyxwiz::GroupedCount(rows) + " \xC3\x97 " + cyxwiz::GroupedCount(cols);  // "5,000 × 100"
}

// A starter table from the node's shape (the builder kept from before TOFIX152).
bool WriteEmbeddingMatrix(const fs::path& path, int rows, int cols, int padding_idx, int init_mode,
                          std::string& error) {
    if (rows <= 0 || cols <= 0) {
        error = "Embedding rows and dimensions must be positive.";
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        error = "Could not open output file: " + path.string();
        return false;
    }
    std::mt19937 rng(42);
    const float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-limit, limit);
    out << "# cyxwiz_embedding rows=" << rows << " dim=" << cols << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float value = 0.0f;
            if (r == padding_idx) value = 0.0f;
            else if (init_mode == 2) value = (r == c) ? 1.0f : 0.0f;
            else if (init_mode == 1) value = uniform(rng);
            else value = normal(rng);
            if (c > 0) out << ' ';
            out << value;
        }
        out << '\n';
    }
    return true;
}

}  // namespace

EmbeddingDialog::EmbeddingDialog(MLNode* node) : NodeConfigDialog("Embedding Configuration", node) { LoadFromNode(); }

void EmbeddingDialog::LoadFromNode() {
    if (!node_) return;
    num_embeddings_ = std::max(2, IntParam(*node_, "num_embeddings", 10000));
    embedding_dim_ = std::max(1, IntParam(*node_, "embedding_dim", 256));
    padding_idx_ = IntParam(*node_, "padding_idx", -1);
    freeze_ = Param(*node_, "freeze") == "true";
    weights_file_ = Param(*node_, "weights_file");
    if (weights_file_.empty()) weights_file_ = Param(*node_, "embedding_weights_file");
    output_file_ = Param(*node_, "output_weights_file");
    max_norm_ = 0.0f;
    try {
        if (!Param(*node_, "max_norm").empty()) max_norm_ = std::stof(Param(*node_, "max_norm"));
    } catch (...) {
    }
    const std::string init = Param(*node_, "init_mode");
    init_mode_ = init == "uniform" ? 1 : init == "one_hot" ? 2 : 0;
    status_message_.clear();
    status_is_error_ = false;
    inspected_path_.clear();
    source_resolved_ = false;
}

void EmbeddingDialog::Apply() {
    if (!node_) return;
    if (num_embeddings_ < 2) num_embeddings_ = 2;
    if (embedding_dim_ < 1) embedding_dim_ = 1;
    if (padding_idx_ >= num_embeddings_) padding_idx_ = num_embeddings_ - 1;
    if (max_norm_ < 0.0f) max_norm_ = 0.0f;
    node_->parameters["num_embeddings"] = std::to_string(num_embeddings_);
    node_->parameters["embedding_dim"] = std::to_string(embedding_dim_);
    node_->parameters["padding_idx"] = std::to_string(padding_idx_);
    node_->parameters["max_norm"] = std::to_string(max_norm_);
    node_->parameters["freeze"] = freeze_ ? "true" : "false";
    node_->parameters["weights_file"] = source_ == Source::Random ? std::string() : weights_file_;
    node_->parameters.erase("embedding_weights_file");
    node_->parameters["output_weights_file"] = output_file_;
    node_->parameters["init_mode"] = init_mode_ == 1 ? "uniform" : init_mode_ == 2 ? "one_hot" : "normal";
    original_params_ = node_->parameters;
    has_changes_ = false;
    spdlog::info("EmbeddingDialog: Applied embedding settings");
}

void EmbeddingDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    has_changes_ = false;
}

std::string EmbeddingDialog::Resolve(const std::string& path) const {
    if (path.empty() || fs::path(path).is_absolute()) return path;
    const std::string& root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
    return root.empty() ? path : (fs::path(root) / path).lexically_normal().string();
}

std::string EmbeddingDialog::VectorsFileOf(const MLNode& vectors_node) const {
    const std::string set = Param(vectors_node, "vectors_path");
    if (!set.empty()) return set;
    // The run's default, project-relative (artifacts/word_embeddings/...).
    return (fs::path("artifacts") / cyxwiz::WordEmbeddingsOperator::DefaultVectorsFile(vectors_node.id, vectors_node.name))
        .generic_string();
}

// Which source the saved weights_file is: none, a Word Embeddings node's file, or a file.
void EmbeddingDialog::ResolveSource() {
    source_resolved_ = true;
    source_node_id_ = -1;
    if (weights_file_.empty()) {
        source_ = Source::Random;
        return;
    }
    source_ = Source::File;
    if (!node_editor_) return;
    const auto target = fs::path(Resolve(weights_file_)).lexically_normal();
    for (const auto& node : node_editor_->GetNodes()) {
        if (node.type != NodeType::WordEmbeddings) continue;
        if (fs::path(Resolve(VectorsFileOf(node))).lexically_normal() == target) {
            source_ = Source::Node;
            source_node_id_ = node.id;
            return;
        }
    }
}

void EmbeddingDialog::InspectFile() {
    inspected_path_ = weights_file_;
    file_rows_ = file_cols_ = 0;
    file_error_.clear();
    if (weights_file_.empty()) return;
    const std::string path = Resolve(weights_file_);
    if (!fs::exists(path)) {
        file_error_ = "Not there yet.";
        return;
    }
    if (!cyxwiz::InspectEmbeddingWeightsFile(path, file_rows_, file_cols_, file_error_)) file_rows_ = file_cols_ = 0;
}

void EmbeddingDialog::UseNode(const MLNode& vectors_node) {
    source_node_id_ = vectors_node.id;
    weights_file_ = VectorsFileOf(vectors_node);
    has_changes_ = true;
    InspectFile();
    // The vectors decide the shape: one row per token id, their width.
    if (file_rows_ > 0) {
        num_embeddings_ = static_cast<int>(file_rows_);
        embedding_dim_ = static_cast<int>(file_cols_);
    }
}

void EmbeddingDialog::RenderContent() {
    if (!source_resolved_) ResolveSource();
    if (inspected_path_ != weights_file_ && !ImGui::IsAnyItemActive()) InspectFile();
    if (ImGui::BeginTabBar("EmbeddingTabs")) {
        if (ImGui::BeginTabItem("Shape")) {
            RenderShapeTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Weights")) {
            RenderWeightsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Advanced")) {
            RenderAdvancedTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void EmbeddingDialog::RenderShapeTab() {
    ImGui::Spacing();
    // A Word Embeddings node's vectors set the shape while they are chosen.
    const bool from_node = source_ == Source::Node && file_rows_ > 0;
    ImGui::BeginDisabled(from_node);
    Label("Vocabulary size");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##num_embeddings", &num_embeddings_, 0, 0)) {
        if (num_embeddings_ < 2) num_embeddings_ = 2;
        has_changes_ = true;
    }
    ui::Tooltip("Number of token ids the embedding table can look up. This must match the tokenizer/vocabulary size.");
    Label("Embedding dimension");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##embedding_dim", &embedding_dim_, 0, 0)) {
        if (embedding_dim_ < 1) embedding_dim_ = 1;
        has_changes_ = true;
    }
    ui::Tooltip("Number of float features learned for each token id.");
    ImGui::EndDisabled();
    if (from_node) {
        ImGui::Indent(kLabelWidth);
        Faint("Set from the Word Embeddings node's vectors (Weights tab).");
        ImGui::Unindent(kLabelWidth);
    }
    Label("Padding index");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##padding_idx", &padding_idx_, 0, 0)) {
        if (padding_idx_ < -1) padding_idx_ = -1;
        has_changes_ = true;
    }
    ui::Tooltip("Token id treated as padding. Its vector is forced to zero so padded positions do not carry text meaning.");
    ImGui::Indent(kLabelWidth);
    Faint("-1 disables padding-zero behavior; 0 is typical for [PAD].");
    ImGui::Unindent(kLabelWidth);
}

void EmbeddingDialog::RenderWeightsTab() {
    const auto& t = ui::CurrentTokens();
    ImGui::Spacing();
    const char* labels[] = {"Random", "From a Word Embeddings node", "File"};
    int selected = static_cast<int>(source_);
    Label("Starting weights");
    if (ui::SegmentedControl("##weights_source", labels, 3, &selected)) {
        source_ = static_cast<Source>(selected);
        has_changes_ = true;
        if (source_ == Source::Random) {
            source_node_id_ = -1;
        } else if (source_ == Source::File && source_node_id_ >= 0) {
            source_node_id_ = -1;  // the same file, now chosen as a file
        }
    }
    ImGui::Dummy(ImVec2(0, t.space_sm));
    switch (source_) {
    case Source::Random:
        Dim("The table starts random and is learned in training.");
        break;
    case Source::Node: RenderNodeSource(); break;
    case Source::File: RenderFileSource(); break;
    }
    if (source_ != Source::Random) {
        ImGui::Dummy(ImVec2(0, t.space_sm));
        if (ImGui::Checkbox("Freeze loaded weights", &freeze_)) has_changes_ = true;
        ImGui::SameLine();
        Faint("(train the rest of the model; the table stays as loaded)");
    }
    ImGui::Dummy(ImVec2(0, t.space_md));
    RenderStarterBuilder();
}

void EmbeddingDialog::RenderNodeSource() {
    const auto& t = ui::CurrentTokens();
    std::vector<const MLNode*> candidates;
    if (node_editor_) {
        for (const auto& node : node_editor_->GetNodes())
            if (node.type == NodeType::WordEmbeddings) candidates.push_back(&node);
    }
    if (candidates.empty()) {
        Dim("No Word Embeddings node in this graph. Add one (Text nodes) to read GloVe / fastText / word2vec "
            "vectors or train word2vec, then choose it here.");
        return;
    }
    const MLNode* chosen = nullptr;
    for (const auto* node : candidates)
        if (node->id == source_node_id_) chosen = node;
    const auto caption = [&](const MLNode& node) {
        std::string text = node.name + " (#" + std::to_string(node.id) + ")";
        if (&node == chosen && file_rows_ > 0) text += kDot + Shape(file_rows_, file_cols_);
        return text;
    };
    Label("Node");
    ImGui::SetNextItemWidth((std::min)(420.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo("##vectors_node", chosen ? caption(*chosen).c_str() : "Choose a Word Embeddings node")) {
        for (const auto* node : candidates) {
            if (ImGui::Selectable(caption(*node).c_str(), node == chosen)) UseNode(*node);
        }
        ImGui::EndCombo();
    }
    if (!chosen) return;
    ImGui::Dummy(ImVec2(0, t.space_xs));
    if (BeginCard("##node_shape")) {
        if (file_rows_ > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.success);
            ImGui::TextUnformatted(ICON_FA_CIRCLE_CHECK);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            {
                ui::FontScope bold(ui::Font::Medium);
                ImGui::TextUnformatted("Shape set from the node");
            }
            Dim("Vocabulary size " + cyxwiz::GroupedCount(file_rows_) + " and dimension " +
                cyxwiz::GroupedCount(file_cols_) +
                " now come from the vectors (the Shape tab shows them read-only while this is chosen).");
            if (static_cast<size_t>(num_embeddings_) != file_rows_ || static_cast<size_t>(embedding_dim_) != file_cols_) {
                num_embeddings_ = static_cast<int>(file_rows_);
                embedding_dim_ = static_cast<int>(file_cols_);
                has_changes_ = true;
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, t.warning);
            ImGui::TextUnformatted(ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            {
                ui::FontScope bold(ui::Font::Medium);
                ImGui::TextUnformatted("The node has not saved its vectors yet");
            }
            Dim("Run it first (Train in its dialog, or run the graph); the shape is then read from " + weights_file_ +
                (file_error_.empty() || file_error_ == "Not there yet." ? std::string(".") : ": " + file_error_));
            if (ui::LinkButton("Check again")) InspectFile();
        }
    }
    EndCard();
}

void EmbeddingDialog::RenderFileSource() {
    const auto& t = ui::CurrentTokens();
    Label("Weights file");
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), "%s", weights_file_.c_str());
    const char* browse_label = ICON_FA_FOLDER_OPEN "##weights_browse";
    const float browse = ui::ButtonWidth(browse_label, ui::ButtonSize::Small);
    ImGui::SetNextItemWidth((std::max)(80.0f, ImGui::GetContentRegionAvail().x - browse - t.space_sm));
    if (ImGui::InputTextWithHint("##weights_file", "a matrix, or word vectors (.vec, .txt)", buffer, sizeof(buffer))) {
        weights_file_ = buffer;
        has_changes_ = true;
    }
    ImGui::SameLine(0.0f, t.space_sm);
    const bool clicked = ui::SecondaryButton(browse_label);
    ui::Tooltip("Choose a file");
    if (clicked) {
        const std::string start = Resolve(weights_file_);
        if (auto picked = cyxwiz::FileDialogs::OpenFile(
                "Choose embedding weights", {{"Embedding weights", "txt,vec,csv"}, {"All Files", "*"}},
                start.empty() ? nullptr : start.c_str())) {
            weights_file_ = *picked;
            has_changes_ = true;
        }
    }
    if (weights_file_.empty()) return;
    ImGui::Indent(kLabelWidth);
    if (!file_error_.empty()) {
        Wrapped(t.caution, std::string(ICON_FA_TRIANGLE_EXCLAMATION "  ") + file_error_);
    } else if (file_rows_ > 0) {
        const bool matches = static_cast<size_t>(num_embeddings_) == file_rows_ &&
                             static_cast<size_t>(embedding_dim_) == file_cols_;
        if (matches) {
            Wrapped(t.success, std::string(ICON_FA_CIRCLE_CHECK "  ") + Shape(file_rows_, file_cols_) +
                                   ", the node's shape.");
        } else {
            Wrapped(t.caution, std::string(ICON_FA_TRIANGLE_EXCLAMATION "  ") + Shape(file_rows_, file_cols_) +
                                   "; the node is " + Shape(num_embeddings_, embedding_dim_) + ".");
            if (ui::LinkButton("Use the file's shape")) {
                num_embeddings_ = static_cast<int>(file_rows_);
                embedding_dim_ = static_cast<int>(file_cols_);
                has_changes_ = true;
            }
        }
    }
    ImGui::Unindent(kLabelWidth);
}

void EmbeddingDialog::RenderStarterBuilder() {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    const bool open = ImGui::TreeNodeEx("Build starter weights", ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleColor();
    if (!open) return;
    Faint("Writes a table from this node's shape and uses it as the weights file.");
    const char* modes[] = {"Random normal", "Random uniform", "One-hot / truncated"};
    Label("Starter");
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::Combo("##init_mode", &init_mode_, modes, 3)) has_changes_ = true;
    Label("Save to");
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), "%s", output_file_.c_str());
    const char* browse_label = ICON_FA_FOLDER_OPEN "##output_browse";
    const float browse = ui::ButtonWidth(browse_label, ui::ButtonSize::Small);
    ImGui::SetNextItemWidth((std::max)(80.0f, ImGui::GetContentRegionAvail().x - browse - t.space_sm));
    if (ImGui::InputText("##output_weights_file", buffer, sizeof(buffer))) {
        output_file_ = buffer;
        has_changes_ = true;
    }
    ImGui::SameLine(0.0f, t.space_sm);
    const bool clicked = ui::SecondaryButton(browse_label);
    ui::Tooltip("Choose where to save");
    if (clicked) {
        const std::string start = Resolve(output_file_);
        if (auto picked = cyxwiz::FileDialogs::SaveFile("Save starter weights", {{"Embedding weights", "txt"}},
                                                         start.empty() ? nullptr : start.c_str(),
                                                         "embedding_weights.txt")) {
            output_file_ = *picked;
            has_changes_ = true;
        }
    }
    ImGui::Indent(kLabelWidth);
    if (ui::SecondaryButton("Build, save and use", !output_file_.empty(), "Choose where to save it"))
        BuildAndSaveWeights();
    if (!status_message_.empty()) Wrapped(status_is_error_ ? t.caution : t.success, status_message_);
    ImGui::Unindent(kLabelWidth);
    ImGui::TreePop();
}

bool EmbeddingDialog::BuildAndSaveWeights() {
    std::string error;
    if (!WriteEmbeddingMatrix(Resolve(output_file_), num_embeddings_, embedding_dim_, padding_idx_, init_mode_, error)) {
        status_message_ = error;
        status_is_error_ = true;
        return false;
    }
    weights_file_ = output_file_;
    source_ = Source::File;
    source_node_id_ = -1;
    status_message_ = "Wrote embedding matrix " + Shape(num_embeddings_, embedding_dim_) + " and chose it.";
    status_is_error_ = false;
    has_changes_ = true;
    return true;
}

void EmbeddingDialog::RenderAdvancedTab() {
    ImGui::Spacing();
    Label("Max norm");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputFloat("##max_norm", &max_norm_, 0.0f, 0.0f, "%.4f")) {
        if (max_norm_ < 0.0f) max_norm_ = 0.0f;
        has_changes_ = true;
    }
    ui::Tooltip("Optional length cap for each token vector during lookup. Use 0 to disable clipping.");
    ImGui::Indent(kLabelWidth);
    Faint("0 disables norm clipping. Positive values clip each vector during lookup.");
    ImGui::Unindent(kLabelWidth);
}

}  // namespace gui
