#include "word_embeddings_dialog.h"

#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include "../core/arrow_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/data_registry.h"
#include "../core/file_dialogs.h"
#include "../core/node_executors/word_embeddings_operator.h"
#include "../core/node_metadata_registry.h"
#include "../core/project_manager.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tokenizer.h>

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace gui {

namespace {

namespace ui = cyxwiz::ui;
namespace fs = std::filesystem;

constexpr float kLabelWidth = 170.0f;
constexpr float kSmallField = 110.0f;
constexpr int kNearest = 10;
constexpr size_t kMissingListed = 2000;  // chips on the Vocabulary tab
const char* const kDot = " \xC2\xB7 ";  // " · "

enum Tab { kSourceTab = 0, kVocabularyTab, kNearestTab, kOutputsTab };

// A tinted card: one step off the window, no border, as tall as its content.
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

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void Dim(const std::string& text) { Wrapped(ui::CurrentTokens().text_dim, text); }
void Faint(const std::string& text) { Wrapped(ui::CurrentTokens().text_faint, text); }

void Heading(const std::string& text) {
    ui::FontScope bold(ui::Font::Medium);
    ImGui::TextUnformatted(text.c_str());
}

// A rounded pill with coloured text, wrapping to the next line when the row
// is full. `first` starts the row.
void Chip(const std::string& text, const ImVec4& colour, bool first) {
    const auto& t = ui::CurrentTokens();
    const ImVec2 pad(10.0f, 2.0f);
    const ImVec2 size(ImGui::CalcTextSize(text.c_str()).x + pad.x * 2.0f, ImGui::GetTextLineHeight() + pad.y * 2.0f);
    if (!first) {
        ImGui::SameLine(0.0f, t.space_md);
        if (ImGui::GetCursorPosX() + size.x > ImGui::GetContentRegionMax().x) ImGui::NewLine();
    }
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ui::ToU32(t.bg_input), size.y * 0.5f);
    draw->AddText(ImVec2(at.x + pad.x, at.y + pad.y), ui::ToU32(colour), text.c_str());
    ImGui::Dummy(size);
}

// A token as the font draws it; one it cannot draw (an emoji) by its code points.
std::string Printable(const std::string& word) {
    ImFont* font = ImGui::GetFont();
    for (const char32_t c : cyxwiz::Utf8CodePoints(word)) {
        if (c > 0xFFFF || font->FindGlyphNoFallback(static_cast<ImWchar>(c)) == nullptr)
            return cyxwiz::CodePointNames(word);
    }
    return word;
}

// The rows of a table: word, v_0 .. v_{d-1}.
std::shared_ptr<const cyxwiz::WordVectors> VectorsFromTable(const std::shared_ptr<arrow::Table>& table) {
    if (!table || table->num_columns() < 2) return nullptr;
    const auto words = table->GetColumnByName("word");
    if (!words) return nullptr;
    auto vectors = std::make_shared<cyxwiz::WordVectors>();
    vectors->dim = table->num_columns() - 1;
    const int64_t rows = table->num_rows();
    vectors->words.reserve(static_cast<size_t>(rows));
    for (const auto& chunk : words->chunks()) {
        const auto strings = std::dynamic_pointer_cast<arrow::StringArray>(chunk);
        if (!strings) return nullptr;
        for (int64_t i = 0; i < strings->length(); ++i) vectors->words.push_back(strings->GetString(i));
    }
    vectors->values.assign(static_cast<size_t>(rows) * vectors->dim, 0.0f);
    for (int d = 0; d < vectors->dim; ++d) {
        const auto column = table->GetColumnByName("v_" + std::to_string(d));
        if (!column) return nullptr;
        int64_t row = 0;
        for (const auto& chunk : column->chunks()) {
            const auto floats = std::dynamic_pointer_cast<arrow::FloatArray>(chunk);
            if (!floats) return nullptr;
            for (int64_t i = 0; i < floats->length(); ++i, ++row)
                vectors->values[static_cast<size_t>(row) * vectors->dim + d] = floats->Value(i);
        }
    }
    vectors->success = true;
    return vectors;
}

bool TaskLive(uint64_t id) {
    if (id == 0) return false;
    const auto task = cyxwiz::AsyncTaskManager::Instance().GetTask(id);
    if (!task) return false;
    const auto state = task->GetState();
    return state == cyxwiz::TaskState::Pending || state == cyxwiz::TaskState::Running;
}

std::string FileName(const std::string& path) {
    const std::string name = fs::path(path).filename().string();
    return name.empty() ? path : name;
}

}  // namespace

WordEmbeddingsDialog::WordEmbeddingsDialog(MLNode* node)
    : NodeConfigDialog("Word Embeddings###word_embeddings_" + std::to_string(node ? node->id : 0), node) {
    if (node_) params_ = node_->parameters;
}

std::string WordEmbeddingsDialog::Get(const std::string& key) const {
    const auto it = params_.find(key);
    if (it != params_.end()) return it->second;
    // The node's metadata default (a graph saved before the parameter existed).
    if (const auto* meta = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(NodeType::WordEmbeddings)) {
        for (const auto& param : meta->parameters) {
            if (param.name == key) return param.default_value;
        }
    }
    return {};
}

void WordEmbeddingsDialog::Set(const std::string& key, const std::string& value) {
    if (Get(key) == value) return;
    params_[key] = value;
    has_changes_ = true;
}

// A relative path is the project's, as the run reads it.
std::string WordEmbeddingsDialog::Resolve(const std::string& path) const {
    if (path.empty() || fs::path(path).is_absolute()) return path;
    const std::string& root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
    return root.empty() ? path : (fs::path(root) / path).lexically_normal().string();
}

void WordEmbeddingsDialog::Apply() {
    if (!node_) return;
    for (const auto& [key, value] : params_) node_->parameters[key] = value;
    original_params_ = node_->parameters;
    has_changes_ = false;
}

void WordEmbeddingsDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    params_ = original_params_;
    has_changes_ = false;
}

// The step's input table (when it has run) gives the text column choices.
void WordEmbeddingsDialog::RefreshInputs() {
    inputs_resolved_ = true;
    input_name_.clear();
    input_columns_.clear();
    input_rows_ = -1;
    if (!node_editor_ || !node_) return;
    const auto context = node_editor_->GetDataStepContext(node_->id);
    if (context.inputs.empty()) return;
    input_name_ = context.inputs.front().name;
    const auto& facts = node_editor_->GetNodeRunFacts();
    const auto it = facts.find(context.inputs.front().id);
    if (it == facts.end() || it->second.output_dataset.empty()) return;
    const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(it->second.output_dataset);
    const auto table = dataset ? dataset->GetArrowTable() : nullptr;
    if (!table) return;
    input_rows_ = table->num_rows();
    for (const auto& field : table->schema()->fields()) {
        const auto id = field->type()->id();
        if (id == arrow::Type::STRING || id == arrow::Type::LARGE_STRING) input_columns_.push_back(field->name());
    }
}

// A Text Tokenizer's vocabulary file in the graph: the default to line up with.
void WordEmbeddingsDialog::FindTokenizerVocabulary() {
    tokenizer_vocab_.clear();
    if (!node_editor_) return;
    for (const auto& node : node_editor_->GetNodes()) {
        if (node.type != NodeType::TextTokenizer) continue;
        const auto it = node.parameters.find("vocab_file");
        if (it != node.parameters.end() && !it->second.empty()) {
            tokenizer_vocab_ = it->second;
            return;
        }
    }
}

std::string WordEmbeddingsDialog::ReadKey() const {
    std::string key;
    for (const char* k : {"vectors_file", "max_words", "vocab_file", "lowercase_fallback", "missing", "unknown", "seed"})
        key += Get(k) + '\n';
    return key;
}

void WordEmbeddingsDialog::StartRead() {
    read_key_ = ReadKey();
    read_error_.clear();
    read_summary_.clear();
    read_vectors_.reset();
    coverage_.reset();
    missing_words_.clear();
    vocab_size_ = 0;
    nearest_.clear();
    const std::string file = Resolve(Get("vectors_file"));
    if (file.empty()) return;
    const std::string vocab = Resolve(Get("vocab_file"));
    size_t max_words = 0;
    try {
        max_words = static_cast<size_t>(std::max(0, std::stoi(Get("max_words"))));
    } catch (...) {
    }
    cyxwiz::WordAlignmentOptions options;
    options.lowercase_fallback = Get("lowercase_fallback") != "false";
    options.missing = Get("missing") == "random" ? "random" : "zeros";
    options.unknown = Get("unknown") == "missing" ? "missing" : "mean";
    try {
        options.seed = static_cast<unsigned>(std::max(0, std::stoi(Get("seed"))));
    } catch (...) {
    }
    read_ = std::make_shared<ReadJob>();
    const auto job = read_;
    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Word Embeddings: read " + FileName(file), [job, file, vocab, max_words, options](cyxwiz::LambdaTask& task) {
            try {
                // The alignment runs where training runs: a worker starts on
                // ArrayFire's default backend, not the selected device.
                cyxwiz::Device::ActivateProcessDeviceOnThisThread();
                task.ReportProgress(0.1f, "Reading " + FileName(file));
                std::string first_line;
                {
                    std::ifstream in(file, std::ios::binary);
                    std::getline(in, first_line);
                }
                const auto start = std::chrono::steady_clock::now();
                auto vectors = std::make_shared<cyxwiz::WordVectors>(cyxwiz::WordEmbeddings::Load(file, max_words));
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (!vectors->success) throw std::runtime_error(vectors->error_message);
                const std::string summary = cyxwiz::VectorFileSummary(cyxwiz::VectorFileFormat(file, first_line),
                                                                      vectors->words.size(), vectors->dim, seconds);
                std::optional<cyxwiz::WordCoverageView> coverage;
                std::vector<std::string> missing;
                size_t vocab_size = 0;
                if (!vocab.empty()) {
                    task.ReportProgress(0.8f, "Lining up with " + FileName(vocab));
                    cyxwiz::Vocabulary vocabulary;
                    if (!vocabulary.LoadFromFile(vocab)) throw std::runtime_error("cannot read the vocabulary " + vocab);
                    auto aligned_options = options;
                    aligned_options.pad_id = vocabulary.PadIndex();
                    aligned_options.unknown_id = vocabulary.UnkIndex();
                    const auto& words = vocabulary.GetWords();
                    const auto aligned = cyxwiz::WordEmbeddings::Align(*vectors, words, aligned_options);
                    if (!aligned.success) throw std::runtime_error(aligned.error_message);
                    const auto word_at = [&](int id) {
                        return id >= 0 && static_cast<size_t>(id) < words.size() ? words[static_cast<size_t>(id)]
                                                                                 : std::string();
                    };
                    coverage = cyxwiz::BuildWordCoverageView(aligned.exact, aligned.lowercase, aligned.missing,
                                                             aligned.missing_words, word_at(aligned_options.pad_id),
                                                             word_at(aligned_options.unknown_id), options.missing,
                                                             options.unknown);
                    missing = aligned.missing_words;
                    vocab_size = words.size();
                }
                std::lock_guard<std::mutex> lock(job->mutex);
                job->summary = summary;
                job->vectors = std::move(vectors);
                job->coverage = std::move(coverage);
                job->missing_words = std::move(missing);
                job->vocab_size = vocab_size;
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->error = e.what();
                }
                job->finished = true;
                throw;
            }
            job->finished = true;
        });
}

void WordEmbeddingsDialog::PollRead() {
    if (!read_ || !read_->finished.load()) return;
    std::lock_guard<std::mutex> lock(read_->mutex);
    read_error_ = read_->error;
    read_summary_ = read_->summary;
    read_vectors_ = read_->vectors;
    coverage_ = read_->coverage;
    missing_words_ = std::move(read_->missing_words);
    vocab_size_ = read_->vocab_size;
    read_.reset();
}

void WordEmbeddingsDialog::StartTraining() {
    if (!node_editor_ || !node_) {
        run_error_ = "Open the Word Embeddings dialog from its node in the graph.";
        return;
    }
    if (node_editor_->IsDataPipelineRunning()) {
        run_error_ = "A pipeline run is in progress; train when it ends.";
        return;
    }
    Apply();
    run_error_.clear();
    run_outcome_.clear();
    run_vectors_.reset();
    run_finished_at_.clear();
    nearest_.clear();
    run_started_ = node_editor_->ExecuteDataPipelineThrough(node_->id, false);
    run_task_ = run_started_ ? node_editor_->GetDataPipelineTaskId() : 0;
    if (!run_started_) run_error_ = "The step was not run; the notice says why.";
}

// The step's latest run in this session, whoever started it.
void WordEmbeddingsDialog::PollRun() {
    if (!node_editor_ || !node_) return;
    if (run_started_ && TaskLive(run_task_)) return;
    const auto& facts = node_editor_->GetNodeRunFacts();
    const auto it = facts.find(node_->id);
    if (it == facts.end() || (it->second.finished_at.empty() && it->second.error.empty())) return;
    const std::string stamp = it->second.finished_at + '\n' + it->second.error;
    if (stamp == run_finished_at_ || node_editor_->IsDataPipelineRunning()) return;
    run_finished_at_ = stamp;
    run_error_ = it->second.error;
    run_outcome_ = it->second.outcome;
    run_seconds_ = it->second.seconds;
    run_dataset_ = it->second.output_dataset;
    run_started_ = false;
    LoadRunVectors();
    inputs_resolved_ = false;  // the inputs ran too
}

void WordEmbeddingsDialog::LoadRunVectors() {
    run_vectors_.reset();
    if (!run_error_.empty() || run_dataset_.empty()) return;
    const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(run_dataset_);
    run_vectors_ = VectorsFromTable(dataset ? dataset->GetArrowTable() : nullptr);
}

void WordEmbeddingsDialog::RenderContent() {
    PollRead();
    PollRun();
    if (!inputs_resolved_ && (!node_editor_ || !node_editor_->IsDataPipelineRunning())) {
        RefreshInputs();
        FindTokenizerVocabulary();
        if (Get("vocab_file").empty() && !tokenizer_vocab_.empty()) Set("vocab_file", tokenizer_vocab_);
    }
    // Read a pretrained file again when what it depends on changed (not while
    // a field is being typed in).
    if (!Training() && !read_ && !ImGui::IsAnyItemActive() && ReadKey() != read_key_) StartRead();

    if (!ImGui::BeginTabBar("##word_embeddings_tabs")) return;
    const char* names[] = {"Source", "Vocabulary", "Nearest words", "Outputs"};
    for (int tab = 0; tab < 4; ++tab) {
        const ImGuiTabItemFlags flags = tab_request_ == tab ? ImGuiTabItemFlags_SetSelected : 0;
        if (!ImGui::BeginTabItem(names[tab], nullptr, flags)) continue;
        ImGui::Dummy(ImVec2(0, ui::CurrentTokens().space_sm));
        ImGui::BeginChild("##tab_body", ImVec2(0, 0), ImGuiChildFlags_None);
        switch (tab) {
        case kSourceTab: RenderSource(); break;
        case kVocabularyTab: RenderVocabulary(); break;
        case kNearestTab: RenderNearest(); break;
        default: RenderOutputs(); break;
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
    tab_request_ = -1;
    ImGui::EndTabBar();
}

void WordEmbeddingsDialog::Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::SameLine(kLabelWidth);
}

bool WordEmbeddingsDialog::FileField(const char* id, const char* key, const char* title, bool save, const char* hint) {
    const auto& t = ui::CurrentTokens();
    std::string value = Get(key);
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), "%s", value.c_str());
    const std::string browse_label = std::string(ICON_FA_FOLDER_OPEN "##") + id;
    const float browse = ui::ButtonWidth(browse_label.c_str(), ui::ButtonSize::Small);
    ImGui::SetNextItemWidth((std::max)(80.0f, ImGui::GetContentRegionAvail().x - browse - t.space_sm));
    bool changed = false;
    if (ImGui::InputTextWithHint(id, hint ? hint : "", buffer, sizeof(buffer))) {
        Set(key, buffer);
        changed = true;
    }
    if (!value.empty() && ImGui::IsItemHovered()) ui::Tooltip(value.c_str());
    ImGui::SameLine(0.0f, t.space_sm);
    const bool clicked = ui::SecondaryButton(browse_label.c_str());
    ui::Tooltip("Choose a file");
    if (clicked) {
        const std::string start = Resolve(value);
        const char* at = start.empty() ? nullptr : start.c_str();
        std::optional<std::string> picked;
        if (save) {
            picked = cyxwiz::FileDialogs::SaveFile(title, {{"Word vectors (.vec)", "vec"}}, at, "word_vectors.vec");
        } else if (std::strcmp(key, "vocab_file") == 0) {
            picked = cyxwiz::FileDialogs::OpenFile(title, {{"Vocabulary", "txt"}, {"All Files", "*"}}, at);
        } else {
            picked = cyxwiz::FileDialogs::OpenFile(title, {{"Word vectors", "txt,vec,bin"}, {"All Files", "*"}}, at);
        }
        if (picked) {
            Set(key, *picked);
            changed = true;
        }
    }
    return changed;
}

bool WordEmbeddingsDialog::IntField(const char* id, const char* key, float width) {
    int value = 0;
    try {
        value = std::stoi(Get(key));
    } catch (...) {
    }
    ImGui::SetNextItemWidth(width);
    if (!ImGui::InputInt(id, &value, 0, 0)) return false;
    Set(key, std::to_string(value));
    return true;
}

bool WordEmbeddingsDialog::FloatField(const char* id, const char* key, float width) {
    double value = 0.0;
    try {
        value = std::stod(Get(key));
    } catch (...) {
    }
    ImGui::SetNextItemWidth(width);
    if (!ImGui::InputDouble(id, &value, 0.0, 0.0, "%g")) return false;
    char text[32];
    std::snprintf(text, sizeof(text), "%g", value);
    Set(key, text);
    return true;
}

bool WordEmbeddingsDialog::Choice(const char* id, const char* key, const char* const* labels, const char* const* values,
                                  int count) {
    const std::string current = Get(key);
    int selected = 0;
    for (int i = 0; i < count; ++i) {
        if (current == values[i]) selected = i;
    }
    if (!ui::SegmentedControl(id, labels, count, &selected)) return false;
    Set(key, values[selected]);
    return true;
}

void WordEmbeddingsDialog::RenderSource() {
    const char* labels[] = {"Pretrained file", "Train word2vec", "Train GloVe"};
    const char* values[] = {"pretrained", "word2vec", "glove"};
    Label("Source");
    Choice("##source", "source", labels, values, 3);
    ImGui::Dummy(ImVec2(0, ui::CurrentTokens().space_sm));
    if (Training()) {
        RenderTraining();
    } else {
        RenderPretrained();
    }
}

void WordEmbeddingsDialog::RenderPretrained() {
    const auto& t = ui::CurrentTokens();
    Label("Vector file");
    FileField("##vectors_file", "vectors_file", "Choose word vectors", false);
    Label("Read");
    if (Get("vectors_file").empty()) {
        Faint("Choose a GloVe, fastText or word2vec file.");
    } else if (read_) {
        Wrapped(t.accent_text, std::string(ICON_FA_SPINNER) + "  Reading...");
    } else if (!read_error_.empty()) {
        Wrapped(t.caution, std::string(ICON_FA_TRIANGLE_EXCLAMATION) + "  " + read_error_);
    } else {
        Dim(read_summary_);
    }
    Label("Read at most");
    IntField("##max_words", "max_words", kSmallField);
    ImGui::Indent(kLabelWidth);
    Faint("0 reads every word. Formats: GloVe .txt, fastText / word2vec .vec, word2vec binary .bin.");
    ImGui::Unindent(kLabelWidth);
    ImGui::Dummy(ImVec2(0, t.space_sm));
    RenderCoverage(false);
    ImGui::Dummy(ImVec2(0, t.space_sm));

    if (ImGui::BeginTable("##alignment", 2, ImGuiTableFlags_SizingStretchSame)) {
        const char* missing_labels[] = {"Zeros", "Random"};
        const char* missing_values[] = {"zeros", "random"};
        const char* unknown_labels[] = {"Mean vector", "As missing"};
        const char* unknown_values[] = {"mean", "missing"};
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        bool lowercase = Get("lowercase_fallback") != "false";
        if (ImGui::Checkbox("Match a word in lowercase when not found as written", &lowercase))
            Set("lowercase_fallback", lowercase ? "true" : "false");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted("Missing words");
        ImGui::PopStyleColor();
        ImGui::SameLine(120.0f);
        Choice("##missing", "missing", missing_labels, missing_values, 2);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted("Unknown token");
        ImGui::PopStyleColor();
        ImGui::SameLine(120.0f);
        Choice("##unknown", "unknown", unknown_labels, unknown_values, 2);
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted("Random seed");
        ImGui::PopStyleColor();
        ImGui::SameLine(120.0f);
        IntField("##seed", "seed", kSmallField);
        ImGui::EndTable();
    }
}

void WordEmbeddingsDialog::RenderCoverage(bool full_list) {
    const auto& t = ui::CurrentTokens();
    if (Training()) return;
    if (Get("vocab_file").empty()) {
        Faint("No vocabulary: the vectors keep the file's own words, in its order. Choose one on the Vocabulary tab "
              "to line them up with the Text Tokenizer's token ids.");
        return;
    }
    if (!coverage_) return;
    const auto& view = *coverage_;
    if (BeginCard("##coverage")) {
        Heading("Lined up with the vocabulary");
        const std::string vocab = Get("vocab_file");
        const bool from_tokenizer = !tokenizer_vocab_.empty() && vocab == tokenizer_vocab_;
        Dim(std::string(from_tokenizer ? "From the Text Tokenizer: " : "Vocabulary file: ") + FileName(vocab) + kDot +
            cyxwiz::GroupedCount(vocab_size_) + " tokens");
        // The bar: found, found in lowercase, missing.
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        const float height = 8.0f;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.bg_input), 4.0f);
        const float found = width * view.found_share;
        const float lower = width * view.lowercase_share;
        draw->AddRectFilled(at, ImVec2(at.x + found + lower, at.y + height), ui::ToU32(t.info), 4.0f,
                            ImDrawFlags_RoundCornersLeft);
        draw->AddRectFilled(at, ImVec2(at.x + found, at.y + height), ui::ToU32(t.success), 4.0f,
                            ImDrawFlags_RoundCornersLeft);
        if (found + lower < width)
            draw->AddRectFilled(ImVec2(at.x + found + lower, at.y), ImVec2(at.x + width, at.y + height),
                                ui::ToU32(t.caution), 4.0f, ImDrawFlags_RoundCornersRight);
        ImGui::Dummy(ImVec2(width, height));
        Chip(std::string(ICON_FA_CIRCLE_CHECK " ") + view.found, t.success, true);
        Chip(view.lowercase, t.text, false);
        Chip(std::string(ICON_FA_TRIANGLE_EXCLAMATION " ") + view.missing, t.caution, false);
        if (!view.special.empty()) Chip(view.special, t.info, false);
        if (!view.missing_preview.empty()) {
            if (full_list) {
                std::string heading = view.missing + ", in vocabulary order";
                if (!view.missing_kind.empty()) heading += " (" + view.missing_kind + ")";
                Dim(heading + ":");
                ImGui::BeginChild("##missing_words", ImVec2(0, 220.0f), ImGuiChildFlags_None);
                const size_t shown = (std::min)(missing_words_.size(), kMissingListed);
                for (size_t i = 0; i < shown; ++i) Chip(Printable(missing_words_[i]), t.text, i == 0);
                if (shown < missing_words_.size())
                    Faint("and " + cyxwiz::GroupedCount(missing_words_.size() - shown) + " more.");
                ImGui::EndChild();
            } else {
                Dim(view.missing_preview);
                for (size_t i = 0; i < view.preview_words.size(); ++i)
                    Chip(Printable(view.preview_words[i]), t.text, i == 0);
                if (missing_words_.size() > view.preview_words.size()) Chip("...", t.text_dim, false);
                if (!view.missing_kind.empty()) Dim("Mostly " + view.missing_kind.substr(std::strlen("mostly ")) + ".");
                if (ui::LinkButton("Show all")) tab_request_ = kVocabularyTab;
            }
        }
    }
    EndCard();
}

void WordEmbeddingsDialog::RenderTraining() {
    const auto& t = ui::CurrentTokens();
    Label("Text column");
    const std::string column = Get("text_col");
    if (!input_columns_.empty()) {
        ImGui::SetNextItemWidth((std::min)(320.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::BeginCombo("##text_col", column.empty() ? "Choose a column" : column.c_str())) {
            for (const auto& name : input_columns_) {
                if (ImGui::Selectable(name.c_str(), name == column)) Set("text_col", name);
            }
            ImGui::EndCombo();
        }
    } else {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%s", column.c_str());
        ImGui::SetNextItemWidth((std::min)(320.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::InputTextWithHint("##text_col", "column name", buffer, sizeof(buffer))) Set("text_col", buffer);
        ImGui::SameLine(0.0f, t.space_md);
        const bool can_load = node_editor_ && !input_name_.empty() && !node_editor_->IsDataPipelineRunning();
        if (ui::SecondaryButton("Load inputs", can_load,
                                input_name_.empty() ? "Connect the table with the text to the Text pin"
                                                    : "A pipeline run is in progress")) {
            node_editor_->ExecuteDataPipelineThrough(node_->id, false, false);
            inputs_resolved_ = false;
        }
    }
    Label("Words");
    std::string words = "Split as the Text Tokenizer splits (lowercase)";
    if (input_rows_ >= 0) words += kDot + cyxwiz::GroupedCount(static_cast<size_t>(input_rows_)) + " rows";
    else if (input_name_.empty()) words += std::string(kDot) + "connect a table with the text";
    else words += std::string(kDot) + "load the inputs to list the columns";
    Dim(words);
    if (GloVe()) {
        RenderGloVeSettings();
    } else {
        RenderWord2VecSettings();
    }
    ImGui::Dummy(ImVec2(0, t.space_sm));
    RenderRunCard();
}

void WordEmbeddingsDialog::RenderWord2VecSettings() {
    const auto& t = ui::CurrentTokens();
    const char* model_labels[] = {"Skip-gram", "CBOW"};
    const char* model_values[] = {"skipgram", "cbow"};
    Label("Model");
    Choice("##architecture", "architecture", model_labels, model_values, 2);
    ImGui::Dummy(ImVec2(0, t.space_xs));
    if (ImGui::BeginTable("##training", 2, ImGuiTableFlags_SizingStretchSame)) {
        const struct {
            const char* label;
            const char* key;
        } fields[] = {{"Vector size", "vector_size"}, {"Window", "window"}, {"Minimum count", "min_count"},
                      {"Negative samples", "negative"}, {"Epochs", "epochs"}, {"Seed", "train_seed"}};
        for (const auto& field : fields) {
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
            ImGui::TextUnformatted(field.label);
            ImGui::PopStyleColor();
            ImGui::SameLine(150.0f);
            IntField((std::string("##") + field.key).c_str(), field.key, kSmallField);
        }
        ImGui::EndTable();
    }
    const std::string advanced = "Advanced: down-sampling " + Get("sample") + kDot + "learning rate " + Get("alpha") +
                                 " to " + Get("min_alpha") + kDot + "batch " + Get("batch_size") +
                                 "###advanced";
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    const bool open = ImGui::TreeNodeEx(advanced.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleColor();
    if (open) {
        Label("Down-sampling");
        FloatField("##sample", "sample", kSmallField);
        Label("Learning rate");
        FloatField("##alpha", "alpha", kSmallField);
        ImGui::SameLine(0.0f, t.space_md);
        ImGui::TextUnformatted("to");
        ImGui::SameLine(0.0f, t.space_md);
        FloatField("##min_alpha", "min_alpha", kSmallField);
        Label("Batch");
        IntField("##batch_size", "batch_size", kSmallField);
        Faint("gensim's names and defaults. Training runs on the selected device; the same seed repeats a run.");
        ImGui::TreePop();
    }
}

void WordEmbeddingsDialog::RenderGloVeSettings() {
    const auto& t = ui::CurrentTokens();
    Label("How it learns");
    Dim("Counts how often each pair of words appears within the window (closer pairs count more), then fits "
        "vectors whose dot products match those counts. word2vec instead learns pair by pair.");
    ImGui::Dummy(ImVec2(0, t.space_xs));
    if (ImGui::BeginTable("##glove", 2, ImGuiTableFlags_SizingStretchSame)) {
        const struct {
            const char* label;
            const char* key;
        } fields[] = {{"Vector size", "vector_size"}, {"Window", "glove_window"}, {"Minimum count", "min_count"},
                      {"Epochs", "glove_epochs"}, {"Seed", "train_seed"}};
        for (const auto& field : fields) {
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
            ImGui::TextUnformatted(field.label);
            ImGui::PopStyleColor();
            ImGui::SameLine(150.0f);
            IntField((std::string("##") + field.key).c_str(), field.key, kSmallField);
        }
        ImGui::EndTable();
    }
    const std::string advanced = "Advanced: count cap x_max " + Get("x_max") + kDot + "weighting power " +
                                 Get("weighting_power") + kDot + "learning rate " + Get("glove_learning_rate") +
                                 " (AdaGrad)" + kDot + "batch " + Get("glove_batch") + "###glove_advanced";
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    const bool open = ImGui::TreeNodeEx(advanced.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleColor();
    if (open) {
        Label("Count cap");
        FloatField("##x_max", "x_max", kSmallField);
        Label("Weighting power");
        FloatField("##weighting_power", "weighting_power", kSmallField);
        Label("Learning rate");
        FloatField("##glove_learning_rate", "glove_learning_rate", kSmallField);
        Label("Batch");
        IntField("##glove_batch", "glove_batch", kSmallField);
        Faint("The Stanford GloVe defaults. Pairs seen more than x_max times all weigh the same; rarer pairs weigh "
              "less (the power). Each word's vector is the sum of its two learned vectors, as the reference writes "
              "them. Training runs on the selected device; the same seed repeats a run.");
        ImGui::TreePop();
    }
}

// Training progress or the result of the latest run.
void WordEmbeddingsDialog::RenderRunCard() {
    const auto& t = ui::CurrentTokens();
    const bool running = run_started_ && TaskLive(run_task_);
    if (!running && run_outcome_.empty() && run_error_.empty()) return;
    if (BeginCard("##run")) {
        if (running) {
            const auto task = cyxwiz::AsyncTaskManager::Instance().GetTask(run_task_);
            const std::string status = task ? task->GetInfo().status_message : std::string();
            const float share = cyxwiz::StepShareFromStatus(status);
            const std::string message = share >= 0.0f ? cyxwiz::StepMessageFromStatus(status) : "Running the steps before it";
            // GloVe's two stages: counting, then the epochs.
            if (message.rfind("Training GloVe", 0) == 0) {
                Wrapped(t.success, std::string(ICON_FA_CIRCLE_CHECK "  ") + "Counted word pairs");
            }
            Heading(std::string(ICON_FA_SPINNER "  ") + message);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_input);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent_text);
            ImGui::ProgressBar(share >= 0.0f ? share : 0.0f, ImVec2(-1.0f, 6.0f), "");
            ImGui::PopStyleColor(2);
            Dim("Runs as a task (Task View); the dialog stays usable.");
            if (ui::LinkButton("Cancel training")) cyxwiz::AsyncTaskManager::Instance().Cancel(run_task_);
        } else if (!run_error_.empty()) {
            Wrapped(t.caution, std::string(ICON_FA_TRIANGLE_EXCLAMATION "  ") + run_error_);
        } else {
            char seconds[32];
            std::snprintf(seconds, sizeof(seconds), "%.1f s", run_seconds_);
            {
                ui::FontScope bold(ui::Font::Medium);
                Wrapped(t.success, std::string(ICON_FA_CIRCLE_CHECK "  ") +
                                       (Training() ? "Trained in " : "Read in ") + seconds);
            }
            Dim(cyxwiz::ShortEpochFigures(run_outcome_));
            const auto losses = cyxwiz::WordEmbeddingsOperator::EpochLosses(run_outcome_);
            if (losses.size() >= 2) {
                // Loss per epoch.
                const float height = 60.0f;
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float width = ImGui::GetContentRegionAvail().x;
                const auto [low, high] = std::minmax_element(losses.begin(), losses.end());
                const double span = (*high - *low) > 1e-12 ? (*high - *low) : 1.0;
                const float plot = height - ImGui::GetTextLineHeight() - 6.0f;
                std::vector<ImVec2> points;
                for (size_t i = 0; i < losses.size(); ++i) {
                    const float x = at.x + 8.0f + (width - 16.0f) * static_cast<float>(i) / (losses.size() - 1);
                    const float y = at.y + 4.0f + (plot - 8.0f) * static_cast<float>((*high - losses[i]) / span);
                    points.emplace_back(x, y);
                }
                ImDrawList* draw = ImGui::GetWindowDrawList();
                draw->AddPolyline(points.data(), static_cast<int>(points.size()), ui::ToU32(t.accent_text),
                                  ImDrawFlags_None, 2.0f);
                const std::string last = "epoch " + std::to_string(losses.size());
                draw->AddText(ImVec2(at.x + 8.0f, at.y + plot), ui::ToU32(t.text_faint), "epoch 1");
                draw->AddText(ImVec2(at.x + width - 8.0f - ImGui::CalcTextSize(last.c_str()).x, at.y + plot),
                              ui::ToU32(t.text_faint), last.c_str());
                ImGui::Dummy(ImVec2(width, height));
                if (ImGui::IsItemHovered()) {
                    // Every epoch's figure (the summary above shows the first and last).
                    std::string figures;
                    for (size_t i = 0; i < losses.size(); ++i) {
                        char line[48];
                        std::snprintf(line, sizeof(line), "\nepoch %zu: %.4g", i + 1, losses[i]);
                        figures += line;
                    }
                    const char* what = run_outcome_.find("(GloVe") != std::string::npos
                                           ? "Mean cost per word pair in each epoch"
                                           : "Mean loss per example in each epoch";
                    ImGui::SetTooltip("%s%s", what, figures.c_str());
                }
            }
        }
    }
    EndCard();
}

void WordEmbeddingsDialog::RenderVocabulary() {
    const auto& t = ui::CurrentTokens();
    Label("Vocabulary file");
    FileField("##vocab_file", "vocab_file", "Choose a vocabulary", false);
    const std::string vocab = Get("vocab_file");
    ImGui::Indent(kLabelWidth);
    if (!tokenizer_vocab_.empty() && vocab != tokenizer_vocab_) {
        Faint("The graph's Text Tokenizer uses " + FileName(tokenizer_vocab_) + ".");
        if (ui::LinkButton("Use it")) Set("vocab_file", tokenizer_vocab_);
    } else if (vocab.empty()) {
        Faint("Empty: the vectors keep their own words, in the file's (or the training's) order.");
    } else {
        Faint("Row i of the vectors is token id i of this file, so an Embedding layer can load them.");
    }
    ImGui::Unindent(kLabelWidth);
    ImGui::Dummy(ImVec2(0, t.space_sm));
    if (Training()) {
        if (!run_outcome_.empty() && !vocab.empty()) Dim("Latest run: " + cyxwiz::ShortEpochFigures(run_outcome_));
        else if (!vocab.empty()) Faint("Train to see how the vocabulary lines up.");
    } else {
        RenderCoverage(true);
    }
}

void WordEmbeddingsDialog::RenderNearest() {
    const auto& t = ui::CurrentTokens();
    const auto vectors = Training() ? run_vectors_ : (read_vectors_ ? read_vectors_ : run_vectors_);
    if (Training()) {
        RenderRunCard();
    } else if (!read_summary_.empty()) {
        if (BeginCard("##read")) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.success);
            ImGui::TextUnformatted(ICON_FA_CIRCLE_CHECK);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            Heading(FileName(Get("vectors_file")));
            Dim(read_summary_);
        }
        EndCard();
    }
    if (!vectors) {
        ImGui::Dummy(ImVec2(0, t.space_lg));
        Dim(Training() ? "Train the vectors (Source tab) to look up words here."
                       : "Choose a vector file (Source tab) to look up words here.");
        return;
    }
    ImGui::Dummy(ImVec2(0, t.space_sm));
    Label("Word");
    ImGui::SetNextItemWidth((std::min)(320.0f, ImGui::GetContentRegionAvail().x));
    const bool entered = ImGui::InputTextWithHint("##word", "type a word, then Enter", query_, sizeof(query_),
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    if (entered || (ImGui::IsItemDeactivatedAfterEdit() && query_word_ != query_)) {
        query_word_ = query_;
        std::string error;
        nearest_ = cyxwiz::WordEmbeddings::Nearest(*vectors, query_word_, kNearest, &error);
        nearest_note_ = !error.empty() ? error
                        : nearest_.empty() && !query_word_.empty()
                            ? "No vector for \"" + query_word_ + "\" in these " +
                                  cyxwiz::GroupedCount(vectors->words.size()) + " words."
                            : std::string();
    }
    if (!nearest_note_.empty()) Wrapped(t.caution, nearest_note_);
    if (nearest_.empty()) return;
    if (ImGui::BeginTable("##nearest", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Nearest", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn("Cosine", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableHeadersRow();
        for (const auto& [word, cosine] : nearest_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Printable(word).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", cosine);
        }
        ImGui::EndTable();
    }
    Faint("Cosine similarity on the selected device; " + cyxwiz::GroupedCount(vectors->words.size()) + " words.");
}

void WordEmbeddingsDialog::RenderOutputs() {
    const auto& t = ui::CurrentTokens();
    int dim = 0;
    try {
        dim = std::stoi(Get("vector_size"));
    } catch (...) {
    }
    if (!Training() && read_vectors_) dim = read_vectors_->dim;
    if (run_vectors_) dim = run_vectors_->dim;
    const std::string last = dim > 0 ? "v_" + std::to_string(dim - 1) : std::string("v_{d-1}");
    const std::string last_emb = dim > 0 ? "emb_" + std::to_string(dim - 1) : std::string("emb_{d-1}");
    if (BeginCard("##vectors_output")) {
        Heading("Vectors (output pin)");
        std::string rows = Get("vocab_file").empty() ? "one row per word" : "one row per vocabulary token in id order";
        if (!Training() && vocab_size_ > 0) rows += " (" + cyxwiz::GroupedCount(vocab_size_) + " rows)";
        else if (run_vectors_) rows += " (" + cyxwiz::GroupedCount(run_vectors_->words.size()) + " rows)";
        Dim("A table: word, v_0 ... " + last + ", " + rows + ".");
        Label("Saved as");
        const std::string default_file =
            "artifacts/" + cyxwiz::WordEmbeddingsOperator::DefaultVectorsFile(node_ ? node_->id : 0, node_ ? node_->name : "");
        FileField("##vectors_path", "vectors_path", "Save word vectors", true, default_file.c_str());
        Faint("Empty saves it in the project at " + default_file +
              ", replaced at each run. An Embedding layer loads it: Embedding > Weights > From a Word Embeddings node.");
    }
    EndCard();
    ImGui::Dummy(ImVec2(0, t.space_sm));
    if (BeginCard("##rows_output")) {
        Heading("Row vectors (optional output pin)");
        bool rows = Get("row_vectors") == "true";
        const std::string label = "One averaged vector per text row, as numeric columns emb_0 ... " + last_emb;
        if (ImGui::Checkbox(label.c_str(), &rows)) Set("row_vectors", rows ? "true" : "false");
        Faint("For classic ML on text (Logistic Regression, SVM, KNN). Words without a vector are skipped; a row with "
              "none gets zeros. Needs the text column (Source tab).");
    }
    EndCard();
}

float WordEmbeddingsDialog::FooterHeight() const {
    const auto& t = ui::CurrentTokens();
    return ImGui::GetFontSize() + t.button_padding.y * 2.0f + ImGui::GetStyle().ItemSpacing.y * 3.0f;
}

void WordEmbeddingsDialog::RenderFooter(bool& should_close) {
    ImGui::Dummy(ImVec2(0, 2.0f));
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const bool training = Training();
    const bool running = run_started_ && TaskLive(run_task_);
    // Train while there is no result for these settings; OK once there is.
    const bool trained = training && !running && run_error_.empty() && run_vectors_ && !has_changes_;
    const char* primary = training && !trained ? "Train" : "OK";
    float width = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular) + ui::ButtonWidth(primary, ui::ButtonSize::Regular) +
                  spacing;
    if (!training) width += ui::ButtonWidth("Apply", ui::ButtonSize::Regular) + spacing;
    ImGui::SetCursorPosX((std::max)(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - width));
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) {
        Reset();
        should_close = true;
    }
    if (!training) {
        ImGui::SameLine();
        if (ui::SecondaryButton("Apply", has_changes_, "Nothing changed", ui::ButtonSize::Regular)) Apply();
    }
    ImGui::SameLine();
    if (training && !trained) {
        const bool can = !running && node_editor_ && !Get("text_col").empty();
        const char* why = running ? "Training is running" : "Choose the text column";
        if (ui::PrimaryButton("Train", can, why)) StartTraining();
    } else if (ui::PrimaryButton("OK")) {
        Apply();
        should_close = true;
    }
}

}  // namespace gui
