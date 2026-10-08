#include "language_model_generation_panel.h"
#include "language_model_generation_panel_metadata.h"

#include "../../core/language_model_generation.h"
#include "../../core/training_manager.h"
#include "../../core/file_dialogs.h"
#include "../../core/formats/cyxmodel_format.h"
#include "../../core/model_importer.h"
#include "../../inference/text_inference_input.h"
#include "../../inference/language_model_inference_contract.h"
#include "../icons.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {

namespace {
constexpr const char* kTokenizerLabels[] = {
    "Whitespace", "Word", "Character", "Byte BPE", "WordPiece",
    "SentencePiece BPE (optional)", "SentencePiece Unigram (optional)"};
} // namespace

LanguageModelGenerationPanel::LanguageModelGenerationPanel()
    : Panel("Language Model Generation", false) {
    status_ =
        "Train a decoder model that returns Float32 [1, seq, vocab] logits, "
        "then enter prompt token IDs.";
}

LanguageModelGenerationPanel::~LanguageModelGenerationPanel() = default;

void LanguageModelGenerationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(760, 560), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Language Model Generation###LanguageModelGeneration",
                     &visible_)) {
        ImGui::Text("%s Language Model Generation", ICON_FA_WAND_MAGIC_SPARKLES);
        ImGui::TextWrapped(
            "Current contract: the model must return Float32 logits shaped "
            "[1, seq, vocab]. Text mode uses a CyxWiz vocabulary file; raw "
            "token-ID mode remains available for debugging.");
        ImGui::Separator();

        RenderPrompt();
        ImGui::Separator();
        RenderControls();
        ImGui::Separator();
        RenderResult();
    }
    ImGui::End();
}

void LanguageModelGenerationPanel::InvalidateOutput() {
    action_error_.clear();
    compatibility_status_.clear();
    status_.clear();
    has_result_ = false;
    stop_reason_.clear();
    generated_ids_.clear();
    generated_text_.clear();
    last_candidates_.clear();
    sampling_settings_.clear();
}

std::string LanguageModelGenerationPanel::RequireActiveTokenizerIdentity(
    const Tokenizer& tokenizer,
    const std::shared_ptr<SequentialModel>& session_model) const {
    TextTokenizerPackage prepared;
    const Tokenizer* expected = imported_model_tokenizer_.get();
    if (!use_imported_model_) {
        expected = nullptr;
        const auto info = TrainingManager::Instance().GetActiveModelInfo();
        // Keep the captured model alive and ensure these metadata belong to it.
        // A later replacement is safe: this action continues with the pinned model.
        if (TrainingManager::Instance().GetActiveModel() != session_model)
            throw std::runtime_error("Active model changed during tokenizer validation. Retry with the newly loaded model.");
        if (info.evaluation_config) {
            const auto& sequence = info.evaluation_config->sequence_batch;
            if (!sequence.tokenizer_config_json.empty() ||
                !sequence.tokenizer_vocabulary_artifact.empty()) {
                if (sequence.tokenizer_config_json.empty() ||
                    sequence.tokenizer_vocabulary_artifact.empty())
                    throw std::runtime_error("Active model tokenizer metadata is incomplete: configuration and vocabulary are both required.");
                std::string error;
                if (!LoadTextTokenizerPackage(sequence.tokenizer_config_json,
                        sequence.tokenizer_vocabulary_artifact, prepared, error) ||
                    !prepared.tokenizer || !prepared.has_vocabulary)
                    throw std::runtime_error("Active model tokenizer metadata is incomplete or invalid: " + error);
                expected = prepared.tokenizer.get();
            }
        }
    }
    const auto identity = ValidateLanguageModelTokenizerIdentity(tokenizer, expected,
        use_packaged_tokenizer_ ? packaged_tokenizer_model_data_ : std::string_view{},
        use_imported_model_ ? imported_model_tokenizer_data_ : std::string_view{});
    if (!identity.compatible) throw std::runtime_error(identity.message);
    return identity.message;
}

void LanguageModelGenerationPanel::RenderPrompt() {
    ImGui::Text("%s Prompt", ICON_FA_KEYBOARD);
    if (ImGui::RadioButton("Text prompt", use_text_prompt_)) {
        InvalidateOutput();
        use_text_prompt_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Raw token IDs", !use_text_prompt_)) {
        InvalidateOutput();
        use_text_prompt_ = false;
    }

    if (use_text_prompt_) {
        if (ImGui::InputTextMultiline("##TextPrompt",
                                  text_prompt_,
                                  sizeof(text_prompt_),
                                  ImVec2(-1, 96))) InvalidateOutput();

        if (ImGui::InputText("Vocabulary file", vocab_file_, sizeof(vocab_file_))) InvalidateOutput();
        ImGui::SameLine();
        if (ImGui::Button("Browse##GenerationVocab")) {
            auto result = FileDialogs::OpenFile(
                "Select Vocabulary File",
                {{"Vocabulary", "txt,vocab"},
                 {"All Files", "*"}});
            if (result) {
                InvalidateOutput();
                std::snprintf(vocab_file_,
                              sizeof(vocab_file_),
                              "%s",
                              result->c_str());
            }
        }

        if (ImGui::InputText("CyxModel package", cyxmodel_path_, sizeof(cyxmodel_path_))) InvalidateOutput();
        ImGui::SameLine();
        if (ImGui::Button("Browse##GenerationCyxModel")) {
            auto result = FileDialogs::OpenFile(
                "Select CyxModel Package",
                {{"CyxModel", "cyxmodel"},
                 {"All Files", "*"}});
            if (result) {
                InvalidateOutput();
                std::snprintf(cyxmodel_path_,
                              sizeof(cyxmodel_path_),
                              "%s",
                              result->c_str());
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load tokenizer assets")) {
            LoadTokenizerFromCyxModel();
        }
        ImGui::SameLine();
        if (ImGui::Button("Load model + tokenizer")) {
            LoadModelAndTokenizerFromCyxModel();
        }

        if (ImGui::Checkbox("Use packaged tokenizer assets", &use_packaged_tokenizer_)) InvalidateOutput();
        if (!packaged_tokenizer_summary_.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", packaged_tokenizer_summary_.c_str());
        }
        if (imported_model_) {
            if (ImGui::Checkbox("Use imported .cyxmodel model", &use_imported_model_)) InvalidateOutput();
            ImGui::SameLine();
            ImGui::TextDisabled("%s", imported_model_source_.c_str());
            if (!imported_model_summary_.empty()) {
                ImGui::TextDisabled("%s", imported_model_summary_.c_str());
            }
        }

        if (use_packaged_tokenizer_) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Combo("Tokenizer", &tokenizer_type_idx_, kTokenizerLabels, 7)) {
            InvalidateOutput();
            if (tokenizer_type_idx_ == 3) lowercase_ = false;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(tokenizer_type_idx_ == 3);
        if (ImGui::Checkbox("Lowercase", &lowercase_)) InvalidateOutput();
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Checkbox("BOS", &add_bos_)) InvalidateOutput();
        ImGui::SameLine();
        if (ImGui::Checkbox("EOS", &add_eos_)) InvalidateOutput();
        if (ImGui::InputInt("Max prompt length", &max_length_)) InvalidateOutput();
        if (use_packaged_tokenizer_) {
            ImGui::EndDisabled();
        }
        ImGui::TextDisabled(
            "Manual mode uses the selected vocabulary file. Packaged mode uses "
            "tokenizer/config.json and tokenizer/vocab.txt from a .cyxmodel.");
    } else {
        if (ImGui::InputTextMultiline("##PromptTokenIds",
                                  prompt_ids_,
                                  sizeof(prompt_ids_),
                                  ImVec2(-1, 96))) InvalidateOutput();
        ImGui::TextDisabled(
            "Use spaces, commas, or new lines. Example: 1 42 17");
    }
}

void LanguageModelGenerationPanel::RenderControls() {
    ImGui::Text("%s Generation controls", ICON_FA_SLIDERS);
    ImGui::PushItemWidth(180);
    if (ImGui::InputInt("Max new tokens", &max_new_tokens_)) InvalidateOutput();
    if (ImGui::InputFloat("Temperature", &temperature_, 0.05f, 0.25f, "%.3f")) InvalidateOutput();
    if (ImGui::InputInt("Top-K (0 disables)", &top_k_)) InvalidateOutput();
    if (ImGui::InputFloat("Top-P", &top_p_, 0.05f, 0.25f, "%.3f")) InvalidateOutput();
    if (ImGui::InputInt("EOS token (-1 disables)", &eos_token_id_)) InvalidateOutput();
    if (ImGui::InputInt("Seed", &seed_)) InvalidateOutput();
    ImGui::PopItemWidth();

    if (ImGui::Checkbox("Multinomial sampling", &multinomial_sampling_)) InvalidateOutput();
    ImGui::SameLine();
    if (ImGui::Checkbox("Include prompt in output", &include_prompt_)) InvalidateOutput();

    auto& training = TrainingManager::Instance();
    const auto session_model = training.GetActiveModel();
    const std::weak_ptr<SequentialModel> current_owner = session_model;
    if (observed_session_model_.owner_before(current_owner) ||
        current_owner.owner_before(observed_session_model_)) {
        if (!use_imported_model_) InvalidateOutput();
        observed_session_model_ = session_model;
        const auto info = training.GetActiveModelInfo();
        session_model_status_ = !session_model ? "Active model: none"
            : (info.origin == ActiveModelOrigin::LoadedCheckpoint
                ? "Active checkpoint: " + info.checkpoint_path
                : "Active model: trained in this session");
    }
    const bool has_model = (use_imported_model_ && imported_model_) || session_model;
    active_model_status_ = use_imported_model_ && imported_model_
        ? "Active package: " + imported_model_source_ : session_model_status_;
    ImGui::TextWrapped("%s", active_model_status_.c_str());
    if (!has_model) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.15f, 1.0f),
                           "%s No trained or imported model available",
                           ICON_FA_TRIANGLE_EXCLAMATION);
    }

    if (!has_model) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_FA_STETHOSCOPE " Check compatibility", ImVec2(190, 0))) {
        CheckModelCompatibility();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PLAY " Generate", ImVec2(160, 0))) {
        RunGeneration();
    }
    if (!has_model) {
        ImGui::EndDisabled();
    }

    // Keep actionable errors beside the buttons, outside the output area.
    // Use text and an icon as well as color, and wrap long paths/messages.
    if (!action_error_.empty()) {
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.35f, 1.0f));
        ImGui::Text("%s Action blocked", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::TextWrapped("%s", action_error_.c_str());
        ImGui::PopStyleColor();
        if (ImGui::Button("Copy error")) ImGui::SetClipboardText(action_error_.c_str());
    } else if (!compatibility_status_.empty()) {
        ImGui::TextWrapped("%s", compatibility_status_.c_str());
    }
}

void LanguageModelGenerationPanel::RenderResult() {
    ImGui::Text("%s Result", ICON_FA_TERMINAL);
    if (!status_.empty() && action_error_.empty()) {
        ImGui::TextWrapped("%s", status_.c_str());
    }

    if (!stop_reason_.empty()) {
        ImGui::Text("Stop reason: %s", stop_reason_.c_str());
    }
    if (!has_result_) {
        return;
    }

    ImGui::Text("Prompt length: %zu", last_prompt_length_);
    ImGui::SameLine();
    if (last_max_context_length_ > 0) {
        ImGui::Text("Max context: %zu", last_max_context_length_);
    } else {
        ImGui::Text("Max context: unbounded");
    }
    ImGui::Text("Remaining generation budget: %zu", last_remaining_budget_);
    if (!sampling_settings_.empty()) {
        ImGui::TextWrapped("Sampling settings: %s", sampling_settings_.c_str());
    }

    const std::string joined = JoinTokenIds(generated_ids_);
    ImGui::Text("Generated token IDs");
    ImGui::InputTextMultiline("##GeneratedTokenIds",
                              const_cast<char*>(joined.c_str()),
                              joined.size() + 1,
                              ImVec2(-1, 120),
                              ImGuiInputTextFlags_ReadOnly);
    if (!generated_text_.empty()) {
        ImGui::Text("Decoded text");
        ImGui::InputTextMultiline("##GeneratedText",
                                  const_cast<char*>(generated_text_.c_str()),
                                  generated_text_.size() + 1,
                                  ImVec2(-1, 120),
                                  ImGuiInputTextFlags_ReadOnly);
    }

    if (!last_candidates_.empty() &&
        ImGui::BeginTable("##GenerationCandidateTable",
                          2,
                          ImGuiTableFlags_Borders |
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Token ID");
        ImGui::TableSetupColumn("Probability");
        ImGui::TableHeadersRow();
        const size_t visible_candidates =
            std::min<size_t>(last_candidates_.size(), 8);
        for (size_t i = 0; i < visible_candidates; ++i) {
            const auto& candidate = last_candidates_[i];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%lld", static_cast<long long>(candidate.token_id));
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.6f", candidate.probability);
        }
        ImGui::EndTable();
    }
}

void LanguageModelGenerationPanel::RunGeneration() {
    InvalidateOutput();
    try {
        if (use_imported_model_) {
            if (use_text_prompt_ && !use_packaged_tokenizer_) {
                throw std::runtime_error(
                    "Imported .cyxmodel generation requires packaged tokenizer assets");
            }
            RequireImportedModelPackageContract();
        }

        const auto session_model = TrainingManager::Instance().GetActiveModel();
        auto* model = use_imported_model_ ? imported_model_.get() : session_model.get();
        if (!model) throw std::runtime_error("No trained or imported model is available");
        std::string tokenizer_identity = "Raw token IDs: tokenizer identity not checked.";
        std::unique_ptr<Tokenizer> tokenizer;
        std::vector<int64_t> prompt;
        size_t tokenizer_vocab_size = 0;
        size_t max_sequence_length = 0;
        if (use_text_prompt_) {
            tokenizer = BuildTokenizer();
            tokenizer_identity = RequireActiveTokenizerIdentity(*tokenizer, session_model);
            prompt = EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
            tokenizer_vocab_size = tokenizer->GetVocabularySize();
            max_sequence_length = static_cast<size_t>(std::max(0, tokenizer->GetMaxLength()));
        } else {
            prompt = ParsePromptIds();
        }
        if (use_imported_model_) {
            tokenizer_vocab_size = imported_model_contract_.tokenizer_vocabulary_size;
            max_sequence_length = imported_model_contract_.max_sequence_length;
        }

        const auto prompt_contract = ValidateLanguageModelPromptIds(
            prompt,
            max_sequence_length);
        if (!prompt_contract.compatible) {
            throw std::runtime_error("Invalid prompt: " + prompt_contract.error);
        }

        LanguageModelGenerationConfig config;
        config.max_new_tokens =
            static_cast<size_t>(std::max(1, max_new_tokens_));
        config.temperature = temperature_;
        config.top_k = top_k_ > 0 ? static_cast<size_t>(top_k_) : 0;
        config.top_p = top_p_;
        config.eos_token_id = static_cast<int64_t>(eos_token_id_);
        config.include_prompt = include_prompt_;
        config.max_context_tokens = max_sequence_length;
        config.sampling_mode = multinomial_sampling_
            ? LanguageModelSamplingMode::Multinomial
            : LanguageModelSamplingMode::Greedy;
        RequireValidLanguageModelGenerationConfig(config);
        if (config.max_context_tokens > 0 &&
            prompt.size() >= config.max_context_tokens) {
            throw std::runtime_error(
                "Prompt length leaves no generation budget for max context");
        }


        Tensor contract_input({1, prompt.size()}, prompt.data(), DataType::Int64);
        const Tensor contract_logits = model->Forward(contract_input);
        const auto runtime_contract = ValidateLanguageModelRuntimeOutput(
            contract_logits,
            prompt.size(),
            tokenizer_vocab_size);
        if (!runtime_contract.compatible) {
            throw std::runtime_error(
                "Model output contract failed: " + runtime_contract.error);
        }

        const auto report = GenerateTokenIdsWithReport(
            *model,
            prompt,
            config,
            static_cast<uint32_t>(std::max(0, seed_)));
        generated_ids_ = report.token_ids;
        const auto metadata = BuildLanguageModelGenerationPanelRunMetadata(
            report,
            config,
            max_sequence_length,
            static_cast<uint32_t>(std::max(0, seed_)));
        stop_reason_ = metadata.stop_reason;
        last_prompt_length_ = metadata.prompt_length;
        last_max_context_length_ = metadata.max_context_length;
        last_remaining_budget_ = metadata.remaining_budget;
        last_candidates_ = metadata.last_candidates;
        sampling_settings_ = metadata.sampling_settings;

        generated_text_.clear();
        if (tokenizer) {
            generated_text_ = DecodeGeneratedTokenIds(*tokenizer, generated_ids_);
        }
        has_result_ = true;
        status_ = "Generation completed: " +
                  std::to_string(report.new_token_ids.size()) +
                  " new token IDs. " + tokenizer_identity;
        spdlog::info("Language Model Generation: {}", status_);
    } catch (const std::exception& e) {
        has_result_ = false;
        generated_ids_.clear();
        generated_text_.clear();
        stop_reason_ = "error";
        sampling_settings_.clear();
        last_prompt_length_ = 0;
        last_max_context_length_ = 0;
        last_remaining_budget_ = 0;
        last_candidates_.clear();
        status_ = std::string("Generation failed: ") + e.what();
        action_error_ = status_;
        spdlog::error("Language Model Generation: {}", status_);
    }
}

void LanguageModelGenerationPanel::CheckModelCompatibility() {
    InvalidateOutput();
    try {
        if (use_imported_model_) {
            if (use_text_prompt_ && !use_packaged_tokenizer_) {
                throw std::runtime_error(
                    "Imported .cyxmodel compatibility checks require packaged tokenizer assets");
            }
            RequireImportedModelPackageContract();
        }

        const auto session_model = TrainingManager::Instance().GetActiveModel();
        auto* model = use_imported_model_ ? imported_model_.get() : session_model.get();
        if (!model) throw std::runtime_error("No trained or imported model is available");
        std::string tokenizer_identity = "Raw token IDs: tokenizer identity not checked.";
        std::unique_ptr<Tokenizer> tokenizer;
        std::vector<int64_t> prompt;
        size_t tokenizer_vocab_size = 0;
        size_t max_sequence_length = 0;
        if (use_text_prompt_) {
            tokenizer = BuildTokenizer();
            tokenizer_identity = RequireActiveTokenizerIdentity(*tokenizer, session_model);
            prompt = EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
            tokenizer_vocab_size = tokenizer->GetVocabularySize();
            max_sequence_length = static_cast<size_t>(std::max(0, tokenizer->GetMaxLength()));
        } else {
            prompt = ParsePromptIds();
        }
        if (use_imported_model_) {
            tokenizer_vocab_size = imported_model_contract_.tokenizer_vocabulary_size;
            max_sequence_length = imported_model_contract_.max_sequence_length;
        }

        const auto prompt_contract = ValidateLanguageModelPromptIds(
            prompt,
            max_sequence_length);
        if (!prompt_contract.compatible) {
            throw std::runtime_error("Invalid prompt: " + prompt_contract.error);
        }

        Tensor input({1, prompt.size()}, prompt.data(), DataType::Int64);
        const Tensor logits = model->Forward(input);
        const auto runtime_contract = ValidateLanguageModelRuntimeOutput(
            logits,
            prompt.size(),
            tokenizer_vocab_size);
        if (!runtime_contract.compatible) {
            throw std::runtime_error(runtime_contract.error);
        }

        compatibility_status_ =
            "Compatible: model returned Float32 [1, " +
            std::to_string(runtime_contract.sequence_length) + ", " +
            std::to_string(runtime_contract.vocab_size) + "] logits";
        if (tokenizer_vocab_size > 0) {
            compatibility_status_ +=
                "; tokenizer vocab=" + std::to_string(tokenizer_vocab_size);
            if (use_imported_model_) {
                compatibility_status_ +=
                    ", eos=" + std::to_string(imported_model_contract_.eos_token_id);
            } else if (tokenizer) {
                compatibility_status_ +=
                    ", eos=" + std::to_string(tokenizer->GetEosId());
            }
        }
        compatibility_status_ += ". " + tokenizer_identity;
    } catch (const std::exception& e) {
        compatibility_status_ =
            std::string("Not compatible for generation: ") + e.what();
        action_error_ = compatibility_status_;
        spdlog::error("Language Model Generation: {}", action_error_);
    }
}

void LanguageModelGenerationPanel::RequireImportedModelPackageContract() const {
    if (!use_imported_model_) {
        return;
    }
    if (imported_model_contract_.package_path.empty()) {
        throw std::runtime_error(
            "Imported model package contract is not available");
    }
    if (!imported_model_contract_.compatible) {
        throw std::runtime_error(
            "Package contract failed: " + imported_model_contract_.error);
    }
}

void LanguageModelGenerationPanel::LoadTokenizerFromCyxModel() {
    InvalidateOutput();
    try {
        if (cyxmodel_path_[0] == '\0') {
            throw std::invalid_argument("Choose a .cyxmodel package first");
        }

        formats::CyxModelFormat format;
        std::string config_json;
        std::string vocab_text;
        std::string model_data;
        if (!format.ExtractTextTokenizerAssets(cyxmodel_path_,
                                               config_json,
                                               vocab_text,
                                               model_data)) {
            throw std::runtime_error(
                "No tokenizer assets found in package: " +
                format.GetLastError());
        }

        TextTokenizerPackage package;
        std::string error;
        if (!LoadTextTokenizerPackage(config_json, vocab_text, model_data, package, error)) {
            throw std::runtime_error(error);
        }
        if (!package.has_vocabulary && !package.has_model_artifact) {
            throw std::runtime_error(
                "Package tokenizer assets do not include a vocabulary or model artifact");
        }

        if (package.tokenizer) {
            eos_token_id_ = package.tokenizer->GetEosId();
            max_length_ = package.tokenizer->GetMaxLength();
            lowercase_ = package.tokenizer->GetLowercase();
            switch (package.tokenizer->GetType()) {
                case TokenizerType::Whitespace: tokenizer_type_idx_ = 0; break;
                case TokenizerType::Word: tokenizer_type_idx_ = 1; break;
                case TokenizerType::Character: tokenizer_type_idx_ = 2; break;
                case TokenizerType::ByteBPE: tokenizer_type_idx_ = 3; break;
                case TokenizerType::WordPiece: tokenizer_type_idx_ = 4; break;
                case TokenizerType::SentencePieceBPE: tokenizer_type_idx_ = 5; break;
                case TokenizerType::SentencePieceUnigram: tokenizer_type_idx_ = 6; break;
            }
            packaged_tokenizer_summary_ =
                "packaged tokenizer: vocab=" +
                std::to_string(package.tokenizer->GetVocabularySize()) +
                ", max_len=" + std::to_string(max_length_) +
                ", eos=" + std::to_string(eos_token_id_);
        }

        packaged_tokenizer_config_json_ = std::move(config_json);
        packaged_tokenizer_vocab_text_ = std::move(vocab_text);
        packaged_tokenizer_model_data_ = std::move(model_data);
        use_packaged_tokenizer_ = true;
        status_ = "Loaded packaged tokenizer assets from .cyxmodel.";
    } catch (const std::exception& e) {
        use_packaged_tokenizer_ = false;
        packaged_tokenizer_config_json_.clear();
        packaged_tokenizer_vocab_text_.clear();
        packaged_tokenizer_model_data_.clear();
        packaged_tokenizer_summary_.clear();
        status_ = std::string("Failed to load packaged tokenizer: ") + e.what();
        action_error_ = status_;
    }
}

void LanguageModelGenerationPanel::LoadModelAndTokenizerFromCyxModel() {
    InvalidateOutput();
    try {
        if (cyxmodel_path_[0] == '\0') {
            throw std::invalid_argument("Choose a .cyxmodel package first");
        }

        auto model = std::make_unique<SequentialModel>();
        ModelImporter importer;
        ImportOptions options;
        const auto result = importer.Import(cyxmodel_path_, *model, options);
        if (!result.success) {
            throw std::runtime_error(
                result.error_message.empty()
                    ? importer.GetLastError()
                    : result.error_message);
        }

        LoadTokenizerFromCyxModel();
        if (!use_packaged_tokenizer_) {
            throw std::runtime_error(status_);
        }

        TextTokenizerPackage contract_tokenizer_package;
        std::string contract_tokenizer_error;
        if (!LoadTextTokenizerPackage(packaged_tokenizer_config_json_,
                                      packaged_tokenizer_vocab_text_,
                                      packaged_tokenizer_model_data_,
                                      contract_tokenizer_package,
                                      contract_tokenizer_error)) {
            throw std::runtime_error(contract_tokenizer_error);
        }

        imported_model_ = std::move(model);
        imported_model_source_ = cyxmodel_path_;
        imported_model_summary_.clear();
        const auto probe = importer.ProbeFile(cyxmodel_path_);
        const auto package_contract = ValidateLanguageModelPackageContract(
            probe,
            &contract_tokenizer_package,
            cyxmodel_path_);
        imported_model_contract_ = package_contract;
        imported_model_tokenizer_ = std::move(contract_tokenizer_package.tokenizer);
        imported_model_tokenizer_data_ = packaged_tokenizer_model_data_;
        imported_model_summary_ =
            "package: family=" +
            (package_contract.model_family.empty() ? std::string("unspecified")
                                                   : package_contract.model_family) +
            ", generation=" +
            (package_contract.supports_generation ? std::string("yes")
                                                  : std::string("no")) +
            ", contract=" +
            (package_contract.generation_output_contract.empty()
                 ? std::string("unspecified")
                 : package_contract.generation_output_contract) +
            ", tokenizer_vocab=" +
            std::to_string(package_contract.tokenizer_vocabulary_size) +
            ", max_len=" + std::to_string(package_contract.max_sequence_length) +
            ", eos=" + std::to_string(package_contract.eos_token_id);
        compatibility_status_ = package_contract.compatible
            ? "Package contract is compatible; use Check compatibility to "
              "validate the active runtime graph."
            : "Package contract failed: " + package_contract.error;
        if (!package_contract.compatible) action_error_ = compatibility_status_;
        use_imported_model_ = true;
        spdlog::info("Language Model Generation: loaded '{}' ({} layers); {}",
                     imported_model_source_, imported_model_->Size(), compatibility_status_);
        status_ = package_contract.compatible
            ? "Loaded model and tokenizer assets from .cyxmodel."
            : "Loaded model package, but generation contract failed: " +
                  package_contract.error;
    } catch (const std::exception& e) {
        imported_model_.reset();
        imported_model_source_.clear();
        imported_model_summary_.clear();
        imported_model_contract_ = {};
        imported_model_tokenizer_.reset();
        imported_model_tokenizer_data_.clear();
        use_imported_model_ = false;
        status_ = std::string("Failed to load model package: ") + e.what();
        action_error_ = status_;
        spdlog::error("Language Model Generation: {}", status_);
    }
}

std::vector<int64_t> LanguageModelGenerationPanel::ParsePromptIds() const {
    std::string normalized(prompt_ids_);
    for (char& c : normalized) {
        if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
            c = ' ';
        }
    }

    std::istringstream in(normalized);
    std::vector<int64_t> ids;
    int64_t id = 0;
    while (in >> id) {
        if (id < 0) {
            throw std::invalid_argument("Prompt token IDs must be non-negative");
        }
        ids.push_back(id);
    }
    if (ids.empty()) {
        throw std::invalid_argument("Prompt token IDs are required");
    }
    return ids;
}

std::vector<int64_t> LanguageModelGenerationPanel::CurrentPromptIdsForProbe() const {
    if (use_text_prompt_) {
        auto tokenizer = BuildTokenizer();
        return EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
    }
    return ParsePromptIds();
}

std::unique_ptr<Tokenizer> LanguageModelGenerationPanel::BuildTokenizer() const {
    if (use_packaged_tokenizer_) {
        if (packaged_tokenizer_config_json_.empty() ||
            (packaged_tokenizer_vocab_text_.empty() &&
             packaged_tokenizer_model_data_.empty())) {
            throw std::invalid_argument(
                "Packaged tokenizer mode is enabled but no package assets are loaded");
        }
        TextTokenizerPackage package;
        std::string error;
        if (!LoadTextTokenizerPackage(packaged_tokenizer_config_json_,
                                      packaged_tokenizer_vocab_text_,
                                      packaged_tokenizer_model_data_,
                                      package,
                                      error)) {
            throw std::runtime_error(error);
        }
        if ((!package.has_vocabulary && !package.has_model_artifact) ||
            !package.tokenizer) {
            throw std::runtime_error(
                "Packaged tokenizer does not contain a usable vocabulary or model artifact");
        }
        return std::move(package.tokenizer);
    }

    TokenizerType type = TokenizerType::Word;
    if (tokenizer_type_idx_ == 0) {
        type = TokenizerType::Whitespace;
    } else if (tokenizer_type_idx_ == 2) {
        type = TokenizerType::Character;
    } else if (tokenizer_type_idx_ == 3) {
        type = TokenizerType::ByteBPE;
    } else if (tokenizer_type_idx_ == 4) {
        type = TokenizerType::WordPiece;
    } else if (tokenizer_type_idx_ == 5) {
        type = TokenizerType::SentencePieceBPE;
    } else if (tokenizer_type_idx_ == 6) {
        type = TokenizerType::SentencePieceUnigram;
    }

    auto tokenizer = std::make_unique<Tokenizer>(type);
    tokenizer->SetLowercase(lowercase_);
    tokenizer->SetMaxLength(std::max(1, max_length_));
    tokenizer->SetPadding(true);
    tokenizer->SetTruncation(true);
    tokenizer->SetAddBos(add_bos_);
    tokenizer->SetAddEos(add_eos_);

    if (vocab_file_[0] == '\0') {
        throw std::invalid_argument("Text mode requires a vocabulary file");
    }
    if (!tokenizer->GetVocabulary().LoadFromFile(vocab_file_)) {
        throw std::runtime_error(
            "Failed to load vocabulary file: " + std::string(vocab_file_));
    }
    try {
        tokenizer->ValidateVocabulary();
    } catch (const std::invalid_argument& e) {
        // Validation remains owned by Tokenizer; this only adds GUI guidance.
        const std::string selected = kTokenizerLabels[std::clamp(tokenizer_type_idx_, 0, 6)];
        std::string guidance;
        if (tokenizer->GetVocabulary().IsByteBPE() &&
            !IsSentencePieceTokenizerType(type)) {
            guidance = "Selected " + selected +
                "; this vocabulary requires Byte BPE. Set Tokenizer to Byte BPE "
                "and Lowercase to off.";
        } else if (type == TokenizerType::ByteBPE) {
            guidance = "Selected Byte BPE, but this vocabulary has no BPE merge rules. "
                "Choose the BPE vocabulary used by the active model, or select "
                "the tokenizer type that was used to train it.";
        } else if (IsSentencePieceTokenizerType(type)) {
            guidance = "Selected " + selected +
                ". Use a .cyxmodel package containing tokenizer/model.spm and "
                "a build with SentencePiece support.";
        }
        throw std::invalid_argument(guidance.empty() ? e.what()
            : guidance + " Details: " + e.what());
    }
    return tokenizer;
}

std::string LanguageModelGenerationPanel::JoinTokenIds(
    const std::vector<int64_t>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            out << ' ';
        }
        out << ids[i];
    }
    return out.str();
}

} // namespace cyxwiz
